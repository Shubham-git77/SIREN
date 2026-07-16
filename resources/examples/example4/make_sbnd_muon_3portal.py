"""
SBND prediction in the same 3-panel per-portal style as MiniBooNE_muon_3portal:
(scalar | pseudoscalar | vector) E_vis, MUON-ONLY (K_mu, pi_mu + TOTAL), at the
MiniBooNE best-fit coupling ("if MiniBooNE is this model, what SBND sees").

Reads the fresh analytic-engine npz (SBND_{scalar,pseudo,vector}_analytic.npz);
NO SBND fit (there is no SBND data) -- the SAME coupling fixed by the MiniBooNE
fit is applied to SBND's flux/argon/geometry. Also prints verification numbers
(per-portal total + SBND/MiniBooNE ratio).
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__); OUT = os.path.join(HERE, "output")
PORTALS = [("scalar", "Scalar $\\phi\\to\\gamma$",   "SBND_scalar_analytic.npz"),
           ("pseudo", "Pseudoscalar $a\\to\\gamma$", "SBND_pseudo_analytic.npz"),
           ("vector", "Vector $\\to e^+e^-$",        "SBND_vector_analytic.npz")]
MUON = ["K_mu", "pi_mu"]
COL  = {"K_mu": "C1", "pi_mu": "C3"}
# MiniBooNE best-fit amplitudes (from make_miniboone_muon_3portal fit) -- the SAME
# coupling applied to SBND. coupling-string for scalar/pseudo, rate for vector.
BESTFIT_A = {"scalar": 0.0382, "pseudo": 0.00246, "vector": 36.86}
KIND = {"scalar": "coupling 0.20x", "pseudo": "coupling 0.05x", "vector": "A=36.9x"}
# MiniBooNE muon-only best-fit totals (for the SBND/MB enhancement cross-check)
MB_TOTAL = {"scalar": 373.0, "pseudo": 251.0, "vector": 331.0}

Ebins = np.linspace(0, 2000, 41); EC = 0.5 * (Ebins[:-1] + Ebins[1:])
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
print("VERIFICATION (SBND, muon-only, MiniBooNE best-fit coupling):")
for j, (key, label, fname) in enumerate(PORTALS):
    d = np.load(os.path.join(OUT, fname)); A = BESTFIT_A[key]
    tot = np.zeros(len(EC)); grand = 0.0
    for nm in MUON:
        if f"{nm}_E" not in d or not len(d[f"{nm}_E"]):
            continue
        E = d[f"{nm}_E"] * 1e3; w = d[f"{nm}_w"] * A
        h = np.histogram(E, bins=Ebins, weights=w)[0]
        ax[j].step(EC, h, where="mid", color=COL[nm], lw=1.5, label=nm)
        tot += h; grand += w.sum()
    ax[j].step(EC, tot, where="mid", color="k", lw=2.2, label="TOTAL")
    ax[j].set_xlabel(r"$E_{\rm vis}$ [MeV]"); ax[j].set_ylabel("Counts")
    ax[j].set_xlim(0, 2000); ax[j].set_ylim(bottom=0); ax[j].legend(fontsize=8)
    ratio = grand / MB_TOTAL[key]
    ax[j].set_title("%s — %s, total %.0f ev (SBND/MB = %.1f)" % (label, KIND[key], grand, ratio),
                    fontsize=9.5)
    print("  %-7s : total = %.0f ev   MiniBooNE = %.0f   SBND/MB = %.1fx"
          % (key, grand, MB_TOTAL[key], ratio))
fig.suptitle("SBND prediction at the MiniBooNE best-fit coupling, MUON-ONLY ($g_e{=}0$), per portal "
             "— analytic $\\sigma\\!\\cdot\\!N\\!\\cdot\\!$chord (same coupling as MiniBooNE)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = os.path.join(OUT, "SBND_muon_3portal.png")
fig.savefig(out, dpi=115); plt.close(fig)
print("Saved -> %s" % out)
