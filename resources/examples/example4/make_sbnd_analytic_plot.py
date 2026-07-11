"""
Combined SBND analytic prediction plot for all three portals (scalar phi,
pseudoscalar a, vector double-mediator -> e+e-).

Uses the VALIDATED analytic estimator outputs (sigma*N*chord ray-trace through
the SBND LAr TPC; NOT the directed SIREN sampler, which over-estimates these
rates 60-400x via a boost-Jacobian bias). Benchmark couplings, POT = 6.6e20.

Left panel : absolute E_vis spectra (log-y; totals span ~10^2 - 10^6).
Right panel: area-normalized spectral shapes.

NB scalar/pseudo E_vis = E_gamma (single photon); vector E_vis ~ E_chi (proxy
for the e+e- visible energy). MUON-ONLY (K_mu + pi_mu): the paper's scenario
(g_e=0; the e channels are dropped -- for the pseudoscalar pi_e is unphysically
large and helicity-unsuppressed, so excluding it is essential). For the vector
(kinetic-mixing, charge-coupled) "muon-only" just means the mu production
channels; the e channels are physical there but dropped for a like-for-like
comparison.
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "output")
PORTALS = [
    ("scalar", "Scalar ($\\phi\\to\\gamma$)",            "SBND_scalar_analytic.npz", "C0"),
    ("pseudo", "Pseudoscalar ($a\\to\\gamma$)",          "SBND_pseudo_analytic.npz", "C3"),
    ("vector", "Vector (double-med. $\\to e^+e^-$)",      "SBND_vector_analytic.npz", "C2"),
]
CHANS = ["K_mu", "pi_mu"]            # muon-only (paper scenario, g_e=0)
# MiniBooNE best-fit amplitudes A from make_miniboone_muon_3portal.py (each portal
# FIT to the MiniBooNE excess; muon-only; vector uses the cascade e+e- energy).
# Scaling the SBND benchmark rate by A = the SBND prediction at the SAME coupling
# that explains the MiniBooNE excess ("if MiniBooNE is this model, what does SBND
# see?"). NB vector A=36.9 (rate factor) -- the old 0.079 came from the broken
# directed shape and is wrong.
BESTFIT_A = {"scalar": 0.0382, "pseudo": 0.00246, "vector": 36.86}
EBINS = np.linspace(0, 3000, 61)          # 50 MeV bins, MeV
EC = 0.5 * (EBINS[:-1] + EBINS[1:])

def load_portal(fname):
    d = np.load(os.path.join(OUT, fname))
    E = np.concatenate([d[f"{c}_E"] for c in CHANS if f"{c}_E" in d and len(d[f"{c}_E"])])
    w = np.concatenate([d[f"{c}_w"] for c in CHANS if f"{c}_w" in d and len(d[f"{c}_w"])])
    return E * 1e3, w                       # GeV -> MeV

fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.2))
for key, label, fname, col in PORTALS:
    E, w = load_portal(fname)
    w = w * BESTFIT_A[key]               # scale benchmark -> MiniBooNE best-fit coupling
    h = np.histogram(E, bins=EBINS, weights=w)[0]
    tot = w.sum()
    ax[0].step(EC, h, where="mid", color=col, lw=2,
               label="%s: %.2e ev" % (label, tot))
    if h.sum() > 0:
        ax[1].step(EC, h / h.sum(), where="mid", color=col, lw=2, label=label)

ax[0].set_yscale("log")
ax[0].set_xlabel(r"$E_{\rm vis}$ [MeV]"); ax[0].set_ylabel("Events / 50 MeV bin")
ax[0].set_xlim(0, 3000); ax[0].set_ylim(bottom=1e-2)
ax[0].legend(fontsize=8.5, title="SBND, $6.6\\times10^{20}$ POT, MiniBooNE best-fit coupling")
ax[0].set_title("Absolute prediction (analytic; log scale)")

ax[1].set_xlabel(r"$E_{\rm vis}$ [MeV]"); ax[1].set_ylabel("fraction / 50 MeV bin")
ax[1].set_xlim(0, 3000); ax[1].set_ylim(bottom=0)
ax[1].legend(fontsize=9)
ax[1].set_title("Spectral shape (area-normalized)")

fig.suptitle("SBND prediction at the MiniBooNE best-fit coupling, MUON-ONLY ($g_e{=}0$) — "
             "analytic $\\sigma\\!\\cdot\\!N\\!\\cdot\\!$chord ('if MiniBooNE is this model, what SBND sees')",
             fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = os.path.join(OUT, "SBND_analytic_prediction.png")
fig.savefig(out, dpi=140); plt.close(fig)
print("Saved -> %s" % out)
print("  (muon-only, at MiniBooNE best-fit coupling A=%s)" % BESTFIT_A)
for key, label, fname, col in PORTALS:
    E, w = load_portal(fname); w = w * BESTFIT_A[key]
    print("  %-12s total = %.3e events   <E_vis> = %.0f MeV   (benchmark x A=%.3f)"
          % (key, w.sum(), (E*w).sum()/w.sum(), BESTFIT_A[key]))
