"""
MiniBooNE meson-portal prediction, ALL FOUR channels (K_e, K_mu, pi_e, pi_mu),
for all three portals (scalar | pseudoscalar | vector), 3-panel E_gamma style
with per-channel curves + TOTAL. No MiniBooNE-excess overlay.

Lepton-universal benchmark couplings (g_e = g_mu, Table II) -- the raw prediction.
NB the electron channels are physical here but NOT the paper's muon-only scenario;
for the pseudoscalar pi_e is helicity-UNSUPPRESSED and dominant. Log y-axis so all
four channels are visible across their wide range.

Validated analytic sigma*N*chord estimator (ray-sphere, MiniBooNE oil, carbon);
vector uses the full cascade e+e- visible energy. NOT the directed SIREN sampler.
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from siren import _util

HERE = os.path.dirname(__file__)
MB = _util.load_module("mb3", os.path.join(HERE, "make_miniboone_muon_3portal.py"))
sp_channel, vec_channel = MB.sp_channel, MB.vec_channel

ALL = ["K_e", "K_mu", "pi_e", "pi_mu"]
COL = {"K_e": "C0", "K_mu": "C1", "pi_e": "C2", "pi_mu": "C3"}
NDEC = 300

def main():
    portals = [("Scalar $\\phi\\to\\gamma$",        lambda nm: sp_channel(MB.S_sc, nm, n_dec=NDEC)),
               ("Pseudoscalar $a\\to\\gamma$",      lambda nm: sp_channel(MB.S_ps, nm, n_dec=NDEC)),
               ("Vector $\\to e^+e^-$",             lambda nm: vec_channel(nm, n_dec=NDEC))]
    Ebins = np.linspace(0, 2000, 41); EC = 0.5 * (Ebins[:-1] + Ebins[1:])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for j, (label, fn) in enumerate(portals):
        tot = np.zeros(len(EC)); grand = 0.0
        for nm in ALL:
            E, w = fn(nm)
            if not len(E):
                print("  %-22s %-6s : (none)" % (label, nm)); continue
            h = np.histogram(E * 1e3, bins=Ebins, weights=w)[0]
            ax[j].step(EC, h, where="mid", color=COL[nm], lw=1.4, label=nm)
            tot += h; grand += w.sum()
            print("  %-22s %-6s = %.3e ev" % (label, nm, w.sum()))
        ax[j].step(EC, tot, where="mid", color="k", lw=2.2, label="TOTAL")
        ax[j].set_yscale("log")
        ax[j].set_xlabel(r"$E_\gamma$ [MeV]"); ax[j].set_ylabel("Counts (benchmark)")
        # low log floor for the tiny vector panel so its ~0.1-3 ev/bin channels
        # are visible; normal floor for the large scalar/pseudo panels.
        floor = 0.02 if tot.max() < 100 else 1.0
        ax[j].set_xlim(0, 2000); ax[j].set_ylim(bottom=floor)
        ax[j].legend(fontsize=8)
        ax[j].set_title("%s — total %.2e ev" % (label, grand), fontsize=9.5)
    fig.suptitle("MiniBooNE meson-portal, ALL channels ($\\mu$ and $e$), lepton-universal benchmark couplings "
                 "— analytic $\\sigma\\!\\cdot\\!N\\!\\cdot\\!$chord", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(HERE, "output", "MiniBooNE_all4_3portal.png")
    fig.savefig(out, dpi=115); plt.close(fig); print("Saved -> %s" % out)

if __name__ == "__main__":
    main()
