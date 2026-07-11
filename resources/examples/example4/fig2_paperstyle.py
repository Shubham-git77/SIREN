"""
Dutta-Kim Fig.2-style REBUILD for all three portals (MiniBooNE nu-mode E_vis).
Draws the MiniBooNE background (tan) with OUR signal (red) STACKED on top,
fit-normalized to the excess, and the MiniBooNE data points (black) overlaid --
the way the paper presents Fig.2.

Panels:
  * Scalar  (phi Dark Primakoff, single photon)  -> paper Fig.2 BOTTOM
  * Pseudoscalar (a Dark Primakoff, single photon) -> Fig.3
  * Vector  (double-mediator DM -> e+e-)           -> paper Fig.2 TOP

Background + data points are DIGITIZED from the paper's Fig.2 (scalar, nu-mode
E_vis; arXiv:2110.11944) and reused for all three -- a SHAPE + fit-normalized
comparison, not an independent MiniBooNE data fit.

*** REBUILT 2026-07-11 on the current authoritative engine (sbnd_analytic) ***
All three panels now come from the same estimator that produces the anchored
yields, so the benchmark in-window counts match the final results:
  * scalar/pseudo : analytic_sp_mb, muon-only (g_e=0), MiniBooNE single-photon eff
  * vector        : analytic_vec_mb, all channels, MiniBooNE electron-like (nu_e)
                    eff (the collimated e+e- reconstructs as a single e-like ring)
MINIBOONE_POT = 18.75e20 (the 320-excess exposure) for all three.
"""
import os, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from siren import _util
import sbnd_analytic as SA

HERE = os.path.dirname(os.path.abspath(__file__))
def load(name): return _util.load_module(name, os.path.join(HERE, name+".py"))
S_sc  = load("ScalarPortal_MiniBooNE_multichannel")
S_ps  = load("PseudoscalarPortal_MiniBooNE_multichannel")
S_vec = load("VectorPortal_MiniBooNE_fullchain")

# ---- Digitized MiniBooNE nu-mode E_vis (Dutta-Kim Fig.2 bottom-left) ----
DATA_E   = np.array([225,275,325,375,425,475,525,575,625,675,725,775,825,875,925,975,1025,1075,1125],float)
DATA_N   = np.array([302,402,333,279,189,168,134,118, 81, 83, 75, 84, 57, 61, 38, 52, 26, 19, 19],float)
DATA_ERR = np.array([ 36, 41, 38, 35, 28, 27, 25, 23, 18, 19, 18, 19, 15, 18, 13, 15, 11, 12, 10],float)
BKG      = np.array([255,320,300,250,175,150,120,105, 76, 78, 70, 76, 52, 55, 35, 47, 24, 18, 17],float)
PBINS = np.concatenate([[DATA_E[0]-25], DATA_E+25])   # 200,250,...,1150 MeV
WIN_LO, WIN_HI = 0.140, 0.300                          # GeV signal window


def mb_signal(S, vector=False, muon_only=True, n_dec=400):
    """MiniBooNE E_vis signal from the CURRENT engine. Returns
    (hist over the data bins [events], exact in-window[0.14,0.30] benchmark sum).
    scalar/pseudo: single-photon eff, muon-only; vector: electron-like eff, all channels."""
    fn = SA.analytic_vec_mb if vector else SA.analytic_sp_mb
    chans = list(S.CHANNELS) if (vector or not muon_only) else [c for c in S.CHANNELS if "mu" in c]
    h = np.zeros(len(DATA_E)); inwin = 0.0
    for nm in chans:
        E, w = fn(S, nm, n_dec=n_dec, eff_mode="mb")
        E = np.asarray(E); w = np.asarray(w)
        if E.size == 0:
            continue
        h += np.histogram(E * 1e3, bins=PBINS, weights=w)[0]
        inwin += w[(E >= WIN_LO) & (E <= WIN_HI)].sum()
    return h, inwin


def fit_and_panel(ax, sig, bench_in, title, norm_kind="coupling", color="#c0504d"):
    """sig = absolute benchmark signal in the data bins. Fit overall amplitude A
    to the excess (data - bkg). bench_in = exact in-window[0.14,0.30] benchmark
    events (for the annotation).  For BOTH the single-photon Dark Primakoff
    (N_S ~ (g_mu g_n lambda)^2) and the vector (measured N ~ P^2, P=eps1 eps2
    g'^2/4pi), the coupling combination scales as sqrt(A)."""
    excess = DATA_N - BKG
    wgt = 1.0 / DATA_ERR**2
    denom = np.sum(wgt * sig * sig)
    A = max(np.sum(wgt * sig * excess) / denom, 0.0) if denom > 0 else 0.0
    sig_fit = A * sig
    chi2 = np.sum(((DATA_N - (BKG + sig_fit))**2) * wgt)
    edges = PBINS
    ax.stairs(BKG + sig_fit, edges, fill=True, color=color, alpha=0.9, zorder=1, label="our signal (fit)")
    ax.stairs(BKG, edges, fill=True, color="#cdab7e", zorder=2, label="MiniBooNE bkg")
    ax.stairs(BKG + sig_fit, edges, fill=False, color="0.4", lw=0.8, zorder=3)
    ax.errorbar(DATA_E, DATA_N, yerr=DATA_ERR, fmt="o", color="k", ms=3.5,
                capsize=2, lw=1, zorder=5, label="MiniBooNE data")
    ax.set_xlabel(r"$E_{vis}$ [MeV]"); ax.set_ylabel("Counts")
    ax.set_xlim(DATA_E[0] - 25, 1250); ax.set_ylim(0, 480)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(title, fontsize=10)
    if norm_kind == "coupling":
        txt = ("A = %.3f $\\times$ bench\n$g$-product %.2f $\\times$ bench\n"
               "bench in-win = %.0f ev" % (A, math.sqrt(A), bench_in))
    else:  # vector: N ~ P^2 so product ~ sqrt(A); e-like eff applied
        txt = ("A = %.2f $\\times$ bench\n(double-med.; product\n%.2f $\\times$ bench, $N\\!\\propto\\!P^2$)\n"
               "bench in-win = %.0f ev" % (A, math.sqrt(A), bench_in))
    ax.text(0.96, 0.70, txt, transform=ax.transAxes, ha="right", va="top",
            fontsize=8, bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    return A, chi2


def main():
    print("Scalar (muon-only, single-photon eff) ...")
    hsc, bsc = mb_signal(S_sc)
    print("Pseudoscalar (muon-only, single-photon eff) ...")
    hps, bps = mb_signal(S_ps)
    print("Vector (all channels, electron-like eff) ...")
    hvec, bvec = mb_signal(S_vec, vector=True)

    panels = [("Scalar ($\\phi$ Dark Primakoff)",              hsc,  bsc,  "coupling"),
              ("Pseudoscalar ($a$ Dark Primakoff)",            hps,  bps,  "coupling"),
              ("Vector (double-mediator $\\to e^+e^-$)",        hvec, bvec, "rate")]

    fig, axes = plt.subplots(1, len(panels), figsize=(6.0 * len(panels), 5.0), squeeze=False)
    for ax, (title, sig, bin_, nk) in zip(axes[0], panels):
        A, chi2 = fit_and_panel(ax, sig, bin_, title, norm_kind=nk)
        print("  %-40s A=%.4f  bench-in-win=%.0f  chi2=%.1f" % (title, A, bin_, chi2))
    fig.suptitle(r"MiniBooNE $\nu$-mode, Fig.2 style — our signal fit-normalized to the excess "
                 r"(bkg+data digitized from Dutta-Kim Fig.2)", fontsize=11)
    os.makedirs(os.path.join(HERE, "output"), exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(HERE, "output", "MiniBooNE_fig2_paperstyle.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print("Saved -> %s" % out)


if __name__ == "__main__":
    main()
