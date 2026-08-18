"""
Dutta-Kim Fig.2-style rebuild for all three portals (MiniBooNE nu-mode E_vis).
MiniBooNE background (tan) + OUR signal (red) stacked, fit-normalized to the
excess, with MiniBooNE data points (black) overlaid -- the way the paper's Fig.2
presents it.

Background + data points are DIGITIZED from the paper's Fig.2 (scalar nu-mode
E_vis; arXiv:2110.11944).  The black points are the REAL MiniBooNE data; the tan
is MiniBooNE's background prediction.

RE-RUN 2026-07-24 on the current authoritative engine (AnalyticRate). Config
paths point at the example4 dir; engine loaded from the installed package copy.

Run:  DK2NU_FILE=/home/shubham/nubeamHighSample.dk2nu.root \
      /home/shubham/siren_venv/bin/python fig2_paperstyle.py
"""
import os, math, importlib.util
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("DK2NU_FILE", "/home/shubham/nubeamHighSample.dk2nu.root")
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
PKG  = os.environ.get(
    "SIREN_DNT_DIR",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "processes", "DarkNewsTables",
    ),
)

def load(path, name):
    import sys
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m

SA    = load(os.path.join(PKG, "AnalyticRate.py"), "AnalyticRate")
S_sc  = load(os.path.join(HERE, "ScalarPortal_MiniBooNE_multichannel.py"),      "S_sc")
S_ps  = load(os.path.join(HERE, "PseudoscalarPortal_MiniBooNE_multichannel.py"), "S_ps")
S_vec = load(os.path.join(HERE, "VectorPortal_MiniBooNE_fullchain.py"),          "S_vec")

# ---- Digitized MiniBooNE nu-mode E_vis (Dutta-Kim Fig.2 bottom-left) ----
# MiniBooNE nu-mode data now comes from the shared module. It used to be a
# 19-bin digitization inlined here, which matched no official release and
# silently diverged from the corrected 11-bin HEPData binning used by
# scan_brute_grid.py / mcmc_fit.py -- results from the two were not
# comparable. ERR_MODE=quad adds the MiniBooNE background systematics the
# paper says it used; ERR_MODE=stat (default) keeps the historical
# stat-only weighting.
import os as _os
import miniboone_data as MB
DATA_E, DATA_N, DATA_ERR, BKG, EBINS = MB.DATA_E, MB.DATA_N, MB.DATA_ERR, MB.BKG, MB.EBINS
ERR_MODE = _os.environ.get("ERR_MODE", "stat")
EXCESS = MB.EXCESS
INV2 = 1.0 / MB.errors(ERR_MODE) ** 2
PBINS = np.concatenate([[DATA_E[0]-25], DATA_E+25])   # 200,250,...,1150 MeV
WIN_LO, WIN_HI = 0.140, 0.300                          # GeV signal window


def mb_signal(S, vector=False, muon_only=True, n_dec=400):
    fn = SA.analytic_vec_mb if vector else SA.analytic_sp_mb
    meson_fn = SA._mesons_dk2nu if os.environ.get("FLUX", "dk2nu") == "dk2nu" else None
    chans = list(S.CHANNELS) if (vector or not muon_only) else [c for c in S.CHANNELS if "mu" in c]
    h = np.zeros(len(DATA_E)); inwin = 0.0
    for nm in chans:
        E, w = fn(S, nm, n_dec=n_dec, eff_mode="mb", meson_fn=meson_fn)
        E = np.asarray(E); w = np.asarray(w)
        if E.size == 0:
            continue
        h += np.histogram(E * 1e3, bins=PBINS, weights=w)[0]
        inwin += w[(E >= WIN_LO) & (E <= WIN_HI)].sum()
    return h, inwin


def fit_and_panel(ax, sig, bench_in, title, norm_kind="coupling", color="#c0504d"):
    excess = DATA_N - BKG
    wgt = 1.0 / DATA_ERR**2
    denom = np.sum(wgt * sig * sig)
    A = max(np.sum(wgt * sig * excess) / denom, 0.0) if denom > 0 else 0.0
    sig_fit = A * sig
    chi2 = np.sum(((DATA_N - (BKG + sig_fit))**2) * wgt)
    chi2_null = np.sum((excess**2) * wgt)
    edges = PBINS
    ax.stairs(BKG + sig_fit, edges, fill=True, color=color, alpha=0.9, zorder=1, label="our signal (fit)")
    ax.stairs(BKG, edges, fill=True, color="#cdab7e", zorder=2, label="MiniBooNE bkg")
    ax.stairs(BKG + sig_fit, edges, fill=False, color="0.4", lw=0.8, zorder=3)
    ax.errorbar(DATA_E, DATA_N, yerr=DATA_ERR, fmt="o", color="k", ms=3.5,
                capsize=2, lw=1, zorder=5, label="MiniBooNE data")
    ax.set_xlabel(r"$E_{vis}$ [MeV]"); ax.set_ylabel("Counts")
    ax.set_xlim(DATA_E[0] - 25, 1250); ax.set_ylim(0, 480)
    ax.legend(frameon=False, fontsize=8); ax.set_title(title, fontsize=10)
    if norm_kind == "coupling":
        txt = ("A = %.3f $\\times$ bench\n$g$-product %.2f $\\times$ bench\n"
               "bench in-win = %.0f ev\n$\\chi^2$=%.1f (null %.1f), 18 dof" % (A, math.sqrt(A), bench_in, chi2, chi2_null))
    else:
        txt = ("A = %.2f $\\times$ bench\n(double-med.; product\n%.2f $\\times$ bench, $N\\!\\propto\\!P^2$)\n"
               "bench in-win = %.0f ev\n$\\chi^2$=%.1f (null %.1f)" % (A, math.sqrt(A), bench_in, chi2, chi2_null))
    ax.text(0.96, 0.70, txt, transform=ax.transAxes, ha="right", va="top",
            fontsize=8, bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    return A, chi2


def main():
    print("Scalar ..."); hsc, bsc = mb_signal(S_sc)
    print("Pseudoscalar ..."); hps, bps = mb_signal(S_ps)
    print("Vector ..."); hvec, bvec = mb_signal(S_vec, vector=True)
    panels = [("Scalar ($\\phi$ Dark Primakoff)",       hsc,  bsc,  "coupling"),
              ("Pseudoscalar ($a$ Dark Primakoff)",     hps,  bps,  "coupling"),
              ("Vector (double-mediator $\\to e^+e^-$)", hvec, bvec, "rate")]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.0 * len(panels), 5.0), squeeze=False)
    for ax, (title, sig, bin_, nk) in zip(axes[0], panels):
        A, chi2 = fit_and_panel(ax, sig, bin_, title, norm_kind=nk)
        print("  %-40s A=%.4f  bench-in-win=%.0f  chi2=%.1f" % (title, A, bin_, chi2))
    fig.suptitle(r"MiniBooNE $\nu$-mode, Fig.2 style -- our signal fit-normalized to the excess "
                 r"(bkg+data digitized from Dutta-Kim Fig.2)  [re-run 2026-07-24]", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(HERE, "output", "MiniBooNE_fig2_paperstyle.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print("Saved -> %s" % out)


if __name__ == "__main__":
    main()
