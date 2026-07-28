#!/usr/bin/env python
"""
2D histogram of the N4 flight-time delay (decay_time - upscatter_time) vs the
neutrino energy, from sbnd.py's output/SBND_timing.{csv,parquet}.

Physics: the delay is the N4 lab-frame time-of-flight from the in-detector
upscattering vertex to its decay, = beta*gamma*c*tau / (beta c) ~ gamma*tau.
So a MORE energetic (more boosted) N4 is MORE time-dilated and decays LATER --
the delay INCREASES with energy. The (E, delay) plane is where a slow, delayed
HNL signal separates from prompt, beam-coincident background.

Run:
  /home/shubham/siren_ubaid_venv/bin/python plot_delay_vs_energy.py \
      [--in output/SBND_timing.csv] [--out output/delay_vs_energy.png]
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe


def load(path):
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        t = pq.read_table(path)
        return {c: np.asarray(t[c]) for c in t.column_names}
    d = np.genfromtxt(path, delimiter=",", names=True)
    return {c: d[c] for c in d.dtype.names}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="output/SBND_timing.csv")
    ap.add_argument("--out", default="output/delay_vs_energy.png")
    ap.add_argument("--m4", default="0.14")
    ap.add_argument("--mu", default="1e-6")
    ap.add_argument("--ymax", type=float, default=None,
                    help="y (delay) axis cap [ns]; default = 99th percentile")
    args = ap.parse_args()

    d = load(args.inp)
    E = np.asarray(d["nu_energy"], float)
    delay = np.asarray(d["decay_time"], float) - np.asarray(d["upscatter_time"], float)
    ok = np.isfinite(E) & np.isfinite(delay) & (delay >= 0)
    E, delay = E[ok], delay[ok]
    n = len(E)

    # Cap the y-axis so the dense bulk stays legible; the boosted-N4 tail runs
    # far past it, so note how many events land above the cap.
    ymax = args.ymax if args.ymax is not None else float(np.percentile(delay, 99))
    n_above = int((delay > ymax).sum())

    # Denser hex grid when there are many events; coarse when sparse (proof run).
    gs = (44, 34) if n > 5000 else (26, 22)

    fig, ax = plt.subplots(figsize=(7.6, 5.4), dpi=150)

    # 2D histogram on a hex grid; log color scale copes with the long-tailed
    # density (a few cells hold most events, many hold one).
    hb = ax.hexbin(E, delay, gridsize=gs, cmap="viridis", extent=(E.min(), E.max(), 0, ymax),
                   bins="log", mincnt=1, linewidths=0.2, edgecolors="none")
    cb = fig.colorbar(hb, ax=ax, pad=0.015)
    cb.set_label("events per cell (log scale)", fontsize=10)

    # Overlay the trend: median delay in energy quantile bins. White line with a
    # dark stroke so it reads over any part of the viridis field.
    qedges = np.quantile(E, np.linspace(0, 1, 9))
    xs, ys, lo, hi = [], [], [], []
    for a, b in zip(qedges[:-1], qedges[1:]):
        m = (E >= a) & (E < b) if b < qedges[-1] else (E >= a) & (E <= b)
        if m.sum() < 3:
            continue
        xs.append(0.5 * (a + b))
        ys.append(np.median(delay[m]))
        lo.append(np.percentile(delay[m], 25))
        hi.append(np.percentile(delay[m], 75))
    xs, ys, lo, hi = map(np.array, (xs, ys, lo, hi))
    stroke = [pe.Stroke(linewidth=3.4, foreground="0.15"), pe.Normal()]
    ax.fill_between(xs, lo, hi, color="white", alpha=0.12, zorder=3)
    ax.plot(xs, ys, "-o", color="white", lw=2.0, ms=6, zorder=4,
            path_effects=stroke, markeredgecolor="0.15", markeredgewidth=0.8,
            label="median delay (IQR band)")

    ax.set_ylim(0, ymax)
    ax.set_xlabel("Neutrino energy  [GeV]", fontsize=11)
    ax.set_ylabel("N4 flight-time delay:  decay $-$ upscatter  [ns]", fontsize=11)
    ax.set_title("N4 flight-time delay vs neutrino energy — SBND dipole HNL",
                 fontsize=12)
    ax.legend(loc="upper left", framealpha=0.85, fontsize=9)
    ax.margins(x=0.01)

    r = np.corrcoef(E, delay)[0, 1]
    tail = ("\n%d events (%.1f%%) above %.0f ns (tail to %.0f ns)"
            % (n_above, 100.0 * n_above / n, ymax, delay.max())) if n_above else ""
    ax.text(0.985, 0.03,
            "N = %d\n$m_4$ = %s GeV,  $\\mu_{tr}$ = %s GeV$^{-1}$\n"
            "corr(E, delay) = %+.2f  (boosted N4 decays later)%s"
            % (n, args.m4, args.mu, r, tail),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5,
            bbox=dict(boxstyle="round", fc="white", ec="0.6", alpha=0.85))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out)
    print("wrote %s  (N=%d, corr=%.3f)" % (args.out, n, r))


if __name__ == "__main__":
    main()
