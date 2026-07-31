r"""Standalone log-x plot of the delay relative to a prompt beta=1 particle.

This is panel 3 of the sim figure, pulled out on its own so it gets the whole
canvas and a log x-axis instead of sharing a linear one with three other
panels.  Both legs of the chain are shown against the same axis:

  * nu_mu -> N4   (upscatter_delay_ns)   -- the prompt, SM-like leg
  * N4 -> nu gamma (hnl_decay_delay_ns)  -- the signal photon, later by the
    slow-HNL flight

Why log: the delay piles up against a hard geometric floor (every event pays
the same beta=1 minimum) and then trails out over more than a decade, so a
linear axis spends most of its width on an empty tail.  A dotted line marks
the observed floor -- the shortest delay any event in the file achieved.

Run:
  /home/shubham/siren_pr178_venv/bin/python timing_delay_log.py \
      output/icarus_bnb_darknews_hnl_timing.csv
Out: <csv stem>_delay_log.png (override with --out).
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

# Okabe-Ito, colorblind-safe -- same palette as plot_dirt_timing.py.
C_PRIMARY = "#0072B2"   # blue     : nu_mu -> N4 (upscatter)
C_SECOND = "#D55E00"    # vermilion: N4 -> nu gamma (decay)

SERIES = [
    ("upscatter_delay_ns", r"$\nu_\mu \to N_4$  (prompt)", C_PRIMARY),
    ("hnl_decay_delay_ns", r"$N_4 \to \nu\gamma$  (signal)", C_SECOND),
]


def _weighted_median(v, w):
    order = np.argsort(v)
    v, w = v[order], w[order]
    return v[np.searchsorted(np.cumsum(w), 0.5 * w.sum())]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="timing CSV from the sim")
    ap.add_argument("--out", help="output PNG (default: <csv stem>_delay_log.png)")
    ap.add_argument("--bins", type=int, default=70)
    ap.add_argument("--logy", action="store_true",
                    help="log y as well, to bring out the sparse tail")
    ap.add_argument("--title", help="figure title (default: from the filename)")
    args = ap.parse_args(argv)

    d = np.genfromtxt(args.csv, delimiter=",", names=True)
    w = d["weight"]
    stem = os.path.splitext(args.csv)[0]
    out = args.out or stem + "_delay_log.png"
    title = args.title or os.path.basename(stem).replace("_", " ")

    allv = np.concatenate([d[c] for c, _, _ in SERIES])
    floor = allv[allv > 0].min()
    bins = np.logspace(np.log10(floor * 0.95), np.log10(allv.max() * 1.1),
                       args.bins)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for col, label, color in SERIES:
        v = d[col]
        med = _weighted_median(v, w)
        ax.hist(v, bins=bins, weights=w, density=True, histtype="step",
                lw=1.6, color=color, label="%s   median %.2f ns" % (label, med))
        ax.axvline(med, color=color, ls="--", lw=1.0, alpha=0.7)

    ax.axvline(floor, color="0.35", ls=":", lw=1.2,
               label=r"$\beta=1$ floor  %.2f ns" % floor)

    ax.set_xscale("log")
    if args.logy:
        ax.set_yscale("log")
    ax.set_xlabel(r"delay relative to prompt $\beta=1$ [ns]")
    ax.set_ylabel("weighted density")
    ax.set_title("%s  (N=%d)\ndelay relative to prompt, log axis"
                 % (title, len(w)), fontsize=12)
    ax.grid(alpha=0.25, lw=0.6, which="both")
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("Wrote %s" % os.path.abspath(out))


if __name__ == "__main__":
    main()
