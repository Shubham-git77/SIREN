r"""Re-plot the timing figure with EVENT COUNTS on y instead of density.

Same four panels, same binning (50 bins per series over its own range) and
therefore the same x axes as the sim's own figure -- only the y quantity
changes.  The sim normalizes each histogram to unit area, which hides how many
events actually sit in a bin; this shows the raw population instead, so sparse
bins look sparse.

Two senses of "events", pick with --weighted:
  * default   -- raw Monte Carlo event counts.  Shows the SAMPLING: how many
    simulated events back each bin, i.e. where the plot is statistically thin.
  * --weighted -- summed weights.  Shows the PHYSICAL expectation shape, the
    same information the density plot carries but unnormalized.

They differ a lot here: the weights span ~1e11, so a bin with many MC events
can carry almost no weight and vice versa.  Neither is an absolute rate -- see
the timing-study notes on normalization.

Run:
  /home/shubham/siren_pr178_venv/bin/python timing_counts.py \
      output/icarus_bnb_darknews_hnl_timing_3k.csv
Out: <csv stem>_counts.png (override with --out).
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

# Okabe-Ito, colorblind-safe -- same palette as the other post-processors here.
C_PRIMARY = "#0072B2"   # blue     : nu_mu -> N4 (upscatter)
C_SECOND = "#D55E00"    # vermilion: N4 -> nu gamma (decay)

# Panels, in the sim figure's order: (column(s), xlabel).
PANELS = [
    ([("parent_decay_ns", None, C_PRIMARY)],
     "pion decay time after proton [ns]"),
    ([("upscatter_ns", r"$\nu_\mu \to N_4$", C_PRIMARY),
      ("hnl_decay_ns", r"$N_4 \to \nu\gamma$", C_SECOND)],
     "absolute vertex time after proton [ns]"),
    ([("upscatter_delay_ns", r"$\nu_\mu \to N_4$", C_PRIMARY),
      ("hnl_decay_delay_ns", r"$N_4 \to \nu\gamma$", C_SECOND)],
     r"delay relative to prompt $\beta=1$ [ns]"),
    ([("hnl_flight_ns", None, C_PRIMARY)],
     r"$N_4$ decay time after upscatter [ns]"),
]


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="timing CSV from the sim")
    ap.add_argument("--out", help="output PNG (default: <csv stem>_counts.png)")
    ap.add_argument("--weighted", action="store_true",
                    help="sum weights per bin instead of counting MC events")
    ap.add_argument("--logy", action="store_true",
                    help="log y, to keep sparse bins visible next to full ones")
    ap.add_argument("--bins", type=int, default=50,
                    help="bins per series (default 50, matching the sim)")
    ap.add_argument("--title", help="figure title (default: from the filename)")
    args = ap.parse_args(argv)

    d = np.genfromtxt(args.csv, delimiter=",", names=True)
    w = d["weight"]
    stem = os.path.splitext(args.csv)[0]
    out = args.out or stem + "_counts.png"
    title = args.title or os.path.basename(stem).replace("_", " ")
    ylab = "weighted events" if args.weighted else "events"

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)

    for ax, (series, xlabel) in zip(axes.flat, PANELS):
        for col, label, color in series:
            v = d[col]
            mask = np.isfinite(v) & np.isfinite(w) & (w >= 0.0)
            ax.hist(v[mask], bins=args.bins,
                    weights=w[mask] if args.weighted else None,
                    density=False, histtype="step", lw=1.8,
                    color=color, label=label)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylab)
        if args.logy:
            ax.set_yscale("log")
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
        if any(lb for _, lb, _ in series):
            ax.legend()

    fig.suptitle("%s  (N=%d) -- %s on y, x axes as the sim figure"
                 % (title, len(w), ylab))
    fig.savefig(out, dpi=160)
    print("Wrote %s" % os.path.abspath(out))


if __name__ == "__main__":
    main()
