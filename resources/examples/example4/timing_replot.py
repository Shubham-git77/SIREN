r"""Re-plot the DarkNews HNL timing CSVs with per-panel axis scaling.

Post-processor for the *_bnb_darknews_hnl_timing.csv files.  The sim's own
figure autoscales every panel to the full data range, so at high statistics a
handful of near-zero-weight tail events stretch the axes and squash the bulk
into one or two bins (at 50k ICARUS: 5 events out of 50000, carrying 3e-6 of
the weight, push the N4-decay axis from ~70 ns out to 1858 ns).

Nothing is thrown away -- each panel picks the scale that suits its own data
shape, and any event outside a clipped view is counted in the panel title
along with the weight it carries, so the reader can see what was set aside.

  * pion decay time      -- linear.  Bounded 0-200 ns, no tail problem.
  * absolute vertex time -- linear, CLIPPED to percentiles.  The structure is a
    narrow band at large x (ICARUS 1962-2053 ns); a log axis would compress it
    to ~0.02 decades and destroy it.  Clipping is the only thing that works.
  * delay vs prompt      -- log.  Spans 1.3-94 ns, bulk piled at the low end:
    the textbook case for a log axis, and all values are strictly positive.
  * N4 decay time        -- symlog.  Spans 0-1858 ns but CONTAINS EXACT ZEROS,
    so a pure log axis would drop them; symlog is linear below `linthresh`.

Run:
  /home/shubham/siren_pr178_venv/bin/python timing_replot.py \
      output/icarus_bnb_darknews_hnl_timing.csv
Out: <csv stem>_replot.png (override with --out).
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
C_PRIMARY = "#0072B2"   # blue    : nu_mu -> N4 (upscatter)
C_SECOND = "#D55E00"    # vermilion: N4 -> nu gamma (decay)


def _weighted_hist(ax, values, weights, bins, color, label):
    ax.hist(values, bins=bins, weights=weights, density=True,
            histtype="step", lw=1.4, color=color, label=label)


def _clip_note(values, weights, lo, hi):
    """How many events (and how much weight) fall outside a clipped view."""
    out = (values < lo) | (values > hi)
    n = int(out.sum())
    if n == 0:
        return ""
    frac = weights[out].sum() / weights.sum()
    return "  [%d evt outside, %.1e of weight]" % (n, frac)


def _log_bins(lo, hi, n=60):
    return np.logspace(np.log10(lo), np.log10(hi), n)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="timing CSV from the sim")
    ap.add_argument("--out", help="output PNG (default: <csv stem>_replot.png)")
    ap.add_argument("--clip", type=float, default=99.9,
                    help="upper percentile for the clipped vertex-time panel "
                         "(default: 99.9); the lower edge uses 100-clip")
    ap.add_argument("--title", help="figure title (default: from the filename)")
    args = ap.parse_args(argv)

    d = np.genfromtxt(args.csv, delimiter=",", names=True)
    w = d["weight"]
    stem = os.path.splitext(args.csv)[0]
    out = args.out or stem + "_replot.png"
    title = args.title or os.path.basename(stem).replace("_", " ")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("%s  (N=%d, per-panel axis scaling)" % (title, len(w)),
                 fontsize=13)

    # -- 1. pion decay time: linear, the distribution is naturally bounded ----
    ax = axes[0, 0]
    v = d["parent_decay_ns"]
    _weighted_hist(ax, v, w, np.linspace(0, v.max(), 60), C_PRIMARY, None)
    ax.set_xlabel("pion decay time after proton [ns]")
    ax.set_title("linear -- bounded, no tail", fontsize=10)

    # -- 2. absolute vertex time: linear but CLIPPED (log would destroy it) ---
    ax = axes[0, 1]
    up, dec = d["upscatter_ns"], d["hnl_decay_ns"]
    both = np.concatenate([up, dec])
    lo = np.percentile(both, 100.0 - args.clip)
    hi = np.percentile(both, args.clip)
    pad = 0.02 * (hi - lo)
    bins = np.linspace(lo - pad, hi + pad, 70)
    _weighted_hist(ax, up, w, bins, C_PRIMARY, r"$\nu_\mu \to N_4$")
    _weighted_hist(ax, dec, w, bins, C_SECOND, r"$N_4 \to \nu\gamma$")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_xlabel("absolute vertex time after proton [ns]")
    ax.set_title("linear, clipped to %.4g-%.4g%%%s" % (
        100.0 - args.clip, args.clip,
        _clip_note(both, np.concatenate([w, w]), lo, hi)), fontsize=10)
    ax.legend(frameon=False, fontsize=9)

    # -- 3. delay vs prompt: log, bulk piled at the low end -------------------
    ax = axes[1, 0]
    ud, hd = d["upscatter_delay_ns"], d["hnl_decay_delay_ns"]
    both = np.concatenate([ud, hd])
    pos = both[both > 0]
    bins = _log_bins(pos.min() * 0.9, both.max() * 1.1)
    _weighted_hist(ax, ud, w, bins, C_PRIMARY, r"$\nu_\mu \to N_4$")
    _weighted_hist(ax, hd, w, bins, C_SECOND, r"$N_4 \to \nu\gamma$")
    ax.set_xscale("log")
    ax.set_xlabel(r"delay relative to prompt $\beta=1$ [ns]")
    ax.set_title("log -- spans decades, all values > 0", fontsize=10)
    ax.legend(frameon=False, fontsize=9)

    # -- 4. N4 decay time: symlog, the data contains exact zeros -------------
    ax = axes[1, 1]
    v = d["hnl_flight_ns"]
    linthresh = 1.0
    lin = np.linspace(0, linthresh, 10)
    log = _log_bins(linthresh, v.max() * 1.1, 50)
    _weighted_hist(ax, v, w, np.concatenate([lin, log[1:]]), C_PRIMARY, None)
    ax.set_xscale("symlog", linthresh=linthresh)
    ax.set_xlabel(r"$N_4$ decay time after upscatter [ns]")
    ax.set_title("symlog (linear below %g ns) -- data contains zeros"
                 % linthresh, fontsize=10)

    for ax in axes.ravel():
        ax.set_ylabel("weighted density")
        ax.grid(alpha=0.25, lw=0.6)
        ax.set_axisbelow(True)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=130)
    print("Wrote %s" % os.path.abspath(out))


if __name__ == "__main__":
    main()
