r"""2D vertex-position vs time maps for the BNB DarkNews HNL timing runs.

Post-processor for the *_bnb_darknews_hnl_timing.csv files produced by
DarkNewsHNL_SBND_BNB_timing.py / DarkNewsHNL_ICARUS_BNB_timing.py.

The CSVs store times only, but the vertex position along the line of sight is
recoverable exactly: collect_timing() defines

    delay = t_vertex - |r_vertex - target| / c

so the distance of each vertex from the beam target is

    L = c * (t_vertex - delay)

for both the nu -> N4 upscatter and the N4 -> nu gamma decay.  L is a radial
distance from the target, not a cartesian z, but the detectors are far
downstream and small in transverse extent, so L tracks depth along the beam to
well under the bin width.

Each panel is a weight-summed 2D histogram of L (x) against vertex time (y):
  --time-axis absolute : time after the proton on target [ns].  The dashed
      line is the beta=1 light cone from the target, t = L/c; every vertex sits
      above it by the pion decay time plus any slow-HNL lateness.
  --time-axis delay    : t - L/c, the residual after removing the light cone.
      This is the axis where the HNL lateness is visible (the upscatter row
      collapses to the prompt-neutrino band, the decay row develops the tail).

Colour is the fraction of the total event weight per bin (log scale), so the
maps are shapes, not rates -- same caveat as the rest of the timing study.

Run (uses the two CSVs already in output/):
  /home/shubham/siren_pr178_venv/bin/python timing_position_2d.py
  /home/shubham/siren_pr178_venv/bin/python timing_position_2d.py --time-axis delay \
      --output output/timing_position2d_delay_sbnd_icarus.png
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

C_M_PER_NS = 0.299792458

DETECTORS = [
    ("SBND", "output/sbnd_bnb_darknews_hnl_timing.csv"),
    ("ICARUS", "output/icarus_bnb_darknews_hnl_timing.csv"),
]

# (time column, delay column, panel label)
VERTICES = [
    ("upscatter_ns", "upscatter_delay_ns", r"$\nu_\mu\,\mathrm{Ar}\to N_4$ upscatter"),
    ("hnl_decay_ns", "hnl_decay_delay_ns", r"$N_4\to\nu\gamma$ decay (detected $\gamma$)"),
]


def _load_vertex(csv_path, time_col, delay_col):
    """Return (distance_from_target_m, time_ns, delay_ns, weight) for one vertex."""
    d = np.genfromtxt(csv_path, delimiter=",", names=True)
    w = np.asarray(d["weight"], dtype=float)
    t = np.asarray(d[time_col], dtype=float)
    dly = np.asarray(d[delay_col], dtype=float)
    L = C_M_PER_NS * (t - dly)
    m = np.isfinite(L) & np.isfinite(t) & np.isfinite(dly) & np.isfinite(w) & (w > 0)
    return L[m], t[m], dly[m], w[m]


def _wquantile(x, w, q):
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = np.cumsum(w) / w.sum()
    return float(x[np.searchsorted(c, q, side="left").clip(0, x.size - 1)])


def _range(x, w, qlo, qhi, pad_frac=0.04, min_pad=0.05):
    """Weighted-quantile axis range, padded, robust to zero-weight outliers."""
    lo, hi = _wquantile(x, w, qlo), _wquantile(x, w, qhi)
    if hi <= lo:
        lo, hi = lo - min_pad, hi + min_pad
    pad = max(pad_frac * (hi - lo), min_pad)
    return lo - pad, hi + pad


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--time-axis", choices=("absolute", "delay"), default="absolute",
                    help="y axis: time after proton, or residual t - L/c")
    ap.add_argument("--bins", type=int, default=140,
                    help="bins per axis (default 140)")
    ap.add_argument("--qlo", type=float, default=0.0005,
                    help="lower weighted quantile for the axis ranges")
    ap.add_argument("--qhi", type=float, default=0.9995,
                    help="upper weighted quantile for the axis ranges")
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--output", default=None,
                    help="default: output/timing_position2d[_delay]_sbnd_icarus.png")
    args = ap.parse_args(argv)

    output = args.output
    if output is None:
        suffix = "" if args.time_axis == "absolute" else "_delay"
        output = "output/timing_position2d%s_sbnd_icarus.png" % suffix

    fig, axes = plt.subplots(len(DETECTORS), len(VERTICES),
                             figsize=(13.0, 9.0), constrained_layout=True)
    axes = np.atleast_2d(axes)

    print("vertex position (from target) vs %s time"
          % ("absolute" if args.time_axis == "absolute" else "light-cone-subtracted"))
    print("-" * 82)

    for row, (name, csv) in enumerate(DETECTORS):
        for col, (time_col, delay_col, vlabel) in enumerate(VERTICES):
            ax = axes[row, col]
            if not os.path.exists(csv):
                ax.text(0.5, 0.5, "missing %s" % csv, ha="center", va="center",
                        transform=ax.transAxes)
                continue

            L, t, dly, w = _load_vertex(csv, time_col, delay_col)
            y = t if args.time_axis == "absolute" else dly

            xr = _range(L, w, args.qlo, args.qhi)
            yr = _range(y, w, args.qlo, args.qhi)
            if args.time_axis == "delay":
                # keep the beta=1 reference (delay 0) on screen: the gap between
                # it and the lowest delay is the pion decay time, not zero.
                yr = (min(yr[0], -0.15), yr[1])
            bx = np.linspace(xr[0], xr[1], args.bins + 1)
            by = np.linspace(yr[0], yr[1], args.bins + 1)

            H, _, _ = np.histogram2d(L, y, bins=[bx, by], weights=w)
            H = H / w.sum()                       # fraction of total weight per bin
            H = np.ma.masked_where(H <= 0.0, H)
            vmax = float(H.max())
            mesh = ax.pcolormesh(bx, by, H.T, cmap=args.cmap, shading="flat",
                                 norm=LogNorm(vmin=max(vmax * 1e-6, H.min()),
                                              vmax=vmax))
            cb = fig.colorbar(mesh, ax=ax, pad=0.01)
            cb.set_label("fraction of total weight / bin", fontsize=8)
            cb.ax.tick_params(labelsize=7)

            # beta = 1 reference from the target
            ref = bx / C_M_PER_NS if args.time_axis == "absolute" else np.zeros_like(bx)
            ax.plot(bx, ref, ls="--", lw=1.1, color="w", alpha=0.85)
            ax.plot(bx, ref, ls="--", lw=0.6, color="k", alpha=0.55,
                    label=r"$\beta=1$ from target")

            ax.set_xlim(*xr)
            ax.set_ylim(*yr)
            ax.set_xlabel("vertex distance from BNB target [m]")
            ax.set_ylabel("time after proton on target [ns]" if args.time_axis == "absolute"
                          else r"delay $t - L/c$ [ns]")
            ax.set_title("%s -- %s" % (name, vlabel), fontsize=10)
            ax.legend(fontsize=7, loc="upper left", framealpha=0.75)
            ax.grid(alpha=0.18, color="w", lw=0.4)

            Lmed = _wquantile(L, w, 0.5)
            ymed = _wquantile(y, w, 0.5)
            print("%-7s %-26s | L med=%8.2f m  span[%s]=%7.2f..%7.2f m  "
                  "%s med=%9.3f ns  span=%8.3f..%8.3f ns"
                  % (name, time_col, Lmed,
                     "%.1f%%" % (100 * (args.qhi - args.qlo)),
                     _wquantile(L, w, args.qlo), _wquantile(L, w, args.qhi),
                     "t" if args.time_axis == "absolute" else "dt",
                     ymed, _wquantile(y, w, args.qlo), _wquantile(y, w, args.qhi)))
    print("-" * 82)

    fig.suptitle("BNB DarkNews HNL (dipole, $m_4=0.1$ GeV): vertex position along the "
                 "beam vs %s time -- 50k events/detector, weight-summed shapes"
                 % ("absolute" if args.time_axis == "absolute" else "light-cone-subtracted"),
                 fontsize=12)
    out = os.path.abspath(output)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print("Wrote %s" % out)


if __name__ == "__main__":
    main()
