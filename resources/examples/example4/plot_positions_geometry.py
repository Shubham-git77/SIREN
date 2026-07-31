r"""Dirt-style three-panel geometry figure for the BNB in-argon HNL runs.

Reads the position CSVs written by DarkNewsHNL_SBN_BNB_positions.py and makes
the same figure as plot_dirt_timing.py's geometry panel in the SIREN_ubaid
tree, so the in-argon runs can be compared with the dirt-induced ones directly:

  (a) beam-axis z of the upscatter vertex vs the decay vertex, with the
      fiducial band shaded.
  (b) N4 propagation length, upscatter -> decay.
  (c) top view (x vs z): upscatter and decay points with the fiducial box
      outlined.

These use TRUE COORDINATES from the CSV (ux/uy/uz, dx/dy/dz), not a radius
reconstructed from timing, so panel (c) is a real top view -- the thing
timing_flight_geometry.py cannot produce.

HOW THIS DIFFERS FROM THE DIRT FIGURE: the model is Ar40-only, so upscatter
happens inside the liquid argon rather than tens of metres upstream in rock.
The two populations therefore OVERLAP inside the detector instead of sitting on
opposite sides of its face, and the propagation length is metres rather than
tens of metres.  That is physics, not a plotting difference.

Weighting: histograms are weight-summed by default (the physical shape).  Pass
--unweighted for raw MC counts, which shows the sampling instead -- the two
differ a lot here because the weights span many orders of magnitude.

Run:
  /home/shubham/siren_pr178_venv/bin/python plot_positions_geometry.py \
      output/sbnd_bnb_hnl_positions.csv
Out: <csv stem>_geometry.png
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

# Okabe-Ito, fixed order, matching plot_dirt_timing.py.
OI = {"orange": "#E69F00", "blue": "#0072B2", "vermilion": "#D55E00",
      "gray": "#999999"}

# Mirrored from DarkNewsHNL_SBN_dk2nu_timing.py::_FIDUCIALS (center, widths).
FIDUCIALS = {
    "SBND": ((0.0, 0.59, -0.415), (4.026, 4.074645, 5.01)),
    "ICARUS": ((0.0, 0.0, 0.0), (7.20, 3.16, 17.95)),
}


def _detector_from_path(path):
    name = os.path.basename(path).lower()
    for key in FIDUCIALS:
        if key.lower() in name:
            return key
    raise SystemExit("cannot tell the detector from %r; pass --detector" % path)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="position CSV from DarkNewsHNL_SBN_BNB_positions.py")
    ap.add_argument("--detector", choices=sorted(FIDUCIALS))
    ap.add_argument("--out", help="output PNG (default: <csv stem>_geometry.png)")
    ap.add_argument("--unweighted", action="store_true",
                    help="raw MC counts instead of summed weights")
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--max-points", type=int, default=4000,
                    help="subsample size for the top-view scatter")
    args = ap.parse_args(argv)

    detector = args.detector or _detector_from_path(args.csv)
    center, widths = FIDUCIALS[detector]
    half = 0.5 * np.asarray(widths)
    zlo, zhi = center[2] - half[2], center[2] + half[2]
    xlo, xhi = center[0] - half[0], center[0] + half[0]

    d = np.genfromtxt(args.csv, delimiter=",", names=True)
    w = None if args.unweighted else d["weight"]
    ylab = "events / bin" if args.unweighted else "summed weight / bin"
    sig = d["in_detector"] == 1
    out = args.out or os.path.splitext(args.csv)[0] + "_geometry.png"

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.4), constrained_layout=True)

    # (a) longitudinal z: upscatter vs decay, fiducial band shaded ------------
    lo = min(d["uz"].min(), d["dz"].min())
    hi = max(d["uz"].max(), d["dz"].max())
    bins_z = np.linspace(lo, hi, args.bins)
    ax[0].hist(d["uz"], bins=bins_z, weights=w, histtype="step", lw=2.0,
               color=OI["orange"], label="upscatter vertex (LAr)")
    ax[0].hist(d["dz"][sig], bins=bins_z,
               weights=None if w is None else w[sig],
               histtype="step", lw=2.0, color=OI["blue"],
               label="decay vertex (in fiducial)")
    ax[0].axvspan(zlo, zhi, color=OI["gray"], alpha=0.18, lw=0)
    ax[0].text(0.5 * (zlo + zhi), ax[0].get_ylim()[1], " fiducial",
               color="0.35", fontsize=9, va="top")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("beam-axis position z [m]  (detector frame; beam +z)")
    ax[0].set_ylabel(ylab)
    ax[0].set_title("%s: upscatter and decay INSIDE argon" % detector)
    ax[0].legend(fontsize=8)

    # (b) N4 propagation length ----------------------------------------------
    fl = d["n4_flight_dist"][sig]
    fw = None if w is None else w[sig]
    good = np.isfinite(fl)
    ax[1].hist(fl[good], bins=np.linspace(0.0, np.percentile(fl[good], 99.5), 50),
               weights=None if fw is None else fw[good],
               color="#2CA089", edgecolor="none")
    mean = (np.average(fl[good], weights=fw[good]) if fw is not None
            else float(np.mean(fl[good])))
    ax[1].axvline(mean, color="0.35", ls="--", lw=1.2)
    ax[1].text(mean, ax[1].get_ylim()[1] * 0.94, " mean %.2f m" % mean,
               color="0.35", fontsize=9)
    ax[1].set_xlabel(r"N4 flight distance, upscatter $\to$ decay [m]")
    ax[1].set_ylabel(ylab.replace("events", "signal events"))
    ax[1].set_title("N4 propagation length")

    # (c) top view x-z, with the fiducial box outlined ------------------------
    n = len(d["uz"])
    idx = (np.random.default_rng(0).choice(n, args.max_points, replace=False)
           if n > args.max_points else np.arange(n))
    ax[2].scatter(d["uz"][idx], d["ux"][idx], s=6, color=OI["orange"],
                  alpha=0.5, label="upscatter", linewidths=0)
    ax[2].scatter(d["dz"][idx], d["dx"][idx], s=6, color=OI["blue"],
                  alpha=0.6, label="decay", linewidths=0)
    ax[2].add_patch(Rectangle((zlo, xlo), zhi - zlo, xhi - xlo, fill=False,
                              edgecolor="0.35", lw=1.2, ls="--"))
    ax[2].set_xlabel("z [m]  (beam axis)")
    ax[2].set_ylabel("x [m]")
    ax[2].set_title("Top view: N4 flight inside the detector")
    ax[2].legend(fontsize=8, markerscale=2)

    fig.suptitle("BNB in-argon dipole-portal HNL at %s  "
                 "(%d events, %d decay in fiducial)"
                 % (detector, n, int(sig.sum())), fontsize=13)
    fig.savefig(out, dpi=140)
    print("Wrote %s" % os.path.abspath(out))


if __name__ == "__main__":
    main()
