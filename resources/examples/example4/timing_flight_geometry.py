r"""N4 production/decay geometry along the beam for the BNB HNL timing runs.

Post-processor for the *_bnb_darknews_hnl_timing.csv files.  This is the
in-argon analogue of the usual dirt-sample three-panel figure; see the CAVEATS
below for what differs.

Both quantities are reconstructed from the stored times, not read from
coordinates (the CSVs carry no vertex positions):

  * position along the line of sight, from the delay definition in
    collect_timing(),   L = c * (t_vertex - delay)
    exact, but a RADIUS from the beam target rather than a cartesian z.  It is
    plotted as an offset s = L - L_center about the middle of the upscatter
    support, so the active volume brackets zero as in a detector-frame z plot.
  * N4 propagation length,   d = beta * c * (t_decay - t_upscatter)
    with beta = sqrt(1 - (m4/E_N4)^2) from the stored hnl_energy_GeV.  Exact
    for straight-line flight, which is what the injector generates.

CAVEATS -- this is not a dirt sample:
  * _darknews_bundle() is built with nuclear_targets=["Ar40"], so upscatter
    happens only in liquid argon.  Both vertices are inside the TPC; there is
    no upstream dirt population.  A real dirt study needs upstream nuclei in
    the target list AND an injection volume covering the dirt, then a new run.
  * The transverse coordinate is not recoverable, so the third panel shows
    production vs decay position along the beam instead of a top view.
  * Colour/height is summed weight -- shapes, not rates.

Run:
  /home/shubham/siren_pr178_venv/bin/python timing_flight_geometry.py
Out: output/timing_flight_geometry_sbnd_icarus.png
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

# Fiducial z widths mirrored from DarkNewsHNL_SBN_dk2nu_timing.py::_FIDUCIALS,
# used only to shade the active region.
DETECTORS = [
    ("SBND", "output/sbnd_bnb_darknews_hnl_timing.csv", 5.01, "#1f77b4"),
    ("ICARUS", "output/icarus_bnb_darknews_hnl_timing.csv", 17.95, "#d62728"),
]

UPS_COLOR = "#e8a33d"


def _load(csv_path, m4):
    d = np.genfromtxt(csv_path, delimiter=",", names=True)
    w = np.asarray(d["weight"], dtype=float)
    Lu = C_M_PER_NS * (d["upscatter_ns"] - d["upscatter_delay_ns"])
    Ld = C_M_PER_NS * (d["hnl_decay_ns"] - d["hnl_decay_delay_ns"])
    E = np.asarray(d["hnl_energy_GeV"], dtype=float)
    beta = np.sqrt(np.clip(1.0 - (m4 / E) ** 2, 0.0, 1.0))
    flight = beta * C_M_PER_NS * np.asarray(d["hnl_flight_ns"], dtype=float)
    m = (np.isfinite(Lu) & np.isfinite(Ld) & np.isfinite(flight)
         & np.isfinite(w) & (w > 0))
    return Lu[m], Ld[m], flight[m], E[m], beta[m], w[m]


def _wquantile(x, w, q):
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = np.cumsum(w) / w.sum()
    return float(x[np.searchsorted(c, q, side="left").clip(0, x.size - 1)])


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m4", type=float, default=0.10,
                    help="HNL mass in GeV used for beta (must match the run)")
    ap.add_argument("--qhi", type=float, default=0.995,
                    help="upper weighted quantile for the flight-length axis")
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--output",
                    default="output/timing_flight_geometry_sbnd_icarus.png")
    ap.add_argument("--sbnd-csv", help="override the SBND input CSV")
    ap.add_argument("--icarus-csv", help="override the ICARUS input CSV")
    args = ap.parse_args(argv)

    # Defaults are the 50k CSVs; the overrides let a lower-statistics run
    # (e.g. the *_3k / *_1500 reruns) be plotted without touching them.
    overrides = {"SBND": args.sbnd_csv, "ICARUS": args.icarus_csv}
    detectors = [(name, overrides.get(name) or csv, zwidth, color)
                 for name, csv, zwidth, color in DETECTORS]

    fig, axes = plt.subplots(len(detectors), 3, figsize=(16.5, 8.6),
                             constrained_layout=True)
    axes = np.atleast_2d(axes)

    print("m4 = %.3f GeV;  s = L - L_center (radius from target, not z)" % args.m4)
    print("-" * 92)

    nloaded = []
    for row, (name, csv, zwidth, color) in enumerate(detectors):
        if not os.path.exists(csv):
            for c in range(3):
                axes[row, c].text(0.5, 0.5, "missing %s" % csv, ha="center",
                                  va="center", transform=axes[row, c].transAxes)
            continue

        Lu, Ld, flight, E, beta, w = _load(csv, args.m4)
        nloaded.append((name, len(w)))
        # Anchor s = 0 at the middle of the upscatter support: upscatter is
        # confined to the argon, so this is the active volume's centre.
        lo, hi = _wquantile(Lu, w, 0.001), _wquantile(Lu, w, 0.999)
        center = 0.5 * (lo + hi)
        su, sd = Lu - center, Ld - center
        # Shade the observed upscatter support, which IS the argon along the
        # line of sight.  It runs slightly wider than the geometric z half-width
        # because s is a radius: transverse extent adds to the projection.
        half = 0.5 * (hi - lo)

        # ---- left: production and decay position along the beam ----
        axL = axes[row, 0]
        smax = max(_wquantile(sd, w, args.qhi), half * 1.6)
        bins = np.linspace(-half * 1.6, smax, args.bins)
        axL.hist(su, bins=bins, weights=w, histtype="step", lw=1.9,
                 color=UPS_COLOR, label="upscatter vertex (LAr)")
        axL.hist(sd, bins=bins, weights=w, histtype="step", lw=1.9,
                 color=color, label="decay vertex")
        axL.axvspan(-half, half, color="0.6", alpha=0.28, lw=0)
        axL.text(half, 0.94, " argon", transform=axL.get_xaxis_transform(),
                 fontsize=8, color="0.35", va="top")
        axL.set_yscale("log")
        axL.set_xlabel("beam-axis position $s$ [m]  (0 = active-volume centre)")
        axL.set_ylabel("summed weight / bin")
        axL.set_title("%s: upscatter and decay INSIDE argon" % name, fontsize=10)
        axL.legend(fontsize=8)
        axL.grid(alpha=0.25)

        # ---- middle: N4 propagation length ----
        axM = axes[row, 1]
        fmax = _wquantile(flight, w, args.qhi)
        fb = np.linspace(0.0, fmax, args.bins)
        axM.hist(flight, bins=fb, weights=w, color="#2ca089", lw=0)
        fmean = float(np.average(flight, weights=w))
        fmed = _wquantile(flight, w, 0.5)
        axM.axvline(fmean, ls="--", lw=1.3, color="0.35")
        axM.text(fmean, 0.95, " mean %.2f m" % fmean, color="0.35", fontsize=8,
                 transform=axM.get_xaxis_transform(), va="top")
        axM.set_xlabel(r"$N_4$ flight distance, upscatter $\to$ decay [m]")
        axM.set_ylabel("summed weight / bin")
        axM.set_title(r"%s: $N_4$ propagation length (med %.2f m)"
                      % (name, fmed), fontsize=10)
        axM.grid(alpha=0.25)

        # ---- right: production vs decay position (no transverse coord) ----
        axR = axes[row, 2]
        bx = np.linspace(-half * 1.15, half * 1.15, 90)
        by = np.linspace(-half * 1.15, smax, 90)
        H, _, _ = np.histogram2d(su, sd, bins=[bx, by], weights=w)
        H = np.ma.masked_where(H <= 0, H / w.sum())
        mesh = axR.pcolormesh(bx, by, H.T, cmap="viridis", shading="flat",
                              norm=LogNorm(vmin=max(H.max() * 1e-6, H.min()),
                                           vmax=H.max()))
        cb = fig.colorbar(mesh, ax=axR, pad=0.01)
        cb.set_label("fraction of total weight / bin", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        axR.plot(bx, bx, ls="--", lw=1.0, color="w", alpha=0.8)
        axR.axhline(half, ls=":", lw=1.2, color="w", alpha=0.8)
        axR.text(bx[0], half, " downstream edge", color="w", fontsize=7.5,
                 va="bottom")
        axR.set_xlabel("upscatter position $s$ [m]")
        axR.set_ylabel("decay position $s$ [m]")
        axR.set_title("%s: production vs decay along beam" % name, fontsize=10)
        axR.grid(alpha=0.18, color="w", lw=0.4)

        esc = float(w[sd > half].sum() / w.sum())
        print("%-7s | active support %+.2f..%+.2f m (geom width %.2f m) | "
              "flight med %.2f mean %.2f m | beta med %.4f | decay past "
              "downstream edge %.1f%% of weight"
              % (name, lo - center, hi - center, zwidth, fmed, fmean,
                 float(np.median(beta)), 100 * esc))
    print("-" * 92)

    # Report the event counts actually loaded rather than a hardcoded figure --
    # the same script now serves the 50k runs and the lower-statistics reruns.
    counts = ", ".join("%s %d" % (name, n) for name, n in nloaded)
    fig.suptitle(r"BNB DarkNews HNL (dipole, $m_4=%.2f$ GeV): in-argon production "
                 "and decay geometry -- %s events, weight-summed shapes"
                 % (args.m4, counts), fontsize=12)
    out = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print("Wrote %s" % out)


if __name__ == "__main__":
    main()
