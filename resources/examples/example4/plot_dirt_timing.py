#!/usr/bin/env python
"""
Plots for the DIRT-INDUCED dipole-portal HNL run (sbnd_dirt.py).

Reads output/SBND_dirt_timing.parquet (or .csv) and makes two figures that
tell the story "upscatter OUTSIDE the detector -> long-lived N4 propagates in
-> decay INSIDE SBND":

  output/sbnd_dirt_geometry.png
    (a) longitudinal z of the upscatter vertex (in dirt) vs the decay vertex
        (in SBND, signal), on one axis, with the SBND active-volume band shaded
        -- the two populations sit on opposite sides of the detector face.
    (b) N4 flight distance dirt -> decay (the propagation length), signal only.
    (c) top-view (x vs z) scatter: upscatter points (dirt) and decay points
        (SBND), with the active-volume box outlined -- shows the N4 flying in.

  output/sbnd_dirt_timing.png
    (a) the three-leg timing chain: production t0, upscatter time, decay time.
    (b) N4 time-of-flight (upscatter -> decay), signal only.
    (c) N4 velocity beta = |p|/E, signal only.

"Signal" = in_detector == 1 (decay vertex inside a TPC). N4s whose scattered
direction misses SBND decay in the dirt and are NOT signal; they are shown only
where the full population is relevant (panel a of the geometry figure).

Colorblind-safe categorical colors (Okabe-Ito), assigned in fixed order.

Run:  /home/shubham/siren_ubaid_venv/bin/python plot_dirt_timing.py
"""
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
PARQUET = os.path.join(HERE, "output", "SBND_dirt_timing.parquet")
CSV = os.path.join(HERE, "output", "SBND_dirt_timing.csv")

# Okabe-Ito colorblind-safe palette, fixed order.
OI = {
    "blue":      "#0072B2",
    "orange":    "#E69F00",
    "green":     "#009E73",
    "vermillion":"#D55E00",
    "purple":    "#CC79A7",
    "sky":       "#56B4E9",
    "gray":      "#666666",
}

# SBND active volume extent in the detector frame (from sbnd_dirt.py geometry).
ACTIVE_Z = (-2.92, 2.09)     # z span [m]
ACTIVE_X = (-2.01, 2.01)     # x span [m] (both TPCs)


# ---------------------------------------------------------------------
# Load the per-event table (parquet preferred, csv fallback)
# ---------------------------------------------------------------------
def load():
    cols = ["production_time", "upscatter_time", "nu_tof", "decay_time", "n4_tof",
            "nu_energy", "nu_flight_dist", "n4_flight_dist", "n4_beta",
            "ux", "uy", "uz", "dx", "dy", "dz", "in_detector"]
    if os.path.exists(PARQUET):
        import pyarrow.parquet as pq
        t = pq.read_table(PARQUET)
        d = {c: np.asarray(t[c]) for c in t.column_names}
        src = PARQUET
    elif os.path.exists(CSV):
        arr = np.genfromtxt(CSV, delimiter=",", names=True)
        d = {c: np.asarray(arr[c]) for c in arr.dtype.names}
        src = CSV
    else:
        raise SystemExit("No output/SBND_dirt_timing.{parquet,csv}; run sbnd_dirt.py first.")
    return d, src


def main():
    d, src = load()
    n = len(d["decay_time"])
    sig = d["in_detector"] > 0.5
    ns = int(sig.sum())
    print("loaded %d events from %s  (signal / decay-in-SBND: %d = %.1f%%)"
          % (n, os.path.basename(src), ns, 100.0 * ns / max(n, 1)))
    if ns == 0:
        raise SystemExit("No signal (decay-in-detector) events to plot.")

    # =================================================================
    # Figure 1: geometry / propagation
    # =================================================================
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    # (a) longitudinal z: upscatter (dirt, all) vs decay (SBND, signal)
    zmin = np.floor(min(d["uz"].min(), d["dz"][sig].min()))
    bins_z = np.linspace(zmin, 5.0, 60)
    ax[0].hist(d["uz"], bins=bins_z, histtype="step", lw=2.0, color=OI["orange"],
               label="upscatter vertex (dirt)")
    ax[0].hist(d["dz"][sig], bins=bins_z, histtype="step", lw=2.0, color=OI["blue"],
               label="decay vertex (SBND)")
    ax[0].axvspan(ACTIVE_Z[0], ACTIVE_Z[1], color=OI["gray"], alpha=0.18, lw=0)
    ax[0].text(ACTIVE_Z[1] + 0.5, ax[0].get_ylim()[1] * 0.9, "SBND\nactive",
               color=OI["gray"], fontsize=9, va="top")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("beam-axis position z [m]  (detector frame; beam +z)")
    ax[0].set_ylabel("events / bin")
    ax[0].set_title("Upscatter OUTSIDE, decay INSIDE")
    ax[0].legend(loc="upper left", fontsize=9)

    # (b) N4 flight distance dirt -> decay (signal)
    fl = d["n4_flight_dist"][sig]
    ax[1].hist(fl, bins=np.linspace(0.0, np.percentile(fl, 99.5), 50),
               color=OI["green"], alpha=0.85)
    ax[1].axvline(fl.mean(), color=OI["gray"], ls="--", lw=1.4)
    ax[1].text(fl.mean(), ax[1].get_ylim()[1] * 0.96, "  mean %.1f m" % fl.mean(),
               color=OI["gray"], fontsize=9, va="top")
    ax[1].set_xlabel("N4 flight distance, dirt $\\to$ decay [m]")
    ax[1].set_ylabel("signal events / bin")
    ax[1].set_title("Long-lived N4 propagation length")

    # (c) top view x-z scatter (subsample) showing the N4 flying into SBND
    k = min(1500, ns)
    idx = np.random.default_rng(0).choice(np.flatnonzero(sig), size=k, replace=False)
    ax[2].scatter(d["uz"][idx], d["ux"][idx], s=6, color=OI["orange"], alpha=0.5,
                  label="upscatter (dirt)")
    ax[2].scatter(d["dz"][idx], d["dx"][idx], s=6, color=OI["blue"], alpha=0.6,
                  label="decay (SBND)")
    ax[2].add_patch(Rectangle((ACTIVE_Z[0], ACTIVE_X[0]),
                              ACTIVE_Z[1] - ACTIVE_Z[0], ACTIVE_X[1] - ACTIVE_X[0],
                              fill=False, ec=OI["gray"], lw=1.6))
    ax[2].set_xlabel("z [m]  (beam axis)")
    ax[2].set_ylabel("x [m]")
    ax[2].set_title("Top view: N4 flight into the detector")
    ax[2].legend(loc="upper left", fontsize=9)

    fig.suptitle("Dirt-induced dipole-portal HNL at SBND  "
                 "(%d events, %d decay in SBND)" % (n, ns), fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out1 = os.path.join(HERE, "output", "sbnd_dirt_geometry.png")
    fig.savefig(out1, dpi=130)
    print("wrote", out1)

    # =================================================================
    # Figure 2: timing / kinematics
    # =================================================================
    fig2, ax2 = plt.subplots(1, 3, figsize=(16, 4.6))

    # (a) three-leg timing chain
    t0 = d["production_time"][sig]
    tup = d["upscatter_time"][sig]
    tdec = d["decay_time"][sig]
    lo = np.floor(min(t0.min(), tup.min(), tdec.min()))
    hi = np.ceil(max(t0.max(), tup.max(), tdec.max()))
    tb = np.linspace(lo, hi, 60)
    ax2[0].hist(t0, bins=tb, histtype="step", lw=2.0, color=OI["gray"],
                label="production $t_0$")
    ax2[0].hist(tup, bins=tb, histtype="step", lw=2.0, color=OI["orange"],
                label="upscatter (dirt)")
    ax2[0].hist(tdec, bins=tb, histtype="step", lw=2.0, color=OI["blue"],
                label="decay (SBND)")
    ax2[0].set_xlabel("time [ns]")
    ax2[0].set_ylabel("signal events / bin")
    ax2[0].set_title("Timing chain: $t_0$ $\\to$ upscatter $\\to$ decay")
    ax2[0].legend(loc="upper right", fontsize=9)

    # (b) N4 time of flight
    tof = d["n4_tof"][sig]
    ax2[1].hist(tof, bins=np.linspace(0.0, np.percentile(tof, 99.5), 50),
                color=OI["purple"], alpha=0.85)
    ax2[1].axvline(tof.mean(), color=OI["gray"], ls="--", lw=1.4)
    ax2[1].text(tof.mean(), ax2[1].get_ylim()[1] * 0.96, "  mean %.1f ns" % tof.mean(),
                color=OI["gray"], fontsize=9, va="top")
    ax2[1].set_xlabel("N4 time of flight, dirt $\\to$ decay [ns]")
    ax2[1].set_ylabel("signal events / bin")
    ax2[1].set_title("N4 time of flight")

    # (c) N4 velocity beta
    beta = d["n4_beta"][sig]
    ax2[2].hist(beta, bins=np.linspace(max(0.0, beta.min() - 0.02), 1.0, 50),
                color=OI["sky"], alpha=0.9)
    ax2[2].axvline(beta.mean(), color=OI["gray"], ls="--", lw=1.4)
    ax2[2].text(beta.mean(), ax2[2].get_ylim()[1] * 0.96,
                "mean %.3f  " % beta.mean(), color=OI["gray"], fontsize=9,
                va="top", ha="right")
    ax2[2].set_xlabel(r"N4 velocity $\beta = |p|/E$")
    ax2[2].set_ylabel("signal events / bin")
    ax2[2].set_title("N4 velocity")

    fig2.suptitle("Dirt-induced dipole-portal HNL at SBND: timing & kinematics",
                  fontsize=13)
    fig2.tight_layout(rect=(0, 0, 1, 0.96))
    out2 = os.path.join(HERE, "output", "sbnd_dirt_timing.png")
    fig2.savefig(out2, dpi=130)
    print("wrote", out2)


if __name__ == "__main__":
    main()
