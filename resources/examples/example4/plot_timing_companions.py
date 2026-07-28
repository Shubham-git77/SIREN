#!/usr/bin/env python
"""
Two companion timing figures from sbnd.py's output/SBND_timing.{csv,parquet}:

  1) delay_hist.png    -- distribution of the N4 flight-time delay
                          (decay_time - upscatter_time), log-y, showing the
                          boosted-N4 decay tail.
  2) timing_stages.png -- production_time (t0), upscatter_time, decay_time
                          overlaid, showing the beam spill getting shifted by
                          the neutrino ToF and then smeared by the N4 delay.

Run:
  /home/shubham/siren_ubaid_venv/bin/python plot_timing_companions.py \
      [--in output/SBND_timing.parquet] [--outdir output]
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# dataviz reference palette (light surface): fixed categorical order + ink tokens
BLUE, AQUA, YELLOW = "#2a78d6", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e5e4e0"


def style(ax):
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c4")
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def load(path):
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq
        t = pq.read_table(path)
        return {c: np.asarray(t[c]) for c in t.column_names}
    d = np.genfromtxt(path, delimiter=",", names=True)
    return {c: d[c] for c in d.dtype.names}


def plot_delay_hist(delay, outpath):
    p99 = float(np.percentile(delay, 99))
    n_above = int((delay > p99).sum())
    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=150)
    style(ax)
    bins = np.linspace(0, p99, 70)
    ax.hist(delay, bins=bins, color=BLUE, edgecolor="#1c5cab", lw=0.4, zorder=2)
    ax.set_yscale("log")
    med, mean = np.median(delay), delay.mean()
    for x, lab, c in ((med, "median %.0f ns" % med, INK),
                      (mean, "mean %.0f ns" % mean, INK2)):
        ax.axvline(x, color=c, ls="--", lw=1.3, zorder=3)
        ax.text(x, ax.get_ylim()[1] * 0.6, " " + lab, color=c, fontsize=8.5,
                rotation=90, va="top", ha="left")
    ax.set_xlim(0, p99)
    ax.set_xlabel("N4 flight-time delay:  decay $-$ upscatter  [ns]",
                  fontsize=11, color=INK)
    ax.set_ylabel("events per bin  (log scale)", fontsize=11, color=INK)
    ax.set_title("N4 flight-time delay — SBND dipole HNL", fontsize=12, color=INK)
    ax.text(0.985, 0.93,
            "N = %d\n%d events (%.1f%%) beyond %.0f ns\n(tail to %.0f ns)"
            % (len(delay), n_above, 100.0 * n_above / len(delay), p99, delay.max()),
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5, color=INK2,
            bbox=dict(boxstyle="round", fc="white", ec="0.75", alpha=0.9))
    fig.tight_layout()
    fig.savefig(outpath)
    print("wrote %s" % outpath)


def plot_timing_stages(prod, upsc, decay, outpath):
    hi = float(np.percentile(decay, 99))
    bins = np.linspace(0, hi, 64)
    series = [("production time  $t_0$", prod, BLUE, "#1c5cab"),
              ("upscatter time  ($t_0$ + $\\nu$ ToF)", upsc, AQUA, "#199e70"),
              ("N4 decay time  (+ N4 ToF)", decay, YELLOW, "#c98500")]
    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=150)
    style(ax)
    for lab, v, fc, ec in series:
        ax.hist(v, bins=bins, histtype="stepfilled", facecolor=fc, alpha=0.16, zorder=2)
        ax.hist(v, bins=bins, histtype="step", edgecolor=ec, lw=2.0, zorder=3, label=lab)
    ax.set_xlim(0, hi)
    ax.set_xlabel("lab time  [ns]", fontsize=11, color=INK)
    ax.set_ylabel("events per bin", fontsize=11, color=INK)
    ax.set_title("Timing chain across the beam spill — SBND dipole HNL",
                 fontsize=12, color=INK)
    leg = ax.legend(loc="upper right", framealpha=0.9, fontsize=9.5)
    for t in leg.get_texts():
        t.set_color(INK)
    ax.text(0.015, 0.965,
            "$t_0$: synthetic 0–1600 ns spill\n"
            "mean shifts:  +%.0f ns ($\\nu$ ToF),  +%.0f ns (N4 ToF)"
            % ((upsc - prod).mean(), (decay - upsc).mean()),
            transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color=INK2)
    fig.tight_layout()
    fig.savefig(outpath)
    print("wrote %s" % outpath)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="output/SBND_timing.parquet")
    ap.add_argument("--outdir", default="output")
    args = ap.parse_args()
    if not os.path.exists(args.inp):
        args.inp = "output/SBND_timing.csv"
    d = load(args.inp)
    prod = np.asarray(d["production_time"], float)
    upsc = np.asarray(d["upscatter_time"], float)
    decay = np.asarray(d["decay_time"], float)
    delay = decay - upsc
    ok = np.isfinite(delay) & (delay >= 0)
    os.makedirs(args.outdir, exist_ok=True)
    plot_delay_hist(delay[ok], os.path.join(args.outdir, "delay_hist.png"))
    plot_timing_stages(prod[ok], upsc[ok], decay[ok],
                       os.path.join(args.outdir, "timing_stages.png"))


if __name__ == "__main__":
    main()
