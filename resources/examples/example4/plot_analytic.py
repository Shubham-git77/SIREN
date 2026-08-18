#!/usr/bin/env python3
"""Plot the analytic-engine results for any detector and portal.

All nine portal scripts now default to --engine analytic, the trustworthy rate
path: the SIREN directed sampler over-estimates 60-400x (see AnalyticRate.py).
Measured at MiniBooNE: the analytic engine predicts 534 muon-channel events in
200-3000 MeV at the paper's Table I scalar coupling, against a measured excess
of 534, where the sampler gave 5.2e4. The analytic path writes
<DET>_<portal>_analytic.npz and no figure; this makes the figures.

WHAT IS AND IS NOT HERE: the npz stores per-channel (E_vis, weight) only --
analytic_sp/analytic_vec were called without return_cos, so there is no angular
information to plot. Ask the engine for cos theta if the angular panels are
wanted; it cannot be recovered from these files.

Weights are absolute events at each detector's POT (from sbn_exposures), so the
y axis is counts, not a shape. NB the three portals are each at their OWN
Table I benchmark, so their normalisations are not comparable to each other.

    python plot_analytic.py --detector all --portal all
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sbn_exposures import SBND_POT, ICARUS_BNB_POT, MINIBOONE_POT
import miniboone_data as MB

HERE = os.path.dirname(os.path.abspath(__file__))
CHANNELS = ["K_e", "K_mu", "pi_e", "pi_mu"]
# Dutta-Kim's model is MUON-ONLY (g_e = 0). These configs run g_e = g_mu, which
# adds K_e and pi_e -- and pi->e nu phi is helicity-UNsuppressed, so pi_e is the
# largest single channel. Including them made the plotted TOTAL 2.47x the paper
# for scalar (1353 vs 547 in 200-1250 MeV) while the muon-only total is 525,
# i.e. 0.96 of the measured excess. Default to the paper-comparable subset and
# show the electron channels dashed so nothing is hidden.
MUON_CHANNELS = ["K_mu", "pi_mu"]
ELEC_CHANNELS = ["K_e", "pi_e"]
COLOURS = {"K_e": "tab:blue", "K_mu": "tab:orange",
           "pi_e": "tab:green", "pi_mu": "tab:red"}
LABEL = {"scalar": "scalar Dark Primakoff",
         "pseudo": "pseudoscalar Dark Primakoff",
         "vector": "vector portal (double mediator)"}
DETECTORS = {"MiniBooNE": MINIBOONE_POT, "SBND": SBND_POT, "ICARUS": ICARUS_BNB_POT}
# the coupling each config runs at, for reporting the rescaled equivalent
CONFIG_PRODUCT = {"scalar": 2.2e-8, "pseudo": 6.5e-7, "vector": 1.3e-7}
E_VIS_THRESHOLD = 0.140          # GeV, the selection threshold the scripts apply


def load(detector, portal, indir):
    path = os.path.join(indir, "%s_%s_analytic.npz" % (detector, portal))
    if not os.path.exists(path):
        return None, path
    d = np.load(path)
    out = {}
    for ch in CHANNELS:
        if "%s_E" % ch in d:
            out[ch] = (d["%s_E" % ch], d["%s_w" % ch])
    return out, path


def plot(detector, portal, data, outdir, pot, channels="muon", canonical=True,
         paper_bins=True):
    # Bin on MiniBooNE's OWN variable-width edges over the range Fig.2 shows
    # (200-1250 MeV). Uniform 50 MeV bins starting at 0 are not comparable with
    # the paper: they display the 140-200 MeV region Fig.2 omits and put a
    # different width under every point, which makes counts-per-bin look larger
    # than the paper's for the same total.
    edges = MB.EBINS[:10] if paper_bins else np.linspace(0.0, 2.0, 41)   # GeV
    show = MUON_CHANNELS if channels == "muon" else CHANNELS
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    total = np.zeros(len(edges) - 1)
    for ch in show:
        if ch not in data:
            continue
        E, w = data[ch]
        h, _ = np.histogram(E, bins=edges, weights=w)
        total += h
        ax.step(edges[:-1], h, where="post", lw=1.1, color=COLOURS[ch],
                label="%s  (%.3e)" % (ch, w.sum()))
    ax.step(edges[:-1], total, where="post", lw=2.2, color="k",
            label="TOTAL  (%.3e)" % total.sum())
    if channels == "muon":
        # not part of the paper's model; drawn dashed so the omission is visible
        for ch in ELEC_CHANNELS:
            if ch not in data:
                continue
            E, w = data[ch]
            h, _ = np.histogram(E, bins=edges, weights=w)
            ax.step(edges[:-1], h, where="post", lw=1.0, ls="--", alpha=0.55,
                    color=COLOURS[ch], label="%s  (%.3e, g_e=0 in paper)" % (ch, w.sum()))
    if paper_bins and detector == "MiniBooNE":
        ctr = 0.5 * (edges[:-1] + edges[1:])
        ax.errorbar(ctr, MB.EXCESS[:9], yerr=MB.DATA_ERR[:9], fmt="ko", ms=4,
                    lw=1.0, capsize=2, label="MiniBooNE excess (%.0f total)" % MB.EXCESS[:9].sum(),
                    zorder=5)
    ax.axvline(E_VIS_THRESHOLD, ls="--", lw=1.0, color="grey")
    ax.set_yscale("log")
    ax.set_xlabel(r"$E_{\rm vis}$ [GeV]")
    ax.set_ylabel("Events / bin at %.2e POT" % pot)
    sub = "muon channels only (paper: g_e=0)" if channels == "muon" else "all four channels (g_e=g_mu)"
    ax.set_title("%s %s\nanalytic engine (authoritative), BNB, %.2e POT\n%s"
                 % (detector, LABEL.get(portal, portal), pot, sub), fontsize=9)
    ax.legend(fontsize=7.5, title="channel (total events)", title_fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    # The canonical choice for a portal gets the plain name; only an explicit
    # override is marked, so there is never a "variant"-looking file that is
    # actually the correct one (the vector's all-channel plot IS the right one).
    suffix = "" if canonical else ("_muon" if channels == "muon" else "_allchan")
    suffix += "" if paper_bins else "_uniformbins"
    out = os.path.join(outdir, "%s_%s_analytic_countrate%s.png" % (detector, portal, suffix))
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out, total.sum()


def plot_paper_style(detector, portal, data, outdir, pot, channels, rescale=False):
    """Reproduce the Dutta-Kim Fig.2 nu-mode E_vis panel layout.

    Fig.2 stacks the MiniBooNE background (tan) with the model signal on top
    (red for phi/a Dark Primakoff, blue for chi upscattering) and overlays the
    raw data as black points with error bars, on a LINEAR axis. Plotting the
    signal alone on a log axis -- which the default view does -- is the right
    thing for seeing the model, but it cannot be compared with Fig.2 by eye.
    Only MiniBooNE has the background needed for this, so it is MiniBooNE-only.
    """
    # MB.EBINS is in GeV; Fig.2's axis is in MeV, so convert. (The label used to
    # say MeV while the numbers were GeV -- 0.2 where the paper shows 200.)
    # Fig.2 uses 50 MeV UNIFORM bins (verified by digitising its data points:
    # they sit at 124.7, 174.7, 224.7 ... 1225.3 MeV, and their total over
    # 200-1250 is 2844 against HEPData's 2870 -- the same dataset, 0.9% apart).
    # HEPData's own edges are 75-150 MeV wide, so plotting on them makes every
    # bar look ~2x taller than the paper's for identical physics.
    edges = np.arange(200.0, 1250.0 + 1e-9, 50.0)      # MeV, matching Fig.2
    # background is tabulated on HEPData's variable bins: spread it by density
    # (flat within each release bin) onto the 50 MeV grid.
    _he = MB.EBINS * 1e3
    _dens = MB.BKG / np.diff(_he)                      # counts per MeV
    bkg = np.array([_dens[np.searchsorted(_he, 0.5*(edges[i]+edges[i+1]), "right") - 1]
                    * (edges[i+1] - edges[i]) for i in range(len(edges) - 1)])
    _ddens = MB.DATA_N / np.diff(_he)
    dat = np.array([_ddens[np.searchsorted(_he, 0.5*(edges[i]+edges[i+1]), "right") - 1]
                    * (edges[i+1] - edges[i]) for i in range(len(edges) - 1)])
    ctr = 0.5 * (edges[:-1] + edges[1:])
    width = np.diff(edges)
    show = MUON_CHANNELS if channels == "muon" else CHANNELS
    sig = np.zeros(len(ctr))
    for ch in show:
        if ch in data:
            E, w = data[ch]
            sig += np.histogram(E * 1e3, bins=edges, weights=w)[0]   # E is GeV, edges MeV
    # Optionally rescale the signal to best-fit the measured excess and report the
    # coupling that implies. Rate ~ product^2, so the implied product is
    # P_config * sqrt(scale). This separates SHAPE (is the model the right shape?)
    # from NORMALISATION (is the paper's quoted coupling right?) -- for the
    # pseudoscalar at Table I the shape is fine and only the coupling is off 7x.
    scale = 1.0
    if rescale and sig.sum() > 0:
        scale = float((dat - bkg).sum() / sig.sum())
        sig = sig * scale
    colour = {"scalar": "#c0392b", "pseudo": "#c0392b", "vector": "#2e6da4"}[portal]
    label = {"scalar": r"$\phi$ Dark Primakoff", "pseudo": r"$a$ Dark Primakoff",
             "vector": r"$\chi$ Upscattering"}[portal]
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.bar(ctr, bkg, width=width, color="#d8c9a3", edgecolor="#b9a37a", linewidth=0.4,
           label="MiniBooNE Background", zorder=1)
    ax.bar(ctr, sig, width=width, bottom=bkg, color=colour, alpha=0.9,
           label=label, zorder=2)
    ax.errorbar(ctr, dat, yerr=np.sqrt(np.maximum(dat, 0)), fmt="ko", ms=3.0,
                lw=0.9, capsize=0, zorder=5)
    ax.set_xlabel(r"$E_{vis}$ [MeV]"); ax.set_ylabel("Counts")
    ax.set_xlim(edges[0], edges[-1]); ax.set_ylim(0, None)
    if rescale:
        ttl = ("%s %s  (Fig.2 style, signal RESCALED to the excess)\n"
               "signal %.0f on background %.0f; data %.0f   [coupling x %.3f = %.2e]"
               % (detector, portal, sig.sum(), bkg.sum(), dat.sum(),
                  np.sqrt(scale), CONFIG_PRODUCT.get(portal, np.nan) * np.sqrt(scale)))
    else:
        ttl = ("%s %s  (Fig.2 style, at the paper's Table I coupling)\n"
               "signal %.0f on background %.0f; data %.0f"
               % (detector, portal, sig.sum(), bkg.sum(), dat.sum()))
    ax.set_title(ttl, fontsize=8.5)
    ax.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    out = os.path.join(outdir, "%s_%s_fig2style%s.png"
                       % (detector, portal, "_fitted" if rescale else ""))
    fig.savefig(out, dpi=140); plt.close(fig)
    return out, sig.sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--detector", default="all",
                    choices=["MiniBooNE", "SBND", "ICARUS", "all"])
    ap.add_argument("--portal", default="all",
                    choices=["scalar", "pseudo", "vector", "all"])
    ap.add_argument("--rescale-to-excess", action="store_true",
                    help="with --fig2-style: rescale the signal to fit the measured "
                         "excess and report the coupling that implies")
    ap.add_argument("--fig2-style", action="store_true",
                    help="stacked background+signal with raw data points, linear axis, "
                         "matching Dutta-Kim Fig.2 (MiniBooNE only)")
    ap.add_argument("--uniform-bins", action="store_true",
                    help="use 50 MeV uniform bins from 0 instead of MiniBooNE's own edges")
    ap.add_argument("--channels", default="auto", choices=["auto", "muon", "all"],
                    help="auto (default): muon for (pseudo)scalar since the paper sets "
                         "g_e=0, all four for vector since kinetic mixing is "
                         "lepton-universal. Override with muon/all.")
    ap.add_argument("--input-dir", default=os.path.join(HERE, "output"))
    ap.add_argument("--output-dir", default=os.path.join(HERE, "output"))
    a = ap.parse_args()

    portals = ["scalar", "pseudo", "vector"] if a.portal == "all" else [a.portal]
    dets = list(DETECTORS) if a.detector == "all" else [a.detector]
    os.makedirs(a.output_dir, exist_ok=True)
    missing = []
    for det in dets:
        for p in portals:
            data, path = load(det, p, a.input_dir)
            if not data:
                missing.append(path)
                continue
            ch = a.channels
            if ch == "auto":
                # a kinetically mixed dark photon couples eps*e to every charged
                # lepton, so the vector MUST include K_e/pi_e; the (pseudo)scalar
                # model in the paper has g_e = 0 and must NOT. Getting this wrong
                # is a factor 2.5 on the vector.
                ch = "all" if p == "vector" else "muon"
            if a.fig2_style:
                if det != "MiniBooNE":
                    continue                    # only MiniBooNE has the background
                out, tot = plot_paper_style(det, p, data, a.output_dir, DETECTORS[det], ch,
                                            rescale=a.rescale_to_excess)
                print("  %-10s %-7s signal %.4e -> %s" % (det, p, tot, os.path.basename(out)))
                continue
            out, tot = plot(det, p, data, a.output_dir, DETECTORS[det], ch,
                            canonical=(a.channels == "auto"),
                            paper_bins=not a.uniform_bins)
            print("  %-10s %-7s total %.4e events at %.2e POT  ->  %s"
                  % (det, p, tot, DETECTORS[det], os.path.basename(out)))
    for m in missing:
        print("  MISSING %s -- run that portal script with --engine analytic" % m)


if __name__ == "__main__":
    main()
