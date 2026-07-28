r"""Fold the DarkNews HNL timing into the BNB beam bunch structure.

Post-processor for the *_bnb_darknews_hnl_timing.csv files produced by
DarkNewsHNL_SBND_BNB_timing.py / DarkNewsHNL_ICARUS_BNB_timing.py.  It turns
the raw per-event timing into the observable that actually discriminates a
dark-sector signal from beam neutrinos: arrival phase within one RF bunch.

Signal vs prompt, both taken self-consistently from the SAME events:
  * prompt (beam-neutrino proxy) = the nu upscatter arrival delay
    (upscatter_delay_ns): the neutrino travels at beta ~ 1, so this is the
    prompt, SM-like timing.
  * signal (what a detector sees) = the N4 -> nu gamma photon arrival delay
    (hnl_decay_delay_ns): later than prompt by the slow-HNL flight.

Both delays are measured relative to a prompt beta=1 particle from the beam
target, so a truly prompt massless particle sits near 0; the massive HNL adds
a positive, energy-dependent lateness.

Beam model: BNB delivers ~53.1 MHz micro-bunches (18.83 ns spacing), each of
finite width.  We fold (proton-arrival Gaussian within a bunch) + delay, wrap
modulo the bunch spacing, and compare the signal and prompt phase profiles
inside one bunch.  Shapes are unit-area normalized (no absolute rate is
implied -- see the timing-study notes).

Run (uses the two CSVs already in output/):
  /home/shubham/siren_pr178_venv/bin/python timing_beam_fold.py
Tune: --bunch-spacing, --bunch-sigma, --nsmear, --late-nsigma.
Out:  output/timing_beamfold_sbnd_icarus.png  (+ printed figure-of-merit)
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

# BNB radio-frequency structure.  53.1 MHz -> 18.83 ns bunch spacing; the
# per-bunch proton arrival is roughly Gaussian with ~1.3 ns rms (an effective
# value; override with --bunch-sigma).
BNB_SPACING_NS = 18.83
BNB_SIGMA_NS = 1.3

DETECTORS = [
    ("SBND", "output/sbnd_bnb_darknews_hnl_timing.csv", "#1f77b4"),
    ("ICARUS", "output/icarus_bnb_darknews_hnl_timing.csv", "#d62728"),
]


def _load(csv_path, signal_col, prompt_col):
    d = np.genfromtxt(csv_path, delimiter=",", names=True)
    w = np.asarray(d["weight"], dtype=float)
    sig = np.asarray(d[signal_col], dtype=float)
    pmt = np.asarray(d[prompt_col], dtype=float)
    sm = np.isfinite(sig) & np.isfinite(w) & (w > 0)
    pm = np.isfinite(pmt) & np.isfinite(w) & (w > 0)
    return (sig[sm], w[sm]), (pmt[pm], w[pm])


def _wquantile(x, w, q):
    o = np.argsort(x)
    x, w = x[o], w[o]
    c = np.cumsum(w) / w.sum()
    return x[np.searchsorted(c, q)]


def _fold(delay, weight, spacing, sigma, nsmear, rng):
    """Fold (bunch-Gaussian + delay) modulo the bunch spacing, into a window
    centered on the nominal proton bunch center [-spacing/2, +spacing/2)."""
    d = np.repeat(delay, nsmear)
    w = np.repeat(weight, nsmear)
    t = d + rng.normal(0.0, sigma, size=d.shape)
    phase = np.mod(t + spacing / 2.0, spacing) - spacing / 2.0
    return phase, w


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bunch-spacing", type=float, default=BNB_SPACING_NS,
                    help="RF bunch spacing [ns] (default BNB 18.83)")
    ap.add_argument("--bunch-sigma", type=float, default=BNB_SIGMA_NS,
                    help="per-bunch proton arrival rms [ns] (default 1.3)")
    ap.add_argument("--nsmear", type=int, default=200,
                    help="proton-phase draws per event")
    ap.add_argument("--late-nsigma", type=float, default=2.0,
                    help="late-window cut = prompt median + N*bunch_sigma")
    ap.add_argument("--signal-col", default="hnl_decay_delay_ns")
    ap.add_argument("--prompt-col", default="upscatter_delay_ns")
    ap.add_argument("--output", default="output/timing_beamfold_sbnd_icarus.png")
    args = ap.parse_args(argv)

    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(len(DETECTORS), 2, figsize=(12.5, 8.5),
                             constrained_layout=True)
    if len(DETECTORS) == 1:
        axes = axes[None, :]

    print("Beam model: spacing=%.2f ns, bunch sigma=%.2f ns, late cut=median+%.1f sigma"
          % (args.bunch_spacing, args.bunch_sigma, args.late_nsigma))
    print("-" * 78)

    for row, (name, csv, color) in enumerate(DETECTORS):
        if not os.path.exists(csv):
            axes[row, 0].text(0.5, 0.5, "missing %s" % csv, ha="center",
                              va="center", transform=axes[row, 0].transAxes)
            continue
        (sig, wsig), (pmt, wpmt) = _load(csv, args.signal_col, args.prompt_col)

        # ---- left: raw delay distributions (intrinsic timing) ----
        axL = axes[row, 0]
        lo = min(pmt.min(), sig.min())
        hi = _wquantile(sig, wsig, 0.995)
        bins = np.linspace(lo - 0.2, hi + 0.5, 60)
        axL.hist(pmt, bins=bins, weights=wpmt, density=True, histtype="step",
                 lw=1.8, color="0.45", label=r"prompt $\nu$ (upscatter)")
        axL.hist(sig, bins=bins, weights=wsig, density=True, histtype="step",
                 lw=2.0, color=color, label=r"signal $N_4\to\nu\gamma$")
        pm = _wquantile(pmt, wpmt, 0.5)
        sm = _wquantile(sig, wsig, 0.5)
        axL.axvline(pm, ls="--", lw=0.9, color="0.45")
        axL.axvline(sm, ls="--", lw=0.9, color=color)
        axL.set_xlabel(r"delay relative to prompt $\beta=1$ [ns]")
        axL.set_ylabel("weighted density")
        axL.set_title("%s: intrinsic delay  (prompt med=%.2f, sig med=%.2f, "
                      r"$\Delta$=%.2f ns)" % (name, pm, sm, sm - pm), fontsize=10)
        axL.legend(fontsize=8)
        axL.grid(alpha=0.25)

        # ---- right: folded into one BNB bunch cycle ----
        axR = axes[row, 1]
        ps, wps = _fold(sig, wsig, args.bunch_spacing, args.bunch_sigma,
                        args.nsmear, rng)
        pp, wpp = _fold(pmt, wpmt, args.bunch_spacing, args.bunch_sigma,
                        args.nsmear, rng)
        fb = np.linspace(-args.bunch_spacing / 2, args.bunch_spacing / 2, 90)
        axR.hist(pp, bins=fb, weights=wpp, density=True, histtype="step",
                 lw=1.8, color="0.45", label=r"prompt $\nu$")
        axR.hist(ps, bins=fb, weights=wps, density=True, histtype="stepfilled",
                 lw=2.0, color=color, alpha=0.30)
        axR.hist(ps, bins=fb, weights=wps, density=True, histtype="step",
                 lw=2.0, color=color, label=r"signal $N_4\to\nu\gamma$")

        # late-time window cut and figure-of-merit
        cut = pm + args.late_nsigma * args.bunch_sigma
        axR.axvline(cut, ls=":", lw=1.4, color="k")
        eff_sig = float(wps[ps > cut].sum() / wps.sum())
        leak_pmt = float(wpp[pp > cut].sum() / wpp.sum())
        enh = eff_sig / leak_pmt if leak_pmt > 0 else np.inf
        axR.axvspan(cut, args.bunch_spacing / 2, color="k", alpha=0.05)
        axR.set_xlabel("arrival phase within BNB bunch [ns]")
        axR.set_ylabel("weighted density")
        axR.set_title("%s: folded  (late-win sig eff=%.2f, prompt leak=%.2f, "
                      "enh=%.1fx)" % (name, eff_sig, leak_pmt, enh), fontsize=10)
        axR.legend(fontsize=8)
        axR.grid(alpha=0.25)

        print("%-7s | prompt med=%.2f ns  signal med=%.2f ns  delta=%.2f ns"
              % (name, pm, sm, sm - pm))
        print("        | late window (phase>%.2f ns): signal eff=%.3f, "
              "prompt leak=%.3f, enhancement=%.1fx"
              % (cut, eff_sig, leak_pmt, enh))
    print("-" * 78)

    fig.suptitle("DarkNews HNL timing folded into BNB bunch structure "
                 "(unit-area shapes; no absolute rate)", fontsize=12)
    out = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print("Wrote %s" % out)


if __name__ == "__main__":
    main()
