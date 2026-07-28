"""
AUTHORITATIVE analytic NuMI->ICARUS rate (sigma*N*chord ray-trace), all portals.

The NuMI counterpart of the BNB analytic engine: instead of the SIREN directed
sampler (which over-estimates 60-400x), this uses sbnd_analytic.analytic_sp/vec
-- the validated low-variance sigma*N*chord estimators -- sourced from the REAL
g4numi dk2nu flux and ray-traced through ICARUS's TWO active cryostats.

Portals:
  scalar / pseudo : phi/a + N -> gamma + N  (single-photon Dark Primakoff),
                    muon-only (g_e=0); cos(theta) smeared to the photon dir.
  vector          : full double-mediator chi cascade -> e+e-, all channels.

Two corrections vs VectorPortal_ICARUS_NuMI_dk2nu.py (the SIREN version):
  1. TWO-CRYOSTAT GEOMETRY: the analytic engine's box IS the argon target, so it
     uses the true single active MODULE box (3.00 x 3.16 x 17.95 m; ICARUS
     portal modules, 2026-07-17 geometry fix) called once per cryostat center
     (ICARUS_MODULE_CENTERS_BNB) and summed -- excluding the 1.2 m argon gap.
  2. NuMI -> BNB transform (GNuMIFlux.xml / SBN DocDB 22998-v2) applied to every
     g4numi decay vertex AND parent momentum before the ray-trace.

Flux: g4numi RHC dk2nu, medium-energy, ~1e6 POT/file.
Usage:
    ICARUS_NUMI_POT=3e21 python analytic_NuMI_ICARUS.py [--portal scalar|pseudo|vector|all]
        [--n-dec 200] [--eff raw|lartpc]
"""
import argparse
import glob
import os

import numpy as np

from siren import _util

HERE = os.path.dirname(os.path.abspath(__file__))
SA = _util.load_module("AnalyticRate", os.path.join(
    _util.resource_package_dir(), "processes", "DarkNewsTables", "AnalyticRate.py"))
# reuse the single-photon helpers (Primakoff opening-angle smearing) used by the
# BNB ICARUS/SBND analytic plotters.
from plot_sbnd_analytic import (smear_photon_beam, _get_primakoff, COLORS,
                                mb_inwindow, SINGLE_GAMMA_EFF, SEL_FACTOR,
                                WIN_LO, WIN_HI, MB_EXCESS)
GEO = _util.load_module("sbn_geometry",
                        os.path.join(_util.resource_package_dir(),
                                     "detectors", "SBN", "SBN-v1", "sbn_geometry.py"))

# portal -> (ICARUS model module, is_vector)
PORTALS = {
    "scalar": ("ScalarPortal_ICARUS_multichannel.py", False),
    "pseudo": ("PseudoscalarPortal_ICARUS_multichannel.py", False),
    "vector": ("VectorPortal_ICARUS_fullchain.py", True),
}

# NuMI beam frame -> BNB (SIREN world) rigid transform.
T = GEO.transform("NuMI", "BNB")
_R, _t = np.asarray(T.R, float), np.asarray(T.t, float)

# All g4numi files in sources/NuMI/ are read together (read_dk2nu sums POT over
# the list), unless NUMI_DK2NU_GLOB/NUMI_DK2NU_FILE restricts it. Each file is an
# independent seed of the same production -> more files = more MC statistics
# (smoother spectra), NOT a different physical rate (the weight is per POT).
_glob = os.environ.get("NUMI_DK2NU_GLOB", os.path.join(HERE, "sources", "NuMI", "g4numi*.root"))
NUMI_FILES = ([os.environ["NUMI_DK2NU_FILE"]] if os.environ.get("NUMI_DK2NU_FILE")
              else sorted(glob.glob(_glob)))
ICARUS_NUMI_POT = float(os.environ.get("ICARUS_NUMI_POT", "3.0e21"))


def _beam_tag(files):
    """Horn-current / beam mode from the g4numi filenames, so outputs are
    self-labeled and RHC/FHC runs can never overwrite each other."""
    n = " ".join(os.path.basename(f).lower() for f in files)
    return "RHC" if "_rhc" in n else ("FHC" if "_fhc" in n else "NuMI")


BEAM = _beam_tag(NUMI_FILES)

_NMAX = int(os.environ.get("NUMI_NMAX", "120000"))   # parents/species cap (raise to use multi-file stats)


def numi_meson_fn(S, pdg, n_max=None, seed=42):
    """meson_fn for the analytic engine: g4numi parents in the BNB world frame.

    Thin wrapper over Dk2nuReader.analytic_meson_source: reads ALL NuMI files
    (POT summed, cached) and applies the NuMI->BNB transform (T) to vertices and
    momenta. The flux-read + transform now live in the DarkNewsTables package."""
    return S._DK.analytic_meson_source(
        NUMI_FILES, pdg, beam_transform=T, n_max=(_NMAX if n_max is None else n_max), seed=seed)


def run_portal(key, n_dec, eff, anchored, muon_only=True):
    fname, vector = PORTALS[key]
    S = _util.load_module("ICARUS_%s" % key, os.path.join(HERE, fname))
    fn = SA.analytic_vec if vector else SA.analytic_sp
    dp = None if vector else _get_primakoff(S)
    rng = np.random.default_rng(1234)
    centers = [np.asarray(c, float) for c in S.ICARUS_MODULE_CENTERS_BNB]

    names = list(S.CHANNELS)
    if muon_only and not vector:
        names = [nm for nm in names if "mu" in nm]

    # efficiency treatment: anchored uses the GROUNDED single-gamma eff (scalar/
    # pseudo) applied flat on raw sigma*N*chord; vector keeps legacy lartpc x SEL
    # (e+e- electron-like grounding still pending). capability => as passed.
    if anchored and not vector:
        eff_mode, wscale = "raw", SINGLE_GAMMA_EFF
    elif anchored:
        eff_mode, wscale = "lartpc", SEL_FACTOR
    else:
        eff_mode, wscale = eff, 1.0

    data, per, grand = {}, {}, 0.0
    for nm in names:
        Es, Ws, Cs = [], [], []
        for ctr in centers:                                # sum over cryostats
            E, w, c = fn(S, nm, n_dec=n_dec, det=ctr, pot=ICARUS_NUMI_POT,
                         eff_mode=eff_mode, return_cos=True, meson_fn=numi_meson_fn)
            E, w, c = np.asarray(E), np.asarray(w) * wscale, np.asarray(c)
            if dp is not None and E.size:                  # mediator -> photon dir
                c = smear_photon_beam(c, E, dp, rng)
            Es.append(E); Ws.append(w); Cs.append(c)
        E = np.concatenate(Es) if Es else np.array([])
        w = np.concatenate(Ws) if Ws else np.array([])
        c = np.concatenate(Cs) if Cs else np.array([])
        per[nm] = float(w.sum()); grand += per[nm]; data[nm] = (E, w, c)
    win = sum(d[1][(d[0] >= WIN_LO) & (d[0] <= WIN_HI)].sum() for d in data.values())
    return S, vector, dp, data, per, grand, win


def plot_portal(key, vector, dp, data, per, grand, out, anchored=False, win=None, R=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Ebins = np.linspace(0.0, 2.0, 50)
    Cbins = np.linspace(-1.0, 1.0, 60)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    totE = totC = None
    for nm, (E, w, c) in data.items():
        if w.size == 0:
            continue
        col = COLORS.get(nm, "k")
        hE, _ = np.histogram(E, bins=Ebins, weights=w)
        hC, _ = np.histogram(c, bins=Cbins, weights=w)
        ax[0].step(Ebins[:-1], hE, where="post", color=col, label="%s (%.0f)" % (nm, per[nm]))
        ax[1].step(Cbins[:-1], hC, where="post", color=col)
        totE = hE if totE is None else totE + hE
        totC = hC if totC is None else totC + hC
    if totE is not None:
        ax[0].step(Ebins[:-1], totE, where="post", color="k", lw=2, label="TOTAL (%.0f)" % grand)
        ax[1].step(Cbins[:-1], totC, where="post", color="k", lw=2, label="TOTAL")
    cos_note = "photon" if dp is not None else "e$^+$e$^-$ system"
    if anchored:
        ax[0].axvspan(WIN_LO, WIN_HI, color="gold", alpha=0.20, label="signal window")
    ax[0].set_xlabel(r"$E_{vis}$ [GeV]"); ax[0].set_ylabel("events / bin")
    ax[0].set_title("ICARUS $\\times$ NuMI (RHC): $E_{vis}$  (%.1e POT)" % ICARUS_NUMI_POT)
    ax[0].legend(fontsize=8)
    ax[1].set_xlabel(r"$\cos\theta$ wrt NuMI axis (%s)" % cos_note)
    ax[1].set_ylabel("events / bin"); ax[1].legend(fontsize=8)
    ax[1].set_title("ICARUS $\\times$ NuMI (RHC): $\\cos\\theta$")
    coup = "all channels (lepton-universal)" if vector else r"MUON-ONLY ($g_e=0$)"
    if anchored:
        head = ("MiniBooNE-ANCHORED (coupling cancels; flux does NOT -- NuMI vs BNB): "
                "ICARUS(NuMI)/MB(BNB) ratio=%.3f $\\times$ excess 320 = in-window %.0f events"
                % (R, win))
        eff_note = ("SBND single-$\\gamma$ eff=%.2f (cited)" % SINGLE_GAMMA_EFF
                    if not vector else "e$^+$e$^-$ eff=legacy (grounding pending)")
        fig.suptitle("ICARUS %s Portal from g4numi (two cryostats)  --  %s\n%s  |  %s"
                     % (key, head, eff_note, coup), fontsize=10)
    else:
        fig.suptitle("Analytic $\\sigma\\!\\cdot\\!N\\!\\cdot\\!$chord  --  ICARUS %s Portal from g4numi "
                     "(two cryostats)  --  TOTAL = %.0f events  |  %s"
                     % (key, grand, coup), fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=130)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--portal", default="all", choices=list(PORTALS) + ["all"])
    ap.add_argument("--n-dec", type=int, default=200)
    ap.add_argument("--eff", default="raw", choices=["raw", "lartpc"])
    ap.add_argument("--anchored", action="store_true",
                    help="MiniBooNE-anchored comparable observable: coupling "
                         "cancels vs the MB(BNB) in-window model, grounded eff. "
                         "ICARUS(NuMI)/MB(BNB) ratio x 320 excess. (Flux does NOT "
                         "cancel here -- NuMI vs BNB is a real flux difference.)")
    args = ap.parse_args()

    present = [f for f in NUMI_FILES if os.path.exists(f)]
    if not present:
        raise SystemExit("No NuMI dk2nu files found (glob %r)" % _glob)

    c_num = GEO.detector_center("ICARUS", "NuMI")
    print("Analytic NuMI -> ICARUS  (sigma*N*chord, two cryostats)")
    print("  flux: %d file(s) [%s]   POT: %.2e   %s   n_dec: %d"
          % (len(present), ", ".join(os.path.basename(f)[-9:] for f in present),
             ICARUS_NUMI_POT,
             "MiniBooNE-ANCHORED (grounded eff)" if args.anchored else "eff=%s" % args.eff,
             args.n_dec))
    print("  ICARUS off NuMI axis: %.2f deg @ %.0f m"
          % (np.degrees(np.arccos(c_num[2] / np.linalg.norm(c_num))), np.linalg.norm(c_num)))
    keys = list(PORTALS) if args.portal == "all" else [args.portal]
    os.makedirs(os.path.join(HERE, "output"), exist_ok=True)
    summary = {}
    for key in keys:
        S, vector, dp, data, per, grand, win = run_portal(
            key, args.n_dec, args.eff, args.anchored)
        R = anchored_note = None
        if args.anchored:
            den, mbpot = mb_inwindow(key, args.n_dec, meson_fn=SA._mesons_dk2nu)
            R = win / den if den > 0 else 0.0
            kfac = MB_EXCESS / den if den > 0 else 0.0
            for nm in data:
                E, w, c = data[nm]; data[nm] = (E, w * kfac, c); per[nm] *= kfac
            grand *= kfac; win *= kfac
            anchored_note = ("ICARUS(NuMI)/MB(BNB) R=%.3f, MB model=%.0f=%.1fx320"
                             % (R, den, den / MB_EXCESS))
        print("=" * 60)
        print("  %s portal:%s" % (key.upper(), "  [%s]" % anchored_note if anchored_note else ""))
        for nm in per:
            print("    %-7s : %.4e events" % (nm, per[nm]))
        lbl = "in-window" if args.anchored else "TOTAL"
        print("    %-7s : %.4e events" % (lbl, win if args.anchored else grand))
        tag = ("anchored" if args.anchored else "capability") + "_" + BEAM
        out = os.path.join(HERE, "output", "ICARUS_NuMI_%s_%s.png" % (key, tag))
        plot_portal(key, vector, dp, data, per, grand, out, anchored=args.anchored,
                    win=win, R=R)
        np.savez(os.path.join(HERE, "output", "ICARUS_NuMI_%s_%s.npz" % (key, tag)),
                 pot=ICARUS_NUMI_POT, anchored=args.anchored, **per)
        print("    -> %s" % os.path.basename(out))
        summary[key] = win if args.anchored else grand
    print("=" * 60)
    print("  PORTAL %s (%.1e POT%s): %s"
          % ("IN-WINDOW ANCHORED" if args.anchored else "TOTALS", ICARUS_NUMI_POT,
             ", grounded eff" if args.anchored else ", eff=%s" % args.eff,
             "  ".join("%s=%.0f" % (k, v) for k, v in summary.items())))


if __name__ == "__main__":
    main()
