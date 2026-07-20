"""
AUTHORITATIVE analytic NuMI rate for ANY SBN detector (ICARUS / SBND / MiniBooNE),
all three portals (scalar / pseudo / vector), from the g4numi RHC flux.

Generalizes analytic_NuMI_ICARUS.py to the off-axis NuMI geometry of every SBN
detector (positions from sbn_geometry, NuMI-frame off-axis angles):
    ICARUS     5.8 deg @ 799 m   (box, TWO cryostats)
    SBND      30.2 deg @ 402 m   (box, single LArTPC -- far off the NuMI axis)
    MicroBooNE 7.7 deg @ 682 m   (not run here)
    MiniBooNE  6.4 deg @ 746 m   (sphere, carbon -- HYPOTHETICAL: MiniBooNE took
                                  no NuMI data; shown for a like-for-like rate)

Reuses the verified NuMI->BNB transform + multi-file g4numi reader from
analytic_NuMI_ICARUS (numi_meson_fn). Box detectors use analytic_sp/vec ray-
traced through the true active box(es); MiniBooNE uses the sphere engine
(analytic_sp_mb/vec_mb) at its BNB position. --anchored gives the MiniBooNE(BNB)
comparable observable (coupling cancels; flux does NOT).

Usage:
    ICARUS_NUMI_POT=3e21 python analytic_NuMI_SBN.py --detector SBND --portal all --anchored
"""
import argparse
import os

import numpy as np

from siren import _util

HERE = os.path.dirname(os.path.abspath(__file__))
SA = _util.load_module("AnalyticRate", os.path.join(
    _util.resource_package_dir(), "processes", "DarkNewsTables", "AnalyticRate.py"))
from plot_sbnd_analytic import (smear_photon_beam, _get_primakoff, COLORS,
                                mb_inwindow, SINGLE_GAMMA_EFF, SEL_FACTOR,
                                WIN_LO, WIN_HI, MB_EXCESS)
# reuse the flux reader (multi-file g4numi + NuMI->BNB transform) and its POT.
from analytic_NuMI_ICARUS import numi_meson_fn, ICARUS_NUMI_POT, NUMI_FILES, _glob
GEO = _util.load_module("sbn_geometry",
                        os.path.join(_util.resource_package_dir(),
                                     "detectors", "SBN", "SBN-v1", "sbn_geometry.py"))

# detector -> {portal: (module file, is_vector)}, geometry engine, centers
_BOX = "box"; _SPH = "sphere"
DETECTORS = {
    "ICARUS": (_BOX, {
        "scalar": ("ScalarPortal_ICARUS_multichannel.py", False),
        "pseudo": ("PseudoscalarPortal_ICARUS_multichannel.py", False),
        "vector": ("VectorPortal_ICARUS_fullchain.py", True)}),
    "SBND": (_BOX, {
        "scalar": ("ScalarPortal_SBND_multichannel.py", False),
        "pseudo": ("PseudoscalarPortal_SBND_multichannel.py", False),
        "vector": ("VectorPortal_SBND_fullchain.py", True)}),
    "MiniBooNE": (_SPH, {
        "scalar": ("ScalarPortal_MiniBooNE_multichannel.py", False),
        "pseudo": ("PseudoscalarPortal_MiniBooNE_multichannel.py", False),
        "vector": ("VectorPortal_MiniBooNE_fullchain.py", True)}),
}


def _centers(detector, S):
    """Active-box center(s) in BNB coords for a box detector."""
    if detector == "ICARUS":
        return [np.asarray(c, float) for c in S.ICARUS_MODULE_CENTERS_BNB]
    return [np.asarray(GEO.detector_center(detector, "BNB"), float)]   # SBND: single box


def run_portal(detector, key, n_dec, eff, anchored, muon_only=True):
    engine, portals = DETECTORS[detector]
    fname, vector = portals[key]
    S = _util.load_module("%s_%s" % (detector, key), os.path.join(HERE, fname))
    dp = None if vector else _get_primakoff(S)
    rng = np.random.default_rng(1234)

    names = list(S.CHANNELS)
    if muon_only and not vector:
        names = [nm for nm in names if "mu" in nm]

    # efficiency: anchored -> grounded single-gamma (scalar/pseudo) / legacy e+e-.
    if anchored and not vector:
        eff_mode, wscale = "raw", SINGLE_GAMMA_EFF
    elif anchored:
        eff_mode, wscale = "lartpc", SEL_FACTOR
    else:
        eff_mode, wscale = eff, 1.0

    data, per, grand = {}, {}, 0.0
    for nm in names:
        Es, Ws, Cs = [], [], []
        if engine == _BOX:
            fn = SA.analytic_vec if vector else SA.analytic_sp
            for ctr in _centers(detector, S):              # sum over active boxes
                E, w, c = fn(S, nm, n_dec=n_dec, det=ctr, pot=ICARUS_NUMI_POT,
                             eff_mode=eff_mode, return_cos=True, meson_fn=numi_meson_fn)
                Es.append(np.asarray(E)); Ws.append(np.asarray(w) * wscale); Cs.append(np.asarray(c))
        else:  # MiniBooNE sphere: engine uses MB position + R_OIL; override POT.
            fn = SA.analytic_vec_mb if vector else SA.analytic_sp_mb
            _pot0 = S.MINIBOONE_POT
            S.MINIBOONE_POT = ICARUS_NUMI_POT
            try:
                E, w, c = fn(S, nm, n_dec=n_dec, eff_mode=eff_mode,
                             return_cos=True, meson_fn=numi_meson_fn)
            finally:
                S.MINIBOONE_POT = _pot0
            Es.append(np.asarray(E)); Ws.append(np.asarray(w) * wscale); Cs.append(np.asarray(c))
        E = np.concatenate(Es) if Es else np.array([])
        w = np.concatenate(Ws) if Ws else np.array([])
        c = np.concatenate(Cs) if Cs else np.array([])
        if dp is not None and E.size:                      # mediator -> photon dir
            c = smear_photon_beam(c, E, dp, rng)
        per[nm] = float(w.sum()); grand += per[nm]; data[nm] = (E, w, c)
    win = sum(d[1][(d[0] >= WIN_LO) & (d[0] <= WIN_HI)].sum() for d in data.values())
    return S, vector, dp, data, per, grand, win


def plot_portal(detector, key, vector, dp, data, per, grand, out, anchored, win, R):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    Ebins = np.linspace(0.0, 2.0, 50); Cbins = np.linspace(-1.0, 1.0, 60)
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
    if anchored:
        ax[0].axvspan(WIN_LO, WIN_HI, color="gold", alpha=0.20, label="signal window")
    cos_note = "photon" if dp is not None else "e$^+$e$^-$ system"
    ax[0].set_xlabel(r"$E_{vis}$ [GeV]"); ax[0].set_ylabel("events / bin"); ax[0].legend(fontsize=8)
    ax[0].set_title("%s $\\times$ NuMI (RHC): $E_{vis}$  (%.1e POT)" % (detector, ICARUS_NUMI_POT))
    ax[1].set_xlabel(r"$\cos\theta$ wrt NuMI axis (%s)" % cos_note)
    ax[1].set_ylabel("events / bin"); ax[1].legend(fontsize=8)
    ax[1].set_title("%s $\\times$ NuMI (RHC): $\\cos\\theta$" % detector)
    coup = "all channels (lepton-universal)" if vector else r"MUON-ONLY ($g_e=0$)"
    if anchored:
        head = ("MiniBooNE-ANCHORED (coupling cancels; flux does NOT): "
                "%s(NuMI)/MB(BNB) ratio=%.3f $\\times$ 320 = in-window %.0f events"
                % (detector, R, win))
        eff = ("single-$\\gamma$ eff=%.2f (cited)" % SINGLE_GAMMA_EFF if not vector
               else "e$^+$e$^-$ eff=legacy")
        fig.suptitle("%s %s Portal from g4numi  --  %s\n%s  |  %s"
                     % (detector, key, head, eff, coup), fontsize=10)
    else:
        fig.suptitle("Analytic $\\sigma\\!\\cdot\\!N\\!\\cdot\\!$chord  --  %s %s Portal from g4numi  "
                     "--  TOTAL = %.0f events  |  %s" % (detector, key, grand, coup), fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(out, dpi=130)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--detector", default="SBND", choices=list(DETECTORS))
    ap.add_argument("--portal", default="all", choices=["scalar", "pseudo", "vector", "all"])
    ap.add_argument("--n-dec", type=int, default=150)
    ap.add_argument("--eff", default="raw", choices=["raw", "lartpc"])
    ap.add_argument("--anchored", action="store_true")
    args = ap.parse_args()

    present = [f for f in NUMI_FILES if os.path.exists(f)]
    if not present:
        raise SystemExit("No NuMI dk2nu files (glob %r)" % _glob)
    det = args.detector
    c_num = GEO.detector_center(det, "NuMI")
    print("Analytic NuMI -> %s  (%d g4numi file(s), POT %.2e, %s)"
          % (det, len(present), ICARUS_NUMI_POT,
             "ANCHORED grounded eff" if args.anchored else "eff=%s" % args.eff))
    print("  %s off NuMI axis: %.1f deg @ %.0f m%s"
          % (det, np.degrees(np.arccos(c_num[2] / np.linalg.norm(c_num))),
             np.linalg.norm(c_num),
             "   (HYPOTHETICAL: no NuMI data)" if det == "MiniBooNE" else ""))
    keys = ["scalar", "pseudo", "vector"] if args.portal == "all" else [args.portal]
    os.makedirs(os.path.join(HERE, "output"), exist_ok=True)
    summary = {}
    for key in keys:
        S, vector, dp, data, per, grand, win = run_portal(det, key, args.n_dec, args.eff, args.anchored)
        R = None
        if args.anchored:
            den, _ = mb_inwindow(key, args.n_dec, meson_fn=SA._mesons_dk2nu)
            R = win / den if den > 0 else 0.0
            kfac = MB_EXCESS / den if den > 0 else 0.0
            for nm in data:
                E, w, c = data[nm]; data[nm] = (E, w * kfac, c); per[nm] *= kfac
            grand *= kfac; win *= kfac
        print("=" * 60)
        print("  %s %s:%s" % (det, key.upper(), ("  R=%.3f" % R) if R is not None else ""))
        for nm in per:
            print("    %-7s : %.4e events" % (nm, per[nm]))
        print("    %-7s : %.4e events" % ("in-window" if args.anchored else "TOTAL",
                                          win if args.anchored else grand))
        tag = "anchored" if args.anchored else "capability"
        out = os.path.join(HERE, "output", "%s_NuMI_%s_%s.png" % (det, key, tag))
        plot_portal(det, key, vector, dp, data, per, grand, out, args.anchored, win, R)
        np.savez(os.path.join(HERE, "output", "%s_NuMI_%s_%s.npz" % (det, key, tag)),
                 pot=ICARUS_NUMI_POT, anchored=args.anchored, **per)
        print("    -> %s" % os.path.basename(out))
        summary[key] = win if args.anchored else grand
    print("=" * 60)
    print("  %s NuMI %s (%.1e POT): %s"
          % (det, "IN-WINDOW ANCHORED" if args.anchored else "TOTALS", ICARUS_NUMI_POT,
             "  ".join("%s=%.0f" % (k, v) for k, v in summary.items())))


if __name__ == "__main__":
    main()
