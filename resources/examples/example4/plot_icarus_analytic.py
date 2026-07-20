"""
Clean count-rate plots for ICARUS from the AUTHORITATIVE analytic engine
(sbnd_analytic.py, sigma*N*chord ray-trace) with the REAL high-statistics BNB
dk2nu flux (nubeam12M.dk2nu.root) as the parent-meson source.

ICARUS = the T600 liquid-argon TPC at 600 m on the BNB axis (SBN far detector).
Same engine as SBND; only the detector center / POT / flux change (see
icarus_analytic.py).  For each portal it produces the SBND-style 3-panel figure
(E_vis, cos theta, cos theta zoom) with per-channel + TOTAL, weighted by the
physical event rate at ICARUS_POT.

These are RAW / capability-level spectra (bare sigma*N*chord, or x LArTPC eff).
The MiniBooNE-ANCHORED ICARUS observable (the flux-independent quotable number)
is a separate step, analogous to plot_sbnd_analytic's 'anchored' level.

  DK2NU_FILE (env)  parent flux    (default nubeam12M.dk2nu.root)
  N_DEC      (env)  MC decays/meson (default 200)
  EFF_MODE   (env)  raw | lartpc    (default lartpc)

Usage:  python plot_icarus_analytic.py [scalar|pseudo|vector|all]   (default all)
"""
import importlib.util
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("DK2NU_FILE", "/home/shubham/nubeam12M.dk2nu.root")

from siren import _util as _su
import os as _os
SA = _su.load_module("AnalyticRate", _os.path.join(_su.resource_package_dir(), "processes", "DarkNewsTables", "AnalyticRate.py"))
# reuse the SBND plotter's helpers + the SAME MiniBooNE anchor machinery
# (import-safe: plot_sbnd_analytic is main-guarded).
from plot_sbnd_analytic import (smear_photon_beam, _get_primakoff, COLORS,
                                mb_inwindow, SEL_FACTOR, SINGLE_GAMMA_EFF,
                                WIN_LO, WIN_HI, MB_EXCESS)

HERE = os.path.dirname(os.path.abspath(__file__))

# ICARUS active-box center in BNB beam coords [m] (load_detector origin + GDML box)
DET_ICARUS = np.array([0.0, -0.432, 600.0])

PORTALS = {
    "scalar": ("ScalarPortal_ICARUS_multichannel.py", "ICARUS scalar Dark Primakoff", False),
    "pseudo": ("PseudoscalarPortal_ICARUS_multichannel.py", "ICARUS pseudoscalar Dark Primakoff", False),
    "vector": ("VectorPortal_ICARUS_fullchain.py", "ICARUS vector $\\chi$ upscattering ($e^+e^-$)", True),
}


def load_portal(fname):
    spec = importlib.util.spec_from_file_location("PortalICARUS", os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_BNB_REF = None
def _ensure_bnb(S):
    """Attach the synthetic BNBFlux module to an ICARUS portal (ICARUS scripts load
    _DK but not _BNB), so analytic_sp/analytic_vec's default _mesons (BNBFlux) works.
    Used ONLY for the anchor, where the ICARUS side must share MiniBooNE's flux."""
    global _BNB_REF
    if _BNB_REF is None:
        _BNB_REF = load_portal("ScalarPortal_SBND_multichannel.py")._BNB
    S._BNB = _BNB_REF


def make_plot(key, n_dec, eff_mode="lartpc", muon_only=True, level="capability"):
    fname, label, vector = PORTALS[key]
    S = load_portal(fname)
    fn = SA.analytic_vec if vector else SA.analytic_sp
    pot = S.ICARUS_POT
    dp = None if vector else _get_primakoff(S)
    rng = np.random.default_rng(1234)

    # muon-only (g_e=0) for the muon-coupled scalar/pseudoscalar; vector is
    # lepton-universal (kinetic mixing) -> keep all channels.
    chan_names = list(S.CHANNELS)
    if muon_only and not vector:
        chan_names = [nm for nm in chan_names if "mu" in nm]

    anchored = (level == "anchored")
    # ANCHOR flux: numerator (ICARUS) and denominator (MiniBooNE) MUST share the same
    # flux so it CANCELS exactly in R -> flux-independent. ANCHOR_FLUX=dk2nu (default)
    # sources BOTH sides from the real 12M dk2nu file; =bnb uses synthetic BNBFlux on
    # both. By flux-independence the two give the SAME R (that is the cross-check).
    anchor_flux = os.environ.get("ANCHOR_FLUX", "dk2nu")
    mb_mfn = None
    if anchored:
        # single-photon (scalar/pseudo): grounded flat total single-gamma eff on
        # the raw sigma*N*chord yield (replaces the back-tuned lartpc x SEL). The
        # vector e+e- pair is electron-like (higher eff, no dedicated study), so
        # keep the legacy lartpc x SEL_FACTOR there and flag it for grounding.
        if vector:
            eff_mode = "lartpc"; wscale = SEL_FACTOR
        else:
            eff_mode = "raw"; wscale = SINGLE_GAMMA_EFF
        if anchor_flux == "bnb":
            _ensure_bnb(S); meson_fn = None; mb_mfn = None
        else:
            meson_fn = SA._mesons_dk2nu; mb_mfn = SA._mesons_dk2nu
    else:
        meson_fn = SA._mesons_dk2nu
        wscale = 1.0

    # ICARUS = two separate cryostats. Call the analytic engine once per active
    # MODULE center (single-module _TPC_BOX in the portal module) and sum the
    # per-module yields; a single fat box would include the 1.2 m argon-free gap
    # and the warm vessel. Falls back to DET_ICARUS if the module constants are
    # absent (older portal module without the two-module geometry fix).
    module_centers = [np.asarray(c, float)
                      for c in getattr(S, "ICARUS_MODULE_CENTERS_BNB", [DET_ICARUS])]
    data = {}
    for nm in chan_names:
        Es, Ws, Cs = [], [], []
        for ctr in module_centers:
            E, w, c = fn(S, nm, n_dec=n_dec, return_cos=True, eff_mode=eff_mode,
                         det=ctr, pot=pot, meson_fn=meson_fn)
            Es.append(np.asarray(E)); Ws.append(np.asarray(w) * wscale); Cs.append(np.asarray(c))
        E = np.concatenate(Es); w = np.concatenate(Ws); c = np.concatenate(Cs)
        if dp is not None and E.size:        # scalar/pseudo: mediator dir -> photon dir
            c = smear_photon_beam(c, E, dp, rng)
        data[nm] = (E, w, c)

    win_ev = sum(d[1][(d[0] >= WIN_LO) & (d[0] <= WIN_HI)].sum() for d in data.values())

    # anchor: rescale so ICARUS in-window yield = (ICARUS/MB ratio) x MB excess (320).
    anchor_info = None
    if anchored:
        m_win, mb_pot = mb_inwindow(key, n_dec, meson_fn=mb_mfn)   # MiniBooNE in-window observable
        R = win_ev / m_win if m_win > 0 else 0.0
        kfac = MB_EXCESS / m_win if m_win > 0 else 0.0
        for nm in data:
            E, w, c = data[nm]; data[nm] = (E, w * kfac, c)
        win_ev *= kfac
        anchor_info = (R, m_win, mb_pot, m_win / MB_EXCESS)

    total_ev = sum(d[1].sum() for d in data.values())
    cos_note = "photon" if dp is not None else "$e^+e^-$ system"

    if anchored:
        R, m_win, mb_pot, mb_over = anchor_info
        eff_tag = ("MiniBooNE-ANCHORED (%s flux, both sides): ICARUS/MB ratio=%.2f $\\times$ excess %.0f "
                   "(MB model=%.0f=%.1f$\\times$320 @ MB-POT %.2e; window [%.2f,%.2f]); FLUX CANCELS"
                   % (anchor_flux, R, MB_EXCESS, m_win, mb_over, mb_pot, WIN_LO, WIN_HI))
    elif eff_mode == "lartpc":
        eff_tag = ("ICARUS LArTPC eff (CAPABILITY): fiducial containment (T600 geom) "
                   "$\\times$ generic-LAr reco turn-on")
    else:
        eff_tag = "RAW: no detector efficiency (bare $\\sigma\\!\\cdot\\!N\\!\\cdot\\!$chord)"
    coup_tag = ("all channels (vector: lepton-universal)" if vector else
                r"MUON-ONLY ($g_e=0$, paper coupling)")
    level_suffix = "anchored" if anchored else ("%s_dk2nu" % eff_mode)
    mode_suffix = "%s_%s" % (level_suffix, "allchan" if vector else "muononly")

    Ebins = np.linspace(0.0, 2.0, 60)
    Cbins = np.linspace(-1.0, 1.0, 80)
    allc = np.concatenate([d[2] for d in data.values() if d[2].size]) if data else np.array([])
    allw = np.concatenate([d[1] for d in data.values() if d[2].size]) if data else np.array([])
    if allc.size:
        o = np.argsort(allc); cwz = np.cumsum(allw[o]) / max(allw.sum(), 1e-30)
        zlo = float(np.clip(np.floor(np.interp(0.02, cwz, allc[o]) * 20) / 20, 0.80, 0.98))
    else:
        zlo = 0.80
    Czoom = np.linspace(zlo, 1.0, 60)

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    tot_E = tot_C = tot_Cz = None
    for nm, (E, w, c) in data.items():
        if w.size == 0:
            continue
        col = COLORS.get(nm, "k")
        hE, _ = np.histogram(E, bins=Ebins, weights=w)
        hC, _ = np.histogram(c, bins=Cbins, weights=w)
        hZ, _ = np.histogram(c, bins=Czoom, weights=w)
        ax[0].step(Ebins[:-1], hE, where="post", color=col, label=nm)
        ax[1].step(Cbins[:-1], hC, where="post", color=col, label=nm)
        ax[2].step(Czoom[:-1], hZ, where="post", color=col, label=nm)
        tot_E = hE if tot_E is None else tot_E + hE
        tot_C = hC if tot_C is None else tot_C + hC
        tot_Cz = hZ if tot_Cz is None else tot_Cz + hZ

    for a, tot, bins in ((ax[0], tot_E, Ebins), (ax[1], tot_C, Cbins), (ax[2], tot_Cz, Czoom)):
        if tot is not None:
            a.step(bins[:-1], tot, where="post", color="k", lw=2, label="TOTAL")

    pot_note = "(%.1e POT)" % pot
    if anchored:
        ax[0].axvspan(WIN_LO, WIN_HI, color="gold", alpha=0.20, label="signal window")
    ax[0].set_title("%s: $E_{vis}$ %s" % (label, pot_note))
    ax[0].set_xlabel(r"$E_{vis}$ [GeV]"); ax[0].set_ylabel("events / bin")
    ax[1].set_title("%s: $\\cos\\theta$ %s" % (label, pot_note))
    ax[1].set_xlabel(r"$\cos\theta$ wrt beam (%s)" % cos_note); ax[1].set_ylabel("events / bin")
    ax[2].set_title("%s: $\\cos\\theta$ zoom [%.2f, 1.0]" % (label, zlo))
    ax[2].set_xlabel(r"$\cos\theta$"); ax[2].set_ylabel("events / bin")
    for a in ax:
        a.legend(fontsize=8)
    if anchored:
        head = ("ICARUS MiniBooNE-ANCHORED observable  --  %s  --  in-window[%.2f,%.2f] = %.0f events"
                % (label, WIN_LO, WIN_HI, win_ev))
    else:
        head = ("ICARUS analytic $\\sigma\\!\\cdot\\!N\\!\\cdot\\!$chord (dk2nu flux)  --  %s  "
                "--  TOTAL = %.3e events" % (label, total_ev))
    fig.suptitle("%s\n%s  |  %s" % (head, eff_tag, coup_tag), fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    os.makedirs(os.path.join(HERE, "output"), exist_ok=True)
    out = os.path.join(HERE, "output", "ICARUS_%s_analytic_%s.png" % (key, mode_suffix))
    plt.savefig(out, dpi=130); plt.close(fig)
    if anchored:
        R, m_win, mb_pot, mb_over = anchor_info
        tag = "ANCHORED in-window=%.0f ev (R=%.2f, MB model=%.0f=%.1fx320)" % (win_ev, R, m_win, mb_over)
    else:
        tag = "TOTAL=%.3e" % total_ev
    print("  %-8s %s   -> %s" % (key, tag, os.path.basename(out)))
    return {nm: data[nm][1].sum() for nm in data}


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    n_dec = int(os.environ.get("N_DEC", "200"))
    eff_mode = os.environ.get("EFF_MODE", "lartpc")   # raw | lartpc
    level = os.environ.get("LEVEL", "capability")     # capability | anchored
    muon_only = os.environ.get("MUON_ONLY", "1") != "0"
    keys = list(PORTALS) if which == "all" else [which]

    print("Regenerating ICARUS analytic plots  (n_dec=%d, LEVEL=%s, EFF_MODE=%s, flux=%s)\n"
          % (n_dec, level, eff_mode, os.path.basename(os.environ["DK2NU_FILE"])))
    for k in keys:
        chans = make_plot(k, n_dec, eff_mode=eff_mode, muon_only=muon_only, level=level)
        for nm, s in chans.items():
            print("      %-7s : %.4e events" % (nm, s))
        print()
