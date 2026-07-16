"""
Clean count-rate plots for SBND from the AUTHORITATIVE analytic engine
(sbnd_analytic.py, sigma*N*chord ray-trace). Replaces the stale Jun-2022
*_reliable_countrate.png plots, whose absolute normalisation (~35x high) came
from the retired reliable_analytic_NS.py workflow.

For each portal it produces a 3-panel figure (E_vis, cos theta, cos theta zoom)
with per-channel + TOTAL, weighted by the physical event rate at SBND_POT.

Usage:  python plot_sbnd_analytic.py [scalar|pseudo|vector|all]   (default all)
"""
import importlib.util
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sbnd_analytic as SA

# Real BNB flux: source the parent mesons from the 12M dk2nu file (the physical
# flux) instead of the synthetic Sanford-Wang BNBFlux.  FLUX=bnb reverts to the
# synthetic generator.  The anchor is flux-independent, so the anchored (paper)
# yields are stable either way; only the raw/capability absolute rates move.
os.environ.setdefault("DK2NU_FILE", "/home/shubham/nubeam12M.dk2nu.root")

HERE = os.path.dirname(os.path.abspath(__file__))

PORTALS = {
    "scalar": ("ScalarPortal_SBND_multichannel.py", "SBND scalar Dark Primakoff", False),
    "pseudo": ("PseudoscalarPortal_SBND_multichannel.py", "SBND pseudoscalar Dark Primakoff", False),
    "vector": ("VectorPortal_SBND_fullchain.py", "SBND vector chi upscattering (e+e-)", True),
}
COLORS = {"K_e": "tab:blue", "K_mu": "tab:orange", "pi_e": "tab:green", "pi_mu": "tab:red"}

# --- analysis-level knobs (turn a capability yield into a quotable observable) ---
# PROD_FACTOR : known C-R Eq.25 production prefactor (code is 2x high) -> /2.
# SEL_FACTOR  : background-rejection SELECTION retention ON TOP of the LArTPC reco
#               capability (PID/containment/cosmic+dirt rejection). ~0.15 is a
#               generic-LArTPC single-photon estimate; combined with the ~0.9 reco
#               plateau this gives total single-gamma eff ~0.13 (~MiniBooNE-like).
# WIN         : single-photon signal energy window (MiniBooNE low-E excess region),
#               the relevant comparison region for E_vis ~ E_gamma.
PROD_FACTOR = 0.5
SEL_FACTOR  = 0.15
WIN_LO, WIN_HI = 0.140, 0.300         # GeV

# --- MiniBooNE anchor (coupling-normalized SBND yield) ------------------------
# N_SBND = (SBND/MiniBooNE observable ratio) x (MiniBooNE measured excess).  The
# ratio is from the SAME engine (analytic_sp vs analytic_sp_mb) so coupling,
# production-factor and flux-normalization CANCEL; only geometry/target/POT/eff
# survive.  REQUIREMENT: MINIBOONE_POT (in the MB script) must be the POT of the
# quoted excess -- it drives the over-prediction factor and hence the anchor.
MB_EXCESS = 320.0                     # MiniBooNE single-photon excess (events)
MB_SCRIPTS = {"scalar": "ScalarPortal_MiniBooNE_multichannel.py",
              "pseudo": "PseudoscalarPortal_MiniBooNE_multichannel.py",
              "vector": "VectorPortal_MiniBooNE_fullchain.py"}


def mb_inwindow(key, n_dec, meson_fn=None):
    """MiniBooNE in-window observable model rate, same engine as SBND.
    scalar/pseudo: single-photon, muon-only channels (paper coupling).
    vector: e+e- cascade, ALL channels (kinetic mixing is lepton-universal);
    MiniBooNE Cherenkov counts the collimated e+e- as the same electron-like
    sub-GeV excess, so analytic_vec_mb applies the MiniBooNE single-photon eff.
    meson_fn=None -> synthetic BNBFlux (default); pass _mesons_dk2nu to source the
    MiniBooNE side from the real dk2nu file (for a dk2nu-on-both-sides anchor)."""
    SMB = load_portal(MB_SCRIPTS[key])
    vector = key == "vector"
    chans = list(SMB.CHANNELS) if vector else [c for c in SMB.CHANNELS if "mu" in c]
    tot = 0.0
    for nm in chans:
        if vector:
            E, w = SA.analytic_vec_mb(SMB, nm, n_dec=n_dec, eff_mode="mb", meson_fn=meson_fn)
        else:
            E, w = SA.analytic_sp_mb(SMB, nm, n_dec=n_dec, eff_mode="mb", meson_fn=meson_fn)
        E = np.asarray(E); w = np.asarray(w)
        tot += w[(E >= WIN_LO) & (E <= WIN_HI)].sum()
    return tot, SMB.MINIBOONE_POT


def load_portal(fname):
    spec = importlib.util.spec_from_file_location("PortalSBND", os.path.join(HERE, fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _get_primakoff(S):
    """Return one DarkPrimakoff _dp object for this (scalar/pseudo) portal.
    All channels share the same target + m_phi, so any valid channel works."""
    for nm, (pdg, m_M, m_l, lpdg, nupdg, gsm) in S.CHANNELS.items():
        if (m_M - m_l) > S.M_PHI:
            ch = S.build_onshell_models(pdg, m_M, m_l, lpdg, nupdg)
            return ch["models"]["primakoff"]._dp
    return None


def sample_cos_star(dp, E, rng, nE=48, nt=160):
    """Vectorised sample of the Primakoff opening angle cos(theta*) (photon wrt
    incoming phi) from dsigma/dt, binned in E_phi. cos(theta*) ~ 1 + t/(2 E^2)."""
    E = np.asarray(E, float)
    out = np.ones_like(E)
    if E.size == 0:
        return out
    edges = np.linspace(E.min(), E.max() + 1e-9, nE + 1)
    idx = np.clip(np.digitize(E, edges) - 1, 0, nE - 1)
    for b in range(nE):
        m = idx == b
        if not m.any():
            continue
        Eb = 0.5 * (edges[b] + edges[b + 1])
        s = dp.m_phi ** 2 + dp.MA ** 2 + 2.0 * dp.MA * Eb
        tlo, thi = dp._t_range(s)
        if tlo is None or thi <= tlo:
            continue
        ts = np.linspace(tlo, thi, nt)
        w = np.array([dp._dsigma_dt(s, t) for t in ts])
        if w.sum() <= 0:
            continue
        cdf = np.cumsum(w); cdf /= cdf[-1]
        tsamp = np.interp(rng.random(int(m.sum())), cdf, ts)
        out[m] = np.clip(1.0 + tsamp / (2.0 * Eb * Eb), -1.0, 1.0)
    return out


def smear_photon_beam(cos_med, E, dp, rng):
    """Rotate the mediator direction (cos_med wrt beam) by the sampled Primakoff
    opening angle to get the outgoing-photon cos(theta) wrt the beam."""
    cstar = sample_cos_star(dp, E, rng)
    th_med = np.arccos(np.clip(cos_med, -1.0, 1.0))
    th_star = np.arccos(np.clip(cstar, -1.0, 1.0))
    psi = rng.uniform(0.0, 2.0 * np.pi, size=E.shape)
    return np.clip(np.cos(th_med) * np.cos(th_star)
                   + np.sin(th_med) * np.sin(th_star) * np.cos(psi), -1.0, 1.0)


def make_plot(key, n_dec, eff_mode="lartpc", muon_only=True, level="capability"):
    fname, label, vector = PORTALS[key]
    S = load_portal(fname)
    fn = SA.analytic_vec if vector else SA.analytic_sp
    pot = S.SBND_POT

    dp = None if vector else _get_primakoff(S)
    rng = np.random.default_rng(1234)

    # flux source: real BNB dk2nu (default) or synthetic BNBFlux (FLUX=bnb).
    flux = os.environ.get("FLUX", "dk2nu")
    meson_fn = SA._mesons_dk2nu if flux == "dk2nu" else None
    flux_label = "real dk2nu" if flux == "dk2nu" else "BNBFlux"

    # channel selection: muon-only (g_e=0) drops the electron production channels.
    # Physical for the muon-coupled scalar/pseudoscalar mediator (the paper's model).
    # The vector couples lepton-universally via kinetic mixing, so muon-only is N/A
    # there -- keep all channels and say so in the title.
    chan_names = list(S.CHANNELS)
    if muon_only and not vector:
        chan_names = [nm for nm in chan_names if "mu" in nm]

    # analysis / anchored levels build on the LArTPC capability:
    #   analysis : production/2 x selection factor + signal window (absolute)
    #   anchored : coupling-normalized -- scale the LArTPC(xSEL) spectrum so the
    #              in-window yield = (SBND/MiniBooNE ratio) x MB measured excess.
    analysis = (level == "analysis")
    anchored = (level == "anchored")
    if analysis or anchored:
        eff_mode = "lartpc"
    # SBND per-hit weight scale: analysis => prod/2 x SEL; anchored => SEL only
    # (prod/2 cancels in the SBND/MB ratio); capability/raw => 1.
    wscale = (PROD_FACTOR * SEL_FACTOR) if analysis else (SEL_FACTOR if anchored else 1.0)

    data = {}
    for nm in chan_names:
        E, w, c = fn(S, nm, n_dec=n_dec, return_cos=True, eff_mode=eff_mode, meson_fn=meson_fn)
        E, w, c = np.asarray(E), np.asarray(w) * wscale, np.asarray(c)
        # scalar/pseudo: smear mediator dir by the Primakoff opening angle to
        # recover the true PHOTON cos(theta) wrt beam (forward peak with width).
        if dp is not None and E.size:
            c = smear_photon_beam(c, E, dp, rng)
        data[nm] = (E, w, c)

    win_ev = sum(d[1][(d[0] >= WIN_LO) & (d[0] <= WIN_HI)].sum() for d in data.values())

    # anchor: rescale so in-window yield = (SBND/MB ratio) x MB measured excess.
    anchor_info = None
    if anchored:
        m_win, mb_pot = mb_inwindow(key, n_dec, meson_fn=meson_fn)
        R = win_ev / m_win if m_win > 0 else 0.0
        k = MB_EXCESS / m_win if m_win > 0 else 0.0
        for nm in data:
            E, w, c = data[nm]; data[nm] = (E, w * k, c)
        win_ev *= k
        anchor_info = (R, m_win, mb_pot, m_win / MB_EXCESS)

    total_ev = sum(d[1].sum() for d in data.values())
    cos_note = "photon" if dp is not None else ("$e^+e^-$ system" if vector else "mediator (proxy)")

    # honest mode labels for the title + filename
    if anchored:
        R, m_win, mb_pot, mb_over = anchor_info
        eff_tag = ("MiniBooNE-ANCHORED: SBND/MB ratio=%.2f $\\times$ measured excess %.0f "
                   "(MB model=%.0f = %.1f$\\times$320 at MB-POT %.2e; window [%.2f,%.2f])"
                   % (R, MB_EXCESS, m_win, mb_over, mb_pot, WIN_LO, WIN_HI))
    elif analysis:
        eff_tag = ("ANALYSIS-LEVEL: production/2 $\\times$ selection $\\varepsilon_{sel}$=%.2f "
                   "(on LArTPC reco) $\\times$ window [%.2f,%.2f] GeV"
                   % (SEL_FACTOR, WIN_LO, WIN_HI))
    elif eff_mode == "lartpc":
        gamma = "$\\gamma$-conversion+containment (SBND geom) $\\times$ generic-LAr reco turn-on"
        eff_tag = ("SBND LArTPC eff (CAPABILITY): " + gamma) if not vector else \
                  "SBND LArTPC eff (CAPABILITY): fiducial containment (SBND geom) $\\times$ generic-LAr reco (e+e-)"
    elif eff_mode == "raw":
        eff_tag = "RAW: no detector efficiency (bare $\\sigma\\!\\cdot\\!N\\!\\cdot\\!$chord)"
    else:
        eff_tag = "MiniBooNE single-photon eff PLACEHOLDER + 0.14 GeV thr (not SBND)"
    if vector:
        coup_tag = "all channels (vector: lepton-universal via kinetic mixing)"
    elif muon_only:
        coup_tag = r"MUON-ONLY ($g_e=0$, paper-valid coupling)"
    else:
        coup_tag = r"$g_e=g_\mu$ (incl. $\pi\to e\nu$: PIENU-excluded)"
    level_suffix = "anchored" if anchored else ("analysis" if analysis else eff_mode)
    mode_suffix = "%s_%s" % (level_suffix,
                             "muononly" if (muon_only and not vector) else "allchan")

    Ebins = np.linspace(0.0, 2.0, 60)          # GeV
    Cbins = np.linspace(-1.0, 1.0, 80)
    # data-driven zoom lower edge: frame the forward peak of THIS figure (the
    # vector e+e- system is far more collinear than the Primakoff photon, so a
    # fixed [0.80,1.0] would not resolve it). Use the weighted 2nd percentile,
    # rounded to 0.05, capped at 0.95, never coarser than 0.80.
    allc = np.concatenate([d[2] for d in data.values() if d[2].size]) if data else np.array([])
    allw = np.concatenate([d[1] for d in data.values() if d[2].size]) if data else np.array([])
    if allc.size:
        o = np.argsort(allc); cwz = np.cumsum(allw[o]) / max(allw.sum(), 1e-30)
        zlo = float(np.clip(np.floor(np.interp(0.02, cwz, allc[o]) * 20) / 20, 0.80, 0.95))
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
    if analysis or anchored:
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
        head = ("SBND MiniBooNE-ANCHORED observable  --  %s  --  in-window[%.2f,%.2f] = %.0f events"
                % (label, WIN_LO, WIN_HI, win_ev))
    elif analysis:
        head = ("SBND ANALYSIS-LEVEL observable  --  %s  --  in-window[%.2f,%.2f]=%.3e ev "
                "(full spectrum %.3e)" % (label, WIN_LO, WIN_HI, win_ev, total_ev))
    else:
        head = ("SBND analytic $\\sigma\\!\\cdot\\!N\\!\\cdot\\!$chord  --  %s  --  TOTAL = %.3e events"
                % (label, total_ev))
    fig.suptitle("%s\n%s  |  %s  |  flux: %s" % (head, eff_tag, coup_tag, flux_label), fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    os.makedirs(os.path.join(HERE, "output"), exist_ok=True)
    out = os.path.join(HERE, "output", "SBND_%s_analytic_%s.png" % (key, mode_suffix))
    plt.savefig(out, dpi=130)
    plt.close(fig)
    if anchored:
        R, m_win, mb_pot, mb_over = anchor_info
        tag = "ANCHORED in-window=%.0f ev (R=%.2f, MB model=%.0f=%.1fx320)" % (win_ev, R, m_win, mb_over)
    elif analysis:
        tag = "in-window=%.3e (full %.3e)" % (win_ev, total_ev)
    else:
        tag = "TOTAL=%.3e" % total_ev
    print("  %-8s %s   -> %s" % (key, tag, os.path.basename(out)))
    return {nm: data[nm][1].sum() for nm in data}


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    n_dec = int(os.environ.get("N_DEC", "400"))
    eff_mode = os.environ.get("EFF_MODE", "lartpc")   # raw | mb | lartpc
    level = os.environ.get("LEVEL", "capability")     # capability | analysis
    muon_only = os.environ.get("MUON_ONLY", "1") != "0"
    keys = list(PORTALS) if which == "all" else [which]

    print("Regenerating SBND analytic plots  (n_dec=%d, LEVEL=%s, EFF_MODE=%s, MUON_ONLY=%s)\n"
          % (n_dec, level, eff_mode, muon_only))
    for k in keys:
        chans = make_plot(k, n_dec, eff_mode=eff_mode, muon_only=muon_only, level=level)
        for nm, s in chans.items():
            print("      %-7s : %.4e events" % (nm, s))
        print()
