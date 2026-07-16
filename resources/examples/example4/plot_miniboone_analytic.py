"""
Clean MiniBooNE analytic count-rate plots (E_vis, cos theta, forward zoom) from
the authoritative sbnd_analytic engine -- the MiniBooNE analog of
plot_sbnd_analytic.py. Replaces the stale directed-mode MiniBooNE_*_countrate.png
(spiky angular, non-physical vector normalization) that fed the paper's fig:mb.

  scalar/pseudo : analytic_sp_mb, muon-only (g_e=0), MiniBooNE single-photon eff;
                  the mediator cos(theta) is smeared by the Primakoff opening angle
                  to the true PHOTON direction (as in plot_sbnd_analytic).
  vector        : analytic_vec_mb, all channels, MiniBooNE electron-like (nu_e)
                  eff; cos(theta) is the true e+e- system direction (no smear).
POT = MINIBOONE_POT (18.75e20).

Usage:  python plot_miniboone_analytic.py [scalar|pseudo|vector|all]   (default all)
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sbnd_analytic as SA
from plot_sbnd_analytic import smear_photon_beam, _get_primakoff, load_portal, COLORS

HERE = os.path.dirname(os.path.abspath(__file__))

PORTALS = {
    "scalar": ("ScalarPortal_MiniBooNE_multichannel.py", "MiniBooNE scalar Dark Primakoff", False),
    "pseudo": ("PseudoscalarPortal_MiniBooNE_multichannel.py", "MiniBooNE pseudoscalar Dark Primakoff", False),
    "vector": ("VectorPortal_MiniBooNE_fullchain.py", "MiniBooNE vector $\\chi$ upscattering ($e^+e^-$)", True),
}


def make_plot(key, n_dec=400, muon_only=True):
    fname, label, vector = PORTALS[key]
    S = load_portal(fname)
    pot = S.MINIBOONE_POT
    dp = None if vector else _get_primakoff(S)
    rng = np.random.default_rng(1234)
    flux = os.environ.get("FLUX", "dk2nu")             # real BNB dk2nu (default) | bnb synthetic
    meson_fn = SA._mesons_dk2nu if flux == "dk2nu" else None

    chans = list(S.CHANNELS)
    if muon_only and not vector:
        chans = [nm for nm in chans if "mu" in nm]

    data = {}
    for nm in chans:
        fn = SA.analytic_vec_mb if vector else SA.analytic_sp_mb
        E, w, c = fn(S, nm, n_dec=n_dec, eff_mode="mb", return_cos=True, meson_fn=meson_fn)
        E, w, c = np.asarray(E), np.asarray(w), np.asarray(c)
        if dp is not None and E.size:          # scalar/pseudo -> true photon dir
            c = smear_photon_beam(c, E, dp, rng)
        data[nm] = (E, w, c)

    total = sum(d[1].sum() for d in data.values())
    cos_note = "photon" if dp is not None else "$e^+e^-$ system"
    coup = r"all channels (lepton-universal)" if vector else r"MUON-ONLY ($g_e=0$)"
    eff_tag = ("MiniBooNE single-photon eff" if not vector
               else r"MiniBooNE electron-like ($\nu_e$) eff")

    Ebins = np.linspace(0.0, 2.0, 60)
    Cbins = np.linspace(-1.0, 1.0, 80)
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

    pot_note = "(%.2e POT)" % pot
    ax[0].axvline(0.140, color="0.4", ls="--", lw=1)
    ax[0].set_title("%s: $E_{vis}$ %s" % (label, pot_note))
    ax[0].set_xlabel(r"$E_{vis}$ [GeV]"); ax[0].set_ylabel("events / bin")
    ax[1].set_title("%s: $\\cos\\theta$ %s" % (label, pot_note))
    ax[1].set_xlabel(r"$\cos\theta$ wrt beam (%s)" % cos_note); ax[1].set_ylabel("events / bin")
    ax[2].set_title("%s: $\\cos\\theta$ zoom [%.2f, 1.0]" % (label, zlo))
    ax[2].set_xlabel(r"$\cos\theta$"); ax[2].set_ylabel("events / bin")
    for a in ax:
        a.legend(fontsize=8)
    fig.suptitle("MiniBooNE analytic $\\sigma\\!\\cdot\\!N\\!\\cdot\\!$chord  --  %s  --  TOTAL = %.3e events\n"
                 "%s  |  %s  |  flux: %s" % (label, total, eff_tag, coup,
                 "real dk2nu" if flux == "dk2nu" else "BNBFlux"), fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.93])

    os.makedirs(os.path.join(HERE, "output"), exist_ok=True)
    out = os.path.join(HERE, "output", "MiniBooNE_%s_analytic_countrate.png" % key)
    plt.savefig(out, dpi=130); plt.close(fig)
    print("  %-8s TOTAL=%.3e  -> %s" % (key, total, os.path.basename(out)))
    return {nm: data[nm][1].sum() for nm in data}


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    n_dec = int(os.environ.get("N_DEC", "400"))
    keys = list(PORTALS) if which == "all" else [which]
    print("MiniBooNE analytic count-rate plots (n_dec=%d)\n" % n_dec)
    for k in keys:
        make_plot(k, n_dec=n_dec)
