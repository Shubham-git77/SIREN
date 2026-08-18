"""
#3: SBND / ICARUS REACH for the (pseudo)scalar Dark Primakoff.

If the model explains the MiniBooNE 320-event excess, how many signal events do
SBND (BNB) and ICARUS (NuMI) see?  The coupling is fixed by MiniBooNE at each m_Zp
(the #2 fit curve).  Both detector and MB rates scale as (product)^2, so the ratio
is coupling-INDEPENDENT:
    N_det(m_Zp) = 320 * [N_det/prod_det^2] / [N_MB/prod_MB^2]   (in-window [0.14,0.30])
This is the MiniBooNE-anchored ratio the validated scripts compute at fixed
couplings, here scanned across m_Zp with the #2 cached-sigma trick.

Beams/geometry (real configs): MB = BNB carbon sphere (mb eff); SBND = BNB argon
box, SBND_POT, single-gamma eff 0.10; ICARUS = NuMI argon box x2 cryostats,
ICARUS_NUMI_POT, single-gamma eff 0.10.

Run:  DK2NU_FILE=/home/shubham/nubeamHighSample.dk2nu.root \
      /home/shubham/siren_venv/bin/python scan_reach_sbnd_icarus.py
Outputs: output/reach_sbnd_icarus.{npz,png}
"""
import os, importlib.util, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

os.environ.setdefault("DK2NU_FILE", "/home/shubham/nubeamHighSample.dk2nu.root")   # BNB side
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
PKG  = os.environ.get(
    "SIREN_DNT_DIR",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "processes", "DarkNewsTables",
    ),
)

def load(path, name):
    import sys
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m

SA = load(os.path.join(PKG, "AnalyticRate.py"), "AnalyticRate")
from analytic_NuMI_ICARUS import numi_meson_fn, ICARUS_NUMI_POT     # NuMI flux + NuMI->BNB transform
SINGLE_GAMMA_EFF = float(os.environ.get("SINGLE_GAMMA_EFF", "0.10"))

# ANCHOR (reviewed 2026-08-17). 320 is the paper's "MiniBooNE observed 320 excess
# events below 300 MeV visible energy" [its ref 3]. It is NOT derivable from the
# official release we now use: HEPData ins1804293 starts at 200 MeV and gives
# 204.8 in 200-300, so the paper's 320 must include the 150-200 MeV region that
# release does not cover. Keeping 320 preserves the historical result and the
# paper's own statement; deriving it from miniboone_data over [200,300] would
# instead give 204.8 and shift every fitted coupling by sqrt(320/204.8) = 1.25.
# Set EXCESS_WINDOW_ANCHOR=data to use the release value instead.
import os as _os
import miniboone_data as _MB
_ANCHOR = _os.environ.get("EXCESS_WINDOW_ANCHOR", "paper320")
WIN = (0.140, 0.300); NDEC = 300
EXCESS = 320.0 if _ANCHOR == "paper320" else float(_MB.EXCESS[0])
ET_ENGINE = np.concatenate([np.linspace(0.001, 0.3, 120), np.linspace(0.31, 9, 160)])
ET_SCAN   = np.linspace(0.13, 0.32, 60)
MZP_GRID  = np.geomspace(0.030, 0.200, 24)

def cache(S, kind, meson_fn, pot, centers, eff_mode, wscale):
    """Return (E_vis[], G[], dp, product) for the muon channels of detector S.
    G = w*wscale / sigma_ref(E)  strips the (nucleus-correct) cross-section."""
    product = S.G_MU_PROD * S.G_N * (S.LAMBDA * 1e-3)   # MeV^-1
    ref_mzp = S.M_ZP; Els = []; Gs = []; dp = None
    for ch in [c for c in S.CHANNELS if "mu" in c]:
        pdg, m_M, m_l, lpdg, nupdg, gsm = S.CHANNELS[ch]
        dp = S.build_onshell_models(pdg, m_M, m_l, lpdg, nupdg)["models"]["primakoff"]._dp
        if kind == "sphere":
            El, w = SA.analytic_sp_mb(S, ch, n_dec=NDEC, eff_mode=eff_mode, meson_fn=meson_fn)
            El, w = np.asarray(El), np.asarray(w) * wscale
        else:
            Ee, ww = [], []
            for ctr in (centers if centers else [None]):
                E2, w2 = SA.analytic_sp(S, ch, n_dec=NDEC, eff_mode=eff_mode, det=ctr, pot=pot, meson_fn=meson_fn)
                Ee.append(np.asarray(E2)); ww.append(np.asarray(w2) * wscale)
            El, w = np.concatenate(Ee), np.concatenate(ww)
        dp.m_Zp = ref_mzp
        st_ref = np.array([dp.total_xsec(float(e)) for e in ET_ENGINE])
        sig = np.interp(El, ET_ENGINE, st_ref); ok = sig > 0
        Els.append(El[ok]); Gs.append(w[ok] / sig[ok])
    return np.concatenate(Els), np.concatenate(Gs), dp, product

def rate_over_p2(cacheobj):
    """N_inwindow(m_Zp) / product^2  vs MZP_GRID  (coupling-stripped rate)."""
    El, G, dp, product = cacheobj
    m = (El >= WIN[0]) & (El <= WIN[1]); Elw, Gw = El[m], G[m]
    out = np.empty_like(MZP_GRID)
    for i, mzp in enumerate(MZP_GRID):
        dp.m_Zp = float(mzp)
        st = np.array([dp.total_xsec(float(e)) for e in ET_SCAN])
        out[i] = float(np.sum(Gw * np.interp(Elw, ET_SCAN, st))) / product**2
    return out

import sys
res = {}
for portal, (mb_f, sb_f, ic_f) in {
    "scalar": ("ScalarPortal_MiniBooNE_multichannel.py", "ScalarPortal_SBND_multichannel.py", "ScalarPortal_ICARUS_multichannel.py"),
    "pseudo": ("PseudoscalarPortal_MiniBooNE_multichannel.py", "PseudoscalarPortal_SBND_multichannel.py", "PseudoscalarPortal_ICARUS_multichannel.py"),
}.items():
    print("caching %s: MB ..." % portal); sys.stdout.flush()
    MB = cache(load(mb_f, "MB_"+portal), "sphere", SA._mesons_dk2nu, None, None, "mb", 1.0)
    print("caching %s: SBND ..." % portal); sys.stdout.flush()
    Ssb = load(sb_f, "SB_"+portal)
    SB = cache(Ssb, "box", SA._mesons_dk2nu, None, None, "raw", SINGLE_GAMMA_EFF)     # BNB, SBND_POT
    print("caching %s: ICARUS ..." % portal); sys.stdout.flush()
    Sic = load(ic_f, "IC_"+portal)
    IC = cache(Sic, "box", numi_meson_fn, ICARUS_NUMI_POT, Sic.ICARUS_MODULE_CENTERS_BNB, "raw", SINGLE_GAMMA_EFF)
    print("scanning %s ..." % portal); sys.stdout.flush()
    r_mb = rate_over_p2(MB); r_sb = rate_over_p2(SB); r_ic = rate_over_p2(IC)
    res[portal] = {"N_sbnd": EXCESS * r_sb / r_mb, "N_icarus": EXCESS * r_ic / r_mb}
    print("  %s  SBND N(49/85..)=%.2g..%.2g   ICARUS=%.2g..%.2g"
          % (portal, res[portal]["N_sbnd"][0], res[portal]["N_sbnd"][-1],
             res[portal]["N_icarus"][0], res[portal]["N_icarus"][-1])); sys.stdout.flush()

np.savez(os.path.join(HERE, "output", "reach_sbnd_icarus.npz"), mzp_MeV=MZP_GRID*1e3,
         **{f"{k}_{p}": res[p][k] for p in res for k in res[p]})

fig, ax = plt.subplots(1, 2, figsize=(14, 5.6), sharey=True)
sty = {"scalar": ("#009E73", "Scalar"), "pseudo": ("#5B2C8D", "Pseudo")}
for j, det in enumerate(["N_sbnd", "N_icarus"]):
    for p, (col, lab) in sty.items():
        ax[j].plot(MZP_GRID*1e3, res[p][det], color=col, lw=2.2, label=lab)
    for y, t in [(1, "1 ev"), (3, "3 ev"), (10, "10 ev")]:
        ax[j].axhline(y, color="gray", ls=":", lw=1); ax[j].text(31, y*1.05, t, fontsize=7, color="gray")
    ax[j].set_xscale("log"); ax[j].set_yscale("log")
    ax[j].set_xlabel(r"$m_{Z'}$ [MeV]")
    ax[j].set_title(("SBND (BNB, %.1e POT)" % 6.6e20) if det=="N_sbnd" else ("ICARUS (NuMI, %.1e POT)" % ICARUS_NUMI_POT))
    ax[j].grid(True, which="both", alpha=0.3); ax[j].legend(fontsize=9)
ax[0].set_ylabel("expected signal events\n(if model explains MiniBooNE 320 excess)")
fig.suptitle("SBND / ICARUS reach for the (pseudo)scalar Dark Primakoff, MiniBooNE-anchored  (single-$\\gamma$ eff=%.2f)" % SINGLE_GAMMA_EFF, fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(HERE, "output", "reach_sbnd_icarus.png"), dpi=120)
print("wrote output/reach_sbnd_icarus.{npz,png}")
