"""
#2 warm-up: coupling-vs-mass FIT CURVE for the (pseudo)scalar Dark Primakoff.

For each Z' mass, find the coupling product g_mu*g_n*lambda [MeV^-1] that makes the
MiniBooNE in-window [0.14,0.30] muon-only prediction equal the 320-event excess.

CACHED design: production/decay kinematics + geometry are m_Zp-INDEPENDENT, so we
run the analytic engine ONCE per channel to get per-hit (E_vis, geometry-weight
G = w / sigma_ref(E_vis)), then for each m_Zp only re-evaluate the cross-section
sigma(m_Zp, E_vis) and re-weight.  Rate ~ (product)^2, so the fit product at each
mass is closed-form:  product_fit(m_Zp) = sqrt(320 / R(m_Zp)),
R(m_Zp) = sum_inwindow[ G * sigma(m_Zp,E) ] / product_ref^2.

Run:  DK2NU_FILE=/home/shubham/nubeamHighSample.dk2nu.root \
      /home/shubham/siren_venv/bin/python scan_fit_product.py
Outputs: output/fit_product_vs_mZp.{npz,png}
"""
import os, importlib.util, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

os.environ.setdefault("DK2NU_FILE", "/home/shubham/nubeamHighSample.dk2nu.root")
HERE = os.path.dirname(os.path.abspath(__file__))
PKG  = os.environ.get(
    "SIREN_DNT_DIR",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "processes", "DarkNewsTables",
    ),
)

def load(path, name):
    import sys
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m

SA = load(os.path.join(PKG, "AnalyticRate.py"), "AnalyticRate")

WIN    = (0.140, 0.300)          # GeV, MiniBooNE single-photon window
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
EXCESS = 320.0 if _ANCHOR == "paper320" else float(_MB.EXCESS[0])   # events
NDEC   = 400
MESON  = SA._mesons_dk2nu        # real dk2nu flux (cached inside the process)
# engine's own sigma-table energy grid (must match to strip sigma_ref exactly):
ET_ENGINE = np.concatenate([np.linspace(0.001, 0.3, 120), np.linspace(0.31, 9, 160)])
# focused grid for the scan (only in-window E_vis is summed):
ET_SCAN = np.linspace(0.13, 0.32, 60)
MZP_GRID = np.geomspace(0.030, 0.200, 30)   # GeV, paper Fig.3 x-range 30-200 MeV

def cache_production(S):
    """Run the engine once per muon channel; return (E_vis, G) per hit and the dp
    object + reference product.  G = w / sigma_ref(E_vis) strips the cross-section."""
    ref_mzp = S.M_ZP
    product_ref = S.G_MU_PROD * S.G_N * (S.LAMBDA * 1e-3)     # MeV^-1
    dp = None; Els = []; Gs = []
    for ch in [c for c in S.CHANNELS if "mu" in c]:
        pdg, m_M, m_l, lpdg, nupdg, gsm = S.CHANNELS[ch]
        chm = S.build_onshell_models(pdg, m_M, m_l, lpdg, nupdg)
        dp = chm["models"]["primakoff"]._dp                  # same carbon dp for all channels
        El, w = SA.analytic_sp_mb(S, ch, n_dec=NDEC, eff_mode="mb", meson_fn=MESON)
        El, w = np.asarray(El), np.asarray(w)
        dp.m_Zp = ref_mzp
        st_ref = np.array([dp.total_xsec(float(e)) for e in ET_ENGINE])
        sig_ref = np.interp(El, ET_ENGINE, st_ref)
        good = sig_ref > 0
        Els.append(El[good]); Gs.append(w[good] / sig_ref[good])
    return np.concatenate(Els), np.concatenate(Gs), dp, product_ref

def scan(El, G, dp, product_ref):
    inwin = (El >= WIN[0]) & (El <= WIN[1])
    Elw, Gw = El[inwin], G[inwin]
    fit = np.empty_like(MZP_GRID)
    for i, mzp in enumerate(MZP_GRID):
        dp.m_Zp = float(mzp)
        st = np.array([dp.total_xsec(float(e)) for e in ET_SCAN])
        sig = np.interp(Elw, ET_SCAN, st)
        N = float(np.sum(Gw * sig))                           # events at product_ref
        R = N / product_ref**2
        fit[i] = np.sqrt(EXCESS / R) if R > 0 else np.nan
    return fit

print("caching scalar production ..."); import sys; sys.stdout.flush()
Ssc = load(os.path.join(HERE, "ScalarPortal_MiniBooNE_multichannel.py"), "Ssc")
El_s, G_s, dp_s, pref_s = cache_production(Ssc)
print("caching pseudo production ..."); sys.stdout.flush()
Sps = load(os.path.join(HERE, "PseudoscalarPortal_MiniBooNE_multichannel.py"), "Sps")
El_p, G_p, dp_p, pref_p = cache_production(Sps)

print("scanning m_Zp ..."); sys.stdout.flush()
fit_s = scan(El_s, G_s, dp_s, pref_s)
fit_p = scan(El_p, G_p, dp_p, pref_p)

# self-check at each config's own m_Zp (should ~reproduce today's validated fit products)
def at(mzp_grid, fit, mzp): return float(np.interp(mzp, mzp_grid, fit))
print("\nSELF-CHECK (fit product at config m_Zp):")
print("  scalar @ m_Zp=49 MeV : %.3e MeV^-1  (paper Table I 2.2e-8)" % at(MZP_GRID, fit_s, 0.049))
print("  pseudo @ m_Zp=85 MeV : %.3e MeV^-1  (today's validated fit ~1.3e-7)" % at(MZP_GRID, fit_p, 0.085))

out_npz = os.path.join(HERE, "output", "fit_product_vs_mZp.npz")
np.savez(out_npz, mzp_MeV=MZP_GRID*1e3, fit_scalar=fit_s, fit_pseudo=fit_p)
print("wrote", out_npz)

fig, ax = plt.subplots(figsize=(7.5, 6))
ax.plot(MZP_GRID*1e3, fit_s, color="#009E73", lw=2.2, label="Scalar  ($m_\\phi=1$ MeV)")
ax.plot(MZP_GRID*1e3, fit_p, color="#5B2C8D", lw=2.2, label="Pseudoscalar  ($m_a=1$ MeV)")
# paper Table I benchmark points + today's validated pseudo fit
ax.scatter([49], [2.2e-8], color="#009E73", marker="*", s=180, zorder=5, edgecolor="k",
           label="paper Table I scalar (2.2e-8)")
ax.scatter([85], [5.9e-7], color="#5B2C8D", marker="*", s=180, zorder=5, edgecolor="k",
           label="paper Table I pseudo (5.9e-7)")
ax.scatter([85], [1.3e-7], color="#D55E00", marker="o", s=90, zorder=5, edgecolor="k",
           label="our validated pseudo fit (1.3e-7)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$m_{Z'}$ [MeV]"); ax.set_ylabel(r"fit product $g_\mu g_n \lambda$ [MeV$^{-1}$]  (gives 320 in-window)")
ax.set_title("MiniBooNE Dark-Primakoff: coupling product that fits the 320 excess\n(cached scan, muon-only, real dk2nu)")
ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
fig.tight_layout()
out_png = os.path.join(HERE, "output", "fit_product_vs_mZp.png")
fig.savefig(out_png, dpi=120); print("wrote", out_png)
