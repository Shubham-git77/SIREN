"""
#1: MiniBooNE credible regions (Dutta-Kim Fig.3 clone) for the (pseudo)scalar.

For each (m_Zp, coupling product P), predict the MiniBooNE nu-mode E_vis signal
over the 19 digitized data bins, form chi2 against the digitized excess (data-bkg,
per-bin errors from fig2_paperstyle), and draw Delta-chi2 = 2.30 (68%) / 6.18 (95%)
contours in the (m_Zp, P) plane, overlaid on the #2 fit lines.

Signal shape per m_Zp comes from the #2 cached hits; normalization scales as P^2
(N ~ (g_mu g_n lambda)^2).  Digitized data/bkg = fig2_paperstyle (nu-mode E_vis;
black points are REAL MiniBooNE data, tan is MiniBooNE bkg -- from Dutta-Kim Fig.2).

Run:  DK2NU_FILE=/home/shubham/nubeamHighSample.dk2nu.root \
      /home/shubham/siren_venv/bin/python scan_credible_region.py
Outputs: output/credible_region_mZp_product.{npz,png}
"""
import os, importlib.util, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

os.environ.setdefault("DK2NU_FILE", "/home/shubham/nubeamHighSample.dk2nu.root")
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

# ---- digitized MiniBooNE nu-mode E_vis (same as fig2_paperstyle) ----
# MiniBooNE nu-mode data now comes from the shared module. It used to be a
# 19-bin digitization inlined here, which matched no official release and
# silently diverged from the corrected 11-bin HEPData binning used by
# scan_brute_grid.py / mcmc_fit.py -- results from the two were not
# comparable. ERR_MODE=quad adds the MiniBooNE background systematics the
# paper says it used; ERR_MODE=stat (default) keeps the historical
# stat-only weighting.
import os as _os
import miniboone_data as MB
DATA_E, DATA_N, DATA_ERR, BKG, EBINS = MB.DATA_E, MB.DATA_N, MB.DATA_ERR, MB.BKG, MB.EBINS
ERR_MODE = _os.environ.get("ERR_MODE", "stat")
EXCESS = MB.EXCESS
INV2 = 1.0 / MB.errors(ERR_MODE) ** 2
PBINS = np.concatenate([[DATA_E[0]-25], DATA_E+25]) / 1e3   # GeV edges 0.200..1.150

NDEC = 400
ET_ENGINE = np.concatenate([np.linspace(0.001, 0.3, 120), np.linspace(0.31, 9, 160)])
ET_SIG    = np.linspace(0.13, 1.60, 90)                 # sigma grid over the full data-bin range
MZP_GRID  = np.geomspace(0.030, 0.200, 28)
PROD_GRID = np.geomspace(3e-9, 1.2e-6, 90)              # MeV^-1

def cache(S):
    """All muon-channel hits (E_vis, G=w/sigma_ref) + dp + reference product."""
    product = S.G_MU_PROD * S.G_N * (S.LAMBDA * 1e-3)
    ref_mzp = S.M_ZP; Els=[]; Gs=[]; dp=None
    for ch in [c for c in S.CHANNELS if "mu" in c]:
        pdg, m_M, m_l, lpdg, nupdg, gsm = S.CHANNELS[ch]
        dp = S.build_onshell_models(pdg, m_M, m_l, lpdg, nupdg)["models"]["primakoff"]._dp
        El, w = SA.analytic_sp_mb(S, ch, n_dec=NDEC, eff_mode="mb", meson_fn=SA._mesons_dk2nu)
        El, w = np.asarray(El), np.asarray(w)
        dp.m_Zp = ref_mzp
        sig = np.interp(El, ET_ENGINE, np.array([dp.total_xsec(float(e)) for e in ET_ENGINE]))
        ok = sig > 0; Els.append(El[ok]); Gs.append(w[ok] / sig[ok])
    return np.concatenate(Els), np.concatenate(Gs), dp, product

def shape_bins(cacheobj, mzp):
    """Signal histogram over the 19 data bins at product_ref, at this m_Zp."""
    El, G, dp, product = cacheobj
    dp.m_Zp = float(mzp)
    sig = np.interp(El, ET_SIG, np.array([dp.total_xsec(float(e)) for e in ET_SIG]))
    return np.histogram(El, bins=PBINS, weights=G * sig)[0]

def chi2_grid(cacheobj):
    """chi2 over the (MZP_GRID, PROD_GRID) plane."""
    El, G, dp, product = cacheobj
    C = np.empty((len(MZP_GRID), len(PROD_GRID)))
    for i, mzp in enumerate(MZP_GRID):
        s_ref = shape_bins(cacheobj, mzp)                 # at product
        for j, P in enumerate(PROD_GRID):
            s = s_ref * (P / product)**2                  # scale N ~ P^2
            C[i, j] = np.sum((EXCESS - s)**2 * INV2)
    return C

import sys
CFG = {"scalar": "ScalarPortal_MiniBooNE_multichannel.py",
       "pseudo": "PseudoscalarPortal_MiniBooNE_multichannel.py"}
grids = {}; fitlines = {}
chi2_null = np.sum(EXCESS**2 * INV2)
for portal, f in CFG.items():
    print("caching + scanning %s ..." % portal); sys.stdout.flush()
    cob = cache(load(f, "C_"+portal))
    C = chi2_grid(cob); grids[portal] = C
    imin = np.unravel_index(np.argmin(C), C.shape)
    # best-fit product per m_Zp (the #2 spine, via chi2 min over P)
    fitlines[portal] = PROD_GRID[np.argmin(C, axis=1)]
    # dof was hardcoded 18 from the superseded 19-bin digitisation; the official
    # release has 11 bins and this fit floats (m_Zp, product), so it is 11-2 = 9.
    print("  %s  chi2_min=%.2f at m_Zp=%.0f MeV, P=%.2e  (null %.1f, %d dof)"
          % (portal, C.min(), MZP_GRID[imin[0]]*1e3, PROD_GRID[imin[1]], chi2_null,
             len(MB.DATA_N)-2)); sys.stdout.flush()

np.savez(os.path.join(HERE, "output", "credible_region_mZp_product.npz"),
         mzp_MeV=MZP_GRID*1e3, prod=PROD_GRID,
         chi2_scalar=grids["scalar"], chi2_pseudo=grids["pseudo"])

fig, ax = plt.subplots(figsize=(8, 6.4))
X, Y = np.meshgrid(MZP_GRID*1e3, PROD_GRID, indexing="ij")
sty = {"scalar": ("#009E73", "Scalar"), "pseudo": ("#5B2C8D", "Pseudoscalar")}
for portal, (col, lab) in sty.items():
    C = grids[portal]; D = C - C.min()
    ax.contourf(X, Y, D, levels=[0, 6.18], colors=[col], alpha=0.18)     # 95%
    ax.contourf(X, Y, D, levels=[0, 2.30], colors=[col], alpha=0.30)     # 68%
    ax.contour(X, Y, D, levels=[2.30, 6.18], colors=col, linewidths=[1.6, 0.9])
    ax.plot(MZP_GRID*1e3, fitlines[portal], color=col, lw=1.2, ls="--", label="%s best-fit" % lab)
# paper Table I benchmarks
ax.scatter([49], [2.2e-8], color="#009E73", marker="*", s=200, edgecolor="k", zorder=6, label="paper scalar (2.2e-8)")
ax.scatter([85], [5.9e-7], color="#5B2C8D", marker="*", s=200, edgecolor="k", zorder=6, label="paper pseudo (5.9e-7)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel(r"$m_{Z'}$ [MeV]"); ax.set_ylabel(r"$g_\mu g_n \lambda$ [MeV$^{-1}$]")
ax.set_title("MiniBooNE credible regions (Fig.3 clone), $\\nu$-mode $E_{vis}$ only\n68%% (dark) / 95%% (light) from $\\Delta\\chi^2$; digitized data")
ax.set_xlim(30, 200); ax.set_ylim(3e-9, 1.2e-6)
ax.grid(True, which="both", alpha=0.25); ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "output", "credible_region_mZp_product.png"), dpi=130)
print("wrote output/credible_region_mZp_product.{npz,png}")
