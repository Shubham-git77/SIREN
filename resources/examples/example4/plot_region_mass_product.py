"""
68%/95% credible region in the (mediator mass, coupling product) plane, built from
FIXED-MASS coupling fits -- one closed-form fit at every mass on the response grid.

Same style as the brute-grid region plots (region_<portal>_<err>_cos-off.png), but
computed from the fixed-mass machinery: at each mass the coupling enters only as a
normalisation, so chi2 is exactly quadratic in it and the whole 2-D map is a closed
form. No sampler, no profiling over a nuisance mass -- the OTHER masses are held at
the paper's benchmark, which is what "fixed-mass run" means here.

  vector : m_V2 scanned (n38 ridge grid), m_chi=8, m_chi'=50, m_V1=17 MeV fixed.
           y = P = eps1 eps2 g'^2/4pi, response anchored at P0 = 1.3e-7 (Table I).
  pseudo : m_Zp scanned, m_a = 1 MeV fixed.
           y = g_mu g_n lambda [MeV^-1], response anchored at P_ref = 6.5e-7.

Env: PORTAL=vector|pseudo, ERR_MODE=quad|stat, COS=off
Out: output/fixedmass_2026-08-27/region_<portal>_<err>_cos-off.png (+ .npz)
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE)
L = os.path.join(HERE, "output", "fixedmass_2026-08-27")
import miniboone_data as MB

PORTAL = os.environ.get("PORTAL", "pseudo")
ERR_MODE = os.environ.get("ERR_MODE", "quad")
NP_ = int(os.environ.get("N_PROD", "600"))

STAR_Y_LO = None
if PORTAL == "vector":
    r = np.load(os.path.join(L, "ridge_response_n38.npz"))
    MASS = np.asarray(r["MV2_GRID"]) * 1e3
    EH = r["EHIST"]; P_REF = 1.3e-7
    STAR = (200.0, 1.3e-7); STAR_LAB = "paper Table I (200 MeV, 1.3e-07)"
    XL = r"$m_{V_2}$ [MeV]"; YL = r"$\epsilon_1\epsilon_2 g'^2/4\pi$"
    FIXED = r"$m_\chi{=}8$, $m_{\chi'}{=}50$, $m_{V_1}{=}17$ MeV fixed"
    XLIM = (10, 2000)
else:
    r = np.load(os.path.join(L, "respscan_%s_ma1.npz" % PORTAL))
    MASS = np.asarray(r["MZP_MEV"]); EH = r["EHIST"]; P_REF = float(r["P_REF"])
    STAR = (85.0, 5.9e-7); STAR_LAB = "paper Table I (85 MeV, 5.9e-07)"
    XL = r"$m_{Z'}$ [MeV]"; YL = r"$g_\mu g_n \lambda$ [MeV$^{-1}$]"
    FIXED = r"$m_a{=}1$ MeV fixed"
    XLIM = (MASS[0], MASS[-1])

EXCESS = MB.EXCESS; INV2 = 1.0 / MB.errors(ERR_MODE) ** 2

# closed-form best-fit product at every mass
u = np.array([np.sum(EXCESS * s * INV2) / np.sum(s * s * INV2) for s in EH])
PHAT = P_REF * np.sqrt(np.clip(u, 0, None))
# keep the star in frame but do not leave decades of empty space
PROD = np.geomspace(min(PHAT.min() / 2.5, STAR[1] / 2.5),
                    max(PHAT.max() * 2.5, STAR[1] * 2.5), NP_)

# chi2 over the full (mass, product) plane
scale = (PROD[None, :] / P_REF) ** 2                       # (nmass=1, nprod)
CHI2 = np.empty((len(MASS), NP_))
for a, s in enumerate(EH):
    pred = scale * s[:, None, None]                        # (nbin, 1, nprod)
    CHI2[a] = np.einsum("i,ijk->k", INV2, (EXCESS[:, None, None] - pred) ** 2)
D = CHI2 - CHI2.min()
ia, ip = np.unravel_index(np.argmin(CHI2), CHI2.shape)
print("[region-%s] %s errors : chi2_min=%.2f at m=%.1f MeV, P=%.3e (chi2_null=%.2f)"
      % (PORTAL, ERR_MODE, CHI2.min(), MASS[ia], PROD[ip], np.sum(EXCESS ** 2 * INV2)))
# Delta-chi2 at the paper point, computed from the RESPONSE (log-log interpolated
# in mass) and not by interpolating chi2 itself -- chi2 varies steeply here, so
# interpolating it directly over-estimates between grid masses (4.91 vs 2.51 for
# the vector, against the exact 200 MeV response).
_lm = np.log(MASS); _b = np.clip(np.searchsorted(MASS, STAR[0]) - 1, 0, len(MASS) - 2)
_f = np.clip((np.log(STAR[0]) - _lm[_b]) / (_lm[_b + 1] - _lm[_b]), 0, 1)
_s = np.exp((1 - _f) * np.log(np.maximum(EH[_b], 1e-300))
            + _f * np.log(np.maximum(EH[_b + 1], 1e-300)))
_chi2_star = np.sum((EXCESS - _s * (STAR[1] / P_REF) ** 2) ** 2 * INV2)
d_star = _chi2_star - CHI2.min()
print("[region-%s] paper point: chi2=%.2f -> Delta-chi2 = %.2f" % (PORTAL, _chi2_star, d_star))

SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
B68, B95, STARC = "#7fa9dd", "#dbe6f4", "#d1342f"
X, Y = np.meshgrid(MASS, PROD, indexing="ij")
fig, ax = plt.subplots(figsize=(7.6, 6.9))
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
ax.contourf(X, Y, D, levels=[0, 2.30, 6.18], colors=[B68, B95], zorder=2)
ax.contour(X, Y, D, levels=[2.30, 6.18], colors=[INK, INK2],
           linewidths=[1.7, 1.0], zorder=3)
ax.plot(MASS, PHAT, "--", color="#111111", lw=1.8, zorder=4,
        label="best-fit product vs %s" % XL.split("$")[1].join(["$", "$"]))
ax.plot(*STAR, "*", ms=24, color=STARC, mec=SURF, mew=1.0, zorder=6, label=STAR_LAB)

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(*XLIM); ax.set_ylim(PROD[0], PROD[-1])
ax.set_xlabel(XL, fontsize=11.5, color=INK); ax.set_ylabel(YL, fontsize=11.5, color=INK)
ax.set_title("%s: 68%%/95%% region from FIXED-MASS coupling fits\n"
             "%s errors; cos-theta dropped\n%s"
             % (PORTAL, "stat + bkg syst (quadrature)" if ERR_MODE == "quad" else "stat",
                FIXED),
             fontsize=10.5, color=INK, pad=10)
ax.grid(True, which="major", color="#e0dfda", lw=0.8, zorder=0)
ax.grid(True, which="minor", color="#eeede9", lw=0.5, zorder=0)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"): ax.spines[sp].set_color("#c9c8c2")
ax.tick_params(colors=INK2, labelsize=9.5)
ax.legend(frameon=True, facecolor=SURF, edgecolor="#c9c8c2", fontsize=9.5,
          loc="upper left", labelcolor=INK2)
_ds = "%.2f" % d_star if d_star < 1e3 else "%.1e" % d_star
_edge = ("\nEDGE-LIMITED: the minimum sits on the %s mass boundary — the region is bounded "
         "by the grid, not the data." % ("lower" if ia == 0 else "upper")
         if ia in (0, len(MASS) - 1) else "")
fig.text(0.012, 0.012,
         "At each mass the coupling enters only as a normalisation, so $\\chi^2$ is exactly quadratic in it "
         "and the map is a\nclosed form. Paper point $\\Delta\\chi^2$ = %s. Levels are 2-parameter "
         "(2.30 / 6.18).%s" % (_ds, _edge),
         fontsize=7.6, color=MUTED, ha="left", va="bottom", linespacing=1.5)
fig.subplots_adjust(left=0.145, right=0.965, top=0.855, bottom=0.175)
out = os.path.join(L, "region_%s_%s_cos-off.png" % (PORTAL, ERR_MODE))
fig.savefig(out, dpi=150, facecolor=SURF)
np.savez(os.path.join(L, "region_%s_%s_cos-off.npz" % (PORTAL, ERR_MODE)),
         mass_MeV=MASS, prod=PROD, chi2=CHI2, dchi2=D, phat=PHAT,
         P_REF=P_REF, err_mode=ERR_MODE, dchi2_paper=d_star)
print("wrote", out)
