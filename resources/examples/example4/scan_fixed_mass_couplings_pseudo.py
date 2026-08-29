"""
FIXED-MASS COUPLING-PLANE SCAN for the (pseudo)scalar Dark-Primakoff MiniBooNE fit
-- the pseudoscalar twin of scan_fixed_mass_couplings.py.

Freeze (m_a, m_Zp) at the paper's Table I benchmark, then scan the coupling plane
directly. All three couplings enter the rate only through the single product

    rate  ~  (g_mu * g_n * lambda)^2        (VERIFICATION.md: all exponents 2.00)

so chi2 is quadratic in that product and the region is a closed form on a grid.
Plane: (log10 g_mu, log10 g_n) with lambda PROFILED over [LAM_MIN, LAM_MAX].

Needs the response from build_pseudo_response_fixed.py (11-bin HEPData binning).

Env: PORTAL=pseudo|scalar, ERR_MODE=stat|quad, NGRID
Out: output/fixedmass_2026-08-27/coupling_plane_<portal>_<tag>.{png,npz}
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
ERR_MODE = os.environ.get("ERR_MODE", "stat")
NG = int(os.environ.get("NGRID", "420"))
LAM_MIN, LAM_MAX = 1e-4, 1e-1                 # MeV^-1 ; config benchmark = 6.5e-3

r = np.load(os.path.join(L, "resp_%s_mzp85_ma1.npz" % PORTAL))
s = r["EHIST"]; P_REF = float(r["P_REF"])
G_MU_0, G_N_0, LAM_0 = float(r["g_mu"]), float(r["g_n"]), float(r["lam_GeV"]) * 1e-3
EXCESS = MB.EXCESS; INV2 = 1.0 / MB.errors(ERR_MODE) ** 2
u_hat = np.sum(EXCESS * s * INV2) / np.sum(s * s * INV2)
P_hat = P_REF * np.sqrt(max(u_hat, 0.0))
print("[cp-%s] m_Zp=%.0f MeV m_a=%.0f MeV : %.1f events at P_ref=%.3e"
      % (PORTAL, r["m_zp"] * 1e3, r["m_phi"] * 1e3, s.sum(), P_REF))
print("[cp-%s] u_hat=%.4g -> P_hat=%.4e MeV^-1  (paper Table I: 5.9e-07, ratio %.2f)"
      % (PORTAL, u_hat, P_hat, P_hat / 5.9e-7))

# ---- scan (g_mu, g_n), profiling lambda ----
lgmu = np.linspace(-5.0, -0.5, NG)
lgn = np.linspace(-5.0, -0.5, NG)
X, Y = np.meshgrid(lgmu, lgn, indexing="ij")
gg = 10 ** X * 10 ** Y                                   # g_mu * g_n


def dchi2(scale):
    c = np.einsum("i,ijk->jk", INV2,
                  (EXCESS[:, None, None] - scale[None, :, :] * s[:, None, None]) ** 2)
    return c, c - c.min()


# PRIMARY band: lambda FIXED at the paper's own benchmark. Profiling lambda over
# three decades makes the band three decades wide and lets almost any (g_mu, g_n)
# be compensated -- true, but it hides the comparison the figure is for.
CHI2, D = dchi2((gg * LAM_0 / P_REF) ** 2)
# CONTEXT envelope: the same region with lambda free over [LAM_MIN, LAM_MAX].
lam_star = np.clip(np.sqrt(max(u_hat, 0.0)) * P_REF / np.maximum(gg, 1e-300), LAM_MIN, LAM_MAX)
_, D_prof = dchi2((gg * lam_star / P_REF) ** 2)
print("[cp-%s] chi2_min=%.2f  chi2_null=%.2f" % (PORTAL, CHI2.min(), np.sum(EXCESS ** 2 * INV2)))

SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
BAND68, BAND95, LIMIT = "#2a78d6", "#a8c8ee", "#b3261e"
fig, ax = plt.subplots(figsize=(7.8, 7.0))
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
ax.contourf(X, Y, D_prof, levels=[0, 5.99], colors=["#dfe9f7"], zorder=1.5)
ax.contour(X, Y, D_prof, levels=[5.99], colors=[INK2], linewidths=0.9,
           linestyles="--", zorder=3)
ax.contourf(X, Y, D, levels=[0, 2.30, 5.99], colors=[BAND68, BAND95], zorder=2)
ax.contour(X, Y, D, levels=[2.30, 5.99], colors=[INK, INK2], linewidths=[1.6, 1.0], zorder=3)

# g_mu ceiling from the paper's OWN Table II: K -> mu nu a, BR ~ g_mu^2.
# predicted 1.00e-6 at g_mu=1e-2 against the 90% CL limit (parenthesised value
# 3.0e-6 is the tighter of the two quoted).
GMU_MAX_LOOSE = G_MU_0 * np.sqrt(2.0e-5 / 1.00e-6)
GMU_MAX_TIGHT = G_MU_0 * np.sqrt(3.0e-6 / 1.00e-6)
for v, ls, lab in ((np.log10(GMU_MAX_TIGHT), "-", "Table II $K\\to\\mu\\nu a$ (tight, %.1e)" % GMU_MAX_TIGHT),
                   (np.log10(GMU_MAX_LOOSE), "--", "Table II (loose, %.1e)" % GMU_MAX_LOOSE)):
    ax.axvline(v, color=LIMIT, lw=2, ls=ls, zorder=5)
ax.fill_betweenx([-5, -0.5], np.log10(GMU_MAX_LOOSE), -0.5, color=LIMIT, alpha=0.06, lw=0, zorder=1)
ax.text(-0.58, -0.58,
        "$g_\\mu$ ceilings, paper's own Table II\n($BR\\sim g_\\mu^2$; $g_n,\\lambda$ not constrained there)",
        color=LIMIT, fontsize=8.8, ha="right", va="top")

ax.plot(np.log10(G_MU_0), np.log10(G_N_0), "*", ms=22, color="#111111", mec=SURF, mew=1.2,
        zorder=6, label="paper Table I benchmark ($10^{-2}$, $10^{-2}$)")
ax.plot([], [], "s", ms=11, color=BAND68, label="68% CL ($\\Delta\\chi^2=2.30$)")
ax.plot([], [], "s", ms=11, color=BAND95, label="95% CL ($\\Delta\\chi^2=5.99$)")

# where the paper's own benchmark lambda would put you
lam_paper = LAM_0
gg_needed = P_hat / lam_paper
ax.plot([], [], "s", ms=11, color="#dfe9f7", label="95% CL if $\\lambda$ free over 3 decades")

ax.set_xlim(-5.0, -0.5); ax.set_ylim(-5.0, -0.5)
ax.set_xlabel(r"$\log_{10} g_\mu$   ($a$–muon, sets production)", fontsize=11, color=INK)
ax.set_ylabel(r"$\log_{10} g_n$   ($Z'$–nucleon, sets scattering)", fontsize=11, color=INK)
ax.set_title("Pseudoscalar Dark Primakoff: coupling plane at FIXED "
             "$m_{Z'}=85$, $m_a=1$ MeV\n"
             r"MiniBooNE $\nu$-mode $E_{vis}$, %s errors, $\lambda$ FIXED at the "
             r"paper's 6.5 GeV$^{-1}$" % ERR_MODE,
             fontsize=12, color=INK, loc="left", pad=12)
ax.grid(True, color="#e3e2dd", lw=0.7, zorder=0)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"): ax.spines[sp].set_color("#c9c8c2")
ax.tick_params(colors=INK2, labelsize=9)
ax.legend(frameon=False, fontsize=8.8, loc="lower left", labelcolor=INK2, borderaxespad=0.8)
fig.text(0.01, 0.012,
         "Only the product $g_\\mu g_n\\lambda$ enters the rate, so the allowed set is a hyperbolic BAND, not a blob.\n"
         "At the paper's own $\\lambda$ our best fit needs $P=%.2e$ MeV$^{-1}$ vs their Table I $5.9\\times10^{-7}$ — a factor\n"
         "%.1f in coupling, %.0f in RATE — so their benchmark sits ABOVE the band. Let $\\lambda$ float 3 decades (pale band)\n"
         "and the point is reachable again: it is the PRODUCT that is too large, not any one coupling."
         % (P_hat, 5.9e-7 / P_hat, (5.9e-7 / P_hat) ** 2),
         fontsize=7.6, color=MUTED, ha="left", va="bottom", linespacing=1.5)
fig.subplots_adjust(left=0.115, right=0.975, top=0.875, bottom=0.20)
tag = "mzp85_%s" % ERR_MODE
out = os.path.join(L, "coupling_plane_%s_%s.png" % (PORTAL, tag))
fig.savefig(out, dpi=150, facecolor=SURF)
np.savez(os.path.join(L, "coupling_plane_%s_%s.npz" % (PORTAL, tag)),
         lg_gmu=lgmu, lg_gn=lgn, chi2=CHI2, dchi2=D, lam_profiled=lam_star,
         u_hat=u_hat, P_hat=P_hat, P_REF=P_REF, err_mode=ERR_MODE)
print("wrote", out)
