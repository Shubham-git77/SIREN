"""
FIXED-MASS COUPLING-PLANE SCAN for the vector double-mediator MiniBooNE fit.

Freeze m_V2, then scan the coupling plane directly and draw Delta-chi2 contours.
No sampler: at fixed mass the rate is an exact closed form in the couplings,

    rate  ~  eps1^2 * eps2^2 * alpha_D          (VERIFICATION.md, exponents exact
                                                 to 0.01)

so chi2 is quadratic in that combination and the whole 2-D region is evaluated on
a grid in seconds. This is prior-free -- no MCMC convergence question, none of the
seeding traps that made the marginal posteriors unreliable.

Plane: (log10 eps1, log10 eps2), with alpha_D PROFILED at every grid point over
[ALPHAD_MIN, ALPHAD_MAX]. Because only the product combination enters the rate,
the allowed region is a hyperbolic band, and profiling alpha_D widens it by the
allowed alpha_D range -- that band shape IS the degeneracy, not a plotting artefact.

Why this plane: NA64/BaBar constrain eps1 and eps2 INDIVIDUALLY, so the exclusion
is a pair of straight lines here and can be read off directly against the band.

Env: MV2 (MeV, default 200 = paper Table I), ERR_MODE=stat|quad, NGRID
Run: /home/shubham/siren_venv/bin/python scan_fixed_mass_couplings.py
Out: output/fixedmass_2026-08-27/coupling_plane_vector_m<MV2>_<ERR_MODE>.{png,npz}
"""
import os, sys, importlib.util
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
sys.path.insert(0, HERE)
L = os.path.join(HERE, "output", "fixedmass_2026-08-27")
import miniboone_data as MB

MV2 = float(os.environ.get("MV2", "200"))
ERR_MODE = os.environ.get("ERR_MODE", "stat")
NG = int(os.environ.get("NGRID", "420"))
ALPHAD_MIN, ALPHAD_MAX = 1e-3, 0.5   # 0.5 = the paper's stated convention;
                                     # 1e-3 ~ the tableI_fit split's 2.2e-3

# ---- benchmark couplings the response grid was simulated at ----
_spec = importlib.util.spec_from_file_location("S_ref", "VectorPortal_MiniBooNE_fullchain.py")
_S = importlib.util.module_from_spec(_spec); sys.modules["S_ref"] = _S
_spec.loader.exec_module(_S)
EPS1_0, EPS2_0 = float(_S.EPSILON_1), float(_S.EPSILON_2)
ALPHAD_0 = float(_S.G_D) ** 2 / (4.0 * np.pi)
print("[cp] benchmark split: eps1=%.4e eps2=%.4e alpha_D=%.4e" % (EPS1_0, EPS2_0, ALPHAD_0))

# ---- response at the requested fixed mass (11-bin HEPData; no simulation) ----
r7 = np.load(os.path.join(L, "fixedmass_response_d100_31.4_60_100_150_200_300_500.npz"))
m7 = np.asarray(r7["MV2_GRID"]) * 1e3
j = int(np.argmin(np.abs(m7 - MV2)))
assert abs(m7[j] - MV2) < 1e-6, "no exact response at %.1f MeV (have %s)" % (MV2, m7)
s = r7["EHIST"][j]
EXCESS = MB.EXCESS; INV2 = 1.0 / MB.errors(ERR_MODE) ** 2
u_hat = np.sum(EXCESS * s * INV2) / np.sum(s * s * INV2)
print("[cp] m_V2=%.1f MeV : %.1f predicted events at the benchmark split, u_hat=%.4g"
      % (MV2, s.sum(), u_hat))

# ---- scan the plane, profiling alpha_D ----
lg1 = np.linspace(-6.0, -1.0, NG)
lg2 = np.linspace(-6.0, -1.0, NG)
X, Y = np.meshgrid(lg1, lg2, indexing="ij")
ratio = (10 ** X / EPS1_0) ** 2 * (10 ** Y / EPS2_0) ** 2      # scale without alpha_D
# chi2 is minimised where scale == u_hat -> solve for alpha_D, then clip to range
aD_star = np.clip(u_hat * ALPHAD_0 / np.maximum(ratio, 1e-300), ALPHAD_MIN, ALPHAD_MAX)
scale = ratio * (aD_star / ALPHAD_0)
CHI2 = np.einsum("i,ijk->jk", INV2,
                 (EXCESS[:, None, None] - scale[None, :, :] * s[:, None, None]) ** 2)
D = CHI2 - CHI2.min()
print("[cp] chi2_min=%.2f  chi2_null=%.2f" % (CHI2.min(), np.sum(EXCESS ** 2 * INV2)))

# ---- figure ----
SURF, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
BAND68, BAND95 = "#2a78d6", "#a8c8ee"     # sequential: one hue, light->dark
LIMIT = "#b3261e"
fig, ax = plt.subplots(figsize=(7.8, 7.0))
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
ax.contourf(X, Y, D, levels=[0, 2.30, 5.99], colors=[BAND68, BAND95], zorder=2)
ax.contour(X, Y, D, levels=[2.30, 5.99], colors=[INK, INK2],
           linewidths=[1.6, 1.0], zorder=3)

# existing limits act on eps1 and eps2 individually -> straight lines here
NA64_17 = np.log10(2.9e-5)      # digitised NA64 2023 Fig.3 at m = m_V1 = 17 MeV
BABAR   = np.log10(1.0e-3)      # BaBar invisible, the flat line on that figure
ax.axvline(NA64_17, color=LIMIT, lw=2, zorder=5)
ax.axhline(BABAR, color=LIMIT, lw=2, ls="--", zorder=5)
ax.fill_betweenx([-6, -1], NA64_17, -1, color=LIMIT, alpha=0.06, lw=0, zorder=1)
ax.fill_between([-6, -1], BABAR, -1, color=LIMIT, alpha=0.06, lw=0, zorder=1)
# the box that survives BOTH limits
ax.add_patch(plt.Rectangle((-6, -6), NA64_17 + 6, BABAR + 6, fill=False,
                           ec='#1b7f4d', lw=2.2, zorder=5))
ax.text(-5.9, BABAR - 0.22, 'allowed by NA64 + BaBar', color='#1b7f4d',
        fontsize=9.5, ha='left', va='top', weight='bold')
ax.text(NA64_17 + 0.09, -1.55, "NA64 2023\n$\\epsilon_1\\leq2.9{\\times}10^{-5}$ at $m_{V_1}{=}17$ MeV",
        color=LIMIT, fontsize=9, ha="left", va="top")
ax.text(-1.05, BABAR + 0.10, "BaBar invisible: $\\epsilon_2\\leq10^{-3}$", color=LIMIT,
        fontsize=9, ha="right", va="bottom")

ax.plot(np.log10(7e-5), np.log10(1e-4), "*", ms=20, color="#111111", mec=SURF, mew=1.2,
        zorder=6, label="paper Table II split (7e-5, 1e-4)")
ax.plot(np.log10(EPS1_0), np.log10(EPS2_0), "o", ms=10, color="#eb6834", mec=SURF, mew=1.5,
        zorder=6, label="config `tableI_fit` split (7.6e-3, 7.6e-3)")
ax.plot([], [], "s", ms=11, color=BAND68, label="68% CL ($\\Delta\\chi^2=2.30$)")
ax.plot([], [], "s", ms=11, color=BAND95, label="95% CL ($\\Delta\\chi^2=5.99$)")

ax.set_xlim(-6.0, -1.0); ax.set_ylim(-6.0, -1.0)
ax.set_xlabel(r"$\log_{10}\epsilon_1$   (mixing of $V_1$, $m=17$ MeV)", fontsize=11, color=INK)
ax.set_ylabel(r"$\log_{10}\epsilon_2$   (mixing of $V_2$, $m=%.0f$ MeV)" % MV2,
              fontsize=11, color=INK)
ax.set_title("Vector double mediator: coupling plane at FIXED $m_{V_2}=%.0f$ MeV\n"
             r"MiniBooNE $\nu$-mode $E_{vis}$, %s errors, $\alpha_D$ profiled over "
             r"[%g, %g]" % (MV2, ERR_MODE, ALPHAD_MIN, ALPHAD_MAX),
             fontsize=12, color=INK, loc="left", pad=12)
ax.grid(True, color="#e3e2dd", lw=0.7, zorder=0)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"): ax.spines[sp].set_color("#c9c8c2")
ax.tick_params(colors=INK2, labelsize=9)
ax.legend(frameon=False, fontsize=8.8, loc="lower right", labelcolor=INK2,
          borderaxespad=0.8)
fig.text(0.01, 0.012,
         "Only the combination $\\epsilon_1^2\\epsilon_2^2\\alpha_D$ enters the rate, so the allowed set is a "
         "hyperbolic BAND, not a blob: the fit pins the\nproduct and says nothing about the split. The limits "
         "act on $\\epsilon_1$ and $\\epsilon_2$ separately — the band must reach the box below BOTH lines.\n"
         "NA64 at $m_{V_2}=200$ MeV was not digitised, so the $\\epsilon_2$ line shown is BaBar, the weaker bound.",
         fontsize=7.6, color=MUTED, ha="left", va="bottom", linespacing=1.5)
I, J = np.meshgrid(lg1, lg2, indexing="ij")
band_lo = (I + J)[D < 2.30].min()
best_allowed = NA64_17 + BABAR                       # top corner of the allowed box
gap_dec = band_lo - best_allowed                     # decades in lg(eps1*eps2)
print("[cp] 68%% band needs lg(e1*e2) >= %.2f ; limits allow at most %.2f -> gap %.2f decades"
      % (band_lo, best_allowed, gap_dec))
print("[cp] shortfall in RATE (~eps^4) = %.3g" % 10 ** (2 * gap_dec))
ax.annotate("", xy=(NA64_17, BABAR), xytext=(band_lo / 2, band_lo / 2),
            arrowprops=dict(arrowstyle="<->", color=INK, lw=1.6), zorder=7)
ax.text(-4.15, -3.05, "gap: %.1f decades in $\\epsilon_1\\epsilon_2$\n"
        "$\\Rightarrow$ %.0e too little RATE" % (abs(gap_dec), 10 ** (2 * abs(gap_dec))),
        fontsize=9.5, color=INK, ha="center", va="center", weight="bold",
        bbox=dict(fc=SURF, ec="none", alpha=0.85, pad=2.5), zorder=8)
fig.subplots_adjust(left=0.115, right=0.975, top=0.885, bottom=0.175)
tag = "m%.0f_%s" % (MV2, ERR_MODE)
out = os.path.join(L, "coupling_plane_vector_%s.png" % tag)
fig.savefig(out, dpi=150, facecolor=SURF)
np.savez(os.path.join(L, "coupling_plane_vector_%s.npz" % tag),
         lg_eps1=lg1, lg_eps2=lg2, chi2=CHI2, dchi2=D, alphaD_profiled=aD_star,
         mv2=MV2, err_mode=ERR_MODE, u_hat=u_hat, s=s)
print("wrote", out)
