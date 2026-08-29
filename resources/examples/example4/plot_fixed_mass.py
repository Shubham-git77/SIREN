"""
Plot the FIXED-MASS coupling fit (fit_fixed_mass.py) against Dutta-Kim Table I.

Two stacked panels sharing the mass axis -- deliberately NOT a dual-axis chart,
since P_hat and chi2/dof have unrelated scales.
  top    : P_hat(m_V2) with its 68% CL band, stat and quad, plus the paper's two
           Table I vector benchmarks.
  bottom : chi2/dof of the same fits.

Run: /home/shubham/siren_venv/bin/python plot_fixed_mass.py
Out: output/fixedmass_2026-08-27/fixed_mass_coupling.png
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
L = os.path.join(HERE, "output", "fixedmass_2026-08-27")

# --- categorical slots 1 and 2, fixed order (validated: normal dE 33.6, CVD dE >= 30)
C_STAT, C_QUAD = "#2a78d6", "#eb6834"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
P_PAPER_DOUBLE, P_PAPER_SINGLE = 1.3e-7, 3.6e-9

d = {m: np.load(os.path.join(L, "fixed_mass_fit_%s.npz" % m)) for m in ("stat", "quad")}
sgl = np.load(os.path.join(L, "SINGLE_fixed_mass_fit_stat.npz"))
m = d["stat"]["masses"]

# ---- the full 10-2000 MeV ridge, from the n38 profile-scan response grid ----
# Same closed form as fit_fixed_mass.py, evaluated at every grid mass, so the
# 7 explicitly-run fixed-mass points below must land ON this curve.
sys.path.insert(0, HERE)
import miniboone_data as MB
_r = np.load(os.path.join(L, "ridge_response_n38.npz"))
RM, RE = np.asarray(_r["MV2_GRID"]) * 1e3, _r["EHIST"]
ridge = {}
for mode in ("stat", "quad"):
    iv = 1.0 / MB.errors(mode) ** 2
    A = np.array([np.sum(MB.EXCESS * s * iv) / np.sum(s * s * iv) for s in RE])
    ridge[mode] = dict(
        P=1.3e-7 * np.sqrt(np.clip(A, 0, None)),
        chi2=np.array([np.sum((MB.EXCESS - a * s) ** 2 * iv) for a, s in zip(A, RE)]))

fig, (ax, bx) = plt.subplots(2, 1, figsize=(8.2, 7.4), sharex=True,
                             gridspec_kw=dict(height_ratios=[2.35, 1], hspace=0.10))
fig.patch.set_facecolor("#fcfcfb")
for a in (ax, bx):
    a.set_facecolor("#fcfcfb")
    a.grid(True, which="major", color="#e3e2dd", lw=0.8, zorder=0)
    a.grid(True, which="minor", color="#efeeea", lw=0.5, zorder=0)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        a.spines[s].set_color("#c9c8c2")
    a.tick_params(colors=INK2, labelsize=9)

# ---- top: coupling ----
# The two central curves agree to <1%, so they would hide each other: give quad
# the wider (real) band and a fatter line so it survives as a halo under stat.
ax.axvspan(100, 2000, color="#e3e2dd", alpha=0.5, lw=0, zorder=0)
ax.text(450, 1.35e-5, "range plotted in the paper's Fig. 3 (left)",
        fontsize=8, color=MUTED, ha="center", va="top")
for mode, col, lw, mk, ms, z in (("quad", C_QUAD, 5.0, "s", 10, 3),
                                 ("stat", C_STAT, 1.8, "o", 6, 5)):
    r = d[mode]
    ax.plot(RM, ridge[mode]["P"], "-", color=col, lw=lw, zorder=z,
            solid_capstyle="round", label="our fit, %s errors" % mode)
    ax.fill_between(m, r["P_lo"], r["P_hi"], color=col, alpha=0.16, lw=0, zorder=z - 1)
    ax.plot(m, r["P_hat"], ls="none", marker=mk, color=col, ms=ms, mec="#fcfcfb", mew=1.4,
            zorder=z + 2)

ax.plot(200, P_PAPER_DOUBLE, "*", ms=22, color="#111111", mec="#fcfcfb", mew=1.2,
        zorder=6, label="Dutta-Kim Table I benchmark")
ax.annotate("double-mediator\npaper $1.3{\\times}10^{-7}$\nours $1.32{\\times}10^{-7}$"
            "  —  ratio 1.02",
            xy=(215, 1.45e-7), xytext=(330, 2.4e-8),
            fontsize=9, color=INK, ha="left",
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))

# single-mediator benchmark: ours vs paper at m_V1 = 17 MeV
ax.plot([17, 17], [P_PAPER_SINGLE, sgl["P_hat"][0]], color=MUTED, lw=1.2, ls=":", zorder=3)
ax.plot(17, P_PAPER_SINGLE, "*", ms=22, color="#111111", mec="#fcfcfb", mew=1.2, zorder=6)
ax.plot(17, sgl["P_hat"][0], "o", ms=8, color=C_STAT, mec="#fcfcfb", mew=1.5, zorder=6)
ax.annotate("single-mediator benchmark (17, $-$, 8, 40) MeV\nours $2.5\\times$ high",
            xy=(18.6, 4.8e-9), xytext=(27, 2.9e-9),
            fontsize=9, color=INK, ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_ylabel(r"$\hat{P}=\epsilon_1\epsilon_2 g'^2/4\pi$", fontsize=11, color=INK)
ax.set_title("Fixed-mass coupling fit vs Dutta-Kim Table I\n"
             "freeze $m_{V_2}$, fit only the coupling product  ·  MiniBooNE "
             r"$\nu$-mode $E_{vis}$, 11 HEPData bins",
             fontsize=12, color=INK, loc="left", pad=12)
leg = ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK2,
                borderaxespad=0.9)
ax.set_ylim(1.8e-9, 2.6e-5)


# ---- bottom: goodness of fit ----
bx.axvspan(100, 2000, color="#e3e2dd", alpha=0.55, lw=0, zorder=0)
for mode, col in (("stat", C_STAT), ("quad", C_QUAD)):
    bx.plot(RM, ridge[mode]["chi2"] / 10.0, "-", color=col, lw=2, zorder=4)
    bx.plot(m, d[mode]["chi2_min"] / 10.0, ls="none", marker="o", color=col, ms=6,
            mec="#fcfcfb", mew=1.4, zorder=5)
bx.axvline(200, color=MUTED, lw=1, ls=":", zorder=1)
bx.set_yscale("log")
bx.set_xlabel(r"$m_{V_2}$  [MeV]   (held fixed)", fontsize=11, color=INK)
bx.set_ylabel(r"$\chi^2/\mathrm{dof}$", fontsize=11, color=INK)
bx.set_xticks([10, 20, 31.4, 60, 100, 200, 500, 1000, 2000])
bx.set_xticklabels(["10", "20", "31.4", "60", "100", "200", "500", "1000", "2000"])
bx.set_yticks([0.1, 0.2, 0.5, 1.0])
bx.set_yticklabels(["0.1", "0.2", "0.5", "1.0"])
bx.set_ylim(0.09, 1.8)
bx.annotate("our shape fit prefers ~31 MeV", xy=(31.4, 0.87), xytext=(46, 0.30),
            fontsize=9, color=INK2,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
bx.text(1500, 1.60, "stat", fontsize=9, color=C_STAT, ha="center", va="center")
bx.text(1500, 0.115, "quad", fontsize=9, color=C_QUAD, ha="center", va="center")

fig.text(0.012, 0.012,
         "The coupling enters only as a normalisation $A=(P/P_0)^2$, so at fixed mass $\\chi^2$ is exactly\n"
         "quadratic in $A$ and $\\hat{P}$ has a closed form. Anchor $P_0$ is the paper's own Table I product,\n"
         "so ratio 1.00 = exact agreement. $\\chi^2/\\mathrm{dof}$ is OURS only — not comparable with the\n"
         "paper's four-distribution fit (see VERIFICATION.md). The stat and quad central curves\n"
         "agree to <1%; only their 68% bands differ, so quad is drawn as the wider halo.",
         fontsize=7.4, color=MUTED, ha="left", va="bottom", linespacing=1.5)
ax.set_xlim(9.2, 2400)
fig.subplots_adjust(left=0.125, right=0.965, top=0.885, bottom=0.205)
out = os.path.join(L, "fixed_mass_coupling.png")
fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
print("wrote", out)
