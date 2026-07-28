"""
Overlay OUR #1 MiniBooNE credible region on a REDRAWN Dutta-Kim Fig.3 RIGHT panel
(Model II, long-lived (pseudo)scalar).  Paper blobs are digitized approximations
of arXiv:2110.11944 Fig.3-right; our regions are the Delta-chi2 contours from
scan_credible_region.py (output/credible_region_mZp_product.npz).

We only have the m=1 MeV cases (scalar m_phi=1, pseudo m_a=1), matching the paper's
green (scalar 1 MeV) and purple (pseudo 1 MeV) blobs.

Run: /home/shubham/siren_venv/bin/python overlay_fig3_right.py
Out: output/overlay_fig3_right.png
"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "output", "credible_region_mZp_product.npz"))
mzp = d["mzp_MeV"]; prod = d["prod"]
X, Y = np.meshgrid(mzp, prod, indexing="ij")

fig, ax = plt.subplots(figsize=(8.4, 6.6))

# ---- paper Fig.3-right blobs, MEASURED by pixel-digitizing the high-res panel ----
# (fig3_ellipses.json). center (m_Zp[MeV], coupling[MeV^-1]); half-widths in log10:
# hx95/hy68 from pixel extents; 68%x = 0.6*95%x, 95%y = 1.64*68%y (contamination-safe).
# (center_mZp, center_coup, hx95_log, hy68_log, color)
PAPER = {
    "Scalar ($m_\\phi=1$)":  (45.9, 2.20e-8, 0.155, 0.034, "#2ca25f"),
    "Scalar ($m_\\phi=25$)": (69.0, 1.18e-7, 0.314, 0.036, "#bcbd22"),
    "Pseudo ($m_a=1$)":      (63.6, 5.68e-7, 0.292, 0.030, "#6a51a3"),
    "Pseudo ($m_a=25$)":     (85.0, 8.84e-7, 0.159, 0.032, "#238b8b"),
}
for lab, (mc, pc, hx95, hy68, col) in PAPER.items():
    hy95 = 1.64 * hy68; hx68 = 0.60 * hx95            # 2-param CL scaling / core fraction
    for hx, hy, alpha in [(hx95, hy95, 0.16), (hx68, hy68, 0.34)]:  # 95% then 68%
        e = Ellipse((np.log10(mc), np.log10(pc)), 2*hx, 2*hy,
                    facecolor=col, edgecolor="none", alpha=alpha, transform=ax.transData, zorder=1)
        ax.add_patch(e)
    ax.plot([], [], color=col, lw=6, alpha=0.5, label="paper "+lab)

# ---- our #1 Delta-chi2 credible contours (log-log axes handled via log10 coords) ----
def our_region(chi2, col, lab):
    D = chi2 - chi2.min()
    lx, ly = np.log10(X), np.log10(Y)
    ax.contourf(lx, ly, D, levels=[0, 6.18], colors=[col], alpha=0.0)  # (keep for extents)
    ax.contour(lx, ly, D, levels=[2.30], colors=col, linewidths=2.4)
    ax.contour(lx, ly, D, levels=[6.18], colors=col, linewidths=1.2, linestyles="--")
    ax.plot([], [], color=col, lw=2.4, label="OUR "+lab+" (68% solid / 95% dashed)")

our_region(d["chi2_scalar"], "#00701a", "scalar")
our_region(d["chi2_pseudo"], "#3b0a70", "pseudo")

# ---- paper Table I benchmark stars ----
ax.scatter(np.log10(49), np.log10(2.2e-8), marker="*", s=260, color="#2ca25f",
           edgecolor="k", zorder=6, label="paper Table I scalar (49, 2.2e-8)")
ax.scatter(np.log10(85), np.log10(5.9e-7), marker="*", s=260, color="#6a51a3",
           edgecolor="k", zorder=6, label="paper Table I pseudo (85, 5.9e-7)")

# ---- log tick formatting on the manual log10 axes ----
ax.set_xlim(np.log10(30), np.log10(200)); ax.set_ylim(np.log10(3e-9), np.log10(1.2e-6))
xt = [30,40,60,100,200]; ax.set_xticks([np.log10(v) for v in xt]); ax.set_xticklabels([str(v) for v in xt])
yt = [1e-8,1e-7,1e-6]; ax.set_yticks([np.log10(v) for v in yt]); ax.set_yticklabels([r"$10^{-8}$",r"$10^{-7}$",r"$10^{-6}$"])
ax.set_xlabel(r"$m_{Z'}$ [MeV]"); ax.set_ylabel(r"$\lambda g_\mu g_n$ [MeV$^{-1}$]")
ax.set_title("Our #1 MiniBooNE credible region vs Dutta-Kim Fig.3 (right, Model II)\n"
             "paper blobs pixel-measured from Fig.3; ours = $\\Delta\\chi^2$, $\\nu$-mode $E_{vis}$ only", fontsize=11)
ax.grid(True, alpha=0.25)
ax.legend(fontsize=7.5, loc="lower right", ncol=1, framealpha=0.9)
fig.tight_layout()
out = os.path.join(HERE, "output", "overlay_fig3_right.png")
fig.savefig(out, dpi=130); print("wrote", out)
