"""
OUR Fig.3-right (Model II) with ALL FOUR regions {scalar,pseudo} x {m=1,25 MeV},
paper-style filled 68%/95% shading, PLUS the paper's pixel-measured blobs overlaid
faintly and the Table I benchmark stars.  Direct "ours vs Dutta-Kim" in one frame.

Needs: output/credible_region_mZp_product.npz (m=1) + output/credible_region_m25.npz
Run: /home/shubham/siren_venv/bin/python our_fig3_full.py
Out: output/our_fig3_full.png
"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

HERE = os.path.dirname(os.path.abspath(__file__))
d1 = np.load(os.path.join(HERE, "output", "credible_region_mZp_product.npz"))
d2 = np.load(os.path.join(HERE, "output", "credible_region_m25.npz"))
mzp = d1["mzp_MeV"]; prod = d1["prod"]
X, Y = np.meshgrid(mzp, prod, indexing="ij")

fig, ax = plt.subplots(figsize=(7.6, 7.0))

# ---- paper measured blobs (faint, underneath), from pixel digitization ----
# (center m_Zp, center coup, hx95_log, hy68_log, color)
PAPER = {
    "scalar1":  (45.9, 2.20e-8, 0.155, 0.034, "#2ca25f"),
    "scalar25": (69.0, 1.18e-7, 0.314, 0.036, "#bcbd22"),
    "pseudo1":  (63.6, 5.68e-7, 0.292, 0.030, "#6a51a3"),
    "pseudo25": (85.0, 8.84e-7, 0.159, 0.032, "#238b8b"),
}
for name,(mc,pc,hx95,hy68,col) in PAPER.items():
    hy95=1.64*hy68; hx68=0.60*hx95
    for hx,hy,al in [(hx95,hy95,0.10),(hx68,hy68,0.20)]:
        ax.add_patch(Ellipse((np.log10(mc),np.log10(pc)),2*hx,2*hy,
                     facecolor=col,edgecolor="none",alpha=al,zorder=1))

# ---- our four credible regions (filled 95% light + 68% dark), paper style ----
OURS = [
    (d1["chi2_scalar"], "Scalar ($m_\\phi=1$)",       "#00701a"),
    (d2["chi2_scalar25"],"Scalar ($m_\\phi=25$)",     "#7f7f16"),
    (d1["chi2_pseudo"], "Pseudoscalar ($m_a=1$)",     "#3b0a70"),
    (d2["chi2_pseudo25"],"Pseudoscalar ($m_a=25$)",   "#0f6b6b"),
]
lx, ly = np.log10(X), np.log10(Y)
for C, lab, col in OURS:
    D = C - C.min()
    ax.contourf(lx, ly, D, levels=[0, 6.18], colors=[col], alpha=0.22)   # 95%
    ax.contourf(lx, ly, D, levels=[0, 2.30], colors=[col], alpha=0.50)   # 68%
    ax.plot([], [], color=col, lw=8, alpha=0.6, label="OUR "+lab)

# paper Table I stars
ax.scatter(np.log10(49), np.log10(2.2e-8), marker="*", s=240, color="#2ca25f", edgecolor="k", zorder=7)
ax.scatter(np.log10(85), np.log10(5.9e-7), marker="*", s=240, color="#6a51a3", edgecolor="k", zorder=7)
ax.plot([], [], "k*", ms=13, label="paper Table I (scalar 49 / pseudo 85)")
ax.plot([], [], color="0.5", lw=8, alpha=0.35, label="paper Fig.3 blobs (measured)")

# log tick formatting on manual log10 axes
ax.set_xlim(np.log10(30), np.log10(200)); ax.set_ylim(np.log10(1e-8), np.log10(1e-5))
xt=[30,40,60,100,200]; ax.set_xticks([np.log10(v) for v in xt])
ax.set_xticklabels([r"$3\times10^1$",r"$4\times10^1$",r"$6\times10^1$",r"$10^2$",r"$2\times10^2$"])
yt=[1e-8,1e-7,1e-6,1e-5]; ax.set_yticks([np.log10(v) for v in yt])
ax.set_yticklabels([r"$10^{-8}$",r"$10^{-7}$",r"$10^{-6}$",r"$10^{-5}$"])
ax.set_xlabel(r"$m_{Z'}$ [MeV]", fontsize=12); ax.set_ylabel(r"$\lambda g_\mu g_n$ [MeV$^{-1}$]", fontsize=12)
ax.set_title("Our MiniBooNE credible regions (Fig.3-right style, all 4 cases)\n"
             "68% dark / 95% light;  paper Fig.3 blobs faint underneath;  $\\nu$-mode $E_{vis}$ fit", fontsize=10.5)
ax.legend(fontsize=8, loc="lower right", framealpha=0.95)
ax.grid(True, alpha=0.2)
fig.tight_layout()
out = os.path.join(HERE, "output", "our_fig3_full.png")
fig.savefig(out, dpi=140); print("wrote", out)
