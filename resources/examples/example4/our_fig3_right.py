"""
OUR version of Dutta-Kim Fig.3 RIGHT panel (Model II, long-lived (pseudo)scalar),
drawn in the PAPER'S STYLE: filled credible regions, dark=68% / light=95%, on the
same log-log (m_Zp, lambda*g_mu*g_n) axes.

Regions = our #1 Delta-chi2 fit to the digitized MiniBooNE nu-mode excess
(output/credible_region_mZp_product.npz).  Only the m_phi=m_a=1 MeV cases (what we
scanned).  Paper Table I benchmark stars shown for reference.

Run: /home/shubham/siren_venv/bin/python our_fig3_right.py
Out: output/our_fig3_right.png
"""
import os, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d = np.load(os.path.join(HERE, "output", "credible_region_mZp_product.npz"))
mzp = d["mzp_MeV"]; prod = d["prod"]
X, Y = np.meshgrid(mzp, prod, indexing="ij")

fig, ax = plt.subplots(figsize=(6.6, 6.4))

# paper-style colors: scalar = green, pseudoscalar = purple
SERIES = [
    ("chi2_scalar", "Scalar ($m_\\phi = 1$ MeV)",       "#2ca25f"),
    ("chi2_pseudo", "Pseudoscalar ($m_a = 1$ MeV)",      "#6a51a3"),
]
for key, lab, col in SERIES:
    D = d[key] - d[key].min()
    # light (95%) first, then dark (68%) on top -- paper look
    ax.contourf(X, Y, D, levels=[0, 6.18], colors=[col], alpha=0.25)   # 95%
    ax.contourf(X, Y, D, levels=[0, 2.30], colors=[col], alpha=0.55)   # 68%
    ax.plot([], [], color=col, lw=8, alpha=0.6, label=lab)             # legend proxy

# Table I benchmark points (paper)
ax.scatter([49], [2.2e-8], marker="*", s=210, color="#2ca25f", edgecolor="k", zorder=6,
           label="paper Table I scalar")
ax.scatter([85], [5.9e-7], marker="*", s=210, color="#6a51a3", edgecolor="k", zorder=6,
           label="paper Table I pseudo")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlim(30, 200); ax.set_ylim(1e-8, 1e-5)
xt = [30, 40, 60, 100, 200]
ax.set_xticks(xt); ax.set_xticklabels([r"$3\times10^1$", r"$4\times10^1$", r"$6\times10^1$",
                                       r"$10^2$", r"$2\times10^2$"])
ax.set_xlabel(r"$m_{Z'}$ [MeV]", fontsize=12)
ax.set_ylabel(r"$\lambda g_\mu g_n$ [MeV$^{-1}$]", fontsize=12)
ax.set_title("Our MiniBooNE credible regions (Fig.3-right style)\n"
             "68% (dark) / 95% (light);  $\\nu$-mode $E_{vis}$ fit", fontsize=11)
ax.legend(fontsize=9, loc="lower right", framealpha=0.95)
ax.grid(True, which="both", alpha=0.2)
fig.tight_layout()
out = os.path.join(HERE, "output", "our_fig3_right.png")
fig.savefig(out, dpi=140); print("wrote", out)
