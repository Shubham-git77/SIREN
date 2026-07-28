"""
Standalone plotting for the MiniBooNE Vector Portal K+ -> e+ simulation.

Reads the observables saved by VectorPortal_MiniBooNE_kaon_eplus.py
(output/MiniBooNE_VectorPortal_kaon_e_full_observables.npz) and produces
visible-energy and angular plots styled to match the paper's Fig. 2
(Dutta et al., PRL 129, 111803).

Usage:
    python plot_miniboone_vectorportal.py
    python plot_miniboone_vectorportal.py path/to/observables.npz

Paper conventions applied (vector-portal, nu-mode, top panels of Fig. 2):
  - 140 MeV visible-energy threshold.
  - E_vis range 0-1250 MeV.
  - cos(theta) range -1..1.
The absolute normalization here is NOT paper-comparable (first-run g_D=1.0,
no POT scaling); the SHAPES are the comparison. Histograms are area-
normalized so the shape can be compared independent of the overall rate.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Locate the observables file ---
default_path = "output/MiniBooNE_VectorPortal_kaon_e_full_observables.npz"
obs_path = sys.argv[1] if len(sys.argv) > 1 else default_path
if not os.path.exists(obs_path):
    sys.exit("Observables file not found: %s\n"
             "Run VectorPortal_MiniBooNE_kaon_eplus.py first." % obs_path)

data   = np.load(obs_path)
E_vis  = data["E_vis"] * 1e3      # GeV -> MeV
cos_t  = data["cos_theta"]
weight = data["weight"]

print("Loaded %d signal events from %s" % (len(weight), obs_path))

# --- Paper analysis selections ---
E_THRESHOLD = 140.0   # MeV  visible-energy threshold (paper)
E_MAX       = 1250.0  # MeV  nu-mode E_vis axis range (paper top-left)

sel = (E_vis >= E_THRESHOLD)
E_vis_s  = E_vis[sel]
cos_t_s  = cos_t[sel]
weight_s = weight[sel]
print("After 140 MeV threshold: %d events (%.1f%%)"
      % (sel.sum(), 100.0 * sel.sum() / max(len(weight), 1)))

if sel.sum() == 0:
    sys.exit("No events survive the 140 MeV threshold; nothing to plot.")

# --- Binning (paper-like) ---
E_BINS   = np.linspace(0.0, E_MAX, 12)      # ~115 MeV bins, like Fig. 2 top-left
COS_BINS = np.linspace(-1.0, 1.0, 11)        # 10 bins, like Fig. 2 top-right

# ===========================================================================
# Figure 1: raw weighted counts (this simulation's own normalization)
# ===========================================================================
fig, ax = plt.subplots(1, 2, figsize=(12, 4.8))

ax[0].hist(E_vis_s, bins=E_BINS, weights=weight_s,
           histtype="stepfilled", color="#4C72B0", alpha=0.85,
           edgecolor="black", linewidth=0.7)
ax[0].axvline(E_THRESHOLD, color="grey", ls="--", lw=1,
              label="140 MeV threshold")
ax[0].set_xlabel(r"$E_{\rm vis}$  [MeV]")
ax[0].set_ylabel("Weighted events")
ax[0].set_title(r"MiniBooNE  $K^+\!\to e^+$ :  visible energy")
ax[0].set_xlim(0, E_MAX)
ax[0].legend(frameon=False, fontsize=9)

ax[1].hist(cos_t_s, bins=COS_BINS, weights=weight_s,
           histtype="stepfilled", color="#4C72B0", alpha=0.85,
           edgecolor="black", linewidth=0.7)
ax[1].set_xlabel(r"$\cos\theta$  (wrt beam)")
ax[1].set_ylabel("Weighted events")
ax[1].set_title(r"MiniBooNE  $K^+\!\to e^+$ :  angular")
ax[1].set_xlim(-1, 1)

fig.suptitle(r"Vector-portal $\chi$ upscattering signal "
             r"($m_\chi$=8, $m_{\chi'}$=50, $m_{V_1}$=17, $m_{V_2}$=200 MeV)",
             fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out1 = os.path.splitext(obs_path)[0].replace("_observables", "") + "_counts.png"
fig.savefig(out1, dpi=140)
print("Saved counts plot      -> %s" % out1)

# ===========================================================================
# Figure 2: SHAPE comparison (area-normalized) — the paper-comparable view
# ===========================================================================
fig2, ax2 = plt.subplots(1, 2, figsize=(12, 4.8))

ax2[0].hist(E_vis_s, bins=E_BINS, weights=weight_s, density=True,
            histtype="step", color="#C44E52", linewidth=1.8)
ax2[0].axvline(E_THRESHOLD, color="grey", ls="--", lw=1)
ax2[0].set_xlabel(r"$E_{\rm vis}$  [MeV]")
ax2[0].set_ylabel("Normalized rate  [arb.]")
ax2[0].set_title("Visible energy (shape)")
ax2[0].set_xlim(0, E_MAX)

ax2[1].hist(cos_t_s, bins=COS_BINS, weights=weight_s, density=True,
            histtype="step", color="#C44E52", linewidth=1.8)
ax2[1].set_xlabel(r"$\cos\theta$  (wrt beam)")
ax2[1].set_ylabel("Normalized rate  [arb.]")
ax2[1].set_title("Angular (shape)")
ax2[1].set_xlim(-1, 1)

fig2.suptitle("Shape comparison to paper Fig. 2 (top, nu-mode). "
              "Absolute rate not calibrated.", fontsize=11)
fig2.tight_layout(rect=[0, 0, 1, 0.96])
out2 = os.path.splitext(obs_path)[0].replace("_observables", "") + "_shapes.png"
fig2.savefig(out2, dpi=140)
print("Saved shape plot       -> %s" % out2)

# --- Quick numeric summary ---
def wfrac(mask):
    return weight_s[mask].sum() / weight_s.sum()

print("\nForward/backward split (weighted):")
print("  cos(theta) > 0 (forward) : %.1f%%" % (100 * wfrac(cos_t_s > 0)))
print("  cos(theta) < 0 (backward): %.1f%%" % (100 * wfrac(cos_t_s < 0)))
print("  cos(theta) > 0.5         : %.1f%%" % (100 * wfrac(cos_t_s > 0.5)))
emean = np.average(E_vis_s, weights=weight_s)
print("Weighted-mean E_vis: %.1f MeV" % emean)
print("Done.")
