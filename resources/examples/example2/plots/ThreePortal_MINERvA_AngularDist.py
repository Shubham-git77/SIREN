import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======================================
# CONFIGURATION
# ======================================

POT = 1.2e21
BINS = np.linspace(-1, 1, 45)

PDG_HNL = 5914
PDG_NU  = 5910     # IMPORTANT: DarkNews ν_light
PDG_E_MINUS = 11
PDG_E_PLUS  = -11

# ======================================
# LOAD DATA
# ======================================

df = pd.read_parquet("MINERvA_ThreePortal_M2.00e-01_eps5.00e-04_gD5.00e-01.parquet")

# ======================================
# cos(theta)
# ======================================

def cos_theta(p):
    p = np.array(p)

    if len(p) == 4:
        _, px, py, pz = p
    elif len(p) == 3:
        px, py, pz = p
    else:
        return np.nan

    p_mag = np.sqrt(px**2 + py**2 + pz**2)
    if p_mag == 0:
        return np.nan

    return pz / p_mag

# ======================================
# CONTAINERS
# ======================================

cos_init_all, cos_init_fid = [], []
cos_N_all, cos_N_fid = [], []
cos_nu_all, cos_nu_fid = [], []
cos_em_all, cos_em_fid = [], []
cos_ep_all, cos_ep_fid = [], []

w_all, w_fid = [], []

# ======================================
# LOOP
# ======================================

for i in range(len(df)):

    weight = df["event_weight"][i] * POT

    # ---- Fiducial ----
    fid_arr = df["in_fiducial"][i]

    if isinstance(fid_arr, (list, np.ndarray)) and len(fid_arr) > 0:
        in_fid = bool(fid_arr[1]) if len(fid_arr) > 1 else bool(fid_arr[0])
    else:
        in_fid = False

    # ---- Initial ν ----
    c0 = cos_theta(df["primary_momentum"][i])

    cos_init_all.append(c0)
    w_all.append(weight)

    if in_fid:
        cos_init_fid.append(c0)
        w_fid.append(weight)

    # ---- Secondary particles ----
    types_blocks = df["secondary_types"][i]
    mom_blocks   = df["secondary_momenta"][i]

    for block_types, block_mom in zip(types_blocks, mom_blocks):

        for t, p in zip(block_types, block_mom):

            t = int(t)
            c = cos_theta(p)

            if t == PDG_HNL:
                cos_N_all.append(c)
                if in_fid: cos_N_fid.append(c)

            elif t == PDG_NU:
                cos_nu_all.append(c)
                if in_fid: cos_nu_fid.append(c)

            elif t == PDG_E_MINUS:
                cos_em_all.append(c)
                if in_fid: cos_em_fid.append(c)

            elif t == PDG_E_PLUS:
                cos_ep_all.append(c)
                if in_fid: cos_ep_fid.append(c)

# ======================================
# PLOT
# ======================================

plt.figure(figsize=(8,6))

# Initial ν
plt.hist(cos_init_all, bins=BINS, weights=w_all,
         histtype="step", linewidth=2.2, color="orange",
         label="Initial ν")
plt.hist(cos_init_fid, bins=BINS, weights=w_fid,
         histtype="step", linestyle="--", linewidth=2.2,
         color="orange")

# Upscattered N
plt.hist(cos_N_all, bins=BINS,
         histtype="step", linewidth=2.2, color="cyan",
         label="Upscattered N")
plt.hist(cos_N_fid, bins=BINS,
         histtype="step", linestyle="--", linewidth=2.2,
         color="cyan")

# Final ν
plt.hist(cos_nu_all, bins=BINS,
         histtype="step", linewidth=2.2, color="blue",
         label="Final ν")
plt.hist(cos_nu_fid, bins=BINS,
         histtype="step", linestyle="--", linewidth=2.2,
         color="blue")

# e-
plt.hist(cos_em_all, bins=BINS,
         histtype="step", linewidth=2.2, color="red",
         label="e⁻")
plt.hist(cos_em_fid, bins=BINS,
         histtype="step", linestyle="--", linewidth=2.2,
         color="red")

# e+
plt.hist(cos_ep_all, bins=BINS,
         histtype="step", linewidth=2.2, color="magenta",
         label="e⁺")
plt.hist(cos_ep_fid, bins=BINS,
         histtype="step", linestyle="--", linewidth=2.2,
         color="magenta")

# Style
plt.yscale("log")
plt.xlim(-1, 1)
plt.ylim(1e1, 1e6)

plt.xlabel(r"$\cos\theta$", fontsize=14)
plt.ylabel(r"Event Rate in $1.2\times10^{21}$ POT", fontsize=14)

plt.legend(frameon=False, fontsize=11)
plt.grid(True, which="both", linestyle="--", alpha=0.25)

plt.tight_layout()
plt.savefig("MINERvA_ThreePortal_angular.png", dpi=400)
plt.show()
