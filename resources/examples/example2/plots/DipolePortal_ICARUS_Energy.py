import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======================================
# CONFIGURATION
# ======================================

POT = 1.2e21
BINS = np.linspace(0, 20, 45)

PDG_HNL = 5914
PDG_NU  = 22     # swapped (correct physical neutrino)
PDG_GAM = 5910   # swapped (correct photon)

# ======================================
# Load Data
# ======================================

df = pd.read_parquet("ICARUS_Dipole.parquet")

# ======================================
# Energy extractor
# ======================================

def energy(p):
    p = np.array(p)

    if len(p) == 4:
        E = p[0]
        return E
    return np.nan

# ======================================
# Containers
# ======================================

E_init_all, E_init_fid = [], []
E_N_all, E_N_fid = [], []
E_nu_all, E_nu_fid = [], []
E_g_all, E_g_fid = [], []

w_all, w_fid = [], []

# ======================================
# Loop Over Events
# ======================================

for i in range(len(df)):

    weight = df["event_weight"][i] * POT

    # ---- SAFE Fiducial Handling ----
    in_fid_array = df["in_fiducial"][i]

    if isinstance(in_fid_array, (list, np.ndarray)) and len(in_fid_array) > 0:
        if len(in_fid_array) > 1:
            in_fid = bool(in_fid_array[1])  # decay block
        else:
            in_fid = bool(in_fid_array[0])
    else:
        in_fid = False

    # ---- Primary neutrino ----
    p0 = df["primary_momentum"][i]
    e0 = energy(p0)

    E_init_all.append(e0)
    w_all.append(weight)

    if in_fid:
        E_init_fid.append(e0)
        w_fid.append(weight)

    # ---- Secondary blocks ----
    types_blocks = df["secondary_types"][i]
    mom_blocks   = df["secondary_momenta"][i]

    for block_types, block_mom in zip(types_blocks, mom_blocks):

        for t, p in zip(block_types, block_mom):

            t = int(t)
            e = energy(p)

            if t == PDG_HNL:
                E_N_all.append(e)
                if in_fid:
                    E_N_fid.append(e)

            elif t == PDG_NU:
                E_nu_all.append(e)
                if in_fid:
                    E_nu_fid.append(e)

            elif t == PDG_GAM:
                E_g_all.append(e)
                if in_fid:
                    E_g_fid.append(e)

# ======================================
# Plot (Paper Style)
# ======================================

plt.figure(figsize=(8,6))

# Initial ν
plt.hist(E_init_all, bins=BINS, weights=w_all,
         histtype="step", linewidth=2.2, color="#d17c00",
         label="Initial ν")
plt.hist(E_init_fid, bins=BINS, weights=w_fid,
         histtype="step", linestyle="--", linewidth=2.2,
         color="#d17c00")

# Upscattered N
plt.hist(E_N_all, bins=BINS,
         histtype="step", linewidth=2.2, color="cyan",
         label="Upscattered N")
plt.hist(E_N_fid, bins=BINS,
         histtype="step", linestyle="--", linewidth=2.2,
         color="cyan")

# Outgoing ν
plt.hist(E_nu_all, bins=BINS,
         histtype="step", linewidth=2.2, color="blue",
         label="Outgoing ν")
plt.hist(E_nu_fid, bins=BINS,
         histtype="step", linestyle="--", linewidth=2.2,
         color="blue")

# Outgoing γ
plt.hist(E_g_all, bins=BINS,
         histtype="step", linewidth=2.2, color="magenta",
         label="Outgoing γ")
plt.hist(E_g_fid, bins=BINS,
         histtype="step", linestyle="--", linewidth=2.2,
         color="magenta")

plt.yscale("log")
plt.xlim(0, 20)
plt.ylim(1e1, 1e5)

plt.xlabel("Energy [GeV]", fontsize=14)
plt.ylabel(r"Event Rate in $1.2\times10^{21}$ POT", fontsize=14)

plt.legend(frameon=False, fontsize=11)
plt.grid(True, which="both", linestyle="--", alpha=0.25)

plt.tight_layout()
plt.savefig("ICARUS_Fig8_Energy_FINAL.png", dpi=400)
plt.show()
