import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ======================================
# CONFIG
# ======================================

POT = 1.2e21
BINS = np.linspace(0, 20, 45)

PDG_HNL = 5914
PDG_NU  = 5910      # light neutrino 
PDG_E_MINUS = 11
PDG_E_PLUS  = -11

# ======================================
# LOAD DATA
# ======================================

df = pd.read_parquet("MINERvA_ThreePortal_M2.00e-01_eps5.00e-04_gD5.00e-01.parquet")

# ======================================
# ENERGY FUNCTION
# ======================================

def energy(p):
    p = np.array(p)
    if len(p) == 4:
        return p[0]
    return np.nan

# ======================================
# CONTAINERS
# ======================================

E_init, E_init_fid = [], []
E_N, E_N_fid = [], []
E_nu, E_nu_fid = [], []
E_em, E_em_fid = [], []
E_ep, E_ep_fid = [], []

w_all, w_fid = [], []

# ======================================
# LOOP
# ======================================

for i in range(len(df)):

    weight = df["event_weight"][i] * POT

    # --- Fiducial ---
    fid_arr = df["in_fiducial"][i]
    if isinstance(fid_arr, (list, np.ndarray)) and len(fid_arr) > 0:
        in_fid = bool(fid_arr[1]) if len(fid_arr) > 1 else bool(fid_arr[0])
    else:
        in_fid = False

    # --- Initial ν ---
    e0 = energy(df["primary_momentum"][i])
    E_init.append(e0)
    w_all.append(weight)

    if in_fid:
        E_init_fid.append(e0)
        w_fid.append(weight)

    # --- Secondaries ---
    types_blocks = df["secondary_types"][i]
    mom_blocks   = df["secondary_momenta"][i]

    for block_types, block_mom in zip(types_blocks, mom_blocks):

        for t, p in zip(block_types, block_mom):

            t = int(t)
            e = energy(p)

            if t == PDG_HNL:
                E_N.append(e)
                if in_fid: E_N_fid.append(e)

            elif t == PDG_NU:
                E_nu.append(e)
                if in_fid: E_nu_fid.append(e)

            elif t == PDG_E_MINUS:
                E_em.append(e)
                if in_fid: E_em_fid.append(e)

            elif t == PDG_E_PLUS:
                E_ep.append(e)
                if in_fid: E_ep_fid.append(e)

# ======================================
# PLOT
# ======================================

plt.figure(figsize=(8,6))

plt.hist(E_init, bins=BINS, weights=w_all,
         histtype="step", lw=2, color="orange", label="Initial ν")

plt.hist(E_N, bins=BINS,
         histtype="step", lw=2, color="cyan", label="Upscattered N")

plt.hist(E_nu, bins=BINS,
         histtype="step", lw=2, color="blue", label="Final ν")

plt.hist(E_em, bins=BINS,
         histtype="step", lw=2, color="red", label="e⁻")

plt.hist(E_ep, bins=BINS,
         histtype="step", lw=2, color="magenta", label="e⁺")

plt.yscale("log")
plt.xlabel("Energy [GeV]")
plt.ylabel("Event Rate")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("MINERvA_ThreePortal_energy.png", dpi=300)
plt.show()
