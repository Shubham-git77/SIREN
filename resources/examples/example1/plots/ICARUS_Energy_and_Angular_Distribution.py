import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------------
# Load ICARUS data
# --------------------------------------------------
df = pd.read_parquet("output/UnidirectionalICARUS_DIS.parquet")

# Event weights
W_evt = df["event_weight"].to_numpy(dtype=float)

# --------------------------------------------------
# INITIAL NEUTRINO
# --------------------------------------------------
E_nu = []
cos_nu = []

for p in df["primary_momentum"]:
    E, px, py, pz = p[0]
    pmod = np.sqrt(px**2 + py**2 + pz**2)

    E_nu.append(E)
    cos_nu.append(pz / pmod)

E_nu = np.array(E_nu)
cos_nu = np.array(cos_nu)

# --------------------------------------------------
# SECONDARIES (CORRECT WEIGHT HANDLING)
# --------------------------------------------------
E_mu, cos_mu, W_mu = [], [], []
E_had, cos_had, W_had = [], [], []

for ie, (moms, types) in enumerate(zip(df["secondary_momenta"], df["secondary_types"])):

    w = W_evt[ie]

    had_E, had_px, had_py, had_pz = 0, 0, 0, 0

    for (E, px, py, pz), pid in zip(moms[0], types[0]):
        pmod = np.sqrt(px**2 + py**2 + pz**2)

        if abs(pid) == 13:  # muon
            E_mu.append(E)
            cos_mu.append(pz / pmod)
            W_mu.append(w)

        else:
            had_E  += E
            had_px += px
            had_py += py
            had_pz += pz

    if had_E > 0:
        had_p = np.sqrt(had_px**2 + had_py**2 + had_pz**2)
        E_had.append(had_E)
        cos_had.append(had_pz / had_p)
        W_had.append(w)

E_mu  = np.array(E_mu)
cos_mu = np.array(cos_mu)
W_mu  = np.array(W_mu)

E_had = np.array(E_had)
cos_had = np.array(cos_had)
W_had = np.array(W_had)

# --------------------------------------------------
# BINS (same as paper)
# --------------------------------------------------
E_bins   = np.logspace(-2, 6, 50)
cos_bins = np.linspace(0.80, 1.00, 50)

# --------------------------------------------------
# HIST FUNCTION
# --------------------------------------------------
def weighted_hist(x, w, bins):
    h, _ = np.histogram(x, bins=bins, weights=w)
    return h / np.diff(bins)

# --------------------------------------------------
# HISTOGRAMS
# --------------------------------------------------
dNdE_nu  = weighted_hist(E_nu,  W_evt, E_bins)
dNdE_mu  = weighted_hist(E_mu,  W_mu,  E_bins)
dNdE_had = weighted_hist(E_had, W_had, E_bins)

dNdC_nu  = weighted_hist(cos_nu,  W_evt, cos_bins)
dNdC_mu  = weighted_hist(cos_mu,  W_mu,  cos_bins)
dNdC_had = weighted_hist(cos_had, W_had, cos_bins)

# --------------------------------------------------
# PLOT
# --------------------------------------------------
fig, ax = plt.subplots(2, 1, figsize=(6.5, 7.5))

# Energy
ax[0].step(E_bins[:-1], dNdE_nu,  where="post", label="Initial ν")
ax[0].step(E_bins[:-1], dNdE_mu,  where="post", label="Outgoing μ")
ax[0].step(E_bins[:-1], dNdE_had, where="post", label="Outgoing Hadrons")

ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[0].set_xlabel("Energy [GeV]")
ax[0].set_ylabel("dN/dE")
ax[0].legend()

# Angle
ax[1].step(cos_bins[:-1], dNdC_nu,  where="post", label="Initial ν")
ax[1].step(cos_bins[:-1], dNdC_mu,  where="post", label="Outgoing μ")
ax[1].step(cos_bins[:-1], dNdC_had, where="post", label="Outgoing Hadrons")

ax[1].set_yscale("log")
ax[1].set_xlabel("cosθ")
ax[1].set_ylabel("dN/dcosθ")
ax[1].legend()

plt.tight_layout()
plt.savefig("UnidirectionalICARUS_Figure6.png", dpi=300)
plt.close()
