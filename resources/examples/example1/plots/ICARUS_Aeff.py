import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12,
    "axes.linewidth": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

# -----------------------------
# Utility functions
# -----------------------------
def extract_energy(df):
    # primary_momentum[i] = [ [E, px, py, pz] ]
    return np.array([p[0][0] for p in df["primary_momentum"]], dtype=float)

def effective_area(E, W, Wg, bins):
    A = np.zeros(len(bins) - 1)
    for i in range(len(bins) - 1):
        mask = (E >= bins[i]) & (E < bins[i + 1])
        Ngen = np.count_nonzero(mask)
        if Ngen > 0:
            A[i] = np.sum(W[mask] / Wg[mask]) / Ngen
        else:
            A[i] = np.nan
    return A

# -----------------------------
# Load data
# -----------------------------
datasets = {
    "ICARUS":pd.read_parquet("output/ICARUS_DIS.parquet"),
    
}

# -----------------------------
# Energy binning (paper)
# -----------------------------
bins = np.logspace(3, 6, 21)
bin_centers = np.sqrt(bins[:-1] * bins[1:])

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(6.8, 5.0))

for label, df in datasets.items():
    E  = extract_energy(df)
    W  = df["event_weight"].to_numpy(dtype=float)
    Wg = df["event_weight_time"].to_numpy(dtype=float)

    Aeff = effective_area(E, W, Wg, bins)

    plt.step(
        bin_centers,
        Aeff,
        where="mid",
        lw=2.0,
        label=label
    )

plt.xscale("log")
plt.yscale("log")
plt.xlabel(r"$E_\nu$ [GeV]")
plt.ylabel(r"$A_{\mathrm{eff}}$ [m$^2$]")
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig("Aeff_ICARUS.png", dpi=300)
plt.close()
