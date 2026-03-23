import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------
# Load datasets
# -------------------------------------------------
datasets = {
    "ICARUS": pd.read_parquet("output/ICARUS_DIS.parquet"),
}

colors = {
    "ICARUS": "#E69F00",
    
}

# -------------------------------------------------
# Exact paper binning
# -------------------------------------------------
time_bins = np.logspace(-6, -2, 60)

# -------------------------------------------------
# Median ± 1σ (paper definition)
# -------------------------------------------------
def median_sigma(x):
    med = np.median(x)
    lo  = med - np.percentile(x, 16)
    hi  = np.percentile(x, 84) - med
    return med, lo, hi

# -------------------------------------------------
# Plot
# -------------------------------------------------
fig, ax = plt.subplots(2, 1, figsize=(6.8, 8.2), sharex=False)

# ============================
# TOP: Event generation time
# ============================
for name, df in datasets.items():
    t_gen = df["event_gen_time"].to_numpy()

    med, lo, hi = median_sigma(t_gen)

    label = (
        f"{name}\n"
        r"$\tau$ = "
        f"{med/1e-5:.2f}"
        r"$^{+"
        f"{hi/1e-5:.2f}"
        r"}_{-"
        f"{lo/1e-5:.2f}"
        r"} \times 10^{-5}$ s"
    )

    ax[0].hist(
        t_gen,
        bins=time_bins,
        histtype="stepfilled",
        alpha=0.6,
        color=colors[name],
        label=label,
    )

ax[0].set_xscale("log")
ax[0].set_yscale("log")
ax[0].set_ylabel("Events")
ax[0].set_xlabel("Event generation time [s]")
ax[0].legend(frameon=False)

# ============================
# BOTTOM: Event weight time
# ============================
for name, df in datasets.items():
    t_wgt = df["event_weight_time"].to_numpy()

    med, lo, hi = median_sigma(t_wgt)

    label = (
        f"{name}\n"
        r"$\tau$ = "
        f"{med/1e-5:.2f}"
        r"$^{+"
        f"{hi/1e-5:.2f}"
        r"}_{-"
        f"{lo/1e-5:.2f}"
        r"} \times 10^{-5}$ s"
    )

    ax[1].hist(
        t_wgt,
        bins=time_bins,
        histtype="stepfilled",
        alpha=0.6,
        color=colors[name],
        label=label,
    )

ax[1].set_xscale("log")
ax[1].set_yscale("log")
ax[1].set_ylabel("Events")
ax[1].set_xlabel("Event weight calculation time [s]")
ax[1].legend(frameon=False)

# -------------------------------------------------
# Finalize
# -------------------------------------------------
plt.tight_layout()
plt.savefig("Timing_ICARUS.png", dpi=300)
plt.close()
