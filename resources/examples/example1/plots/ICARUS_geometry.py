import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Load parquet file
# =========================
file_path = "output/ICARUS_DIS.parquet"
df = pd.read_parquet(file_path)

print("\nColumns:", df.columns)

# =========================
# Extract vertices (FIXED)
# =========================
vertices = []

for v in df["vertex"]:
    try:
        v = list(v)              # convert to list
        v = np.array(v[0])       # extract actual [x,y,z]

        if v.shape[0] == 3:
            vertices.append(v)

    except:
        continue

vertices = np.array(vertices)

if len(vertices) == 0:
    raise ValueError(" No valid vertices extracted")

print("Vertices extracted:", len(vertices))
print("Sample:", vertices[:3])

# =========================
# Split coordinates
# =========================
x = vertices[:, 0]
y = vertices[:, 1]
z = vertices[:, 2]

print("\nDEBUG:")
print("X:", np.min(x), np.max(x))
print("Y:", np.min(y), np.max(y))
print("Z:", np.min(z), np.max(z))

# =========================
# Fiducial mask (ICARUS geometry)
# =========================
# Left module
fid_left = (
    (x > -3.5) & (x < -0.5) &
    (y > -1.5) & (y < 1.5) &
    (z > -8)   & (z < 8)
)

# Right module
fid_right = (
    (x > 0.5) & (x < 3.5) &
    (y > -1.5) & (y < 1.5) &
    (z > -8)   & (z < 8)
)

fid_mask = fid_left | fid_right

print("Fiducial events:", np.sum(fid_mask))

# =========================
# Create plots
# =========================
plt.figure(figsize=(18, 5))

# -------------------------
# XZ projection
# -------------------------
plt.subplot(1, 3, 1)
plt.scatter(y, z, s=1, alpha=0.1, color="blue", label="All")
plt.scatter(x[fid_mask], z[fid_mask], s=2, color="crimson", label="Fiducial")
plt.xlabel("X [m]")
plt.ylabel("Z [m]")
plt.title("XZ view")
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.legend()

# -------------------------
# XY projection (key one)
# -------------------------
#plt.subplot(1, 3, 2)
#plt.scatter(x, y, s=1, alpha=0.05, label="All")
#plt.scatter(x[fid_mask], y[fid_mask], s=2, color="red", label="Fiducial")
#plt.xlabel("X [m]")
#plt.ylabel("Y [m]")
#plt.title("XY view (ICARUS modules)")
#plt.legend()

# -------------------------
# YZ projection
# -------------------------
plt.subplot(1, 3, 3)
plt.scatter(y, z, s=1, alpha=0.1, color="blue", label="All")
plt.scatter(y[fid_mask], z[fid_mask], s=2, color="crimson", label="Fiducial")
plt.xlabel("Y [m]")
plt.ylabel("Z [m]")
plt.title("YZ view")
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.legend()

# =========================
# Save plot
# =========================
plt.tight_layout()
plt.savefig("plots/ICARUS_all_projections.png", dpi=300)

print("\n Plot saved: plots/ICARUS_all_projections.png")
