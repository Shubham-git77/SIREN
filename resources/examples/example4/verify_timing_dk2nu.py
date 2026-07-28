#!/usr/bin/env python
"""
End-to-end verification of the PrimaryExternalDistribution `t0` (initial-time)
feature using real mesons from a dk2nu flux file.

Pipeline:
  1. Read parent mesons (pi+/K+) from a dk2nu.root file: decay vertex, momentum,
     energy, parent type -> PDG mass.
  2. Build a PrimaryExternalDistribution CSV: production point x0,y0,z0 (target
     origin), decay point x,y,z (the dk2nu decay vertex), px,py,pz, E, m, and a
     synthetic per-meson initial time t0 (uniform over the BNB spill by default).
  3. Inject each meson with PrimaryExternalDistribution -> the record gets
     initial_time = t0 and interaction_time = t0 + time-of-flight to the vertex.
  4. Write the timing (primary_initial_time, vertex_time, ...) to parquet + hdf5.
  5. Read the timing back FROM the files and check
        vertex_time - primary_initial_time == distance / (beta * c)
     with beta = sqrt(E^2 - m^2)/E  (SIREN uses the energy/mass velocity, NOT the
     stored 3-momentum; dk2nu mesons are typically a few % off-shell).

Units: SIREN Constants.c ~ 0.29979 => positions in METRES, times in NANOSECONDS
(t0 in ns; the docstring convention 1 s = 1e9).

Run with the from-source siren build that has the t0 feature, e.g.:
    /home/shubham/siren_ubaid_venv/bin/python verify_timing_dk2nu.py \
        --dk2nu /home/shubham/nubeam12M.dk2nu.root --n 3000 --outdir ./timing_out
"""
import argparse
import os
import numpy as np

from siren import distributions, dataclasses, utilities

PDG_MASS = {211: 0.13957, -211: 0.13957, 321: 0.49368, -321: 0.49368}  # GeV
SPECIES = {211: "pi+", 321: "K+", -211: "pi-", -321: "K-"}


def build_csv(dk2nu_path, n, csv_path, spill_ns, seed, species):
    """Read mesons from dk2nu and write the PrimaryExternalDistribution CSV."""
    import uproot
    t = uproot.open(dk2nu_path)["dk2nuTree"]
    # read enough entries to collect n of the requested species
    chunk = max(n * 80, 200000)
    b = lambda name: t["dk2nu/decay/decay." + name].array(entry_stop=chunk, library="np")
    pt = b("ptype")
    keep = np.isin(pt, species)
    idx = np.where(keep)[0][:n]
    if len(idx) < n:
        print("  WARNING: only %d mesons of the requested species in the first %d entries"
              % (len(idx), chunk))
    vx, vy, vz = b("vx")[idx], b("vy")[idx], b("vz")[idx]        # cm
    px, py, pz = b("pdpx")[idx], b("pdpy")[idx], b("pdpz")[idx]  # GeV
    E = b("ppenergy")[idx]                                       # GeV
    m = np.array([PDG_MASS[int(p)] for p in pt[idx]])
    rng = np.random.default_rng(seed)
    prod = np.zeros((len(idx), 3))                              # production ~ target origin [m]
    dec = np.stack([vx, vy, vz], axis=1) / 100.0                # cm -> m
    t0 = rng.uniform(0.0, spill_ns, size=len(idx))              # ns (synthetic beam spill)
    data = np.column_stack([prod, dec, np.stack([px, py, pz], 1), E, m, t0])
    np.savetxt(csv_path, data, delimiter=",",
               header="x0,y0,z0,x,y,z,px,py,pz,E,m,t0", comments="", fmt="%.8g")
    counts = {SPECIES.get(int(k), int(k)): int((pt[idx] == k).sum()) for k in np.unique(pt[idx])}
    print("  CSV: %d mesons %s -> %s" % (len(idx), counts, csv_path))
    return len(idx)


def inject(csv_path, n_sample, seed):
    """Sample n_sample records through PrimaryExternalDistribution; collect timing."""
    d = distributions.PrimaryExternalDistribution(csv_path)
    rng = utilities.SIREN_random(seed)
    rows = []
    for _ in range(n_sample):
        r = dataclasses.PrimaryDistributionRecord(dataclasses.ParticleType.PiPlus)
        d.Sample(rng, None, None, r)
        ip, vt = r.initial_position, r.interaction_vertex
        rows.append((r.initial_time, r.interaction_time, r.length, r.energy, r.mass,
                     float(np.linalg.norm(r.three_momentum)),
                     ip[0], ip[1], ip[2], vt[0], vt[1], vt[2]))
    a = np.array(rows)
    keys = ("primary_initial_time", "vertex_time", "length", "energy", "mass", "pmag",
            "ix", "iy", "iz", "vx", "vy", "vz")
    D = {k: a[:, i] for i, k in enumerate(keys)}
    C = utilities.Constants.c
    D["dist"] = np.sqrt((D["vx"] - D["ix"])**2 + (D["vy"] - D["iy"])**2 + (D["vz"] - D["iz"])**2)
    p_on = np.sqrt(np.maximum(D["energy"]**2 - D["mass"]**2, 0.0))
    D["beta"] = np.divide(p_on, D["energy"], out=np.zeros_like(D["energy"]), where=D["energy"] > 0)
    D["tof"] = D["vertex_time"] - D["primary_initial_time"]
    D["tof_expected"] = np.where(D["beta"] > 0, D["dist"] / np.maximum(D["beta"] * C, 1e-30), 0.0)
    return D


def write_files(D, outdir):
    import pyarrow as pa, pyarrow.parquet as pq, h5py
    cols = ("primary_initial_time", "vertex_time", "length", "dist", "energy",
            "mass", "pmag", "beta", "tof", "tof_expected")
    pq.write_table(pa.table({k: D[k] for k in cols}), os.path.join(outdir, "meson_timing_events.parquet"))
    with h5py.File(os.path.join(outdir, "meson_timing_events.h5"), "w") as h:
        for k in cols:
            h.create_dataset(k, data=D[k])
    print("  wrote meson_timing_events.parquet + .h5 in %s" % outdir)


def verify_from_files(outdir):
    import pyarrow.parquet as pq, h5py
    P = {k: v.to_numpy() for k, v in zip(pq.read_table(os.path.join(outdir, "meson_timing_events.parquet")).column_names,
                                         pq.read_table(os.path.join(outdir, "meson_timing_events.parquet")).columns)}
    with h5py.File(os.path.join(outdir, "meson_timing_events.h5"), "r") as h:
        H = {k: h[k][:] for k in h.keys()}
    ok = True
    for tag, F in (("parquet", P), ("hdf5", H)):
        tof = F["vertex_time"] - F["primary_initial_time"]
        rel = np.abs(tof - F["tof_expected"]) / np.maximum(np.abs(F["tof_expected"]), 1e-9)
        good = rel.max() < 1e-6
        ok = ok and good
        print("  [%-7s] vertex_time - initial_time == dist/(beta c): max rel resid %.2e -> %s"
              % (tag, rel.max(), "PASS" if good else "FAIL"))
    same = max(np.max(np.abs(P[k] - H[k])) for k in H.keys())
    print("  parquet == hdf5: max diff %.2e -> %s" % (same, "PASS" if same < 1e-12 else "FAIL"))
    print("  <t0>=%.1f  <tof>=%.1f  <vertex_time>=%.1f ns  <beta>=%.3f  dist[med]=%.1f m  (n=%d)"
          % (P["primary_initial_time"].mean(), P["tof"].mean(), P["vertex_time"].mean(),
             P["beta"].mean(), np.median(P["dist"]), len(P["vertex_time"])))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dk2nu", default="/home/shubham/nubeam12M.dk2nu.root", help="dk2nu.root flux file")
    ap.add_argument("--n", type=int, default=3000, help="number of mesons to read")
    ap.add_argument("--n-sample", type=int, default=None, help="injection samples (default = --n)")
    ap.add_argument("--outdir", default="./timing_out", help="output directory")
    ap.add_argument("--spill-ns", type=float, default=1600.0, help="synthetic t0 spread [ns]")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--species", type=int, nargs="+", default=[211, 321], help="parent PDGs (default pi+ K+)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, "meson_timing_input.csv")
    n_sample = args.n_sample or args.n

    print("[1/4] build meson CSV from dk2nu ...")
    build_csv(args.dk2nu, args.n, csv_path, args.spill_ns, args.seed, args.species)
    print("[2/4] inject via PrimaryExternalDistribution ...")
    D = inject(csv_path, n_sample, args.seed)
    print("[3/4] write parquet + hdf5 ...")
    write_files(D, args.outdir)
    print("[4/4] verify timing read back FROM the files ...")
    ok = verify_from_files(args.outdir)
    print("\nRESULT:", "TIMING VERIFIED (t0 -> initial_time -> vertex_time by TOF)" if ok else "FAILED")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
