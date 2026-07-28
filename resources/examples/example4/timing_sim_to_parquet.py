#!/usr/bin/env python
"""
Timing SIMULATION -> parquet.  Injects beams of several particle masses via
PrimaryExternalDistribution (each with a beam-spill start time t0), lets SIREN's
timing feature propagate them a baseline L to the detector, and writes ONE parquet
(+ hdf5) with the per-event timing so a separate plotter can read it back.

Output columns (one row per injected event):
    mass                  primary mass [GeV]   (0 = neutrino)
    energy                primary energy [GeV]
    primary_initial_time  t0 [ns]              (from PrimaryExternalDistribution)
    vertex_time           interaction time [ns] = t0 + L/(beta c)   (SIREN)
    dist                  |vertex - initial| [m]
    beta                  |p|/E = sqrt(E^2-m^2)/E
    tof                   vertex_time - t0 [ns]
    delay                 tof - dist/c [ns]     (delay vs a light-speed particle)

Run:  /home/shubham/siren_ubaid_venv/bin/python timing_sim_to_parquet.py
"""
import argparse
import os
import numpy as np

from siren import distributions, dataclasses, utilities

C = utilities.Constants.c   # m/ns


def inject_mass(m, L, e_lo, e_hi, spill, n, seed, scr):
    """Inject n particles of mass m; return per-event dict arrays."""
    rng = np.random.default_rng(seed)
    E = rng.uniform(max(e_lo, m + 1e-3), e_hi, size=n)
    p = np.sqrt(E**2 - m**2)
    t0 = rng.uniform(0.0, spill, size=n)
    rows = np.column_stack([np.zeros((n, 3)),
                            np.zeros(n), np.zeros(n), np.full(n, L),
                            np.zeros(n), np.zeros(n), p,
                            E, np.full(n, m), t0])
    csv = os.path.join(scr, "beam_m%0.3f.csv" % m)
    np.savetxt(csv, rows, delimiter=",",
               header="x0,y0,z0,x,y,z,px,py,pz,E,m,t0", comments="", fmt="%.8g")
    d = distributions.PrimaryExternalDistribution(csv)
    rnd = utilities.SIREN_random(seed)
    out = {k: [] for k in ("mass", "energy", "primary_initial_time", "vertex_time", "dist")}
    for _ in range(n):
        r = dataclasses.PrimaryDistributionRecord(dataclasses.ParticleType.NuMu)
        d.Sample(rnd, None, None, r)
        ip = np.array(r.initial_position); v = np.array(r.interaction_vertex)
        out["mass"].append(r.mass); out["energy"].append(r.energy)
        out["primary_initial_time"].append(r.initial_time)
        out["vertex_time"].append(r.interaction_time)
        out["dist"].append(float(np.linalg.norm(v - ip)))
    return {k: np.array(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=float, default=110.0, help="baseline [m]")
    ap.add_argument("--e-lo", type=float, default=0.05)
    ap.add_argument("--e-hi", type=float, default=1.0)
    ap.add_argument("--spill-ns", type=float, default=100.0)
    ap.add_argument("--masses", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.20], help="GeV")
    ap.add_argument("--n", type=int, default=4000, help="events per mass")
    ap.add_argument("--outdir", default="output/timing_phys")
    args = ap.parse_args()

    scr = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.outdir)
    os.makedirs(scr, exist_ok=True)

    parts = [inject_mass(m, args.L, args.e_lo, args.e_hi, args.spill_ns, args.n, 10 + i, scr)
             for i, m in enumerate(args.masses)]
    D = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    D["beta"] = np.sqrt(np.maximum(D["energy"]**2 - D["mass"]**2, 0.0)) / D["energy"]
    D["tof"] = D["vertex_time"] - D["primary_initial_time"]
    D["delay"] = D["tof"] - D["dist"] / C

    import pyarrow as pa, pyarrow.parquet as pq, h5py
    cols = ["mass", "energy", "primary_initial_time", "vertex_time", "dist", "beta", "tof", "delay"]
    pq_path = os.path.join(scr, "timing_massive_events.parquet")
    pq.write_table(pa.table({c: D[c] for c in cols}), pq_path)
    with h5py.File(os.path.join(scr, "timing_massive_events.h5"), "w") as h:
        for c in cols:
            h.create_dataset(c, data=D[c])
    # store the baseline as parquet key-value metadata (handy for the plotter)
    print("wrote %d events (%d masses) -> %s" % (len(D["mass"]), len(args.masses), pq_path))
    for m in args.masses:
        sel = D["mass"] == m
        print("  m=%3d MeV:  <beta>=%.3f  median delay=%.1f ns  max delay=%.1f ns  (N=%d)"
              % (m * 1e3, D["beta"][sel].mean(), np.median(D["delay"][sel]), D["delay"][sel].max(), sel.sum()))


if __name__ == "__main__":
    main()
