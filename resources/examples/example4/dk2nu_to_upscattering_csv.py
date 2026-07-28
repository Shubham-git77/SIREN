#!/usr/bin/env python
"""
Convert a dk2nu.root flux file into a PrimaryExternalDistribution CSV of
NEUTRINOS (production point + kinematics + initial time) for SBND, feeding
sbnd.py, where they upscatter to an HNL (nu -> N4) INSIDE the detector.

This is the neutrino-level counterpart of verify_timing_dk2nu.py (which writes
the parent meson). Every CSV row is a neutrino taken from the dk2nu NEUTRINO
branch (`dk2nu/nuray/...`) projected to the SBND location.

Division of labour (IMPORTANT -- read before changing this file)
----------------------------------------------------------------
PrimaryExternalDistribution is NOT a VertexPositionDistribution, so it cannot
be the sole primary distribution: the Injector requires a real vertex
distribution (it calls FindPrimaryVertexDistribution and exit(0)s otherwise).
The intended use of the t0 feature is as an ADD-ON: this CSV supplies the
neutrino's production point (x0,y0,z0), momentum (px,py,pz), energy E, mass m,
and initial time t0; sbnd.py pairs it with a PrimaryBoundedVertexDistribution
that samples the actual upscattering VERTEX inside the SBND active volume,
along the neutrino direction, weighted by cross-section. SIREN then sets
    primary.initial_time     = t0                       (production time)
    primary.interaction_time = t0 + FlightTime(x0 -> vertex)
                             = time the upscattering happens IN SBND
with v = c because m = 0. So this file deliberately does NOT emit x,y,z --
leave the vertex to SIREN.

Pipeline
--------
  1. Read neutrinos from dk2nu:
       - flavor              : dk2nu/decay/decay.ntype        (14 = nu_mu)
       - production vertex   : dk2nu/decay/decay.v{x,y,z}     [cm] (meson decay pt)
       - neutrino kinematics : dk2nu/nuray/nuray.{px,py,pz,E} at the SBND ray
       - flux weight         : nuray.wgt * decay.nimpwt
     The correct nuray index is found by name from dkmetaTree (SBND).
  2. Keep forward-going neutrinos (pz>0 toward SBND) and importance-resample by
     flux weight (default ON) so that PrimaryExternalDistribution -- which draws
     rows UNIFORMLY -- reproduces the physical flux shape (an unknown "weight"
     column would be treated as an interaction parameter, not a sampling weight).
  3. Assign each neutrino a production time t0 ~ U(0, spill) [ns].
  4. Write the CSV: x0,y0,z0,px,py,pz,E,m,t0.

Units and FRAME: positions in METRES in DETECTOR coordinates, momenta/energy
in GeV, time in NANOSECONDS (SIREN Constants.c ~ 0.29979 m/ns).

    COORDINATE FRAME (the bug this fixes): SIREN interprets every record
    position as DETECTOR-frame -- Injector.cxx wraps record.interaction_vertex
    directly in DetectorPosition(), no geometry->detector conversion. dk2nu
    decay vertices are in the BNB beam frame (SBND detector origin sits at
    BNB (0.7378, -0.59, 112.92) m). Writing raw BNB coordinates put the
    neutrino production points ~110 m downstream of truth -- inside/behind
    the detector, flying away from it -- so upscatter vertices landed in the
    dirt (vol_glacial_till) and the SBND rate came out low. The production
    points are now translated BNB -> detector (axes are identical, pure
    translation from sbn_geometry.detector_center("SBND", "BNB")).

Run with the from-source siren venv (has uproot + the t0 feature):
    /home/shubham/siren_ubaid_venv/bin/python dk2nu_to_upscattering_csv.py \
        --dk2nu /home/shubham/nubeam12M.dk2nu.root \
        --n 100000 --out pion_derived_upscattering_events.csv
"""
import argparse
import os
import numpy as np

NU_NAME = {12: "nu_e", -12: "nu_e_bar", 14: "nu_mu", -14: "nu_mu_bar",
           16: "nu_tau", -16: "nu_tau_bar"}


def detector_center_bnb(detector="SBND"):
    """Detector origin (LAr geometric center) in the BNB frame [m], from the
    in-tree SBN frame graph. Detector axes coincide with BNB axes for the
    SBN LArTPCs, so BNB -> detector is the pure translation r_det = r_bnb - c."""
    import importlib.util
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    geo_path = os.path.normpath(os.path.join(
        here, "..", "..", "detectors", "SBN", "SBN-v1", "sbn_geometry.py"))
    spec = importlib.util.spec_from_file_location("sbn_geometry", geo_path)
    geo = importlib.util.module_from_spec(spec)
    sys.modules["sbn_geometry"] = geo   # dataclass decorator needs the module registered
    spec.loader.exec_module(geo)
    return np.asarray(geo.detector_center(detector, "BNB"), dtype=float)


def sbnd_nuray_index(dk2nu_path):
    """Return the nuray sub-index for SBND, found by name in dkmetaTree.

    nuray[0] is the random-decay ray; nuray[1+k] is the neutrino projected to
    the k-th entry of dkmeta/location. So SBND's nuray index = 1 + loc_index.
    """
    import uproot
    meta = uproot.open(dk2nu_path)["dkmetaTree"]
    names = meta["dkmeta/location/location.name"].array(entry_stop=1, library="np")[0]
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in names]
    for k, nm in enumerate(names):
        if nm.strip().upper() == "SBND":
            return 1 + k, names
    raise RuntimeError("No 'SBND' location in dkmeta; found %s" % names)


def read_neutrinos(dk2nu_path, chunk, ntypes, nuray_idx):
    """Read up to `chunk` entries; return per-neutrino arrays at the SBND ray."""
    import uproot
    t = uproot.open(dk2nu_path)["dk2nuTree"]

    ntype = t["dk2nu/decay/decay.ntype"].array(entry_stop=chunk, library="np")
    keep = np.isin(ntype, ntypes)
    idx = np.where(keep)[0]
    if idx.size == 0:
        raise RuntimeError("No neutrinos of ntype %s in first %d entries"
                           % (ntypes, chunk))

    def scalar(br):
        return t[br].array(entry_stop=chunk, library="np")[idx]

    def ray(comp):
        # jagged (n_entries x 6) -> take the SBND sub-index for kept rows
        a = t["dk2nu/nuray/nuray." + comp].array(entry_stop=chunk)
        return np.asarray(a[:, nuray_idx].to_numpy())[idx]

    prod = np.stack([scalar("dk2nu/decay/decay.vx"),
                     scalar("dk2nu/decay/decay.vy"),
                     scalar("dk2nu/decay/decay.vz")], axis=1) / 100.0   # cm -> m
    p = np.stack([ray("px"), ray("py"), ray("pz")], axis=1)            # GeV
    E = ray("E")                                                       # GeV
    wgt = ray("wgt") * scalar("dk2nu/decay/decay.nimpwt")             # flux weight
    ntype_kept = ntype[idx]
    # Real pion decay time after the primary proton [ns] = the last ancestor's
    # start time. Used as the physical production t0 when --real-t0 is passed
    # (instead of a synthetic uniform spill). Jagged branch -> take last entry.
    startt = t["dk2nu/ancestor/ancestor.startt"].array(entry_stop=chunk)
    startt_last = np.asarray(startt[:, -1].to_numpy())[idx]
    return prod, p, E, wgt, ntype_kept, startt_last


def build(dk2nu_path, n, out, ntypes, spill_ns, seed, use_weight, chunk, real_t0=False):
    nuray_idx, loc_names = sbnd_nuray_index(dk2nu_path)
    print("  dkmeta locations:", loc_names, "-> SBND is nuray[%d]" % nuray_idx)

    prod, p, E, wgt, ntype_kept, startt = read_neutrinos(
        dk2nu_path, chunk, ntypes, nuray_idx)

    # BNB beam frame -> SBND detector frame (see docstring: SIREN record
    # positions are DETECTOR coordinates). Pure translation; axes coincide.
    c_det = detector_center_bnb("SBND")
    prod = prod - c_det
    print("  BNB -> detector translation: -(%.4f, %.4f, %.4f) m" % tuple(c_det))
    print("  read %d neutrinos %s"
          % (len(E), {NU_NAME.get(int(k), int(k)): int((ntype_kept == k).sum())
                      for k in np.unique(ntype_kept)}))

    # forward-going toward SBND (nuray z-momentum > 0); |p| > 0
    pmag = np.linalg.norm(p, axis=1)
    good = (pmag > 0) & (p[:, 2] > 0)
    prod, p, E, wgt, startt = prod[good], p[good], E[good], wgt[good], startt[good]

    rng = np.random.default_rng(seed)

    # select n rows: importance-sample by flux weight (so uniform draws in
    # PrimaryExternalDistribution reproduce the flux), else first n.
    if use_weight:
        w = np.clip(wgt, 0, None)
        if w.sum() <= 0:
            raise RuntimeError("All flux weights <= 0; use --no-weight")
        sel = rng.choice(len(E), size=n, replace=True, p=w / w.sum())
    else:
        if len(E) < n:
            print("  WARNING: only %d forward neutrinos (< n=%d); using all"
                  % (len(E), n))
        sel = np.arange(min(n, len(E)))

    prod, p, E, startt = prod[sel], p[sel], E[sel], startt[sel]
    if real_t0:
        # Physical production time: the real pion decay time after the proton.
        t0 = startt
        print("  t0 = REAL pion decay time (ancestor.startt): [%.1f, %.1f] ns, mean %.1f"
              % (t0.min(), t0.max(), t0.mean()))
    else:
        t0 = rng.uniform(0.0, spill_ns, size=len(E))    # ns, synthetic beam spill

    # Massless neutrino: m=0 forces the flight-time velocity to c. WITHOUT an
    # explicit m column the record would back-compute m = sqrt(E^2 - |p|^2)
    # from the independently rounded E and px,py,pz; since the neutrino is
    # on-shell, |p| > E about half the time after 8-sig-fig rounding, giving a
    # nan/>=E mass -> momentum 0 -> FlightTime() returns 0 (no ToF added).
    m = np.zeros(len(E))

    data = np.column_stack([prod, p, E, m, t0])
    header = "x0,y0,z0,px,py,pz,E,m,t0"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savetxt(out, data, delimiter=",", header=header, comments="", fmt="%.8g")
    print("  wrote %d rows -> %s  (header: %s)" % (len(data), out, header))
    print("  <E>=%.3f GeV  production z in [%.1f, %.1f] m  <t0>=%.1f ns"
          % (E.mean(), prod[:, 2].min(), prod[:, 2].max(), t0.mean()))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dk2nu", default="/home/shubham/nubeam12M.dk2nu.root")
    ap.add_argument("--n", type=int, default=100000,
                    help="number of CSV rows to write (default matches sbnd.py)")
    ap.add_argument("--out", default="pion_derived_upscattering_events.csv",
                    help="output CSV (default = the file sbnd.py loads)")
    ap.add_argument("--ntype", type=int, nargs="+", default=[14],
                    help="neutrino PDGs to keep (default 14 = nu_mu)")
    ap.add_argument("--spill-ns", type=float, default=1600.0,
                    help="synthetic beam-spill width for t0 [ns] (ignored with --real-t0)")
    ap.add_argument("--real-t0", action="store_true",
                    help="use the REAL pion decay time (dk2nu ancestor.startt) as t0 "
                         "instead of a synthetic uniform spill (physical timing chain)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-weight", dest="weight", action="store_false",
                    help="disable flux-weight resampling (take first n)")
    ap.add_argument("--chunk", type=int, default=None,
                    help="dk2nu entries to scan (default max(n*8, 400000))")
    args = ap.parse_args()

    chunk = args.chunk or max(args.n * 8, 400000)
    print("[dk2nu -> upscattering CSV] scanning up to %d entries ..." % chunk)
    build(args.dk2nu, args.n, args.out, args.ntype,
          args.spill_ns, args.seed, args.weight, chunk, real_t0=args.real_t0)


if __name__ == "__main__":
    main()
