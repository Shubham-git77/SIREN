#!/usr/bin/env python
"""
DIF/DAR-capable variant of dk2nu_to_upscattering_csv.py.

Same output as dk2nu_to_upscattering_csv.py -- a PrimaryExternalDistribution CSV
of neutrinos (x0,y0,z0,px,py,pz,E,m,t0) in SBND DETECTOR coordinates, feeding
sbnd.py / sbnd_dirt.py -- but it can MERGE a split decay-in-flight (DIF) and
decay-at-rest (DAR) dk2nu production the way SIREN PR #178 does.

Why a separate script: the original dk2nu_to_upscattering_csv.py is kept intact
(single production, no POT bookkeeping). This one additionally reads the parent
species/momentum and the per-sample POT so it can classify DIF vs DAR by parent
kinetic energy and normalize consistently. The merge itself lives in the shared
helper _beam_samples.py (ported from PR #178, adapted to our neutrino-level
sample dicts).

The merge (see _beam_samples.combine_dif_dar):
  - keep DIF rows with parent kinetic energy >= --dar-ke-cut (default 0.05 GeV,
    the g4numi kill threshold);
  - keep DAR rows with parent kinetic energy < the cut;
  - rescale DAR flux weights by dif_pot / dar_pot so the merged sample is
    normalized to the DIF POT, with no double counting.
Downstream flux-weight resampling then draws from the merged, correctly
normalized weights exactly as before.

Examples
--------
  # single production (behaves like dk2nu_to_upscattering_csv.py)
  python dk2nu_to_upscattering_csv_difdar.py \
      --dk2nu /home/shubham/nubeam12M.dk2nu.root \
      --n 100000 --out pion_derived_upscattering_events.csv

  # merged DIF + DAR
  python dk2nu_to_upscattering_csv_difdar.py \
      --dk2nu '/data/g4numi/*_fhc_1*.root' \
      --dar-files '/data/g4numi/*_fhc_dar_*.root' \
      --dar-ke-cut 0.05 \
      --n 100000 --out pion_derived_upscattering_events_difdar.csv
"""
import argparse
import glob
import os

import numpy as np

# Reuse the frame/index helpers from the original ingestion script (unchanged),
# and the DIF/DAR merge from the shared helper. Both resolve because Python puts
# this script's own directory on sys.path when it is run as a script.
from dk2nu_to_upscattering_csv import (
    NU_NAME, detector_center_bnb, sbnd_nuray_index)
import _beam_samples


def _expand(paths):
    out = []
    for value in paths:
        matches = sorted(glob.glob(value))
        if matches:
            out.extend(matches)
        elif os.path.isfile(value):
            out.append(value)
    return list(dict.fromkeys(os.path.abspath(p) for p in out))


def read_sample(files, chunk, ntypes):
    """Read a NEUTRINO-level sample dict from one or more dk2nu files.

    Returns the dict shape consumed by _beam_samples.combine_dif_dar:
    prod,p,E,wgt,ntype,ptype,parent_pmag (per-neutrino arrays) and pot (float,
    summed over the file set).
    """
    import uproot
    prods, ps, Es, wgts, ntys, ptys, pmags = [], [], [], [], [], [], []
    pot = 0.0
    for path in files:
        nuray_idx, loc_names = sbnd_nuray_index(path)
        f = uproot.open(path)
        pot += float(np.sum(f["dkmetaTree"]["dkmeta/pots"].array(library="np")))
        t = f["dk2nuTree"]
        ntype = t["dk2nu/decay/decay.ntype"].array(entry_stop=chunk, library="np")
        idx = np.where(np.isin(ntype, ntypes))[0]
        if idx.size == 0:
            continue

        def scalar(br):
            return t[br].array(entry_stop=chunk, library="np")[idx]

        def ray(comp):
            a = t["dk2nu/nuray/nuray." + comp].array(entry_stop=chunk)
            return np.asarray(a[:, nuray_idx].to_numpy())[idx]

        prods.append(np.stack([scalar("dk2nu/decay/decay.vx"),
                               scalar("dk2nu/decay/decay.vy"),
                               scalar("dk2nu/decay/decay.vz")], axis=1) / 100.0)
        ps.append(np.stack([ray("px"), ray("py"), ray("pz")], axis=1))
        Es.append(ray("E"))
        wgts.append(ray("wgt") * scalar("dk2nu/decay/decay.nimpwt"))
        ntys.append(ntype[idx])
        ptys.append(scalar("dk2nu/decay/decay.ptype"))
        pd = np.stack([scalar("dk2nu/decay/decay.pdpx"),
                       scalar("dk2nu/decay/decay.pdpy"),
                       scalar("dk2nu/decay/decay.pdpz")], axis=1)
        pmags.append(np.linalg.norm(pd, axis=1))
    if not prods:
        raise RuntimeError("No neutrinos of ntype %s in the given files" % (ntypes,))
    return dict(
        prod=np.concatenate(prods), p=np.concatenate(ps), E=np.concatenate(Es),
        wgt=np.concatenate(wgts), ntype=np.concatenate(ntys),
        ptype=np.concatenate(ptys), parent_pmag=np.concatenate(pmags), pot=pot)


def build(dif_files, dar_files, n, out, ntypes, spill_ns, seed, use_weight,
          chunk, ke_cut):
    dif = read_sample(dif_files, chunk, ntypes)
    print("  DIF sample: %d rows, POT=%.4g" % (len(dif["E"]), dif["pot"]))

    if dar_files:
        dar = read_sample(dar_files, chunk, ntypes)
        print("  DAR sample: %d rows, POT=%.4g" % (len(dar["E"]), dar["pot"]))
        data = _beam_samples.combine_dif_dar(dif, dar, kinetic_energy_cut=ke_cut)
        print("  merged: %d rows (%d DIF + %d DAR); normalized to DIF POT=%.4g; "
              "DAR flux weights x %.4g"
              % (len(data["E"]), int(np.sum(~data["dar"])), int(np.sum(data["dar"])),
                 data["pot"], dif["pot"] / dar["pot"]))
    else:
        data = dif
        print("  (no --dar-files: single production)")

    prod, p, E, wgt = data["prod"], data["p"], data["E"], data["wgt"]
    ntype_kept = data["ntype"]

    # BNB beam frame -> SBND detector frame (SIREN record positions are DETECTOR
    # coordinates). Pure translation; axes coincide.
    c_det = detector_center_bnb("SBND")
    prod = prod - c_det
    print("  BNB -> detector translation: -(%.4f, %.4f, %.4f) m" % tuple(c_det))
    print("  neutrino species: %s"
          % {NU_NAME.get(int(k), int(k)): int((ntype_kept == k).sum())
             for k in np.unique(ntype_kept)})

    # forward-going toward SBND (nuray z-momentum > 0); |p| > 0
    pmag = np.linalg.norm(p, axis=1)
    good = (pmag > 0) & (p[:, 2] > 0)
    prod, p, E, wgt = prod[good], p[good], E[good], wgt[good]

    rng = np.random.default_rng(seed)
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

    prod, p, E = prod[sel], p[sel], E[sel]
    t0 = rng.uniform(0.0, spill_ns, size=len(E))    # ns, synthetic beam spill
    m = np.zeros(len(E))                             # massless -> ToF at c

    data_out = np.column_stack([prod, p, E, m, t0])
    header = "x0,y0,z0,px,py,pz,E,m,t0"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savetxt(out, data_out, delimiter=",", header=header, comments="", fmt="%.8g")
    print("  wrote %d rows -> %s  (header: %s)" % (len(data_out), out, header))
    print("  <E>=%.3f GeV  production z in [%.1f, %.1f] m  <t0>=%.1f ns"
          % (E.mean(), prod[:, 2].min(), prod[:, 2].max(), t0.mean()))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dk2nu", nargs="+",
                    default=["/home/shubham/nubeam12M.dk2nu.root"],
                    help="decay-in-flight dk2nu file(s) or quoted glob(s)")
    ap.add_argument("--dar-files", nargs="+",
                    help="decay-at-rest dk2nu file(s)/glob(s) to merge in")
    ap.add_argument("--dar-ke-cut", type=float, default=0.05,
                    help="parent kinetic-energy boundary [GeV] (default 0.05, "
                         "the g4numi kill threshold)")
    ap.add_argument("--n", type=int, default=100000,
                    help="number of CSV rows to write")
    ap.add_argument("--out", default="pion_derived_upscattering_events_difdar.csv")
    ap.add_argument("--ntype", type=int, nargs="+", default=[14],
                    help="neutrino PDGs to keep (default 14 = nu_mu)")
    ap.add_argument("--spill-ns", type=float, default=1600.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-weight", dest="weight", action="store_false",
                    help="disable flux-weight resampling (take first n)")
    ap.add_argument("--chunk", type=int, default=None,
                    help="dk2nu entries to scan per file (default max(n*8, 400000))")
    args = ap.parse_args()

    dif_files = _expand(args.dk2nu)
    if not dif_files:
        ap.error("none of the --dk2nu paths/globs matched a file")
    dar_files = _expand(args.dar_files) if args.dar_files else None
    if args.dar_files and not dar_files:
        ap.error("none of the --dar-files paths/globs matched a file")

    chunk = args.chunk or max(args.n * 8, 400000)
    print("[dk2nu DIF/DAR -> upscattering CSV] scanning up to %d entries/file ..."
          % chunk)
    build(dif_files, dar_files, args.n, args.out, args.ntype,
          args.spill_ns, args.seed, args.weight, chunk, args.dar_ke_cut)


if __name__ == "__main__":
    main()
