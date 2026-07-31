r"""BNB DarkNews HNL run that records VERTEX POSITIONS as well as times.

Same physics as DarkNewsHNL_SBND_BNB_timing.py / DarkNewsHNL_ICARUS_BNB_timing.py
-- it imports the model, vertices, bias and flux handling from
DarkNewsHNL_SBN_dk2nu_timing.py unchanged.  The only difference is what gets
written out: the reference collect_timing() collapses each vertex to a time and
a distance, so the CSV carries no coordinates and position plots have to
reconstruct a radius from the timing.  This driver keeps the raw
`interaction_vertex` of every leg, so real geometry plots are possible.

Those existing scripts are left untouched; this one writes its own CSV.

EXTRA COLUMNS (on top of every column the reference CSV has, so the existing
post-processors still run on this file unchanged):
    parent_vx/vy/vz  pion decay vertex        [m, detector frame]
    ux/uy/uz         upscatter vertex         [m]   nu -> N4 production
    dx/dy/dz         N4 decay vertex          [m]   N4 -> nu gamma
    n4_px/py/pz      N4 momentum              [GeV]
    n4_flight_dist   |decay - upscatter|      [m]
    n4_beta          sqrt(1 - (m4/E)^2)
    in_detector      1 if the decay vertex is inside the fiducial box

The ux/uy/uz, dx/dy/dz, n4_flight_dist, n4_beta and in_detector names match
sbnd_dirt.py in the SIREN_ubaid tree on purpose, so the dirt plotting
conventions carry over to these in-argon runs.

NOTE ON WHAT THIS RUN IS: the model is built with nuclear_targets=["Ar40"],
so upscatter happens only in liquid argon -- both vertices are in the TPC and
there is no upstream dirt population.  See the caveats in
timing_flight_geometry.py.

Run (one detector per invocation):
  /home/shubham/siren_pr178_venv/bin/python DarkNewsHNL_SBN_BNB_positions.py \
      --detector ICARUS --events 1500
Out: output/<detector>_bnb_hnl_positions.csv  (+ the standard timing PNG)
"""
import argparse
import os

import numpy as np

import siren
from siren import dk2nu
from siren.injection import Injector
from siren.Weighter import Weighter

import DarkNewsHNL_SBN_dk2nu_timing as ref

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

BEAM_FRAME = "BNB"
DEFAULT_FLUX = "/home/shubham/nubeam12M.dk2nu.root"

_NAN3 = (np.nan, np.nan, np.nan)


def _vertex(record):
    """The interaction vertex of a record as a plain 3-vector, or NaNs."""
    if record is None:
        return _NAN3
    v = np.asarray(record.interaction_vertex, dtype=float)
    return (float(v[0]), float(v[1]), float(v[2]))


def collect_positions(results, detector_model, fiducial_center,
                      fiducial_widths, m4,
                      prompt_origin_geometry=ref.BNB_TARGET_CENTER):
    """collect_timing(), plus the vertex coordinates of every leg."""
    columns = ref.collect_timing(
        results, detector_model, prompt_origin_geometry=prompt_origin_geometry)

    extra = {name: [] for name in (
        "parent_vx", "parent_vy", "parent_vz",
        "ux", "uy", "uz", "dx", "dy", "dz",
        "n4_px", "n4_py", "n4_pz",
        "n4_flight_dist", "n4_beta", "in_detector")}

    center = np.asarray(fiducial_center, dtype=float)
    half = 0.5 * np.asarray(fiducial_widths, dtype=float)

    for event, _weight in results:
        root = event.tree[0].record
        upscatter = ref._record_for_primary(event, siren.particles.NuMu)
        hnl_decay = ref._record_for_primary(event, siren.particles.N4)

        pv = _vertex(root)
        uv = _vertex(upscatter)
        dv = _vertex(hnl_decay)

        n4_p = _NAN3
        beta = np.nan
        if hnl_decay is not None:
            p = np.asarray(hnl_decay.primary_momentum, dtype=float)
            n4_p = (float(p[1]), float(p[2]), float(p[3]))
            # beta from energy and mass, matching SIREN's own time-of-flight
            # convention (dk2nu momenta are slightly off shell).
            energy = float(p[0])
            if energy > m4:
                beta = float(np.sqrt(1.0 - (m4 / energy) ** 2))

        flight = np.nan
        if not np.isnan(uv[0]) and not np.isnan(dv[0]):
            flight = float(np.linalg.norm(np.array(dv) - np.array(uv)))

        # Real containment test against the fiducial box, not a timing proxy.
        inside = 0.0
        if not np.isnan(dv[0]):
            inside = float(np.all(np.abs(np.array(dv) - center) <= half))

        for name, value in (
                ("parent_vx", pv[0]), ("parent_vy", pv[1]), ("parent_vz", pv[2]),
                ("ux", uv[0]), ("uy", uv[1]), ("uz", uv[2]),
                ("dx", dv[0]), ("dy", dv[1]), ("dz", dv[2]),
                ("n4_px", n4_p[0]), ("n4_py", n4_p[1]), ("n4_pz", n4_p[2]),
                ("n4_flight_dist", flight), ("n4_beta", beta),
                ("in_detector", inside)):
            extra[name].append(value)

    columns.update({k: np.asarray(v, dtype=float) for k, v in extra.items()})
    return columns


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dk2nu_files", nargs="*",
                        help="BNB dk2nu ROOT file(s) or glob(s); default: %s"
                             % DEFAULT_FLUX)
    parser.add_argument("--detector", choices=sorted(ref._FIDUCIALS),
                        default="SBND")
    parser.add_argument("--events", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--entry-stop", type=int,
                        help="read at most this many rows per ROOT file")
    parser.add_argument("--m4", type=float, default=0.10,
                        help="HNL mass in GeV (default: 0.10)")
    parser.add_argument("--mu-tr-mu4", type=float, default=2.5e-6,
                        help="transition dipole coupling in GeV^-1")
    parser.add_argument("--output", help="CSV path (default: "
                                         "output/<detector>_bnb_hnl_positions.csv)")
    parser.add_argument("--table-emax", type=float)
    parser.add_argument("--no-precompute-tables", action="store_true")
    args = parser.parse_args(argv)

    detector = args.detector

    files = ref._expand_input_paths(args.dk2nu_files or [DEFAULT_FLUX])
    if not files:
        parser.error("none of the dk2nu paths/globs matched a file")

    data = dk2nu.read_dk2nu(files, parent_pdg=dk2nu.PTYPE_PIPLUS,
                            decay_modes=13, nu_pdg=14, read_time=True,
                            entry_stop=args.entry_stop)
    dk2nu.print_summary(data)
    if "t0" not in data:
        raise RuntimeError("These dk2nu files have no ancestor start-time "
                           "branches; this example needs them.")
    if len(data["E"]) == 0:
        raise RuntimeError("No pi+ -> mu+ nu_mu rows survived the filters")

    detector_model = siren.load_detector("SBN", detector=detector)
    frame = siren.resources.detectors.SBN.geo.transform(BEAM_FRAME, "BNB")
    bias = ref._make_threshold_bias(args.m4)
    open_rows = bias(data["E"], data["px"], data["py"], data["pz"],
                     data["vx"], data["vy"], data["vz"])
    if int(np.count_nonzero(open_rows)) == 0:
        raise RuntimeError("No dk2nu row can reach the N4 threshold for "
                           "m4 = %g GeV" % args.m4)
    print("Threshold bias keeps %d of %d rows for sampling"
          % (int(np.count_nonzero(open_rows)), len(open_rows)))

    external = dk2nu.dk2nu_to_primary_distribution(
        data, detector_model, frame=frame, sampling_bias=bias)
    prompt_origin = ref._prompt_origin_geometry(BEAM_FRAME, frame)
    spec = ref._FIDUCIALS[detector]
    fiducial = siren.geometry.Box(widths=spec["widths"], center=spec["center"])

    if args.no_precompute_tables:
        table_emax = None
    else:
        table_emax = args.table_emax or ref._max_neutrino_energy(data)
        print("Precomputing DarkNews tables up to %.3f GeV" % table_emax)

    primary, secondaries = ref.build_vertices(
        detector_model, external, fiducial, args.m4, args.mu_tr_mu4,
        table_emax=table_emax)

    injector = Injector(detector=detector_model, primary=primary,
                        secondaries=secondaries, events=args.events,
                        seed=args.seed)
    weighter = Weighter(injector, primary_physical=primary.physical)
    results = siren.generate(injector, weighter, events=args.events,
                             on_shortfall="warn")
    results.summary()
    print(injector.report())

    columns = collect_positions(
        results, detector_model, spec["center"], spec["widths"], args.m4,
        prompt_origin_geometry=prompt_origin)

    csv_path = args.output or os.path.join(
        "output", "%s_bnb_hnl_positions.csv" % detector.lower())
    ref.save_timing_csv(columns, csv_path)
    png = ref.plot_timing(columns, os.path.splitext(csv_path)[0] + ".png",
                          detector, beam_frame=BEAM_FRAME)

    inside = np.nansum(columns["in_detector"])
    total = np.sum(np.isfinite(columns["dx"]))
    print("Decay vertex inside the fiducial box: %d of %d completed decays"
          % (int(inside), int(total)))
    print("Wrote %s" % os.path.abspath(csv_path))
    print("Wrote %s" % png)


if __name__ == "__main__":
    main()
