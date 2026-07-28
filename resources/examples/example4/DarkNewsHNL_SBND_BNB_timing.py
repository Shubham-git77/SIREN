r"""BNB dk2nu -> SBND DarkNews HNL timing (standalone SBND-only driver).

A single-detector variant of DarkNewsHNL_SBN_dk2nu_timing.py with the detector
pinned to SBND and the coordinate frame pinned to BNB, so it runs on a BNB
dk2nu file with no --detector / --beam-frame flags.  The physics chain,
threshold biasing, DarkNews table precompute, timing extraction and plotting
are imported unchanged from that reference example -- only the driver differs.
See DarkNewsHNL_SBN_dk2nu_timing.py and README_beam_timing.md for the details.

    dk2nu pi+ -> mu+ nu_mu -> N4 + Ar -> nu + gamma        (SBND, BNB frame)

Every SBN detector model uses the BNB/SAND frame for geometry, so a G4BNB
file needs no coordinate transform.  Times are nanoseconds relative to the
primary proton; the dk2nu ancestor chain supplies the pion decay time and
SIREN propagates the neutrino and HNL times of flight to later vertices.

Example::

    python DarkNewsHNL_SBND_BNB_timing.py                       # default BNB flux
    python DarkNewsHNL_SBND_BNB_timing.py /path/nubeam.dk2nu.root \
        --events 500 --m4 0.10 --mu-tr-mu4 2.5e-6
"""

import argparse
import os

import numpy as np

import siren
from siren import dk2nu
from siren.Injector import Injector
from siren.Weighter import Weighter

# Reuse the reference example's physics + analysis, exactly as
# VectorPortal_SBN_dk2nu_timing.py reuses VectorPortal_SBND_dk2nu.py.
import DarkNewsHNL_SBN_dk2nu_timing as ref

# This driver is SBND-only, BNB-only.
DETECTOR = "SBND"
BEAM_FRAME = "BNB"
# Local BNB dk2nu flux (override by passing a positional path/glob).
DEFAULT_FLUX = "/home/shubham/nubeam12M.dk2nu.root"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "dk2nu_files", nargs="*",
        help=("BNB dk2nu ROOT file(s) or quoted glob(s); "
              "default: %s" % DEFAULT_FLUX))
    parser.add_argument("--events", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--entry-stop", type=int,
                        help="read at most this many rows per ROOT file")
    parser.add_argument("--m4", type=float, default=0.10,
                        help="HNL mass in GeV (default: 0.10)")
    parser.add_argument("--mu-tr-mu4", type=float, default=2.5e-6,
                        help="transition dipole coupling in GeV^-1")
    parser.add_argument("--output",
                        help=("PNG path (default: output/%s_bnb_"
                              "darknews_hnl_timing.png)" % DETECTOR.lower()))
    parser.add_argument(
        "--table-emax", type=float,
        help=("fill the DarkNews interpolation tables up to this neutrino "
              "energy in GeV before injecting (default: the forward Doppler "
              "bound of the loaded dk2nu rows)"))
    parser.add_argument(
        "--no-precompute-tables", action="store_true",
        help=("skip the up-front table fill and save; build the DarkNews "
              "tables lazily during injection"))
    args = parser.parse_args(argv)

    files = ref._expand_input_paths(args.dk2nu_files or [DEFAULT_FLUX])
    if not files:
        parser.error("none of the dk2nu paths/globs matched a file "
                     "(default %s)" % DEFAULT_FLUX)

    read_kwargs = dict(
        parent_pdg=dk2nu.PTYPE_PIPLUS,
        decay_modes=13,
        nu_pdg=14,
        read_time=True,
        entry_stop=args.entry_stop,
    )
    data = dk2nu.read_dk2nu(files, **read_kwargs)
    dk2nu.print_summary(data)
    if "t0" not in data:
        raise RuntimeError(
            "These dk2nu files have no ancestor start-time branches. "
            "They can drive event generation, but not this timing example.")
    if len(data["E"]) == 0:
        raise RuntimeError("No pi+ -> mu+ nu_mu rows survived the filters")

    detector_model = siren.load_detector("SBN", detector=DETECTOR)
    # BNB file already in the SBN geometry frame: transform("BNB", "BNB") is
    # the identity, kept explicit to match the reference driver.
    frame = siren.resources.detectors.SBN.geo.transform(BEAM_FRAME, "BNB")
    bias = ref._make_threshold_bias(args.m4)
    open_rows = bias(data["E"], data["px"], data["py"], data["pz"],
                     data["vx"], data["vy"], data["vz"])
    n_open = int(np.count_nonzero(open_rows))
    if n_open == 0:
        raise RuntimeError(
            "No dk2nu row can reach the N4 threshold for m4 = %g GeV"
            % args.m4)
    print("Threshold bias keeps %d of %d rows for sampling"
          % (n_open, len(open_rows)))
    external = dk2nu.dk2nu_to_primary_distribution(
        data, detector_model, frame=frame, sampling_bias=bias)
    prompt_origin = ref._prompt_origin_geometry(BEAM_FRAME, frame)
    spec = ref._FIDUCIALS[DETECTOR]
    fiducial = siren.geometry.Box(
        widths=spec["widths"], center=spec["center"])
    if args.no_precompute_tables:
        table_emax = None
    else:
        table_emax = args.table_emax or ref._max_neutrino_energy(data)
        print("Precomputing DarkNews tables up to %.3f GeV" % table_emax)
    primary, secondaries = ref.build_vertices(
        detector_model, external, fiducial, args.m4, args.mu_tr_mu4,
        table_emax=table_emax)

    injector = Injector(
        detector=detector_model,
        primary=primary,
        secondaries=secondaries,
        events=args.events,
        seed=args.seed,
    )
    weighter = Weighter(injector, primary_physical=primary.physical)
    results = siren.generate(
        injector, weighter, events=args.events, on_shortfall="warn")
    results.summary()
    print(injector.report())

    columns = ref.collect_timing(
        results, detector_model, prompt_origin_geometry=prompt_origin)
    output = args.output or os.path.join(
        "output", "%s_bnb_darknews_hnl_timing.png" % DETECTOR.lower())
    output = ref.plot_timing(
        columns, output, DETECTOR, beam_frame=BEAM_FRAME)
    csv_path = os.path.splitext(output)[0] + ".csv"
    ref.save_timing_csv(columns, csv_path)
    print("Wrote %s" % output)
    print("Wrote %s" % csv_path)


if __name__ == "__main__":
    main()
