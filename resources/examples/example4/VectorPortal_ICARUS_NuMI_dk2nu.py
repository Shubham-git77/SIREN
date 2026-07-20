"""
VectorPortal_ICARUS_NuMI_dk2nu.py — Dutta-Kim vector-portal K+ chain at
ICARUS driven by the REAL NuMI beam simulation (g4numi dk2nu).

This is the NuMI counterpart of VectorPortal_SBND_fullchain.py (whose
channel machinery it reuses): kaon kinematics are read from g4numi dk2nu
files, whose vertices/momenta are expressed in NuMI beam coordinates
(origin at MCZERO, z along the NuMI axis).  They are transformed into the
SIREN world frame (BNB) with sbn_geometry.transform("NuMI", "BNB") — the
ICARUS-NuMI rotation/translation from icaruscode GNuMIFlux.xml, confirmed
by SBN DocDB 22998-v2 — via the Dk2nuReader beam_transform hook, and
injected through the composite SBN geometry (BNB + NuMI beamlines +
ICARUS detector, all in BNB coordinates).

Flux files (Austin Schneider, "SBN Global Fits" Drive folder / g4numi):
  g4numiNone_downstream_me000z-200i_rhc_2001-2005.root
  medium-energy target, -200 kA horns = RHC (antineutrino mode),
  1e6 POT per file, ~15k K+ decays per file.

Usage:
  python VectorPortal_ICARUS_NuMI_dk2nu.py --n-events 2000
  python VectorPortal_ICARUS_NuMI_dk2nu.py --channel K_mu --n-events 10000
"""

import argparse
import glob
import os

import numpy as np

import siren
from siren import _util

_EX4 = os.path.dirname(os.path.realpath(__file__))

# Reuse the validated SBND full-chain machinery (models, phase spaces,
# injector loop, cuts); everything detector/beam-specific is overridden
# below through module attributes.
fc = _util.load_module(
    "VP_SBND_fullchain", os.path.join(_EX4, "VectorPortal_SBND_fullchain.py"))
_DK = fc._DK

geo = _util.load_module(
    "sbn_geometry",
    os.path.join(_util.resource_package_dir(),
                 "detectors", "SBN", "SBN-v1", "sbn_geometry.py"))

# ------------------------------------------------------------------ #
#  NuMI flux files                                                     #
# ------------------------------------------------------------------ #
NUMI_GLOB = os.environ.get(
    "NUMI_DK2NU_GLOB", os.path.join(_EX4, "sources", "NuMI", "g4numi*.root"))
NUMI_FILES = sorted(glob.glob(NUMI_GLOB))

# ICARUS NuMI exposure. PLACEHOLDER at the SBN-proposal scale — set
# ICARUS_NUMI_POT in the environment for a real projection.
ICARUS_NUMI_POT = float(os.environ.get("ICARUS_NUMI_POT", 3.0e21))

# ------------------------------------------------------------------ #
#  ICARUS overrides of the SBND-shaped module constants                #
# ------------------------------------------------------------------ #
# Active LAr envelope from sbn_geometry.DETECTORS["ICARUS"]:
#   x: +/-3.60 m,  y: -0.202 +/- 1.58 m,  z: +/-8.975 m
# (detector frame is centered on the LAr geometric center, so the box is
# origin-centered in detector coordinates).
# NOTE on geometry: this box is only the DetectorDirected SAMPLING ENVELOPE
# (it directs the mediator/chi toward ICARUS and bounds the vertex sampler).
# The real two-cryostat geometry is enforced downstream by the GDML
# volTPCActive sector cut in fc.upscatter_in_lar/signal_eepair_observables,
# which operates on the composite ICARUS GDML (both C0 and C1 drift volumes).
# So a single envelope spanning both cryostats + the 1.2 m argon-free gap is
# fine here -- gap interactions are rejected by the sector cut. (The analytic
# engine is different: there the box IS the target, so it uses the true
# single-module box summed over ICARUS_MODULE_CENTERS_BNB -- see
# analytic_NuMI_ICARUS.py.)
fc._TPC_BOX_X = 7.20     # sampling-envelope width  [m] (spans both cryostats)
fc._TPC_BOX_Y = 3.16     # sampling-envelope height [m]
fc._TPC_BOX_Z = 17.95    # sampling-envelope length [m]
# Sphere enclosing the farthest active-volume corner (~9.80 m).
fc.R_LAR_INJECT = 10.5
fc.SBND_POT = ICARUS_NUMI_POT   # folded into w_abs by run_channel

# NuMI beam frame -> BNB (SIREN world) rigid transform.
T_NUMI_TO_BNB = geo.transform("NuMI", "BNB")


def load_numi_mesons(dk2nu_data, parent_pdg, detector_model, upscatter=None):
    """NuMI version of fc.load_dk2nu_mesons: identical sampling bias
    (sigma(E) * max(cos,0), evaluated in the raw NuMI beam frame where
    "forward" means along the NuMI axis), plus the NuMI->BNB transform."""
    sigma_fn = (fc._build_sigma_interp(upscatter)
                if upscatter is not None else None)
    return _DK.dk2nu_to_primary_distribution(
        dk2nu_data, detector_model, parent_pdg=parent_pdg,
        sampling_bias=fc.make_meson_bias(sigma_fn),
        flux_weighted_sampling=True,
        beam_transform=T_NUMI_TO_BNB)


# run_channel resolves load_dk2nu_mesons from its module globals.
fc.load_dk2nu_mesons = load_numi_mesons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", choices=list(fc.CHANNELS) + ["all"],
                    default="K_e")
    ap.add_argument("--n-events", type=int, default=2000)
    ap.add_argument("--n-files", type=int, default=1,
                    help="how many g4numi files to read")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if not NUMI_FILES:
        raise SystemExit("No NuMI dk2nu files match %s" % NUMI_GLOB)
    files = NUMI_FILES[:args.n_files]

    print("NuMI -> BNB transform (GNuMIFlux.xml / DocDB 22998-v2):")
    print("  MCZERO in BNB frame: %s m" % np.round(T_NUMI_TO_BNB.t, 3))
    print("  ICARUS in NuMI frame: %s m  (baseline %.1f m, off-axis %.2f deg)"
          % (np.round(geo.detector_center("ICARUS", "NuMI"), 2),
             np.linalg.norm(geo.detector_center("ICARUS", "NuMI")),
             np.degrees(np.arccos(
                 geo.detector_center("ICARUS", "NuMI")[2]
                 / np.linalg.norm(geo.detector_center("ICARUS", "NuMI"))))))

    print("\nLoading composite ICARUS model (BNB + NuMI beamlines + detector) ...")
    detector_model = siren.utilities.load_detector("SBN", detector="ICARUS")

    print("Reading NuMI dk2nu: %d file(s)" % len(files))
    dk2nu_data = _DK.read_dk2nu(files)
    try:
        _DK.print_summary(dk2nu_data)
    except Exception:
        pass

    names = list(fc.CHANNELS) if args.channel == "all" else [args.channel]
    per_channel = {}
    for name in names:
        per_channel[name] = fc.run_channel(
            name, dk2nu_data, detector_model, args.n_events, debug=args.debug)

    print("\n" + "=" * 64)
    print("  ICARUS x NuMI (RHC)  —  Vector Portal (full chain)")
    print("  POT: %.2e   files: %d" % (ICARUS_NUMI_POT, len(files)))
    print("=" * 64)
    for n in per_channel:
        Ev, cs, wv = per_channel[n]
        print("  %-6s : %4d events   sum(w)=%.3e"
              % (n, len(Ev), wv.sum() if len(wv) else 0.0))

    os.makedirs(os.path.join(_EX4, "output"), exist_ok=True)
    stem = os.path.join(_EX4, "output", "ICARUS_VectorPortal_NuMI_dk2nu")
    np.savez(stem + "_observables.npz",
             **{f"{n}_E": per_channel[n][0] for n in per_channel},
             **{f"{n}_c": per_channel[n][1] for n in per_channel},
             **{f"{n}_w": per_channel[n][2] for n in per_channel})
    print("  Saved -> %s_observables.npz" % stem)


if __name__ == "__main__":
    main()
