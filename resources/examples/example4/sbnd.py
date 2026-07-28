"""
Example: DarkNews dipole-portal HNL (N4) production in SBND, using
PrimaryExternalDistribution to load pre-computed upscattering kinematics
(including an initial-time column, t0) instead of sampling position/energy
from a flux + range distribution.

Physics picture (see dk2nu_to_upscattering_csv.py):
  - Each CSV row is one neutrino, produced at its meson decay vertex
    (x0, y0, z0) at lab time t0 (the beam-spill production time), with momentum
    (px, py, pz), energy E, mass m=0. It flies to SBND and undergoes
    dipole-portal upscattering (nu -> N4) INSIDE the detector.
  - PrimaryExternalDistribution injects x0, momentum, E, t0; a paired
    PrimaryBoundedVertexDistribution samples the upscattering VERTEX inside the
    SBND active volume, along the neutrino direction, weighted by cross-section
    (PrimaryExternalDistribution is not itself a vertex distribution, which the
    Injector requires -- see section 4). SIREN then sets:
        primary.initial_time     = t0                        (production time)
        primary.interaction_time = t0 + FlightTime(x0 -> vertex)
                                 = upscattering time IN SBND  (neutrino ToF)
    v = c for the massless neutrino. You do NOT compute this by hand.
  - N4 then propagates and decays (N4 -> nu + gamma, or nu + e+e-) at some
    later point, which the injector samples inside SBND. SIREN's
    InteractionRecord machinery automatically sets
        secondary.initial_time  = primary interaction_time   (== t0)
        secondary.interaction_time = secondary.initial_time + FlightTime(...)
    i.e. decay time = t0 + time-of-flight of the produced N4, computed
    automatically from its sampled momentum/mass and the sampled decay
    length. 


"""

import os

import numpy as np

import siren
from siren.SIREN_Controller import SIREN_Controller

# ---------------------------------------------------------------------
# 1. DarkNews dipole-portal model parameters
# ---------------------------------------------------------------------
model_kwargs = {
    # SIREN-SBN benchmark point (matches example2/DipolePortal_MiniBooNE.py
    # and the SIREN-SBN reference figure sbnd_hnl_timing.png). The previous 0.140 GeV / 1e-6 came
    # from the commented-out alternative in that example, not a deliberate
    # choice; its DarkNews tables remain cached under Dipole_M1.40e-01_mu1.00e-06.
    "m4": 0.47,              # HNL mass [GeV]
    "mu_tr_mu4": 2.5e-6,     # transition magnetic moment [GeV^-1]
    "UD4": 0,
    "Umu4": 0,
    "epsilon": 0.0,
    "gD": 0.0,
    "decay_product": "photon",   # or "e+e-"
    "noHC": True,
    "HNLtype": "dirac",
}

# Overridable for quick end-to-end proofs; the DarkNews table-build cost is
# independent of this (tables cache to disk on first use), so a small value
# exercises the full chain fast once tables exist.
events_to_inject = int(os.environ.get("SBND_N_EVENTS", "100000"))

# ---------------------------------------------------------------------
# 2. Load SBND geometry
# ---------------------------------------------------------------------
# NOTE: SBND is NOT loaded via experiment="SBND". It lives in the shared
# "SBN" GDML-composite loader (ICARUS / SBND / MicroBooNE + BNB/NuMI
# beamlines all in one frame). See resources/detectors/SBN/SBN-v1/detector.py.
#
#   earth_model=False -> local site geology only (~500 m around detectors).
#                         Fine if the upscattering vertex in your CSV is
#                         within that range of SBND (true for BNB dirt/
#                         decay-pipe upscattering).
#   earth_model=True  -> full PREM Earth model; only needed if your
#                         upscattering vertices are farther away.
detector_model = siren.utilities.load_detector(
    "SBN", detector="SBND", earth_model=False
)

# The controller must be constructed with detector_model= (not
# experiment="SBND"), since there is no legacy materials.dat/densities.dat
# for the GDML composite geometry.
controller = SIREN_Controller(
    events_to_inject, detector_model=detector_model
)

# ---------------------------------------------------------------------
# 3. DarkNews cross section / decay tables
# ---------------------------------------------------------------------
primary_type = siren.dataclasses.Particle.ParticleType.NuMu

xs_path = siren.utilities.get_processes_model_path(
    f"DarkNewsTables-v{siren.utilities.darknews_version()}", must_exist=False
)
table_dir = os.path.join(
    xs_path,
    "Dipole_M%2.2e_mu%2.2e" % (model_kwargs["m4"], model_kwargs["mu_tr_mu4"]),
)
# upscattering=True builds the primary (nu -> N4) process,
# decay=True builds the secondary (N4 -> nu + photon) process.
controller.InputDarkNewsModel(primary_type, table_dir, **model_kwargs)

# ---------------------------------------------------------------------
# 4. Primary (upscattering) distributions
# ---------------------------------------------------------------------
# The primary process needs TWO cooperating distributions:
#
#   (a) PrimaryExternalDistribution -- supplies, per neutrino (from dk2nu via
#       dk2nu_to_upscattering_csv.py): the PRODUCTION point x0,y0,z0, the
#       momentum px,py,pz, energy E, mass m (=0), and production time t0.
#       CSV columns: x0,y0,z0,px,py,pz,E,m,t0  (NOTE: no x,y,z -- see (b)).
#
#   (b) PrimaryBoundedVertexDistribution -- the actual VERTEX distribution.
#       PrimaryExternalDistribution is NOT a VertexPositionDistribution, and
#       the Injector requires one (FindPrimaryVertexDistribution -> exit(0)
#       otherwise). This one shoots a ray from x0 along the neutrino direction
#       and samples the upscattering vertex INSIDE the SBND active volume,
#       weighted by cross-section. It is added below (section 5) once the
#       active-volume geometry is built.
#
# Order matters: (a) must run before (b) so the vertex sampler can read the
# initial position and direction. The controller assembles the list as
#   [PrimaryMass(0), Helicity, external, position]
# from the insertion order of this dict, so list "external" before "position".
#
# SIREN then sets, on the finalized record:
#   primary.initial_time     = t0                       (production time)
#   primary.interaction_time = t0 + FlightTime(x0 -> vertex)
#                            = the time the upscattering happens IN SBND
# (v = c because m = 0).
#
# Regenerate the CSV with:
#   /home/shubham/siren_ubaid_venv/bin/python dk2nu_to_upscattering_csv.py \
#       --n 100000 --out pion_derived_upscattering_events.csv
external_csv = "pion_derived_upscattering_events.csv"  # from dk2nu_to_upscattering_csv.py

external_dist = siren.distributions.PrimaryExternalDistribution(external_csv)
# primary_injection_distributions is completed in section 5, after the SBND
# active-volume geometry (needed by the vertex distribution) is constructed.

# ---------------------------------------------------------------------
# 5. Secondary (decay) process: N4 -> nu + photon, decays inside SBND
# ---------------------------------------------------------------------
secondary_types = [siren.dataclasses.Particle.ParticleType.N4]

# fid_vol_secondary=True is the *default* SetProcesses() behavior, but for a
# GDML composite detector model (as opposed to a legacy experiment="X"
# string load), SIREN_Controller.GetFiducialVolume() returns None (there is
# no densities.dat "fiducial" line to parse). That means the automatic
# "confine decay to fiducial volume" behavior silently does nothing for
# SBND, and decays would be sampled via SecondaryPhysicalVertexDistribution
# instead (NOT guaranteed to land inside SBND). So we build and pass the
# fiducial geometry explicitly instead of relying on fid_vol_secondary.
#
# Confirmed by inspecting controller.detector_model.Sectors on a real SBND
# load: SBND has exactly two TPC-active sectors (one per drift volume,
# left/right of the central cathode), named:
active_sector_names = ("volTPCActive", "volTPCActive_2")

active_geos = [
    sector.geo
    for sector in controller.detector_model.Sectors
    if sector.name in active_sector_names
]
if len(active_geos) != len(active_sector_names):
    found = {s.name for s in controller.detector_model.Sectors}
    missing = set(active_sector_names) - found
    raise RuntimeError(
        f"Expected SBND active-volume sectors {active_sector_names}, "
        f"missing {missing}. GDML volume naming may have changed -- "
        f"re-check with: for s in controller.detector_model.Sectors: print(s.name)"
    )

# COORDINATE FRAME: sector.geo placements are in the GDML WORLD frame (= BNB
# beam frame; volTPCActive sits at z ~ 112.5 m there), but the bounded vertex
# distributions interpret their geometry in DETECTOR coordinates (the record
# positions the Injector consumes are DetectorPosition; SBND's detector origin
# is the LAr center at BNB (0.7378, -0.59, 112.92)). Using the raw sector.geo
# made the sampler target a phantom TPC ~112 m BEHIND the real detector, so
# every "in-detector" vertex actually landed in glacial till -- the low-SBND-
# rate bug. Rebuild each box with its placement converted world -> detector
# via the detector model itself (SBND's DetectorRotation is the identity, so
# translating the placement position is exact).
def _geo_in_detector_frame(detector_model, geo):
    p = geo.placement.Position
    dp = detector_model.GeoPositionToDetPosition(
        siren.detector.GeometryPosition(p)).get()
    return siren.geometry.Box(
        siren.geometry.Placement(dp, geo.placement.Quaternion),
        geo.X, geo.Y, geo.Z,
    )

active_geos = [
    _geo_in_detector_frame(controller.detector_model, g) for g in active_geos
]

# Union the two TPC-active volumes into a single fiducial geometry so both the
# primary upscattering vertex AND the N4 decay are confined to a drift volume
# (i.e. "inside SBND").
sbnd_active_fid_vol = siren.geometry.BooleanGeometry(
    siren.geometry.BooleanOperation.UNION,
    active_geos[0],
    active_geos[1],
)

# Primary vertex distribution (see section 4b). max_length is the ray length
# searched from the neutrino production point x0 for the active volume; the
# BNB baseline is ~110 m and production is upstream of that, so 500 m safely
# reaches and passes through the detector for every neutrino.
primary_vertex_dist = siren.distributions.PrimaryBoundedVertexDistribution(
    sbnd_active_fid_vol, 500.0
)
# Insertion order = application order: external (sets x0, direction, E, t0)
# BEFORE the vertex sampler (reads them to place the in-detector vertex).
primary_injection_distributions = {
    "external": external_dist,
    "position": primary_vertex_dist,
}
primary_physical_distributions = {
    "external": external_dist,
    "position": primary_vertex_dist,
}

secondary_injection_distributions = [[
    siren.distributions.SecondaryBoundedVertexDistribution(sbnd_active_fid_vol)
]]
secondary_physical_distributions = [[]]

controller.SetProcesses(
    primary_type,
    primary_injection_distributions,
    primary_physical_distributions,
    secondary_types,
    secondary_injection_distributions,
    secondary_physical_distributions,
    # We already supplied our own SecondaryBoundedVertexDistribution above.
    # If fid_vol_secondary stayed True, SetInjectionProcesses() would ALSO
    # append its own fallback distribution (SecondaryPhysicalVertexDistribution,
    # since self.fid_vol is None for a GDML composite detector model),
    # giving the secondary process two position distributions at once --
    # so we turn the automatic behavior off here.
    fid_vol_secondary=False,
)

controller.Initialize()


def stop(tree, datum, i):
    secondary_type = datum.record.signature.secondary_types[i]
    return secondary_type != siren.dataclasses.Particle.ParticleType.N4


controller.SetInjectorStoppingCondition(stop)

events = controller.GenerateEvents(fill_tables_at_exit=False)

# ---------------------------------------------------------------------
# 6. Record the timing chain: production time t0 and in-detector
#    upscattering time (= t0 + neutrino time-of-flight)
# ---------------------------------------------------------------------
# For every event we walk the interaction tree. The ROOT datum (tree[0]) is
# the primary nu -> N4 upscattering; on the finalized InteractionRecord:
#   record.primary_initial_time = t0   (neutrino PRODUCTION time; from the CSV)
#   record.interaction_time     = t0 + FlightTime(x0 -> x)
#                               = the time the upscattering happens IN SBND.
# (Note: the finalized record exposes `primary_initial_time`, NOT the
# `initial_time` name used on the intermediate PrimaryDistributionRecord.)
# Any further datum (tree[1:]) is the N4 -> nu + photon decay; its
# interaction_time = upscattering time + N4 time-of-flight, recorded too.
N4 = int(siren.dataclasses.Particle.ParticleType.N4)
C = siren.utilities.Constants.c   # m/ns
rows = []
n_empty = 0
for ev in events:
    # The injector returns an EMPTY tree when a primary injection fails
    # (e.g. the bounded vertex sampler finds no in-volume path, or a zero
    # total cross section along it), so skip those before indexing tree[0].
    if len(ev.tree) == 0:
        n_empty += 1
        continue
    root = ev.tree[0]
    r = root.record
    E = float(r.primary_momentum[0])                 # neutrino energy [GeV]
    x0 = np.array(r.primary_initial_position, float)  # production point [m]
    xup = np.array(r.interaction_vertex, float)       # upscattering vertex [m]
    production_time = float(r.primary_initial_time)   # t0
    upscatter_time = float(r.interaction_time)        # t0 + neutrino ToF
    dist = float(np.linalg.norm(xup - x0))
    # decay time of the N4 (first secondary datum, if the decay was sampled)
    decay_time = np.nan
    for datum in ev.tree[1:]:
        if int(datum.record.signature.primary_type) == N4:
            decay_time = float(datum.record.interaction_time)
            break
    rows.append((production_time, upscatter_time, upscatter_time - production_time,
                 decay_time, E, dist, xup[0], xup[1], xup[2]))

cols = ["production_time", "upscatter_time", "nu_tof", "decay_time",
        "nu_energy", "nu_flight_dist", "vx", "vy", "vz"]
if not rows:
    raise SystemExit("No events with a primary interaction (all %d trees empty). "
                     "Check the vertex distribution / cross sections." % n_empty)
timing = np.array(rows, float)

# quick consistency check: upscatter_time - t0 == neutrino ToF == dist/c (v=c)
tof = timing[:, 1] - timing[:, 0]
rel = np.abs(tof - timing[:, 5] / C) / np.maximum(timing[:, 5] / C, 1e-12)
print("recorded %d events (%d empty trees skipped)" % (len(timing), n_empty))
print("  <production t0>=%.1f ns  <nu ToF>=%.1f ns  <upscatter time>=%.1f ns"
      % (timing[:, 0].mean(), tof.mean(), timing[:, 1].mean()))
print("  upscatter_time == t0 + dist/c : max rel resid %.2e -> %s"
      % (rel.max(), "PASS" if rel.max() < 1e-6 else "FAIL"))

os.makedirs("output", exist_ok=True)
np.savetxt("output/SBND_timing.csv", timing, delimiter=",",
           header=",".join(cols), comments="", fmt="%.8g")
print("  wrote per-event timing -> output/SBND_timing.csv")
try:
    import pyarrow as pa, pyarrow.parquet as pq
    pq.write_table(pa.table({c: timing[:, i] for i, c in enumerate(cols)}),
                   "output/SBND_timing.parquet")
    print("  wrote per-event timing -> output/SBND_timing.parquet")
except ImportError:
    print("  (pyarrow not available; skipped parquet)")

# Drop failed-injection (empty) trees before the native writer: SaveEvents /
# SaveInteractionTrees walks each tree's records and segfaults on empty ones.
controller.events = [ev for ev in controller.events if len(ev.tree) > 0]

# Native SIREN event output is OFF by default: SaveInteractionTrees segfaults
# inside the C++ serializer on this build (a separate, pre-existing bug -- it
# crashes even after empty trees are filtered), and it is not needed for the
# timing analysis above (output/SBND_timing.*). Set SBND_SAVE_EVENTS=1 to try it.
if os.environ.get("SBND_SAVE_EVENTS", "0") != "0":
    controller.SaveEvents(
        "output/SBND_Dipole_M%2.2e_mu%2.2e_external_time"
        % (model_kwargs["m4"], model_kwargs["mu_tr_mu4"]),
        fill_tables_at_exit=False,
    )
