"""
Example: DIRT-INDUCED dipole-portal HNL (N4) in SBND.

Difference from sbnd.py (read that first)
-----------------------------------------
sbnd.py upscatters nu -> N4 INSIDE the SBND active volume and decays N4 there
too. This script moves the PRIMARY interaction OUTSIDE the detector:

  - nu -> N4 upscattering happens in the DIRT/ROCK upstream of SBND (glacial
    till / berm), NOT in the liquid argon.
  - The (long-lived) N4 then PROPAGATES from the dirt into the detector.
  - N4 -> nu + gamma DECAY happens INSIDE the SBND active volume.

This is the classic "dirt-induced" topology, and it only produces a visible
event when the HNL is long-lived enough to survive the flight from the dirt
into the LAr before decaying. The N4 survival x decay-in-detector probability
is folded into the event weight automatically (see section 5).

Only ONE thing changes mechanically vs sbnd.py: the PRIMARY vertex
distribution. Instead of

    PrimaryBoundedVertexDistribution(sbnd_active_fid_vol, 500.0)   # sbnd.py

we sample the upscattering vertex inside an upstream DIRT box:

    PrimaryBoundedVertexDistribution(dirt_box, 500.0)             # this file

The SECONDARY decay distribution is UNCHANGED --
SecondaryBoundedVertexDistribution(sbnd_active_fid_vol) already confines the
decay to SBND and, with its default max_length = infinity, will find the
detector no matter how far the N4 has to travel from the dirt (it shoots the
N4 direction, clips to the active-volume intersections, and samples the decay
point weighted by the HNL decay length -- i.e. the lifetime). A long-lived N4
gives a near-uniform decay distribution across the active volume.

Cross sections in the dirt: SIREN_Controller.InputDarkNewsModel builds
DarkNews upscattering tables for EVERY nuclear target in the detector model
(GetDetectorModelTargets over all sectors), so the Si/O/... nuclei of the
dirt already have cross sections -- no model change is needed. (The first run
that touches a new nucleus will build & cache its table.)

Timing chain (three legs now):
  primary.initial_time     = t0                                (nu production)
  primary.interaction_time = t0 + FlightTime(x0 -> upscatter)  (v=c; in DIRT)
  secondary.initial_time   = primary.interaction_time          (N4 born in dirt)
  secondary.interaction_time = upscatter_time
                             + FlightTime(upscatter -> decay)   (v_N4 = p/E c;
                                                                 decay IN SBND)

Regenerate the input CSV (neutrino production points/kinematics/t0) with:
  /home/shubham/siren_ubaid_venv/bin/python dk2nu_to_upscattering_csv.py \
      --dk2nu /home/shubham/nubeam12M.dk2nu.root \
      --n 100000 --out pion_derived_upscattering_events.csv
"""

import os

import numpy as np

import siren
from siren.SIREN_Controller import SIREN_Controller

# ---------------------------------------------------------------------
# 1. DarkNews dipole-portal model parameters
# ---------------------------------------------------------------------
# For a DIRT-induced signal the HNL must be LONG-LIVED: it has to survive the
# ~5-52 m flight from the upstream dirt into the LAr before decaying. The
# transition moment mu_tr_mu4 sets BOTH the production rate (~mu^2) and the
# decay rate (Gamma ~ mu^2 m4^3), so smaller mu -> longer-lived (more dirt N4
# reach the detector) but rarer production; smaller m4 also lengthens the
# lifetime (Gamma ~ m4^3).
#
# DEFAULT is a LONG-LIVED point (m4=0.15 GeV, mu=1e-7 GeV^-1): lab decay length
# of order a few hundred m, so a real (non-negligibly weighted) fraction of the
# dirt N4 reach SBND before decaying -- the physically appropriate regime for a
# dirt-induced search. NOTE: the SIREN-SBN benchmark 0.47/2.5e-6 used by sbnd.py
# has a ~mm lab decay length, so an N4 born in the dirt decays essentially at the
# upscatter point and its in-SBND weight is ~0 (fine for IN-detector upscattering,
# WRONG for dirt-induced). The dirt->detector survival x decay-in-SBND probability
# is captured in the event weight, so trust the acceptance/weights, not a guess.
# Override either parameter via env vars for a scan:
#     SBND_M4=0.10 SBND_MU=5e-8 python sbnd_dirt.py
model_kwargs = {
    "m4":        float(os.environ.get("SBND_M4", "0.15")),   # HNL mass [GeV]
    "mu_tr_mu4": float(os.environ.get("SBND_MU", "1e-7")),   # transition moment [GeV^-1]
    "UD4": 0,
    "Umu4": 0,
    "epsilon": 0.0,
    "gD": 0.0,
    "decay_product": "photon",   # or "e+e-"
    "noHC": True,
    "HNLtype": "dirac",
}

events_to_inject = int(os.environ.get("SBND_N_EVENTS", "100000"))

# Rough decay-length guide -- ORDER OF MAGNITUDE ONLY. The real per-event N4
# kinematics and the exact DarkNews width set the true rate; trust the acceptance
# printed at the end, NOT this number. Dipole transition-moment width
# Gamma(N->nu gamma) ~ mu^2 m4^3 / (4 pi) [GeV]; lab length L = gamma*beta*c*tau,
# with c*tau = hbar*c / Gamma and hbar*c = 1.9733e-16 GeV*m.
_m4, _mu = model_kwargs["m4"], model_kwargs["mu_tr_mu4"]
_Gamma = _mu ** 2 * _m4 ** 3 / (4.0 * np.pi)                 # GeV (order of magnitude)
_ctau = 1.9733e-16 / _Gamma if _Gamma > 0 else np.inf        # rest-frame ctau [m]
_E_ref = 1.0                                                  # GeV, representative N4 energy
_gammabeta = np.sqrt(max((_E_ref / _m4) ** 2 - 1.0, 0.0)) if _m4 > 0 else np.inf
print("HNL point: m4=%.3f GeV  mu=%.2e GeV^-1  ->  ctau~%.3g m,  "
      "L_lab~%.3g m @ E_N=%.1f GeV (rough)   [dirt standoff ~5-52 m]"
      % (_m4, _mu, _ctau, _ctau * _gammabeta, _E_ref))

# ---------------------------------------------------------------------
# 2. Load SBND geometry (shared SBN GDML composite; see sbnd.py notes)
# ---------------------------------------------------------------------
# earth_model=False keeps the local site geology (~500 m around the detectors),
# which is exactly the dirt/rock we now want to upscatter IN, so it must stay
# False (True/PREM is only needed for even farther upstream vertices).
detector_model = siren.utilities.load_detector(
    "SBN", detector="SBND", earth_model=False
)

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

# The external neutrino CSV (also loaded in section 5). Read here so we can size
# the DarkNews interpolation tables to the actual beam energy range up front.
external_csv = os.environ.get("SBND_EXTERNAL_CSV",
                              "pion_derived_upscattering_events.csv")

# Fill the DarkNews cross-section/decay interpolation tables BEFORE injection,
# up to the beam's maximum neutrino energy, instead of building them lazily
# inside the injection loop. This pays the (one-time, then disk-cached) table
# cost up front so a NEW parameter point doesn't stall mid-injection; later runs
# with the same m4/mu load the cached tables in seconds either way. Set
# SBND_FILL_TABLES=0 to revert to lazy in-loop building; SBND_TABLE_EMAX
# overrides the fill range [GeV] (default: max CSV neutrino energy + 5% headroom).
fill_tables = os.environ.get("SBND_FILL_TABLES", "1") != "0"
table_emax = None
if fill_tables:
    emax_env = os.environ.get("SBND_TABLE_EMAX")
    if emax_env is not None:
        table_emax = float(emax_env)
    else:
        # Peek at the CSV's neutrino energies (column 6, "E") for the beam max.
        _E = np.genfromtxt(external_csv, delimiter=",", names=True)["E"]
        table_emax = float(np.max(_E)) * 1.05
    print("Precomputing DarkNews tables up to Emax=%.3f GeV before injection"
          % table_emax)

# Restrict the DarkNews cross-section build to the DOMINANT dirt nuclei so the
# table build stays FAST. The full SBND detector model has ~50 nuclei, and
# building DarkNews tables for all of them at a new mass point is very slow (and
# can thrash memory). For a dirt-induced study the upscattering rate is carried
# by the abundant, higher-Z dirt/rock nuclei below; the long tail of rare
# isotopes contributes negligibly. Set SBND_NUCLEAR_TARGETS="all" to use every
# detector nucleus, or a comma-separated list to customize.
_DEFAULT_NUCLEI = ["O16", "Si28", "Al27", "Ca40", "Fe56", "Mg24", "Na23",
                   "K39", "C12", "Ar40"]
_nt_env = os.environ.get("SBND_NUCLEAR_TARGETS", "").strip()
if _nt_env.lower() == "all":
    nuclear_targets = None                      # controller then uses ALL nuclei
elif _nt_env:
    nuclear_targets = [s.strip() for s in _nt_env.split(",") if s.strip()]
else:
    nuclear_targets = _DEFAULT_NUCLEI
print("DarkNews nuclear targets:",
      nuclear_targets if nuclear_targets else "ALL detector nuclei (~50, slow)")

controller.InputDarkNewsModel(
    primary_type, table_dir,
    fill_tables_at_start=fill_tables, Emax=table_emax,
    nuclear_targets=nuclear_targets,
    **model_kwargs,
)

# ---------------------------------------------------------------------
# 4. Build the SBND active volume (used ONLY for the DECAY now)
# ---------------------------------------------------------------------
# Same construction as sbnd.py: the two TPC-active sectors, each rebuilt in the
# DETECTOR frame (sector.geo placements are in the GDML WORLD/BNB frame, but
# the vertex distributions interpret geometry in detector coordinates). See the
# long comment in sbnd.py for why the world->detector conversion is essential.
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
        f"missing {missing}. GDML volume naming may have changed."
    )


def _geo_in_detector_frame(detector_model, geo):
    """Rebuild a Box geometry with its placement converted world -> detector."""
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

sbnd_active_fid_vol = siren.geometry.BooleanGeometry(
    siren.geometry.BooleanOperation.UNION,
    active_geos[0],
    active_geos[1],
)

# ---------------------------------------------------------------------
# 4b. Build the UPSTREAM DIRT box (used for the PRIMARY upscattering)
# ---------------------------------------------------------------------
# This is the whole point of this script: the nu -> N4 upscattering vertex is
# sampled here, OUTSIDE the detector, in the dirt/rock upstream of SBND.
#
# Frame & placement (all DETECTOR coordinates, metres; SBND rotation is the
# identity so BNB and detector axes coincide):
#   * The BNB beam travels along +z; "upstream" is therefore -z.
#   * The SBND active volume upstream face sits at z ~ -2.92 m (measured from
#     the geometry above); the LAr center is the detector origin.
#   * The beam axis passes through detector-frame (x ~ -0.74, y ~ +0.59).
#   * A density scan along the upstream beam axis (see probe) shows solid dirt
#     (rho ~ 1.7-1.9 g/cm^3, glacial till / dolomite) over z in [-52, -8] m,
#     with the detector hall/berm just upstream of the LAr (z ~ -8 to -3) and
#     AIR beyond z ~ -60 m (the local geology model ends). So the dirt box must
#     live inside z in [-52, -8] to actually sit in rock.
#
# Defaults put the box solidly in that dirt window, centred transversely on the
# beam so every SBND-bound neutrino ray passes through it. All overridable via
# env vars for a standoff / thickness scan.
Z_NEAR = float(os.environ.get("DIRT_Z_NEAR", "-8.0"))    # det z of downstream (near) face [m]
Z_FAR  = float(os.environ.get("DIRT_Z_FAR",  "-52.0"))   # det z of upstream (far) face [m]
HALF_X = float(os.environ.get("DIRT_HALF_X", "6.0"))     # transverse half-width in x [m]
HALF_Y = float(os.environ.get("DIRT_HALF_Y", "6.0"))     # transverse half-width in y [m]
X_C    = float(os.environ.get("DIRT_X_C", "0.0"))        # transverse centre x [m]
Y_C    = float(os.environ.get("DIRT_Y_C", "0.5"))        # transverse centre y [m] (~beam axis)

z_center = 0.5 * (Z_NEAR + Z_FAR)
z_full   = abs(Z_NEAR - Z_FAR)
if not (Z_FAR < Z_NEAR <= -3.0):
    raise ValueError(
        f"DIRT box z-range [{Z_FAR}, {Z_NEAR}] must be upstream of the "
        f"detector (both < ~-3 m) with Z_FAR < Z_NEAR.")

dirt_box = siren.geometry.Box(
    siren.geometry.Placement(
        siren.detector.DetectorPosition(
            siren.math.Vector3D(X_C, Y_C, z_center)).get(),
        # axis-aligned box: identity orientation
        siren.math.Quaternion(0.0, 0.0, 0.0, 1.0),
    ),
    2.0 * HALF_X, 2.0 * HALF_Y, z_full,   # Box takes FULL side lengths
)
print("dirt box (detector frame): x in [%.1f, %.1f], y in [%.1f, %.1f], "
      "z in [%.1f, %.1f] m  (upstream of SBND active face at z~-2.9 m)"
      % (X_C - HALF_X, X_C + HALF_X, Y_C - HALF_Y, Y_C + HALF_Y, Z_FAR, Z_NEAR))

# ---------------------------------------------------------------------
# 5. Distributions
# ---------------------------------------------------------------------
# external_csv was defined in section 3 (so the table fill could read its energy
# range); reuse it here.
external_dist = siren.distributions.PrimaryExternalDistribution(external_csv)

# PRIMARY vertex: sample the upscattering point inside the upstream DIRT box,
# along the neutrino direction, weighted by cross section. 500 m ray length
# covers the neutrino flight from its far-upstream production point x0 (~110 m)
# to the dirt region. <<< THE ONE CHANGE vs sbnd.py (was sbnd_active_fid_vol) >>>
primary_vertex_dist = siren.distributions.PrimaryBoundedVertexDistribution(
    dirt_box, 500.0
)

primary_injection_distributions = {
    "external": external_dist,
    "position": primary_vertex_dist,
}
primary_physical_distributions = {
    "external": external_dist,
    "position": primary_vertex_dist,
}

# SECONDARY (decay) process: N4 -> nu + photon, decays INSIDE SBND.
# UNCHANGED from sbnd.py. Default max_length = infinity, so the N4 ray from the
# dirt upscatter vertex will always reach the SBND active volume; the decay
# point is then sampled inside it, weighted by the N4 decay length (lifetime).
secondary_types = [siren.dataclasses.Particle.ParticleType.N4]
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
    fid_vol_secondary=False,   # we supplied our own secondary vertex dist above
)

controller.Initialize()


def stop(tree, datum, i):
    secondary_type = datum.record.signature.secondary_types[i]
    return secondary_type != siren.dataclasses.Particle.ParticleType.N4


controller.SetInjectorStoppingCondition(stop)

events = controller.GenerateEvents(fill_tables_at_exit=False)

# ---------------------------------------------------------------------
# 6. Record the timing / propagation chain
# ---------------------------------------------------------------------
# ROOT datum (tree[0]) = nu -> N4 upscattering, now IN THE DIRT:
#   r.primary_initial_position = x0            (nu production, far upstream)
#   r.interaction_vertex       = upscatter pt  (IN DIRT, outside detector)
#   r.primary_initial_time     = t0
#   r.interaction_time         = t0 + nu ToF   (upscatter time)
# Secondary datum (N4 -> nu + photon) = the DECAY:
#   d.primary_initial_position = upscatter pt  (N4 birth, in dirt)
#   d.interaction_vertex       = decay pt
#   d.primary_momentum         = N4 4-momentum (for its velocity)
#   d.primary_initial_time     = upscatter time
#   d.interaction_time         = decay time = upscatter time + N4 ToF
#
# NOTE on acceptance: SecondaryBoundedVertexDistribution only *confines* the
# decay to SBND for N4s whose (scattered) direction actually crosses the active
# volume; an N4 born in the dirt whose direction MISSES SBND has no fiducial
# intersection and simply decays along its path in the dirt. Those are real MC
# events but NOT the dirt-induced-signal topology, so we tag each event with
# `in_detector` (decay vertex inside a TPC) and report the acceptance. The
# in_detector==1 subset is the signal (upscatter in dirt -> N4 flight -> decay
# in SBND); cut on it (with the event weights) for any rate.

# Real 3D containment against the two TPC boxes (detector frame, axis-aligned).
def _box_bounds(box):
    c = box.placement.Position
    return (c.GetX() - 0.5 * box.X, c.GetX() + 0.5 * box.X,
            c.GetY() - 0.5 * box.Y, c.GetY() + 0.5 * box.Y,
            c.GetZ() - 0.5 * box.Z, c.GetZ() + 0.5 * box.Z)


_tpc_bounds = [_box_bounds(g) for g in active_geos]


def _in_detector(p):
    for (x0, x1, y0, y1, z0, z1) in _tpc_bounds:
        if x0 <= p[0] <= x1 and y0 <= p[1] <= y1 and z0 <= p[2] <= z1:
            return True
    return False


N4 = int(siren.dataclasses.Particle.ParticleType.N4)
C = siren.utilities.Constants.c   # m/ns
rows = []
n_empty = 0
n_no_decay = 0
for ev in events:
    if len(ev.tree) == 0:
        n_empty += 1
        continue
    root = ev.tree[0]
    r = root.record
    E = float(r.primary_momentum[0])                     # nu energy [GeV]
    x0 = np.array(r.primary_initial_position, float)      # nu production [m]
    xup = np.array(r.interaction_vertex, float)           # upscatter vertex (dirt) [m]
    production_time = float(r.primary_initial_time)        # t0
    upscatter_time = float(r.interaction_time)            # t0 + nu ToF
    nu_dist = float(np.linalg.norm(xup - x0))

    # N4 leg: born at xup (dirt), decays at xdec (SBND)
    decay_time = np.nan
    xdec = np.array([np.nan, np.nan, np.nan])
    n4_flight = np.nan
    n4_beta = np.nan
    for datum in ev.tree[1:]:
        d = datum.record
        if int(d.signature.primary_type) != N4:
            continue
        decay_time = float(d.interaction_time)
        xdec = np.array(d.interaction_vertex, float)
        n4_flight = float(np.linalg.norm(xdec - xup))     # dirt -> decay [m]
        p4 = np.array(d.primary_momentum, float)          # (E, px, py, pz) GeV
        EN, pN = p4[0], np.linalg.norm(p4[1:])
        n4_beta = pN / EN if EN > 0 else np.nan           # v_N4 / c
        break
    if not np.isfinite(decay_time):
        n_no_decay += 1
        continue

    in_det = 1.0 if _in_detector(xdec) else 0.0
    rows.append((production_time, upscatter_time, upscatter_time - production_time,
                 decay_time, decay_time - upscatter_time, E, nu_dist, n4_flight,
                 n4_beta, xup[0], xup[1], xup[2], xdec[0], xdec[1], xdec[2], in_det))

cols = ["production_time", "upscatter_time", "nu_tof", "decay_time", "n4_tof",
        "nu_energy", "nu_flight_dist", "n4_flight_dist", "n4_beta",
        "ux", "uy", "uz", "dx", "dy", "dz", "in_detector"]
if not rows:
    raise SystemExit(
        "No dirt-induced events (%d empty trees, %d without an N4 decay). "
        "If ALL trees are empty the neutrino rays may miss the dirt box, or the "
        "upscattering cross section in the dirt nuclei may be negligible for "
        "this mass; if trees exist but none decay in SBND the HNL may be too "
        "short-lived to reach it (lower mu / mass for a longer lifetime)."
        % (n_empty, n_no_decay))
timing = np.array(rows, float)

# ---- consistency checks -------------------------------------------------
# leg 1 (neutrino, v=c): upscatter_time - t0 == nu_dist / c
tof_nu = timing[:, 1] - timing[:, 0]
rel_nu = np.abs(tof_nu - timing[:, 6] / C) / np.maximum(timing[:, 6] / C, 1e-12)
# leg 2 (N4, v=beta c): decay_time - upscatter_time == n4_dist / (beta c)
vN4 = timing[:, 8] * C
tof_n4 = timing[:, 3] - timing[:, 1]
rel_n4 = np.abs(tof_n4 - timing[:, 7] / np.maximum(vN4, 1e-12)) / np.maximum(tof_n4, 1e-12)
# every upscatter must be OUTSIDE the detector; signal = decay INSIDE a TPC.
in_det = timing[:, 15] > 0.5
n_up_outside = int(np.sum(~np.array([_in_detector(p) for p in timing[:, 9:12]])))
n_dec_inside = int(np.sum(in_det))

print("recorded %d dirt upscattering events (%d empty trees, %d no-decay skipped)"
      % (len(timing), n_empty, n_no_decay))
print("  --- ALL events ---")
print("  <t0>=%.1f ns  <nu ToF>=%.1f ns  <upscatter t>=%.1f ns  <N4 ToF>=%.1f ns  <decay t>=%.1f ns"
      % (timing[:, 0].mean(), tof_nu.mean(), timing[:, 1].mean(),
         tof_n4.mean(), timing[:, 3].mean()))
print("  upscatter_time == t0 + nu_dist/c        : max rel resid %.2e -> %s"
      % (rel_nu.max(), "PASS" if rel_nu.max() < 1e-6 else "FAIL"))
print("  decay_time     == upscatter + n4/(beta c): max rel resid %.2e -> %s"
      % (rel_n4.max(), "PASS" if rel_n4.max() < 1e-4 else "FAIL"))
print("  upscatter OUTSIDE detector: %d/%d" % (n_up_outside, len(timing)))
print("  --- SIGNAL subset: decay INSIDE SBND active volume (in_detector==1) ---")
print("  acceptance (N4 aimed at & decaying in SBND): %d/%d = %.1f%%"
      % (n_dec_inside, len(timing), 100.0 * n_dec_inside / len(timing)))
if n_dec_inside:
    print("  <N4 flight dirt->decay>=%.2f m  <beta_N4>=%.3f  <decay t>=%.1f ns"
          % (timing[in_det, 7].mean(), timing[in_det, 8].mean(), timing[in_det, 3].mean()))

os.makedirs("output", exist_ok=True)
np.savetxt("output/SBND_dirt_timing.csv", timing, delimiter=",",
           header=",".join(cols), comments="", fmt="%.8g")
print("  wrote per-event timing -> output/SBND_dirt_timing.csv")
try:
    import pyarrow as pa, pyarrow.parquet as pq
    pq.write_table(pa.table({c: timing[:, i] for i, c in enumerate(cols)}),
                   "output/SBND_dirt_timing.parquet")
    print("  wrote per-event timing -> output/SBND_dirt_timing.parquet")
except ImportError:
    print("  (pyarrow not available; skipped parquet)")

# Drop failed-injection (empty) trees before any native writer (segfaults on
# empty trees; native SaveEvents also crashes in the serializer on this build,
# so it stays OFF unless SBND_SAVE_EVENTS=1 -- see sbnd.py).
controller.events = [ev for ev in controller.events if len(ev.tree) > 0]
if os.environ.get("SBND_SAVE_EVENTS", "0") != "0":
    controller.SaveEvents(
        "output/SBND_dirt_Dipole_M%2.2e_mu%2.2e_external_time"
        % (model_kwargs["m4"], model_kwargs["mu_tr_mu4"]),
        fill_tables_at_exit=False,
    )
