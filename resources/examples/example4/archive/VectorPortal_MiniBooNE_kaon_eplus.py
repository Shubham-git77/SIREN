"""
Vector-portal dark matter at MiniBooNE  —  K+ -> e+ nu_e V1 channel.

Chain:
    K+ -> e+ nu_e V1        (three-body production)
    V1 -> chi chi           (dark photon decay, folded into chi_flux)
    chi + N -> chi' + N      (upscattering in mineral oil: C12 + H1)
    chi' -> chi V1           (de-excitation)
    V1 -> e+ e-              (visible signal)

Reference: Dutta et al., PRL 129, 111803 (2022) [arXiv:2110.11944].

FIRST-RUN VERSION (per plan):
  - Tabulated PionKaon kaon flux (not dk2nu).
  - Single g_D through the all-in-one factory (production and upscattering
    share one coupling). This is NOT the split-coupling benchmark; the
    absolute rate will be refined later by (a) splitting G1 (production,
    BR-fixed) from G2P (upscattering) and (b) using dk2nu flux directly.
  - The factory's compute_chi_flux already includes the BR(V1->2chi)=0.5
    factor (hardcoded *0.5), and auto-selects targets (C12, H1, ...) from
    the loaded MiniBooNE GDML, so no manual nuclear target is needed.
"""

import os
import math
import numpy as np

import siren
from siren import utilities
from siren import _util as _siren_util

# ---------------------------------------------------------------------------
# Load the vector-portal factory from the branch's processes.py
# ---------------------------------------------------------------------------
_dt_base = os.path.join(
    _siren_util.resource_package_dir(), "processes", "DarkNewsTables",
)
_mod_processes = _siren_util.load_module(
    "siren.resources.processes.DarkNewsTables.processes",
    os.path.join(_dt_base, "processes.py"),
)
load_vector_portal = _mod_processes.load_vector_portal

# ---------------------------------------------------------------------------
# Model parameters (Dutta et al. Table I, double-mediator benchmark)
# ---------------------------------------------------------------------------
M_CHI       = 8e-3     # GeV  chi   (DM ground state)
M_CHI_PRIME = 50e-3    # GeV  chi'  (DM excited state)
M_V1        = 17e-3    # GeV  V1    (light dark photon, visible)
M_V2        = 200e-3   # GeV  V2    (heavy upscattering mediator)
EPSILON_1   = 7e-5     # kinetic mixing for V1
EPSILON_2   = 1e-4     # kinetic mixing for V2

# FIRST-RUN single coupling. The example uses g_D = 1.0; we keep that here
# to reproduce the example's behaviour. NOTE: physically the V1 production
# coupling is the BR-fixed G1 ~ 1.08e-4, NOT 1.0 — this is the main thing to
# correct in the split-coupling refinement. Documented, not silently wrong.
G_D         = 1.0

# Production channel: K+ -> e+ nu_e V1
M_MESON  = 0.49368     # GeV  K+
M_LEPTON = 0.000511    # GeV  e+   (electron mass -> nu_e channel)

# FLUX TAG: the PionKaon flux loader keys on NEUTRINO FLAVOR, not the parent
# meson. Valid particles are ['numu','numubar','nue','nuebar'], tag form is
# "<mode>_<particle>". For K+ -> e+ nu_e the associated neutrino is nu_e, so:
#   FHC_nue   (confirmed to resolve to a TabulatedFluxDistribution)
# The parent-meson contributions (_pion/_kaon/_all .dat files) are resolved
# internally; the channel (kaon vs pion) is fixed by m_meson/m_lepton below.
FLUX_TAG = "FHC_nue"

PDGID_CHI       = 5917
PDGID_CHI_PRIME = 5918
PDGID_V1        = 5922    # this factory (load_vector_portal) uses ONE V1 code
# NOTE: plain load_vector_portal uses pdgid_V1=5922 for BOTH production and
# signal V1 (no separate 5923). The signal vertex is therefore a 5922 vertex
# whose secondaries are e+e-. We identify the signal by the e+e- secondaries
# directly, which is robust to the shared PDG code.

events_to_inject = 10_000
experiment       = "MiniBooNE"
MAX_ENERGY       = 3.0     # GeV  upper bound for chi flux table
R_FID            = 5.0     # m    spherical fiducial radius (inside 6.1 m tank)
R_OIL            = 5.746   # m    inner mineral-oil (signal) radius (tank GDML)

# Oil-constituent nuclei: the paper's coherent upscattering is on the mineral
# oil (CH2) -> carbon (and hydrogen). Upscatters on the surrounding hall
# materials (O16/Si28/Ca40/Fe56 in concrete/steel) are NOT part of the paper's
# signal and produce off-axis e+e- that distort the angular distribution.
OIL_TARGET_PDGS  = {1000060120, 1000010010}   # C12, H1

output_stem = "output/MiniBooNE_VectorPortal_kaon_e_full"
os.makedirs("output", exist_ok=True)

# ---------------------------------------------------------------------------
# Detector (MiniBooNE mineral-oil tank, via GDML loader)
# ---------------------------------------------------------------------------
print("Loading MiniBooNE detector model (GDML) ...")
detector_model = utilities.load_detector("SBN", detector=experiment)

chi_type       = siren.dataclasses.Particle.ParticleType(PDGID_CHI)
chi_prime_type = siren.dataclasses.Particle.ParticleType(PDGID_CHI_PRIME)
v1_type        = siren.dataclasses.Particle.ParticleType(PDGID_V1)

# ---------------------------------------------------------------------------
# Build the full vector-portal process chain via the factory
# ---------------------------------------------------------------------------
print("Building vector-portal processes (factory) ...")
primary_processes, secondary_processes, chi_flux = load_vector_portal(
    m_chi          = M_CHI,
    m_chi_prime    = M_CHI_PRIME,
    m_V1           = M_V1,
    m_V2           = M_V2,
    g_D            = G_D,
    epsilon_1      = EPSILON_1,
    epsilon_2      = EPSILON_2,
    detector_model = detector_model,
    flux_tag       = FLUX_TAG,
    m_meson        = M_MESON,
    m_lepton       = M_LEPTON,
    max_energy     = MAX_ENERGY,
)

n_targets = len(primary_processes.get(chi_type, []))
print("  Upscattering targets built from GDML materials: %d" % n_targets)
print("  Secondary decays: chi' (%d), V1 (%d)"
      % (len(secondary_processes.get(chi_prime_type, [])),
         len(secondary_processes.get(v1_type, []))))
if n_targets == 0:
    raise RuntimeError(
        "No upscattering targets — the MiniBooNE GDML materials did not map "
        "to any entry in the factory target_db (expected C12, H1). Check that "
        "load_detector returned oil sectors.")

# ---------------------------------------------------------------------------
# Fiducial volume: 5 m sphere at the tank center
# ---------------------------------------------------------------------------
# The example builds Sphere(5.0, 0.0) in detector-local coordinates; the
# detector model's DetectorOrigin places it at the tank center in the BNB
# frame, so a detector-local sphere is centered on the tank automatically.
fiducial_volume = siren.geometry.Sphere(R_FID, 0.0)
print("  Fiducial: %.1f m sphere at tank center" % R_FID)

# Mineral-oil signal volume (for the upscatter-in-oil cut). The upscatter
# (chi -> chi') must happen in the oil on a C/H target to be a paper signal.
oil_volume = siren.geometry.Sphere(R_OIL, 0.0)
print("  Oil volume: %.3f m sphere (upscatter must occur here, on C/H)" % R_OIL)

# ---------------------------------------------------------------------------
# Injection distributions
#
# A primary needs a FULL stack: mass + energy + direction + position.
# chi_flux is a TabulatedFluxDistribution -> ENERGY ONLY, so we add the rest
# explicitly (matching the working DuttaKim_SBND_full_chain pattern):
#   - PrimaryMass(M_CHI)
#   - chi_flux                       (energy)
#   - FixedDirection(+z, beam axis)
#   - PointSourcePositionDistribution(origin, max_distance)
# ---------------------------------------------------------------------------
from siren import distributions as _dists
from siren.math import Vector3D as _Vec3

# Source point and reach. Detector-local convention (the model places the
# tank at DetectorOrigin), beam along +z. max_distance must span the tank.
_SOURCE_POINT   = [0.0, 0.0, 0.0]   # detector-local origin (tank center frame)
_MAX_DISTANCE   = 50.0              # m, generous reach across the ~12 m tank

primary_injection_distributions = [
    _dists.PrimaryMass(M_CHI),
    chi_flux,
    _dists.FixedDirection(_Vec3(0.0, 0.0, 1.0)),
    _dists.PointSourcePositionDistribution(_SOURCE_POINT, _MAX_DISTANCE),
]
# Physical distributions: mass + energy + direction (no position weighting
# needed on the physical side here; the example uses energy+direction).
primary_physical_distributions = [
    _dists.PrimaryMass(M_CHI),
    chi_flux,
    _dists.FixedDirection(_Vec3(0.0, 0.0, 1.0)),
]

# ---------------------------------------------------------------------------
# Secondary vertex distributions (must have the SAME KEYS as
# secondary_interactions). Most secondaries decay/scatter anywhere along the
# path (SecondaryPhysicalVertexDistribution); the chi upscatter vertex is
# BOUNDED to the fiducial so chi actually interacts in the oil
# (SecondaryBoundedVertexDistribution), per the SBND chain.
# ---------------------------------------------------------------------------
_sv         = _dists.SecondaryPhysicalVertexDistribution()
_sv_bounded = _dists.SecondaryBoundedVertexDistribution(fiducial_volume)

secondary_injection_distributions = {
    pt: [_sv] for pt in secondary_processes
}
# chi (the upscattering primary-secondary) gets the bounded vertex.
if chi_type in secondary_injection_distributions:
    secondary_injection_distributions[chi_type] = [_sv_bounded]

secondary_physical_distributions = {
    pt: [_sv] for pt in secondary_processes
}
if chi_type in secondary_physical_distributions:
    secondary_physical_distributions[chi_type] = [_sv_bounded]

# ---------------------------------------------------------------------------
# Injector  (primary = chi; production + V1->chichi are folded into chi_flux)
# ---------------------------------------------------------------------------
print("\nBuilding injector (%d events) ..." % events_to_inject)
injector = siren.injection.Injector()
injector.number_of_events                  = events_to_inject
injector.detector_model                    = detector_model
injector.primary_type                      = chi_type
injector.primary_interactions              = primary_processes[chi_type]
# ---------------------------------------------------------------------------
# Stopping condition — REQUIRED so the chain propagates through all vertices.
# Without it the injector stops after the first (chi upscatter) vertex and
# never produces chi' / V1_signal / e+e-. Taken from the working SBND chain's
# onshell_stopping_condition. Returns False = keep propagating this secondary,
# True = stop.
# ---------------------------------------------------------------------------
def onshell_stopping_condition(datum, i):
    sec    = int(datum.record.signature.secondary_types[i])
    parent = int(datum.record.signature.primary_type)
    if sec == 5922:  return False          # V1 (prod AND signal) -> keep going
    if sec == 5917:  return parent != 5922 or i != 0   # chi only from V1 decay
    if sec == 5918:  return False          # chi' -> de-excites
    return True

injector.primary_injection_distributions   = primary_injection_distributions
injector.secondary_interactions            = secondary_processes
injector.secondary_injection_distributions = secondary_injection_distributions
injector.stopping_condition                = onshell_stopping_condition

# ---------------------------------------------------------------------------
# Event generation  (the shipped example calls generate_events() which does
# NOT exist — iterate the injector, the SBND-chain pattern).
# ---------------------------------------------------------------------------
print("Generating events ...")
events = []
try:
    for event in injector:
        events.append(event)
except StopIteration:
    pass
except Exception as e:
    # Fall back to single-event generation if direct iteration is unsupported
    print("  (direct iteration failed: %r — trying generate_event loop)" % e)
    while len(events) < events_to_inject:
        try:
            events.append(injector.generate_event())
        except Exception:
            break
print("  Generated %d events." % len(events))

# ---------------------------------------------------------------------------
# Weighter
# ---------------------------------------------------------------------------
print("\nBuilding weighter ...")
weighter = siren.injection.Weighter()
weighter.injectors                        = [injector]
weighter.detector_model                   = detector_model
weighter.primary_type                     = chi_type
weighter.primary_interactions             = primary_processes[chi_type]
weighter.primary_physical_distributions   = primary_physical_distributions
weighter.secondary_interactions           = secondary_processes
weighter.secondary_physical_distributions = secondary_physical_distributions

# ---------------------------------------------------------------------------
# Compute weights and apply the fiducial cut
# Accessors confirmed against DuttaKim_SBND_full_chain make_fiducial_metric:
#   for datum in ev.tree: r = datum.record
#   r.signature.primary_type ; r.interaction_vertex[0..2]
# ---------------------------------------------------------------------------
print("Computing weights ...")
from siren.math import Vector3D as _MV3

_n_nonempty   = 0
_n_rawpos     = 0
_seen_types   = {}   # primary_type -> count, across first events (diagnostic)
_seen_secs    = {}   # secondary_type -> count, across first events (diagnostic)

raw_weights = []
for idx, ev in enumerate(events):
    # tree present?
    try:
        tree_ok = bool(ev.tree)
    except Exception:
        tree_ok = False
    if tree_ok:
        _n_nonempty += 1
        # record which vertex primary/secondary types appear (first 200 events)
        if idx < 200:
            try:
                for datum in ev.tree:
                    pt = int(datum.record.signature.primary_type)
                    _seen_types[pt] = _seen_types.get(pt, 0) + 1
                    for sp in datum.record.signature.secondary_types:
                        s = int(sp)
                        _seen_secs[s] = _seen_secs.get(s, 0) + 1
            except Exception:
                pass
    # raw weight
    try:
        w = weighter(ev)
    except Exception as e:
        if idx < 3:
            print("  [diag] weighter(ev) raised on event %d: %r" % (idx, e))
        w = 0.0
    if not np.isfinite(w):
        w = 0.0
    if w > 0:
        _n_rawpos += 1
    raw_weights.append(w)
raw_weights = np.array(raw_weights)

print("  [diag] events with non-empty tree : %d / %d" % (_n_nonempty, len(events)))
print("  [diag] events with raw weight > 0  : %d" % _n_rawpos)
print("  [diag] vertex primary types seen (first 200 ev): %s"
      % {k: _seen_types[k] for k in sorted(_seen_types)})
print("  [diag] secondary types seen (first 200 ev): %s"
      % {k: _seen_secs[k] for k in sorted(_seen_secs)})

def signal_in_fiducial(event):
    """True if the visible e+e- vertex is inside the fiducial volume.
    Identified by a vertex whose secondaries include BOTH e+ and e-
    (robust to the shared V1 PDG code 5922)."""
    try:
        for datum in event.tree:
            r = datum.record
            secs = [int(s) for s in r.signature.secondary_types]
            if 11 in secs and -11 in secs:   # e- and e+ in this vertex
                vtx = _MV3(r.interaction_vertex[0],
                           r.interaction_vertex[1],
                           r.interaction_vertex[2])
                if fiducial_volume.IsInside(vtx):
                    return True
    except Exception:
        return False
    return False

def upscatter_in_oil(event):
    """True if the chi upscatter (chi -> chi') vertex is on a C/H target
    AND inside the mineral-oil volume. This restricts the signal to coherent
    upscattering in the oil, matching the paper (excludes hall O/Si/Ca/Fe)."""
    try:
        for datum in event.tree:
            r = datum.record
            secs = [int(s) for s in r.signature.secondary_types]
            # upscatter vertex: produces chi' (5918) and recoils a nucleus
            if 5918 in secs and any(s in OIL_TARGET_PDGS for s in secs):
                vtx = _MV3(r.interaction_vertex[0],
                           r.interaction_vertex[1],
                           r.interaction_vertex[2])
                if oil_volume.IsInside(vtx):
                    return True
    except Exception:
        return False
    return False

def signal_passes(event):
    """Full signal selection: upscatter on C/H in the oil AND visible
    e+e- vertex in the fiducial."""
    return upscatter_in_oil(event) and signal_in_fiducial(event)

fid_mask     = np.array([signal_passes(ev) for ev in events])
# diagnostic breakdown of each cut independently
_n_upscatter = int(np.sum([upscatter_in_oil(ev) for ev in events]))
_n_eevtx     = int(np.sum([signal_in_fiducial(ev) for ev in events]))
print("  [diag] upscatter on C/H in oil    : %d" % _n_upscatter)
print("  [diag] e+e- vertex in fiducial     : %d" % _n_eevtx)
print("  [diag] BOTH (signal selection)     : %d" % int(fid_mask.sum()))
weights  = np.where(fid_mask, raw_weights, 0.0)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  RESULTS  —  MiniBooNE Vector Portal  K+ -> e+ channel")
print("=" * 60)
finite = weights[(weights > 0) & np.isfinite(weights)]
print("  Events generated        : %d" % len(events))
print("  In-fiducial events      : %d" % int(fid_mask.sum()))
print("  Finite positive weights : %d" % len(finite))
if len(finite):
    print("  Weight range            : %.3e - %.3e" % (finite.min(), finite.max()))
    print("  Total expected signal   : %.3e events" % finite.sum())
else:
    print("  WARNING: no positive in-fiducial weights")
print("=" * 60)

# ---------------------------------------------------------------------------
# MiniBooNE reconstruction cuts (paper, "Fits and discussions")
# ---------------------------------------------------------------------------
E_VIS_THRESHOLD = 0.140     # GeV   visible-energy threshold (140 MeV)
E_PAIR_MAX_DEG  = 10.0      # deg   max e+e- opening angle (single-ring)

# Energy-dependent detection efficiency epsilon(E_vis), refs [76,77] in the
# paper. The published curve must be DIGITIZED and supplied here; the stub
# below returns 1.0 (no efficiency reweighting) and MUST be replaced before
# comparing absolute bin heights to Fig. 2 — a flat efficiency will not
# reproduce the spectral shape. Provide (E_GeV, eff) sample points and the
# function linearly interpolates.
#
#   _EFF_TABLE = np.array([[0.10, 0.0], [0.15, 0.30], [0.30, 0.45],
#                          [0.50, 0.50], [1.00, 0.45], [1.50, 0.35]])
#
_EFF_TABLE = None           # <-- set to the digitized [[E_GeV, eff], ...] array

def detection_efficiency(E_vis_gev):
    """Linear-interpolated detection efficiency at visible energy [GeV].
    Returns 1.0 if no table is supplied (placeholder — see note above)."""
    if _EFF_TABLE is None:
        return 1.0
    E = np.asarray(_EFF_TABLE)[:, 0]
    eff = np.asarray(_EFF_TABLE)[:, 1]
    return float(np.interp(E_vis_gev, E, eff, left=eff[0], right=eff[-1]))

# ---------------------------------------------------------------------------
# Count-rate plots: visible energy and angle of the e+e- signal.
# Scaffold only — refine binning and POT normalization to match the
# paper's MiniBooNE excess panels.
# ---------------------------------------------------------------------------
def signal_eepair_observables(event):
    """Return (E_vis, cos_theta) of the e+e- pair from the SIGNAL vertex,
    AFTER applying the MiniBooNE reconstruction cuts (paper, 'Fits and
    discussions'):
      - E_vis >= 140 MeV threshold
      - e+e- opening angle < 10 deg  (single-ring requirement; the boosted
        V1 -> e+e- pair must overlap into one Cherenkov ring)
    The pair MUST come from a single V1->e+e- vertex (same datum) inside the
    fiducial volume. Momentum convention [E, px, py, pz] (slot0=E, slot3=pz).
    Returns None if no vertex passes all cuts."""
    try:
        for datum in event.tree:
            r = datum.record
            secs = [int(s) for s in r.signature.secondary_types]
            if not (11 in secs and -11 in secs):
                continue
            # this vertex has BOTH e- and e+ : a V1 -> e+e- vertex
            vtx = _MV3(r.interaction_vertex[0],
                       r.interaction_vertex[1],
                       r.interaction_vertex[2])
            if not fiducial_volume.IsInside(vtx):
                continue
            # collect the e+ and e- 3-momenta FROM THIS SAME vertex
            P = np.zeros(4)
            n_e = 0
            e_p3 = []
            for i, sp in enumerate(r.signature.secondary_types):
                if int(sp) in (11, -11):
                    p = r.secondary_momenta[i]
                    P = P + np.array([p[0], p[1], p[2], p[3]])
                    e_p3.append(np.array([p[1], p[2], p[3]]))
                    n_e += 1
            if n_e < 2 or len(e_p3) < 2:
                continue

            # ── CUT 1: e+e- opening angle < 10 deg ───────────────────────────
            p1, p2 = e_p3[0], e_p3[1]
            n1, n2 = np.linalg.norm(p1), np.linalg.norm(p2)
            if n1 <= 0.0 or n2 <= 0.0:
                continue
            cos_open = float(np.dot(p1, p2) / (n1 * n2))
            cos_open = max(-1.0, min(1.0, cos_open))
            open_deg = math.degrees(math.acos(cos_open))
            if open_deg > E_PAIR_MAX_DEG:
                continue

            # ── CUT 2: visible-energy threshold ──────────────────────────────
            E_vis = P[0]                       # GeV
            if E_vis < E_VIS_THRESHOLD:
                continue

            pmag = math.sqrt(P[1]**2 + P[2]**2 + P[3]**2)
            cth = P[3] / pmag if pmag > 0 else 0.0
            return E_vis, cth
    except Exception:
        return None
    return None

E_vis_list, cos_list, w_list = [], [], []
for ev, w in zip(events, weights):
    if w <= 0:
        continue
    obs = signal_eepair_observables(ev)
    if obs is None:
        continue
    E_vis_gev = obs[0]
    # ── CUT 3: energy-dependent detection efficiency (multiplies weight) ─────
    w_eff = w * detection_efficiency(E_vis_gev)
    if w_eff <= 0:
        continue
    E_vis_list.append(E_vis_gev)
    cos_list.append(obs[1])
    w_list.append(w_eff)

# Save observables so the standalone plotting script can read them
# (avoids re-running the simulation just to restyle plots).
_obs_path = output_stem + "_observables.npz"
np.savez(_obs_path,
         E_vis=np.array(E_vis_list),   # GeV
         cos_theta=np.array(cos_list),
         weight=np.array(w_list))
print("  Saved observables -> %s  (%d signal events)"
      % (_obs_path, len(w_list)))

if w_list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    E_vis = np.array(E_vis_list)
    cos_t = np.array(cos_list)
    wv    = np.array(w_list)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].hist(E_vis * 1e3, bins=30, weights=wv, histtype="step", color="C3")
    ax[0].set_xlabel("E_vis (e+e-)  [MeV]")
    ax[0].set_ylabel("Weighted events")
    ax[0].set_title("MiniBooNE  K+ -> e+ : visible energy")

    ax[1].hist(cos_t, bins=30, range=(-1, 1), weights=wv,
               histtype="step", color="C0")
    ax[1].set_xlabel("cos(theta)  wrt beam")
    ax[1].set_ylabel("Weighted events")
    ax[1].set_title("MiniBooNE  K+ -> e+ : angular")

    fig.tight_layout()
    plot_path = output_stem + "_countrate.png"
    fig.savefig(plot_path, dpi=130)
    print("  Saved count-rate plots -> %s" % plot_path)
else:
    print("  No positive-weight signal events to plot.")

print("  Done.")
