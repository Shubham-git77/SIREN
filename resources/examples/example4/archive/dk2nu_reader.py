"""
VectorPortal_ICARUS_kaon_eplus_dk2nu.py
========================================
Full on-shell Dutta-Kim Vector Portal chain at ICARUS, K+ channel,
reading directly from dk2nu ROOT files.

Injection chain (kaon-first, on-shell chi'):
    1. Read K+ from BNB dk2nu ROOT file via Dk2nuReader
    2. K+ -> e+ nu_e V1              (three-body decay, Dalitz)
    3. V1 -> chi chi                  (dark photon -> DM pair)
    4. chi + Ar -> chi' + Ar          (upscattering via V2)
    5. chi' -> chi + V1_signal        (chi' decay)
    6. V1_signal -> e+ e-             (visible signal)

Key differences from SBND pion script:
    - K+ parent (ptype=321) instead of pi+ (ptype=211)
    - e+ lepton from kaon decay (not mu+)
    - ICARUS detector geometry (600 m baseline, larger fiducial)
    - Ar40 nuclear target (same as SBND version)

Reference: Dutta et al., PRL 129, 111803 (2022) [arXiv:2110.11944]
"""

import os
import sys
import glob
import numpy as np

import siren
from siren import utilities
from siren._util import GenerateEvents, SaveEvents
from siren.math import Vector3D
from siren.geometry import Box

# ---------------------------------------------------------------------------
# Import model classes via SIREN module loader
# ---------------------------------------------------------------------------
from siren import _util as _siren_util

_dt_base = os.path.join(
    _siren_util.resource_package_dir(), "processes", "DarkNewsTables",
)

_mod_mp = _siren_util.load_module(
    "siren.resources.processes.DarkNewsTables.MesonProduction",
    os.path.join(_dt_base, "MesonProduction.py"),
)
_mod_vp = _siren_util.load_module(
    "siren.resources.processes.DarkNewsTables.VectorPortal",
    os.path.join(_dt_base, "VectorPortal.py"),
)
_mod_dk = _siren_util.load_module(
    "siren.resources.processes.DarkNewsTables.Dk2nuReader",
    os.path.join(_dt_base, "Dk2nuReader.py"),
)

MesonThreeBodySIRENDecay  = _mod_mp.MesonThreeBodySIRENDecay
DarkPhotonToChiDecay      = _mod_vp.DarkPhotonToChiDecay
ChiPrimeDecay             = _mod_vp.ChiPrimeDecay
DarkPhotonDecay           = _mod_vp.DarkPhotonDecay
VectorPortalUpscatteringXS = _mod_vp.VectorPortalUpscatteringXS

read_dk2nu                 = _mod_dk.read_dk2nu
dk2nu_to_primary_distribution = _mod_dk.dk2nu_to_primary_distribution
print_summary              = _mod_dk.print_summary

# PDG code for K+ in dk2nu (ptype field)
PTYPE_KPLUS = 321

# ---------------------------------------------------------------------------
# Physics parameters  (Table I, arXiv:2110.11944 Model 1b)
# ---------------------------------------------------------------------------
M_CHI        = 8e-3      # GeV  chi   dark matter ground state
M_CHI_PRIME  = 50e-3     # GeV  chi'  dark matter excited state
M_V1         = 17e-3     # GeV  V1    light dark photon (visible)
M_V2         = 200e-3    # GeV  V2    heavy upscattering mediator
G_D          = 1.0       # dark gauge coupling
EPSILON_1    = 7e-5      # V1 -> e+e- kinetic mixing
EPSILON_2    = 1e-4      # chi upscattering kinetic mixing
G_MU         = 1e-3      # K+ -> l nu V1 effective coupling

# Particle masses
M_KAON  = 0.49368    # GeV  K+
M_EPLUS = 0.000511   # GeV  e+
M_ARGON = 37.215     # GeV  Ar40 nuclear mass

# PDG IDs
PDGID_KAON      = 321
PDGID_EPLUS     = -11
PDGID_NUE       = 12
PDGID_V1_PROD   = 5922   # V1 from kaon decay
PDGID_CHI       = 5917
PDGID_CHI_PRIME = 5918
PDGID_V1_SIGNAL = 5923   # V1 from chi' decay (visible)

# Particle types
PT = lambda pdg: siren.dataclasses.Particle.ParticleType(pdg)
KAON_TYPE      = PT(PDGID_KAON)
V1_PROD_TYPE   = PT(PDGID_V1_PROD)
CHI_TYPE       = PT(PDGID_CHI)
CHI_PRIME_TYPE = PT(PDGID_CHI_PRIME)
V1_SIG_TYPE    = PT(PDGID_V1_SIGNAL)

# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------
events_to_inject = 10_000

# dk2nu ROOT file — same file used for pion/kaon flux extraction
# dk2nu ROOT file — same directory as this script
DK2NU_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "nubeamHighSample.dk2nu.root"
)
# Fallback to home directory if not found locally
if not os.path.exists(DK2NU_FILE):
    DK2NU_FILE = "/home/shubham/nubeamHighSample.dk2nu.root"

# Output
output_stem = "output/ICARUS_VectorPortal_kaon_e_dk2nu"
os.makedirs("output", exist_ok=True)

# ---------------------------------------------------------------------------
# Verify kinematic feasibility
# ---------------------------------------------------------------------------
assert M_KAON > M_EPLUS + M_V1, \
    "K+ -> e+ nu V1 kinematically forbidden: M_K=%.1f < M_e+M_V1=%.1f MeV" \
    % (M_KAON*1e3, (M_EPLUS+M_V1)*1e3)
assert M_CHI_PRIME > M_CHI + M_V1, \
    "chi' -> chi V1 forbidden: m_chi'=%.0f < m_chi+m_V1=%.0f MeV" \
    % (M_CHI_PRIME*1e3, (M_CHI+M_V1)*1e3)
assert M_V1 > 2 * 0.000511, "V1 -> e+e- forbidden"

# ---------------------------------------------------------------------------
# 1. Read K+ from dk2nu file
# ---------------------------------------------------------------------------
print("Reading K+ from dk2nu file: %s" % DK2NU_FILE)
if not os.path.exists(DK2NU_FILE):
    print("ERROR: dk2nu file not found: %s" % DK2NU_FILE)
    sys.exit(1)

dk2nu_data = read_dk2nu([DK2NU_FILE], parent_pdg=[PTYPE_KPLUS])
print_summary(dk2nu_data)

total_pot    = dk2nu_data["pot"]
n_kaons      = len(dk2nu_data["E"])
pot_per_kaon = total_pot / n_kaons if n_kaons > 0 else 0.0
print("K+ events    : %d" % n_kaons)
print("Total POT    : %.3e" % total_pot)
print("POT per kaon : %.3e" % pot_per_kaon)

# ---------------------------------------------------------------------------
# 2. Load ICARUS detector model
# ---------------------------------------------------------------------------
print("\nLoading ICARUS detector model ...")
base = os.environ.get(
    "ICARUS_MODEL_DIR",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "..", "detectors", "ICARUS", "ICARUS-v1",
    ),
)
detector_model = siren.detector.DetectorModel(
    os.path.join(base, "densities.dat"),
    os.path.join(base, "materials.dat"),
)

# ICARUS fiducial: active LAr 2×(2.56 m × 1.58 m × 8.97 m)
fiducial = Box(2.56, 1.58, 8.97)

# ---------------------------------------------------------------------------
# 3. Build physics processes
# ---------------------------------------------------------------------------
print("\nBuilding physics processes ...")

# Vertex 1: K+ -> e+ + nu_e + V1  (three-body Dalitz, Carlson-Rislow)
kaon_decay = MesonThreeBodySIRENDecay(
    m_mediator     = M_V1,
    g_mu           = G_MU,
    pdgid_meson    = PDGID_KAON,
    pdgid_lepton   = PDGID_EPLUS,
    pdgid_neutrino = PDGID_NUE,
    pdgid_mediator = PDGID_V1_PROD,
    m_meson        = M_KAON,
    m_lepton       = M_EPLUS,
)
print("  K+ BSM decay width  : %.4e GeV" % kaon_decay._total_width)

# Vertex 2: V1 -> chi + chi  (dark photon to DM pair)
v1_to_chi = DarkPhotonToChiDecay(
    M_V1, M_CHI, G_D,
    pdgid_V1  = PDGID_V1_PROD,
    pdgid_chi = PDGID_CHI,
)
print("  V1->chi chi width   : %.4e GeV" % v1_to_chi._total_width)

# Vertex 3: chi + Ar40 -> chi' + Ar40  (upscattering via V2)
upscatter = VectorPortalUpscatteringXS(
    M_CHI, M_CHI_PRIME, M_V2, G_D, EPSILON_2,
    pdgid_chi       = PDGID_CHI,
    pdgid_chi_prime = PDGID_CHI_PRIME,
    nuclear_pdgid   = 1000180400,
    nuclear_mass    = M_ARGON,
    nuclear_name    = "Ar40",
    A = 40, Z = 18,
)
print("  chi threshold       : %.3f MeV" % (upscatter.Ethreshold * 1e3))

# Vertex 4: chi' -> chi + V1_signal
chi_prime_decay = ChiPrimeDecay(
    M_CHI, M_CHI_PRIME, M_V1, G_D,
    pdgid_chi_prime = PDGID_CHI_PRIME,
    pdgid_chi       = PDGID_CHI,
    pdgid_V1        = PDGID_V1_SIGNAL,
)

# Vertex 5: V1_signal -> e+ + e-  (visible signal)
visible_decay = DarkPhotonDecay(
    M_V1, EPSILON_1, pdgid_V1 = PDGID_V1_SIGNAL,
)
print("  V1->e+e- width      : %.4e GeV" % visible_decay._total_width)
print("  All 5 processes built.")

# Process maps
primary_processes = {KAON_TYPE: [kaon_decay]}
secondary_processes = {
    V1_PROD_TYPE:   [v1_to_chi],
    CHI_TYPE:       [upscatter],
    CHI_PRIME_TYPE: [chi_prime_decay],
    V1_SIG_TYPE:    [visible_decay],
}

# ---------------------------------------------------------------------------
# 4. Primary distribution from dk2nu
#    Dk2nuReader.dk2nu_to_primary_distribution handles:
#      - Coordinate transform: BNB beamline -> detector frame
#      - Vertex projection to world boundary
#      - Kaon sampling bias (E^2 × cos_theta × exp(-r/200))
#      - PrimaryExternalDistribution construction
# ---------------------------------------------------------------------------
print("\nBuilding primary distribution from dk2nu ...")
primary_dist = dk2nu_to_primary_distribution(
    dk2nu_data,
    detector_model,
    parent_pdg = PTYPE_KPLUS,
    # sampling_bias=None uses default bias: E^2 * cos_theta * exp(-r/200)
)
print("  Loaded %d K+ events for injection" % primary_dist.GetPhysicalNumEvents()
      if hasattr(primary_dist, 'GetPhysicalNumEvents') else
      "  Primary distribution built")

primary_injection_distributions  = [primary_dist]
primary_physical_distributions   = [primary_dist]

# ---------------------------------------------------------------------------
# 5. Secondary vertex distributions
# ---------------------------------------------------------------------------
sv         = siren.distributions.SecondaryPhysicalVertexDistribution()
sv_bounded = siren.distributions.SecondaryBoundedVertexDistribution(fiducial)

secondary_injection_distributions = {
    V1_PROD_TYPE:   [sv],          # V1 decays anywhere along kaon path
    CHI_TYPE:       [sv_bounded],  # upscatter confined to ICARUS fiducial
    CHI_PRIME_TYPE: [sv],          # chi' decays in-place (sub-micron)
    V1_SIG_TYPE:    [sv],          # V1_signal decays ~mm from upscatter
}

# ---------------------------------------------------------------------------
# 6. Stopping condition
# ---------------------------------------------------------------------------
def stop(datum, i):
    sec    = int(datum.record.signature.secondary_types[i])
    parent = int(datum.record.signature.primary_type)
    if sec == PDGID_V1_PROD:   return False   # V1_prod -> continue to chi chi
    if sec == PDGID_CHI:       return parent != PDGID_V1_PROD or i != 0  # chi[0] from V1_prod only
    if sec == PDGID_CHI_PRIME: return False   # chi' -> continue to decay
    if sec == PDGID_V1_SIGNAL: return False   # V1_signal -> continue to e+e-
    return True                                # e+, nu_e, recoil Ar, e+e- -> stop

# ---------------------------------------------------------------------------
# 7. Injector
# ---------------------------------------------------------------------------
print("\nBuilding injector (%d events) ..." % events_to_inject)
injector = siren.injection.Injector()
injector.number_of_events                  = events_to_inject
injector.detector_model                    = detector_model
injector.primary_type                      = KAON_TYPE
injector.primary_interactions              = primary_processes[KAON_TYPE]
injector.primary_injection_distributions   = primary_injection_distributions
injector.secondary_interactions            = secondary_processes
injector.secondary_injection_distributions = secondary_injection_distributions
injector.stopping_condition                = stop

print("Generating %d events ..." % events_to_inject)
events, gen_times = GenerateEvents(injector)
print("Generated %d event trees." % len(events))

# ---------------------------------------------------------------------------
# 8. Weighter
# ---------------------------------------------------------------------------
weighter = siren.injection.Weighter()
weighter.injectors                        = [injector]
weighter.detector_model                   = detector_model
weighter.primary_type                     = KAON_TYPE
weighter.primary_interactions             = primary_processes[KAON_TYPE]
weighter.secondary_interactions           = secondary_processes
weighter.primary_physical_distributions   = primary_physical_distributions
weighter.secondary_physical_distributions = {}

# ---------------------------------------------------------------------------
# 9. Save events
# ---------------------------------------------------------------------------
SaveEvents(
    events, weighter, gen_times,
    fid_vol         = fiducial,
    output_filename = output_stem,
)

# ---------------------------------------------------------------------------
# 10. Summary
# ---------------------------------------------------------------------------
weights     = np.array([weighter(ev) for ev in events])
finite_mask = np.isfinite(weights) & (weights > 0)
valid_w     = weights[finite_mask]

def effective_sample_fraction(w_all):
    w = w_all[(np.isfinite(w_all)) & (w_all > 0)]
    if len(w) == 0: return 0.0
    return (w.sum()**2) / (len(w) * (w**2).sum()) * len(w) / len(w_all)

eff = effective_sample_fraction(weights) * 100.0

# Count signal vertices
n_chi_scatter = 0
n_v1_signal   = 0
for event in events:
    for datum in event.tree:
        ptype = int(datum.record.signature.primary_type)
        if ptype == PDGID_CHI:       n_chi_scatter += 1
        if ptype == PDGID_V1_SIGNAL: n_v1_signal   += 1

print()
print("=" * 60)
print("  RESULTS  —  ICARUS Vector Portal  K+ channel  (dk2nu)")
print("=" * 60)
print("  Chain: K+(321) -> e+ nu_e V1 -> chi chi")
print("         chi + Ar40 -> chi' + Ar40")
print("         chi' -> chi + V1_sig -> e+e-  [VISIBLE]")
print("  m_chi=%.0f MeV  m_chi'=%.0f MeV  m_V1=%.0f MeV  m_V2=%.0f MeV"
      % (M_CHI*1e3, M_CHI_PRIME*1e3, M_V1*1e3, M_V2*1e3))
print()
print("  dk2nu K+ events      : %d" % n_kaons)
print("  Total POT            : %.3e" % total_pot)
print("  Events generated     : %d" % len(events))
print("  Chi scatter vertices : %d" % n_chi_scatter)
print("  V1_signal vertices   : %d" % n_v1_signal)
print("  Finite pos. weights  : %d / %d" % (finite_mask.sum(), len(weights)))
if len(valid_w) > 0:
    print("  Weight range         : %.3e - %.3e"
          % (valid_w.min(), valid_w.max()))
    print("  Effective sample fr. : %.1f%%" % eff)
    print("  Expected signal (POT): %.3e events" % valid_w.sum())
print("  Injection rate       : %.0f events/s"
      % (len(events) / sum(gen_times) if sum(gen_times) > 0 else 0))
print("  Output               : %s.*" % output_stem)
print("=" * 60)

# POT-scaled signal estimate
if n_kaons > 0 and len(valid_w) > 0:
    print()
    print("  POT-scaled signal estimates:")
    for target_pot in [6e20, 1.2e21, 6e21]:
        scale = target_pot / total_pot if total_pot > 0 else 0
        print("    At %.0e POT: %.3e signal events"
              % (target_pot, valid_w.sum() * scale))
