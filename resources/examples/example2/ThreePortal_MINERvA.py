"""
ThreePortal_MINERvA.py
======================
HNL production + ThreePortalDecay in MINERvA.

Physics chain
-------------
  Primary   :  nu_mu  A  ->  N4  A       (Primakoff upscattering, DarkNews)
  Secondary :  N4  ->  nu  e-  e+        (ThreePortalDecay via PyDarkNewsDecay)

Structure mirrors DipolePortal_MINERvA.py exactly.
Uses siren._util, utilities.load_detector/load_processes/load_flux,
siren.injection.Injector and Weighter directly. No SIREN_Controller.
"""

import os
import numpy as np
import siren
from siren import utilities
from siren._util import GenerateEvents, SaveEvents, get_processes_model_path

SaveDarkNewsProcesses = siren.resources.processes.DarkNewsTables.SaveDarkNewsProcesses

# ---------------------------------------------------------------------------
# 0.  Load NewDarkNewsDecay (PyDarkNewsDecay with isinstance fix)
# ---------------------------------------------------------------------------
_DECAY_MOD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "processes", "DarkNewsTables", "DarkNewsDecay.py")
from siren import _util as _siren_util
_siren_util.load_module("DarkNewsDecay", _DECAY_MOD_PATH)
from DarkNewsDecay import PyDarkNewsDecay

from DarkNews.processes import ThreePortalDecay
from DarkNews import model as DKmodel

# ---------------------------------------------------------------------------
# 1.  Physics parameters
# ---------------------------------------------------------------------------
M_HNL   = 0.200        # HNL mass [GeV]
UMU4    = 1.0e-3       # |U_mu4|
UD4     = 0.0
EPSILON = 5.0e-4       # kinetic mixing
GD      = 0.5          # dark gauge coupling
MZP     = 0.250        # Z' mass [GeV]  > M_HNL => vector off-shell
MHPRIME = 0.300        # h' mass [GeV]  > M_HNL => scalar off-shell
KS      = 1.0e-3       # scalar mixing (ThreePortalModel only)

# ---------------------------------------------------------------------------
# 2.  model_kwargs
# ---------------------------------------------------------------------------
model_kwargs = {
    "m4"           : M_HNL,
    "Umu4"         : UMU4,
    "UD4"          : UD4,
    "epsilon"      : EPSILON,
    "gD"           : GD,
    "mzprime"      : MZP,
    "mhprime"      : MHPRIME,
    "decay_product": "e+e-",
    "noHC"         : True,
    "HNLtype"      : "dirac",
}

# ---------------------------------------------------------------------------
# 3.  Experiment setup
# ---------------------------------------------------------------------------
events_to_inject = 10000
experiment       = "MINERvA"
detector_model   = utilities.load_detector(experiment)

primary_type = siren.dataclasses.Particle.ParticleType.NuMu

table_name  = f"DarkNewsTables-v{siren.utilities.darknews_version()}/"
table_name += "ThreePortal_M%2.2e_eps%2.2e_gD%2.2e" % (M_HNL, EPSILON, GD)
table_dir   = os.path.join(get_processes_model_path("DarkNewsTables"), table_name)
os.makedirs(table_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# 4.  Load DarkNews processes
# ---------------------------------------------------------------------------
primary_processes, secondary_processes, primary_ups_keys, secondary_dec_keys = \
    utilities.load_processes(
        "DarkNewsTables",
        primary_type   = primary_type,
        detector_model = detector_model,
        table_name     = table_name,
        **model_kwargs,
    )

print("[INFO] primary_processes  :", primary_processes)
print("[INFO] secondary_processes:", secondary_processes)


# ---------------------------------------------------------------------------
# REPLACE DEFAULT DECAY WITH ThreePortalDecay
# ---------------------------------------------------------------------------

from DarkNews.processes import ThreePortalDecay
from DarkNewsDecay import PyDarkNewsDecay

# Get wrapped decay
wrapped_decay = list(secondary_processes.values())[0][0]

# Extract actual DarkNews decay
existing_dec_case = wrapped_decay.dec_case   

print("[DEBUG] Existing decay:", existing_dec_case)

# Extract particles
nu_parent   = existing_dec_case.nu_parent
nu_daughter = existing_dec_case.nu_daughter
secondaries = existing_dec_case.secondaries

# Create ThreePortalDecay
threeportal_decay = ThreePortalDecay(
    nu_parent=nu_parent,
    nu_daughter=nu_daughter,
    final_lepton1=secondaries[0],
    final_lepton2=secondaries[1],
    TheoryModel=existing_dec_case.TheoryModel
)

print("[INFO] ThreePortalDecay instantiated successfully")

# Wrap into SIREN decay
wrapped_new_decay = PyDarkNewsDecay(threeportal_decay)

# Replace all secondary decays
for key in secondary_processes:
    secondary_processes[key] = [wrapped_new_decay]

print("[INFO] Replaced default decay with ThreePortalDecay")
# ---------------------------------------------------------------------------
# 5.  Replace secondary decay with ThreePortalDecay
#
#     N4_type defined HERE immediately after load_processes so it is
#     available throughout the rest of the file.
#
#     KEY INSIGHT — why we reuse particles from existing_dec_case:
#     ThreePortalDecay.__init__ -> FermionDileptonDecay.__init__ calls:
#         get_HNL_index(nu_daughter)
#         which does: int(particle.name.strip("nuN")) - 1
#     This requires particle.name to be "N1","N2","N3","N4","N5","N6".
#     pdg.numu.name = "nu(mu)"  ->  strip("nuN") = "(mu)"  ->  int FAILS.
#     The existing dec_case from load_processes already holds the correct
#     particle objects (nu_parent.name="N4", nu_daughter.name="N1" etc.)
#     We reuse them directly — no pdg name guessing needed.
# ---------------------------------------------------------------------------
N4_type = siren.dataclasses.Particle.ParticleType.N4

# Extract existing dec_case that load_processes built for N4
existing_pydecay  = secondary_processes[N4_type][0]
existing_dec_case = existing_pydecay.dec_case

print("[INFO] Existing dec_case type    : %s" % type(existing_dec_case).__name__)
print("[INFO] nu_parent  name           : %s" % existing_dec_case.nu_parent.name)
print("[INFO] nu_daughter name          : %s" % existing_dec_case.nu_daughter.name)
print("[INFO] secondaries               : %s" % existing_dec_case.secondaries)

# Build ThreePortalModel with the same couplings as the upscattering step
tp_model = DKmodel.ThreePortalModel(
    m4      = M_HNL,
    Umu4    = UMU4,
    epsilon = EPSILON,
    gD      = GD,
    mzprime = MZP,
    kappa   = KS,
    mhprime = MHPRIME,
    HNLtype = "dirac",
)

# Instantiate ThreePortalDecay reusing the particle objects from existing_dec_case
# so get_HNL_index receives particles with parseable names ("N4", "N1" etc.)
tp_decay_case = ThreePortalDecay(
    nu_parent     = existing_dec_case.nu_parent,       # N4  (name="N4") ✓
    nu_daughter   = existing_dec_case.nu_daughter,     # HNL-indexed    ✓
    final_lepton1 = existing_dec_case.secondaries[0],  # e-
    final_lepton2 = existing_dec_case.secondaries[1],  # e+
    TheoryModel   = tp_model,
)

print("[INFO] ThreePortalDecay instantiated successfully")
print("[INFO] vector_on_shell=%s  scalar_on_shell=%s"
      % (tp_decay_case.vector_on_shell, tp_decay_case.scalar_on_shell))
print("[INFO] m_parent=%.4f GeV  mm=%.6f GeV"
      % (tp_decay_case.m_parent, tp_decay_case.mm))

# Wrap in PyDarkNewsDecay (isinstance fix applied)
tp_siren_decay = PyDarkNewsDecay(tp_decay_case)

# Load cached VEGAS integrator if available
decay_table_dir = os.path.join(table_dir, "decay_ThreePortal_ee")
os.makedirs(decay_table_dir, exist_ok=True)
tp_siren_decay.load_from_table(decay_table_dir)

# Replace N4 secondary with ThreePortalDecay
secondary_processes[N4_type] = [tp_siren_decay]

# ---------------------------------------------------------------------------
# 6.  Compute DN_min_decay_width (after replacing secondary_processes)
# ---------------------------------------------------------------------------
DN_min_decay_width = np.inf
for secondary_type, decays in secondary_processes.items():
    for decay in decays:
        is_decay = False
        for signature in decay.GetPossibleSignatures():
            is_decay |= (
                signature.target_type == siren.dataclasses.Particle.ParticleType.Decay
            )
        if not is_decay:
            continue
        decay_width = decay.TotalDecayWidth(secondary_type)
        DN_min_decay_width = min(DN_min_decay_width, decay_width)

print("[INFO] Minimum decay width: %s" % DN_min_decay_width)
assert DN_min_decay_width < np.inf, \
    "No decays found. Check secondary_processes and model parameters."

# ---------------------------------------------------------------------------
# 7.  Primary distributions — identical to DipolePortal_MINERvA.py
# ---------------------------------------------------------------------------
mass_ddist = siren.distributions.PrimaryMass(0)

edist     = siren.utilities.load_flux("NUMI", tag="FHC_ME_numu",
                                      physically_normalized=True)
edist_gen = siren.utilities.load_flux("NUMI", tag="FHC_ME_numu",
                                      min_energy=model_kwargs["m4"],
                                      max_energy=20,
                                      physically_normalized=False)

direction_distribution = siren.distributions.FixedDirection(
    siren.math.Vector3D(0, 0, 1.0)
)

decay_range_func = siren.distributions.DecayRangeFunction(
    model_kwargs["m4"], DN_min_decay_width, 3, 240
)
position_distribution = siren.distributions.DecayRangePositionDistribution(
    1.24, 5.0, decay_range_func,
)

primary_injection_distributions = [
    mass_ddist,
    edist_gen,
    direction_distribution,
    position_distribution,
]

primary_physical_distributions = [
    edist,
    direction_distribution,
]

# ---------------------------------------------------------------------------
# 8.  Secondary distributions — identical to DipolePortal_MINERvA.py
# ---------------------------------------------------------------------------
fiducial_volume = siren.utilities.get_fiducial_volume(experiment)
secondary_injection_distributions = {}
for secondary_type in secondary_processes.keys():
    secondary_injection_distributions[secondary_type] = [
        siren.distributions.SecondaryBoundedVertexDistribution(fiducial_volume)
    ]

# ---------------------------------------------------------------------------
# 9.  Stopping condition — identical to DipolePortal_MINERvA.py
# ---------------------------------------------------------------------------
def stop(datum, i):
    secondary_type = datum.record.signature.secondary_types[i]
    return secondary_type != siren.dataclasses.Particle.ParticleType.N4

# ---------------------------------------------------------------------------
# 10.  Injector — identical to DipolePortal_MINERvA.py
# ---------------------------------------------------------------------------
injector = siren.injection.Injector()
injector.number_of_events                  = events_to_inject
injector.detector_model                    = detector_model
injector.primary_type                      = primary_type
injector.primary_interactions              = primary_processes[primary_type]
injector.primary_injection_distributions   = primary_injection_distributions
injector.secondary_interactions            = secondary_processes
injector.secondary_injection_distributions = secondary_injection_distributions
injector.stopping_condition                = stop

# ---------------------------------------------------------------------------
# 11.  Generate
# ---------------------------------------------------------------------------
print("[INFO] Generating %d events in %s ..." % (events_to_inject, experiment))
print("[INFO] First ~1000 events may be slow — VEGAS initialising 3-body integrator.")
events, gen_times = GenerateEvents(injector)
print("[INFO] %d events generated." % len(events))

# ---------------------------------------------------------------------------
# 12.  Weighter
# ---------------------------------------------------------------------------
os.makedirs("output", exist_ok=True)

weighter = siren.injection.Weighter()
weighter.injectors                        = [injector]
weighter.detector_model                   = detector_model
weighter.primary_type                     = primary_type
weighter.primary_interactions             = primary_processes[primary_type]
weighter.secondary_interactions           = secondary_processes
weighter.primary_physical_distributions   = primary_physical_distributions
weighter.secondary_physical_distributions = {}

# ---------------------------------------------------------------------------
# 13.  Save events
# ---------------------------------------------------------------------------
SaveEvents(
    events, weighter, gen_times,
    output_filename="output/MINERvA_ThreePortal_M%2.2e_eps%2.2e_gD%2.2e"
                    % (M_HNL, EPSILON, GD),
)

# ---------------------------------------------------------------------------
# 14.  Save tables
# ---------------------------------------------------------------------------
tp_siren_decay.save_to_table(decay_table_dir)

SaveDarkNewsProcesses(
    table_dir,
    primary_processes,
    primary_ups_keys,
    secondary_processes,
    secondary_dec_keys,
)

# ---------------------------------------------------------------------------
# 15.  Weights
# ---------------------------------------------------------------------------
weights = [weighter(event) for event in events]
print("[INFO] Done. %d weights computed." % len(weights))
