"""
ThreePortal_MiniBooNE.py
=======================
ThreePortalDecay for MiniBooNE detector

Structure:
- MiniBooNE setup from DipolePortal_MiniBooNE
- ThreePortalDecay from MINERvA version
"""

import os
import numpy as np
import siren
from siren import utilities
from siren._util import GenerateEvents, SaveEvents, get_processes_model_path

SaveDarkNewsProcesses = siren.resources.processes.DarkNewsTables.SaveDarkNewsProcesses

# ---------------------------------------------------------------------------
# Load NewDarkNewsDecay
# ---------------------------------------------------------------------------
_THIS_DIR       = os.path.dirname(os.path.abspath(__file__))
_DECAY_MOD_PATH = os.path.join(_THIS_DIR, "NewDarkNewsDecay.py")

from siren import _util as _siren_util
_siren_util.load_module("NewDarkNewsDecay", _DECAY_MOD_PATH)

from NewDarkNewsDecay import PyDarkNewsDecay
from DarkNews.processes import ThreePortalDecay
from DarkNews import model as DKmodel

# ---------------------------------------------------------------------------
# PHYSICS PARAMETERS 
# ---------------------------------------------------------------------------
M_HNL   = 0.200
UMU4    = 1.0e-3
UD4     = 0.0
EPSILON = 5.0e-4
GD      = 0.5
MZP     = 0.250
MHPRIME = 0.300
KS      = 1.0e-3

model_kwargs = {
    "m4": M_HNL,
    "Umu4": UMU4,
    "UD4": UD4,
    "epsilon": EPSILON,
    "gD": GD,
    "mzprime": MZP,
    "mhprime": MHPRIME,
    "decay_product": "e+e-",
    "noHC": True,
    "HNLtype": "dirac",
}

# ---------------------------------------------------------------------------
# MiniBooNE setup 
# ---------------------------------------------------------------------------
events_to_inject = 10000
experiment = "MiniBooNE"
detector_model = utilities.load_detector(experiment)

primary_type = siren.dataclasses.Particle.ParticleType.NuMu

# ---------------------------------------------------------------------------
# Table setup
# ---------------------------------------------------------------------------
table_name  = f"DarkNewsTables-v{siren.utilities.darknews_version()}/"
table_name += "ThreePortal_M%2.2e_eps%2.2e_gD%2.2e" % (M_HNL, EPSILON, GD)

table_dir = os.path.join(get_processes_model_path("DarkNewsTables"), table_name)
os.makedirs(table_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Load processes
# ---------------------------------------------------------------------------
primary_processes, secondary_processes, primary_ups_keys, secondary_dec_keys = \
    utilities.load_processes(
        "DarkNewsTables",
        primary_type=primary_type,
        detector_model=detector_model,
        table_name=table_name,
        **model_kwargs,
    )

# ---------------------------------------------------------------------------
# Replace decay → ThreePortalDecay
# ---------------------------------------------------------------------------
N4_type = siren.dataclasses.Particle.ParticleType.N4

existing_pydecay  = secondary_processes[N4_type][0]
existing_dec_case = existing_pydecay.dec_case

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

tp_decay_case = ThreePortalDecay(
    nu_parent     = existing_dec_case.nu_parent,
    nu_daughter   = existing_dec_case.nu_daughter,
    final_lepton1 = existing_dec_case.secondaries[0],
    final_lepton2 = existing_dec_case.secondaries[1],
    TheoryModel   = tp_model,
)

tp_siren_decay = PyDarkNewsDecay(tp_decay_case)

# Load decay tables if exist
decay_table_dir = os.path.join(table_dir, "decay_ThreePortal_ee")
os.makedirs(decay_table_dir, exist_ok=True)
tp_siren_decay.load_from_table(decay_table_dir)

secondary_processes[N4_type] = [tp_siren_decay]

# ---------------------------------------------------------------------------
# Compute decay width
# ---------------------------------------------------------------------------
DN_min_decay_width = np.inf

for secondary_type, decays in secondary_processes.items():
    for decay in decays:
        is_decay = False
        for signature in decay.GetPossibleSignatures():
            is_decay |= (
                signature.target_type ==
                siren.dataclasses.Particle.ParticleType.Decay
            )
        if not is_decay:
            continue

        decay_width = decay.TotalDecayWidth(secondary_type)
        DN_min_decay_width = min(DN_min_decay_width, decay_width)

assert DN_min_decay_width < np.inf

# ---------------------------------------------------------------------------
# Primary distributions (UNCHANGED MiniBooNE)
# ---------------------------------------------------------------------------
mass_ddist = siren.distributions.PrimaryMass(0)

edist = siren.utilities.load_flux("BNB", tag="FHC_numu", physically_normalized=True)
edist_gen = siren.utilities.load_flux(
    "BNB",
    tag="FHC_numu",
    min_energy=model_kwargs["m4"],
    max_energy=10,
    physically_normalized=False,
)

direction_distribution = siren.distributions.FixedDirection(
    siren.math.Vector3D(0, 0, 1.0)
)

decay_range_func = siren.distributions.DecayRangeFunction(
    model_kwargs["m4"], DN_min_decay_width, 3, 541
)

position_distribution = siren.distributions.DecayRangePositionDistribution(
    6.2,
    6.2,
    decay_range_func,
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
# Secondary distributions
# ---------------------------------------------------------------------------
fiducial_volume = utilities.get_fiducial_volume(experiment)

secondary_injection_distributions = {}
for secondary_type in secondary_processes.keys():
    secondary_injection_distributions[secondary_type] = [
        siren.distributions.SecondaryBoundedVertexDistribution(fiducial_volume)
    ]

# ---------------------------------------------------------------------------
# Stop condition
# ---------------------------------------------------------------------------
def stop(datum, i):
    secondary_type = datum.record.signature.secondary_types[i]
    return secondary_type != siren.dataclasses.Particle.ParticleType.N4

# ---------------------------------------------------------------------------
# Injector
# ---------------------------------------------------------------------------
injector = siren.injection.Injector()
injector.number_of_events = events_to_inject
injector.detector_model = detector_model
injector.primary_type = primary_type
injector.primary_interactions = primary_processes[primary_type]
injector.primary_injection_distributions = primary_injection_distributions
injector.secondary_interactions = secondary_processes
injector.secondary_injection_distributions = secondary_injection_distributions
injector.stopping_condition = stop

# ---------------------------------------------------------------------------
# Generate events
# ---------------------------------------------------------------------------
print("[INFO] Generating events...")
events, gen_times = GenerateEvents(injector)

# ---------------------------------------------------------------------------
# Save output
# ---------------------------------------------------------------------------
os.makedirs("output", exist_ok=True)

weighter = siren.injection.Weighter()
weighter.injectors = [injector]
weighter.detector_model = detector_model
weighter.primary_type = primary_type
weighter.primary_interactions = primary_processes[primary_type]
weighter.secondary_interactions = secondary_processes
weighter.primary_physical_distributions = primary_physical_distributions
weighter.secondary_physical_distributions = {}

SaveEvents(
    events,
    weighter,
    gen_times,
    output_filename="output/MiniBooNE_ThreePortal",
)

# Save tables
tp_siren_decay.save_to_table(decay_table_dir)

SaveDarkNewsProcesses(
    table_dir,
    primary_processes,
    primary_ups_keys,
    secondary_processes,
    secondary_dec_keys,
)

weights = [weighter(event) for event in events]

print("[INFO] Done.")
