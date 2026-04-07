import os
import siren
from siren._util import GenerateEvents, SaveEvents

# -------------------------------
# Number of events
# -------------------------------
events_to_inject = int(1e5)

# -------------------------------
# Load SBND detector
# -------------------------------
base = "/home/subham-patra/SIREN/resources/detectors/SBND/SBNDv1"

detector_model = siren.detector.DetectorModel(
    os.path.join(base, "SBND-v1.dat"),
    os.path.join(base, "densities.dat"),
    os.path.join(base, "materials.dat"),
)

# -------------------------------
# Primary particle
# -------------------------------
primary_type = siren.dataclasses.Particle.ParticleType.NuMu

# -------------------------------
# Load DIS cross-section
# -------------------------------
cross_section_model = "CSMSDISSplines"

primary_processes, _ = siren.utilities.load_processes(
    cross_section_model,
    primary_types=[primary_type],
    target_types=[siren.dataclasses.Particle.ParticleType.Nucleon],
    isoscalar=True,
    process_types=["CC"]
)

primary_cross_sections = primary_processes[primary_type]

# -------------------------------
# Injector
# -------------------------------
injector = siren.injection.Injector()
injector.number_of_events = events_to_inject
injector.detector_model = detector_model
injector.primary_type = primary_type
injector.primary_interactions = primary_cross_sections

# -------------------------------
# Define BOX geometry for injection
# -------------------------------
center = siren.math.Vector3D(0.0, 0.0, 2.5)

# Identity quaternion (no rotation)
rotation = siren.math.Quaternion(1.0, 0.0, 0.0, 0.0)

placement = siren.geometry.Placement(center, rotation)

# SBND active volume box: 4m x 4m x 5m
box = siren.geometry.Box(placement, 4.0, 4.0, 5.0)

# -------------------------------
# Distributions (BOX injection)
# -------------------------------
injector.primary_injection_distributions = [
    siren.distributions.PrimaryMass(0),

    # Energy distribution
    siren.distributions.PowerLaw(1, 1e4, 1e7),

    # Beam direction
    siren.distributions.FixedDirection(
        siren.math.Vector3D(0, 0, 1)
    ),

    # True BOX vertex injection
    siren.distributions.BoxVolumePositionDistribution(box)
]

# -------------------------------
# Generate events
# -------------------------------
events, gen_times = GenerateEvents(injector)

num_interacting = sum(len(e.tree) > 0 for e in events)
num_total = len(events)

print(f"\nTotal events generated: {num_total}")
print(f"Events with interactions: {num_interacting}")
print(f"Interaction fraction: {num_interacting/num_total:.6f}")

# -------------------------------
# Weighter
# -------------------------------
weighter = siren.injection.Weighter()
weighter.injectors = [injector]
weighter.detector_model = detector_model
weighter.primary_type = primary_type
weighter.primary_interactions = primary_cross_sections

weighter.primary_physical_distributions = [
    siren.distributions.PowerLaw(1, 1e3, 1e6),
    siren.distributions.IsotropicDirection()
]

# -------------------------------
# Save events
# -------------------------------
os.makedirs("output", exist_ok=True)

print("Number of events:", len(events))

SaveEvents(events, weighter, gen_times,
           output_filename="output/UnidirectionalSBND_DIS_BOX")
