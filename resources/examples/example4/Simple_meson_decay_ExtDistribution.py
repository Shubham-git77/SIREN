import os
import numpy as np

import siren
from siren import utilities
from siren._util import GenerateEvents, SaveEvents
from siren.dataclasses import Particle

from DarkNews import pdg as dn_pdg
from DarkNews.processes import MesonSimpleDecay

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
events_to_inject = 10_000
experiment       = "MiniBooNE"
CSV_FILE         = "/home/shubham/SIREN_latest/resources/fluxes/Pions/Pions-v1.0/pion_for_siren.csv"
output_stem      = "output/MiniBooNE_pion_decay"

PDGID_PION = 211    # pi+
PDGID_MUON = -13    # mu+
PDGID_NUMU =  14    # nu_mu

# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
detector_model = utilities.load_detector(experiment)
pion_type      = Particle.ParticleType(PDGID_PION)
muon_type      = Particle.ParticleType(PDGID_MUON)
nu_type        = Particle.ParticleType(PDGID_NUMU)

# ---------------------------------------------------------------------------
# Decay process  (pi+ -> mu+ + nu_mu)
# MesonSimpleDecay now inherits DarkNewsDecay directly and implements
# SampleFinalState analytically — no VEGAS, no PyDarkNewsDecay wrapper,
# no table_dir needed.
# ---------------------------------------------------------------------------
decay = MesonSimpleDecay(
    nu_parent   = dn_pdg.piplus,
    nu_daughter = dn_pdg.numu,
)

primary_processes = {pion_type: [decay]}

# ---------------------------------------------------------------------------
# PrimaryExternalDistribution reads kinematics from CSV.
# Columns: E, px, py, pz, x0, y0, z0, m  (GeV / cm)
# ---------------------------------------------------------------------------
primary_dist = siren.distributions.PrimaryExternalDistribution(CSV_FILE)

# ---------------------------------------------------------------------------
# Fiducial volume
# ---------------------------------------------------------------------------
fiducial_volume = utilities.get_fiducial_volume(experiment)

primary_injection_distributions = [primary_dist]
primary_physical_distributions  = [primary_dist]

secondary_processes                = {}
secondary_injection_distributions  = {}

def stop(datum, i):
    # mu+ and nu_mu are stable final states — stop after first decay
    return True

# ---------------------------------------------------------------------------
# Injector
# ---------------------------------------------------------------------------
injector = siren.injection.Injector()
injector.number_of_events                  = events_to_inject
injector.detector_model                    = detector_model
injector.primary_type                      = pion_type
injector.primary_interactions              = primary_processes[pion_type]
injector.primary_injection_distributions   = primary_injection_distributions
injector.secondary_interactions            = secondary_processes
injector.secondary_injection_distributions = secondary_injection_distributions
injector.stopping_condition                = stop

print("Generating %d events ..." % events_to_inject)
events, gen_times = GenerateEvents(injector)
print("Generated %d event trees." % len(events))

os.makedirs("output", exist_ok=True)

# ---------------------------------------------------------------------------
# Weighter
# ---------------------------------------------------------------------------
weighter = siren.injection.Weighter()
weighter.injectors                        = [injector]
weighter.detector_model                   = detector_model
weighter.primary_type                     = pion_type
weighter.primary_interactions             = primary_processes[pion_type]
weighter.secondary_interactions           = secondary_processes
weighter.primary_physical_distributions   = primary_physical_distributions
weighter.secondary_physical_distributions = {}

SaveEvents(
    events, weighter, gen_times,
    fid_vol         = fiducial_volume,
    output_filename = output_stem,
)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
weights     = np.array([weighter(ev) for ev in events])
finite_mask = np.isfinite(weights) & (weights > 0)

print("Events generated        :", len(events))
print("Finite positive weights :", finite_mask.sum(), "/", len(weights))
if finite_mask.any():
    print("Weight range            : %.3e - %.3e"
          % (weights[finite_mask].min(), weights[finite_mask].max()))
    print("Total expected signal   : %.3e events" % weights[finite_mask].sum())
print("Output ->", output_stem + ".*")
