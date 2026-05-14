import os
import numpy as np

import siren
from siren import utilities
from siren._util import GenerateEvents, SaveEvents, get_processes_model_path

SaveDarkNewsProcesses = siren.resources.processes.DarkNewsTables.SaveDarkNewsProcesses

from siren.SIREN_DarkNews import PyDarkNewsCrossSection, VectorPortalUpsCase
from DarkNews import phase_space
from DarkNews.processes import ChiPrimeDecay, DarkPhotonDecay

M_CHI        = 8e-3      # GeV  χ   dark matter ground state
M_CHI_PRIME  = 50e-3     # GeV  χ'  dark matter excited state
M_V1         = 17e-3     # GeV  V₁  light dark photon
M_V2         = 200e-3    # GeV  V₂  heavy upscattering mediator
G_D          = 1.0
EPSILON_1    = 7e-5
EPSILON_2    = 1e-4

PDGID_CHI       = 5917
PDGID_CHI_PRIME = 5918
PDGID_V1        = 5922

NUCLEAR_PDGID    = 1000060120
NUCLEAR_MASS_GEV = 11.178
NUCLEAR_NAME     = "C12"
NUCLEAR_A        = 12

events_to_inject = 100_000
experiment       = "MiniBooNE"


M_LEPTON    = 0.000511    # GeV — 0.10566 for μ, 0.000511 for e
M_MESON     = 0.13957     # GeV — 0.13957 for π, 0.49368 for K
flux_tag    = "pion_nue"     # pion_numu / pion_nue / kaon_numu / kaon_nue
output_stem = "output/MiniBooNE_Vectorportal_pi_e"

detector_model = utilities.load_detector(experiment)
primary_type   = siren.dataclasses.Particle.ParticleType(PDGID_CHI)
chi_prime_type = siren.dataclasses.Particle.ParticleType(PDGID_CHI_PRIME)
v1_type        = siren.dataclasses.Particle.ParticleType(PDGID_V1)

table_name  = f"DarkNewsTables-v{siren.utilities.darknews_version()}/"
table_name += "VectorPortal_mchi%2.2e_mV1%2.2e_mV2%2.2e" % (M_CHI, M_V1, M_V2)
table_dir   = os.path.join(get_processes_model_path("DarkNewsTables"), table_name)
os.makedirs(table_dir, exist_ok=True)


def compute_chi_flux(flux_tag, m_meson, m_lepton, m_V1, min_energy, max_energy,
                     physically_normalized):

    raw_flux = siren.utilities.load_flux(
        "PionKaon",
        tag                   = flux_tag,
        physically_normalized = physically_normalized,
    )
    meson_energies = list(raw_flux.GetEnergyNodes())

 
    available = m_meson - m_lepton
    if available <= m_V1:
        raise RuntimeError(
            f"Channel kinematically forbidden!\n"
            f"  m_meson={m_meson*1e3:.1f} MeV, m_lepton={m_lepton*1e3:.1f} MeV, "
            f"m_V1={m_V1*1e3:.1f} MeV\n"
            f"  Need m_meson > m_lepton + m_V1"
        )

    E_V_rest = (m_meson**2 + m_V1**2 - m_lepton**2) / (2.0 * m_meson)

    chi_energies = []
    chi_flux_vals = []

    for E_meson in meson_energies:
        if E_meson < m_meson:
            continue

        # Lorentz boost factor of meson in lab frame
        gamma_meson = E_meson / m_meson

        # V₁ energy in lab frame (forward boost approximation)
        E_V_lab = gamma_meson * E_V_rest

        # χ energy: each χ gets half of V₁ in V₁ rest frame, boosted to lab
        E_chi = E_V_lab * 0.5

        # Apply energy cuts
        if E_chi < min_energy or E_chi > max_energy:
            continue

        # Get flux value at this meson energy
        flux_val = raw_flux.SamplePDF(E_meson)

        chi_energies.append(E_chi)
        chi_flux_vals.append(flux_val)

    if len(chi_energies) == 0:
        raise RuntimeError(
            f"No χ energies in range [{min_energy*1e3:.1f}, {max_energy*1e3:.1f}] MeV!\n"
            f"  E_V_rest = {E_V_rest*1e3:.1f} MeV → check mass parameters."
        )

    idx          = np.argsort(chi_energies)
    chi_energies  = np.array(chi_energies)[idx].tolist()
    chi_flux_vals = np.array(chi_flux_vals)[idx].tolist()

    return chi_energies, chi_flux_vals

ups_case = VectorPortalUpsCase(
    m_chi           = M_CHI,
    m_chi_prime     = M_CHI_PRIME,
    m_V             = M_V2,
    g_D             = G_D,
    epsilon         = EPSILON_2,
    pdgid_chi       = PDGID_CHI,
    pdgid_chi_prime = PDGID_CHI_PRIME,
    nuclear_pdgid   = NUCLEAR_PDGID,
    nuclear_mass    = NUCLEAR_MASS_GEV,
    nuclear_name    = NUCLEAR_NAME,
    A               = NUCLEAR_A,
)
E_thresh = ups_case.Ethreshold
print("χ upscattering threshold (from ups_case): %.2f MeV" % (E_thresh * 1e3))


print("Computing χ flux for channel: %s  (m_lepton=%.2f MeV) ..." % (
    flux_tag, M_LEPTON * 1e3))

chi_E_phys, chi_F_phys = compute_chi_flux(
    flux_tag, M_MESON, M_LEPTON, M_V1,
    min_energy            = E_thresh,
    max_energy            = 3.0,
    physically_normalized = True,
)

chi_E_gen, chi_F_gen = compute_chi_flux(
    flux_tag, M_MESON, M_LEPTON, M_V1,
    min_energy            = E_thresh,
    max_energy            = 3.0,
    physically_normalized = False,
)

# Build TabulatedFluxDistribution directly from χ energies
edist_phys = siren.distributions.TabulatedFluxDistribution(
    E_thresh, 3.0, chi_E_phys, chi_F_phys, True
)
edist_gen = siren.distributions.TabulatedFluxDistribution(
    E_thresh, 3.0, chi_E_gen, chi_F_gen, False
)

print("χ flux built: %d energy bins  E ∈ [%.2f MeV, %.2f GeV]" % (
    len(chi_E_phys), min(chi_E_phys)*1e3, max(chi_E_phys)))

xs_table_dir = os.path.join(table_dir, "CrossSection_chi_C12")
xs = PyDarkNewsCrossSection(
    ups_case,
    table_dir          = xs_table_dir,
    tolerance          = 1e-6,
    interp_tolerance   = 5e-2,
    always_interpolate = True,
)

primary_processes = {primary_type: [xs]}
primary_ups_keys  = {primary_type: [[xs.ups_case.nuclear_target]]}

print("Building secondary decays ...")
chi_prime_decay = ChiPrimeDecay(
    m_chi           = M_CHI,
    m_chi_prime     = M_CHI_PRIME,
    m_V1            = M_V1,
    g_D             = G_D,
    pdgid_chi_prime = PDGID_CHI_PRIME,
    pdgid_chi       = PDGID_CHI,
    pdgid_V1        = PDGID_V1,
    table_dir       = os.path.join(table_dir, "Decay_ChiPrime"),
)

dark_photon_decay = DarkPhotonDecay(
    m_V1      = M_V1,
    epsilon   = EPSILON_1,
    pdgid_V1  = PDGID_V1,
    table_dir = os.path.join(table_dir, "Decay_V1"),
)

secondary_processes = {
    chi_prime_type: [chi_prime_decay],
    v1_type:        [dark_photon_decay],
}
secondary_dec_keys = {k: [[k]] for k in secondary_processes.keys()}

DN_min_decay_width = np.inf
for sec_type, decays in secondary_processes.items():
    for decay in decays:
        is_decay = any(
            sig.target_type == siren.dataclasses.Particle.ParticleType.Decay
            for sig in decay.GetPossibleSignatures()
        )
        if not is_decay:
            continue
        DN_min_decay_width = min(DN_min_decay_width, decay.TotalDecayWidth(sec_type))

print("Minimum decay width: %.3e GeV" % DN_min_decay_width)
assert DN_min_decay_width < np.inf, (
    "No valid decay widths found. Check mass hierarchy:\n"
    "  Need m_χ' > m_χ + m_V₁  and  m_V₁ > 2 m_e"
)

mass_ddist             = siren.distributions.PrimaryMass(M_CHI)
direction_distribution = siren.distributions.FixedDirection(
    siren.math.Vector3D(0, 0, 1.0)
)
fiducial_volume        = utilities.get_fiducial_volume(experiment)
position_distribution  = siren.distributions.ColumnDepthPositionDistribution(
    6.2, 6.2, siren.distributions.LeptonDepthFunction()
)

primary_injection_distributions = [
    mass_ddist,
    edist_gen,
    direction_distribution,
    position_distribution,
]
primary_physical_distributions = [
    edist_phys,
    direction_distribution,
]

secondary_injection_distributions = {
    sec_type: [siren.distributions.SecondaryBoundedVertexDistribution(fiducial_volume)]
    for sec_type in secondary_processes
}

def stop(datum, i):
    sec_type = datum.record.signature.secondary_types[i]
    return sec_type not in [chi_prime_type, v1_type]


injector = siren.injection.Injector()
injector.number_of_events                  = events_to_inject
injector.detector_model                    = detector_model
injector.primary_type                      = primary_type
injector.primary_interactions              = primary_processes[primary_type]
injector.primary_injection_distributions   = primary_injection_distributions
injector.secondary_interactions            = secondary_processes
injector.secondary_injection_distributions = secondary_injection_distributions
injector.stopping_condition                = stop


print("Generating %d events ..." % events_to_inject)
events, gen_times = GenerateEvents(injector)
print("Generated %d event trees." % len(events))

os.makedirs("output", exist_ok=True)

weighter = siren.injection.Weighter()
weighter.injectors                        = [injector]
weighter.detector_model                   = detector_model
weighter.primary_type                     = primary_type
weighter.primary_interactions             = primary_processes[primary_type]
weighter.secondary_interactions           = secondary_processes
weighter.primary_physical_distributions   = primary_physical_distributions
weighter.secondary_physical_distributions = {}

SaveEvents(
    events, weighter, gen_times,
    fid_vol         = fiducial_volume,
    output_filename = output_stem,
)

xs.SaveInterpolationTables()
SaveDarkNewsProcesses(
    table_dir,
    primary_processes,   primary_ups_keys,
    secondary_processes, secondary_dec_keys,
)

weights     = np.array([weighter(ev) for ev in events])
finite_mask = np.isfinite(weights) & (weights > 0)

print("Events generated        :", len(events))
print("Finite positive weights :", finite_mask.sum(), "/", len(weights))
if finite_mask.any():
    print("Weight range            : %.3e – %.3e"
          % (weights[finite_mask].min(), weights[finite_mask].max()))
    print("Total expected signal   : %.3e events" % weights[finite_mask].sum())
print("Output →", output_stem + ".*")
