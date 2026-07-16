"""
Dutta-Kim vector-portal K+ chain at ICARUS from the REAL NuMI beam
simulation (g4numi dk2nu), in spec form.

On-shell chain (5 vertices):

    K+ -> e+ nu_e V1;   V1 -> chi chi;   chi Ar -> chi' Ar;
    chi' -> chi V1_sig;   V1_sig -> e+ e-

The kaon kinematics come from g4numi dk2nu files whose vertices and
momenta are expressed in NuMI beam coordinates (origin at MCZERO, z along
the NuMI axis).  They are mapped into the SIREN world frame (BNB) with
sbn_geometry.transform("NuMI", "BNB") — the ICARUS-NuMI rotation and
translation from icaruscode GNuMIFlux.xml, confirmed by SBN DocDB
22998-v2 — through the Dk2nuReader beam_transform hook, and injected via
PrimaryExternalDistribution through the composite SBN geometry
(BNB + NuMI beamlines + ICARUS detector, all in BNB coordinates).

Flux files (g4numi, medium-energy target, RHC / antineutrino mode,
1e6 POT and ~15k K+ decays per file):
  g4numiNone_downstream_me000z-200i_rhc_2001-2005.root
Set NUMI_DK2NU_GLOB to point elsewhere.

Usage:
    python VectorPortal_ICARUS_NuMI_dk2nu.py [--events N] [--seed S]
        [--n-files K] [--tune]
"""

import argparse
import glob
import os

import numpy as np

import siren
from siren import _util, channels, dataclasses, distributions, expand
from siren.Injector import Injector
from siren.Weighter import Weighter

_PROC = os.path.join(_util.resource_package_dir(), "processes", "DarkNewsTables")
_MESON = _util.load_module("DuttaKim_MesonProduction",
                           os.path.join(_PROC, "MesonProduction.py"))
_VP = _util.load_module("DuttaKim_VectorPortal",
                        os.path.join(_PROC, "VectorPortal.py"))
_DK = _util.load_module("DuttaKim_Dk2nuReader",
                        os.path.join(_PROC, "Dk2nuReader.py"))
_GEO = _util.load_module("sbn_geometry",
                         os.path.join(_util.resource_package_dir(),
                                      "detectors", "SBN", "SBN-v1",
                                      "sbn_geometry.py"))

# ---------------------------------------------------------------------- #
#  Model parameters (Dutta-Kim Table I/II, double-mediator)                #
# ---------------------------------------------------------------------- #
M_KAON = 0.49368
M_ELEC = 0.000511
M_CHI, M_CHI_PRIME = 8e-3, 50e-3
M_V1, M_V2 = 17e-3, 200e-3
M_ARGON40 = 37.215
G_D, EPSILON_1, EPSILON_2, G_MU = 1.0, 7e-5, 1e-4, 1e-3

# PDG K+ total width (tau = 1.238e-8 s) for the BSM branching ratio.
GAMMA_KAON_SM = 5.3167e-17

KAON = dataclasses.Particle.ParticleType(321)
V1_PROD = siren.particles.define("V1_prod", 5922, M_V1)
CHI = siren.particles.define("chi", 5917, M_CHI)
CHI_PRIME = siren.particles.define("chi_prime", 5918, M_CHI_PRIME)
V1_SIG = siren.particles.define("V1_sig", 5923, M_V1)

# ---------------------------------------------------------------------- #
#  NuMI flux                                                               #
# ---------------------------------------------------------------------- #
_EX4 = os.path.dirname(os.path.realpath(__file__))
NUMI_GLOB = os.environ.get(
    "NUMI_DK2NU_GLOB", os.path.join(_EX4, "sources", "NuMI", "g4numi*.root"))

# NuMI beam frame -> BNB (SIREN world frame).
T_NUMI_TO_BNB = _GEO.transform("NuMI", "BNB")


def build_onshell_models():
    """Physics models for the on-shell K+ chain (vector production)."""
    kaon_decay = _MESON.MesonThreeBodySIRENDecay(
        M_KAON, M_ELEC, M_V1, G_MU, "vector",
        pdgid_meson=321, pdgid_lepton=-11,
        pdgid_neutrino=12, pdgid_mediator=5922)
    v1_to_chi = _VP.DarkPhotonToChiDecay(
        M_V1, M_CHI, G_D, pdgid_V1=5922, pdgid_chi=5917)
    upscatter = _VP.VectorPortalUpscatteringXS(
        M_CHI, M_CHI_PRIME, M_V2, G_D, EPSILON_2,
        pdgid_chi=5917, pdgid_chi_prime=5918,
        nuclear_pdgid=1000180400, nuclear_mass=M_ARGON40, A=40, Z=18)
    chi_prime_decay = _VP.ChiPrimeDecay(
        M_CHI, M_CHI_PRIME, M_V1, G_D,
        pdgid_chi_prime=5918, pdgid_chi=5917, pdgid_V1=5923)
    visible_decay = _VP.DarkPhotonDecay(M_V1, EPSILON_1, pdgid_V1=5923)
    return {
        "kaon_decay": kaon_decay,
        "models": {
            "v1_to_chi": v1_to_chi,
            "upscatter": upscatter,
            "chi_prime_decay": chi_prime_decay,
            "visible_decay": visible_decay,
        },
    }


def build_sX_cdf(meson_decay, n_nodes=257):
    """CDF of the physical s_X = M^2(l, nu) marginal for the directed
    primary channel's pair-mass proposal (exact for any masses)."""
    d = meson_decay._decay
    s_min, s_max = d.m_l ** 2, (d.m_M - d.m_phi) ** 2
    eps = 1e-9 * (s_max - s_min)
    s = np.linspace(s_min + eps, s_max - eps, n_nodes)
    E_V1 = (d.m_M ** 2 + d.m_phi ** 2 - s) / (2.0 * d.m_M)
    dens = np.clip(np.asarray(d.differential_decay_rate(E_V1), float), 0.0, None)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (dens[1:] + dens[:-1]) * np.diff(s))])
    return s.tolist(), cdf.tolist()


def numi_kaon_bias(sigma_fn=None):
    """Importance shape for the dk2nu K+ sample, evaluated in the RAW NuMI
    beam frame (before beam_transform): forward along the NuMI axis, times
    the upscatter cross-section trend when available."""
    def bias(E, px, py, pz, vx, vy, vz):
        p = np.sqrt(px ** 2 + py ** 2 + pz ** 2)
        cos_theta = np.divide(pz, p, out=np.zeros_like(p), where=(p > 0))
        fwd = np.maximum(cos_theta, 0.0)
        if sigma_fn is None:
            return fwd
        return sigma_fn(np.asarray(E, dtype=float)) * fwd
    return bias


def build_sigma_interp(upscatter, e_lo=0.06, e_hi=10.0, n=120):
    grid = np.linspace(e_lo, e_hi, n)
    sg = np.array([upscatter._ups.total_xsec(float(e)) for e in grid])
    smax = float(np.max(sg))
    if smax > 0:
        sg = sg / smax

    def sigma_fn(E):
        return np.interp(np.clip(E, e_lo, e_hi), grid, sg)
    return sigma_fn


def build_vertices(models, fiducial, meson_dist, br_bsm):
    """The chain as spec-form Vertex objects with channels algebra.

    The primary kaon comes from the dk2nu external distribution (position
    and 4-momentum fixed by the beam simulation, already transformed into
    the world frame), so the primary Vertex carries no synthetic energy/
    direction/position distributions and uses Fixed vertex weighting."""
    m = models["models"]
    sx = channels.PairMass.tabulated(*build_sX_cdf(models["kaon_decay"]))
    sv = distributions.SecondaryPhysicalVertexDistribution

    primary = siren.Vertex(
        KAON, models["kaon_decay"],
        distributions=[meson_dist],
        physical=[meson_dist,
                  distributions.NormalizationConstant(br_bsm)],
        weighting=siren.Fixed(),
        kinematics=0.98 * channels.toward_3body("V1_prod", fiducial,
                                                strategy="direct", pair_mass=sx)
                   + 0.02 * channels.physical(),
        expand=(expand.child("V1_prod"),))

    v1 = siren.Vertex(
        "V1_prod", m["v1_to_chi"], position=sv(),
        kinematics=0.98 * channels.toward(0, fiducial)
                   + 0.02 * channels.physical(),
        expand=(expand.child("chi", index=0),))

    chi = siren.Vertex(
        "chi", m["upscatter"],
        position=distributions.SecondaryBoundedVertexDistribution(fiducial),
        kinematics=channels.physical(),
        expand=(expand.child("chi_prime"),))

    chip = siren.Vertex(
        "chi_prime", m["chi_prime_decay"], position=sv(),
        kinematics=channels.physical(),
        expand=(expand.child("V1_sig"),))

    v1s = siren.Vertex(
        "V1_sig", m["visible_decay"], position=sv(),
        kinematics=0.5 * channels.physical() + 0.5 * channels.isotropic(0),
        expand=(expand.depth_below(0),))  # terminal: e+ e- are final

    return primary, (v1, chi, chip, v1s)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--events", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-files", type=int, default=1,
                    help="how many g4numi files to read")
    ap.add_argument("--tune", action="store_true",
                    help="Kleiss-Pittau channel-weight tuning before production")
    args = ap.parse_args()

    files = sorted(glob.glob(NUMI_GLOB))[:args.n_files]
    if not files:
        raise SystemExit("No NuMI dk2nu files match %s" % NUMI_GLOB)

    c_num = _GEO.detector_center("ICARUS", "NuMI")
    print("NuMI -> BNB transform (GNuMIFlux.xml / DocDB 22998-v2):")
    print("  MCZERO in BNB frame : %s m" % np.round(T_NUMI_TO_BNB.t, 3))
    print("  ICARUS in NuMI frame: %s m  (baseline %.1f m, off-axis %.2f deg)"
          % (np.round(c_num, 2), np.linalg.norm(c_num),
             np.degrees(np.arccos(c_num[2] / np.linalg.norm(c_num)))))

    print("\nLoading composite ICARUS model (BNB + NuMI beamlines + detector) ...")
    detector = siren.utilities.load_detector("SBN", detector="ICARUS")

    # ICARUS active LAr envelope (sbn_geometry.DETECTORS["ICARUS"]):
    # x +/-3.60 m, y +/-1.58 m, z +/-8.975 m about the detector origin.
    fiducial = siren.geometry.Box(widths=(7.20, 3.16, 17.95))

    print("Reading NuMI dk2nu: %d file(s)" % len(files))
    dk2nu_data = _DK.read_dk2nu(files, parent_pdg=[321])
    _DK.print_summary(dk2nu_data)

    models = build_onshell_models()
    br_bsm = models["kaon_decay"]._total_width / GAMMA_KAON_SM
    print("  BSM 3-body width: %.4e GeV   BR(K+ -> e nu V1): %.4e"
          % (models["kaon_decay"]._total_width, br_bsm))

    sigma_fn = build_sigma_interp(models["models"]["upscatter"])
    meson_dist = _DK.dk2nu_to_primary_distribution(
        dk2nu_data, detector, parent_pdg=[321],
        sampling_bias=numi_kaon_bias(sigma_fn),
        flux_weighted_sampling=True,
        beam_transform=T_NUMI_TO_BNB)
    print("  Loaded %d kaon entries into PrimaryExternalDistribution"
          % meson_dist.GetPhysicalNumEvents())

    primary, secondaries = build_vertices(models, fiducial, meson_dist, br_bsm)

    injector = Injector(detector=detector, primary=primary,
                        secondaries=secondaries,
                        events=args.events, seed=args.seed)
    weighter = Weighter(injector, primary_physical=primary.physical)

    if args.tune:
        print(siren.tune.tune(injector, weighter, events=200, rounds=3))
        injector.reset()

    results = siren.generate(injector, weighter,
                             events=args.events, on_shortfall="warn")
    results.summary()
    print(results.variance_report())
    print(injector.report())


if __name__ == "__main__":
    main()
