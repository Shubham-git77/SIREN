"""
Full Dutta-Kim vector-portal chain at ICARUS (LAr)  —  MULTICHANNEL sum.

Adapted from DuttaKim_SBND_full_chain.py to:
  - sum the four charged-meson channels K/pi x e/mu (Eq.3, arXiv:2110.11944),
  - use the VALIDATED vector three-body production (mediator_type="vector",
    C-R Eq.27, Table II anchored) instead of the scalar default,
  - read both pi+ (211) and K+ (321) from dk2nu per channel,
  - target ICARUS argon (Ar40),
  - apply ICARUS cuts (volTPCActive sector + 140 MeV + 10deg) and sum observables.

Architecture (meson-as-primary, the SBND pattern):
  primary = meson, primary_interactions = [meson_decay(vector)],
  production happens IN-SIM; BSM branching ratio added via NormalizationConstant
  with VertexWeightingMode.Fixed (dk2nu pions already SM-decayed).

Per-channel verification: run_channel() runs and returns observables for one
channel; the bottom loops all four and sums. Run with --channel K_e (etc.) to
test a single channel first, or no flag to run all four.
"""

import argparse
import glob
import math
import os
import sys
import numpy as np

import siren
from siren import _util, dataclasses, distributions, injection
from siren.Injector import Injector
from siren.Weighter import Weighter

# ------------------------------------------------------------------ #
#  Load physics modules (validated MesonProduction + VectorPortal)     #
# ------------------------------------------------------------------ #
_PROC_DIR = os.path.join(_util.resource_package_dir(), "processes", "DarkNewsTables")
_MESON = _util.load_module("DuttaKim_MesonProduction",
                           os.path.join(_PROC_DIR, "MesonProduction.py"))
_VP = _util.load_module("DuttaKim_VectorPortal",
                        os.path.join(_PROC_DIR, "VectorPortal.py"))
_DK = _util.load_module("DuttaKim_Dk2nuReader",
                        os.path.join(_PROC_DIR, "Dk2nuReader.py"))

# Validated absolute-normalization constant (bridges C-R production to Table II)
CALIB_VECTOR = getattr(_MESON, "CALIB_VECTOR", 2412.0)

# ------------------------------------------------------------------ #
#  Constants                                                           #
# ------------------------------------------------------------------ #
M_PION = 0.13957039
M_KAON = 0.49368
M_MUON = 0.10565837
M_ELEC = 0.000511

# SM two-body total widths [GeV] for the BSM branching ratio denominator.
GAMMA_PION_SM = 2.5281e-17   # PDG pi+ total width
GAMMA_KAON_SM = 5.3167e-17   # PDG K+ total width (tau = 1.238e-8 s)

M_CHI = 8e-3
M_CHI_PRIME = 50e-3
M_V1 = 17e-3
M_V2 = 200e-3
M_ARGON40 = 37.224           # GeV, Ar40 nuclear mass (ICARUS liquid argon)

# Dutta-Kim Table II benchmark (double-mediator scenario, arXiv:2110.11944):
#   epsilon_1 = 7e-5   (V1 kinetic mixing: production + visible decay)
#   epsilon_2 = 1e-4   (V2 kinetic mixing: upscattering)
#   g'_2^2/(4pi) = 0.5 (dark gauge coupling of V2 to chi-chi')
#   -> G_D = sqrt(4*pi*0.5) = sqrt(2*pi)
G_D = math.sqrt(2.0 * math.pi)   # g'_2^2/(4pi) = 0.5
EPSILON_1 = 7e-5
EPSILON_2 = 1e-4
G_MU = EPSILON_1             # production coupling slot = kinetic mixing (vector)

PT = lambda pdg: dataclasses.Particle.ParticleType(pdg)
V1_PROD = PT(5922)
CHI = PT(5917)
CHI_PRIME = PT(5918)
V1_SIGNAL = PT(5923)

# ICARUS LAr geometry (active-volume cut via GDML volTPCActive sectors)
# Injection geometry: sphere/box enclosing all 8 volTPCActive sectors.
# In detector coordinates (z-offset of 600 m subtracted from GDML world):
#   TPC x-extent: [-4.32, +4.32] m   y: [-3.40, +2.94] m   z: [-13.42, +13.42] m
#   Farthest corner from origin: 14.51 m  -> R_LAR_INJECT must be > 14.51 m.
# The old value of 12.0 m was too small (missed TPC z-corners), causing the
# chi bounded vertex distribution to sample only ~83% of the active z-range.
# ICARUS is TWO separate cryostats (C0, C1) with a ~1.2 m argon-free gap, NOT
# one monolithic box. _TPC_BOX is ONE active module (3.0 x 3.16 x 17.95 m from
# sbn_geometry ICARUS_C0/C1); the analytic engine is called once per module
# center (ICARUS_MODULE_CENTERS_BNB) and the yields are summed. The old single
# 8.64 x 6.34 x 26.84 m box (1469 m^3) was the warm-vessel envelope -> 2051 t of
# phantom argon (4.3x the true 476 t across both modules), inflating every rate.
R_LAR_INJECT = 10.0  # encloses one module's far corner (9.23 m)
_TPC_BOX_X = 3.00    # single-module x-extent [m]  (covers ±1.50 m)
_TPC_BOX_Y = 3.16    # single-module y-extent [m]
_TPC_BOX_Z = 17.95   # single-module z-extent [m]  (covers ±8.975 m)
# Two active-module centers in the BNB frame [m] (sbn_geometry ICARUS_C0/C1).
ICARUS_MODULE_CENTERS_BNB = [(-2.10215, -0.202, 600.0), (2.10215, -0.202, 600.0)]
LAR_TARGET_PDGS = {1000180400}               # Ar40 (upscatter target in LAr)
events_to_inject = 10_000

# Reconstruction cuts
E_VIS_THRESHOLD = 0.140
E_PAIR_MAX_DEG = 10.0

# ---- Absolute normalization ----
# CALIB_VECTOR: the validated overall constant bridging the C-R production
# width to Dutta-Kim Table II (imported from MesonProduction). Applied once
# per event weight so the production rate is on the Table II scale.
# (Loaded after _MESON import below.)
#
# ICARUS NuMI exposure (SBN programme nominal, neutrino mode).
ICARUS_POT = 6e20
#
# Energy-dependent detection efficiency eps(E_vis) from refs [76,77].
# Digitize and fill as [[E_GeV, eff], ...]; None -> efficiency 1.0 (a flat
# efficiency will NOT reproduce the exact shape; this is a placeholder).
_EFF_TABLE = None

def detection_efficiency(E_vis_gev):
    if _EFF_TABLE is None:
        return 1.0
    E = np.asarray(_EFF_TABLE)[:, 0]; eff = np.asarray(_EFF_TABLE)[:, 1]
    return float(np.interp(E_vis_gev, E, eff, left=eff[0], right=eff[-1]))

DK2NU_FILE = os.environ.get("DK2NU_FILE", "/home/shubham/nubeamHighSample.dk2nu.root")

# The four channels: name -> (parent_pdg, m_meson, m_lepton, lepton_pdg,
#                              nu_pdg, gamma_sm)
CHANNELS = {
    "K_e":   (321, M_KAON, M_ELEC, -11, 12, GAMMA_KAON_SM),
    "K_mu":  (321, M_KAON, M_MUON, -13, 14, GAMMA_KAON_SM),
    "pi_e":  (211, M_PION, M_ELEC, -11, 12, GAMMA_PION_SM),
    "pi_mu": (211, M_PION, M_MUON, -13, 14, GAMMA_PION_SM),
}


def _mc(channels, weights):
    m = injection.MultiChannelPhaseSpace()
    m.channels = channels
    m.weights = weights
    return m


def make_meson_bias(sigma_fn=None):
    """Build the parent-meson importance shape used with flux_weighted_sampling.

    The realized sampling weight is  nimpwt * bias  -- i.e. sample PROPORTIONAL
    TO PHYSICAL RATE, focused forward where the boosted V1/chi can reach ICARUS.

    bias = sigma(E) * max(cos,0)   (candidate "C")
      sigma is the chi-upscatter total cross-section sigma(E_chi); the meson
      energy E proxies E_chi (looser than the scalar case because of the
      V1->chi chi split, but it still tracks the trend). Folding it in makes
      the sampling track the true per-event weight.  DO NOT reintroduce an E**n
      factor: dividing the flux weight by E**n (the old E**2*cos bias)
      manufactures a heavy weight tail; sigma already carries the correct
      energy dependence.

    If sigma_fn is None, fall back to the parameter-free forward shape max(cos,0).
    """
    def bias(E, px, py, pz, vx, vy, vz):
        p = np.sqrt(px**2 + py**2 + pz**2)
        cos_theta = np.divide(pz, p, out=np.zeros_like(p), where=(p > 0))
        fwd = np.maximum(cos_theta, 0.0)
        if sigma_fn is None:
            return fwd
        return sigma_fn(np.asarray(E, dtype=float)) * fwd
    return bias


# ------------------------------------------------------------------ #
#  Channel-parameterized model building (VECTOR production)            #
# ------------------------------------------------------------------ #
def build_onshell_models(parent_pdg, m_meson, m_lepton, lepton_pdg, nu_pdg):
    """On-shell chain models for a given production channel, with the
    VALIDATED vector three-body production (mediator_type='vector')."""
    meson_decay = _MESON.MesonThreeBodySIRENDecay(
        m_meson, m_lepton, M_V1, G_MU, "vector",   # positional: m_meson, m_lepton, m_mediator, g_mu, mediator_type
        pdgid_meson=parent_pdg, pdgid_lepton=lepton_pdg,
        pdgid_neutrino=nu_pdg, pdgid_mediator=5922)   # validated vector ME
    v1_to_chi = _VP.DarkPhotonToChiDecay(
        M_V1, M_CHI, G_D, pdgid_V1=5922, pdgid_chi=5917)
    upscatter = _VP.VectorPortalUpscatteringXS(
        M_CHI, M_CHI_PRIME, M_V2, G_D, EPSILON_2,
        pdgid_chi=5917, pdgid_chi_prime=5918,
        nuclear_pdgid=1000180400, nuclear_mass=M_ARGON40, A=40, Z=18)  # Ar40
    chi_prime_decay = _VP.ChiPrimeDecay(
        M_CHI, M_CHI_PRIME, M_V1, G_D,
        pdgid_chi_prime=5918, pdgid_chi=5917, pdgid_V1=5923)
    visible_decay = _VP.DarkPhotonDecay(M_V1, EPSILON_1, pdgid_V1=5923)
    return {
        "meson_decay": meson_decay,
        "secondary_interactions": {
            V1_PROD: [v1_to_chi],
            CHI: [upscatter],
            CHI_PRIME: [chi_prime_decay],
            V1_SIGNAL: [visible_decay],
        },
        "models": {
            "v1_to_chi": v1_to_chi,
            "upscatter": upscatter,
            "chi_prime_decay": chi_prime_decay,
            "visible_decay": visible_decay,
        },
    }


def build_geometric_targets(detector_model, fiducial):
    """Single fiducial-directed target (simplified; the SBND 13-probe basis
    is an optimization nicety, not required for a correct sum)."""
    return {"fiducial": fiducial}


def build_sX_cdf_table(meson_decay, n_nodes=257):
    """CDF of s_X = M^2(l,nu) built from the validated matrix element, so the
    DetectorDirected3Body proposal samples proportional to the physical
    marginal (flat weights). Validated to produce a monotonic 0->1 CDF for
    all four channels. Uses the real API: _matel_sq(E_nu,E_phi),
    _E_phi_limits(E_nu), E_nu_max."""
    _trapz = getattr(np, "trapezoid", getattr(np, "trapz"))
    eng = meson_decay._decay
    m_M = eng.m_M
    E_nu_max = eng.E_nu_max
    g = np.linspace(1e-6, E_nu_max, n_nodes)
    marg = []
    for E_nu in g:
        lims = eng._E_phi_limits(E_nu)
        lo, hi = (lims if lims is not None else (None, None))
        if lo is None or hi is None or hi <= lo:
            marg.append(0.0); continue
        EE = np.linspace(lo, hi, 24)
        vals = [eng._matel_sq(E_nu, e) for e in EE]
        marg.append(_trapz(vals, EE))
    marg = np.array(marg)
    if marg.max() <= 0:
        return [], []
    sX = m_M**2 - 2.0 * m_M * g
    order = np.argsort(sX)
    sX_s = sX[order]; m_s = marg[order]
    cdf = np.concatenate([[0.0],
          np.cumsum(0.5 * (m_s[1:] + m_s[:-1]) * np.diff(sX_s))])
    if cdf[-1] <= 0:
        return [], []
    return list(sX_s), list(cdf / cdf[-1])


def build_primary_phase_spaces(targets, meson_decay):
    sig = meson_decay.GetPossibleSignatures()[0]
    geo_list = list(targets.values())
    cdf_nodes, cdf_values = build_sX_cdf_table(meson_decay)
    # Tabulated needs a wide enough s_X window; razor-thin windows (e.g.
    # pi->mu, where m_pi - m_mu leaves ~0.004 GeV^2) make TabulatedMapping
    # degenerate ("non-positive cumulative mass"). Fall back to Uniform when
    # the window is too narrow or the CDF is degenerate.
    sX_span = (cdf_nodes[-1] - cdf_nodes[0]) if len(cdf_nodes) > 1 else 0.0
    use_tab = len(cdf_nodes) > 1 and sX_span > 0.02   # GeV^2 threshold
    mode = (injection.InvariantMassMode.Tabulated if use_tab
            else injection.InvariantMassMode.Uniform)
    print("    mass-mode: %s (s_X span = %.4f GeV^2)"
          % ("Tabulated" if use_tab else "Uniform", sX_span))
    channels = [injection.PhysicalDecayChannel(meson_decay, sig)]
    for target in geo_list:
        channels.append(
            injection.DetectorDirected3BodyChannel(
                target, directed_index=2,
                mass_mode=mode,
                resonance_mass=0.0, resonance_width=0.0,
                power_law_nu=-8.6, power_law_offset=0.0,
                topology=injection.PhaseSpaceTopology.Decay3Body,
                mass_cdf_nodes=(cdf_nodes if use_tab else []),
                mass_cdf_values=(cdf_values if use_tab else [])))
    n = len(channels)
    weights = [0.02] + [(1.0 - 0.02) / (n - 1)] * (n - 1)
    return {sig: _mc(channels, weights)}


def _build_2body_channels(geo_list, idx, model, sig):
    channels = [injection.PhysicalDecayChannel(model, sig)]
    for target in geo_list:
        channels.append(injection.DetectorDirected2BodyChannel(target, idx))
    n = len(channels)
    weights = [0.02] + [(1.0 - 0.02) / (n - 1)] * (n - 1)
    return _mc(channels, weights)


def build_onshell_phase_spaces(targets, models):
    m = models["models"]
    v1_sig = m["v1_to_chi"].GetPossibleSignatures()[0]
    chi_sig = m["upscatter"].GetPossibleSignatures()[0]
    chip_sig = m["chi_prime_decay"].GetPossibleSignatures()[0]
    vis_sig = m["visible_decay"].GetPossibleSignatures()[0]
    geo_list = list(targets.values())
    return {
        V1_PROD: {v1_sig: _build_2body_channels(geo_list, 0, m["v1_to_chi"], v1_sig)},
        CHI: {chi_sig: _mc([
            injection.PhysicalCrossSectionChannel(m["upscatter"], chi_sig)], [1.0])},
        CHI_PRIME: {chip_sig: _mc([
            injection.PhysicalDecayChannel(m["chi_prime_decay"], chip_sig)], [1.0])},
        V1_SIGNAL: {vis_sig: _mc([
            injection.PhysicalDecayChannel(m["visible_decay"], vis_sig),
            injection.Isotropic2BodyChannel(0)], [0.50, 0.50])},
    }


def onshell_stopping_condition(datum, i):
    sec = int(datum.record.signature.secondary_types[i])
    parent = int(datum.record.signature.primary_type)
    if sec == 5922:   return False
    if sec == 5917:   return parent != 5922 or i != 0
    if sec == 5918:   return False
    if sec == 5923:   return False
    return True


def _build_sigma_interp(upscatter, e_lo=0.06, e_hi=8.0, n=120):
    """Tabulate the chi-upscatter total cross-section sigma(E) on a grid and
    return a fast clipped-linear interpolator (the energy shape of the meson
    sampling bias). Built once per channel; normalized to O(1) so the tiny raw
    sigma cannot push the sampling weights into precision loss."""
    grid = np.linspace(e_lo, e_hi, n)
    sg = np.array([upscatter._ups.total_xsec(float(e)) for e in grid])
    smax = float(np.max(sg))
    if smax > 0:
        sg = sg / smax
    def sigma_fn(E):
        return np.interp(np.clip(E, e_lo, e_hi), grid, sg)
    return sigma_fn


def load_dk2nu_mesons(dk2nu_data, parent_pdg, detector_model, upscatter=None):
    """PrimaryExternalDistribution of a given parent from already-read dk2nu.

    When `upscatter` is supplied, the meson sampling bias is sigma(E)*max(cos,0)
    (candidate C) so the sampling tracks the true per-event weight. Otherwise the
    parameter-free forward shape max(cos,0) is used."""
    sigma_fn = _build_sigma_interp(upscatter) if upscatter is not None else None
    return _DK.dk2nu_to_primary_distribution(
        dk2nu_data, detector_model, parent_pdg=parent_pdg,
        sampling_bias=make_meson_bias(sigma_fn), flux_weighted_sampling=True)


# ------------------------------------------------------------------ #
#  Cut / observable helpers (ICARUS LAr, GDML volTPCActive sectors)    #
# ------------------------------------------------------------------ #
from siren.math import Vector3D as _MV3

_DEBUG_LAR = False  # set True to surface GetContainingSector exceptions

def _containing_sector(detector_model, vtx):
    """Return the sector containing detector-frame point `vtx` (a Vector3D).

    GetContainingSector requires a DetectorPosition (NOT a GeometryPosition);
    the interaction_vertex stored in the record is already in the detector
    frame, so it wraps directly into DetectorPosition.  If the local
    DetectorPosition constructor differs, fall back to converting a
    GeometryPosition through the model's frame transform.
    """
    from siren.detector import DetectorPosition
    try:
        return detector_model.GetContainingSector(DetectorPosition(vtx))
    except TypeError:
        from siren.detector import GeometryPosition
        try:
            dpos = detector_model.GeoPositionToDetPosition(
                GeometryPosition(vtx))
        except Exception:
            dpos = DetectorPosition(vtx[0], vtx[1], vtx[2])
        return detector_model.GetContainingSector(dpos)

def _in_lar_active(detector_model, vtx):
    """True if detector-frame point vtx is inside one of the 8 ICARUS
    volTPCActive liquid-argon sectors (the GDML active volume)."""
    try:
        sec = _containing_sector(detector_model, vtx)
        return (sec is not None) and sec.name.startswith("volTPCActive")
    except Exception as _e:
        if _DEBUG_LAR:
            print("  [lar-diag] _in_lar_active EXCEPTION: %r" % _e)
        return False

def upscatter_in_lar(event, detector_model, debug=False):
    """Did the chi+Ar -> chi'+Ar upscatter (producing chi'=5918) happen on Ar
    inside the LAr active volume?  GDML sector cut (volTPCActive)."""
    try:
        for _di, datum in enumerate(event.tree):
            r = datum.record
            secs = [int(s) for s in r.signature.secondary_types]
            if debug:
                try:
                    iv = r.interaction_vertex
                    vx, vy, vz = iv[0], iv[1], iv[2]
                    sec = _containing_sector(detector_model, _MV3(vx, vy, vz))
                    sec_name = sec.name if sec is not None else None
                    _vtx_str = "(%.3f,%.3f,%.3f)" % (vx, vy, vz)
                except Exception as _e:
                    _vtx_str = "?"; sec_name = "<sector-err:%r>" % _e
                print("  [vtx-diag] vtx%d prim=%d secs=%s  pos=%s  sector=%s"
                      % (_di, int(r.signature.primary_type), secs,
                         _vtx_str, sec_name))
            # upscatter vertex: chi' (5918) produced on an Ar nucleus
            if 5918 in secs and any(s in LAR_TARGET_PDGS for s in secs):
                vtx = _MV3(r.interaction_vertex[0], r.interaction_vertex[1],
                           r.interaction_vertex[2])
                if _in_lar_active(detector_model, vtx):
                    return True
    except Exception as _e:
        if debug:
            print("  [vtx-diag] EXCEPTION in upscatter_in_lar: %r" % _e)
        return False
    return False

def signal_eepair_observables(event, detector_model):
    """(E_vis, cos_theta) of the e+e- pair from the V1(5923)->e+e- visible
    decay, requiring the pair vertex inside volTPCActive, after the 140 MeV
    visible-energy and 10deg opening-angle cuts."""
    try:
        for datum in event.tree:
            r = datum.record
            secs = [int(s) for s in r.signature.secondary_types]
            if not (11 in secs and -11 in secs):
                continue
            vtx = _MV3(r.interaction_vertex[0], r.interaction_vertex[1],
                       r.interaction_vertex[2])
            if not _in_lar_active(detector_model, vtx):
                continue
            P = np.zeros(4); e_p3 = []
            for i, sp in enumerate(r.signature.secondary_types):
                if int(sp) in (11, -11):
                    p = r.secondary_momenta[i]
                    P = P + np.array([p[0], p[1], p[2], p[3]])
                    e_p3.append(np.array([p[1], p[2], p[3]]))
            if len(e_p3) < 2:
                continue
            p1, p2 = e_p3[0], e_p3[1]
            n1, n2 = np.linalg.norm(p1), np.linalg.norm(p2)
            if n1 <= 0 or n2 <= 0:
                continue
            cos_open = max(-1.0, min(1.0, float(np.dot(p1, p2) / (n1 * n2))))
            if math.degrees(math.acos(cos_open)) > E_PAIR_MAX_DEG:
                continue
            E_vis = P[0]
            if E_vis < E_VIS_THRESHOLD:
                continue
            pmag = math.sqrt(P[1]**2 + P[2]**2 + P[3]**2)
            cth = P[3] / pmag if pmag > 0 else 0.0
            return E_vis, cth
    except Exception:
        return None
    return None


# ------------------------------------------------------------------ #
#  Per-channel run                                                     #
# ------------------------------------------------------------------ #
def run_channel(name, dk2nu_data, detector_model, n_events=events_to_inject, debug=False):
    parent_pdg, m_meson, m_lepton, lepton_pdg, nu_pdg, gamma_sm = CHANNELS[name]
    print("\n" + "=" * 64)
    print("  CHANNEL %s : parent=%d  m_meson=%.4f  m_lepton=%.5f"
          % (name, parent_pdg, m_meson, m_lepton))
    print("=" * 64)

    if (m_meson - m_lepton) <= M_V1:
        print("  kinematically forbidden -> skip"); return np.array([]), np.array([]), np.array([])

    # Injection / importance-sampling geometry.
    # Two complementary volumes are used:
    #
    #   inject_sphere  — a sphere of R=R_LAR_INJECT enclosing all TPC corners.
    #                    Used as the SecondaryBoundedVertexDistribution for chi
    #                    (bounding box for the upscatter vertex sampler).
    #
    #   inject_box     — a tight axis-aligned Box that matches the actual ICARUS
    #                    TPC active volume envelope (_TPC_BOX_X × Y × Z).
    #                    Used as the DetectorDirected target for V1 and chi, so
    #                    that the direction biasing points precisely into LAr
    #                    instead of into the larger sphere dead-zones.
    #                    Volume ≈ 1469 m³ vs sphere ≈ 7238 m³ → ~5x better
    #                    directional efficiency.
    #
    # Neither is a physical cut; the real signal selection is the volTPCActive
    # GDML sector check applied below in upscatter_in_lar / signal_eepair_observables.
    inject_sphere = siren.geometry.Sphere(R_LAR_INJECT, 0.0)
    inject_box    = siren.geometry.Box(_TPC_BOX_X, _TPC_BOX_Y, _TPC_BOX_Z)

    # The directed channels use the tight TPC bounding box; the bounded vertex
    # distribution uses the sphere (Box.IsInside is stricter and would reject
    # valid corners when the chi path is slightly off-axis).
    targets = build_geometric_targets(detector_model, inject_box)
    chain = build_onshell_models(parent_pdg, m_meson, m_lepton, lepton_pdg, nu_pdg)
    meson_decay = chain["meson_decay"]
    secondary_interactions = chain["secondary_interactions"]
    phase_spaces = build_onshell_phase_spaces(targets, chain)
    primary_ps = build_primary_phase_spaces(targets, meson_decay)

    bsm_width = meson_decay._total_width
    br_bsm = bsm_width / gamma_sm
    print("  BSM 3-body width (calibrated): %.4e GeV   SM total width: %.4e   BR: %.4e"
          % (bsm_width * CALIB_VECTOR, gamma_sm, bsm_width * CALIB_VECTOR / gamma_sm))

    meson_dist = load_dk2nu_mesons(dk2nu_data, parent_pdg, detector_model,
                                   upscatter=chain["models"]["upscatter"])
    primary_dists = [meson_dist]
    br_dist = distributions.NormalizationConstant(br_bsm)
    physical_dists = [meson_dist, br_dist]
    primary_mode = injection.VertexWeightingMode.Fixed()

    sv = distributions.SecondaryPhysicalVertexDistribution()
    # Use the sphere (R=15m) for the chi vertex bound: the sphere is more
    # forgiving for the sampler (avoids edge effects from the box's sharp
    # corners), while still correctly enclosing all 8 TPC sectors.
    sv_bounded = distributions.SecondaryBoundedVertexDistribution(inject_sphere)
    sec_dists = {pt: [sv] for pt in secondary_interactions}
    sec_dists[CHI] = [sv_bounded]

    print("  Injection sphere R=%.1fm  box=%.2fx%.2fx%.2fm" %
          (R_LAR_INJECT, _TPC_BOX_X, _TPC_BOX_Y, _TPC_BOX_Z))
    print("  Building injector (%d events) ..." % n_events)
    injector = Injector(
        number_of_events=n_events, detector_model=detector_model, seed=42,
        primary_type=PT(parent_pdg), primary_interactions=[meson_decay],
        primary_injection_distributions=primary_dists,
        primary_weighting_mode=primary_mode,
        secondary_interactions=secondary_interactions,
        secondary_injection_distributions=sec_dists,
        secondary_phase_spaces=phase_spaces,
        primary_phase_spaces=primary_ps,
        stopping_condition=onshell_stopping_condition,
    )
    # Force-init, tolerant of a pathological first event (large-phase-space
    # channels like K->e can throw on the first GenerateEvent; that must not
    # leave the level stack half-entered).
    try:
        for ev in injector:
            break
    except RuntimeError as e:
        print("  (force-init first event threw: %r — continuing)" % e)
    try:
        injector._Injector__injector.ResetInjectedEvents(n_events)
    except Exception:
        pass

    weighter = Weighter(
        injectors=[injector], detector_model=detector_model,
        primary_type=PT(parent_pdg), primary_interactions=[meson_decay],
        primary_physical_distributions=physical_dists,
        secondary_interactions=secondary_interactions,
    )

    print("  Generating events ...")
    events = []
    it = iter(injector)
    n_skipped = 0
    while len(events) < n_events:
        try:
            event = next(it)
        except StopIteration:
            break
        except RuntimeError:
            n_skipped += 1
            if n_skipped > 5 * n_events:   # avoid an infinite skip loop
                print("  (too many failed events — stopping generation)")
                break
            continue
        if event.tree:
            events.append(event)
    if n_skipped:
        print("  (skipped %d kinematically pathological events)" % n_skipped)
    print("  Generated %d events" % len(events))

    raw = np.array([weighter(ev) for ev in events])
    if debug:
        globals()["_DEBUG_LAR"] = True
        for _ev in events[:5]:
            upscatter_in_lar(_ev, detector_model, debug=True)
        globals()["_DEBUG_LAR"] = False
    n_lar = int(np.sum([upscatter_in_lar(ev, detector_model) for ev in events]))
    print("  [diag] upscatter on Ar in LAr active: %d" % n_lar)

    Ev, cs, wv = [], [], []
    for ev, w in zip(events, raw):
        if not np.isfinite(w) or w <= 0:
            continue
        if not upscatter_in_lar(ev, detector_model):
            continue
        obs = signal_eepair_observables(ev, detector_model)
        if obs is None:
            continue
        E_vis = obs[0]
        # Absolute normalization: calibrated C-R constant × delivered POT × efficiency.
        # Factor 2: V1_PROD -> chi + chi pair; either chi can upscatter
        # (Dutta-Kim Eq. 4 prefactor). Only chi[0] propagates via stopping condition.
        w_abs = (w * 2.0 * CALIB_VECTOR * ICARUS_POT
                 * detection_efficiency(E_vis))
        if not np.isfinite(w_abs) or w_abs <= 0:
            continue
        Ev.append(E_vis); cs.append(obs[1]); wv.append(w_abs)
    Ev, cs, wv = np.array(Ev), np.array(cs), np.array(wv)
    # ---- weight-health diagnostics (validated-script convention) ----
    if len(wv) > 0:
        w_mean = wv.mean()
        w_rel  = wv.std() / w_mean if w_mean > 0 else float("inf")
        # effective sample fraction: (sum w)^2 / (N * sum w^2)
        esf = (wv.sum()**2) / (len(wv) * np.sum(wv**2)) if np.sum(wv**2) > 0 else 0.0
        order = np.argsort(wv)[::-1]
        print("  [wstat] range %.3e - %.3e   mean %.3e   stddev/mean %.2f   ESF %.1f%%"
              % (wv.min(), wv.max(), w_mean, w_rel, 100.0 * esf))
        topw = wv[order[:5]]
        print("  [wstat] top-5 weights: %s  (=%.1f%% of sum)"
              % (np.array2string(topw, precision=2),
                 100.0 * topw.sum() / wv.sum()))
    print("  [result] plottable signal: %d   sum(w_abs)=%.3e events"
          % (len(Ev), wv.sum() if len(wv) else 0.0))
    return Ev, cs, wv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", choices=list(CHANNELS) + ["all"], default="all")
    ap.add_argument("--n-events", type=int, default=events_to_inject)
    ap.add_argument("--debug", action="store_true",
                    help="Print per-vertex LAr sector diagnostics for first 5 events per channel")
    args = ap.parse_args()

    print("Loading ICARUS detector (GDML) ...")
    detector_model = siren.utilities.load_detector("SBN", detector="ICARUS")

    print("Reading dk2nu (all parents) ...")
    dk2nu_data = _DK.read_dk2nu(DK2NU_FILE)
    try:
        _DK.print_summary(dk2nu_data)
    except Exception:
        pass

    names = list(CHANNELS) if args.channel == "all" else [args.channel]
    per_channel = {}
    for name in names:
        per_channel[name] = run_channel(name, dk2nu_data, detector_model, args.n_events,
                                        debug=args.debug)

    E_all = np.concatenate([per_channel[n][0] for n in per_channel]) \
            if any(len(per_channel[n][0]) for n in per_channel) else np.array([])
    c_all = np.concatenate([per_channel[n][1] for n in per_channel]) if E_all.size else np.array([])
    w_all = np.concatenate([per_channel[n][2] for n in per_channel]) if E_all.size else np.array([])

    print("\n" + "=" * 64)
    print("  MULTICHANNEL SUM  —  ICARUS Vector Portal (full chain)")
    print("=" * 64)
    for n in per_channel:
        Ev, cs, wv = per_channel[n]
        print("  %-6s : %4d events   sum(w)=%.3e"
              % (n, len(Ev), wv.sum() if len(wv) else 0.0))
    print("  %-6s : %4d events   sum(w)=%.3e"
          % ("TOTAL", len(E_all), w_all.sum() if w_all.size else 0.0))
    print("=" * 64)

    os.makedirs("output", exist_ok=True)
    stem = "output/ICARUS_VectorPortal_multichannel_fullchain"
    np.savez(stem + "_observables.npz", E_vis=E_all, cos_theta=c_all, weight=w_all,
             **{f"{n}_E": per_channel[n][0] for n in per_channel},
             **{f"{n}_w": per_channel[n][2] for n in per_channel})
    print("  Saved -> %s_observables.npz" % stem)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Bin definitions matching Dutta-Kim Fig. 2 style:
    #   Energy:   0-2000 MeV, 50 MeV bins (Fig. 2 x-axis)
    #   cos theta: -1 to 1, 60 bins (full range, as in Fig. 2)
    #   cos theta (zoom): 0.8-1.0, 40 bins  (ICARUS/NuMI forward-peak detail)
    E_bins  = np.linspace(0.0, 2.0, 41)          # 40 bins × 50 MeV
    c_bins  = np.linspace(-1.0, 1.0, 61)          # 60 bins × 0.033
    cz_bins = np.linspace(0.80, 1.0, 41)          # 40 bins × 0.005 (zoom)
    colors  = {"K_e": "C0", "K_mu": "C1", "pi_e": "C2", "pi_mu": "C3"}

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    for n in per_channel:
        Ev, cs, wv = per_channel[n]
        if len(Ev):
            ax[0].hist(Ev*1e3, bins=E_bins*1e3, weights=wv, histtype="step",
                       color=colors.get(n), label=n)
            ax[1].hist(cs, bins=c_bins, weights=wv, histtype="step",
                       color=colors.get(n), label=n)
            ax[2].hist(cs, bins=cz_bins, weights=wv, histtype="step",
                       color=colors.get(n), label=n)

    if E_all.size:
        ax[0].hist(E_all*1e3, bins=E_bins*1e3, weights=w_all, histtype="step",
                   color="k", lw=2, label="TOTAL")
        ax[1].hist(c_all, bins=c_bins, weights=w_all, histtype="step",
                   color="k", lw=2, label="TOTAL")
        ax[2].hist(c_all, bins=cz_bins, weights=w_all, histtype="step",
                   color="k", lw=2, label="TOTAL")

    # --- Axis labels matching Dutta-Kim Fig. 2 exactly ---
    # "Counts" matches the Dutta-Kim Fig.2 y-axis label exactly.
    # The POT exposure (6e20) is already folded into w_abs; the histogram
    # bins show the total expected signal events for the full ICARUS exposure.
    y_label = "Counts"
    pot_note = r"(%.0e POT, ICARUS NuMI)" % ICARUS_POT

    ax[0].set_xlabel(r"$E_\mathrm{vis}$ [MeV]", fontsize=12)
    ax[0].set_ylabel(y_label, fontsize=12)
    ax[0].set_xlim(0, 2000)
    ax[0].set_ylim(bottom=0)
    ax[0].axvline(E_VIS_THRESHOLD * 1e3, color="gray", ls="--", lw=0.8,
                  label="140 MeV threshold")
    ax[0].set_title(r"ICARUS $\chi$ upscattering: $E_\mathrm{vis}\ (e^+e^-)$ " + pot_note,
                    fontsize=10)
    ax[0].legend(fontsize=8)

    ax[1].set_xlabel(r"$\cos\theta$", fontsize=12)
    ax[1].set_ylabel(y_label, fontsize=12)
    ax[1].set_xlim(-1, 1)
    ax[1].set_ylim(bottom=0)
    ax[1].set_title(r"ICARUS $\chi$ upscattering: $\cos\theta$ wrt beam " + pot_note,
                    fontsize=10)
    ax[1].legend(fontsize=8)

    ax[2].set_xlabel(r"$\cos\theta$", fontsize=12)
    ax[2].set_ylabel(y_label, fontsize=12)
    ax[2].set_xlim(0.80, 1.0)
    ax[2].set_ylim(bottom=0)
    ax[2].set_title(r"ICARUS $\chi$ upscattering: $\cos\theta$ zoomed $[0.80,1.0]$ " + pot_note,
                    fontsize=10)
    ax[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(stem + "_countrate.png", dpi=130)
    print("  Saved -> %s_countrate.png" % stem)
    print("  Done.")


if __name__ == "__main__":
    main()
