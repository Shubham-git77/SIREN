"""
Long-lived SCALAR (phi) Dark Primakoff chain at ICARUS (LAr)  —  MULTICHANNEL sum.

Model ii) of Dutta et al. (arXiv:2110.11944): Fig.1(a)-phi production +
Fig.1(c) Dark Primakoff scattering. Reproduces the Fig.2 BOTTOM panels.

Chain (SHORTER than the vector portal -- one secondary vertex, no cascade):
    pi+/K+ -> l+ nu phi        (three-body SCALAR production, validated ME)
    phi propagates to detector
    phi N -> gamma N           (Dark Primakoff, Eq.C2/C3, DarkPrimakoffUpsCase)
    signal = single PHOTON (E_gamma ~ E_phi, coherent recoil keV-MeV)

Adapted from the working vector multichannel script:
  - production mediator_type="scalar", pdgid_mediator=5919 (phi)
  - secondary chain = {phi: [DarkPrimakoff]} ONLY (no chi'/V1/e+e-)
  - observable = single photon (E_gamma, cos_theta), not e+e- pair
  - sums the four channels K/pi x e/mu.

Benchmark (Table I/II scalar): m_phi=1 MeV, m_Z'=49 MeV,
    (g_mu, g_n, lambda) = (5e-3, 1e-2, 0.44 GeV^-1).

PHYSICS NOTE: the scalar couples to muons (g_mu phi mu-bar mu), so K->mu and
pi->mu are the natural channels. The e channels are included for completeness
using the same scalar ME with m_l = m_e; whether they carry the muon coupling
or a separate g_e is a modeling choice -- they are expected to be small.
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
#  Load physics modules (MesonProduction + DarkPrimakoff + Dk2nuReader) #
# ------------------------------------------------------------------ #
_PROC_DIR = os.path.join(_util.resource_package_dir(), "processes", "DarkNewsTables")
_MESON = _util.load_module("DuttaKim_MesonProduction",
                           os.path.join(_PROC_DIR, "MesonProduction.py"))
_DK = _util.load_module("DuttaKim_Dk2nuReader",
                        os.path.join(_PROC_DIR, "Dk2nuReader.py"))
_DP = _util.load_module("DuttaKim_DarkPrimakoff",
                        os.path.join(_PROC_DIR, "DarkPrimakoff.py"))

# Scalar production normalization. The vector model needed an empirical
# constant (2412) to bridge the C-R vector convention to Table II. The scalar
# ME (_matel_sq_scalar) matches C-R Eq.25 DIRECTLY, so it should NOT carry the
# vector's bridge factor. Set to 1.0 until validated against the paper's
# scalar production prediction (Fig.4 / Table II scalar rows).
# TODO: validate BR(K->mu nu phi) against the paper and set this if needed.
CALIB_SCALAR = 1.0

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

# ---- Scalar model (Table I/II of arXiv:2110.11944): m_phi=1 MeV, m_Z'=49 MeV ----
# Table I best-fit (scalar): (m_Z', m_phi) = (49, 1) MeV
# Table II individual benchmark (scalar):
#   g_mu = 5e-3,  g_n = 1e-2,  lambda = 4.4e-4 MeV^-1 = 0.44 GeV^-1
# NOTE: the PSEUDOSCALAR benchmark uses (m_Z'=85 MeV, g_mu=1e-2, lambda=6.5 GeV^-1) —
# all three differ from the scalar values above.
M_PHI = 1e-3                 # GeV  scalar mediator (phi)
M_ZP  = 49e-3                # GeV  Z' dark Primakoff mediator (scalar: 49 MeV)
M_ARGON40  = 37.224          # GeV, Ar40 nuclear mass (ICARUS liquid argon)

# Couplings (Table II scalar benchmark)
G_MU_PROD = 5e-3             # phi-muon coupling (g_mu);  pseudoscalar had 1e-2
G_N       = 1e-2             # Z'-nucleon coupling (g_n); same for both
LAMBDA    = 0.44             # GeV^-1 (= 4.4e-4 MeV^-1); pseudoscalar had 6.5 GeV^-1

PT = lambda pdg: dataclasses.Particle.ParticleType(pdg)
PHI = PT(5919)             # scalar mediator (single secondary vertex)

# ICARUS LAr geometry (active-volume cut via GDML volTPCActive sectors).
# In detector coordinates the 8 TPC corners reach R = 14.51 m, so
# R_LAR_INJECT must exceed 14.51 m to enclose the full active volume.
# Tight bounding box (X×Y×Z) encloses all 8 TPC sectors with minimal deadzone.
# ICARUS is TWO separate cryostats (C0, C1) with a ~1.2 m argon-free gap, NOT
# one monolithic box. _TPC_BOX is ONE active module (3.0 x 3.16 x 17.95 m from
# sbn_geometry ICARUS_C0/C1); the analytic engine is called once per module
# center (ICARUS_MODULE_CENTERS_BNB) and the yields are summed. The old single
# 8.64 x 6.34 x 26.84 m box was the warm-vessel envelope -> 2051 t of phantom
# argon (4.3x the true 476 t across both modules), inflating every ICARUS rate.
R_LAR_INJECT = 10.0   # sphere radius [m] — encloses one module's far corner (9.23 m)
_TPC_BOX_X   = 3.00   # single-module x-extent [m]  (covers ±1.50 m)
_TPC_BOX_Y   = 3.16   # single-module y-extent [m]
_TPC_BOX_Z   = 17.95  # single-module z-extent [m]  (covers ±8.975 m)
# Two active-module centers in the BNB frame [m] (sbn_geometry ICARUS_C0/C1).
ICARUS_MODULE_CENTERS_BNB = [(-2.10215, -0.202, 600.0), (2.10215, -0.202, 600.0)]
LAR_TARGET_PDGS = {1000180400}                # Ar40
events_to_inject = 10_000

# Reconstruction cuts (single-photon / single-shower selection)
E_VIS_THRESHOLD = 0.140

# ICARUS NuMI exposure (SBN programme nominal, neutrino mode).
ICARUS_POT = 6e20
#
# Energy-dependent detection efficiency eps(E_vis). NOTE: the digitized MiniBooNE
# single-photon efficiency (mb_eff, ~0.1) does NOT apply here -- ICARUS is a LArTPC
# with its own (much higher, energy-dependent) EM-shower efficiency. Fill with an
# ICARUS-specific curve when available; None -> efficiency 1.0 (placeholder).
_EFF_TABLE = None

def detection_efficiency(E_vis_gev):
    if _EFF_TABLE is None:
        return 1.0
    E = np.asarray(_EFF_TABLE)[:, 0]; eff = np.asarray(_EFF_TABLE)[:, 1]
    return float(np.interp(E_vis_gev, E, eff, left=eff[0], right=eff[-1]))

DK2NU_FILE = os.environ.get("DK2NU_FILE", "/home/shubham/nubeamHighSample.dk2nu.root")

# The four channels: name -> (parent_pdg, m_meson, m_lepton, lepton_pdg,
#                              nu_pdg, gamma_sm)
# All four channels included (lepton-universal coupling, g_e = g_mu). NB the paper
# (Dutta et al. 2110.11944) is muon-only (g_e=0); with g_e=g_mu the helicity-
# unsuppressed pi->e nu phi is large. For non-universal coupling, scale the
# electron-channel production coupling by g_e/g_mu.
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
    TO PHYSICAL RATE, focused forward where the boosted phi can reach ICARUS.

    bias = sigma(E_phi) * max(cos,0)   (candidate "C")
      The Dark Primakoff scatter probability scales with sigma(E_phi), which
      DECREASES with energy. Folding sigma in (proxying E_phi by the meson
      energy E, with which it is tightly correlated) makes the sampling track
      the true per-event weight: it down-weights high-E mesons (small sigma)
      that the forward-only shape over-sampled, removing the residual K_mu tail
      (ESF 0.7% -> O(30%)).  DO NOT reintroduce an E**n factor: sigma already
      carries the correct (decreasing) energy dependence.

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
#  Channel-parameterized model building (SCALAR production + Primakoff) #
# ------------------------------------------------------------------ #
def build_onshell_models(parent_pdg, m_meson, m_lepton, lepton_pdg, nu_pdg):
    """Scalar chain: meson -> l nu phi (validated scalar ME), then a SINGLE
    Dark Primakoff vertex phi N -> gamma N. No chi'/V1/e+e- cascade."""
    meson_decay = _MESON.MesonThreeBodySIRENDecay(
        m_meson, m_lepton, M_PHI, G_MU_PROD, "scalar",   # SCALAR production
        pdgid_meson=parent_pdg, pdgid_lepton=lepton_pdg,
        pdgid_neutrino=nu_pdg, pdgid_mediator=5919)        # phi = 5919
    primakoff = _DP.DarkPrimakoffUpsCase(
        M_PHI, M_ZP, G_N, LAMBDA,
        nuclear_pdgid=1000180400, nuclear_mass=M_ARGON40,
        A=40, Z=18, pdgid_phi=5919)                        # Ar40, phi->gamma
    return {
        "meson_decay": meson_decay,
        "secondary_interactions": {
            PHI: [primakoff],          # single secondary vertex
        },
        "models": {
            "primakoff": primakoff,
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
                power_law_nu=-2.0, power_law_offset=0.0,
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
    prim_sig = m["primakoff"].GetPossibleSignatures()[0]
    geo_list = list(targets.values())
    # Single secondary vertex: phi N -> gamma N (a 2->2 scatter).
    return {
        PHI: {prim_sig: _mc([
            injection.PhysicalCrossSectionChannel(m["primakoff"], prim_sig)], [1.0])},
    }



def onshell_stopping_condition(datum, i):
    sec = int(datum.record.signature.secondary_types[i])
    # phi (5919) must scatter (don't stop); photon (22) and nucleus are final.
    if sec == 5919:
        return False          # let phi propagate + Primakoff-scatter
    return True               # stop at photon / nucleus / everything else


def _build_sigma_interp(primakoff, e_lo=0.06, e_hi=8.0, n=120):
    """Tabulate the Dark Primakoff total cross-section sigma(E) on a grid and
    return a fast clipped-linear interpolator (used as the energy shape of the
    meson sampling bias). Built once per channel."""
    grid = np.linspace(e_lo, e_hi, n)
    sg = np.array([primakoff._dp.total_xsec(float(e)) for e in grid])
    # Normalize to O(1): the sampling bias is only defined up to an overall
    # scale, but the raw sigma (~1e-32 cm^2) would push the sampling weights to
    # ~1e-33 and risk precision loss in the sampler's internal normalization.
    smax = float(np.max(sg))
    if smax > 0:
        sg = sg / smax
    def sigma_fn(E):
        return np.interp(np.clip(E, e_lo, e_hi), grid, sg)
    return sigma_fn


def load_dk2nu_mesons(dk2nu_data, parent_pdg, detector_model, primakoff=None):
    """PrimaryExternalDistribution of a given parent from already-read dk2nu.

    When `primakoff` is supplied, the meson sampling bias is sigma(E)*max(cos,0)
    (candidate C) so the sampling tracks the true per-event weight (Dark
    Primakoff sigma decreases with E). Otherwise the parameter-free forward
    shape max(cos,0) is used."""
    sigma_fn = _build_sigma_interp(primakoff) if primakoff is not None else None
    return _DK.dk2nu_to_primary_distribution(
        dk2nu_data, detector_model, parent_pdg=parent_pdg,
        sampling_bias=make_meson_bias(sigma_fn), flux_weighted_sampling=True)


# ------------------------------------------------------------------ #
#  Cut / observable helpers (MiniBooNE)                                #
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
        # Fallback: some builds expose DetectorPosition(x, y, z) only.
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

def primakoff_in_lar(event, detector_model, debug=False):
    """Did a phi->gamma Primakoff scatter happen on Ar inside the LAr active
    volume?  Uses the GDML sector cut (volTPCActive) instead of a sphere."""
    try:
        from siren.detector import GeometryPosition
        for _di, datum in enumerate(event.tree):
            r = datum.record
            secs = [int(s) for s in r.signature.secondary_types]
            if debug:
                # Dump EVERY vertex's signature + its containing sector, so we
                # see whether the photon+Ar vertex exists and where it lands.
                try:
                    iv = r.interaction_vertex
                    vx, vy, vz = iv[0], iv[1], iv[2]
                    vtx_d = _MV3(vx, vy, vz)
                    sec = _containing_sector(detector_model, vtx_d)
                    sec_name = sec.name if sec is not None else None
                    _vtx_str = "(%.3f,%.3f,%.3f)" % (vx, vy, vz)
                except Exception as _e:
                    _vtx_str = "?"
                    sec_name = "<sector-err:%r>" % _e
                print("  [vtx-diag] vtx%d prim=%d secs=%s  pos=%s  sector=%s"
                      % (_di, int(r.signature.primary_type), secs,
                         _vtx_str, sec_name))
            # Primakoff vertex: photon (22) + nucleus, primary was phi (5919)
            if 22 in secs and any(s in LAR_TARGET_PDGS for s in secs):
                vtx = _MV3(r.interaction_vertex[0], r.interaction_vertex[1],
                           r.interaction_vertex[2])
                if _in_lar_active(detector_model, vtx):
                    return True
    except Exception as _e:
        if debug:
            print("  [vtx-diag] EXCEPTION in primakoff_in_lar: %r" % _e)
        return False
    return False

def signal_photon_observables(event, detector_model):
    """(E_gamma, cos_theta) of the single Primakoff photon at a fiducial
    vertex, after the E_vis threshold. Single-shower selection (one photon)."""
    try:
        for datum in event.tree:
            r = datum.record
            secs = [int(s) for s in r.signature.secondary_types]
            if 22 not in secs:
                continue
            vtx = _MV3(r.interaction_vertex[0], r.interaction_vertex[1],
                       r.interaction_vertex[2])
            if not _in_lar_active(detector_model, vtx):
                continue
            # the photon four-momentum
            for i, sp in enumerate(r.signature.secondary_types):
                if int(sp) == 22:
                    p = r.secondary_momenta[i]
                    E_gamma = p[0]
                    if E_gamma < E_VIS_THRESHOLD:
                        continue
                    pmag = math.sqrt(p[1]**2 + p[2]**2 + p[3]**2)
                    cth = p[3] / pmag if pmag > 0 else 0.0
                    return E_gamma, cth
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

    if (m_meson - m_lepton) <= M_PHI:
        print("  kinematically forbidden -> skip"); return np.array([]), np.array([]), np.array([])

    # Injection geometry — same two-volume approach as the vector portal:
    #   inject_sphere (R=15m): bounds phi scatter vertex (conservative, no edge effects)
    #   inject_box  (TPC envelope): directed channel target for phi and V1 — aims
    #               secondaries into the actual LAr volume, not into dead zones.
    inject_sphere = siren.geometry.Sphere(R_LAR_INJECT, 0.0)
    inject_box    = siren.geometry.Box(_TPC_BOX_X, _TPC_BOX_Y, _TPC_BOX_Z)
    targets = build_geometric_targets(detector_model, inject_box)

    chain       = build_onshell_models(parent_pdg, m_meson, m_lepton, lepton_pdg, nu_pdg)
    meson_decay = chain["meson_decay"]
    primakoff   = chain["models"]["primakoff"]
    secondary_interactions = chain["secondary_interactions"]

    # --- MESON-AS-PRIMARY ARCHITECTURE (mirrors the vector portal) -----------
    # Primary  = meson (from dk2nu), Fixed mode (decay position from dk2nu).
    # Primary interaction = meson -> l nu phi  (three-body scalar ME).
    # Secondary = phi travels to TPC and undergoes Dark Primakoff phi N -> gamma N.
    # This uses the actual dk2nu meson positions/momenta directly and avoids the
    # phi-as-primary approximations (wrong origin, BNB flux tables, BR double-count).
    bsm_width = meson_decay._total_width
    br_bsm    = bsm_width / gamma_sm
    print("  BSM 3-body width (calibrated): %.4e GeV   SM total width: %.4e   BR: %.4e"
          % (bsm_width * CALIB_SCALAR, gamma_sm, bsm_width * CALIB_SCALAR / gamma_sm))

    meson_dist  = load_dk2nu_mesons(dk2nu_data, parent_pdg, detector_model,
                                    primakoff=primakoff)
    br_dist     = distributions.NormalizationConstant(br_bsm)
    primary_injection_dists = [meson_dist]
    primary_physical_dists  = [meson_dist, br_dist]
    primary_mode = injection.VertexWeightingMode.Fixed()

    # phi propagates physically; bound its scatter vertex to the TPC sphere.
    sv         = distributions.SecondaryPhysicalVertexDistribution()
    sv_bounded = distributions.SecondaryBoundedVertexDistribution(inject_sphere)
    sec_dists  = {PHI: [sv_bounded]}

    # Phase spaces
    primary_ps   = build_primary_phase_spaces(targets, meson_decay)
    secondary_ps = build_onshell_phase_spaces(targets, chain)

    print("  Injection sphere R=%.1fm  box=%.2fx%.2fx%.2fm" %
          (R_LAR_INJECT, _TPC_BOX_X, _TPC_BOX_Y, _TPC_BOX_Z))
    print("  Building injector (%d events) ..." % n_events)
    injector = Injector(
        number_of_events=n_events, detector_model=detector_model, seed=42,
        primary_type=PT(parent_pdg), primary_interactions=[meson_decay],
        primary_injection_distributions=primary_injection_dists,
        primary_weighting_mode=primary_mode,
        secondary_interactions=secondary_interactions,
        secondary_injection_distributions=sec_dists,
        secondary_phase_spaces=secondary_ps,
        primary_phase_spaces=primary_ps,
        stopping_condition=onshell_stopping_condition,
    )
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
        primary_physical_distributions=primary_physical_dists,
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
            if n_skipped > 5 * n_events:
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
            primakoff_in_lar(_ev, detector_model, debug=True)
        globals()["_DEBUG_LAR"] = False

    n_lar = int(np.sum([primakoff_in_lar(ev, detector_model) for ev in events]))
    print("  [diag] Primakoff scatter on Ar in LAr active: %d" % n_lar)

    Ev, cs, wv = [], [], []
    for ev, w in zip(events, raw):
        if not np.isfinite(w) or w <= 0:
            continue
        if not primakoff_in_lar(ev, detector_model):
            continue
        obs = signal_photon_observables(ev, detector_model)
        if obs is None:
            continue
        E_vis = obs[0]
        # Absolute normalization: C-R calibration × delivered NuMI POT × efficiency.
        w_abs = w * CALIB_SCALAR * ICARUS_POT * detection_efficiency(E_vis)
        if not np.isfinite(w_abs) or w_abs <= 0:
            continue
        Ev.append(E_vis); cs.append(obs[1]); wv.append(w_abs)
    Ev, cs, wv = np.array(Ev), np.array(cs), np.array(wv)

    if len(wv) > 0:
        w_mean = wv.mean()
        w_rel  = wv.std() / w_mean if w_mean > 0 else float("inf")
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
                    help="Print per-vertex LAr sector diagnostics for first 5 events")
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
        per_channel[name] = run_channel(name, dk2nu_data, detector_model,
                                        args.n_events, debug=args.debug)

    E_all = np.concatenate([per_channel[n][0] for n in per_channel]) \
            if any(len(per_channel[n][0]) for n in per_channel) else np.array([])
    c_all = np.concatenate([per_channel[n][1] for n in per_channel]) if E_all.size else np.array([])
    w_all = np.concatenate([per_channel[n][2] for n in per_channel]) if E_all.size else np.array([])

    print("\n" + "=" * 64)
    print("  MULTICHANNEL SUM  —  ICARUS Scalar Dark Primakoff")
    print("=" * 64)
    for n in per_channel:
        Ev, cs, wv = per_channel[n]
        print("  %-6s : %4d events   sum(w)=%.3e"
              % (n, len(Ev), wv.sum() if len(wv) else 0.0))
    print("  %-6s : %4d events   sum(w)=%.3e"
          % ("TOTAL", len(E_all), w_all.sum() if w_all.size else 0.0))
    print("=" * 64)

    os.makedirs("output", exist_ok=True)
    stem = "output/ICARUS_ScalarPrimakoff_multichannel"
    np.savez(stem + "_observables.npz", E_vis=E_all, cos_theta=c_all, weight=w_all,
             **{f"{n}_E": per_channel[n][0] for n in per_channel},
             **{f"{n}_w": per_channel[n][2] for n in per_channel})
    print("  Saved -> %s_observables.npz" % stem)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Bins matching Dutta-Kim Fig. 2 style
    E_bins  = np.linspace(0.0, 2.0, 41)    # 40 × 50 MeV bins, 0-2000 MeV
    c_bins  = np.linspace(-1.0, 1.0, 61)   # 60 bins, full cos theta range
    cz_bins = np.linspace(0.80, 1.0, 41)   # 40 bins, forward zoom
    colors  = {"K_e": "C0", "K_mu": "C1", "pi_e": "C2", "pi_mu": "C3"}
    pot_note = r"(%.0e POT, ICARUS NuMI)" % ICARUS_POT

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    for n in per_channel:
        Ev, cs, wv = per_channel[n]
        if len(Ev):
            ax[0].hist(Ev*1e3, bins=E_bins*1e3, weights=wv, histtype="step",
                       color=colors.get(n), label=n)
            ax[1].hist(cs, bins=c_bins,  weights=wv, histtype="step",
                       color=colors.get(n), label=n)
            ax[2].hist(cs, bins=cz_bins, weights=wv, histtype="step",
                       color=colors.get(n), label=n)
    if E_all.size:
        ax[0].hist(E_all*1e3, bins=E_bins*1e3, weights=w_all, histtype="step",
                   color="k", lw=2, label="TOTAL")
        ax[1].hist(c_all, bins=c_bins,  weights=w_all, histtype="step",
                   color="k", lw=2, label="TOTAL")
        ax[2].hist(c_all, bins=cz_bins, weights=w_all, histtype="step",
                   color="k", lw=2, label="TOTAL")

    ax[0].set_xlabel(r"$E_\mathrm{vis}$ [MeV]", fontsize=12)
    ax[0].set_ylabel("Counts", fontsize=12)
    ax[0].set_xlim(0, 2000); ax[0].set_ylim(bottom=0)
    ax[0].axvline(E_VIS_THRESHOLD * 1e3, color="gray", ls="--", lw=0.8,
                  label="140 MeV threshold")
    ax[0].set_title(r"ICARUS scalar Dark Primakoff: $E_\gamma$ " + pot_note,
                    fontsize=10)
    ax[0].legend(fontsize=8)

    ax[1].set_xlabel(r"$\cos\theta$", fontsize=12)
    ax[1].set_ylabel("Counts", fontsize=12)
    ax[1].set_xlim(-1, 1); ax[1].set_ylim(bottom=0)
    ax[1].set_title(r"ICARUS scalar Dark Primakoff: $\cos\theta$ wrt beam " + pot_note,
                    fontsize=10)
    ax[1].legend(fontsize=8)

    ax[2].set_xlabel(r"$\cos\theta$", fontsize=12)
    ax[2].set_ylabel("Counts", fontsize=12)
    ax[2].set_xlim(0.80, 1.0); ax[2].set_ylim(bottom=0)
    ax[2].set_title(r"ICARUS scalar Dark Primakoff: $\cos\theta$ zoom $[0.80,1.0]$ " + pot_note,
                    fontsize=10)
    ax[2].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(stem + "_countrate.png", dpi=130)
    print("  Saved -> %s_countrate.png" % stem)
    print("  Done.")


if __name__ == "__main__":
    main()
