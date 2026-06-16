"""
Long-lived PSEUDOSCALAR (a) Dark Primakoff chain at ICARUS (LAr)  —  MULTICHANNEL sum.

Model ii) of Dutta et al. (arXiv:2110.11944): Fig.1(a)-phi production +
Fig.1(c) Dark Primakoff scattering. Reproduces the Fig.2 BOTTOM panels.

Chain (SHORTER than the vector portal -- one secondary vertex, no cascade):
    pi+/K+ -> l+ nu phi        (three-body SCALAR production, validated ME)
    phi propagates to detector
    phi N -> gamma N           (Dark Primakoff, Eq.C2/C3, DarkPrimakoffUpsCase)
    signal = single PHOTON (E_gamma ~ E_phi, coherent recoil keV-MeV)

Adapted from the working vector multichannel script:
  - production mediator_type="pseudoscalar", pdgid_mediator=5919 (a)
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
#  Load physics modules (validated MesonProduction + VectorPortal)     #
# ------------------------------------------------------------------ #
_PROC_DIR = os.path.join(_util.resource_package_dir(), "processes", "DarkNewsTables")
_MESON = _util.load_module("DuttaKim_MesonProduction",
                           os.path.join(_PROC_DIR, "MesonProduction.py"))
_VP = _util.load_module("DuttaKim_VectorPortal",
                        os.path.join(_PROC_DIR, "VectorPortal.py"))
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

# ---- Scalar model (Table I/II): m_phi=1 MeV, m_Z'=49 MeV ----
M_PHI = 1e-3                 # GeV  scalar mediator
M_ZP  = 49e-3               # GeV  Z' (Dark Primakoff mediator)
M_ARGON40  = 37.224          # GeV, Ar40 nuclear mass (ICARUS liquid argon)

# Couplings (Table II scalar benchmark)
G_MU_PROD = 5e-3            # phi-muon production coupling (g_mu)
G_N       = 1e-2           # Z'-nucleon coupling (g_n)
LAMBDA    = 0.44           # GeV^-1  (= 4.4e-4 MeV^-1; unit-converted!)

PT = lambda pdg: dataclasses.Particle.ParticleType(pdg)
PHI = PT(5919)             # scalar mediator (single secondary vertex)

# ICARUS LAr fiducial (active-volume cut via GDML sectors)
R_FID = 5.0          # (legacy, unused for ICARUS sector cut)
R_OIL = 5.746        # (legacy, unused for ICARUS sector cut)
R_LAR_INJECT = 12.0  # fallback injector fiducial radius [m] if ParseFiducialVolume fails
LAR_TARGET_PDGS = {1000180400}                # Ar40
events_to_inject = 10_000

# Reconstruction cuts (single-photon / single-shower selection)
E_VIS_THRESHOLD = 0.140

# ---- Absolute normalization ----
# CALIB_VECTOR: the validated overall constant bridging the C-R production
# width to Dutta-Kim Table II (imported from MesonProduction). Applied once
# per event weight so the production rate is on the Table II scale.
# (Loaded after _MESON import below.)
#
# MiniBooNE delivered POT (neutrino mode, Ref [3]): 6.46e20.
MINIBOONE_POT = 6.46e20
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

# Flux tag for build_phi_flux (keys on neutrino flavor of the production channel).
CHANNEL_FLUX_TAG = {
    "K_e":   "FHC_nue",
    "K_mu":  "FHC_numu",
    "pi_e":  "FHC_nue",
    "pi_mu": "FHC_numu",
}


def _mc(channels, weights):
    m = injection.MultiChannelPhaseSpace()
    m.channels = channels
    m.weights = weights
    return m


def meson_bias(E, px, py, pz, vx, vy, vz):
    """Forward/energetic importance bias for the parent meson.

    NOTE: the vector script used E**2 * cos, tuned to the steeply-falling
    chi spectrum. The scalar phi flux is flatter in energy, so E**2 OVER-biases
    toward high-E forward production and leaves the populated moderate-E /
    moderate-angle region undersampled -> a few events there carry runaway
    importance weights (heavy weight tail). A gentler bias (E**1, softer cos
    floor) matches the phi flux better and keeps the weights flat.
    """
    p = np.sqrt(px**2 + py**2 + pz**2)
    cos_theta = np.divide(pz, p, out=np.zeros_like(p), where=(p > 0))
    r_trans = np.sqrt(vx**2 + vy**2)
    return E**1.5 * np.maximum(cos_theta, 0.02) * np.exp(-r_trans / 200.0)


# ------------------------------------------------------------------ #
#  Channel-parameterized model building (SCALAR production + Primakoff) #
# ------------------------------------------------------------------ #
def build_onshell_models(parent_pdg, m_meson, m_lepton, lepton_pdg, nu_pdg):
    """Scalar chain: meson -> l nu phi (validated scalar ME), then a SINGLE
    Dark Primakoff vertex phi N -> gamma N. No chi'/V1/e+e- cascade."""
    meson_decay = _MESON.MesonThreeBodySIRENDecay(
        m_meson, m_lepton, M_PHI, G_MU_PROD, "pseudoscalar",   # PSEUDOSCALAR production
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


def build_primary_primakoff_phase_spaces(targets, models):
    """Phi-as-primary architecture: the Dark Primakoff phi N -> gamma N is the
    PRIMARY interaction (phi injected directly from build_phi_flux). The phi is
    already aimed at the detector by FixedDirection + PointSource, so the scatter
    phase space is just the physical cross-section channel (no DetectorDirected
    channel -- that one is Decay2Body topology and clashes with Scatter2to2)."""
    m = models["models"]
    prim_sig = m["primakoff"].GetPossibleSignatures()[0]
    return {prim_sig: _mc(
        [injection.PhysicalCrossSectionChannel(m["primakoff"], prim_sig)], [1.0])}


def onshell_stopping_condition(datum, i):
    sec = int(datum.record.signature.secondary_types[i])
    # phi (5919) must scatter (don't stop); photon (22) and nucleus are final.
    if sec == 5919:
        return False          # let phi propagate + Primakoff-scatter
    return True               # stop at photon / nucleus / everything else


def load_dk2nu_mesons(dk2nu_data, parent_pdg, detector_model):
    """PrimaryExternalDistribution of a given parent from already-read dk2nu."""
    return _DK.dk2nu_to_primary_distribution(
        dk2nu_data, detector_model, parent_pdg=parent_pdg,
        sampling_bias=meson_bias)


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
def run_channel(name, dk2nu_data, detector_model, n_events=events_to_inject):
    parent_pdg, m_meson, m_lepton, lepton_pdg, nu_pdg, gamma_sm = CHANNELS[name]
    print("\n" + "=" * 64)
    print("  CHANNEL %s : parent=%d  m_meson=%.4f  m_lepton=%.5f"
          % (name, parent_pdg, m_meson, m_lepton))
    print("=" * 64)

    if (m_meson - m_lepton) <= M_PHI:
        print("  kinematically forbidden -> skip"); return np.array([]), np.array([]), np.array([])

    # Injector fiducial: ICARUS LAr active volume from the GDML.
    # (The physical signal selection is the volTPCActive sector cut below.)
    try:
        fiducial_box = detector_model.ParseFiducialVolume(
            "volTPCActive", "")
    except Exception:
        fiducial_box = siren.geometry.Sphere(R_LAR_INJECT, 0.0)
    targets = build_geometric_targets(detector_model, fiducial_box)
    chain = build_onshell_models(parent_pdg, m_meson, m_lepton, lepton_pdg, nu_pdg)
    meson_decay = chain["meson_decay"]
    primakoff = chain["models"]["primakoff"]

    # --- PHI-AS-PRIMARY ARCHITECTURE (mirrors the vector portal) -------------
    # The K/pi -> l nu phi production is precomputed into a phi flux at the
    # detector (build_phi_flux convolves the parent meson spectrum with the
    # validated three-body differential rate and boosts to lab).  We then
    # inject phi (5919) DIRECTLY with that flux and make the Dark Primakoff
    # phi N -> gamma N the PRIMARY interaction.  This removes the live meson-
    # decay vertex whose isotropic phys/gen mismatch produced the ~1e9 weight
    # spike for forward phi -- exactly how the vector portal stays well-behaved.
    bsm_width = meson_decay._total_width
    br_bsm = bsm_width / gamma_sm
    print("  BSM 3-body width: %.4e GeV   SM 2-body width: %.4e   BR: %.4e"
          % (bsm_width, gamma_sm, br_bsm))

    flux_tag = CHANNEL_FLUX_TAG[name]
    print("  Building phi flux (build_phi_flux, tag=%s) ..." % flux_tag)
    phi_flux = _MESON.build_phi_flux(
        m_meson=m_meson, m_lepton=m_lepton, m_phi=M_PHI,
        g_mu=G_MU_PROD, mediator_type="pseudoscalar",
        flux_tag=flux_tag, min_energy=0.0, max_energy=3.0,
        n_bins=50, physically_normalized=True)

    # Primakoff as a PRIMARY process, detector-directed toward the targets.
    primary_primakoff_ps = build_primary_primakoff_phase_spaces(targets, chain)

    # Injection distributions: mass + flux + direction + position.
    # Physical distributions: same MINUS position, PLUS the BSM branching-ratio
    # normalization that the precomputed flux does not itself carry.
    from siren.math import Vector3D as _V3
    _SRC = [0.0, 0.0, 0.0]
    _MAXD = 50.0
    primary_injection_distributions = [
        distributions.PrimaryMass(M_PHI), phi_flux,
        distributions.FixedDirection(_V3(0.0, 0.0, 1.0)),
        distributions.PointSourcePositionDistribution(_SRC, _MAXD),
    ]
    br_dist = distributions.NormalizationConstant(br_bsm)
    primary_physical_distributions = [
        distributions.PrimaryMass(M_PHI), phi_flux,
        distributions.FixedDirection(_V3(0.0, 0.0, 1.0)),
        br_dist,
    ]

    print("  Building injector (%d events) ..." % n_events)
    injector = Injector(
        number_of_events=n_events, detector_model=detector_model, seed=42,
        primary_type=PHI, primary_interactions=[primakoff],
        primary_injection_distributions=primary_injection_distributions,
        primary_phase_spaces=primary_primakoff_ps,
    )
    # Force-init, tolerant of a pathological first event.
    try:
        for ev in injector:
            break
    except RuntimeError as e:
        print("  (force-init first event threw: %r — continuing)" % e)

    weighter = Weighter(
        injectors=[injector], detector_model=detector_model,
        primary_type=PHI, primary_interactions=[primakoff],
        primary_physical_distributions=primary_physical_distributions,
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

    # Normalize the weighter by the ACTUAL number of events generated, so the
    # rate is invariant to n_events (acceptance-limited generation otherwise
    # makes sum(w) scale as 1/n_events). Use the raw generated count = events
    # that came out of the injector (successful GenerateEvent calls).
    n_generated = len(events)
    try:
        injector._Injector__injector.ResetInjectedEvents(max(n_generated, 1))
    except Exception:
        pass

    raw = np.array([weighter(ev) for ev in events])
    # Print vertex/sector for the first few events to confirm where the
    # Primakoff scatter actually lands (geometry-offset diagnosis).
    globals()["_DEBUG_LAR"] = True
    for _ev in events[:5]:
        primakoff_in_lar(_ev, detector_model, debug=True)
    globals()["_DEBUG_LAR"] = False
    n_lar = int(np.sum([primakoff_in_lar(ev, detector_model) for ev in events]))
    print("  [diag] Primakoff gamma on Ar in LAr active: %d" % n_lar)

    # ---- TEMP normalization diagnostic ----
    _prim = chain["models"]["primakoff"]
    print("  [diag] standalone sigma(0.5 GeV) = %.4e cm^2"
          % _prim._dp.total_xsec(0.5))
    _pos = raw[np.isfinite(raw) & (raw > 0)]
    if len(_pos):
        print("  [diag] raw weight: min=%.3e median=%.3e max=%.3e mean=%.3e"
              % (_pos.min(), np.median(_pos), _pos.max(), _pos.mean()))
        _med = np.median(_pos)
        # Does a single event dominate the sum? (one-event-estimate test)
        _sum = _pos.sum()
        _topfrac = _pos.max() / _sum if _sum > 0 else float("nan")
        print("  [diag] top event is %.1f%% of sum(raw>0)   (N_pos=%d)"
              % (100.0 * _topfrac, len(_pos)))
        # Identify and dump the kinematics of the single max-weight event.
        # raw can contain non-finite/<=0 entries, so mask before argmax.
        _safe = np.where(np.isfinite(raw) & (raw > 0), raw, -np.inf)
        _imax = int(np.argmax(_safe))
        _ev = events[_imax]
        print("  [diag] MAX-weight event idx=%d  w=%.3e  (%.0fx median)"
              % (_imax, raw[_imax], raw[_imax] / _med if _med > 0 else float("nan")))
        # Pull the PRIMARY meson 4-momentum from the first datum's record,
        # using the same record API the observable extractors use.
        try:
            _prec = _ev.tree[0].record
            _pmom = _prec.primary_momentum   # [E, px, py, pz] in GeV
            _E  = _pmom[0]
            _px, _py, _pz = _pmom[1], _pmom[2], _pmom[3]
            _pmag = math.sqrt(_px**2 + _py**2 + _pz**2)
            _cth = (_pz / _pmag) if _pmag > 0 else 0.0
            try:
                _vtx = _prec.interaction_vertex
                _rt = math.sqrt(_vtx[0]**2 + _vtx[1]**2)
            except Exception:
                _rt = float("nan")
            _ptype = int(_prec.signature.primary_type)
            print("  [diag]   primary pdg=%d  E=%.4f GeV  cos_theta=%.5f  "
                  "r_trans=%.3f m  bias~%.3e"
                  % (_ptype, _E, _cth, _rt,
                     (_E**1.5 * max(_cth, 0.02) * math.exp(-_rt / 200.0))))
        except Exception as _e:
            print("  [diag]   (could not read primary kinematics: %r)" % _e)

        # --- Decompose the max-weight event over its interaction tree. ---
        # The runaway is suspected to be a Lorentz-boost / phase-space Jacobian
        # on the ULTRALIGHT phi (M_PHI=1 MeV -> gamma_phi = E_phi/M_PHI can be
        # ~1000s for a multi-GeV forward phi). Print each vertex's secondaries,
        # their energies, and the phi boost factor so the blown-up vertex is
        # obvious.
        try:
            print("  [diag]   --- tree decomposition of max-weight event ---")
            for _di, _datum in enumerate(_ev.tree):
                _r = _datum.record
                _sig = _r.signature
                _prim_t = int(_sig.primary_type)
                _sec_t = [int(s) for s in _sig.secondary_types]
                try:
                    _pe = _r.primary_momentum[0]
                except Exception:
                    _pe = float("nan")
                # gamma factor if the primary at this vertex is the phi (5919)
                _gam = (_pe / M_PHI) if (_prim_t == 5919 and M_PHI > 0) else float("nan")
                _se = []
                try:
                    for _i in range(len(_sec_t)):
                        _se.append(_r.secondary_momenta[_i][0])
                except Exception:
                    pass
                print("  [diag]     vtx%d  prim=%d (E=%.4f, gamma_phi=%.1f)  "
                      "secs=%s  E_secs=%s"
                      % (_di, _prim_t, _pe, _gam, _sec_t,
                         ["%.4f" % e for e in _se]))
                for _i, _e2 in enumerate(_se):
                    if np.isfinite(_e2) and _e2 > 50.0:   # >50 GeV is unphysical here
                        _stp = _sec_t[_i] if _i < len(_sec_t) else -1
                        print("  [diag]       !! secondary %d E=%.3f GeV "
                              "EXCEEDS sane range -> boost/Jacobian blow-up"
                              % (_stp, _e2))
        except Exception as _e:
            print("  [diag]   (tree decomposition failed: %r)" % _e)
    # ---- end diagnostic ----

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
        # Absolute normalization: production CALIB x POT x efficiency.
        # The g_mu^2 production coupling and g_n^2 lambda^2 Primakoff coupling
        # are already inside the matrix elements; CALIB bridges the C-R
        # production convention; POT scales the per-POT dk2nu flux.
        w_abs = (w * CALIB_SCALAR * MINIBOONE_POT
                 * detection_efficiency(E_vis))
        if not np.isfinite(w_abs) or w_abs <= 0:
            continue
        Ev.append(E_vis); cs.append(obs[1]); wv.append(w_abs)
    Ev, cs, wv = np.array(Ev), np.array(cs), np.array(wv)
    print("  [result] plottable signal: %d   sum(w_abs)=%.3e events"
          % (len(Ev), wv.sum() if len(wv) else 0.0))
    return Ev, cs, wv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", choices=list(CHANNELS) + ["all"], default="all")
    ap.add_argument("--n-events", type=int, default=events_to_inject)
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
        per_channel[name] = run_channel(name, dk2nu_data, detector_model, args.n_events)

    E_all = np.concatenate([per_channel[n][0] for n in per_channel]) \
            if any(len(per_channel[n][0]) for n in per_channel) else np.array([])
    c_all = np.concatenate([per_channel[n][1] for n in per_channel]) if E_all.size else np.array([])
    w_all = np.concatenate([per_channel[n][2] for n in per_channel]) if E_all.size else np.array([])

    print("\n" + "=" * 64)
    print("  MULTICHANNEL SUM  —  ICARUS Pseudoscalar Dark Primakoff")
    print("=" * 64)
    for n in per_channel:
        Ev, cs, wv = per_channel[n]
        print("  %-6s : %4d events   sum(w)=%.3e"
              % (n, len(Ev), wv.sum() if len(wv) else 0.0))
    print("  %-6s : %4d events   sum(w)=%.3e"
          % ("TOTAL", len(E_all), w_all.sum() if w_all.size else 0.0))
    print("=" * 64)

    os.makedirs("output", exist_ok=True)
    stem = "output/ICARUS_PseudoscalarPrimakoff_multichannel"
    np.savez(stem + "_observables.npz", E_vis=E_all, cos_theta=c_all, weight=w_all,
             **{f"{n}_E": per_channel[n][0] for n in per_channel},
             **{f"{n}_w": per_channel[n][2] for n in per_channel})
    print("  Saved -> %s_observables.npz" % stem)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    E_bins = np.linspace(0.140, 2.4, 40); c_bins = np.linspace(-1, 1, 40)
    colors = {"K_e": "C0", "K_mu": "C1", "pi_e": "C2", "pi_mu": "C3"}
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for n in per_channel:
        Ev, cs, wv = per_channel[n]
        if len(Ev):
            ax[0].hist(Ev*1e3, bins=E_bins*1e3, weights=wv, histtype="step",
                       color=colors.get(n), label=n)
            ax[1].hist(cs, bins=c_bins, weights=wv, histtype="step",
                       color=colors.get(n), label=n)
    if E_all.size:
        ax[0].hist(E_all*1e3, bins=E_bins*1e3, weights=w_all, histtype="step",
                   color="k", lw=2, label="TOTAL")
        ax[1].hist(c_all, bins=c_bins, weights=w_all, histtype="step",
                   color="k", lw=2, label="TOTAL")
    ax[0].set_xlabel("E_gamma [MeV]"); ax[0].set_ylabel("Events / %g POT" % MINIBOONE_POT)
    ax[0].set_title("ICARUS Pseudoscalar Primakoff : photon energy"); ax[0].legend(fontsize=8)
    ax[1].set_xlabel("cos(theta) wrt beam"); ax[1].set_ylabel("Events / %g POT" % MINIBOONE_POT)
    ax[1].set_title("ICARUS Pseudoscalar Primakoff : photon angular"); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(stem + "_countrate.png", dpi=130)
    print("  Saved -> %s_countrate.png" % stem)
    print("  Done.")


if __name__ == "__main__":
    main()
