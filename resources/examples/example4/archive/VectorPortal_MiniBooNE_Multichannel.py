"""
Vector-portal dark matter at MiniBooNE  —  MULTICHANNEL sum.

Sums the four charged-meson production channels (Eq. 3 of arXiv:2110.11944):
    K+  -> e+  nu_e  V1
    K+  -> mu+ nu_mu V1
    pi+ -> e+  nu_e  V1
    pi+ -> mu+ nu_mu V1
each through the validated vector three-body production (C-R Eq.27, anchored
to Table II) inside the factory's compute_chi_flux, then the same upscatter
-> chi' -> V1 -> e+e- chain, the same fiducial + oil + 140 MeV + 10deg cuts,
and sums the weighted E_vis and cos(theta) histograms.

Built by mirroring VectorPortal_MiniBooNE_kaon_eplus.py exactly, wrapped in a
per-channel loop.

CAVEAT (tabulated-flux parent mixing): this uses the tabulated PionKaon flux
keyed on neutrino flavor (FHC_nue / FHC_numu). That flux already mixes K and
pi parents, so the K-vs-pi ABSOLUTE normalization carries the proxy
approximation of the original single-channel script. The RELATIVE e-vs-mu
weighting (which the validated production BR controls) is correct. For exact
K/pi normalization, switch to the dk2nu route (compute_chi_flux_from_dk2nu).
"""

import os
import math
import numpy as np

import siren
from siren import utilities
from siren import _util as _siren_util

# ---------------------------------------------------------------------------
# Load the vector-portal factory
# ---------------------------------------------------------------------------
_dt_base = os.path.join(
    _siren_util.resource_package_dir(), "processes", "DarkNewsTables",
)
_mod_processes = _siren_util.load_module(
    "siren.resources.processes.DarkNewsTables.processes",
    os.path.join(_dt_base, "processes.py"),
)
load_vector_portal = _mod_processes.load_vector_portal

# ---------------------------------------------------------------------------
# Model parameters (Dutta et al. Table I, double-mediator benchmark)
# ---------------------------------------------------------------------------
M_CHI       = 8e-3
M_CHI_PRIME = 50e-3
M_V1        = 17e-3
M_V2        = 200e-3
EPSILON_1   = 7e-5
EPSILON_2   = 1e-4
G_D         = 1.0

PDGID_CHI       = 5917
PDGID_CHI_PRIME = 5918
PDGID_V1        = 5922

events_to_inject = 10_000
experiment       = "MiniBooNE"
MAX_ENERGY       = 3.0
R_FID            = 5.0
R_OIL            = 5.746
OIL_TARGET_PDGS  = {1000060120, 1000010010}   # C12, H1

# Particle masses for the channels
M_K  = 0.49368
M_PI = 0.13957039
M_E  = 0.000511
M_MU = 0.10565837

# ---------------------------------------------------------------------------
# The four production channels: (name, m_meson, m_lepton, flux_tag)
# flux_tag keys on the associated neutrino flavor:
#   e-channels  -> nu_e  -> FHC_nue
#   mu-channels -> nu_mu -> FHC_numu
# ---------------------------------------------------------------------------
CHANNELS = [
    ("K_e",   M_K,  M_E,  "FHC_nue"),
    ("K_mu",  M_K,  M_MU, "FHC_numu"),
    ("pi_e",  M_PI, M_E,  "FHC_nue"),
    ("pi_mu", M_PI, M_MU, "FHC_numu"),
]

output_stem = "output/MiniBooNE_VectorPortal_multichannel"
os.makedirs("output", exist_ok=True)

# ---------------------------------------------------------------------------
# Detector + particle types (shared across channels)
# ---------------------------------------------------------------------------
print("Loading MiniBooNE detector model (GDML) ...")
detector_model = utilities.load_detector("SBN", detector=experiment)

chi_type       = siren.dataclasses.Particle.ParticleType(PDGID_CHI)
chi_prime_type = siren.dataclasses.Particle.ParticleType(PDGID_CHI_PRIME)
v1_type        = siren.dataclasses.Particle.ParticleType(PDGID_V1)

fiducial_volume = siren.geometry.Sphere(R_FID, 0.0)
oil_volume      = siren.geometry.Sphere(R_OIL, 0.0)

from siren import distributions as _dists
from siren.math import Vector3D as _Vec3
from siren.math import Vector3D as _MV3

_SOURCE_POINT = [0.0, 0.0, 0.0]
_MAX_DISTANCE = 50.0

# MiniBooNE reconstruction cuts
E_VIS_THRESHOLD = 0.140     # GeV
E_PAIR_MAX_DEG  = 10.0      # deg
_EFF_TABLE = None           # digitize refs [76,77]; None -> efficiency 1.0

def detection_efficiency(E_vis_gev):
    if _EFF_TABLE is None:
        return 1.0
    E = np.asarray(_EFF_TABLE)[:, 0]; eff = np.asarray(_EFF_TABLE)[:, 1]
    return float(np.interp(E_vis_gev, E, eff, left=eff[0], right=eff[-1]))


# ===========================================================================
# Cut / observable helpers (identical to the single-channel script)
# ===========================================================================
def signal_in_fiducial(event):
    try:
        for datum in event.tree:
            r = datum.record
            secs = [int(s) for s in r.signature.secondary_types]
            if 11 in secs and -11 in secs:
                vtx = _MV3(r.interaction_vertex[0], r.interaction_vertex[1],
                           r.interaction_vertex[2])
                if fiducial_volume.IsInside(vtx):
                    return True
    except Exception:
        return False
    return False

def upscatter_in_oil(event):
    try:
        for datum in event.tree:
            r = datum.record
            secs = [int(s) for s in r.signature.secondary_types]
            if 5918 in secs and any(s in OIL_TARGET_PDGS for s in secs):
                vtx = _MV3(r.interaction_vertex[0], r.interaction_vertex[1],
                           r.interaction_vertex[2])
                if oil_volume.IsInside(vtx):
                    return True
    except Exception:
        return False
    return False

def signal_passes(event):
    return upscatter_in_oil(event) and signal_in_fiducial(event)

def signal_eepair_observables(event):
    """(E_vis, cos_theta) of the e+e- pair from a single fiducial V1->e+e-
    vertex, AFTER the 140 MeV threshold and <10deg opening-angle cuts.
    Returns None if no vertex passes."""
    try:
        for datum in event.tree:
            r = datum.record
            secs = [int(s) for s in r.signature.secondary_types]
            if not (11 in secs and -11 in secs):
                continue
            vtx = _MV3(r.interaction_vertex[0], r.interaction_vertex[1],
                       r.interaction_vertex[2])
            if not fiducial_volume.IsInside(vtx):
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
            cos_open = float(np.dot(p1, p2) / (n1 * n2))
            cos_open = max(-1.0, min(1.0, cos_open))
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


# ===========================================================================
# Per-channel run: factory -> injector -> events -> weighter -> cuts -> obs
# ===========================================================================
def run_channel(name, m_meson, m_lepton, flux_tag):
    print("\n" + "=" * 64)
    print("  CHANNEL: %s   (m_meson=%.4f, m_lepton=%.5f, flux=%s)"
          % (name, m_meson, m_lepton, flux_tag))
    print("=" * 64)

    # kinematic guard
    if (m_meson - m_lepton) <= M_V1:
        print("  channel kinematically forbidden -> skipping")
        return np.array([]), np.array([]), np.array([])

    print("  Building vector-portal processes (factory) ...")
    try:
        primary_processes, secondary_processes, chi_flux = load_vector_portal(
            m_chi=M_CHI, m_chi_prime=M_CHI_PRIME, m_V1=M_V1, m_V2=M_V2,
            g_D=G_D, epsilon_1=EPSILON_1, epsilon_2=EPSILON_2,
            detector_model=detector_model, flux_tag=flux_tag,
            m_meson=m_meson, m_lepton=m_lepton, max_energy=MAX_ENERGY,
        )
    except Exception as e:
        print("  factory failed for this channel: %r -> skipping" % e)
        return np.array([]), np.array([]), np.array([])

    n_targets = len(primary_processes.get(chi_type, []))
    print("  Upscattering targets: %d" % n_targets)
    if n_targets == 0:
        print("  no targets -> skipping")
        return np.array([]), np.array([]), np.array([])

    # injection distributions
    primary_injection_distributions = [
        _dists.PrimaryMass(M_CHI), chi_flux,
        _dists.FixedDirection(_Vec3(0.0, 0.0, 1.0)),
        _dists.PointSourcePositionDistribution(_SOURCE_POINT, _MAX_DISTANCE),
    ]
    primary_physical_distributions = [
        _dists.PrimaryMass(M_CHI), chi_flux,
        _dists.FixedDirection(_Vec3(0.0, 0.0, 1.0)),
    ]
    _sv = _dists.SecondaryPhysicalVertexDistribution()
    _sv_bounded = _dists.SecondaryBoundedVertexDistribution(fiducial_volume)
    secondary_injection_distributions = {pt: [_sv] for pt in secondary_processes}
    if chi_type in secondary_injection_distributions:
        secondary_injection_distributions[chi_type] = [_sv_bounded]
    secondary_physical_distributions = {pt: [_sv] for pt in secondary_processes}
    if chi_type in secondary_physical_distributions:
        secondary_physical_distributions[chi_type] = [_sv_bounded]

    def onshell_stopping_condition(datum, i):
        sec    = int(datum.record.signature.secondary_types[i])
        parent = int(datum.record.signature.primary_type)
        if sec == 5922:  return False
        if sec == 5917:  return parent != 5922 or i != 0
        if sec == 5918:  return False
        return True

    print("  Building injector (%d events) ..." % events_to_inject)
    injector = siren.injection.Injector()
    injector.number_of_events                  = events_to_inject
    injector.detector_model                    = detector_model
    injector.primary_type                      = chi_type
    injector.primary_interactions              = primary_processes[chi_type]
    injector.primary_injection_distributions   = primary_injection_distributions
    injector.secondary_interactions            = secondary_processes
    injector.secondary_injection_distributions = secondary_injection_distributions
    injector.stopping_condition                = onshell_stopping_condition

    print("  Generating events ...")
    events = []
    try:
        for event in injector:
            events.append(event)
    except StopIteration:
        pass
    except Exception as e:
        print("  (iteration failed: %r — trying generate_event)" % e)
        while len(events) < events_to_inject:
            try:
                events.append(injector.generate_event())
            except Exception:
                break
    print("  Generated %d events." % len(events))

    weighter = siren.injection.Weighter()
    weighter.injectors                        = [injector]
    weighter.detector_model                   = detector_model
    weighter.primary_type                     = chi_type
    weighter.primary_interactions             = primary_processes[chi_type]
    weighter.primary_physical_distributions   = primary_physical_distributions
    weighter.secondary_interactions           = secondary_processes
    weighter.secondary_physical_distributions = secondary_physical_distributions

    raw_weights = []
    for ev in events:
        try:
            w = weighter(ev)
        except Exception:
            w = 0.0
        if not np.isfinite(w):
            w = 0.0
        raw_weights.append(w)
    raw_weights = np.array(raw_weights)

    fid_mask = np.array([signal_passes(ev) for ev in events])
    n_oil  = int(np.sum([upscatter_in_oil(ev) for ev in events]))
    n_eevx = int(np.sum([signal_in_fiducial(ev) for ev in events]))
    print("  [diag] upscatter on C/H in oil : %d" % n_oil)
    print("  [diag] e+e- vertex in fiducial  : %d" % n_eevx)
    print("  [diag] BOTH (signal selection)  : %d" % int(fid_mask.sum()))
    sel_weights = np.where(fid_mask, raw_weights, 0.0)

    # observables (with the 140 MeV + 10deg cuts + efficiency)
    Ev, cs, wv = [], [], []
    for ev, w in zip(events, sel_weights):
        if w <= 0:
            continue
        obs = signal_eepair_observables(ev)
        if obs is None:
            continue
        weff = w * detection_efficiency(obs[0])
        if weff <= 0:
            continue
        Ev.append(obs[0]); cs.append(obs[1]); wv.append(weff)
    Ev, cs, wv = np.array(Ev), np.array(cs), np.array(wv)
    print("  [result] plottable signal events: %d   sum(w)=%.3e"
          % (len(Ev), wv.sum() if len(wv) else 0.0))
    return Ev, cs, wv


# ===========================================================================
# Run all channels and sum
# ===========================================================================
per_channel = {}
for name, mM, mL, tag in CHANNELS:
    Ev, cs, wv = run_channel(name, mM, mL, tag)
    per_channel[name] = (Ev, cs, wv)

# combined arrays
E_all  = np.concatenate([per_channel[n][0] for n in per_channel]) \
         if any(len(per_channel[n][0]) for n in per_channel) else np.array([])
c_all  = np.concatenate([per_channel[n][1] for n in per_channel]) \
         if E_all.size else np.array([])
w_all  = np.concatenate([per_channel[n][2] for n in per_channel]) \
         if E_all.size else np.array([])

print("\n" + "=" * 64)
print("  MULTICHANNEL SUM  —  MiniBooNE Vector Portal")
print("=" * 64)
for name in per_channel:
    Ev, cs, wv = per_channel[name]
    print("  %-6s : %4d events   sum(w)=%.3e"
          % (name, len(Ev), wv.sum() if len(wv) else 0.0))
print("  %-6s : %4d events   sum(w)=%.3e"
      % ("TOTAL", len(E_all), w_all.sum() if w_all.size else 0.0))
print("=" * 64)

# save combined + per-channel observables
np.savez(output_stem + "_observables.npz",
         E_vis=E_all, cos_theta=c_all, weight=w_all,
         **{f"{n}_E": per_channel[n][0] for n in per_channel},
         **{f"{n}_w": per_channel[n][2] for n in per_channel})
print("  Saved observables -> %s_observables.npz" % output_stem)

# ---------------------------------------------------------------------------
# Plot: stacked/overlaid per-channel + total
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

E_bins  = np.linspace(0.140, 2.4, 40)
c_bins  = np.linspace(-1.0, 1.0, 40)
colors  = {"K_e": "C0", "K_mu": "C1", "pi_e": "C2", "pi_mu": "C3"}

fig, ax = plt.subplots(1, 2, figsize=(13, 5))

# E_vis: per-channel step + total
for name in per_channel:
    Ev, cs, wv = per_channel[name]
    if len(Ev):
        ax[0].hist(Ev * 1e3, bins=E_bins * 1e3, weights=wv, histtype="step",
                   color=colors.get(name), label=name)
if E_all.size:
    ax[0].hist(E_all * 1e3, bins=E_bins * 1e3, weights=w_all, histtype="step",
               color="k", lw=2, label="TOTAL")
ax[0].set_xlabel("E_vis (e+e-)  [MeV]")
ax[0].set_ylabel("Weighted events")
ax[0].set_title("MiniBooNE multichannel : visible energy")
ax[0].legend(fontsize=8)

# cos(theta)
for name in per_channel:
    Ev, cs, wv = per_channel[name]
    if len(cs):
        ax[1].hist(cs, bins=c_bins, weights=wv, histtype="step",
                   color=colors.get(name), label=name)
if c_all.size:
    ax[1].hist(c_all, bins=c_bins, weights=w_all, histtype="step",
               color="k", lw=2, label="TOTAL")
ax[1].set_xlabel("cos(theta)  wrt beam")
ax[1].set_ylabel("Weighted events")
ax[1].set_title("MiniBooNE multichannel : angular")
ax[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(output_stem + "_countrate.png", dpi=130)
print("  Saved count-rate plots -> %s_countrate.png" % output_stem)
print("  Done.")
