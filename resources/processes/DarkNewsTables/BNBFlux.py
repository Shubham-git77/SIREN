"""
BNB (Booster Neutrino Beam) parent-meson flux generator for SIREN.

Goal: produce a BNB pi+/-, K+ parent-meson sample in the SAME format SIREN's
NuMI dk2nu path uses ( E,px,py,pz,x0,y0,z0,m,ptype,nimpwt ), so the example4
Dark-Primakoff scripts can inject mesons from the Booster beam at MiniBooNE
exactly as they do from NuMI at ICARUS.

Physics (MiniBooNE flux paper, Phys.Rev.D 79, 072002 = arXiv:0806.1449):
  * 8.89 GeV/c protons on a beryllium target.
  * pi+/- production: Sanford-Wang double-differential cross section (Eq. 11),
      d2sigma/dp dOmega = c1 p^c2 (1 - p/(pB-c9))
                          * exp[ -c3 p^c4 / pB^c5 - c6 theta (p - c7 pB cos^c8 theta) ]
    (theta in rad, p in GeV/c, pB = 8.89 GeV/c; for pions c9 = 1, c3 fixed).
  * K+ production: Feynman-scaling parameterization (Eq. 15),
      d2sigma/dp dOmega = (p^2/E) * c1 (1-|xF|)
                          * exp[ -c2 pT - c3 |xF|^c4 - c5 pT^2 - c7 |pT xF|^c6 ]
    with xF = pL*/(sqrt(s)/2) (CM longitudinal momentum scaled), pT transverse.

STAGE 1 (this file, for now): the production model + a rejection sampler over
(p, theta).  Geometry / horn focusing / decay / CSV writing / neutrino-flux
validation are added in later stages.

NB: the absolute production normalization (mb / GeV / sr) is only meaningful up
to the overall constant that the later neutrino-flux validation fixes; the SHAPE
is what these parameterizations provide.
"""

import math
import numpy as np

# ------------------------------------------------------------------ #
#  Beam / particle constants                                          #
# ------------------------------------------------------------------ #
P_BEAM   = 8.89          # GeV/c, BNB primary proton momentum
M_PROTON = 0.9382720813  # GeV
M_NEUTRON= 0.9395654133  # GeV
M_PION   = 0.13957039    # GeV
M_KAON   = 0.493677      # GeV

# Sanford-Wang parameters c1..c9 (c9=1 fixed for pions), Phys.Rev.D 79, 072002.
SW_PARAMS = {
    "pi+": (220.7, 1.080, 1.000, 1.978, 1.32,  5.572, 0.0868,  9.686, 1.0),
    "pi-": (213.7, 0.9379, 5.454, 1.210, 1.284, 4.781, 0.07338, 8.329, 1.0),
}

# Feynman-scaling parameters c1..c7 for K+ (Table VIII).
FS_KAON = (15.130, 1.975, 4.084, 0.928, 0.731, 4.362, 0.048)

# pdg codes
PDG = {"pi+": 211, "pi-": -211, "K+": 321}
MASS = {"pi+": M_PION, "pi-": M_PION, "K+": M_KAON}


# ------------------------------------------------------------------ #
#  Double-differential production cross sections  d2sigma/dp dOmega   #
#  (arbitrary overall normalization; SHAPE is what matters)           #
# ------------------------------------------------------------------ #
def dsigma_sw(p, theta, species):
    """Sanford-Wang d2sigma/dp dOmega for pi+ / pi- [arb. units].

    p     : meson momentum [GeV/c]   (scalar or ndarray)
    theta : lab angle wrt beam [rad] (scalar or ndarray)
    """
    c1, c2, c3, c4, c5, c6, c7, c8, c9 = SW_PARAMS[species]
    p = np.asarray(p, dtype=float)
    theta = np.asarray(theta, dtype=float)
    # kinematic cutoff: the (1 - p/(pB - c9)) factor must stay >= 0
    bracket = 1.0 - p / (P_BEAM - c9)
    expo = (-c3 * np.power(p, c4) / np.power(P_BEAM, c5)
            - c6 * theta * (p - c7 * P_BEAM * np.power(np.cos(theta), c8)))
    val = c1 * np.power(p, c2) * bracket * np.exp(expo)
    return np.where(bracket > 0.0, np.maximum(val, 0.0), 0.0)


def _kaon_xF_pT(p, theta):
    """Feynman x (scaled CM longitudinal momentum) and transverse momentum pT
    for a K+ produced at lab momentum p, angle theta, on a nucleon at rest."""
    E = np.sqrt(p * p + M_KAON * M_KAON)
    pL = p * np.cos(theta)
    pT = p * np.sin(theta)
    # p-nucleon CM (target nucleon at rest)
    E_beam = math.sqrt(P_BEAM * P_BEAM + M_PROTON * M_PROTON)
    s = M_PROTON ** 2 + M_NEUTRON ** 2 + 2.0 * M_NEUTRON * E_beam
    roots = math.sqrt(s)
    beta = P_BEAM / (E_beam + M_NEUTRON)
    gamma = (E_beam + M_NEUTRON) / roots
    pL_star = gamma * (pL - beta * E)
    xF = pL_star / (roots / 2.0)          # scaled to max CM longitudinal momentum
    return xF, pT


def dsigma_kaon(p, theta):
    """Feynman-scaling d2sigma/dp dOmega for K+ [arb. units]."""
    c1, c2, c3, c4, c5, c6, c7 = FS_KAON
    p = np.asarray(p, dtype=float)
    theta = np.asarray(theta, dtype=float)
    E = np.sqrt(p * p + M_KAON * M_KAON)
    xF, pT = _kaon_xF_pT(p, theta)
    axF = np.abs(xF)
    expo = (-c2 * pT - c3 * np.power(axF, c4) - c5 * pT * pT
            - c7 * np.power(np.abs(pT * xF), c6))
    val = (p * p / E) * c1 * (1.0 - axF) * np.exp(expo)
    return np.where(axF < 1.0, np.maximum(val, 0.0), 0.0)


def dsigma(p, theta, species):
    """Dispatch to the right production model for a species."""
    if species in SW_PARAMS:
        return dsigma_sw(p, theta, species)
    elif species == "K+":
        return dsigma_kaon(p, theta)
    raise ValueError("unknown species %r" % species)


# ------------------------------------------------------------------ #
#  Rejection sampler over (p, theta) ~ d2sigma/dp dOmega * sin(theta) #
#  (sin theta is the solid-angle Jacobian dOmega = sin th dth dphi)   #
# ------------------------------------------------------------------ #
def sample_production(species, n, theta_max=0.35, rng=None, batch=200000):
    """Draw n mesons (p, theta, phi) from the production distribution.

    theta_max [rad]: only sample the forward cone that the horn/decay-pipe can
    deliver to the detector (the BNB decay region + 541 m baseline accept a few
    hundred mrad of production angle; high-angle mesons miss). Returns arrays
    p[GeV/c], theta[rad], phi[rad].
    """
    if rng is None:
        rng = np.random.default_rng(0)
    p_hi = P_BEAM
    # find an envelope constant M >= f(p,theta) on a grid
    pg = np.linspace(1e-3, p_hi, 400)
    tg = np.linspace(0.0, theta_max, 200)
    PG, TG = np.meshgrid(pg, tg, indexing="ij")
    F = dsigma(PG, TG, species) * np.sin(TG)
    M = 1.25 * float(np.max(F))
    if M <= 0:
        raise RuntimeError("envelope is zero for %s" % species)

    out_p, out_t = [], []
    need = n
    while need > 0:
        pp = rng.uniform(1e-3, p_hi, batch)
        tt = rng.uniform(0.0, theta_max, batch)
        ff = dsigma(pp, tt, species) * np.sin(tt)
        keep = rng.uniform(0.0, M, batch) < ff
        out_p.append(pp[keep]); out_t.append(tt[keep])
        need -= int(np.sum(keep))
    p = np.concatenate(out_p)[:n]
    th = np.concatenate(out_t)[:n]
    ph = rng.uniform(0.0, 2.0 * math.pi, n)
    return p, th, ph


# ------------------------------------------------------------------ #
#  Stage 2: geometry, horn focusing, decay, meson sample              #
# ------------------------------------------------------------------ #
# BNB beamline geometry (SBN loader: bsim/G4BNB survey, PRD 79 072002).
L_DECAY_PIPE = 50.0                              # m, decay region length
DET_CENTER   = np.array([0.0, 1.896, 541.34])    # m, MiniBooNE center in BNB coords
DET_RADIUS   = 6.096                             # m, oil-tank inner radius
CTAU = {"pi+": 7.8045, "pi-": 7.8045, "K+": 3.711}   # m, c*tau
# Horn kick + flux normalization CALIBRATED in Stage 3 against the shipped
# BNB_FHC.dat numu flux (validate_bnb_flux.py): PT_KICK=0.35 GeV/c reproduces
# the spectrum shape (peak ~0.5 GeV) and BNB_FLUX_NORM maps the arb production
# units onto physical mesons/POT (total numu flux then matches data to ~4%).
PT_KICK_DEFAULT = 0.35                            # GeV/c, calibrated horn transverse kick
BNB_FLUX_NORM   = 2.4322e-3                       # arb -> mesons/POT (from numu-flux fit)
# Right-sign (focused) species in FHC = positive horn current.
FOCUS_SIGN = {"pi+": +1.0, "K+": +1.0, "pi-": -1.0}  # +1 focus, -1 defocus


def total_xsec(species, theta_max=0.35, n_p=300, n_t=200):
    """Integrate d2sigma/dp dOmega * sin(theta) over the forward cone -> the
    production cross section into the cone [arb. mb units, COMPARABLE across
    species since the SW/Feynman fits are in the same physical units]. Sets the
    relative pi+ : pi- : K+ composition of the sample."""
    pg = np.linspace(1e-3, P_BEAM, n_p)
    tg = np.linspace(0.0, theta_max, n_t)
    PG, TG = np.meshgrid(pg, tg, indexing="ij")
    integrand = dsigma(PG, TG, species) * np.sin(TG)
    # 2*pi from the trivial phi integral
    return 2.0 * math.pi * np.trapz(np.trapz(integrand, tg, axis=1), pg)


def _apply_horn(p, theta, phi, species, pt_kick):
    """Idealized thin-horn focusing: a transverse-momentum kick pt_kick toward
    the axis rotates the meson by dtheta = pt_kick/p (|p| conserved). Right-sign
    mesons focus (theta decreases), wrong-sign defocus. Returns unit direction."""
    sign = FOCUS_SIGN[species]
    dtheta = sign * pt_kick / np.maximum(p, 1e-6)
    th2 = theta - dtheta
    ph2 = np.where(th2 < 0.0, phi + math.pi, phi)
    th2 = np.abs(th2)
    dx = np.sin(th2) * np.cos(ph2)
    dy = np.sin(th2) * np.sin(ph2)
    dz = np.cos(th2)
    return np.stack([dx, dy, dz], axis=1)


def _sample_decay(p, m, species, direction, rng):
    """Sample a decay point inside the decay pipe and the decay-in-pipe weight.

    Path length s ~ Exp(L) with L = (p/m) c*tau, restricted to [0, L_pipe] via
    inverse-CDF; weight = P_pipe = 1 - exp(-L_pipe/L) (fraction that decay before
    the beam stop). Position = s * direction (target at origin, m)."""
    betagamma = p / m
    L = betagamma * CTAU[species]                       # lab decay length [m]
    P_pipe = 1.0 - np.exp(-L_DECAY_PIPE / L)
    u = rng.uniform(0.0, 1.0, len(p))
    s = -L * np.log(1.0 - u * P_pipe)                   # decay path in [0, L_pipe]
    pos = direction * s[:, None]                        # [m]
    return pos, P_pipe


def generate_bnb_sample(n_per_species=200000, pot=1.0, pt_kick=PT_KICK_DEFAULT,
                        theta_max=0.35, seed=0,
                        species_list=("pi+", "pi-", "K+")):
    """Generate a BNB parent-meson sample as a dk2nu_data-compatible dict
    (keys: ptype,E,px,py,pz,vx,vy,vz,nimpwt,pot) in BNB beam coords [cm],
    ready for dk2nu_to_primary_distribution. nimpwt encodes the relative physical
    rate (species cross section x decay-in-pipe prob); the absolute scale is
    fixed later by the neutrino-flux validation (Stage 3)."""
    rng = np.random.default_rng(seed)
    sig = {sp: total_xsec(sp, theta_max) for sp in species_list}
    cols = {k: [] for k in ("ptype", "E", "px", "py", "pz", "vx", "vy", "vz", "nimpwt")}
    for sp in species_list:
        m = MASS[sp]
        p, th, ph = sample_production(sp, n_per_species, theta_max=theta_max, rng=rng)
        d = _apply_horn(p, th, ph, sp, pt_kick)         # (n,3) unit dirs
        pos, P_pipe = _sample_decay(p, m, sp, d, rng)   # [m]
        E = np.sqrt(p * p + m * m)
        # per-meson physical weight: (sigma_species / n_sampled) * decay-in-pipe,
        # times the Stage-3 calibration that maps arb production units -> mesons/POT.
        w = BNB_FLUX_NORM * (sig[sp] / n_per_species) * P_pipe
        cols["ptype"].append(np.full(n_per_species, PDG[sp]))
        cols["E"].append(E)
        cols["px"].append(p * d[:, 0]); cols["py"].append(p * d[:, 1]); cols["pz"].append(p * d[:, 2])
        cols["vx"].append(pos[:, 0] * 100.0)            # m -> cm
        cols["vy"].append(pos[:, 1] * 100.0)
        cols["vz"].append(pos[:, 2] * 100.0)
        cols["nimpwt"].append(w)
    data = {k: np.concatenate(v) for k, v in cols.items()}
    data["pot"] = pot
    return data


# ------------------------------------------------------------------ #
#  Stage 3: meson -> neutrino flux at the detector (for validation)   #
# ------------------------------------------------------------------ #
M_MUON = 0.1056583745

def _e_star(m_parent, m_lep):
    """Neutrino energy in the parent rest frame for 2-body M -> l nu."""
    return (m_parent ** 2 - m_lep ** 2) / (2.0 * m_parent)

# parent pdg -> (parent mass, rest-frame nu energy, BR) for the numu channel
_NUMU_DECAYS = {
    211: (M_PION, _e_star(M_PION, M_MUON), 0.9999),   # pi+ -> mu+ numu
    321: (M_KAON, _e_star(M_KAON, M_MUON), 0.6356),   # K+  -> mu+ numu
}


def compute_numu_flux(data):
    """Project the meson sample onto the numu flux AT the MiniBooNE detector.

    For each pi+/K+ decay, the neutrino aimed at the detector has
        E_nu = E* / [gamma (1 - beta cos t)]
    and contributes to the per-area flux with the relativistic-beaming weight
        w_nu = nimpwt * BR / (4 pi L^2 [gamma(1-beta cos t)]^2),
    where t is the lab angle between the meson direction and the line to the
    detector, and L is that distance. Returns (E_nu[GeV], w_nu) arrays."""
    pt = data["ptype"]
    E_all = data["E"]
    px, py, pz = data["px"], data["py"], data["pz"]
    pos = np.stack([data["vx"], data["vy"], data["vz"]], axis=1) / 100.0   # cm->m
    Lv = DET_CENTER[None, :] - pos
    L = np.linalg.norm(Lv, axis=1)
    det_dir = Lv / L[:, None]
    p = np.sqrt(px * px + py * py + pz * pz)
    d = np.stack([px, py, pz], axis=1) / p[:, None]
    cost = np.sum(d * det_dir, axis=1)
    Ev, Wv = [], []
    for pdg, (mpar, estar, br) in _NUMU_DECAYS.items():
        m = pt == pdg
        if not np.any(m):
            continue
        beta = p[m] / E_all[m]
        gamma = E_all[m] / mpar
        denom = gamma * (1.0 - beta * cost[m])
        E_nu = estar / denom
        w_nu = data["nimpwt"][m] * br / (4.0 * math.pi * L[m] ** 2 * denom ** 2)
        Ev.append(E_nu); Wv.append(w_nu)
    return np.concatenate(Ev), np.concatenate(Wv)


def load_bnb_reference_numu(dat_path):
    """Read the shipped BNB_FHC.dat numu column -> (E_centers[GeV], flux[arb])."""
    lines = open(dat_path).read().splitlines()
    hdr = lines[0].split()
    j = hdr.index("numu")
    E, F = [], []
    for ln in lines[1:]:
        r = ln.split()
        if len(r) <= j:
            continue
        E.append(0.5 * (float(r[0]) + float(r[1])))
        F.append(float(r[j]))
    return np.array(E), np.array(F)


# ------------------------------------------------------------------ #
#  Self test (Stages 1+2)                                             #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    print("BNB flux self-test (Stages 1+2)   pB = %.2f GeV/c" % P_BEAM)
    print("\n[Stage 1] production spectra:")
    for sp in ("pi+", "pi-", "K+"):
        p, th, ph = sample_production(sp, 50000, rng=rng)
        print("  %-4s <p>=%.2f  p[5,50,95]=%s GeV/c  <theta>=%.0f mrad"
              % (sp, p.mean(), np.round(np.percentile(p, [5, 50, 95]), 2), 1e3 * th.mean()))

    print("\n[Stage 1] relative production cross sections (cone integral):")
    sig = {sp: total_xsec(sp) for sp in ("pi+", "pi-", "K+")}
    tot = sum(sig.values())
    for sp in ("pi+", "pi-", "K+"):
        print("  %-4s sigma=%.3e  (%.1f%% of pi+/pi-/K+)" % (sp, sig[sp], 100 * sig[sp] / tot))

    print("\n[Stage 2] meson sample (horn + decay):")
    d = generate_bnb_sample(n_per_species=100000, seed=1)
    for sp, pdg in (("pi+", 211), ("pi-", -211), ("K+", 321)):
        mask = d["ptype"] == pdg
        E = d["E"][mask]; pz = d["pz"][mask]; p = np.sqrt(d["px"][mask]**2 + d["py"][mask]**2 + pz**2)
        cth = pz / p
        zdec = d["vz"][mask] / 100.0
        w = d["nimpwt"][mask]
        print("  %-4s <E>=%.2f GeV  <cos>=%.4f  fwd(cos>0.99)=%.0f%%  <z_decay>=%.1f m  sum(w)=%.3e"
              % (sp, E.mean(), cth.mean(), 100 * np.mean(cth > 0.99), zdec.mean(), w.sum()))
    print("  total mesons:", len(d["E"]))
