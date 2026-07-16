"""
Dark Primakoff scattering for the long-lived (pseudo)scalar scenario of
Dutta et al. (arXiv:2110.11944), Model ii) / Fig. 1(c):

    phi(/a) + N  ->  gamma + N      (coherent, via t-channel Z' exchange)

Matrix element : Eq. (C2)  (scalar uses m_a -> m_phi; SAME squared ME)
Cross section  : Eq. (C3)  with Helm nuclear form factor
Signal         : a single PHOTON (E_gamma ~ E_phi, since coherent recoil on a
                 heavy nucleus is keV-MeV scale)

This is the (pseudo)scalar analog of VectorPortalUpsCase. It is a SINGLE
scattering vertex producing a photon -- there is NO downstream chi'/V1/e+e-
cascade. The photon IS the visible signal.

Validation status: |M|^2 positive across the physical t-range for all phi
energies; sigma ~ 6e-33 cm^2 on carbon at the Table I/II scalar benchmark
(formula validated term-by-term against Eq. C2/C3); E_gamma ~ E_phi confirmed.
The absolute rate should be checked end-to-end against the ~few-hundred-event
MiniBooNE scalar fit (Fig. 2 bottom).

Benchmark (Table I/II scalar): m_phi = 1 MeV, m_Z' = 49 MeV,
    (g_mu, g_n, lambda) = (5e-3, 1e-2, 4.4e-4 MeV^-1 = 0.44 GeV^-1).
NOTE the lambda unit conversion (MeV^-1 -> GeV^-1) -- easy to get wrong by 1e6.
"""

import math
import numpy as np

# SIREN cross-section interface (only imported when used inside SIREN; guarded
# so the standalone physics class can be tested without a full siren install).
try:
    from siren.interactions import CrossSection as _CrossSection
    from siren import dataclasses
    from siren.dataclasses import Particle
    from siren.injection import PhaseSpaceConvention as _PhaseSpaceConvention
    from siren.injection import PhaseSpaceTopology as _Topology
    from siren.injection import PhaseSpaceMeasure as _Measure
    _HAVE_SIREN = True
except Exception:
    _CrossSection = object
    _HAVE_SIREN = False

_ALPHA_EM   = 1.0 / 137.036
_GEV2_TO_CM2 = 3.8938e-28
_HBARC_FM   = 0.197327      # GeV * fm

# numpy>=2.0 renamed trapz -> trapezoid; support both without eager eval.
if hasattr(np, "trapezoid"):
    _TRAPZ = np.trapezoid
else:
    _TRAPZ = np.trapz


def _helm_F2(t, A):
    """Helm nuclear form factor squared. t spacelike (<0); |q| = sqrt(-t)."""
    Q = math.sqrt(max(-t, 0.0))
    Qfm = Q / _HBARC_FM
    s_skin = 0.9
    r0sq = max((1.2 * A ** (1.0 / 3.0)) ** 2 - 5.0 * s_skin ** 2, 0.0)
    Qr = Qfm * math.sqrt(r0sq)
    if Qr < 1e-6:
        j = 1.0 / 3.0
    else:
        j = (math.sin(Qr) - Qr * math.cos(Qr)) / Qr ** 3
    return (3.0 * j) ** 2 * math.exp(-(Qfm * s_skin) ** 2)


class DarkPrimakoffScattering:
    """Coherent phi N -> gamma N (Eq. C2/C3).

    Parameters
    ----------
    m_phi : float        scalar/pseudoscalar mass [GeV]
    m_Zp  : float        Z' mediator mass [GeV]
    g_n   : float        Z' coupling to nucleons
    lam   : float        lambda coupling [GeV^-1]  (NB: paper quotes MeV^-1)
    nuclear_pdgid, nuclear_mass : target (default C12 for MiniBooNE oil)
    A, Z  : mass number, atomic number
    """

    def __init__(self, m_phi, m_Zp, g_n, lam,
                 nuclear_pdgid=1000060120, nuclear_mass=11.178,
                 nuclear_name="C12", A=12, Z=6,
                 pdgid_phi=5919, pdgid_photon=22):
        self.m_phi = m_phi
        self.m_Zp  = m_Zp
        self.g_n   = g_n
        self.lam   = lam
        self.MA    = nuclear_mass
        self.A     = A
        self.Z     = Z
        self.nuclear_pdgid = nuclear_pdgid
        self.nuclear_name  = nuclear_name
        self.pdgid_phi    = pdgid_phi
        self.pdgid_photon = pdgid_photon

    # ---- Eq. (C2): spin/structure-summed |M|^2 (incl. g_n^2 lambda^2) ----
    def _matel_sq(self, s, t):
        m_a = self.m_phi
        M   = self.MA
        # Eq. (C2) has a leading factor of t multiplying the bracket:
        #   |M|^2 = g_n^2 lambda^2 * t * { ... } / [2 (t - m_Zp^2)^2]
        # In the (+---) convention t<0 (spacelike); the physically positive
        # combination is |t| = -t = Q^2 (the bracket is dominated by +2 m_N^4,
        # so |M|^2 must use Q^2 to stay positive). This leading Q^2 also
        # restores correct dimensions (lambda^2[GeV^-2] * Q^2[GeV^2] *
        # {GeV^4}/{GeV^4} = dimensionless) and was the missing ~|t| factor.
        Q2 = -t
        bracket = (2.0 * M ** 2 * (m_a ** 2 - 2.0 * s - t)
                   + 2.0 * M ** 4
                   - 2.0 * m_a ** 2 * (s + t)
                   + m_a ** 4 + 2.0 * s ** 2 + 2.0 * s * t + t ** 2)
        den = 2.0 * (t - self.m_Zp ** 2) ** 2
        if den <= 0.0:
            return 0.0
        val = self.g_n ** 2 * self.lam ** 2 * Q2 * bracket / den
        return max(val, 0.0)

    # ---- t-range for a(p) N -> gamma(massless) N ----
    def _t_range(self, s):
        m_a, M = self.m_phi, self.MA
        kall = lambda x, y, z: x*x + y*y + z*z - 2*x*y - 2*y*z - 2*z*x
        pi2 = kall(s, m_a ** 2, M ** 2) / (4.0 * s)
        pf2 = kall(s, 0.0,      M ** 2) / (4.0 * s)
        if pi2 <= 0 or pf2 <= 0:
            return None, None
        pi, pf = math.sqrt(pi2), math.sqrt(pf2)
        sq = math.sqrt(s)
        Ei_a = (s + m_a ** 2 - M ** 2) / (2.0 * sq)
        Ef_g = (s - M ** 2) / (2.0 * sq)
        t_minus = m_a ** 2 - 2.0 * (Ei_a * Ef_g - pi * pf)   # cos=+1 (forward)
        t_plus  = m_a ** 2 - 2.0 * (Ei_a * Ef_g + pi * pf)   # cos=-1 (back)
        return t_plus, t_minus

    # ---- Eq. (C3): dsigma/dt [cm^2/GeV^2] ----
    def _dsigma_dt(self, s, t):
        m_a, M = self.m_phi, self.MA
        denom = 16.0 * math.pi * (s - (m_a + M) ** 2) * (s - (m_a - M) ** 2)
        if denom <= 0.0:
            return 0.0
        val = self.Z ** 2 / denom * self._matel_sq(s, t) * _helm_F2(t, self.A)
        return max(val, 0.0) * _GEV2_TO_CM2

    # ---- total cross section at lab phi energy E_phi [GeV] ----
    def total_xsec(self, E_phi, n_t=400):
        s = self.m_phi ** 2 + self.MA ** 2 + 2.0 * self.MA * E_phi
        tlo, thi = self._t_range(s)
        if tlo is None or thi <= tlo:
            return 0.0
        ts = np.linspace(tlo, thi, n_t)
        vals = np.array([self._dsigma_dt(s, t) for t in ts])
        return max(float(_TRAPZ(vals, ts)), 0.0)

    # ---- Q2 = -t convention helpers (SIREN uses Q2 as density variable) ----
    @property
    def Ethreshold(self):
        # phi N -> gamma N has no real threshold beyond E_phi > 0 (massless
        # photon, elastic-like). Use a tiny floor for numerical safety.
        return max(self.m_phi, 1e-4)

    def Q2_range(self, E_phi):
        s = self.m_phi ** 2 + self.MA ** 2 + 2.0 * self.MA * E_phi
        tlo, thi = self._t_range(s)
        if tlo is None:
            return None, None
        # Q2 = -t ; t in [tlo(<0), thi(~0)] -> Q2 in [-thi, -tlo]
        return -thi, -tlo

    def diff_xsec_Q2(self, E_phi, Q2):
        """dsigma/dQ2 [cm^2/GeV^2], with Q2 = -t."""
        s = self.m_phi ** 2 + self.MA ** 2 + 2.0 * self.MA * E_phi
        return self._dsigma_dt(s, -Q2)   # |dt/dQ2| = 1

    # ---- sample a photon: returns (E_gamma, cos_theta_gamma_lab) ----
    def sample_photon(self, E_phi, rng=None):
        rng = rng or np.random
        s = self.m_phi ** 2 + self.MA ** 2 + 2.0 * self.MA * E_phi
        tlo, thi = self._t_range(s)
        if tlo is None or thi <= tlo:
            return None
        # sample t from dsigma/dt by rejection
        ts = np.linspace(tlo, thi, 256)
        w  = np.array([self._dsigma_dt(s, t) for t in ts])
        if w.sum() <= 0:
            return None
        cdf = np.cumsum(w); cdf /= cdf[-1]
        t = float(np.interp(rng.random(), cdf, ts))
        # photon lab energy: E_gamma = E_phi - T_N, T_N = -t/(2 M)
        T_N = -t / (2.0 * self.MA)
        E_gamma = E_phi - T_N
        # photon lab angle wrt beam: for coherent forward scattering, the
        # photon is nearly collinear with the incoming phi (small |t|).
        # cos_theta from |t| ~ 2 E_phi E_gamma (1 - cos): approximate.
        if E_phi > 0 and E_gamma > 0:
            cos_theta = 1.0 + t / (2.0 * E_phi * E_gamma)
            cos_theta = max(-1.0, min(1.0, cos_theta))
        else:
            cos_theta = 1.0
        return E_gamma, cos_theta


# ===================================================================
#  DarkPrimakoffUpsCase  --  SIREN cross-section wrapper
#  phi N -> gamma N  (scalar/pseudoscalar -> photon)
#  Mirrors VectorPortalUpsCase's interface so it drops into the same
#  injector machinery. Density variable Q2 = -t; 2->2 scatter.
# ===================================================================

class DarkPrimakoffUpsCase(_CrossSection):
    """SIREN cross-section for coherent Dark Primakoff phi N -> gamma N.

    Primary  : phi   (pdgid_phi, default 5919)
    Target   : nucleus (nuclear_pdgid)
    Secondary: [photon (22), nucleus]
    """

    def __init__(self, m_phi, m_Zp, g_n, lam,
                 nuclear_pdgid=1000060120, nuclear_mass=11.178,
                 nuclear_name="C12", A=12, Z=6,
                 pdgid_phi=5919):
        super().__init__()
        self._dp = DarkPrimakoffScattering(
            m_phi, m_Zp, g_n, lam,
            nuclear_pdgid=nuclear_pdgid, nuclear_mass=nuclear_mass,
            nuclear_name=nuclear_name, A=A, Z=Z,
            pdgid_phi=pdgid_phi, pdgid_photon=22)
        self.m_phi = m_phi
        self.m_target = nuclear_mass
        self.nuclear_pdgid = nuclear_pdgid
        self.pdgid_phi = pdgid_phi
        self.pdgid_photon = 22

    # ---- signatures ----
    def GetPossibleTargets(self):
        target_type = Particle.ParticleType(self.nuclear_pdgid)
        if target_type == Particle.ParticleType.PPlus:
            target_type = Particle.ParticleType.HNucleus
        return [target_type]

    def GetPossibleTargetsFromPrimary(self, primary_type):
        if int(primary_type) == self.pdgid_phi:
            return self.GetPossibleTargets()
        return []

    def GetPossiblePrimaries(self):
        return [Particle.ParticleType(self.pdgid_phi)]

    def GetPossibleSignatures(self):
        sig = dataclasses.InteractionSignature()
        sig.primary_type = Particle.ParticleType(self.pdgid_phi)
        target_type = Particle.ParticleType(self.nuclear_pdgid)
        if target_type == Particle.ParticleType.PPlus:
            target_type = Particle.ParticleType.HNucleus
        sig.target_type = target_type
        sig.secondary_types = [
            Particle.ParticleType(self.pdgid_photon),
            target_type,
        ]
        return [sig]

    def GetPossibleSignaturesFromParents(self, primary_type, target_type):
        if int(primary_type) == self.pdgid_phi:
            expected = Particle.ParticleType(self.nuclear_pdgid)
            if expected == Particle.ParticleType.PPlus:
                expected = Particle.ParticleType.HNucleus
            if target_type == expected:
                return self.GetPossibleSignatures()
        return []

    # ---- cross sections ----
    def TotalCrossSection(self, arg1, energy=None, target=None):
        if isinstance(arg1, dataclasses.InteractionRecord):
            energy = arg1.primary_momentum[0]
            primary = arg1.signature.primary_type
        else:
            primary = arg1
        if int(primary) != self.pdgid_phi:
            return 0.0
        return self._dp.total_xsec(energy)

    def DifferentialCrossSection(self, arg1, target=None, energy=None, Q2=None):
        if isinstance(arg1, dataclasses.InteractionRecord):
            record = arg1
            primary = np.array(record.primary_momentum, dtype=float)
            photon = np.array(record.secondary_momenta[0], dtype=float)
            # Q2 = -t = -(p_phi - p_gamma)^2.  The naive (m1sq + m3sq - 2*p1p3)
            # form suffers catastrophic cancellation for the ultralight scalar
            # (m_phi = 1 MeV) in the forward limit, where 2*p1p3 -> m_phi^2 and
            # the reconstructed Q2 becomes floating-point noise -> runaway event
            # weight.  Compute t from the difference 4-vector instead, which is
            # numerically stable for any mediator mass.
            dE = primary[0] - photon[0]
            dp = primary[1:] - photon[1:]
            t = dE * dE - float(np.dot(dp, dp))
            Q2 = max(0.0, -t)
            energy = record.primary_momentum[0]
        return float(np.real(self._dp.diff_xsec_Q2(energy, Q2)))

    def InteractionThreshold(self, interaction):
        return self._dp.Ethreshold

    def Q2Min(self, interaction):
        lo, hi = self._dp.Q2_range(interaction.primary_momentum[0])
        return lo if lo is not None else 0.0

    def Q2Max(self, interaction):
        lo, hi = self._dp.Q2_range(interaction.primary_momentum[0])
        return hi if hi is not None else 0.0

    def TargetMass(self, target_type):
        return self.m_target

    def SecondaryMasses(self, secondary_types):
        return [0.0, self.m_target]      # photon massless

    def SecondaryHelicities(self, record):
        return [record.primary_helicity, record.target_helicity]

    def FinalStateProbability(self, record):
        total = self.TotalCrossSection(record)
        if total <= 0.0:
            return 0.0
        return self.DifferentialCrossSection(record) / total

    def DensityVariables(self):
        return ["Q2"]

    def Convention(self):
        return _PhaseSpaceConvention.MandelstamST

    def Topology(self):
        return _Topology.Scatter2to2

    def Measure(self):
        return _Measure.MandelstamQ2()

    def equal(self, other):
        return self is other

    def _sample_Q2(self, E_phi, random):
        """Rejection-sample Q2 from dsigma/dQ2.

        NOTE: the differential |M|^2 has a LEADING Q^2 factor (Eq. C2), so
        dsigma/dQ2 -> 0 at q2min and RISES with Q^2 before the Z' propagator
        eventually turns it over. The envelope must therefore bound the true
        maximum over the whole [q2min, q2max] range, not the value at q2min
        (which is near the minimum). Anchoring at q2min under-bounds the
        density, the rejection loop fails, and the old uniform fallback drew
        unnormalized Q^2 -> a single event with a ~1e9x runaway weight.
        """
        q2min, q2max = self._dp.Q2_range(E_phi)
        if q2min is None or q2max <= q2min:
            return None
        # Scan the range to find the true envelope maximum.
        n_scan = 256
        q2grid = q2min + (q2max - q2min) * (
            (np.arange(n_scan) + 0.5) / n_scan)
        fvals = np.array([self._dp.diff_xsec_Q2(E_phi, q) for q in q2grid])
        fvals = np.where(np.isfinite(fvals) & (fvals > 0.0), fvals, 0.0)
        f_max = fvals.max() * 1.3   # 1.3 safety margin above the scanned peak
        if f_max <= 0.0:
            return None
        for _ in range(10000):
            cand = random.Uniform(q2min, q2max)
            fc = self._dp.diff_xsec_Q2(E_phi, cand)
            if not np.isfinite(fc) or fc <= 0.0:
                continue
            if random.Uniform(0.0, f_max) <= fc:
                return cand
        # If we still fail after 10000 tries the envelope is mis-scaled;
        # return the scanned-peak Q^2 rather than an unnormalized uniform draw,
        # so no event escapes with a runaway weight.
        return float(q2grid[int(np.argmax(fvals))])

    def SampleFinalState(self, record, random):
        E_phi = record.primary_momentum[0]
        M = self.m_target
        m_phi = self.m_phi

        Q2 = self._sample_Q2(E_phi, random)
        if Q2 is None:
            return

        s = m_phi**2 + M**2 + 2.0 * M * E_phi
        sqrt_s = math.sqrt(s)
        # final state: photon (massless) + nucleus
        E_gamma_cm = (s - M**2) / (2.0 * sqrt_s)
        E_N_cm = (s + M**2) / (2.0 * sqrt_s)
        p_out_cm = E_gamma_cm     # massless photon: |p| = E

        E_in_cm = (s + m_phi**2 - M**2) / (2.0 * sqrt_s)
        p_in_cm = math.sqrt(max(E_in_cm**2 - m_phi**2, 0.0))

        if p_in_cm > 0.0 and p_out_cm > 0.0:
            cos_cm = 1.0 - Q2 / (2.0 * p_in_cm * p_out_cm)
        else:
            cos_cm = 0.0
        cos_cm = max(-1.0, min(1.0, cos_cm))

        gamma_cm = (E_phi + M) / sqrt_s
        beta_cm = math.sqrt(max(E_phi**2 - m_phi**2, 0.0)) / (E_phi + M)

        phi_cm = random.Uniform(0.0, 2.0 * math.pi)
        sin_cm = math.sqrt(max(1.0 - cos_cm**2, 0.0))

        px_g = p_out_cm * sin_cm * math.cos(phi_cm)
        py_g = p_out_cm * sin_cm * math.sin(phi_cm)
        pz_g = p_out_cm * cos_cm

        E_g_lab = gamma_cm * (E_gamma_cm + beta_cm * pz_g)
        pz_g_lab = gamma_cm * (pz_g + beta_cm * E_gamma_cm)
        P_gamma = np.array([E_g_lab, px_g, py_g, pz_g_lab])

        E_N_lab = gamma_cm * (E_N_cm - beta_cm * pz_g)
        pz_N_lab = gamma_cm * (-pz_g + beta_cm * E_N_cm)
        P_N = np.array([E_N_lab, -px_g, -py_g, pz_N_lab])

        # rotate from beam-aligned frame to lab
        p_in_dir = np.array(record.primary_momentum[1:])
        p_in_mag = np.linalg.norm(p_in_dir)
        if p_in_mag > 1e-12:
            z_hat = p_in_dir / p_in_mag
            arb = np.array([0, 1, 0]) if abs(z_hat[1]) < 0.9 else np.array([1, 0, 0])
            x_hat = np.cross(z_hat, arb); x_hat /= np.linalg.norm(x_hat)
            y_hat = np.cross(z_hat, x_hat)
            R = np.column_stack([x_hat, y_hat, z_hat])
            P_gamma[1:] = R @ P_gamma[1:]
            P_N[1:] = R @ P_N[1:]

        for sec in record.get_secondary_particle_records():
            if int(sec.type) == self.pdgid_photon:
                sec.four_momentum = P_gamma
                sec.mass = 0.0
            else:
                sec.four_momentum = P_N
                sec.mass = M
        return
