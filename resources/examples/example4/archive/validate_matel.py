"""
Pin down whether the Dark Primakoff |M|^2 (_matel_sq) is the source of the
~model-dependent over-prediction.

Checks:
  1. Bracket transcription: at t=0 the paper bracket must equal 2(s-M^2)^2.
  2. Leading factor: paper has |M|^2 ~ t*{bracket}; with t<0 and bracket>0 this
     is NEGATIVE (unphysical). Code uses Q2=-t. Verify code == |paper| (magnitude),
     i.e. the t->|t| swap is a pure positivity fix, not a magnitude change.
  3. Sign of bracket across the physical t-range (is paper's t*bracket ever the
     correct sign on its own?).
  4. sigma scaling: must be exactly ∝ lambda^2 (code), and check the m_Zp
     (propagator) scaling against the analytic expectation, for scalar vs pseudo.
"""
import os, math
import numpy as np
from siren import _util

S = _util.load_module("ScalarMB", os.path.join(
    os.path.dirname(__file__), "ScalarPortal_MiniBooNE_multichannel.py"))
DP = _util.load_module("DP", os.path.join(
    _util.resource_package_dir(), "processes", "DarkNewsTables", "DarkPrimakoff.py"))

GEV2_CM2 = 3.8938e-28

def paper_matel(dp, s, t):
    """Paper Eq.21 EXACTLY as written: leading factor is t (not -t)."""
    m_a, M = dp.m_phi, dp.MA
    bracket = (2.0*M**2*(m_a**2 - 2.0*s - t) + 2.0*M**4
               - 2.0*m_a**2*(s + t) + m_a**4 + 2.0*s**2 + 2.0*s*t + t**2)
    den = 2.0*(t - dp.m_Zp**2)**2
    return dp.g_n**2 * dp.lam**2 * t * bracket / den, bracket

def main():
    # scalar benchmark
    dp = DP.DarkPrimakoffScattering(m_phi=1e-3, m_Zp=49e-3, g_n=1e-2, lam=0.44,
                                    nuclear_mass=11.178, A=12, Z=6)
    E = 1.0
    s = dp.m_phi**2 + dp.MA**2 + 2.0*dp.MA*E
    tlo, thi = dp._t_range(s)
    print("=== scalar benchmark, E_phi=1 GeV ===")
    print("  s=%.4f GeV^2   t-range=[%.4e, %.4e]   m_Zp^2=%.4e" % (s, tlo, thi, dp.m_Zp**2))

    # (1) bracket at t=0
    _, brk0 = paper_matel(dp, s, 0.0)
    print("\n(1) bracket(t=0) = %.6e   vs   2(s-M^2)^2 = %.6e   ratio=%.6f"
          % (brk0, 2.0*(s-dp.MA**2)**2, brk0/(2.0*(s-dp.MA**2)**2)))

    # (2)+(3) leading factor / sign across range
    print("\n(2/3) per-t: code _matel_sq vs paper (t*bracket); bracket sign")
    print("   %-12s %-14s %-14s %-14s %-8s" % ("t", "code|M|^2", "paper t*brk", "bracket", "code/|paper|"))
    for frac in (0.001, 0.05, 0.3, 0.6, 0.95):
        t = thi + (tlo - thi)*frac   # thi~0 (fwd) -> tlo (back)
        code = dp._matel_sq(s, t)
        paper, brk = paper_matel(dp, s, t)
        ratio = code/abs(paper) if paper != 0 else float('nan')
        print("   %-12.4e %-14.4e %-14.4e %-14.4e %-8.4f" % (t, code, paper, brk, ratio))

    # (4) sigma scaling
    print("\n(4) sigma scaling checks (scalar benchmark, E=1 GeV):")
    s0 = dp.total_xsec(E)
    dp2 = DP.DarkPrimakoffScattering(m_phi=1e-3, m_Zp=49e-3, g_n=1e-2, lam=0.88,
                                     nuclear_mass=11.178, A=12, Z=6)
    print("   lambda x2 -> sigma x %.3f (expect 4.0 if ∝ lam^2)" % (dp2.total_xsec(E)/s0))
    for mzp in (49e-3, 85e-3, 98e-3):
        dpm = DP.DarkPrimakoffScattering(m_phi=1e-3, m_Zp=mzp, g_n=1e-2, lam=0.44,
                                         nuclear_mass=11.178, A=12, Z=6)
        print("   m_Zp=%.3f GeV -> sigma=%.4e cm^2  (1/m_Zp^4 norm: %.4e)"
              % (mzp, dpm.total_xsec(E), dpm.total_xsec(E)*mzp**4))

    # (5) absolute sigma for scalar vs pseudo at their OWN benchmarks
    print("\n(5) absolute sigma at each model's benchmark, E_phi=1 GeV:")
    sc = DP.DarkPrimakoffScattering(1e-3, 49e-3, 1e-2, 0.44, nuclear_mass=11.178, A=12, Z=6)
    ps = DP.DarkPrimakoffScattering(1e-3, 85e-3, 1e-2, 6.5, nuclear_mass=11.178, A=12, Z=6)
    ssc, sps = sc.total_xsec(E), ps.total_xsec(E)
    print("   scalar (m_Zp=49,lam=0.44): %.4e cm^2" % ssc)
    print("   pseudo (m_Zp=85,lam=6.5):  %.4e cm^2" % sps)
    print("   pseudo/scalar sigma ratio = %.2f" % (sps/ssc))
    print("   analytic ∝ (lam_p/lam_s)^2 * (m_Zp_s/m_Zp_p)^4 = %.2f"
          % ((6.5/0.44)**2 * (49.0/85.0)**4))

if __name__ == "__main__":
    main()
