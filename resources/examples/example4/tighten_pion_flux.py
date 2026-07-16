"""
Tighten the BNB pi (and K) flux normalization separately, instead of the single
global BNB_FLUX_NORM fit to TOTAL numu.

numu comes from pi+->mu nu (dominates E_nu < ~1.3 GeV) and K+->mu nu (high-E tail).
We compute the two parent contributions separately, then least-squares fit
   F_official(E) ~ a_pi * h_pi(E) + a_K * h_K(E)
over the reference range. a_pi, a_K are multiplicative corrections to the CURRENT
(already-BNB_FLUX_NORM'd) pi and K normalizations. a_pi != 1 => the pion norm
(which sets the dominant pi_mu Dark-Primakoff channel) needs adjusting; its spread
across fit ranges is the pion-flux systematic.
"""
import os, math
import numpy as np
from siren import _util
_BNB = _util.load_module("BNBFlux", os.path.join(
    _util.resource_package_dir(), "processes", "DarkNewsTables", "BNBFlux.py"))
_REF = _util.load_module("BNBref", os.path.join(
    _util.resource_package_dir(), "fluxes", "BNB", "BNB-v1.0", "flux.py"))

def parent_numu(data, pdg):
    sub = {k: (v[data["ptype"] == pdg] if hasattr(v, "__len__") and k != "pot" else v)
           for k, v in data.items()}
    if sub["E"].size == 0:
        return np.array([]), np.array([])
    return _BNB.compute_numu_flux(sub)

def main(n=400000, pt_kick=_BNB.PT_KICK_DEFAULT, seed=1):
    # official reference numu spectrum (nu/m^2/GeV/POT), loaded from BNB_FHC.dat
    _REF.fetch_data()
    dat = os.path.join(_REF._get_abs_dir(), "BNB_FHC.dat")
    raw = np.loadtxt(dat, skiprows=1)   # cols: Elo Ehi numu numubar nue nuebar
    E_ref = 0.5*(raw[:,0]+raw[:,1])
    F_ref = raw[:,2] / 50.0 * 1000.0 * 1e4   # numu -> nu/m^2/GeV/POT
    BINS = np.arange(0.0, 7.001, 0.05); CENT = 0.5*(BINS[:-1]+BINS[1:])
    F_ref_c = np.interp(CENT, E_ref, F_ref, left=0, right=0)

    data = _BNB.generate_bnb_sample(n_per_species=n, pt_kick=pt_kick, seed=seed)
    def hist(pdg):
        E, w = parent_numu(data, pdg)
        if E.size == 0: return np.zeros_like(CENT)
        h, _ = np.histogram(E, bins=BINS, weights=w); return h/np.diff(BINS)
    h_pi = hist(211); h_K = hist(321)

    # 2-component non-negative least squares over the populated range
    def fit(lo, hi):
        m = (CENT>=lo)&(CENT<=hi)&(F_ref_c>0)
        A = np.vstack([h_pi[m], h_K[m]]).T; b = F_ref_c[m]
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        pred = A@coef
        resid = np.sqrt(np.mean(((pred-b)/(b+1e-30))**2))
        return coef, resid
    print("BNB numu: separate pi/K normalization fit (a=correction to current norm)")
    print("  current totals (this sample): sum h_pi=%.3e  sum h_K=%.3e  ratio K/pi=%.3f"
          % (h_pi.sum(), h_K.sum(), h_K.sum()/max(h_pi.sum(),1e-30)))
    for lo,hi,lab in [(0.2,1.0,"pi-dominated [0.2,1.0]"),
                      (0.2,3.0,"full [0.2,3.0]"),
                      (1.5,4.0,"K-tail [1.5,4.0]")]:
        (a_pi,a_K),r = fit(lo,hi)
        print("  %-22s : a_pi=%.3f  a_K=%.3f   (rms frac resid=%.2f)" % (lab,a_pi,a_K,r))
    # headline pion correction from the pi-dominated low-E region
    (a_pi_lo,_),_ = fit(0.2,1.0)
    (a_pi_full,a_K_full),_ = fit(0.2,3.0)
    print("\n  => PION norm correction a_pi ~ %.2f (low-E) / %.2f (full fit)"
          % (a_pi_lo, a_pi_full))
    print("     spread of a_pi across ranges = pion-flux systematic")
    print("     (apply a_pi as a multiplicative factor on the pi_mu Dark-Primakoff N_S)")

if __name__=="__main__":
    main()
