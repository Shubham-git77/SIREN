"""
Derive the Dark Primakoff scattering CALIB.

Logic established 2026-06-21:
  * N_S  proportional to  (g_mu g_n lambda)^2   [production g_mu^2 x scatter g_n^2 lam^2]
  * Table II benchmark product == Table I best-fit product (scalar 2.2e-8 MeV^-1,
    pseudo 6.5e-7 vs 5.9e-7 MeV^-1) => at these couplings the model, by definition
    of "best fit", must reproduce the ~320-event MiniBooNE excess.
  * Therefore any factor between SIREN's full-chain rate and ~320 is a real
    normalization bug:  CALIB_DP = N_paper / N_SIREN.

This script breaks N_S into its Eq.5 factors evaluated INDEPENDENTLY of SIREN, so
we can see which factor (sigma, BR, N_T, acceptance) carries the discrepancy.
Run for scalar K_mu (the best-conditioned channel).
"""
import os, math
import numpy as np
from siren import _util

S = _util.load_module("ScalarMB", os.path.join(
    os.path.dirname(__file__), "ScalarPortal_MiniBooNE_multichannel.py"))
BNB = S._BNB

DET = np.array([0.0, 1.896, 541.34])        # m, MiniBooNE center (BNB coords)
GEV2_CM2 = 3.8938e-28

N_C = 3.6312e22           # carbon nuclei / cm^3 (from detector model)
POT = S.MINIBOONE_POT
R_FID_m = S.R_FID
L_typ = (4.0/3.0) * R_FID_m * 100.0   # cm, mean chord of a sphere = 4R/3

def acceptance(data, parent_pdg):
    """Fraction of mesons whose forward line of flight passes within R_fid of
    the detector (phi||meson approx).  Same A_fid SIREN computes per event."""
    is_p = data["ptype"] == parent_pdg
    pK = np.stack([data["px"][is_p], data["py"][is_p], data["pz"][is_p]], axis=1)
    vK = np.stack([data["vx"][is_p], data["vy"][is_p], data["vz"][is_p]], axis=1) / 100.0
    nrm = np.linalg.norm(pK, axis=1); good = nrm > 0
    dirK = pK[good] / nrm[good, None]; vKg = vK[good]
    rel = DET[None, :] - vKg
    tcl = np.einsum("ij,ij->i", rel, dirK)
    closest = vKg + tcl[:, None] * dirK
    miss = np.linalg.norm(closest - DET[None, :], axis=1)
    return np.mean((tcl > 0) & (miss < R_FID_m)), data["E"][is_p].mean()

def derive(portal, S_mod, data):
    print("\n" + "#"*64)
    print("#  %s  Dark Primakoff CALIB derivation" % portal)
    print("#"*64)
    total = 0.0
    for name in S_mod.CHANNELS:
        parent_pdg, m_meson, m_lepton, lepton_pdg, nu_pdg, gamma_sm = S_mod.CHANNELS[name]
        chain = S_mod.build_onshell_models(parent_pdg, m_meson, m_lepton, lepton_pdg, nu_pdg)
        meson_decay = chain["meson_decay"]; dp = chain["models"]["primakoff"]._dp
        br = meson_decay._total_width / gamma_sm
        is_p = data["ptype"] == parent_pdg
        if is_p.sum() == 0:
            print("  %-6s : no %d in BNB sample, skip" % (name, parent_pdg)); continue
        N_meson = data["nimpwt"][is_p].sum() * POT
        N_phi = N_meson * br
        A_fid, Emean = acceptance(data, parent_pdg)
        # phi carries ~half the meson energy on average in 3-body; use 0.5*E as proxy
        E_phi = 0.5 * Emean
        sig = dp.total_xsec(E_phi)
        P_sc = sig * N_C * L_typ
        N_S = N_phi * A_fid * P_sc
        total += N_S
        print("  %-6s : BR=%.2e  N_phi=%.2e  A_fid=%.2e  sig(%.2f)=%.2e  N_S=%.3e"
              % (name, br, N_phi, A_fid, E_phi, sig, N_S))
    print("  %-6s : N_S(total) = %.3e" % ("SUM", total))
    print("  Paper best-fit anchor (model reproduces ~320 excess BEFORE eff/cuts)")
    print("  => CALIB_%s = 320 / N_S = %.3e   (eff~0.1-0.2 + E_vis<0.3GeV cut are SEPARATE)"
          % (portal[:6].upper(), 320.0 / total if total else 0))
    return total

def main():
    global gamma_sm
    data = BNB.generate_bnb_sample(n_per_species=50000, seed=7)
    # gamma_sm is per-channel; set inside derive via closure-free lookup
    import importlib.util
    P = _util.load_module("PseudoMB", os.path.join(
        os.path.dirname(__file__), "PseudoscalarPortal_MiniBooNE_multichannel.py"))
    # patch module-level gamma_sm usage: derive() reads it from CHANNELS tuple instead
    derive("Scalar", S, data)
    derive("Pseudoscalar", P, data)

if __name__ == "__main__":
    main()
