"""
Build the (pseudo)scalar Dark-Primakoff response at ONE fixed (m_a, m_Zp) point,
on the corrected 11-bin HEPData binning, for scan_fixed_mass_couplings.py.

WHY THIS EXISTS: the only cached (pseudo)scalar chi2 cube in output/ is
`credible_region_mZp_product.npz`, dated 2026-07-26 -- built when
scan_credible_region.py still used the superseded 19-bin digitisation. That
script now imports miniboone_data, but line ~51 still derives its histogram
edges as DATA_E +/- 25 MeV from what are now VARIABLE-width bin centres, so its
edges match neither binning. Nothing cached is usable; rebuild here.

Muon channels only -- the paper's model is g_e = 0.

Env: PORTAL=pseudo|scalar, M_ZP_MEV, M_PHI_MEV, NDEC, DK2NU_FILE
Run: DK2NU_FILE=/home/shubham/nubeam12M.dk2nu.root \
     /home/shubham/siren_venv/bin/python build_pseudo_response_fixed.py
Out: output/fixedmass_2026-08-27/resp_<portal>_mzp<..>_ma<..>.npz
"""
import os, sys, importlib.util
import numpy as np

os.environ.setdefault("DK2NU_FILE", "/home/shubham/nubeam12M.dk2nu.root")
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE); sys.path.insert(0, HERE)
PKG = os.environ.get("SIREN_DNT_DIR", os.path.join(HERE, "..", "..", "processes", "DarkNewsTables"))
L = os.path.join(HERE, "output", "fixedmass_2026-08-27"); os.makedirs(L, exist_ok=True)

import miniboone_data as MB


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m
    spec.loader.exec_module(m); return m


SA = load(os.path.join(PKG, "AnalyticRate.py"), "AnalyticRate")
DP = load(os.path.join(PKG, "DarkPrimakoff.py"), "DarkPrimakoff")

PORTAL = os.environ.get("PORTAL", "pseudo")
CFG = {"scalar": "ScalarPortal_MiniBooNE_multichannel.py",
       "pseudo": "PseudoscalarPortal_MiniBooNE_multichannel.py"}[PORTAL]
NDEC = int(os.environ.get("NDEC", "300"))
WIN = (0.14, 0.30)
NCB = 20; COS_EDGES = np.linspace(-1, 1, NCB + 1)
ET_ENGINE = np.concatenate([np.linspace(0.001, 0.3, 120), np.linspace(0.31, 9, 160)])
ET_SIG = np.concatenate([np.linspace(0.13, 1.60, 90), np.linspace(1.65, 3.20, 32)])
RNG = np.random.default_rng(11)

S = load(CFG, "S_fix")
M_PHI = float(os.environ.get("M_PHI_MEV", S.M_PHI * 1e3)) / 1e3
M_ZP = float(os.environ.get("M_ZP_MEV", S.M_ZP * 1e3)) / 1e3
S.M_PHI = M_PHI; S.M_ZP = M_ZP
P_REF = S.G_MU_PROD * S.G_N * (S.LAMBDA * 1e-3)          # MeV^-1
print("[resp] portal=%s  m_a=%.1f MeV  m_Zp=%.1f MeV  NDEC=%d" % (PORTAL, M_PHI*1e3, M_ZP*1e3, NDEC))
print("[resp] benchmark couplings: g_mu=%.3g g_n=%.3g lambda=%.3g GeV^-1 -> P_ref=%.4e MeV^-1"
      % (S.G_MU_PROD, S.G_N, S.LAMBDA, P_REF))
print("[resp] flux: %s" % os.environ["DK2NU_FILE"])

Els, Cs, Gs, dp = [], [], [], None
for ch in [c for c in S.CHANNELS if "mu" in c]:          # muon-only: the paper has g_e = 0
    pdg, m_M, m_l, lp, nu, g = S.CHANNELS[ch]
    if (m_M - m_l) <= S.M_PHI:
        continue
    dp = S.build_onshell_models(pdg, m_M, m_l, lp, nu)["models"]["primakoff"]._dp
    E, w, c = SA.analytic_sp_mb(S, ch, n_dec=NDEC, eff_mode="mb", return_cos=True,
                                meson_fn=SA._mesons_dk2nu)
    E, w, c = np.asarray(E), np.asarray(w), np.asarray(c)
    dp.m_Zp = S.M_ZP
    sig = np.interp(E, ET_ENGINE, np.array([dp.total_xsec(float(e)) for e in ET_ENGINE]))
    ok = sig > 0
    Els.append(E[ok]); Cs.append(c[ok]); Gs.append(w[ok] / sig[ok])
    print("  %-8s %6d hits" % (ch, ok.sum()), flush=True)

El = np.concatenate(Els); Cmed = np.concatenate(Cs); G = np.concatenate(Gs)
dp.m_Zp = float(M_ZP)
sig = np.interp(El, ET_SIG, np.array([dp.total_xsec(float(e)) for e in ET_SIG]))
w = G * sig
EH = np.histogram(El, bins=MB.EBINS, weights=w)[0]
mwin = (El >= WIN[0]) & (El <= WIN[1])
cg = DP.smear_photon_beam(Cmed[mwin], El[mwin], dp, RNG)
CH = np.histogram(cg, bins=COS_EDGES, weights=w[mwin])[0]
CH = CH / CH.sum() if CH.sum() > 0 else CH

out = os.path.join(L, "resp_%s_mzp%.0f_ma%.0f.npz" % (PORTAL, M_ZP * 1e3, M_PHI * 1e3))
np.savez(out, EHIST=EH, CHIST=CH, EBINS=MB.EBINS, P_REF=P_REF, m_zp=M_ZP, m_phi=M_PHI,
         g_mu=S.G_MU_PROD, g_n=S.G_N, lam_GeV=S.LAMBDA, n_dec=NDEC,
         flux=os.environ["DK2NU_FILE"])
print("[resp] total %.1f events at P_ref ; in-window[0.14,0.30] %.1f"
      % (EH.sum(), np.histogram(El[mwin], bins=MB.EBINS, weights=w[mwin])[0].sum()))
print("[resp] wrote", out)
