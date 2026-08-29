"""
Build the (pseudo)scalar Dark-Primakoff response over a GRID of m_Zp at fixed m_a,
on the corrected 11-bin HEPData binning.

Cheap because the production/decay kinematics and the geometry are m_Zp-INDEPENDENT:
run the engine ONCE to cache per-hit (E, cos, G = w/sigma_ref), then per m_Zp only
mutate dp.m_Zp and recompute the cross-section. Same trick scan_fit_product.py uses.

Feeds plot_region_mass_product.py (the 68%/95% region in the mass-product plane).

Env: PORTAL=pseudo|scalar, MZP_LO/MZP_HI/N_MZP (MeV), M_PHI_MEV, NDEC, DK2NU_FILE
Out: output/fixedmass_2026-08-27/respscan_<portal>_ma<..>.npz
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
MZP = np.geomspace(float(os.environ.get("MZP_LO", "30")),
                   float(os.environ.get("MZP_HI", "200")),
                   int(os.environ.get("N_MZP", "36"))) / 1e3          # GeV
WIN = (0.14, 0.30); NCB = 20; COS_EDGES = np.linspace(-1, 1, NCB + 1)
ET_ENGINE = np.concatenate([np.linspace(0.001, 0.3, 120), np.linspace(0.31, 9, 160)])
ET_SIG = np.concatenate([np.linspace(0.13, 1.60, 90), np.linspace(1.65, 3.20, 32)])
RNG = np.random.default_rng(11)

S = load(CFG, "S_scan")
M_PHI = float(os.environ.get("M_PHI_MEV", S.M_PHI * 1e3)) / 1e3
S.M_PHI = M_PHI
P_REF = S.G_MU_PROD * S.G_N * (S.LAMBDA * 1e-3)
print("[scan] portal=%s m_a=%.1f MeV  m_Zp %.0f-%.0f MeV (%d pts)  NDEC=%d"
      % (PORTAL, M_PHI * 1e3, MZP[0] * 1e3, MZP[-1] * 1e3, len(MZP), NDEC), flush=True)
print("[scan] P_ref=%.4e MeV^-1   flux=%s" % (P_REF, os.environ["DK2NU_FILE"]), flush=True)

# ---- one engine pass: cache the m_Zp-independent hits (muon channels; g_e = 0) ----
Els, Cs, Gs, dp = [], [], [], None
for ch in [c for c in S.CHANNELS if "mu" in c]:
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
    print("  cached %-8s %6d hits" % (ch, ok.sum()), flush=True)
El = np.concatenate(Els); Cmed = np.concatenate(Cs); G = np.concatenate(Gs)

EH = np.zeros((len(MZP), len(MB.EXCESS))); CH = np.zeros((len(MZP), NCB))
mwin = (El >= WIN[0]) & (El <= WIN[1])
for b, mzp in enumerate(MZP):
    dp.m_Zp = float(mzp)
    sig = np.interp(El, ET_SIG, np.array([dp.total_xsec(float(e)) for e in ET_SIG]))
    w = G * sig
    EH[b] = np.histogram(El, bins=MB.EBINS, weights=w)[0]
    cg = DP.smear_photon_beam(Cmed[mwin], El[mwin], dp, RNG)
    h = np.histogram(cg, bins=COS_EDGES, weights=w[mwin])[0]
    CH[b] = h / h.sum() if h.sum() > 0 else h
    print("  m_Zp=%6.1f MeV -> %10.1f events at P_ref" % (mzp * 1e3, EH[b].sum()), flush=True)

out = os.path.join(L, "respscan_%s_ma%.0f.npz" % (PORTAL, M_PHI * 1e3))
np.savez(out, EHIST=EH, CHIST=CH, MZP_MEV=MZP * 1e3, EBINS=MB.EBINS, P_REF=P_REF,
         m_phi=M_PHI, g_mu=S.G_MU_PROD, g_n=S.G_N, lam_GeV=S.LAMBDA, n_dec=NDEC,
         flux=os.environ["DK2NU_FILE"])
print("[scan] wrote", out)
