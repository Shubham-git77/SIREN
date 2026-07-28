"""
Extend #1 to the m_phi/m_a = 25 MeV case (paper Fig.3 shows m=1 AND m=25).
Reuses scan_credible_region machinery but overrides S.M_PHI = 25 MeV.
Saves output/credible_region_m25.npz (chi2_scalar25, chi2_pseudo25) on the SAME
(m_Zp, product) grid as the m=1 run so they can be plotted together.
"""
import os, importlib.util, numpy as np
os.environ.setdefault("DK2NU_FILE", "/home/shubham/nubeamHighSample.dk2nu.root")
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
PKG  = "/home/shubham/siren_venv/lib/python3.12/site-packages/siren/resources/processes/DarkNewsTables"

def load(path, name):
    import sys
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m

SA = load(os.path.join(PKG, "AnalyticRate.py"), "AnalyticRate")

# same digitized MiniBooNE nu-mode data + same grids as scan_credible_region.py
DATA_N = np.array([302,402,333,279,189,168,134,118,81,83,75,84,57,61,38,52,26,19,19],float)
DATA_E = np.array([225,275,325,375,425,475,525,575,625,675,725,775,825,875,925,975,1025,1075,1125],float)
DATA_ERR=np.array([36,41,38,35,28,27,25,23,18,19,18,19,15,18,13,15,11,12,10],float)
BKG    = np.array([255,320,300,250,175,150,120,105,76,78,70,76,52,55,35,47,24,18,17],float)
PBINS = np.concatenate([[DATA_E[0]-25], DATA_E+25]) / 1e3
EXCESS = DATA_N - BKG; INV2 = 1.0/DATA_ERR**2
NDEC=400
ET_ENGINE = np.concatenate([np.linspace(0.001,0.3,120), np.linspace(0.31,9,160)])
ET_SIG = np.linspace(0.13,1.60,90)
MZP_GRID = np.geomspace(0.030,0.200,28)
PROD_GRID = np.geomspace(3e-9,1.2e-6,90)

def cache(S):
    product = S.G_MU_PROD*S.G_N*(S.LAMBDA*1e-3); ref=S.M_ZP; Els=[];Gs=[];dp=None
    for ch in [c for c in S.CHANNELS if "mu" in c]:
        pdg,m_M,m_l,lp,nu,g = S.CHANNELS[ch]
        if (m_M-m_l) <= S.M_PHI:      # heavier phi may close pion channels
            continue
        dp = S.build_onshell_models(pdg,m_M,m_l,lp,nu)["models"]["primakoff"]._dp
        El,w = SA.analytic_sp_mb(S,ch,n_dec=NDEC,eff_mode="mb",meson_fn=SA._mesons_dk2nu)
        El,w=np.asarray(El),np.asarray(w); dp.m_Zp=ref
        sig=np.interp(El,ET_ENGINE,np.array([dp.total_xsec(float(e)) for e in ET_ENGINE]))
        ok=sig>0; Els.append(El[ok]); Gs.append(w[ok]/sig[ok])
    return np.concatenate(Els),np.concatenate(Gs),dp,product

def chi2_grid(cob):
    El,G,dp,product=cob; C=np.empty((len(MZP_GRID),len(PROD_GRID)))
    for i,mzp in enumerate(MZP_GRID):
        dp.m_Zp=float(mzp)
        s_ref=np.histogram(El,bins=PBINS,weights=G*np.interp(El,ET_SIG,np.array([dp.total_xsec(float(e)) for e in ET_SIG])))[0]
        for j,P in enumerate(PROD_GRID):
            s=s_ref*(P/product)**2; C[i,j]=np.sum((EXCESS-s)**2*INV2)
    return C

import sys
out={}
for portal,f in [("scalar25","ScalarPortal_MiniBooNE_multichannel.py"),
                 ("pseudo25","PseudoscalarPortal_MiniBooNE_multichannel.py")]:
    print("scanning %s (m_phi=25 MeV) ..."%portal); sys.stdout.flush()
    S=load(f,"M25_"+portal); S.M_PHI=25e-3        # override to 25 MeV
    C=chi2_grid(cache(S)); out["chi2_"+portal]=C
    imin=np.unravel_index(np.argmin(C),C.shape)
    print("  %s chi2_min=%.2f at m_Zp=%.0f, P=%.2e"%(portal,C.min(),MZP_GRID[imin[0]]*1e3,PROD_GRID[imin[1]])); sys.stdout.flush()
np.savez(os.path.join(HERE,"output","credible_region_m25.npz"),
         mzp_MeV=MZP_GRID*1e3, prod=PROD_GRID, **out)
print("wrote output/credible_region_m25.npz")
