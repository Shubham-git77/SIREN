"""
BRUTE-FORCE dense re-simulating grid scan of the (pseudo)scalar Dark-Primakoff
MiniBooNE fit -- the literal "slow" version (NO tabulation, NO interpolation, NO
analytic coupling solve).

Contrast with the fast paths:
  scan_fit_product.py / scan_credible_region.py : cache the MC hits ONCE, solve the
      coupling analytically, only re-evaluate a cheap sigma per m_Zp.
  mcmc_fit.py : tabulate a response grid ONCE, MCMC interpolates it.
THIS script instead RE-RUNS THE FULL ENGINE at every mass grid point:
  for each (m_phi, m_Zp): fresh analytic_sp_mb() -> fresh meson-decay sampling
  (n_dec loop) + Primakoff scatter + ray-trace + cross-section.  Nothing is cached
  across points except the one-time dk2nu flux READ (re-reading a 10 GB file per
  point is absurd and no analysis does it; set FRESH_FLUX_PER_POINT=1 to force it).

Parameters actually re-simulated: (m_phi, m_Zp).  The couplings enter the rate ONLY
as the product P=g_mu*g_n*lambda via P^2 (the MC is coupling-independent), so P is
swept on a cheap grid at each mass point -- gridding g_mu,g_n,lambda separately would
be pure degeneracy.  Result: a full chi2 cube over (m_phi, m_Zp, P).

Likelihood = E_vis (real nu-mode data, 19 bins) + cos-theta (paper template), same
as mcmc_fit.py, so the brute-force credible region is directly comparable.

*** DEFAULT GRID IS DENSE -> HOURS. Estimate is printed before the scan starts. ***
Tune via env:  N_MPHI, N_MZP, N_PROD, NDEC, PORTAL, BRUTE_TEST=1 (tiny smoke grid).

Run (when ready):
  PORTAL=scalar DK2NU_FILE=/home/shubham/nubeam12M.dk2nu.root \
      /home/shubham/siren_venv/bin/python scan_brute_grid.py
Outputs (checkpointed): output/brute_<portal>_chi2cube.npz + _region.png
"""
import os, json, time, importlib.util, numpy as np

PORTAL = os.environ.get("PORTAL", "scalar")
os.environ.setdefault("DK2NU_FILE", "/home/shubham/nubeam12M.dk2nu.root")
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
PKG  = os.environ.get(
    "SIREN_DNT_DIR",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "processes", "DarkNewsTables",
    ),
)

def load(path, name):
    import sys
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m

SA = load(os.path.join(PKG, "AnalyticRate.py"), "AnalyticRate")
DP = load(os.path.join(PKG, "DarkPrimakoff.py"), "DPmod")
CFG = {"scalar":"ScalarPortal_MiniBooNE_multichannel.py",
       "pseudo":"PseudoscalarPortal_MiniBooNE_multichannel.py"}[PORTAL]

# ---------- data + cos template ----------
DATA_N = np.array([302,402,333,279,189,168,134,118,81,83,75,84,57,61,38,52,26,19,19],float)
DATA_E = np.array([225,275,325,375,425,475,525,575,625,675,725,775,825,875,925,975,1025,1075,1125],float)
DATA_ERR=np.array([36,41,38,35,28,27,25,23,18,19,18,19,15,18,13,15,11,12,10],float)
BKG    = np.array([255,320,300,250,175,150,120,105,76,78,70,76,52,55,35,47,24,18,17],float)
EBINS  = np.concatenate([[DATA_E[0]-25], DATA_E+25]) / 1e3
EXCESS = DATA_N-BKG; INV2 = 1.0/DATA_ERR**2
ct = json.load(open("cos_template_nu.json"))
COS_TGT = np.array(ct["shape"]); COS_TGT/=COS_TGT.sum()
NCB=len(COS_TGT); COS_EDGES=np.linspace(-1,1,NCB+1)
COS_SIG = np.sqrt(COS_TGT*(1-COS_TGT)/320.0)+0.01
WIN=(0.14,0.30)

# ---------- grid (DENSE by default) ----------
TEST = os.environ.get("BRUTE_TEST","0")=="1"
N_MPHI = int(os.environ.get("N_MPHI", "3" if TEST else "15"))
N_MZP  = int(os.environ.get("N_MZP",  "4" if TEST else "30"))
N_PROD = int(os.environ.get("N_PROD", "20" if TEST else "60"))
NDEC   = int(os.environ.get("NDEC",   "100" if TEST else "400"))
MPHI_GRID = np.linspace(1e-3, 100e-3, N_MPHI)          # GeV, scalar/pseudo mass
MZP_GRID  = np.geomspace(30e-3, 200e-3, N_MZP)         # GeV, Z' mass
PROD_GRID = np.geomspace(3e-9, 1.2e-6, N_PROD)         # MeV^-1, coupling product
FRESH_FLUX_PER_POINT = os.environ.get("FRESH_FLUX_PER_POINT","0")=="1"
CKPT_EVERY = int(os.environ.get("CKPT_EVERY","10"))    # save partial cube every N m_phi rows

S = load(CFG, "S_brute")
P_REF = S.G_MU_PROD * S.G_N * (S.LAMBDA*1e-3)          # config product [MeV^-1]
MUON_CHANNELS = [c for c in S.CHANNELS if "mu" in c]
RNG = np.random.default_rng(0)

def forward_point(mphi, mzp):
    """FRESH full-engine re-simulation at (m_phi, m_Zp): returns (E-hist over data
    bins at P_REF, normalized cos-theta shape).  No caching of hits, no sigma table."""
    S.M_PHI = float(mphi); S.M_ZP = float(mzp)         # build_onshell_models reads these globals
    if FRESH_FLUX_PER_POINT:
        SA._DK2NU_CACHE.clear()                         # force a real flux re-read (very slow)
    Ehist = np.zeros(len(DATA_N)); cos_all=[]; w_all=[]
    for ch in MUON_CHANNELS:
        pdg,m_M,m_l,lp,nu,g = S.CHANNELS[ch]
        if (m_M-m_l) <= S.M_PHI: continue               # channel closed for heavy phi
        E,w,c = SA.analytic_sp_mb(S, ch, n_dec=NDEC, eff_mode="mb",
                                  return_cos=True, meson_fn=SA._mesons_dk2nu)
        E,w,c = np.asarray(E),np.asarray(w),np.asarray(c)
        if E.size==0: continue
        dp = S.build_onshell_models(pdg,m_M,m_l,lp,nu)["models"]["primakoff"]._dp  # has m_phi,m_Zp set
        cg = DP.smear_photon_beam(c, E, dp, RNG)         # mediator dir -> photon dir
        Ehist += np.histogram(E, bins=EBINS, weights=w)[0]
        m=(E>=WIN[0])&(E<=WIN[1]); cos_all.append(cg[m]); w_all.append(w[m])
    if cos_all:
        cc=np.concatenate(cos_all); ww=np.concatenate(w_all)
        ch,_=np.histogram(cc,bins=COS_EDGES,weights=ww); cosshape=ch/ch.sum() if ch.sum()>0 else ch
    else:
        cosshape=np.zeros(NCB)
    return Ehist, cosshape

def chi2_over_product(Ehist, cosshape):
    """chi2 vs (E_vis data + cos template) for every product on PROD_GRID."""
    chi2_c = np.sum((COS_TGT - cosshape)**2 / COS_SIG**2)   # product-independent (shape)
    out = np.empty(len(PROD_GRID))
    for k,P in enumerate(PROD_GRID):
        sig = Ehist * (P/P_REF)**2
        out[k] = np.sum((EXCESS - sig)**2 * INV2) + chi2_c
    return out

def main():
    npt = N_MPHI * N_MZP
    print("="*64)
    print(" BRUTE-FORCE re-simulating grid  [%s]" % PORTAL)
    print("  m_phi: %d pts (%.0f-%.0f MeV)   m_Zp: %d pts (%.0f-%.0f MeV)   product: %d pts"
          % (N_MPHI, MPHI_GRID[0]*1e3, MPHI_GRID[-1]*1e3, N_MZP, MZP_GRID[0]*1e3, MZP_GRID[-1]*1e3, N_PROD))
    print("  full engine re-runs: %d   n_dec=%d   fresh_flux_per_point=%s" % (npt, NDEC, FRESH_FLUX_PER_POINT))
    print("="*64)
    # ---- time ONE point to print an ETA, then scan ----
    t0=time.time(); E0,c0 = forward_point(MPHI_GRID[0], MZP_GRID[len(MZP_GRID)//2]); dt=time.time()-t0
    eta = dt*npt + (dt*npt if FRESH_FLUX_PER_POINT else 0)
    print("  1 engine re-run = %.1f s  ->  ESTIMATED TOTAL ~ %.0f min (%.1f h)"
          % (dt, eta/60, eta/3600)); import sys; sys.stdout.flush()

    CUBE = np.full((N_MPHI, N_MZP, N_PROD), np.nan)
    done=0
    for a,mphi in enumerate(MPHI_GRID):
        for b,mzp in enumerate(MZP_GRID):
            Eh,cs = forward_point(mphi, mzp)
            CUBE[a,b] = chi2_over_product(Eh, cs)
            done+=1
        if (a+1)%CKPT_EVERY==0 or a==N_MPHI-1:
            np.savez("output/brute_%s_chi2cube.npz"%PORTAL, chi2=CUBE,
                     mphi_MeV=MPHI_GRID*1e3, mzp_MeV=MZP_GRID*1e3, prod=PROD_GRID)
            el=time.time()-t0
            print("  m_phi row %d/%d done  (%d/%d pts, %.0f min elapsed, ~%.0f min left)"
                  % (a+1,N_MPHI, done,npt, el/60, el/60*(npt-done)/max(done,1))); sys.stdout.flush()

    # ---- marginalize to (m_Zp, product): profile over m_phi (min chi2) ----
    prof = np.nanmin(CUBE, axis=0)                      # [m_Zp, product]
    np.savez("output/brute_%s_chi2cube.npz"%PORTAL, chi2=CUBE, prof_mZp_prod=prof,
             mphi_MeV=MPHI_GRID*1e3, mzp_MeV=MZP_GRID*1e3, prod=PROD_GRID)
    _plot_region(prof)
    imin=np.unravel_index(np.nanargmin(CUBE), CUBE.shape)
    print("  BEST FIT: chi2=%.2f at m_phi=%.0f, m_Zp=%.0f MeV, product=%.2e"
          % (CUBE[imin], MPHI_GRID[imin[0]]*1e3, MZP_GRID[imin[1]]*1e3, PROD_GRID[imin[2]]))
    print("  wrote output/brute_%s_chi2cube.npz + _region.png" % PORTAL)

def _plot_region(prof):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    D = prof - np.nanmin(prof)
    X,Y = np.meshgrid(MZP_GRID*1e3, PROD_GRID, indexing="ij")
    fig,ax=plt.subplots(figsize=(7,6))
    ax.contourf(X,Y,D,levels=[0,2.30],colors=["#69c"],alpha=0.55)     # 68%
    ax.contourf(X,Y,D,levels=[0,6.18],colors=["#bcd"],alpha=0.30)     # 95%
    ax.contour(X,Y,D,levels=[2.30,6.18],colors="k",linewidths=[1.5,0.8])
    star={"scalar":(49,2.2e-8),"pseudo":(85,5.9e-7)}[PORTAL]
    ax.plot(*star,"r*",ms=18,label="paper Table I")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$m_{Z'}$ [MeV]"); ax.set_ylabel(r"$g_\mu g_n\lambda$ [MeV$^{-1}$]")
    ax.set_title("BRUTE-FORCE re-simulating grid (%s): 68%%/95%% region\n(profiled over $m_\\phi$; full engine re-run per mass point)"%PORTAL)
    ax.legend(); ax.grid(True,which="both",alpha=0.3)
    fig.tight_layout(); fig.savefig("output/brute_%s_region.png"%PORTAL,dpi=120)

if __name__ == "__main__":
    main()
