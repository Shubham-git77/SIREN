"""
REAL multi-parameter MCMC fit of the (pseudo)scalar Dark-Primakoff model to the
MiniBooNE nu-mode excess -- the "slow"/full-likelihood kind of parameter scan.

Unlike scan_fit_product.py (fast: coupling solved analytically, only m_Zp scanned),
here ALL FIVE parameters are sampled and the coupling product is WALKED, not solved:

    theta = ( log10 g_mu , log10 g_n , log10 lambda[MeV^-1] , m_Zp[MeV] , m_phi[MeV] )

Likelihood = full E_vis (real data, 19 bins) + cos-theta (paper nu-mode template).
Sampler = affine-invariant ensemble (emcee stretch move), implemented here (no deps).

Forward model is TABULATED once as a response grid over (m_phi, m_Zp) -- literally
re-running the flux MC at every MCMC step would take days; tabulation is standard.
The MCMC still explores the true 5-D posterior (product degeneracy included).

Run:  PORTAL=scalar DK2NU_FILE=/home/shubham/nubeam12M.dk2nu.root \
      /home/shubham/siren_venv/bin/python mcmc_fit.py
Out:  output/mcmc_<portal>_chain.npz, output/mcmc_<portal>_corner.png,
      output/mcmc_<portal>_mZp_product.png
"""
import os, json, importlib.util, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

PORTAL = os.environ.get("PORTAL", "scalar")
# OUT_DIR redirects every artefact. Tests and throwaway runs MUST set it: this
# script overwrites output/mcmc_<portal>_{chain.npz,corner.png,mZp_product.png}
# unconditionally, and output/ is gitignored, so a short test run silently
# destroys a real result. That is exactly how the 2026-07-27 scalar chain was
# lost on 2026-08-16.
OUT_DIR = os.environ.get("MCMC_OUT", "output")
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

# ---------- data ----------
# Arrays now come from the shared module -- they used to be inlined here as a
# 19-bin digitization that no longer matches the official release the paper cites.
import miniboone_data as MB
DATA_N, DATA_ERR, BKG, EBINS, EXCESS = MB.DATA_N, MB.DATA_ERR, MB.BKG, MB.EBINS, MB.EXCESS

# ERR_MODE=stat reproduces the historical (stat-only) likelihood; ERR_MODE=quad adds
# the MiniBooNE background systematics the paper says it added in quadrature, which
# is 5-12x less chi2 weight per bin.
ERR_MODE = os.environ.get("ERR_MODE", "stat")
INV2 = 1.0 / MB.errors(ERR_MODE) ** 2

# COS_WEIGHT scales the cos-theta template errors: 1 = original weight, "off" drops
# the term. It is NOT data -- it is a pixel-extracted shape of the paper's own Fig.2
# signal band with an invented 320-event error -- and on the brute grid it supplies
# 145 chi2 units of spread against 61 from the real E_vis data, i.e. it outvotes the
# measurement and is what drives the fitted m_Zp high. Under ERR_MODE=quad the
# imbalance gets worse (~14:1), so vary the two together, not separately.
COS_WEIGHT = os.environ.get("COS_WEIGHT", "1")
ct = json.load(open("cos_template_nu.json"))
COS_TGT = np.array(ct["shape"]); COS_TGT/=COS_TGT.sum()
NCB=len(COS_TGT); COS_EDGES=np.linspace(-1,1,NCB+1)
COS_OFF = (COS_WEIGHT == "off")
# Vocabulary matches rederive_regions.py --cos: "off" drops the term, "on" (or "1")
# keeps its original weight, any other number scales the template errors.
if COS_OFF or COS_WEIGHT == "on":
    _cos_scale = 1.0
else:
    try:
        _cos_scale = float(COS_WEIGHT)
    except ValueError:
        raise SystemExit("COS_WEIGHT must be 'off', 'on', or a number; got %r" % COS_WEIGHT)
COS_SIG = (np.sqrt(COS_TGT*(1-COS_TGT)/320.0)+0.01) * _cos_scale
WIN=(0.14,0.30)

# ---------- response-grid precompute over (m_phi, m_Zp) ----------
NDEC=120
MPHI_GRID = np.array([1,10,20,30,45,60,80,100],float)/1e3     # GeV
MZP_GRID  = np.geomspace(0.030,0.200,36)                       # GeV
ET_ENGINE = np.concatenate([np.linspace(0.001,0.3,120), np.linspace(0.31,9,160)])
# ET_SIG must span the DATA bins: np.interp clamps beyond its range, so a grid
# ending at 1.60 GeV froze sigma across the whole 1500-3000 MeV overflow bin of
# the corrected binning (harmless under the old 19-bin max of 1.15 GeV).
ET_SIG    = np.concatenate([np.linspace(0.13,1.60,90), np.linspace(1.65,3.20,32)])
RNG = np.random.default_rng(11)

def cache_hits(mphi):
    """Per-hit (E, cos_med, G=w/sigma_ref, dp) for muon channels at this m_phi."""
    S=load(CFG,"S_%d"%int(mphi*1e3)); S.M_PHI=float(mphi)
    Els=[];Cs=[];Gs=[];dp=None
    for ch in [c for c in S.CHANNELS if "mu" in c]:
        pdg,m_M,m_l,lp,nu,g=S.CHANNELS[ch]
        if (m_M-m_l)<=S.M_PHI: continue
        dp=S.build_onshell_models(pdg,m_M,m_l,lp,nu)["models"]["primakoff"]._dp
        E,w,c=SA.analytic_sp_mb(S,ch,n_dec=NDEC,eff_mode="mb",return_cos=True,meson_fn=SA._mesons_dk2nu)
        E,w,c=np.asarray(E),np.asarray(w),np.asarray(c); dp.m_Zp=S.M_ZP
        sig=np.interp(E,ET_ENGINE,np.array([dp.total_xsec(float(e)) for e in ET_ENGINE]))
        ok=sig>0; Els.append(E[ok]);Cs.append(c[ok]);Gs.append(w[ok]/sig[ok])
    S0=load(CFG,"S0"); P_ref=S0.G_MU_PROD*S0.G_N*(S0.LAMBDA*1e-3)
    if not Els: return None
    return np.concatenate(Els),np.concatenate(Cs),np.concatenate(Gs),dp,P_ref

# RESPONSE_CUBE=<brute cube .npz> reuses a scan_brute_grid.py cube as the response
# grid instead of tabulating here. Those cubes hold the SAME objects this block
# builds (per-mass-point E_vis histogram + normalized cos shape) but on a finer
# m_phi grid (15 pts vs 8) at 3.3x the decay statistics (NDEC=400 vs 120), and they
# already exist -- so this skips the expensive engine pass entirely and drops the
# dk2nu/SIREN dependency for a pure re-fit. Without it, behaviour is unchanged
# except that the freshly built grid is now SAVED for reuse rather than discarded.
import sys
RESPONSE_CUBE = os.environ.get("RESPONSE_CUBE", "")
if RESPONSE_CUBE:
    _d = np.load(RESPONSE_CUBE)
    for _k in ("Ehist", "cosshape"):
        if _k not in _d:
            sys.exit("ERROR: %s has no %s -- it predates the raw-output patch in "
                     "scan_brute_grid.py and cannot serve as a response grid."
                     % (RESPONSE_CUBE, _k))
    EHIST = _d["Ehist"]; CHIST = _d["cosshape"]
    MPHI_GRID = _d["mphi_MeV"] / 1e3          # bilinear() reads these globals
    MZP_GRID  = _d["mzp_MeV"] / 1e3
    P_REF = float(_d["P_ref"])
    if EHIST.shape[-1] != len(DATA_N):
        sys.exit("ERROR: cube has %d E bins, data has %d -- cube was built against a "
                 "different binning." % (EHIST.shape[-1], len(DATA_N)))
    PLOT_MZP = (float(MZP_GRID[0] * 1e3), float(MZP_GRID[-1] * 1e3))
    PLOT_PROD = (float(_d["prod"][0]), float(_d["prod"][-1])) if "prod" in _d else None
    print("[%s] response grid from %s: %d m_phi x %d m_Zp, P_ref=%.3e"
          % (PORTAL, RESPONSE_CUBE, len(MPHI_GRID), len(MZP_GRID), P_REF))
else:
    print("[%s] precomputing response grid over %d m_phi x %d m_Zp ..."
          % (PORTAL, len(MPHI_GRID), len(MZP_GRID)))
    sys.stdout.flush()
    EHIST=np.zeros((len(MPHI_GRID),len(MZP_GRID),len(DATA_N)))     # signal events at P_ref
    CHIST=np.zeros((len(MPHI_GRID),len(MZP_GRID),NCB))            # normalized cos shape
    P_REF=None
    for a,mphi in enumerate(MPHI_GRID):
        hit=cache_hits(mphi)
        if hit is None: continue
        El,Cmed,G,dp,P_REF=hit
        for b,mzp in enumerate(MZP_GRID):
            dp.m_Zp=float(mzp)
            sig=np.interp(El,ET_SIG,np.array([dp.total_xsec(float(e)) for e in ET_SIG]))
            w=G*sig
            EHIST[a,b]=np.histogram(El,bins=EBINS,weights=w)[0]
            m=(El>=WIN[0])&(El<=WIN[1])
            cg=DP.smear_photon_beam(Cmed[m],El[m],dp,RNG)
            hc,_=np.histogram(cg,bins=COS_EDGES,weights=w[m])
            CHIST[a,b]=hc/hc.sum() if hc.sum()>0 else hc
        print("  m_phi=%3.0f MeV done"%(mphi*1e3)); sys.stdout.flush()
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(os.path.join(OUT_DIR,"mcmc_%s_response.npz"%PORTAL), Ehist=EHIST, cosshape=CHIST,
             mphi_MeV=MPHI_GRID*1e3, mzp_MeV=MZP_GRID*1e3, P_ref=P_REF)
    print("  saved %s/mcmc_%s_response.npz (reusable via RESPONSE_CUBE=)"%(OUT_DIR,PORTAL))
    PLOT_MZP = (float(MZP_GRID[0]*1e3), float(MZP_GRID[-1]*1e3)); PLOT_PROD = None

def bilinear(grid, mphi, mzp):
    """Interpolate response grid[a,b,:] at (mphi[GeV], mzp[GeV])."""
    a=np.clip(np.searchsorted(MPHI_GRID,mphi)-1,0,len(MPHI_GRID)-2)
    b=np.clip(np.searchsorted(MZP_GRID,mzp)-1,0,len(MZP_GRID)-2)
    fa=(mphi-MPHI_GRID[a])/(MPHI_GRID[a+1]-MPHI_GRID[a])
    fb=(mzp -MZP_GRID[b]) /(MZP_GRID[b+1]-MZP_GRID[b])
    fa=np.clip(fa,0,1); fb=np.clip(fb,0,1)
    g=grid
    return ((1-fa)*(1-fb)*g[a,b] + fa*(1-fb)*g[a+1,b] + (1-fa)*fb*g[a,b+1] + fa*fb*g[a+1,b+1])

# ---------- priors + likelihood over theta=(lgmu,lgn,llam,mZp[MeV],mphi[MeV]) ----------
PLO=np.array([-4.0,-3.0,-4.0, 30.0,  1.0])
PHI=np.array([-1.0,-1.0,-2.0,200.0,100.0])
def logprob(th):
    if np.any(th<PLO) or np.any(th>PHI): return -np.inf
    lgmu,lgn,llam,mZp,mphi = th
    P = 10**lgmu * 10**lgn * 10**llam                 # product [MeV^-1]
    Ev = bilinear(EHIST, mphi/1e3, mZp/1e3)*(P/P_REF)**2
    chi2_E = np.sum((EXCESS-Ev)**2*INV2)
    cs = bilinear(CHIST, mphi/1e3, mZp/1e3); s=cs.sum()
    cs = cs/s if s>0 else cs
    chi2_c = 0.0 if COS_OFF else np.sum((COS_TGT-cs)**2/COS_SIG**2)
    return -0.5*(chi2_E+chi2_c)

# ---------- affine-invariant ensemble sampler (emcee stretch move) ----------
def run_mcmc(nwalkers=int(os.environ.get("MCMC_WALKERS", "40")),
             nsteps=int(os.environ.get("MCMC_STEPS", "4000")), seed=1):
    rng=np.random.default_rng(seed); ndim=5
    ctr=0.5*(PLO+PHI); span=(PHI-PLO)
    pos=ctr+0.25*span*(rng.random((nwalkers,ndim))-0.5)     # start near center
    lp=np.array([logprob(p) for p in pos])
    # nudge any -inf starts inward
    for k in range(nwalkers):
        while not np.isfinite(lp[k]):
            pos[k]=ctr+0.1*span*(rng.random(ndim)-0.5); lp[k]=logprob(pos[k])
    chain=np.empty((nsteps,nwalkers,ndim)); a=2.0
    acc=0
    for st in range(nsteps):
        for k in range(nwalkers):
            j=rng.integers(nwalkers)
            while j==k: j=rng.integers(nwalkers)
            z=((a-1.0)*rng.random()+1.0)**2/a
            prop=pos[j]+z*(pos[k]-pos[j])
            lpp=logprob(prop)
            if np.log(rng.random()) < (ndim-1)*np.log(z)+lpp-lp[k]:
                pos[k]=prop; lp[k]=lpp; acc+=1
        chain[st]=pos
        if (st+1)%500==0: print("  step %d/%d  acc=%.2f"%(st+1,nsteps,acc/((st+1)*nwalkers))); sys.stdout.flush()
    return chain

print("[%s] running MCMC ..."%PORTAL); sys.stdout.flush()
chain=run_mcmc()
burn=chain.shape[0]//3
flat=chain[burn:].reshape(-1,5)
lgP = flat[:,0]+flat[:,1]+flat[:,2]                        # log10 product
samples=np.column_stack([flat, lgP])                      # add derived product
labels=[r"$\log_{10}g_\mu$",r"$\log_{10}g_n$",r"$\log_{10}\lambda$",r"$m_{Z'}$",r"$m_\phi$",r"$\log_{10}(g_\mu g_n\lambda)$"]
os.makedirs(OUT_DIR, exist_ok=True); np.savez(os.path.join(OUT_DIR,"mcmc_%s_chain.npz"%PORTAL), chain=chain, flat=flat, lgP=lgP, labels=labels)

# ---------- corner plot ----------
def corner(s, labels, truths=None):
    n=s.shape[1]; fig,ax=plt.subplots(n,n,figsize=(2.2*n,2.2*n))
    for i in range(n):
        for j in range(n):
            a=ax[i,j]
            if j>i: a.axis("off"); continue
            if i==j:
                a.hist(s[:,i],bins=45,color="#3b6",histtype="stepfilled",alpha=0.7)
                q=np.percentile(s[:,i],[16,50,84]); [a.axvline(v,ls="--",c="k",lw=0.7) for v in q]
                a.set_title("%s = %.2f$^{+%.2f}_{-%.2f}$"%(labels[i],q[1],q[2]-q[1],q[1]-q[0]),fontsize=8)
            else:
                a.hist2d(s[:,j],s[:,i],bins=45,cmap="viridis")
                if truths is not None and truths[j] is not None and truths[i] is not None:
                    a.plot(truths[j],truths[i],"*",color="red",ms=12)
            if i==n-1: a.set_xlabel(labels[j],fontsize=8)
            else: a.set_xticklabels([])
            if j==0 and i>0: a.set_ylabel(labels[i],fontsize=8)
            else: a.set_yticklabels([])
            a.tick_params(labelsize=6)
    fig.suptitle("MCMC posterior -- %s Dark Primakoff (MiniBooNE nu E_vis + cos-theta)"%PORTAL,fontsize=12)
    fig.tight_layout(); return fig
# paper Table I truth (scalar 49/2.2e-8, pseudo 85/5.9e-7); log couplings unknown individually
paperP = {"scalar":(49.0,np.log10(2.2e-8)),"pseudo":(85.0,np.log10(5.9e-7))}[PORTAL]
truths=[None,None,None,paperP[0],None,paperP[1]]
corner(samples,labels,truths).savefig(os.path.join(OUT_DIR,"mcmc_%s_corner.png"%PORTAL),dpi=110)
print("wrote %s/mcmc_%s_corner.png"%(OUT_DIR,PORTAL))

# ---------- (m_Zp, product) 2D posterior with 68/95 contours ----------
fig,ax=plt.subplots(figsize=(7,6))
H,xe,ye=np.histogram2d(flat[:,3],lgP,bins=60)
# credible levels from the histogram
Hs=np.sort(H.ravel())[::-1]; cum=np.cumsum(Hs)/H.sum()
l68=Hs[np.searchsorted(cum,0.68)]; l95=Hs[np.searchsorted(cum,0.95)]
X,Y=np.meshgrid(0.5*(xe[:-1]+xe[1:]),0.5*(ye[:-1]+ye[1:]),indexing="ij")
ax.contourf(X,Y,H,levels=[l95,l68,H.max()],colors=["#bcd","#69c"],alpha=0.7)
ax.contour(X,Y,H,levels=[l95,l68],colors="k",linewidths=[0.8,1.4])
ax.plot(paperP[0],paperP[1],"*",color="red",ms=18,label="paper Table I")
ax.set_xscale("log")
# Pin the axes to the response-grid extent. Auto-scaling to the posterior made
# the paper star land in a corner with no sense of how far outside it sat, and
# made these plots impossible to compare with the brute-grid region figures,
# which use exactly these limits.
ax.set_xlim(*PLOT_MZP)
if PLOT_PROD is not None:
    ax.set_ylim(np.log10(PLOT_PROD[0]), np.log10(PLOT_PROD[1]))
ax.set_xlabel(r"$m_{Z'}$ [MeV]"); ax.set_ylabel(r"$\log_{10}(g_\mu g_n\lambda)$ [MeV$^{-1}$]")
ax.set_title("MCMC posterior (%s): 68%%/95%% credible region\nfull 5-param fit, coupling WALKED not solved"%PORTAL)
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR,"mcmc_%s_mZp_product.png"%PORTAL),dpi=120)
print("wrote %s/mcmc_%s_mZp_product.png"%(OUT_DIR,PORTAL))
print("[%s] DONE. product log10 = %.2f +/- %.2f ; m_Zp = %.0f +/- %.0f MeV ; m_phi = %.0f +/- %.0f"
      %(PORTAL, np.median(lgP), np.std(lgP), np.median(flat[:,3]), np.std(flat[:,3]),
        np.median(flat[:,4]), np.std(flat[:,4])))
