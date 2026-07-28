"""
Vector-portal (double-mediator) MCMC fit to the MiniBooNE nu-mode excess -- the
vector analog of mcmc_fit.py, mapping to Dutta-Kim Fig.3 LEFT panel (Model I).

Model I signal: meson->l nu V1 ; V1->chi chi ; chi N -> chi' N (via V2) ; chi'->chi
V1_sig -> e+e-.  Engine = analytic_vec_mb (full cascade, e+e- system).
Fixed benchmark masses: m_chi=8, m_chi'=50, m_V1=17 MeV (paper double-mediator).
FREE: m_V2 (x-axis) and the coupling product P = eps1 * eps2 * g'^2/(4pi) (y-axis).

Likelihood = E_vis (real nu-mode data) + cos-theta (VECTOR template, Fig.2 top-row
blue chi-upscattering band, pixel-extracted -> cos_template_vector_nu.json).

*** CAVEAT (vector normalization): the vector rate uses the EMPIRICAL CALIB_VECTOR
bridge (not first-principles).  We anchor the config-benchmark calibrated rate to
the paper's double-mediator Table I product P0=1.3e-7 and scale signal as (P/P0)^2.
So the m_V2 localization is robust, but the absolute coupling axis inherits the known
~few-x vector normalization uncertainty. ***

Run: DK2NU_FILE=/home/shubham/nubeam12M.dk2nu.root \
     /home/shubham/siren_venv/bin/python mcmc_fit_vector.py
Out: output/mcmc_vector_{chain.npz,corner.png,mV2_product.png}
"""
import os, json, importlib.util, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

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
CFG = "VectorPortal_MiniBooNE_fullchain.py"

# ---- data + vector cos template ----
DATA_N = np.array([302,402,333,279,189,168,134,118,81,83,75,84,57,61,38,52,26,19,19],float)
DATA_E = np.array([225,275,325,375,425,475,525,575,625,675,725,775,825,875,925,975,1025,1075,1125],float)
DATA_ERR=np.array([36,41,38,35,28,27,25,23,18,19,18,19,15,18,13,15,11,12,10],float)
BKG    = np.array([255,320,300,250,175,150,120,105,76,78,70,76,52,55,35,47,24,18,17],float)
EBINS  = np.concatenate([[DATA_E[0]-25], DATA_E+25]) / 1e3
EXCESS = DATA_N-BKG; INV2 = 1.0/DATA_ERR**2
ct = json.load(open("cos_template_vector_nu.json"))
COS_TGT = np.array(ct["shape"]); COS_TGT/=COS_TGT.sum()
NCB=len(COS_TGT); COS_EDGES=np.linspace(-1,1,NCB+1)
COS_SIG = np.sqrt(COS_TGT*(1-COS_TGT)/320.0)+0.01
WIN=(0.14,0.30)

NDEC=100
MV2_GRID = np.geomspace(0.060, 2.0, 26)         # GeV, Fig.3-left x-range
P0 = 1.3e-7                                       # paper double-mediator Table I product (anchor)
RNG = np.random.default_rng(5)

# ---- response grid over m_V2 (only free mass); Ehist(calibrated) + cos shape ----
import sys
RESP = "output/mcmc_vector_response.npz"
if os.path.exists(RESP):
    print("[vector] loading cached response grid %s (skip precompute)"%RESP); sys.stdout.flush()
    _r=np.load(RESP); EHIST=_r["EHIST"]; CHIST=_r["CHIST"]
    assert EHIST.shape[0]==len(MV2_GRID), "cached grid shape mismatch; delete %s to rebuild"%RESP
else:
    print("[vector] precomputing response grid over %d m_V2 ..."%len(MV2_GRID)); sys.stdout.flush()
    S = load(CFG, "S_vec")
    CHANS = list(S.CHANNELS)                      # vector is lepton-universal (all channels)
    EHIST=np.zeros((len(MV2_GRID),len(DATA_N))); CHIST=np.zeros((len(MV2_GRID),NCB))
    for a,mv2 in enumerate(MV2_GRID):
        S.M_V2=float(mv2)
        Eh=np.zeros(len(DATA_N)); cos_all=[]; w_all=[]
        for ch in CHANS:
            pdg,m_M,m_l,lp,nu,g=S.CHANNELS[ch]
            if (m_M-m_l)<=S.M_V1: continue
            E,w,c=SA.analytic_vec_mb(S,ch,n_dec=NDEC,eff_mode="mb",return_cos=True,meson_fn=SA._mesons_dk2nu)
            E,w,c=np.asarray(E),np.asarray(w),np.asarray(c)
            if E.size==0: continue
            Eh+=np.histogram(E,bins=EBINS,weights=w)[0]
            m=(E>=WIN[0])&(E<=WIN[1]); cos_all.append(c[m]); w_all.append(w[m])
        EHIST[a]=Eh
        if cos_all:
            cc=np.concatenate(cos_all); ww=np.concatenate(w_all)
            h,_=np.histogram(cc,bins=COS_EDGES,weights=ww); CHIST[a]=h/h.sum() if h.sum()>0 else h
        print("  m_V2=%.0f MeV done  (in-window bench=%.2f ev)"%(mv2*1e3, Eh[(DATA_E/1e3>=WIN[0])&(DATA_E/1e3<=WIN[1])].sum())); sys.stdout.flush()
    np.savez(RESP, EHIST=EHIST, CHIST=CHIST, MV2_GRID=MV2_GRID)   # checkpoint so a rerun skips the slow precompute
    print("[vector] saved response grid -> %s"%RESP); sys.stdout.flush()

def interp(grid, mv2):
    b=np.clip(np.searchsorted(MV2_GRID,mv2)-1,0,len(MV2_GRID)-2)
    f=np.clip((mv2-MV2_GRID[b])/(MV2_GRID[b+1]-MV2_GRID[b]),0,1)
    return (1-f)*grid[b]+f*grid[b+1]

# ---- priors + likelihood: theta=(m_V2[MeV], log10 P) ----
PLO=np.array([60.0, -9.5]); PHI=np.array([2000.0, -5.0])
def logprob(th):
    if np.any(th<PLO) or np.any(th>PHI): return -np.inf
    mv2, lgP = th; P=10**lgP
    Ev = interp(EHIST, mv2/1e3) * (P/P0)**2
    chi2_E = np.sum((EXCESS-Ev)**2*INV2)
    cs = interp(CHIST, mv2/1e3); s=cs.sum(); cs=cs/s if s>0 else cs
    chi2_c = np.sum((COS_TGT-cs)**2/COS_SIG**2)
    return -0.5*(chi2_E+chi2_c)

def run_mcmc(nwalkers=32, nsteps=4000, seed=1):
    rng=np.random.default_rng(seed); ndim=2
    ctr=np.array([300.0,-6.5]); pos=ctr+np.array([200.0,0.6])*(rng.random((nwalkers,ndim))-0.5)
    pos[:,0]=np.clip(pos[:,0], PLO[0], PHI[0])
    lp=np.array([logprob(p) for p in pos])
    for k in range(nwalkers):
        while not np.isfinite(lp[k]):
            pos[k]=ctr+np.array([100.0,0.4])*(rng.random(ndim)-0.5); lp[k]=logprob(pos[k])
    chain=np.empty((nsteps,nwalkers,ndim)); a=2.0; acc=0
    for st in range(nsteps):
        for k in range(nwalkers):
            j=rng.integers(nwalkers)
            while j==k: j=rng.integers(nwalkers)
            z=((a-1)*rng.random()+1)**2/a
            prop=pos[j]+z*(pos[k]-pos[j])
            lpp=logprob(prop)
            if np.log(rng.random())<(ndim-1)*np.log(z)+lpp-lp[k]:
                pos[k]=prop; lp[k]=lpp; acc+=1
        chain[st]=pos
        if (st+1)%1000==0: print("  step %d/%d acc=%.2f"%(st+1,nsteps,acc/((st+1)*nwalkers))); sys.stdout.flush()
    return chain

print("[vector] running MCMC ..."); sys.stdout.flush()
chain=run_mcmc(); burn=chain.shape[0]//3; flat=chain[burn:].reshape(-1,2)
np.savez("output/mcmc_vector_chain.npz", chain=chain, flat=flat, mV2_grid=MV2_GRID, P0=P0)

# corner (2x2)
labels=[r"$m_{V_2}$ [MeV]", r"$\log_{10}(\epsilon_1\epsilon_2 g'^2/4\pi)$"]
fig,ax=plt.subplots(2,2,figsize=(8,8))
for i in range(2):
    for j in range(2):
        a=ax[i,j]
        if j>i: a.axis("off"); continue
        if i==j:
            a.hist(flat[:,i],bins=45,color="#69c",alpha=0.8)
            q=np.percentile(flat[:,i],[16,50,84]); [a.axvline(v,ls="--",c="k",lw=0.7) for v in q]
            a.set_title("%s = %.2f$^{+%.2f}_{-%.2f}$"%(labels[i],q[1],q[2]-q[1],q[1]-q[0]),fontsize=9)
        else:
            a.hist2d(flat[:,j],flat[:,i],bins=45,cmap="viridis")
        if i==1: a.set_xlabel(labels[j],fontsize=9)
        if j==0 and i==1: a.set_ylabel(labels[i],fontsize=9)
fig.suptitle("MCMC posterior -- Vector double-mediator (MiniBooNE nu E_vis + cos-theta)",fontsize=11)
fig.tight_layout(); fig.savefig("output/mcmc_vector_corner.png",dpi=110)

# (m_V2, product) posterior region
fig,ax=plt.subplots(figsize=(7,6))
H,xe,ye=np.histogram2d(flat[:,0],flat[:,1],bins=55)
Hs=np.sort(H.ravel())[::-1]; cum=np.cumsum(Hs)/H.sum()
l68=Hs[np.searchsorted(cum,0.68)]; l95=Hs[np.searchsorted(cum,0.95)]
X,Y=np.meshgrid(0.5*(xe[:-1]+xe[1:]),0.5*(ye[:-1]+ye[1:]),indexing="ij")
ax.contourf(X,Y,H,levels=[l95,l68,H.max()],colors=["#bcd","#69c"],alpha=0.7)
ax.contour(X,Y,H,levels=[l95,l68],colors="k",linewidths=[0.8,1.4])
ax.plot(200,np.log10(1.3e-7),"r*",ms=18,label="paper double (200,1.3e-7)")
ax.plot(17,np.log10(3.6e-9),"m*",ms=15,label="paper single (17,3.6e-9)")
ax.set_xscale("log")
ax.set_xlabel(r"$m_{V_2}$ [MeV]"); ax.set_ylabel(r"$\log_{10}(\epsilon_1\epsilon_2 g'^2/4\pi)$")
ax.set_title("Vector MCMC posterior (MiniBooNE): 68%/95% region\n[Fig.3-LEFT analog; abs. axis via empirical CALIB_VECTOR]")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("output/mcmc_vector_mV2_product.png",dpi=120)
print("wrote output/mcmc_vector_{corner,mV2_product}.png")
print("[vector] DONE. m_V2 = %.0f +/- %.0f MeV ; log10 product = %.2f +/- %.2f"
      %(np.median(flat[:,0]),np.std(flat[:,0]),np.median(flat[:,1]),np.std(flat[:,1])))
