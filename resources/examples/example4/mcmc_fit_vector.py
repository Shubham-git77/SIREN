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
# MiniBooNE nu-mode data now comes from the shared module. It used to be a
# 19-bin digitization inlined here, which matched no official release and
# silently diverged from the corrected 11-bin HEPData binning used by
# scan_brute_grid.py / mcmc_fit.py -- results from the two were not
# comparable. ERR_MODE=quad adds the MiniBooNE background systematics the
# paper says it used; ERR_MODE=stat (default) keeps the historical
# stat-only weighting.
import os as _os
import miniboone_data as MB
DATA_E, DATA_N, DATA_ERR, BKG, EBINS = MB.DATA_E, MB.DATA_N, MB.DATA_ERR, MB.BKG, MB.EBINS
ERR_MODE = _os.environ.get("ERR_MODE", "stat")
EXCESS = MB.EXCESS
INV2 = 1.0 / MB.errors(ERR_MODE) ** 2
# COS_WEIGHT: "off" drops the angular term, "on"/1 keeps it, a number scales its
# errors. The template is a pixel-extracted shape of the paper's OWN Fig.2
# vector signal band with an invented error -- on the scalar side the
# equivalent term supplied more chi2 leverage than the real data and drove the
# fitted mass. Make it switchable rather than assumed.
COS_WEIGHT = os.environ.get("COS_WEIGHT", "1")
COS_OFF = (COS_WEIGHT == "off")
_cos_scale = 1.0 if (COS_OFF or COS_WEIGHT == "on") else float(COS_WEIGHT)
OUT_DIR = os.environ.get("MCMC_OUT", "output")
ct = json.load(open("cos_template_vector_nu.json"))
COS_TGT = np.array(ct["shape"]); COS_TGT/=COS_TGT.sum()
NCB=len(COS_TGT); COS_EDGES=np.linspace(-1,1,NCB+1)
COS_SIG = (np.sqrt(COS_TGT*(1-COS_TGT)/320.0)+0.01) * _cos_scale
WIN=(0.14,0.30)

NDEC=100
# Range was 0.060-2.0 (the paper's Fig.3-left x-range) until 2026-08-23, when the
# best fit was found pinned on the 60 MeV LOWER EDGE -- a grid-truncation artefact.
# m_V2 is a t-channel exchange mass with no on-shell threshold, so going below the
# paper's plot range is legitimate; the real minimum sits near 31-36 MeV.
MV2_LO = float(os.environ.get("MV2_LO", "0.010"))
MV2_HI = float(os.environ.get("MV2_HI", "2.0"))
N_MV2  = int(os.environ.get("N_MV2", "38"))
MV2_GRID = np.geomspace(MV2_LO, MV2_HI, N_MV2)
P0 = 1.3e-7                                       # paper double-mediator Table I product (anchor)
RNG = np.random.default_rng(5)

# ---- response grid over m_V2 (only free mass); Ehist(calibrated) + cos shape ----
import sys
_stale = False
# Key the cache by the grid it was built for. The mass RANGE is configurable now,
# so a fixed filename plus a count-only staleness check would let a 26-point
# 60-2000 grid load for a 26-point 10-2000 request -- same shape, wrong physics.
_isdefault = (N_MV2==26 and abs(MV2_LO-0.060)<1e-12 and abs(MV2_HI-2.0)<1e-12)
RESP = os.path.join(OUT_DIR, "mcmc_vector_response.npz" if _isdefault else
                    "vector_response_n%d_d%d_lo%.0f_hi%.0f.npz"
                    % (N_MV2, NDEC, MV2_LO*1e3, MV2_HI*1e3))
if os.path.exists(RESP):
    print("[vector] loading cached response grid %s (skip precompute)"%RESP); sys.stdout.flush()
    _r=np.load(RESP); EHIST=_r["EHIST"]; CHIST=_r["CHIST"]
    # Validate BOTH axes. The mass-axis-only check used to let a cache built against
    # the old 19-bin inlined digitization load against the corrected 11-bin HEPData
    # binning, which then died in logprob with an opaque broadcast error. Treat any
    # shape mismatch as "stale" and rebuild rather than assert.
    _stale = (EHIST.shape[0]!=len(MV2_GRID) or EHIST.shape[1]!=len(DATA_N)
              or CHIST.shape[0]!=len(MV2_GRID) or CHIST.shape[1]!=NCB)
    # Shape alone is not identity: check the mass VALUES too.
    if not _stale:
        _stale = ("MV2_GRID" not in _r.files
                  or not np.allclose(_r["MV2_GRID"], MV2_GRID))
        if _stale:
            print("[vector] cached grid covers different MASSES -> rebuilding"); sys.stdout.flush()
    if _stale:
        print("[vector] cached response %s is STALE (EHIST %s, need (%d,%d)) -> rebuilding"
              % (RESP, EHIST.shape, len(MV2_GRID), len(DATA_N))); sys.stdout.flush()
if (not os.path.exists(RESP)) or _stale:
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
    """Log-log interpolation in mass.

    The response falls as steeply as m_V2^-4, so LINEAR interpolation between
    log-spaced grid points is a chord above a convex curve and over-estimates the
    rate by up to ~4% at bin midpoints -- a systematic bias, always in the same
    direction. Interpolating log(rate) vs log(m) removes it. Zeros are floored
    rather than dropped so an empty high-mass bin stays at ~0 instead of -inf.
    """
    b = np.clip(np.searchsorted(MV2_GRID, mv2) - 1, 0, len(MV2_GRID) - 2)
    lm, lm0, lm1 = np.log(mv2), np.log(MV2_GRID[b]), np.log(MV2_GRID[b + 1])
    f = np.clip((lm - lm0) / (lm1 - lm0), 0, 1)
    FL = 1e-300
    g0 = np.log(np.maximum(grid[b], FL))
    g1 = np.log(np.maximum(grid[b + 1], FL))
    out = np.exp((1 - f) * g0 + f * g1)
    return np.where(out <= FL * 10, 0.0, out)

# ---- priors + likelihood: theta=(m_V2[MeV], log10 P) ----
PLO=np.array([MV2_LO*1e3, -9.5]); PHI=np.array([MV2_HI*1e3, -5.0])
MASS_PRIOR = os.environ.get("MASS_PRIOR", "log")   # see mcmc_fit_vector_full.py
def logprob(th):
    if np.any(th<PLO) or np.any(th>PHI): return -np.inf
    mv2, lgP = th; P=10**lgP
    lp_mass = -np.log(mv2) if MASS_PRIOR == "log" else 0.0
    Ev = interp(EHIST, mv2/1e3) * (P/P0)**2
    chi2_E = np.sum((EXCESS-Ev)**2*INV2)
    cs = interp(CHIST, mv2/1e3); s=cs.sum(); cs=cs/s if s>0 else cs
    chi2_c = 0.0 if COS_OFF else np.sum((COS_TGT-cs)**2/COS_SIG**2)
    return -0.5*(chi2_E+chi2_c) + lp_mass

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
np.savez(os.path.join(OUT_DIR,"mcmc_vector_chain.npz"), chain=chain, flat=flat, mV2_grid=MV2_GRID, P0=P0)

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
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR,"mcmc_vector_corner.png"),dpi=110)

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
fig.tight_layout(); fig.savefig(os.path.join(OUT_DIR,"mcmc_vector_mV2_product.png"),dpi=120)
print("wrote output/mcmc_vector_{corner,mV2_product}.png")
print("[vector] DONE. m_V2 = %.0f +/- %.0f MeV ; log10 product = %.2f +/- %.2f"
      %(np.median(flat[:,0]),np.std(flat[:,0]),np.median(flat[:,1]),np.std(flat[:,1])))
