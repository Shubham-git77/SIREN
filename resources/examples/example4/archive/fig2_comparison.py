"""
Dutta-Kim Fig.2-style comparison: MiniBooNE E_vis distribution of the predicted
dark-Primakoff single-photon signal, with ALL corrections applied so the scale
is directly comparable to the ~320-event excess:
  * muon-only (g_e=0): only K->mu nu X and pi->mu nu X (the paper's coupling),
  * MiniBooNE single-photon efficiency eps(E_gamma),
  * best-fit couplings (scalar = benchmark; pseudo product 5.9/6.5 -> N_S x0.824).
NB: the old ad-hoc production "/2" is gone -- that factor-2 over-prediction is now
fixed at source (MesonProduction _matel_sq_* prefactor 8.0->4.0, validated vs a
first-principles Dirac trace), so the production rate here is already corrected.
Uses the low-variance estimator (n_dec decays/meson) so the curves are smooth.
Overlays scalar and pseudoscalar; shades the [140,300] MeV excess window.
"""
import os, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from siren import _util

HERE = os.path.dirname(__file__)
def load(name): return _util.load_module(name, os.path.join(HERE, name+".py"))
S_sc = load("ScalarPortal_MiniBooNE_multichannel")
S_ps = load("PseudoscalarPortal_MiniBooNE_multichannel")
BNB  = S_sc._BNB

DET   = np.array([0.0, 1.896, 541.34])   # m, MiniBooNE center (BNB coords)
R_OIL = S_sc.R_OIL                        # m
N_C   = 3.6312e22                         # carbon nuclei/cm^3
POT   = S_sc.MINIBOONE_POT
_EFF_E = np.array([0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.90])
_EFF   = np.array([0.089,0.135,0.139,0.131,0.123,0.116,0.106,0.102])
def mb_eff(E):
    e = np.interp(E,_EFF_E,_EFF,left=_EFF[0],right=_EFF[-1])
    return np.where(E >= 0.140, e, 0.0)

EBINS = np.linspace(0, 1500, 61)          # MeV (25 MeV bins)
EC = 0.5*(EBINS[:-1]+EBINS[1:])

def channel_hist(S, channel, corr, n_dec=400):
    """Return events/bin histogram of E_vis for one channel, with corrections."""
    pdg,m_M,m_l,lpdg,nupdg,gsm = S.CHANNELS[channel]
    ch = S.build_onshell_models(pdg,m_M,m_l,lpdg,nupdg)
    md = ch["meson_decay"]._decay; dp = ch["models"]["primakoff"]._dp
    m_phi = md.m_phi; br = ch["meson_decay"]._total_width/gsm
    d = BNB.generate_bnb_sample(n_per_species=50000, seed=7)
    isp = d["ptype"]==pdg
    p = np.stack([d["px"][isp],d["py"][isp],d["pz"][isp]],axis=1)
    v = np.stack([d["vx"][isp],d["vy"][isp],d["vz"][isp]],axis=1)/100.0
    E = d["E"][isp]; w = d["nimpwt"][isp]
    pmag = np.linalg.norm(p,axis=1); dirK = p/pmag[:,None]
    Emax=(m_M**2+m_phi**2-m_l**2)/(2*m_M); Eg=np.linspace(m_phi+1e-5,Emax-1e-5,400)
    dN=np.array([max(md.differential_decay_rate([e])[0],0.0) for e in Eg])
    dN=np.where(np.isfinite(dN)&(dN>0),dN,0.0); cdf=np.cumsum(dN); cdf/=cdf[-1]
    Et=np.concatenate([np.linspace(0.001,0.3,120),np.linspace(0.31,8,120)])
    st=np.array([dp.total_xsec(float(e)) for e in Et])
    rng=np.random.default_rng(1); nK=len(pmag)
    arb=np.tile(np.array([0.,1.,0.]),(nK,1)); mm=np.abs(dirK[:,1])>0.9; arb[mm]=np.array([1.,0.,0.])
    xc=np.cross(dirK,arb); xc/=np.linalg.norm(xc,axis=1)[:,None]; yc=np.cross(dirK,xc)
    gK=E/m_M; bK=pmag/E; prefm=w*POT*br*corr
    h=np.zeros(len(EC))
    for _ in range(n_dec):
        u=rng.random(nK); Es=np.interp(u,cdf,Eg); ps=np.sqrt(np.maximum(Es**2-m_phi**2,0))
        c=rng.uniform(-1,1,nK); s=np.sqrt(1-c**2); az=rng.uniform(0,2*math.pi,nK)
        El=gK*(Es+bK*ps*c); pll=gK*(ps*c+bK*Es); ptt=ps*s
        pph=pll[:,None]*dirK+(ptt*np.cos(az))[:,None]*xc+(ptt*np.sin(az))[:,None]*yc
        dph=pph/np.linalg.norm(pph,axis=1)[:,None]
        mv=v-DET[None,:]; bd=np.einsum("ij,ij->i",mv,dph)
        disc=bd**2-(np.einsum("ij,ij->i",mv,mv)-R_OIL**2); hit=(disc>0)&(-bd>0)
        chord=np.where(hit,2*np.sqrt(np.maximum(disc,0)),0.0)*100.0
        sig=np.interp(El,Et,st); base=np.where(hit,sig*N_C*chord,0.0)
        wd=prefm*base*mb_eff(El)/n_dec
        h+=np.histogram(El*1e3,bins=EBINS,weights=wd)[0]
    return h

# scalar: best-fit = benchmark (x1.0).
# pseudo: best-fit product 5.9/6.5 -> N_S x (5.9/6.5)^2 = 0.824.
# NB the old ad-hoc production "/2" was REMOVED: the factor-2 over-prediction it
# compensated is now fixed at source (MesonProduction _matel_sq_* prefactor
# 8.0->4.0), so `br` already carries the corrected (halved) production rate.
hsc = channel_hist(S_sc,"K_mu",1.0) + channel_hist(S_sc,"pi_mu",1.0)
hps = (channel_hist(S_ps,"K_mu",0.824) + channel_hist(S_ps,"pi_mu",0.824))
win=(EC>=140)&(EC<=300)
isc, ips = hsc[win].sum(), hps[win].sum()
fsc, fps = isc/hsc.sum(), ips/hps.sum()
print("in-window [140,300] MeV: scalar=%.0f (x%.1f)  pseudo=%.0f (x%.1f)  excess~320"
      %(isc,isc/320,ips,ips/320))
print("in-window fraction of total: scalar=%.1f%%  pseudo=%.1f%%"%(100*fsc,100*fps))

fig,ax=plt.subplots(1,2,figsize=(13.5,5.2))
# --- left: absolute prediction with honest residual ---
ax[0].step(EC,hsc,where="mid",color="C0",lw=2,label="Scalar")
ax[0].step(EC,hps,where="mid",color="C3",lw=2,label="Pseudoscalar (muon-only)")
ax[0].axvspan(140,300,color="gray",alpha=0.15); ax[0].axvline(140,color="gray",ls="--",lw=1)
ax[0].set_xlabel(r"$E_{\rm vis}\simeq E_\gamma$ [MeV]"); ax[0].set_ylabel("Counts / 25 MeV bin")
ax[0].set_xlim(0,1500); ax[0].set_ylim(bottom=0); ax[0].legend(fontsize=9)
ax[0].set_title("Absolute prediction (leading corrections applied)")
ax[0].text(0.96,0.96,"in-window [140,300] MeV:\n scalar %.0f  ($\\times$%.1f)\n pseudo %.0f  ($\\times$%.1f)\n"
           " MiniBooNE excess $\\approx$320"%(isc,isc/320,ips,ips/320),
           transform=ax[0].transAxes,ha="right",va="top",fontsize=8.5,
           bbox=dict(boxstyle="round",fc="white",ec="0.7"))
# --- right: shape comparison (area-normalized) ---
ax[1].step(EC,hsc/hsc.sum(),where="mid",color="C0",lw=2,label="Scalar (%.0f%% in window)"%(100*fsc))
ax[1].step(EC,hps/hps.sum(),where="mid",color="C3",lw=2,label="Pseudoscalar (%.0f%% in window)"%(100*fps))
ax[1].axvspan(140,300,color="gray",alpha=0.15,label="excess window [140,300] MeV")
ax[1].axvline(140,color="gray",ls="--",lw=1)
ax[1].set_xlabel(r"$E_{\rm vis}\simeq E_\gamma$ [MeV]"); ax[1].set_ylabel("Counts (area-normalized)")
ax[1].set_xlim(0,1500); ax[1].set_ylim(bottom=0); ax[1].legend(fontsize=9)
ax[1].set_title("Spectral shape (area-normalized)")
fig.suptitle(r"MiniBooNE dark-Primakoff vs the $\sim$320 excess "
             r"($E_{\rm vis}$ variable, confirmed from Fig.2; muon-only, eff, best-fit)",
             fontsize=11)
os.makedirs("output",exist_ok=True)
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig("output/MiniBooNE_fig2_comparison.png",dpi=140); plt.close(fig)
print("Saved -> output/MiniBooNE_fig2_comparison.png")
