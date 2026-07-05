"""
Dutta-Kim Fig.2-style REBUILD for all three portals (MiniBooNE nu-mode E_vis).
Draws the MiniBooNE background (tan) with OUR signal (red) STACKED on top,
fit-normalized to the excess, and the MiniBooNE data points (black) overlaid --
the way the paper presents Fig.2.

Panels:
  * Scalar  (phi Dark Primakoff, single photon)  -> paper Fig.2 BOTTOM
  * Pseudoscalar (a Dark Primakoff, single photon) -> not a Fig.2 panel (Fig.3)
  * Vector  (double-mediator DM -> e+e-)           -> paper Fig.2 TOP

Background + data points are DIGITIZED from the paper's Fig.2 bottom-left panel
(scalar, nu-mode E_vis; arXiv:2110.11944) and reused for all three -- so this is
a SHAPE + fit-normalized comparison, not an independent MiniBooNE data fit.

Scalar/pseudoscalar shapes come from the low-variance estimator (as in
fig2_comparison.py, muon-only K_mu+pi_mu). Vector shape comes from the saved
full-chain observables (lepton-universal; the e+e- visible energy).
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

DET   = np.array([0.0, 1.896, 541.34])
R_OIL = S_sc.R_OIL
N_C   = 3.6312e22
POT   = S_sc.MINIBOONE_POT
_EFF_E = np.array([0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.90])
_EFF   = np.array([0.089,0.135,0.139,0.131,0.123,0.116,0.106,0.102])
def mb_eff(E):
    e = np.interp(E,_EFF_E,_EFF,left=_EFF[0],right=_EFF[-1])
    return np.where(E >= 0.140, e, 0.0)

EBINS = np.linspace(0, 1500, 61)
EC    = 0.5*(EBINS[:-1]+EBINS[1:])

def channel_hist(S, channel, n_dec=400):
    """Single-photon Dark-Primakoff signal E_vis hist (counts/fine-bin), benchmark."""
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
    gK=E/m_M; bK=pmag/E; prefm=w*POT*br
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

# ---- Digitized MiniBooNE nu-mode E_vis (Dutta-Kim Fig.2 bottom-left) ----
DATA_E   = np.array([225,275,325,375,425,475,525,575,625,675,725,775,825,875,925,975,1025,1075,1125],float)
DATA_N   = np.array([302,402,333,279,189,168,134,118, 81, 83, 75, 84, 57, 61, 38, 52, 26, 19, 19],float)
DATA_ERR = np.array([ 36, 41, 38, 35, 28, 27, 25, 23, 18, 19, 18, 19, 15, 18, 13, 15, 11, 12, 10],float)
BKG      = np.array([255,320,300,250,175,150,120,105, 76, 78, 70, 76, 52, 55, 35, 47, 24, 18, 17],float)
PBINS = np.concatenate([[DATA_E[0]-25], DATA_E+25])

def rebin_fine(fine_hist):
    idx = np.digitize(EC, PBINS)-1
    out = np.zeros(len(DATA_E))
    for i,b in enumerate(idx):
        if 0 <= b < len(out): out[b]+=fine_hist[i]
    return out

def vector_hist():
    """Vector portal (e+e-) E_vis histogram in ABSOLUTE event units, from the
    full-chain observables (POT=18.75e20, CALIB_VECTOR applied) -> data bins."""
    f = "output/MiniBooNE_VectorPortal_multichannel_SBND_observables.npz"  # NB stem mislabeled _SBND
    d = np.load(f)
    Ev = d["E_vis"]*1e3; w = d["weight"]          # GeV->MeV, w already in events
    return np.histogram(Ev, bins=PBINS, weights=w)[0]

def fit_and_panel(ax, sig, title, norm_kind="coupling", color="#c0504d"):
    """sig is in absolute benchmark event units. Fit overall amplitude A to the
    excess. norm_kind:
      'coupling' (scalar/pseudo): N_S ∝ (g_mu g_n lambda)^2, so coupling ∝ sqrt(A).
      'rate'     (vector, double-mediator): report A as a rate factor only; the
                 coupling scaling is model-specific, not a single sqrt(A)."""
    excess = DATA_N - BKG
    wgt = 1.0/DATA_ERR**2
    denom = np.sum(wgt*sig*sig)
    A = max(np.sum(wgt*sig*excess)/denom, 0.0) if denom>0 else 0.0
    sig_fit = A*sig
    chi2 = np.sum(((DATA_N-(BKG+sig_fit))**2)*wgt)
    inwin = (DATA_E>=140)&(DATA_E<=300)
    edges = PBINS
    ax.stairs(BKG+sig_fit, edges, fill=True, color=color, alpha=0.9, zorder=1,
              label="our signal (fit)")
    ax.stairs(BKG, edges, fill=True, color="#cdab7e", zorder=2, label="MiniBooNE bkg")
    ax.stairs(BKG+sig_fit, edges, fill=False, color="0.4", lw=0.8, zorder=3)
    ax.errorbar(DATA_E, DATA_N, yerr=DATA_ERR, fmt="o", color="k", ms=3.5,
                capsize=2, lw=1, zorder=5, label="MiniBooNE data")
    ax.set_xlabel(r"$E_{vis}$ [MeV]"); ax.set_ylabel("Counts")
    ax.set_xlim(DATA_E[0]-25, 1250); ax.set_ylim(0, 480)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title(title, fontsize=10)
    frac_in = sig[inwin].sum()/sig.sum() if sig.sum()>0 else 0.0
    bench_in = sig[inwin].sum()           # benchmark events in [140,300] before fit
    if norm_kind == "coupling":
        txt = ("A = %.3f $\\times$ bench\n$g$-product %.2f $\\times$ bench\n"
               "bench in-win = %.0f ev" % (A, math.sqrt(A), bench_in))
    else:  # rate
        txt = ("A = %.3f $\\times$ bench\n(double-mediator:\ncoupling map differs)\n"
               "bench in-win = %.0f ev" % (A, bench_in))
    ax.text(0.96,0.70, txt, transform=ax.transAxes, ha="right", va="top",
            fontsize=8, bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    return A, chi2, frac_in

def main():
    print("Scalar shape ...");      hsc = channel_hist(S_sc,"K_mu") + channel_hist(S_sc,"pi_mu")
    print("Pseudoscalar shape ..."); hps = channel_hist(S_ps,"K_mu") + channel_hist(S_ps,"pi_mu")
    print("Vector hist (from full-chain, event units) ...")
    try:    hvec = vector_hist()
    except Exception as e:
        print("  vector npz unavailable (%s) -- skipping vector panel" % e); hvec=None

    panels = [("Scalar ($\\phi$ Dark Primakoff)", rebin_fine(hsc), "coupling"),
              ("Pseudoscalar ($a$ Dark Primakoff)", rebin_fine(hps), "coupling")]
    if hvec is not None:
        panels.append(("Vector (double-mediator $\\to e^+e^-$)", hvec, "rate"))

    fig, axes = plt.subplots(1, len(panels), figsize=(6.0*len(panels), 5.0), squeeze=False)
    for ax,(title,sig,nk) in zip(axes[0], panels):
        A,chi2,fin = fit_and_panel(ax, sig, title, norm_kind=nk)
        print("  %-40s A=%.3f  g-prod=%.2fx  in-window=%.0f%%  chi2=%.1f"
              %(title, A, math.sqrt(A), 100*fin, chi2))
    fig.suptitle(r"MiniBooNE $\nu$-mode, Fig.2 style — our signal fit-normalized to the excess "
                 r"(bkg+data digitized from Dutta-Kim Fig.2)", fontsize=11)
    os.makedirs("output", exist_ok=True)
    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig("output/MiniBooNE_fig2_paperstyle.png", dpi=140); plt.close(fig)
    print("Saved -> output/MiniBooNE_fig2_paperstyle.png")

if __name__ == "__main__":
    main()
