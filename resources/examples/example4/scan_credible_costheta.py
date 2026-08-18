"""
#1-plus: MiniBooNE credible regions WITH the cos-theta constraint -> localized BLOBS.

Joint chi2 = chi2(E_vis, real nu-mode data) + chi2(cos-theta shape vs paper template).
The E_vis term fixes the coupling at each m_Zp (a ribbon); the cos-theta term is a
mass PRIOR (its shape depends on m_Zp via the Primakoff opening angle) that carves
the ribbon into a compact blob -- the mechanism the paper's multi-observable fit uses.

cos-theta target = paper Fig.2 (scalar, nu-mode) RED signal-band shape, pixel-
extracted (cos_template_nu.json).  nu-bar mode is NOT added: our BNB dk2nu flux is
nu-mode only.  Errors: multinomial (Neff=320 excess) + 0.01 digitization floor.

Caching: production hits (E_vis, cos_mediator, G=w/sigma_ref) are m_Zp-independent;
per m_Zp we only re-weight by sigma(E,m_Zp) and re-smear cos by the m_Zp Primakoff angle.

Run:  DK2NU_FILE=/home/shubham/nubeamHighSample.dk2nu.root \
      /home/shubham/siren_venv/bin/python scan_credible_costheta.py
Out: output/credible_region_costheta.npz + output/our_fig3_blobs.png
"""
import os, json, importlib.util, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

os.environ.setdefault("DK2NU_FILE", "/home/shubham/nubeamHighSample.dk2nu.root")
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
DP = load(os.path.join(PKG, "DarkPrimakoff.py"), "DP")

# --- E_vis data (nu-mode, digitized) ---
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
PBINS = np.concatenate([[DATA_E[0]-25], DATA_E+25]) / 1e3

# --- cos-theta template (paper nu-mode red band) ---
ct = json.load(open(os.path.join(HERE, "cos_template_nu.json")))
COS_CTR = np.array(ct["cos"]); COS_TGT = np.array(ct["shape"]); COS_TGT/=COS_TGT.sum()
NCB = len(COS_CTR)
COS_EDGES = np.linspace(-1,1,NCB+1)
COS_SIG = np.sqrt(COS_TGT*(1-COS_TGT)/320.0) + 0.01     # multinomial + digitization floor

NDEC=200; WIN=(0.14,0.30)
ET_ENGINE = np.concatenate([np.linspace(0.001,0.3,120), np.linspace(0.31,9,160)])
ET_SIG = np.linspace(0.13,1.60,90)
MZP_GRID = np.geomspace(0.030,0.200,26)
PROD_GRID = np.geomspace(3e-9,1.2e-6,80)
RNG = np.random.default_rng(7)

def cache(S):
    product = S.G_MU_PROD*S.G_N*(S.LAMBDA*1e-3); ref=S.M_ZP
    Els=[];Cmed=[];Gs=[];dp=None
    for ch in [c for c in S.CHANNELS if "mu" in c]:
        pdg,m_M,m_l,lp,nu,g=S.CHANNELS[ch]
        if (m_M-m_l)<=S.M_PHI: continue
        dp=S.build_onshell_models(pdg,m_M,m_l,lp,nu)["models"]["primakoff"]._dp
        E,w,c=SA.analytic_sp_mb(S,ch,n_dec=NDEC,eff_mode="mb",return_cos=True,meson_fn=SA._mesons_dk2nu)
        E,w,c=np.asarray(E),np.asarray(w),np.asarray(c); dp.m_Zp=ref
        sig=np.interp(E,ET_ENGINE,np.array([dp.total_xsec(float(e)) for e in ET_ENGINE]))
        ok=sig>0; Els.append(E[ok]);Cmed.append(c[ok]);Gs.append(w[ok]/sig[ok])
    return np.concatenate(Els),np.concatenate(Cmed),np.concatenate(Gs),dp,product

def joint_chi2(cob):
    El,Cmed,G,dp,product=cob
    C=np.empty((len(MZP_GRID),len(PROD_GRID)))
    for i,mzp in enumerate(MZP_GRID):
        dp.m_Zp=float(mzp)
        sig=np.interp(El,ET_SIG,np.array([dp.total_xsec(float(e)) for e in ET_SIG]))
        w=G*sig
        # E_vis histogram (signal at product_ref)
        sE=np.histogram(El,bins=PBINS,weights=w)[0]
        # cos-theta shape (in-window), smeared by m_Zp Primakoff angle
        m=(El>=WIN[0])&(El<=WIN[1])
        cg=DP.smear_photon_beam(Cmed[m],El[m],dp,RNG)
        hc,_=np.histogram(cg,bins=COS_EDGES,weights=w[m])
        hc=hc/hc.sum() if hc.sum()>0 else hc
        chi2_cos=np.sum((COS_TGT-hc)**2/COS_SIG**2)
        for j,P in enumerate(PROD_GRID):
            s=sE*(P/product)**2
            chi2_E=np.sum((EXCESS-s)**2*INV2)
            C[i,j]=chi2_E+chi2_cos      # cos term = mass prior (product-independent)
    return C

import sys
CFG={"scalar":"ScalarPortal_MiniBooNE_multichannel.py","pseudo":"PseudoscalarPortal_MiniBooNE_multichannel.py"}
grids={}
for portal,f in CFG.items():
    print("caching+joint-scanning %s ..."%portal); sys.stdout.flush()
    C=joint_chi2(cache(load(f,"CT_"+portal))); grids[portal]=C
    imin=np.unravel_index(np.argmin(C),C.shape)
    print("  %s chi2_min=%.2f at m_Zp=%.0f MeV, P=%.2e"%(portal,C.min(),MZP_GRID[imin[0]]*1e3,PROD_GRID[imin[1]])); sys.stdout.flush()

np.savez(os.path.join(HERE,"output","credible_region_costheta.npz"),
         mzp_MeV=MZP_GRID*1e3,prod=PROD_GRID,chi2_scalar=grids["scalar"],chi2_pseudo=grids["pseudo"])

# ---- Fig.3-style blob plot ----
fig,ax=plt.subplots(figsize=(7.4,6.6))
X,Y=np.meshgrid(MZP_GRID*1e3,PROD_GRID,indexing="ij"); lx,ly=np.log10(X),np.log10(Y)
# paper measured blobs (faint)
PAPER={"scalar1":(45.9,2.20e-8,0.155,0.034,"#2ca25f"),"pseudo1":(63.6,5.68e-7,0.292,0.030,"#6a51a3")}
for nm,(mc,pc,hx95,hy68,col) in PAPER.items():
    for hx,hy,al in [(hx95,1.64*hy68,0.10),(0.6*hx95,hy68,0.20)]:
        ax.add_patch(Ellipse((np.log10(mc),np.log10(pc)),2*hx,2*hy,facecolor=col,edgecolor="none",alpha=al,zorder=1))
for key,lab,col in [("scalar","Scalar ($m_\phi=1$)","#00701a"),("pseudo","Pseudoscalar ($m_a=1$)","#3b0a70")]:
    D=grids[key]-grids[key].min()
    ax.contourf(lx,ly,D,levels=[0,6.18],colors=[col],alpha=0.25)
    ax.contourf(lx,ly,D,levels=[0,2.30],colors=[col],alpha=0.55)
    ax.plot([],[],color=col,lw=8,alpha=0.6,label="OUR "+lab)
ax.scatter(np.log10(49),np.log10(2.2e-8),marker="*",s=230,color="#2ca25f",edgecolor="k",zorder=7)
ax.scatter(np.log10(85),np.log10(5.9e-7),marker="*",s=230,color="#6a51a3",edgecolor="k",zorder=7)
ax.plot([],[],"k*",ms=13,label="paper Table I")
ax.plot([],[],color="0.5",lw=8,alpha=0.35,label="paper Fig.3 blobs")
ax.set_xlim(np.log10(30),np.log10(200)); ax.set_ylim(np.log10(1e-8),np.log10(1e-6))
xt=[30,40,60,100,200]; ax.set_xticks([np.log10(v) for v in xt])
ax.set_xticklabels([r"$3\times10^1$",r"$4\times10^1$",r"$6\times10^1$",r"$10^2$",r"$2\times10^2$"])
yt=[1e-8,1e-7,1e-6]; ax.set_yticks([np.log10(v) for v in yt]); ax.set_yticklabels([r"$10^{-8}$",r"$10^{-7}$",r"$10^{-6}$"])
ax.set_xlabel(r"$m_{Z'}$ [MeV]",fontsize=12); ax.set_ylabel(r"$\lambda g_\mu g_n$ [MeV$^{-1}$]",fontsize=12)
ax.set_title("Our MiniBooNE credible BLOBS (E_vis + cos$\\theta$ joint fit)\n"
             "cos$\\theta$ localizes the mass;  68% dark / 95% light;  paper blobs faint",fontsize=10.5)
ax.legend(fontsize=8,loc="lower right",framealpha=0.95); ax.grid(True,alpha=0.2)
fig.tight_layout(); fig.savefig(os.path.join(HERE,"output","our_fig3_blobs.png"),dpi=140)
print("wrote output/our_fig3_blobs.png + credible_region_costheta.npz")
