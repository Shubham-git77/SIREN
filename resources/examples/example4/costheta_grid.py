"""
Signal cos(theta) distributions for ALL detectors x models, nu and nu-bar modes.
3x3 grid: rows = {MiniBooNE, SBND, ICARUS}, cols = {scalar, pseudo, vector}.
In each panel: nu-mode (solid) + nu-bar-mode (dashed), area-normalized SHAPE.

Fluxes (real where available):
  MiniBooNE/SBND (BNB): nu = nubeam12M (pi+/K+ focused).  nu-bar: NO RHC-BNB file
     -> charge-conjugate approx (signal cos-theta is C-symmetric: RHC focuses pi-/K-
     with the same boost band as FHC pi+/K+), drawn == nu shape and LABELLED as approx.
  ICARUS (NuMI): nu = FHC (g4numi_fhc_dif_1008), nu-bar = RHC (sources/NuMI/*rhc*).
     BOTH REAL. nu selects +mesons, nu-bar selects -mesons (charge-conjugate production).

cos-theta = outgoing PHOTON (scalar/pseudo, via smear_photon_beam) or e+e- system
(vector), for in-window E_vis in [0.14,0.30] GeV.  Shapes only (POT/coupling irrelevant).

Run: /home/shubham/siren_venv/bin/python costheta_grid.py
Out: output/costheta_grid.png
"""
import os, glob, importlib.util, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

os.environ.setdefault("DK2NU_FILE", "/home/shubham/nubeam12M.dk2nu.root")   # BNB nu-mode
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

SA  = load(os.path.join(PKG, "AnalyticRate.py"), "AnalyticRate")
DP  = load(os.path.join(PKG, "DarkPrimakoff.py"), "DPmod")
from siren import _util
GEO = _util.load_module("sbn_geometry", os.path.join(_util.resource_package_dir(),
                        "detectors", "SBN", "SBN-v1", "sbn_geometry.py"))
T   = GEO.transform("NuMI", "BNB")

NUMI_FHC = ["/home/shubham/g4numi_fhc_dif_1008.dk2nu.root"]
NUMI_RHC = sorted(glob.glob(os.path.join(HERE, "sources", "NuMI", "g4numi*rhc*.root")))
WIN=(0.14,0.30); NDEC=150; RNG=np.random.default_rng(3)
COS_EDGES=np.linspace(-1,1,25); COS_CTR=0.5*(COS_EDGES[:-1]+COS_EDGES[1:])

# --- meson_fn factories ---
def bnb_nu(S,pdg,**k):    return SA._mesons_dk2nu(S, pdg)               # pi+/K+ from nubeam12M
def numi_fhc(S,pdg,**k):  return S._DK.analytic_meson_source(NUMI_FHC, pdg,  beam_transform=T, n_max=120000)
def numi_rhc(S,pdg,**k):  return S._DK.analytic_meson_source(NUMI_RHC, -pdg, beam_transform=T, n_max=120000)  # -mesons focused

CFG = {  # portal -> (scalar/pseudo config per detector, vector config)
 "MiniBooNE": {"scalar":"ScalarPortal_MiniBooNE_multichannel.py","pseudo":"PseudoscalarPortal_MiniBooNE_multichannel.py","vector":"VectorPortal_MiniBooNE_fullchain.py","geom":"sphere"},
 "SBND":      {"scalar":"ScalarPortal_SBND_multichannel.py","pseudo":"PseudoscalarPortal_SBND_multichannel.py","vector":"VectorPortal_SBND_fullchain.py","geom":"box"},
 "ICARUS":    {"scalar":"ScalarPortal_ICARUS_multichannel.py","pseudo":"PseudoscalarPortal_ICARUS_multichannel.py","vector":"VectorPortal_ICARUS_fullchain.py","geom":"boxI"},
}

def cos_shape(det, portal, mode):
    """Return normalized cos-theta histogram (over COS_CTR) for det/portal/mode."""
    cfg=CFG[det]; vector=(portal=="vector"); geom=cfg["geom"]
    S=load(cfg[portal], f"{det}_{portal}_{mode}")
    # pick meson_fn + engine per detector/mode
    if det in ("MiniBooNE","SBND"):
        mf = bnb_nu          # nu-mode; nu-bar handled as charge-conj (== nu) by caller
    else:
        mf = numi_fhc if mode=="nu" else numi_rhc
    cos_all=[]; w_all=[]
    chans = list(S.CHANNELS) if vector else [c for c in S.CHANNELS if "mu" in c]
    for ch in chans:
        pdg,m_M,m_l,lp,nu,g=S.CHANNELS[ch]
        thr = S.M_V1 if vector else S.M_PHI
        if (m_M-m_l)<=thr: continue
        if geom=="sphere":
            fn = SA.analytic_vec_mb if vector else SA.analytic_sp_mb
            E,w,c = fn(S,ch,n_dec=NDEC,eff_mode="mb",return_cos=True,meson_fn=mf)
        else:
            fn = SA.analytic_vec if vector else SA.analytic_sp
            det_ctr = np.asarray(GEO.detector_center("SBND","BNB"),float) if geom=="box" \
                      else None   # ICARUS: sum modules below
            if geom=="boxI":
                E=[];w=[];c=[]
                for ctr in S.ICARUS_MODULE_CENTERS_BNB:
                    e2,w2,c2=fn(S,ch,n_dec=NDEC,return_cos=True,eff_mode="raw",det=np.asarray(ctr,float),pot=1.0,meson_fn=mf)
                    E.append(np.asarray(e2));w.append(np.asarray(w2));c.append(np.asarray(c2))
                E=np.concatenate(E);w=np.concatenate(w);c=np.concatenate(c)
            else:
                E,w,c = fn(S,ch,n_dec=NDEC,return_cos=True,eff_mode="raw",det=det_ctr,pot=1.0,meson_fn=mf)
        E,w,c=np.asarray(E),np.asarray(w),np.asarray(c)
        if E.size==0: continue
        if not vector:                       # scalar/pseudo: mediator dir -> photon dir
            dp=S.build_onshell_models(pdg,m_M,m_l,lp,nu)["models"]["primakoff"]._dp
            c=DP.smear_photon_beam(c,E,dp,RNG)
        m=(E>=WIN[0])&(E<=WIN[1])
        cos_all.append(c[m]); w_all.append(w[m])
    if not cos_all: return np.zeros(len(COS_CTR))
    c=np.concatenate(cos_all); w=np.concatenate(w_all)
    h,_=np.histogram(c,bins=COS_EDGES,weights=w)
    return h/h.sum() if h.sum()>0 else h

import sys
DETS=["MiniBooNE","SBND","ICARUS"]; PORTALS=["scalar","pseudo","vector"]
results={}
for det in DETS:
    for portal in PORTALS:
        print(f"computing {det} {portal} nu ..."); sys.stdout.flush()
        nu = cos_shape(det,portal,"nu")
        if det=="ICARUS":
            print(f"computing {det} {portal} nubar (RHC) ..."); sys.stdout.flush()
            nb = cos_shape(det,portal,"nubar")
            approx=False
        else:
            nb = nu.copy(); approx=True        # charge-conjugate approx (no RHC-BNB)
        results[(det,portal)]=(nu,nb,approx)

fig,axes=plt.subplots(3,3,figsize=(13,10.5),sharex=True)
for i,det in enumerate(DETS):
    for j,portal in enumerate(PORTALS):
        ax=axes[i,j]; nu,nb,approx=results[(det,portal)]
        ax.step(COS_CTR,nu,where="mid",color="#0072B2",lw=2,label=r"$\nu$-mode")
        ax.step(COS_CTR,nb,where="mid",color="#D55E00",lw=2,ls="--",
                label=(r"$\bar\nu$ (C-conj approx)" if approx else r"$\bar\nu$-mode (RHC)"))
        if i==0: ax.set_title(portal, fontsize=12)
        if j==0: ax.set_ylabel(f"{det}\nnorm. events", fontsize=10)
        if i==2: ax.set_xlabel(r"cos$\theta$ (photon / $e^+e^-$)")
        ax.legend(fontsize=7); ax.grid(alpha=0.25); ax.set_xlim(-1,1)
fig.suptitle("Signal cos$\\theta$ shapes -- all detectors x models, $\\nu$ / $\\bar\\nu$ modes  "
             "(in-window $E_{vis}$; shapes normalized)\n"
             "ICARUS $\\bar\\nu$ = real RHC NuMI; BNB $\\bar\\nu$ = charge-conjugate approx (no RHC-BNB flux)", fontsize=11)
fig.tight_layout(rect=[0,0,1,0.96])
fig.savefig(os.path.join(HERE,"output","costheta_grid.png"),dpi=130)
print("wrote output/costheta_grid.png")
