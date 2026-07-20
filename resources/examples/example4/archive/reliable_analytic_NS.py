"""
Low-variance RELIABLE analytic N_S for scalar K_mu (MiniBooNE/BNB), folding in the
full phi energy spectrum + decay angular smearing + per-phi chord through the oil.
This replaces the high-variance SIREN physical-only estimator (which is dominated by
a handful of rare contributing phi).

N_S = sum_kaons nimpwt*POT*BR * <hit ? sigma(E_phi_lab)*n_C*chord_oil : 0>_decays

Uses N_DEC decays per kaon (isotropic in K rest frame, E*_phi from the real 3-body
spectrum), boosts to lab, ray-intersects the R_OIL sphere at the detector.
"""
import os, math
import numpy as np
from siren import _util

_SCRIPT = os.environ.get("PORTAL_SCRIPT", "ScalarPortal_MiniBooNE_multichannel.py")
_LABEL = "pseudoscalar" if "Pseudo" in _SCRIPT else "scalar"
S = _util.load_module("PortalMB", os.path.join(os.path.dirname(__file__), _SCRIPT))
# COUPLING_SCALE multiplies lambda => scales the product (g_mu g_n lambda) and hence
# N_S by COUPLING_SCALE^2 (sigma ~ lambda^2). Used to go from the Table-II benchmark
# to the Table-I best-fit product: scalar 1.0 (2.2e-8==2.2e-8), pseudo 5.9/6.5=0.9077.
_CSCALE = float(os.environ.get("COUPLING_SCALE", "1.0"))
if _CSCALE != 1.0 and hasattr(S, "LAMBDA"):
    S.LAMBDA = S.LAMBDA * _CSCALE
    print("  [COUPLING_SCALE=%.4f -> LAMBDA=%.4g, best-fit product]" % (_CSCALE, S.LAMBDA))
BNB = S._BNB
DET = np.array([0.0, 1.896, 541.34])      # m
R_OIL = S.R_OIL                            # m (5.746)
N_C = 3.6312e22                            # carbon nuclei /cm^3

# MiniBooNE single-photon detection efficiency vs E_gamma [GeV] (E_gamma ~ E_phi).
# Source: panorama review arXiv:2308.02543 (from MiniBooNE single-photon analyses);
# ~flat 0.09-0.14, 15% systematic; eff=0.102 above 0.9 GeV; analysis threshold 0.14 GeV.
_EFF_E = np.array([0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.90])
_EFF   = np.array([0.089,0.135,0.139,0.131,0.123,0.116,0.106,0.102])
_E_THRESH = 0.140
def mb_eff(E):
    e = np.interp(E, _EFF_E, _EFF, left=_EFF[0], right=_EFF[-1])
    return np.where(E >= _E_THRESH, e, 0.0)

def main(n_dec=400, channel="K_mu"):
    parent_pdg,m_meson,m_lepton,lepton_pdg,nu_pdg,gamma_sm = S.CHANNELS[channel]
    chain = S.build_onshell_models(parent_pdg,m_meson,m_lepton,lepton_pdg,nu_pdg)
    md = chain["meson_decay"]._decay
    dp = chain["models"]["primakoff"]._dp
    m_phi = md.m_phi
    br = chain["meson_decay"]._total_width/gamma_sm
    POT = S.MINIBOONE_POT
    data = BNB.generate_bnb_sample(n_per_species=50000, seed=7)
    isp = data["ptype"]==parent_pdg
    p = np.stack([data["px"][isp],data["py"][isp],data["pz"][isp]],axis=1)
    v = np.stack([data["vx"][isp],data["vy"][isp],data["vz"][isp]],axis=1)/100.0  # m
    E = data["E"][isp]; w = data["nimpwt"][isp]
    pmag = np.linalg.norm(p,axis=1); dirK = p/pmag[:,None]
    N_meson = w.sum()*POT; N_phi = N_meson*br
    nK = len(pmag)

    # phi rest-frame energy spectrum
    Emax = (m_meson**2+m_phi**2-m_lepton**2)/(2*m_meson)
    Eg = np.linspace(m_phi+1e-5, Emax-1e-5, 500)
    dN = np.array([max(md.differential_decay_rate([e])[0],0.0) for e in Eg])
    dN = np.where(np.isfinite(dN)&(dN>0),dN,0.0); cdf=np.cumsum(dN); cdf/=cdf[-1]
    # sigma interpolation table (cm^2) vs lab E_phi
    Etab = np.concatenate([np.linspace(0.001,0.3,120), np.linspace(0.31,8,120)])
    stab = np.array([dp.total_xsec(float(e)) for e in Etab])

    rng = np.random.default_rng(1)
    # transverse basis per kaon
    arb = np.tile(np.array([0.,1.,0.]),(nK,1)); m=np.abs(dirK[:,1])>0.9
    arb[m]=np.array([1.,0.,0.])
    xc=np.cross(dirK,arb); xc/=np.linalg.norm(xc,axis=1)[:,None]; yc=np.cross(dirK,xc)
    gK=E/m_meson; bK=pmag/E
    acc = np.zeros(nK)            # raw: mean (hit? sigma*nC*chord) per kaon
    acc_eff = np.zeros(nK)        # x mb_eff(E_gamma), full range above threshold
    acc_effw = np.zeros(nK)       # x mb_eff, restricted to [0.140,0.300] GeV window
    Ehit=[]
    for _ in range(n_dec):
        u=rng.random(nK); Estar=np.interp(u,cdf,Eg)
        pstar=np.sqrt(np.maximum(Estar**2-m_phi**2,0))
        cth=rng.uniform(-1,1,nK); sth=np.sqrt(1-cth**2); az=rng.uniform(0,2*math.pi,nK)
        pl=pstar*cth; pt=pstar*sth
        Elab=gK*(Estar+bK*pl); pl_lab=gK*(pl+bK*Estar)
        pphi=pl_lab[:,None]*dirK + (pt*np.cos(az))[:,None]*xc + (pt*np.sin(az))[:,None]*yc
        dphi=pphi/np.linalg.norm(pphi,axis=1)[:,None]
        # ray (v + t dphi) intersect sphere(DET,R_OIL): chord = 2 sqrt(disc)
        mvec=v-DET[None,:]; bdot=np.einsum("ij,ij->i",mvec,dphi)
        disc=bdot**2-(np.einsum("ij,ij->i",mvec,mvec)-R_OIL**2)
        tmid=-bdot
        hit=(disc>0)&(tmid>0)
        chord=np.where(hit,2*np.sqrt(np.maximum(disc,0)),0.0)*100.0  # m->cm
        sig=np.interp(Elab,Etab,stab)
        base=np.where(hit, sig*N_C*chord, 0.0)
        eff=mb_eff(Elab)
        acc += base
        acc_eff += base*eff                                   # eff-weighted, full range
        inwin = (Elab>=0.140)&(Elab<=0.300)
        acc_effw += np.where(inwin, base*eff, 0.0)            # eff-weighted, excess window
        if len(Ehit)<5000: Ehit.extend(Elab[hit].tolist())
    acc/=n_dec; acc_eff/=n_dec; acc_effw/=n_dec
    # N_S = sum over kaons of nimpwt*POT*BR * <scatter prob>
    pref = w*POT*br
    N_S       = (pref*acc).sum()
    N_S_eff   = (pref*acc_eff).sum()
    N_S_effw  = (pref*acc_effw).sum()
    Ehit=np.array(Ehit)
    print("==== RELIABLE ANALYTIC N_S  (%s %s) ====" % (_LABEL, channel))
    print("  N_K=%.3e  BR=%.3e  N_phi=%.3e   (n_dec=%d, %d kaons => %.1e samples)"
          %(N_meson,br,N_phi,n_dec,nK,n_dec*nK))
    # overall acceptance & mean scatter prob for reporting
    A_fid = ((acc>0).sum()/nK)  # rough; better: hit fraction
    print("  <E_phi|hit>=%.3f GeV  median=%.3f  [5,95]=%s"
          %(Ehit.mean(),np.median(Ehit),np.round(np.percentile(Ehit,[5,95]),3)))
    print("  N_S(%s): raw=%.3e  effwtd(full)=%.3e  effwtd(window)=%.3e"
          % (channel, N_S, N_S_eff, N_S_effw))
    return N_S, N_S_eff, N_S_effw

def all_channels(n_dec=400):
    print("\n" + "="*60)
    print("  RELIABLE ANALYTIC N_S  -  all channels (%s, CALIB=1)" % _LABEL)
    print("="*60)
    muon_only = os.environ.get("MUON_ONLY", "0") == "1"
    chans = [c for c in S.CHANNELS if (not muon_only or c.endswith("_mu"))]
    if muon_only: print("  [MUON-ONLY: g_e=0, paper couples mediator only to muons]")
    tot=0.0; tote=0.0; totew=0.0; rows=[]
    for ch in chans:
        try:
            ns,nse,nsew=main(n_dec=n_dec, channel=ch)
            tot+=ns; tote+=nse; totew+=nsew; rows.append((ch,ns,nse,nsew))
        except Exception as e:
            print("  %s: SKIP (%s)"%(ch,e)); rows.append((ch,0.0,0.0,0.0))
    print("\n" + "="*64)
    print("  SUMMARY (%s, code couplings/BR, CALIB=1) + MiniBooNE single-gamma eff" % _LABEL)
    print("="*64)
    print("  %-6s   %-11s %-13s %-13s" % ("chan","raw","eff*full","eff*[.14,.30]"))
    for ch,ns,nse,nsew in rows:
        print("  %-6s : %.3e   %.3e     %.3e" % (ch, ns, nse, nsew))
    print("  %-6s : %.3e   %.3e     %.3e" % ("SUM", tot, tote, totew))
    print("  ---")
    print("  paper excess = 320 events (E_vis < 300 MeV)")
    print("  predicted (eff-weighted, [0.14,0.30] window) = %.0f events  -> %.1fx over 320" % (totew, totew/320.0))
    print("  after production factor-2 correction          = %.0f events  -> %.1fx over 320" % (totew/2.0, totew/2.0/320.0))
    return tot, tote, totew

if __name__=="__main__":
    all_channels()
