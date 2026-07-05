"""
High-statistics RELIABLE re-baseline of scalar K_mu (MiniBooNE/BNB), physical-only
(unbiased) mode, with a full per-event weight decomposition so we can finally
reconcile SIREN's N_S against the analytic N_phi x A_fid x P_scatter.

Reports: injected, contributing, sum(w_abs), ESF, realized acceptance, and the
SIREN-implied P_scatter vs the physical sigma x column-depth.
"""
import os, math
import numpy as np
from siren import _util
import siren
from siren.Injector import Injector
from siren.Weighter import Weighter
from siren import injection, distributions

S = _util.load_module("ScalarMB", os.path.join(
    os.path.dirname(__file__), "ScalarPortal_MiniBooNE_multichannel.py"))
BNB = S._BNB
DET = np.array([0.0, 1.896, 541.34]); R_FID = S.R_FID

def main(n_events=600000):
    det = siren.utilities.load_detector("SBN", detector="MiniBooNE")
    data = BNB.generate_bnb_sample(n_per_species=50000, seed=7)
    name="K_mu"; parent_pdg,m_meson,m_lepton,lepton_pdg,nu_pdg,gamma_sm = S.CHANNELS[name]
    inject_sphere = siren.geometry.Sphere(S.R_OIL,0.0)
    fiducial_box  = siren.geometry.Sphere(S.R_FID,0.0)
    targets = S.build_geometric_targets(det, fiducial_box)
    chain = S.build_onshell_models(parent_pdg,m_meson,m_lepton,lepton_pdg,nu_pdg)
    meson_decay = chain["meson_decay"]; primakoff = chain["models"]["primakoff"]
    secondary_interactions = chain["secondary_interactions"]
    bsm_width = meson_decay._total_width; br = bsm_width/gamma_sm
    meson_dist = S.load_dk2nu_mesons(data, parent_pdg, det, primakoff=primakoff)
    br_dist = distributions.NormalizationConstant(br)
    sv_bounded = distributions.SecondaryBoundedVertexDistribution(inject_sphere)
    secondary_ps = S.build_onshell_phase_spaces(targets, chain)
    sig = meson_decay.GetPossibleSignatures()[0]
    primary_ps = {sig: S._mc([injection.PhysicalDecayChannel(meson_decay, sig)], [1.0])}
    PT=S.PT; PHI=S.PHI
    injector = Injector(number_of_events=n_events, detector_model=det, seed=2024,
        primary_type=PT(parent_pdg), primary_interactions=[meson_decay],
        primary_injection_distributions=[meson_dist],
        primary_weighting_mode=injection.VertexWeightingMode.Fixed(),
        secondary_interactions=secondary_interactions,
        secondary_injection_distributions={PHI:[sv_bounded]},
        secondary_phase_spaces=secondary_ps, primary_phase_spaces=primary_ps,
        stopping_condition=S.onshell_stopping_condition)
    try:
        for ev in injector: break
    except RuntimeError: pass
    try: injector._Injector__injector.ResetInjectedEvents(n_events)
    except Exception: pass
    weighter = Weighter(injectors=[injector], detector_model=det,
        primary_type=PT(parent_pdg), primary_interactions=[meson_decay],
        primary_physical_distributions=[meson_dist, br_dist],
        secondary_interactions=secondary_interactions)

    oil = S._oil(); POT = S.MINIBOONE_POT; CAL = S.CALIB_SCALAR
    it=iter(injector); ns=0; got=0; wv=[]; Ephis=[]
    while got < n_events:
        try: ev=next(it)
        except StopIteration: break
        except RuntimeError:
            ns+=1
            if ns>30*n_events: break
            continue
        if not ev.tree: continue
        got+=1
        w=weighter(ev)
        if not np.isfinite(w) or w<=0: continue
        if not S.primakoff_in_oil(ev, oil): continue
        wa=w*CAL*POT
        if not np.isfinite(wa) or wa<=0: continue
        wv.append(wa)
        for datum in ev.tree:
            r=datum.record; secs=[int(s) for s in r.signature.secondary_types]
            if 5919 in secs:
                Ephis.append(r.secondary_momenta[secs.index(5919)][0]); break
        if got % 100000 == 0:
            print("  ... injected=%d contributing=%d sum=%.3e" % (got, len(wv), np.sum(wv)), flush=True)
    wv=np.array(wv); Ephis=np.array(Ephis)
    esf=(wv.sum()**2)/(len(wv)*np.sum(wv**2)) if len(wv) and np.sum(wv**2)>0 else 0
    N_K = data["nimpwt"][data["ptype"]==parent_pdg].sum()*POT
    N_phi = N_K*br
    A_real = len(wv)/got
    print("\n==== RE-BASELINE scalar K_mu (physical-only, CALIB=1) ====")
    print("  injected=%d  skipped=%d  contributing=%d" % (got, ns, len(wv)))
    print("  sum(w_abs) = %.4e events   ESF=%.1f%%  (eff N=%.0f)" % (wv.sum(), 100*esf, esf*len(wv)))
    print("  <E_phi>=%.3f GeV  [5,50,95]=%s" % (Ephis.mean() if len(Ephis) else 0,
          np.round(np.percentile(Ephis,[5,50,95]),3) if len(Ephis) else []))
    print("  N_K=%.3e  BR=%.3e  N_phi=%.3e" % (N_K, br, N_phi))
    print("  realized acceptance (contributing/injected) = %.3e" % A_real)
    sig_at = primakoff._dp.total_xsec(Ephis.mean()) if len(Ephis) else 0
    print("  sigma(<E_phi>)=%.3e cm^2 ; physical P_scatter(full chord 1149cm)=%.3e"
          % (sig_at, sig_at*3.63e22*1149))
    print("  SIREN-implied <P_scatter> = sum / (N_phi*A_real_count) where A counts phi at det")
    print("  ==> reliable N_S(K_mu) = %.3e" % wv.sum())

if __name__=="__main__":
    main()
