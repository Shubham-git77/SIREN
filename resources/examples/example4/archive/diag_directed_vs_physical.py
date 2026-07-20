"""
Decisive test of the directed-channel acceptance: importance sampling is
unbiased, so the DIRECTED primary phase space and a PHYSICAL-ONLY one must give
the same physical sum(w_abs). If they differ, the ratio is the directed-channel
over-count factor (and the physical run is the correct answer).

Runs scalar K_mu both ways and compares sum(w_abs).
"""
import os, numpy as np
from siren import _util
import siren
from siren.Injector import Injector
from siren.Weighter import Weighter
from siren import injection, distributions

S = _util.load_module("ScalarMB", os.path.join(
    os.path.dirname(__file__), "ScalarPortal_MiniBooNE_multichannel.py"))
BNB = S._BNB


def run(mode, n_events, seed=11):
    det = siren.utilities.load_detector("SBN", detector="MiniBooNE")
    data = BNB.generate_bnb_sample(n_per_species=50000, seed=3)
    name="K_mu"; parent_pdg,m_meson,m_lepton,lepton_pdg,nu_pdg,gamma_sm = S.CHANNELS[name]
    inject_sphere = siren.geometry.Sphere(S.R_OIL,0.0)
    fiducial_box  = siren.geometry.Sphere(S.R_FID,0.0)
    targets = S.build_geometric_targets(det, fiducial_box)
    chain = S.build_onshell_models(parent_pdg,m_meson,m_lepton,lepton_pdg,nu_pdg)
    meson_decay = chain["meson_decay"]; primakoff = chain["models"]["primakoff"]
    secondary_interactions = chain["secondary_interactions"]
    br_bsm = meson_decay._total_width/gamma_sm
    meson_dist = S.load_dk2nu_mesons(data, parent_pdg, det, primakoff=primakoff)
    br_dist = distributions.NormalizationConstant(br_bsm)
    sv_bounded = distributions.SecondaryBoundedVertexDistribution(inject_sphere)
    secondary_ps = S.build_onshell_phase_spaces(targets, chain)

    sig = meson_decay.GetPossibleSignatures()[0]
    if mode == "directed":
        primary_ps = S.build_primary_phase_spaces(targets, meson_decay)
    else:  # physical-only: no DetectorDirected channel
        primary_ps = {sig: S._mc([injection.PhysicalDecayChannel(meson_decay, sig)], [1.0])}

    PT=S.PT; PHI=S.PHI
    injector = Injector(number_of_events=n_events, detector_model=det, seed=seed,
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

    oil=S._oil()
    it=iter(injector); ns=0; got=0; wsum=0.0; wv=[]
    POT=S.MINIBOONE_POT; CAL=S.CALIB_SCALAR
    while got < n_events:
        try: ev=next(it)
        except StopIteration: break
        except RuntimeError:
            ns+=1
            if ns>20*n_events: break
            continue
        if not ev.tree: continue
        got+=1
        w=weighter(ev)
        if not np.isfinite(w) or w<=0: continue
        if not S.primakoff_in_oil(ev, oil): continue
        wa=w*CAL*POT
        if not np.isfinite(wa) or wa<=0: continue
        wsum+=wa; wv.append(wa)
    wv=np.array(wv)
    esf=(wv.sum()**2)/(len(wv)*np.sum(wv**2)) if len(wv) and np.sum(wv**2)>0 else 0
    print("  [%-9s] generated=%d skipped=%d  contributing=%d  sum(w_abs)=%.3e  ESF=%.1f%%"
          %(mode, got, ns, len(wv), wsum, 100*esf))
    return wsum


if __name__=="__main__":
    print("Directed vs Physical-only primary phase space (scalar K_mu):")
    d = run("directed", 2000)
    p = run("physical", 40000)
    if p>0:
        print("\n  RATIO directed/physical = %.2f"%(d/p))
        print("  => directed-channel over-counts by this factor (physical = unbiased truth)")
