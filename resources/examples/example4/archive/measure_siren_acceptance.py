"""
Measure SIREN's REALIZED phi acceptance directly from the event tree, to decide
whether the chain reproduces the true decay-smeared A_fid = 1.3e-3 (measure_
acceptance.py) or over-counts it ~40x (the phi||meson value 5.2e-2).

Runs the scalar K_mu PHYSICAL-ONLY injector, extracts each event's phi momentum
direction and meson-decay vertex, and asks what fraction of generated phi point
within R_FID of the detector (= the directional acceptance SIREN actually used).
"""
import os, math
import numpy as np
from siren import _util
import siren
from siren.Injector import Injector
from siren import injection, distributions

S = _util.load_module("ScalarMB", os.path.join(
    os.path.dirname(__file__), "ScalarPortal_MiniBooNE_multichannel.py"))
BNB = S._BNB
DET = np.array([0.0, 1.896, 541.34]); R_FID = S.R_FID

def main(n_events=20000):
    det = siren.utilities.load_detector("SBN", detector="MiniBooNE")
    data = BNB.generate_bnb_sample(n_per_species=50000, seed=7)
    name="K_mu"; parent_pdg,m_meson,m_lepton,lepton_pdg,nu_pdg,gamma_sm = S.CHANNELS[name]
    inject_sphere = siren.geometry.Sphere(S.R_OIL,0.0)
    fiducial_box  = siren.geometry.Sphere(S.R_FID,0.0)
    targets = S.build_geometric_targets(det, fiducial_box)
    chain = S.build_onshell_models(parent_pdg,m_meson,m_lepton,lepton_pdg,nu_pdg)
    meson_decay = chain["meson_decay"]; primakoff = chain["models"]["primakoff"]
    secondary_interactions = chain["secondary_interactions"]
    meson_dist = S.load_dk2nu_mesons(data, parent_pdg, det, primakoff=primakoff)
    sv_bounded = distributions.SecondaryBoundedVertexDistribution(inject_sphere)
    secondary_ps = S.build_onshell_phase_spaces(targets, chain)
    sig = meson_decay.GetPossibleSignatures()[0]
    # physical-only primary phase space (no DetectorDirected bias)
    primary_ps = {sig: S._mc([injection.PhysicalDecayChannel(meson_decay, sig)], [1.0])}
    PT=S.PT; PHI=S.PHI
    injector = Injector(number_of_events=n_events, detector_model=det, seed=11,
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

    it=iter(injector); ns=0; got=0; hit=0
    while got < n_events:
        try: ev=next(it)
        except StopIteration: break
        except RuntimeError:
            ns+=1
            if ns>20*n_events: break
            continue
        if not ev.tree: continue
        got+=1
        # extract phi momentum direction + meson-decay vertex
        dpos=None; pdir=None
        for datum in ev.tree:
            r=datum.record; secs=[int(s) for s in r.signature.secondary_types]
            if 5919 in secs:
                i=secs.index(5919); pp=r.secondary_momenta[i]
                pv=np.array([pp[1],pp[2],pp[3]]); n=np.linalg.norm(pv)
                if n>0:
                    pdir=pv/n
                    dpos=np.array(r.interaction_vertex[:3])/100.0  # cm->m
                break
        if pdir is None: continue
        rel = DET - dpos; tcl = float(rel @ pdir)
        miss = np.linalg.norm((dpos + tcl*pdir) - DET)
        if tcl>0 and miss<R_FID: hit+=1

    A_siren = hit/got if got else 0
    print("generated=%d  phi pointing at detector=%d" % (got, hit))
    print("SIREN realized directional A_fid = %.3e" % A_siren)
    print("  true decay-smeared A_fid       = 1.30e-03")
    print("  phi||meson (collinear) A_fid   = 5.23e-02")
    print("  isotropic Omega/4pi            = 2.14e-05")
    if A_siren>0:
        print("  SIREN/true = %.1fx" % (A_siren/1.30e-3))

if __name__=="__main__":
    main()
