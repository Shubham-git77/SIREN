"""
Directed-channel acceptance diagnostic (scalar K_mu, MiniBooNE/BNB).

Density + sigma are already confirmed correct, so the ~10x over-normalization
must sit in the DetectorDirected channel's solid-angle weighting. This script
rebuilds the K_mu injector/weighter, extracts per event the phi four-momentum,
the meson decay position, and the SIREN weight, and compares SIREN's effective
flux of phi reaching the detector to the analytic geometric expectation.
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

DET = np.array([0.0, 1.896, 541.34])   # m, MiniBooNE center (BNB coords)
R_FID = S.R_FID
N_C = 3.6312e22                         # /cm^3 (confirmed from detector model)

def main(n_events=600):
    det = siren.utilities.load_detector("SBN", detector="MiniBooNE")
    data = BNB.generate_bnb_sample(n_per_species=50000, seed=7)

    name="K_mu"; parent_pdg,m_meson,m_lepton,lepton_pdg,nu_pdg,gamma_sm = S.CHANNELS[name]
    inject_sphere = siren.geometry.Sphere(S.R_OIL,0.0)
    fiducial_box  = siren.geometry.Sphere(S.R_FID,0.0)
    targets = S.build_geometric_targets(det, fiducial_box)
    chain = S.build_onshell_models(parent_pdg,m_meson,m_lepton,lepton_pdg,nu_pdg)
    meson_decay = chain["meson_decay"]; primakoff = chain["models"]["primakoff"]
    secondary_interactions = chain["secondary_interactions"]
    bsm_width = meson_decay._total_width; br_bsm = bsm_width/gamma_sm
    meson_dist = S.load_dk2nu_mesons(data, parent_pdg, det, primakoff=primakoff)
    br_dist = distributions.NormalizationConstant(br_bsm)
    primary_mode = injection.VertexWeightingMode.Fixed()
    sv_bounded = distributions.SecondaryBoundedVertexDistribution(inject_sphere)
    primary_ps = S.build_primary_phase_spaces(targets, meson_decay)
    secondary_ps = S.build_onshell_phase_spaces(targets, chain)
    PT = S.PT; PHI = S.PHI

    injector = Injector(number_of_events=n_events, detector_model=det, seed=7,
        primary_type=PT(parent_pdg), primary_interactions=[meson_decay],
        primary_injection_distributions=[meson_dist], primary_weighting_mode=primary_mode,
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

    events=[]; it=iter(injector); ns=0
    while len(events)<n_events:
        try: e=next(it)
        except StopIteration: break
        except RuntimeError:
            ns+=1
            if ns>5*n_events: break
            continue
        if e.tree: events.append(e)

    oil = S._oil(); fidv = S._fiducial()
    POT = S.MINIBOONE_POT; CAL = S.CALIB_SCALAR
    rows=[]
    for ev in events:
        w = weighter(ev)
        if not np.isfinite(w) or w<=0: continue
        if not S.primakoff_in_oil(ev, oil): continue
        # extract phi four-momentum + meson decay position from the tree
        Ephi=None; dpos=None
        for datum in ev.tree:
            r=datum.record; secs=[int(s) for s in r.signature.secondary_types]
            if 5919 in secs and dpos is None:    # meson decay vertex
                dpos=np.array([r.interaction_vertex[0],r.interaction_vertex[1],r.interaction_vertex[2]])
                i=secs.index(5919); p=r.secondary_momenta[i]
                Ephi=p[0]; pdir=np.array([p[1],p[2],p[3]]); pdir=pdir/np.linalg.norm(pdir)
        if Ephi is None: continue
        rows.append((w, Ephi, dpos, pdir))

    print("scattered events extracted:", len(rows))
    w_abs = np.array([r[0] for r in rows])*CAL*POT
    Ephi = np.array([r[1] for r in rows])
    print("SIREN sum(w_abs) over these = %.3e events"%w_abs.sum())
    print("  <E_phi>=%.2f GeV  E_phi[5,50,95]=%s"%(Ephi.mean(), np.round(np.percentile(Ephi,[5,50,95]),2)))

    # analytic per-event geometric acceptance + scatter probability
    sig_fn = S._build_sigma_interp(primakoff)  # NB: normalized to O(1); need raw sigma:
    raw_sigma = lambda E: primakoff._dp.total_xsec(float(E))
    Ageo=[]; Psc=[]
    for (w,E,dpos,pdir) in rows:
        dpos_m = np.array(dpos)/100.0
        Lv = DET - dpos_m; L = np.linalg.norm(Lv)
        Omega = math.pi*R_FID**2 / L**2          # detector solid angle [sr]
        # boost enhancement: phi forward-peaked; use dN/dOmega(0)/(1/4pi) ~ 4 gamma^2
        # approximate gamma from E_phi and a typical phi mass scale -> use meson boost proxy
        # here measure enhancement empirically instead: acceptance ~ Omega/4pi * enh
        Ageo.append(Omega/(4*math.pi))           # ISOTROPIC baseline (no boost)
        L_oil = 2*S.R_OIL*100                     # cm, max chord
        Psc.append(raw_sigma(E)*N_C*L_oil)
    Ageo=np.array(Ageo); Psc=np.array(Psc)
    print("\n  [geom] isotropic acceptance Omega/4pi: mean=%.3e"%Ageo.mean())
    print("  [scat] P_scatter (sigma*n_C*2R_oil): mean=%.3e"%Psc.mean())
    print("  Note: SIREN weight already includes flux*BR*acceptance*scatter.")
    print("  Effective SIREN (accept*scatter) per produced phi = sum(w_abs)/N_phi_produced")
    Nphi = data["nimpwt"][data["ptype"]==321].sum()*br_bsm*POT
    print("  N_phi_produced(K_mu) = %.3e ; SIREN accept*scatter = %.3e"%(Nphi, w_abs.sum()/Nphi))
    print("  analytic isotropic accept*scatter = %.3e (x boost-enh ~%.0f gives forward value)"
          %((Ageo*Psc).mean(), 0))

if __name__=="__main__":
    main()
