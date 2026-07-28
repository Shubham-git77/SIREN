#!/usr/bin/env python
"""
MiniBooNE detector simulation with a PRIMARY interaction (neutrino upscattering
to a heavy neutrino N4) and a SECONDARY interaction (N4 -> nu gamma decay),
instrumented to record the TIMING of every interaction vertex.

This is the dipole-portal heavy-neutrino chain (SIREN example2), run at MiniBooNE
via SIREN_Controller.  It is structurally the same as our meson-portal chain
(production -> propagation -> upscatter/decay), but uses the DarkNews models that
ship with the base SIREN build, so it runs against the timing branch.

Chain per event:
   nu_mu (BNB flux)  --upscatter-->  N4  --propagates-->  N4 -> nu + gamma
                     [interaction 0]        [interaction 1, a distance downstream]

SaveEvents writes vertex_time (time at each interaction vertex), secondary_times
(production time of each secondary), and primary_initial_time.  The companion
verify step checks the timing chain:
   t(decay vertex) == t(upscatter vertex) + |x_decay - x_upscatter| / (beta_N4 c)

Run (from this directory):
   /home/shubham/siren_ubaid_venv/bin/python timing_miniboone_chain.py --n 2000
"""
import argparse
import os
import numpy as np

import siren
from siren.SIREN_Controller import SIREN_Controller


def run_sim(n_events, outstem):
    model_kwargs = {
        "m4": 0.47, "mu_tr_mu4": 2.50e-6, "UD4": 0, "Umu4": 0,
        "epsilon": 0.0, "gD": 0.0, "decay_product": "photon",
        "noHC": True, "HNLtype": "dirac",
    }
    # MiniBooNE is not a stand-alone detector model in this build; load it the
    # way our SIREN_interface example4 scripts do -- the SBN GDML composite that
    # includes the MiniBooNE oil-sphere enclosure -- and hand it to the Controller.
    detector_model = siren.utilities.load_detector("SBN", detector="MiniBooNE")
    controller = SIREN_Controller(n_events, detector_model=detector_model)
    primary_type = siren.dataclasses.Particle.ParticleType.NuMu

    # DarkNews cross-section/decay tables are generated into (and cached in) this
    # local dir on first run; reused afterwards.
    # Use the PRE-GENERATED DarkNews tables shipped in resources/processes (fast:
    # loads tables, no vegas integration).  Generating from scratch over all the
    # MiniBooNE oil + dirt nuclei takes hours -- see the pre-built Dipole tables.
    proc_dir = siren.utilities.get_processes_model_path(
        "DarkNewsTables-v%s" % siren.utilities.darknews_version(), must_exist=False)
    table_dir = os.path.join(proc_dir, "Dipole_M%2.2e_mu%2.2e" % (model_kwargs["m4"], model_kwargs["mu_tr_mu4"]))
    controller.InputDarkNewsModel(primary_type, table_dir, **model_kwargs)

    inj, phys = {}, {}
    phys["energy"] = siren.utilities.load_flux("BNB", tag="FHC_numu", physically_normalized=True)
    inj["energy"] = siren.utilities.load_flux("BNB", tag="FHC_numu", min_energy=model_kwargs["m4"],
                                              max_energy=10, physically_normalized=False)
    d = siren.distributions.FixedDirection(siren.math.Vector3D(0, 0, 1.0))
    inj["direction"] = phys["direction"] = d
    decay_range_func = siren.distributions.DecayRangeFunction(
        model_kwargs["m4"], controller.DN_min_decay_width, 3, 541)
    # Restrict the upscatter to the OIL nuclei (C12, H1) -- the actual MiniBooNE
    # detector medium.  This is physical AND fast: the surrounding dirt/berm has
    # exotic nuclei (Be, K, Mg, Cr, Na, Ca) with no pre-built DarkNews tables, so
    # sampling upscatter there would trigger hours of on-the-fly integration.
    PT = siren.dataclasses.Particle.ParticleType
    oil_targets = {t for t in controller.GetDetectorModelTargets()[0]
                   if t in (PT.C12Nucleus, PT.HNucleus)}
    inj["position"] = siren.distributions.RangePositionDistribution(
        6.2, 6.2, decay_range_func, oil_targets)

    controller.SetProcesses(primary_type, inj, phys)
    controller.Initialize()

    def stop(tree, datum, i):
        return datum.record.signature.secondary_types[i] != siren.dataclasses.Particle.ParticleType.N4
    controller.SetInjectorStoppingCondition(stop)

    events = controller.GenerateEvents(fill_tables_at_exit=False)
    os.makedirs(os.path.dirname(outstem) or ".", exist_ok=True)
    controller.SaveEvents(outstem, fill_tables_at_exit=False)
    return outstem


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="events to inject")
    ap.add_argument("--outstem", default="output/MiniBooNE_timing_chain")
    args = ap.parse_args()
    print("[sim] MiniBooNE dipole-portal chain, %d events ..." % args.n)
    stem = run_sim(args.n, args.outstem)
    print("[sim] saved events with stem:", stem)
