#!/usr/bin/env python
"""
Authoritative analytic ICARUS meson-portal yields -- all three channels
(scalar, pseudoscalar, vector) -- using the REAL high-statistics BNB dk2nu flux
(nubeam12M.dk2nu.root) as the parent-meson source.

ICARUS = the T600 liquid-argon TPC at 600 m on the BNB axis (the SBN far
detector).  It reuses the SAME validated sigma*N*chord ray-trace engine
(sbnd_analytic) as SBND -- ICARUS is also an argon LArTPC, so the target,
box ray-trace, LArTPC efficiency and cascade physics are identical.  Only three
things change vs SBND, all passed in explicitly so the SBND path is untouched:

    det       DET_ICARUS = (0.0, -0.432, 600.0) m   (box center, beam coords;
              derived from load_detector("SBN","ICARUS") origin + GDML box)
    pot       S.ICARUS_POT
    meson_fn  _mesons_dk2nu  (parents from the dk2nu file, w = nimpwt/pot_tot),
              instead of the synthetic BNBFlux used for SBND.

For each channel it ALSO prints the synthetic-BNBFlux yield (same geometry) as a
flux cross-check, so any dk2nu-vs-BNBFlux normalization factor is caught here.

  DK2NU_FILE  (env)  parent flux file    (default: nubeam12M.dk2nu.root)
  N_DEC       (env)  MC decays per meson (default 200)
  EFF_MODE    (env)  raw | mb | lartpc   (default lartpc, the physical one)
  XCHECK      (env)  1 -> also run the BNBFlux cross-check (slower)

Run:  DK2NU_FILE=/home/shubham/nubeam12M.dk2nu.root \
      /home/shubham/siren_venv/bin/python icarus_analytic.py
"""
import os
import sys
import numpy as np

os.environ.setdefault("DK2NU_FILE", "/home/shubham/nubeam12M.dk2nu.root")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sbnd_analytic as A
import ScalarPortal_ICARUS_multichannel as SC
import PseudoscalarPortal_ICARUS_multichannel as PS
import VectorPortal_ICARUS_fullchain as VE

# ICARUS active-box center in BNB beam coords [m] (see module docstring).
DET_ICARUS = np.array([0.0, -0.432, 600.0])

N_DEC  = int(os.environ.get("N_DEC", "200"))
EFF    = os.environ.get("EFF_MODE", "lartpc")
XCHECK = os.environ.get("XCHECK", "0") == "1"

# For the BNBFlux cross-check the ICARUS modules need a _BNB (they normally read
# the dk2nu file, not BNBFlux). Borrow the one the SBND scalar script loads.
if XCHECK:
    import ScalarPortal_SBND_multichannel as _SBND
    for _m in (SC, PS, VE):
        _m._BNB = _SBND._BNB


def run(S, label, vector=False, meson_fn=None, tag="dk2nu"):
    fn = A.analytic_vec if vector else A.analytic_sp
    pot = S.ICARUS_POT
    # muon-only (g_e=0) for the muon-coupled scalar/pseudoscalar (paper model);
    # the vector couples lepton-universally (kinetic mixing) -> keep all channels.
    chans = list(S.CHANNELS)
    if not vector:
        chans = [nm for nm in chans if "mu" in nm]
    print("=" * 70)
    print("  ICARUS  %-14s  (POT=%.2e, eff=%s, flux=%s, chans=%s)"
          % (label, pot, EFF, tag, ",".join(chans)))
    print("=" * 70)
    tot = 0.0
    res = {}
    for nm in chans:
        E, w = fn(S, nm, n_dec=N_DEC, eff_mode=EFF,
                  det=DET_ICARUS, pot=pot, meson_fn=meson_fn)
        s = float(w.sum()); tot += s; res[nm] = s
        ev = (E * w).sum() / s if s > 0 else 0.0
        print("   %-7s : %.4e ev    <E_vis>=%3.0f MeV    (hits=%d)"
              % (nm, s, ev * 1e3, len(E)))
    print("   " + "-" * 50)
    print("   %-7s : %.4e ev" % ("TOTAL", tot))
    return tot, res


def main():
    print("\nDK2NU_FILE = %s" % os.environ["DK2NU_FILE"])
    print("Building ICARUS analytic yields (all 3 channels)...\n")
    totals = {}
    totals["scalar"] = run(SC, "SCALAR",       vector=False, meson_fn=A._mesons_dk2nu)[0]
    totals["pseudo"] = run(PS, "PSEUDOSCALAR", vector=False, meson_fn=A._mesons_dk2nu)[0]
    totals["vector"] = run(VE, "VECTOR",       vector=True,  meson_fn=A._mesons_dk2nu)[0]

    if XCHECK:
        print("\n### BNBFlux cross-check (same geometry, synthetic flux) ###")
        b = {}
        b["scalar"] = run(SC, "SCALAR",       vector=False, meson_fn=None, tag="BNBFlux")[0]
        b["pseudo"] = run(PS, "PSEUDOSCALAR", vector=False, meson_fn=None, tag="BNBFlux")[0]
        b["vector"] = run(VE, "VECTOR",       vector=True,  meson_fn=None, tag="BNBFlux")[0]
        print("\n  flux-normalization cross-check  dk2nu / BNBFlux :")
        for k in ("scalar", "pseudo", "vector"):
            r = totals[k] / b[k] if b[k] else float("nan")
            print("    %-7s : dk2nu=%.3e  BNBFlux=%.3e  ratio=%.3f" % (k, totals[k], b[k], r))

    print("\n" + "=" * 70)
    print("  ICARUS TOTALS (dk2nu, eff=%s, POT=%.1e):" % (EFF, SC.ICARUS_POT))
    for k in ("scalar", "pseudo", "vector"):
        print("    %-7s : %.4e events" % (k, totals[k]))
    print("=" * 70)


if __name__ == "__main__":
    main()
