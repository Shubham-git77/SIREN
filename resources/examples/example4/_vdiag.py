"""Quantify the MB(BNB) vector anchor-denominator scatter (the vector warning)."""
import os, time
import numpy as np
os.environ.setdefault("DK2NU_FILE", "/home/shubham/nubeamHighSample.dk2nu.root")
from siren import _util
SA = _util.load_module("sa", "sbnd_analytic.py")
S = _util.load_module("MBv", "VectorPortal_MiniBooNE_fullchain.py")
WIN = (0.140, 0.300)


def denom(nd, seed):
    tot = 0.0; nhit = 0
    for nm in S.CHANNELS:
        E, w = SA.analytic_vec_mb(S, nm, n_dec=nd, seed=seed, eff_mode="mb",
                                  meson_fn=SA._mesons_dk2nu)
        E = np.asarray(E); w = np.asarray(w)
        m = (E >= WIN[0]) & (E <= WIN[1])
        tot += w[m].sum(); nhit += int(m.sum())
    return tot, nhit


print("MB(BNB) VECTOR in-window anchor DENOMINATOR -- seed-scatter quantifies the vector warning:",
      flush=True)
for nd in (150, 600):
    t0 = time.time(); vals = []; hits = 0
    for s in (7, 17, 27):
        d, h = denom(nd, s); vals.append(d); hits = h
    print("  n_dec=%4d : mean %.2f  range [%.2f,%.2f]  seed-scatter %.0f%%  (~%d MC hits, %.0fs)"
          % (nd, np.mean(vals), min(vals), max(vals),
             100 * np.std(vals) / max(np.mean(vals), 1e-9), hits, time.time() - t0),
          flush=True)
print("=== VDIAG DONE ===", flush=True)
