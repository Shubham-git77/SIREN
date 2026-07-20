"""Deterministic physics-regression check for the analytic engine.

Fixed synthetic kaon input + fixed seed -> exact-reproducible weights. Run this
BEFORE and AFTER each refactor stage; the numbers MUST match to machine precision
(same seed, same math). It auto-finds the engine whether it lives in example4
(sbnd_analytic.py) or the package (DarkNewsTables/AnalyticRate.py), so the SAME
script verifies both sides of the move.
"""
import os
import numpy as np
from siren import _util

HERE = os.path.dirname(os.path.abspath(__file__))
_PROC = os.path.join(_util.resource_package_dir(), "processes", "DarkNewsTables")


def load_engine():
    p = os.path.join(_PROC, "AnalyticRate.py")
    if os.path.exists(p):
        return _util.load_module("AnalyticRate", p), "package/AnalyticRate.py"
    return (_util.load_module("sbnd_analytic", os.path.join(HERE, "sbnd_analytic.py")),
            "example4/sbnd_analytic.py")


def fixed_kaons(S, pdg):
    """Deterministic 300-kaon sample (2 GeV, forward, upstream) in BNB frame."""
    n = 300
    E = np.full(n, 2.0)
    m_K = 0.49368
    pmag = np.full(n, np.sqrt(2.0 ** 2 - m_K ** 2))
    d = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
    v = np.tile(np.array([0.7, 0.0, 50.0]), (n, 1))
    w = np.ones(n)
    return E, pmag, d, v, w


def main():
    SA, src = load_engine()
    det = np.array([0.7378, 0.0, 110.0])   # SBND center, BNB frame
    print("ENGINE:", src)

    Ssp = _util.load_module("SBNDp", os.path.join(HERE, "PseudoscalarPortal_SBND_multichannel.py"))
    E, w = SA.analytic_sp(Ssp, "K_mu", n_dec=50, det=det, pot=6.6e20,
                          eff_mode="raw", meson_fn=fixed_kaons)
    ev = (E * w).sum() / w.sum() if w.sum() > 0 else 0.0
    print("PSEUDO K_mu : sum(w)=%.10e  n=%d  <Evis>=%.8f" % (np.sum(w), len(w), ev))

    Sv = _util.load_module("SBNDv", os.path.join(HERE, "VectorPortal_SBND_fullchain.py"))
    E2, w2 = SA.analytic_vec(Sv, "K_mu", n_dec=50, det=det, pot=6.6e20,
                             eff_mode="raw", meson_fn=fixed_kaons)
    ev2 = (E2 * w2).sum() / w2.sum() if w2.sum() > 0 else 0.0
    print("VECTOR K_mu : sum(w)=%.10e  n=%d  <Evis>=%.8f" % (np.sum(w2), len(w2), ev2))
    print("VERIFY DONE")


if __name__ == "__main__":
    main()
