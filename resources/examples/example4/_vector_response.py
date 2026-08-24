"""
Shared m_V2 RESPONSE-GRID builder for the vector double-mediator MiniBooNE fit.

Two analyses need the same expensive object and must not drift apart:
  mcmc_fit_vector.py  -- Bayesian posterior over (m_V2, P)
  scan_vector_grid.py -- profile-likelihood Delta-chi2 grid over (m_V2, P)

The object: for each m_V2, (a) the calibrated E_vis histogram on the MiniBooNE
bins at the anchor coupling P0, and (b) the normalised in-window cos-theta shape.
Everything else in the fit is analytic, because the coupling product
P = eps1*eps2*g'^2/(4pi) enters the rate ONLY as (P/P0)^2 -- a pure normalisation
that leaves both shapes untouched.

WHY THIS MODULE EXISTS: the E_vis binning moved from an old 19-bin digitisation to
the corrected 11-bin HEPData set, but the cached response npz was guarded by a
check on the MASS axis only. The stale 19-bin cache kept loading and killed
mcmc_fit_vector.py with an opaque broadcast error (11 vs 19) -- the vector fit was
simply unrunnable. `load_or_build` validates EVERY axis and rebuilds on mismatch.
"""
import os, sys, importlib.util
import numpy as np


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def build(mv2_grid, ebins, cos_edges, n_dec, win, cfg_path, pkg_dir, log=print):
    """Simulate the response at every m_V2. Returns (EHIST, CHIST).

    EHIST[a] = calibrated E_vis histogram (len(ebins)-1 bins) at benchmark couplings.
    CHIST[a] = in-window cos-theta shape, normalised to unit sum (0 if no events).
    """
    SA = _load(os.path.join(pkg_dir, "AnalyticRate.py"), "AnalyticRate")
    S = _load(cfg_path, "S_vec")

    # Optional benchmark-mass overrides (GeV). Dutta-Kim Fig.3-LEFT shows TWO vector
    # benchmark sets: (m_chi=8, m_chi'=50, m_V1=17) MeV -- the config default -- and
    # (m_chi=4, m_chi'=100, m_V1=10) MeV. None of these is used to derive anything at
    # config import time (verified: every use is inside build_onshell_models, which
    # analytic_vec_mb calls fresh), so setting the module attribute here is safe --
    # the same reason mutating S.M_V2 per grid point works.
    for env, attr in (("VEC_M_CHI", "M_CHI"), ("VEC_M_CHI_PRIME", "M_CHI_PRIME"),
                      ("VEC_M_V1", "M_V1")):
        if os.environ.get(env):
            setattr(S, attr, float(os.environ[env]))
    build.last_bench = np.array([S.M_CHI, S.M_CHI_PRIME, S.M_V1], float)
    log("[resp] benchmark: m_chi=%.1f  m_chi'=%.1f  m_V1=%.1f MeV"
        % (S.M_CHI * 1e3, S.M_CHI_PRIME * 1e3, S.M_V1 * 1e3))
    if S.M_V1 <= 2 * S.M_CHI:
        log("[resp] WARNING: V1 -> chi chi is CLOSED (m_V1=%.1f <= 2*m_chi=%.1f MeV)"
            % (S.M_V1 * 1e3, 2 * S.M_CHI * 1e3))
    if S.M_CHI_PRIME <= S.M_CHI + S.M_V1:
        log("[resp] WARNING: chi' -> chi V1 is CLOSED (m_chi'=%.1f <= %.1f MeV)"
            % (S.M_CHI_PRIME * 1e3, (S.M_CHI + S.M_V1) * 1e3))

    chans = list(S.CHANNELS)              # vector portal is lepton-universal
    nE, nC = len(ebins) - 1, len(cos_edges) - 1
    EHIST = np.zeros((len(mv2_grid), nE))
    CHIST = np.zeros((len(mv2_grid), nC))

    for a, mv2 in enumerate(mv2_grid):
        S.M_V2 = float(mv2)
        Eh = np.zeros(nE)
        cos_all, w_all = [], []
        for ch in chans:
            pdg, m_M, m_l, lp, nu, g = S.CHANNELS[ch]
            if (m_M - m_l) <= S.M_V1:     # channel closed for this mediator mass
                continue
            E, w, c = SA.analytic_vec_mb(S, ch, n_dec=n_dec, eff_mode="mb",
                                         return_cos=True, meson_fn=SA._mesons_dk2nu)
            E, w, c = np.asarray(E), np.asarray(w), np.asarray(c)
            if E.size == 0:
                continue
            Eh += np.histogram(E, bins=ebins, weights=w)[0]
            m = (E >= win[0]) & (E <= win[1])
            cos_all.append(c[m]); w_all.append(w[m])
        EHIST[a] = Eh
        if cos_all:
            cc = np.concatenate(cos_all); ww = np.concatenate(w_all)
            h, _ = np.histogram(cc, bins=cos_edges, weights=ww)
            CHIST[a] = h / h.sum() if h.sum() > 0 else h
        log("  m_V2=%7.1f MeV  ->  in-window %8.2f ev" % (mv2 * 1e3, Eh.sum()))
    return EHIST, CHIST


def load_or_build(path, mv2_grid, ebins, cos_edges, n_dec, win,
                  cfg_path, pkg_dir, log=print):
    """Reuse `path` only if EVERY axis matches what the caller needs."""
    nE, nC = len(ebins) - 1, len(cos_edges) - 1
    want = ((len(mv2_grid), nE), (len(mv2_grid), nC))
    if os.path.exists(path):
        d = np.load(path)
        EHIST, CHIST = d["EHIST"], d["CHIST"]
        got = (EHIST.shape, CHIST.shape)
        same_masses = ("MV2_GRID" in d and len(d["MV2_GRID"]) == len(mv2_grid)
                       and np.allclose(d["MV2_GRID"], mv2_grid))
        # The PHYSICS the grid was simulated at must match too, not just its shape.
        # Shape-only validation is exactly what let a stale 19-bin cache load and
        # kill mcmc_fit_vector.py; benchmark masses and the cos window are the same
        # kind of silent-mismatch hazard, so check them here rather than relying on
        # callers to encode everything in the filename.
        want_bench = np.array([float(os.environ.get(e, "nan")) for e in
                               ("VEC_M_CHI", "VEC_M_CHI_PRIME", "VEC_M_V1")])
        same_bench = True
        if "BENCH" in d and np.isfinite(want_bench).any():
            got_b = np.asarray(d["BENCH"], float)
            m = np.isfinite(want_bench)
            same_bench = np.allclose(got_b[m], want_bench[m], rtol=1e-9)
        elif "BENCH" not in d and np.isfinite(want_bench).any():
            same_bench = False              # cache predates the tag; cannot verify
        same_win = ("WIN" not in d) or np.allclose(np.asarray(d["WIN"], float), win)
        if not same_bench:
            log("[resp] %s was built at a DIFFERENT benchmark %s (want %s) -> rebuilding"
                % (path, d["BENCH"] if "BENCH" in d else "unrecorded", want_bench))
        if not same_win:
            log("[resp] %s used cos window %s, want %s -> rebuilding"
                % (path, d["WIN"], win))
        if got == want and same_masses and same_bench and same_win:
            log("[resp] reusing %s  EHIST%s" % (path, EHIST.shape))
            return EHIST, CHIST
        log("[resp] %s is STALE (have EHIST%s masses=%s, need EHIST%s matching grid)"
            " -> rebuilding" % (path, got[0], "ok" if same_masses else "differ", want[0]))
    log("[resp] building response over %d m_V2 points (n_dec=%d) ..." % (len(mv2_grid), n_dec))
    EHIST, CHIST = build(mv2_grid, ebins, cos_edges, n_dec, win, cfg_path, pkg_dir, log=log)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez(path, EHIST=EHIST, CHIST=CHIST, MV2_GRID=np.asarray(mv2_grid), N_DEC=n_dec,
             BENCH=getattr(build, "last_bench", np.array([np.nan] * 3)), WIN=np.asarray(win))
    log("[resp] saved -> %s" % path)
    return EHIST, CHIST
