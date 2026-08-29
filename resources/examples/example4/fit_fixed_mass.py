"""
FIXED-MASS coupling fit for the vector double-mediator MiniBooNE analysis.

Question this answers: if we FREEZE m_V2 at the paper's Table I benchmark
(200 MeV) instead of fitting it, does the coupling product come out at the
paper's value P = 1.3e-7?

Why this is a clean test.  The coupling enters the predicted spectrum ONLY as
an overall factor A = (P/P0)^2 -- it changes no shape.  So at fixed mass

    chi2_E(A) = sum_i (d_i - A*s_i)^2 / sigma_i^2

is an EXACT quadratic in A, minimised at

    A_hat = sum(d*s/sig^2) / sum(s*s/sig^2),      sigma_A = 1/sqrt(sum(s*s/sig^2))

and the cos-theta term is independent of A, so it only shifts chi2 by a
constant and cannot move the coupling.  An MCMC over the coupling therefore has
a closed-form answer; we run one anyway (flat prior in log10 P, the same prior
mcmc_fit_vector.py uses) to confirm the sampler reproduces it and to get the
Bayesian credible interval, which differs slightly from the Delta-chi2 interval
because of the log Jacobian.

The anchor P0 is the paper's own Table I product, so the comparison is direct:
    A_hat = 1        <=>  we need exactly the paper's coupling
    sqrt(A_hat)      =    P_hat / P_paper
    A_hat            =    the RATE ratio (rate ~ P^2)

Env: MASSES (MeV, comma-separated; default the paper benchmark plus context)
     ERR_MODE=stat|quad   NDEC   N_MCMC
Run: DK2NU_FILE=/home/shubham/nubeam12M.dk2nu.root \
     /home/shubham/siren_venv/bin/python fit_fixed_mass.py
Out: output/fixed_mass_fit_<ERR_MODE>.npz  + a table on stdout
"""
import os, sys, importlib.util
import numpy as np

os.environ.setdefault("DK2NU_FILE", "/home/shubham/nubeam12M.dk2nu.root")
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
PKG = os.environ.get("SIREN_DNT_DIR",
                     os.path.join(HERE, "..", "..", "processes", "DarkNewsTables"))
OUT = os.environ.get("MCMC_OUT", "output"); os.makedirs(OUT, exist_ok=True)

import _vector_response as VR
import miniboone_data as MB

CFG      = "VectorPortal_MiniBooNE_fullchain.py"
P0       = 1.3e-7                      # paper double-mediator Table I product = the anchor
WIN      = (0.14, 0.30)
NDEC     = int(os.environ.get("NDEC", "100"))
ERR_MODE = os.environ.get("ERR_MODE", "stat")
N_MCMC   = int(os.environ.get("N_MCMC", "60000"))
MASSES   = [float(x) for x in os.environ.get(
    "MASSES", "31.4,60,100,150,200,300,500").split(",")]        # MeV

EXCESS = MB.EXCESS
SIG    = MB.errors(ERR_MODE)
INV2   = 1.0 / SIG ** 2
EBINS  = MB.EBINS
NCB    = 15
COS_EDGES = np.linspace(-1, 1, NCB + 1)

def log(m):
    print(m, flush=True)

log("[fixed] ERR_MODE=%s  NDEC=%d  masses=%s MeV" % (ERR_MODE, NDEC, MASSES))
log("[fixed] anchor P0 = %.3g (paper Table I double-mediator product)" % P0)

# ---- response at exactly these masses (cache keyed by the mass list) ----------
tag = "_".join("%g" % m for m in MASSES)
RESP = os.path.join(OUT, "fixedmass_response_d%d_%s.npz" % (NDEC, tag))
grid = np.array(MASSES) / 1e3          # GeV
EHIST, CHIST = VR.load_or_build(RESP, grid, EBINS, COS_EDGES, NDEC, WIN,
                                CFG, PKG, log=log)

# ---- closed-form coupling fit at each fixed mass ------------------------------
def mcmc_logP(s, nsteps, seed=3):
    """1-D Metropolis over lg = log10 P, flat prior on [-9.5, -5.0]."""
    rng = np.random.default_rng(seed)
    lo, hi = -9.5, -5.0
    def lnp(lg):
        if not (lo <= lg <= hi):
            return -np.inf
        A = (10 ** lg / P0) ** 2
        return -0.5 * np.sum((EXCESS - A * s) ** 2 * INV2)
    lg = -6.9; cur = lnp(lg); out = np.empty(nsteps); acc = 0
    for i in range(nsteps):
        prop = lg + 0.12 * rng.standard_normal()
        lpp = lnp(prop)
        if np.log(rng.random()) < lpp - cur:
            lg, cur = prop, lpp; acc += 1
        out[i] = lg
    return out[nsteps // 3:], acc / nsteps

rows = []
for a, m in enumerate(MASSES):
    s = EHIST[a]                                  # predicted excess at P = P0
    Sss = np.sum(s * s * INV2); Sds = np.sum(EXCESS * s * INV2)
    A_hat = Sds / Sss
    sA    = 1.0 / np.sqrt(Sss)                    # Delta-chi2 = 1
    chi2_min  = np.sum((EXCESS - A_hat * s) ** 2 * INV2)
    chi2_null = np.sum(EXCESS ** 2 * INV2)        # no signal
    chi2_pap  = np.sum((EXCESS - 1.0 * s) ** 2 * INV2)   # A=1 <=> P = paper value
    dof = len(EXCESS) - 1
    A_lo, A_hi = max(A_hat - sA, 0.0), A_hat + sA
    P_hat = P0 * np.sqrt(max(A_hat, 0.0))
    P_lo, P_hi = P0 * np.sqrt(A_lo), P0 * np.sqrt(A_hi)
    ch, acc = mcmc_logP(s, N_MCMC)
    q = np.percentile(ch, [16, 50, 84])
    rows.append(dict(m=m, s_tot=s.sum(), A_hat=A_hat, sA=sA, P_hat=P_hat,
                     P_lo=P_lo, P_hi=P_hi, chi2_min=chi2_min, dof=dof,
                     chi2_pap=chi2_pap, chi2_null=chi2_null,
                     mcmc=q, acc=acc))

log("")
log("=" * 108)
log("FIXED-MASS COUPLING FIT   (ERR_MODE=%s, %d bins, 1 free parameter -> dof=%d)"
    % (ERR_MODE, len(EXCESS), len(EXCESS) - 1))
log("=" * 108)
log("%8s %10s %10s %11s %22s %10s %10s %9s" %
    ("m_V2", "pred@P0", "A_hat", "P_hat/P_pap", "P_hat [68% CL]", "chi2/dof",
     "chi2(pap)", "Dchi2_pap"))
log("%8s %10s %10s %11s %22s %10s %10s %9s" %
    ("[MeV]", "[events]", "rate rat", "", "", "", "/dof", ""))
log("-" * 108)
for r in rows:
    log("%8.1f %10.1f %10.3g %11.3f %10.3g [%.3g, %.3g] %10.2f %10.2f %9.1f" %
        (r["m"], r["s_tot"], r["A_hat"], np.sqrt(max(r["A_hat"], 0)), r["P_hat"],
         r["P_lo"], r["P_hi"], r["chi2_min"] / r["dof"],
         r["chi2_pap"] / r["dof"], r["chi2_pap"] - r["chi2_min"]))
log("-" * 108)
log("MCMC over log10 P at each fixed mass (flat prior, %d steps, 1/3 burn):" % N_MCMC)
for r in rows:
    log("  m_V2=%7.1f MeV : log10 P = %.3f +%.3f -%.3f  (P = %.3g)   closed form %.3f   acc=%.2f"
        % (r["m"], r["mcmc"][1], r["mcmc"][2] - r["mcmc"][1],
           r["mcmc"][1] - r["mcmc"][0], 10 ** r["mcmc"][1],
           np.log10(r["P_hat"]) if r["P_hat"] > 0 else float("nan"), r["acc"]))
log("")
log("paper Table I : m_V2 = 200 MeV, P = 1.3e-7 (log10 = %.3f), chi2/dof = 2.2 (2.6 stat-only)"
    % np.log10(1.3e-7))

np.savez(os.path.join(OUT, "fixed_mass_fit_%s.npz" % ERR_MODE),
         masses=np.array(MASSES), EHIST=EHIST,
         A_hat=np.array([r["A_hat"] for r in rows]),
         P_hat=np.array([r["P_hat"] for r in rows]),
         P_lo=np.array([r["P_lo"] for r in rows]),
         P_hi=np.array([r["P_hi"] for r in rows]),
         chi2_min=np.array([r["chi2_min"] for r in rows]),
         chi2_pap=np.array([r["chi2_pap"] for r in rows]),
         mcmc_q=np.array([r["mcmc"] for r in rows]), P0=P0, err_mode=ERR_MODE)
log("[fixed] wrote %s/fixed_mass_fit_%s.npz" % (OUT, ERR_MODE))
