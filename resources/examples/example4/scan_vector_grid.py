"""
PROFILE-LIKELIHOOD grid scan of the VECTOR double-mediator MiniBooNE fit
-- the vector analog of scan_brute_grid.py, and the frequentist counterpart of
mcmc_fit_vector.py.  Maps to Dutta-Kim (arXiv:2110.11944) Fig.3 LEFT (Model I).

Model I: meson -> l nu V1 ; V1 -> chi chi ; chi N -> chi' N (via V2) ; chi' -> chi V1_sig ;
V1_sig -> e+e-.  Fixed benchmark masses m_chi=8, m_chi'=50, m_V1=17 MeV.
FREE: m_V2 (x-axis) and the coupling product P = eps1*eps2*g'^2/(4pi) (y-axis).

WHY A PROFILE SCAN AND NOT JUST THE MCMC
----------------------------------------
The coupling enters the rate ONLY as (P/P0)^2, a pure normalisation, so at fixed
m_V2 the chi2 is an exact quadratic in A=(P/P0)^2 and the best-fit coupling has a
CLOSED FORM -- no sampling needed.  That matters here because the vector response
dies off with mass (the benchmark in-window rate falls ~200 events at m_V2=60 MeV
to <0.1 above ~1 GeV).  Where the signal is tiny the data constrain the coupling
only through the product A*s, so the ALLOWED REGION legitimately runs off to large
P -- but a MARGINAL posterior integrates that unbounded volume and reports it as
"preferred mass", which is prior volume, not evidence.  Profiling reports the same
physics without that artefact, and is what Fig.3-style exclusion contours mean.

Likelihood (identical to mcmc_fit_vector.py, so the two are comparable):
  chi2 = chi2_Evis(m_V2, P) + chi2_costheta(m_V2)
  E_vis    : real nu-mode excess, 11-bin HEPData binning (miniboone_data).
  cos-theta: VECTOR template (Fig.2 top-row blue band, pixel-extracted).
             Shape-normalised => INDEPENDENT of P; it constrains mass only.

*** CAVEAT (inherited, unchanged): the vector rate uses the EMPIRICAL CALIB_VECTOR
bridge, anchored to the paper's Table I product P0=1.3e-7.  Mass localisation is
robust; the ABSOLUTE coupling axis carries the known ~few-x vector normalisation
uncertainty. ***

Tune via env:  N_MV2 (mass points, default 26 = reuses the MCMC response cache),
NDEC, N_P (coupling points), ERR_MODE=stat|quad, COS_WEIGHT=off|on|<float>.

Run:
  DK2NU_FILE=/home/shubham/nubeam12M.dk2nu.root \
      /home/shubham/siren_venv/bin/python scan_vector_grid.py
Outputs: output/vector_chi2grid.npz + output/vector_region.png
"""
import os, sys, json, time
from datetime import datetime
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

os.environ.setdefault("DK2NU_FILE", "/home/shubham/nubeam12M.dk2nu.root")
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
PKG = os.environ.get("SIREN_DNT_DIR",
                     os.path.join(HERE, "..", "..", "processes", "DarkNewsTables"))
OUT = os.environ.get("SCAN_OUT", "output")
os.makedirs(OUT, exist_ok=True)


def log(msg):
    line = "[%s] %s" % (datetime.now().strftime("%H:%M:%S"), msg)
    print(line, flush=True)
    with open(os.path.join(OUT, "scan_vector_grid.log"), "a") as f:
        f.write(line + "\n")


# ---------------- data + templates (same modules as the MCMC) ----------------
import miniboone_data as MB
import _vector_response as VR

ERR_MODE = os.environ.get("ERR_MODE", "stat")
EXCESS, EBINS = MB.EXCESS, MB.EBINS
SIG = MB.errors(ERR_MODE)
INV2 = 1.0 / SIG ** 2

COS_WEIGHT = os.environ.get("COS_WEIGHT", "1")
COS_OFF = (COS_WEIGHT == "off")
_cos_scale = 1.0 if (COS_OFF or COS_WEIGHT == "on") else float(COS_WEIGHT)
ct = json.load(open("cos_template_vector_nu.json"))
COS_TGT = np.array(ct["shape"]); COS_TGT /= COS_TGT.sum()
NCB = len(COS_TGT); COS_EDGES = np.linspace(-1, 1, NCB + 1)
COS_SIG = (np.sqrt(COS_TGT * (1 - COS_TGT) / 320.0) + 0.01) * _cos_scale

WIN = (0.14, 0.30)
P0 = 1.3e-7                                  # paper Table I double-mediator anchor
CFG = "VectorPortal_MiniBooNE_fullchain.py"

N_MV2 = int(os.environ.get("N_MV2", "26"))
NDEC = int(os.environ.get("NDEC", "100"))
N_P = int(os.environ.get("N_P", "241"))
# Mass range in GeV. Default 0.060-2.0 = the paper's Fig.3-left x-range. The
# 26-point default fit ran into the LOWER edge (best fit pinned at 60 MeV), so the
# floor is configurable: m_V2 is a t-channel exchange mass with no on-shell
# threshold, and nothing in the model forbids going below the paper's plot range.
MV2_LO = float(os.environ.get("MV2_LO", "0.060"))
MV2_HI = float(os.environ.get("MV2_HI", "2.0"))
MV2_GRID = np.geomspace(MV2_LO, MV2_HI, N_MV2)
LGP_GRID = np.linspace(-9.5, -5.0, N_P)      # same prior span as the MCMC

# Reuse the MCMC's cache only when the request is EXACTLY the MCMC's grid;
# otherwise keep a private one keyed by every axis parameter.
# Benchmark-mass overrides (MeV in, GeV out) -- Fig.3-LEFT has two vector sets.
BENCH = {k: os.environ.get(v) for k, v in
         (("m_chi", "VEC_M_CHI"), ("m_chip", "VEC_M_CHI_PRIME"), ("m_V1", "VEC_M_V1"))}
BTAG = ("_b%s" % "-".join("%.0f" % (float(v) * 1e3) for v in BENCH.values())
        if any(BENCH.values()) else "")

_is_mcmc_grid = (N_MV2 == 26 and NDEC == 100 and not BTAG
                 and abs(MV2_LO - 0.060) < 1e-12 and abs(MV2_HI - 2.0) < 1e-12)
RESP = (os.path.join(OUT, "mcmc_vector_response.npz") if _is_mcmc_grid
        else os.path.join(OUT, "vector_response_n%d_d%d_lo%.0f_hi%.0f%s.npz"
                          % (N_MV2, NDEC, MV2_LO * 1e3, MV2_HI * 1e3, BTAG)))

log("VECTOR profile scan: %d m_V2 in [%.0f, %.0f] MeV x %d P   "
    "ERR_MODE=%s COS_WEIGHT=%s NDEC=%d"
    % (N_MV2, MV2_LO * 1e3, MV2_HI * 1e3, N_P, ERR_MODE, COS_WEIGHT, NDEC))
t0 = time.time()
EHIST, CHIST = VR.load_or_build(RESP, MV2_GRID, EBINS, COS_EDGES, NDEC, WIN,
                                CFG, PKG, log=log)
log("response ready in %.1f s" % (time.time() - t0))

# ---------------- chi2 surface ----------------
# A = (P/P0)^2 scales the signal; the cos term is shape-normalised so it is
# P-independent and enters as a per-mass offset.
A_GRID = (10 ** LGP_GRID / P0) ** 2

chi2_cos = np.zeros(N_MV2)
if not COS_OFF:
    for a in range(N_MV2):
        s = CHIST[a].sum()
        cs = CHIST[a] / s if s > 0 else CHIST[a]
        chi2_cos[a] = np.sum((COS_TGT - cs) ** 2 / COS_SIG ** 2)

# PRIOR=physical: profile over the COUPLING SPLIT at each (m_V2, P), subject to
# existing limits, instead of assuming the benchmark split.
#   rate ~ eps1^2 eps2^2 alpha_D = (P/alpha_D)^2 alpha_D = P^2 / alpha_D
# so at fixed P the achievable rate is set by how far alpha_D can move, and
# alpha_D is bounded both directly (perturbativity / paper convention) and
# indirectly through eps1*eps2 = P/alpha_D having to stay inside its own limits:
#   alpha_D >= P / (eps1_max eps2_max)      and      alpha_D <= alpha_D_max
# A P above eps1_max*eps2_max*alpha_D_max is UNREACHABLE at any split -> excluded
# outright. This is the frequentist counterpart of the physical-prior MCMC, and
# unlike a marginal credible region it is immune to the no-signal prior volume
# that made that posterior's contours break up.
PRIOR = os.environ.get("PRIOR", "none")
EPS1_MAX = float(os.environ.get("EPS1_MAX", "4.7e-3"))
_e2spec = os.environ.get("EPS2_MAX", "1.0e-3")          # may be a mass table "m:e,..."
EPS2_MAX = float(_e2spec) if ":" not in _e2spec else None  # None => use _eps2_ceiling()
ALPHAD_MAX = float(os.environ.get("ALPHAD_MAX", "0.5"))
ALPHAD_MIN = float(os.environ.get("ALPHAD_MIN", "1e-5"))

# EPS2_MAX may be given as a single number OR as a mass-dependent table
# "m1:e1,m2:e2,..." (masses in MeV, step-wise: the ceiling for a given m_V2 is the
# entry with the largest mass <= m_V2). V2 mixes kinetically with the photon, so
# it is produced on shell in dark-photon searches whatever its role here, and the
# applicable bound is mass dependent -- BaBar's invisible-A' search gives
# eps < 1e-3 across this range, while NA64 is stronger below ~100 MeV, reaching
# ~1e-4 below 10 MeV. *** Neither is a digitised curve; the default is the
# conservative flat BaBar value and the table exists so a real curve can be
# dropped in. Run both and quote the sensitivity. ***
def _eps2_ceiling(mv2_gev):
    spec = os.environ.get("EPS2_MAX", "1.0e-3")
    if ":" not in spec:
        return np.full_like(mv2_gev, float(spec))
    pts = sorted((float(a), float(b)) for a, b in
                 (kv.split(":") for kv in spec.split(",")))
    out = np.full_like(mv2_gev, pts[0][1])
    for m_mev, e in pts:
        out[mv2_gev * 1e3 >= m_mev] = e
    return out


if PRIOR == "physical":
    import importlib.util as _ilu
    _sp = _ilu.spec_from_file_location("S_ref", CFG)
    _S = _ilu.module_from_spec(_sp); sys.modules["S_ref"] = _S; _sp.loader.exec_module(_S)
    ALPHAD_0 = float(_S.G_D) ** 2 / (4.0 * np.pi)
    P_LIN = 10 ** LGP_GRID
    E2CEIL = _eps2_ceiling(MV2_GRID)                     # (N_MV2,)
    # eps2 ceiling is now per-mass, so the reachable alpha_D window is 2-D
    aD_lo = np.maximum(ALPHAD_MIN,
                       P_LIN[None, :] / (EPS1_MAX * E2CEIL[:, None]))   # (N_MV2, N_P)
    aD_hi = ALPHAD_MAX
    FEASIBLE2D = aD_lo <= aD_hi
    FEASIBLE = FEASIBLE2D.any(axis=0)
    P_MAX_PHYS = EPS1_MAX * float(np.max(E2CEIL)) * ALPHAD_MAX
    log("eps2 ceiling: %s" % ("flat %.2e" % E2CEIL[0] if np.ptp(E2CEIL) == 0 else
        "mass-dependent %.2e..%.2e" % (E2CEIL.min(), E2CEIL.max())))
    # rate scale u = (P/P0)^2 * (alpha_D0/alpha_D): decreasing in alpha_D
    with np.errstate(divide="ignore", invalid="ignore"):
        U_LO = (P_LIN[None, :] / P0) ** 2 * (ALPHAD_0 / aD_hi)       # (1, N_P) bcast
        U_LO = np.broadcast_to(U_LO, (N_MV2, N_P)).copy()
        U_HI = (P_LIN[None, :] / P0) ** 2 * (ALPHAD_0 / aD_lo)       # (N_MV2, N_P)
    log("PHYSICAL profile: eps1<=%.2e eps2<=%s alpha_D in [%.1e, %.2f]"
        % (EPS1_MAX, ("%.2e" % EPS2_MAX) if EPS2_MAX is not None else _e2spec,
           ALPHAD_MIN, ALPHAD_MAX))
    log("  => P is unreachable above %.3e (log10 %.2f); %d/%d grid P values feasible"
        % (P_MAX_PHYS, np.log10(P_MAX_PHYS), FEASIBLE.sum(), N_P))
else:
    FEASIBLE = np.ones(N_P, dtype=bool)

CHI2_E = np.empty((N_MV2, N_P))                           # E_vis term alone
for a in range(N_MV2):
    if PRIOR == "physical":
        den_a = np.sum(EHIST[a] ** 2 * INV2)
        num_a = np.sum(EHIST[a] * EXCESS * INV2)
        u_hat = num_a / den_a if den_a > 0 else 0.0
        u = np.clip(u_hat, U_LO[a], U_HI[a])              # best split within limits
        pred = u[:, None] * EHIST[a]
    else:
        pred = np.outer(A_GRID, EHIST[a])                 # (N_P, nbins)
    CHI2_E[a] = np.sum((EXCESS - pred) ** 2 * INV2, axis=1)
if PRIOR == "physical":
    CHI2_E[~FEASIBLE2D] = np.inf                          # per (mass, P) reachability
else:
    CHI2_E[:, ~FEASIBLE] = np.inf
CHI2 = CHI2_E + chi2_cos[:, None]

# Closed-form best coupling per mass: d/dA sum (EXCESS - A s)^2/sig^2 = 0
num = (EHIST * EXCESS * INV2).sum(1)
den = (EHIST ** 2 * INV2).sum(1)
with np.errstate(divide="ignore", invalid="ignore"):
    A_HAT = np.where(den > 0, num / den, np.nan)
A_HAT = np.where(np.isfinite(A_HAT) & (A_HAT > 0), A_HAT, np.nan)   # A>=0 physical
P_HAT = P0 * np.sqrt(A_HAT)
chi2_hat = np.array([
    np.sum((EXCESS - (A_HAT[a] if np.isfinite(A_HAT[a]) else 0.0) * EHIST[a]) ** 2 * INV2)
    + chi2_cos[a] for a in range(N_MV2)])
if PRIOR == "physical":
    # The analytic A_HAT above ignores the coupling limits, so it would leave the
    # 1-D panel UNCONSTRAINED while the 2-D region next to it is constrained --
    # two different likelihoods in one figure. Take the 1-D profile from the same
    # constrained surface instead (min over feasible P at each mass).
    chi2_hat = np.min(CHI2, axis=1)
    # ...and P_HAT must come from the same surface, or the reported "best profiled
    # P" is the unconstrained solution sitting next to a constrained chi2 (they
    # differed by 1.5x before this fix).
    _jhat = np.argmin(np.where(np.isfinite(CHI2), CHI2, np.inf), axis=1)
    P_HAT = np.where(np.isfinite(chi2_hat), 10 ** LGP_GRID[_jhat], np.nan)

# The no-signal reference must be compared LIKE FOR LIKE. With COS_WEIGHT on,
# the cos term adds a per-mass offset that the signal-free hypothesis has no
# counterpart for (no events => no angular shape), so mixing them made
# "improvement over no-signal" come out negative. Compare on the E_vis term only.
CHI2_NULL = float(np.sum(EXCESS ** 2 * INV2))             # E_vis, no signal
imin = np.unravel_index(np.argmin(CHI2), CHI2.shape)
CHI2_MIN = float(CHI2[imin])
CHI2_E_MIN = float(CHI2_E[imin])
DCHI2 = CHI2 - CHI2_MIN
NDOF = len(EXCESS) - 2

log("chi2_null (no signal)      = %.2f  (%d bins)" % (CHI2_NULL, len(EXCESS)))
log("chi2_min  on grid          = %.2f  at m_V2=%.1f MeV, P=%.3e  (chi2/dof=%.2f)"
    % (CHI2_MIN, MV2_GRID[imin[0]] * 1e3, 10 ** LGP_GRID[imin[1]], CHI2_MIN / NDOF))
log("chi2_Evis at that point     = %.2f  -> improvement over no-signal = %.2f"
    % (CHI2_E_MIN, CHI2_NULL - CHI2_E_MIN))
log("cos-theta offset           = %.3f %s"
    % (chi2_cos[imin[0]],
       "(CONSTANT across mass => zero leverage)"
       if (not COS_OFF and np.ptp(chi2_cos) < 1e-6) else ""))
log("best profiled mass point   : m_V2=%.1f MeV  P_hat=%.3e  chi2=%.2f"
    % (MV2_GRID[np.nanargmin(chi2_hat)] * 1e3,
       P_HAT[np.nanargmin(chi2_hat)], np.nanmin(chi2_hat)))

_ibest = int(np.nanargmin(chi2_hat))
if _ibest in (0, N_MV2 - 1):
    log("WARNING: best fit sits ON THE %s MASS EDGE (%.1f MeV) -- boundary-limited, "
        "widen the range" % ("LOWER" if _ibest == 0 else "UPPER", MV2_GRID[_ibest] * 1e3))

# where does the paper's benchmark sit?
ip = int(np.argmin(np.abs(MV2_GRID - 0.200)))
jp = int(np.argmin(np.abs(LGP_GRID - np.log10(1.3e-7))))
log("paper double-mediator (200 MeV, 1.3e-7): chi2=%.2f  Delta-chi2=%.2f"
    % (CHI2[ip, jp], DCHI2[ip, jp]))

np.savez(os.path.join(OUT, "vector_chi2grid%s.npz"
                      % ("_physprior" if PRIOR == "physical" else "")),
         chi2=CHI2, chi2_E=CHI2_E, dchi2=DCHI2, mv2_grid=MV2_GRID, lgP_grid=LGP_GRID,
         chi2_cos=chi2_cos, chi2_hat=chi2_hat, P_hat=P_HAT, A_hat=A_HAT,
         EHIST=EHIST, CHIST=CHIST, chi2_null=CHI2_NULL, chi2_min=CHI2_MIN,
         mv2_lo=MV2_LO, mv2_hi=MV2_HI, feasible=FEASIBLE, prior=PRIOR,
         P0=P0, err_mode=ERR_MODE, cos_weight=COS_WEIGHT, n_dec=NDEC)
log("wrote %s/vector_chi2grid%s.npz" % (OUT, "_physprior" if PRIOR == "physical" else ""))

# ---------------- figure ----------------
L68, L95 = 2.30, 5.99                        # 2 d.o.f.
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.6),
                               gridspec_kw={"width_ratios": [1.35, 1]})

X, Y = np.meshgrid(MV2_GRID * 1e3, LGP_GRID, indexing="ij")
axL.contourf(X, Y, DCHI2, levels=[0, L68, L95], colors=["#69c", "#bcd"], alpha=0.75)
axL.contour(X, Y, DCHI2, levels=[L68, L95], colors="k", linewidths=[1.4, 0.8])
axL.plot(MV2_GRID[imin[0]] * 1e3, LGP_GRID[imin[1]], "k+", ms=12, mew=2, label="grid best fit")
good = np.isfinite(P_HAT)
axL.plot(MV2_GRID[good] * 1e3, np.log10(P_HAT[good]), "w-", lw=1.6, alpha=0.9)
axL.plot(MV2_GRID[good] * 1e3, np.log10(P_HAT[good]), "k--", lw=1.0,
         label=(r"profiled $\hat{P}(m_{V_2})$"
                + (" (within limits)" if PRIOR == "physical" else "")))
axL.plot(200, np.log10(1.3e-7), "r*", ms=18, label="paper double (200, 1.3e-7)")
axL.plot(17, np.log10(3.6e-9), "m*", ms=14, label="paper single (17, 3.6e-9)")
axL.set_xscale("log")
axL.set_xlabel(r"$m_{V_2}$ [MeV]")
axL.set_ylabel(r"$\log_{10}(\epsilon_1\epsilon_2 g'^2/4\pi)$")
axL.set_title("Vector double-mediator, MiniBooNE $\\nu$-mode\n"
              "profile-likelihood 68%/95% ($\\Delta\\chi^2$=2.30/5.99, 2 d.o.f.)")
axL.legend(fontsize=8, loc="lower right"); axL.grid(alpha=0.3)

axR.plot(MV2_GRID * 1e3, chi2_hat - np.nanmin(chi2_hat), "o-", color="#248", ms=3.5,
         label=r"profiled over $P$")
axR.axhline(1.0, ls=":", c="k", lw=0.9); axR.axhline(3.84, ls="--", c="k", lw=0.9)
axR.text(MV2_GRID[0] * 1e3 * 1.05, 1.05, r"$\Delta\chi^2=1$ (68%)", fontsize=7)
axR.text(MV2_GRID[0] * 1e3 * 1.05, 3.95, r"$\Delta\chi^2=3.84$ (95%)", fontsize=7)
axR.axvline(200, color="r", ls="-.", lw=1.0, alpha=0.7, label="paper 200 MeV")
axR.set_xscale("log"); axR.set_xlabel(r"$m_{V_2}$ [MeV]")
axR.set_ylabel(r"$\Delta\chi^2$ (profiled)")
axR.set_ylim(0, max(12.0, float(np.nanmin([np.nanmax(chi2_hat - np.nanmin(chi2_hat)), 40.0]))))
axR.set_title("1-D mass profile\n(in-window benchmark rate annotated)")
axR.legend(fontsize=8); axR.grid(alpha=0.3)
ax2 = axR.twinx()
ax2.plot(MV2_GRID * 1e3, np.maximum(EHIST.sum(1), 1e-3), color="#c60", lw=1.0, alpha=0.6)
ax2.set_yscale("log"); ax2.set_ylabel("benchmark events (all bins)", color="#c60", fontsize=8)
ax2.tick_params(axis="y", labelcolor="#c60", labelsize=7)

fig.suptitle("scan_vector_grid.py  [ERR_MODE=%s, COS_WEIGHT=%s, n_dec=%d%s]  "
             "$\\chi^2_{null}$=%.1f, $\\chi^2_{min}$=%.1f"
             % (ERR_MODE, COS_WEIGHT, NDEC,
                (", bench " + BTAG.lstrip("_b")) if BTAG else "", CHI2_NULL, CHI2_MIN),
             fontsize=10)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(OUT, "vector_region%s.png"
                        % ("_physprior" if PRIOR == "physical" else "")), dpi=130)
log("wrote %s/vector_region%s.png" % (OUT, "_physprior" if PRIOR == "physical" else ""))
log("SCAN DONE")
