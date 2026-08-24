"""
FULL-PARAMETER vector-portal MCMC for MiniBooNE -- the vector analog of
mcmc_fit.py's 5-parameter scalar fit, producing the same two figures:
a corner plot of the posterior and a (mass, product) 68%/95% credible region.

WHY THIS EXISTS ALONGSIDE mcmc_fit_vector.py
--------------------------------------------
mcmc_fit_vector.py walks only (m_V2, P): it treats the coupling product P as THE
parameter. For the scalar that is legitimate -- the rate there depends on the
couplings solely through (g_mu g_n lambda)^2. **For the vector it is not.**
Measured scaling (VERIFICATION.md, exact to 0.01):

    rate  ~  eps1^2 * eps2^2 * g_D^2        with   P = eps1 * eps2 * alpha_D

so, eliminating eps1*eps2 at fixed P,

    rate  ~  P^2 / alpha_D

-- the rate is INVERSELY proportional to the dark coupling at fixed product. Two
parameter sets quoting the same Table I product differ by 550x in prediction.
Walking (eps1, eps2, alpha_D) separately instead of collapsing them into P makes
that degeneracy an explicit, plottable feature of the posterior rather than a
hidden assumption -- which is the whole point of the §3 finding.

Walked:  log10 eps1, log10 eps2, log10 alpha_D, m_V2 [MeV]
Derived: log10 P = log10(eps1 * eps2 * alpha_D)   (last corner row, like the
         scalar's log10(g_mu g_n lambda))
Fixed:   the benchmark masses (m_chi, m_chi', m_V1), overridable via
         VEC_M_CHI / VEC_M_CHI_PRIME / VEC_M_V1 -- Fig.3-LEFT has two such sets.

The m_V2 response grid is shared with scan_vector_grid.py via _vector_response,
so no re-simulation is needed if a matching grid already exists.

Run:
  DK2NU_FILE=<the VALIDATED flux, not the script default> \
  SIREN_DNT_DIR=<...>/DarkNewsTables \
      python mcmc_fit_vector_full.py
Out: $MCMC_OUT/mcmc_vector_full_{chain.npz,corner.png,mV2_product.png}
Priors: PRIOR=physical (default, limit-based) | flat. See the prior block below.
"""
import os, sys, json
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

os.environ.setdefault("DK2NU_FILE", "/home/shubham/nubeam12M.dk2nu.root")
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
PKG = os.environ.get("SIREN_DNT_DIR",
                     os.path.join(HERE, "..", "..", "processes", "DarkNewsTables"))
OUT = os.environ.get("MCMC_OUT", "output")
os.makedirs(OUT, exist_ok=True)

import miniboone_data as MB
import _vector_response as VR

ERR_MODE = os.environ.get("ERR_MODE", "stat")
EXCESS, EBINS = MB.EXCESS, MB.EBINS
INV2 = 1.0 / MB.errors(ERR_MODE) ** 2

COS_WEIGHT = os.environ.get("COS_WEIGHT", "1")
COS_OFF = (COS_WEIGHT == "off")
_cos_scale = 1.0 if (COS_OFF or COS_WEIGHT == "on") else float(COS_WEIGHT)
ct = json.load(open("cos_template_vector_nu.json"))
COS_TGT = np.array(ct["shape"]); COS_TGT /= COS_TGT.sum()
NCB = len(COS_TGT); COS_EDGES = np.linspace(-1, 1, NCB + 1)
COS_SIG = (np.sqrt(COS_TGT * (1 - COS_TGT) / 320.0) + 0.01) * _cos_scale

WIN = (0.14, 0.30)
CFG = "VectorPortal_MiniBooNE_fullchain.py"
N_MV2 = int(os.environ.get("N_MV2", "38"))
NDEC = int(os.environ.get("NDEC", "100"))
MV2_LO = float(os.environ.get("MV2_LO", "0.010"))
MV2_HI = float(os.environ.get("MV2_HI", "2.0"))
MV2_GRID = np.geomspace(MV2_LO, MV2_HI, N_MV2)

BENCH = {k: os.environ.get(v) for k, v in
         (("m_chi", "VEC_M_CHI"), ("m_chip", "VEC_M_CHI_PRIME"), ("m_V1", "VEC_M_V1"))}
BTAG = ("_b%s" % "-".join("%.0f" % (float(v) * 1e3) for v in BENCH.values())
        if any(BENCH.values()) else "")
RESP = os.path.join(OUT, "vector_response_n%d_d%d_lo%.0f_hi%.0f%s.npz"
                    % (N_MV2, NDEC, MV2_LO * 1e3, MV2_HI * 1e3, BTAG))

# ---- benchmark couplings the response grid was simulated at ----
import importlib.util
_spec = importlib.util.spec_from_file_location("S_ref", CFG)
_S = importlib.util.module_from_spec(_spec); sys.modules["S_ref"] = _S
_spec.loader.exec_module(_S)
EPS1_0, EPS2_0 = float(_S.EPSILON_1), float(_S.EPSILON_2)
ALPHAD_0 = float(_S.G_D) ** 2 / (4.0 * np.pi)
P0 = EPS1_0 * EPS2_0 * ALPHAD_0
print("[full] benchmark couplings: eps1=%.4e eps2=%.4e alpha_D=%.4e -> P0=%.4e"
      % (EPS1_0, EPS2_0, ALPHAD_0, P0), flush=True)

EHIST, CHIST = VR.load_or_build(RESP, MV2_GRID, EBINS, COS_EDGES, NDEC, WIN,
                                CFG, PKG, log=lambda m: print(m, flush=True))


def interp(grid, mv2):
    """Log-log interpolation in mass.

    The response falls as steeply as m_V2^-4, so LINEAR interpolation between
    log-spaced grid points is a chord above a convex curve and over-estimates the
    rate by up to ~4% at bin midpoints -- a systematic bias, always in the same
    direction. Interpolating log(rate) vs log(m) removes it. Zeros are floored
    rather than dropped so an empty high-mass bin stays at ~0 instead of -inf.
    """
    b = np.clip(np.searchsorted(MV2_GRID, mv2) - 1, 0, len(MV2_GRID) - 2)
    lm, lm0, lm1 = np.log(mv2), np.log(MV2_GRID[b]), np.log(MV2_GRID[b + 1])
    f = np.clip((lm - lm0) / (lm1 - lm0), 0, 1)
    FL = 1e-300
    g0 = np.log(np.maximum(grid[b], FL))
    g1 = np.log(np.maximum(grid[b + 1], FL))
    out = np.exp((1 - f) * g0 + f * g1)
    return np.where(out <= FL * 10, 0.0, out)


# ---- priors: theta = (lg eps1, lg eps2, lg alpha_D, m_V2[MeV]) ----
# PRIOR=flat      : wide log-flat box. The couplings then pile up against the
#                   ceiling, because the likelihood constrains only the
#                   combination entering the rate -- the marginals are prior
#                   volume, not measurement. Kept for comparison.
# PRIOR=physical  : ceilings from EXISTING LIMITS (default).
#
# eps1 -- from Dutta-Kim Table II, which lists 90% CL limits on exotic pi/K decays
# together with the BR their DOUBLE-mediator benchmark predicts at
# (eps1, eps2, alpha_D) = (7.0e-5, 1.0e-4, 0.5). V1 is radiated off the charged
# meson leg, so production scales as BR ~ eps1^2 and each row gives a ceiling
# eps1_max = 7.0e-5 * sqrt(limit / predicted):
#
#   channel              limit        predicted(double)   -> eps1_max
#   K -> mu nu V(phi)    3.0e-6       6.8e-10                4.7e-3   <-- binding
#   K -> e nu nu nu      6.0e-5       7.2e-10                2.0e-2
#   pi -> e nu X         5.0e-7       3.4e-13                8.5e-2
#
# The K -> l nu e+e- rows are nominally tighter (eps1 <~ 4.3e-4) but they need
# V1 -> e+e-, so they scale as eps1^2 * BR(V1->ee), and BR(V1->ee) itself falls as
# alpha_D rises (V1 -> chi chi opens up, m_V1=17 > 2*m_chi=16 MeV). Applying them
# with BR(V1->ee) frozen at the benchmark would be wrong across the alpha_D range
# we are walking, so the default uses only the clean eps1^2 rows. Set
# EPS1_MAX=4.3e-4 to impose the visible-decay bound instead.
#
# eps2 -- NOT constrained by Table II: V2 is exchanged in the t-channel, never
# produced in meson decay. The default ceiling is a generic visible dark-photon
# kinetic-mixing bound over 0.2-2 GeV. *** This is an order-of-magnitude envelope,
# NOT a number taken from the paper -- replace it with the specific limit you cite. ***
#
# alpha_D -- 0.5 is the paper's stated convention; perturbativity would allow ~1.
PRIOR = os.environ.get("PRIOR", "physical")
EPS1_MAX = float(os.environ.get("EPS1_MAX", "4.7e-3"))
EPS2_MAX = float(os.environ.get("EPS2_MAX", "1.0e-3"))
ALPHAD_MAX = float(os.environ.get("ALPHAD_MAX", "0.5"))

if PRIOR == "physical":
    PLO = np.array([-8.0, -8.0, -5.0, MV2_LO * 1e3])
    PHI = np.array([np.log10(EPS1_MAX), np.log10(EPS2_MAX),
                    np.log10(ALPHAD_MAX), MV2_HI * 1e3])
    print("[full] PHYSICAL priors: eps1<=%.2e (Table II, K->mu nu V), "
          "eps2<=%.2e (generic dark-photon envelope), alpha_D<=%.2f"
          % (EPS1_MAX, EPS2_MAX, ALPHAD_MAX), flush=True)
else:
    PLO = np.array([-5.0, -5.0, -5.0, MV2_LO * 1e3])
    PHI = np.array([-1.0, -1.0, np.log10(0.5), MV2_HI * 1e3])
    print("[full] FLAT priors (couplings will pile up at the ceiling)", flush=True)
LABELS = [r"$\log_{10}\epsilon_1$", r"$\log_{10}\epsilon_2$",
          r"$\log_{10}\alpha_D$", r"$m_{V_2}$ [MeV]",
          r"$\log_{10}(\epsilon_1\epsilon_2\alpha_D)$"]


# MASS_PRIOR: "log" (default) or "linear". The mass spans 2.3 decades, so a
# flat-LINEAR prior puts 95% of its volume above 100 MeV and only 4.5% below --
# that alone pushes the m_V2 marginal to high mass, on top of the genuine
# mass-coupling degeneracy. Log-flat puts 43% below 100 MeV and matches the
# log-spaced response grid. The profile-likelihood scan is prior-free either way.
MASS_PRIOR = os.environ.get("MASS_PRIOR", "log")
print("[full] mass prior: %s-flat over [%.0f, %.0f] MeV"
      % (MASS_PRIOR, MV2_LO * 1e3, MV2_HI * 1e3), flush=True)


def logprob(th):
    if np.any(th < PLO) or np.any(th > PHI):
        return -np.inf
    lg1, lg2, lgaD, mv2 = th
    # Jacobian for a log-flat mass prior sampled in linear mass: p(m) ~ 1/m
    lp_mass = -np.log(mv2) if MASS_PRIOR == "log" else 0.0
    # rate ~ eps1^2 eps2^2 g_D^2, and g_D^2 = 4 pi alpha_D
    scale = (10 ** lg1 / EPS1_0) ** 2 * (10 ** lg2 / EPS2_0) ** 2 * (10 ** lgaD / ALPHAD_0)
    Ev = interp(EHIST, mv2 / 1e3) * scale
    chi2_E = np.sum((EXCESS - Ev) ** 2 * INV2)
    cs = interp(CHIST, mv2 / 1e3); s = cs.sum(); cs = cs / s if s > 0 else cs
    chi2_c = 0.0 if COS_OFF else np.sum((COS_TGT - cs) ** 2 / COS_SIG ** 2)
    return -0.5 * (chi2_E + chi2_c) + lp_mass


# ---- where to start the walkers -------------------------------------------
# The centre of the prior box sits at eps ~ 1e-5, a rate scale ~1e-13, i.e. deep
# in the flat no-signal plateau. Starting every walker there left 39% of the
# post-burn samples stranded in that plateau after 6000 steps, even though the
# signal mode beats it by Delta-chi2 ~ 116 (a likelihood ratio of e^58) and the
# prior volume favours the plateau by only ~1e2. The reported mode weights were
# therefore an initialisation artefact, not the posterior.
#
# Fix: seed a fraction of the walkers AT the closed-form best fit (the same
# u_hat = argmin chi2 that scan_vector_grid.py profiles analytically) and spread
# the rest over the box, so the likelihood -- not the starting point -- sets the
# relative weight of the two modes.
_den = (EHIST ** 2 * INV2).sum(1)
_num = (EHIST * EXCESS * INV2).sum(1)
with np.errstate(divide="ignore", invalid="ignore"):
    _u = np.where(_den > 0, _num / _den, 0.0)
_u = np.clip(np.nan_to_num(_u), 0.0, None)
_chi2 = np.array([np.sum((EXCESS - _u[a] * EHIST[a]) ** 2 * INV2)
                  for a in range(len(MV2_GRID))])
_abest = int(np.nanargmin(_chi2))
U_BEST = float(_u[_abest]); MV2_BEST = float(MV2_GRID[_abest] * 1e3)
# scale = (eps1/eps1_0)^2 (eps2/eps2_0)^2 (alpha_D/alpha_D_0); take eps1=eps2=eps
# and alpha_D near the top of its allowed range (minimises the eps needed).
_aD = min(10 ** PHI[2], 0.5)
_eps = (U_BEST * EPS1_0 ** 2 * EPS2_0 ** 2 * ALPHAD_0 / _aD) ** 0.25
SEED = np.clip(np.array([np.log10(_eps), np.log10(_eps), np.log10(_aD), MV2_BEST]),
               PLO, PHI)
print("[full] best-fit seed: m_V2=%.1f MeV  eps=%.3e  alpha_D=%.3f  (chi2=%.2f, logprob=%.2f)"
      % (MV2_BEST, 10 ** SEED[0], 10 ** SEED[2], _chi2[_abest], logprob(SEED)), flush=True)


def run_mcmc(nwalkers=int(os.environ.get("MCMC_WALKERS", "40")),
             nsteps=int(os.environ.get("MCMC_STEPS", "6000")), seed=1):
    rng = np.random.default_rng(seed); ndim = 4
    ctr = 0.5 * (PLO + PHI); span = PHI - PLO
    # Default 1.0 = seed every walker at the best fit. Justified by measurement,
    # not taste: with ERR_MODE=stat the signal mode beats the no-signal plateau by
    # Delta-chi2 ~ 116, and the flow is strictly ONE-WAY -- seeding all walkers in
    # the signal mode gives 1.000 signal fraction with ZERO transitions out, while
    # seeding them all in the plateau gives 0.61 with 1647 transitions, all inward.
    # The plateau population is therefore un-burned-in initialisation, not
    # posterior mass, and starting at the MAP removes it.
    # WITH ERR_MODE=quad THIS IS NOT TRUE: the barrier is only Delta-chi2 ~ 17,
    # walkers cross both ways, and the two seedings disagree (0.62 vs 0.20) -- that
    # posterior is genuinely bimodal and needs a much longer chain. Prefer the
    # profile-likelihood region (scan_vector_grid.py PRIOR=physical) there.
    frac = float(os.environ.get("MCMC_SEED_FRAC", "1.0"))
    n_seed = int(round(nwalkers * frac))
    pos = ctr + 0.25 * span * (rng.random((nwalkers, ndim)) - 0.5)   # explorers
    if n_seed:                                                       # seeded at best fit
        jitter = np.zeros((n_seed, ndim))
        jitter[:, :3] = 0.30 * (rng.random((n_seed, 3)) - 0.5)       # +/-0.15 dex
        jitter[:, 3] = MV2_BEST * 0.30 * (rng.random(n_seed) - 0.5)  # +/-15% in mass
        pos[:n_seed] = np.clip(SEED + jitter, PLO, PHI)
    lp = np.array([logprob(p) for p in pos])
    for k in range(nwalkers):
        tries = 0
        while not np.isfinite(lp[k]) and tries < 500:
            pos[k] = ctr + 0.15 * span * (rng.random(ndim) - 0.5)
            lp[k] = logprob(pos[k]); tries += 1
    chain = np.empty((nsteps, nwalkers, ndim)); a = 2.0; acc = 0
    for st in range(nsteps):
        for k in range(nwalkers):
            j = rng.integers(nwalkers)
            while j == k:
                j = rng.integers(nwalkers)
            z = ((a - 1) * rng.random() + 1) ** 2 / a
            prop = pos[j] + z * (pos[k] - pos[j])
            lpp = logprob(prop)
            if np.log(rng.random()) < (ndim - 1) * np.log(z) + lpp - lp[k]:
                pos[k] = prop; lp[k] = lpp; acc += 1
        chain[st] = pos
        if (st + 1) % 1000 == 0:
            print("  step %d/%d acc=%.2f" % (st + 1, nsteps, acc / ((st + 1) * nwalkers)),
                  flush=True)
    return chain


print("[full] running MCMC (4 walked params + 1 derived) ...", flush=True)
chain = run_mcmc(); burn = chain.shape[0] // 3
flat = chain[burn:].reshape(-1, 4)
lgP = flat[:, 0] + flat[:, 1] + flat[:, 2]                 # log10(eps1 eps2 alpha_D)
samples = np.column_stack([flat, lgP])
np.savez(os.path.join(OUT, "mcmc_vector_full_chain.npz"),
         chain=chain, flat=flat, lgP=lgP, labels=np.array(LABELS, dtype=object),
         mv2_grid=MV2_GRID, P0=P0, eps1_0=EPS1_0, eps2_0=EPS2_0, alphaD_0=ALPHAD_0,
         prior=PRIOR, plo=PLO, phi=PHI, mass_prior=MASS_PRIOR)


def corner(s, labels, truths=None):
    n = s.shape[1]; fig, ax = plt.subplots(n, n, figsize=(2.2 * n, 2.2 * n))
    for i in range(n):
        for j in range(n):
            a = ax[i, j]
            if j > i:
                a.axis("off"); continue
            if i == j:
                a.hist(s[:, i], bins=45, color="#3b6", histtype="stepfilled", alpha=0.7)
                q = np.percentile(s[:, i], [16, 50, 84])
                [a.axvline(v, ls="--", c="k", lw=0.7) for v in q]
                a.set_title("%s = %.2f$^{+%.2f}_{-%.2f}$"
                            % (labels[i], q[1], q[2] - q[1], q[1] - q[0]), fontsize=8)
            else:
                a.hist2d(s[:, j], s[:, i], bins=45, cmap="viridis")
                if truths is not None and truths[j] is not None and truths[i] is not None:
                    a.plot(truths[j], truths[i], "*", color="red", ms=12)
            if i == n - 1:
                a.set_xlabel(labels[j], fontsize=8)
            else:
                a.set_xticklabels([])
            if j == 0 and i > 0:
                a.set_ylabel(labels[i], fontsize=8)
            else:
                a.set_yticklabels([])
            a.tick_params(labelsize=6)
    fig.suptitle("MCMC posterior -- vector double mediator "
                 "(MiniBooNE nu E_vis + cos-theta)%s"
                 % ("  [%s priors]" % PRIOR
                    + ("  [bench %s]" % BTAG.lstrip("_b") if BTAG else "")), fontsize=12)
    fig.tight_layout(); return fig


# paper Table I: only the product and m_V2 are quoted; eps1/eps2/alpha_D individually
# are NOT determined by it -- that is exactly the §3 ambiguity, so no truth marker
# is placed on the first three axes.
PAPER_MV2, PAPER_LGP = 200.0, np.log10(1.3e-7)
truths = [None, None, None, PAPER_MV2, PAPER_LGP]
corner(samples, LABELS, truths).savefig(
    os.path.join(OUT, "mcmc_vector_full_corner.png"), dpi=110)
print("wrote %s/mcmc_vector_full_corner.png" % OUT)

# ---------- (m_V2, product) 2D posterior with 68/95 contours ----------
fig, ax = plt.subplots(figsize=(7, 6))
H, xe, ye = np.histogram2d(flat[:, 3], lgP, bins=60)
Hs = np.sort(H.ravel())[::-1]; cum = np.cumsum(Hs) / H.sum()
l68 = Hs[np.searchsorted(cum, 0.68)]; l95 = Hs[np.searchsorted(cum, 0.95)]
X, Y = np.meshgrid(0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:]), indexing="ij")
ax.contourf(X, Y, H, levels=[l95, l68, H.max()], colors=["#bcd", "#69c"], alpha=0.7)
ax.contour(X, Y, H, levels=[l95, l68], colors="k", linewidths=[0.8, 1.4])
ax.plot(PAPER_MV2, PAPER_LGP, "*", color="red", ms=18, label="paper Table I")
ax.set_xscale("log")
# Pin to the response-grid extent so this is comparable with the profile-scan
# figures, rather than auto-scaling to wherever the posterior happens to sit.
ax.set_xlim(MV2_LO * 1e3, MV2_HI * 1e3)
ax.set_xlabel(r"$m_{V_2}$ [MeV]")
ax.set_ylabel(r"$\log_{10}(\epsilon_1\epsilon_2 g'^2/4\pi)$")
_ttl = ("MCMC posterior (vector): 68%/95% credible region\n"
        "full 4-param fit, couplings WALKED not solved  ["
        + PRIOR + " priors]"
        + ("  [bench " + BTAG.lstrip("_b") + "]" if BTAG else ""))
ax.set_title(_ttl)
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "mcmc_vector_full_mV2_product.png"), dpi=120)
print("wrote %s/mcmc_vector_full_mV2_product.png" % OUT)

# convergence diagnostic: how the two modes are populated, and whether walkers mix
_sig = lgP > -10.0
_pw = (chain[burn:, :, 0] + chain[burn:, :, 1] + chain[burn:, :, 2]) > -10.0
_frac = _pw.mean(axis=0)
print("[full] mode split: %.3f of post-burn samples in the SIGNAL mode" % _sig.mean())
print("[full]   walkers always-signal %d / always-plateau %d / mixing %d (of %d)"
      % ((_frac > 0.98).sum(), (_frac < 0.02).sum(),
         ((_frac >= 0.02) & (_frac <= 0.98)).sum(), _pw.shape[1]))
print("[full]   mode transitions post-burn: %d"
      % np.abs(np.diff(_pw.astype(int), axis=0)).sum())

q = np.percentile(samples, [16, 50, 84], axis=0)
print("[full] DONE.")
for i, L in enumerate(LABELS):
    print("   %-34s %.3f +%.3f -%.3f" % (L, q[1, i], q[2, i] - q[1, i], q[1, i] - q[0, i]))
