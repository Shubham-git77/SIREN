"""
BRUTE-FORCE dense re-simulating grid scan of the (pseudo)scalar Dark-Primakoff
MiniBooNE fit -- the literal "slow" version (NO tabulation, NO interpolation, NO
analytic coupling solve).

Contrast with the fast paths:
  scan_fit_product.py / scan_credible_region.py : cache the MC hits ONCE, solve the
      coupling analytically, only re-evaluate a cheap sigma per m_Zp.
  mcmc_fit.py : tabulate a response grid ONCE, MCMC interpolates it.
THIS script instead RE-RUNS THE FULL ENGINE at every mass grid point:
  for each (m_phi, m_Zp): fresh analytic_sp_mb() -> fresh meson-decay sampling
  (n_dec loop) + Primakoff scatter + ray-trace + cross-section.  Nothing is cached
  across points except the one-time dk2nu flux READ (re-reading a 10 GB file per
  point is absurd and no analysis does it; set FRESH_FLUX_PER_POINT=1 to force it).

Parameters actually re-simulated: (m_phi, m_Zp).  The couplings enter the rate ONLY
as the product P=g_mu*g_n*lambda via P^2 (the MC is coupling-independent), so P is
swept on a cheap grid at each mass point -- gridding g_mu,g_n,lambda separately would
be pure degeneracy.  Result: a full chi2 cube over (m_phi, m_Zp, P).

Likelihood = E_vis (real nu-mode data, 11 HEPData bins) + cos-theta (paper template),
same as mcmc_fit.py, so the brute-force credible region is directly comparable.

KNOWN GAPS vs the paper's own fit (arXiv:2110.11944, "Fits and discussions", p.3),
which fits FOUR distributions -- E_vis and cos-theta in BOTH nu and nubar modes
(its Fig. 2 is 4 columns x 2 rows: vector portal on top, scalar below):
  1. We fit nu-mode only. No RHC-BNB flux file exists, so nubar cannot currently
     be simulated at all. Expected to cost statistics rather than bias the angle:
     the signal's cos-theta is C-symmetric (horn focuses the same |p| band for
     either charge, and M -> l nu phi is C-symmetric).
  2. The paper uses statistical AND systematic uncertainties "added by quadrature"
     (approximating background systematics from Table I of its ref [3]). DATA_ERR
     here is STAT-ONLY, which is very likely why our 68% band is implausibly tight
     (~+/-10% in coupling) for a systematics-dominated measurement.
  3. Our cos-theta term is not data: it is a pixel-extracted shape of the paper's
     own Fig. 2 signal band with an invented error (COS_SIG below, a made-up
     320-event normalization). Measured leverage: it contributes 145 chi2 units of
     spread across the grid against 61 from the real E_vis data, i.e. it OUTVOTES
     the data and is what pulls the fitted m_Zp up to ~97 MeV. With it removed the
     E_vis-only fit lands on the paper's Table I scalar benchmark (2.25e-8 vs
     2.2e-8 at m_Zp=49 MeV; paper point at Delta-chi2=2.96, inside 95%).

*** DEFAULT GRID IS DENSE -> HOURS. Estimate is printed before the scan starts. ***
Tune via env:  N_MPHI, N_MZP, N_PROD, NDEC, PORTAL, BRUTE_TEST=1 (tiny smoke grid).

Run (when ready):
  PORTAL=scalar DK2NU_FILE=/home/shubham/nubeam12M.dk2nu.root \
      /home/shubham/siren_venv/bin/python scan_brute_grid.py
Outputs (checkpointed): output/brute_<portal>_chi2cube.npz + _region.png
"""
import os, sys, json, time, traceback, importlib.util, numpy as np
from datetime import datetime

def log(msg):
    """Timestamped, immediately-flushed print -> stdout AND a log file, so
    progress survives even if the terminal/tmux pane is lost or a broken
    pipe kills the SSH session before stdout gets flushed."""
    line = "[%s] %s" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    os.makedirs("output", exist_ok=True)
    with open(os.path.join("output", "scan_brute_grid.log"), "a") as f:
        f.write(line + "\n")

def heartbeat(state):
    """Overwrite a small JSON file every point so you can check *right now*
    whether the process is alive and how far it's gotten, without waiting
    for the next print or checkpoint. Compare its mtime to `date` to see
    if it's stale (stuck) or fresh (actively running)."""
    os.makedirs("output", exist_ok=True)
    tmp = os.path.join("output", "heartbeat.json.tmp")
    with open(tmp, "w") as f:
        json.dump(state, f)
    os.replace(tmp, os.path.join("output", "heartbeat.json"))

PORTAL = os.environ.get("PORTAL", "scalar")
os.environ.setdefault("DK2NU_FILE", "/home/shubham/nubeam12M.dk2nu.root")
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
PKG  = os.environ.get(
    "SIREN_DNT_DIR",
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "..", "processes", "DarkNewsTables",
    ),
)

def load(path, name):
    import sys
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m

SA = load(os.path.join(PKG, "AnalyticRate.py"), "AnalyticRate")
DP = load(os.path.join(PKG, "DarkPrimakoff.py"), "DPmod")
CFG = {"scalar":"ScalarPortal_MiniBooNE_multichannel.py",
       "pseudo":"PseudoscalarPortal_MiniBooNE_multichannel.py"}[PORTAL]

# ---------- data + cos template ----------
# CORRECTED 2026-08-10: replaced with the official public MiniBooNE nu-mode
# release (HEPData ins1804293, Table t6 "NuE data and background", tied to
# arXiv:2006.16883 -- the exact reference [3] the Dutta-Kim paper cites for
# its nu-mode data). The previous 19x50MeV-uniform-bin arrays did not match
# any official release binning (11 variable-width bins, 200-3000 MeV with an
# overflow tail) and were likely a mis-digitization.
DATA_E = np.array([250.0,337.5,425.0,512.5,612.5,737.5,875.0,1025.0,1175.0,1375.0,2250.0])
DATA_N = np.array([732,426,444,248,281,236,201,164,138,144,188], float)
# stat errors symmetrized (avg of HEPData stat+/stat-; asymmetry is small)
DATA_ERR = np.array([27.83,21.33,21.83,16.33,17.33,15.83,14.83,13.33,12.33,12.82,14.33])
BKG    = np.array([527.164624,315.423689,349.644825,186.21197,261.441799,
                    195.534193,203.008745,165.664396,118.581365,143.989367,201.450357])
EBINS  = np.array([200,300,375,475,550,675,800,950,1100,1250,1500,3000]) / 1e3  # GeV, 11 bins
EXCESS = DATA_N-BKG; INV2 = 1.0/DATA_ERR**2
ct = json.load(open("cos_template_nu.json"))
COS_TGT = np.array(ct["shape"]); COS_TGT/=COS_TGT.sum()
NCB=len(COS_TGT); COS_EDGES=np.linspace(-1,1,NCB+1)
COS_SIG = np.sqrt(COS_TGT*(1-COS_TGT)/320.0)+0.01
WIN=(0.14,0.30)

# --- real-data cos(theta) channel (miniboone_costheta.py, FIG. 8 of 2006.16883) ---
# The legacy cosshape above is a SHAPE in 15 bins over WIN=(0.14,0.30) GeV, which
# matches neither the binning nor the event selection of the real angular data.
# These store the prediction on the DATA's own footprint: 20 uniform bins over
# the figure's 200 < E < 1250 MeV range, as ABSOLUTE counts at P_REF (so they
# scale by (P/P_REF)^2 exactly like Ehist, and the angular term constrains the
# coupling instead of only the shape).
COS20_EDGES = np.linspace(-1.0, 1.0, 21)
COS20_WIN = (0.200, 1.250)

# ---------- grid (DENSE by default) ----------
TEST = os.environ.get("BRUTE_TEST","0")=="1"
N_MPHI = int(os.environ.get("N_MPHI", "3" if TEST else "15"))
N_MZP  = int(os.environ.get("N_MZP",  "4" if TEST else "30"))
N_PROD = int(os.environ.get("N_PROD", "20" if TEST else "60"))
NDEC   = int(os.environ.get("NDEC",   "100" if TEST else "400"))
MPHI_GRID = np.linspace(1e-3, 100e-3, N_MPHI)          # GeV, scalar/pseudo mass
MZP_GRID  = np.geomspace(30e-3, 200e-3, N_MZP)         # GeV, Z' mass
PROD_GRID = np.geomspace(3e-9, 1.2e-6, N_PROD)         # MeV^-1, coupling product
FRESH_FLUX_PER_POINT = os.environ.get("FRESH_FLUX_PER_POINT","0")=="1"
CKPT_EVERY = int(os.environ.get("CKPT_EVERY","10"))    # save partial cube every N m_phi rows

S = load(CFG, "S_brute")
P_REF = S.G_MU_PROD * S.G_N * (S.LAMBDA*1e-3)          # config product [MeV^-1]
MUON_CHANNELS = [c for c in S.CHANNELS if "mu" in c]
# Photon-smearing RNG, re-seeded per grid point inside forward_point() and
# exposed as RNG_SEED so a run can be repeated under a different realization.
#
# MEASURED 2026-08-16 on an identical 12-point m_Zp slice at NDEC=400. Recording
# what was tested, including two refuted hypotheses, so nobody re-runs them:
#   * per-run MC noise in the chi2 surface is sigma ~ 0.86 units, from
#     RMS(seed1 - seed0) = 1.213 over two RNG_SEED values. That is ~40% of the
#     Delta-chi2 = 2.30 68% level -- small, but enough to break a FLAT surface
#     into disconnected contour islands, which is what the cos-theta-dominated
#     region is.
#   * REFUTED: "the shared module-level RNG desynchronizes the stream across
#     grid points". Per-point re-seeding changed nothing (3.213 -> 3.100).
#     sample_cos_star() draws by inverse-CDF, one uniform per event, not by
#     rejection, so there was little desync to remove.
#   * REFUTED: "the nt=160 t-grid in sample_cos_star is too coarse near the
#     1/(t - m_Zp^2)^2 propagator peak". A 10x refinement changed nothing
#     (nt=160 -> 3.100, nt=1600 -> 3.425).
#   * Residual patterns correlate 0.92 between seeds, i.e. most apparent
#     "roughness" is REPRODUCIBLE STRUCTURE in chi2_cos(m_Zp), not scatter --
#     a degree-4 polynomial baseline just cannot fit the true curve. Do not
#     quote polynomial-residual RMS as a noise estimate; use seed-to-seed.
# The E_vis term is exactly smooth (0.00 scatter) because analytic_sp_mb
# re-seeds internally (seed=7) and production kinematics are m_Zp-independent.
RNG_SEED = int(os.environ.get("RNG_SEED", "0"))

def forward_point(mphi, mzp):
    """FRESH full-engine re-simulation at (m_phi, m_Zp): returns (E-hist over data
    bins at P_REF, normalized cos-theta shape).  No caching of hits, no sigma table."""
    S.M_PHI = float(mphi); S.M_ZP = float(mzp)         # build_onshell_models reads these globals
    rng = np.random.default_rng(RNG_SEED)              # same stream at every mass point
    if FRESH_FLUX_PER_POINT:
        SA._DK2NU_CACHE.clear()                         # force a real flux re-read (very slow)
    Ehist = np.zeros(len(DATA_N)); cos_all=[]; w_all=[]
    cos20 = np.zeros(len(COS20_EDGES)-1)
    for ch in MUON_CHANNELS:
        pdg,m_M,m_l,lp,nu,g = S.CHANNELS[ch]
        if (m_M-m_l) <= S.M_PHI: continue               # channel closed for heavy phi
        E,w,c = SA.analytic_sp_mb(S, ch, n_dec=NDEC, eff_mode="mb",
                                  return_cos=True, meson_fn=SA._mesons_dk2nu)
        E,w,c = np.asarray(E),np.asarray(w),np.asarray(c)
        if E.size==0: continue
        dp = S.build_onshell_models(pdg,m_M,m_l,lp,nu)["models"]["primakoff"]._dp  # has m_phi,m_Zp set
        cg = DP.smear_photon_beam(c, E, dp, rng)         # mediator dir -> photon dir
        Ehist += np.histogram(E, bins=EBINS, weights=w)[0]
        m=(E>=WIN[0])&(E<=WIN[1]); cos_all.append(cg[m]); w_all.append(w[m])
        m20=(E>=COS20_WIN[0])&(E<=COS20_WIN[1])
        cos20 += np.histogram(cg[m20], bins=COS20_EDGES, weights=w[m20])[0]
    if cos_all:
        cc=np.concatenate(cos_all); ww=np.concatenate(w_all)
        ch,_=np.histogram(cc,bins=COS_EDGES,weights=ww); cosshape=ch/ch.sum() if ch.sum()>0 else ch
    else:
        cosshape=np.zeros(NCB)
    return Ehist, cosshape, cos20

def chi2_over_product(Ehist, cosshape):
    """chi2 vs (E_vis data + cos template) for every product on PROD_GRID."""
    chi2_c = np.sum((COS_TGT - cosshape)**2 / COS_SIG**2)   # product-independent (shape)
    out = np.empty(len(PROD_GRID))
    for k,P in enumerate(PROD_GRID):
        sig = Ehist * (P/P_REF)**2
        out[k] = np.sum((EXCESS - sig)**2 * INV2) + chi2_c
    return out

def main():
    npt = N_MPHI * N_MZP
    log("="*64)
    log(" BRUTE-FORCE re-simulating grid  [%s]" % PORTAL)
    log("  m_phi: %d pts (%.0f-%.0f MeV)   m_Zp: %d pts (%.0f-%.0f MeV)   product: %d pts"
          % (N_MPHI, MPHI_GRID[0]*1e3, MPHI_GRID[-1]*1e3, N_MZP, MZP_GRID[0]*1e3, MZP_GRID[-1]*1e3, N_PROD))
    log("  full engine re-runs: %d   n_dec=%d   fresh_flux_per_point=%s" % (npt, NDEC, FRESH_FLUX_PER_POINT))
    log("="*64)
    heartbeat({"stage":"starting", "pid": os.getpid(), "npt": npt, "done": 0})

    # ---- time ONE point to print an ETA, then scan ----
    log("  timing first engine re-run ...")
    t0=time.time(); E0,c0,_ = forward_point(MPHI_GRID[0], MZP_GRID[len(MZP_GRID)//2]); dt=time.time()-t0
    eta = dt*npt + (dt*npt if FRESH_FLUX_PER_POINT else 0)
    log("  1 engine re-run = %.1f s  ->  ESTIMATED TOTAL ~ %.0f min (%.1f h)"
          % (dt, eta/60, eta/3600))

    CUBE = np.full((N_MPHI, N_MZP, N_PROD), np.nan)
    # Save the RAW forward-model output per mass point, not just the collapsed
    # chi2. The chi2 cube alone is a dead end: chi2_E only survives in it as
    # B=sum(excess*Ehist/err^2) and C=sum(Ehist^2/err^2), two numbers that
    # cannot be inverted back to the 11 per-bin predictions. Keeping Ehist and
    # cosshape (26 floats per point, negligible) makes ANY later change of
    # error model -- e.g. adding the background systematics the paper adds in
    # quadrature, which we currently omit -- a seconds-long recomputation
    # instead of another multi-hour re-simulation.
    EHIST = np.full((N_MPHI, N_MZP, len(DATA_N)), np.nan)
    COSSH = np.full((N_MPHI, N_MZP, NCB), np.nan)
    COS20 = np.full((N_MPHI, N_MZP, len(COS20_EDGES)-1), np.nan)
    done=0
    t_start = time.time()
    n_errors = 0
    for a,mphi in enumerate(MPHI_GRID):
        for b,mzp in enumerate(MZP_GRID):
            pt_t0 = time.time()
            try:
                Eh,cs,c20 = forward_point(mphi, mzp)
                CUBE[a,b] = chi2_over_product(Eh, cs)
                EHIST[a,b] = Eh; COSSH[a,b] = cs; COS20[a,b] = c20
            except Exception as e:
                # Don't let one bad grid point silently kill or hang a
                # multi-hour run -- log it, leave that cell as NaN, move on.
                n_errors += 1
                log("  !! ERROR at m_phi=%.1f MeV, m_Zp=%.1f MeV: %s"
                      % (mphi*1e3, mzp*1e3, repr(e)))
                log(traceback.format_exc())
            done += 1
            pt_dt = time.time() - pt_t0
            # Per-point heartbeat: cheap, always up to date, tells you
            # immediately (via mtime + this JSON) whether it's alive.
            heartbeat({
                "stage": "running",
                "pid": os.getpid(),
                "portal": PORTAL,
                "done": done, "total": npt,
                "last_point": {"m_phi_MeV": mphi*1e3, "m_Zp_MeV": mzp*1e3},
                "last_point_seconds": round(pt_dt, 2),
                "errors_so_far": n_errors,
                "updated_at": datetime.now().isoformat(),
            })
            # Print every point too (not just every 10 rows) -- cheap, and
            # means a `tail -f output/scan_brute_grid.log` shows continuous
            # progress instead of long silent gaps.
            el = time.time() - t_start
            rate = done / el if el > 0 else 0
            eta_min = (npt - done) / rate / 60 if rate > 0 else float("nan")
            log("  pt %d/%d  m_phi=%.1f MeV m_Zp=%.1f MeV  took=%.1fs  elapsed=%.0fmin  eta=%.0fmin  errors=%d"
                  % (done, npt, mphi*1e3, mzp*1e3, pt_dt, el/60, eta_min, n_errors))
        if (a+1)%CKPT_EVERY==0 or a==N_MPHI-1:
            np.savez("output/brute_%s_chi2cube.npz"%PORTAL, chi2=CUBE,
                     Ehist=EHIST, cosshape=COSSH, cos20=COS20, cos20_edges=COS20_EDGES,
                     cos20_win=np.array(COS20_WIN), Ebins_GeV=EBINS, cos_edges=COS_EDGES,
                     excess=EXCESS, data_err=DATA_ERR, P_ref=P_REF,
                     mphi_MeV=MPHI_GRID*1e3, mzp_MeV=MZP_GRID*1e3, prod=PROD_GRID)
            el=time.time()-t_start
            log("  CHECKPOINT SAVED: m_phi row %d/%d done  (%d/%d pts, %.0f min elapsed, ~%.0f min left, %d errors)"
                  % (a+1,N_MPHI, done,npt, el/60, el/60*(npt-done)/max(done,1), n_errors))

    # ---- marginalize to (m_Zp, product): profile over m_phi (min chi2) ----
    prof = np.nanmin(CUBE, axis=0)                      # [m_Zp, product]
    np.savez("output/brute_%s_chi2cube.npz"%PORTAL, chi2=CUBE, prof_mZp_prod=prof,
             Ehist=EHIST, cosshape=COSSH, cos20=COS20, cos20_edges=COS20_EDGES,
                     cos20_win=np.array(COS20_WIN), Ebins_GeV=EBINS, cos_edges=COS_EDGES,
             excess=EXCESS, data_err=DATA_ERR, P_ref=P_REF,
             mphi_MeV=MPHI_GRID*1e3, mzp_MeV=MZP_GRID*1e3, prod=PROD_GRID)
    _plot_region(prof)
    imin=np.unravel_index(np.nanargmin(CUBE), CUBE.shape)
    log("  BEST FIT: chi2=%.2f at m_phi=%.0f, m_Zp=%.0f MeV, product=%.2e"
          % (CUBE[imin], MPHI_GRID[imin[0]]*1e3, MZP_GRID[imin[1]]*1e3, PROD_GRID[imin[2]]))
    log("  wrote output/brute_%s_chi2cube.npz + _region.png" % PORTAL)
    heartbeat({"stage":"done", "pid": os.getpid(), "done": npt, "total": npt,
               "updated_at": datetime.now().isoformat()})

def _plot_region(prof):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    D = prof - np.nanmin(prof)
    X,Y = np.meshgrid(MZP_GRID*1e3, PROD_GRID, indexing="ij")
    fig,ax=plt.subplots(figsize=(7,6))
    ax.contourf(X,Y,D,levels=[0,2.30],colors=["#69c"],alpha=0.55)     # 68%
    ax.contourf(X,Y,D,levels=[0,6.18],colors=["#bcd"],alpha=0.30)     # 95%
    ax.contour(X,Y,D,levels=[2.30,6.18],colors="k",linewidths=[1.5,0.8])
    star={"scalar":(49,2.2e-8),"pseudo":(85,5.9e-7)}[PORTAL]
    ax.plot(*star,"r*",ms=18,label="paper Table I")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$m_{Z'}$ [MeV]"); ax.set_ylabel(r"$g_\mu g_n\lambda$ [MeV$^{-1}$]")
    ax.set_title("BRUTE-FORCE re-simulating grid (%s): 68%%/95%% region\n(profiled over $m_\\phi$; full engine re-run per mass point)"%PORTAL)
    ax.legend(); ax.grid(True,which="both",alpha=0.3)
    fig.tight_layout(); fig.savefig("output/brute_%s_region.png"%PORTAL,dpi=120)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log("  !! FATAL ERROR, run aborted: %s" % repr(e))
        log(traceback.format_exc())
        heartbeat({"stage": "crashed", "pid": os.getpid(),
                   "error": repr(e), "updated_at": datetime.now().isoformat()})
        raise
