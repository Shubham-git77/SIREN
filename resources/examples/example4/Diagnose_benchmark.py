"""
DIAGNOSTIC (fast, single-point): run the SAME forward_point() engine used by
scan_brute_grid.py, but only ONCE, at the paper's own claimed best-fit
benchmark (m_phi=1 MeV, m_Zp=49 MeV, product=2.2e-8 MeV^-1 -- Table I,
"Scalar" row of arXiv:2110.11944), and plot the predicted E_vis histogram
directly against the real data (DATA_N - BKG) and the cos-theta template.

Why: three separate corrections (rebinned data, real MiniBooNE efficiency
curve) all left the *grid-scan* best fit stuck at m_Zp=97 MeV. Rather than
keep guessing which file has the bug, this script asks the direct question:
"if I force the paper's own numbers in, does my engine reproduce their
excess shape?" If yes -> the physics/engine is fine and the discrepancy is
purely a fitting/likelihood-weighting issue. If no -> the engine itself
doesn't reproduce the paper's claimed signal, which is a much more basic and
important thing to know before trusting any grid-scan result.

Run (same env as scan_brute_grid.py):
  PORTAL=scalar DK2NU_FILE=/home/shubham/nubeam12M.dk2nu.root \
      python diagnose_benchmark.py
Output: output/diagnostic_benchmark.png (+ printed numeric comparison)
"""
import os, sys, json, importlib.util, numpy as np

PORTAL = os.environ.get("PORTAL", "scalar")
os.environ.setdefault("DK2NU_FILE", "/home/shubham/nubeam12M.dk2nu.root")
HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
PKG  = os.environ.get(
    "SIREN_DNT_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "processes", "DarkNewsTables"),
)

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m

SA = load(os.path.join(PKG, "AnalyticRate.py"), "AnalyticRate")
DP = load(os.path.join(PKG, "DarkPrimakoff.py"), "DPmod")
CFG = {"scalar":"ScalarPortal_MiniBooNE_multichannel.py",
       "pseudo":"PseudoscalarPortal_MiniBooNE_multichannel.py"}[PORTAL]

# ---- same corrected data as scan_brute_grid.py ----
DATA_E = np.array([250.0,337.5,425.0,512.5,612.5,737.5,875.0,1025.0,1175.0,1375.0,2250.0])
DATA_N = np.array([732,426,444,248,281,236,201,164,138,144,188], float)
DATA_ERR = np.array([27.83,21.33,21.83,16.33,17.33,15.83,14.83,13.33,12.33,12.82,14.33])
BKG    = np.array([527.164624,315.423689,349.644825,186.21197,261.441799,
                    195.534193,203.008745,165.664396,118.581365,143.989367,201.450357])
EBINS  = np.array([200,300,375,475,550,675,800,950,1100,1250,1500,3000]) / 1e3
EXCESS = DATA_N - BKG
ct = json.load(open("cos_template_nu.json"))
COS_TGT = np.array(ct["shape"]); COS_TGT /= COS_TGT.sum()
NCB = len(COS_TGT); COS_EDGES = np.linspace(-1,1,NCB+1)
WIN = (0.14, 999.0)

S = load(CFG, "S_brute")
P_REF = S.G_MU_PROD * S.G_N * (S.LAMBDA*1e-3)
MUON_CHANNELS = [c for c in S.CHANNELS if "mu" in c]
RNG = np.random.default_rng(0)

# ---- THE PAPER'S OWN BENCHMARK (Table I, scalar row) ----
M_PHI_PAPER = 1e-3    # GeV
M_ZP_PAPER  = 49e-3   # GeV
PROD_PAPER  = 2.2e-8  # MeV^-1  (g_mu * g_n * lambda)

def forward_point(mphi, mzp, ndec=400):
    S.M_PHI = float(mphi); S.M_ZP = float(mzp)
    Ehist = np.zeros(len(DATA_N)); cos_all=[]; w_all=[]
    for ch in MUON_CHANNELS:
        pdg,m_M,m_l,lp,nu,g = S.CHANNELS[ch]
        if (m_M-m_l) <= S.M_PHI: continue
        E,w,c = SA.analytic_sp_mb(S, ch, n_dec=ndec, eff_mode="mb",
                                  return_cos=True, meson_fn=SA._mesons_dk2nu)
        E,w,c = np.asarray(E),np.asarray(w),np.asarray(c)
        if E.size==0: continue
        dp = S.build_onshell_models(pdg,m_M,m_l,lp,nu)["models"]["primakoff"]._dp
        cg = DP.smear_photon_beam(c, E, dp, RNG)
        Ehist += np.histogram(E, bins=EBINS, weights=w)[0]
        m=(E>=WIN[0])&(E<=WIN[1]); cos_all.append(cg[m]); w_all.append(w[m])
    if cos_all:
        cc=np.concatenate(cos_all); ww=np.concatenate(w_all)
        ch_,_=np.histogram(cc,bins=COS_EDGES,weights=ww); cosshape=ch_/ch_.sum() if ch_.sum()>0 else ch_
    else:
        cosshape=np.zeros(NCB)
    return Ehist, cosshape

def main():
    print("="*64)
    print(" DIAGNOSTIC: forcing paper's own benchmark point")
    print("  m_phi=%.1f MeV  m_Zp=%.1f MeV  product=%.2e MeV^-1" %
          (M_PHI_PAPER*1e3, M_ZP_PAPER*1e3, PROD_PAPER))
    print("="*64)

    Ehist_ref, cosshape = forward_point(M_PHI_PAPER, M_ZP_PAPER, ndec=400)
    sig = Ehist_ref * (PROD_PAPER / P_REF) ** 2

    print("\n%-12s %10s %10s %10s %10s" % ("E bin (MeV)", "data-bkg", "predicted", "data", "bkg"))
    for i in range(len(DATA_E)):
        print("%-12.0f %10.1f %10.1f %10.1f %10.1f" %
              (DATA_E[i], EXCESS[i], sig[i], DATA_N[i], BKG[i]))

    chi2 = np.sum((EXCESS - sig)**2 / DATA_ERR**2)
    print("\nchi2 (E_vis only, stat errors) at paper's own benchmark = %.2f  (ndof=%d)" %
          (chi2, len(DATA_E)))

    os.makedirs("output", exist_ok=True)
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12,5))

    ax = axes[0]
    ax.errorbar(DATA_E, EXCESS, yerr=DATA_ERR, fmt="ko", label="data - bkg (real excess)")
    ax.plot(DATA_E, sig, "r-s", label="model @ paper benchmark\n(m_Zp=49 MeV)")
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlabel("E_vis [MeV]"); ax.set_ylabel("excess events")
    ax.set_title("E_vis: does the engine reproduce the paper's excess?")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    cos_centers = 0.5*(COS_EDGES[:-1]+COS_EDGES[1:])
    ax.plot(cos_centers, COS_TGT, "ko-", label="paper cos-theta template")
    ax.plot(cos_centers, cosshape, "r-s", label="model @ paper benchmark")
    ax.set_xlabel("cos(theta)"); ax.set_ylabel("normalized shape")
    ax.set_title("cos-theta shape comparison")
    ax.legend(); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("output/diagnostic_benchmark.png", dpi=120)
    print("\nwrote output/diagnostic_benchmark.png")

if __name__ == "__main__":
    main()
