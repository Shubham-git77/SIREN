"""Re-derive credible regions from a saved brute-grid cube WITHOUT re-simulating.

The expensive part of scan_brute_grid.py is the forward model: the per-mass-point
E_vis histogram and cos-theta shape. Everything downstream of those -- the error
model, whether systematics are included, how hard the cos-theta term is weighted,
even the coupling grid -- is arithmetic. This script redoes that arithmetic in
seconds against a cube that saved its raw forward output.

WHY IT CAN: the E_vis chi2 is exactly quadratic in x = (P/P_ref)^2,
    chi2_E(x) = e^T W e - 2 x h^T W e + x^2 h^T W h        (e = excess, h = Ehist)
so with h stored per cell, any weight matrix W (diagonal stat, diagonal
stat+syst, or a full covariance inverse) is a closed-form re-evaluation. The
cos-theta term is product-independent and adds a per-cell constant.

REQUIRES a cube written by the patched scan_brute_grid.py, which saves Ehist and
cosshape. Cubes from before that patch hold only the collapsed chi2, where the
E_vis term survives as just B = h^T W e and C = h^T W h -- two numbers that
cannot be inverted back to 11 per-bin predictions. Those cubes must be
regenerated; this script says so rather than guessing.

Usage:
  python rederive_regions.py --cube output/brute_pseudo_chi2cube.npz \
      --errors quad --cos off --out output/pseudo_quad_nocos.png

  --errors stat   data statistical errors only (what the original runs used)
          quad   stat + MiniBooNE background systematics in quadrature
                 <- matches what Dutta-Kim state they did
          cov    full correlated background covariance (see the caution in
                 miniboone_systematics: unconstrained, will overcover)
  --cos   off    drop the cos-theta template term entirely
          on     keep it at its original (invented) weight
          xN     inflate the cos-theta errors by N, e.g. --cos x3
          data   REAL MiniBooNE cos-theta data (FIG. 8 of 2006.16883), stat only
          data-syst  same, plus an assumed angular background systematic

The "data" modes are qualitatively different from the template modes. The
template is a SHAPE with an invented error, so its chi2 is product-independent
and it constrains only the mass. The real data are ABSOLUTE counts, so the
angular term scales with (P/P_ref)^2 like the energy term and constrains the
COUPLING too -- it enters the same quadratic in x=(P/P_ref)^2 rather than adding
a constant. They need a cube carrying `cos20` (the prediction on the data's own
20-bin, 200-1250 MeV footprint); older cubes stored only the 15-bin shape over a
140-300 MeV window, which matches neither the binning nor the selection.
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

PAPER = {"scalar": (49.0, 2.2e-8), "pseudo": (85.0, 5.9e-7)}


def load_cube(path):
    d = np.load(path)
    missing = [k for k in ("Ehist", "cosshape") if k not in d]
    if missing:
        sys.exit(
            "ERROR: %s lacks %s.\n"
            "It predates the raw-output patch, so its per-bin predictions are\n"
            "unrecoverable -- the collapsed chi2 keeps only h^T W e and h^T W h.\n"
            "Re-run scan_brute_grid.py with the current script to regenerate it."
            % (path, " and ".join(missing))
        )
    return d


def weight_matrix(mode, excess_err):
    """Return (W, label). W is the inverse-covariance used by the chi2."""
    import miniboone_systematics as MS

    if mode == "stat":
        return np.diag(1.0 / excess_err ** 2), "stat only"
    if mode == "quad":
        err = np.hypot(excess_err, MS.SYS_ABS)
        return np.diag(1.0 / err ** 2), "stat + bkg syst (quadrature)"
    if mode == "cov":
        return np.linalg.inv(MS.total_cov(include_stat=True)), "full correlated covariance"
    raise ValueError(mode)


def cos_data_weights(mode):
    """(excess, weights, label) for the REAL angular data."""
    import miniboone_costheta as MC
    exc = MC.EXCESS
    if mode == "data":
        sig = MC.STAT
        lab = "REAL cos-theta data, stat only"
    else:
        # MiniBooNE publishes no per-angular-bin systematic. Scale a flat
        # fractional background error until the TOTAL excess uncertainty
        # reproduces the published 638.0 +/- 132.8. An assumption, but anchored
        # to a published number rather than invented like the old template's
        # 320-event normalisation.
        need = max(132.8 ** 2 - MC.DATA.sum(), 0.0)
        f = np.sqrt(need / (MC.BKG ** 2).sum())
        sig = np.hypot(MC.STAT, f * MC.BKG)
        lab = "REAL cos-theta data + %.1f%% bkg syst (tuned to published 132.8)" % (100 * f)
    return exc, 1.0 / sig ** 2, lab


def cos_term(cosshape, cos_mode):
    """Product-independent chi2 contribution from the cos-theta template."""
    if cos_mode == "off":
        return np.zeros(cosshape.shape[:2]), "cos-theta dropped"
    scale = 1.0
    if cos_mode.startswith("x"):
        scale = float(cos_mode[1:])
    ct = json.load(open(os.path.join(HERE, "cos_template_nu.json")))
    tgt = np.array(ct["shape"], float); tgt /= tgt.sum()
    sig = (np.sqrt(tgt * (1 - tgt) / 320.0) + 0.01) * scale
    chi2 = (((tgt[None, None, :] - cosshape) / sig[None, None, :]) ** 2).sum(axis=2)
    lab = "cos-theta at original weight" if scale == 1 else "cos-theta errors x%g" % scale
    return chi2, lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cube", required=True)
    ap.add_argument("--errors", default="quad", choices=("stat", "quad", "cov"))
    ap.add_argument("--cos", default="off",
                    help="off | on | xN | data | data-syst")
    ap.add_argument("--out", default=None, help="output PNG (default: alongside the cube)")
    ap.add_argument("--portal", default=None, help="scalar|pseudo (default: infer from filename)")
    a = ap.parse_args()

    portal = a.portal or ("pseudo" if "pseudo" in os.path.basename(a.cube) else "scalar")
    d = load_cube(a.cube)
    H = d["Ehist"]                      # (n_mphi, n_mzp, n_bins)
    CS = d["cosshape"]                  # (n_mphi, n_mzp, n_cosbins)
    mzp = d["mzp_MeV"]; mphi = d["mphi_MeV"]; prod = d["prod"]
    excess = d["excess"]; err = d["data_err"]; P_ref = float(d["P_ref"])

    W, wlab = weight_matrix(a.errors, err)

    # chi2_E(x) = A - 2 B x + C x^2, with x = (P/P_ref)^2
    A = float(excess @ W @ excess)
    B = np.einsum("abi,ij,j->ab", H, W, excess)
    C = np.einsum("abi,ij,abj->ab", H, W, H)

    if a.cos in ("data", "data-syst"):
        if "cos20" not in d:
            sys.exit(
                "ERROR: %s has no `cos20`.\n"
                "The real-data angular modes need the prediction on the DATA's own\n"
                "footprint (20 bins, 200-1250 MeV). This cube stores only the legacy\n"
                "15-bin shape over a 140-300 MeV window, which matches neither the\n"
                "binning nor the selection. Re-run scan_brute_grid.py to regenerate."
                % a.cube)
        pred = d["cos20"]
        e_c, w_c, clab = cos_data_weights(a.cos)
        # the angular term is ALSO quadratic in x, so it folds into A/B/C
        A = A + float((e_c ** 2 * w_c).sum())
        B = B + np.einsum("abi,i,i->ab", pred, w_c, e_c)
        C = C + np.einsum("abi,i,abi->ab", pred, w_c, pred)
        cc = np.zeros(B.shape)
    else:
        cc, clab = cos_term(CS, "on" if a.cos == "on" else a.cos)

    x = (prod / P_ref) ** 2
    CUBE = (A - 2.0 * B[..., None] * x + C[..., None] * x ** 2) + cc[..., None]

    prof = np.nanmin(CUBE, axis=0)                 # profile over m_phi
    D = prof - np.nanmin(prof)

    i = np.unravel_index(np.nanargmin(CUBE), CUBE.shape)
    print("cube      : %s   (%d x %d x %d)" % (a.cube, *CUBE.shape))
    print("errors    : %s" % wlab)
    print("cos-theta : %s" % clab)
    print("null chi2 (no signal) = %.2f" % A)
    print("BEST FIT  : chi2=%.2f  m_phi=%.0f MeV  m_Zp=%.1f MeV  product=%.3e"
          % (CUBE[i], mphi[i[0]], mzp[i[1]], prod[i[2]]))

    # analytic best-fit product vs m_Zp (profiled over m_phi), and the paper point
    xbest = np.where(C > 0, B / np.where(C > 0, C, 1.0), np.nan)
    Pbest = P_ref * np.sqrt(np.clip(xbest, 0, None))
    pm, pp = PAPER[portal]
    j = int(np.argmin(abs(mzp - pm)))
    dchi = float(np.interp(pp, prod, prof[j]) - np.nanmin(prof))
    verdict = ("inside 68%" if dchi < 2.30 else
               "inside 95%" if dchi < 6.18 else "OUTSIDE 95%")
    print("paper pt  : (%.0f MeV, %.1e) -> Delta-chi2 = %.2f   [%s]" % (pm, pp, dchi, verdict))
    a0 = int(np.nanargmin(np.nanmin(CUBE, axis=(1, 2))))
    print("our best-fit product at the paper mass = %.3e  (paper %.1e, ratio %.2f)"
          % (Pbest[a0, j], pp, Pbest[a0, j] / pp))

    out = a.out or os.path.join(os.path.dirname(a.cube),
                                "region_%s_%s_cos-%s.png" % (portal, a.errors, a.cos))
    _plot(D, mzp, prod, Pbest[a0], portal, wlab, clab, out)
    print("wrote %s" % out)


def _plot(D, mzp, prod, Pbest, portal, wlab, clab, out):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    X, Y = np.meshgrid(mzp, prod, indexing="ij")
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contourf(X, Y, D, levels=[0, 2.30], colors=["#69c"], alpha=0.55)
    ax.contourf(X, Y, D, levels=[0, 6.18], colors=["#bcd"], alpha=0.30)
    ax.contour(X, Y, D, levels=[2.30, 6.18], colors="k", linewidths=[1.5, 0.8])
    ax.plot(mzp, Pbest, "w-", lw=3)
    ax.plot(mzp, Pbest, "k--", lw=1.4, label=r"best-fit product vs $m_{Z'}$")
    pm, pp = PAPER[portal]
    ax.plot(pm, pp, "r*", ms=18, label="paper Table I (%.0f MeV, %.1e)" % (pm, pp))
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(mzp[0], mzp[-1]); ax.set_ylim(prod[0], prod[-1])
    ax.set_xlabel(r"$m_{Z'}$ [MeV]"); ax.set_ylabel(r"$g_\mu g_n\lambda$ [MeV$^{-1}$]")
    ax.set_title("%s: 68%%/95%% region (profiled over $m_\\phi$)\n%s; %s"
                 % (portal, wlab, clab), fontsize=10)
    ax.legend(loc="upper left", fontsize=9); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=120)


if __name__ == "__main__":
    main()
