"""MiniBooNE nu-mode nu_e BACKGROUND systematics, from the official public release.

PROVENANCE (fetched 2026-08-16): HEPData record ins1804293 (DOI 10.17182/hepdata.114365.v1),
table t14 "Neutrino fractional covariance matrix", tied to arXiv:2006.16883 -- the same
reference [3] that Dutta-Kim arXiv:2110.11944 cites for its nu-mode data AND for the
background systematics it approximates.

The HEPData table is a 30x30 FRACTIONAL covariance sigma^2_ij/(mu_i mu_j) in three
side-by-side diagonal blocks, in this order:
    bins  0-10 : nu_e "full transmutation" signal   (11 E_vis bins)
    bins 11-21 : nu_e BACKGROUND prediction         (11 E_vis bins)  <-- what we want
    bins 22-29 : nu_mu CCQE                         (8 bins)
Only the background block is reproduced here. Bin edges match EBINS in scan_brute_grid.py.

WHY THIS EXISTS: scan_brute_grid.py (and the other example4 fit scripts) use STAT-ONLY
errors, while the paper uses "statistical and systematic uncertainties added by quadrature"
(arXiv:2110.11944 p.3, "Fits and discussions"). Stat-only is why our 68% band comes out at
an implausible ~+/-10% in coupling for a systematics-dominated measurement.

CAUTION -- read before using TOTAL_COV:
  * These systematics are UNCONSTRAINED. MiniBooNE's published excess (638.0 +/- 132.8)
    is quoted AFTER the nu_mu CCQE constraint, which is what block 3 and its cross-
    correlations exist to apply. Propagating the correlated background block alone gives
    +/-313 on the excess; quadrature-summing the per-bin diagonals gives +/-161; stat-only
    gives +/-58.7. Use SYS_ABS in quadrature to MATCH THE PAPER's stated approximation;
    use TOTAL_COV only if you also implement the nu_mu constraint, or you will overcover.
  * The background block is correlated bin-to-bin (mean off-diagonal correlation 0.38), so
    treating SYS_ABS as independent is itself an approximation -- the same one the paper made.

EFFECT ON THE FIT (measured 2026-08-16): these errors are 2.2x-3.5x the stat errors, so each
bin's chi2 weight drops 5-12x. That shrinks the E_vis chi2 spread across the mass grid from
~61 to ~10 while leaving the cos-theta template term's ~145 untouched (its error is invented,
see COS_SIG in scan_brute_grid.py). Adding these systematics WITHOUT also fixing the
cos-theta term therefore makes the fitted m_Zp MORE paper-discrepant, not less.
"""
import numpy as np

# Correlation structure of the background block (mean off-diagonal 0.38). Stored as the
# fractional covariance so it can be rescaled if the background prediction is ever updated.
_FRACCOV_BKG = np.array([
    [ 0.02373900,  0.01202300,  0.01204000,  0.00852700,  0.01048200,  0.00284400,  0.00109500, -0.00423500, -0.00465200, -0.00476600,  0.00454700],
    [ 0.01202300,  0.01910700,  0.01140200,  0.01416000,  0.00928000,  0.00679100,  0.00486100,  0.00789500, -0.00170800,  0.00723400,  0.00127300],
    [ 0.01204000,  0.01140200,  0.02247400,  0.01140800,  0.01339600,  0.01039200,  0.00958100,  0.00854300,  0.01067700,  0.00715000,  0.01233800],
    [ 0.00852700,  0.01416000,  0.01140800,  0.02979000,  0.01220800,  0.01368200,  0.01557900,  0.01388400,  0.01412900,  0.01517500,  0.01462800],
    [ 0.01048200,  0.00928000,  0.01339600,  0.01220800,  0.02932800,  0.01256200,  0.01867700,  0.01226500,  0.01945100,  0.02013800,  0.02084600],
    [ 0.00284400,  0.00679100,  0.01039200,  0.01368200,  0.01256200,  0.03082500,  0.02430800,  0.01958700,  0.02246300,  0.02076900,  0.02237300],
    [ 0.00109500,  0.00486100,  0.00958100,  0.01557900,  0.01867700,  0.02430800,  0.04014500,  0.02658300,  0.02912800,  0.02758600,  0.03070300],
    [-0.00423500,  0.00789500,  0.00854300,  0.01388400,  0.01226500,  0.01958700,  0.02658300,  0.04090400,  0.02198500,  0.03418700,  0.01963800],
    [-0.00465200, -0.00170800,  0.01067700,  0.01412900,  0.01945100,  0.02246300,  0.02912800,  0.02198500,  0.06027600,  0.03814200,  0.04017600],
    [-0.00476600,  0.00723400,  0.00715000,  0.01517500,  0.02013800,  0.02076900,  0.02758600,  0.03418700,  0.03814200,  0.05996900,  0.03247000],
    [ 0.00454700,  0.00127300,  0.01233800,  0.01462800,  0.02084600,  0.02237300,  0.03070300,  0.01963800,  0.04017600,  0.03247000,  0.05601500],
])

# E_vis bin edges [MeV], 11 bins -- identical to EBINS*1e3 in scan_brute_grid.py
EDGES_MEV = np.array([200, 300, 375, 475, 550, 675, 800, 950, 1100, 1250, 1500, 3000], float)

# MiniBooNE nu-mode nu_e background prediction per bin [events]
BKG = np.array([527.164624, 315.423689, 349.644825, 186.21197, 261.441799,
                195.534193, 203.008745, 165.664396, 118.581365, 143.989367, 201.450357])

# sqrt of the diagonal of the fractional covariance, background block (bins 11-21).
# 15% at low E rising to ~25% above 1 GeV. DERIVED from _FRACCOV_BKG below rather
# than transcribed, so it can never drift from the matrix it came from.
FRAC_SYS = np.sqrt(np.diag(_FRACCOV_BKG))

# Absolute background systematic per bin [events] = FRAC_SYS * BKG
SYS_ABS = FRAC_SYS * BKG

# Statistical errors on the DATA, symmetrized from the HEPData stat+/stat- columns.
STAT = np.array([27.83, 21.33, 21.83, 16.33, 17.33, 15.83, 14.83, 13.33, 12.33, 12.82, 14.33])

# What the paper says it does: stat and syst added in quadrature.
TOTAL_ERR = np.hypot(STAT, SYS_ABS)


def total_cov(include_stat=True):
    """Full correlated background covariance [events^2], optionally with stat on the
    diagonal. Requires the nu_mu constraint to be defensible -- see module docstring."""
    cov = _FRACCOV_BKG * np.outer(BKG, BKG)
    if include_stat:
        cov = cov + np.diag(STAT ** 2)
    return cov



if __name__ == "__main__":
    excess_err_stat = np.sqrt((STAT ** 2).sum())
    excess_err_quad = np.sqrt((TOTAL_ERR ** 2).sum())
    excess_err_corr = np.sqrt(total_cov().sum())
    print("per-bin sys/stat ratio :", " ".join("%.1f" % r for r in TOTAL_ERR / STAT))
    print("error on total excess  : stat-only %.1f | quadrature %.1f | correlated %.1f"
          % (excess_err_stat, excess_err_quad, excess_err_corr))
    print("chi2 weight reduction  : %.1fx - %.1fx per bin"
          % (((TOTAL_ERR / STAT) ** 2).min(), ((TOTAL_ERR / STAT) ** 2).max()))
