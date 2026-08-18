"""MiniBooNE nu-mode cos(theta) DATA, extracted from FIG. 8 of arXiv:2006.16883.

This replaces the cos_template_nu.json "template", which was a pixel-extracted
shape of the DUTTA-KIM PAPER'S OWN Fig. 2 signal band carrying an invented error
(sqrt(p(1-p)/320)+0.01). Using a paper's model prediction to constrain a test of
that paper is circular, and measurably so: on the brute grid that template
supplied 145 chi2 units of spread against 61 from the real E_vis data, i.e. it
outvoted the measurement and drove the fitted m_Zp from 30 to ~100 MeV.

These are real measured counts with real statistical errors.

SOURCE: arXiv:2006.16883 (Phys. Rev. D 103, 052002) FIG. 8 -- "The MiniBooNE
neutrino mode cos(theta) distributions, corresponding to the total 18.75e20 POT
data in the 200 < E_nu^QE < 1250 MeV energy range, for nu_e CCQE data (points
with statistical errors) and background (colored histogram)."

HOW (not a pixel digitisation): the paper's figures are VECTOR graphics, so the
data marker centres and the stacked-histogram staircase vertices are exact PDF
coordinates. They were read with PyMuPDF get_drawings() and mapped to data
coordinates by a linear fit to the tick-label bounding-box centres (x residual
0.0035 in cos, y residual 0.31 events over the full range). The background is
the per-bin maximum over the seven stacked components, which in a ROOT THStack
is the cumulative total. Statistical errors: the drawn error bars were verified
to equal sqrt(N) on the 12 bins where the bar is taller than the marker
(extracted = sqrt(N) + 0.55 constant line-width offset); below that the marker
covers the bar, so sqrt(N) is used throughout on that demonstrated basis.

VALIDATION -- three independent checks, none of them tuned:
  1. The same method applied to FIG. 9 on the same page (E_vis, for which we
     have exact HEPData values) reproduces them to 0.45% RMS -- 13x smaller than
     the statistical error. See miniboone_data.py for those numbers.
  2. Extracted counts come out as INTEGERS (a constant +0.08 calibration offset
     removed), as event counts must be.
  3. The extracted excess integrates to 545.7 +/- 53.4 (stat) against
     MiniBooNE's published nu-mode excess of 638.0 +/- 132.8 (PRD 103 052002
     abstract, 18.75e20 POT). NB our 545.6 covers 200-1250 MeV only; the
     published 638.0 also includes 150-200 MeV, which HEPData t6 does not
     tabulate -- so this is consistency, not an identity.

CAVEATS:
  * FIG. 9's tail is drawn on a COMPRESSED x-axis (its last three markers sit at
    1200/1400/1600 MeV against release centres 1175/1375/2250). FIG. 8 is not
    affected -- its 20 cos bins were verified uniform to 0.0999 width -- but it
    is why bin edges must always be read from the figure, never assumed.
  * The extracted FIG. 8 data total (2850) is 0.7% below the HEPData E_vis total
    over the same 200-1250 MeV range (2870). Unexplained; too small to matter at
    these statistics, but not zero, so do not treat the two as the same sample.
  * These are STATISTICAL errors only. The published excess error (132.8) is
    2.2x the statistical one, so an angular systematic almost certainly belongs
    here too -- MiniBooNE does not provide one per angular bin. Treat a fit
    using stat-only angular errors as OVER-constraining, in the same way
    stat-only E_vis errors were.
  * E_vis and cos(theta) are two projections of the SAME events. Using both as
    independent chi2 terms double-counts the data. The paper does this anyway;
    the clean alternative is FIG. 13's 2D (E_vis, cos theta) excess, which is a
    colour map and therefore much lossier to read.
"""
import numpy as np

# 20 uniform bins, verified from the marker spacing (width 0.0999 +/- 0.000)
EDGES = np.linspace(-1.0, 1.0, 21)
CENTRES = 0.5 * (EDGES[:-1] + EDGES[1:])

# nu_e CCQE candidates, 200 < E_nu^QE < 1250 MeV, 18.75e20 POT
DATA = np.array([35, 40, 33, 44, 49, 62, 50, 53, 65, 91,
                 103, 110, 87, 126, 188, 187, 215, 336, 400, 576], float)

# total predicted background (top of the 7-component stack)
BKG = np.array([25.8, 22.3, 26.0, 33.4, 39.0, 33.4, 37.8, 43.6, 59.2, 55.9,
                75.8, 84.4, 83.8, 133.3, 129.6, 174.2, 209.9, 257.3, 337.4, 442.3])

EXCESS = DATA - BKG
STAT = np.sqrt(DATA)                      # verified Poisson, see module docstring


def shape():
    """Excess normalised to unit sum -- for comparing against a shape prediction."""
    return EXCESS / EXCESS.sum()


if __name__ == "__main__":
    print("bins            : %d uniform, %.1f to %.1f" % (len(DATA), EDGES[0], EDGES[-1]))
    print("data total      : %.0f" % DATA.sum())
    print("background total: %.1f" % BKG.sum())
    print("EXCESS          : %+.1f +/- %.1f (stat)" % (EXCESS.sum(), np.sqrt(DATA.sum())))
    print("published       : 638.0 +/- 132.8 (incl. 150-200 MeV we do not have)")
    fwd = EXCESS[CENTRES > 0.9].sum() / EXCESS.sum()
    print("forward fraction: %.1f%% of the excess sits at cos(theta) > 0.9" % (100 * fwd))
