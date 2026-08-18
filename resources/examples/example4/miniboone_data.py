"""MiniBooNE nu-mode E_vis data, background and errors -- ONE authoritative copy.

Every example4 fit script used to carry its own inline copy of these arrays. That is
how the repository ended up with two different binnings in circulation at once:
scan_brute_grid.py was corrected to the official release on 2026-08-10 while
scan_credible_region.py, scan_credible_costheta.py, scan_credible_m25.py,
mcmc_fit.py, mcmc_fit_vector.py and fig2_paperstyle.py kept a 19-bin
digitization, so their results silently stopped being comparable. Import from
here instead of pasting arrays.

SOURCE: HEPData ins1804293 (DOI 10.17182/hepdata.114365.v1) table t6 "NuE data and
background", tied to arXiv:2006.16883 -- the reference [3] that Dutta-Kim
arXiv:2110.11944 cites for its nu-mode data. 11 variable-width bins, 200-3000 MeV,
the last one an overflow bin.

The SUPERSEDED 19x50 MeV uniform-bin arrays matched no official release binning and
were most likely a mis-digitization; they are kept below as LEGACY_* only so old
results can be reproduced for comparison. Do not use them for new work.

Systematics live in miniboone_systematics.py, deliberately separate: these are the
measurement, those are an error-model choice.
"""
import numpy as np

# --- official release, 11 bins -------------------------------------------------
# bin centres [MeV] (the last is an overflow bin, plotted at its centroid)
DATA_E = np.array([250.0, 337.5, 425.0, 512.5, 612.5, 737.5, 875.0,
                   1025.0, 1175.0, 1375.0, 2250.0])
# bin edges [GeV] -- what np.histogram wants
EBINS = np.array([200, 300, 375, 475, 550, 675, 800, 950, 1100, 1250, 1500, 3000]) / 1e3
# observed nu_e candidates
DATA_N = np.array([732, 426, 444, 248, 281, 236, 201, 164, 138, 144, 188], float)
# statistical errors, symmetrized from the HEPData stat+/stat- columns (asymmetry is small)
DATA_ERR = np.array([27.83, 21.33, 21.83, 16.33, 17.33, 15.83, 14.83, 13.33, 12.33, 12.82, 14.33])
# predicted background
BKG = np.array([527.164624, 315.423689, 349.644825, 186.21197, 261.441799,
                195.534193, 203.008745, 165.664396, 118.581365, 143.989367, 201.450357])

EXCESS = DATA_N - BKG
INV2 = 1.0 / DATA_ERR ** 2          # stat-only weights; see errors() for the alternatives


def errors(mode="stat"):
    """Per-bin uncertainty on the excess.

    'stat'  statistical only -- what every script did historically.
    'quad'  stat + MiniBooNE background systematics in quadrature. This is what
            Dutta-Kim state they did ("statistical and systematic uncertainties
            added by quadrature", arXiv:2110.11944 p.3), and it is 2.2x-3.5x
            larger per bin, i.e. 5-12x less chi2 weight.
    """
    if mode == "stat":
        return DATA_ERR.copy()
    if mode == "quad":
        import miniboone_systematics as MS
        return np.hypot(DATA_ERR, MS.SYS_ABS)
    raise ValueError("unknown error mode %r" % mode)


# --- SUPERSEDED 19-bin digitization, for reproducing old results only ----------
LEGACY_DATA_N = np.array([302, 402, 333, 279, 189, 168, 134, 118, 81, 83, 75, 84,
                          57, 61, 38, 52, 26, 19, 19], float)
LEGACY_DATA_E = np.array([225, 275, 325, 375, 425, 475, 525, 575, 625, 675, 725,
                          775, 825, 875, 925, 975, 1025, 1075, 1125], float)
LEGACY_DATA_ERR = np.array([36, 41, 38, 35, 28, 27, 25, 23, 18, 19, 18, 19, 15, 18,
                            13, 15, 11, 12, 10], float)
LEGACY_BKG = np.array([255, 320, 300, 250, 175, 150, 120, 105, 76, 78, 70, 76, 52,
                       55, 35, 47, 24, 18, 17], float)
LEGACY_EBINS = np.concatenate([[LEGACY_DATA_E[0] - 25], LEGACY_DATA_E + 25]) / 1e3

if __name__ == "__main__":
    print("official : %d bins, %.0f-%.0f MeV, excess %.1f +/- %.1f (stat)"
          % (len(DATA_N), EBINS[0] * 1e3, EBINS[-1] * 1e3, EXCESS.sum(),
             np.sqrt((DATA_ERR ** 2).sum())))
    print("           with bkg systematics in quadrature: +/- %.1f"
          % np.sqrt((errors("quad") ** 2).sum()))
    print("legacy   : %d bins, %.0f-%.0f MeV, excess %.1f  (SUPERSEDED)"
          % (len(LEGACY_DATA_N), LEGACY_EBINS[0] * 1e3, LEGACY_EBINS[-1] * 1e3,
             (LEGACY_DATA_N - LEGACY_BKG).sum()))
