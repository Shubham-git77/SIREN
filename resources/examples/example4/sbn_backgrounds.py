"""SBND and ICARUS nu_e charged-current background predictions.

SOURCE: SBN Proposal, arXiv:1503.01520, TABLE IX (p. I-43) -- "Event rates in the
nu_e charged-current candidate sample in the range 200-3000 MeV reconstructed
neutrino energy for 6.6e20 protons on target in LAr1-ND and the ICARUS-T600".

Two things make this directly usable here:
  * the exposure is 6.6e20 POT, exactly SBND_POT / ICARUS_BNB_POT in
    sbn_exposures.py, so no rescaling is needed;
  * the range 200-3000 MeV is the same range as the MiniBooNE release in
    miniboone_data.py, so signal/background comparisons are like for like.

LAr1-ND is the detector that became SBND.

WHAT THIS IS NOT: a spectrum. TABLE IX gives TOTALS per background category, not
a binned distribution, so it cannot by itself produce a Fig.2-style stacked plot
-- that needs the shape, which lives in the proposal's Figures 21/22 and would
have to be digitised. Use these for normalisation, signal-to-background and
significance statements, not for drawing a background histogram.

CAVEATS:
  * Cosmogenic rates are quoted with, and in parentheses without, a 95%-efficient
    time-based ID system. Both are kept below; COSMIC_TAGGED is the realistic one.
  * The proposal notes an additional 1.5% (LAr1-ND) / 3% (ICARUS) reduction of all
    beam-related categories from veto cuts, "not shown for clarity". Not applied
    here, so these totals are ~2-3% conservative.
  * These are the collaborations' pre-data PREDICTIONS, not measurements. Unlike
    the MiniBooNE background in miniboone_data.py, nothing has been observed yet.
"""
import numpy as np

POT = 6.6e20            # matches sbn_exposures.SBND_POT / ICARUS_BNB_POT
E_RANGE_MEV = (200.0, 3000.0)

# TABLE IX, event counts
SBND = {
    "mu -> nu_e":      6712.0,
    "K+ -> nu_e":      7333.0,
    "K0 -> nu_e":      1786.0,
    "NC pi0 -> gamma gamma": 1356.0,
    "NC Delta -> gamma":       87.0,
    "nu_mu CC":         484.0,
    "dirt":              44.0,
}
ICARUS = {
    "mu -> nu_e":       607.0,
    "K+ -> nu_e":       706.0,
    "K0 -> nu_e":       180.0,
    "NC pi0 -> gamma gamma":  149.0,
    "NC Delta -> gamma":        9.0,
    "nu_mu CC":          51.0,
    "dirt":              67.0,
}
COSMIC = {"SBND": 170.0, "ICARUS": 204.0}          # no time-based ID
COSMIC_TAGGED = {"SBND": 9.0, "ICARUS": 10.0}      # with 95%-efficient ID


def total(detector, cosmics="tagged"):
    """Total nu_e CC candidate background over 200-3000 MeV at 6.6e20 POT."""
    d = {"SBND": SBND, "ICARUS": ICARUS}[detector]
    c = {"tagged": COSMIC_TAGGED, "untagged": COSMIC, "none": {detector: 0.0}}[cosmics]
    return sum(d.values()) + c[detector]


def beam_related(detector):
    """Beam-related only -- the part a beam-coincident signal competes with."""
    return sum({"SBND": SBND, "ICARUS": ICARUS}[detector].values())


if __name__ == "__main__":
    for det in ("SBND", "ICARUS"):
        print("%-7s beam-related %8.0f | +cosmics(tagged) %8.0f | +cosmics(raw) %8.0f"
              % (det, beam_related(det), total(det), total(det, "untagged")))
    print("\nintrinsic nu_e (mu + K+ + K0) fraction of beam-related:")
    for det, d in (("SBND", SBND), ("ICARUS", ICARUS)):
        intr = d["mu -> nu_e"] + d["K+ -> nu_e"] + d["K0 -> nu_e"]
        print("  %-7s %.0f / %.0f = %.1f%%" % (det, intr, beam_related(det),
                                               100 * intr / beam_related(det)))
