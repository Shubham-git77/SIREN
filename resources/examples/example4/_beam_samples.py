"""Merge split decay-in-flight / decay-at-rest dk2nu productions.

Analysis-level helper for this repo's SBND dk2nu upscattering-CSV pipeline
(dk2nu_to_upscattering_csv_difdar.py). Ported from the SIREN Dutta-Kim branch
PR #178 `_beam_samples.py` and adapted to OUR NEUTRINO-level sample dicts
(per-neutrino arrays plus the parent species/momentum needed to classify
decay-in-flight vs decay-at-rest by parent kinetic energy).

The standard g4numi production kills every particle other than a neutrino once
its kinetic energy falls below KillTrackingThreshold (0.05 GeV), so its files
contain no decays from parents below that energy. A dedicated decay-at-rest
production tracks everything to rest and records decays at all parent energies
-- the two samples overlap above the threshold, and they carry different
protons-on-target (POT). The merge here makes the split exact and the
normalization consistent:

- decay-in-flight rows are kept where the parent kinetic energy is >= the cut;
- decay-at-rest rows are kept where it is < the cut;
- decay-at-rest importance weights are rescaled by the POT ratio, so the merged
  rows behave exactly like a single sample whose POT is the decay-in-flight POT.

Downstream consumers that form per-POT weights as nimpwt/pot therefore need no
changes: the merged dict already carries the rescaled weights and pot = dif_pot.

A "sample dict" has these keys (all numpy arrays of equal length N, plus pot):
    prod        (N,3) neutrino production vertex [m, BNB frame]
    p           (N,3) neutrino momentum at the detector ray [GeV]
    E           (N,)  neutrino energy [GeV]
    wgt         (N,)  flux weight (nuray.wgt * decay.nimpwt)
    ntype       (N,)  neutrino PDG
    ptype       (N,)  PARENT PDG (for the KE classification)
    parent_pmag (N,)  |parent momentum at decay| [GeV] (for the KE classification)
    pot         float simulated protons-on-target for the sample
"""

import numpy as np

# Parent rest masses [GeV] for the kinetic-energy classification.
PARENT_MASSES = {
    13: 0.1056583755,     # mu
    211: 0.13957039,      # pi+-
    321: 0.493677,        # K+-
    130: 0.497611,        # K0L
    310: 0.497611,        # K0S
    2112: 0.9395654205,   # n
    2212: 0.9382720882,   # p
}

# Per-neutrino arrays carried through the merge (wgt handled separately so the
# decay-at-rest side can be POT-rescaled).
_ARRAY_KEYS = ("prod", "p", "E", "ntype", "ptype", "parent_pmag")


def parent_kinetic_energy(data):
    """Parent kinetic energy at decay [GeV] per row: sqrt(|p|^2 + m^2) - m.

    Uses data["ptype"] (parent PDG) and data["parent_pmag"] (|parent momentum
    at decay|). Raises KeyError for a parent species without a rest mass.
    """
    codes = np.abs(np.asarray(data["ptype"], dtype=np.int64))
    masses = np.empty(codes.shape, dtype=float)
    for code in np.unique(codes):
        if int(code) not in PARENT_MASSES:
            raise KeyError(
                "No rest mass on record for dk2nu parent pdg %d; extend "
                "PARENT_MASSES to classify it" % int(code))
        masses[codes == code] = PARENT_MASSES[int(code)]
    pmag = np.asarray(data["parent_pmag"], dtype=float)
    return np.sqrt(pmag ** 2 + masses ** 2) - masses


def combine_dif_dar(dif_data, dar_data, kinetic_energy_cut=0.05):
    """Merge a decay-in-flight and a decay-at-rest neutrino sample dict.

    Returns a merged sample dict (same array keys) plus:
        dar     : bool array, True for rows from the decay-at-rest sample
        pot     : the decay-in-flight POT (the merged set's normalization)
        dif_pot, dar_pot : the raw per-sample POT
    """
    dif_pot = float(dif_data.get("pot", 0.0))
    dar_pot = float(dar_data.get("pot", 0.0))
    if not (dif_pot > 0.0) or not (dar_pot > 0.0):
        raise ValueError(
            "combine_dif_dar requires a positive POT in both samples "
            "(got dif_pot=%r, dar_pot=%r); the input files carried no usable "
            "dkmetaTree/pots metadata" % (dif_pot, dar_pot))

    dif_keep = parent_kinetic_energy(dif_data) >= kinetic_energy_cut
    dar_keep = parent_kinetic_energy(dar_data) < kinetic_energy_cut

    n_below = int(np.count_nonzero(~dif_keep))
    if n_below > 0:
        print("combine_dif_dar: dropped %d decay-in-flight rows below the "
              "%.3g GeV cut (the decay-at-rest sample covers them)"
              % (n_below, kinetic_energy_cut))

    merged = {}
    for key in _ARRAY_KEYS:
        merged[key] = np.concatenate(
            [np.asarray(dif_data[key])[dif_keep],
             np.asarray(dar_data[key])[dar_keep]])
    # Rescale the decay-at-rest flux weights by the POT ratio so the merged
    # sample is normalized consistently to the decay-in-flight POT.
    merged["wgt"] = np.concatenate(
        [np.asarray(dif_data["wgt"])[dif_keep],
         np.asarray(dar_data["wgt"])[dar_keep] * (dif_pot / dar_pot)])
    merged["dar"] = np.concatenate(
        [np.zeros(int(np.count_nonzero(dif_keep)), dtype=bool),
         np.ones(int(np.count_nonzero(dar_keep)), dtype=bool)])
    merged["pot"] = dif_pot
    merged["dif_pot"] = dif_pot
    merged["dar_pot"] = dar_pot
    return merged
