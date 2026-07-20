"""
Read parent meson kinematics from dk2nu ROOT files.

dk2nu is a standardized format for neutrino flux simulation output
(see https://github.com/NuSoftHEP/dk2nu).  Each entry describes a
neutrino produced in a meson decay, recording the parent meson's
identity, momentum, decay vertex, and importance weight.

This module extracts the parent meson information and provides it
in forms suitable for SIREN flux construction:

    - As numpy arrays for direct use in flux calculations
    - As a TabulatedFluxDistribution for the SIREN injector
    - As energy spectra binned by parent species
"""

import math
import numpy as np


# PDG codes for parent mesons
PTYPE_PIPLUS = 211
PTYPE_PIMINUS = -211
PTYPE_KPLUS = 321
PTYPE_KMINUS = -321
PTYPE_K0L = 130

_PARENT_NAMES = {
    PTYPE_PIPLUS: "pi+",
    PTYPE_PIMINUS: "pi-",
    PTYPE_KPLUS: "K+",
    PTYPE_KMINUS: "K-",
    PTYPE_K0L: "K0L",
    13: "mu-",
    -13: "mu+",
}

# dk2nu branch paths (nested under dk2nu/ in NuMI files, flat in BNB)
_BRANCH_SETS = [
    {
        "ptype": "dk2nu/decay/decay.ptype",
        "pdpx": "dk2nu/decay/decay.pdpx",
        "pdpy": "dk2nu/decay/decay.pdpy",
        "pdpz": "dk2nu/decay/decay.pdpz",
        "ppenergy": "dk2nu/decay/decay.ppenergy",
        "vx": "dk2nu/decay/decay.vx",
        "vy": "dk2nu/decay/decay.vy",
        "vz": "dk2nu/decay/decay.vz",
        "nimpwt": "dk2nu/decay/decay.nimpwt",
        "ntype": "dk2nu/decay/decay.ntype",
    },
    {
        "ptype": "decay.ptype",
        "pdpx": "decay.pdpx",
        "pdpy": "decay.pdpy",
        "pdpz": "decay.pdpz",
        "ppenergy": "decay.ppenergy",
        "vx": "decay.vx",
        "vy": "decay.vy",
        "vz": "decay.vz",
        "nimpwt": "decay.nimpwt",
        "ntype": "decay.ntype",
    },
]


def _detect_branches(tree):
    """Auto-detect which branch naming convention the file uses."""
    keys = set(tree.keys())
    for bset in _BRANCH_SETS:
        if all(v in keys for v in bset.values()):
            return bset
    raise ValueError(
        "Could not find dk2nu decay branches. "
        f"Available branches: {sorted(keys)[:20]}..."
    )


def read_dk2nu(
    filenames,
    parent_pdg=None,
    entry_start=None,
    entry_stop=None,
):
    """
    Read parent meson kinematics from one or more dk2nu ROOT files.

    Parameters
    ----------
    filenames : str or list of str
        Path(s) to dk2nu ROOT files.
    parent_pdg : int or list of int, optional
        Filter to specific parent PDG code(s).  Default: all parents.
    entry_start, entry_stop : int, optional
        Limit the number of entries read (per file).

    Returns
    -------
    dict with keys:
        ptype      : int array, parent PDG code
        E          : float array, parent energy [GeV]
        px, py, pz : float arrays, parent momentum components [GeV]
        vx, vy, vz : float arrays, decay vertex [cm]
        nimpwt     : float array, importance weight
        ntype      : int array, neutrino PDG code
        pot        : float, total simulated POT across all files
    """
    try:
        import uproot
    except ImportError:
        raise ImportError(
            "uproot is required to read dk2nu files: pip install uproot"
        )

    if isinstance(filenames, str):
        filenames = [filenames]
    if parent_pdg is not None and not hasattr(parent_pdg, "__iter__"):
        parent_pdg = [parent_pdg]

    all_data = {k: [] for k in [
        "ptype", "E", "px", "py", "pz", "vx", "vy", "vz", "nimpwt", "ntype"
    ]}
    total_pot = 0.0

    for fname in filenames:
        f = uproot.open(fname)
        if "dk2nuTree" not in [k.split(";")[0] for k in f.keys()]:
            continue
        tree = f["dk2nuTree"]
        branches = _detect_branches(tree)

        kw = {}
        if entry_start is not None:
            kw["entry_start"] = entry_start
        if entry_stop is not None:
            kw["entry_stop"] = entry_stop

        data = tree.arrays(
            list(branches.values()),
            library="np",
            **kw,
        )

        ptype = data[branches["ptype"]].astype(int)
        E = data[branches["ppenergy"]]
        px = data[branches["pdpx"]]
        py = data[branches["pdpy"]]
        pz = data[branches["pdpz"]]
        vx = data[branches["vx"]]
        vy = data[branches["vy"]]
        vz = data[branches["vz"]]
        nimpwt = data[branches["nimpwt"]]
        ntype = data[branches["ntype"]].astype(int)

        if parent_pdg is not None:
            mask = np.isin(ptype, parent_pdg)
            ptype = ptype[mask]
            E = E[mask]
            px = px[mask]
            py = py[mask]
            pz = pz[mask]
            vx = vx[mask]
            vy = vy[mask]
            vz = vz[mask]
            nimpwt = nimpwt[mask]
            ntype = ntype[mask]

        all_data["ptype"].append(ptype)
        all_data["E"].append(E)
        all_data["px"].append(px)
        all_data["py"].append(py)
        all_data["pz"].append(pz)
        all_data["vx"].append(vx)
        all_data["vy"].append(vy)
        all_data["vz"].append(vz)
        all_data["nimpwt"].append(nimpwt)
        all_data["ntype"].append(ntype)

        # uproot method to get total POT from dkmetaTree (if available)
        if "dkmetaTree" in f:
            meta_tree = f["dkmetaTree"]
            print(list(meta_tree.keys()))
            if "dkmeta/pots" in meta_tree.keys():
                pots = meta_tree["dkmeta/pots"].array(library="np")
                if len(pots) > 0:
                    pots = pots[0]
                else:
                    pots = 0.0
                total_pot += pots

    result = {k: np.concatenate(v) for k, v in all_data.items()}
    result["pot"] = total_pot
    return result


def meson_energy_spectrum(
    dk2nu_data,
    parent_pdg,
    n_bins=100,
    E_range=None,
):
    """
    Build a weighted energy spectrum of a given parent meson from dk2nu data.

    Parameters
    ----------
    dk2nu_data : dict
        Output of read_dk2nu().
    parent_pdg : int
        PDG code of the parent meson to histogram.
    n_bins : int
        Number of energy bins.
    E_range : tuple of (float, float), optional
        Energy range in GeV.  Default: auto from data.

    Returns
    -------
    E_centers : array, bin centers [GeV]
    dN_dE     : array, weighted counts per GeV per POT
    """
    mask = dk2nu_data["ptype"] == parent_pdg
    E = dk2nu_data["E"][mask]
    w = dk2nu_data["nimpwt"][mask]
    pot = dk2nu_data["pot"]

    if E_range is None:
        E_range = (E.min() * 0.9, E.max() * 1.1) if len(E) > 0 else (0.0, 5.0)

    counts, bin_edges = np.histogram(E, bins=n_bins, range=E_range, weights=w)
    dE = bin_edges[1] - bin_edges[0]
    E_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    if pot > 0:
        dN_dE = counts / (dE * pot)
    else:
        dN_dE = counts / dE

    return E_centers, dN_dE


def dk2nu_to_tabulated_flux(
    dk2nu_data,
    parent_pdg,
    n_bins=100,
    E_range=None,
    physically_normalized=True,
):
    """
    Convert dk2nu parent meson data into a SIREN TabulatedFluxDistribution
    (energy spectrum of the parent meson, per POT).

    Parameters
    ----------
    dk2nu_data : dict
        Output of read_dk2nu().
    parent_pdg : int
        PDG code of the parent meson.
    n_bins : int
        Number of energy bins.
    E_range : tuple, optional
        Energy range in GeV.
    physically_normalized : bool
        Whether the flux is physically normalized.

    Returns
    -------
    siren.distributions.TabulatedFluxDistribution
    """
    import siren

    E_centers, dN_dE = meson_energy_spectrum(
        dk2nu_data, parent_pdg, n_bins=n_bins, E_range=E_range
    )

    return siren.distributions.TabulatedFluxDistribution(
        float(E_centers[0]),
        float(E_centers[-1]),
        list(E_centers),
        list(dN_dE),
        physically_normalized,
    )


def dk2nu_to_primary_distribution(
    dk2nu_data,
    detector_model,
    parent_pdg=None,
    sampling_bias=None,
    flux_weighted_sampling=False,
    beam_transform=None,
):
    """
    Build a PrimaryExternalDistribution directly from dk2nu data.

    Converts positions from BNB (geometry) coordinates to detector-local
    coordinates using the detector model's ToDet transform, and from cm
    to meters.  No intermediate CSV file is needed.

    Parameters
    ----------
    dk2nu_data : dict
        Output of read_dk2nu().
    detector_model : siren.detector.DetectorModel
        Detector model (provides the geometry-to-detector transform).
    parent_pdg : int or list of int, optional
        Filter to specific parent PDG code(s).
    sampling_bias : callable, optional
        Function f(E, px, py, pz, vx, vy, vz) -> weight that computes
        per-entry sampling weights from the pion kinematics (in the raw
        dk2nu beam frame, before any transform).  Entries are selected
        with probability proportional to these weights.  The generation
        probability accounts for the bias so event weights remain correct.
        Arguments are numpy arrays; the return value should broadcast to
        the same length.  When None (default), uniform selection is used.
    beam_transform : Transform, optional
        Rigid transform (with .R rotation matrix and .t translation in
        METERS) from the dk2nu beam frame to the SIREN geometry (world)
        frame.  Required for flux files whose coordinates are not already
        in the geometry frame — e.g. NuMI g4numi dk2nu files, whose
        vertices/momenta are in NuMI beam coordinates (origin at MCZERO,
        z along the NuMI axis).  Use
        sbn_geometry.transform("NuMI", "BNB") for NuMI files with the
        SBN detector models (world frame = BNB).  Positions are converted
        cm -> m first, then r_geo = R @ r_beam + t; momenta are rotated
        p_geo = R @ p_beam.  When None (default), dk2nu coordinates are
        assumed to already be in the geometry frame (correct for BNB
        G4BNB files).

    Returns
    -------
    siren.distributions.PrimaryExternalDistribution
    """
    import siren
    from siren.detector import GeometryPosition, GeometryDirection
    from siren.math import Vector3D

    ptype = dk2nu_data["ptype"]
    if parent_pdg is not None:
        if not hasattr(parent_pdg, "__iter__"):
            parent_pdg = [parent_pdg]
        mask = np.isin(ptype, parent_pdg)
    else:
        mask = np.ones(len(ptype), dtype=bool)

    simulated_pot = dk2nu_data["pot"]

    E = dk2nu_data["E"][mask]
    px = dk2nu_data["px"][mask]
    py = dk2nu_data["py"][mask]
    pz = dk2nu_data["pz"][mask]
    vx = dk2nu_data["vx"][mask]
    vy = dk2nu_data["vy"][mask]
    vz = dk2nu_data["vz"][mask]
    nimpwt = dk2nu_data["nimpwt"][mask]
    pt = ptype[mask]

    weight = nimpwt / simulated_pot

    # Beam frame -> geometry (world) frame.  dk2nu positions are in cm;
    # beam_transform.t is in meters, so convert units first.
    if beam_transform is not None:
        R = np.asarray(beam_transform.R, dtype=float)
        t = np.asarray(beam_transform.t, dtype=float)
        pos_geo = R @ np.stack([vx, vy, vz]) * 0.01 + t[:, None]
        gx, gy, gz = pos_geo
        mom_geo = R @ np.stack([px, py, pz])
        gpx, gpy, gpz = mom_geo
    else:
        gx, gy, gz = vx * 0.01, vy * 0.01, vz * 0.01
        gpx, gpy, gpz = px, py, pz

    mass_map = {
        211: 0.13957039, -211: 0.13957039,
        321: 0.49368,    -321: 0.49368,
        130: 0.49761,
        13: 0.10566,     -13: 0.10566,
    }

    keys = ["E", "px", "py", "pz", "x", "y", "z", "m", "weight"]
    data = []
    for i in range(len(E)):
        # Convert position from geometry (BNB) to detector coordinates
        geo_pos = GeometryPosition(Vector3D(gx[i], gy[i], gz[i]))
        det_pos = detector_model.GeoPositionToDetPosition(geo_pos).get()

        # Convert momentum direction from geometry to detector coordinates.
        # Energy is a scalar and is unchanged; the 3-momentum direction
        # must be rotated if the detector axes differ from geometry axes.
        p_mag = math.sqrt(float(gpx[i])**2 + float(gpy[i])**2 + float(gpz[i])**2)
        if p_mag > 0:
            geo_dir = GeometryDirection(Vector3D(
                float(gpx[i]) / p_mag, float(gpy[i]) / p_mag, float(gpz[i]) / p_mag))
            det_dir = detector_model.GeoDirectionToDetDirection(geo_dir).get()
            px_det = det_dir.GetX() * p_mag
            py_det = det_dir.GetY() * p_mag
            pz_det = det_dir.GetZ() * p_mag
        else:
            px_det = py_det = pz_det = 0.0

        m = mass_map.get(int(pt[i]), 0.13957)
        # dk2nu stores momentum at decay (pdpx/pdpy/pdpz) but energy
        # at production (ppenergy). Compute on-shell energy from the
        # decay-point momentum and known mass.
        E_decay = math.sqrt(p_mag * p_mag + m * m)
        data.append([
            E_decay, px_det, py_det, pz_det,
            det_pos.GetX(), det_pos.GetY(), det_pos.GetZ(),
            m, float(weight[i]),
        ])

    if sampling_bias is not None or flux_weighted_sampling:
        if sampling_bias is not None:
            sw = np.asarray(
                sampling_bias(E, px, py, pz, vx, vy, vz),
                dtype=float,
            )
        else:
            sw = np.ones(len(E), dtype=float)
        np.maximum(sw, 0.0, out=sw)
        # Fold the physical flux weight (nimpwt/POT) into the sampling weight so
        # entries are drawn PROPORTIONAL TO THEIR PHYSICAL RATE. Without this the
        # importance weights inherit the full nimpwt dynamic range (~2e3x) and the
        # sampling_bias (e.g. E^1.5) can divide it into a runaway heavy tail. The
        # generation probability still accounts for sw, so event weights stay
        # correct; this only flattens the realized weight distribution.
        if flux_weighted_sampling:
            sw = sw * np.asarray(weight, dtype=float)
        return siren.distributions.PrimaryExternalDistribution(
            keys, data, sw.tolist()
        )

    return siren.distributions.PrimaryExternalDistribution(keys, data)


def dk2nu_to_csv(
    dk2nu_data,
    output_path,
    parent_pdg=None,
    position_transform=None,
    units_cm=True,
    beam_transform=None,
):
    """
    Write dk2nu parent meson kinematics to a CSV file suitable for
    SIREN's PrimaryExternalDistribution.

    The CSV has columns: E, px, py, pz, x0, y0, z0, m, nimpwt

    Parameters
    ----------
    dk2nu_data : dict
        Output of read_dk2nu().
    output_path : str
        Path to write the CSV file.
    parent_pdg : int or list of int, optional
        Filter to specific parent PDG code(s).  Default: use all entries
        in dk2nu_data (which may already be filtered).
    position_transform : callable, optional
        Function that takes (vx, vy, vz) arrays in dk2nu coordinates
        and returns (x0, y0, z0) arrays in detector coordinates.
        dk2nu positions are in cm.  If None, positions are used as-is.
        Applied after beam_transform (if both are given).
    units_cm : bool
        If True (default), positions in the CSV are in cm.
        If False, positions are converted to meters.
    beam_transform : Transform, optional
        Rigid transform (with .R rotation matrix and .t translation in
        METERS) from the dk2nu beam frame to the SIREN geometry (world)
        frame — e.g. sbn_geometry.transform("NuMI", "BNB") for NuMI
        g4numi files.  Positions get r_geo = R @ r_beam + t (translation
        applied in cm internally); momenta get p_geo = R @ p_beam.
        When None (default), dk2nu coordinates are assumed to already be
        in the geometry frame (correct for BNB G4BNB files).

    Returns
    -------
    int
        Number of rows written.
    """
    ptype = dk2nu_data["ptype"]
    if parent_pdg is not None:
        if not hasattr(parent_pdg, "__iter__"):
            parent_pdg = [parent_pdg]
        mask = np.isin(ptype, parent_pdg)
    else:
        mask = np.ones(len(ptype), dtype=bool)

    E = dk2nu_data["E"][mask]
    px = dk2nu_data["px"][mask]
    py = dk2nu_data["py"][mask]
    pz = dk2nu_data["pz"][mask]
    vx = dk2nu_data["vx"][mask]
    vy = dk2nu_data["vy"][mask]
    vz = dk2nu_data["vz"][mask]
    nimpwt = dk2nu_data["nimpwt"][mask]
    pt = ptype[mask]

    # Beam frame -> geometry (world) frame.  Positions here are in cm,
    # so the transform translation (meters) is scaled to cm.
    if beam_transform is not None:
        R = np.asarray(beam_transform.R, dtype=float)
        t = np.asarray(beam_transform.t, dtype=float)
        vx, vy, vz = R @ np.stack([vx, vy, vz]) + t[:, None] * 100.0
        px, py, pz = R @ np.stack([px, py, pz])

    if position_transform is not None:
        vx, vy, vz = position_transform(vx, vy, vz)

    scale = 1.0 if units_cm else 0.01

    mass_map = {
        211: 0.13957039, -211: 0.13957039,
        321: 0.49368,    -321: 0.49368,
        130: 0.49761,
        13: 0.10566,     -13: 0.10566,
    }

    with open(output_path, "w") as f:
        f.write("E,px,py,pz,x,y,z,m,nimpwt\n")
        for i in range(len(E)):
            m = mass_map.get(int(pt[i]), 0.13957)
            f.write(
                f"{E[i]:.8e},{px[i]:.8e},{py[i]:.8e},{pz[i]:.8e},"
                f"{vx[i]*scale:.8e},{vy[i]*scale:.8e},{vz[i]*scale:.8e},"
                f"{m:.8e},{nimpwt[i]:.8e}\n"
            )

    return int(np.sum(mask))


def print_summary(dk2nu_data):
    """Print a summary of the dk2nu data."""
    ptypes = dk2nu_data["ptype"]
    unique, counts = np.unique(ptypes, return_counts=True)
    n_total = len(ptypes)
    pot = dk2nu_data["pot"]

    print(f"Total entries: {n_total}")
    print(f"Total POT: {pot:.3e}")
    print(f"Parent breakdown:")
    for pdg, count in sorted(zip(unique, counts), key=lambda x: -x[1]):
        name = _PARENT_NAMES.get(int(pdg), str(int(pdg)))
        frac = 100.0 * count / n_total if n_total > 0 else 0
        E = dk2nu_data["E"][ptypes == pdg]
        print(f"  {name:>5s} ({int(pdg):>4d}): {count:>7d} ({frac:5.1f}%)  "
              f"E = [{E.min():.3f}, {E.max():.3f}] GeV")


_ANALYTIC_MESON_CACHE = {}


def analytic_meson_source(files, pdg, beam_transform=None, n_max=120000, seed=42):
    """Parent-meson kinematics for the analytic sigma*N*chord engine.

    Reads one or more dk2nu files (POT summed, cached) and returns the tuple
    (E[GeV], |p|[GeV], dir[N,3], vertex[N,3] in METRES, w) the analytic engine
    consumes, where w = nimpwt / POT_simulated (mesons per POT).

    If beam_transform is given -- a rigid Transform with .R (3x3) and .t (metres),
    e.g. sbn_geometry.transform("NuMI", "BNB") -- the decay vertices (cm->m) and
    parent momenta are mapped into the SIREN world (BNB) frame:
    r_world = R r + t, p_world = R p. Needed for g4numi NuMI files (NuMI beam
    coordinates); omit for BNB g4bnb files (already in the world frame).

    Large species are sub-sampled to n_max parents with an unbiased M/n weight
    rescale so the per-POT normalization is preserved.
    """
    key = tuple(files) if isinstance(files, (list, tuple)) else (files,)
    if key not in _ANALYTIC_MESON_CACHE:
        _ANALYTIC_MESON_CACHE[key] = read_dk2nu(list(key))
    d = _ANALYTIC_MESON_CACHE[key]
    pot = float(d["pot"])
    isp = d["ptype"] == pdg
    E = d["E"][isp]
    px, py, pz = d["px"][isp], d["py"][isp], d["pz"][isp]
    vx, vy, vz = d["vx"][isp], d["vy"][isp], d["vz"][isp]
    w = d["nimpwt"][isp] / pot
    M = len(E)
    if M > n_max:
        idx = np.random.default_rng(seed).choice(M, n_max, replace=False)
        E, px, py, pz, vx, vy, vz = (a[idx] for a in (E, px, py, pz, vx, vy, vz))
        w = w[idx] * (M / n_max)
    if beam_transform is not None:
        R = np.asarray(beam_transform.R, dtype=float)
        t = np.asarray(beam_transform.t, dtype=float)
        pos = R @ (np.stack([vx, vy, vz]) * 0.01) + t[:, None]   # (3,N) m, world
        mom = R @ np.stack([px, py, pz])                         # (3,N) GeV, world
        v, p = pos.T, mom.T
    else:
        v = np.stack([vx, vy, vz], axis=1) * 0.01
        p = np.stack([px, py, pz], axis=1)
    pmag = np.linalg.norm(p, axis=1)
    ok = (pmag > 0) & np.isfinite(E)
    return E[ok], pmag[ok], p[ok] / pmag[ok, None], v[ok], w[ok]
