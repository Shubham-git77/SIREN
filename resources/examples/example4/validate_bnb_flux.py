"""
Stage 3 validation: decay the BNBFlux meson sample to numu and compare the
predicted spectrum at MiniBooNE to SIREN's shipped BNB_FHC.dat. Scan the horn
kick PT_KICK for the best SHAPE match (chi2 on area-normalized spectra), and
report the global normalization factor that maps our arb-unit flux onto the
official nu/m^2/GeV/POT units. That normalization + PT_KICK calibrate the sample.
"""
import os
import numpy as np
from siren import _util

_BNB = _util.load_module(
    "BNBFlux", os.path.join(_util.resource_package_dir(),
                            "processes", "DarkNewsTables", "BNBFlux.py"))
_BNBFLUX = _util.load_module(
    "BNBflux_ref", os.path.join(_util.resource_package_dir(),
                                "fluxes", "BNB", "BNB-v1.0", "flux.py"))

# shipped reference
_BNBFLUX.fetch_data()
DAT = _BNB.resolve = os.path.join(_BNBFLUX._get_abs_dir(), "BNB_FHC.dat")
E_ref, F_ref = _BNB.load_bnb_reference_numu(DAT)
# reference flux.py converts the raw column by /50*1000*1e4 -> nu/m^2/GeV/POT
F_ref_phys = F_ref / 50.0 * 1000.0 * 1e4

# histogram binning matched to the reference (0..7 GeV, 0.05 GeV bins)
BINS = np.arange(0.0, 7.001, 0.05)
CENT = 0.5 * (BINS[:-1] + BINS[1:])


def predicted_spectrum(pt_kick, n=300000, seed=1):
    data = _BNB.generate_bnb_sample(n_per_species=n, pt_kick=pt_kick, seed=seed)
    E_nu, w = _BNB.compute_numu_flux(data)
    h, _ = np.histogram(E_nu, bins=BINS, weights=w)
    h = h / np.diff(BINS)            # per-GeV
    return h


def shape_chi2(h_pred, F_ref_on_cent):
    # compare area-normalized shapes over the populated reference range
    mask = (CENT >= 0.1) & (CENT <= 3.0) & (F_ref_on_cent > 0)
    a = h_pred[mask] / h_pred[mask].sum()
    b = F_ref_on_cent[mask] / F_ref_on_cent[mask].sum()
    return np.sum((a - b) ** 2 / (b + 1e-12))


def main():
    # interpolate reference onto our bin centers
    F_ref_c = np.interp(CENT, E_ref, F_ref_phys, left=0, right=0)

    print("Stage 3: BNB numu flux validation vs BNB_FHC.dat")
    print("scanning horn kick PT_KICK ...")
    best = None
    for pk in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        h = predicted_spectrum(pk)
        c2 = shape_chi2(h, F_ref_c)
        # peak position comparison
        epk_pred = CENT[np.argmax(h)]
        print("  PT_KICK=%.2f GeV/c : shape-chi2=%.4f  peak@%.2f GeV" % (pk, c2, epk_pred))
        if best is None or c2 < best[1]:
            best = (pk, c2, h)
    pk, c2, h = best
    epk_ref = CENT[np.argmax(F_ref_c)]
    print("\nBEST PT_KICK=%.2f GeV/c (shape-chi2=%.4f)" % (pk, c2))
    print("  predicted peak @ %.2f GeV ; reference peak @ %.2f GeV" % (CENT[np.argmax(h)], epk_ref))
    # global normalization to physical units (fit over 0.1-3 GeV)
    m = (CENT >= 0.1) & (CENT <= 3.0) & (F_ref_c > 0)
    norm = np.sum(h[m] * F_ref_c[m]) / np.sum(h[m] * h[m])
    print("  global norm (arb -> nu/m^2/GeV/POT) = %.4e" % norm)
    print("  integral check: pred=%.3e  ref=%.3e nu/m^2/POT"
          % (np.sum(h * norm * np.diff(BINS)), np.sum(F_ref_c * np.diff(BINS))))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].step(CENT, h * norm, where="mid", label="BNBFlux (this work)")
    ax[0].step(CENT, F_ref_c, where="mid", label="BNB_FHC.dat (SIREN)")
    ax[0].set_xlabel(r"$E_\nu$ [GeV]"); ax[0].set_ylabel(r"$\nu_\mu$ flux [/m$^2$/GeV/POT]")
    ax[0].set_xlim(0, 3.5); ax[0].legend(); ax[0].set_title("BNB $\\nu_\\mu$ flux (PT_KICK=%.2f)" % pk)
    ax[1].step(CENT, h / h.sum(), where="mid", label="BNBFlux (shape)")
    ax[1].step(CENT, F_ref_c / F_ref_c.sum(), where="mid", label="BNB_FHC.dat (shape)")
    ax[1].set_xlabel(r"$E_\nu$ [GeV]"); ax[1].set_ylabel("area-normalized")
    ax[1].set_xlim(0, 3.5); ax[1].legend(); ax[1].set_title("shape comparison")
    os.makedirs("output", exist_ok=True)
    plt.tight_layout(); plt.savefig("output/BNB_numu_validation.png", dpi=130)
    print("  saved -> output/BNB_numu_validation.png")


if __name__ == "__main__":
    main()
