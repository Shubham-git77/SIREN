"""
Independent low-variance ANALYTIC estimator of the SBND scalar K_mu rate, to
settle the directed-vs-cone importance-sampler discrepancy.

Method (identical in spirit to the MiniBooNE fig2 channel_hist, no SIREN
injector): for each K+ from the BNB flux, sample many phi decays, boost to lab,
ray-trace the phi toward the SBND LAr TPC (axis-aligned box), and accumulate
   dN = flux * POT * BR * [sigma(E_phi) * N_Ar * chord] * eff(E_vis) / n_dec.
The phi rest-frame emission is isotropic (spin-0 parent); the E_phi marginal
carries the matrix-element dynamics. This avoids the directed/cone proposals
entirely, so it is the ground truth to calibrate them against.

Reference numbers to compare:
   directed sampler (ESF 0.3%, 30k ev): 1.15e6 events
   cone sampler     (ESF 97%, normalization proposal-dependent): 1e7-1e8
"""
import os, math, argparse
import numpy as np
from siren import _util

HERE = os.path.dirname(__file__)
_MODULES = {
    "scalar": "ScalarPortal_SBND_multichannel.py",
    "pseudo": "PseudoscalarPortal_SBND_multichannel.py",
}
# directed-sampler totals for comparison (factor-2-fixed; scalar=2.5k all-ch run,
# pseudo = Jun-22 run /2 for the production fix).
_DIRECTED_TOT = {"scalar": 2.494e6, "pseudo": 2.597e9 / 2.0}

# --- SBND geometry (BNB frame, metres); same for both portals ---
DET  = np.array([0.7378, 0.0, 110.0])            # SBND center (G4BNB location)
N_AR = 2.1036e22                                  # argon nuclei / cm^3 (rho=1.3954 g/cc)


def ray_box_chord(v, d, center, half):
    """Vectorized AABB slab intersection. v,d: (N,3). Returns chord length [m]
    of the forward ray (origin v, unit dir d) through the box, 0 if it misses."""
    lo = center - half; hi = center + half
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        t1 = (lo - v) * inv
        t2 = (hi - v) * inv
    tmin = np.minimum(t1, t2)
    tmax = np.maximum(t1, t2)
    tenter = np.nanmax(tmin, axis=1)
    texit  = np.nanmin(tmax, axis=1)
    hit = (tenter < texit) & (texit > 0)
    tenter_c = np.clip(tenter, 0.0, None)         # if origin inside, start at 0
    chord = np.where(hit, texit - tenter_c, 0.0)
    return np.maximum(chord, 0.0)


# vectorized detection efficiency (same table/threshold as the script)
_EE = np.array([0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.90])
_EF = np.array([0.089, 0.135, 0.139, 0.131, 0.123, 0.116, 0.106, 0.102])
def eff_vec(Earr):
    e = np.interp(Earr, _EE, _EF, left=_EF[0], right=_EF[-1])
    return np.where(Earr >= 0.140, e, 0.0)

def compute_channel(S, HALF, name, bnb, n_dec=600, seed=7):
    pdg, m_M, m_l, lpdg, nupdg, gsm = S.CHANNELS[name]
    if (m_M - m_l) <= S.M_PHI:
        return None
    ch = S.build_onshell_models(pdg, m_M, m_l, lpdg, nupdg)
    md = ch["meson_decay"]._decay
    dp = ch["models"]["primakoff"]._dp
    m_phi = md.m_phi
    br = ch["meson_decay"]._total_width / gsm

    isp = bnb["ptype"] == pdg
    p = np.stack([bnb["px"][isp], bnb["py"][isp], bnb["pz"][isp]], axis=1)
    v = np.stack([bnb["vx"][isp], bnb["vy"][isp], bnb["vz"][isp]], axis=1) / 100.0
    E = bnb["E"][isp]; w = bnb["nimpwt"][isp]
    pmag = np.linalg.norm(p, axis=1); dirK = p / pmag[:, None]
    nK = len(pmag)

    # E_phi (rest-frame) CDF from the validated matrix element
    Emax = (m_M**2 + m_phi**2 - m_l**2) / (2 * m_M)
    Eg = np.linspace(m_phi + 1e-5, Emax - 1e-5, 400)
    dN = np.array([max(md.differential_decay_rate([e])[0], 0.0) for e in Eg])
    dN = np.where(np.isfinite(dN) & (dN > 0), dN, 0.0)
    cdf = np.cumsum(dN); cdf /= cdf[-1]

    # sigma(E_phi) on argon (depends only on E_phi, not the lepton)
    Et = np.concatenate([np.linspace(0.001, 0.3, 120), np.linspace(0.31, 9, 160)])
    st = np.array([dp.total_xsec(float(e)) for e in Et])

    rng = np.random.default_rng(seed)
    arb = np.tile(np.array([0., 1., 0.]), (nK, 1))
    mm = np.abs(dirK[:, 1]) > 0.9; arb[mm] = np.array([1., 0., 0.])
    xc = np.cross(dirK, arb); xc /= np.linalg.norm(xc, axis=1)[:, None]
    yc = np.cross(dirK, xc)
    gK = E / m_M; bK = pmag / E
    prefm = w * S.SBND_POT * br

    total = 0.0; wsum2 = 0.0; ncontrib = 0; Ev_hit = []; w_hit = []
    for _ in range(n_dec):
        u = rng.random(nK); Es = np.interp(u, cdf, Eg)
        ps = np.sqrt(np.maximum(Es**2 - m_phi**2, 0))
        c = rng.uniform(-1, 1, nK); s = np.sqrt(1 - c**2)
        az = rng.uniform(0, 2 * math.pi, nK)
        El = gK * (Es + bK * ps * c)
        pll = gK * (ps * c + bK * Es); ptt = ps * s
        pph = (pll[:, None] * dirK + (ptt * np.cos(az))[:, None] * xc
               + (ptt * np.sin(az))[:, None] * yc)
        dph = pph / np.linalg.norm(pph, axis=1)[:, None]
        chord_m = ray_box_chord(v, dph, DET, HALF)
        sig = np.interp(El, Et, st)
        Pscat = sig * N_AR * (chord_m * 100.0)
        wd = prefm * Pscat * eff_vec(El) / n_dec
        total += wd.sum()
        m = wd > 0
        wsum2 += np.sum((wd[m] * n_dec)**2); ncontrib += int(m.sum())
        Ev_hit.append(El[m]); w_hit.append(wd[m])
    # per-hit ESF = (sum w)^2 / (N * sum w^2); per-hit weight = wd*n_dec, sum = total*n_dec
    esf = (100.0 * (total * n_dec)**2 / (ncontrib * wsum2)) if (ncontrib and wsum2 > 0) else 0.0
    return dict(name=name, br=br, total=total, esf=esf, ncontrib=ncontrib,
                E=np.concatenate(Ev_hit), w=np.concatenate(w_hit))


def main(portal="scalar", n_dec=600):
    S = _util.load_module("SBND_%s" % portal, os.path.join(HERE, _MODULES[portal]))
    HALF = np.array([S._TPC_BOX_X, S._TPC_BOX_Y, S._TPC_BOX_Z]) / 2.0
    bnb = S._BNB.generate_bnb_sample(n_per_species=50000, seed=42)
    print("\n==== ANALYTIC SBND %s, all channels (POT=%.2e, benchmark) ===="
          % (portal, S.SBND_POT))
    print("  %-7s %-11s %-11s %-9s" % ("chan", "BR", "analytic", "ESF%"))
    grand = 0.0; results = {}
    for name in ["K_e", "K_mu", "pi_e", "pi_mu"]:
        r = compute_channel(S, HALF, name, bnb, n_dec=n_dec)
        if r is None:
            print("  %-7s (kinematically forbidden)" % name); continue
        results[name] = r; grand += r["total"]
        print("  %-7s %-11.3e %-11.3e %-9.1f" % (name, r["br"], r["total"], r["esf"]))
    print("  " + "-" * 40)
    print("  %-7s %-11s %-11.3e" % ("TOTAL", "", grand))
    dir_tot = _DIRECTED_TOT.get(portal)
    print("\n  => TRUE SBND %s total (analytic) = %.3e events" % (portal, grand))
    if dir_tot:
        pi = sum(results[n]["total"] for n in results if n.startswith("pi"))
        K  = sum(results[n]["total"] for n in results if n.startswith("K"))
        print("     directed-sampler total ~%.3e  (%.0fx high)" % (dir_tot, dir_tot / grand))
        print("     analytic composition: pion %.0f%%, kaon %.0f%%"
              % (100 * pi / grand, 100 * K / grand))
    out = "output/SBND_%s_analytic.npz" % portal
    np.savez(out, **{f"{n}_E": results[n]["E"] for n in results},
             **{f"{n}_w": results[n]["w"] for n in results})
    print("  Saved spectra -> %s" % out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--portal", choices=["scalar", "pseudo"], default="scalar")
    ap.add_argument("--n-dec", type=int, default=600)
    args = ap.parse_args()
    main(portal=args.portal, n_dec=args.n_dec)
