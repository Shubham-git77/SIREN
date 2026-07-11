"""
Independent ANALYTIC estimator for the SBND VECTOR-portal (e+e-) rate, to check
the directed sampler (overall ESF 0.06%, K_mu ESF 0.1% -> over-estimated).

Chain: meson -> mu nu V1 ; V1 -> chi chi ; chi + Ar -> chi' + Ar (upscatter) ;
chi' -> chi V1_sig ; V1_sig -> e+e-.  All daughter decays are prompt and their
BRs ~ 1 (V1->chichi dominant via G_D >> eps; chi' and V1_sig are single-channel),
so the rate is production x [2 chi] x (sigma_ups * N_Ar * chord).

Method (two nested boosts, no SIREN injector):
  1. sample V1 from meson decay (vector ME spectrum) -> E_V1, dir   [boost #1]
  2. V1 -> chi chi: chi fixed energy M_V1/2 in V1 frame, isotropic; both chi
     boosted to lab                                                  [boost #2]
  3. ray-trace each chi to the SBND LAr box -> chord
  4. P_ups = sigma_ups(E_chi) * N_Ar * chord  (E_chi > Ethreshold)
  dN = flux * POT * BR(M->mu nu V1)*CALIB_VECTOR * P_ups * eff(E_vis~E_chi)

E_vis is approximated by E_chi (the visible e+e- energy is set by the cascade;
this is a proxy for the efficiency cut only). VALIDATION: the analytic K_e is
compared to the directed K_e (8988, ESF 21% -> reliable). A match calibrates the
normalization and the E_vis proxy.
"""
import os, math, argparse
import numpy as np
from siren import _util

HERE = os.path.dirname(__file__)
V = _util.load_module("SBND_vector_fullchain",
                      os.path.join(HERE, "VectorPortal_SBND_fullchain.py"))

DET  = np.array([0.7378, 0.0, 110.0])
N_AR = 2.1036e22
POT  = V.SBND_POT
HALF = np.array([V._TPC_BOX_X, V._TPC_BOX_Y, V._TPC_BOX_Z]) / 2.0

_EE = np.array([0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.90])
_EF = np.array([0.089, 0.135, 0.139, 0.131, 0.123, 0.116, 0.106, 0.102])
def eff_vec(E):
    e = np.interp(E, _EE, _EF, left=_EF[0], right=_EF[-1])
    return np.where(E >= 0.140, e, 0.0)

# directed reference (Jun-22 fullchain run)
_DIRECTED = {"K_e": 8.988e3, "K_mu": 2.855e4, "pi_e": 9.85e2, "pi_mu": 0.0}


def ray_box_chord(v, d, center, half):
    lo = center - half; hi = center + half
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        t1 = (lo - v) * inv; t2 = (hi - v) * inv
    tmin = np.minimum(t1, t2); tmax = np.maximum(t1, t2)
    tenter = np.nanmax(tmin, axis=1); texit = np.nanmin(tmax, axis=1)
    hit = (tenter < texit) & (texit > 0)
    return np.where(hit, texit - np.clip(tenter, 0.0, None), 0.0)


def basis(dirv):
    """orthonormal (x,y) perpendicular to each row of dirv (N,3)."""
    arb = np.tile(np.array([0., 1., 0.]), (len(dirv), 1))
    arb[np.abs(dirv[:, 1]) > 0.9] = np.array([1., 0., 0.])
    xc = np.cross(dirv, arb); xc /= np.linalg.norm(xc, axis=1)[:, None]
    yc = np.cross(dirv, xc)
    return xc, yc


def boost_daughter(E_par, p_par, dir_par, xc, yc, Estar, pstar, c, az):
    """Boost a daughter (rest-frame energy Estar, momentum pstar, angle c=cos,
    az) from a parent (E_par,p_par,dir_par) to the lab. Returns E_lab, dir_lab."""
    g = E_par / np.sqrt(np.maximum(E_par**2 - p_par**2, 1e-18))
    b = p_par / E_par
    s = np.sqrt(np.maximum(1 - c**2, 0))
    El = g * (Estar + b * pstar * c)
    pll = g * (pstar * c + b * Estar)
    ptt = pstar * s
    plab = (pll[:, None] * dir_par + (ptt * np.cos(az))[:, None] * xc
            + (ptt * np.sin(az))[:, None] * yc)
    dl = plab / np.linalg.norm(plab, axis=1)[:, None]
    return El, dl


def compute_channel(name, bnb, n_dec=400, seed=7):
    pdg, m_M, m_l, lpdg, nupdg, gsm = V.CHANNELS[name]
    if (m_M - m_l) <= V.M_V1:
        return None
    ch = V.build_onshell_models(pdg, m_M, m_l, lpdg, nupdg)
    md = ch["meson_decay"]._decay
    ups = ch["models"]["upscatter"]._ups
    m_V1 = V.M_V1; m_chi = V.M_CHI
    br_prod = ch["meson_decay"]._total_width * V.CALIB_VECTOR / gsm

    isp = bnb["ptype"] == pdg
    p = np.stack([bnb["px"][isp], bnb["py"][isp], bnb["pz"][isp]], axis=1)
    vtx = np.stack([bnb["vx"][isp], bnb["vy"][isp], bnb["vz"][isp]], axis=1) / 100.0
    E = bnb["E"][isp]; w = bnb["nimpwt"][isp]
    pmag = np.linalg.norm(p, axis=1); dirM = p / pmag[:, None]; nM = len(pmag)

    # V1 rest-frame energy spectrum from the vector ME (E_V1 in meson rest frame)
    Emax = (m_M**2 + m_V1**2 - m_l**2) / (2 * m_M)
    Eg = np.linspace(m_V1 + 1e-5, Emax - 1e-5, 400)
    dN = np.array([max(md.differential_decay_rate([e])[0], 0.0) for e in Eg])
    dN = np.where(np.isfinite(dN) & (dN > 0), dN, 0.0); cdf = np.cumsum(dN); cdf /= cdf[-1]

    # sigma_ups(E_chi)
    Et = np.linspace(ups.Ethreshold, 9.0, 300)
    st = np.array([ups.total_xsec(float(e)) for e in Et])

    # chi rest-frame kinematics in V1 frame (fixed)
    Echi_star = m_V1 / 2.0
    pchi_star = math.sqrt(max(Echi_star**2 - m_chi**2, 0.0))

    xcM, ycM = basis(dirM)
    gM = E / m_M; bM = pmag / E
    prefm = w * POT * br_prod
    rng = np.random.default_rng(seed)

    total = 0.0; wsum2 = 0.0; nhit = 0; Ev = []; wv = []
    for _ in range(n_dec):
        # boost #1: meson -> V1
        u = rng.random(nM); EsV = np.interp(u, cdf, Eg)
        psV = np.sqrt(np.maximum(EsV**2 - m_V1**2, 0))
        cV = rng.uniform(-1, 1, nM); azV = rng.uniform(0, 2 * math.pi, nM)
        EV1, dV1 = boost_daughter(E, pmag, dirM, xcM, ycM, EsV, psV, cV, azV)
        xcV, ycV = basis(dV1)
        pV1 = np.sqrt(np.maximum(EV1**2 - m_V1**2, 0))
        # boost #2: V1 -> chi chi (two back-to-back chi)
        cc = rng.uniform(-1, 1, nM); azc = rng.uniform(0, 2 * math.pi, nM)
        for sign in (+1.0, -1.0):
            Echi, dchi = boost_daughter(EV1, pV1, dV1, xcV, ycV,
                                        Echi_star, pchi_star, sign * cc, azc)
            chord = ray_box_chord(vtx, dchi, DET, HALF)
            sig = np.interp(Echi, Et, st, left=0.0, right=st[-1])
            sig = np.where(Echi >= ups.Ethreshold, sig, 0.0)
            Pups = sig * N_AR * (chord * 100.0)
            wd = prefm * Pups * eff_vec(Echi) / n_dec
            total += wd.sum()
            m = wd > 0
            wsum2 += np.sum((wd[m] * n_dec)**2); nhit += int(m.sum())
            Ev.append(Echi[m]); wv.append(wd[m])
    esf = (100.0 * (total * n_dec)**2 / (nhit * wsum2)) if (nhit and wsum2 > 0) else 0.0
    return dict(name=name, br=br_prod, total=total, esf=esf,
                E=np.concatenate(Ev) if Ev else np.array([]),
                w=np.concatenate(wv) if wv else np.array([]))


def main(n_dec=400):
    bnb = V._BNB.generate_bnb_sample(n_per_species=50000, seed=42)
    print("\n==== ANALYTIC SBND vector (e+e-), all channels (POT=%.2e) ====" % POT)
    print("  %-7s %-11s %-11s %-9s %-11s %-9s" %
          ("chan", "BR*CALIB", "analytic", "ESF%", "directed", "dir/anal"))
    grand = 0.0; results = {}
    for name in ["K_e", "K_mu", "pi_e", "pi_mu"]:
        r = compute_channel(name, bnb, n_dec=n_dec)
        if r is None:
            print("  %-7s (forbidden: m_M - m_l < M_V1)" % name); continue
        results[name] = r; grand += r["total"]
        d = _DIRECTED.get(name, float("nan"))
        ratio = (d / r["total"]) if r["total"] > 0 else float("inf")
        print("  %-7s %-11.3e %-11.3e %-9.1f %-11.3e %-9.1f" %
              (name, r["br"], r["total"], r["esf"], d, ratio))
    print("  " + "-" * 62)
    dir_tot = sum(_DIRECTED.values())
    print("  %-7s %-11s %-11.3e %-9s %-11.3e %-9.1f" %
          ("TOTAL", "", grand, "", dir_tot, dir_tot / grand if grand else 0))
    print("\n  => TRUE SBND vector total (analytic) = %.3e events" % grand)
    print("  VALIDATION: analytic K_e vs directed K_e (ESF 21%%, reliable):")
    if "K_e" in results:
        print("     analytic=%.3e  directed=8.99e3  ratio=%.2f  (==1 => calibrated)"
              % (results["K_e"]["total"], 8.988e3 / max(results["K_e"]["total"], 1e-30)))
    np.savez("output/SBND_vector_analytic.npz",
             **{f"{n}_E": results[n]["E"] for n in results},
             **{f"{n}_w": results[n]["w"] for n in results})
    print("  Saved -> output/SBND_vector_analytic.npz")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-dec", type=int, default=400)
    args = ap.parse_args()
    main(n_dec=args.n_dec)
