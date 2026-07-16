"""
MiniBooNE scalar Dark-Primakoff per-channel observables, in the 3-panel style:
  (1) E_gamma [MeV]      (2) cos(theta) wrt beam      (3) cos(theta) zoom [0.80,1]
each with the four production channels (K_e, K_mu, pi_e, pi_mu) overlaid + TOTAL.

Uses the VALIDATED analytic sigma*N*chord estimator (ray-sphere through the
MiniBooNE oil tank, carbon target) -- NOT the directed SIREN sampler. Benchmark
couplings, POT = MINIBOONE_POT. The photon direction ~ the phi direction
(coherent Dark Primakoff is forward); cos(theta) is wrt the beam (+z).
"""
import os, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from siren import _util

HERE = os.path.dirname(__file__)
S = _util.load_module("ScalarPortal_MiniBooNE_multichannel",
                      os.path.join(HERE, "ScalarPortal_MiniBooNE_multichannel.py"))
BNB = S._BNB

DET   = np.array([0.0, 1.896, 541.34])      # MiniBooNE center (BNB frame, m)
R_OIL = S.R_OIL                              # mineral-oil sphere radius [m]
N_C   = 3.6312e22                            # carbon nuclei / cm^3 (mineral oil)
POT   = S.MINIBOONE_POT
CHANS = ["K_e", "K_mu", "pi_e", "pi_mu"]
COL   = {"K_e": "C0", "K_mu": "C1", "pi_e": "C2", "pi_mu": "C3"}

_EE = np.array([0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.90])
_EF = np.array([0.089, 0.135, 0.139, 0.131, 0.123, 0.116, 0.106, 0.102])
def eff_vec(E):
    e = np.interp(E, _EE, _EF, left=_EF[0], right=_EF[-1])
    return np.where(E >= 0.140, e, 0.0)


def compute_channel(name, n_dec=400, seed=7):
    """Return per-hit (E_vis [GeV], cos_theta wrt beam, weight) for one channel."""
    pdg, m_M, m_l, lpdg, nupdg, gsm = S.CHANNELS[name]
    if (m_M - m_l) <= S.M_PHI:
        return np.array([]), np.array([]), np.array([])
    ch = S.build_onshell_models(pdg, m_M, m_l, lpdg, nupdg)
    md = ch["meson_decay"]._decay; dp = ch["models"]["primakoff"]._dp
    m_phi = md.m_phi; br = ch["meson_decay"]._total_width / gsm

    d = BNB.generate_bnb_sample(n_per_species=50000, seed=42)
    isp = d["ptype"] == pdg
    p = np.stack([d["px"][isp], d["py"][isp], d["pz"][isp]], axis=1)
    v = np.stack([d["vx"][isp], d["vy"][isp], d["vz"][isp]], axis=1) / 100.0
    E = d["E"][isp]; w = d["nimpwt"][isp]
    pmag = np.linalg.norm(p, axis=1); dirK = p / pmag[:, None]; nK = len(pmag)

    Emax = (m_M**2 + m_phi**2 - m_l**2) / (2 * m_M)
    Eg = np.linspace(m_phi + 1e-5, Emax - 1e-5, 400)
    dN = np.array([max(md.differential_decay_rate([e])[0], 0.0) for e in Eg])
    dN = np.where(np.isfinite(dN) & (dN > 0), dN, 0.0); cdf = np.cumsum(dN); cdf /= cdf[-1]
    Et = np.concatenate([np.linspace(0.001, 0.3, 120), np.linspace(0.31, 9, 160)])
    st = np.array([dp.total_xsec(float(e)) for e in Et])

    rng = np.random.default_rng(seed)
    arb = np.tile(np.array([0., 1., 0.]), (nK, 1))
    arb[np.abs(dirK[:, 1]) > 0.9] = np.array([1., 0., 0.])
    xc = np.cross(dirK, arb); xc /= np.linalg.norm(xc, axis=1)[:, None]; yc = np.cross(dirK, xc)
    gK = E / m_M; bK = pmag / E; prefm = w * POT * br
    Ehit, Chit, Whit = [], [], []
    for _ in range(n_dec):
        u = rng.random(nK); Es = np.interp(u, cdf, Eg); ps = np.sqrt(np.maximum(Es**2 - m_phi**2, 0))
        c = rng.uniform(-1, 1, nK); s = np.sqrt(1 - c**2); az = rng.uniform(0, 2 * math.pi, nK)
        El = gK * (Es + bK * ps * c); pll = gK * (ps * c + bK * Es); ptt = ps * s
        pph = (pll[:, None] * dirK + (ptt * np.cos(az))[:, None] * xc + (ptt * np.sin(az))[:, None] * yc)
        dph = pph / np.linalg.norm(pph, axis=1)[:, None]
        mv = v - DET[None, :]; bd = np.einsum("ij,ij->i", mv, dph)
        disc = bd**2 - (np.einsum("ij,ij->i", mv, mv) - R_OIL**2)
        hit = (disc > 0) & (-bd > 0)
        chord = np.where(hit, 2 * np.sqrt(np.maximum(disc, 0)), 0.0) * 100.0   # cm
        sig = np.interp(El, Et, st)
        wd = prefm * np.where(hit, sig * N_C * chord, 0.0) * eff_vec(El) / n_dec
        m = wd > 0
        Ehit.append(El[m]); Chit.append(dph[m, 2]); Whit.append(wd[m])   # cos = dph_z (beam +z)
    return (np.concatenate(Ehit), np.concatenate(Chit), np.concatenate(Whit))


def main():
    data = {}
    for nm in CHANS:
        print("computing %s ..." % nm)
        data[nm] = compute_channel(nm)
        print("  %s total = %.3e events" % (nm, data[nm][2].sum()))

    Ebins = np.linspace(0, 2000, 41)            # MeV
    cbins = np.linspace(-1, 1, 61)
    czoom = np.linspace(0.80, 1.0, 41)
    fig, ax = plt.subplots(1, 3, figsize=(19, 4.8))

    def panel(a, x_of, bins, scale=1.0, xlabel=""):
        tot = np.zeros(len(bins) - 1)
        for nm in CHANS:
            E, c, w = data[nm]
            if not len(E): continue
            x = (E * 1e3 if x_of == "E" else c)
            h = np.histogram(x, bins=bins, weights=w)[0]
            a.step(0.5 * (bins[:-1] + bins[1:]), h, where="mid", color=COL[nm], lw=1.4, label=nm)
            tot += h
        a.step(0.5 * (bins[:-1] + bins[1:]), tot, where="mid", color="k", lw=2.2, label="TOTAL")
        a.set_xlabel(xlabel); a.set_ylabel("Counts"); a.legend(fontsize=8)

    panel(ax[0], "E", Ebins, xlabel=r"$E_\gamma$ [MeV]")
    ax[0].set_title(r"MiniBooNE scalar Dark Primakoff: $E_\gamma$ (%.2e POT)" % POT, fontsize=10)
    panel(ax[1], "c", cbins, xlabel=r"$\cos\theta$ wrt beam")
    ax[1].set_title(r"MiniBooNE scalar: $\cos\theta$ (%.2e POT)" % POT, fontsize=10)
    panel(ax[2], "c", czoom, xlabel=r"$\cos\theta$")
    ax[2].set_title(r"MiniBooNE scalar: $\cos\theta$ zoom [0.80, 1.0]", fontsize=10)
    for a in ax: a.set_ylim(bottom=0)

    fig.tight_layout()
    out = os.path.join(HERE, "output", "MiniBooNE_scalar_perchannel.png")
    fig.savefig(out, dpi=130); plt.close(fig)
    print("Saved -> %s" % out)


if __name__ == "__main__":
    main()
