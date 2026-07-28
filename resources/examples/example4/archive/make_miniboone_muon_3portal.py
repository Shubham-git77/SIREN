"""
MiniBooNE meson-portal prediction, MUON-ONLY (K_mu + pi_mu, g_e=0), at the
MiniBooNE best-fit coupling (Fig. 2 normalization), for all three portals.

3-panel E_gamma figure (scalar | pseudoscalar | vector), each showing the two
muon production channels (K_mu, pi_mu) + TOTAL, in the per-channel style.

Validated analytic sigma*N*chord estimator (ray-sphere through the MiniBooNE oil
tank, carbon target) -- NOT the directed SIREN sampler. Each portal's analytic
shape is FIT to the digitized MiniBooNE excess (data - bkg), i.e. the Fig. 2
normalization, so the meaningful output is the coupling each portal requires.

E_vis: scalar/pseudo = E_gamma (single photon, =E_phi); vector = the e+e- visible
energy from the full cascade (chi upscatters -> chi'(E~E_chi) -> chi + V1_sig
[two-body] -> e+e-, so E_vis = E_{V1_sig}, which is ~0.5 E_chi -- softer than the
old E_chi proxy and matching the excess).
"""
import os, math
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from siren import _util

HERE = os.path.dirname(__file__)
def load(n): return _util.load_module(n, os.path.join(HERE, n + ".py"))
S_sc  = load("ScalarPortal_MiniBooNE_multichannel")
S_ps  = load("PseudoscalarPortal_MiniBooNE_multichannel")
S_vec = load("VectorPortal_MiniBooNE_fullchain")
BNB = S_sc._BNB

DET   = np.array([0.0, 1.896, 541.34])
R_OIL = S_sc.R_OIL
N_C   = 3.6312e22
POT   = S_sc.MINIBOONE_POT
MUON  = ["K_mu", "pi_mu"]
COL   = {"K_mu": "C1", "pi_mu": "C3"}
BESTFIT_A = {"scalar": 0.038, "pseudo": 0.002, "vector": 0.079}

_EE = np.array([0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.90])
_EF = np.array([0.089, 0.135, 0.139, 0.131, 0.123, 0.116, 0.106, 0.102])
def eff_vec(E):
    e = np.interp(E, _EE, _EF, left=_EF[0], right=_EF[-1])
    return np.where(E >= 0.140, e, 0.0)

def basis(dirv):
    arb = np.tile(np.array([0., 1., 0.]), (len(dirv), 1))
    arb[np.abs(dirv[:, 1]) > 0.9] = np.array([1., 0., 0.])
    xc = np.cross(dirv, arb); xc /= np.linalg.norm(xc, axis=1)[:, None]
    return xc, np.cross(dirv, xc)

def boost(E_par, p_par, dirp, xc, yc, Estar, pstar, c, az):
    g = E_par / np.sqrt(np.maximum(E_par**2 - p_par**2, 1e-18)); b = p_par / E_par
    s = np.sqrt(np.maximum(1 - c**2, 0))
    El = g * (Estar + b * pstar * c); pll = g * (pstar * c + b * Estar); ptt = pstar * s
    plab = pll[:, None] * dirp + (ptt * np.cos(az))[:, None] * xc + (ptt * np.sin(az))[:, None] * yc
    return El, plab / np.linalg.norm(plab, axis=1)[:, None]

def ray_sphere_chord(v, d, C, R):
    mv = v - C[None, :]; bd = np.einsum("ij,ij->i", mv, d)
    disc = bd**2 - (np.einsum("ij,ij->i", mv, mv) - R**2)
    hit = (disc > 0) & (-bd > 0)
    return np.where(hit, 2 * np.sqrt(np.maximum(disc, 0)), 0.0)

def _meson_arrays(S, pdg):
    d = BNB.generate_bnb_sample(n_per_species=50000, seed=42)
    isp = d["ptype"] == pdg
    p = np.stack([d["px"][isp], d["py"][isp], d["pz"][isp]], axis=1)
    v = np.stack([d["vx"][isp], d["vy"][isp], d["vz"][isp]], axis=1) / 100.0
    E = d["E"][isp]; w = d["nimpwt"][isp]
    pmag = np.linalg.norm(p, axis=1)
    return E, pmag, p / pmag[:, None], v, w

# ---- scalar / pseudoscalar single-photon ----
def sp_channel(S, name, n_dec=400, seed=7):
    pdg, m_M, m_l, lpdg, nupdg, gsm = S.CHANNELS[name]
    ch = S.build_onshell_models(pdg, m_M, m_l, lpdg, nupdg)
    md = ch["meson_decay"]._decay; dp = ch["models"]["primakoff"]._dp
    m_phi = md.m_phi; br = ch["meson_decay"]._total_width / gsm
    E, pmag, dirK, v, w = _meson_arrays(S, pdg); nK = len(E)
    Emax = (m_M**2 + m_phi**2 - m_l**2) / (2 * m_M)
    Eg = np.linspace(m_phi + 1e-5, Emax - 1e-5, 400)
    dN = np.array([max(md.differential_decay_rate([e])[0], 0.0) for e in Eg])
    dN = np.where(np.isfinite(dN) & (dN > 0), dN, 0.0); cdf = np.cumsum(dN); cdf /= cdf[-1]
    Et = np.concatenate([np.linspace(0.001, 0.3, 120), np.linspace(0.31, 9, 160)])
    st = np.array([dp.total_xsec(float(e)) for e in Et])
    xc, yc = basis(dirK); gK = E / m_M; bK = pmag / E; prefm = w * POT * br
    rng = np.random.default_rng(seed); Eh, Wh = [], []
    for _ in range(n_dec):
        u = rng.random(nK); Es = np.interp(u, cdf, Eg); ps = np.sqrt(np.maximum(Es**2 - m_phi**2, 0))
        c = rng.uniform(-1, 1, nK); az = rng.uniform(0, 2 * math.pi, nK)
        El, dph = boost(E, pmag, dirK, xc, yc, Es, ps, c, az)
        chord = ray_sphere_chord(v, dph, DET, R_OIL) * 100.0
        wd = prefm * np.interp(El, Et, st) * N_C * chord * eff_vec(El) / n_dec
        m = wd > 0; Eh.append(El[m]); Wh.append(wd[m])
    return np.concatenate(Eh), np.concatenate(Wh)

# ---- vector cascade (meson->V1->chi chi -> upscatter -> e+e-) ----
def vec_channel(name, n_dec=400, seed=7):
    pdg, m_M, m_l, lpdg, nupdg, gsm = S_vec.CHANNELS[name]
    ch = S_vec.build_onshell_models(pdg, m_M, m_l, lpdg, nupdg)
    md = ch["meson_decay"]._decay; ups = ch["models"]["upscatter"]._ups
    m_V1 = S_vec.M_V1; m_chi = S_vec.M_CHI
    br = ch["meson_decay"]._total_width * S_vec.CALIB_VECTOR / gsm
    E, pmag, dirK, v, w = _meson_arrays(S_vec, pdg); nK = len(E)
    Emax = (m_M**2 + m_V1**2 - m_l**2) / (2 * m_M)
    Eg = np.linspace(m_V1 + 1e-5, Emax - 1e-5, 400)
    dN = np.array([max(md.differential_decay_rate([e])[0], 0.0) for e in Eg])
    dN = np.where(np.isfinite(dN) & (dN > 0), dN, 0.0); cdf = np.cumsum(dN); cdf /= cdf[-1]
    Et = np.linspace(ups.Ethreshold, 9.0, 300); st = np.array([ups.total_xsec(float(e)) for e in Et])
    Echi_star = m_V1 / 2.0; pchi_star = math.sqrt(max(Echi_star**2 - m_chi**2, 0))
    # cascade for the VISIBLE e+e- energy: chi upscatters to chi'(E~E_chi), then
    # chi' -> chi + V1_sig (two-body), V1_sig -> e+e-, so E_vis = E_{V1_sig}.
    m_cp = S_vec.M_CHI_PRIME
    E_Vs = (m_cp**2 + m_V1**2 - m_chi**2) / (2 * m_cp)      # V1_sig energy in chi' rest frame
    p_Vs = math.sqrt(max(E_Vs**2 - m_V1**2, 0))
    xcK, ycK = basis(dirK); gK = E / m_M; bK = pmag / E; prefm = w * POT * br
    rng = np.random.default_rng(seed); Eh, Wh = [], []
    for _ in range(n_dec):
        u = rng.random(nK); EsV = np.interp(u, cdf, Eg); psV = np.sqrt(np.maximum(EsV**2 - m_V1**2, 0))
        cV = rng.uniform(-1, 1, nK); azV = rng.uniform(0, 2 * math.pi, nK)
        EV1, dV1 = boost(E, pmag, dirK, xcK, ycK, EsV, psV, cV, azV)
        xcV, ycV = basis(dV1); pV1 = np.sqrt(np.maximum(EV1**2 - m_V1**2, 0))
        cc = rng.uniform(-1, 1, nK); azc = rng.uniform(0, 2 * math.pi, nK)
        for sgn in (1.0, -1.0):
            Ech, dch = boost(EV1, pV1, dV1, xcV, ycV, Echi_star, pchi_star, sgn * cc, azc)
            chord = ray_sphere_chord(v, dch, DET, R_OIL) * 100.0
            sig = np.where(Ech >= ups.Ethreshold, np.interp(Ech, Et, st, left=0, right=st[-1]), 0.0)
            # chi'(E~E_chi) -> chi V1_sig: boost the V1_sig energy to the lab (isotropic decay)
            gp = np.maximum(Ech / m_cp, 1.0); bp = np.sqrt(np.maximum(1 - 1 / gp**2, 0))
            cstar = rng.uniform(-1, 1, nK)
            E_vis = gp * (E_Vs + bp * p_Vs * cstar)          # = E_{V1_sig} = e+e- visible energy
            wd = prefm * sig * N_C * chord * eff_vec(E_vis) / n_dec
            m = wd > 0; Eh.append(E_vis[m]); Wh.append(wd[m])
    return np.concatenate(Eh), np.concatenate(Wh)

# Digitized MiniBooNE nu-mode E_vis (Dutta-Kim Fig.2 bottom-left): the excess to fit.
DATA_E   = np.array([225,275,325,375,425,475,525,575,625,675,725,775,825,875,925,975,1025,1075,1125],float)
DATA_N   = np.array([302,402,333,279,189,168,134,118, 81, 83, 75, 84, 57, 61, 38, 52, 26, 19, 19],float)
DATA_ERR = np.array([ 36, 41, 38, 35, 28, 27, 25, 23, 18, 19, 18, 19, 15, 18, 13, 15, 11, 12, 10],float)
BKG      = np.array([255,320,300,250,175,150,120,105, 76, 78, 70, 76, 52, 55, 35, 47, 24, 18, 17],float)
PBINS = np.concatenate([[DATA_E[0]-25], DATA_E+25])
EXCESS = DATA_N - BKG; WGT = 1.0/DATA_ERR**2

def fit_to_excess(E_mev, w):
    """Least-squares amplitude A so that A*signal best matches the MiniBooNE
    excess (data-bkg) over the digitized bins -- the Fig.2 fit."""
    sig = np.histogram(E_mev, bins=PBINS, weights=w)[0]
    denom = np.sum(WGT*sig*sig)
    A = max(np.sum(WGT*sig*EXCESS)/denom, 0.0) if denom > 0 else 0.0
    chi2 = np.sum(((EXCESS - A*sig)**2)*WGT)
    return A, chi2

def main():
    portals = [("scalar", "Scalar $\\phi\\to\\gamma$", lambda nm: sp_channel(S_sc, nm), "coupling"),
               ("pseudo", "Pseudoscalar $a\\to\\gamma$", lambda nm: sp_channel(S_ps, nm), "coupling"),
               ("vector", "Vector $\\to e^+e^-$", vec_channel, "rate")]
    Ebins = np.linspace(0, 2000, 41); EC = 0.5 * (Ebins[:-1] + Ebins[1:])
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for j, (key, label, fn, kind) in enumerate(portals):
        # benchmark per-channel signal, then fit overall amplitude to the excess
        chan = {nm: fn(nm) for nm in MUON}
        E_all = np.concatenate([chan[nm][0] for nm in MUON]) * 1e3
        w_all = np.concatenate([chan[nm][1] for nm in MUON])
        A, chi2 = fit_to_excess(E_all, w_all)
        tot = np.zeros(len(EC)); grand = 0.0
        for nm in MUON:
            E, wts = chan[nm]; wts = wts * A
            h = np.histogram(E * 1e3, bins=Ebins, weights=wts)[0]
            ax[j].step(EC, h, where="mid", color=COL[nm], lw=1.5, label=nm)
            tot += h; grand += wts.sum()
        ax[j].step(EC, tot, where="mid", color="k", lw=2.2, label="TOTAL")
        ax[j].set_xlabel(r"$E_\gamma$ [MeV]"); ax[j].set_ylabel("Counts"); ax[j].legend(fontsize=8)
        cstr = ("coupling %.2f$\\times$bench" % math.sqrt(A)) if kind == "coupling" else ("A=%.1f$\\times$bench" % A)
        ax[j].set_title("%s — fit %s, total %.0f ev, $\\chi^2$/bin %.1f"
                        % (label, cstr, grand, chi2/len(DATA_E)), fontsize=9.5)
        ax[j].set_ylim(bottom=0)
        print("  %-7s fit A=%.3e (%s)  total=%.0f ev  chi2/bin=%.1f"
              % (key, A, cstr, grand, chi2/len(DATA_E)))
    fig.suptitle("MiniBooNE meson-portal, MUON-ONLY, each portal FIT to the MiniBooNE excess (Fig. 2 normalization) "
                 "— analytic $\\sigma\\!\\cdot\\!N\\!\\cdot\\!$chord", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(HERE, "output", "MiniBooNE_muon_3portal.png")
    fig.savefig(out, dpi=115); plt.close(fig); print("Saved -> %s" % out)

if __name__ == "__main__":
    main()
