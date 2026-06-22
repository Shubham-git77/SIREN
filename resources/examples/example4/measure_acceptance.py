"""
Measure the TRUE phi acceptance A_fid (fraction of phi from meson decay that
reach the MiniBooNE detector), INCLUDING the decay angular smearing that the
phi||meson approximation in derive_dp_calib.py ignored.

A K+ (gamma_K~4.5) emits phi into a ~1/gamma ~ 220 mrad cone; the detector
subtends ~11 mrad at 541 m, so even a kaon aimed straight at the detector
delivers only a small fraction of its phi. This script:
  1. reports the BNB meson angular distribution (over-collimated?),
  2. MC-decays each meson isotropically-in-rest-frame (phi E from the real
     3-body spectrum), boosts to lab, and counts phi that hit the fiducial vol,
  3. compares true A_fid to (a) phi||meson (5.3%) and (b) isotropic Omega/4pi.
"""
import os, math
import numpy as np
from siren import _util

S = _util.load_module("ScalarMB", os.path.join(
    os.path.dirname(__file__), "ScalarPortal_MiniBooNE_multichannel.py"))
BNB = S._BNB
DET = np.array([0.0, 1.896, 541.34])      # m
R_FID = S.R_FID                            # m (5.0)

def line_hits(pos, dirn, det, R):
    """ray pos + t*dirn (t>0) passes within R of det?"""
    rel = det[None, :] - pos
    tcl = np.einsum("ij,ij->i", rel, dirn)
    closest = pos + tcl[:, None] * dirn
    miss = np.linalg.norm(closest - det[None, :], axis=1)
    return (tcl > 0) & (miss < R)

def main(n_dec=200):
    name = "K_mu"
    parent_pdg, m_meson, m_lepton, lepton_pdg, nu_pdg, gamma_sm = S.CHANNELS[name]
    chain = S.build_onshell_models(parent_pdg, m_meson, m_lepton, lepton_pdg, nu_pdg)
    md = chain["meson_decay"]._decay
    m_phi = md.m_phi
    data = BNB.generate_bnb_sample(n_per_species=50000, seed=7)
    isK = data["ptype"] == parent_pdg
    p = np.stack([data["px"][isK], data["py"][isK], data["pz"][isK]], axis=1)
    v = np.stack([data["vx"][isK], data["vy"][isK], data["vz"][isK]], axis=1)/100.0  # m
    E = data["E"][isK]; w = data["nimpwt"][isK]
    pmag = np.linalg.norm(p, axis=1)
    dirK = p/pmag[:, None]

    # (1) meson angular distribution wrt beam (+z) and wrt the line to detector
    th_beam = np.arccos(np.clip(dirK[:, 2], -1, 1))*1e3   # mrad
    to_det = DET[None, :] - v; to_det /= np.linalg.norm(to_det, axis=1)[:, None]
    ang_det = np.arccos(np.clip(np.einsum("ij,ij->i", dirK, to_det), -1, 1))*1e3
    det_halfang = math.atan(R_FID/541.0)*1e3
    print("=== BNB K+ meson angular distribution ===")
    print("  theta wrt beam [mrad]: mean=%.1f  median=%.1f  [5,95]=%s"
          % (th_beam.mean(), np.median(th_beam), np.round(np.percentile(th_beam,[5,95]),1)))
    print("  angle to detector line [mrad]: mean=%.1f  median=%.1f"
          % (ang_det.mean(), np.median(ang_det)))
    print("  detector fiducial half-angle at 541m = %.1f mrad" % det_halfang)
    print("  gamma_K: mean=%.2f  => 1/gamma decay cone ~ %.0f mrad"
          % ((E/m_meson).mean(), 1e3/(E/m_meson).mean()))

    # phi||meson acceptance (the OLD approx)
    hit_par = line_hits(v, dirK, DET, R_FID)
    A_par = np.average(hit_par, weights=w)
    print("\n=== acceptance comparison (nimpwt-weighted) ===")
    print("  A_fid [phi||meson approx, OLD] = %.3e" % A_par)

    # (2) TRUE A_fid: MC-decay each kaon n_dec times
    # sample E*_phi from the real 3-body spectrum, isotropic direction in K frame
    Eg = np.linspace(m_phi+1e-4, (m_meson**2+m_phi**2-m_lepton**2)/(2*m_meson)-1e-4, 400)
    try:
        dN = np.array([max(md.differential_decay_rate([e])[0], 0.0) for e in Eg])
    except Exception:
        dN = np.ones_like(Eg)
    dN = np.where(np.isfinite(dN) & (dN > 0), dN, 0.0)
    cdf = np.cumsum(dN); cdf = cdf/cdf[-1]
    rng = np.random.default_rng(1)
    hits = np.zeros(len(pmag)); ntot = 0
    Estar_phi_mean = 0.0
    for _ in range(n_dec):
        u = rng.random(len(pmag))
        Estar = np.interp(u, cdf, Eg)               # phi energy in K rest frame
        Estar_phi_mean += Estar.mean()
        pstar = np.sqrt(np.maximum(Estar**2 - m_phi**2, 0.0))
        cth = rng.uniform(-1, 1, len(pmag)); sth = np.sqrt(1-cth**2)
        az = rng.uniform(0, 2*math.pi, len(pmag))
        # phi momentum in K rest frame (z* along K direction)
        pl = pstar*cth; pt = pstar*sth
        # boost along K direction
        gK = E/m_meson; bK = pmag/E
        Elab = gK*(Estar + bK*pl)
        pl_lab = gK*(pl + bK*Estar)
        # build lab momentum: along dirK (pl_lab) + transverse
        # transverse basis
        zc = dirK
        arb = np.tile(np.array([0.0,1.0,0.0]), (len(pmag),1))
        mask = np.abs(zc[:,1])>0.9
        arb[mask] = np.array([1.0,0.0,0.0])
        xc = np.cross(zc, arb); xc /= np.linalg.norm(xc, axis=1)[:,None]
        yc = np.cross(zc, xc)
        ptx = pt*np.cos(az); pty = pt*np.sin(az)
        pphi = pl_lab[:,None]*zc + ptx[:,None]*xc + pty[:,None]*yc
        dphi = pphi/np.linalg.norm(pphi, axis=1)[:,None]
        hits += line_hits(v, dphi, DET, R_FID).astype(float)
        ntot += 1
    A_true_per = hits/ntot
    A_true = np.average(A_true_per, weights=w)
    iso = (math.pi*R_FID**2)/(4*math.pi*541.0**2)
    print("  A_fid [TRUE, decay-smeared]    = %.3e   (<E*_phi>=%.3f GeV)" % (A_true, Estar_phi_mean/n_dec))
    print("  A_fid [isotropic Omega/4pi]    = %.3e" % iso)
    print("\n  OVER-COUNT of old phi||meson vs true = %.1fx" % (A_par/A_true if A_true>0 else 0))
    print("  true/isotropic boost enhancement     = %.1fx" % (A_true/iso))

if __name__ == "__main__":
    main()
