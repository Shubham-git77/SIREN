#!/usr/bin/env python
"""
Physics of the timing feature: a slow massive dark-sector particle arrives LATE
relative to a prompt neutrino.  We inject beams of particles of several masses via
PrimaryExternalDistribution (each carries a beam-spill start time t0), let SIREN's
timing machinery propagate them a baseline L to the detector, and read the vertex
(arrival) time it computes: t_vertex = t0 + L/(beta c), beta = sqrt(E^2-m^2)/E.

Produces two plots (the standard timing-study observables, e.g. arXiv:2006.09386):
  (1) arrival-time spectrum  -- prompt nu peak + delayed massive tails
  (2) time delay dt vs energy -- the dt = (L/c)(1/beta - 1) ~ (L/c) m^2/(2E^2) law

Setup: SBND-like baseline L, sub-GeV energies, BNB-like spill; masses 0 (nu) and a
few dark-sector values.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from siren import distributions, dataclasses, utilities

C = utilities.Constants.c                # m/ns  (~0.29979)
L = 110.0                                # baseline [m] (SBND-like near detector)
E_LO, E_HI = 0.05, 1.0                   # GeV
SPILL_NS = 100.0                         # beam-spill width for t0
MASSES = [0.0, 0.05, 0.10, 0.20]         # GeV  (0 = neutrino; others = dark-sector)
COLORS = {0.0: "k", 0.05: "tab:blue", 0.10: "tab:green", 0.20: "tab:red"}
N = 4000
SCR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "timing_phys")


def sample_population(m, seed):
    """Inject N particles of mass m via PrimaryExternalDistribution; return arrays
    (E, t0, vertex_time, dist) as computed by SIREN's timing feature."""
    os.makedirs(SCR, exist_ok=True)
    rng = np.random.default_rng(seed)
    E = rng.uniform(max(E_LO, m + 1e-3), E_HI, size=N)     # ensure E > m
    p = np.sqrt(E**2 - m**2)                                # |momentum|
    t0 = rng.uniform(0.0, SPILL_NS, size=N)
    # production at origin; interaction vertex on-axis at baseline L -> dist = L
    rows = np.column_stack([np.zeros((N, 3)),               # x0,y0,z0
                            np.zeros(N), np.zeros(N), np.full(N, L),   # x,y,z (vertex)
                            np.zeros(N), np.zeros(N), p,      # px,py,pz (forward)
                            E, np.full(N, m), t0])
    csv = os.path.join(SCR, "beam_m%0.3f.csv" % m)
    np.savetxt(csv, rows, delimiter=",",
               header="x0,y0,z0,x,y,z,px,py,pz,E,m,t0", comments="", fmt="%.8g")
    d = distributions.PrimaryExternalDistribution(csv)
    rnd = utilities.SIREN_random(seed)
    vt, tt, EE, dd = [], [], [], []
    for _ in range(N):
        r = dataclasses.PrimaryDistributionRecord(dataclasses.ParticleType.NuMu)
        d.Sample(rnd, None, None, r)
        ip = np.array(r.initial_position); v = np.array(r.interaction_vertex)
        vt.append(r.initial_time); tt.append(r.interaction_time)
        EE.append(r.energy); dd.append(float(np.linalg.norm(v - ip)))
    return np.array(EE), np.array(vt), np.array(tt), np.array(dd)


def main():
    data = {m: sample_population(m, seed=10 + i) for i, m in enumerate(MASSES)}

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))

    # ---- Plot 1: arrival-time spectrum ----
    tbins = np.linspace(L / C - 10, L / C + SPILL_NS + 700, 90)
    for m in MASSES:
        E, t0, tv, dist = data[m]
        lab = r"$\nu$ ($m=0$)" if m == 0 else r"dark, $m=%d$ MeV" % (m * 1e3)
        ax[0].hist(tv, bins=tbins, histtype="step", lw=2, color=COLORS[m], label=lab)
    ax[0].axvspan(L / C, L / C + SPILL_NS, color="0.85", zorder=0, label="prompt window")
    ax[0].set_xlabel("arrival (vertex) time  $t_0 + L/(\\beta c)$  [ns]")
    ax[0].set_ylabel("events / bin")
    ax[0].set_title("Timing spectrum: prompt $\\nu$ + delayed massive tail\n(L=%.0f m, spill=%.0f ns)" % (L, SPILL_NS))
    ax[0].legend(fontsize=9)

    # ---- Plot 2: delay vs energy ----
    for m in MASSES:
        if m == 0:
            continue
        E, t0, tv, dist = data[m]
        dt = (tv - t0) - dist / C                       # delay vs a light-speed particle
        ax[1].scatter(E, dt, s=6, alpha=0.25, color=COLORS[m])
        Eg = np.linspace(m + 1e-3, E_HI, 200)
        beta = np.sqrt(Eg**2 - m**2) / Eg
        ax[1].plot(Eg, (L / C) * (1.0 / beta - 1.0), color=COLORS[m], lw=2,
                   label=r"$m=%d$ MeV" % (m * 1e3))
    ax[1].set_xlabel("energy $E$ [GeV]")
    ax[1].set_ylabel(r"delay $\Delta t = (L/c)(1/\beta - 1)$  [ns]")
    ax[1].set_title(r"Time delay vs energy  ($\Delta t \approx \frac{L}{c}\frac{m^2}{2E^2}$)")
    ax[1].set_yscale("log"); ax[1].legend(fontsize=9)

    fig.suptitle("SIREN timing feature: slow massive dark-sector particles arrive late "
                 "(injected via PrimaryExternalDistribution t0)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(SCR, exist_ok=True)
    out = os.path.join(SCR, "timing_massive_signature.png")
    fig.savefig(out, dpi=140); plt.close(fig)
    print("saved ->", out)

    # quick numeric readout
    for m in MASSES:
        E, t0, tv, dist = data[m]
        dt = (tv - t0) - dist / C
        print("  m=%3d MeV:  <beta>=%.3f  median delay=%.1f ns  max delay=%.1f ns"
              % (m * 1e3, np.mean(np.sqrt(np.maximum(E**2 - m**2, 0)) / E), np.median(dt), dt.max()))


if __name__ == "__main__":
    main()
