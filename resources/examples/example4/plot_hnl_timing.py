#!/usr/bin/env python
"""
HNL timing figure for SBND in the SIREN-SBN reference style (sbnd_hnl_timing.png):

  Left : interaction times in the SBND active volume, on a "time since
         proton on target" axis (per-event t0 removed; the meson leg
         target -> production point is added back at speed c, a sub-ns
         approximation for relativistic BNB mesons), for both the
         upscatter (nu Ar -> N Ar) and the decay (N -> nu gamma).
         Dashed line: light from the target reaching the detector center.
  Right: N decay delay relative to light arriving at the decay point
         (log-y), i.e. the pure HNL time-of-flight delay.

Run (quick):  SBND_N_EVENTS=3000 .../python plot_hnl_timing.py
"""
import os
import numpy as np

os.environ.setdefault("SBND_N_EVENTS", "3000")

# Execute sbnd.py up to (and including) event generation, reusing its full
# injection setup; stop before its own timing extraction.
_src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sbnd.py")).read()
_ns = {"__name__": "sbnd_setup"}
exec(_src[:_src.index("# 6. Record the timing chain")], _ns)

siren = _ns["siren"]
events = _ns["events"]
model_kwargs = _ns["model_kwargs"]

N4 = int(siren.dataclasses.Particle.ParticleType.N4)
C = siren.utilities.Constants.c            # m/ns

# BNB target in the SBND detector frame: BNB origin minus the detector
# center (0.7378, -0.59, 112.92) -- same translation used by the CSV maker.
TARGET_DET = np.array([-0.7378, +0.59, -112.92])

rows = []
for ev in events:
    if len(ev.tree) == 0:
        continue
    r0 = ev.tree[0].record
    t0 = float(r0.primary_initial_time)
    x0 = np.array(r0.primary_initial_position, float)   # nu production point
    xup = np.array(r0.interaction_vertex, float)
    t_up = float(r0.interaction_time)
    row = [t0, np.linalg.norm(x0 - TARGET_DET), t_up, np.nan, np.nan, np.nan, np.nan, np.nan]
    for datum in ev.tree[1:]:
        r = datum.record
        if int(r.signature.primary_type) != N4:
            continue
        xdec = np.array(r.interaction_vertex, float)
        row[3] = float(r.interaction_time)
        row[4:7] = xdec
        row[7] = np.linalg.norm(xdec - TARGET_DET)
        break
    rows.append(row)

a = np.array(rows, float)
t0, d_meson, t_up, t_dec = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
d_dec_target = a[:, 7]

# Time since proton on target: strip the synthetic spill t0, add the
# target -> production leg back at c (meson beta ~ 1 approximation).
t_pot_up = (t_up - t0) + d_meson / C
t_pot_dec = (t_dec - t0) + d_meson / C

# Light from target to the detector center (det origin):
t_light_center = np.linalg.norm(TARGET_DET) / C

# N decay delay relative to light reaching the decay point:
delay = t_pot_dec - d_dec_target / C

ok = np.isfinite(t_pot_dec)
print("events: %d  (with decay: %d)" % (len(a), ok.sum()))
print("t_pot upscatter: [%.1f, %.1f] ns   light @ center: %.1f ns"
      % (t_pot_up.min(), t_pot_up.max(), t_light_center))
print("N decay delay vs light: mean %.4f ns  p99 %.4f ns  max %.4f ns"
      % (np.nanmean(delay), np.nanpercentile(delay[ok], 99), np.nanmax(delay)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
lo = np.floor(min(t_pot_up.min(), np.nanmin(t_pot_dec)))
hi = np.ceil(max(t_pot_up.max(), np.nanmax(t_pot_dec)))
bins = np.linspace(lo, hi, 45)
ax[0].hist(t_pot_up, bins=bins, histtype="step", lw=1.6, color="C0",
           label=r"$\nu_\mu\,{\rm Ar} \to N\,{\rm Ar}$")
ax[0].hist(t_pot_dec[ok], bins=bins, histtype="step", lw=1.6, color="C3",
           ls="--", label=r"$N \to \nu\gamma$")
ax[0].axvline(t_light_center, color="gray", ls=":", lw=1.2)
ax[0].text(t_light_center, ax[0].get_ylim()[1] * 0.97, " light from target",
           color="gray", fontsize=9, va="top")
ax[0].set_xlabel("time since proton on target [ns]")
ax[0].set_ylabel("events / bin")
ax[0].set_title("Interaction times in the SBND active volume")
ax[0].legend(loc="lower center")

dmax = np.nanpercentile(delay[ok], 99.9)
ax[1].hist(delay[ok], bins=np.linspace(0.0, max(dmax, 1e-3), 60),
           color="teal", alpha=0.85)
ax[1].set_yscale("log")
ax[1].set_xlabel("N decay delay relative to light [ns]")
ax[1].set_ylabel("events / bin")
ax[1].set_title("HNL time-of-flight delay")

fig.suptitle(r"Dipole-portal HNL at SBND: $m_4$ = %.2f GeV, $\mu_{tr}$ = %.1e GeV$^{-1}$"
             % (model_kwargs["m4"], model_kwargs["mu_tr_mu4"]))
fig.tight_layout()
out = "output/sbnd_hnl_timing_lightpoint.png"
fig.savefig(out, dpi=130)
print("wrote", out)
