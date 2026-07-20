"""
ICARUS_VectorPortal_kaon_e_full_plot.py
========================================
Plotting script for ICARUS_VectorPortal_kaon_e_full output.

Reads from .parquet file (kinematics + weights).
Falls back to .siren_events if parquet kinematics are empty.

Corrections applied:
  1. Flux correction factor (N_unbiased / N_biased) = 320800 / 44093 = 7.28
     The simulation used a biased kaon CSV; this restores the true flux norm.
  2. All secondary histograms weighted by the same per-event weight as the
     primary K+ (correct: one weight per event, not per particle).
  3. 5-element momentum vector handled ([E, px, py, pz, helicity] -> take [:4])
  4. Fiducial flag: True if the V1_signal vertex is inside ICARUS LAr.

Usage:
    python ICARUS_VectorPortal_kaon_e_full_plot.py
    python ICARUS_VectorPortal_kaon_e_full_plot.py \\
        --stem output/ICARUS_VectorPortal_kaon_e_full \\
        --outdir plots/ --pot 6e20
    python ICARUS_VectorPortal_kaon_e_full_plot.py --diagnose
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--stem",     default="output/ICARUS_VectorPortal_kaon_e_full")
parser.add_argument("--outdir",   default=".")
parser.add_argument("--pot",      type=float, default=6.0e20)
parser.add_argument("--n-unbiased", type=int, default=320800,
                    help="Total K+ in original unbiased CSV (default: 320800)")
parser.add_argument("--n-biased",   type=int, default=44093,
                    help="Effective K+ sample size after bias (default: 44093)")
parser.add_argument("--diagnose", action="store_true")
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)
POT = args.pot

# ── Flux correction for kaon sampling bias ────────────────────────────────────
# The simulation used a biased CSV (preferentially forward, high-E kaons).
# Both injection and physical distributions used the same biased CSV, so
# the per-event weights do NOT include the flux normalisation correction.
# We apply it here: multiply all weights by (N_unbiased / N_biased).
FLUX_CORRECTION = args.n_unbiased / args.n_biased
print("Flux correction factor : %.4f  (%d / %d)" %
      (FLUX_CORRECTION, args.n_unbiased, args.n_biased))

# ── PDG codes ─────────────────────────────────────────────────────────────────
PDG_KAON      = 321
PDG_CHI       = 5917
PDG_CHI_PRIME = 5918
PDG_V1_PROD   = 5922
PDG_V1_SIGNAL = 5923
PDG_ELECTRON  = 11
PDG_POSITRON  = -11
ALL_PDGS      = {PDG_KAON, PDG_CHI, PDG_CHI_PRIME,
                 PDG_V1_PROD, PDG_V1_SIGNAL, PDG_POSITRON, PDG_ELECTRON}

# ── Load parquet ──────────────────────────────────────────────────────────────
parquet_path = args.stem + ".parquet"
df = pd.read_parquet(parquet_path)

if args.diagnose:
    print("=== Columns ===", df.columns.tolist())
    print("=== Shape ===", df.shape)
    for col in df.columns:
        v = df[col].iloc[0]
        print(f"\n--- {col} ---  type={type(v)}  value={v}")
    sys.exit(0)

# ── Weight extraction ─────────────────────────────────────────────────────────
def _scalar(x):
    """Extract single float from any nested structure."""
    if isinstance(x, (int, float, np.floating, np.integer)):
        return float(x)
    try:
        arr = np.asarray(x, dtype=float).flatten()
        return float(arr[0]) if arr.size > 0 else np.nan
    except Exception:
        return np.nan

w_raw       = np.array([_scalar(x) for x in df["event_weight"].values])
finite_mask = np.isfinite(w_raw) & (w_raw > 0)
print("event_weight: %d finite positive / %d total" % (finite_mask.sum(), len(w_raw)))
print("  sample (first 3): %s" % str(w_raw[:3]))

# If all weights are nan (weighter failed), use flat weight = 1/N_events
# This gives correct SHAPES but not absolute normalisation.
# For absolute normalisation, re-run the simulation with fixed weighter.
if finite_mask.sum() == 0:
    print("WARNING: all weights are NaN — using flat weight = 1/N for shape plots")
    print("         Absolute normalisation will be incorrect.")
    print("         Re-run simulation with fixed weighter for correct rates.")
    w_raw       = np.ones(len(df)) / len(df)
    finite_mask = np.ones(len(df), dtype=bool)
    FLUX_CORRECTION = 1.0   # already baked into flat weight
    # Override POT label to indicate unweighted shapes
    _POT_LABEL = "unweighted (shape only)"
else:
    _POT_LABEL = "%.0e POT" % POT

# ── Check kinematics ──────────────────────────────────────────────────────────
sample_pm     = df["primary_momentum"].iloc[0]
has_kine      = hasattr(sample_pm, '__len__') and len(sample_pm) > 0
print("primary_momentum has data:", has_kine)

USE_SIREN = False
if not has_kine:
    siren_path = args.stem + ".siren_events"
    if not os.path.exists(siren_path):
        print("ERROR: no kinematics in parquet and .siren_events not found.")
        sys.exit(1)
    import siren as _siren
    from siren._util import LoadEvents
    events    = LoadEvents(siren_path)
    USE_SIREN = True
    print("Loaded %d events from .siren_events" % len(events))

# ── 4-vector helpers ──────────────────────────────────────────────────────────
def _floats(obj):
    out = []
    if isinstance(obj, (int, float, np.floating, np.integer)):
        out.append(float(obj))
    elif hasattr(obj, '__iter__'):
        for x in obj:
            out.extend(_floats(x))
    return out

def get_4vec(p):
    s = _floats(p)
    if len(s) >= 4: return s[:4]   # [E, px, py, pz]  (5th = helicity, ignored)
    if len(s) == 3: return [0.0] + s
    return None

def E_kin(p):
    v = get_4vec(p); return v[0] if v else np.nan

def cos_th(p):
    v = get_4vec(p)
    if not v: return np.nan
    _, px, py, pz = v
    pm = np.sqrt(px**2 + py**2 + pz**2)
    return float(pz / pm) if pm > 1e-30 else np.nan

# ── Containers ────────────────────────────────────────────────────────────────
# Each species: ([E_all], [E_fid], [w_all], [w_fid])
# We store per-particle weights so secondaries are also correctly normalised.
Eall = {p: [] for p in ALL_PDGS}
Efid = {p: [] for p in ALL_PDGS}
Call = {p: [] for p in ALL_PDGS}
Cfid = {p: [] for p in ALL_PDGS}
Wall = {p: [] for p in ALL_PDGS}
Wfid = {p: [] for p in ALL_PDGS}

m_ee_all, m_ee_fid, m_ee_w_all, m_ee_w_fid = [], [], [], []
n_skip = 0

# ── Fiducial flag helper (parquet) ────────────────────────────────────────────
def _in_fid(raw):
    try:
        if hasattr(raw, '__len__') and len(raw) > 0:
            return any(bool(x) for x in raw)
        return bool(raw)
    except Exception:
        return False

# ── Secondary iterator (parquet) ─────────────────────────────────────────────
def _iter_sec(types_raw, momenta_raw):
    try:
        for bt, bp in zip(types_raw, momenta_raw):
            if hasattr(bt, '__iter__') and not isinstance(bt, (int, float)):
                for t, p in zip(bt, bp):
                    yield int(t), p
            else:
                yield int(bt), bp
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════════════════════
# EVENT LOOP
# ══════════════════════════════════════════════════════════════════════════════
if USE_SIREN:
    print("Processing %d events from .siren_events ..." % len(events))
    try:
        fid_box = _siren.geometry.Box(2.56, 1.58, 8.97)
    except Exception:
        fid_box = None

    for idx, event in enumerate(events):
        if not (idx < len(w_raw) and finite_mask[idx]):
            n_skip += 1;  continue
        w = w_raw[idx] * POT * FLUX_CORRECTION

        in_fid    = False
        ep4, em4  = None, None

        for datum in event.tree:
            rec   = datum.record
            ptype = int(rec.signature.primary_type)
            p4    = list(rec.primary_momentum)
            vtx   = list(rec.interaction_vertex)

            # Update fiducial flag on V1_signal vertex
            if ptype == PDG_V1_SIGNAL and fid_box is not None:
                try:
                    pos = _siren.math.Vector3D(*vtx[:3])
                    in_fid = bool(fid_box.IsInside(pos))
                except Exception:
                    pass

            # Primary of this vertex
            if ptype in ALL_PDGS:
                Eall[ptype].append(E_kin(p4));  Call[ptype].append(cos_th(p4))
                Wall[ptype].append(w)

            # Secondaries of this vertex
            for st, sp in zip(rec.signature.secondary_types, rec.secondary_momenta):
                t = int(st);  p = list(sp)
                if t not in ALL_PDGS: continue
                # For e+/e-: ONLY include those from V1_signal decay (signal)
                # Exclude kaon-decay e+ (parent=K+) from energy/angular plots
                if t in (PDG_POSITRON, PDG_ELECTRON) and ptype != PDG_V1_SIGNAL:
                    continue
                Eall[t].append(E_kin(p));  Call[t].append(cos_th(p))
                Wall[t].append(w)
                # Collect signal e+/e- for invariant mass
                if ptype == PDG_V1_SIGNAL:
                    if t == PDG_POSITRON and ep4 is None: ep4 = p
                    if t == PDG_ELECTRON and em4 is None: em4 = p

        # Fiducial copies
        if in_fid:
            for ptype in ALL_PDGS:
                if Wall[ptype] and Wall[ptype][-1] == w:
                    Efid[ptype].append(Eall[ptype][-1])
                    Cfid[ptype].append(Call[ptype][-1])
                    Wfid[ptype].append(w)

        # Invariant mass
        if ep4 and em4:
            v1, v2 = get_4vec(ep4), get_4vec(em4)
            if v1 and v2:
                Et = v1[0]+v2[0]; pxt=v1[1]+v2[1]; pyt=v1[2]+v2[2]; pzt=v1[3]+v2[3]
                mee = np.sqrt(max(Et**2-pxt**2-pyt**2-pzt**2, 0.0)) * 1e3
                m_ee_all.append(mee);  m_ee_w_all.append(w)
                if in_fid: m_ee_fid.append(mee); m_ee_w_fid.append(w)

else:
    print("Processing %d events from parquet ..." % len(df))
    for i in range(len(df)):
        raw_w = _scalar(df["event_weight"].iloc[i])
        if not (np.isfinite(raw_w) and raw_w > 0):
            n_skip += 1;  continue
        w      = raw_w * POT * FLUX_CORRECTION
        in_fid = _in_fid(df["in_fiducial"].iloc[i])

        # Primary K+
        prim = df["primary_momentum"].iloc[i]
        Eall[PDG_KAON].append(E_kin(prim));  Call[PDG_KAON].append(cos_th(prim))
        Wall[PDG_KAON].append(w)
        if in_fid:
            Efid[PDG_KAON].append(E_kin(prim));  Cfid[PDG_KAON].append(cos_th(prim))
            Wfid[PDG_KAON].append(w)

        ep4, em4 = None, None
        pt_raw = df["primary_type"].iloc[i]

        def _iter_sec_parent(pt_raw, bt_raw, bp_raw):
            try:
                for parent, block_t, block_p in zip(pt_raw, bt_raw, bp_raw):
                    ppd = int(parent) if not hasattr(parent,"__iter__") else int(list(parent)[0])
                    if hasattr(block_t,"__iter__") and not isinstance(block_t,(int,float)):
                        for t, p in zip(block_t, block_p):
                            yield ppd, int(t), p
                    else:
                        yield ppd, int(block_t), block_p
            except Exception:
                pass

        for ppd, t, p in _iter_sec_parent(pt_raw,
                                           df["secondary_types"].iloc[i],
                                           df["secondary_momenta"].iloc[i]):
            if t not in ALL_PDGS: continue
            # For e+/e-: ONLY include those from V1_signal decay (signal)
            # Exclude kaon-decay e+ (parent=K+) from energy/angular plots
            if t in (PDG_POSITRON, PDG_ELECTRON) and ppd != PDG_V1_SIGNAL:
                continue
            Eall[t].append(E_kin(p)); Call[t].append(cos_th(p)); Wall[t].append(w)
            if in_fid: Efid[t].append(E_kin(p)); Cfid[t].append(cos_th(p)); Wfid[t].append(w)
            # Collect signal e+/e- for invariant mass
            if ppd == PDG_V1_SIGNAL:
                if t == PDG_POSITRON and ep4 is None: ep4 = get_4vec(p)
                if t == PDG_ELECTRON and em4 is None: em4 = get_4vec(p)

        if ep4 and em4:
            Et=ep4[0]+em4[0]; pxt=ep4[1]+em4[1]; pyt=ep4[2]+em4[2]; pzt=ep4[3]+em4[3]
            mee = np.sqrt(max(Et**2-pxt**2-pyt**2-pzt**2, 0.0)) * 1e3
            m_ee_all.append(mee);  m_ee_w_all.append(w)
            if in_fid: m_ee_fid.append(mee); m_ee_w_fid.append(w)

# ── Summary ───────────────────────────────────────────────────────────────────
n_used = len(Wall[PDG_KAON])
n_fid  = len(Wfid[PDG_KAON])
w_kaon_all = np.array(Wall[PDG_KAON])
w_kaon_fid = np.array(Wfid[PDG_KAON])
print("Events used    : %d  (skipped %d)" % (n_used, n_skip))
print("Fiducial events: %d" % n_fid)
if n_used:
    print("Total rate     : %.3e events @ %.0e POT (incl. flux corr. x%.2f)"
          % (w_kaon_all.sum(), POT, FLUX_CORRECTION))
if n_fid:
    print("Fiducial rate  : %.3e events" % w_kaon_fid.sum())

# ── Plot style ────────────────────────────────────────────────────────────────
KW = dict(histtype="step", linewidth=2.2)
TITLE_PARAMS = (r"$m_\chi\!=\!8$ MeV, $m_{\chi'}\!=\!50$ MeV, "
                r"$m_{V_1}\!=\!17$ MeV, $m_{V_2}\!=\!200$ MeV, $g_D\!=\!1.0$")
CHANNEL = (r"$K^+\!\to\!e^+\nu_e V_1\!\to\!\chi\bar\chi$, "
           r"$\chi\,\mathrm{Ar}\!\to\!\chi'\mathrm{Ar}$, "
           r"$\chi'\!\to\!\chi V_1^{\rm sig}\!\to\!e^+e^-$")
LEG_EXTRA = [
    Line2D([0],[0], color="gray", lw=1.8, ls="-",  label="All events"),
    Line2D([0],[0], color="gray", lw=1.8, ls="--", label="Fiducial events"),
]
SPECIES = [
    (PDG_KAON,      "gold",   r"Primary $K^+$"),
    (PDG_CHI,       "orange", r"$\chi$ (dark matter)"),
    (PDG_CHI_PRIME, "cyan",   r"$\chi'$ (excited)"),
    (PDG_V1_PROD,   "purple", r"$V_1^{\rm prod}$ (from $K^+$)"),
    (PDG_V1_SIGNAL, "green",  r"$V_1^{\rm sig}$ (from $\chi'$)"),
    (PDG_POSITRON,  "red",    r"$e^+$ (signal, from $V_1^{\rm sig}$)"),
    (PDG_ELECTRON,  "blue",   r"$e^-$ (signal, from $V_1^{\rm sig}$)"),
]

def _arr(d): return np.array(d, dtype=float)

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 1 — Energy spectra
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 6))
EBINS = np.linspace(0, 4.0, 60)

for pdg, color, label in SPECIES:
    ea = _arr(Eall[pdg]);  wa = _arr(Wall[pdg])
    ef = _arr(Efid[pdg]);  wf = _arr(Wfid[pdg])
    if ea.size == 0: continue
    ax.hist(ea, bins=EBINS, weights=wa, color=color, label=label, **KW)
    if ef.size: ax.hist(ef, bins=EBINS, weights=wf, color=color, ls="--", **KW)

ax.set_yscale("log");  ax.set_ylim(bottom=1e-5);  ax.set_xlim(0, 4.0)
ax.set_xlabel("Energy  [GeV]", fontsize=14)
ax.set_ylabel(r"Event rate  (%s)" % _POT_LABEL, fontsize=13)
ax.set_title("ICARUS \u2014 Dutta-Kim Vector Portal  Energy Spectra\n"
             + CHANNEL + "\n" + TITLE_PARAMS, fontsize=9)
h, _ = ax.get_legend_handles_labels()
ax.legend(handles=h+LEG_EXTRA, frameon=False, ncol=2,
          loc="upper right", fontsize=9)
ax.grid(True, which="both", ls="--", alpha=0.25)
fig.tight_layout()
out1 = os.path.join(args.outdir, "ICARUS_VectorPortal_kaon_e_full_EnergySpectra.png")
fig.savefig(out1, dpi=300, bbox_inches="tight");  plt.close(fig)
print("Saved:", out1)

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 2 — Angular distributions
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 6))
CBINS = np.linspace(-1, 1, 50)

for pdg, color, label in SPECIES:
    ca = _arr(Call[pdg]);  wa = _arr(Wall[pdg])
    cf = _arr(Cfid[pdg]);  wf = _arr(Wfid[pdg])
    if ca.size == 0: continue
    ax.hist(ca, bins=CBINS, weights=wa, color=color, label=label, **KW)
    if cf.size: ax.hist(cf, bins=CBINS, weights=wf, color=color, ls="--", **KW)

ax.set_yscale("log");  ax.set_ylim(bottom=1e-5);  ax.set_xlim(-1, 1)
ax.set_xlabel(r"$\cos\theta$  (wrt beam axis)", fontsize=14)
ax.set_ylabel(r"Event rate  (%s)" % _POT_LABEL, fontsize=13)
ax.set_title("ICARUS \u2014 Dutta-Kim Vector Portal  Angular Distributions\n"
             + CHANNEL + "\n" + TITLE_PARAMS, fontsize=9)
h, _ = ax.get_legend_handles_labels()
ax.legend(handles=h+LEG_EXTRA, frameon=False, ncol=2,
          loc="upper left", fontsize=9)
ax.grid(True, which="both", ls="--", alpha=0.25)
fig.tight_layout()
out2 = os.path.join(args.outdir, "ICARUS_VectorPortal_kaon_e_full_AngularDist.png")
fig.savefig(out2, dpi=300, bbox_inches="tight");  plt.close(fig)
print("Saved:", out2)

# ══════════════════════════════════════════════════════════════════════════════
# PLOT 3 — e+e- invariant mass  (key observable — should peak at m_V1=17 MeV)
# ══════════════════════════════════════════════════════════════════════════════
if m_ee_all:
    fig, ax = plt.subplots(figsize=(9, 6))
    MBINS = np.linspace(0, 50, 50)
    wa = np.array(m_ee_w_all);  wf = np.array(m_ee_w_fid)
    ax.hist(m_ee_all, bins=MBINS, weights=wa, color="green",
            histtype="step", lw=2.2, label="All events (weighted)")
    if m_ee_fid:
        ax.hist(m_ee_fid, bins=MBINS, weights=wf, color="green",
                histtype="step", lw=2.2, ls="--", label="Fiducial (weighted)")
    ax.axvline(17.0, color="red", ls=":", lw=1.5,
               label=r"$m_{V_1^{\rm sig}}=17$ MeV")
    ax.set_xlabel(r"$m_{e^+e^-}$  [MeV]", fontsize=14)
    ax.set_ylabel(r"Event rate  (%s)" % _POT_LABEL, fontsize=13)
    ax.set_title("ICARUS \u2014 Dutta-Kim Vector Portal\n"
                 r"$e^+e^-$ invariant mass  ($V_1^{\rm sig}\!\to\!e^+e^-$)"
                 + "\n" + TITLE_PARAMS, fontsize=9)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(True, which="both", ls="--", alpha=0.25)
    fig.tight_layout()
    out3 = os.path.join(args.outdir, "ICARUS_VectorPortal_kaon_e_full_InvMass_ee.png")
    fig.savefig(out3, dpi=300, bbox_inches="tight");  plt.close(fig)
    print("Saved:", out3)
    print("  e+e- pairs: %d all  /  %d fiducial" % (len(m_ee_all), len(m_ee_fid)))

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("  RESULTS  —  ICARUS Vector Portal  K+ -> e+ nu_e V1 -> chi chi")
print("=" * 60)
print("  POT                   : %.2e" % POT)
print("  Flux correction       : x%.4f  (%d / %d kaons)"
      % (FLUX_CORRECTION, args.n_unbiased, args.n_biased))
print("  Events processed      : %d  (skipped %d)" % (n_used, n_skip))
print("  Fiducial events       : %d" % n_fid)
if n_used:
    raw_sum = np.array(Wall[PDG_KAON]).sum() / FLUX_CORRECTION / POT
    print("  Raw weight sum        : %.3e  (before flux corr)" % raw_sum)
    print("  Total signal rate     : %.3e events @ %.0e POT"
          % (w_kaon_all.sum(), POT))
if n_fid:
    print("  Fiducial signal rate  : %.3e events @ %.0e POT"
          % (w_kaon_fid.sum(), POT))
    print("  Fiducial efficiency   : %.1f%%"
          % (100.0 * n_fid / n_used if n_used else 0))
print("  Output dir            : %s" % args.outdir)
print("=" * 60)
