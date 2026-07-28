# Dark-portal analytic analysis — code structure

Two layers, mirroring the official SIREN examples (heavy machinery in the
package, thin run scripts in the example):

```
resources/processes/DarkNewsTables/   <-- REUSABLE LIBRARY (imported)
resources/examples/example4/          <-- THIN RUN SCRIPTS + configs (run these)
```

The authoritative rate is the analytic sigma*N*chord estimator (NOT SIREN's
directed sampler, which over-estimates 60-400x).

---

## Package library — `resources/processes/DarkNewsTables/`

| module | provides |
|--------|----------|
| `MesonProduction.py` | meson 3-body decay `M -> l nu phi`, Carlson-Rislow matrix element (scalar / pseudoscalar / **vector**). |
| `VectorPortal.py`    | vector double-mediator cascade: `V1->chi chi`, chi upscattering, chi' decay. |
| `DarkPrimakoff.py`   | scalar/pseudo dark-Primakoff scatter `phi N -> gamma N` (Z^2, Helm FF) **+ `smear_photon_beam` / `sample_cos_star`** (outgoing-photon angle). |
| `Dk2nuReader.py`     | dk2nu flux reading; `beam_transform` for external distributions; **`analytic_meson_source(files, pdg, beam_transform)`** — parent mesons in the world frame for the engine (NuMI->BNB transform applied here). |
| `AnalyticRate.py`    | **the engine** — `analytic_sp` / `analytic_vec` (box detectors) and `analytic_sp_mb` / `analytic_vec_mb` (MiniBooNE sphere); ray-traces sigma*N*chord, applies efficiency modes (`raw`/`mb`/`lartpc`). Consumes a portal-config object `S`. |

## Example layer — `resources/examples/example4/`

**Run drivers (thin):**
- `analytic_NuMI_ICARUS.py` — ICARUS x NuMI, all portals. Reads g4numi via
  `Dk2nuReader.analytic_meson_source` (NuMI->BNB transform), sums the two
  cryostats, anchors. `--portal all --n-dec N --anchored`.
- `analytic_NuMI_SBN.py` — generalizes to SBND (single box) / MiniBooNE
  (carbon sphere) at their off-axis NuMI positions. `--detector SBND ...`.

**BNB plots + anchor helper:**
- `plot_sbnd_analytic.py` — SBND plots **and** the anchoring helper the drivers
  import: `mb_inwindow` (MiniBooNE denominator), `_get_primakoff`, `load_portal`,
  constants (`SINGLE_GAMMA_EFF=0.10`, `WIN_LO/HI`, `MB_EXCESS`). Re-exports
  `smear_photon_beam` from `DarkPrimakoff`.
- `plot_icarus_analytic.py`, `plot_miniboone_analytic.py` — BNB ICARUS / MiniBooNE.
- `plot_fhc_vs_rhc.py` — neutrino- vs antineutrino-mode comparison bar chart.

**Portal config modules (9)** — `{Scalar,Pseudoscalar}Portal_{ICARUS,SBND,MiniBooNE}_multichannel.py`
and `VectorPortal_{ICARUS,SBND,MiniBooNE}_fullchain.py`. Each is the `S` object:
model params, `CHANNELS`, `build_onshell_models`, detector box/POT, and the two-
cryostat `ICARUS_MODULE_CENTERS_BNB` for ICARUS.

**Diagnostics / regression:**
- `_verify.py` — deterministic engine regression (fixed seed + fixed synthetic
  kaons). Must reproduce: PSEUDO sum(w)=1.1025700832e+11, VECTOR sum(w)=1.6308848208e+07.
- `_vdiag.py` — vector anchor-denominator scatter check.

**Reference (kept, non-authoritative):** `DuttaKim_SBND_full_chain.py`,
`VectorPortal_ICARUS_NuMI_dk2nu.py` (SIREN directed sampler).

`archive/` — 41 superseded experiment/diagnostic scripts. `output/` — plots + npz.
`sources/` — flux .root files (NuMI RHC/FHC, single/multi).

---

## Import wiring (how a run resolves the library)

```
analytic_NuMI_ICARUS.py
  SA  = load_module("AnalyticRate", <pkg>/AnalyticRate.py)      # engine
  from plot_sbnd_analytic import mb_inwindow, SINGLE_GAMMA_EFF, ...   # anchor helper
  GEO = load_module("sbn_geometry", <detectors>/SBN-v1/sbn_geometry.py)  # NuMI->BNB
  numi_meson_fn -> S._DK.analytic_meson_source(files, pdg, beam_transform=T)  # flux+transform (pkg)
  portal config S = load_module(<portal>_multichannel.py)       # example-side config
      S._DK  -> Dk2nuReader (pkg),  S.build_onshell_models -> MesonProduction/DarkPrimakoff/VectorPortal (pkg)
```

Package modules are loaded from the installed venv copy; example scripts run from
this directory. After editing a package module, sync it to
`siren_venv/.../resources/processes/DarkNewsTables/`.

## Run examples

```bash
cd resources/examples/example4
PY=/home/shubham/siren_venv/bin/python

# NuMI -> ICARUS, all portals, anchored (5-file RHC)
DK2NU_FILE=/home/shubham/nubeam12M.dk2nu.root \
NUMI_DK2NU_GLOB="$PWD/sources/NuMI/g4numi*.root" ICARUS_NUMI_POT=3.0e21 \
  $PY analytic_NuMI_ICARUS.py --portal all --anchored

# NuMI -> SBND / MiniBooNE
$PY analytic_NuMI_SBN.py --detector SBND --portal all --anchored

# regression check (must reproduce the two baseline numbers)
DK2NU_FILE=/home/shubham/nubeamHighSample.dk2nu.root $PY _verify.py
```
