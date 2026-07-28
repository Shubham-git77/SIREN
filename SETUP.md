# Setup on a new machine (`shubham/sbnd-timing-study`)

SBND and dirt timing study, built on the `PrimaryExternalDistribution` initial-time
(`t0`) feature on this branch.

## Build

```bash
git clone --recurse-submodules git@github.com:Shubham-git77/SIREN.git
cd SIREN && git checkout shubham/sbnd-timing-study

python3.12 -m venv ~/venv_sbnd_timing
~/venv_sbnd_timing/bin/pip install -r requirements.txt
~/venv_sbnd_timing/bin/pip install -e .
```

`pip install -e .` compiles the C++ extension, so the machine needs CMake and a
C++ compiler. `--recurse-submodules` is required (`vendor/cereal`,
`vendor/photospline`).

**This branch needs its own venv.** The `t0` support is a C++ change in
`PrimaryExternalDistribution`, so a venv built from another branch will silently
lack it — the `t0` CSV column is ignored and every event comes out prompt. An
editable install does not save you here: after any C++ change, rerun
`pip install -e .` to rebuild.

## Environment variables

| Variable | Used for |
| --- | --- |
| `SBND_N_EVENTS` | event count for `sbnd.py` / `sbnd_dirt.py` |
| `SBND_SAVE_EVENTS` | write events to disk |
| `PYTHIA8DATA` | Pythia8 share directory |
| `LHAPDF_DATA_PATH` | LHAPDF set location |
| `SIREN_CHARM_SPLINE_DIR` | charm spline tables |

## Data inputs

Two stages, and only the first needs beam files:

```bash
# 1. dk2nu -> external-distribution CSV (needs the beam file)
python dk2nu_to_upscattering_csv.py --dk2nu /path/to/g4numi_fhc_dif_*.dk2nu.root

# 2. everything downstream reads the CSV, no beam file required
python sbnd.py
```

So if you carry `pion_derived_upscattering_events.csv` (8.9 MB) you can skip the
multi-GB beam files entirely. Re-derive only if you change cuts, POT, the meson
selection, or the DIF/DAR split. Beam files are re-fetchable from `/data/g4numi/`.

Generated outputs (`dn_tables/`, `timing_out/`, the derived CSV) are untracked.

## CSV columns

`PrimaryExternalDistribution` recognizes `x0/y0/z0`, `x/y/z`, `px/py/pz`, `E`,
`m`, and `t0`. `t0` is the primary initial time in SIREN units — **nanoseconds**
(1 s = 1e9) — propagated to the vertex by time of flight.
