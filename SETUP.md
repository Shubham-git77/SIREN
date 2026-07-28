# Setup on a new machine (`detector-icarus`)

NuMI analytic chain, parameter scans, and MCMC fits for the SBN detectors.

## Build

```bash
git clone --recurse-submodules git@github.com:Shubham-git77/SIREN.git
cd SIREN && git checkout detector-icarus

python3.12 -m venv ~/venv_detector_icarus
~/venv_detector_icarus/bin/pip install -r requirements.txt
~/venv_detector_icarus/bin/pip install -e .
```

`pip install -e .` compiles the C++ extension, so the machine needs CMake and a
C++ compiler. The `--recurse-submodules` is not optional: `vendor/cereal` and
`vendor/photospline` are required to build.

Use one venv per branch checkout. The branches carry different siren sources
(`pr178-bnb-hnl-timing` has `siren.dk2nu`, which this branch does not), so a
venv built here will not run those scripts.

## Environment variables

Every script reads these with a fallback, so nothing needs editing — set only
what you want to override.

| Variable | Used for |
| --- | --- |
| `DK2NU_FILE` | beam file for the multichannel portal scripts (10 scripts) |
| `ICARUS_MODEL_DIR` | ICARUS detector model; defaults to the in-repo `resources/detectors/ICARUS/ICARUS-v1` |
| `FLUX` | flux selection |
| `PORTAL` | portal selection (scalar / pseudoscalar / vector) |
| `N_DEC` | number of decays to generate |
| `EFF_MODE`, `SINGLE_GAMMA_EFF`, `MUON_ONLY` | detector efficiency model |
| `LEVEL` | verbosity |

## Data inputs

The repo carries no beam files. Scripts default to paths under `/home/shubham/`
that will not exist elsewhere; point `DK2NU_FILE` at your own copy, or re-fetch
from the beamline area (`/data/g4numi/*dk2nu*.root`).

Referenced by this branch:

- `nubeamHighSample.dk2nu.root` (844 MB) — the multichannel portal scripts
- `g4numi_fhc_dif_1008.dk2nu.root` (2.5 GB) — the NuMI analytic scripts

The detector models (`ICARUS-v1`, `SBND-v1`) **are** tracked here, so those need
no external data.
