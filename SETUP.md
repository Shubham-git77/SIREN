# Setup on a new machine (`shubham/pr178-bnb-hnl-timing`)

SBND and ICARUS BNB timing studies for the DarkNews HNL chain, built on top of
the Dutta-Kim vector-portal stack in PR #178.

## Build

```bash
git clone --recurse-submodules git@github.com:Shubham-git77/SIREN.git
cd SIREN && git checkout shubham/pr178-bnb-hnl-timing

python3.12 -m venv ~/venv_pr178
~/venv_pr178/bin/pip install -r requirements.txt
~/venv_pr178/bin/pip install -e .
```

`pip install -e .` compiles the C++ extension, so the machine needs CMake and a
C++ compiler. `--recurse-submodules` is required (`vendor/cereal`,
`vendor/photospline`).

**This branch needs its own venv.** The scripts import `siren.dk2nu`, which only
exists on the PR178 stack. A venv built from any other branch fails at import
for every script in `example4`, the upstream reference ones included. If you see
`ImportError: cannot import name 'dk2nu'`, the venv was built from the wrong
checkout.

## Environment variables

| Variable | Used for |
| --- | --- |
| `PYTHIA8DATA` | Pythia8 share directory |
| `LHAPDF_DATA_PATH` | LHAPDF set location |
| `SIREN_CHARM_SPLINE_DIR` | charm spline tables |
| `SIREN_PYTHIA_PDF`, `SIREN_PYTHIA_WIDE_SIGMA`, `SIREN_PYTHIA_WIDE_DSDXDY` | Pythia cross-section overrides |
| `SIREN_STRICT` | fail instead of warning on recoverable errors |

## Data inputs

Beam files are not in the repo. The timing scripts take them as arguments:

```bash
~/venv_pr178/bin/python DarkNewsHNL_SBND_BNB_timing.py --dk2nu-dir '/data/g4bnb/*dk2nu*.root'
```

`DEFAULT_FLUX` in `DarkNewsHNL_{SBND,ICARUS}_BNB_timing.py` points at
`nubeam12M.dk2nu.root` (9.9 GB) — pass `--dk2nu-dir` rather than relying on it.
Equivalent files live in the beamline areas `/data/g4bnb/` and `/data/g4numi/`.

Outputs (`logs/`, generated tables) are deliberately untracked and regenerate
from the scripts.
