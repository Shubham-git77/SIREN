# Setup on a new machine (`interface-redesign-wip`)

**This is a WIP backup branch, not a reviewed state.** It is a snapshot of
in-progress work on the Layer-2 interface redesign, pushed so it exists somewhere
other than one disk. Expect rough edges.

## Build

```bash
git clone --recurse-submodules git@github.com:Shubham-git77/SIREN.git
cd SIREN && git checkout interface-redesign-wip

python3.12 -m venv ~/venv_interface
~/venv_interface/bin/pip install -r requirements.txt
~/venv_interface/bin/pip install -e .
```

`pip install -e .` compiles the C++ extension, so the machine needs CMake and a
C++ compiler. `--recurse-submodules` is required (`vendor/cereal`,
`vendor/photospline`).

Use one venv per branch checkout — this branch's siren source differs from the
others.

## State of this branch

- It is **167 commits behind** `Harvard-Neutrino/SIREN:interface-redesign` as of
  2026-07-16. Upstream rebased, so 148 of the commits this branch appears to be
  "ahead" by are stale duplicates of upstream work. Only three commits are
  genuinely local: `f58559c0`, `b919fd81` (NuMI `beam_transform` in
  `Dk2nuReader`), `7c283cea`. Rebasing those three onto current upstream is the
  real catch-up job; expect conflicts in `example4`, which upstream rewrote.
- `vendor/cereal` and `vendor/photospline` pointer bumps were **left
  uncommitted** on purpose — they point at submodule commits that may not be
  pushed anywhere.
- `example4/archive/` holds 41 retired scripts (moved, not deleted). Two of them
  still `import sbnd_analytic`, which was renamed to
  `resources/processes/DarkNewsTables/AnalyticRate.py` — fix the import if you
  revive them.
- Detector models are reorganized under `resources/detectors/SBN/` on this
  branch; there is no `ICARUS/ICARUS-v1`. The archived ICARUS scripts therefore
  need `ICARUS_MODEL_DIR` pointed somewhere real.

## Environment variables

| Variable | Used for |
| --- | --- |
| `DK2NU_FILE` | beam file for the multichannel portal scripts (11 scripts) |
| `ICARUS_MODEL_DIR` | ICARUS detector model (see above — not in-repo on this branch) |
| `FLUX` | flux selection |
| `PORTAL` | portal selection (scalar / pseudoscalar / vector) |
| `N_DEC` | number of decays to generate |
| `EFF_MODE`, `SINGLE_GAMMA_EFF`, `MUON_ONLY` | detector efficiency model |
| `LEVEL` | verbosity |

## Data inputs

Not in the repo, and large. `example4/sources/` (3.9 GB of NuMI root files) and
`resources/fluxes/Mesons/` (112 MB) are untracked. Point `DK2NU_FILE` at your own
copy or re-fetch from `/data/g4numi/`.
