# Verification against Dutta & Kim, arXiv:2110.11944

Status of the three portals against the paper's Table I benchmarks, using the
**analytic engine** (`AnalyticRate.analytic_sp_mb` / `analytic_vec_mb`) driven by
the real dk2nu BNB flux. Numbers are MiniBooNE, muon channels only.

Everything here is reproducible: `python <config>.py --engine analytic`.

## Bottom line

| portal | channels | our result | paper Table I | verdict |
|---|---|---|---|---|
| scalar | muon only | 523 events | 2.2e-08 | **reproduced, ratio 0.96** |
| vector | ALL FOUR | 547.4 events | 1.3e-07 | **reproduced, ratio 1.000** (needs the full coupling set) |
| pseudoscalar | muon only | needs 7.98e-08 | 5.9e-07 | **paper 7.4x high** |

WHICH CHANNELS depends on the portal and getting it wrong is a factor 2.5:
  * (pseudo)scalar -- the paper's model is g_e = 0, so ONLY K_mu and pi_mu count.
  * vector -- a kinetically mixed dark photon couples eps*e to every charged
    lepton, so K_e and pi_e MUST be included. K_e is in fact the largest vector
    channel (338 of 547 events). Applying the scalar's muon-only cut to the
    vector under-counts by 2.49x, which is how an earlier version of this file
    claimed ratio 1.009 for a benchmark that was really 2.49x high.
`plot_analytic.py --channels auto` (the default) picks the right set per portal.

The measured MiniBooNE excess is 547.3 events (200-1250 MeV) / 533.9 (200-3000),
from HEPData ins1804293 -- see `miniboone_data.py`.

## Compare muon channels only

The paper's model is **muon-only, g_e = 0**. These configs run g_e = g_mu, which
adds `K_e` and `pi_e`. Because pi -> e nu phi is helicity-UNsuppressed, `pi_e`
becomes the largest single channel and inflates the total 2.47x. Comparing the
all-channel total against the paper is the single easiest mistake to make here;
`plot_analytic.py` therefore defaults to muon-only and draws the electron
channels dashed.

## Vector: Table I's product does not determine the rate

Measured scaling (doubling each coupling in turn, exact to 0.01):

    rate  ~  eps1^2 * eps2^2 * g_D^2

Table I quotes only `P = eps1 * eps2 * g'^2/(4pi)`. Eliminating eps1*eps2 at
fixed P gives `rate ~ (4 pi P)^2 / g_D^2` -- **inversely** proportional to the
dark coupling. So the same quoted product spans orders of magnitude in rate:

| eps1 | eps2 | alpha_D | P | events |
|---|---|---|---|---|
| 7.0e-05 | 3.71e-03 | 0.500 | 1.3e-07 | 1.0 |
| 1.206e-02 | 1.206e-02 | 8.94e-04 | 1.3e-07 | **552.2** |

Both are the paper's Table I product; they differ by 550x in prediction. Taking
alpha_D = 0.5 by convention lands at the wrong end. Selectable via
`VECTOR_BENCHMARK` (`tableI_fit` is the default, and is the row that reproduces
the excess).

TRAP: `G_MU = EPSILON_1` is a module-level snapshot. Changing `EPSILON_1` without
also setting `G_MU` leaves production untouched -- the measured eps1 exponent
comes out 0.00 instead of 2.00.

## Pseudoscalar: no equivalent freedom, and the deficit is not ours

All three couplings scale with exponent exactly 2.00, so the rate depends on the
single product `(g_mu g_n lambda)^2`. There is no split to choose. Everything
upstream was checked against the paper itself:

- **Production BR vs the paper's own Table II** -- scalar K_mu 1.07, pi_mu 1.01;
  pseudo K_mu 1.06, pi_mu 1.00. All within 7%.
- **Dark Primakoff cross-section at matched parameters** -- pseudo/scalar =
  1.000000 at every energy tested, confirming numerically that the two matrix
  elements are identical on a real photon.
- The sigma ratio at the two benchmarks is 106 against a coupling-ratio^2 of 218;
  the factor 2 is the heavier mediator (85 vs 49 MeV). Self-consistent.

Two independent methods agree on the coupling that *does* fit:

| method | product |
|---|---|
| direct event count | 7.98e-08 |
| brute-grid chi2 fit, 450 points | 7.66e-08 |
| agreement | 1.042 |

against Table I's 5.9e-07. This is not a tuning artefact: the paper's Fig. 3
pseudo blob was pixel-digitised at 5.68e-07, so the paper's own table and figure
agree with each other and disagree with the physics.

## Vector MCMC

`mcmc_fit_vector.py` fits (m_V2, P) against the E_vis data. Its answer depends
entirely on the coupling split its response grid was tabulated at:

| response grid built at | paper point (200 MeV, 1.3e-7) |
|---|---|
| alpha_D = 8.94e-4 (mis-tuned on muon channels) | OUTSIDE 95% |
| **alpha_D = 2.229e-3 (`tableI_fit`)** | **INSIDE 68%, enters at 66.2%** |

Do NOT read the 1D marginal as the comparison. It reports m_V2 = 1022 +682/-730
MeV and P = 2.49e-6, which looks 19x off -- but m_V2 is only weakly constrained,
so with a flat 60-2000 MeV prior the marginal drifts to high mass and the product
follows it up the mass-coupling correlation. The 2D containment is the meaningful
statement, and it accepts the paper point. Same effect as the scalar MCMC.

Note also its cos-theta term evaluates to exactly 93.750 at EVERY mass -- a
constant offset with zero leverage, so `COS_WEIGHT=off` changes nothing and this
is effectively an E_vis-only fit.

## Which engine to trust

`--engine analytic` is authoritative. `--engine siren` runs the directed
importance sampler, which over-estimates 60-400x through an uncancelled
production boost-Jacobian. Measured at MiniBooNE: analytic 534 events at the
paper's scalar coupling against a measured excess of 534; the sampler gives
5.2e4 for the same physics -- 109x high.

## Two environment traps that hid real edits

1. The configs loaded `AnalyticRate` / `MesonProduction` from the **installed**
   `site-packages/siren/resources`, not this source tree, so edits to those
   modules silently had no effect. All configs now honour `SIREN_DNT_DIR`.
2. `analytic_*`'s `meson_fn=None` falls back to the **synthetic** BNB flux, not
   dk2nu -- a 13x normalisation difference at MiniBooNE. `report_detector`
   passes `_mesons_dk2nu` explicitly, as the fitting scripts always did.

Consider `pip install -e` on the source tree so the two copies cannot diverge.
