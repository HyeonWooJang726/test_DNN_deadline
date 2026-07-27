# P3 V-grid and budget-cap impact

## Scope and method

This report uses the committed full-run design (`T=100,000`, ten seeds) and
the same deterministic channel traces for every comparison. The v0.2 result
is read from the pre-change full CSV. Four counterfactual P3 variants were
then evaluated on all 1,680 valid seed-condition rows:

| Variant | V grid | Cap used by the threshold mask | V selection |
|---|---|---|---|
| Legacy/old | `{10,100,1000,10000}` | Total running violations `< floor(εT)` | Minimum energy within `ε+0.005` |
| Legacy/new | New 15-point grid | Total running violations `< floor(εT)` | Minimum energy within `ε+0.005` |
| Fixed/old | `{10,100,1000,10000}` | Selected violations `< floor(εT)-forced_count` | Minimum energy within `ε+0.005` |
| Fixed/new | New 15-point grid | Selected violations `< floor(εT)-forced_count` | Minimum energy within `ε+0.005` |

The fixed/new counterfactual reproduces the final `comparisons.csv` recovery
values to a maximum absolute error of `5.4e-14`. Effect summaries below use
seed-averaged condition rows. They include positive-gap conditions below the
2% plotting mask so that the algorithmic effects are evaluated on the full
valid set; README and figure ranges apply the normal mask.

## Tight high-correlation example

The following rows use `late`, `rho=0.975`, and `epsilon=0.15`. Values are
online recovery percentages.

| D/D_min | v0.2 original | Legacy/old | Legacy/new | Fixed/old | Fixed/new (final) |
|---:|---:|---:|---:|---:|---:|
| 1.1 | 58.03 | 97.00 | 99.48 | 95.74 | 96.52 |
| 1.2 | 62.90 | 94.90 | 98.30 | 93.99 | 96.84 |

The large difference between v0.2 and Legacy/old is B2's corrected
energy-first selection criterion. Comparing columns within the 2×2 grid
separates B1 and B3 without conflating them with B2.

## Factorized effects on recovery

All values are percentage-point changes in recovery.

| Change | Mean | Minimum | Maximum | Condition at largest magnitude |
|---|---:|---:|---:|---|
| B2: new selection vs v0.2 (legacy cap, old grid) | +9.22 | 0.00 | +50.43 | `drop`, D/D_min=1.2, ε=0.10, rho=0.975 |
| B1: old to new grid with legacy cap | +0.45 | 0.00 | +3.62 | `late`, D/D_min=1.2, ε=0.10, rho=0.975 |
| B1: old to new grid with fixed cap | +0.28 | 0.00 | +2.84 | `late`, D/D_min=1.2, ε=0.15, rho=0.975 |
| B3: legacy to fixed cap with old grid | -0.40 | -2.48 | 0.00 | `drop`, D/D_min=1.1, ε=0.15, rho=0.875 |
| B3: legacy to fixed cap with new grid | -0.57 | -4.25 | 0.00 | `late`, D/D_min=1.2, ε=0.10, rho=0.975 |

B1 and B3 act in opposite directions as expected. B2 is the dominant change
in the tight/high-correlation region because the old rate-distance ordering
could reject a much lower-energy candidate for a negligible rate difference.

The legacy cap exceeded `floor(εT)` in 1,078 of 1,680 rows with the old grid
and 1,084 rows with the new grid. Its maximum excesses were 462 and 499
violations, respectively. This also explains counterfactual recovery values
above 1.0. Under the fixed cap, every final row has zero excess; per-seed
recovery is 95.12–99.75%, aggregate recovery is 96.34–99.72%, and no recovery
value exceeds 1.0.

## B4 burn-in

B4 changes only post-processing of burst statistics, so its recovery impact
is exactly zero. The full-run burn-in is `max(200, T//100)=1,000` slots and is
applied uniformly to every policy. At the representative P2/late condition,
the mean count of violation runs of length at least two changes as follows:

| rho | Before burn-in | After burn-in | Maximum run before/after |
|---:|---:|---:|---:|
| 0 | 244.0 | 240.9 | 3.4 / 3.4 |
| 0.975 | 847.2 | 837.9 | 6.8 / 6.8 |

Thus the warm-up correction removes early-run bias without changing energy,
violation rate, total violation count, or recovery.
