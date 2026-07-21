# Energy-efficient DNN split-offloading simulator

This package measures the value of reallocating a long-run deadline-violation
budget over time. It implements P1, random-budget P0, offline P2, a
no-consecutive-violation P2-prime, and virtual-queue P3 over a two-state
Gilbert-Elliott or CSV channel trace.

The local action space is the product of DNN split point and local execution
mode. The default profile provides `normal` and a faster, higher-energy
`boost` mode. `D_min` is intentionally defined using normal mode only. The
synthetic channel is parameterized directly by lag-1 state autocorrelation
`rho` and applies independent clipped-lognormal rate jitter within each state.

## Run

```powershell
python -m pip install -r requirements.txt
python -m pytest
python run_sweep.py --mode smoke
python run_sweep.py --mode quick
python run_sweep.py --mode full
```

- `smoke`: T=10,000, two seeds, rho in `{0, 0.75, 0.975}`.
- `quick`: T=10,000, three seeds, full rho/deadline/epsilon grid.
- `full`: T=100,000, ten seeds, full grid.

Outputs go to `results/<mode>`; use `--output PATH` to override. Sanity
assertions are strict by default and can be collected without stopping via
`--no-strict-sanity`. Plotting can be disabled with `--no-plots`.

## Main outputs

- `policy_runs.csv`, `policy_aggregate.csv`: policy energy, violation, burst,
  selected-V, and boost-use metrics.
- `comparisons.csv`, `comparison_aggregate.csv`: discard gain, temporal
  targeting gain, offline oracle gap, online recovery, and P2-prime cost.
- `preflight.csv`: invalid combinations and exact exclusion reasons under the
  expected forced-violation `< epsilon/2` rule.
- `channel_stats.csv`: state occupancy, specified/observed rho, and jitter
  P10/P50/P90.
- `saving_diagnostics.csv`: saving P10/P50/P90 and unique-value counts.
- `mode_usage.csv`: local-mode use by policy and channel state.
- `deadline_axis_diagnostics.csv`, `diagnostic_warnings.csv`: inactive-axis,
  degenerate-saving, and unused-boost diagnostics.
- `smoke_acceptance.csv`: the four smoke acceptance checks.
- `sanity_checks.csv`, `reproducibility_hashes.csv`: invariant and fixed-seed
  reproducibility checks.
- `run_parameters.json`: complete configuration and derived normal-mode
  `D_min` values.

PNG figures beside the CSV files visualize the energy decomposition, deadline
heatmaps, rho-dependent online recovery and burstiness, the P2/P2-prime gap,
and the oracle-gap sanity diagnostic.
