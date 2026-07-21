# Energy-efficient DNN split-offloading simulator

This package measures the value of reallocating a long-run deadline-violation
budget over time. It implements P1, random-budget P0, offline P2, a
no-consecutive-violation P2′, and virtual-queue P3 over a Gilbert–Elliott or CSV
channel trace.

## Run

```powershell
python -m pip install -r requirements.txt
python -m pytest
python run_sweep.py --mode quick
python run_sweep.py --mode full
```

`quick` uses T=10,000 and three seeds while retaining every requested sweep
axis. `full` uses T=100,000 and ten seeds. Outputs go to `results/quick` or
`results/full`; use `--output PATH` to override. Sanity assertions are strict by
default and can be collected without stopping via `--no-strict-sanity`.

## Output interpretation

- `preflight.csv` retains invalid combinations and the exact exclusion reason.
- `policy_runs.csv` and `policy_aggregate.csv` contain per-seed and mean±SD
  policy metrics.
- `comparisons.csv` contains the P1−P0 discard gain, P0−P2 pure temporal
  targeting gain, online oracle recovery, and P2′ constraint cost.
- violation patterns, interval histograms, channel statistics, sanity results,
  stable hashes, JSON parameters, and all requested PNG plots are saved beside
  them.

The specified transition equations imply lag-1 correlation −0.25 at L=1 when
πB=0.2, rather than exactly zero. The simulator preserves the equations and
records this fact in both tests and run metadata. The large-T P2′ path uses the
requested Lagrangian relaxation; the T=20 validation path is an exact
cardinality DP. Since burst-capable P3 and no-adjacent P2′ optimize over
different feasible sets, `P2prime <= P3` is recorded as a diagnostic rather
than a required theorem; all other requested order/budget assertions are
strict. P3 also uses a known-horizon terminal budget cap in addition to the
specified virtual-queue threshold, ensuring its finite-run rate cannot exceed
`floor(epsilon*T)/T` on valid traces.
