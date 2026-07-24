> **STALE (pre-parameter-update).** Generated under the previous boost/channel parameters. Regenerate after this update.
# Cross-seed robustness of P3 V calibration

## Method

This check uses `T=10,000`, seeds 1701–1705, the configured 15-point
P3 V grid, and `violation_tolerance=0.005`. For each evaluation seed
`s_i`, post-hoc recovery uses the V calibrated on `s_i`; cross-seed
recovery uses the V calibrated on `s_(i+1) mod 5`. Only V transfers.
Both evaluations recompute `floor(epsilon*T)-forced_count` from the
evaluation trace. P1, P2, channel generation, slot costs, and P3's
threshold mask are imported from the simulator modules.

The gap is post-hoc recovery minus cross-seed recovery, in percentage
points. V match is the fraction of the five transfers whose source V
equals the evaluation seed's post-hoc V.

## Results

| rho | D/D_min | epsilon | skip | Post-hoc mean | Cross-seed mean | Gap (pp) | V match |
|---:|---:|---:|---|---:|---:|---:|---:|
| 0 | 1.1 | 0.15 | drop | 98.7696% | 98.7451% | +0.0245 | 60.0% |
| 0 | 1.1 | 0.15 | late | 97.9295% | 97.9295% | +0.0000 | 100.0% |
| 0 | 1.2 | 0.15 | drop | 99.1622% | 99.0546% | +0.1076 | 20.0% |
| 0 | 1.2 | 0.15 | late | 98.6421% | 98.6328% | +0.0094 | 60.0% |
| 0 | 1.5 | 0.05 | drop | 97.8876% | 97.3972% | +0.4903 | 40.0% |
| 0 | 1.5 | 0.05 | late | 97.7828% | 97.4890% | +0.2939 | 20.0% |
| 0.75 | 1.1 | 0.15 | drop | 97.3017% | 96.8857% | +0.4160 | 20.0% |
| 0.75 | 1.1 | 0.15 | late | 96.0043% | 95.8927% | +0.1115 | 40.0% |
| 0.75 | 1.2 | 0.15 | drop | 97.3980% | 97.3980% | +0.0000 | 100.0% |
| 0.75 | 1.2 | 0.15 | late | 96.6443% | 96.6443% | +0.0000 | 100.0% |
| 0.75 | 1.5 | 0.05 | drop | 95.3256% | 94.8719% | +0.4537 | 40.0% |
| 0.75 | 1.5 | 0.05 | late | 95.5756% | 95.0771% | +0.4986 | 40.0% |
| 0.975 | 1.1 | 0.15 | drop | 90.0505% | 89.6926% | +0.3579 | 60.0% |
| 0.975 | 1.1 | 0.15 | late | 88.7275% | 88.4944% | +0.2331 | 60.0% |
| 0.975 | 1.2 | 0.15 | drop | 90.0415% | 89.2084% | +0.8330 | 20.0% |
| 0.975 | 1.2 | 0.15 | late | 89.6953% | 88.9183% | +0.7770 | 60.0% |
| 0.975 | 1.5 | 0.05 | drop | 89.1090% | 88.6897% | +0.4193 | 60.0% |
| 0.975 | 1.5 | 0.05 | late | 90.5003% | 89.8677% | +0.6326 | 60.0% |

## Summary

- Maximum absolute gap: **0.8330 pp** at `rho=0.975, D/D_min=1.2, epsilon=0.15, skip=drop`.
- Every tested condition has `|gap| < 1 pp`: **yes**.
