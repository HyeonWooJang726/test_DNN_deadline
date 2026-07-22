"""One-time v0.1-h1 versus regenerated-full regression gate."""

from __future__ import annotations

import argparse
import io
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


OLD_TAG = "v0.1-h1"
POLICY_KEYS = [
    "device",
    "rho",
    "seed",
    "deadline_ratio",
    "epsilon",
    "skip_mode",
    "policy",
]
COMBINATION_KEYS = [
    "device",
    "rho",
    "deadline_ratio",
    "epsilon",
    "skip_mode",
]
UNCHANGED_POLICIES = ("P1", "P2", "P2prime", "P3")
MAX_P0_DERIVED_DELTA_PERCENTAGE_POINTS = 0.3


def _tagged_csv(path: str) -> pd.DataFrame:
    completed = subprocess.run(
        ["git", "show", f"{OLD_TAG}:{path}"],
        check=True,
        stdout=subprocess.PIPE,
    )
    return pd.read_csv(io.BytesIO(completed.stdout), float_precision="round_trip")


def _float_bits(series: pd.Series) -> np.ndarray:
    return series.to_numpy(dtype=np.float64).view(np.uint64)


def _bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series
    return series.astype(str).str.lower().eq("true")


def run_check(results_dir: Path, report_path: Path) -> bool:
    old_policy = _tagged_csv("results/full/policy_runs.csv")
    new_policy = pd.read_csv(
        results_dir / "policy_runs.csv", float_precision="round_trip"
    )
    old_comparison = _tagged_csv("results/full/comparison_aggregate.csv")
    new_comparison = pd.read_csv(
        results_dir / "comparison_aggregate.csv", float_precision="round_trip"
    )
    sanity = pd.read_csv(results_dir / "sanity_checks.csv")
    runtime_failures = pd.read_csv(results_dir / "runtime_failures.csv")

    lines = [
        "H1 v0.1-h1 -> v0.2-h1 regression check",
        f"old_source={OLD_TAG}:results/full",
        f"new_source={results_dir.as_posix()}",
        "",
        "Non-P0 mean_energy_j bitwise comparison:",
    ]
    failures: list[str] = []

    for policy in UNCHANGED_POLICIES:
        old = (
            old_policy.loc[old_policy["policy"] == policy, POLICY_KEYS + ["mean_energy_j"]]
            .sort_values(POLICY_KEYS)
            .reset_index(drop=True)
        )
        new = (
            new_policy.loc[new_policy["policy"] == policy, POLICY_KEYS + ["mean_energy_j"]]
            .sort_values(POLICY_KEYS)
            .reset_index(drop=True)
        )
        keys_equal = old[POLICY_KEYS].equals(new[POLICY_KEYS])
        bitwise_mismatches = (
            int(np.count_nonzero(_float_bits(old["mean_energy_j"]) != _float_bits(new["mean_energy_j"])))
            if keys_equal
            else -1
        )
        passed = keys_equal and bitwise_mismatches == 0
        lines.append(
            f"  {policy}: {'PASS' if passed else 'FAIL'}; rows={len(new)}; "
            f"keys_equal={keys_equal}; bitwise_mismatches={bitwise_mismatches}"
        )
        if not passed:
            failures.append(f"{policy} mean_energy_j changed")

    old_p0 = (
        old_policy.loc[old_policy["policy"] == "P0", POLICY_KEYS + ["mean_energy_j"]]
        .sort_values(POLICY_KEYS)
        .reset_index(drop=True)
    )
    new_p0 = (
        new_policy.loc[new_policy["policy"] == "P0", POLICY_KEYS + ["mean_energy_j"]]
        .sort_values(POLICY_KEYS)
        .reset_index(drop=True)
    )
    p0_keys_equal = old_p0[POLICY_KEYS].equals(new_p0[POLICY_KEYS])
    p0_changed = (
        int(np.count_nonzero(_float_bits(old_p0["mean_energy_j"]) != _float_bits(new_p0["mean_energy_j"])))
        if p0_keys_equal
        else -1
    )
    p0_passed = p0_keys_equal and p0_changed > 0
    lines.extend(
        [
            "",
            "P0 intentional RNG-entropy change:",
            f"  P0: {'PASS' if p0_passed else 'FAIL'}; rows={len(new_p0)}; "
            f"keys_equal={p0_keys_equal}; changed_rows={p0_changed}",
            "",
            "P0-derived aggregate metric deltas (new-old, percentage points):",
        ]
    )
    if not p0_passed:
        failures.append("P0 did not show the intended isolated change")

    old_indexed = old_comparison.set_index(COMBINATION_KEYS).sort_index()
    new_indexed = new_comparison.set_index(COMBINATION_KEYS).sort_index()
    comparison_keys_equal = old_indexed.index.equals(new_indexed.index)
    if not comparison_keys_equal:
        failures.append("comparison_aggregate combination keys changed")
    for metric in ("discard_gain_percent_mean", "targeting_gain_percent_mean"):
        if comparison_keys_equal:
            delta = new_indexed[metric] - old_indexed[metric]
            absolute = delta.abs()
            max_delta = float(absolute.max())
            worst_key = tuple(
                value.item() if isinstance(value, np.generic) else value
                for value in absolute.idxmax()
            )
            passed = max_delta <= MAX_P0_DERIVED_DELTA_PERCENTAGE_POINTS + 1e-12
        else:
            max_delta = float("nan")
            worst_key = "keys differ"
            passed = False
        lines.append(
            f"  {metric}: {'PASS' if passed else 'FAIL'}; "
            f"max_abs_delta_pp={max_delta:.12g}; worst_combination={worst_key}"
        )
        if not passed:
            failures.append(f"{metric} exceeded +/-0.3 percentage points")

    severity = sanity["severity"].fillna("")
    required_failures = sanity[
        severity.eq("required") & ~_bool_series(sanity["passed"])
    ]
    lines.extend(
        [
            "",
            "Sanity/runtime gate:",
            f"  required sanity failures: {'PASS' if required_failures.empty else 'FAIL'}; "
            f"count={len(required_failures)}",
            f"  runtime-invalid rows: {'PASS' if runtime_failures.empty else 'FAIL'}; "
            f"count={len(runtime_failures)}",
        ]
    )
    if not required_failures.empty:
        failures.append("required sanity failures are present")
    if not runtime_failures.empty:
        failures.append("runtime-invalid rows are present in the full run")

    passed = not failures
    lines.extend(
        [
            "",
            f"OVERALL: {'PASS' if passed else 'FAIL'}",
        ]
    )
    if failures:
        lines.append("Failures:")
        lines.extend(f"  - {failure}" for failure in failures)

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return passed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results/full"))
    parser.add_argument(
        "--report", type=Path, default=Path("regression_check.txt")
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if not run_check(arguments.results, arguments.report):
        raise SystemExit(1)
