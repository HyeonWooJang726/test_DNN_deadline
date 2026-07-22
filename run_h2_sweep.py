"""CLI for H2 multi-device shared-server sweeps."""

from __future__ import annotations

import argparse
import json
import math
import platform
import re
import shutil
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from config import DeviceConfig
from h2_channel import generate_h2_trace, generate_h2_traces, trace_hash
from h2_config import (
    H2ExperimentConfig,
    H2_PREREGISTRATION,
    default_h2_experiment,
    default_h2_output_dir,
)
from h2_plotting import create_h2_plots
from h2_simulator import (
    h2_decomposition_metrics,
    h2_preflight_check,
    h2_sanity_rows,
    independent_policy_sets,
    minimum_h2_deadline_ms,
    simulate_h2_trace,
    stable_h2_digest,
)
from metrics import assert_sanity, summarize_policy
from simulator import simulate_trace


H2_CHECKPOINT_VERSION = 1
H2_CHECKPOINT_TABLES = (
    "policy_rows",
    "decomposition_rows",
    "sanity_rows",
    "digest_rows",
    "runtime_rows",
    "runtime_failure_rows",
)
H2_RUNTIME_FAILURE_COLUMNS = [
    "N",
    "channel_sync",
    "rho",
    "seed",
    "deadline_ratio",
    "deadline_ms",
    "epsilon",
    "skip_mode",
    "status",
    "exception_type",
    "exception_message",
]


def _aggregate(
    frame: pd.DataFrame, group_columns: list[str], value_columns: list[str]
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    grouped = (
        frame.groupby(group_columns, dropna=False)[value_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.columns = [
        column if isinstance(column, str) else "_".join(part for part in column if part)
        for column in grouped.columns
    ]
    return grouped


def _group_dir(
    root: Path,
    n_devices: int,
    ratio: float,
    epsilon: float,
    skip_mode: str,
    channel_sync: str,
) -> Path:
    return root / (
        f"N-{n_devices:02d}"
        f"__ratio-{round(ratio * 10_000):08d}"
        f"__epsilon-{round(epsilon * 1_000_000):08d}"
        f"__skip-{re.sub(r'[^A-Za-z0-9_.-]', '_', skip_mode)}"
        f"__sync-{re.sub(r'[^A-Za-z0-9_.-]', '_', channel_sync)}"
    )


def _group_identity(config: H2ExperimentConfig, **group) -> dict[str, object]:
    sweep = config.sweep
    return {
        "version": H2_CHECKPOINT_VERSION,
        "mode": sweep.mode,
        "t_slots": sweep.t_slots,
        "seeds": list(sweep.seeds),
        "rho_values": list(sweep.rho_values),
        "capacity_bins": sweep.capacity_bins,
        "lambda_iterations": sweep.lambda_iterations,
        **group,
    }


def _checkpoint_complete(path: Path, identity: dict[str, object]) -> bool:
    marker = path / "_complete.json"
    if not marker.exists():
        return False
    try:
        return json.loads(marker.read_text(encoding="utf-8")) == identity
    except (OSError, ValueError, TypeError):
        return False


def _write_checkpoint(
    path: Path,
    identity: dict[str, object],
    tables: dict[str, list[dict[str, object]]],
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    marker = path / "_complete.json"
    marker.unlink(missing_ok=True)
    for name in H2_CHECKPOINT_TABLES:
        target = path / f"{name}.csv"
        if not tables[name]:
            target.unlink(missing_ok=True)
            continue
        temporary = target.with_suffix(".csv.tmp")
        pd.DataFrame(tables[name]).to_csv(
            temporary, index=False, float_format="%.17g"
        )
        temporary.replace(target)
    temporary_marker = marker.with_suffix(".json.tmp")
    temporary_marker.write_text(
        json.dumps(identity, sort_keys=True, indent=2), encoding="utf-8"
    )
    temporary_marker.replace(marker)


def _read_checkpoint(path: Path) -> dict[str, list[dict[str, object]]]:
    tables = {}
    for name in H2_CHECKPOINT_TABLES:
        source = path / f"{name}.csv"
        tables[name] = (
            pd.read_csv(source, float_precision="round_trip").to_dict("records")
            if source.exists()
            else []
        )
    return tables


def _n1_reduction_check(config: H2ExperimentConfig) -> tuple[bool, str]:
    trace = generate_h2_trace(config.channel, 0.75, 2_000, 9917, 0)
    d_min = minimum_h2_deadline_ms(
        config.profile, config.channel, config.sweep.server_total_speedup, 1
    )
    deadline = 2.0 * d_min
    h1 = simulate_trace(
        trace,
        DeviceConfig(profile=config.profile),
        config.channel,
        deadline,
        2.0,
        0.15,
        "drop",
        (10.0, 100.0, 1_000.0, 10_000.0),
        9917,
        0.75,
        0,
    )
    i1, i2, _ = independent_policy_sets(
        (trace,),
        config.profile,
        config.channel,
        deadline,
        0.15,
        "drop",
        config.sweep.server_total_speedup,
    )
    comparisons = []
    for h1_name, h2_result in (("P1", i1.per_device[0]), ("P2", i2.per_device[0])):
        expected = h1.policies[h1_name]
        comparisons.extend(
            [
                np.array_equal(expected.violate, h2_result.violate),
                np.array_equal(expected.split_p, h2_result.split_p),
                np.array_equal(expected.local_mode, h2_result.local_mode),
                np.array_equal(expected.energy_j, h2_result.energy_j),
            ]
        )
    return bool(all(comparisons)), "P1/I1 and P2/I2 arrays compared bitwise"


def _runtime_projection(
    runtime_df: pd.DataFrame, current_t_slots: int
) -> dict[str, object]:
    if runtime_df.empty:
        return {}
    means = runtime_df.groupby("N")["elapsed_seconds"].mean().to_dict()
    seconds_n2 = float(means.get(2, float("nan")))
    seconds_n4 = float(means.get(4, float("nan")))
    if np.isfinite(seconds_n2) and np.isfinite(seconds_n4):
        slope = max(0.0, (seconds_n4 - seconds_n2) / 2.0)
        extrapolated_n8 = seconds_n4 + 4.0 * slope
        n_linear_floor = seconds_n4 * 2.0
        seconds_n8 = max(extrapolated_n8, n_linear_floor)
    elif np.isfinite(seconds_n4):
        seconds_n8 = seconds_n4 * 2.0
    else:
        seconds_n8 = float("nan")
    full_seed_combinations_per_n = 3 * 3 * 2 * 2 * 2 * 10
    t_scale = 50_000 / current_t_slots
    projected = {
        2: seconds_n2 * t_scale * full_seed_combinations_per_n,
        4: seconds_n4 * t_scale * full_seed_combinations_per_n,
        8: seconds_n8 * t_scale * full_seed_combinations_per_n,
    }
    total = float(sum(projected.values()))
    return {
        "smoke_mean_seconds_per_combination_seed_by_N": {
            str(key): float(value) for key, value in means.items()
        },
        "projected_full_seconds_by_N": {
            str(key): float(value) for key, value in projected.items()
        },
        "projected_full_hours": total / 3600.0,
        "n8_method": "max(linear slope extrapolation, N-linear floor from N=4)",
        "exceeds_four_hours": total > 4 * 3600,
    }


def _preregistration_assessment(decomposition: pd.DataFrame) -> dict[str, object]:
    if decomposition.empty:
        return {"status": "not_assessable"}
    region = decomposition[
        (decomposition["channel_sync"] == "independent")
        & (decomposition["skip_mode"] == "late")
        & (decomposition["N"].isin([4, 8]))
        & (decomposition["rho"] >= 0.75)
        & (decomposition["deadline_ratio"] <= 1.35 + 1e-12)
    ]
    if region.empty:
        return {"status": "not_assessable", "reason": "high-load region absent"}
    maximum = float(region["interaction_percent_points_mean"].max())
    if maximum >= 2.0:
        decision = "adopt"
    elif maximum >= 0.5:
        decision = "conditional"
    else:
        decision = "reject_on_observed_region"
    paired = decomposition[
        decomposition["skip_mode"].eq("late")
    ].pivot_table(
        index=["N", "rho", "deadline_ratio", "epsilon"],
        columns="channel_sync",
        values="interaction_percent_points_mean",
    )
    if {"independent", "common"}.issubset(paired.columns):
        sync_difference = paired["independent"] - paired["common"]
        h2b_pairs = int(len(sync_difference))
        h2b_positive = int((sync_difference > 0.0).sum())
        h2b_share = float(h2b_positive / h2b_pairs) if h2b_pairs else None
        h2b_mean = float(sync_difference.mean()) if h2b_pairs else None
    else:
        h2b_pairs = 0
        h2b_positive = 0
        h2b_share = None
        h2b_mean = None

    fairness = decomposition[
        decomposition["skip_mode"].eq("late")
        & decomposition["channel_sync"].eq("independent")
    ]
    if len(fairness):
        spread_not_worse = (
            fairness["J2_violation_spread_mean"]
            <= fairness["I2_violation_spread_mean"] + 1e-12
        )
        jain_not_worse = (
            fairness["J2_jain_index_mean"]
            >= fairness["I2_jain_index_mean"] - 1e-12
        )
        fairness_not_worse = spread_not_worse & jain_not_worse
        h2c_share = float(fairness_not_worse.mean())
        h2c_max_spread_increase = float(
            (
                fairness["J2_violation_spread_mean"]
                - fairness["I2_violation_spread_mean"]
            ).max()
        )
        h2c_min_jain_change = float(
            (
                fairness["J2_jain_index_mean"]
                - fairness["I2_jain_index_mean"]
            ).min()
        )
    else:
        h2c_share = None
        h2c_max_spread_increase = None
        h2c_min_jain_change = None

    return {
        "status": "provisional" if 8 not in set(decomposition["N"]) else "assessed",
        "H2a_decision": decision,
        "maximum_high_load_interaction_percent_points": maximum,
        "H2b_independent_higher_pair_count": h2b_positive,
        "H2b_paired_combination_count": h2b_pairs,
        "H2b_independent_higher_share": h2b_share,
        "H2b_mean_independent_minus_common_percent_points": h2b_mean,
        "H2c_fairness_not_worse_share": h2c_share,
        "H2c_max_violation_spread_increase": h2c_max_spread_increase,
        "H2c_min_jain_index_change": h2c_min_jain_change,
    }


def run_h2_sweep(
    mode: str,
    output_dir: Path,
    strict_sanity: bool = True,
    make_plots: bool = True,
    resume: bool = False,
) -> dict[str, Path]:
    config = default_h2_experiment(mode)
    sweep = config.sweep
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_root = output_dir / "checkpoint"
    if checkpoint_root.exists() and not resume:
        shutil.rmtree(checkpoint_root)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    d_min_by_n = {
        n: minimum_h2_deadline_ms(
            config.profile, config.channel, sweep.server_total_speedup, n
        )
        for n in sweep.n_devices
    }
    preflight_rows = []
    valid_lookup = {}
    for n_devices in sweep.n_devices:
        for rho in sweep.rho_values:
            for ratio in sweep.deadline_ratios:
                deadline = ratio * d_min_by_n[n_devices]
                for epsilon in sweep.epsilons:
                    check = h2_preflight_check(
                        config.profile,
                        config.channel,
                        sweep.server_total_speedup,
                        n_devices,
                        deadline,
                        epsilon,
                    )
                    for skip_mode in sweep.skip_modes:
                        for channel_sync in sweep.channel_sync_values:
                            key = (n_devices, rho, ratio, epsilon, skip_mode, channel_sync)
                            valid_lookup[key] = check.valid
                            preflight_rows.append(
                                {
                                    "N": n_devices,
                                    "channel_sync": channel_sync,
                                    "rho": rho,
                                    "deadline_ratio": ratio,
                                    "deadline_ms": deadline,
                                    "d_min_n_ms": d_min_by_n[n_devices],
                                    "epsilon": epsilon,
                                    "skip_mode": skip_mode,
                                    "forced_violation_expected": check.forced_violation_expected,
                                    "valid": check.valid,
                                    "exclusion_reason": check.reason,
                                }
                            )

    channel_rows = []
    channel_sanity_rows = []
    checkpoint_groups = []
    total_groups = (
        len(sweep.n_devices)
        * len(sweep.deadline_ratios)
        * len(sweep.epsilons)
        * len(sweep.skip_modes)
        * len(sweep.channel_sync_values)
    )
    group_number = 0

    for n_devices in sweep.n_devices:
        trace_cache = {}
        for channel_sync in sweep.channel_sync_values:
            for rho in sweep.rho_values:
                for seed in sweep.seeds:
                    traces = generate_h2_traces(
                        config.channel,
                        rho,
                        sweep.t_slots,
                        seed,
                        n_devices,
                        channel_sync,
                    )
                    trace_cache[(channel_sync, rho, seed)] = traces
                    hashes = [trace_hash(trace) for trace in traces]
                    sync_passed = (
                        len(set(hashes)) == 1
                        if channel_sync == "common"
                        else len(set(hashes)) == n_devices
                    )
                    channel_sanity_rows.append(
                        {
                            "N": n_devices,
                            "channel_sync": channel_sync,
                            "rho": rho,
                            "seed": seed,
                            "check": f"channel_hashes_{channel_sync}",
                            "passed": sync_passed,
                            "detail": json.dumps(hashes),
                            "severity": "required",
                        }
                    )
                    if strict_sanity:
                        assert_sanity(channel_sanity_rows[-1:])
                    for device, trace in enumerate(traces):
                        channel_rows.append(
                            {
                                "N": n_devices,
                                "channel_sync": channel_sync,
                                "rho": rho,
                                "seed": seed,
                                "device": f"device_{device}",
                                "device_index": device,
                                "trace_sha256": hashes[device],
                                **trace.metadata,
                            }
                        )

        for ratio in sweep.deadline_ratios:
            deadline = ratio * d_min_by_n[n_devices]
            for epsilon in sweep.epsilons:
                for skip_mode in sweep.skip_modes:
                    for channel_sync in sweep.channel_sync_values:
                        group_number += 1
                        group_path = _group_dir(
                            checkpoint_root,
                            n_devices,
                            ratio,
                            epsilon,
                            skip_mode,
                            channel_sync,
                        )
                        identity = _group_identity(
                            config,
                            N=n_devices,
                            deadline_ratio=ratio,
                            epsilon=epsilon,
                            skip_mode=skip_mode,
                            channel_sync=channel_sync,
                        )
                        checkpoint_groups.append(group_path)
                        if resume and _checkpoint_complete(group_path, identity):
                            print(
                                f"[H2 checkpoint {group_number:03d}/{total_groups:03d}] "
                                f"resume-skip N={n_devices} D/Dmin={ratio:g} epsilon={epsilon:g} "
                                f"skip={skip_mode} sync={channel_sync}",
                                flush=True,
                            )
                            continue
                        tables = {name: [] for name in H2_CHECKPOINT_TABLES}
                        for rho in sweep.rho_values:
                            key = (n_devices, rho, ratio, epsilon, skip_mode, channel_sync)
                            if not valid_lookup[key]:
                                continue
                            for seed in sweep.seeds:
                                traces = trace_cache[(channel_sync, rho, seed)]
                                common = {
                                    "N": n_devices,
                                    "channel_sync": channel_sync,
                                    "rho": rho,
                                    "seed": seed,
                                    "deadline_ratio": ratio,
                                    "deadline_ms": deadline,
                                    "d_min_n_ms": d_min_by_n[n_devices],
                                    "epsilon": epsilon,
                                    "skip_mode": skip_mode,
                                }
                                started = time.perf_counter()
                                try:
                                    simulation = simulate_h2_trace(
                                        traces,
                                        config.profile,
                                        config.channel,
                                        deadline,
                                        epsilon,
                                        skip_mode,
                                        sweep,
                                    )
                                except Exception as exc:
                                    tables["runtime_failure_rows"].append(
                                        {
                                            **common,
                                            "status": "runtime-invalid",
                                            "exception_type": type(exc).__name__,
                                            "exception_message": str(exc),
                                        }
                                    )
                                    print(
                                        f"WARNING: H2 runtime-invalid {common}: "
                                        f"{type(exc).__name__}: {exc}",
                                        file=sys.stderr,
                                        flush=True,
                                    )
                                    continue
                                elapsed = time.perf_counter() - started
                                tables["runtime_rows"].append(
                                    {
                                        **common,
                                        "elapsed_seconds": elapsed,
                                        "slots": sweep.t_slots,
                                        "J2_iterations": simulation.j2_iterations,
                                    }
                                )
                                for policy_name, policy in simulation.policies.items():
                                    for device, result in enumerate(policy.per_device):
                                        summary = summarize_policy(
                                            result, traces[device].state
                                        )
                                        allocation = policy.allocation_fraction[device]
                                        tables["policy_rows"].append(
                                            {
                                                **common,
                                                "device": f"device_{device}",
                                                "device_index": device,
                                                **summary,
                                                "allocation_mean": float(allocation.mean()),
                                                "allocation_max": float(allocation.max()),
                                                "allocation_nonzero_rate": float(np.mean(allocation > 0)),
                                            }
                                        )
                                metrics = h2_decomposition_metrics(simulation)
                                tables["decomposition_rows"].append(
                                    {**common, **metrics}
                                )
                                checks = h2_sanity_rows(
                                    simulation,
                                    epsilon,
                                    sweep.relative_energy_tolerance,
                                    sweep.violation_tolerance,
                                )
                                for check in checks:
                                    tables["sanity_rows"].append({**common, **check})
                                if strict_sanity:
                                    assert_sanity(checks)
                                tables["digest_rows"].append(
                                    {**common, "sha256": stable_h2_digest(simulation)}
                                )
                        _write_checkpoint(group_path, identity, tables)
                        print(
                            f"[H2 checkpoint {group_number:03d}/{total_groups:03d}] "
                            f"saved N={n_devices} D/Dmin={ratio:g} epsilon={epsilon:g} "
                            f"skip={skip_mode} sync={channel_sync}",
                            flush=True,
                        )
        del trace_cache

    raw = {name: [] for name in H2_CHECKPOINT_TABLES}
    for group_path in checkpoint_groups:
        tables = _read_checkpoint(group_path)
        for name in H2_CHECKPOINT_TABLES:
            raw[name].extend(tables[name])

    policy_df = pd.DataFrame(raw["policy_rows"])
    runs_df = pd.DataFrame(raw["decomposition_rows"])
    sanity_df = pd.DataFrame(raw["sanity_rows"] + channel_sanity_rows)
    digest_df = pd.DataFrame(raw["digest_rows"])
    runtime_df = pd.DataFrame(raw["runtime_rows"])
    failures_df = pd.DataFrame(
        raw["runtime_failure_rows"], columns=H2_RUNTIME_FAILURE_COLUMNS
    )
    preflight_df = pd.DataFrame(preflight_rows)

    invalid_keys = {
        (
            int(row["N"]),
            str(row["channel_sync"]),
            float(row["rho"]),
            float(row["deadline_ratio"]),
            float(row["epsilon"]),
            str(row["skip_mode"]),
        )
        for row in raw["runtime_failure_rows"]
    }
    for invalid in invalid_keys:
        n_devices, channel_sync, rho, ratio, epsilon, skip_mode = invalid
        mask = (
            (preflight_df["N"] == n_devices)
            & (preflight_df["channel_sync"] == channel_sync)
            & np.isclose(preflight_df["rho"], rho)
            & np.isclose(preflight_df["deadline_ratio"], ratio)
            & np.isclose(preflight_df["epsilon"], epsilon)
            & (preflight_df["skip_mode"] == skip_mode)
        )
        preflight_df.loc[mask, "valid"] = False
        preflight_df.loc[mask, "exclusion_reason"] = "runtime-invalid; see runtime_failures.csv"

    def analysis_rows(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or not invalid_keys:
            return frame
        keep = []
        for row in frame.itertuples(index=False):
            key = (
                int(row.N),
                str(row.channel_sync),
                float(row.rho),
                float(row.deadline_ratio),
                float(row.epsilon),
                str(row.skip_mode),
            )
            keep.append(key not in invalid_keys)
        return frame.loc[keep].copy()

    policy_analysis = analysis_rows(policy_df)
    runs_analysis = analysis_rows(runs_df)
    policy_group = [
        "N",
        "channel_sync",
        "rho",
        "deadline_ratio",
        "epsilon",
        "skip_mode",
        "device",
        "device_index",
        "policy",
    ]
    policy_values = [
        "mean_energy_j",
        "violation_rate",
        "max_violation_run",
        "burst_count_ge2",
        "violation_run_count",
        "violation_state_bad_share",
        "boost_use_rate",
        "allocation_mean",
        "allocation_max",
        "allocation_nonzero_rate",
    ]
    policy_aggregate = _aggregate(policy_analysis, policy_group, policy_values)
    decomposition_group = [
        "N",
        "channel_sync",
        "rho",
        "deadline_ratio",
        "epsilon",
        "skip_mode",
    ]
    decomposition_values = [
        "I1_energy_j",
        "J1_energy_j",
        "I2_energy_j",
        "J2_energy_j",
        "server_only_percent",
        "targeting_indep_percent",
        "targeting_joint_percent",
        "interaction_percent_points",
        "total_percent",
        "identity_error_percent_points",
        "I1_violation_rate_mean",
        "J1_violation_rate_mean",
        "I2_violation_rate_mean",
        "J2_violation_rate_mean",
        "I2_violation_spread",
        "J2_violation_spread",
        "I2_jain_index",
        "J2_jain_index",
        "J2_heuristic_selected",
        "J2_I2_guard_selected",
        "J2_J1_guard_selected",
        "J2_repair_success",
    ]
    decomposition = _aggregate(
        runs_analysis, decomposition_group, decomposition_values
    )
    if not decomposition.empty:
        seed_counts = (
            runs_analysis.groupby(decomposition_group, dropna=False)
            .size()
            .rename("seed_count")
            .reset_index()
        )
        decomposition = decomposition.merge(seed_counts, on=decomposition_group)
    fairness = decomposition[
        decomposition_group
        + [
            "I2_violation_spread_mean",
            "J2_violation_spread_mean",
            "I2_jain_index_mean",
            "J2_jain_index_mean",
        ]
    ].copy() if not decomposition.empty else pd.DataFrame()

    n1_passed, n1_detail = _n1_reduction_check(config)
    n1_row = {
        "check": "N1_policy_path_bitwise_reduction",
        "passed": n1_passed,
        "detail": n1_detail,
        "severity": "required",
    }
    sanity_df = pd.concat([sanity_df, pd.DataFrame([n1_row])], ignore_index=True)
    if strict_sanity:
        assert_sanity([n1_row])

    required_failures = sanity_df[
        sanity_df["severity"].fillna("required").eq("required")
        & ~sanity_df["passed"].astype(bool)
    ]
    heuristic_share = (
        float(runs_analysis["J2_heuristic_selected"].mean())
        if len(runs_analysis)
        else float("nan")
    )
    warning_rows = []
    if np.isfinite(heuristic_share) and heuristic_share < sweep.heuristic_adoption_warning_threshold:
        warning_rows.append(
            {
                "warning": "low J2 heuristic adoption",
                "detail": f"share={heuristic_share:.6f} < {sweep.heuristic_adoption_warning_threshold}",
            }
        )
        print(f"WARNING: {warning_rows[-1]['detail']}", file=sys.stderr)
    j1_axis_alive = bool(
        len(runs_analysis)
        and (runs_analysis["J1_energy_j"] < runs_analysis["I1_energy_j"] - 1e-12).any()
    )
    i2_axis_alive = bool(
        len(runs_analysis)
        and (runs_analysis["I2_energy_j"] < runs_analysis["I1_energy_j"] - 1e-12).any()
    )
    runtime_projection = _runtime_projection(runtime_df, sweep.t_slots)
    smoke_rows = []
    if mode == "smoke":
        smoke_rows = [
            {"condition": "a_N1_bitwise_reduction", "passed": n1_passed, "detail": n1_detail},
            {
                "condition": "b_required_capacity_budget_sanity",
                "passed": required_failures.empty,
                "detail": f"required_failures={len(required_failures)}",
            },
            {
                "condition": "c1_J1_strictly_improves_I1_somewhere",
                "passed": j1_axis_alive,
                "detail": f"strict_improvement_exists={j1_axis_alive}",
            },
            {
                "condition": "c2_I2_strictly_improves_I1_somewhere",
                "passed": i2_axis_alive,
                "detail": f"strict_improvement_exists={i2_axis_alive}",
            },
            {
                "condition": "d_J2_heuristic_adoption_reported",
                "passed": np.isfinite(heuristic_share),
                "detail": f"heuristic_share={heuristic_share:.6f}",
            },
            {
                "condition": "e_full_runtime_projection_reported",
                "passed": bool(runtime_projection),
                "detail": json.dumps(runtime_projection, sort_keys=True),
            },
        ]
        failed_gate = [row for row in smoke_rows[:4] if not row["passed"]]
        if strict_sanity and failed_gate:
            raise AssertionError(f"H2 smoke gate failures: {failed_gate}")

    outputs = {
        "policy_runs": output_dir / "policy_runs.csv",
        "policy_aggregate": output_dir / "policy_aggregate.csv",
        "h2_runs": output_dir / "h2_runs.csv",
        "h2_decomposition": output_dir / "h2_decomposition.csv",
        "h2_fairness": output_dir / "h2_fairness.csv",
        "preflight": output_dir / "preflight.csv",
        "channel_stats": output_dir / "channel_stats.csv",
        "violation_patterns": output_dir / "violation_pattern_stats.csv",
        "sanity": output_dir / "sanity_checks.csv",
        "digests": output_dir / "reproducibility_hashes.csv",
        "runtime": output_dir / "runtime_stats.csv",
        "runtime_failures": output_dir / "runtime_failures.csv",
        "warnings": output_dir / "diagnostic_warnings.csv",
        "smoke_acceptance": output_dir / "smoke_acceptance.csv",
        "parameters": output_dir / "run_parameters.json",
        "preregistration": output_dir / "preregistration_assessment.json",
    }
    policy_df.to_csv(outputs["policy_runs"], index=False)
    policy_aggregate.to_csv(outputs["policy_aggregate"], index=False)
    runs_df.to_csv(outputs["h2_runs"], index=False)
    decomposition.to_csv(outputs["h2_decomposition"], index=False)
    fairness.to_csv(outputs["h2_fairness"], index=False)
    preflight_df.to_csv(outputs["preflight"], index=False)
    pd.DataFrame(channel_rows).to_csv(outputs["channel_stats"], index=False)
    policy_df.to_csv(outputs["violation_patterns"], index=False)
    sanity_df.to_csv(outputs["sanity"], index=False)
    digest_df.to_csv(outputs["digests"], index=False)
    runtime_df.to_csv(outputs["runtime"], index=False)
    failures_df.to_csv(outputs["runtime_failures"], index=False)
    pd.DataFrame(warning_rows, columns=["warning", "detail"]).to_csv(outputs["warnings"], index=False)
    pd.DataFrame(smoke_rows, columns=["condition", "passed", "detail"]).to_csv(outputs["smoke_acceptance"], index=False)

    preregistration = _preregistration_assessment(decomposition)
    outputs["preregistration"].write_text(
        json.dumps(
            {"criteria": H2_PREREGISTRATION, "observed": preregistration},
            indent=2,
        ),
        encoding="utf-8",
    )
    parameter_payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "config": config.to_dict(),
        "derived_d_min_ms_by_N": {str(key): value for key, value in d_min_by_n.items()},
        "runtime_projection": runtime_projection,
        "J2_source_adoption": {
            "heuristic": heuristic_share,
            "I2_guard": float(runs_analysis["J2_I2_guard_selected"].mean()) if len(runs_analysis) else None,
            "J1_guard": float(runs_analysis["J2_J1_guard_selected"].mean()) if len(runs_analysis) else None,
        },
        "notes": {
            "server_model": "Server is a GPS-style proportional-share processor: concurrent execution, linear scaling in f, no partitioning granularity/interference/switching overhead. All policies share this idealization; relaxing it is future work.",
            "late": "A late violation runs in best-effort/idle time outside real-time capacity and consumes no server capacity in its missed slot; drop also consumes none.",
            "D_min": "D_min(N) uses normal mode, nominal Good rate, and fixed equal share f=1/N.",
            "channel_sync": "independent uses device-index entropy; common reuses device-index-zero state and jitter arrays for every device.",
            "capacity_grid": f"minimum shares are rounded upward to 1/{sweep.capacity_bins}; all configured 1/N shares are exact grid points.",
            "performance_optimization": "Meet-option Pareto dominance pruning is enabled because the initial unpruned smoke projected 6.08 full hours (>4 h); the violation option is never pruned against meet options.",
            "tie_rotation": "At slot t the joint DP processes devices in order (i+t) mod N to avoid device-zero tie bias.",
            "J2": {
                "lambda_initial": "per-device median positive equal-share saving",
                "lambda_min_factor": sweep.lambda_min_factor,
                "eta_schedule": "0.5/sqrt(iteration+1)",
                "max_full_horizon_iterations": sweep.lambda_iterations,
                "repair": "lowest equal-share-saving violations first; each forced-meet change re-solves the joint slot",
                "guard": "minimum-energy feasible member of heuristic, I2, and J1",
            },
        },
    }
    outputs["parameters"].write_text(
        json.dumps(parameter_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if make_plots:
        create_h2_plots(decomposition, fairness, output_dir)

    print(f"Completed H2 {mode} sweep. Outputs: {output_dir.resolve()}")
    if runtime_projection:
        print(
            f"Projected H2 full runtime: {runtime_projection['projected_full_hours']:.3f} h "
            f"(N=8 corrected by {runtime_projection['n8_method']})"
        )
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "quick", "full"), default="smoke")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--strict-sanity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_h2_sweep(
        args.mode,
        args.output or default_h2_output_dir(args.mode),
        strict_sanity=args.strict_sanity,
        make_plots=args.plots,
        resume=args.resume,
    )
