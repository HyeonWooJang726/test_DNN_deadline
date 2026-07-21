"""CLI entry point: python run_sweep.py --mode {quick|full}."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from channel import GilbertElliottChannel
from config import default_experiment, default_output_dir
from dnn_profile import minimum_good_deadline_ms
from metrics import (
    assert_sanity,
    combination_sanity_rows,
    interval_histogram,
    stable_simulation_digest,
    summarize_policy,
)
from plotting import create_all_plots
from simulator import preflight_check, simulate_trace


def _aggregate(frame: pd.DataFrame, group_columns: list[str], value_columns: list[str]) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    grouped = frame.groupby(group_columns, dropna=False)[value_columns].agg(["mean", "std"]).reset_index()
    grouped.columns = [
        column if isinstance(column, str) else "_".join(part for part in column if part)
        for column in grouped.columns
    ]
    return grouped


def run_sweep(mode: str, output_dir: Path, strict_sanity: bool = True, make_plots: bool = True) -> dict[str, Path]:
    config = default_experiment(mode)
    sweep = config.sweep
    channel_config = config.channel
    output_dir.mkdir(parents=True, exist_ok=True)

    policy_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    preflight_rows: list[dict[str, object]] = []
    channel_rows: list[dict[str, object]] = []
    interval_rows: list[dict[str, object]] = []
    sanity_rows: list[dict[str, object]] = []
    digest_rows: list[dict[str, object]] = []

    first_repro_args = None
    first_repro_digest = None
    device_dmins: dict[str, float] = {}
    total_trace_count = len(config.devices) * len(sweep.l_values) * len(sweep.seeds)
    trace_number = 0

    for device in config.devices:
        d_min = minimum_good_deadline_ms(
            device.profile, channel_config.r_good_mbps, channel_config.tx_power_w
        )
        device_dmins[device.name] = d_min

        # Pre-flight is analytical and therefore repeated in the table for each
        # L/skip combination even though feasibility itself is L-invariant.
        valid_lookup: dict[tuple[int, float, float, str], bool] = {}
        for l_value in sweep.l_values:
            for ratio in sweep.deadline_ratios:
                deadline_ms = ratio * d_min
                for epsilon in sweep.epsilons:
                    check = preflight_check(device, channel_config, deadline_ms, epsilon)
                    for skip_mode in sweep.skip_modes:
                        key = (l_value, ratio, epsilon, skip_mode)
                        valid_lookup[key] = check.valid
                        preflight_rows.append(
                            {
                                "device": device.name,
                                "L": l_value,
                                "deadline_ratio": ratio,
                                "deadline_ms": deadline_ms,
                                "epsilon": epsilon,
                                "skip_mode": skip_mode,
                                "good_feasible": check.good_feasible,
                                "bad_feasible": check.bad_feasible,
                                "good_feasible_points": json.dumps(check.good_points),
                                "bad_feasible_points": json.dumps(check.bad_points),
                                "forced_violation_expected": check.forced_violation_expected,
                                "valid": check.valid,
                                "exclusion_reason": check.reason,
                            }
                        )

        for l_value in sweep.l_values:
            source = GilbertElliottChannel(
                r_good_mbps=channel_config.r_good_mbps,
                r_bad_mbps=channel_config.r_bad_mbps,
                pi_bad=channel_config.pi_bad,
                mean_bad_dwell_slots=l_value,
                marginal_tolerance=channel_config.enforce_marginal_tolerance,
                max_resamples=channel_config.max_trace_resamples,
            )
            for seed in sweep.seeds:
                trace = source.generate(sweep.t_slots, seed)
                trace_number += 1
                print(f"[{trace_number:02d}/{total_trace_count}] device={device.name} L={l_value} seed={seed}", flush=True)
                channel_rows.append(
                    {
                        "device": device.name,
                        "L": l_value,
                        "seed": seed,
                        **trace.metadata,
                    }
                )
                for ratio in sweep.deadline_ratios:
                    deadline_ms = ratio * d_min
                    for epsilon in sweep.epsilons:
                        for skip_mode in sweep.skip_modes:
                            if not valid_lookup[(l_value, ratio, epsilon, skip_mode)]:
                                continue
                            simulation = simulate_trace(
                                trace,
                                device,
                                channel_config,
                                deadline_ms,
                                ratio,
                                epsilon,
                                skip_mode,
                                sweep.p3_v_values,
                                seed,
                            )
                            common = {
                                "device": device.name,
                                "L": l_value,
                                "seed": seed,
                                "deadline_ratio": ratio,
                                "deadline_ms": deadline_ms,
                                "epsilon": epsilon,
                                "skip_mode": skip_mode,
                                "r_good_mbps": channel_config.r_good_mbps,
                                "r_bad_mbps": channel_config.r_bad_mbps,
                                "rate_ratio": channel_config.r_good_mbps / channel_config.r_bad_mbps,
                                "pi_bad_observed": trace.metadata["pi_bad_observed"],
                                "lag1_autocorr": trace.metadata["lag1_autocorr"],
                                "forced_count": simulation.forced_count,
                                "discretionary_budget": simulation.discretionary_budget,
                            }
                            for result in simulation.policies.values():
                                summary = summarize_policy(result, trace.state)
                                policy_rows.append({**common, **summary})
                                if result.name in ("P2", "P3"):
                                    for hist in interval_histogram(result):
                                        interval_rows.append({**common, **hist})

                            energies = {name: result.mean_energy_j for name, result in simulation.policies.items()}
                            p1 = energies["P1"]
                            oracle_denominator = energies["P1"] - energies["P2"]
                            oracle_gap_percent = 100.0 * oracle_denominator / max(p1, 1e-15)
                            comparison_rows.append(
                                {
                                    **common,
                                    "discard_gain_percent": 100.0 * (energies["P1"] - energies["P0"]) / max(p1, 1e-15),
                                    "targeting_gain_percent": 100.0 * (energies["P0"] - energies["P2"]) / max(p1, 1e-15),
                                    "oracle_gap_percent": oracle_gap_percent,
                                    "online_recovery": (
                                        (energies["P1"] - energies["P3"]) / oracle_denominator
                                        if oracle_gap_percent >= sweep.oracle_gap_mask_percent and oracle_denominator > 0
                                        else np.nan
                                    ),
                                    "p2prime_gap_percent": 100.0 * (energies["P2prime"] - energies["P2"]) / max(p1, 1e-15),
                                }
                            )
                            checks = combination_sanity_rows(
                                simulation,
                                epsilon,
                                sweep.relative_energy_tolerance,
                                sweep.violation_tolerance,
                            )
                            # P3 has no no-adjacent constraint (its bursts are an
                            # H1b outcome), so it can legitimately beat P2prime.
                            # Keep the requested ordering as a visible diagnostic,
                            # but do not mislabel it as a mathematical assertion.
                            for check in checks:
                                check["severity"] = (
                                    "diagnostic"
                                    if check["check"] == "energy_P2prime_le_P3"
                                    else "required"
                                )
                            for check in checks:
                                sanity_rows.append({**common, **check})
                            if strict_sanity:
                                assert_sanity(checks)

                            digest = stable_simulation_digest(simulation)
                            digest_rows.append({**common, "sha256": digest})
                            if first_repro_args is None:
                                first_repro_args = (trace, device, channel_config, deadline_ms, ratio, epsilon, skip_mode, sweep.p3_v_values, seed)
                                first_repro_digest = digest

    # Channel assertions are evaluated after all seeds are present.
    channel_df = pd.DataFrame(channel_rows)
    for row in channel_rows:
        passed = abs(float(row["pi_bad_observed"]) - channel_config.pi_bad) <= 0.01 + 1e-12
        sanity_rows.append({"device": row["device"], "L": row["L"], "seed": row["seed"], "check": "channel_pi_bad_within_0.01", "passed": passed, "detail": f"observed={row['pi_bad_observed']:.6f}"})
        if strict_sanity:
            assert passed
    autocorr_means = channel_df.groupby("L")["lag1_autocorr"].mean()
    l1_passed = abs(float(autocorr_means.loc[1])) <= 0.30
    corr_order_passed = float(autocorr_means.loc[50]) > float(autocorr_means.loc[5])
    channel_global_checks = [
        {"check": "L1_autocorrelation_approximately_zero", "passed": l1_passed, "detail": f"rho1={autocorr_means.loc[1]:.6f}; formula implies -0.25 at pi_B=0.2"},
        {"check": "autocorrelation_L50_gt_L5", "passed": corr_order_passed, "detail": f"rho50={autocorr_means.loc[50]:.6f}, rho5={autocorr_means.loc[5]:.6f}"},
    ]
    sanity_rows.extend(channel_global_checks)
    if strict_sanity:
        assert_sanity(channel_global_checks)

    # Re-run one full combination, including random P0, and compare hashes.
    if first_repro_args is not None:
        repeated = simulate_trace(*first_repro_args)
        repeated_digest = stable_simulation_digest(repeated)
        repro_passed = repeated_digest == first_repro_digest
        sanity_rows.append({"check": "fixed_seed_pipeline_hash_reproducible", "passed": repro_passed, "detail": f"{first_repro_digest} == {repeated_digest}"})
        if strict_sanity:
            assert repro_passed

    policy_df = pd.DataFrame(policy_rows)
    comparison_df = pd.DataFrame(comparison_rows)
    preflight_df = pd.DataFrame(preflight_rows)
    interval_df = pd.DataFrame(interval_rows)
    sanity_df = pd.DataFrame(sanity_rows)
    digest_df = pd.DataFrame(digest_rows)

    policy_group = ["device", "L", "deadline_ratio", "epsilon", "skip_mode", "policy"]
    policy_values = ["mean_energy_j", "violation_rate", "max_violation_run", "burst_count_ge2", "violation_state_bad_share", "selected_v"]
    policy_aggregate = _aggregate(policy_df, policy_group, policy_values)
    comparison_group = ["device", "L", "deadline_ratio", "epsilon", "skip_mode"]
    comparison_values = ["discard_gain_percent", "targeting_gain_percent", "oracle_gap_percent", "online_recovery", "p2prime_gap_percent", "pi_bad_observed", "lag1_autocorr"]
    comparison_aggregate = _aggregate(comparison_df, comparison_group, comparison_values)

    outputs = {
        "policy_runs": output_dir / "policy_runs.csv",
        "policy_aggregate": output_dir / "policy_aggregate.csv",
        "comparisons": output_dir / "comparisons.csv",
        "comparison_aggregate": output_dir / "comparison_aggregate.csv",
        "preflight": output_dir / "preflight.csv",
        "channel_stats": output_dir / "channel_stats.csv",
        "violation_patterns": output_dir / "violation_pattern_stats.csv",
        "interval_histograms": output_dir / "violation_interval_histograms.csv",
        "sanity": output_dir / "sanity_checks.csv",
        "digests": output_dir / "reproducibility_hashes.csv",
        "parameters": output_dir / "run_parameters.json",
    }
    policy_df.to_csv(outputs["policy_runs"], index=False)
    policy_aggregate.to_csv(outputs["policy_aggregate"], index=False)
    comparison_df.to_csv(outputs["comparisons"], index=False)
    comparison_aggregate.to_csv(outputs["comparison_aggregate"], index=False)
    preflight_df.to_csv(outputs["preflight"], index=False)
    channel_df.to_csv(outputs["channel_stats"], index=False)
    policy_df[policy_df["policy"].isin(["P2", "P3"])].to_csv(outputs["violation_patterns"], index=False)
    interval_df.to_csv(outputs["interval_histograms"], index=False)
    sanity_df.to_csv(outputs["sanity"], index=False)
    digest_df.to_csv(outputs["digests"], index=False)

    parameter_payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "config": config.to_dict(),
        "derived_d_min_ms": device_dmins,
        "notes": {
            "L1_autocorrelation": "The specified transition formula implies rho=-0.25, not exactly zero, for L=1 and pi_B=0.2; tolerance is |rho|<=0.30.",
            "P2prime": "T>2000 uses cardinality-priced (Lagrangian) exact path DP plus deterministic safe augmentation.",
            "invalid": "Combinations with expected forced violation rate >= epsilon/2 are excluded from policy simulation and retained in preflight.csv.",
        },
    }
    with outputs["parameters"].open("w", encoding="utf-8") as handle:
        json.dump(parameter_payload, handle, ensure_ascii=False, indent=2)

    if make_plots:
        create_all_plots(
            policy_aggregate,
            comparison_aggregate,
            preflight_df,
            channel_df,
            sweep.deadline_ratios,
            sweep.epsilons,
            sweep.skip_modes,
            sweep.oracle_gap_mask_percent,
            output_dir,
        )
    print(f"Completed {mode} sweep. Outputs: {output_dir.resolve()}")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--strict-sanity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_sweep(
        args.mode,
        args.output or default_output_dir(args.mode),
        strict_sanity=args.strict_sanity,
        make_plots=args.plots,
    )
