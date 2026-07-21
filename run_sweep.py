"""CLI entry point: python run_sweep.py --mode {smoke|quick|full}."""

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
    mode_usage_rows,
    stable_simulation_digest,
    summarize_saving,
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


def _deadline_axis_diagnostics(
    comparison_aggregate: pd.DataFrame,
    tolerance_percent: float,
) -> pd.DataFrame:
    """Check whether adjacent deadline ratios change the offline oracle gap."""
    rows: list[dict[str, object]] = []
    if comparison_aggregate.empty:
        return pd.DataFrame(rows)
    for (device, epsilon, skip_mode), group in comparison_aggregate.groupby(
        ["device", "epsilon", "skip_mode"], dropna=False
    ):
        by_deadline = (
            group.groupby("deadline_ratio", as_index=False)["oracle_gap_percent_mean"]
            .mean()
            .sort_values("deadline_ratio")
        )
        ratios = by_deadline["deadline_ratio"].to_numpy(dtype=float)
        gaps = by_deadline["oracle_gap_percent_mean"].to_numpy(dtype=float)
        differences = np.abs(np.diff(gaps))
        assessable = len(differences) > 0
        max_difference = float(np.max(differences)) if assessable else float("nan")
        active = bool(assessable and max_difference > tolerance_percent)
        rows.append(
            {
                "device": device,
                "epsilon": epsilon,
                "skip_mode": skip_mode,
                "deadline_ratios_json": json.dumps(ratios.tolist()),
                "oracle_gap_percent_json": json.dumps(gaps.tolist()),
                "adjacent_abs_differences_percent_json": json.dumps(differences.tolist()),
                "max_adjacent_abs_difference_percent": max_difference,
                "tolerance_percent": tolerance_percent,
                "assessable": assessable,
                "deadline_axis_active": active,
                "warning": "deadline axis inactive" if assessable and not active else "",
            }
        )
    return pd.DataFrame(rows)


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
    mode_usage_output_rows: list[dict[str, object]] = []
    saving_rows: list[dict[str, object]] = []
    sanity_rows: list[dict[str, object]] = []
    digest_rows: list[dict[str, object]] = []
    warning_rows: list[dict[str, object]] = []
    saving_seen: set[tuple[str, float, int, float, str]] = set()

    first_repro_args = None
    first_repro_digest = None
    device_dmins: dict[str, float] = {}
    total_trace_count = len(config.devices) * len(sweep.rho_values) * len(sweep.seeds)
    trace_number = 0

    for device in config.devices:
        d_min = minimum_good_deadline_ms(
            device.profile, channel_config.r_good_mbps, channel_config.tx_power_w
        )
        device_dmins[device.name] = d_min

        # Pre-flight is analytical and therefore repeated in the table for each
        # rho/skip combination even though feasibility itself is rho-invariant.
        valid_lookup: dict[tuple[float, float, float, str], bool] = {}
        for rho in sweep.rho_values:
            for ratio in sweep.deadline_ratios:
                deadline_ms = ratio * d_min
                for epsilon in sweep.epsilons:
                    check = preflight_check(device, channel_config, deadline_ms, epsilon)
                    for skip_mode in sweep.skip_modes:
                        key = (rho, ratio, epsilon, skip_mode)
                        valid_lookup[key] = check.valid
                        preflight_rows.append(
                            {
                                "device": device.name,
                                "rho": rho,
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

        for rho in sweep.rho_values:
            source = GilbertElliottChannel(
                r_good_mbps=channel_config.r_good_mbps,
                r_bad_mbps=channel_config.r_bad_mbps,
                pi_bad=channel_config.pi_bad,
                rho=rho,
                rate_jitter_sigma_log=channel_config.rate_jitter_sigma_log,
                marginal_tolerance=channel_config.enforce_marginal_tolerance,
                max_resamples=channel_config.max_trace_resamples,
            )
            for seed in sweep.seeds:
                trace = source.generate(sweep.t_slots, seed)
                trace_number += 1
                print(f"[{trace_number:02d}/{total_trace_count}] device={device.name} rho={rho:g} seed={seed}", flush=True)
                channel_rows.append(
                    {
                        "device": device.name,
                        "rho": rho,
                        "seed": seed,
                        **trace.metadata,
                    }
                )
                for ratio in sweep.deadline_ratios:
                    deadline_ms = ratio * d_min
                    for epsilon in sweep.epsilons:
                        for skip_mode in sweep.skip_modes:
                            if not valid_lookup[(rho, ratio, epsilon, skip_mode)]:
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
                                "rho": rho,
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
                                for usage in mode_usage_rows(result, trace.state):
                                    mode_usage_output_rows.append({**common, **usage})
                                if result.name in ("P2", "P3"):
                                    for hist in interval_histogram(result):
                                        interval_rows.append({**common, **hist})

                            saving_key = (device.name, rho, seed, ratio, skip_mode)
                            if saving_key not in saving_seen:
                                saving_seen.add(saving_key)
                                saving_summary = summarize_saving(simulation.costs)
                                saving_rows.append(
                                    {
                                        "device": device.name,
                                        "rho": rho,
                                        "seed": seed,
                                        "deadline_ratio": ratio,
                                        "deadline_ms": deadline_ms,
                                        "skip_mode": skip_mode,
                                        **saving_summary,
                                    }
                                )

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
        pi_passed = abs(float(row["pi_bad_observed"]) - channel_config.pi_bad) <= 0.01 + 1e-12
        rho_passed = abs(float(row["lag1_autocorr"]) - float(row["rho"])) <= 0.03 + 1e-12
        theory_passed = abs(float(row["lag1_autocorr_theory"]) - float(row["rho"])) <= 1e-12
        channel_checks = [
            {
                "device": row["device"],
                "rho": row["rho"],
                "seed": row["seed"],
                "check": "channel_pi_bad_within_0.01",
                "passed": pi_passed,
                "detail": f"observed={row['pi_bad_observed']:.6f}",
            },
            {
                "device": row["device"],
                "rho": row["rho"],
                "seed": row["seed"],
                "check": "channel_rho_within_0.03",
                "passed": rho_passed,
                "detail": f"observed={row['lag1_autocorr']:.6f}, specified={row['rho']:.6f}",
            },
            {
                "device": row["device"],
                "rho": row["rho"],
                "seed": row["seed"],
                "check": "channel_rho_theory_matches_specification",
                "passed": theory_passed,
                "detail": f"theory={row['lag1_autocorr_theory']:.6f}, specified={row['rho']:.6f}",
            },
        ]
        sanity_rows.extend(channel_checks)
        if strict_sanity:
            assert_sanity(channel_checks)

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
    mode_usage_df = pd.DataFrame(mode_usage_output_rows)
    saving_df = pd.DataFrame(saving_rows)
    digest_df = pd.DataFrame(digest_rows)

    policy_group = ["device", "rho", "deadline_ratio", "epsilon", "skip_mode", "policy"]
    policy_values = [
        "mean_energy_j",
        "violation_rate",
        "max_violation_run",
        "burst_count_ge2",
        "violation_state_bad_share",
        "boost_use_rate",
        "boost_use_rate_good",
        "boost_use_rate_bad",
        "selected_v",
    ]
    policy_aggregate = _aggregate(policy_df, policy_group, policy_values)
    comparison_group = ["device", "rho", "deadline_ratio", "epsilon", "skip_mode"]
    comparison_values = ["discard_gain_percent", "targeting_gain_percent", "oracle_gap_percent", "online_recovery", "p2prime_gap_percent", "pi_bad_observed", "lag1_autocorr"]
    comparison_aggregate = _aggregate(comparison_df, comparison_group, comparison_values)
    deadline_axis_df = _deadline_axis_diagnostics(
        comparison_aggregate, sweep.deadline_axis_tolerance_percent
    )

    for row in saving_rows:
        if bool(row["saving_degenerate"]):
            detail = (
                f"saving degenerate: device={row['device']} rho={row['rho']} "
                f"seed={row['seed']} D/Dmin={row['deadline_ratio']} "
                f"skip={row['skip_mode']} unique={row['saving_unique_count']}"
            )
            warning_rows.append({"warning": "saving degenerate", "detail": detail})
            print(f"WARNING: {detail}", file=sys.stderr, flush=True)

    if not deadline_axis_df.empty:
        for row in deadline_axis_df[deadline_axis_df["warning"] != ""].to_dict("records"):
            detail = (
                f"deadline axis inactive: device={row['device']} epsilon={row['epsilon']} "
                f"skip={row['skip_mode']} max adjacent gap change="
                f"{row['max_adjacent_abs_difference_percent']:.6g}%"
            )
            warning_rows.append({"warning": "deadline axis inactive", "detail": detail})
            print(f"WARNING: {detail}", file=sys.stderr, flush=True)

    p1_rows = policy_df[policy_df["policy"] == "P1"]
    boost_used_in_bad = bool(
        len(p1_rows) and (p1_rows["boost_use_rate_bad"].fillna(0.0) > 0.0).any()
    )
    if not boost_used_in_bad:
        detail = "boost unused: P1 boost use rate is zero in Bad state for every setting"
        warning_rows.append({"warning": "boost unused", "detail": detail})
        print(f"WARNING: {detail}", file=sys.stderr, flush=True)

    smoke_rows: list[dict[str, object]] = []
    if mode == "smoke":
        valid_tight_ratios = [
            float(value)
            for value in sorted(
                preflight_df.loc[
                    preflight_df["valid"] & (preflight_df["deadline_ratio"] <= 1.35 + 1e-12),
                    "deadline_ratio",
                ].unique()
            )
        ]
        valid_below_1_35 = [value for value in valid_tight_ratios if value < 1.35 - 1e-12]
        minimum_saving_unique = int(saving_df["saving_unique_count"].min()) if len(saving_df) else 0
        dmin_unchanged = all(
            abs(value - 36.045) <= 1e-12 for value in device_dmins.values()
        )
        smoke_rows = [
            {
                "condition": "a_valid_tight_deadline_ratios_at_least_2",
                "passed": len(valid_tight_ratios) >= 2 and len(valid_below_1_35) >= 2,
                "detail": (
                    f"valid ratios <=1.35: {valid_tight_ratios}; "
                    f"strict ladder target <1.35: {valid_below_1_35}"
                ),
            },
            {
                "condition": "b_saving_unique_values_at_least_20",
                "passed": minimum_saving_unique >= 20,
                "detail": f"minimum unique count={minimum_saving_unique}",
            },
            {
                "condition": "c_p1_boost_used_in_bad_state",
                "passed": boost_used_in_bad,
                "detail": f"maximum Bad-state rate={p1_rows['boost_use_rate_bad'].max() if len(p1_rows) else 0.0}",
            },
            {
                "condition": "d_normal_dmin_is_36_045_ms",
                "passed": dmin_unchanged,
                "detail": json.dumps(device_dmins, sort_keys=True),
            },
        ]
        for row in smoke_rows:
            sanity_rows.append(
                {
                    "check": row["condition"],
                    "passed": row["passed"],
                    "detail": row["detail"],
                    "severity": "acceptance",
                }
            )
            status = "PASS" if row["passed"] else "FAIL"
            print(f"[smoke] {status} {row['condition']}: {row['detail']}", flush=True)

    sanity_df = pd.DataFrame(sanity_rows)
    warning_df = pd.DataFrame(warning_rows, columns=["warning", "detail"])
    smoke_df = pd.DataFrame(smoke_rows, columns=["condition", "passed", "detail"])

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
        "saving_diagnostics": output_dir / "saving_diagnostics.csv",
        "mode_usage": output_dir / "mode_usage.csv",
        "deadline_axis_diagnostics": output_dir / "deadline_axis_diagnostics.csv",
        "diagnostic_warnings": output_dir / "diagnostic_warnings.csv",
        "smoke_acceptance": output_dir / "smoke_acceptance.csv",
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
    saving_df.to_csv(outputs["saving_diagnostics"], index=False)
    mode_usage_df.to_csv(outputs["mode_usage"], index=False)
    deadline_axis_df.to_csv(outputs["deadline_axis_diagnostics"], index=False)
    warning_df.to_csv(outputs["diagnostic_warnings"], index=False)
    smoke_df.to_csv(outputs["smoke_acceptance"], index=False)

    parameter_payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "config": config.to_dict(),
        "derived_d_min_ms": device_dmins,
        "notes": {
            "rho": "The state transition probabilities are inverted directly from stationary pi_B and specified lag-1 rho; rho=0 is i.i.d.",
            "D_min": "D_min uses normal mode only and is fixed at 36.045 ms for the default profile.",
            "rate_jitter": "Rates use independent clipped lognormal multiplicative jitter; state occupancy and autocorrelation checks use state labels only.",
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
    failed_smoke = [row for row in smoke_rows if not bool(row["passed"])]
    if mode == "smoke" and strict_sanity and failed_smoke:
        raise AssertionError(
            "smoke acceptance failures: "
            + "; ".join(f"{row['condition']} ({row['detail']})" for row in failed_smoke)
        )
    print(f"Completed {mode} sweep. Outputs: {output_dir.resolve()}")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "quick", "full"), default="quick")
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
