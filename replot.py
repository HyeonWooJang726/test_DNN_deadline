"""Regenerate figures from an existing simulator results directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from plotting import create_all_plots


def replot(results_dir: Path) -> None:
    policy_aggregate = pd.read_csv(results_dir / "policy_aggregate.csv")
    policy_runs = pd.read_csv(results_dir / "policy_runs.csv")
    comparison_aggregate = pd.read_csv(results_dir / "comparison_aggregate.csv")
    preflight = pd.read_csv(results_dir / "preflight.csv")
    channel_stats = pd.read_csv(results_dir / "channel_stats.csv")

    valid_keys = (
        preflight.loc[
            preflight["valid"].astype(bool),
            ["device", "rho", "deadline_ratio", "epsilon", "skip_mode"],
        ].drop_duplicates()
    )
    policy_analysis_df = policy_runs.merge(
        valid_keys,
        on=["device", "rho", "deadline_ratio", "epsilon", "skip_mode"],
        how="inner",
        validate="many_to_one",
    )

    deadline_ratios = tuple(sorted(float(v) for v in preflight["deadline_ratio"].unique()))
    epsilons = tuple(sorted(float(v) for v in preflight["epsilon"].unique()))
    skip_modes = tuple(sorted(str(v) for v in preflight["skip_mode"].unique()))
    create_all_plots(
        policy_aggregate,
        policy_analysis_df,
        comparison_aggregate,
        preflight,
        channel_stats,
        deadline_ratios,
        epsilons,
        skip_modes,
        gap_mask_percent=2.0,
        output_dir=results_dir,
    )
    figures_dir = Path("figures") / results_dir.name
    print(f"Regenerated plots in: {figures_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    replot(args.results)
