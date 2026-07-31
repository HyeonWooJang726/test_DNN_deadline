"""Gate 0 drop-only offline energy-saving ceiling experiment.

Run from the repository root with::

    python experiments/gate0_drop_ceiling/run.py
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from channel import GilbertElliottChannel
from config import ChannelConfig, DNNProfileConfig, default_experiment
from dnn_profile import compute_slot_costs, minimum_good_deadline_ms


EPSILONS = (0.001, 0.0025, 0.005, 0.0075, 0.01)
DEADLINE_RATIOS = (1.2, 1.35, 1.5, 2.0, 2.5, 3.0, 4.0)
N_SAMPLES = 2_000_000
RANDOM_SEED = 20260731
XCHECK_TOL_PP = 0.3
ENERGY_UNIT = "J"
DEFAULT_CHUNK_SIZE = 100_000

EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
FIGURES_DIR = EXPERIMENT_DIR / "figures"
RESULT_COLUMNS = [
    "profile",
    "sample_count",
    "random_seed",
    "energy_unit",
    "d_min_ms",
    "deadline_ratio",
    "deadline_ms",
    "epsilon",
    "epsilon_min",
    "discretionary_budget",
    "n_discretionary_drops",
    "status",
    "p1_energy_mean",
    "p2_energy_mean",
    "p2_saving_percent",
    "boundary_saving_lambda",
    "additional_saving_percent_per_0_1pct_budget",
]
SUMMARY_COLUMNS = [
    "profile",
    "deadline_ratio",
    "deadline_ms",
    "epsilon_min",
    "saving_at_0_1pct",
    "saving_at_0_25pct",
    "saving_at_0_5pct",
    "saving_at_0_75pct",
    "saving_at_1pct",
]
SUMMARY_SAVING_COLUMNS = {
    0.001: "saving_at_0_1pct",
    0.0025: "saving_at_0_25pct",
    0.005: "saving_at_0_5pct",
    0.0075: "saving_at_0_75pct",
    0.01: "saving_at_1pct",
}


@dataclass(frozen=True)
class DeadlineEvaluation:
    """P1 costs and the feasible-slot P2 frontier for one deadline."""

    sample_count: int
    epsilon_min: float
    feasible_count: int
    p1_energy_mean: float
    savings_descending: np.ndarray
    cumulative_savings: np.ndarray


def profile_identifier(profile: DNNProfileConfig) -> str:
    """Build the CSV identifier from the configured profile and mode scales."""

    scales = ",".join(
        f"{mode.name}:{mode.energy_scale:g}" for mode in profile.local_modes
    )
    return f"{profile.name}|energy_scale={scales}"


def sample_stationary_rates(
    sample_count: int,
    seed: int,
    channel_config: ChannelConfig | None = None,
) -> np.ndarray:
    """Generate i.i.d. stationary-marginal rates via the existing channel path."""

    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if channel_config is None:
        channel_config = default_experiment().channel
    source = GilbertElliottChannel(
        r_good_mbps=channel_config.r_good_mbps,
        r_bad_mbps=channel_config.r_bad_mbps,
        pi_bad=channel_config.pi_bad,
        rho=0.0,
        rate_jitter_sigma_log=channel_config.rate_jitter_sigma_log,
        marginal_tolerance=channel_config.enforce_marginal_tolerance,
        max_resamples=channel_config.max_trace_resamples,
    )
    return source.generate(sample_count, seed).rate_mbps


def evaluate_deadline(
    rate_mbps: np.ndarray,
    profile: DNNProfileConfig,
    deadline_ms: float,
    tx_power_w: float,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> DeadlineEvaluation:
    """Evaluate forced drops, P1 energy, and savings using existing slot costs."""

    rates = np.asarray(rate_mbps, dtype=np.float64)
    if rates.ndim != 1 or len(rates) == 0 or np.any(rates <= 0):
        raise ValueError("rate_mbps must be a non-empty positive 1-D array")
    if deadline_ms <= 0:
        raise ValueError("deadline_ms must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    forced_count = 0
    p1_energy_total = 0.0
    saving_chunks: list[np.ndarray] = []
    for start in range(0, len(rates), chunk_size):
        costs = compute_slot_costs(
            profile,
            rates[start : start + chunk_size],
            deadline_ms,
            tx_power_w,
            skip_mode="drop",
        )
        forced_count += int((~costs.feasible).sum())
        p1_energy_total += float(costs.meet_energy_j.sum(dtype=np.float64))
        saving_chunks.append(costs.saving_j[costs.feasible])

    if saving_chunks:
        savings = np.concatenate(saving_chunks)
    else:  # The rates input is non-empty, but keep the all-forced case explicit.
        savings = np.empty(0, dtype=np.float64)
    savings.sort()
    savings = savings[::-1].copy()
    cumulative = np.cumsum(savings, dtype=np.float64)
    sample_count = len(rates)
    return DeadlineEvaluation(
        sample_count=sample_count,
        epsilon_min=forced_count / sample_count,
        feasible_count=len(savings),
        p1_energy_mean=p1_energy_total / sample_count,
        savings_descending=savings,
        cumulative_savings=cumulative,
    )


def _fractional_saving_sum(
    savings_descending: np.ndarray,
    cumulative_savings: np.ndarray,
    requested_count: float,
) -> float:
    """Return S(x), including the exact fractional boundary item."""

    count = min(max(float(requested_count), 0.0), float(len(savings_descending)))
    whole = math.floor(count)
    total = float(cumulative_savings[whole - 1]) if whole else 0.0
    if whole < len(savings_descending):
        total += (count - whole) * float(savings_descending[whole])
    return total


def compute_fractional_oracle(
    savings_descending: np.ndarray,
    n_discretionary_drops: float,
    sample_count: int,
    p1_energy_mean: float,
    cumulative_savings: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute exact fractional P2 saving, boundary statistic, and +0.1 pp gain.

    ``boundary_saving_lambda`` is an order statistic on the offline sample-path
    frontier.  It is not the future causal CMDP-LP dual ``lambda_V*``.
    """

    savings = np.asarray(savings_descending, dtype=np.float64)
    if savings.ndim != 1 or np.any(savings < 0):
        raise ValueError("savings_descending must be a nonnegative 1-D array")
    if len(savings) > 1 and np.any(savings[:-1] < savings[1:]):
        raise ValueError("savings_descending must be sorted in descending order")
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if n_discretionary_drops < 0:
        raise ValueError("n_discretionary_drops must be nonnegative")
    if p1_energy_mean < 0:
        raise ValueError("p1_energy_mean must be nonnegative")

    if cumulative_savings is None:
        cumulative = np.cumsum(savings, dtype=np.float64)
    else:
        cumulative = np.asarray(cumulative_savings, dtype=np.float64)
        if cumulative.shape != savings.shape:
            raise ValueError("cumulative_savings must match savings_descending")

    boundary_count = min(float(n_discretionary_drops), float(len(savings)))
    total_saving = _fractional_saving_sum(savings, cumulative, boundary_count)
    p2_energy_mean = p1_energy_mean - total_saving / sample_count
    if p1_energy_mean > 0:
        p2_saving_percent = 100.0 * total_saving / (
            sample_count * p1_energy_mean
        )
    else:
        p2_saving_percent = float("nan")

    boundary_index = math.floor(boundary_count)
    # Once the entire feasible set has been dropped, there is no next order
    # statistic, so the boundary value is intentionally NaN.
    if boundary_index >= len(savings):
        boundary_lambda = float("nan")
    else:
        boundary_lambda = float(savings[boundary_index])

    # The +0.1 pp endpoint is clipped at F because no additional feasible slot
    # remains available beyond that point.
    extended_count = min(boundary_count + 0.001 * sample_count, len(savings))
    extended_saving = _fractional_saving_sum(savings, cumulative, extended_count)
    if p1_energy_mean > 0:
        additional_saving = 100.0 * (extended_saving - total_saving) / (
            sample_count * p1_energy_mean
        )
    else:
        additional_saving = float("nan")

    return {
        "p2_energy_mean": p2_energy_mean,
        "p2_saving_percent": p2_saving_percent,
        "boundary_saving_lambda": boundary_lambda,
        "additional_saving_percent_per_0_1pct_budget": additional_saving,
    }


def evaluate_epsilon(
    evaluation: DeadlineEvaluation,
    epsilon: float,
) -> dict[str, float | str]:
    """Evaluate one total drop budget against a deadline frontier."""

    discretionary_budget = float(epsilon) - evaluation.epsilon_min
    n_discretionary_drops = evaluation.sample_count * discretionary_budget
    base: dict[str, float | str] = {
        "epsilon": float(epsilon),
        "epsilon_min": evaluation.epsilon_min,
        "discretionary_budget": discretionary_budget,
        "n_discretionary_drops": n_discretionary_drops,
        "p1_energy_mean": evaluation.p1_energy_mean,
    }
    if epsilon < evaluation.epsilon_min:
        return {
            **base,
            "status": "infeasible",
            "p2_energy_mean": float("nan"),
            "p2_saving_percent": float("nan"),
            "boundary_saving_lambda": float("nan"),
            "additional_saving_percent_per_0_1pct_budget": float("nan"),
        }

    oracle = compute_fractional_oracle(
        evaluation.savings_descending,
        n_discretionary_drops,
        evaluation.sample_count,
        evaluation.p1_energy_mean,
        evaluation.cumulative_savings,
    )
    return {**base, "status": "feasible", **oracle}


def calculate_experiment(
    sample_count: int = N_SAMPLES,
    seed: int = RANDOM_SEED,
    deadline_ratios: Iterable[float] = DEADLINE_RATIOS,
    epsilons: Iterable[float] = EPSILONS,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> pd.DataFrame:
    """Calculate the Gate 0 table without writing files or figures."""

    config = default_experiment()
    ratios = tuple(float(value) for value in deadline_ratios)
    budgets = tuple(float(value) for value in epsilons)
    if not ratios or min(ratios) <= 0:
        raise ValueError("deadline_ratios must be non-empty and positive")
    if not budgets or min(budgets) < 0 or max(budgets) > 1:
        raise ValueError("epsilons must be non-empty and lie in [0, 1]")

    rates = sample_stationary_rates(sample_count, seed, config.channel)
    rows: list[dict[str, object]] = []
    for device in config.devices:
        profile = device.profile
        identifier = profile_identifier(profile)
        d_min_ms = minimum_good_deadline_ms(
            profile,
            config.channel.r_good_mbps,
            config.channel.tx_power_w,
        )
        for ratio in ratios:
            deadline_ms = ratio * d_min_ms
            evaluation = evaluate_deadline(
                rates,
                profile,
                deadline_ms,
                config.channel.tx_power_w,
                chunk_size,
            )
            for epsilon in budgets:
                row = evaluate_epsilon(evaluation, epsilon)
                rows.append(
                    {
                        "profile": identifier,
                        "sample_count": sample_count,
                        "random_seed": seed,
                        "energy_unit": ENERGY_UNIT,
                        "d_min_ms": d_min_ms,
                        "deadline_ratio": ratio,
                        "deadline_ms": deadline_ms,
                        **row,
                    }
                )
    return (
        pd.DataFrame(rows, columns=RESULT_COLUMNS)
        .sort_values(["profile", "deadline_ratio", "epsilon"])
        .reset_index(drop=True)
    )


def build_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Build one deadline row with named saving columns for each epsilon."""

    rows: list[dict[str, object]] = []
    group_columns = ["profile", "deadline_ratio", "deadline_ms", "epsilon_min"]
    for keys, group in results.groupby(group_columns, sort=True, dropna=False):
        row: dict[str, object] = dict(zip(group_columns, keys))
        for epsilon, column in SUMMARY_SAVING_COLUMNS.items():
            match = group[np.isclose(group["epsilon"], epsilon)]
            row[column] = (
                float(match.iloc[0]["p2_saving_percent"])
                if len(match)
                else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS).sort_values(
        ["profile", "deadline_ratio"]
    )


def plot_saving_frontier(results: pd.DataFrame, output_path: Path) -> None:
    """Plot P2 ceiling by total drop budget, leaving infeasible gaps."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8.5, 5.5), layout="constrained")
    for ratio, group in results.groupby("deadline_ratio", sort=True):
        ordered = group.sort_values("epsilon")
        axis.plot(
            100.0 * ordered["epsilon"],
            ordered["p2_saving_percent"],
            marker="o",
            label=f"D/D_min={ratio:g}",
        )
    axis.set_xlabel("Total drop budget epsilon (%)")
    axis.set_ylabel("P2 maximum device-energy saving (%)")
    axis.set_title("Drop-only offline energy-saving ceiling")
    axis.grid(alpha=0.25)
    axis.legend(ncols=2)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_forced_drop_frontier(results: pd.DataFrame, output_path: Path) -> None:
    """Plot the forced-drop floor versus deadline ratio."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    points = (
        results[["deadline_ratio", "epsilon_min"]]
        .drop_duplicates()
        .sort_values("deadline_ratio")
    )
    figure, axis = plt.subplots(figsize=(8.5, 5.0), layout="constrained")
    axis.plot(
        points["deadline_ratio"],
        100.0 * points["epsilon_min"],
        marker="o",
        color="#1f77b4",
        label="forced drop rate",
    )
    for level, style in ((0.1, "--"), (0.5, "-."), (1.0, ":")):
        axis.axhline(level, linestyle=style, color="#666666", label=f"{level:g}%")
    axis.set_xlabel("Deadline ratio D/D_min")
    axis.set_ylabel("Forced drop rate epsilon_min (%)")
    axis.set_title("Deadline feasibility floor")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _committed_cross_check_rows(repo_root: Path) -> tuple[Path | None, pd.DataFrame]:
    """Read the committed sweep's directly normalized P2 saving rows."""

    path = repo_root / "results" / "full" / "comparison_aggregate.csv"
    if not path.exists():
        return None, pd.DataFrame()
    frame = pd.read_csv(path)
    required = {
        "rho",
        "deadline_ratio",
        "epsilon",
        "skip_mode",
        "oracle_gap_percent_mean",
    }
    if not required.issubset(frame.columns):
        return path, pd.DataFrame()
    rows = frame[
        (frame["skip_mode"] == "drop")
        & np.isclose(frame["deadline_ratio"], 1.5)
        & np.isclose(frame["epsilon"], 0.01)
    ].copy()
    return path, rows.sort_values("rho")


def cross_check_against_sweep(
    results: pd.DataFrame,
    repo_root: Path = REPO_ROOT,
    tolerance_pp: float = XCHECK_TOL_PP,
) -> dict[str, object] | None:
    """Compare Gate 0 with the committed sweep and print a soft warning only."""

    path, sweep_rows = _committed_cross_check_rows(repo_root)
    if path is None:
        print("Cross-check skipped: committed sweep CSV was not found.")
        return None
    if sweep_rows.empty:
        print(
            "Cross-check skipped: required columns/target rows were not found in "
            f"{path}."
        )
        return None

    gate_rows = results[
        np.isclose(results["deadline_ratio"], 1.5)
        & np.isclose(results["epsilon"], 0.01)
        & (results["status"] == "feasible")
    ]
    if gate_rows.empty:
        print("Cross-check skipped: Gate 0 target condition is not feasible.")
        return None
    gate_saving = float(gate_rows.iloc[0]["p2_saving_percent"])

    print(f"Cross-check source: {path}")
    for row in sweep_rows.itertuples(index=False):
        print(
            "  committed sweep "
            f"rho={float(row.rho):g}: P2 saving="
            f"{float(row.oracle_gap_percent_mean):.6f}%"
        )
    representative = sweep_rows[
        np.isclose(sweep_rows["rho"], 0.75)
    ]
    if representative.empty:
        representative = sweep_rows.iloc[[0]]
    sweep_saving = float(representative.iloc[0]["oracle_gap_percent_mean"])
    representative_rho = float(representative.iloc[0]["rho"])
    difference_pp = gate_saving - sweep_saving
    print(
        f"  representative rho={representative_rho:g}: Gate 0={gate_saving:.6f}%, "
        f"sweep={sweep_saving:.6f}%, difference={difference_pp:+.6f} pp"
    )
    if abs(difference_pp) > tolerance_pp:
        print(
            f"WARNING: cross-check difference exceeds {tolerance_pp:g} pp "
            "(soft warning; run continues)."
        )
    # Small differences are expected from finite-T path sampling, per-path
    # floor(epsilon*T) integer budgets versus this fractional budget, and the
    # committed sweep's burn-in exclusion convention.
    return {
        "source": path,
        "gate0_saving_percent": gate_saving,
        "sweep_saving_percent": sweep_saving,
        "representative_rho": representative_rho,
        "difference_pp": difference_pp,
        "warning": abs(difference_pp) > tolerance_pp,
    }


def _print_results_log(results: pd.DataFrame) -> None:
    for ratio, group in results.groupby("deadline_ratio", sort=True):
        epsilon_min = float(group.iloc[0]["epsilon_min"])
        print(
            f"D/D_min={ratio:.2f}: epsilon_min={100.0 * epsilon_min:.6f}%"
        )
        for row in group.sort_values("epsilon").itertuples(index=False):
            if row.status == "infeasible":
                print(
                    f"  epsilon={100.0 * row.epsilon:.2f}%: infeasible "
                    f"(epsilon_min={100.0 * row.epsilon_min:.6f}%)"
                )
            else:
                print(
                    f"  epsilon={100.0 * row.epsilon:.2f}%: "
                    f"P2 ceiling={row.p2_saving_percent:.6f}%"
                )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    config = default_experiment()
    profile = config.devices[0].profile
    d_min_ms = minimum_good_deadline_ms(
        profile,
        config.channel.r_good_mbps,
        config.channel.tx_power_w,
    )
    print(f"profile: {profile_identifier(profile)} / energy_unit: {ENERGY_UNIT}")
    print(f"sample count: {N_SAMPLES} / random seed: {RANDOM_SEED}")
    print(f"D_min: {d_min_ms:.9g} ms")
    print(f"output directory: {EXPERIMENT_DIR}")

    results = calculate_experiment()
    summary = build_summary(results)
    result_path = RESULTS_DIR / "drop_ceiling.csv"
    summary_path = RESULTS_DIR / "drop_ceiling_summary.csv"
    figure_path = FIGURES_DIR / "p2_saving_frontier.png"
    forced_path = FIGURES_DIR / "forced_drop_frontier.png"
    results.to_csv(result_path, index=False, na_rep="NaN")
    summary.to_csv(summary_path, index=False, na_rep="NaN")
    plot_saving_frontier(results, figure_path)
    plot_forced_drop_frontier(results, forced_path)

    _print_results_log(results)
    cross_check_against_sweep(results)
    print(f"results: {result_path}")
    print(f"summary: {summary_path}")
    print(f"figure: {figure_path}")
    print(f"figure: {forced_path}")
    print("CSV preview:")
    print(results.head().to_string(index=False))


if __name__ == "__main__":
    main()
