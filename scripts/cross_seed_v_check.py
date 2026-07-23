"""Measure the robustness of post-hoc P3 V calibration across random seeds."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np

from channel import GilbertElliottChannel
from config import default_experiment
from dnn_profile import SlotCosts, compute_slot_costs, minimum_good_deadline_ms
from policies import P1AlwaysMeet, P2OfflineOracle, _threshold_mask


T_SLOTS = 10_000
SEEDS = (1701, 1702, 1703, 1704, 1705)
RHO_VALUES = (0.0, 0.75, 0.975)
DEADLINE_EPSILON_PAIRS = ((1.1, 0.15), (1.2, 0.15), (1.5, 0.05))
SKIP_MODES = ("drop", "late")
REPORT_PATH = ROOT_DIR / "docs" / "reports" / "cross_seed_v_robustness.md"


@dataclass(frozen=True)
class Candidate:
    v: float
    mean_energy_j: float
    violation_rate: float


@dataclass(frozen=True)
class SeedEvaluation:
    costs: SlotCosts
    discretionary_budget: int
    p1_energy_j: float
    p2_energy_j: float
    selected_v: float
    post_hoc_recovery: float


def _evaluate_v(
    costs: SlotCosts,
    epsilon: float,
    v_parameter: float,
    discretionary_budget: int,
) -> Candidate:
    raw, _, _ = _threshold_mask(
        costs.feasible,
        costs.saving_j,
        epsilon,
        v_parameter,
        discretionary_budget,
    )
    violate = (~costs.feasible) | raw.astype(bool)
    energy = np.where(violate, costs.skip_energy_j, costs.meet_energy_j)
    return Candidate(
        v=float(v_parameter),
        mean_energy_j=float(np.mean(energy)),
        violation_rate=float(np.mean(violate)),
    )


def _select_candidate(
    costs: SlotCosts,
    epsilon: float,
    v_values: tuple[float, ...],
    discretionary_budget: int,
    violation_tolerance: float,
) -> Candidate:
    candidates = [
        _evaluate_v(costs, epsilon, v, discretionary_budget) for v in v_values
    ]
    within_budget = [
        candidate
        for candidate in candidates
        if candidate.violation_rate <= epsilon + violation_tolerance
    ]
    if within_budget:
        return min(within_budget, key=lambda candidate: (candidate.mean_energy_j, candidate.v))
    return min(
        candidates,
        key=lambda candidate: (
            abs(candidate.violation_rate - epsilon),
            candidate.mean_energy_j,
            candidate.v,
        ),
    )


def _recovery(
    p1_energy_j: float,
    p2_energy_j: float,
    p3_energy_j: float,
) -> float:
    denominator = p1_energy_j - p2_energy_j
    if denominator <= 0.0:
        raise ValueError("P1-P2 recovery denominator must be positive")
    return (p1_energy_j - p3_energy_j) / denominator


def _generate_traces(channel_config) -> dict[tuple[float, int], object]:
    traces = {}
    for rho in RHO_VALUES:
        source = GilbertElliottChannel(
            r_good_mbps=channel_config.r_good_mbps,
            r_bad_mbps=channel_config.r_bad_mbps,
            pi_bad=channel_config.pi_bad,
            rho=rho,
            rate_jitter_sigma_log=channel_config.rate_jitter_sigma_log,
            marginal_tolerance=channel_config.enforce_marginal_tolerance,
            max_resamples=channel_config.max_trace_resamples,
        )
        for seed in SEEDS:
            traces[(rho, seed)] = source.generate(T_SLOTS, seed)
    return traces


def _evaluate_seed(
    trace,
    device,
    channel_config,
    deadline_ms: float,
    epsilon: float,
    skip_mode: str,
    v_values: tuple[float, ...],
    violation_tolerance: float,
) -> SeedEvaluation:
    costs = compute_slot_costs(
        device.profile,
        trace.rate_mbps,
        deadline_ms,
        channel_config.tx_power_w,
        skip_mode,
    )
    forced_count = int((~costs.feasible).sum())
    total_budget = int(np.floor(epsilon * T_SLOTS))
    discretionary_budget = total_budget - forced_count
    if discretionary_budget < 0:
        raise ValueError(
            f"forced count {forced_count} exceeds floor(epsilon*T)={total_budget}"
        )

    p1 = P1AlwaysMeet().run(costs)
    p2 = P2OfflineOracle(discretionary_budget).run(costs)
    selected = _select_candidate(
        costs,
        epsilon,
        v_values,
        discretionary_budget,
        violation_tolerance,
    )
    return SeedEvaluation(
        costs=costs,
        discretionary_budget=discretionary_budget,
        p1_energy_j=p1.mean_energy_j,
        p2_energy_j=p2.mean_energy_j,
        selected_v=selected.v,
        post_hoc_recovery=_recovery(
            p1.mean_energy_j,
            p2.mean_energy_j,
            selected.mean_energy_j,
        ),
    )


def _report_markdown(rows: list[dict[str, float | str]]) -> str:
    maximum = max(rows, key=lambda row: abs(float(row["gap_pp"])))
    all_below_one = all(abs(float(row["gap_pp"])) < 1.0 for row in rows)
    lines = [
        "# Cross-seed robustness of P3 V calibration",
        "",
        "## Method",
        "",
        "This check uses `T=10,000`, seeds 1701–1705, the configured 15-point",
        "P3 V grid, and `violation_tolerance=0.005`. For each evaluation seed",
        "`s_i`, post-hoc recovery uses the V calibrated on `s_i`; cross-seed",
        "recovery uses the V calibrated on `s_(i+1) mod 5`. Only V transfers.",
        "Both evaluations recompute `floor(epsilon*T)-forced_count` from the",
        "evaluation trace. P1, P2, channel generation, slot costs, and P3's",
        "threshold mask are imported from the simulator modules.",
        "",
        "The gap is post-hoc recovery minus cross-seed recovery, in percentage",
        "points. V match is the fraction of the five transfers whose source V",
        "equals the evaluation seed's post-hoc V.",
        "",
        "## Results",
        "",
        "| rho | D/D_min | epsilon | skip | Post-hoc mean | Cross-seed mean | Gap (pp) | V match |",
        "|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {float(row['rho']):g} | {float(row['deadline_ratio']):g} | "
            f"{float(row['epsilon']):g} | {row['skip_mode']} | "
            f"{100.0 * float(row['post_hoc_mean']):.4f}% | "
            f"{100.0 * float(row['cross_seed_mean']):.4f}% | "
            f"{float(row['gap_pp']):+.4f} | "
            f"{100.0 * float(row['v_match_rate']):.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Maximum absolute gap: **{abs(float(maximum['gap_pp'])):.4f} pp** "
            f"at `rho={float(maximum['rho']):g}, "
            f"D/D_min={float(maximum['deadline_ratio']):g}, "
            f"epsilon={float(maximum['epsilon']):g}, "
            f"skip={maximum['skip_mode']}`.",
            f"- Every tested condition has `|gap| < 1 pp`: "
            f"**{'yes' if all_below_one else 'no'}**.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    config = default_experiment("smoke")
    sweep = config.sweep
    device = config.devices[0]
    channel_config = config.channel
    d_min = minimum_good_deadline_ms(
        device.profile,
        channel_config.r_good_mbps,
        channel_config.tx_power_w,
    )
    traces = _generate_traces(channel_config)
    rows: list[dict[str, float | str]] = []

    for rho in RHO_VALUES:
        for deadline_ratio, epsilon in DEADLINE_EPSILON_PAIRS:
            deadline_ms = deadline_ratio * d_min
            for skip_mode in SKIP_MODES:
                evaluations = {
                    seed: _evaluate_seed(
                        traces[(rho, seed)],
                        device,
                        channel_config,
                        deadline_ms,
                        epsilon,
                        skip_mode,
                        sweep.p3_v_values,
                        sweep.violation_tolerance,
                    )
                    for seed in SEEDS
                }
                cross_recoveries = []
                matches = []
                for index, seed in enumerate(SEEDS):
                    source_seed = SEEDS[(index + 1) % len(SEEDS)]
                    target = evaluations[seed]
                    transferred_v = evaluations[source_seed].selected_v
                    transferred = _evaluate_v(
                        target.costs,
                        epsilon,
                        transferred_v,
                        target.discretionary_budget,
                    )
                    cross_recoveries.append(
                        _recovery(
                            target.p1_energy_j,
                            target.p2_energy_j,
                            transferred.mean_energy_j,
                        )
                    )
                    matches.append(transferred_v == target.selected_v)

                post_hoc_mean = float(
                    np.mean(
                        [evaluation.post_hoc_recovery for evaluation in evaluations.values()]
                    )
                )
                cross_seed_mean = float(np.mean(cross_recoveries))
                rows.append(
                    {
                        "rho": rho,
                        "deadline_ratio": deadline_ratio,
                        "epsilon": epsilon,
                        "skip_mode": skip_mode,
                        "post_hoc_mean": post_hoc_mean,
                        "cross_seed_mean": cross_seed_mean,
                        "gap_pp": 100.0 * (post_hoc_mean - cross_seed_mean),
                        "v_match_rate": float(np.mean(matches)),
                    }
                )

    report = _report_markdown(rows)
    REPORT_PATH.write_text(report, encoding="utf-8")
    maximum = max(rows, key=lambda row: abs(float(row["gap_pp"])))
    print(f"Wrote {REPORT_PATH}")
    print(f"Evaluated {len(rows)} conditions across {len(SEEDS)} seeds")
    print(
        "Maximum absolute recovery gap: "
        f"{abs(float(maximum['gap_pp'])):.4f} pp "
        f"(rho={float(maximum['rho']):g}, "
        f"D/D_min={float(maximum['deadline_ratio']):g}, "
        f"epsilon={float(maximum['epsilon']):g}, "
        f"skip={maximum['skip_mode']})"
    )


if __name__ == "__main__":
    main()
