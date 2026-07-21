"""Metrics, violation-pattern summaries, sanity checks, and stable hashing."""

from __future__ import annotations

import hashlib
import json
from typing import Iterable

import numpy as np

from channel import BAD, GOOD
from policies import PolicyResult
from simulator import SimulationResult


INTERVAL_BINS = np.array([1, 2, 3, 5, 10, 20, 50, 100, 250, 500, np.inf])


def run_statistics(violations: np.ndarray) -> tuple[int, int, int]:
    """Return maximum run, number of >=2 runs, and number of all runs."""
    x = np.asarray(violations, dtype=np.int8)
    if not np.any(x):
        return 0, 0, 0
    padded = np.pad(x, (1, 1))
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    lengths = ends - starts
    return int(lengths.max()), int(np.sum(lengths >= 2)), int(len(lengths))


def summarize_policy(
    result: PolicyResult,
    channel_states: np.ndarray,
) -> dict[str, float | int | str]:
    maximum, bursts, runs = run_statistics(result.violate)
    at_violation = channel_states[result.violate]
    known = at_violation[at_violation >= 0]
    bad_share = float(np.mean(known == BAD)) if len(known) else float("nan")
    return {
        "policy": result.name,
        "mean_energy_j": result.mean_energy_j,
        "total_energy_j": float(result.energy_j.sum()),
        "violation_rate": result.violation_rate,
        "violation_count": int(result.violate.sum()),
        "max_violation_run": maximum,
        "burst_count_ge2": bursts,
        "violation_run_count": runs,
        "violation_state_bad_share": bad_share,
        "violation_state_good_count": int(np.sum(at_violation == GOOD)),
        "violation_state_bad_count": int(np.sum(at_violation == BAD)),
        "selected_v": float(result.metadata.get("selected_v", np.nan)),
        "q_final": float(result.metadata.get("q_final", np.nan)),
        "q_max": float(result.metadata.get("q_max", np.nan)),
        "policy_metadata_json": json.dumps(result.metadata, sort_keys=True),
    }


def interval_histogram(result: PolicyResult) -> list[dict[str, float | int | str]]:
    indices = np.flatnonzero(result.violate)
    intervals = np.diff(indices)
    rows = []
    for low, high in zip(INTERVAL_BINS[:-1], INTERVAL_BINS[1:]):
        count = int(np.sum((intervals >= low) & (intervals < high)))
        high_label = "inf" if np.isinf(high) else str(int(high))
        rows.append(
            {
                "policy": result.name,
                "interval_low_inclusive": int(low),
                "interval_high_exclusive": high_label,
                "count": count,
                "fraction": float(count / len(intervals)) if len(intervals) else 0.0,
            }
        )
    return rows


def _leq(left: float, right: float, scale: float, relative_tolerance: float) -> bool:
    return left <= right + relative_tolerance * max(abs(scale), 1e-12)


def combination_sanity_rows(
    simulation: SimulationResult,
    epsilon: float,
    relative_energy_tolerance: float,
    violation_tolerance: float,
) -> list[dict[str, object]]:
    p = simulation.policies
    e = {name: value.mean_energy_j for name, value in p.items()}
    scale = e["P1"]
    checks: list[tuple[str, bool, str]] = [
        ("energy_P2_le_P2prime", _leq(e["P2"], e["P2prime"], scale, relative_energy_tolerance), f"{e['P2']:.9g} <= {e['P2prime']:.9g}"),
        ("energy_P2prime_le_P3", _leq(e["P2prime"], e["P3"], scale, relative_energy_tolerance), f"{e['P2prime']:.9g} <= {e['P3']:.9g}"),
        ("energy_P2_le_P0", _leq(e["P2"], e["P0"], scale, relative_energy_tolerance), f"{e['P2']:.9g} <= {e['P0']:.9g}"),
        ("energy_P0_le_P1", _leq(e["P0"], e["P1"], scale, relative_energy_tolerance), f"{e['P0']:.9g} <= {e['P1']:.9g}"),
    ]
    for name in ("P0", "P2", "P2prime", "P3"):
        rate = p[name].violation_rate
        checks.append(
            (f"violation_{name}_within_budget", rate <= epsilon + violation_tolerance + 1e-12, f"{rate:.9g} <= {epsilon + violation_tolerance:.9g}")
        )
    p1_rate = p["P1"].violation_rate
    forced_rate = simulation.forced_count / len(p["P1"].violate)
    checks.append(
        ("P1_equals_forced", abs(p1_rate - forced_rate) <= 1.0 / len(p["P1"].violate), f"{p1_rate:.9g} == {forced_rate:.9g}")
    )
    return [{"check": name, "passed": bool(passed), "detail": detail} for name, passed, detail in checks]


def assert_sanity(rows: Iterable[dict[str, object]]) -> None:
    failures = [
        row
        for row in rows
        if row.get("severity", "required") == "required" and not bool(row["passed"])
    ]
    assert not failures, "sanity failures: " + "; ".join(
        f"{row['check']} ({row['detail']})" for row in failures
    )


def stable_simulation_digest(simulation: SimulationResult) -> str:
    digest = hashlib.sha256()
    digest.update(np.asarray(simulation.forced_count, dtype=np.int64).tobytes())
    digest.update(np.asarray(simulation.discretionary_budget, dtype=np.int64).tobytes())
    for name in sorted(simulation.policies):
        result = simulation.policies[name]
        digest.update(name.encode("utf-8"))
        digest.update(np.ascontiguousarray(result.violate).tobytes())
        digest.update(np.ascontiguousarray(result.split_p).tobytes())
        digest.update(np.ascontiguousarray(result.energy_j).tobytes())
        digest.update(json.dumps(result.metadata, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()
