"""Policy implementations sharing a common decide/run interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from dnn_profile import SlotCosts

try:
    from numba import njit
except ImportError:  # pragma: no cover
    def njit(*args, **kwargs):
        def decorate(fn):
            return fn
        return decorate


@dataclass(frozen=True)
class SlotState:
    t: int
    feasible: bool
    meet_p: int
    meet_local_mode: int
    meet_energy_j: float
    skip_p: int
    skip_local_mode: int
    skip_energy_j: float
    saving_j: float


@dataclass
class PolicyResult:
    name: str
    violate: np.ndarray
    split_p: np.ndarray
    local_mode: np.ndarray
    energy_j: np.ndarray
    mode_names: tuple[str, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def violation_rate(self) -> float:
        return float(np.mean(self.violate))

    @property
    def mean_energy_j(self) -> float:
        return float(np.mean(self.energy_j))


class Policy(ABC):
    name: str

    @abstractmethod
    def decide(self, state: SlotState) -> tuple[bool, int, int, float]:
        """Return (violate, split point, local mode, energy) for one slot."""

    @abstractmethod
    def run(self, costs: SlotCosts) -> PolicyResult:
        raise NotImplementedError


def _result_from_optional_mask(name: str, costs: SlotCosts, optional: np.ndarray, **metadata) -> PolicyResult:
    forced = ~costs.feasible
    violate = forced | optional
    p = np.where(violate, costs.skip_p, costs.meet_p).astype(np.int16)
    local_mode = np.where(
        violate, costs.skip_local_mode, costs.meet_local_mode
    ).astype(np.int8)
    energy = np.where(violate, costs.skip_energy_j, costs.meet_energy_j)
    return PolicyResult(name, violate, p, local_mode, energy, costs.mode_names, metadata)


class P1AlwaysMeet(Policy):
    name = "P1"

    def decide(self, state: SlotState) -> tuple[bool, int, int, float]:
        if state.feasible:
            return False, state.meet_p, state.meet_local_mode, state.meet_energy_j
        return True, state.skip_p, state.skip_local_mode, state.skip_energy_j

    def run(self, costs: SlotCosts) -> PolicyResult:
        return _result_from_optional_mask(self.name, costs, np.zeros(len(costs.feasible), dtype=bool))


class _PreparedMaskPolicy(Policy):
    selected: np.ndarray | None = None

    def decide(self, state: SlotState) -> tuple[bool, int, int, float]:
        optional = bool(self.selected is not None and self.selected[state.t])
        violate = (not state.feasible) or optional
        if violate:
            return True, state.skip_p, state.skip_local_mode, state.skip_energy_j
        return False, state.meet_p, state.meet_local_mode, state.meet_energy_j


class P0RandomDrop(_PreparedMaskPolicy):
    """Randomly spends the same discretionary budget as P2.

    The central decomposition is intentionally explicit:
      P1-P0 = benefit of allowing violations at all;
      P0-P2 = pure benefit of targeting those violations in time.
    """

    name = "P0"

    def __init__(self, budget: int, rng: np.random.Generator):
        self.budget = int(budget)
        self.rng = rng

    def run(self, costs: SlotCosts) -> PolicyResult:
        eligible = np.flatnonzero(costs.feasible)
        count = min(self.budget, len(eligible))
        self.selected = np.zeros(len(costs.feasible), dtype=bool)
        if count:
            self.selected[self.rng.choice(eligible, size=count, replace=False)] = True
        return _result_from_optional_mask(self.name, costs, self.selected, budget=count)


class P2OfflineOracle(_PreparedMaskPolicy):
    name = "P2"

    def __init__(self, budget: int):
        self.budget = int(budget)

    def run(self, costs: SlotCosts) -> PolicyResult:
        eligible = np.flatnonzero(costs.feasible)
        count = min(self.budget, len(eligible))
        self.selected = np.zeros(len(costs.feasible), dtype=bool)
        if count:
            # Slots are independent and every violation has unit budget cost.
            # Therefore selecting the B largest nonnegative savings is exactly
            # the unit-weight knapsack optimum (not merely a heuristic).
            order = eligible[np.lexsort((eligible, -costs.saving_j[eligible]))]
            self.selected[order[:count]] = True
        return _result_from_optional_mask(self.name, costs, self.selected, budget=count)


def exact_burst_selection(weights: np.ndarray, allowed: np.ndarray, budget: int) -> np.ndarray:
    """Exact O(T*B) weighted path independent-set DP, used for small cases/tests."""
    w = np.asarray(weights, dtype=np.float64)
    ok = np.asarray(allowed, dtype=bool)
    n = len(w)
    bmax = min(int(budget), (n + 1) // 2)
    neg_inf = -np.inf
    dp = np.full((n + 1, bmax + 1), neg_inf, dtype=np.float64)
    take = np.zeros((n + 1, bmax + 1), dtype=bool)
    dp[:, 0] = 0.0
    for i in range(1, n + 1):
        dp[i] = dp[i - 1]
        if ok[i - 1]:
            base_row = dp[i - 2] if i >= 2 else dp[0]
            for b in range(1, bmax + 1):
                candidate = base_row[b - 1] + w[i - 1]
                if candidate > dp[i, b] + 1e-15:
                    dp[i, b] = candidate
                    take[i, b] = True
    b = int(np.nanargmax(dp[n]))
    selected = np.zeros(n, dtype=bool)
    i = n
    while i > 0 and b >= 0:
        if take[i, b]:
            selected[i - 1] = True
            b -= 1
            i -= 2
        else:
            i -= 1
    return selected


@njit(cache=True)
def _mwis_at_price(weights, allowed, price):
    """Maximum adjusted-weight independent set on a path."""
    n = len(weights)
    choose = np.zeros(n, dtype=np.uint8)
    dp_im2 = 0.0
    dp_im1 = 0.0
    for i in range(n):
        take_value = dp_im2 + weights[i] - price if allowed[i] else -1e300
        if take_value > dp_im1 + 1e-14:
            current = take_value
            choose[i] = 1
        else:
            current = dp_im1
        dp_im2 = dp_im1
        dp_im1 = current
    selected = np.zeros(n, dtype=np.uint8)
    i = n - 1
    count = 0
    while i >= 0:
        if choose[i] == 1:
            selected[i] = 1
            count += 1
            i -= 2
        else:
            i -= 1
    return selected, count


def lagrangian_burst_selection(weights: np.ndarray, allowed: np.ndarray, budget: int) -> np.ndarray:
    """Scalable cardinality-priced DP followed by deterministic augmentation.

    Each price evaluation is an exact path DP.  Binary search finds the
    Lagrange price whose solution is on the <=B side; zero-reduced-cost ties
    can make the cardinality jump, so safe highest-saving augmentation fills
    any remainder.  This is the requested Lagrangian relaxation for large T.
    """
    w = np.asarray(weights, dtype=np.float64)
    ok = np.asarray(allowed, dtype=np.bool_)
    if budget <= 0 or not np.any(ok):
        return np.zeros(len(w), dtype=bool)
    raw, count = _mwis_at_price(w, ok, 0.0)
    if count <= budget:
        return raw.astype(bool)
    lo = 0.0
    hi = float(np.max(w)) + 1e-12
    best = np.zeros(len(w), dtype=np.uint8)
    for _ in range(42):
        mid = (lo + hi) / 2.0
        candidate, count = _mwis_at_price(w, ok, mid)
        if count > budget:
            lo = mid
        else:
            hi = mid
            best = candidate
    selected = best.astype(bool)
    remaining = budget - int(selected.sum())
    if remaining > 0:
        eligible = np.flatnonzero(ok & ~selected)
        order = eligible[np.lexsort((eligible, -w[eligible]))]
        for i in order:
            if remaining == 0:
                break
            if (i == 0 or not selected[i - 1]) and (i + 1 == len(w) or not selected[i + 1]):
                selected[i] = True
                remaining -= 1
    return selected


class P2BurstOracle(_PreparedMaskPolicy):
    name = "P2prime"

    def __init__(self, budget: int, exact_cutoff: int = 2_000):
        self.budget = int(budget)
        self.exact_cutoff = exact_cutoff

    def run(self, costs: SlotCosts) -> PolicyResult:
        forced = ~costs.feasible
        allowed = costs.feasible.copy()
        # Avoid creating an optional violation adjacent to a forced one. Forced-
        # forced adjacency itself is unavoidable and is retained in statistics.
        allowed[1:] &= ~forced[:-1]
        allowed[:-1] &= ~forced[1:]
        if len(allowed) <= self.exact_cutoff:
            self.selected = exact_burst_selection(costs.saving_j, allowed, self.budget)
            method = "exact_cardinality_dp"
        else:
            self.selected = lagrangian_burst_selection(costs.saving_j, allowed, self.budget)
            method = "lagrangian_path_dp"
        return _result_from_optional_mask(
            self.name,
            costs,
            self.selected,
            budget=int(self.selected.sum()),
            method=method,
        )


@njit(cache=True)
def _threshold_mask(feasible, saving, epsilon, v_parameter, discretionary_budget):
    n = len(feasible)
    optional = np.zeros(n, dtype=np.uint8)
    queue = 0.0
    queue_max = 0.0
    selected_count = 0
    for t in range(n):
        if feasible[t]:
            violate = 1 if (
                saving[t] > queue / v_parameter
                and selected_count < discretionary_budget
            ) else 0
            optional[t] = violate
            selected_count += violate
        else:
            violate = 1
        queue = queue + violate - epsilon
        if queue < 0.0:
            queue = 0.0
        if queue > queue_max:
            queue_max = queue
    return optional, queue, queue_max


class P3OnlineThreshold(_PreparedMaskPolicy):
    """Calibrated-V upper bound of online performance.

    V is selected post hoc per combination. A coarse V grid can push this
    optimistic upper bound downward and invalidate the intended a-fortiori
    comparison, so calibration uses the dense grid supplied by SweepConfig.

    Every candidate caps optional violations at floor(epsilon*T) minus the
    trace's complete forced-violation count. Using that future forced count is
    offline information, like post-hoc V calibration, and is part of P3's
    optimistic-upper-bound framing. A deployable policy must reserve budget
    for future forced violations and therefore performs at or below P3.
    """

    name = "P3"

    def __init__(
        self,
        epsilon: float,
        v_values: tuple[float, ...],
        discretionary_budget: int,
        violation_tolerance: float,
    ):
        self.epsilon = float(epsilon)
        self.v_values = tuple(float(v) for v in v_values)
        self.discretionary_budget = int(discretionary_budget)
        self.violation_tolerance = float(violation_tolerance)

    def run(self, costs: SlotCosts) -> PolicyResult:
        candidates = []
        for v in self.v_values:
            raw, q_final, q_max = _threshold_mask(
                costs.feasible,
                costs.saving_j,
                self.epsilon,
                v,
                self.discretionary_budget,
            )
            optional = raw.astype(bool)
            result = _result_from_optional_mask("P3", costs, optional)
            candidates.append((abs(result.violation_rate - self.epsilon), result.mean_energy_j, v, optional, q_final, q_max, result.violation_rate))
        # V is calibrated per (trace, D, epsilon, skip mode), never globally.
        within_budget = [
            candidate
            for candidate in candidates
            if candidate[6] <= self.epsilon + self.violation_tolerance
        ]
        if within_budget:
            within_budget.sort(key=lambda item: (item[1], item[2]))
            selected = within_budget[0]
            selection_rule = "minimum_energy_within_violation_tolerance"
        else:
            candidates.sort(key=lambda item: (item[0], item[1], item[2]))
            selected = candidates[0]
            selection_rule = "closest_violation_rate_fallback"
        _, _, v, self.selected, q_final, q_max, rate = selected
        forced_count = int((~costs.feasible).sum())
        return _result_from_optional_mask(
            self.name,
            costs,
            self.selected,
            selected_v=v,
            q_final=float(q_final),
            q_max=float(q_max),
            calibrated_violation_rate=float(rate),
            horizon_budget_cap=forced_count + self.discretionary_budget,
            discretionary_budget_cap=self.discretionary_budget,
            post_hoc_forced_count=forced_count,
            violation_tolerance=self.violation_tolerance,
            selection_rule=selection_rule,
            candidate_rates={str(c[2]): float(c[6]) for c in candidates},
        )
