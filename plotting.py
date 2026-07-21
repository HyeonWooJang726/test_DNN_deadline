"""Matplotlib figures required by H1a/H1b and sanity diagnostics."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


POLICY_COLORS = {
    "P1": "#636363",
    "P0": "#9ecae1",
    "P2": "#2171b5",
    "P2prime": "#6baed6",
    "P3": "#f28e2b",
}


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_gap_heatmaps(
    comparison_aggregate: pd.DataFrame,
    preflight: pd.DataFrame,
    deadline_ratios: tuple[float, ...],
    epsilons: tuple[float, ...],
    skip_modes: tuple[str, ...],
    output_dir: Path,
) -> None:
    for skip_mode in skip_modes:
        valid_data = comparison_aggregate[comparison_aggregate["skip_mode"] == skip_mode]
        # H1a's offline quantity is averaged over L: with fixed stationary
        # marginal it should not depend on correlation time.
        means = valid_data.groupby(["deadline_ratio", "epsilon"], as_index=False)[
            "targeting_gain_percent_mean"
        ].mean()
        pre = preflight[preflight["skip_mode"] == skip_mode]
        grid = np.full((len(epsilons), len(deadline_ratios)), np.nan)
        for iy, eps in enumerate(epsilons):
            for ix, ratio in enumerate(deadline_ratios):
                validity = pre[
                    np.isclose(pre["epsilon"], eps)
                    & np.isclose(pre["deadline_ratio"], ratio)
                ]["valid"]
                if len(validity) and bool(validity.any()):
                    row = means[
                        np.isclose(means["epsilon"], eps)
                        & np.isclose(means["deadline_ratio"], ratio)
                    ]
                    if len(row):
                        grid[iy, ix] = float(row.iloc[0]["targeting_gain_percent_mean"])

        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("#d9d9d9")
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        image = ax.imshow(np.ma.masked_invalid(grid), origin="lower", aspect="auto", cmap=cmap)
        ax.set_xticks(range(len(deadline_ratios)), [str(v) for v in deadline_ratios])
        ax.set_yticks(range(len(epsilons)), [str(v) for v in epsilons])
        ax.set_xlabel(r"Deadline tightness $D/D_{min}$")
        ax.set_ylabel(r"Violation budget $\epsilon$")
        ax.set_title(f"Pure temporal targeting gain: {skip_mode} (rG/rB=4)")
        for iy in range(len(epsilons)):
            for ix in range(len(deadline_ratios)):
                if np.isnan(grid[iy, ix]):
                    label = "invalid"
                    ax.add_patch(
                        plt.Rectangle((ix - 0.5, iy - 0.5), 1, 1, fill=False, hatch="///", edgecolor="#777777", linewidth=0)
                    )
                else:
                    label = f"{grid[iy, ix]:.2f}%"
                ax.text(ix, iy, label, ha="center", va="center", fontsize=9, color="black")
        fig.colorbar(image, ax=ax, label=r"$(E_{P0}-E_{P2})/E_{P1}$ [%]")
        _save(fig, output_dir / f"gap_heatmap_{skip_mode}.png")


def plot_decomposition(
    policy_aggregate: pd.DataFrame,
    output_dir: Path,
    representative_ratio: float = 1.5,
    representative_epsilon: float = 0.05,
) -> None:
    data = policy_aggregate[
        np.isclose(policy_aggregate["deadline_ratio"], representative_ratio)
        & np.isclose(policy_aggregate["epsilon"], representative_epsilon)
        & policy_aggregate["L"].isin([1, 10, 50])
    ].copy()
    if data.empty:
        return
    policies = ["P1", "P0", "P2", "P2prime", "P3"]
    panels = sorted(data[["skip_mode", "L"]].drop_duplicates().itertuples(index=False, name=None))
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    axes = axes.ravel()
    all_values = []
    for ax, (skip_mode, l_value) in zip(axes, panels):
        subset = data[(data["skip_mode"] == skip_mode) & (data["L"] == l_value)].set_index("policy")
        if "P1" not in subset.index:
            ax.axis("off")
            continue
        baseline = float(subset.loc["P1", "mean_energy_j_mean"])
        values = [100.0 * float(subset.loc[p, "mean_energy_j_mean"]) / baseline for p in policies]
        all_values.extend(values)
        errors = [100.0 * float(subset.loc[p, "mean_energy_j_std"]) / baseline for p in policies]
        ax.bar(policies, values, yerr=errors, capsize=3, color=[POLICY_COLORS[p] for p in policies])
        ax.set_title(f"{skip_mode}, L={l_value}")
        ax.grid(axis="y", alpha=0.25)
    if all_values:
        axes[0].set_ylim(bottom=max(0.0, min(all_values) - 3.0), top=102.0)
    for ax in axes[::3]:
        ax.set_ylabel("Energy normalized by P1 [%]")
    fig.suptitle(
        f"Energy decomposition (D/Dmin={representative_ratio}, epsilon={representative_epsilon})\n"
        "P1-P0: discard benefit; P0-P2: temporal-targeting benefit",
        y=1.02,
    )
    _save(fig, output_dir / "energy_decomposition.png")


def plot_l_dependence(
    comparison_aggregate: pd.DataFrame,
    policy_aggregate: pd.DataFrame,
    output_dir: Path,
    gap_mask_percent: float,
    representative_ratio: float = 1.5,
    representative_epsilon: float = 0.05,
) -> None:
    comp = comparison_aggregate[
        np.isclose(comparison_aggregate["deadline_ratio"], representative_ratio)
        & np.isclose(comparison_aggregate["epsilon"], representative_epsilon)
    ].sort_values("L")
    pol = policy_aggregate[
        np.isclose(policy_aggregate["deadline_ratio"], representative_ratio)
        & np.isclose(policy_aggregate["epsilon"], representative_epsilon)
        & policy_aggregate["policy"].isin(["P2", "P3"])
    ].sort_values("L")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for skip_mode, group in comp.groupby("skip_mode"):
        y = group["online_recovery_mean"].to_numpy(copy=True)
        oracle_gap = group["oracle_gap_percent_mean"].to_numpy()
        y[oracle_gap < gap_mask_percent] = np.nan
        ax.errorbar(group["L"], y, yerr=group["online_recovery_std"], marker="o", capsize=3, label=skip_mode)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set(xlabel="Bad-state mean dwell L [slots]", ylabel=r"Online recovery $(P1-P3)/(P1-P2)$", title=f"H1b(a): online oracle recovery (gap < {gap_mask_percent:g}% masked)")
    ax.set_xscale("log")
    ax.set_xticks(sorted(comp["L"].unique()), [str(int(v)) for v in sorted(comp["L"].unique())])
    ax.grid(alpha=0.25)
    ax.legend()
    _save(fig, output_dir / "l_online_recovery.png")

    for skip_mode, group in pol.groupby("skip_mode"):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for policy, part in group.groupby("policy"):
            color = POLICY_COLORS[policy]
            axes[0].errorbar(part["L"], part["max_violation_run_mean"], yerr=part["max_violation_run_std"], marker="o", capsize=3, color=color, label=policy)
            axes[1].errorbar(part["L"], part["burst_count_ge2_mean"], yerr=part["burst_count_ge2_std"], marker="o", capsize=3, color=color, label=policy)
        for ax in axes:
            ax.set_xscale("log")
            ax.set_xticks(sorted(group["L"].unique()), [str(int(v)) for v in sorted(group["L"].unique())])
            ax.set_xlabel("Bad-state mean dwell L [slots]")
            ax.grid(alpha=0.25)
            ax.legend()
        axes[0].set_ylabel("Maximum consecutive violations")
        axes[1].set_ylabel("Number of violation bursts (length >= 2)")
        fig.suptitle(f"H1b(b): violation burstiness ({skip_mode})")
        _save(fig, output_dir / f"l_burstiness_{skip_mode}.png")

    # At epsilon=0.05 there are enough isolated high-saving Bad slots for the
    # no-adjacent constraint to be inactive. epsilon=0.10 deliberately exposes
    # the P2-P2prime correlation cost requested in H1b(c).
    comp_gap = comparison_aggregate[
        np.isclose(comparison_aggregate["deadline_ratio"], representative_ratio)
        & np.isclose(comparison_aggregate["epsilon"], 0.10)
    ].sort_values("L")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for skip_mode, group in comp_gap.groupby("skip_mode"):
        ax.errorbar(group["L"], group["p2prime_gap_percent_mean"], yerr=group["p2prime_gap_percent_std"], marker="o", capsize=3, label=skip_mode)
    ax.set_xscale("log")
    ax.set_xticks(sorted(comp_gap["L"].unique()), [str(int(v)) for v in sorted(comp_gap["L"].unique())])
    ax.set(xlabel="Bad-state mean dwell L [slots]", ylabel=r"$(P2'-P2)/P1$ [%]", title="H1b(c): no-consecutive-violation cost (epsilon=0.10)")
    ax.grid(alpha=0.25)
    ax.legend()
    _save(fig, output_dir / "l_p2prime_gap.png")


def plot_oracle_invariance(
    comparison_aggregate: pd.DataFrame,
    channel_stats: pd.DataFrame,
    output_dir: Path,
    representative_ratio: float = 1.5,
    representative_epsilon: float = 0.05,
) -> None:
    data = comparison_aggregate[
        np.isclose(comparison_aggregate["deadline_ratio"], representative_ratio)
        & np.isclose(comparison_aggregate["epsilon"], representative_epsilon)
    ].sort_values("L")
    channel_mean = channel_stats.groupby("L", as_index=False)["pi_bad_observed"].mean()
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for skip_mode, group in data.groupby("skip_mode"):
        ax.errorbar(group["L"], group["oracle_gap_percent_mean"], yerr=group["oracle_gap_percent_std"], marker="o", capsize=3, label=f"oracle gap ({skip_mode})")
    ax.set_xscale("log")
    ax.set_xticks(sorted(data["L"].unique()), [str(int(v)) for v in sorted(data["L"].unique())])
    ax.set_xlabel("Bad-state mean dwell L [slots]")
    ax.set_ylabel(r"Offline oracle gap $(P1-P2)/P1$ [%]")
    ax.grid(alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(channel_mean["L"], channel_mean["pi_bad_observed"], color="#d62728", marker="s", linestyle="--", label="observed pi_B")
    ax2.axhline(0.2, color="#d62728", linewidth=0.8, alpha=0.5)
    ax2.set_ylabel("Observed Bad-state fraction", color="#d62728")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")
    ax.set_title("Sanity: offline oracle gap should be flat in L")
    _save(fig, output_dir / "sanity_oracle_gap_vs_l.png")


def create_all_plots(
    policy_aggregate: pd.DataFrame,
    comparison_aggregate: pd.DataFrame,
    preflight: pd.DataFrame,
    channel_stats: pd.DataFrame,
    deadline_ratios: tuple[float, ...],
    epsilons: tuple[float, ...],
    skip_modes: tuple[str, ...],
    gap_mask_percent: float,
    output_dir: Path,
) -> None:
    plot_gap_heatmaps(comparison_aggregate, preflight, deadline_ratios, epsilons, skip_modes, output_dir)
    plot_decomposition(policy_aggregate, output_dir)
    plot_l_dependence(comparison_aggregate, policy_aggregate, output_dir, gap_mask_percent)
    plot_oracle_invariance(comparison_aggregate, channel_stats, output_dir)
