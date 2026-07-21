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
RHO_AXIS_LABEL = "Channel state correlation (mean bad-burst length)"


def _save(fig: plt.Figure, path: Path, *, tight_layout: bool = True) -> None:
    if tight_layout:
        fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def rho_label(rho, pi_bad=0.2):
    """Return a listener-friendly label for a state-correlation value."""
    rho = float(rho)
    if np.isclose(rho, 0.0):
        return "i.i.d."
    mean_bad_burst = 1.0 / ((1.0 - rho) * (1.0 - pi_bad))
    return f"ρ={rho:g} (≈{mean_bad_burst:.0f}-slot bursts)"


def _ordinal(ax, values):
    """비등간격 스윕 값을 등간격 위치에 매핑하고 눈금을 설정."""
    order = sorted(set(float(v) for v in values))
    pos = {v: i for i, v in enumerate(order)}
    ax.set_xticks(
        range(len(order)),
        [rho_label(v) for v in order],
        rotation=20,
        ha="right",
    )
    ax.tick_params(axis="x", labelsize=9)
    return pos


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
        # H1a's offline quantity is averaged over rho: with fixed stationary
        # marginal it should not depend on state correlation.
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
        ax.set_title(f"Pure temporal targeting gain: {skip_mode}")
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
        _save(fig, output_dir / f"fig_h1a_gap_heatmap_{skip_mode}.png")


def plot_decomposition(
    policy_aggregate: pd.DataFrame,
    output_dir: Path,
    representative_ratio: float = 1.5,
    representative_epsilon: float = 0.05,
) -> None:
    data = policy_aggregate[
        np.isclose(policy_aggregate["deadline_ratio"], representative_ratio)
        & np.isclose(policy_aggregate["epsilon"], representative_epsilon)
        & policy_aggregate["rho"].isin([0.0, 0.75, 0.975])
    ].copy()
    if data.empty:
        return
    policies = ["P1", "P0", "P2", "P2prime", "P3"]
    panels = sorted(data[["skip_mode", "rho"]].drop_duplicates().itertuples(index=False, name=None))
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(14.5, 8),
        sharey=True,
        layout="constrained",
    )
    axes = axes.ravel()
    all_values = []
    for ax, (skip_mode, rho) in zip(axes, panels):
        subset = (
            data[(data["skip_mode"] == skip_mode) & np.isclose(data["rho"], rho)]
            .drop_duplicates("policy", keep="first")
            .set_index("policy")
            .reindex(policies)
        )
        baseline_value = subset.loc["P1", "mean_energy_j_mean"]
        baseline = float(baseline_value) if pd.notna(baseline_value) else float("nan")
        values = []
        errors = []
        for policy in policies:
            mean_value = subset.loc[policy, "mean_energy_j_mean"]
            std_value = subset.loc[policy, "mean_energy_j_std"]
            if pd.isna(mean_value) or not np.isfinite(baseline) or baseline <= 0.0:
                values.append(float("nan"))
                errors.append(float("nan"))
            else:
                values.append(100.0 * float(mean_value) / baseline)
                errors.append(
                    100.0 * float(std_value) / baseline
                    if pd.notna(std_value)
                    else 0.0
                )
        finite_values = [value for value in values if np.isfinite(value)]
        all_values.extend(finite_values)
        x = np.arange(len(policies))
        ax.bar(
            x,
            values,
            yerr=errors,
            capsize=3,
            color=[POLICY_COLORS[p] for p in policies],
        )
        ax.set_xticks(x, policies)
        ax.set_xlim(-0.6, len(policies) - 0.4)
        for index, value in enumerate(values):
            if not np.isfinite(value):
                ax.text(
                    index,
                    0.5,
                    "n/a",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="center",
                    color="#666666",
                    fontsize=9,
                )
        channel_description = rho_label(rho)
        if channel_description != "i.i.d.":
            channel_description = f"correlated {channel_description[channel_description.index('('):]}"
        ax.set_title(f"{skip_mode} — {channel_description}")
        ax.grid(axis="y", alpha=0.25)
    if all_values:
        axes[0].set_ylim(bottom=max(0.0, min(all_values) - 3.0), top=102.0)
    for ax in axes[::3]:
        ax.set_ylabel("Energy normalized by P1 [%]")
    fig.suptitle(
        f"Energy decomposition (D/Dmin={representative_ratio:g}, "
        f"eps={representative_epsilon:g}, skip=drop/late)"
    )
    _save(
        fig,
        output_dir / "fig_h1a_energy_decomposition.png",
        tight_layout=False,
    )


def plot_rho_dependence(
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
    ].sort_values("rho")
    pol = policy_aggregate[
        np.isclose(policy_aggregate["deadline_ratio"], representative_ratio)
        & np.isclose(policy_aggregate["epsilon"], representative_epsilon)
        & policy_aggregate["policy"].isin(["P2", "P3"])
    ].sort_values("rho")

    fig, ax = plt.subplots(figsize=(11, 5.7))
    pos = _ordinal(ax, comp["rho"])
    for skip_mode, group in comp.groupby("skip_mode"):
        y = group["online_recovery_mean"].to_numpy(copy=True)
        oracle_gap = group["oracle_gap_percent_mean"].to_numpy()
        y[oracle_gap < gap_mask_percent] = np.nan
        x = [pos[float(value)] for value in group["rho"]]
        ax.errorbar(x, y, yerr=group["online_recovery_std"], marker="o", capsize=3, label=skip_mode)
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set(
        xlabel=RHO_AXIS_LABEL,
        ylabel=r"Online recovery $(P1-P3)/(P1-P2)$",
        title=(
            "Online policy's recovery of oracle savings\n"
            f"(D/Dmin={representative_ratio:g}, eps={representative_epsilon:g}, "
            f"skip=drop/late; gap < {gap_mask_percent:g}% masked)"
        ),
    )
    ax.grid(alpha=0.25)
    ax.legend()
    _save(fig, output_dir / "fig_h1b1_online_recovery.png")

    for skip_mode, group in pol.groupby("skip_mode"):
        fig, axes = plt.subplots(1, 2, figsize=(17, 5.5))
        pos = _ordinal(axes[0], group["rho"])
        _ordinal(axes[1], group["rho"])
        for policy, part in group.groupby("policy"):
            color = POLICY_COLORS[policy]
            x = [pos[float(value)] for value in part["rho"]]
            axes[0].errorbar(x, part["max_violation_run_mean"], yerr=part["max_violation_run_std"], marker="o", capsize=3, color=color, label=policy)
            axes[1].errorbar(x, part["burst_count_ge2_mean"], yerr=part["burst_count_ge2_std"], marker="o", capsize=3, color=color, label=policy)
        for ax in axes:
            ax.set_xlabel(RHO_AXIS_LABEL)
            ax.grid(alpha=0.25)
            ax.legend()
        axes[0].set_ylabel("Maximum consecutive violations")
        axes[1].set_ylabel("Number of violation bursts (length >= 2)")
        fig.suptitle(
            "Violation clustering under energy-optimal budget spending\n"
            f"(D/Dmin={representative_ratio:g}, eps={representative_epsilon:g}, "
            f"skip={skip_mode})"
        )
        _save(fig, output_dir / f"fig_h1b2_burstiness_{skip_mode}.png")

    # epsilon=0.15 exceeds pi_B/2 and exposes the P2-P2prime trade-off.
    comp_gap = comparison_aggregate[
        np.isclose(comparison_aggregate["deadline_ratio"], representative_ratio)
        & np.isclose(comparison_aggregate["epsilon"], 0.15)
    ].sort_values("rho")
    fig, ax = plt.subplots(figsize=(11, 5.7))
    pos = _ordinal(ax, comp_gap["rho"])
    for skip_mode, group in comp_gap.groupby("skip_mode"):
        x = [pos[float(value)] for value in group["rho"]]
        ax.errorbar(x, group["p2prime_gap_percent_mean"], yerr=group["p2prime_gap_percent_std"], marker="o", capsize=3, label=skip_mode)
    ax.set(
        xlabel=RHO_AXIS_LABEL,
        ylabel=r"$(P2'-P2)/P1$ [%]",
        title=(
            "Energy cost of banning consecutive violations (ε=0.15)\n"
            f"(D/Dmin={representative_ratio:g}, eps=0.15, skip=drop/late)"
        ),
    )
    ax.grid(alpha=0.25)
    ax.legend()
    _save(fig, output_dir / "fig_h1b3_burst_ban_cost.png")


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
    ].sort_values("rho")
    channel_mean = channel_stats.groupby("rho", as_index=False)["pi_bad_observed"].mean()
    fig, ax = plt.subplots(figsize=(11, 5.7))
    pos = _ordinal(ax, pd.concat([data["rho"], channel_mean["rho"]]))
    for skip_mode, group in data.groupby("skip_mode"):
        x = [pos[float(value)] for value in group["rho"]]
        ax.errorbar(x, group["oracle_gap_percent_mean"], yerr=group["oracle_gap_percent_std"], marker="o", capsize=3, label=f"oracle gap ({skip_mode})")
    ax.set_xlabel(RHO_AXIS_LABEL)
    ax.set_ylabel(r"Offline oracle gap $(P1-P2)/P1$ [%]")
    ax.grid(alpha=0.25)
    ax2 = ax.twinx()
    channel_x = [pos[float(value)] for value in channel_mean["rho"]]
    ax2.plot(channel_x, channel_mean["pi_bad_observed"], color="#d62728", marker="s", linestyle="--", label="observed pi_B")
    ax2.axhline(0.2, color="#d62728", linewidth=0.8, alpha=0.5)
    ax2.set_ylim(0.18, 0.22)
    ax2.set_ylabel("Observed Bad-state fraction", color="#d62728")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")
    ax.set_title("Check: offline oracle gap is invariant to channel correlation")
    _save(fig, output_dir / "fig_sanity_oracle_flatness.png")


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
    plot_rho_dependence(comparison_aggregate, policy_aggregate, output_dir, gap_mask_percent)
    plot_oracle_invariance(comparison_aggregate, channel_stats, output_dir)
