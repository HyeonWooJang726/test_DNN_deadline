"""Figures for H2 server-sharing decomposition, interaction, and fairness."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plotting import rho_label


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_h2_decomposition(
    decomposition: pd.DataFrame, output_dir: Path
) -> None:
    if decomposition.empty:
        return
    preferred = decomposition[
        (decomposition["channel_sync"] == "independent")
        & (decomposition["skip_mode"] == "late")
        & np.isclose(decomposition["deadline_ratio"], 1.35)
        & np.isclose(decomposition["epsilon"], 0.15)
    ]
    row = (preferred if len(preferred) else decomposition).sort_values(
        ["N", "rho"]
    ).iloc[-1]
    i1 = float(row["I1_energy_j_mean"])
    names = ["I1", "J1", "I2", "J2"]
    values = [100.0 * float(row[f"{name}_energy_j_mean"]) / i1 for name in names]
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    bars = ax.bar(names, values, color=["#636363", "#9ecae1", "#6baed6", "#2171b5"])
    ax.set_ylabel("System energy (% of I1)")
    ax.set_ylim(0, 108)
    ax.set_title(
        "Shared-server energy decomposition\n"
        f"N={int(row['N'])}, {rho_label(row['rho'])}, D/D_min={row['deadline_ratio']:g}, "
        f"epsilon={row['epsilon']:g}, {row['skip_mode']}, {row['channel_sync']}"
    )
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1, f"{value:.1f}%", ha="center")
    _save(fig, output_dir / "fig_h2_decomposition.png")


def plot_interaction_heatmaps(
    decomposition: pd.DataFrame, output_dir: Path
) -> None:
    if decomposition.empty:
        return
    data = decomposition[
        np.isclose(decomposition["deadline_ratio"], 1.35)
        & np.isclose(decomposition["epsilon"], 0.15)
    ]
    sync_values = [value for value in ("independent", "common") if value in set(data["channel_sync"])]
    skip_values = [value for value in ("drop", "late") if value in set(data["skip_mode"])]
    fig, axes = plt.subplots(
        len(skip_values), len(sync_values), figsize=(5.2 * len(sync_values), 4.0 * len(skip_values)), squeeze=False
    )
    n_values = sorted(data["N"].unique())
    rho_values = sorted(data["rho"].unique())
    for iy, skip_mode in enumerate(skip_values):
        for ix, sync in enumerate(sync_values):
            ax = axes[iy, ix]
            panel = data[(data["skip_mode"] == skip_mode) & (data["channel_sync"] == sync)]
            grid = np.full((len(rho_values), len(n_values)), np.nan)
            for row_i, rho in enumerate(rho_values):
                for col_i, n_devices in enumerate(n_values):
                    row = panel[np.isclose(panel["rho"], rho) & (panel["N"] == n_devices)]
                    if len(row):
                        grid[row_i, col_i] = float(row.iloc[0]["interaction_percent_points_mean"])
            image = ax.imshow(np.ma.masked_invalid(grid), origin="lower", aspect="auto", cmap="coolwarm")
            ax.set_xticks(range(len(n_values)), [str(int(n)) for n in n_values])
            ax.set_yticks(range(len(rho_values)), [rho_label(rho) for rho in rho_values])
            ax.set_xlabel("Number of devices N")
            ax.set_ylabel("Channel correlation")
            ax.set_title(f"{sync}, {skip_mode}; D/D_min=1.35, epsilon=0.15")
            for row_i in range(len(rho_values)):
                for col_i in range(len(n_values)):
                    label = "n/a" if np.isnan(grid[row_i, col_i]) else f"{grid[row_i, col_i]:.2f} pp"
                    ax.text(col_i, row_i, label, ha="center", va="center", fontsize=9)
            fig.colorbar(image, ax=ax, label="Interaction (percentage points)")
    _save(fig, output_dir / "fig_h2_interaction_heatmaps.png")


def plot_sync_pairs(decomposition: pd.DataFrame, output_dir: Path) -> None:
    if decomposition.empty:
        return
    data = decomposition[
        (decomposition["skip_mode"] == "late")
        & np.isclose(decomposition["deadline_ratio"], 1.35)
        & np.isclose(decomposition["epsilon"], 0.15)
    ]
    pivot = data.pivot_table(
        index=["N", "rho"],
        columns="channel_sync",
        values="interaction_percent_points_mean",
    ).dropna()
    if pivot.empty or not {"independent", "common"}.issubset(pivot.columns):
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    x = np.arange(len(pivot))
    for index, ((n_devices, rho), row) in enumerate(pivot.iterrows()):
        ax.plot([index, index], [row["common"], row["independent"]], color="#969696")
        ax.scatter(index, row["common"], color="#9ecae1", marker="s", label="common" if index == 0 else None)
        ax.scatter(index, row["independent"], color="#2171b5", marker="o", label="independent" if index == 0 else None)
    ax.axhline(0, color="#777777", linewidth=0.8)
    ax.set_xticks(x, [f"N={int(n)}\n{rho_label(rho)}" for n, rho in pivot.index], rotation=20, ha="right")
    ax.set_ylabel("Interaction (percentage points)")
    ax.set_title("Asynchrony control: independent versus common\nlate, D/D_min=1.35, epsilon=0.15")
    ax.legend()
    _save(fig, output_dir / "fig_h2_sync_pairs.png")


def plot_fairness_table(fairness: pd.DataFrame, output_dir: Path) -> None:
    if fairness.empty:
        return
    data = fairness[
        (fairness["skip_mode"] == "late")
        & np.isclose(fairness["deadline_ratio"], 1.35)
        & np.isclose(fairness["epsilon"], 0.15)
    ].sort_values(["channel_sync", "N", "rho"])
    columns = ["Sync", "N", "rho", "I2 spread", "J2 spread", "I2 Jain", "J2 Jain"]
    cells = []
    for row in data.itertuples(index=False):
        cells.append(
            [
                row.channel_sync,
                str(int(row.N)),
                f"{row.rho:g}",
                f"{row.I2_violation_spread_mean:.4f}",
                f"{row.J2_violation_spread_mean:.4f}",
                f"{row.I2_jain_index_mean:.4f}",
                f"{row.J2_jain_index_mean:.4f}",
            ]
        )
    fig_height = max(2.5, 0.38 * len(cells) + 1.4)
    fig, ax = plt.subplots(figsize=(10.2, fig_height))
    ax.axis("off")
    table = ax.table(cellText=cells, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    ax.set_title("Per-device violation fairness\nlate, D/D_min=1.35, epsilon=0.15", pad=16)
    _save(fig, output_dir / "fig_h2_fairness_table.png")


def create_h2_plots(
    decomposition: pd.DataFrame, fairness: pd.DataFrame, output_dir: Path
) -> None:
    plot_h2_decomposition(decomposition, output_dir)
    plot_interaction_heatmaps(decomposition, output_dir)
    plot_sync_pairs(decomposition, output_dir)
    plot_fairness_table(fairness, output_dir)
