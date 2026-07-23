from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plotting


def test_rho_label_uses_axis_and_panel_styles():
    assert plotting.rho_label(0.0) == "i.i.d."
    assert plotting.rho_label(0.75) == "ρ=0.75 (run 5)"
    assert plotting.rho_label(0.975) == "ρ=0.975 (run 50)"
    assert (
        plotting.rho_label(0.0, style="panel")
        == "i.i.d. (mean bad-state run: 1.25 slots)"
    )
    assert (
        plotting.rho_label(0.75, style="panel")
        == "ρ=0.75 (mean bad-state run: 5 slots)"
    )
    assert (
        plotting.rho_label(0.975, style="panel")
        == "ρ=0.975 (mean bad-state run: 50 slots)"
    )


def test_decomposition_has_five_policy_ticks_in_every_panel(monkeypatch, tmp_path: Path):
    policies = ["P1", "P0", "P2", "P2prime", "P3"]
    rows = []
    for skip_mode in ("drop", "late"):
        for rho in (0.0, 0.75, 0.975):
            for index, policy in enumerate(policies):
                # Exercise the explicit n/a path in one panel.
                if skip_mode == "late" and rho == 0.975 and policy == "P3":
                    continue
                rows.append(
                    {
                        "deadline_ratio": 1.5,
                        "epsilon": 0.05,
                        "rho": rho,
                        "skip_mode": skip_mode,
                        "policy": policy,
                        "mean_energy_j_mean": 1.0 - 0.04 * index,
                        "mean_energy_j_std": 0.01,
                    }
                )
    frame = pd.DataFrame(rows)
    captured = {}

    def capture(fig, path, **kwargs):
        captured["fig"] = fig
        captured["path"] = path

    monkeypatch.setattr(plotting, "_save", capture)
    plotting.plot_decomposition(frame, tmp_path)

    fig = captured["fig"]
    assert len(fig.axes) == 6
    assert all(len(ax.get_xticklabels()) == 5 for ax in fig.axes)
    assert all(
        [tick.get_text() for tick in ax.get_xticklabels()] == policies
        for ax in fig.axes
    )
    assert any(text.get_text() == "n/a" for ax in fig.axes for text in ax.texts)
    assert np.allclose(fig.axes[0].get_xlim(), (-0.6, 4.6))
    plt.close(fig)
