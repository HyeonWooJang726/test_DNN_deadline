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


def test_decomposition_has_four_policy_ticks_in_every_panel(monkeypatch, tmp_path: Path):
    policies = ["P1", "P0", "P2", "P2prime"]
    rows = []
    for skip_mode in ("drop", "late"):
        for rho in (0.75, 0.975):
            for epsilon in (0.01, 0.05, 0.1, 0.15):
                for seed in (1701, 1702):
                    for index, policy in enumerate(policies):
                        rows.append(
                            {
                                "device": "device_0",
                                "deadline_ratio": 1.5,
                                "epsilon": epsilon,
                                "rho": rho,
                                "seed": seed,
                                "skip_mode": skip_mode,
                                "policy": policy,
                                "mean_energy_j": 1.0 - 0.04 * index,
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
    assert len(fig.axes) == 8
    assert all(len(ax.get_xticklabels()) == 4 for ax in fig.axes)
    assert all(
        [tick.get_text() for tick in ax.get_xticklabels()] == policies
        for ax in fig.axes
    )
    assert np.allclose(fig.axes[0].get_xlim(), (-0.6, 3.6))
    plt.close(fig)
