import sys
from dataclasses import replace
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
H2_DIR = ROOT_DIR / "h2"
for path in (ROOT_DIR, H2_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import pandas as pd
import pytest

import run_h2_sweep as h2_sweep_module
from h2_config import default_h2_experiment


def _small_h2_experiment():
    config = default_h2_experiment("smoke")
    sweep = replace(
        config.sweep,
        t_slots=200,
        seeds=(1701,),
        n_devices=(2,),
        rho_values=(0.0,),
        deadline_ratios=(1.35,),
        epsilons=(0.15,),
        skip_modes=("drop", "late"),
        channel_sync_values=("independent",),
        lambda_iterations=2,
    )
    return replace(config, sweep=sweep)


def test_h2_runtime_projection_does_not_rescale_full_horizon_twice():
    runtimes = pd.DataFrame(
        {"N": [2, 4], "elapsed_seconds": [2.0, 6.0]}
    )
    quick = h2_sweep_module._runtime_projection(runtimes, 10_000)
    full = h2_sweep_module._runtime_projection(runtimes, 50_000)
    assert quick["projected_full_hours"] == 5.0 * full["projected_full_hours"]


def test_h2_runtime_failure_isolated_and_non_strict_checkpoint_survives(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        h2_sweep_module,
        "default_h2_experiment",
        lambda mode: _small_h2_experiment(),
    )
    real_simulate = h2_sweep_module.simulate_h2_trace

    def fail_drop(*args):
        if args[5] == "drop":
            raise ValueError("injected H2 runtime failure")
        return real_simulate(*args)

    monkeypatch.setattr(h2_sweep_module, "simulate_h2_trace", fail_drop)
    monkeypatch.setattr(
        h2_sweep_module,
        "h2_sanity_rows",
        lambda *args: [
            {
                "check": "injected_required_failure",
                "passed": False,
                "detail": "checkpoint must survive in non-strict mode",
                "severity": "required",
            }
        ],
    )

    output = tmp_path / "isolated"
    h2_sweep_module.run_h2_sweep(
        "smoke", output, strict_sanity=False, make_plots=False
    )

    failures = pd.read_csv(output / "runtime_failures.csv")
    assert len(failures) == 1
    assert failures.loc[0, "status"] == "runtime-invalid"
    assert failures.loc[0, "exception_type"] == "ValueError"
    preflight = pd.read_csv(output / "preflight.csv")
    drop = preflight[preflight["skip_mode"] == "drop"]
    assert not drop["valid"].any()
    aggregate = pd.read_csv(output / "h2_decomposition.csv")
    assert set(aggregate["skip_mode"]) == {"late"}
    sanity = pd.read_csv(output / "sanity_checks.csv")
    injected = sanity[sanity["check"] == "injected_required_failure"]
    assert len(injected) == 1
    assert not injected["passed"].any()
    assert len(list((output / "checkpoint").rglob("_complete.json"))) == 2


def test_h2_interrupted_resume_matches_uninterrupted_deterministic_csvs(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        h2_sweep_module,
        "default_h2_experiment",
        lambda mode: _small_h2_experiment(),
    )
    real_simulate = h2_sweep_module.simulate_h2_trace

    baseline = tmp_path / "baseline"
    h2_sweep_module.run_h2_sweep(
        "smoke", baseline, strict_sanity=False, make_plots=False
    )

    interrupted = tmp_path / "interrupted"
    raised = False

    def interrupt_late(*args):
        nonlocal raised
        if not raised and args[5] == "late":
            raised = True
            raise KeyboardInterrupt("injected H2 interruption")
        return real_simulate(*args)

    monkeypatch.setattr(h2_sweep_module, "simulate_h2_trace", interrupt_late)
    with pytest.raises(KeyboardInterrupt, match="injected H2 interruption"):
        h2_sweep_module.run_h2_sweep(
            "smoke", interrupted, strict_sanity=False, make_plots=False
        )
    assert len(list((interrupted / "checkpoint").rglob("_complete.json"))) == 1

    monkeypatch.setattr(h2_sweep_module, "simulate_h2_trace", real_simulate)
    h2_sweep_module.run_h2_sweep(
        "smoke",
        interrupted,
        strict_sanity=False,
        make_plots=False,
        resume=True,
    )
    deterministic = (
        "policy_runs.csv",
        "policy_aggregate.csv",
        "h2_runs.csv",
        "h2_decomposition.csv",
        "sanity_checks.csv",
        "reproducibility_hashes.csv",
    )
    for name in deterministic:
        assert (interrupted / name).read_bytes() == (baseline / name).read_bytes(), name
