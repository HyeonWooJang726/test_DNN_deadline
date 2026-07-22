from dataclasses import replace

import pandas as pd
import pytest

import run_sweep as sweep_module
from config import default_experiment


def _small_experiment():
    config = default_experiment("quick")
    sweep = replace(
        config.sweep,
        t_slots=1_000,
        seeds=(1701, 1702),
        rho_values=(0.0,),
        deadline_ratios=(2.0,),
        epsilons=(0.05, 0.1),
        skip_modes=("drop",),
    )
    return replace(config, sweep=sweep)


def test_runtime_failure_isolated_and_non_strict_sanity_checkpointed(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(sweep_module, "default_experiment", lambda mode: _small_experiment())
    real_simulate = sweep_module.simulate_trace

    def fail_one_combination(*args):
        epsilon = args[5]
        seed = args[8]
        if epsilon == 0.05 and seed == 1701:
            raise ValueError("forced count exceeds budget")
        return real_simulate(*args)

    monkeypatch.setattr(sweep_module, "simulate_trace", fail_one_combination)
    monkeypatch.setattr(
        sweep_module,
        "combination_sanity_rows",
        lambda *args: [
            {
                "check": "injected_required_failure",
                "passed": False,
                "detail": "checkpoint must survive with --no-strict-sanity",
            }
        ],
    )

    output = tmp_path / "isolated"
    sweep_module.run_sweep(
        "quick", output, strict_sanity=False, make_plots=False
    )

    failures = pd.read_csv(output / "runtime_failures.csv")
    assert len(failures) == 1
    assert failures.loc[0, "status"] == "runtime-invalid"
    assert failures.loc[0, "seed"] == 1701

    preflight = pd.read_csv(output / "preflight.csv")
    runtime_invalid = preflight[preflight["epsilon"] == 0.05]
    assert not runtime_invalid["valid"].any()
    assert runtime_invalid["exclusion_reason"].str.startswith("runtime-invalid:").all()

    aggregate = pd.read_csv(output / "comparison_aggregate.csv")
    assert set(aggregate["epsilon"]) == {0.1}
    sanity = pd.read_csv(output / "sanity_checks.csv")
    assert (sanity["check"] == "injected_required_failure").any()
    assert not sanity.loc[
        sanity["check"] == "injected_required_failure", "passed"
    ].any()

    checkpoint_markers = list((output / "checkpoint").glob("*/_complete.json"))
    assert len(checkpoint_markers) == 2


def test_interrupted_resume_matches_uninterrupted_csvs(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep_module, "default_experiment", lambda mode: _small_experiment())
    real_simulate = sweep_module.simulate_trace

    baseline = tmp_path / "baseline"
    sweep_module.run_sweep(
        "quick", baseline, strict_sanity=False, make_plots=False
    )

    interrupted = tmp_path / "interrupted"
    raised = False

    def interrupt_second_group(*args):
        nonlocal raised
        if not raised and args[5] == 0.1:
            raised = True
            raise KeyboardInterrupt("test interruption")
        return real_simulate(*args)

    monkeypatch.setattr(sweep_module, "simulate_trace", interrupt_second_group)
    with pytest.raises(KeyboardInterrupt, match="test interruption"):
        sweep_module.run_sweep(
            "quick", interrupted, strict_sanity=False, make_plots=False
        )

    assert len(list((interrupted / "checkpoint").glob("*/_complete.json"))) == 1
    monkeypatch.setattr(sweep_module, "simulate_trace", real_simulate)
    resumed_outputs = sweep_module.run_sweep(
        "quick", interrupted, strict_sanity=False, make_plots=False, resume=True
    )

    for resumed_path in resumed_outputs.values():
        if resumed_path.suffix == ".csv":
            baseline_path = baseline / resumed_path.name
            assert resumed_path.read_bytes() == baseline_path.read_bytes(), resumed_path.name
