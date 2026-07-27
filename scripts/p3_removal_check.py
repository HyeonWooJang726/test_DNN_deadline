"""P3 removal gate: the four retained policies must be unchanged."""
import io
import subprocess

import numpy as np
import pandas as pd


TAG = "before-p3-removal"
KEYS = ["device", "rho", "seed", "deadline_ratio", "epsilon", "skip_mode", "policy"]
COLS = [
    "mean_energy_j",
    "violation_rate",
    "violation_count",
    "max_violation_run",
    "burst_count_ge2",
    "violation_run_count",
    "violation_state_bad_count",
    "violation_state_good_count",
    "boost_use_rate",
    "policy_metadata_json",
]


def tagged_csv(path):
    out = subprocess.run(
        ["git", "show", f"{TAG}:{path}"],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    return pd.read_csv(io.BytesIO(out), float_precision="round_trip")


before = tagged_csv("results/full/policy_runs.csv")
after = pd.read_csv("results/full/policy_runs.csv", float_precision="round_trip")

assert not (after["policy"] == "P3").any(), "P3 행이 남아 있음"

for policy in ("P1", "P0", "P2", "P2prime"):
    old = before[before.policy == policy].sort_values(KEYS).reset_index(drop=True)
    new = after[after.policy == policy].sort_values(KEYS).reset_index(drop=True)
    assert len(old) == len(new) and len(old) > 0, f"{policy}: 행 개수 불일치"
    assert old[KEYS].equals(new[KEYS]), f"{policy}: 조합 key 불일치"
    for column in COLS:
        if column not in old.columns:
            continue
        if old[column].dtype.kind in "OSU":
            assert old[column].equals(new[column]), f"{policy}.{column} 변경됨"
        else:
            np.testing.assert_allclose(
                old[column],
                new[column],
                rtol=0,
                atol=0,
                err_msg=f"{policy}.{column} 변경됨",
            )

lost = set(before.columns) - set(after.columns)
added = set(after.columns) - set(before.columns)
assert lost <= {"selected_v", "q_final", "q_max"}, f"예상 밖 열 삭제: {lost}"
assert not added, f"예상 밖 열 추가: {added}"

print("PASS: P1/P0/P2/P2prime 결과 불변, P3 전용 열만 제거됨")
