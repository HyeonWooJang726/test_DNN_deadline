"""One-time sorted-content check for interrupted/resumed versus clean smoke runs."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import pandas as pd


FILES = (
    "policy_runs.csv",
    "comparison_aggregate.csv",
    "sanity_checks.csv",
    "reproducibility_hashes.csv",
)


def _normalized(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, float_precision="round_trip")
    return frame.sort_values(
        list(frame.columns), kind="mergesort", na_position="first"
    ).reset_index(drop=True)


def _digest(frame: pd.DataFrame) -> str:
    payload = frame.to_csv(index=False, float_format="%.17g").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def compare(left: Path, right: Path, report: Path, interrupted_markers: int) -> bool:
    lines = [
        "Resume equivalence test",
        "=======================",
        "",
        f"Interrupted run: {left.as_posix()}",
        f"Uninterrupted run: {right.as_posix()}",
        f"Completed checkpoints before forced termination: {interrupted_markers}/48",
        "Resume command: python run_sweep.py --mode smoke --output results/resume_a --resume",
        "",
        "Sorted exact-content comparisons:",
    ]
    failures = []
    for filename in FILES:
        left_frame = _normalized(left / filename)
        right_frame = _normalized(right / filename)
        try:
            pd.testing.assert_frame_equal(
                left_frame,
                right_frame,
                check_exact=True,
                check_dtype=True,
                check_like=False,
            )
            equal = True
            detail = ""
        except AssertionError as exc:
            equal = False
            detail = str(exc).splitlines()[0]
            failures.append(f"{filename}: {detail}")
        left_digest = _digest(left_frame)
        right_digest = _digest(right_frame)
        digest_equal = left_digest == right_digest
        if not digest_equal and equal:
            failures.append(f"{filename}: canonical digests differ")
        lines.append(
            f"  {filename}: {'PASS' if equal and digest_equal else 'FAIL'}; "
            f"rows={len(left_frame)}; sha256={left_digest}"
        )

    left_hashes = _normalized(left / "reproducibility_hashes.csv")
    right_hashes = _normalized(right / "reproducibility_hashes.csv")
    hash_columns_equal = left_hashes.equals(right_hashes)
    lines.extend(
        [
            "",
            "Reproducibility hashes:",
            f"  {'PASS' if hash_columns_equal else 'FAIL'}; "
            f"rows={len(left_hashes)}; all keys and sha256 values identical",
            "",
            f"OVERALL: {'PASS' if not failures and hash_columns_equal else 'FAIL'}",
        ]
    )
    if failures:
        lines.append("Mismatches:")
        lines.extend(f"  - {failure}" for failure in failures)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return not failures and hash_columns_equal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resumed", type=Path, default=Path("results/resume_a"))
    parser.add_argument("--clean", type=Path, default=Path("results/resume_b"))
    parser.add_argument(
        "--report", type=Path, default=Path("resume_equivalence_report.txt")
    )
    parser.add_argument("--interrupted-markers", type=int, default=26)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if not compare(
        arguments.resumed,
        arguments.clean,
        arguments.report,
        arguments.interrupted_markers,
    ):
        raise SystemExit(1)
