#!/usr/bin/env python3
"""Compare M2V_SCHED_STATS logs from two otherwise identical Reims runs.

These lines report one-second aggregates, NOT per-frame latencies or individual
submission percentiles. This script cannot demonstrate a frame-time improvement.
It does not launch a VM, modify files or apply system settings.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SCHED_LINE = re.compile(
    r"m2v-sched: (?P<submits>\d+) submits, (?P<batches>\d+) batches "
    r"\([\d.]+ a submit\), (?P<mean_us>[\d.]+) us a submit; ends: "
    r"queue (?P<queue>\d+) limit (?P<limit>\d+) "
    r"conflict (?P<conflict>\d+) fence (?P<fence>\d+) pass (?P<pass>\d+)"
)

COUNTS = ("queue", "limit", "conflict", "fence", "pass")


def parse_line(line: str) -> dict[str, float | int] | None:
    """Read one scheduler summary, ignoring unrelated VM or kernel messages."""
    match = SCHED_LINE.search(line)
    if not match:
        return None
    fields: dict[str, float | int] = {
        key: int(value) for key, value in match.groupdict().items() if key != "mean_us"
    }
    fields["mean_us"] = float(match["mean_us"])
    return fields


def summarize(lines: list[str]) -> dict[str, float | int | dict[str, int]]:
    """Summarize reported non-empty one-second windows; missing windows are unknown."""
    rows = [parsed for line in lines if (parsed := parse_line(line)) is not None]
    if not rows:
        raise ValueError("No 'm2v-sched:' statistics found; set M2V_SCHED_STATS=1")
    submits = sum(int(r["submits"]) for r in rows)
    batches = sum(int(r["batches"]) for r in rows)
    if submits == 0:
        raise ValueError("No Vulkan submissions recorded")
    total_submit_us_estimate = sum(float(r["mean_us"]) * int(r["submits"]) for r in rows)
    return {
        "reported_windows": len(rows),
        "total_submits": submits,
        "total_batches": batches,
        "batches_per_submit": round(batches / submits, 3),
        "submits_per_reported_window": round(submits / len(rows), 3),
        "estimated_mean_submit_us": round(total_submit_us_estimate / submits, 3),
        "estimated_total_submit_time_ms": round(total_submit_us_estimate / 1000, 3),
        "end_reasons": {key: sum(int(r[key]) for r in rows) for key in COUNTS},
    }


def delta_percent(baseline: float, test: float) -> float | None:
    if baseline == 0:
        return None
    return round((test / baseline - 1) * 100, 2)


def comparison(
    baseline: dict[str, float | int | dict[str, int]],
    test: dict[str, float | int | dict[str, int]],
) -> dict[str, float | None]:
    """Percentage changes in aggregate scheduler metrics, not a speedup claim."""
    keys = (
        "batches_per_submit",
        "submits_per_reported_window",
        "estimated_mean_submit_us",
        "estimated_total_submit_time_ms",
    )
    return {
        key: delta_percent(float(baseline[key]), float(test[key])) for key in keys
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path, help="M2V_SCHED_STATS=1 log (default limit 64)")
    parser.add_argument("experimental", type=Path, help="same workload with opt-in AMD limit")
    parser.add_argument("--json", action="store_true", help="emit machine-readable comparison")
    args = parser.parse_args(argv)
    try:
        base = summarize(args.baseline.read_text(encoding="utf-8", errors="replace").splitlines())
        exp = summarize(args.experimental.read_text(encoding="utf-8", errors="replace").splitlines())
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    output = {
        "baseline": base,
        "experimental": exp,
        "percentage_change_experimental_vs_baseline": comparison(base, exp),
        "limitations": (
            "Logs contain nonempty 1-second windows only; aggregate submission time is "
            "estimated from rounded window averages. This is NOT a measure of p95/p99 "
            "frame times or smoothness. Match host, Mesa, guest, workload and run duration."
        ),
    }
    if args.json:
        print(json.dumps(output, indent=2, sort_keys=True))
    else:
        for name in ("baseline", "experimental"):
            result = output[name]
            print(
                f"{name}: {result['total_submits']} submits, "
                f"{result['total_batches']} batches, "
                f"{result['batches_per_submit']} batches/submit, "
                f"~{result['estimated_mean_submit_us']} us/submit "
                f"across {result['reported_windows']} nonempty windows"
            )
        print("Percent changes:", json.dumps(output["percentage_change_experimental_vs_baseline"]))
        print(output["limitations"], file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
