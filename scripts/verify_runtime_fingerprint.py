#!/usr/bin/env python3
"""Recompute/verify validation's runtime-source fingerprint, matching build.rs.

Uses the explicit listed inputs, every non-test Rust file under translator bind/,
and every non-test runtime src/ file. Hash order, names and bytes follow hash_files.
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CRATES = ROOT / "crates"
BUILD_RS = CRATES / "metal2vulkan-validation" / "build.rs"
PIN = CRATES / "metal2vulkan-validation" / "runtime-sources.sha256"


def listed_inputs(build_text: str) -> list[str]:
    start_marker, end_marker = "    let listed = [", "    ];"
    start = build_text.find(start_marker)
    if start < 0:
        raise ValueError("Cannot find runtime input list in build.rs")
    start += len(start_marker)
    end = build_text.find(end_marker, start)
    if end < 0:
        raise ValueError("Cannot find the end of build.rs runtime input list")
    body = build_text[start:end]
    paths = re.findall(r'^\s*"([^"\n]+)",\s*$', body, flags=re.MULTILINE)
    other = re.sub(r'^\s*"[^"\n]+",\s*$', "", body, flags=re.MULTILINE)
    if other.strip() or not paths:
        raise ValueError("Unsupported input-list syntax; update the checksum helper")
    return paths


def walk(directory: str, rust_only: bool) -> list[str]:
    base = CRATES / directory
    if not base.is_dir():
        raise FileNotFoundError(base)
    paths = []
    for candidate in base.rglob("*"):
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(CRATES)
        if any(part.startswith(".") or part == "tests" for part in relative.parts):
            continue
        if candidate.name == "tests.rs" or candidate.name.endswith("_tests.rs"):
            continue
        if rust_only and candidate.suffix != ".rs":
            continue
        paths.append(relative.as_posix())
    return paths


def fingerprint() -> tuple[str, int]:
    paths = listed_inputs(BUILD_RS.read_text(encoding="utf-8"))
    paths.extend(walk("metal2vulkan/src/bind", rust_only=True))
    paths.extend(walk("metal2vulkan-runtime/src", rust_only=False))
    paths.sort()
    digest = hashlib.sha256()
    for relative in paths:
        content = (CRATES / relative).read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(content)
    return digest.hexdigest(), len(paths)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--check", action="store_true")
    modes.add_argument("--print", dest="print_only", action="store_true")
    modes.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    try:
        computed, total = fingerprint()
        expected = PIN.read_text(encoding="ascii").strip()
        if not re.fullmatch(r"[a-f0-9]{64}", expected):
            raise ValueError("Pinned runtime fingerprint is not a 64-digit SHA-256")
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"runtime sources: {total}; computed sha256: {computed}")
    if args.print_only:
        return 0
    if args.write:
        PIN.write_text(computed + "\n", encoding="ascii")
        print(f"Updated {PIN.relative_to(ROOT)}")
        return 0
    if computed != expected:
        print(
            f"Pinned runtime sha256: {expected}\n"
            f"Computed runtime sha256: {computed}\n"
            "Run: python3 scripts/verify_runtime_fingerprint.py --write",
            file=sys.stderr,
        )
        return 1
    print("Pinned runtime fingerprint matches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
