#!/usr/bin/env python3
"""Regenerate the CPU-reproducible Paper-A metrics and diff them against disk.

Reads ``paper_a/repro/metric_manifest.json`` (the single list of every artifact
Paper A cites, with its producing command) and, for the CPU-only entries that are
marked reproducible, runs the producer into a scratch directory and compares the
result with the on-disk artifact using a tolerance-based structural diff
(``scripts/_metric_diff.py``).

The on-disk artifact is NEVER overwritten (AGENTS.md §5.5): output goes to
``--out-dir`` (default ``/tmp/paper_a_repro``), and only the comparison is
reported. Exit code is 0 when every executed entry matches, 1 otherwise. GPU
entries are skipped unless ``--include-gpu-long`` is passed.

Usage
-----
::

    .venv/bin/python scripts/reproduce_paper_a.py            # CPU subset
    .venv/bin/python scripts/reproduce_paper_a.py --dry-run  # print the plan
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts._metric_diff import structural_diff  # noqa: E402

DEFAULT_MANIFEST = PROJECT_ROOT / "paper_a" / "repro" / "metric_manifest.json"
DEFAULT_OUT_DIR = Path("/tmp/paper_a_repro")

CPU_CLASSES = {"cpu_seconds", "cpu_minutes"}


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text())


def _skip_reason(entry: Dict[str, object], include_gpu_long: bool) -> Optional[str]:
    if entry.get("cost_class") == "blocked":
        return "blocked (producer not pinned)"
    if not entry.get("reproducible_from_committed_code"):
        return "producer not reproducible from committed code"
    if not entry.get("exact_command"):
        return "no exact_command in manifest"
    cost = str(entry.get("cost_class", ""))
    if cost.startswith("gpu") and not include_gpu_long:
        return f"{cost} (pass --include-gpu-long to run)"
    if cost not in CPU_CLASSES and not cost.startswith("gpu"):
        return f"unknown cost_class {cost!r}"
    return None


def _build_command(entry: Dict[str, object], out_file: Path, folds_dir: Path) -> List[str]:
    cmd = []
    for tok in entry["exact_command"]:  # type: ignore[index]
        cmd.append(str(tok).replace("{OUT}", str(out_file)).replace("{FOLDS}", str(folds_dir)))
    return cmd


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR),
                   help="scratch dir for regenerated metrics (never the on-disk path)")
    p.add_argument("--folds-dir", default=str(PROJECT_ROOT / "paper_a" / "repro" / "folds"),
                   help="DOI-split fold files for the lightgbm entry")
    p.add_argument("--only", nargs="+", default=None, help="run only these manifest entry names")
    p.add_argument("--include-gpu-long", action="store_true",
                   help="also run GPU entries (hours; skipped by default)")
    p.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    args = p.parse_args(argv)

    manifest = _load_json(Path(args.manifest))
    entries: List[Dict[str, object]] = manifest["entries"]  # type: ignore[assignment]
    if args.only:
        wanted = set(args.only)
        entries = [e for e in entries if e["name"] in wanted]

    env = dict(os.environ)
    for k, v in (manifest.get("env") or {}).items():  # type: ignore[union-attr]
        env.setdefault(k, str(v))

    out_dir = Path(args.out_dir)
    folds_dir = Path(args.folds_dir)
    results: List[tuple] = []

    for entry in entries:
        name = str(entry["name"])
        reason = _skip_reason(entry, args.include_gpu_long)
        if reason is None and "{FOLDS}" in " ".join(map(str, entry["exact_command"])) and not folds_dir.exists():
            reason = f"fold files not found at {folds_dir}"
        if reason is not None:
            print(f"SKIP  {name}: {reason}")
            results.append((name, "SKIP", reason))
            continue

        out_file = out_dir / f"{name}.json"
        cmd = _build_command(entry, out_file, folds_dir)
        if args.dry_run:
            print(f"PLAN  {name}: {' '.join(cmd)}")
            results.append((name, "PLAN", None))
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env,
                              capture_output=True, text=True)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
            print(f"FAIL  {name}: producer exited {proc.returncode}")
            for line in tail:
                print(f"        {line}")
            results.append((name, "FAIL", f"producer exit {proc.returncode}"))
            continue

        on_disk = PROJECT_ROOT / str(entry["metric_file"])
        if not on_disk.exists():
            print(f"FAIL  {name}: on-disk artifact missing ({on_disk})")
            results.append((name, "FAIL", "on-disk missing"))
            continue

        ignore = entry.get("diff_ignore", []) or []
        abs_tol = float(entry.get("abs_tol", 1e-6))
        diffs = structural_diff(_load_json(on_disk), _load_json(out_file),
                                abs_tol=abs_tol, ignore=ignore)
        if diffs:
            print(f"FAIL  {name}: {len(diffs)} difference(s) vs {entry['metric_file']}")
            for d in diffs[:8]:
                print(f"        {d}")
            if len(diffs) > 8:
                print(f"        ... and {len(diffs) - 8} more")
            results.append((name, "FAIL", f"{len(diffs)} diffs"))
        else:
            print(f"PASS  {name}: matches {entry['metric_file']} (abs_tol {abs_tol:g})")
            results.append((name, "PASS", None))

    n_pass = sum(1 for _, s, _ in results if s == "PASS")
    n_fail = sum(1 for _, s, _ in results if s == "FAIL")
    n_skip = sum(1 for _, s, _ in results if s == "SKIP")
    n_plan = sum(1 for _, s, _ in results if s == "PLAN")
    print(f"\n[paper-a-repro] {n_pass} PASS, {n_fail} FAIL, {n_skip} SKIP"
          + (f", {n_plan} PLAN" if n_plan else ""))
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
