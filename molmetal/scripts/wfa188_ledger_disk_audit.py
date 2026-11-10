#!/usr/bin/env python3
# Copyright 2026 Mol-Metal contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wf-A188 -- ledger-vs-DISK vs registry coverage audit (READ-ONLY, CPU-only).

WHY THIS EXISTS
---------------
The repository carries THREE different notions of "is this metric's provenance
intact", and they disagree:

* ``registry.py check`` validates the REGISTRY ROWS only (does a registry row
  point at a file whose sha still matches?).
* ``run_ledger.staleness()`` compares the LEDGER's LATEST sha against the
  REGISTRY's sha -- never against DISK.  A row whose sha matches the registry
  is ``current`` even when the file on disk has since been re-serialised.
* ``run_ledger.uncommitted_provenance()`` compares the ledger's LATEST sha
  against DISK and against GIT HISTORY.

None of them answers the question this module answers: **which metric files
on disk, which registry rows, and which ledger rows do not line up with one
another?**  That gap is where a metric artefact can be published with no
ledger row at all (the ``no_run`` class), or with a ledger row whose sha
identifies an OLDER SERIALIZATION of the file (the ``drift`` class), and
neither is visible to the launch gate.

WHAT IT MEASURES
----------------
``audit()`` returns three lists, each with the DENOMINATOR alongside it:

1. ``drift``     -- the ledger's LATEST row per metric names a
   ``source_json_sha256`` that differs from the sha256 of that file ON DISK.
   (A sub-case, ``file_missing``, is a ledger row whose source file is gone.)
2. ``no_run``    -- a registry metric row has NO ledger row at all.  These are
   reported by ``run_ledger.staleness()`` as ``no_run`` and are explicitly
   NON-BLOCKING there: an artefact that predates the ledger legitimately has
   no row.
3. ``uncovered`` -- a metric file on disk has NO registry row.

NOT A GATE -- DELIBERATELY
---------------------------
This module is **NOT** wired into ``molmetal/scripts/pre_flight_check.py``.
Registering a new pre-flight gate changes ``EXPECTED_GATE_COUNTS``, which is
a USER decision (see the Wf-A172 / 6.267 dead-gate history in
``docs/HONEST_NEGATIVE_CATALOG.md``).  This audit is therefore a REPORTING
surface: run it, read the numbers, decide separately whether any of them
should become blocking.

The stricter git-provenance rule lives in
``molmetal/scripts/wfa166_run_ledger_provenance.py``; this module is
deliberately complementary (file-level coverage), not a replacement.

COST
----
One YAML read, one JSONL read, one sha256 per referenced metric file.  No
git calls, no network, no GPU, no torch.

MEASURED 2026-11-01 on this working tree (after the Wf-A188 repair):
``drift=0``, ``no_run=183``, ``uncovered=0``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:  # PyYAML is present in the project venv; the module degrades gracefully.
    import yaml
except ImportError:  # pragma: no cover - exercised only without PyYAML
    yaml = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = REPO_ROOT / "molmetal" / "reports" / "run_ledger.jsonl"
REGISTRY_METRICS_PATH = REPO_ROOT / "registry" / "metrics.yaml"
BY_METRIC_DIR = REPO_ROOT / "metrics" / "by_metric"


def _sha256_file(path: Path) -> Optional[str]:
    """sha256 of ``path`` on disk, or ``None`` when it is not a readable file."""
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _resolve_source_json(source_json: str, repo_root: Path = REPO_ROOT) -> Path:
    """Resolve a ledger ``source_json`` string to an absolute path.

    Mirrors ``molmetal/scripts/run_ledger.py::_resolve_source_json``:
    a repo-relative path is taken as-is; a bare stem resolves under
    ``metrics/by_metric/``.
    """
    p = Path(source_json)
    if p.is_absolute():
        return p
    if "/" in source_json or p.suffix == ".json":
        return repo_root / p
    return repo_root / "metrics" / "by_metric" / f"{source_json}.json"


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Read a YAML mapping, or ``{}`` when the file (or PyYAML) is absent."""
    if yaml is None or not path.is_file():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def _read_ledger(
    ledger_path: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Read the ledger JSONL.  Returns ``(rows, corrupt)``.

    A corrupt line is a FINDING, not a crash: an audit that dies on a bad
    byte cannot report anything else it found.
    """
    rows: List[Dict[str, Any]] = []
    corrupt: List[Dict[str, Any]] = []
    if not ledger_path.is_file():
        return rows, corrupt
    text = ledger_path.read_text(encoding="utf-8", errors="replace")
    for lineno, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            corrupt.append({"line": lineno, "error": str(exc)})
            continue
        if not isinstance(obj, dict):
            corrupt.append({"line": lineno, "error": "not a JSON object"})
            continue
        rows.append(obj)
    return rows, corrupt


def latest_by_metric(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """The LAST row per ``metric_id`` in FILE ORDER.

    The ledger is append-only (AGENTS.md 5.6), so file order IS time order.
    Grouping by sha instead would order by content, not by time, and the
    "latest row" would be wrong.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        out[str(row.get("metric_id"))] = row
    return out


def audit(
    registry_metrics_path: Optional[Path] = None,
    *,
    ledger_path: Optional[Path] = None,
    repo_root: Optional[Path] = None,
    by_metric_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Audit the ledger against DISK and against the registry.

    Parameters
    ----------
    registry_metrics_path
        Path to ``registry/metrics.yaml``.  Defaults to the repo copy.
    ledger_path
        Path to the ledger JSONL.  Defaults to the repo copy.
    repo_root
        Git working tree the ledger's relative paths resolve against.
    by_metric_dir
        Directory holding the metric JSON files.  Defaults to
        ``<repo_root>/metrics/by_metric``.

    Returns
    -------
    dict with ``counts``, ``drift``, ``no_run``, ``uncovered``,
    ``registry_source_missing_on_disk``, ``corrupt_ledger_lines`` and
    ``paper_line_no_run``.
    """
    registry_metrics_path = Path(
        registry_metrics_path or REGISTRY_METRICS_PATH
    )
    ledger_path = Path(ledger_path or LEDGER_PATH)
    repo_root = Path(repo_root or REPO_ROOT)
    by_metric_dir = Path(by_metric_dir or (repo_root / "metrics" / "by_metric"))

    registry = _load_yaml(registry_metrics_path)
    metric_rows: Dict[str, Dict[str, Any]] = registry.get("metrics") or {}
    rows, corrupt = _read_ledger(ledger_path)
    latest = latest_by_metric(rows)

    # ---- 1. ledger LATEST sha vs DISK sha ----------------------------
    drift: List[Dict[str, Any]] = []
    for mid, row in sorted(latest.items()):
        src = row.get("source_json")
        sha = row.get("source_json_sha256")
        if not src or not sha:
            continue
        disk = _sha256_file(_resolve_source_json(str(src), repo_root=repo_root))
        if disk == str(sha):
            continue
        drift.append(
            {
                "metric_id": mid,
                "source_json": str(src),
                "ledger_sha256": str(sha),
                "disk_sha256": disk,
                "kind": "sha_differs" if disk is not None else "file_missing",
                "ledger_timestamp": row.get("timestamp"),
                "ledger_git_sha": row.get("git_sha"),
            }
        )

    # ---- 2. registry rows with NO ledger row -------------------------
    no_run: List[str] = sorted(mid for mid in metric_rows if mid not in latest)
    paper_line_no_run: List[str] = sorted(
        mid
        for mid in no_run
        if (metric_rows.get(mid) or {}).get("registry_role") == "paper_line"
    )

    # ---- 3. metric files on disk with NO registry row ----------------
    registry_source_names = {
        Path(str(rec.get("source_json"))).name
        for rec in metric_rows.values()
        if rec.get("source_json")
    }
    on_disk = (
        sorted(p.name for p in by_metric_dir.glob("*.json"))
        if by_metric_dir.is_dir()
        else []
    )
    uncovered = sorted(name for name in on_disk if name not in registry_source_names)

    # ---- extra: registry rows whose source file is gone --------------
    registry_source_missing_on_disk = sorted(
        mid
        for mid, rec in metric_rows.items()
        if rec.get("source_json")
        and not _resolve_source_json(
            str(rec["source_json"]), repo_root=repo_root
        ).is_file()
    )

    counts = {
        "registry_metric_rows": len(metric_rows),
        "ledger_rows": len(rows),
        "ledger_latest_rows": len(latest),
        "ledger_distinct_metric_ids": len(latest),
        "metric_files_on_disk": len(on_disk),
        "drift": len(drift),
        "no_run": len(no_run),
        "no_run_paper_line": len(paper_line_no_run),
        "uncovered": len(uncovered),
        "registry_source_missing_on_disk": len(registry_source_missing_on_disk),
        "corrupt_ledger_lines": len(corrupt),
    }

    return {
        "counts": counts,
        "drift": drift,
        "no_run": no_run,
        "paper_line_no_run": paper_line_no_run,
        "uncovered": uncovered,
        "registry_source_missing_on_disk": registry_source_missing_on_disk,
        "corrupt_ledger_lines": corrupt,
        "paths": {
            "registry_metrics": str(registry_metrics_path),
            "ledger": str(ledger_path),
            "repo_root": str(repo_root),
            "by_metric_dir": str(by_metric_dir),
        },
    }


def render_summary(report: Dict[str, Any], *, max_items: int = 20) -> str:
    """Render ``report`` as a human-readable, plain-text summary."""
    c = report["counts"]
    lines: List[str] = []
    lines.append("Wf-A188 ledger-vs-disk vs registry coverage audit (READ-ONLY)")
    lines.append(
        "  registry_metric_rows={registry_metric_rows} ledger_rows={ledger_rows} "
        "ledger_latest_rows={ledger_latest_rows} metric_files_on_disk={metric_files_on_disk}".format(
            **c
        )
    )
    lines.append(
        "  DRIFT (ledger LATEST sha != disk sha) = {drift} | "
        "NO_RUN (registry row, no ledger row) = {no_run} "
        "(paper-line {no_run_paper_line}) | "
        "UNCOVERED (file on disk, no registry row) = {uncovered}".format(**c)
    )
    lines.append(
        "  registry rows whose source file is missing on disk = "
        "{registry_source_missing_on_disk} | corrupt ledger lines = "
        "{corrupt_ledger_lines}".format(**c)
    )

    drift = report.get("drift") or []
    lines.append("")
    lines.append("DRIFT list ({n}):".format(n=len(drift)))
    if not drift:
        lines.append("  (none)")
    for item in drift[:max_items]:
        lines.append(
            "  - {metric_id}: {source_json} ledger={ledger_sha256} disk={disk_sha256} ({kind})".format(
                **item
            )
        )
    if len(drift) > max_items:
        lines.append("  ... and {n} more".format(n=len(drift) - max_items))

    no_run = report.get("no_run") or []
    lines.append("")
    lines.append("NO_RUN list ({n}):".format(n=len(no_run)))
    if not no_run:
        lines.append("  (none)")
    for mid in no_run[:max_items]:
        lines.append("  - {m}".format(m=mid))
    if len(no_run) > max_items:
        lines.append("  ... and {n} more".format(n=len(no_run) - max_items))

    uncovered = report.get("uncovered") or []
    lines.append("")
    lines.append("UNCOVERED list ({n}):".format(n=len(uncovered)))
    if not uncovered:
        lines.append("  (none)")
    for name in uncovered[:max_items]:
        lines.append("  - {n}".format(n=name))
    if len(uncovered) > max_items:
        lines.append("  ... and {n} more".format(n=len(uncovered) - max_items))

    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Audit the run ledger against DISK and against the metric registry "
            "(read-only, CPU-only)."
        )
    )
    ap.add_argument("--registry", type=Path, default=None, help="registry/metrics.yaml path")
    ap.add_argument("--ledger", type=Path, default=None, help="ledger JSONL path")
    ap.add_argument("--repo-root", type=Path, default=None, help="git work tree")
    ap.add_argument("--json", action="store_true", help="print the full report as JSON")
    args = ap.parse_args(argv)

    report = audit(
        args.registry,
        ledger_path=args.ledger,
        repo_root=args.repo_root,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print(render_summary(report))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
