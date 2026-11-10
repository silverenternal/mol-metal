# SPDX-License-Identifier: Apache-2.0
"""Wf-A170 — the census-freshness guard.

THE DEFECT CLASS
----------------
A *census artefact* is a JSON file that records the SHA-256 of a file it
described.  When the described file is still under active edit, the
artefact goes stale on every unrelated edit, and it goes stale
**silently**: the artefact still says ``PASS`` and still carries a
verdict derived from a version of the source that no longer exists.

That is the Wf-A159 shape (``timeout_verdict_census.json`` pinned a
``target_sha256`` that matched neither HEAD nor the working tree), and it
is the same shape as the §6.262 / §6.264 incidents.

WHAT THIS FILE GUARDS
---------------------
1. **NEGATIVE CONTROL** — a synthetic artefact whose recorded
   ``target_sha256`` does NOT match the file it names MUST be flagged
   ``STALE``.  Without this, the guard could be a no-op that always says
   "fresh".
2. **POSITIVE CONTROL** — a synthetic artefact whose recorded
   ``target_sha256`` DOES match the file it names MUST NOT be flagged.
   Without this, the guard could be a no-op that always says "stale".
3. **VACUITY GUARD** — every count the guard reports must equal the
   number of entries in the list it reports.  A guard that reports
   ``n_stale = 3`` while listing one entry is not a measurement.
4. **CPU-ONLY GUARD** — an AST check over the guard's IMPORT GRAPH
   (transitively, over local repo modules) plus a runtime probe: no
   ``torch``, no ``cupy``, and the guard spawns no subprocess.
5. **READ-ONLY GUARD** — scanning must not modify, touch or truncate
   anything it scans.
6. **TRUTHFULNESS** — the SHIPPED report must agree with a LIVE re-check
   of the files it names.  The guard's report is required to be TRUE, not
   required to be clean: a stale pin is a finding to report, not a
   reason to make this test brittle.

CPU-ONLY: every subprocess here is a pure ``sys.executable -c`` probe
with ``CUDA_VISIBLE_DEVICES=""``.  No GPU is touched, no checkpoint is
loaded, no ``torch`` is imported.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GUARD_REL = "molmetal/scripts/wfa170_census_freshness_guard.py"
GUARD_PATH = REPO_ROOT / GUARD_REL
REPORT_REL = "molmetal/reports/wfa170_census_freshness/report.json"

#: Import roots that would make this guard non-CPU-only.
FORBIDDEN_IMPORT_ROOTS = ("torch", "torchvision", "torchaudio", "cupy", "jax", "pynvml")

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _load_guard():
    """Import the guard module by path (the scripts are not a package)."""
    spec = importlib.util.spec_from_file_location("wfa170_guard", GUARD_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["wfa170_guard"] = mod
    spec.loader.exec_module(mod)
    return mod


guard = _load_guard()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_artefact(root: Path, name: str, payload: Dict[str, Any]) -> Path:
    """Write a synthetic artefact under ``root`` and return its path."""
    artefact = root / "synthetic_wave" / name
    artefact.parent.mkdir(parents=True, exist_ok=True)
    artefact.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return artefact


# ---------------------------------------------------------------------------
# 1 + 2 — NEGATIVE and POSITIVE controls
# ---------------------------------------------------------------------------
def test_a_stale_sha_pin_is_flagged(tmp_path) -> None:
    """NEGATIVE CONTROL: a recorded sha that does not match MUST be STALE."""
    reports_root = tmp_path / "reports"
    target = tmp_path / "target_source.py"
    target.write_text("VERSION_ONE = 1\n", encoding="utf-8")
    on_disk = _sha256_bytes(target.read_bytes())
    stale_sha = _sha256_bytes(b"VERSION_TWO = 2\n")
    assert stale_sha != on_disk

    _write_artefact(
        reports_root,
        "stale_census.json",
        {"target_file": str(target), "target_sha256": stale_sha},
    )

    report = guard.scan_reports(reports_root)
    assert report["status"] == "FAIL", report["summary"]
    assert report["scope_primary"]["n_stale"] == 1, report["scope_primary"]
    stale = report["stale"]
    assert len(stale) == 1, stale
    assert stale[0]["verdict"] == "STALE"
    assert stale[0]["artefact"].endswith("stale_census.json")
    assert stale[0]["recorded_sha256"] == stale_sha
    assert stale[0]["current_sha256"] == on_disk


def test_a_matching_sha_pin_is_not_flagged(tmp_path) -> None:
    """POSITIVE CONTROL: a recorded sha that matches MUST NOT be flagged."""
    reports_root = tmp_path / "reports"
    target = tmp_path / "target_source.py"
    target.write_text("VERSION_ONE = 1\n", encoding="utf-8")
    on_disk = _sha256_bytes(target.read_bytes())

    _write_artefact(
        reports_root,
        "fresh_census.json",
        {"target_file": str(target), "target_sha256": on_disk},
    )

    report = guard.scan_reports(reports_root)
    assert report["status"] == "PASS", report["summary"]
    assert report["scope_primary"]["n_stale"] == 0, report["scope_primary"]
    assert report["scope_primary"]["n_fresh"] == 1, report["scope_primary"]
    assert report["stale"] == []
    assert report["scope_primary"]["artefacts"][0]["current_sha256"] == on_disk


def test_a_missing_target_is_reported_as_missing_not_stale(tmp_path) -> None:
    """A named file that is not on disk is MISSING_TARGET, never STALE."""
    reports_root = tmp_path / "reports"
    _write_artefact(
        reports_root,
        "missing_census.json",
        {
            "target_file": str(tmp_path / "does_not_exist.py"),
            "target_sha256": _sha256_bytes(b"whatever"),
        },
    )
    report = guard.scan_reports(reports_root)
    assert report["scope_primary"]["n_stale"] == 0, report["scope_primary"]
    assert report["scope_primary"]["n_missing_target"] == 1, report["scope_primary"]


def test_a_sha_pin_with_no_named_file_is_unpaired_not_stale(tmp_path) -> None:
    """A sha with no sibling path key is a blind spot, never counted STALE."""
    reports_root = tmp_path / "reports"
    _write_artefact(
        reports_root,
        "unpaired_census.json",
        {"live_worktree_sha256_at_probe_time": _sha256_bytes(b"x")},
    )
    report = guard.scan_reports(reports_root)
    assert report["scope_primary"]["n_stale"] == 0, report["scope_primary"]
    assert report["summary"]["n_unpaired_sha_pins"] == 1, report["summary"]
    unpaired = report["scope_generalized"]["unpaired_sha_pins"]
    assert len(unpaired) == 1, unpaired
    assert unpaired[0]["verdict"] == "UNPAIRED"


# ---------------------------------------------------------------------------
# 3 — VACUITY GUARD
# ---------------------------------------------------------------------------
def test_every_reported_count_equals_its_list_length(tmp_path) -> None:
    """VACUITY GUARD: n_stale == len(stale), and so on, for every scope."""
    reports_root = tmp_path / "reports"
    target = tmp_path / "target_source.py"
    target.write_text("VERSION_ONE = 1\n", encoding="utf-8")
    on_disk = _sha256_bytes(target.read_bytes())
    stale_sha = _sha256_bytes(b"VERSION_TWO = 2\n")

    _write_artefact(
        reports_root,
        "stale_census.json",
        {"target_file": str(target), "target_sha256": stale_sha},
    )
    _write_artefact(
        reports_root,
        "fresh_census.json",
        {"target_file": str(target), "target_sha256": on_disk},
    )

    report = guard.scan_reports(reports_root)
    s = report["summary"]
    all_records = (
        report["scope_primary"]["artefacts"]
        + report["scope_generalized"]["artefacts"]
    )
    assert s["n_stale"] == len(report["stale"]) == sum(
        1 for r in all_records if r["verdict"] == "STALE"
    )
    assert s["n_fresh"] == sum(1 for r in all_records if r["verdict"] == "FRESH")
    assert s["n_missing_target"] == sum(
        1 for r in all_records if r["verdict"] == "MISSING_TARGET"
    )
    assert s["n_skipped_too_large"] == sum(
        1 for r in all_records if r["verdict"] == "SKIPPED_TOO_LARGE"
    )
    assert s["n_sha_pins_checked"] == len(all_records)
    assert s["n_stale_artefacts"] == len(report["stale_artefacts"])
    assert report["scope_primary"]["n_pins"] == len(
        report["scope_primary"]["artefacts"]
    )
    assert report["scope_primary"]["n_stale"] == len(
        [
            r
            for r in report["scope_primary"]["artefacts"]
            if r["verdict"] == "STALE"
        ]
    )
    assert report["scope_primary"]["n_artefacts"] == len(
        {r["artefact"] for r in report["scope_primary"]["artefacts"]}
    )
    assert report["scope_generalized"]["n_pins"] == len(
        report["scope_generalized"]["artefacts"]
    )
    assert report["scope_generalized"]["n_stale"] == len(
        [
            r
            for r in report["scope_generalized"]["artefacts"]
            if r["verdict"] == "STALE"
        ]
    )
    assert report["scope_generalized"]["n_unpaired_sha_pins"] == len(
        report["scope_generalized"]["unpaired_sha_pins"]
    )


def test_the_shipped_report_is_internally_consistent() -> None:
    """VACUITY GUARD, on the shipped artefact: counts must match lists."""
    report_path = REPO_ROOT / REPORT_REL
    assert report_path.is_file(), (
        f"the Wf-A170 report is missing on disk: {report_path} — run "
        f"`python {GUARD_REL}`"
    )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    s = report["summary"]
    assert s["n_stale"] == len(report["stale"]), (s["n_stale"], len(report["stale"]))
    assert s["n_stale_artefacts"] == len(report["stale_artefacts"])
    assert s["n_sha_pins_checked"] == len(report["scope_primary"]["artefacts"]) + len(
        report["scope_generalized"]["artefacts"]
    )
    assert report["scope_primary"]["n_stale"] == len(
        [
            r
            for r in report["scope_primary"]["artefacts"]
            if r["verdict"] == "STALE"
        ]
    )
    assert report["scope_generalized"]["n_stale"] == len(
        [
            r
            for r in report["scope_generalized"]["artefacts"]
            if r["verdict"] == "STALE"
        ]
    )
    assert report["advisory"]["n_stale_today"] == s["n_stale"]
    assert report["advisory"]["would_block_launch_if_fail"] == (s["n_stale"] > 0)


# ---------------------------------------------------------------------------
# 4 — CPU-ONLY GUARD
# ---------------------------------------------------------------------------
def _local_import_roots(tree: ast.Module) -> List[str]:
    """Every module named by an ``import`` / ``from ... import`` in ``tree``."""
    roots: List[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                roots.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import — not resolvable as a repo module
                continue
            if node.module:
                roots.append(node.module.split(".")[0])
    return roots


def _import_graph(root: Path) -> Set[str]:
    """The transitive import roots reachable from ``root``.

    Local repo modules are followed (a module that imports torch through
    a sibling would still be caught); stdlib modules are leaves.
    """
    seen: Set[str] = set()
    queue: List[Path] = [root]
    while queue:
        current = queue.pop()
        if current in seen or not current.is_file():
            continue
        seen.add(current)
        tree = ast.parse(current.read_text(encoding="utf-8"))
        for name in _local_import_roots(tree):
            if name in sys.stdlib_module_names:
                continue
            candidate = REPO_ROOT / name
            if candidate.is_dir():
                candidate = candidate / "__init__.py"
            if candidate.is_file():
                queue.append(candidate)
            else:
                seen.add(candidate)  # unresolvable: record the root name
    return seen


def test_the_guard_imports_no_torch_anywhere_in_its_import_graph() -> None:
    """CPU-ONLY GUARD (AST): no torch/cupy/jax in the transitive import graph."""
    roots: Set[str] = set()
    for path in _import_graph(GUARD_PATH):
        if path.suffix == ".py":
            tree = ast.parse(path.read_text(encoding="utf-8"))
            roots.update(_local_import_roots(tree))
        else:
            roots.add(path.name)
    offending = sorted(
        r for r in roots if any(r.startswith(bad) for bad in FORBIDDEN_IMPORT_ROOTS)
    )
    assert not offending, (
        f"the census-freshness guard must be CPU-only, but its import graph "
        f"reaches {offending}"
    )


def test_the_guard_spawns_no_subprocess_and_imports_no_torch() -> None:
    """CPU-ONLY GUARD (AST): the guard must not spawn a subprocess."""
    src = GUARD_PATH.read_text(encoding="utf-8")
    tree = ast.parse(src)
    assert "import torch" not in src
    assert "import subprocess" not in src, (
        "the census-freshness guard must not import subprocess — it reads the "
        "git HEAD from files, it does not shell out"
    )
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fname = None
            f = node.func
            if isinstance(f, ast.Name):
                fname = f.id
            elif isinstance(f, ast.Attribute):
                fname = f.attr
            assert fname not in ("run", "Popen", "check_output", "check_call"), (
                f"the guard must not spawn a subprocess (found {fname!r} at "
                f"line {node.lineno})"
            )


def test_importing_the_guard_touches_no_torch(tmp_path) -> None:
    """CPU-ONLY GUARD (runtime): importing the guard imports no torch."""
    probe = tmp_path / "probe_import.py"
    probe.write_text(
        "import importlib.util, sys\n"
        "spec = importlib.util.spec_from_file_location('wfa170_g', "
        f"{str(GUARD_PATH)!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "sys.modules['wfa170_g'] = mod\n"
        "spec.loader.exec_module(mod)\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
        "assert 'cupy' not in sys.modules, 'cupy was imported'\n"
        "print('NO_TORCH')\n",
        encoding="utf-8",
    )
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    proc = subprocess.run(
        [sys.executable, str(probe)],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )
    assert proc.returncode == 0, (
        f"the CPU-only import probe failed (rc={proc.returncode}): {proc.stderr[-2000:]}"
    )
    assert "NO_TORCH" in proc.stdout


# ---------------------------------------------------------------------------
# 5 — READ-ONLY GUARD
# ---------------------------------------------------------------------------
def test_scanning_does_not_modify_anything_it_scans(tmp_path) -> None:
    """The guard must not write to the tree it measures."""
    reports_root = tmp_path / "reports"
    target = tmp_path / "target_source.py"
    target.write_text("VERSION_ONE = 1\n", encoding="utf-8")
    artefact = _write_artefact(
        reports_root,
        "census.json",
        {
            "target_file": str(target),
            "target_sha256": _sha256_bytes(target.read_bytes()),
        },
    )
    before = {
        p: (p.stat().st_mtime_ns, p.stat().st_size, p.read_bytes())
        for p in (target, artefact)
    }
    guard.scan_reports(reports_root)
    for path, (mtime, size, content) in before.items():
        assert path.stat().st_mtime_ns == mtime, f"{path} was rewritten"
        assert path.stat().st_size == size, f"{path} changed size"
        assert path.read_bytes() == content, f"{path} content changed"


# ---------------------------------------------------------------------------
# 6 — TRUTHFULNESS of the shipped report
# ---------------------------------------------------------------------------
def test_the_shipped_report_is_internally_truthful() -> None:
    """The report's verdict must follow from the hashes the report records.

    This is the "the report does not lie about itself" check: a record
    whose ``current_sha256`` equals its ``recorded_sha256`` MUST be
    FRESH, and one where they differ MUST be STALE.  It does NOT require
    the tree to be clean — a stale pin is a finding the report is
    SUPPOSED to carry.
    """
    report_path = REPO_ROOT / REPORT_REL
    assert report_path.is_file(), report_path
    report = json.loads(report_path.read_text(encoding="utf-8"))
    records = (
        report["scope_primary"]["artefacts"]
        + report["scope_generalized"]["artefacts"]
    )
    assert records, "the shipped report names no pinned files at all"
    for record in records:
        verdict = record["verdict"]
        if verdict in ("FRESH", "STALE"):
            expected = (
                "FRESH"
                if record["current_sha256"] == record["recorded_sha256"]
                else "STALE"
            )
            assert verdict == expected, record
        else:
            assert verdict in ("MISSING_TARGET", "SKIPPED_TOO_LARGE"), record
            assert record["current_sha256"] is None, record


def test_the_shipped_report_agrees_with_a_live_recheck_of_unchanged_targets() -> None:
    """Live re-check, limited to targets that did NOT move since the report.

    The report is a SNAPSHOT of a tree other waves keep editing.  If a
    named file's live content still matches the ``current_sha256`` the
    report recorded, then the report's verdict for it must still be the
    verdict the file earns.  A file that MOVED after the report is
    skipped: the report was true when it was written.
    """
    report_path = REPO_ROOT / REPORT_REL
    assert report_path.is_file(), report_path
    report = json.loads(report_path.read_text(encoding="utf-8"))
    records = (
        report["scope_primary"]["artefacts"]
        + report["scope_generalized"]["artefacts"]
    )
    checked = 0
    for record in records:
        resolved = Path(record["resolved_target"])
        if not resolved.is_file():
            assert record["verdict"] == "MISSING_TARGET", record
            continue
        if record["verdict"] == "SKIPPED_TOO_LARGE":
            continue
        live = _sha256_bytes(resolved.read_bytes())
        if live != record["current_sha256"]:
            continue  # the target moved after the report was written
        expected = "FRESH" if live == record["recorded_sha256"] else "STALE"
        assert record["verdict"] == expected, (
            f"{record['artefact']} {record['json_pointer']}: the shipped report "
            f"says {record['verdict']} but the live hash says {expected}"
        )
        checked += 1
    assert checked > 0, "no target was live-rechecked — the guard would be theatre"


def test_the_primary_scope_covers_the_wfa159_census_artefact() -> None:
    """The Wf-A159 artefact is the canonical member of the primary scope."""
    report_path = REPO_ROOT / REPORT_REL
    assert report_path.is_file(), report_path
    report = json.loads(report_path.read_text(encoding="utf-8"))
    artefacts = {r["artefact"] for r in report["scope_primary"]["artefacts"]}
    assert (
        "molmetal/reports/wfa159_timeout_verdict_census/timeout_verdict_census.json"
        in artefacts
    ), sorted(artefacts)


def test_the_guard_reads_the_git_head_without_a_subprocess() -> None:
    """The git HEAD is read as FILES — a spot-check of the reader."""
    head = guard._git_head()
    assert isinstance(head, str)
    assert head != ""


def test_the_guard_reports_its_own_denominator_not_an_impression() -> None:
    """The guard's denominator must be the artefact count it really scanned."""
    reports_root = REPO_ROOT / "molmetal/reports"
    if not reports_root.is_dir():
        pytest.skip("no reports root on disk")
    report = guard.scan_reports(reports_root, max_bytes=1024 * 1024)
    s = report["summary"]
    assert s["n_json_artefacts_scanned"] >= s["n_json_artefacts_parsed"]
    assert (
        s["n_json_artefacts_scanned"]
        == s["n_json_artefacts_parsed"] + s["n_json_parse_failures"]
    )
    assert s["n_artefacts_with_a_sha_pin"] >= 1
    assert s["n_sha_pins_checked"] >= s["n_artefacts_with_a_sha_pin"]
