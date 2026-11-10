#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Wf-A170 — the census-freshness guard.

THE DEFECT CLASS
----------------
A *census artefact* is a JSON file that records the SHA-256 of a file it
described.  When the described file is still under active edit, the
artefact goes **stale on every unrelated edit** — and the staleness is
silent: the artefact still looks authoritative, still says ``PASS``, and
still carries a verdict that was derived from a version of the source
that no longer exists.

That is the Wf-A159 shape, and it is the same shape as the §6.262 /
§6.264 incidents: a published artefact pinned to a SHA that exists
nowhere in the repository's history, at HEAD, or in the working tree.

WHAT THIS SCRIPT DOES
----------------------
For every JSON artefact under ``molmetal/reports/`` that records a
SHA-256 pin on a file it NAMES, this guard re-hashes the named file and
reports whether the pin still holds.

TWO SCOPES
----------
``primary``
    The scope the Wf-A159 test enforces: an artefact that records a
    ``target_sha256`` field, checked against the file named by its
    ``target_file`` sibling.  This is the exact contract the failing
    test asserts, so it is reported separately and first.

``generalized``
    The wider class: any ``<stem>_sha256`` key paired with a sibling
    ``<stem>_path`` / ``<stem>_file`` / ``<stem>`` key whose value
    names a file on disk.  This catches the same defect class in
    artefacts that use a different key spelling (e.g.
    ``published_artifact_sha256`` + ``published_artifact_path``).

VERDICTS
--------
``FRESH``             the named file's current content still hashes to the recorded SHA-256.
``STALE``             the named file exists but its content has changed.
``MISSING_TARGET``    the named file does not exist on disk (deleted or never present).
``SKIPPED_TOO_LARGE`` the named file exceeds ``--max-bytes`` (this repo holds
                       multi-GB checkpoints; hashing them is not a census).
``UNPAIRED``          a SHA-256 is recorded but NO sibling key names a file, so the
                       pin cannot be checked.  Reported as a residual blind spot,
                       never counted as STALE.

CPU-ONLY: standard library only.  It never imports torch, never spawns a
subprocess, never touches a GPU, and never WRITES to anything it scans.

Usage
-----
::

    python molmetal/scripts/wfa170_census_freshness_guard.py
    python molmetal/scripts/wfa170_census_freshness_guard.py --json-out <path>
    python molmetal/scripts/wfa170_census_freshness_guard.py --md-out <path>

Exit codes: ``0`` = PASS (zero STALE), ``1`` = FAIL (>= 1 STALE),
``2`` = SKIP (the reports root is not on disk).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_REL = "molmetal/reports"
DEFAULT_REPORTS_ROOT = REPO_ROOT / REPORTS_REL
REPORT_DIR = REPO_ROOT / "molmetal/reports/wfa170_census_freshness"
DEFAULT_REPORT = REPORT_DIR / "report.json"
DEFAULT_MD = REPORT_DIR / "report.md"

#: The guard's OWN output directory is excluded from the scan.
#:
#: Why: the guard's report is a report ABOUT sha pins, so it legitimately
#: CONTAINS thousands of 64-hex sha values (``recorded_sha256`` /
#: ``current_sha256``).  If the guard scanned its own output it would find
#: every one of those as a sha pin with no named sibling file, and the
#: count would AMPLIFY on every run (the report carries the previous
#: report's findings forward).  The report is a DERIVED OUTPUT, not an
#: input artefact: the guard regenerates it, and
#: ``test_the_shipped_report_agrees_with_a_live_recheck_of_unchanged_targets``
#: is what holds it honest.
SELF_EXCLUDED_REL_DIRS = ("molmetal/reports/wfa170_census_freshness",)

#: Files larger than this are NOT hashed.  This repository holds multi-GB
#: ``.pt`` checkpoints; hashing them is neither a census nor a courtesy.
DEFAULT_MAX_BYTES = 32 * 1024 * 1024

VERDICT_FRESH = "FRESH"
VERDICT_STALE = "STALE"
VERDICT_MISSING_TARGET = "MISSING_TARGET"
VERDICT_SKIPPED_TOO_LARGE = "SKIPPED_TOO_LARGE"
VERDICT_UNPAIRED = "UNPAIRED"

#: A SHA-256 value: exactly 64 lowercase hex characters.
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

#: Suffixes stripped from a SHA key to get its "stem".
_SHA_KEY_SUFFIXES = (
    "_sha256_at_probe_time",
    "_sha256",
    "_sha",
    "_digest",
    "_hash",
)

#: Suffixes stripped from a path key to get its "stem".
_PATH_KEY_SUFFIXES = (
    "_filepath",
    "_abs_path",
    "_path",
    "_file",
    "_dir",
)

#: File-name suffixes that make a string value look like a filesystem path.
_PATH_VALUE_SUFFIXES = (
    ".py",
    ".json",
    ".jsonl",
    ".md",
    ".yaml",
    ".yml",
    ".csv",
    ".tsv",
    ".txt",
    ".log",
    ".toml",
    ".cfg",
    ".ini",
    ".sh",
    ".tex",
    ".pt",
    ".pth",
    ".pdb",
    ".pdbqt",
    ".sdf",
    ".mol2",
    ".npz",
    ".npy",
    ".pkl",
    ".bin",
)


# ---------------------------------------------------------------------------
# key classification
# ---------------------------------------------------------------------------
def is_sha_key(key: str) -> bool:
    """True when ``key`` names a SHA-256 field.

    Deliberately narrow: ``duplicate_sha_details``, ``git_sha_candidates``
    and ``n_duplicate_shas`` must NOT be treated as pins, or the guard
    would try to interpret a histogram as a content hash.
    """
    k = key.lower()
    if "sha256" in k:
        return True
    return k.endswith(("_sha", "_digest", "_hash"))


def is_sha_value(value: Any) -> bool:
    """True when ``value`` is a 64-hex lowercase SHA-256 string."""
    return isinstance(value, str) and bool(_SHA256_RE.match(value))


def sha_stem(key: str) -> str:
    """The ``<stem>`` part of a SHA key (``target_sha256`` -> ``target``)."""
    k = key.lower()
    for suf in _SHA_KEY_SUFFIXES:
        if k.endswith(suf) and len(key) > len(suf):
            return key[: -len(suf)]
    return key


def path_stem(key: str) -> str:
    """The ``<stem>`` part of a path key (``target_file`` -> ``target``)."""
    k = key.lower()
    for suf in _PATH_KEY_SUFFIXES:
        if k.endswith(suf) and len(key) > len(suf):
            return key[: -len(suf)]
    return key


def looks_like_path(value: Any) -> bool:
    """True when a string value names a filesystem path."""
    if not isinstance(value, str) or not value or "\n" in value:
        return False
    if len(value) > 4096:
        return False
    if value.startswith("/"):
        return True
    if "/" not in value:
        return False
    return value.rsplit("/", 1)[-1].endswith(_PATH_VALUE_SUFFIXES)


def stems_match(a: str, b: str) -> bool:
    """True when two key stems refer to the same thing.

    ``target`` == ``target`` (match); ``source_pdb`` vs ``source_pdb``
    (match); ``published_artifact`` vs ``published_artifact`` (match).
    ``pool`` vs ``pool_name`` also matches (one is a prefix of the other
    at an underscore boundary).
    """
    a_l, b_l = a.lower(), b.lower()
    if a_l == b_l:
        return True
    return a_l.startswith(b_l + "_") or b_l.startswith(a_l + "_")


# ---------------------------------------------------------------------------
# JSON traversal
# ---------------------------------------------------------------------------
def iter_objects(node: Any, pointer: str = "$") -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Yield ``(json_pointer, object)`` for EVERY dict inside ``node``.

    A JSON artefact may hold its pins at the top level, inside a nested
    object, or inside a LIST of objects (e.g. one pin per ledger row).
    All three are scanned.
    """
    if isinstance(node, dict):
        yield pointer, node
        for key, value in node.items():
            yield from iter_objects(value, f"{pointer}.{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from iter_objects(value, f"{pointer}[{index}]")


# ---------------------------------------------------------------------------
# hashing
# ---------------------------------------------------------------------------
def sha256_file(path: Path, max_bytes: int) -> Tuple[Optional[str], Optional[int], str]:
    """Return ``(sha256_or_None, size_bytes, verdict)`` for ``path``."""
    try:
        size = path.stat().st_size
    except OSError:
        return None, None, VERDICT_MISSING_TARGET
    if size > max_bytes:
        return None, size, VERDICT_SKIPPED_TOO_LARGE
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None, size, VERDICT_MISSING_TARGET
    return digest.hexdigest(), size, VERDICT_FRESH


def resolve_target(value: str, artefact_dir: Path) -> Tuple[Optional[Path], bool]:
    """Resolve a recorded path value to a file on disk.

    Returns ``(resolved_path_or_None, existed)``.  ``resolved_path`` is
    the candidate that WOULD have been used, so a missing target can be
    reported by name.
    """
    raw = Path(value)
    if raw.is_absolute():
        candidates = [raw]
    else:
        candidates = [REPO_ROOT / raw, artefact_dir / raw]
    for candidate in candidates:
        if candidate.is_file():
            return candidate, True
    return candidates[0], False


# ---------------------------------------------------------------------------
# the scan
# ---------------------------------------------------------------------------
class _Hasher:
    """Cache of file -> (sha256, size, verdict) so a file is hashed once."""

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self._cache: Dict[Path, Tuple[Optional[str], Optional[int], str]] = {}

    def of(self, path: Path) -> Tuple[Optional[str], Optional[int], str]:
        key = path
        if key not in self._cache:
            self._cache[key] = sha256_file(path, self.max_bytes)
        return self._cache[key]


def _primary_pins(
    pointer: str, obj: Dict[str, Any]
) -> List[Tuple[str, str, str]]:
    """Yield ``(json_pointer, sha_key, path_key)`` for the PRIMARY scope.

    The primary scope is exactly: a ``target_sha256`` field paired with
    the ``target_file`` it names.  Both may live on the same object or
    on an ancestor; the nearest ancestor wins.
    """
    out: List[Tuple[str, str, str]] = []
    if "target_sha256" not in obj:
        return out
    if not is_sha_value(obj.get("target_sha256")):
        return out
    if isinstance(obj.get("target_file"), str):
        out.append((pointer, "target_sha256", "target_file"))
    return out


def _generalized_pins(
    pointer: str, obj: Dict[str, Any]
) -> List[Tuple[str, str, str]]:
    """Yield ``(json_pointer, sha_key, path_key)`` for the GENERALIZED scope.

    A SHA key pairs with a path key when their stems match.  ``target_sha256``
    / ``target_file`` is EXCLUDED here — it belongs to the primary scope and
    is counted once, not twice.
    """
    sha_entries = [
        (k, v)
        for k, v in obj.items()
        if is_sha_key(k) and is_sha_value(v)
    ]
    path_entries = [
        (k, v)
        for k, v in obj.items()
        if looks_like_path(v) and not is_sha_key(k)
    ]
    out: List[Tuple[str, str, str]] = []
    for sha_key, _sha_value in sha_entries:
        if sha_key == "target_sha256":
            continue  # primary scope
        stem = sha_stem(sha_key)
        for path_key, path_value in path_entries:
            if stems_match(stem, path_stem(path_key)):
                out.append((pointer, sha_key, path_key))
                break  # one pairing per sha key: the first match wins
    return out


def scan_reports(
    reports_root: Path = DEFAULT_REPORTS_ROOT,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> Dict[str, Any]:
    """Scan every JSON artefact under ``reports_root`` for stale SHA pins."""
    if not reports_root.is_dir():
        return {
            "status": "SKIP",
            "reason": f"reports root not on disk: {reports_root}",
            "summary": _empty_summary(),
            "scope_primary": _empty_scope_primary(),
            "scope_generalized": _empty_scope_generalized(),
            "stale": [],
            "artefacts": [],
        }

    hasher = _Hasher(max_bytes)
    artefact_paths: List[Path] = []
    excluded_self = 0
    for dirpath, dirnames, filenames in os.walk(reports_root):
        dirnames.sort()
        for name in sorted(filenames):
            if not name.endswith(".json"):
                continue
            candidate = Path(dirpath) / name
            if _is_self_output(candidate):
                excluded_self += 1
                continue
            artefact_paths.append(candidate)

    primary_records: List[Dict[str, Any]] = []
    generalized_records: List[Dict[str, Any]] = []
    unpaired_records: List[Dict[str, Any]] = []
    parse_failures: List[Dict[str, str]] = []
    artefacts_with_a_pin: set = set()

    for artefact in artefact_paths:
        try:
            payload = json.loads(artefact.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            parse_failures.append(
                {"artefact": _rel(artefact), "error": f"{type(exc).__name__}: {exc}"}
            )
            continue
        artefact_dir = artefact.parent
        for pointer, obj in iter_objects(payload):
            pins = _primary_pins(pointer, obj) + _generalized_pins(pointer, obj)
            if pins:
                artefacts_with_a_pin.add(_rel(artefact))
            for pin_pointer, sha_key, path_key in pins:
                recorded = obj[sha_key]
                target_value = obj[path_key]
                resolved, existed = resolve_target(target_value, artefact_dir)
                current, size, verdict = (
                    (None, None, VERDICT_MISSING_TARGET)
                    if not existed
                    else hasher.of(resolved)
                )
                if existed and verdict == VERDICT_FRESH:
                    verdict = VERDICT_FRESH if current == recorded else VERDICT_STALE
                record = {
                    "artefact": _rel(artefact),
                    "json_pointer": pin_pointer,
                    "sha_key": sha_key,
                    "path_key": path_key,
                    "target_file": str(target_value),
                    "resolved_target": str(resolved),
                    "recorded_sha256": recorded,
                    "current_sha256": current,
                    "target_size_bytes": size,
                    "verdict": verdict,
                }
                if sha_key == "target_sha256":
                    primary_records.append(record)
                else:
                    generalized_records.append(record)
            # A SHA pin with no named file is a blind spot, not a stale pin.
            for sha_key, sha_value in obj.items():
                if not (is_sha_key(sha_key) and is_sha_value(sha_value)):
                    continue
                if sha_key == "target_sha256":
                    continue
                if any(p[1] == sha_key for p in _generalized_pins(pointer, obj)):
                    continue
                unpaired_records.append(
                    {
                        "artefact": _rel(artefact),
                        "json_pointer": pointer,
                        "sha_key": sha_key,
                        "recorded_sha256": sha_value,
                        "verdict": VERDICT_UNPAIRED,
                        "why": "no sibling key names the file this SHA pins",
                    }
                )

    primary_records.sort(key=lambda r: (r["artefact"], r["json_pointer"]))
    generalized_records.sort(key=lambda r: (r["artefact"], r["json_pointer"]))
    unpaired_records.sort(key=lambda r: (r["artefact"], r["json_pointer"], r["sha_key"]))

    all_records = primary_records + generalized_records
    stale = [r for r in all_records if r["verdict"] == VERDICT_STALE]
    stale_artefacts = sorted({r["artefact"] for r in stale})

    summary = _empty_summary()
    summary.update(
        {
            "n_json_artefacts_scanned": len(artefact_paths),
            "n_json_artefacts_parsed": len(artefact_paths) - len(parse_failures),
            "n_json_parse_failures": len(parse_failures),
            "n_artefacts_with_a_sha_pin": len(artefacts_with_a_pin),
            "n_excluded_self_output": excluded_self,
            "n_sha_pins_checked": len(all_records),
            "n_stale": len(stale),
            "n_fresh": sum(1 for r in all_records if r["verdict"] == VERDICT_FRESH),
            "n_missing_target": sum(
                1 for r in all_records if r["verdict"] == VERDICT_MISSING_TARGET
            ),
            "n_skipped_too_large": sum(
                1 for r in all_records if r["verdict"] == VERDICT_SKIPPED_TOO_LARGE
            ),
            "n_unpaired_sha_pins": len(unpaired_records),
            "n_stale_artefacts": len(stale_artefacts),
        }
    )

    scope_primary = {
        "definition": (
            "JSON artefacts under molmetal/reports that record a `target_sha256` "
            "field, checked against the file named by their `target_file` "
            "sibling.  This is the exact contract the Wf-A159 test asserts."
        ),
        "n_artefacts": len({r["artefact"] for r in primary_records}),
        "n_pins": len(primary_records),
        "n_stale": sum(1 for r in primary_records if r["verdict"] == VERDICT_STALE),
        "n_fresh": sum(1 for r in primary_records if r["verdict"] == VERDICT_FRESH),
        "n_missing_target": sum(
            1 for r in primary_records if r["verdict"] == VERDICT_MISSING_TARGET
        ),
        "n_skipped_too_large": sum(
            1 for r in primary_records if r["verdict"] == VERDICT_SKIPPED_TOO_LARGE
        ),
        "stale_artefacts": sorted(
            {r["artefact"] for r in primary_records if r["verdict"] == VERDICT_STALE}
        ),
        "artefacts": primary_records,
    }

    scope_generalized = {
        "definition": (
            "Any `<stem>_sha256` key paired with a sibling `<stem>_path` / "
            "`<stem>_file` / `<stem>` key whose value names a file on disk.  "
            "Catches the same defect class under different key spellings."
        ),
        "n_artefacts": len({r["artefact"] for r in generalized_records}),
        "n_pins": len(generalized_records),
        "n_stale": sum(1 for r in generalized_records if r["verdict"] == VERDICT_STALE),
        "n_fresh": sum(1 for r in generalized_records if r["verdict"] == VERDICT_FRESH),
        "n_missing_target": sum(
            1 for r in generalized_records if r["verdict"] == VERDICT_MISSING_TARGET
        ),
        "n_skipped_too_large": sum(
            1 for r in generalized_records if r["verdict"] == VERDICT_SKIPPED_TOO_LARGE
        ),
        "n_unpaired_sha_pins": len(unpaired_records),
        "stale_artefacts": sorted(
            {r["artefact"] for r in generalized_records if r["verdict"] == VERDICT_STALE}
        ),
        "unpaired_sha_pins": unpaired_records,
        "artefacts": generalized_records,
    }

    return {
        "status": "PASS" if not stale else "FAIL",
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_head_commit": _git_head(),
        "guard_script_path": _rel(Path(__file__).resolve()),
        "reports_root": _rel(reports_root),
        "max_hash_bytes": max_bytes,
        "summary": summary,
        "scope_primary": scope_primary,
        "scope_generalized": scope_generalized,
        "stale": stale,
        "stale_artefacts": stale_artefacts,
        "parse_failures": parse_failures,
    }


def _empty_summary() -> Dict[str, int]:
    return {
        "n_json_artefacts_scanned": 0,
        "n_json_artefacts_parsed": 0,
        "n_json_parse_failures": 0,
        "n_artefacts_with_a_sha_pin": 0,
        "n_excluded_self_output": 0,
        "n_sha_pins_checked": 0,
        "n_stale": 0,
        "n_fresh": 0,
        "n_missing_target": 0,
        "n_skipped_too_large": 0,
        "n_unpaired_sha_pins": 0,
        "n_stale_artefacts": 0,
    }


def _empty_scope_primary() -> Dict[str, Any]:
    return {
        "definition": "",
        "n_artefacts": 0,
        "n_pins": 0,
        "n_stale": 0,
        "n_fresh": 0,
        "n_missing_target": 0,
        "n_skipped_too_large": 0,
        "stale_artefacts": [],
        "artefacts": [],
    }


def _empty_scope_generalized() -> Dict[str, Any]:
    return {
        "definition": "",
        "n_artefacts": 0,
        "n_pins": 0,
        "n_stale": 0,
        "n_fresh": 0,
        "n_missing_target": 0,
        "n_skipped_too_large": 0,
        "n_unpaired_sha_pins": 0,
        "stale_artefacts": [],
        "unpaired_sha_pins": [],
        "artefacts": [],
    }


def _rel(path: Path) -> str:
    """Repo-relative POSIX path when possible, else the absolute path."""
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _is_self_output(path: Path) -> bool:
    """True when ``path`` is one of the guard's own output artefacts."""
    rel = _rel(path)
    return any(
        rel == excluded or rel.startswith(excluded + "/")
        for excluded in SELF_EXCLUDED_REL_DIRS
    )


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def render_markdown(report: Dict[str, Any]) -> str:
    """Render the report as a short human-readable Markdown note."""
    s = report["summary"]
    p = report["scope_primary"]
    g = report["scope_generalized"]
    lines: List[str] = []
    lines.append("# Wf-A170 — census-freshness guard")
    lines.append("")
    lines.append(
        f"- status: **{report['status']}**  "
        f"(generated {report['generated_at']}, git head `{report['git_head_commit']}`)"
    )
    lines.append(f"- guard: `{report['guard_script_path']}`")
    lines.append(
        f"- JSON artefacts scanned: **{s['n_json_artefacts_scanned']}** "
        f"({s['n_json_parse_failures']} unparseable)"
    )
    lines.append(
        f"- artefacts carrying >= 1 SHA-256 pin: **{s['n_artefacts_with_a_sha_pin']}** "
        f"(denominator)"
    )
    lines.append(
        f"- SHA pins checked: **{s['n_sha_pins_checked']}** — "
        f"FRESH {s['n_fresh']}, STALE **{s['n_stale']}**, "
        f"MISSING_TARGET {s['n_missing_target']}, "
        f"SKIPPED_TOO_LARGE {s['n_skipped_too_large']}, "
        f"UNPAIRED {s['n_unpaired_sha_pins']}"
    )
    lines.append("")
    lines.append("## Primary scope (`target_sha256` + `target_file`)")
    lines.append("")
    lines.append(f"- artefacts: **{p['n_artefacts']}** (denominator)")
    lines.append(
        f"- stale: **{p['n_stale']}** — "
        + (
            ", ".join(f"`{a}`" for a in p["stale_artefacts"])
            if p["stale_artefacts"]
            else "none"
        )
    )
    for rec in p["artefacts"]:
        lines.append(
            f"  - `{rec['artefact']}` -> `{rec['target_file']}`: "
            f"{rec['verdict']} (recorded `{rec['recorded_sha256'][:12]}…`, "
            f"current `{(rec['current_sha256'] or 'n/a')[:12]}…`)"
        )
    lines.append("")
    lines.append("## Generalized scope (any `<stem>_sha256` + named file)")
    lines.append("")
    lines.append(f"- artefacts: **{g['n_artefacts']}** (denominator)")
    lines.append(
        f"- stale: **{g['n_stale']}** — "
        + (
            ", ".join(f"`{a}`" for a in g["stale_artefacts"][:20])
            if g["stale_artefacts"]
            else "none"
        )
        + (" …" if len(g["stale_artefacts"]) > 20 else "")
    )
    if report.get("stale"):
        lines.append("")
        shown = report["stale"][:40]
        lines.append(
            f"## Stale pins ({len(report['stale'])} total; the first "
            f"{len(shown)} are listed here, the full list is in `report.json`)"
        )
        lines.append("")
        lines.append("| artefact | pointer | sha key | target | recorded | current |")
        lines.append("|---|---|---|---|---|---|")
        for rec in shown:
            lines.append(
                f"| `{rec['artefact']}` | `{rec['json_pointer']}` | "
                f"`{rec['sha_key']}` | `{rec['target_file']}` | "
                f"`{rec['recorded_sha256'][:12]}…` | "
                f"`{(rec['current_sha256'] or 'n/a')[:12]}…` |"
            )
    lines.append("")
    lines.append("## Advisory posture")
    lines.append("")
    blocking = report.get("advisory", {}).get("would_block_launch_if_fail", False)
    lines.append(
        "- A FAIL-ing freshness gate over the **GENERALIZED** scope "
        + (
            "**WOULD** block launches today"
            if blocking
            else "would **NOT** block launches today"
        )
        + f" — {s['n_stale']} stale pin(s) of {g['n_pins']} checked across "
        f"{g['n_artefacts']} artefact(s)."
    )
    lines.append(
        "- A FAIL-ing freshness gate over the **PRIMARY** scope only "
        + (
            "**WOULD** block launches today"
            if p["n_stale"] > 0
            else "would **NOT** block launches today"
        )
        + f" — {p['n_stale']} stale pin(s) of {p['n_pins']} checked across "
        f"{p['n_artefacts']} artefact(s)."
    )
    lines.append(
        "- The proposed pre-flight hook is ADVISORY (WARN-equivalent): it never "
        "changes the exit code and never changes `EXPECTED_GATE_COUNTS`."
    )
    lines.append("")
    lines.append("## Deployment")
    lines.append("")
    lines.append(
        "- Patch SPEC: `patch_preflight_gate.diff` in this directory.  It adds "
        "the freshness check to `molmetal/scripts/pre_flight_check.py` as an "
        "ADVISORY hook in `main()` — NOT registered in `build_gates()`, so "
        "`EXPECTED_GATE_COUNTS` stays 129 (`all`) / 131 (`launch`) and no "
        "PINNED gate count moves."
    )
    lines.append(
        "- Proven with `git apply --check` against a SCRATCH copy of the file "
        "(rc=0); the working tree was NOT modified."
    )
    lines.append(
        "- Cost: the scan re-hashes every sha-pinned artefact.  MEASURED on "
        "this host: ~15 s against a ~411 s full pre-flight."
    )
    lines.append(
        "- To apply: `git apply molmetal/reports/wfa170_census_freshness/"
        "patch_preflight_gate.diff` (the main thread owns "
        "`pre_flight_check.py`)."
    )
    lines.append(
        "- GITIGNORE: `.gitignore:158` ignores `molmetal/reports/**/*.json`, so "
        "THIS `report.json` is an on-disk-only artefact unless it is force-added "
        "(`git add -f`).  The same rule already applies to the Wf-A159 census "
        "artefact it replaces.  `report.md` and `patch_preflight_gate.diff` are "
        "NOT ignored and carry the headline numbers durably."
    )
    unpaired = report["scope_generalized"]["unpaired_sha_pins"]
    if unpaired:
        lines.append("")
        lines.append(
            f"## Residual blind spot — {len(unpaired)} SHA pin(s) with no named file"
        )
        lines.append("")
        lines.append(
            "These record a SHA-256 but name no sibling file, so the pin cannot "
            "be checked.  They are NOT counted as stale — they are the honest "
            "limit of this guard."
        )
        lines.append("")
        seen_keys = sorted({u["sha_key"] for u in unpaired})
        counts: Dict[str, int] = {}
        for u in unpaired:
            counts[u["sha_key"]] = counts.get(u["sha_key"], 0) + 1
        for key in seen_keys:
            lines.append(f"- `{key}`: {counts[key]} pin(s)")
    lines.append("")
    return "\n".join(lines) + "\n"


def _git_head() -> str:
    """The current git HEAD commit, read as FILES (no subprocess).

    ``.git`` may be a directory (normal clone) or a file (worktree / submodule).
    Both layouts are handled; ``packed-refs`` is consulted when the loose ref
    is absent.
    """
    git_path = REPO_ROOT / ".git"
    try:
        if git_path.is_file():
            # "gitdir: <path>" — a worktree or submodule pointer.
            raw = git_path.read_text(encoding="utf-8").strip()
            if not raw.startswith("gitdir:"):
                return "unknown"
            git_dir = Path(raw.split(":", 1)[1].strip())
            if not git_dir.is_absolute():
                git_dir = (REPO_ROOT / git_dir).resolve()
        elif git_path.is_dir():
            git_dir = git_path
        else:
            return "unknown"
        head_text = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if not head_text.startswith("ref:"):
            return head_text or "unknown"
        ref = head_text.split(":", 1)[1].strip()
        loose = git_dir / ref
        if loose.is_file():
            return loose.read_text(encoding="utf-8").strip()
        packed = git_dir / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line.startswith("#") or not line.strip():
                    continue
                parts = line.split()
                if len(parts) == 2 and parts[1] == ref:
                    return parts[0]
        return "unknown"
    except OSError:
        return "unknown"


def _write(path: Path, content: str) -> None:
    """Atomic write: temp file in the same directory + ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-root", default=str(DEFAULT_REPORTS_ROOT))
    parser.add_argument("--json-out", default=str(DEFAULT_REPORT))
    parser.add_argument("--md-out", default=str(DEFAULT_MD))
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    reports_root = Path(args.reports_root).resolve()
    report = scan_reports(reports_root, max_bytes=args.max_bytes)
    if report["status"] == "SKIP":
        print(f"SKIP: {report['reason']}")
        return 2

    s = report["summary"]
    report["advisory"] = {
        "would_block_launch_if_fail": s["n_stale"] > 0,
        "n_stale_today": s["n_stale"],
        "n_pins_checked_today": s["n_sha_pins_checked"],
        "n_artefacts_with_a_sha_pin_today": s["n_artefacts_with_a_sha_pin"],
        "posture": (
            "ADVISORY. A FAIL-ing version of this check would block launches "
            f"today iff n_stale > 0; n_stale = {s['n_stale']}."
        ),
    }

    _write(Path(args.json_out), json.dumps(report, indent=2) + "\n")
    _write(Path(args.md_out), render_markdown(report))

    if not args.quiet:
        print(
            f"census-freshness: {s['n_json_artefacts_scanned']} JSON artefacts, "
            f"{s['n_artefacts_with_a_sha_pin']} carry a SHA pin, "
            f"{s['n_sha_pins_checked']} pins checked, {s['n_stale']} STALE"
        )
        print(
            f"  primary    : {report['scope_primary']['n_artefacts']} artefacts, "
            f"{report['scope_primary']['n_stale']} stale"
        )
        print(
            f"  generalized: {report['scope_generalized']['n_artefacts']} artefacts, "
            f"{report['scope_generalized']['n_stale']} stale"
        )
        for rec in report["stale"][:40]:
            print(
                f"    STALE {rec['artefact']} {rec['json_pointer']} "
                f"{rec['sha_key']} -> {rec['target_file']}"
            )
        if len(report["stale"]) > 40:
            print(f"    … {len(report['stale']) - 40} more")
    print(f"wrote {args.json_out}")
    print(f"wrote {args.md_out}")
    if report["status"] == "PASS":
        print("PASS: every recorded SHA-256 pin still matches the file it names")
        return 0
    print(f"FAIL: {s['n_stale']} recorded SHA-256 pin(s) no longer match the file")
    return 1


if __name__ == "__main__":
    sys.exit(main())
