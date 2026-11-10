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

"""Wf-A156 -- ledger provenance guard.

WHAT THIS GUARDS
----------------
``molmetal/reports/run_ledger.jsonl`` is the append-only provenance record for
every metric value (AGENTS.md SS1 rule 1 / SS5.2 honest framing).  Each row
carries a ``source_json_sha256``: the content hash of the metric JSON at the
moment the row was written.  That hash is the ONLY content-level pointer in
the ledger, and it is the one that must be *provable* -- it has to name a state
of the file that this repository actually held, so a reader can reproduce the
row from git.

TWO EXISTING CHECKS CANNOT SEE AN UNPROVABLE HASH

* ``registry.py check`` validates the REGISTRY rows, never the ledger's.
* ``run_ledger.staleness()`` compares the LEDGER's hash with the REGISTRY's
  hash.  Two hashes that are wrong in the SAME way compare EQUAL, so a
  ledger row and a registry row that agree on a non-existent file version
  sail through both.

Only a comparison against GIT HISTORY finds it.

THE DEFECT THIS GUARD CATCHES
------------------------------
MEASURED 2026-11-01 by this script on the real ledger: 418 rows / 416
distinct ``(source_json, source_json_sha256)`` pairs / 398 distinct source
paths / 396 distinct metrics.  9 rows name a file version that is neither the
file's current content nor any committed version of it -- i.e. a state of the
file this repository has NEVER held.  All 9 are SUPERSEDED rows (a later row
for the same ``metric_id`` carries a provable hash), so no metric's live
provenance is broken; the bad rows are history that the append-only contract
can never erase.  That is why they are reported, counted, and NOT repaired by
an in-place edit.

VERDICTS (per row)
------------------
``on_disk``
    the recorded sha IS the file's current content on disk.  The provenance
    holds today.
``committed``
    the recorded sha is the content of some COMMITTED version of that path,
    but not the current one: the file legitimately changed AFTER the row was
    written.  This is HONEST HISTORY, not a defect (repair class ``c``).
``unprovable``
    the recorded sha matches NEITHER the file on disk NOR any committed
    version of it.  THIS IS THE DEFECT.

REPAIR CLASSES (what to do about an ``unprovable`` row)
-------------------------------------------------------
``a_documentation_note``
    the row is SUPERSEDED -- a later row for the same ``metric_id`` carries a
    provable hash, so the metric's LIVE provenance is intact.  The honest
    repair is a DOCUMENTATION NOTE.  The ledger is append-only: editing or
    deleting the bad row would rewrite history.
``b_append_corrected_row``
    the row is the LATEST row for its metric: the live provenance IS broken
    and the recorded hash is wrong.  The repair is an APPEND of one corrected
    row through ``run_ledger.record_run`` (the sanctioned writer).  Never an
    in-place edit.
``c_changed_since_record``
    the row's hash is provable against git history; the file changed
    legitimately after the record.  Not a defect -- reported so the reader
    can tell "changed since" from "never existed".

TWO INDEPENDENT GIT RESOLVERS, AND WHY BOTH
--------------------------------------------
A ledger row is provable when its hash is in the set of content hashes of
that path's committed versions.  That set is computed TWICE, by two methods
that do not share code:

``bulk``
    ONE ``git log --all --format= --raw --no-abbrev --no-renames -m -- <paths>``
    call yields the blob sha of every committed version of every path, and ONE
    ``git cat-file --batch`` call turns those blobs into content hashes.
``per_path``
    ``git rev-list --all -- <path>`` per DISTINCT path (cached) plus
    ``git cat-file --batch`` over ``<commit>:<path>`` revspecs.

THE ``-m`` IS LOAD-BEARING.  ``git log --raw`` prints NO diff at all for a
MERGE commit, so a content version that a merge introduced (a conflict
resolution, a file merged in from a side branch) is INVISIBLE without
``-m``.  MEASURED on this repo: 398 ledger paths, the no-``-m`` raw log sees
700 blobs, the ``-m`` raw log sees 754 -- and the 754 agree EXACTLY with the
per-path ``rev-list`` method.  A resolver that omits ``-m`` therefore reports
54 paths as having fewer committed versions than they have, and calls a
provable row "unprovable".  The guard runs BOTH resolvers and exits 2 if they
disagree, so this class of false positive cannot hide.

EXIT CODES
----------
0  every row is provable (``on_disk`` or ``committed``)
1  at least one unprovable row (ledger and/or registry)
2  SKIP / ERROR -- missing ledger, git failure, corrupt row, or the two
   resolvers disagreeing

CPU-ONLY.  No torch, no GPU, no network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = REPO_ROOT / "molmetal" / "reports" / "run_ledger.jsonl"
REGISTRY_METRICS_PATH = REPO_ROOT / "registry" / "metrics.yaml"

EXIT_OK = 0
EXIT_DEFECT = 1
EXIT_ERROR = 2

VERDICT_ON_DISK = "on_disk"
VERDICT_COMMITTED = "committed"
VERDICT_UNPROVABLE = "unprovable"

REPAIR_DOC_NOTE = "a_documentation_note"
REPAIR_APPEND = "b_append_corrected_row"
REPAIR_CHANGED_SINCE = "c_changed_since_record"

ZERO_SHA = "0" * 40
SCHEMA = "wfa156.ledger_provenance.v1"

_GIT_TIMEOUT = 900


class GuardError(RuntimeError):
    """The guard cannot render a verdict (missing input, git failure, ...)."""


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path, *, chunk: int = 1 << 20) -> Optional[str]:
    """sha256 of a file, or None when it does not exist / cannot be read."""
    try:
        if not path.is_file():
            return None
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(chunk), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


def resolve_source_json(source_json: str, repo_root: Path = REPO_ROOT) -> Path:
    """Resolve a ledger/registry ``source_json`` string to an absolute path.

    Mirrors ``run_ledger._resolve_source_json`` on purpose: an absolute path
    is taken as-is, a repo-relative path (``metrics/by_metric/x.json``) is
    taken under the repo root, and a bare stem (``x``) resolves under
    ``metrics/by_metric/``.
    """
    p = Path(source_json)
    if p.is_absolute():
        return p
    if "/" in source_json or p.suffix == ".json":
        return repo_root / p
    return repo_root / "metrics" / "by_metric" / f"{source_json}.json"


def _repo_relative(repo_root: Path, path: Path) -> Optional[str]:
    """POSIX-style path of ``path`` relative to ``repo_root``, or None."""
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except (ValueError, OSError):
        return None


def _git(
    repo_root: Path,
    args: Sequence[str],
    *,
    input_bytes: Optional[bytes] = None,
    timeout: int = _GIT_TIMEOUT,
) -> bytes:
    """Run one git command in ``repo_root``; raise GuardError on failure."""
    cmd = ["git", "-C", str(repo_root)] + list(args)
    try:
        proc = subprocess.run(
            cmd,
            input=input_bytes,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise GuardError(f"git {' '.join(list(args)[:2])} could not run: {exc}") from exc
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace").strip().splitlines()
        raise GuardError(
            f"git {' '.join(list(args)[:2])} failed (rc={proc.returncode}): "
            f"{err[-1][:300] if err else 'no stderr'}"
        )
    return proc.stdout


def _cat_file_content_shas(
    repo_root: Path, revspecs: Sequence[str]
) -> Tuple[Dict[str, str], List[str]]:
    """sha256 of the CONTENT of each revspec, in ONE ``git cat-file --batch``.

    Returns ``(mapping, missing)``.  ``missing`` holds the revspecs git could
    not resolve (e.g. ``<commit>:<path>`` for a path deleted at that commit);
    they contribute no content version and are reported, not swallowed.
    """
    if not revspecs:
        return {}, []
    payload = ("\n".join(revspecs) + "\n").encode("utf-8")
    out = _git(repo_root, ["cat-file", "--batch"], input_bytes=payload)
    mapping: Dict[str, str] = {}
    missing: List[str] = []
    pos = 0
    for want in revspecs:
        nl = out.find(b"\n", pos)
        if nl < 0:
            raise GuardError(f"truncated git cat-file --batch output at {want!r}")
        header = out[pos:nl].decode("utf-8", "replace").split()
        if len(header) < 3:
            missing.append(want)
            pos = nl + 1
            continue
        try:
            size = int(header[2])
        except ValueError as exc:
            raise GuardError(
                f"git cat-file --batch returned an unparsable header for "
                f"{want!r}: {header!r}"
            ) from exc
        body = out[nl + 1 : nl + 1 + size]
        if len(body) != size:
            raise GuardError(f"truncated git cat-file --batch body for {want!r}")
        mapping[want] = hashlib.sha256(body).hexdigest()
        pos = nl + 1 + size + 1  # each body is followed by a newline
    return mapping, missing


# --------------------------------------------------------------------------
# resolver 1 -- bulk: one git log --raw -m + one git cat-file --batch
# --------------------------------------------------------------------------
def committed_content_shas_bulk(
    repo_root: Path, git_paths: Sequence[str]
) -> Dict[str, Set[str]]:
    """``path -> {content sha256}`` for every committed version of the path.

    ONE ``git log --all --format= --raw --no-abbrev --no-renames -m -- <paths>``
    call yields the blob sha of every committed version of every path; ONE
    ``git cat-file --batch`` call turns those blobs into content hashes.

    ``-m`` is load-bearing: without it git prints no diff for a MERGE commit,
    so a content version introduced by a merge is invisible.  See the module
    docstring for the measurement.

    API-SHAPE CONTRACT (2026-11-04, Wf-A269 wave-23): this function returns
    an entry for EVERY REQUESTED path.  A path with no committed history maps
    to an EMPTY set, meaning "this path has no committed version" — NOT
    "this path was not asked about".  This module was the ONLY one of the
    five ledger-provenance implementations that honoured the contract; the
    other four now do too, so the five can be compared directly.  A naive
    comparison used to manufacture a 4-path "disagreement" that was pure API
    shape and said nothing about git.
    """
    if not git_paths:
        return {}
    raw = _git(
        repo_root,
        [
            "log",
            "--all",
            "--format=",
            "--raw",
            "--no-abbrev",
            "--no-renames",
            "-m",
            "--",
        ]
        + list(git_paths),
    )
    blobs: Dict[str, Set[str]] = {}
    for line in raw.decode("utf-8", "replace").splitlines():
        if not line.startswith(":"):
            continue
        meta, _, path = line.partition("\t")
        parts = meta.split()
        if len(parts) < 4 or not path:
            continue
        new_sha = parts[3]
        if len(new_sha) != 40 or set(new_sha) == {"0"}:
            continue  # a deletion, not a version
        blobs.setdefault(path, set()).add(new_sha)
    all_blobs = sorted({b for v in blobs.values() for b in v})
    contents, _missing = _cat_file_content_shas(repo_root, all_blobs)
    out: Dict[str, Set[str]] = {}
    for path, shas in blobs.items():
        out[path] = {contents[b] for b in shas if b in contents}
    for path in git_paths:
        out.setdefault(path, set())
    return out


# --------------------------------------------------------------------------
# resolver 2 -- per path: git rev-list --all -- <path> + <commit>:<path>
# --------------------------------------------------------------------------
def committed_content_shas_per_path(
    repo_root: Path, git_paths: Sequence[str]
) -> Dict[str, Set[str]]:
    """``path -> {content sha256}`` via one ``git rev-list`` per DISTINCT path.

    Independent of the bulk resolver: it never looks at a diff, only at the
    file's content at each commit that touched it.  This is the method the
    Wf-A156 brief specifies, and it is the ground truth the bulk resolver is
    checked against.

    API-SHAPE CONTRACT (2026-11-04, Wf-A269 wave-23): an entry for EVERY
    REQUESTED path (empty set = "no committed version"), same as
    :func:`committed_content_shas_bulk`.
    """
    if not git_paths:
        return {}
    revspecs: List[str] = []
    per_path_commits: Dict[str, List[str]] = {}
    for path in git_paths:
        out = _git(repo_root, ["rev-list", "--all", "--", path]).decode("utf-8", "replace")
        commits = [c for c in out.split() if c]
        per_path_commits[path] = commits
        revspecs.extend(f"{c}:{path}" for c in commits)
    contents, _missing = _cat_file_content_shas(repo_root, revspecs)
    out: Dict[str, Set[str]] = {}
    for path, commits in per_path_commits.items():
        out[path] = {
            contents[f"{c}:{path}"]
            for c in commits
            if f"{c}:{path}" in contents
        }
    return out


# --------------------------------------------------------------------------
# ledger + registry readers
# --------------------------------------------------------------------------
def load_rows(ledger_path: Path) -> List[Tuple[int, Dict[str, Any]]]:
    """Read the ledger as ``(line_number, entry)`` pairs, in file order.

    A corrupt line raises GuardError: a ledger that cannot be parsed is a
    real problem, not noise.
    """
    if not ledger_path.is_file():
        raise GuardError(f"ledger not found: {ledger_path}")
    rows: List[Tuple[int, Dict[str, Any]]] = []
    text = ledger_path.read_text(encoding="utf-8")
    for lineno, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            entry = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise GuardError(f"corrupt JSON on {ledger_path}:{lineno}: {exc}") from exc
        if not isinstance(entry, dict):
            raise GuardError(f"ledger row {lineno} is not a JSON object")
        rows.append((lineno, entry))
    return rows


def load_registry_rows(registry_path: Path) -> List[Tuple[str, str, Optional[str]]]:
    """``(metric_id, source_json, source_json_sha256)`` from registry/metrics.yaml."""
    try:
        import yaml  # type: ignore
    except ImportError as exc:  # pragma: no cover - PyYAML is a hard dep
        raise GuardError(f"PyYAML is required for the registry check: {exc}") from exc
    if not registry_path.is_file():
        raise GuardError(f"registry not found: {registry_path}")
    data = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    metrics = data.get("metrics") or {}
    out: List[Tuple[str, str, Optional[str]]] = []
    for mid, rec in metrics.items():
        rec = rec or {}
        out.append((str(mid), str(rec.get("source_json") or ""), rec.get("source_json_sha256")))
    return out


# --------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------
def _verdict_for(sha: Optional[str], disk_sha: Optional[str], committed: Set[str]) -> str:
    if not sha:
        return VERDICT_UNPROVABLE
    if disk_sha is not None and sha == disk_sha:
        return VERDICT_ON_DISK
    if sha in committed:
        return VERDICT_COMMITTED
    return VERDICT_UNPROVABLE


def _row_state(index: int, metric_id: str, last_index: Dict[str, int]) -> str:
    """``latest`` when the row is the metric's newest ledger row, else ``superseded``.

    The ledger is append-only, so the LAST row for a metric in file order is
    its current one -- the same rule ``run_ledger.latest()`` applies.
    """
    return "latest" if last_index.get(metric_id) == index else "superseded"


def _repair_class(verdict: str, row_state: str) -> Optional[str]:
    if verdict == VERDICT_UNPROVABLE:
        return REPAIR_APPEND if row_state == "latest" else REPAIR_DOC_NOTE
    if verdict == VERDICT_COMMITTED:
        return REPAIR_CHANGED_SINCE
    return None


def _suggested_append(
    entry: Dict[str, Any], disk_sha: Optional[str], committed: Set[str]
) -> Dict[str, Any]:
    """The exact ``record_run`` call that repairs an OPEN unprovable row.

    The ledger is append-only: the repair is an APPEND, never an in-place
    edit.  The hash to carry is the file's CURRENT content -- but only if
    that content is itself a committed version, otherwise the appended row
    would re-create the very defect this guard reports.
    """
    src = str(entry.get("source_json") or "")
    sha: Optional[str] = None
    note = ""
    if disk_sha is not None and disk_sha in committed:
        sha = disk_sha
        note = (
            "the current on-disk content is a committed version of this path, "
            "so it is fully reproducible from git"
        )
    elif committed:
        sha = None
        note = (
            "the current on-disk content is NOT a committed version of this "
            "path; commit the source file first, then re-point"
        )
    else:
        note = (
            "this path has NO committed version at all; commit it before "
            "re-pointing the metric"
        )
    return {
        "record_run_kwargs": {
            "metric_id": str(entry.get("metric_id") or ""),
            "value": entry.get("value"),
            "source_json": src,
            "source_json_sha256": sha,
            "data_source": (entry.get("extra") or {}).get("data_source"),
            "timestamp": entry.get("timestamp"),
            "seed": True,
        },
        "note": note,
    }


def analyse_ledger(
    ledger_path: Path,
    repo_root: Path,
    *,
    cross_check: bool = True,
) -> Dict[str, Any]:
    """Classify every ledger row against git history.  Returns the report."""
    rows = load_rows(ledger_path)

    # ---- the set of paths we need history for -------------------------
    disk_paths: Dict[str, Path] = {}
    git_paths: Dict[str, str] = {}  # raw source_json -> repo-relative git path
    for _lineno, entry in rows:
        src = str(entry.get("source_json") or "")
        if not src or src in disk_paths:
            continue
        resolved = resolve_source_json(src, repo_root=repo_root)
        disk_paths[src] = resolved
        rel = _repo_relative(repo_root, resolved)
        if rel is not None:
            git_paths[src] = rel

    unique_git_paths = sorted(set(git_paths.values()))
    bulk = committed_content_shas_bulk(repo_root, unique_git_paths)
    per_path: Dict[str, Set[str]] = {}
    resolver_agreement: Dict[str, Any] = {
        "bulk_paths": len(unique_git_paths),
        "per_path_paths": 0,
        "cross_check": bool(cross_check),
        "agree": True,
        "disagreements": [],
    }
    if cross_check:
        per_path = committed_content_shas_per_path(repo_root, unique_git_paths)
        resolver_agreement["per_path_paths"] = len(unique_git_paths)
        disagreements = []
        for path in unique_git_paths:
            b = bulk.get(path, set())
            p = per_path.get(path, set())
            if b != p:
                disagreements.append(
                    {
                        "path": path,
                        "bulk_only": sorted(b - p),
                        "per_path_only": sorted(p - b),
                    }
                )
        resolver_agreement["agree"] = not disagreements
        resolver_agreement["disagreements"] = disagreements

    # ---- the authoritative set: the UNION of both resolvers ------------
    # A row is provable if EITHER resolver can prove it.  Using the union
    # makes the guard conservative: it can only miss a defect, never invent
    # one.  When the two resolvers agree (the normal case) the union equals
    # each of them.
    committed_by_src: Dict[str, Set[str]] = {}
    for src, git_path in git_paths.items():
        merged = set(bulk.get(git_path, set()))
        if cross_check:
            merged |= set(per_path.get(git_path, set()))
        committed_by_src[src] = merged

    disk_sha_by_src = {src: sha256_file(p) for src, p in disk_paths.items()}

    # ---- ledger order: the LAST row for a metric is its current one -----
    last_index: Dict[str, int] = {}
    for idx, (_lineno, entry) in enumerate(rows):
        last_index[str(entry.get("metric_id"))] = idx

    row_records: List[Dict[str, Any]] = []
    findings: List[Dict[str, Any]] = []
    unprovable: List[Tuple[int, Dict[str, Any]]] = []
    counts = {
        "on_disk": 0,
        "committed": 0,
        "unprovable": 0,
        "open_unprovable": 0,
        "superseded_unprovable": 0,
    }
    distinct_pairs = set()
    for idx, (lineno, entry) in enumerate(rows):
        src = str(entry.get("source_json") or "")
        sha = entry.get("source_json_sha256")
        sha = str(sha) if sha else None
        if sha:
            distinct_pairs.add((src, sha))
        committed = committed_by_src.get(src, set())
        disk_sha = disk_sha_by_src.get(src)
        verdict = _verdict_for(sha, disk_sha, committed)
        counts[verdict] += 1
        metric_id = str(entry.get("metric_id"))
        state = _row_state(idx, metric_id, last_index)
        repair = _repair_class(verdict, state)
        if verdict == VERDICT_UNPROVABLE:
            if state == "latest":
                counts["open_unprovable"] += 1
            else:
                counts["superseded_unprovable"] += 1
        record = {
            "line": lineno,
            "metric_id": metric_id,
            "source_json": src,
            "source_json_sha256": sha,
            "verdict": verdict,
            "row_state": state,
            "repair_class": repair,
            "on_disk_sha256": disk_sha,
            "committed_versions": len(committed),
            "path_tracked": bool(committed),
            "git_sha": entry.get("git_sha"),
            "timestamp": entry.get("timestamp"),
            "value": entry.get("value"),
        }
        row_records.append(record)
        if verdict == VERDICT_UNPROVABLE:
            unprovable.append((idx, record))

    # Second pass: a finding needs the metric's LATEST row, which is only
    # known once every row has been recorded.
    for idx, record in unprovable:
        metric_id = record["metric_id"]
        finding = dict(record)
        # evidence that the metric's LIVE provenance is intact
        latest = row_records[last_index[metric_id]] if metric_id in last_index else None
        finding["latest_row_line"] = latest["line"] if latest else None
        finding["latest_row_verdict"] = latest["verdict"] if latest else None
        finding["latest_row_sha256"] = latest["source_json_sha256"] if latest else None
        if record["row_state"] == "latest":
            finding["suggested_repair"] = _suggested_append(
                rows[idx][1], disk_sha_by_src.get(record["source_json"]), committed_by_src.get(record["source_json"], set())
            )
        else:
            finding["suggested_repair"] = {
                "record_run_kwargs": None,
                "note": (
                    "DO NOT EDIT THE LEDGER.  A later row for this metric "
                    "already carries a provable hash, so the metric's live "
                    "provenance is intact; record this as a documentation "
                    "note instead."
                ),
            }
        findings.append(finding)

    counts["ledger_rows"] = len(rows)
    counts["distinct_pairs"] = len(distinct_pairs)
    counts["distinct_source_paths"] = len(disk_paths)
    counts["distinct_metrics"] = len(last_index)

    status = "FAIL" if counts["unprovable"] else "PASS"
    return {
        "schema": SCHEMA,
        "generated_at": _now_iso(),
        "ledger_path": str(ledger_path),
        "repo_root": str(repo_root),
        "counts": counts,
        "resolvers": resolver_agreement,
        "rows": row_records,
        "findings": findings,
        "status": status,
    }


def analyse_registry(
    registry_path: Path, repo_root: Path
) -> Dict[str, Any]:
    """Classify every registry row's ``source_json_sha256`` the same way.

    READ-ONLY: this function never writes to ``registry/**``.
    """
    entries = load_registry_rows(registry_path)
    git_paths: Dict[str, str] = {}
    for _mid, src, _sha in entries:
        if not src or src in git_paths:
            continue
        rel = _repo_relative(repo_root, resolve_source_json(src, repo_root=repo_root))
        if rel is not None:
            git_paths[src] = rel
    unique = sorted(set(git_paths.values()))
    bulk = committed_content_shas_bulk(repo_root, unique)
    per_path = committed_content_shas_per_path(repo_root, unique)

    counts = {"on_disk": 0, "committed": 0, "unprovable": 0, "no_sha": 0}
    findings: List[Dict[str, Any]] = []
    for mid, src, sha in entries:
        if not sha:
            counts["no_sha"] += 1
            continue
        git_path = git_paths.get(src)
        committed = set(bulk.get(git_path, set())) | set(per_path.get(git_path, set()))
        disk_sha = sha256_file(resolve_source_json(src, repo_root=repo_root))
        verdict = _verdict_for(str(sha), disk_sha, committed)
        counts[verdict] += 1
        if verdict == VERDICT_UNPROVABLE:
            findings.append(
                {
                    "metric_id": mid,
                    "source_json": src,
                    "source_json_sha256": str(sha),
                    "on_disk_sha256": disk_sha,
                    "committed_versions": len(committed),
                }
            )
    return {
        "registry_path": str(registry_path),
        "counts": counts,
        "findings": findings,
        "status": "FAIL" if counts["unprovable"] else "PASS",
    }


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
def _short(sha: Optional[str], n: int = 16) -> str:
    if not sha:
        return "<none>"
    return sha[:n] + "..." if len(sha) > n else sha


def render_text(report: Dict[str, Any], registry: Optional[Dict[str, Any]]) -> str:
    c = report["counts"]
    r = report["resolvers"]
    lines: List[str] = []
    lines.append("Wf-A156 ledger provenance guard")
    lines.append(f"  ledger     : {report['ledger_path']}")
    lines.append(f"  repo root  : {report['repo_root']}")
    lines.append(
        f"  rows       : {c['ledger_rows']}  "
        f"(distinct pairs {c['distinct_pairs']}, "
        f"paths {c['distinct_source_paths']}, "
        f"metrics {c['distinct_metrics']})"
    )
    if r["cross_check"]:
        verdict_txt = "AGREE" if r["agree"] else "DISAGREE"
        lines.append(
            f"  resolvers  : bulk(1 git call) + per-path(git rev-list) "
            f"{verdict_txt} on {r['bulk_paths']} path(s)"
        )
        if not r["agree"]:
            for d in r["disagreements"]:
                lines.append(
                    f"      DISAGREEMENT {d['path']}: "
                    f"bulk_only={len(d['bulk_only'])} "
                    f"per_path_only={len(d['per_path_only'])}"
                )
    else:
        lines.append("  resolvers  : bulk only (cross-check disabled)")
    lines.append(
        f"  verdicts   : on_disk={c['on_disk']}  committed={c['committed']}  "
        f"unprovable={c['unprovable']}"
    )
    lines.append("      on_disk    = the recorded sha IS the file's current content")
    lines.append(
        "      committed  = the recorded sha is a COMMITTED version, not the"
        " current one"
    )
    lines.append(
        "                   (the file legitimately changed after the row) --"
        " honest history"
    )
    lines.append(
        "      unprovable = the recorded sha matches NEITHER disk NOR any"
        " commit -- THE DEFECT"
    )
    findings = report["findings"]
    lines.append("")
    if not findings:
        lines.append("UNPROVABLE ROWS: 0")
    else:
        lines.append(f"UNPROVABLE ROWS ({len(findings)}):")
        for f in findings:
            lines.append(
                f"  [{f['repair_class']}] line {f['line']}  metric={f['metric_id']}"
            )
            lines.append(f"      source_json : {f['source_json']}")
            lines.append(f"      recorded    : {f['source_json_sha256']}")
            lines.append(f"      on disk     : {_short(f['on_disk_sha256'])}")
            lines.append(
                f"      committed   : {f['committed_versions']} version(s) of that"
                " path exist in git history"
            )
            lines.append(
                f"      row state   : {f['row_state'].upper()}  "
                f"(latest row for this metric: line {f['latest_row_line']}, "
                f"verdict {f['latest_row_verdict']})"
            )
            lines.append(f"      repair      : {f['suggested_repair']['note']}")
            if f["suggested_repair"].get("record_run_kwargs"):
                lines.append(
                    "      record_run  : "
                    + json.dumps(
                        f["suggested_repair"]["record_run_kwargs"],
                        sort_keys=True,
                        ensure_ascii=False,
                    )
                )
    if registry is not None:
        rc = registry["counts"]
        lines.append("")
        lines.append(
            f"REGISTRY {registry['registry_path']} (read-only): "
            f"rows={sum(rc.values())} on_disk={rc['on_disk']} "
            f"committed={rc['committed']} unprovable={rc['unprovable']} "
            f"no_sha={rc['no_sha']}"
        )
        for f in registry["findings"]:
            lines.append(
                f"      UNPROVABLE {f['metric_id']}  {f['source_json']}  "
                f"{f['source_json_sha256']}"
            )
    lines.append("")
    lines.append(
        f"LEDGER PROVENANCE: {report['status']} "
        f"({c['unprovable']} unprovable row(s); "
        f"{c['open_unprovable']} open / {c['superseded_unprovable']} superseded)"
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _atomic_write(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` via the repo's atomic-write helper."""
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from wf386a3_atomic_write import atomic_write_file  # type: ignore

        atomic_write_file(path, content)
        return
    except Exception:  # pragma: no cover - fall back to a plain write
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="wfa156_ledger_provenance_guard",
        description=(
            "Guard the run ledger's source_json_sha256 provenance against git "
            "history."
        ),
    )
    p.add_argument("--ledger", type=Path, default=LEDGER_PATH, help="ledger jsonl path")
    p.add_argument("--repo-root", type=Path, default=REPO_ROOT, help="repository root")
    p.add_argument(
        "--registry",
        type=Path,
        default=REGISTRY_METRICS_PATH,
        help="registry metrics.yaml to check (read-only)",
    )
    p.add_argument(
        "--no-registry",
        action="store_true",
        help="skip the registry check",
    )
    p.add_argument(
        "--no-cross-check",
        action="store_true",
        help=(
            "skip the independent per-path resolver (faster, but the bulk "
            "resolver's merge-commit blind spot is then unverified)"
        ),
    )
    p.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="write the machine-readable JSON summary to this path",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="print the JSON summary to stdout instead of the text report",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    try:
        report = analyse_ledger(
            args.ledger.resolve(),
            repo_root,
            cross_check=not args.no_cross_check,
        )
    except GuardError as exc:
        print(f"LEDGER PROVENANCE: SKIP/ERROR -- {exc}", file=sys.stderr)
        return EXIT_ERROR
    except OSError as exc:
        print(f"LEDGER PROVENANCE: SKIP/ERROR -- {exc}", file=sys.stderr)
        return EXIT_ERROR

    if not report["resolvers"]["agree"]:
        print(
            "LEDGER PROVENANCE: SKIP/ERROR -- the two git-history resolvers "
            "disagree; the guard cannot render a trustworthy verdict",
            file=sys.stderr,
        )
        for d in report["resolvers"]["disagreements"]:
            print(f"  {d['path']}", file=sys.stderr)
        return EXIT_ERROR

    registry_report: Optional[Dict[str, Any]] = None
    if not args.no_registry:
        try:
            registry_report = analyse_registry(args.registry.resolve(), repo_root)
        except GuardError as exc:
            print(f"registry check SKIPPED: {exc}", file=sys.stderr)
            registry_report = None

    payload = dict(report)
    payload["registry"] = registry_report
    if registry_report is not None and registry_report["counts"]["unprovable"]:
        payload["status"] = "FAIL"

    if args.out_json is not None:
        _atomic_write(
            args.out_json,
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        )

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print(render_text(report, registry_report))

    n_unprovable = report["counts"]["unprovable"]
    if registry_report is not None:
        n_unprovable += registry_report["counts"]["unprovable"]
    return EXIT_DEFECT if n_unprovable else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
