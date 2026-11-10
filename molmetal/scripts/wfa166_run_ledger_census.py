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

"""Wf-A166 -- the run-ledger provenance CENSUS.

WHAT THIS IS
------------
``molmetal/reports/run_ledger.jsonl`` is the append-only provenance record
(one row per recorded metric value).  This script enumerates EVERY row and
classifies it on seven independent axes, so that the ledger's ``git_sha``
column can be described by what it IS rather than by what it was meant to be.

THE AXES (one verdict per row, each with the row index and the file:line it
came from)

A. ``sha_matches_disk``            -- the row's ``source_json_sha256`` equals
                                       the sha256 of the file on disk NOW.
B. ``sha_matches_a_committed_version`` -- the row's sha256 equals the CONTENT
                                       of some committed version of that path
                                       (git history), i.e. the row is provable
                                       against git.
C. ``commit_exists``               -- the row's ``git_sha`` resolves to a
                                       commit object.
D. ``commit_touches_source_json``  -- the named commit's changed-file list
                                       contains the row's ``source_json``.
E. ``commit_reachable``            -- the named commit is contained in some
                                       branch or tag.
F. ``git_sha_is_a_bulk_constant``  -- the row's git_sha equals the modal
                                       (majority) git_sha of the whole ledger.
G. ``config_sha`` / ``data_sha`` resolvability -- the referenced file exists
                                       and its content hash still equals the
                                       recorded hash.

TWO INDEPENDENT GIT RESOLVERS FOR AXIS B
-----------------------------------------
The set of committed content versions of a path is computed TWICE, by two
methods that share no code:

``bulk``      ONE ``git log --all --format= --raw --no-abbrev --no-renames -m
              -- <paths>`` (the ``-m`` is load-bearing: ``git log --raw``
              prints no diff for a MERGE commit, so a merge-introduced
              version would be invisible) plus ONE ``git cat-file --batch``.
``per_path``  ``git rev-list --all -- <path>`` per distinct path plus ONE
              ``git cat-file --batch`` over ``<commit>:<path>`` revspecs.

If the two resolvers disagree on any path the script reports
``resolver_agreement = "DISAGREE"`` and exits 2 -- a census that rests on a
resolver which is known to be incomplete is not a census.

READ-ONLY.  CPU-only.  No torch, no GPU, no network, no writes to the ledger.

EXIT CODES
----------
0  census written
2  a git resolver failed or the two resolvers disagreed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = REPO_ROOT / "molmetal" / "reports" / "run_ledger.jsonl"
BY_METRIC_DIR = REPO_ROOT / "metrics" / "by_metric"
PROTOCOLS_PATH = REPO_ROOT / "registry" / "protocols.yaml"
OUT_DIR = REPO_ROOT / "molmetal" / "reports" / "wfa166_run_ledger_census"

_ZERO_SHA = "0" * 40

#: The COLUMN CONTRACT -- what each ledger column MEANS, read off the WRITER
#: (``molmetal/scripts/run_ledger.py``), its tests
#: (``molmetal/tests/test_run_ledger.py``) and the repo's own docs.
#: ``kind`` is ``"historical"`` (a snapshot of the run) or ``"live"`` (a
#: pointer into the current tree).
COLUMN_CONTRACT: List[Dict[str, Any]] = [
    {
        "column": "metric_id",
        "written_at": "molmetal/scripts/run_ledger.py:214",
        "producer": "the caller's metric id; the seed path takes the registry/metrics.yaml keys (run_ledger.py:627)",
        "kind": "live pointer",
        "validated_by": "run_ledger.py:305-326 (staleness() joins the ledger to the registry rows on this key)",
        "notes": "the only join key between ledger and registry",
    },
    {
        "column": "value",
        "written_at": "molmetal/scripts/run_ledger.py:215",
        "producer": "the recorded number; the seed path copies the registry row's value verbatim (run_ledger.py:642)",
        "kind": "historical",
        "validated_by": "run_ledger.py:322 (_same_value) -- decides the 'reformatted' vs 'stale' split",
        "notes": "the payload; the only column that carries the number itself",
    },
    {
        "column": "source_json",
        "written_at": "molmetal/scripts/run_ledger.py:216",
        "producer": "a repo-relative path or a bare stem; the seed path copies the registry row's source_json (run_ledger.py:630)",
        "kind": "live pointer",
        "validated_by": "run_ledger.py:107-118 (_resolve_source_json) + :517-528 (uncommitted_provenance resolves it and compares shas)",
        "notes": "a path into the tree, resolved at READ time, not a frozen artifact",
    },
    {
        "column": "source_json_sha256",
        "written_at": "molmetal/scripts/run_ledger.py:217",
        "producer": "_sha256_file(src_path) (run_ledger.py:199-200); the seed path passes the REGISTRY's sha through verbatim (run_ledger.py:644)",
        "kind": "historical",
        "validated_by": "run_ledger.py:320 (vs the registry row) and run_ledger.py:523-528 (vs disk AND vs git history)",
        "notes": "the only CONTENT-level pointer; it is what the two pre-flight gates check",
    },
    {
        "column": "git_sha",
        "written_at": "molmetal/scripts/run_ledger.py:218",
        "producer": "_git_sha() == 'git rev-parse HEAD' AT THE MOMENT THE LEDGER ROW WAS WRITTEN (run_ledger.py:89-104, 201-202)",
        "kind": "historical",
        "validated_by": "NOTHING. It is read only to be echoed into a report (run_ledger.py:547, :874; wfa156_ledger_provenance_guard.py:576). registry.py never opens the ledger (0 hits).",
        "notes": "NOT the commit that produced the value; NOT the commit that touched the metric file",
    },
    {
        "column": "config_sha",
        "written_at": "molmetal/scripts/run_ledger.py:220",
        "producer": "sha256 of config_path (run_ledger.py:207-212); the seed path takes trainer_config from registry/protocols.yaml (run_ledger.py:639, 646)",
        "kind": "historical",
        "validated_by": "NOTHING in the repo. The PATH is NOT stored in the row, so the hash is not self-describing.",
        "notes": "measured here: 6/418 rows carry one; 0/6 still match the file the path resolves to now",
    },
    {
        "column": "data_sha",
        "written_at": "molmetal/scripts/run_ledger.py:219",
        "producer": "sha256 of the file resolved from the metric's data_source string (run_ledger.py:203-206, 121-143)",
        "kind": "historical",
        "validated_by": "NOTHING in the repo. The PATH is not stored in the row either (only echoed into extra.data_source, run_ledger.py:651).",
        "notes": "measured here: 30/418 rows carry one; 30/30 still match",
    },
    {
        "column": "timestamp",
        "written_at": "molmetal/scripts/run_ledger.py:223",
        "producer": "_now_utc() (run_ledger.py:75-76) UNLESS overridden; the seed path passes the SOURCE JSON's own timestamp, else the registry's metric_timestamp / last_refreshed (run_ledger.py:633-638)",
        "kind": "historical",
        "validated_by": "NOTHING. Echoed at run_ledger.py:317. Never parsed, never checked monotonic.",
        "notes": "for seed rows it dates the VALUE, not the row; the row's real write time is not recorded",
    },
    {
        "column": "protocol_id",
        "written_at": "molmetal/scripts/run_ledger.py:221",
        "producer": "the protocol_id kwarg (run_ledger.py:176) -- NO CALLER EVER PASSES IT",
        "kind": "UNDEFINED",
        "validated_by": "NOTHING. No producer, no consumer, no validator, no doc.",
        "notes": "null in 418/418 rows; registry/protocols.yaml has its own protocol rows that this column never links to",
    },
    {
        "column": "host",
        "written_at": "molmetal/scripts/run_ledger.py:222",
        "producer": "os.uname().nodename at write time",
        "kind": "historical",
        "validated_by": "NOTHING.",
        "notes": "'ZhaoLab' in 418/418 rows -- it cannot discriminate anything",
    },
    {
        "column": "seed",
        "written_at": "molmetal/scripts/run_ledger.py:224",
        "producer": "bool(seed) (run_ledger.py:181); True for seed_from_disk (run_ledger.py:648) and repoint_metric (run_ledger.py:741)",
        "kind": "historical",
        "validated_by": "NOTHING. seed_from_disk skips on metric_id, not on this flag (run_ledger.py:626-629).",
        "notes": "'this row is a baseline re-point, not a claimed run' -- but 408/418 rows carry it, so the flag cannot separate runs from baselines",
    },
    {
        "column": "extra",
        "written_at": "molmetal/scripts/run_ledger.py:226-227",
        "producer": "free-form dict, only written when non-empty; the seed path puts value_key + data_source in it (run_ledger.py:651)",
        "kind": "historical",
        "validated_by": "NOTHING. Read only to be echoed (run_ledger.py:547).",
        "notes": "the 9 A123 correction rows carry correction / committed_git_sha / committed_sha256 / correction_of_ledger_sha256 here",
    },
]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def git(*args: str, timeout: int = 300) -> Tuple[int, str]:
    """Run a git command in the repo root.  Returns (returncode, stdout)."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, f"__GIT_ERROR__ {type(exc).__name__}: {exc}"
    return proc.returncode, proc.stdout.decode("utf-8", "replace")


def sha256_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def resolve_source_json(source_json: str) -> Path:
    """Mirror of ``run_ledger._resolve_source_json``.

    A repo-relative path (contains "/" or ends in ".json") resolves against
    the repo root; a bare stem resolves under ``metrics/by_metric/``.
    """
    p = Path(source_json)
    if p.is_absolute():
        return p
    if "/" in source_json or p.suffix == ".json":
        return REPO_ROOT / p
    return BY_METRIC_DIR / f"{source_json}.json"


def resolve_data_source(data_source: Any) -> Optional[Path]:
    """Mirror of ``run_ledger._resolve_data_source``.

    The metric ``data_source`` is often prose; the leading token is tried
    against four known roots.  Never fabricates a path.
    """
    if not data_source:
        return None
    token = str(data_source).strip().split()[0].split("(")[0].strip().strip(",;")
    if not token:
        return None
    for cand in (
        Path(token),
        REPO_ROOT / token,
        REPO_ROOT / ".data" / "molmetal" / Path(token).name,
        REPO_ROOT / "molmetal" / "data" / Path(token).name,
    ):
        if cand.is_file():
            return cand
    return None


# ---------------------------------------------------------------------------
# git resolvers
# ---------------------------------------------------------------------------
def bulk_committed_content_shas(paths: Sequence[str]) -> Tuple[Dict[str, set], int]:
    """Resolver ``bulk``: committed CONTENT shas per path.

    ONE ``git log --all --format= --raw --no-abbrev --no-renames -m -- <paths>``
    yields the blob sha of every committed version of every path; ONE
    ``git cat-file --batch`` turns those blobs into content shas.
    """
    if not paths:
        return {}, 0
    rc, out = git(
        "log",
        "--all",
        "--format=",
        "--raw",
        "--no-abbrev",
        "--no-renames",
        "-m",
        "--",
        *paths,
    )
    if rc != 0:
        raise RuntimeError(f"git log --raw failed rc={rc}: {out[:400]}")
    blob_by_path: Dict[str, set] = {}
    for line in out.splitlines():
        if not line.startswith(":"):
            continue
        meta, _, path = line.partition("\t")
        parts = meta.split()
        if len(parts) < 4 or not path:
            continue
        new_sha = parts[3]
        if new_sha == _ZERO_SHA:
            continue  # a deletion, not a version
        blob_by_path.setdefault(path, set()).add(new_sha)
    all_blobs = sorted({b for v in blob_by_path.values() for b in v})
    content = batch_content_shas(all_blobs)
    by_path: Dict[str, set] = {}
    for path, blobs in blob_by_path.items():
        shas = {content[b] for b in blobs if b in content}
        shas.discard(None)  # type: ignore[arg-type]
        by_path[path] = shas
    # API-SHAPE CONTRACT (2026-11-04, Wf-A269 wave-23): an entry for EVERY
    # REQUESTED path.  A path with no committed history maps to an EMPTY
    # set, meaning "this path has no committed version" — NOT "this path was
    # not asked about".  The contract is normalised at the API so the five
    # ledger-provenance implementations can be compared without a
    # comparison-site compensation (a naive comparison used to manufacture a
    # 4-path "disagreement" that was pure API shape).
    for path in paths:
        by_path.setdefault(path, set())
    return by_path, len(all_blobs)


def batch_content_shas(revs: Sequence[str]) -> Dict[str, str]:
    """sha256 of the CONTENT of each git revspec, in ONE cat-file --batch."""
    if not revs:
        return {}
    # ``git cat-file --batch`` needs stdin, so it cannot go through git() and
    # is called directly with the revspec list on stdin.
    try:
        proc = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "cat-file", "--batch"],
            input=("\n".join(revs) + "\n").encode("utf-8"),
            capture_output=True,
            timeout=600,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"git cat-file --batch failed: {exc}") from exc
    data = proc.stdout
    out_map: Dict[str, str] = {}
    pos = 0
    for want in revs:
        nl = data.find(b"\n", pos)
        if nl < 0:
            break
        header = data[pos:nl].decode("utf-8", "replace").split()
        if len(header) < 3:
            break
        size = int(header[2])
        body = data[nl + 1 : nl + 1 + size]
        out_map[want] = sha256_bytes(body)
        pos = nl + 1 + size + 1
    return out_map


def per_path_committed_content_shas(paths: Sequence[str]) -> Dict[str, set]:
    """Resolver ``per_path``: committed CONTENT shas per path, independently.

    ``git rev-list --all -- <path>`` per distinct path, then
    ``git cat-file --batch`` over ``<commit>:<path>`` revspecs.
    """
    revspecs: List[str] = []
    commit_by_path: Dict[str, List[str]] = {}
    for path in paths:
        rc, out = git("rev-list", "--all", "--", path)
        if rc != 0:
            raise RuntimeError(f"git rev-list failed rc={rc} for {path}")
        commits = [c.strip() for c in out.splitlines() if c.strip()]
        commit_by_path[path] = commits
        for c in commits:
            revspecs.append(f"{c}:{path}")
    content = batch_content_shas(revspecs)
    by_path: Dict[str, set] = {}
    for path, commits in commit_by_path.items():
        shas = {content[f"{c}:{path}"] for c in commits if f"{c}:{path}" in content}
        by_path[path] = shas
    return by_path


def per_path_commit_content_map(
    paths: Sequence[str],
) -> Dict[str, Dict[str, str]]:
    """``path -> {commit: content_sha256}`` for every committed version.

    This is the RAW MATERIAL for the recoverability question: given a row's
    ``source_json_sha256``, the commits whose version of that path has
    exactly that content are the commits that could have produced the row's
    file state.
    """
    revspecs: List[str] = []
    commit_by_path: Dict[str, List[str]] = {}
    for path in paths:
        rc, out = git("rev-list", "--all", "--", path)
        if rc != 0:
            raise RuntimeError(f"git rev-list failed rc={rc} for {path}")
        commits = [c.strip() for c in out.splitlines() if c.strip()]
        commit_by_path[path] = commits
        for c in commits:
            revspecs.append(f"{c}:{path}")
    content = batch_content_shas(revspecs)
    out_map: Dict[str, Dict[str, str]] = {}
    for path, commits in commit_by_path.items():
        out_map[path] = {
            c: content[f"{c}:{path}"]
            for c in commits
            if f"{c}:{path}" in content
        }
    return out_map


def commit_dates(shas: Sequence[str]) -> Dict[str, str]:
    """``commit -> committer date (ISO-8601)`` for the given commits."""
    out: Dict[str, str] = {}
    if not shas:
        return out
    try:
        proc = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "log", "--format=%H%x1f%cI", *shas],
            capture_output=True,
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return out
    for line in proc.stdout.decode("utf-8", "replace").splitlines():
        parts = line.split("\x1f")
        if len(parts) >= 2:
            out[parts[0]] = parts[1]
    return out


def commit_changed_files(sha: str) -> Optional[List[str]]:
    """The changed-file list of a commit, or None if it does not exist."""
    rc, out = git("log", "-1", "--format=", "--name-only", "--no-renames", "-m", "--root", sha)
    if rc != 0:
        return None
    files = []
    for line in out.splitlines():
        line = line.strip()
        if line and not line.startswith(":"):
            files.append(line)
    return files


# ---------------------------------------------------------------------------
# the census
# ---------------------------------------------------------------------------
def build_census(
    ledger_path: Path = LEDGER_PATH,
    repo_root: Path = REPO_ROOT,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    raw_lines = ledger_path.read_text(encoding="utf-8").splitlines()
    line_no = 0
    for raw in raw_lines:
        if not raw.strip():
            continue
        line_no += 1
        try:
            rows.append({"_line": line_no, "_row": json.loads(raw)})
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"corrupt JSON on {ledger_path}:{line_no}: {exc}") from exc

    n_rows = len(rows)

    # ---- which metric rows are superseded by a later row of the same metric
    last_line_for_metric: Dict[str, int] = {}
    for r in rows:
        last_line_for_metric[str(r["_row"].get("metric_id"))] = r["_line"]

    # ---- distinct source paths
    src_paths = sorted({str(r["_row"].get("source_json")) for r in rows if r["_row"].get("source_json")})

    # ---- axis B resolvers (two independent methods)
    bulk_by_path, n_blobs = bulk_committed_content_shas(src_paths)
    per_path_by_path = per_path_committed_content_shas(src_paths)
    commit_content_map = per_path_commit_content_map(src_paths)
    disagreements: List[Dict[str, Any]] = []
    for path in src_paths:
        b = bulk_by_path.get(path, set())
        p = per_path_by_path.get(path, set())
        if b != p:
            disagreements.append(
                {
                    "source_json": path,
                    "bulk_only": sorted(b - p),
                    "per_path_only": sorted(p - b),
                }
            )

    # ---- disk shas per source path
    disk_sha: Dict[str, Optional[str]] = {}
    disk_path: Dict[str, str] = {}
    for path in src_paths:
        p = resolve_source_json(path)
        disk_sha[path] = sha256_file(p)
        try:
            disk_path[path] = str(p.relative_to(repo_root))
        except ValueError:
            disk_path[path] = str(p)

    # ---- commit facts (per distinct git_sha)
    git_shas = sorted({str(r["_row"].get("git_sha")) for r in rows})
    commit_info: Dict[str, Dict[str, Any]] = {}
    for sha in git_shas:
        if sha in ("None", ""):
            commit_info[sha] = {
                "commit_exists": False,
                "commit_reachable": False,
                "changed_files": [],
                "n_changed_files": 0,
                "subject": None,
                "committed_at": None,
            }
            continue
        rc, out = git("cat-file", "-t", sha)
        exists = rc == 0 and out.strip() == "commit"
        info: Dict[str, Any] = {
            "commit_exists": exists,
            "changed_files": [],
            "n_changed_files": 0,
            "subject": None,
            "committed_at": None,
        }
        if exists:
            files = commit_changed_files(sha) or []
            info["changed_files"] = files
            info["n_changed_files"] = len(files)
            rc2, out2 = git("log", "-1", "--format=%H%x1f%cI%x1f%s", sha)
            if rc2 == 0 and out2.strip():
                parts = out2.strip().split("\x1f")
                if len(parts) >= 3:
                    info["committed_at"] = parts[1]
                    info["subject"] = parts[2]
            # reachability: contained in some branch or tag (rev-list --all
            # covers refs/heads + refs/tags + refs/remotes)
            rc3, out3 = git("rev-list", "--all")
            info["commit_reachable"] = sha in set(out3.split())
            rc4, out4 = git("branch", "-a", "--contains", sha)
            info["branches_containing"] = [b.strip() for b in out4.splitlines() if b.strip()]
            rc5, out5 = git("tag", "--contains", sha)
            info["tags_containing"] = [t.strip() for t in out5.splitlines() if t.strip()]
        else:
            info["commit_reachable"] = False
            info["branches_containing"] = []
            info["tags_containing"] = []
        commit_info[sha] = info

    # ---- bulk constant: the modal git_sha
    sha_counts = Counter(str(r["_row"].get("git_sha")) for r in rows)
    bulk_sha, bulk_n = sha_counts.most_common(1)[0]

    # ---- axis G: config_sha / data_sha resolvability
    protocols: Dict[str, Any] = {}
    try:
        import yaml  # type: ignore

        if PROTOCOLS_PATH.is_file():
            protocols = (yaml.safe_load(PROTOCOLS_PATH.read_text(encoding="utf-8")) or {}).get(
                "protocols"
            ) or {}
    except Exception:  # noqa: BLE001 — the census degrades, it does not lie
        protocols = {}

    config_path_cache: Dict[str, Optional[Path]] = {}

    def _config_path_for(metric_id: str) -> Optional[Path]:
        if metric_id in config_path_cache:
            return config_path_cache[metric_id]
        rec = protocols.get(metric_id) or {}
        rel = rec.get("trainer_config")
        p: Optional[Path] = None
        if rel:
            cand = Path(str(rel))
            if not cand.is_absolute():
                cand = repo_root / cand
            p = cand if cand.is_file() else None
        config_path_cache[metric_id] = p
        return p

    data_path_cache: Dict[Any, Optional[Path]] = {}

    def _data_path_for(ds: Any) -> Optional[Path]:
        key = str(ds)
        if key not in data_path_cache:
            data_path_cache[key] = resolve_data_source(ds)
        return data_path_cache[key]

    # ---- recoverability: which commits HOLD the row's file content?
    all_commits = sorted({c for m in commit_content_map.values() for c in m})
    commit_date_map = commit_dates(all_commits)

    # ---- candidate B: the metric FILE's own provenance.git_commit
    metric_commit_cache: Dict[str, Optional[str]] = {}

    def _metric_file_commit(src: str) -> Optional[str]:
        if src in metric_commit_cache:
            return metric_commit_cache[src]
        p = REPO_ROOT / src
        val: Optional[str] = None
        if p.is_file() and p.suffix == ".json":
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                gc = (d.get("provenance") or {}).get("git_commit")
                val = str(gc) if gc else None
            except Exception:  # noqa: BLE001
                val = None
        metric_commit_cache[src] = val
        return val

    # ---- per-row table
    table: List[Dict[str, Any]] = []
    for item in rows:
        e = item["_row"]
        mid = str(e.get("metric_id"))
        src = str(e.get("source_json"))
        sha = e.get("source_json_sha256")
        sha = str(sha) if sha is not None else ""
        gsha = str(e.get("git_sha")) if e.get("git_sha") is not None else ""
        cinfo = commit_info.get(gsha, {"commit_exists": False, "commit_reachable": False, "changed_files": []})
        extra = e.get("extra") or {}

        touches = src in (cinfo.get("changed_files") or [])

        # recoverability of a TRUE producing commit for this row: the commits
        # whose committed version of `source_json` has EXACTLY the content the
        # row's sha names.
        holders = [
            c for c, s in (commit_content_map.get(src) or {}).items() if s == sha
        ]
        holders.sort(key=lambda c: commit_date_map.get(c, ""))
        latest_holder = holders[-1] if holders else None

        ds = extra.get("data_source")
        dpath = _data_path_for(ds)
        data_sha_rec = e.get("data_sha")
        data_resolvable = bool(dpath) and dpath.is_file()
        data_sha_match = (
            bool(data_resolvable)
            and data_sha_rec is not None
            and sha256_file(dpath) == str(data_sha_rec)
        )

        cpath = _config_path_for(mid)
        config_sha_rec = e.get("config_sha")
        config_resolvable = bool(cpath) and cpath.is_file()
        config_sha_match = (
            bool(config_resolvable)
            and config_sha_rec is not None
            and sha256_file(cpath) == str(config_sha_rec)
        )

        table.append(
            {
                "line": item["_line"],
                "metric_id": mid,
                "source_json": src,
                "source_json_sha256": sha or None,
                "git_sha": gsha or None,
                "timestamp": e.get("timestamp"),
                "value": e.get("value"),
                "seed": e.get("seed"),
                "host": e.get("host"),
                "protocol_id": e.get("protocol_id"),
                "is_latest_row_for_metric": item["_line"] == last_line_for_metric.get(mid),
                "axis_A_sha_matches_disk": bool(sha) and disk_sha.get(src) == sha,
                "axis_B_sha_matches_a_committed_version": sha
                in (bulk_by_path.get(src, set()) or set()),
                "axis_B_per_path_resolver_agrees": not any(
                    d["source_json"] == src for d in disagreements
                ),
                "axis_C_commit_exists": bool(cinfo.get("commit_exists")),
                "axis_D_commit_touches_source_json": bool(touches),
                "axis_E_commit_reachable": bool(cinfo.get("commit_reachable")),
                "axis_F_git_sha_is_a_bulk_constant": gsha == bulk_sha,
                "axis_G_data_sha_present": data_sha_rec is not None,
                "axis_G_data_source_path_resolvable": data_resolvable,
                "axis_G_data_sha_matches_file": data_sha_match,
                "axis_G_config_sha_present": config_sha_rec is not None,
                "axis_G_config_path_resolvable": config_resolvable,
                "axis_G_config_sha_matches_file": config_sha_match,
                "extra_correction": bool(extra.get("correction")),
                "extra_committed_git_sha": extra.get("committed_git_sha"),
                "recoverable_n_commits_holding_this_content": len(holders),
                "recoverable_latest_holder": latest_holder,
                "recoverable_latest_holder_date": (
                    commit_date_map.get(latest_holder) if latest_holder else None
                ),
                "recoverable_latest_holder_equals_row_git_sha": (
                    latest_holder == gsha if latest_holder else None
                ),
                "candidate_B_metric_file_provenance_git_commit": _metric_file_commit(src),
            }
        )

    def _count(key: str) -> int:
        return sum(1 for r in table if r[key])

    n_sha_nonnull = sum(1 for r in table if r["source_json_sha256"])
    n_git_nonnull = sum(1 for r in table if r["git_sha"])

    census: Dict[str, Any] = {
        "schema": "wfa166.run_ledger_census.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(repo_root),
        "ledger_path": str(ledger_path),
        "denominator": {
            "ledger_rows": n_rows,
            "distinct_metrics": len({r["metric_id"] for r in table}),
            "distinct_source_json_paths": len(src_paths),
            "distinct_git_sha_values": len(git_shas),
            "rows_with_git_sha": n_git_nonnull,
            "rows_with_source_json_sha256": n_sha_nonnull,
            "distinct_source_json_sha256_values": len(
                {r["source_json_sha256"] for r in table if r["source_json_sha256"]}
            ),
        },
        "axis_counts": {
            "A_sha_matches_disk": _count("axis_A_sha_matches_disk"),
            "B_sha_matches_a_committed_version": _count(
                "axis_B_sha_matches_a_committed_version"
            ),
            "C_commit_exists": _count("axis_C_commit_exists"),
            "D_commit_touches_source_json": _count("axis_D_commit_touches_source_json"),
            "E_commit_reachable": _count("axis_E_commit_reachable"),
            "F_git_sha_is_a_bulk_constant": _count("axis_F_git_sha_is_a_bulk_constant"),
            "G_data_sha_present": _count("axis_G_data_sha_present"),
            "G_data_source_path_resolvable": _count(
                "axis_G_data_source_path_resolvable"
            ),
            "G_data_sha_matches_file": _count("axis_G_data_sha_matches_file"),
            "G_config_sha_present": _count("axis_G_config_sha_present"),
            "G_config_path_resolvable": sum(
                1
                for r in table
                if r["axis_G_config_sha_present"] and r["axis_G_config_path_resolvable"]
            ),
            "G_config_sha_matches_file": sum(
                1
                for r in table
                if r["axis_G_config_sha_present"] and r["axis_G_config_sha_matches_file"]
            ),
            "rows_where_source_json_file_is_missing_on_disk": sum(
                1 for p in src_paths if disk_sha.get(p) is None
            ),
        },
        "git_sha_histogram": dict(sha_counts),
        "bulk_constant": {
            "git_sha": bulk_sha,
            "rows": bulk_n,
            "fraction_of_rows": (bulk_n / n_rows) if n_rows else None,
            "distinct_git_sha_values": len(git_shas),
        },
        "commit_facts": {
            sha: {
                "commit_exists": info.get("commit_exists"),
                "commit_reachable": info.get("commit_reachable"),
                "n_changed_files": info.get("n_changed_files"),
                "committed_at": info.get("committed_at"),
                "subject": info.get("subject"),
                "branches_containing": info.get("branches_containing"),
                "tags_containing": info.get("tags_containing"),
                "touches_by_metric_count": sum(
                    1
                    for r in table
                    if r["git_sha"] == sha and r["axis_D_commit_touches_source_json"]
                ),
                "rows_naming_this_sha": sha_counts.get(sha, 0),
            }
            for sha, info in commit_info.items()
        },
        "resolver_agreement": {
            "status": "DISAGREE" if disagreements else "AGREE",
            "bulk_blobs_seen": n_blobs,
            "n_paths_compared": len(src_paths),
            "disagreements": disagreements,
        },
        "rows": table,
    }

    # ---- recoverability summary (the Part-3 question)
    n_recoverable = sum(1 for r in table if r["recoverable_n_commits_holding_this_content"])
    n_unique_holder = sum(
        1 for r in table if r["recoverable_n_commits_holding_this_content"] == 1
    )
    n_holder_equals_row_sha = sum(
        1
        for r in table
        if r["recoverable_latest_holder_equals_row_git_sha"] is True
    )
    n_holder_differs_from_row_sha = sum(
        1
        for r in table
        if r["recoverable_latest_holder"]
        and not r["recoverable_latest_holder_equals_row_git_sha"]
    )
    census["recoverability"] = {
        "definition": (
            "a TRUE producing commit for a row is recoverable iff some commit "
            "in this repository holds a version of the row's source_json whose "
            "content sha256 equals the row's source_json_sha256"
        ),
        "rows_with_a_recoverable_producing_commit": n_recoverable,
        "rows_with_exactly_ONE_holding_commit": n_unique_holder,
        "rows_whose_latest_holder_IS_the_recorded_git_sha": n_holder_equals_row_sha,
        "rows_whose_latest_holder_DIFFERS_from_the_recorded_git_sha": n_holder_differs_from_row_sha,
        "rows_with_NO_recoverable_producing_commit": n_rows - n_recoverable,
        "note": (
            "recovering a holding commit identifies the commit whose TREE held "
            "the metric file at that content; it does NOT identify the commit "
            "whose tree the measurement RAN on.  The ledger never recorded "
            "the latter for any row."
        ),
    }

    # ---- the three candidate commits, side by side
    rows_with_b = [
        r for r in table if r["candidate_B_metric_file_provenance_git_commit"]
    ]
    a_eq_b = sum(
        1
        for r in rows_with_b
        if r["git_sha"] == r["candidate_B_metric_file_provenance_git_commit"]
    )
    a_eq_c = sum(
        1
        for r in table if r["recoverable_latest_holder_equals_row_git_sha"] is True
    )
    b_eq_c = sum(
        1
        for r in rows_with_b
        if r["recoverable_latest_holder"]
        and r["recoverable_latest_holder"]
        == r["candidate_B_metric_file_provenance_git_commit"]
    )
    census["git_sha_candidates"] = {
        "A_ledger_git_sha": "the commit recorded in the ledger row (HEAD at ledger-row-write time)",
        "B_metric_file_provenance_git_commit": "the commit the metric JSON itself records in provenance.git_commit (only some metric JSONs carry it)",
        "C_unique_holder": "the unique commit whose version of source_json has exactly the row's source_json_sha256",
        "rows_with_a_candidate_B": len(rows_with_b),
        "A_equals_B": a_eq_b,
        "A_equals_C": a_eq_c,
        "B_equals_C": b_eq_c,
        "verdict": (
            "the three candidates are three different questions (when the LEDGER "
            "ROW was appended / when the METRIC FILE was written / which commit "
            "HELD that file content); they do not agree, so the ledger's git_sha "
            "cannot be read as any of the other two"
        ),
    }
    census["column_contract"] = COLUMN_CONTRACT
    return census


# ---------------------------------------------------------------------------
def _render_markdown(census: Dict[str, Any]) -> str:
    d = census["denominator"]
    a = census["axis_counts"]
    rows = census["rows"]
    n = d["ledger_rows"]
    lines: List[str] = []
    lines.append("# Wf-A166 -- run-ledger provenance census")
    lines.append("")
    lines.append(f"MEASURED {census['generated_at']} -- READ-ONLY, CPU-only, 0 GPU.")
    lines.append("")
    lines.append(f"- Ledger: `{census['ledger_path']}`")
    lines.append(f"- **Denominator: {n} rows** / {d['distinct_metrics']} distinct metrics /")
    lines.append(
        f"  {d['distinct_source_json_paths']} distinct `source_json` paths / "
        f"{d['distinct_git_sha_values']} distinct `git_sha` values."
    )
    lines.append("")
    lines.append("## Per-axis counts (denominator = %d rows)" % n)
    lines.append("")
    lines.append("| axis | count | share |")
    lines.append("|---|---|---|")
    order = [
        ("A_sha_matches_disk", "A. row sha == file on disk now"),
        (
            "B_sha_matches_a_committed_version",
            "B. row sha == some committed version of that path",
        ),
        ("C_commit_exists", "C. row `git_sha` resolves to a commit object"),
        (
            "D_commit_touches_source_json",
            "D. that commit's changed files contain the row's `source_json`",
        ),
        ("E_commit_reachable", "E. that commit is in some branch or tag"),
        ("F_git_sha_is_a_bulk_constant", "F. row `git_sha` == the ledger's modal sha"),
        ("G_data_sha_present", "G1. row carries a non-null `data_sha`"),
        (
            "G_data_source_path_resolvable",
            "G2. the `data_source` path resolves to a file on disk",
        ),
        ("G_data_sha_matches_file", "G3. that file's sha256 == the row's `data_sha`"),
        ("G_config_sha_present", "G4. row carries a non-null `config_sha`"),
        (
            "G_config_path_resolvable",
            "G5. (of the rows carrying a config_sha) the protocol's `trainer_config` path resolves on disk",
        ),
        (
            "G_config_sha_matches_file",
            "G6. (of the rows carrying a config_sha) that file's sha256 == the row's `config_sha`",
        ),
    ]
    for key, label in order:
        c = int(a.get(key, 0))
        lines.append(f"| {label} | {c} | {100.0 * c / n:.1f}% |")
    lines.append("")
    lines.append(
        f"Distinct `source_json` paths whose file is MISSING on disk: "
        f"{a['rows_where_source_json_file_is_missing_on_disk']}."
    )
    lines.append("")
    lines.append("## `git_sha` histogram")
    lines.append("")
    lines.append("| git_sha | rows | share | commit exists | reachable |")
    lines.append("|---|---|---|---|---|")
    for sha, c in sorted(
        census["git_sha_histogram"].items(), key=lambda kv: (-kv[1], kv[0])
    ):
        f = census["commit_facts"].get(sha, {})
        lines.append(
            f"| `{sha}` | {c} | {100.0 * c / n:.1f}% | "
            f"{f.get('commit_exists')} | {f.get('commit_reachable')} |"
        )
    lines.append("")
    lines.append("## Rows whose named commit DOES touch their `source_json`")
    lines.append("")
    hit = [r for r in rows if r["axis_D_commit_touches_source_json"]]
    if hit:
        lines.append("| line | metric_id | source_json | git_sha |")
        lines.append("|---|---|---|---|")
        for r in hit:
            lines.append(
                f"| {r['line']} | {r['metric_id']} | `{r['source_json']}` | `{r['git_sha']}` |"
            )
    else:
        lines.append("(none)")
    lines.append("")
    lines.append("## Rows whose named commit is UNREACHABLE from any branch or tag")
    lines.append("")
    unreach = [r for r in rows if not r["axis_E_commit_reachable"]]
    if unreach:
        lines.append("| line | metric_id | source_json | git_sha |")
        lines.append("|---|---|---|---|")
        for r in unreach:
            lines.append(
                f"| {r['line']} | {r['metric_id']} | `{r['source_json']}` | `{r['git_sha']}` |"
            )
    else:
        lines.append("(none)")
    lines.append("")
    lines.append(
        f"Resolver agreement: **{census['resolver_agreement']['status']}** "
        f"({census['resolver_agreement']['n_paths_compared']} paths, "
        f"{census['resolver_agreement']['bulk_blobs_seen']} blobs)."
    )
    lines.append("")
    lines.append("## The column contract")
    lines.append("")
    lines.append("| column | writer | kind | validated by |")
    lines.append("|---|---|---|---|")
    for c in census["column_contract"]:
        lines.append(
            f"| `{c['column']}` | `{c['written_at']}` | {c['kind']} | {c['validated_by']} |"
        )
    lines.append("")
    lines.append("## The three candidate commits, side by side")
    lines.append("")
    gc = census["git_sha_candidates"]
    lines.append(
        f"- **A** -- {gc['A_ledger_git_sha']} (recorded in every row)."
    )
    lines.append(f"- **B** -- {gc['B_metric_file_provenance_git_commit']}.")
    lines.append(f"- **C** -- {gc['C_unique_holder']}.")
    lines.append("")
    lines.append(
        f"Rows carrying a candidate B: **{gc['rows_with_a_candidate_B']}**. "
        f"A == B in **{gc['A_equals_B']}** of them; A == C in **{gc['A_equals_C']}** "
        f"of all {n} rows; B == C in **{gc['B_equals_C']}**."
    )
    lines.append("")
    rec = census["recoverability"]
    lines.append(
        f"Recoverability: **{rec['rows_with_a_recoverable_producing_commit']}/{n}** rows "
        f"have a recoverable producing commit "
        f"(**{rec['rows_with_exactly_ONE_holding_commit']}** of them have EXACTLY ONE "
        f"holding commit); **{rec['rows_with_NO_recoverable_producing_commit']}/{n}** have none."
    )
    lines.append("")
    lines.append(f"> {rec['note']}")
    lines.append("")
    lines.append("The full per-row table (every axis, every row) is `census.json`.")
    lines.append("")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Wf-A166 run-ledger provenance census")
    ap.add_argument("--ledger", default=str(LEDGER_PATH))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args(argv)

    ledger = Path(args.ledger)
    if not ledger.is_file():
        print(f"ledger not found: {ledger}", file=sys.stderr)
        return 2
    try:
        census = build_census(ledger_path=ledger, repo_root=REPO_ROOT)
    except Exception as exc:  # noqa: BLE001
        print(f"census FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    if census["resolver_agreement"]["status"] != "AGREE":
        print(
            "census FAILED: the two git resolvers disagree on "
            f"{len(census['resolver_agreement']['disagreements'])} path(s); "
            "the per-axis counts would not be trustworthy",
            file=sys.stderr,
        )
        return 2

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "census.json"
    md_path = out_dir / "census.md"

    try:
        from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

        atomic_write_file(
            json_path, json.dumps(census, indent=2, sort_keys=False, ensure_ascii=False) + "\n"
        )
        atomic_write_file(md_path, _render_markdown(census))
    except Exception:  # noqa: BLE001 — fall back to a plain write
        json_path.write_text(
            json.dumps(census, indent=2, sort_keys=False, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        md_path.write_text(_render_markdown(census), encoding="utf-8")

    a = census["axis_counts"]
    n = census["denominator"]["ledger_rows"]
    print(f"Wf-A166 run-ledger census: rows={n} (denominator)")
    print(f"  A sha_matches_disk                 {a['A_sha_matches_disk']}/{n}")
    print(
        f"  B sha_matches_a_committed_version  {a['B_sha_matches_a_committed_version']}/{n}"
    )
    print(f"  C commit_exists                    {a['C_commit_exists']}/{n}")
    print(
        f"  D commit_touches_source_json       {a['D_commit_touches_source_json']}/{n}"
    )
    print(f"  E commit_reachable                 {a['E_commit_reachable']}/{n}")
    print(
        f"  F git_sha_is_a_bulk_constant       {a['F_git_sha_is_a_bulk_constant']}/{n}"
    )
    print(f"  G data_sha present/match           {a['G_data_sha_present']}/{a['G_data_sha_matches_file']}")
    print(
        f"  G config_sha present/match         {a['G_config_sha_present']}/{a['G_config_sha_matches_file']}"
    )
    bc = census["bulk_constant"]
    print(
        f"  bulk constant git_sha              {bc['git_sha']} ({bc['rows']}/{n} rows)"
    )
    print(f"  resolvers                          {census['resolver_agreement']['status']}")
    print(f"  wrote {json_path}")
    print(f"  wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
