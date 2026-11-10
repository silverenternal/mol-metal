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

"""Wf-A166 — run-ledger GIT-PROVENANCE validator (READ-ONLY, CPU-only, torch-free).

WHY THIS EXISTS
---------------
``molmetal/reports/run_ledger.jsonl`` records, for every metric value, the
``git_sha`` of the repository state at record time, the ``source_json`` path
and the ``source_json_sha256`` of the file that carried the number.  Nothing
in the repo checks that the ``git_sha`` a row names is a real commit, is
reachable, or is related to the file the row points at.

``molmetal/scripts/run_ledger.py::uncommitted_provenance`` (wired into the
pre-flight as ``p3_run_ledger_provenance``) checks ONE axis: does the row's
``source_json_sha256`` match the file on disk or any committed version of it.
It does NOT check the ``git_sha`` at all.  So a row can name a commit that
does not exist, a commit that was rewritten away, or a commit that has
nothing to do with the file the row points at — and the pre-flight stays
green.

This module is the missing validator: a READ-ONLY, CPU-only, torch-free
audit of the ledger that measures every provenance axis per row and per
metric, and returns the aggregate counts **with the denominator** so a
verdict can never be quoted without it.

SEMANTICS OF ``git_sha`` (MEASURED — read this before writing a rule)
----------------------------------------------------------------------
``run_ledger.record_run`` sets ``git_sha = git rev-parse HEAD`` unless the
caller supplies one.  It therefore means **"the repository state at the
moment the row was recorded"**, NOT **"the commit that authored the
source_json"**.  The two are different questions and the distinction decides
whether a gate is useful or destructive:

* ``commit_content_matches`` — "the sha256 of ``source_json`` AT
  ``git_sha`` equals the row's ``source_json_sha256``" — is the TRUE
  provenance claim.  It is reproducible from git alone.
* ``commit_touches_source`` — "the diff of ``git_sha`` includes
  ``source_json``" — is a claim about WHICH COMMIT WROTE the file.  A row
  recorded after an unrelated commit does not satisfy it, and does not need
  to.

Both are measured here.  The gate proposal in
``molmetal/reports/wfa166_run_ledger_provenance/`` uses both, and the
report states which one blocks launches.

APPEND-ONLY SCOPING
--------------------
The ledger is append-only (AGENTS.md §5.6).  A finding on a metric's LATEST
row is the LIVE defect; a finding on an earlier row is HISTORY, which can
never be repaired (the row stays in the file forever) and is therefore
reported as ``superseded`` and never counted as blocking.

COST
----
Five batched git calls total, not one per row:
``cat-file --batch-check`` (commit types), ``rev-list --branches --tags``
(reachability), ``log --all --raw`` (which commit touched which path + every
committed blob sha per path), and two ``cat-file --batch`` calls (blob
contents + ``<sha>:<path>`` contents).  MEASURED 2026-11-01 on this
repository: 418 ledger rows audited in ~2 s wall.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = REPO_ROOT / "molmetal" / "reports" / "run_ledger.jsonl"
BY_METRIC_DIR = REPO_ROOT / "metrics" / "by_metric"

#: sha256 of the empty file and of ``b"{}"`` — the two shas a placeholder /
#: never-written file would carry.  A row whose ``source_json_sha256`` is one
#: of these names a file that does not exist.
CONSTANT_PLACEHOLDER_SHAS = frozenset(
    {
        hashlib.sha256(b"").hexdigest(),
        hashlib.sha256(b"{}").hexdigest(),
        hashlib.sha256(b"[]").hexdigest(),
        hashlib.sha256(b"null").hexdigest(),
    }
)

#: Axes whose violation on a metric's LATEST row is BLOCKING (the gate FAILs).
BLOCKING_AXES = (
    "commit_unresolvable",
    "commit_unreachable",
    "commit_never_touched_source",
)

#: Axes whose violation is ADVISORY (reported, counted, never blocks).
ADVISORY_AXES = (
    "sha_never_committed",
    "sha_not_on_disk",
    "commit_content_differs",
    "bulk_constant_sha",
    "config_sha_unresolved",
    "data_sha_unresolved",
    "ledger_corrupt_line",
)

#: Every axis this module can emit.
ALL_AXES = BLOCKING_AXES + ADVISORY_AXES


# ---------------------------------------------------------------------------
# git plumbing
# ---------------------------------------------------------------------------
def _git(
    repo_root: Path,
    args: Sequence[str],
    *,
    stdin: Optional[bytes] = None,
    timeout: int = 300,
) -> Optional[subprocess.CompletedProcess]:
    """Run one git command in ``repo_root``.  Returns None on any failure."""
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), *args],
            input=stdin,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _commit_types(repo_root: Path, shas: Sequence[str]) -> Dict[str, str]:
    """Map each sha to its git object type ("commit", "blob", "missing", ...).

    ONE ``git cat-file --batch-check`` call for all shas.
    """
    out: Dict[str, str] = {}
    shas = [s for s in shas if s]
    if not shas:
        return out
    proc = _git(
        repo_root,
        ["cat-file", "--batch-check"],
        stdin=("\n".join(shas) + "\n").encode("utf-8"),
    )
    if proc is None:
        return {s: "unknown" for s in shas}
    lines = proc.stdout.decode("utf-8", "replace").splitlines()
    for sha, line in zip(shas, lines):
        parts = line.split()
        if len(parts) >= 2:
            out[sha] = parts[1]
        else:
            out[sha] = "missing"
    return out


def _reachable_commits(repo_root: Path) -> Set[str]:
    """Every commit reachable from a BRANCH or a TAG.

    ``git rev-list --branches --tags`` walks local branches and tags only;
    that is exactly the "reachable from every branch and tag" question.  A
    commit that survives only in the reflog (a rewritten / rebased-away
    commit) is NOT in this set, which is the point.
    """
    for args in (["rev-list", "--branches", "--tags"], ["rev-list", "--all"]):
        proc = _git(repo_root, args)
        if proc is None or proc.returncode != 0:
            continue
        text = proc.stdout.decode("utf-8", "replace")
        if text.strip():
            return {ln.strip() for ln in text.splitlines() if ln.strip()}
    return set()


def _history_raw(
    repo_root: Path, paths: Optional[Sequence[str]] = None
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """ONE ``git log --all --raw`` call -> two maps.

    Returns ``(commit_paths, path_blob_shas)``:

    * ``commit_paths[commit_sha]`` — the set of repo-relative paths that
      commit's diff touched (added / modified / deleted).
    * ``path_blob_shas[path]`` — every committed CONTENT version of that
      path, as git blob shas (the NEW-side sha of each raw diff line; a
      deletion contributes nothing).

    Merge commits are diffed against their first parent (``--diff-merges=
    first-parent``) so a merge that carries a file change is attributed.

    ``paths`` restricts the walk to the ledger's own source paths.  That is
    a pure optimisation: the "did this commit touch P" answer is identical
    (MEASURED 2026-11-01: 418/418 rows agree between the restricted and the
    unrestricted walk), but the restricted walk touches 754 blobs instead of
    17 308, which takes the batched ``cat-file`` call from 7.0 s to 0.04 s.
    """
    commit_paths: Dict[str, Set[str]] = {}
    path_blobs: Dict[str, Set[str]] = {}
    args = [
        "log",
        "--all",
        "--format=%H",
        "--raw",
        "--no-abbrev",
        "--no-renames",
        "--diff-merges=first-parent",
    ]
    if paths:
        args += ["--", *paths]
    proc = _git(repo_root, args)
    if proc is None:
        return commit_paths, path_blobs
    current: Optional[str] = None
    for line in proc.stdout.decode("utf-8", "replace").splitlines():
        stripped = line.strip()
        if len(stripped) == 40 and all(c in "0123456789abcdef" for c in stripped):
            current = stripped
            commit_paths.setdefault(current, set())
            continue
        if not line.startswith(":"):
            continue
        meta, _, path = line.partition("\t")
        parts = meta.split()
        if len(parts) < 4 or not path:
            continue
        new_sha = parts[3]
        if set(new_sha) == {"0"}:
            continue  # a deletion, not a version
        if current is not None:
            commit_paths[current].add(path)
        path_blobs.setdefault(path, set()).add(new_sha)
    return commit_paths, path_blobs


def _batch_content_shas(
    repo_root: Path, specs: Sequence[str]
) -> Dict[str, Optional[str]]:
    """sha256 of the CONTENT of each git object / ``<sha>:<path>`` spec.

    ONE ``git cat-file --batch`` call for all specs.  A spec that does not
    resolve maps to ``None``.
    """
    out: Dict[str, Optional[str]] = {}
    specs = [s for s in dict.fromkeys(specs) if s]
    if not specs:
        return out
    proc = _git(
        repo_root,
        ["cat-file", "--batch"],
        stdin=("\n".join(specs) + "\n").encode("utf-8"),
    )
    if proc is None:
        return {s: None for s in specs}
    data = proc.stdout
    pos = 0
    for spec in specs:
        nl = data.find(b"\n", pos)
        if nl < 0:
            out[spec] = None
            break
        header = data[pos:nl].decode("utf-8", "replace").split()
        pos = nl + 1
        if len(header) < 3:
            # "<spec> missing" / "<spec> ambiguous" / "<spec> dangling"
            out[spec] = None
            continue
        try:
            size = int(header[2])
        except ValueError:
            out[spec] = None
            continue
        body = data[pos : pos + size]
        out[spec] = hashlib.sha256(body).hexdigest()
        pos = pos + size + 1
    return out


def _sha256_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# REACHABILITY — the whole object database, not only refs-reachable history
# ---------------------------------------------------------------------------
# WHY (2026-11-04, Wf-A269 wave-23): "is this file version committed?" was
# decided over REFS-REACHABLE history only (``git log --all``).  Content that
# WAS committed and is now UNREACHABLE (rewritten away, amended, rebased) was
# therefore invisible, and the finding whose text says "matches neither the
# file on disk nor any committed version" was emitted on top of that blind
# spot.  MEASURED on this repository: the object database holds 1011 commit
# objects while ``git rev-list --all`` reaches 843, so 168 commits (16.6 %)
# are unreachable.
#
# The three-valued verdict below is the fix.  The LEGACY axis
# (``sha_vs_committed``) keeps its NAME and its VALUE — it is quoted VERBATIM
# in ``docs/HONEST_NEGATIVE_CATALOG.md`` and ``TODO/INDEX.md`` as
# ``OPEN_sha_never_committed=14`` / ``superseded_sha_never_committed=9`` —
# and is now labelled REACHABLE-SCOPED in the text.  The three-valued axis is
# ADDED alongside it.
COMMIT_STATUS_COMMITTED = "committed"
COMMIT_STATUS_COMMITTED_UNREACHABLE = "committed_unreachable"
COMMIT_STATUS_NEVER_COMMITTED = "never_committed"
COMMIT_STATUS_VALUES = (
    COMMIT_STATUS_COMMITTED,
    COMMIT_STATUS_COMMITTED_UNREACHABLE,
    COMMIT_STATUS_NEVER_COMMITTED,
)


def _odb_commits(repo_root: Path) -> Set[str]:
    """Every commit object in the OBJECT DATABASE, reachable or not.

    ONE ``git cat-file --batch-all-objects --batch-check`` call.  MEASURED
    2026-11-04 on this repository: 25 494 objects = 17 492 blobs + 6991 trees
    + 1011 commits, in 0.024 s.  (RE-MEASURED later the same day: 25 500
    objects = 17 498 blobs + 6991 trees + 1011 commits — the BLOB count moves
    with the tree; the COMMIT count is what the reachability question uses and
    it was stable.)
    """
    proc = _git(
        repo_root,
        ["cat-file", "--batch-all-objects", "--batch-check=%(objectname) %(objecttype)"],
    )
    out: Set[str] = set()
    if proc is None or proc.returncode != 0:
        return out
    for line in proc.stdout.decode("utf-8", "replace").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == "commit":
            out.add(parts[0])
    return out


def _unreachable_commits(repo_root: Path) -> Set[str]:
    """ODB commits that NO ref reaches.

    ``git rev-list --all`` is the reachability question (``--branches
    --tags`` is a SUBSET of it and would leave remote refs / stash counted as
    unreachable, so the wider form is used here).
    """
    odb = _odb_commits(repo_root)
    if not odb:
        return set()
    proc = _git(repo_root, ["rev-list", "--all"])
    if proc is None or proc.returncode != 0:
        return set()
    reachable = {
        ln.strip()
        for ln in proc.stdout.decode("utf-8", "replace").splitlines()
        if ln.strip()
    }
    return odb - reachable


def _unreachable_path_blobs(
    repo_root: Path, paths: Sequence[str]
) -> Dict[str, Set[str]]:
    """Every committed version of each path held by an UNREACHABLE commit.

    ONE ``git ls-tree -r <commit> -- <paths>`` call PER unreachable commit.
    The tree is read directly — no diff machinery, no history simplification,
    no merge heuristics — so the answer is exact by construction: "the blob at
    path P in commit C's tree".  (MEASURED 2026-11-04: 168 unreachable commits
    x 401 requested paths = 0.69 s.)

    WHY NOT ONE BATCHED ``git ls-tree -r C1 C2 ... -- paths``: that form
    DEDUPES across the trees and returns only 397 of the 505 real
    (path, blob) pairs — a silent under-count, which is the exact failure mode
    this function exists to prevent.

    API-SHAPE CONTRACT: an entry for EVERY REQUESTED path (empty set = "no
    unreachable committed version").
    """
    out: Dict[str, Set[str]] = {p: set() for p in paths}
    for commit in sorted(_unreachable_commits(repo_root)):
        proc = _git(repo_root, ["ls-tree", "-r", commit, "--", *paths])
        if proc is None or proc.returncode != 0:
            continue
        for line in proc.stdout.decode("utf-8", "replace").splitlines():
            meta, _, path = line.partition("\t")
            parts = meta.split()
            if len(parts) != 3 or parts[1] != "blob" or not path:
                continue
            out.setdefault(path, set()).add(parts[2])
    return out


def committed_content_shas(
    repo_root: Path,
    paths: Sequence[str],
    path_blobs: Optional[Dict[str, Set[str]]] = None,
) -> Dict[str, Set[str]]:
    """``path -> {content sha256}`` for EVERY REQUESTED path.

    API-SHAPE CONTRACT (2026-11-04, Wf-A269 wave-23): this function returns
    an entry for EVERY REQUESTED path.  A path with no committed history maps
    to an EMPTY set, which means "this path has no committed version" — it is
    NOT the same as "this path was not asked about".  The contract is
    normalised HERE, at the API, rather than at each comparison site: the
    sibling implementations (``wfa156_ledger_provenance_guard.py``,
    ``run_ledger.py``, ``wfa198_ledger_pin_consistency.py``,
    ``wfa166_run_ledger_census.py``) all return an entry for every requested
    path, so a naive comparison of the five used to manufacture a 4-path
    "disagreement" that was pure API shape and said nothing about git.

    ``path_blobs`` lets a caller that ALREADY walked the history pass the
    result in, so the walk is not repeated.
    """
    out: Dict[str, Set[str]] = {p: set() for p in paths}
    if not paths:
        return out
    if path_blobs is None:
        _commit_paths, path_blobs = _history_raw(repo_root, list(paths))
    blob_specs = sorted({b for v in path_blobs.values() for b in v})
    blob_content = _batch_content_shas(repo_root, blob_specs)
    for path, blobs in path_blobs.items():
        shas_here = {
            blob_content.get(b) for b in blobs if blob_content.get(b) is not None
        }
        shas_here.discard(None)
        out.setdefault(path, set()).update(shas_here)
    for path in paths:
        out.setdefault(path, set())
    return out


def committed_content_shas_unreachable(
    repo_root: Path, paths: Sequence[str]
) -> Dict[str, Set[str]]:
    """``path -> {content sha256}`` for versions held by UNREACHABLE commits.

    API-SHAPE CONTRACT: an entry for EVERY REQUESTED path (empty set = "no
    unreachable committed version"), same as :func:`committed_content_shas`.
    """
    out: Dict[str, Set[str]] = {p: set() for p in paths}
    if not paths:
        return out
    blobs_by_path = _unreachable_path_blobs(repo_root, list(paths))
    blob_specs = sorted({b for v in blobs_by_path.values() for b in v})
    blob_content = _batch_content_shas(repo_root, blob_specs)
    for path, blobs in blobs_by_path.items():
        shas_here = {
            blob_content.get(b) for b in blobs if blob_content.get(b) is not None
        }
        shas_here.discard(None)
        out.setdefault(path, set()).update(shas_here)
    for path in paths:
        out.setdefault(path, set())
    return out


# ---------------------------------------------------------------------------
# path resolution (mirrors molmetal/scripts/run_ledger.py semantics)
# ---------------------------------------------------------------------------
def _resolve_source_json(source_json: str, repo_root: Path = REPO_ROOT) -> Path:
    """Resolve a ledger ``source_json`` string to an absolute path.

    Same semantics as ``run_ledger._resolve_source_json``: a repo-relative
    path is taken as-is; a bare stem resolves under ``metrics/by_metric/``.
    """
    p = Path(source_json)
    if p.is_absolute():
        return p
    if "/" in source_json or p.suffix == ".json":
        return repo_root / p
    return repo_root / "metrics" / "by_metric" / f"{source_json}.json"


def _resolve_data_source(data_source: str, repo_root: Path = REPO_ROOT) -> Optional[Path]:
    """Resolve a ledger ``extra.data_source`` string to an on-disk file.

    ``data_source`` is often prose ("MetalCytoToxDB.csv (precious-metal
    subset)"); take the leading token and try the known roots.  Returns None
    when nothing resolves — never fabricates a path.
    """
    if not data_source:
        return None
    token = str(data_source).strip().split()[0].split("(")[0].strip().strip(",;")
    if not token:
        return None
    candidates = [
        Path(token),
        repo_root / token,
        repo_root / ".data" / "molmetal" / Path(token).name,
        repo_root / "molmetal" / "data" / Path(token).name,
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def _config_index(repo_root: Path) -> Dict[str, str]:
    """sha256 -> repo-relative path, over the tracked+untracked ``configs/``.

    Used by the ``config_sha_resolvable`` axis: a ledger row's
    ``config_sha`` is only meaningful if it can be pointed at a file.
    """
    proc = _git(
        repo_root,
        ["ls-files", "--cached", "--others", "--exclude-standard", "--", "configs"],
    )
    index: Dict[str, str] = {}
    if proc is None:
        return index
    for rel in proc.stdout.decode("utf-8", "replace").splitlines():
        rel = rel.strip()
        if not rel:
            continue
        sha = _sha256_file(repo_root / rel)
        if sha:
            index.setdefault(sha, rel)
    return index


# ---------------------------------------------------------------------------
# the audit
# ---------------------------------------------------------------------------
def _read_ledger(
    ledger_path: Path,
) -> Tuple[List[Tuple[int, Dict[str, Any]]], List[Tuple[int, str]]]:
    """Read the ledger.  Returns ``(rows, corrupt)``.

    A corrupt line is a FINDING, not a crash: a ledger that cannot be parsed
    is exactly the kind of thing this validator exists to surface, and a gate
    must be able to report it instead of dying.
    """
    rows: List[Tuple[int, Dict[str, Any]]] = []
    corrupt: List[Tuple[int, str]] = []
    if not ledger_path.is_file():
        return rows, corrupt
    for lineno, raw in enumerate(
        ledger_path.read_text(encoding="utf-8", errors="replace").splitlines(),
        start=1,
    ):
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            corrupt.append((lineno, str(exc)))
            continue
        if not isinstance(obj, dict):
            corrupt.append((lineno, "not a JSON object"))
            continue
        rows.append((lineno, obj))
    return rows, corrupt


def audit_ledger(
    ledger_path: Optional[Path] = None,
    repo_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Audit the run ledger's provenance.  READ-ONLY, CPU-only, torch-free.

    Parameters
    ----------
    ledger_path
        Path to the JSONL ledger.  Defaults to
        ``molmetal/reports/run_ledger.jsonl``.
    repo_root
        Git working tree to audit against.  Defaults to this repository.

    Returns
    -------
    dict
        ``{status, counts, rows, metrics, findings, gate_verdict}``.

        ``counts`` carries the DENOMINATOR alongside every numerator
        (``ledger_rows``, ``distinct_metrics``, ``latest_rows``) so no count
        can be quoted without it.  ``findings`` is the flat list; every
        finding carries ``scope`` ("latest" / "superseded"), ``severity``
        ("blocking" / "advisory"), ``axis``, ``metric_id``, ``row_index`` and
        a human-readable ``detail``.
    """
    ledger_path = Path(ledger_path) if ledger_path is not None else LEDGER_PATH
    repo_root = Path(repo_root) if repo_root is not None else REPO_ROOT

    raw_rows, corrupt = _read_ledger(ledger_path)
    rows = [obj for _, obj in raw_rows]
    linenos = [ln for ln, _ in raw_rows]

    findings: List[Dict[str, Any]] = []

    def _add(axis: str, scope: str, row_index: Optional[int], obj: Dict[str, Any],
             detail: str) -> None:
        findings.append(
            {
                "axis": axis,
                "severity": "blocking" if axis in BLOCKING_AXES else "advisory",
                "scope": scope,
                "row_index": row_index,
                "metric_id": obj.get("metric_id"),
                "source_json": obj.get("source_json"),
                "git_sha": obj.get("git_sha"),
                "source_json_sha256": obj.get("source_json_sha256"),
                "detail": detail,
            }
        )

    for lineno, msg in corrupt:
        findings.append(
            {
                "axis": "ledger_corrupt_line",
                "severity": "advisory",
                "scope": "latest",
                "row_index": lineno,
                "metric_id": None,
                "source_json": None,
                "git_sha": None,
                "source_json_sha256": None,
                "detail": f"{ledger_path}:{lineno} is not parseable JSON: {msg}",
            }
        )

    # ---- which row is the LATEST for each metric?  The ledger is
    # append-only, so the last row in FILE ORDER for a metric is its
    # current one.  (Grouping by sha would order by content, not by time.)
    latest_index: Dict[str, int] = {}
    for idx, obj in enumerate(rows):
        latest_index[str(obj.get("metric_id"))] = idx

    # ---- git facts (5 batched calls, not one per row) -------------------
    shas = sorted({str(o.get("git_sha")) for o in rows if o.get("git_sha")})
    src_paths = sorted({str(o.get("source_json")) for o in rows if o.get("source_json")})
    ctypes = _commit_types(repo_root, shas)
    reachable = _reachable_commits(repo_root)
    commit_paths, path_blobs = _history_raw(repo_root, src_paths)

    # content sha of every committed version of every referenced path.
    # API-SHAPE CONTRACT (2026-11-04, Wf-A269 wave-23): an entry for EVERY
    # REQUESTED path — an empty set means "this path has no committed
    # version", NOT "this path was not asked about".  The contract is
    # normalised at the API, not at the comparison sites.
    committed_content: Dict[str, Set[str]] = committed_content_shas(
        repo_root, src_paths, path_blobs
    )
    # The versions held by commits that NO ref reaches — the class the
    # refs-reachable-only query could not see.
    unreachable_content: Dict[str, Set[str]] = committed_content_shas_unreachable(
        repo_root, src_paths
    )

    # content sha of `<git_sha>:<source_json>` for every row
    rev_specs = sorted(
        {
            f"{o.get('git_sha')}:{o.get('source_json')}"
            for o in rows
            if o.get("git_sha") and o.get("source_json")
        }
    )
    rev_content = _batch_content_shas(repo_root, rev_specs)

    # disk sha of every referenced source path
    disk_sha: Dict[str, Optional[str]] = {
        p: _sha256_file(_resolve_source_json(p, repo_root=repo_root)) for p in src_paths
    }

    # bulk-constant detection: one sha claimed by several DIFFERENT paths
    sha_to_paths: Dict[str, Set[str]] = {}
    for obj in rows:
        sha = obj.get("source_json_sha256")
        src = obj.get("source_json")
        if sha and src:
            sha_to_paths.setdefault(str(sha), set()).add(str(src))

    # config / data sha resolvability
    config_index = _config_index(repo_root)
    data_cache: Dict[str, Optional[str]] = {}

    # ---- per-row evaluation --------------------------------------------
    row_records: List[Dict[str, Any]] = []
    for idx, obj in enumerate(rows):
        mid = str(obj.get("metric_id"))
        src = obj.get("source_json")
        sha = obj.get("source_json_sha256")
        gsha = obj.get("git_sha")
        scope = "latest" if latest_index.get(mid) == idx else "superseded"

        verdicts: Dict[str, str] = {}
        details: Dict[str, Any] = {}

        # -- commit axes
        ctype = ctypes.get(str(gsha)) if gsha else "missing"
        if not gsha:
            verdicts["commit_exists"] = "missing"
        elif ctype == "commit":
            verdicts["commit_exists"] = "commit"
        elif ctype == "unknown":
            verdicts["commit_exists"] = "unknown"
        else:
            verdicts["commit_exists"] = ctype

        if verdicts["commit_exists"] == "commit":
            if str(gsha) in reachable:
                verdicts["commit_reachable"] = "reachable"
            else:
                verdicts["commit_reachable"] = "unreachable"
            touched = commit_paths.get(str(gsha), set())
            # An empty commit has no entry in `commit_paths`; an unreachable
            # commit is absent from `git log --all` entirely.  Both mean the
            # commit demonstrably did not touch the path.
            verdicts["commit_touches_source"] = (
                "touched" if src in touched else "not_touched"
            )
        else:
            verdicts["commit_reachable"] = "unknown"
            verdicts["commit_touches_source"] = "unknown"

        # -- content at that commit
        if gsha and src:
            at_commit = rev_content.get(f"{gsha}:{src}")
            if at_commit is None:
                verdicts["commit_content_matches"] = "absent"
            elif sha and at_commit == str(sha):
                verdicts["commit_content_matches"] = "match"
            else:
                verdicts["commit_content_matches"] = "differ"
        else:
            verdicts["commit_content_matches"] = "unknown"

        # -- sha vs disk / vs committed
        if sha is None:
            verdicts["sha_vs_disk"] = "unknown"
            verdicts["sha_vs_committed"] = "unknown"
            verdicts["sha_commit_status"] = "unknown"
        else:
            verdicts["sha_vs_disk"] = (
                "match"
                if disk_sha.get(str(src)) == str(sha)
                else ("file_missing" if disk_sha.get(str(src)) is None else "differ")
            )
            # LEGACY axis — REACHABLE-SCOPED.  Its NAME and its VALUE are
            # preserved verbatim (``OPEN_sha_never_committed=14`` /
            # ``superseded_sha_never_committed=9`` are published in
            # ``docs/HONEST_NEGATIVE_CATALOG.md`` and ``TODO/INDEX.md``);
            # the finding TEXT now says "REFS-REACHABLE" so the label
            # matches what was actually checked.
            verdicts["sha_vs_committed"] = (
                "match"
                if str(sha) in committed_content.get(str(src), set())
                else "differ"
            )
            # The THREE-VALUED whole-object-database verdict (2026-11-04,
            # Wf-A269 wave-23), added ALONGSIDE the legacy axis.
            if str(sha) in committed_content.get(str(src), set()):
                verdicts["sha_commit_status"] = COMMIT_STATUS_COMMITTED
            elif str(sha) in unreachable_content.get(str(src), set()):
                verdicts["sha_commit_status"] = COMMIT_STATUS_COMMITTED_UNREACHABLE
            else:
                verdicts["sha_commit_status"] = COMMIT_STATUS_NEVER_COMMITTED

        # -- bulk-constant
        n_paths = len(sha_to_paths.get(str(sha), set())) if sha else 0
        if sha and str(sha) in CONSTANT_PLACEHOLDER_SHAS:
            verdicts["bulk_constant_sha"] = "constant_placeholder"
            details["bulk_constant_sha_paths"] = n_paths
        elif n_paths > 1:
            verdicts["bulk_constant_sha"] = "shared_across_paths"
            details["bulk_constant_sha_paths"] = n_paths
        else:
            verdicts["bulk_constant_sha"] = "unique"

        # -- config / data sha resolvability
        cfg_sha = obj.get("config_sha")
        if not cfg_sha:
            verdicts["config_sha_resolvable"] = "n/a"
        elif str(cfg_sha) in config_index:
            verdicts["config_sha_resolvable"] = "match"
            details["config_path"] = config_index[str(cfg_sha)]
        else:
            verdicts["config_sha_resolvable"] = "unresolved"
        dat_sha = obj.get("data_sha")
        if not dat_sha:
            verdicts["data_sha_resolvable"] = "n/a"
        else:
            ds = (obj.get("extra") or {}).get("data_source")
            if ds is None:
                verdicts["data_sha_resolvable"] = "no_data_source"
            else:
                if ds not in data_cache:
                    data_cache[ds] = _sha256_file(
                        _resolve_data_source(ds, repo_root=repo_root)
                        or Path("/nonexistent")
                    )
                if data_cache[ds] == str(dat_sha):
                    verdicts["data_sha_resolvable"] = "match"
                elif data_cache[ds] is None:
                    verdicts["data_sha_resolvable"] = "unresolved"
                else:
                    verdicts["data_sha_resolvable"] = "differ"

        # -- emit findings
        if verdicts["commit_exists"] != "commit":
            _add(
                "commit_unresolvable",
                scope,
                linenos[idx],
                obj,
                f"git_sha {gsha!r} does not resolve to a git commit "
                f"(git object type = {verdicts['commit_exists']!r})",
            )
        if verdicts["commit_reachable"] == "unreachable":
            _add(
                "commit_unreachable",
                scope,
                linenos[idx],
                obj,
                f"git_sha {gsha!r} is not reachable from any branch or tag",
            )
        if verdicts["commit_touches_source"] == "not_touched":
            _add(
                "commit_never_touched_source",
                scope,
                linenos[idx],
                obj,
                f"commit {gsha} does not touch {src!r} "
                f"(the diff of that commit names other paths only)",
            )
        if verdicts["sha_vs_committed"] == "differ":
            # D1 FIX (2026-11-04, Wf-A269 wave-23): the old text asserted a
            # DISJUNCTION ("matches neither the file on disk nor any
            # committed version") but only the committed half was decided at
            # this point — the disk half is a SEPARATE verdict
            # (``sha_vs_disk``), and for 12 of the 23 findings that verdict
            # is "match", so the disk clause was FALSE.  The two halves are
            # now reported INDEPENDENTLY, each from its own verdict, and the
            # three-valued whole-ODB status is carried alongside.
            #
            # The AXIS NAME and the severity/blocking status are UNCHANGED:
            # the published counts ``OPEN_sha_never_committed=14`` /
            # ``superseded_sha_never_committed=9`` depend on this axis.
            _disk_clause = {
                "match": "MATCHES the file on disk",
                "differ": "does NOT match the file on disk",
                "file_missing": "the file on disk is MISSING",
                "unknown": "the file on disk could not be read",
            }.get(verdicts["sha_vs_disk"], "the file on disk is UNKNOWN")
            _add(
                "sha_never_committed",
                scope,
                linenos[idx],
                obj,
                f"source_json_sha256 {sha}: {_disk_clause}; matches NO "
                f"REFS-REACHABLE committed version of {src!r}; "
                f"whole-object-database commit status = "
                f"{verdicts.get('sha_commit_status', 'unknown')}",
            )
        if verdicts["sha_vs_disk"] == "differ":
            _add(
                "sha_not_on_disk",
                scope,
                linenos[idx],
                obj,
                f"source_json_sha256 {sha} does not match the file on disk "
                f"({src!r})",
            )
        if verdicts["commit_content_matches"] == "differ":
            _add(
                "commit_content_differs",
                scope,
                linenos[idx],
                obj,
                f"the content of {src!r} at commit {gsha} has a different "
                f"sha256 than the row's source_json_sha256",
            )
        if verdicts["bulk_constant_sha"] in ("constant_placeholder", "shared_across_paths"):
            _add(
                "bulk_constant_sha",
                scope,
                linenos[idx],
                obj,
                f"source_json_sha256 {sha} is a constant placeholder / is "
                f"claimed by {n_paths} distinct source paths",
            )
        if verdicts["config_sha_resolvable"] == "unresolved":
            _add(
                "config_sha_unresolved",
                scope,
                linenos[idx],
                obj,
                f"config_sha {cfg_sha} matches no file under configs/",
            )
        if verdicts["data_sha_resolvable"] in ("unresolved", "no_data_source", "differ"):
            _add(
                "data_sha_unresolved",
                scope,
                linenos[idx],
                obj,
                f"data_sha {dat_sha} could not be re-derived from the row's "
                f"data_source {ds!r} (verdict "
                f"{verdicts['data_sha_resolvable']})",
            )

        row_records.append(
            {
                "row_index": linenos[idx],
                "metric_id": mid,
                "source_json": src,
                "source_json_sha256": sha,
                "git_sha": gsha,
                "timestamp": obj.get("timestamp"),
                "seed": obj.get("seed"),
                "scope": scope,
                "verdicts": verdicts,
                "details": details,
            }
        )

    # ---- aggregates ------------------------------------------------------
    def _tally(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
        out: Dict[str, Dict[str, int]] = {}
        for rec in records:
            for axis, verdict in rec["verdicts"].items():
                out.setdefault(axis, {})
                out[axis][verdict] = out[axis].get(verdict, 0) + 1
        return out

    latest_records = [r for r in row_records if r["scope"] == "latest"]
    latest_findings = [f for f in findings if f["scope"] == "latest"]
    superseded_findings = [f for f in findings if f["scope"] == "superseded"]

    blocking_latest = [f for f in latest_findings if f["severity"] == "blocking"]
    advisory_latest = [f for f in latest_findings if f["severity"] == "advisory"]

    blocking_metrics: Dict[str, List[str]] = {}
    for f in blocking_latest:
        blocking_metrics.setdefault(f["axis"], [])
        if f["metric_id"] not in blocking_metrics[f["axis"]]:
            blocking_metrics[f["axis"]].append(str(f["metric_id"]))

    per_axis_metric_counts = {axis: len(v) for axis, v in sorted(blocking_metrics.items())}

    # the strictest single-axis verdict (the gate rule proposed in the report)
    strict_axis_open = sorted(
        mid
        for axis in BLOCKING_AXES
        for mid in blocking_metrics.get(axis, [])
    )
    strict_axis_open = sorted(set(strict_axis_open))

    # the PROVENANCE-TRUE rule: the sha must be the content of the file at
    # the named commit (plus commit exists / reachable).
    true_open: List[str] = []
    true_axes = ("commit_unresolvable", "commit_unreachable", "commit_content_differs")
    for f in latest_findings:
        if f["axis"] in true_axes and str(f["metric_id"]) not in true_open:
            true_open.append(str(f["metric_id"]))

    counts = {
        "ledger_rows": len(rows),
        "distinct_metrics": len(latest_index),
        "latest_rows": len(latest_records),
        "superseded_rows": len(row_records) - len(latest_records),
        "distinct_git_shas": len(shas),
        "distinct_source_paths": len(src_paths),
        "distinct_source_json_sha256": len(sha_to_paths),
        "findings_total": len(findings),
        "findings_latest_blocking": len(blocking_latest),
        "findings_latest_advisory": len(advisory_latest),
        "findings_superseded_blocking": sum(
            1 for f in superseded_findings if f["severity"] == "blocking"
        ),
        "findings_superseded_advisory": sum(
            1 for f in superseded_findings if f["severity"] == "advisory"
        ),
        "blocking_latest_metrics": len({f["metric_id"] for f in blocking_latest}),
        "blocking_latest_metrics_by_axis": per_axis_metric_counts,
        "strict_rule_open_metrics": len(strict_axis_open),
        "provenance_true_rule_open_metrics": len(true_open),
        "axis_counts_all_rows": _tally(row_records),
        "axis_counts_latest_rows": _tally(latest_records),
    }

    return {
        "status": "FAIL" if blocking_latest else "PASS",
        "ledger_path": str(ledger_path),
        "repo_root": str(repo_root),
        "counts": counts,
        "gate_verdict": {
            "strict_axes": list(BLOCKING_AXES),
            "strict_open_metrics": strict_axis_open,
            "strict_open_count": len(strict_axis_open),
            "provenance_true_axes": list(true_axes),
            "provenance_true_open_metrics": sorted(true_open),
            "provenance_true_open_count": len(true_open),
        },
        "metrics": {r["metric_id"]: r for r in latest_records},
        "rows": row_records,
        "findings": findings,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Audit the run ledger's git provenance (read-only, CPU-only)."
    )
    ap.add_argument("--ledger", type=Path, default=None, help="ledger JSONL path")
    ap.add_argument("--repo-root", type=Path, default=None, help="git work tree")
    ap.add_argument("--out", type=Path, default=None, help="write the audit JSON here")
    ap.add_argument("--quiet", action="store_true", help="print only the summary")
    args = ap.parse_args(argv)

    rep = audit_ledger(ledger_path=args.ledger, repo_root=args.repo_root)
    text = json.dumps(rep, indent=2, sort_keys=True, ensure_ascii=False)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    if args.quiet:
        print(json.dumps({"status": rep["status"], "counts": rep["counts"]},
                         indent=2, sort_keys=True))
    else:
        print(text)
    return 0 if rep["status"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
