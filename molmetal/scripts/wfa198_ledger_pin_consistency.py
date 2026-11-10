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

"""Wf-A198 -- the run ledger's VERSION PIN is a PAIR; only the sha half is checked.

WHY THIS EXISTS
---------------
``molmetal/reports/run_ledger.jsonl`` pins every recorded metric value to a
VERSION with a PAIR of columns:

* ``git_sha``          -- the commit the row claims the run came from, and
* ``source_json_sha256`` -- the sha256 of the ``source_json`` file at that
  version.

The repository's pre-flight gate ``p3_run_ledger_provenance``
(``molmetal/scripts/pre_flight_check.py:9986``) delegates to
``molmetal/scripts/run_ledger.py::uncommitted_provenance`` (defined at
``:469``), which groups the rows by the pair ``(source_json,
source_json_sha256)`` and classifies each SHA as

* ``on_disk``      -- it equals the content of the file on disk,
* ``committed``    -- it equals the content of SOME committed version of that
  path,
* ``uncommitted``  -- it equals neither.

**It never reads the ``git_sha`` column.**  So the gate validates HALF of the
pin.  A row whose ``git_sha`` does not point at the version its
``source_json_sha256`` names is INVISIBLE to it: the SHA half can be
"reproducible" (it matches disk, or matches some OTHER commit) while the
commit half names a different, unreproducible version.  Checking out the
``git_sha`` a row names does NOT reproduce the version the row records.

WHAT IT MEASURES
----------------
For every ledger row that has BOTH a ``git_sha`` and a ``source_json_sha256``:

* ``blob_at_commit``  -- the git blob id of ``source_json`` at ``git_sha``
  (``git cat-file --batch-check`` with ``<git_sha>:<path>``);
* ``sha_at_commit``   -- the sha256 of that blob's CONTENT
  (``git cat-file --batch`` + sha256);
* ``exists_at_commit`` -- whether ``source_json`` exists at ``git_sha`` AT
  ALL;
* ``pin_consistent``  -- ``exists_at_commit AND sha_at_commit ==
  source_json_sha256`` -- i.e. checking out the pinned commit DOES reproduce
  the recorded version;
* ``gate_classification`` -- the gate's OWN three-way verdict for the row's
  SHA, replicated here from the same primitives (disk first, then committed
  history, then neither), so the two views can be compared;
* ``sha_reproducible_elsewhere`` -- the row's SHA equals the content of SOME
  committed version of that path (the notion METHOD calls "reproducible
  elsewhere").

READ-ONLY BY CONSTRUCTION
-------------------------
This module opens NOTHING for writing.  It has no ``open(..., "w")``, no
``write_text``, no ``mkdir``, no ``unlink``; it only READS the ledger and the
git object store, and it SHELLS OUT to read-only git plumbing
(``cat-file``, ``log``, ``ls-files``, ``check-ignore``).  It never writes to
the ledger: the ledger is append-only, and the rows this audit finds
inconsistent are REPORTED, never rewritten.

NOT A GATE -- DELIBERATELY
---------------------------
This module is **NOT** wired into ``molmetal/scripts/pre_flight_check.py`` and
does **NOT** change ``pre_flight_check.EXPECTED_GATE_COUNTS``.  Registering a
new pre-flight gate is a USER decision (see the Wf-A172 / catalogue 6.267
dead-gate history).  This is a REPORTING surface: run it, read the numbers,
decide separately whether the pin should become blocking.

USAGE
-----
    CUDA_VISIBLE_DEVICES="" .venv/bin/python \
        molmetal/scripts/wfa198_ledger_pin_consistency.py            # human summary
    CUDA_VISIBLE_DEVICES="" .venv/bin/python \
        molmetal/scripts/wfa198_ledger_pin_consistency.py --json     # machine summary
    CUDA_VISIBLE_DEVICES="" .venv/bin/python \
        molmetal/scripts/wfa198_ledger_pin_consistency.py --cli-compare
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = REPO_ROOT / "molmetal" / "reports" / "run_ledger.jsonl"
RUN_LEDGER_CLI = REPO_ROOT / "molmetal" / "scripts" / "run_ledger.py"

SCHEMA = "wfa198.ledger_pin_consistency.v1"

#: The gate's three-way classification, replicated.  The ORDER is the gate's
#: own order (``run_ledger.uncommitted_provenance._classify``, at
#: ``molmetal/scripts/run_ledger.py:523``): disk FIRST, then committed
#: history.  A SHA that is both on disk and in some commit is therefore
#: reported as ``on_disk`` by the gate -- not as ``committed``.
CLASS_DISK_FIRST = "disk_first"

#: The token the ledger CLI prints for a quantity it did NOT measure.
NOT_MEASURED = "NOT_MEASURED"

_GIT_TIMEOUT = 600
_ZERO_SHA = "0" * 40


# ---------------------------------------------------------------------------
# small read-only helpers
# ---------------------------------------------------------------------------
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path, *, chunk: int = 1 << 20) -> Optional[str]:
    """sha256 of a file, or None when it is not a regular file."""
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _repo_relative(source_json: str, repo_root: Path) -> str:
    """The repo-relative path of a ledger ``source_json``.

    Mirrors ``run_ledger._resolve_source_json``: a path with a slash (or a
    ``.json`` suffix) is repo-relative; a bare stem resolves under
    ``metrics/by_metric/``.  An ABSOLUTE path is made relative to
    ``repo_root`` when it lives under it (git object names are always
    repo-relative); otherwise it is returned unchanged and the row is
    reported as unresolvable rather than silently mis-looked-up.
    """
    p = Path(source_json)
    if p.is_absolute():
        try:
            return str(p.relative_to(repo_root))
        except ValueError:
            return source_json
    if "/" in source_json or p.suffix == ".json":
        return source_json
    return f"metrics/by_metric/{source_json}.json"


def _git(
    repo_root: Path,
    args: List[str],
    *,
    input_text: Optional[str] = None,
    timeout: int = _GIT_TIMEOUT,
) -> subprocess.CompletedProcess:
    """Run a read-only git command.  Never mutates the work tree."""
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        input=input_text.encode("utf-8") if input_text is not None else None,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _load_rows(ledger_path: Path) -> Tuple[List[Dict[str, Any]], int]:
    """Read the ledger JSONL.  Returns (rows, n_corrupt_lines)."""
    rows: List[Dict[str, Any]] = []
    corrupt = 0
    with ledger_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                corrupt += 1
                continue
            if isinstance(obj, dict):
                rows.append(obj)
            else:
                corrupt += 1
    return rows, corrupt


def _blobs_at_commit(
    repo_root: Path, pairs: List[Tuple[str, str]]
) -> Dict[Tuple[str, str], Optional[str]]:
    """git blob id of each ``<git_sha>:<path>``, or None when absent.

    ONE ``git cat-file --batch-check`` call for every distinct pair.
    """
    uniq = sorted(set(pairs))
    out: Dict[Tuple[str, str], Optional[str]] = {}
    if not uniq:
        return out
    payload = "\n".join(f"{g}:{p}" for g, p in uniq) + "\n"
    proc = _git(repo_root, ["cat-file", "--batch-check"], input_text=payload)
    lines = proc.stdout.decode("utf-8", "replace").splitlines()
    if len(lines) != len(uniq):
        raise RuntimeError(
            f"git cat-file --batch-check returned {len(lines)} lines for "
            f"{len(uniq)} inputs (rc={proc.returncode})"
        )
    for (g, p), line in zip(uniq, lines):
        if line.endswith(" missing"):
            out[(g, p)] = None
            continue
        parts = line.rsplit(" ", 2)
        out[(g, p)] = parts[0] if len(parts) == 3 and parts[1] == "blob" else None
    return out


def _blob_content_shas(
    repo_root: Path, blob_shas: List[str]
) -> Dict[str, str]:
    """sha256 of each blob's CONTENT, in ONE ``git cat-file --batch`` call.

    Byte-for-byte the same computation as
    ``run_ledger._blob_content_shas`` -- the "is this SHA a committed version
    of that path?" set MUST be computed identically, or the two views would
    not be comparable.
    """
    uniq = sorted({b for b in blob_shas if b})
    out: Dict[str, str] = {}
    if not uniq:
        return out
    payload = "\n".join(uniq) + "\n"
    proc = _git(repo_root, ["cat-file", "--batch"], input_text=payload)
    data = proc.stdout
    pos = 0
    for want in uniq:
        nl = data.find(b"\n", pos)
        if nl < 0:
            break
        header = data[pos:nl].decode("utf-8", "replace").split()
        if len(header) < 3:
            break
        size = int(header[2])
        body = data[nl + 1 : nl + 1 + size]
        out[want] = hashlib.sha256(body).hexdigest()
        pos = nl + 1 + size + 1
    return out


def _committed_blob_shas(
    repo_root: Path, paths: List[str]
) -> Dict[str, List[str]]:
    """Every committed CONTENT version of each path, as git blob ids.

    ONE ``git log --all --format= --raw --no-abbrev --no-renames -- <paths>``
    call: the raw diff lines carry the blob sha of each committed version of
    the file.  A path with NO git history at all (an untracked file)
    contributes no entry -- which is itself a finding, not an error.

    ``--diff-merges=first-parent`` is REQUIRED (2026-11-04, Wf-A269
    wave-22): ``git log --raw`` does **not** diff a MERGE commit by default,
    so a file whose only committed versions arrive through a merge is
    INVISIBLE to this query.  MEASURED on this repository's own ledger: the
    flag adds **45** of the **401** referenced paths (397 vs 352 paths with
    history; 754 vs 700 blobs), and the merge ``8289a7f5`` alone holds
    **561** paths present in NEITHER of its parents.  Without it this
    function UNDER-COLLECTS the committed set, so a row whose sha IS
    reproducible from a commit is reported as "not reproducible" (a FALSE
    NEGATIVE on the auditor's own per-row verdict).

    THE CLASS WAS ALREADY DIAGNOSED IN THIS REPO -- in the SIBLING module
    ``molmetal/scripts/wfa156_ledger_provenance_guard.py``, whose docstring
    says verbatim: *"``-m`` is load-bearing: without it git prints no diff
    for a MERGE commit, so a content version introduced by a merge is
    invisible."*  That sibling and
    ``molmetal/scripts/wfa166_run_ledger_census.py`` carry the flag; this
    module and ``molmetal/scripts/run_ledger.py`` did not.  MEASURED on
    this repository: ``-m`` (``--diff-merges=separate``) and
    ``--diff-merges=first-parent`` return the IDENTICAL set (397 paths /
    754 blobs each); ``--diff-merges=first-parent`` is used here so that all
    four ledger-provenance implementations answer the same question the
    same way.  See ``docs/HONEST_NEGATIVE_CATALOG.md`` §6.269.33.
    """
    out: Dict[str, set] = {}
    if not paths:
        return {}
    proc = _git(
        repo_root,
        [
            "log",
            "--all",
            "--format=",
            "--raw",
            "--no-abbrev",
            "--no-renames",
            "--diff-merges=first-parent",
            "--",
            *paths,
        ],
    )
    for line in proc.stdout.decode("utf-8", "replace").splitlines():
        if not line.startswith(":"):
            continue
        meta, _, path = line.partition("\t")
        parts = meta.split()
        if len(parts) < 4 or not path:
            continue
        new_sha = parts[3]
        if new_sha == _ZERO_SHA:
            continue  # a deletion, not a version
        out.setdefault(path, set()).add(new_sha)
    # API-SHAPE CONTRACT (2026-11-04, Wf-A269 wave-23): an entry for EVERY
    # REQUESTED path.  A path with no committed history maps to an EMPTY
    # list, meaning "this path has no committed version" — NOT "this path
    # was not asked about".  The contract is normalised at the API so the
    # five ledger-provenance implementations can be compared without a
    # comparison-site compensation (a naive comparison used to manufacture a
    # 4-path "disagreement" that was pure API shape).
    for path in paths:
        out.setdefault(path, set())
    return {p: sorted(v) for p, v in out.items()}


def _tracked_paths(repo_root: Path, paths: List[str]) -> List[str]:
    """Which of ``paths`` are TRACKED by git.  ONE ``git ls-files`` call."""
    if not paths:
        return []
    proc = _git(repo_root, ["ls-files", "--", *paths])
    return [ln for ln in proc.stdout.decode("utf-8", "replace").splitlines() if ln]


def _ignored_paths(repo_root: Path, paths: List[str]) -> Dict[str, str]:
    """Which of ``paths`` are GITIGNORED, and by which rule.

    ONE ``git check-ignore -v --stdin`` call.  The ``-v`` output is
    ``<source>:<linenum>:<pattern>\\t<pathname>``.
    """
    out: Dict[str, str] = {}
    if not paths:
        return out
    payload = "\n".join(paths) + "\n"
    proc = _git(repo_root, ["check-ignore", "-v", "--stdin"], input_text=payload)
    for line in proc.stdout.decode("utf-8", "replace").splitlines():
        rule, _, path = line.partition("\t")
        if path:
            out[path] = rule
    return out


# ---------------------------------------------------------------------------
# the audit
# ---------------------------------------------------------------------------
def audit(
    ledger_path: Optional[Path] = None,
    *,
    repo_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Measure the ledger's version-pin consistency.  READ-ONLY.

    Returns a JSON-serialisable report.
    """
    t0 = time.perf_counter()
    ledger_path = Path(ledger_path) if ledger_path else LEDGER_PATH
    repo_root = Path(repo_root) if repo_root else REPO_ROOT

    ledger_sha = _sha256_file(ledger_path)
    rows, corrupt = _load_rows(ledger_path)

    pinned: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows):
        git_sha = row.get("git_sha")
        sha = row.get("source_json_sha256")
        src = row.get("source_json")
        if not (git_sha and sha and src):
            continue
        pinned.append(
            {
                "index": idx,
                "metric_id": row.get("metric_id"),
                "source_json": str(src),
                "repo_relative": _repo_relative(str(src), repo_root),
                "git_sha": str(git_sha),
                "source_json_sha256": str(sha),
                "timestamp": row.get("timestamp"),
                "value": row.get("value"),
            }
        )

    # --- the pin half: what the file at the PINNED COMMIT actually is -----
    blobs = _blobs_at_commit(
        repo_root, [(r["git_sha"], r["repo_relative"]) for r in pinned]
    )
    blob_ids = sorted({b for b in blobs.values() if b})
    content_shas = _blob_content_shas(repo_root, blob_ids)

    # --- the sha half: is the SHA reproducible ANYWHERE? --------------------
    src_paths = sorted({r["repo_relative"] for r in pinned})
    committed_blobs = _committed_blob_shas(repo_root, src_paths)
    all_committed = sorted({b for v in committed_blobs.values() for b in v})
    committed_content = _blob_content_shas(repo_root, all_committed)
    committed_versions: Dict[str, List[str]] = {}
    for path in src_paths:
        shas = {
            committed_content[b]
            for b in committed_blobs.get(path, [])
            if b in committed_content
        }
        committed_versions[path] = sorted(shas)

    disk_sha: Dict[str, Optional[str]] = {
        path: _sha256_file(repo_root / path) for path in src_paths
    }

    def gate_classification(path: str, sha: str) -> str:
        """The GATE's own verdict -- disk FIRST, then committed history.

        Mirrors ``run_ledger.uncommitted_provenance._classify``
        (molmetal/scripts/run_ledger.py:523) exactly, including the order.
        """
        if disk_sha.get(path) == sha:
            return "on_disk"
        if sha in set(committed_versions.get(path, [])):
            return "committed"
        return "uncommitted"

    def gate_classification_committed_first(path: str, sha: str) -> str:
        """The SAME three-way verdict with the OPPOSITE precedence.

        Reported as a SECONDARY view only: a SHA that is both on disk and in
        some commit is ``on_disk`` under the gate's order and ``committed``
        under this one.  The set of REPRODUCIBLE rows is identical either
        way; only the split moves.
        """
        if sha in set(committed_versions.get(path, [])):
            return "committed"
        if disk_sha.get(path) == sha:
            return "on_disk"
        return "uncommitted"

    verdicts: List[Dict[str, Any]] = []
    for r in pinned:
        key = (r["git_sha"], r["repo_relative"])
        blob = blobs.get(key)
        exists = blob is not None
        sha_at = content_shas.get(blob) if blob else None
        cls = gate_classification(r["repo_relative"], r["source_json_sha256"])
        verdicts.append(
            {
                **r,
                "exists_at_commit": exists,
                "blob_at_commit": blob,
                "sha256_at_commit": sha_at,
                "pin_consistent": bool(exists and sha_at == r["source_json_sha256"]),
                "sha_matches_disk": disk_sha.get(r["repo_relative"])
                == r["source_json_sha256"],
                "sha_matches_a_committed_version": r["source_json_sha256"]
                in set(committed_versions.get(r["repo_relative"], [])),
                "sha_reproducible_elsewhere": r["source_json_sha256"]
                in set(committed_versions.get(r["repo_relative"], [])),
                "gate_classification": cls,
                "gate_classification_committed_first": gate_classification_committed_first(
                    r["repo_relative"], r["source_json_sha256"]
                ),
                "gate_says_reproducible": cls in ("on_disk", "committed"),
                "n_committed_versions_of_path": len(
                    committed_versions.get(r["repo_relative"], [])
                ),
                "disk_sha256": disk_sha.get(r["repo_relative"]),
            }
        )

    inconsistent = [v for v in verdicts if not v["pin_consistent"]]
    mismatch = [v for v in inconsistent if v["exists_at_commit"]]
    absent = [v for v in inconsistent if not v["exists_at_commit"]]

    def _tally(rows_: List[Dict[str, Any]]) -> Dict[str, int]:
        out = {"on_disk": 0, "committed": 0, "uncommitted": 0}
        for v in rows_:
            out[v["gate_classification"]] = out.get(v["gate_classification"], 0) + 1
        return out

    def _tally_cf(rows_: List[Dict[str, Any]]) -> Dict[str, int]:
        out = {"on_disk": 0, "committed": 0, "uncommitted": 0}
        for v in rows_:
            k = v["gate_classification_committed_first"]
            out[k] = out.get(k, 0) + 1
        return out

    # --- tracking / gitignore status of every offending path --------------
    offending_paths = sorted({v["repo_relative"] for v in inconsistent})
    tracked = set(_tracked_paths(repo_root, offending_paths))
    ignored = _ignored_paths(repo_root, offending_paths)
    path_status = []
    for path in offending_paths:
        path_status.append(
            {
                "source_json": path,
                "n_offending_rows": sum(
                    1 for v in inconsistent if v["repo_relative"] == path
                ),
                "tracked_by_git": path in tracked,
                "gitignored": path in ignored,
                "gitignore_rule": ignored.get(path),
                "n_committed_versions": len(committed_versions.get(path, [])),
                "exists_on_disk": (repo_root / path).is_file(),
                "disk_sha256": disk_sha.get(path),
            }
        )

    counts = {
        "ledger_rows": len(rows),
        "ledger_corrupt_lines": corrupt,
        "rows_with_a_complete_pin": len(pinned),
        "pin_consistent": len(verdicts) - len(inconsistent),
        "pin_inconsistent": len(inconsistent),
        "sha_mismatch_exists_at_commit": len(mismatch),
        "absent_at_commit": len(absent),
        "gate_classification_of_sha_mismatch": _tally(mismatch),
        "gate_classification_of_absent_at_commit": _tally(absent),
        "gate_classification_of_all_inconsistent": _tally(inconsistent),
        "gate_classification_of_sha_mismatch_committed_first": _tally_cf(mismatch),
        "gate_classification_of_all_inconsistent_committed_first": _tally_cf(
            inconsistent
        ),
        "gate_says_reproducible__sha_mismatch": sum(
            1 for v in mismatch if v["gate_says_reproducible"]
        ),
        "gate_says_reproducible__all_inconsistent": sum(
            1 for v in inconsistent if v["gate_says_reproducible"]
        ),
        "distinct_offending_paths": len(offending_paths),
        "distinct_pinned_commits": len({v["git_sha"] for v in pinned}),
        "distinct_source_paths": len(src_paths),
    }

    return {
        "schema": SCHEMA,
        "generated_at": _now_iso(),
        "read_only": True,
        "statement": (
            "READ-ONLY MEASUREMENT. No ledger row was written, rewritten or "
            "deleted; molmetal/reports/run_ledger.jsonl is APPEND-ONLY and "
            "was read, never touched. The rows this audit reports as "
            "inconsistent are REPORTED, not repaired."
        ),
        "repo_root": str(repo_root),
        "ledger_path": str(ledger_path),
        "ledger_sha256": ledger_sha,
        "ledger_bytes": ledger_path.stat().st_size if ledger_path.is_file() else None,
        "wall_time_sec": round(time.perf_counter() - t0, 3),
        "counts": counts,
        "offending_rows": inconsistent,
        "offending_paths": path_status,
        "row_verdicts": verdicts,
    }


# ---------------------------------------------------------------------------
# the CLI reporting defect (C)
# ---------------------------------------------------------------------------
def cli_compare(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """Run the ledger CLI's ``provenance`` in BOTH modes and capture stdout.

    READ-ONLY: the CLI's ``provenance`` command only reads.
    """
    repo_root = Path(repo_root) if repo_root else REPO_ROOT
    cli_path = repo_root / "molmetal" / "scripts" / "run_ledger.py"
    out: Dict[str, Any] = {
        "cli": str(cli_path),
        "note": (
            "The CLI's DEFAULT mode used to print a NUMERIC superseded count "
            "for a quantity it did not compute: `uncommitted_provenance()` "
            "filters superseded-only metrics out of `detail` unless "
            "`include_superseded=True`, so `superseded_uncommitted_rows` is "
            "0 in default mode BY CONSTRUCTION, not by measurement."
        ),
    }
    for mode, argv in (("default", ["provenance"]), ("all", ["provenance", "--all"])):
        try:
            proc = subprocess.run(
                [sys.executable, str(cli_path), *argv],
                cwd=str(repo_root),
                capture_output=True,
                text=True,
                timeout=_GIT_TIMEOUT,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            out[mode] = {"argv": argv, "error": f"{type(exc).__name__}: {exc}"}
            continue
        lines = proc.stdout.splitlines()
        out[mode] = {
            "argv": argv,
            "returncode": proc.returncode,
            "summary_line": lines[0] if lines else "",
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    return out


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def render_summary(report: Dict[str, Any], *, max_items: int = 40) -> str:
    c = report.get("counts") or {}
    lines: List[str] = []
    lines.append("Wf-A198 -- ledger version-pin consistency (READ-ONLY)")
    lines.append(
        "ledger: {p}".format(
            p=report.get("ledger_path"),
        )
    )
    lines.append(
        "ledger sha256: {s} | rows={r} | rows with a complete pin={p}".format(
            s=report.get("ledger_sha256"),
            r=c.get("ledger_rows"),
            p=c.get("rows_with_a_complete_pin"),
        )
    )
    lines.append(
        "PIN CONSISTENT = {ok} | PIN INCONSISTENT = {bad} of {tot} rows".format(
            ok=c.get("pin_consistent"),
            bad=c.get("pin_inconsistent"),
            tot=c.get("rows_with_a_complete_pin"),
        )
    )
    lines.append(
        "  of the inconsistent: sha-mismatch (file exists at the commit, "
        "different content) = {m} | ABSENT at the commit = {a}".format(
            m=c.get("sha_mismatch_exists_at_commit"),
            a=c.get("absent_at_commit"),
        )
    )
    gm = c.get("gate_classification_of_sha_mismatch") or {}
    ga = c.get("gate_classification_of_all_inconsistent") or {}
    lines.append(
        "  gate classification of the {m} sha-mismatch rows: on_disk={od} "
        "committed={cm} uncommitted={uc}  ->  {rep} are REPRODUCIBLE under "
        "the gate's own classification, so the gate cannot see them".format(
            m=c.get("sha_mismatch_exists_at_commit"),
            od=gm.get("on_disk"),
            cm=gm.get("committed"),
            uc=gm.get("uncommitted"),
            rep=c.get("gate_says_reproducible__sha_mismatch"),
        )
    )
    lines.append(
        "  gate classification of ALL {n} inconsistent rows: on_disk={od} "
        "committed={cm} uncommitted={uc}  ->  {rep} reproducible".format(
            n=c.get("pin_inconsistent"),
            od=ga.get("on_disk"),
            cm=ga.get("committed"),
            uc=ga.get("uncommitted"),
            rep=c.get("gate_says_reproducible__all_inconsistent"),
        )
    )
    lines.append("")
    lines.append("OFFENDING PATHS ({n}):".format(n=len(report.get("offending_paths") or [])))
    for p in report.get("offending_paths") or []:
        lines.append(
            "  - {p}  rows={r} tracked={t} gitignored={g} rule={rule}".format(
                p=p.get("source_json"),
                r=p.get("n_offending_rows"),
                t=p.get("tracked_by_git"),
                g=p.get("gitignored"),
                rule=p.get("gitignore_rule") or "-",
            )
        )
    lines.append("")
    lines.append(
        "OFFENDING ROWS ({n}):".format(n=len(report.get("offending_rows") or []))
    )
    for v in (report.get("offending_rows") or [])[:max_items]:
        lines.append(
            "  - [{i}] {m}  {p}".format(
                i=v.get("index"), m=v.get("metric_id"), p=v.get("source_json")
            )
        )
        lines.append(
            "      git_sha {g} | recorded sha {r} | sha AT THAT COMMIT {a}".format(
                g=v.get("git_sha"),
                r=v.get("source_json_sha256"),
                a=v.get("sha256_at_commit") or "<PATH ABSENT AT THAT COMMIT>",
            )
        )
        lines.append(
            "      gate classification: {c} | committed versions of the path: {n}".format(
                c=v.get("gate_classification"),
                n=v.get("n_committed_versions_of_path"),
            )
        )
    lines.append("")
    lines.append(
        "READ-ONLY: no ledger row was written; the ledger is append-only. "
        "Not registered as a pre-flight gate; EXPECTED_GATE_COUNTS unchanged."
    )
    lines.append("wall time: {w}s".format(w=report.get("wall_time_sec")))
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Audit the run ledger's VERSION PIN (git_sha + source_json_sha256) "
            "for pair consistency. READ-ONLY, CPU-only, not a pre-flight gate."
        )
    )
    ap.add_argument("--ledger", type=Path, default=None, help="ledger JSONL path")
    ap.add_argument("--repo-root", type=Path, default=None, help="git work tree")
    ap.add_argument("--json", action="store_true", help="print the full report as JSON")
    ap.add_argument(
        "--cli-compare",
        action="store_true",
        help="also run the ledger CLI's `provenance` in both modes and capture stdout",
    )
    args = ap.parse_args(argv)

    report = audit(args.ledger, repo_root=args.repo_root)
    if args.cli_compare:
        report["cli_reporting"] = cli_compare(args.repo_root)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print(render_summary(report))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
