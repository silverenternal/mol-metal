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

"""Run ledger — append-only provenance for every metric value.

WHY THIS EXISTS
---------------
``metrics/by_metric/*.json`` records a "current best" for each metric but keeps
NO history: there is no way to tell *when* a number was produced or *from which*
git / data / config state.  ``registry/metrics.yaml`` (the SSOT) carries a
``source_json_sha256`` and a ``last_refreshed`` stamp, but it is regenerated in
place — the previous value is overwritten.  So "which number is real, and when
did it become real?" cannot be answered from disk.

The fix is a single append-only ledger.  Every time a metric value is recorded,
one JSON line is appended to ``molmetal/reports/run_ledger.jsonl`` carrying the
value, the source JSON path, and the provenance hashes (git SHAs + data hash +
config hash) that identify the reproducibility state.  The file is *never*
rewritten; ``latest()`` and ``history()`` read it back.  ``staleness()`` then
answers "is the registry's current number still backed by the newest run?" by
comparing the ledger's latest entry against each registry row.

This is the executable-provenance counterpart to AGENTS.md §5.2 (honest framing
— every number traces to a run) + §5.5 (filesystem-truth) + §5.6 (atomic write,
append-only here).

HONEST FRAMING (AGENTS.md §5.2): every hash written here is MEASURED on the host
at record time; the seed pass copies values verbatim from the source JSON named
beside them and marks them `seed=True`.  Nothing is PROJECTED.  Seeding does NOT
invent history — it records exactly one baseline entry per registry metric from
the current on-disk state.

APPEND-ONLY CONTRACT: ``record_run`` opens the ledger in ``"a"`` mode and writes
after ``fsync``; it never truncates or rewrites.  A single write of one line is
atomic enough for O_APPEND on POSIX (writes < PIPE_BUF are not interleaved).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - PyYAML is a hard dep of this repo
    yaml = None  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[2]
LEDGER_PATH = REPO_ROOT / "molmetal" / "reports" / "run_ledger.jsonl"
REGISTRY_METRICS_PATH = REPO_ROOT / "registry" / "metrics.yaml"
REGISTRY_DATASETS_PATH = REPO_ROOT / "registry" / "datasets.yaml"
BY_METRIC_DIR = REPO_ROOT / "metrics" / "by_metric"

#: env var that turns the staleness pre-flight gate from report-only (SKIP) to
#: enforced.  Opt-in per the AGENTS.md §5.6/§5.7 new-gate pattern.
RUN_LEDGER_ENFORCE_ENV = "WF_RUN_LEDGER"

#: Token the CLI prints for a quantity it did NOT measure.  Wf-A198 (2026-11-02):
#: the ledger CLI's DEFAULT `provenance` mode printed a numeric
#: ``superseded_never_committed=0`` for a quantity it never computed -- the
#: superseded-only metrics are filtered out of ``detail`` before the count is
#: taken unless ``--all`` is passed -- so a reader of the default output
#: reported "0 superseded rows" when the truth was 9.  A number that was not
#: measured is now printed as a token, never as a digit.
NOT_MEASURED_TOKEN = "NOT_MEASURED"


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path, *, chunk: int = 1 << 20) -> Optional[str]:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _git_sha(repo_root: Path = REPO_ROOT) -> Optional[str]:
    """Return the current HEAD sha256-style commit id, or None outside git."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return None
    if out.returncode != 0:
        return None
    sha = out.stdout.strip()
    return sha or None


def _resolve_source_json(source_json: str, repo_root: Path = REPO_ROOT) -> Path:
    """Resolve a source_json string to an absolute path.

    Accepts a repo-relative path (``metrics/by_metric/x.json``) or a bare stem
    (``x`` / ``x.json``), which resolves under ``metrics/by_metric/``.
    """
    p = Path(source_json)
    if p.is_absolute():
        return p
    if "/" in source_json or p.suffix == ".json":
        return repo_root / p
    return BY_METRIC_DIR / f"{source_json}.json"


def _resolve_data_source(data_source: str, repo_root: Path = REPO_ROOT) -> Optional[Path]:
    """Resolve a data_source string (possibly prose) to an on-disk file.

    ``data_source`` in the metric JSONs is often a human sentence, e.g.
    ``"MetalCytoToxDB.csv (precious-metal subset)"``.  Take the leading token
    up to whitespace / paren and try to resolve it against a few known roots;
    return the first that exists, else None.  Never fabricate a path.
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


def _repo_relative(path: Path, repo_root: Path = REPO_ROOT) -> str:
    """Render ``path`` relative to the repo root when it is inside it.

    A ledger row that stores an ABSOLUTE path is a host-bound locator: it
    resolves on the machine that wrote it and nowhere else, which is the
    §6.269.13 HOST-BOUND-PATH defect class.  A repo-relative path survives a
    clone.  A path outside the repo is stored absolute — there is no
    repo-relative rendering of it, and pretending otherwise would fabricate a
    path that does not exist.
    """
    p = Path(path)
    try:
        return str(p.resolve().relative_to(Path(repo_root).resolve()))
    except (ValueError, OSError):
        return str(p)


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _append_line(entry: Dict[str, Any], ledger_path: Path = LEDGER_PATH) -> Path:
    """Atomically append ONE JSON line to the ledger.  Never rewrites."""
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, sort_keys=True, ensure_ascii=False) + "\n"
    # O_APPEND: a single write of a short line is not interleaved with other
    # appenders; fsync flushes before we return so a crash cannot lose it.
    with ledger_path.open("a", encoding="utf-8") as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())
    return ledger_path


def record_run(
    metric_id: str,
    value: Any,
    *,
    source_json: str,
    git_sha: Optional[str] = None,
    data_sha: Optional[str] = None,
    config_sha: Optional[str] = None,
    data_source: Optional[str] = None,
    config_path: Optional[str] = None,
    protocol_id: Optional[str] = None,
    host: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
    source_json_sha256: Optional[str] = None,
    timestamp: Optional[str] = None,
    seed: bool = False,
    ledger_path: Path = LEDGER_PATH,
    repo_root: Path = REPO_ROOT,
) -> Dict[str, Any]:
    """Append ONE entry to the ledger and return the entry that was written.

    ``git_sha`` defaults to ``git rev-parse HEAD`` when not supplied; a
    ``None`` result (outside a git tree) is recorded as ``null`` rather than
    fabricated.  ``source_json_sha256`` defaults to the sha256 of ``source_json``.
    ``data_sha`` / ``config_sha`` are used verbatim when supplied; otherwise they
    are derived by hashing ``data_source`` (a metric ``data_source`` string, a
    possibly-prose value — its leading path token is resolved against the repo)
    and ``config_path`` when those resolve to an on-disk file.  An unresolvable
    data_source / config_path leaves the hash ``None`` rather than fabricated.
    """
    if not metric_id:
        raise ValueError("record_run requires a metric_id")
    src_path = _resolve_source_json(source_json, repo_root=repo_root)
    if source_json_sha256 is None:
        source_json_sha256 = _sha256_file(src_path)
    if git_sha is None:
        git_sha = _git_sha(repo_root)
    if data_sha is None and data_source:
        dp = _resolve_data_source(data_source, repo_root=repo_root)
        if dp is not None:
            data_sha = _sha256_file(dp)
    if config_sha is None and config_path:
        cp = Path(config_path)
        if not cp.is_absolute():
            cp = repo_root / cp
        if cp.is_file():
            config_sha = _sha256_file(cp)
    entry: Dict[str, Any] = {
        "metric_id": metric_id,
        "value": value,
        "source_json": source_json,
        "source_json_sha256": source_json_sha256,
        "git_sha": git_sha,
        "data_sha": data_sha,
        "config_sha": config_sha,
        "protocol_id": protocol_id,
        "host": host if host is not None else os.uname().nodename,
        "timestamp": timestamp if timestamp is not None else _now_utc(),
        "seed": bool(seed),
    }
    # Wf-A269 (2026-11-04) — persist the CONFIG PATH next to the config hash.
    #
    # The API accepted ``config_path`` (:184) and two call sites passed it
    # (:655 ``seed_from_disk``, :748 ``repoint_metric``), but the written row
    # DROPPED it: a row carried a bare ``config_sha`` with no pointer to the
    # file it hashed.  MEASURED 2026-11-04 (Wf-A269 wave-5 ledger audit):
    # 434 ledger rows, 6 with a non-null ``config_sha``, 3 distinct hashes,
    # **0** carrying a config path; all 3 hashes resolve to HISTORICAL git
    # blobs and 0 of them match the file on disk today.  A hash without a
    # path cannot be re-resolved, so the drift was undetectable by any gate.
    #
    # The path is stored REPO-RELATIVE so the row survives a clone on a
    # different host (an absolute path would be a host-bound locator — the
    # §6.269.13 HOST-BOUND-PATH defect class).
    if config_path:
        entry["config_path"] = _repo_relative(Path(config_path), repo_root)
    if extra:
        entry["extra"] = extra
    _append_line(entry, ledger_path=ledger_path)
    return entry


def _iter_entries(ledger_path: Path = LEDGER_PATH) -> List[Dict[str, Any]]:
    """Read every entry.  Skips blank lines; a corrupt line is surfaced, not
    silently dropped (a corrupt ledger is a real problem, not noise)."""
    if not ledger_path.is_file():
        return []
    entries: List[Dict[str, Any]] = []
    for lineno, raw in enumerate(
        ledger_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw.strip():
            continue
        try:
            entries.append(json.loads(raw))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"corrupt JSON on {ledger_path}:{lineno}: {exc}"
            ) from exc
    return entries


def latest(metric_id: str, ledger_path: Path = LEDGER_PATH) -> Optional[Dict[str, Any]]:
    """Return the most recently appended entry for ``metric_id``, or None."""
    found: Optional[Dict[str, Any]] = None
    for entry in _iter_entries(ledger_path):
        if entry.get("metric_id") == metric_id:
            found = entry
    return found


def history(metric_id: str, ledger_path: Path = LEDGER_PATH) -> List[Dict[str, Any]]:
    """Return every entry for ``metric_id``, in append order (oldest first)."""
    return [e for e in _iter_entries(ledger_path) if e.get("metric_id") == metric_id]


def _load_yaml(path: Path) -> Dict[str, Any]:
    if yaml is None or not path.is_file():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def staleness(
    registry_metrics_path: Path = REGISTRY_METRICS_PATH,
    *,
    ledger_path: Path = LEDGER_PATH,
    repo_root: Path = REPO_ROOT,
) -> Dict[str, Dict[str, Any]]:
    """Compare the ledger's latest entry per metric against the registry row.

    For each metric in ``registry/metrics.yaml``:
      - ``no_run``      — no ledger entry exists for this metric.
      - ``stale``       — the registry's ``source_json_sha256`` differs from the
                          ledger's latest ``source_json_sha256`` **and the value
                          differs too** (the registry was refreshed from a source
                          the ledger has not seen, or the source changed since
                          the ledger entry was written).
      - ``reformatted`` — the hashes differ but the recorded value is identical:
                          the JSON gained a field, lost whitespace or was
                          re-serialised. The number is still the one the ledger
                          saw, so this is not a provenance gap.
      - ``current``     — the latest ledger entry's source_json hash matches the
                          registry row.

    ``reformatted`` exists because MEASURED 2026-10-21 three of four "stale"
    rows (m7_diversity, m2_r2_per_cell, m21_nci60_9cell) carried byte-identical
    values and only the file had been rewritten. A staleness gate that cries
    wolf on a reformat is a gate people stop reading, and the one row that had
    really changed (l5_hilbert_r2, -5.6625 -> -6.7126) was hidden among them.

    Returns ``{metric_id: {status, ledger_timestamp, ledger_source_json_sha256,
    registry_source_json_sha256, ledger_value, registry_value}}``.
    """
    reg = _load_yaml(registry_metrics_path)
    out: Dict[str, Dict[str, Any]] = {}
    for mid, rec in (reg.get("metrics") or {}).items():
        reg_sha = rec.get("source_json_sha256")
        entry = latest(mid, ledger_path=ledger_path)
        info: Dict[str, Any] = {
            "status": "no_run",
            "ledger_timestamp": None,
            "ledger_source_json_sha256": None,
            "registry_source_json_sha256": reg_sha,
            "ledger_value": None,
            "registry_value": rec.get("value"),
        }
        if entry is not None:
            info["ledger_timestamp"] = entry.get("timestamp")
            info["ledger_source_json_sha256"] = entry.get("source_json_sha256")
            info["ledger_value"] = entry.get("value")
            if entry.get("source_json_sha256") == reg_sha:
                info["status"] = "current"
            elif _same_value(entry.get("value"), rec.get("value")):
                info["status"] = "reformatted"
            else:
                info["status"] = "stale"
        out[mid] = info
    return out


def _same_value(a: Any, b: Any) -> bool:
    """True when two recorded values are the same number, list or mapping.

    Compared through a canonical JSON dump rather than ``==``: ``0.123`` and
    ``"0.123"`` are not the same measurement, while ``1`` and ``1.0`` are, and
    a nested dict must compare by content rather than by identity.
    """

    def canon(v: Any) -> str:
        try:
            return json.dumps(v, sort_keys=True, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return repr(v)

    if isinstance(a, bool) != isinstance(b, bool):
        return False
    return canon(a) == canon(b)


def staleness_summary(
    registry_metrics_path: Path = REGISTRY_METRICS_PATH,
    *,
    ledger_path: Path = LEDGER_PATH,
    repo_root: Path = REPO_ROOT,
) -> Dict[str, int]:
    """Counts of current / reformatted / stale / no_run across the registry metrics."""
    counts = {"current": 0, "reformatted": 0, "stale": 0, "no_run": 0}
    for info in staleness(
        registry_metrics_path, ledger_path=ledger_path, repo_root=repo_root
    ).values():
        counts[info["status"]] = counts.get(info["status"], 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Wf-A123 FIX 7 -- the ledger's NEVER-COMMITTED shas
# ---------------------------------------------------------------------------
# WHY THIS CHECK LIVES HERE AND NOT IN registry.py
# ------------------------------------------------
# The LEDGER is the artifact that carries ``source_json_sha256``: it is the
# append-only record of "this value was produced from THAT file, and that
# file hashed to THAT sha".  ``registry.py check`` validates the REGISTRY
# rows (one sha per metric, regenerated in place), and
# :func:`staleness` compares the LEDGER's sha with the REGISTRY's sha --
# NEITHER compares a sha with GIT HISTORY.  A ledger row can therefore point
# at a file version that exists in NO commit, and every existing check still
# reports "current".  This function is the one that asks the missing
# question: does the sha the ledger names match ANY committed version of the
# file it names?
#
# MEASURED 2026-10-31: 9 of 408 distinct (source_json, sha256) pairs in
# molmetal/reports/run_ledger.jsonl match neither the file on disk nor any
# committed version of it.
#
# COST: one ``git log --all --raw`` for every source path in ONE call plus
# ONE ``git cat-file --batch`` for every blob.  MEASURED: 397 paths / 698
# blobs in 0.67 s + 0.9 s on this host.

_ZERO_SHA = "0" * 40


def _committed_blob_shas(repo_root: Path, paths: List[str]) -> Dict[str, List[str]]:
    """Every committed CONTENT version of each path, as git blob shas.

    ONE ``git log --all --format= --raw --no-abbrev --no-renames -- <paths>``
    call: the raw diff lines carry the blob sha of each committed version of
    the file, so no per-path subprocess is needed.  A path with NO git
    history at all (an untracked file) simply contributes no entry -- which
    is itself a finding, not an error.

    ``--diff-merges=first-parent`` is REQUIRED (2026-11-04, Wf-A269
    wave-22): ``git log --raw`` does **not** diff a merge commit by default,
    so a file whose only committed versions arrive through a merge is
    INVISIBLE to this query.  MEASURED on this repository's own ledger: the
    flag adds **45** of the **401** referenced paths (397 vs 352 paths with
    history; 754 vs 700 blobs) — and every one of the 45 is a metric file
    that the repository demonstrably holds in a commit.  Without it this
    function UNDER-COLLECTS the committed set, which makes every consumer
    report a false "never committed" (the independent validator
    ``wfa166_run_ledger_provenance.py`` — which has carried the flag since
    its own ``_history_raw`` — disagreed with this producer by 59 vs 14).
    The independent implementation is the reference; see
    ``docs/HONEST_NEGATIVE_CATALOG.md`` §6.269.33.

    CORRECTION (2026-11-04, Wf-A269 wave-22): the first version of this
    docstring pointed at ``§6.269.26``, which is a DIFFERENT entry ("THE
    CATALOGUE APPEND BROKE THE REPO'S OWN PATH-REFERENCE GUARD").  A false
    citation inside a repair is itself a defect of the §6.269.4b class; it is
    corrected here, in place, to the section that actually carries this
    class (§6.269.33).

    API-SHAPE CONTRACT (2026-11-04, Wf-A269 wave-23): this function returns
    an entry for EVERY REQUESTED path.  A path with no committed history maps
    to an EMPTY list, which means "this path has no committed version" — it
    is NOT the same as "this path was not asked about".  The contract is
    normalised HERE, at the API, rather than at each comparison site: the
    sibling implementations (``wfa156_ledger_provenance_guard.py``,
    ``wfa198_ledger_pin_consistency.py``,
    ``wfa166_run_ledger_census.py``) all return an entry for every requested
    path, so a naive comparison of the five used to manufacture a 4-path
    "disagreement" that was pure API shape and said nothing about git.
    Callers must therefore use ``.get(path, [])`` for lookups but MUST NOT
    treat a missing key as "no history" without the normalisation.
    """
    if not paths:
        return {}
    try:
        proc = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
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
            capture_output=True,
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    out: Dict[str, set] = {}
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
    # The contract: EVERY requested path gets an entry (empty = no history).
    for path in paths:
        out.setdefault(path, set())
    return {p: sorted(s) for p, s in out.items()}


# ---------------------------------------------------------------------------
# REACHABILITY — the whole object database, not only refs-reachable history
# ---------------------------------------------------------------------------
# WHY (2026-11-04, Wf-A269 wave-23): "is this file version committed?" was
# decided over REFS-REACHABLE history only (``git log --all``).  A version that
# WAS committed and is now UNREACHABLE (rewritten away, amended, rebased) was
# therefore invisible, and the auditor reported it as "never committed" — a
# FALSE NEGATIVE on the question and, worse, a finding whose TEXT asserted
# more than the measurement supported.  MEASURED on this repository: the object
# database holds 1011 commit objects while ``git rev-list --all`` reaches 843,
# so 168 commits (16.6 %) are unreachable.
#
# The three-valued verdict below is the fix.  The LEGACY axis
# (``sha_vs_committed`` / ``never_committed``) keeps its NAME and its VALUE —
# it is quoted VERBATIM in ``docs/HONEST_NEGATIVE_CATALOG.md`` and
# ``TODO/INDEX.md`` as ``OPEN_sha_never_committed=14`` /
# ``superseded_sha_never_committed=9`` — and is now labelled REACHABLE-SCOPED
# in the text.  The three-valued axis is ADDED alongside it.
COMMIT_STATUS_COMMITTED = "committed"
COMMIT_STATUS_COMMITTED_UNREACHABLE = "committed_unreachable"
COMMIT_STATUS_NEVER_COMMITTED = "never_committed"
COMMIT_STATUS_VALUES = (
    COMMIT_STATUS_COMMITTED,
    COMMIT_STATUS_COMMITTED_UNREACHABLE,
    COMMIT_STATUS_NEVER_COMMITTED,
)


def _odb_commit_objects(repo_root: Path) -> Set[str]:
    """Every commit object in the OBJECT DATABASE, reachable or not.

    ONE ``git cat-file --batch-all-objects --batch-check`` call: the ODB
    scan is the whole reachability question's input.  MEASURED 2026-11-04 on
    this repository: 25 494 objects = 17 492 blobs + 6991 trees + 1011
    commits, in 0.024 s.  (RE-MEASURED later the same day: 25 500 objects =
    17 498 blobs + 6991 trees + 1011 commits — the BLOB count moves with the
    tree; the COMMIT count is what the reachability question uses and it was
    stable.)
    """
    try:
        proc = subprocess.run(
            [
                "git",
                "-C",
                str(repo_root),
                "cat-file",
                "--batch-all-objects",
                "--batch-check=%(objectname) %(objecttype)",
            ],
            capture_output=True,
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return set()
    out: Set[str] = set()
    for line in proc.stdout.decode("utf-8", "replace").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == "commit":
            out.add(parts[0])
    return out


def _reachable_commit_objects(repo_root: Path) -> Set[str]:
    """Every commit reachable from a ref (branch or tag)."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-list", "--all"],
            capture_output=True,
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return set()
    if proc.returncode != 0:
        return set()
    return {ln.strip() for ln in proc.stdout.decode("utf-8", "replace").splitlines() if ln.strip()}


def _unreachable_commit_objects(repo_root: Path) -> List[str]:
    """ODB commits that NO ref reaches.

    These are the commits a refs-reachable-only query cannot see.  Their
    content is still IN the repository — it was committed — so a file version
    held only by one of them must NOT be reported as "never committed".
    """
    odb = _odb_commit_objects(repo_root)
    if not odb:
        return []
    reachable = _reachable_commit_objects(repo_root)
    return sorted(odb - reachable)


def _committed_blob_shas_unreachable(
    repo_root: Path, paths: List[str]
) -> Dict[str, List[str]]:
    """Every committed version of each path that is held by an UNREACHABLE
    commit, as git blob shas.

    ONE ``git ls-tree -r <commit> -- <paths>`` call PER unreachable commit.
    The tree is read directly — no diff machinery, no history simplification,
    no merge heuristics — so the answer is exact by construction: "the blob at
    path P in commit C's tree".  (MEASURED 2026-11-04: 168 unreachable commits
    x 401 requested paths = 0.69 s; the answer was verified against a
    full-tree enumeration on 8 sampled commits.)

    WHY NOT ONE BATCHED ``git ls-tree -r C1 C2 ... -- paths``: that form
    DEDUPES across the trees and returns only 397 of the 505 real
    (path, blob) pairs — a silent under-count, which is the exact failure mode
    this function exists to prevent.

    API-SHAPE CONTRACT: an entry for EVERY REQUESTED path (empty list = "no
    unreachable committed version"), same as :func:`_committed_blob_shas`.
    """
    out: Dict[str, set] = {p: set() for p in paths}
    if not paths:
        return {}
    for commit in _unreachable_commit_objects(repo_root):
        try:
            proc = subprocess.run(
                [
                    "git",
                    "-C",
                    str(repo_root),
                    "ls-tree",
                    "-r",
                    commit,
                    "--",
                    *paths,
                ],
                capture_output=True,
                timeout=300,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            continue
        for line in proc.stdout.decode("utf-8", "replace").splitlines():
            meta, _, path = line.partition("\t")
            parts = meta.split()
            if len(parts) != 3 or parts[1] != "blob" or not path:
                continue
            out.setdefault(path, set()).add(parts[2])
    return {p: sorted(s) for p, s in out.items()}


def _committed_content_shas_by_reachability(
    repo_root: Path, paths: List[str]
) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """-> ``(reachable_content_by_path, unreachable_content_by_path)``.

    Each map holds the sha256 of the CONTENT of every committed version of
    the path, split by whether the holding commit is reachable from a ref or
    not.  Both maps carry an entry for EVERY REQUESTED path (an empty set
    means "no committed version of this path in that class").
    """
    reachable_blobs = _committed_blob_shas(repo_root, list(paths))
    unreachable_blobs = _committed_blob_shas_unreachable(repo_root, list(paths))
    all_blobs = sorted(
        {b for v in reachable_blobs.values() for b in v}
        | {b for v in unreachable_blobs.values() for b in v}
    )
    content = _blob_content_shas(repo_root, all_blobs)

    def _map(blobs: Dict[str, List[str]]) -> Dict[str, Set[str]]:
        out: Dict[str, Set[str]] = {}
        for path in paths:
            out[path] = {
                content[b] for b in blobs.get(path, []) if b in content
            }
        return out

    return _map(reachable_blobs), _map(unreachable_blobs)


def commit_status(
    sha: str,
    reachable_content: Dict[str, Set[str]],
    unreachable_content: Dict[str, Set[str]],
    src: str,
) -> str:
    """The THREE-VALUED "is this file version committed?" verdict.

    ``committed``            — held by a commit REACHABLE from a ref.
    ``committed_unreachable`` — held by a commit in the object database that
                               NO ref reaches (committed, then rewritten away).
    ``never_committed``      — held by NO commit at all.

    The order matters: a version held by BOTH a reachable and an unreachable
    commit is ``committed`` (it is reachable), not ``committed_unreachable``.
    """
    if sha in reachable_content.get(src, set()):
        return COMMIT_STATUS_COMMITTED
    if sha in unreachable_content.get(src, set()):
        return COMMIT_STATUS_COMMITTED_UNREACHABLE
    return COMMIT_STATUS_NEVER_COMMITTED


def _blob_content_shas(repo_root: Path, blob_shas: List[str]) -> Dict[str, str]:
    """sha256 of each blob's CONTENT, in ONE ``git cat-file --batch`` call."""
    if not blob_shas:
        return {}
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "cat-file", "--batch"],
            input=("\n".join(blob_shas) + "\n").encode("utf-8"),
            capture_output=True,
            timeout=300,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    data = proc.stdout
    out: Dict[str, str] = {}
    pos = 0
    for want in blob_shas:
        nl = data.find(b"\n", pos)
        if nl < 0:
            break
        header = data[pos:nl].decode("utf-8", "replace").split()
        if len(header) < 3:
            break
        size = int(header[2])
        body = data[nl + 1 : nl + 1 + size]
        out[want] = hashlib.sha256(body).hexdigest()
        pos = nl + 1 + size + 1  # the batch output ends each body with \n
    return out


def uncommitted_provenance(
    ledger_path: Path = LEDGER_PATH,
    *,
    repo_root: Path = REPO_ROOT,
    include_superseded: bool = False,
) -> Dict[str, Any]:
    """Ledger rows whose ``source_json_sha256`` matches NEITHER the file on
    disk NOR any committed version of it.

    Such a row names a file version that this repository has never held --
    it was produced by an uncommitted working tree, and no commit can
    reproduce it.  ``registry.py check`` cannot see it (it validates
    registry rows) and :func:`staleness` cannot see it (it compares the
    ledger's sha with the registry's sha); only a comparison against GIT
    HISTORY finds it.

    TWO VIEWS, because the ledger is APPEND-ONLY:

    * **open** -- the metric's LATEST row still has an uncommitted
      provenance.  This is the live defect: nothing on disk reproduces the
      number the ledger points at.
    * **superseded** -- an EARLIER row had an uncommitted provenance but a
      later row for the same metric does not.  The bad row stays in the
      history forever (the ledger is never rewritten), so it is reported,
      counted, and NOT counted as an open defect.

    ``status`` is ``"PASS"`` when there is no OPEN finding (unless
    ``include_superseded`` is set, in which case any uncommitted row at all
    makes it ``"WARN"``).

    Returns ``{status, counts, detail}``.
    """
    entries = _iter_entries(ledger_path)
    pairs: Dict[Any, List[Dict[str, Any]]] = {}
    for e in entries:
        src = e.get("source_json")
        sha = e.get("source_json_sha256")
        if not src or not sha:
            continue
        pairs.setdefault((str(src), str(sha)), []).append(e)

    src_paths = sorted({src for src, _ in pairs})
    blobs = _committed_blob_shas(repo_root, src_paths)
    all_blob_shas = sorted({b for v in blobs.values() for b in v})
    content_shas = _blob_content_shas(repo_root, all_blob_shas)

    committed_by_src: Dict[str, set] = {}
    disk_by_src: Dict[str, Optional[str]] = {}
    for src in src_paths:
        shas = {content_shas.get(b) for b in blobs.get(src, [])}
        shas.discard(None)
        committed_by_src[src] = shas
        disk_by_src[src] = _sha256_file(_resolve_source_json(src, repo_root=repo_root))

    # The THREE-VALUED reachability verdict (2026-11-04, Wf-A269 wave-23).
    # ``committed_by_src`` above is REFS-REACHABLE only; this adds the
    # unreachable commits that the object database still holds.
    reachable_content, unreachable_content = _committed_content_shas_by_reachability(
        repo_root, src_paths
    )

    def _classify(src: str, sha: str) -> str:
        if disk_by_src.get(src) == sha:
            return "on_disk"
        if sha in committed_by_src.get(src, set()):
            return "committed"
        return "uncommitted"

    def _never_committed(src: str, sha: str) -> bool:
        """Is this sha reproducible from NO commit of this repository?

        ORTHOGONAL to :func:`_classify` — 2026-11-04 (Wf-A269 wave-22).
        ``_classify`` short-circuits on ``on_disk``, so a row whose content
        is currently in the WORKING TREE but in no commit is classified
        ``"on_disk"`` and is therefore NOT reported by
        ``uncommitted_provenance`` — yet the function's own docstring says
        the finding is "no commit can reproduce it" and "the file version it
        points at exists in NO commit of this repository".  A file version
        that is on disk HAS been held by the repository (the working tree
        holds it); what it has NOT been held in is a COMMIT.  The two axes
        are independent and must not cancel one another:

          * ``never_committed`` — reproducible from HISTORY? (this axis)
          * ``on_disk``         — reproducible from the WORKING TREE?

        This is the axis ``molmetal/scripts/wfa166_run_ledger_provenance.py``
        reports as ``sha_never_committed``; the two implementations must be
        comparable on it, and before this axis existed they were NOT
        (MEASURED 2026-11-04: the validator said 14, this function said 2).

        ADDITIVE: no existing count, verdict or status changes.

        REACHABLE-SCOPED (2026-11-04, Wf-A269 wave-23): this boolean asks
        "is the sha reproducible from NO **refs-reachable** commit?".  It is
        NOT the whole-object-database question: a version that WAS committed
        and has since become unreachable (rewritten away) is reported here as
        ``never_committed`` even though the repository still holds it in its
        object database.  The three-valued ``commit_status`` field on the same
        row carries the whole-ODB answer.  The boolean's NAME and VALUE are
        preserved verbatim — ``OPEN_sha_never_committed=14`` /
        ``superseded_sha_never_committed=9`` are published in
        ``docs/HONEST_NEGATIVE_CATALOG.md`` and ``TODO/INDEX.md``.
        """
        return sha not in committed_by_src.get(src, set())

    def _commit_status(src: str, sha: str) -> str:
        """The three-valued whole-object-database verdict (see
        :func:`commit_status`)."""
        return commit_status(sha, reachable_content, unreachable_content, src)

    def _disk_clause(src: str, sha: str) -> str:
        """The DISK half of the statement, on its own.

        2026-11-04 (Wf-A269 wave-23) — the producer's ``note`` used to
        assert a DISJUNCTION ("matches neither the file on disk nor any
        committed version of it") while only ONE half was decided at
        emission, and — for a ``superseded`` finding — it described the
        LATEST row instead of the row the finding is about (MEASURED on this
        tree: 9 of the 11 ``detail`` entries had a latest row whose sha
        MATCHED the file on disk, so the note contradicted its own row).
        Each clause is now reported from its own verdict, and never asserts
        a half no verdict supports.
        """
        disk_sha = disk_by_src.get(src)
        if disk_sha is None:
            return "the file on disk is MISSING"
        if disk_sha == sha:
            return "MATCHES the file on disk"
        return "does NOT match the file on disk"

    def _row_clause(row: Dict[str, Any], reach: Optional[str] = None) -> str:
        """One row's statement, clause by clause (no unsupported disjunct).

        ``reach`` overrides the REFS-REACHABLE clause; the default is derived
        from the row's own ``never_committed`` verdict.
        """
        src = str(row.get("source_json"))
        sha = str(row.get("source_json_sha256"))
        if reach is None:
            reach = (
                "matches NO REFS-REACHABLE committed version of it"
                if row.get("never_committed")
                else "matches a REFS-REACHABLE committed version of it"
            )
        return (
            f"sha {sha} ({src}): {_disk_clause(src, sha)}; {reach}; "
            f"whole-object-database commit status = {row.get('commit_status')}"
        )

    per_metric: Dict[str, Dict[str, Any]] = {}
    # The rows are collected in TRUE LEDGER ORDER (the file is append-only,
    # so the last row for a metric is its CURRENT one).  Iterating ``pairs``
    # instead would order a metric's rows by sha-group, not by time, and the
    # "latest row" would be wrong.
    for e in entries:
        src = e.get("source_json")
        sha = e.get("source_json_sha256")
        if not src or not sha:
            continue
        mid = str(e.get("metric_id"))
        info = per_metric.setdefault(mid, {"rows": []})
        info["rows"].append(
            {
                "source_json": str(src),
                "source_json_sha256": str(sha),
                "verdict": _classify(str(src), str(sha)),
                "never_committed": _never_committed(str(src), str(sha)),
                "commit_status": _commit_status(str(src), str(sha)),
                "git_sha": e.get("git_sha"),
                "timestamp": e.get("timestamp"),
                "value": e.get("value"),
            }
        )

    detail: List[Dict[str, Any]] = []
    never_committed_detail: List[Dict[str, Any]] = []
    for mid, info in sorted(per_metric.items()):
        rows = info["rows"]
        latest = rows[-1]
        open_finding = latest["verdict"] == "uncommitted"
        superseded = [r for r in rows[:-1] if r["verdict"] == "uncommitted"]
        if not open_finding and not (include_superseded and superseded):
            pass
        else:
            detail.append(
                {
                    "metric_id": mid,
                    "state": "open" if open_finding else "superseded",
                    "latest_row": latest,
                    "n_ledger_rows": len(rows),
                    "n_superseded_uncommitted_rows": len(superseded),
                    "superseded_rows": superseded,
                    "on_disk_sha256": disk_by_src.get(latest["source_json"]),
                    "n_committed_versions": len(
                        committed_by_src.get(latest["source_json"], set())
                    ),
                    "note": " || ".join(
                        _row_clause(r)
                        # The finding is about the rows it actually names:
                        # the LATEST row for an "open" finding, the
                        # SUPERSEDED rows for a "superseded" one.  Describing
                        # the latest row for a superseded finding asserted a
                        # disk clause the latest row itself contradicts.
                        for r in (
                            [latest] if open_finding else superseded
                        )
                    ),
                }
            )
        # The ORTHOGONAL axis (see ``_never_committed``).  Reported separately
        # so that the existing finding set — and therefore the published
        # counts, the PASS/WARN status and every consumer of them — is
        # untouched.
        if latest["never_committed"]:
            never_committed_detail.append(
                {
                    "metric_id": mid,
                    "state": "open",
                    "latest_row": latest,
                    "n_ledger_rows": len(rows),
                    "on_disk_sha256": disk_by_src.get(latest["source_json"]),
                    "on_disk_matches": disk_by_src.get(latest["source_json"])
                    == latest["source_json_sha256"],
                    "n_committed_versions": len(
                        committed_by_src.get(latest["source_json"], set())
                    ),
                    # The three-valued whole-object-database verdict.  The
                    # boolean ``never_committed`` above is REACHABLE-SCOPED;
                    # this field says whether the version is nonetheless held
                    # by an unreachable commit.
                    "commit_status": latest["commit_status"],
                    # The DISK clause is reported here too (it was NOT
                    # asserted before, but the note must not leave the
                    # reader to infer it): the row's sha can be the CURRENT
                    # content of the file on disk while no commit holds it.
                    "note": _row_clause(
                        latest,
                        "no REFS-REACHABLE commit of this repository "
                        "reproduces the sha this row names",
                    ),
                }
            )

    n_open = sum(1 for d in detail if d["state"] == "open")
    n_sup = sum(d["n_superseded_uncommitted_rows"] for d in detail)
    n_never_open = len(never_committed_detail)
    n_never_sup = sum(
        1
        for info in per_metric.values()
        for r in info["rows"][:-1]
        if r["never_committed"]
    )
    # The THREE-VALUED whole-object-database axis (2026-11-04, Wf-A269
    # wave-23).  ADDITIVE: the counts above keep their names and their
    # values; these are NEW keys.  ``committed_unreachable_*`` is the class
    # the refs-reachable-only query could not see.
    _status_rows = [r for info in per_metric.values() for r in info["rows"]]
    _latest_status = [
        info["rows"][-1]["commit_status"] for info in per_metric.values()
    ]
    _sup_status = [
        r["commit_status"]
        for info in per_metric.values()
        for r in info["rows"][:-1]
    ]
    return {
        "status": "PASS" if n_open == 0 else "WARN",
        "counts": {
            "ledger_rows": len(entries),
            "distinct_metrics": len(per_metric),
            "distinct_pairs": len(pairs),
            "distinct_source_paths": len(src_paths),
            "committed_blobs_seen": len(all_blob_shas),
            "open_uncommitted_metrics": n_open,
            "superseded_uncommitted_rows": n_sup,
            # The orthogonal "no commit reproduces this sha" axis.
            "open_never_committed_metrics": n_never_open,
            "superseded_never_committed_rows": n_never_sup,
            "never_committed_rows": n_never_open + n_never_sup,
            # The three-valued whole-object-database axis (NEW, additive).
            "commit_status_rows": len(_status_rows),
            "commit_status_latest": {
                v: _latest_status.count(v) for v in COMMIT_STATUS_VALUES
            },
            "commit_status_superseded": {
                v: _sup_status.count(v) for v in COMMIT_STATUS_VALUES
            },
            "open_committed_unreachable_metrics": sum(
                1 for s in _latest_status if s == COMMIT_STATUS_COMMITTED_UNREACHABLE
            ),
            "superseded_committed_unreachable_rows": sum(
                1 for s in _sup_status if s == COMMIT_STATUS_COMMITTED_UNREACHABLE
            ),
        },
        "detail": detail,
        "never_committed_detail": never_committed_detail,
    }


def _protocol_trainer_configs(repo_root: Path = REPO_ROOT) -> Dict[str, str]:
    """metric_id -> the trainer config the registry declares for it.

    Read from ``registry/protocols.yaml``.  Used ONLY as a DERIVED pointer
    for legacy ledger rows that predate the ``config_path`` field; a row
    that carries its own ``config_path`` always wins.  A derived pointer is
    a fallback, never a substitute for the row's own record.
    """
    out: Dict[str, str] = {}
    path = Path(repo_root) / "registry" / "protocols.yaml"
    try:
        import yaml as _yaml

        data = _yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — a missing registry is not fatal here
        return out
    protocols = (data or {}).get("protocols") or {}
    items = protocols.items() if isinstance(protocols, dict) else (
        (p.get("id"), p) for p in protocols
    )
    for pid, entry in items:
        cfg = (entry or {}).get("trainer_config")
        if pid and cfg:
            out[str(pid)] = str(cfg)
    return out


def config_provenance(
    ledger_path: Path = LEDGER_PATH,
    *,
    repo_root: Path = REPO_ROOT,
) -> Dict[str, Any]:
    """Can every ledger row's ``config_sha`` be resolved back to a FILE?

    Wf-A269 (2026-11-04) — the frontier application.  A provenance record
    that names a CONTENT HASH but not the file it hashed is not re-resolvable:
    NOMAD's DAG and SLSA's ``resolvedDependencies`` both carry the artifact
    IDENTIFIER alongside the digest, because a digest alone cannot be
    dereferenced.  MEASURED 2026-11-04 on this working tree: 434 ledger rows,
    6 with a non-null ``config_sha``, 3 distinct hashes, **0** carrying a
    config path — the producer accepted ``config_path`` and dropped it.

    Four states, one per row:

      CURRENT     — the recorded sha IS the file's sha on disk today.
      HISTORICAL  — the recorded sha IS a COMMITTED version of that file.
                    The row is HONEST: it names a real past version.  The
                    config has changed since, which is a REPRODUCIBILITY
                    fact, not a provenance lie.
      UNRESOLVABLE— the recorded sha matches neither the file on disk nor
                    any committed version of it.  The row names a file
                    version this repository has never held.  THIS IS A LIE.
      UNPOINTERED — no path at all: neither the row nor the registry's
                    ``trainer_config`` says which file the hash is of.

    The pointer is the row's own ``config_path`` when present, else the
    registry's declared ``trainer_config`` for the row's metric_id (a
    DERIVED pointer, used only for legacy rows).
    """
    entries = _iter_entries(ledger_path)
    derived = _protocol_trainer_configs(repo_root)

    rows = [e for e in entries if e.get("config_sha")]
    pointer_paths = sorted(
        {
            str(e.get("config_path") or derived.get(str(e.get("metric_id"))) or "")
            for e in rows
        }
        - {""}
    )
    committed = _committed_blob_shas(Path(repo_root), pointer_paths)
    content = _blob_content_shas(Path(repo_root), sorted({b for v in committed.values() for b in v}))

    detail: List[Dict[str, Any]] = []
    for e in rows:
        sha = str(e.get("config_sha"))
        metric_id = str(e.get("metric_id"))
        own = e.get("config_path")
        ptr = str(own) if own else derived.get(metric_id)
        state = "UNPOINTERED"
        pointer_source = "none"
        if ptr:
            pointer_source = "row" if own else "registry"
            abs_ptr = Path(ptr)
            if not abs_ptr.is_absolute():
                abs_ptr = Path(repo_root) / abs_ptr
            current = _sha256_file(abs_ptr)
            if current and current == sha:
                state = "CURRENT"
            else:
                versions = {
                    content.get(b)
                    for b in committed.get(ptr, [])
                    if content.get(b) == sha
                }
                if versions:
                    state = "HISTORICAL"
                else:
                    state = "UNRESOLVABLE"
        detail.append(
            {
                "metric_id": metric_id,
                "config_sha": sha,
                "config_path": ptr,
                "pointer_source": pointer_source,
                "state": state,
                "timestamp": e.get("timestamp"),
                "source_json": e.get("source_json"),
                "row_has_own_path": bool(own),
            }
        )

    counts = {
        "ledger_rows": len(entries),
        "rows_with_config_sha": len(rows),
        "distinct_config_shas": len({str(e.get("config_sha")) for e in rows}),
        "rows_with_own_config_path": sum(1 for e in rows if e.get("config_path")),
        "current": sum(1 for d in detail if d["state"] == "CURRENT"),
        "historical": sum(1 for d in detail if d["state"] == "HISTORICAL"),
        "unresolvable": sum(1 for d in detail if d["state"] == "UNRESOLVABLE"),
        "unpointed": sum(1 for d in detail if d["state"] == "UNPOINTERED"),
    }
    return {"counts": counts, "detail": detail}


def seed_from_disk(
    registry_metrics_path: Path = REGISTRY_METRICS_PATH,
    *,
    registry_protocols_path: Optional[Path] = None,
    ledger_path: Path = LEDGER_PATH,
    repo_root: Path = REPO_ROOT,
    skip_existing: bool = True,
) -> List[Dict[str, Any]]:
    """Seed the ledger with ONE baseline entry per registry metric.

    For each metric in ``registry/metrics.yaml`` append exactly one entry built
    from the CURRENT on-disk state: the value + sha256 recorded in the registry
    row and the source JSON's own ``timestamp`` (falling back to the registry's
    ``last_refreshed`` stamp, then ``now``).  ``data_sha`` is derived from the
    source JSON's ``data_source``; ``config_sha`` from the metric's
    ``trainer_config`` when ``registry/protocols.yaml`` names one.  This records
    the baseline and does NOT invent history.  With ``skip_existing`` (default) a
    metric that already has a ledger entry is left untouched, so re-running the
    seed never duplicates the baseline.
    """
    reg = _load_yaml(registry_metrics_path)
    protocols_path = (
        registry_protocols_path
        if registry_protocols_path is not None
        else registry_metrics_path.parent / "protocols.yaml"
    )
    protos = (_load_yaml(protocols_path).get("protocols") or {})
    written: List[Dict[str, Any]] = []
    already = {e.get("metric_id") for e in _iter_entries(ledger_path)}
    for mid, rec in (reg.get("metrics") or {}).items():
        if skip_existing and mid in already:
            continue
        src = rec.get("source_json", "")
        src_path = _resolve_source_json(src, repo_root=repo_root)
        payload = _read_json(src_path)
        ts = (
            payload.get("timestamp")
            or rec.get("metric_timestamp")
            or rec.get("last_refreshed")
            or _now_utc()
        )
        trainer_cfg = (protos.get(mid) or {}).get("trainer_config")
        entry = record_run(
            mid,
            rec.get("value"),
            source_json=src,
            source_json_sha256=rec.get("source_json_sha256"),
            data_source=payload.get("data_source"),
            config_path=trainer_cfg,
            timestamp=ts,
            seed=True,
            ledger_path=ledger_path,
            repo_root=repo_root,
            extra={"value_key": rec.get("value_key"), "data_source": payload.get("data_source")},
        )
        written.append(entry)
    return written


def repoint_metric(
    metric_id: str,
    *,
    registry_metrics_path: Path = REGISTRY_METRICS_PATH,
    ledger_path: Path = LEDGER_PATH,
    repo_root: Path = REPO_ROOT,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Append ONE ledger row that re-points a metric at the file ON DISK.

    Wf-A125 G11c — the run-ledger staleness gate has to FAIL on a stale row
    (a broken invariant: the registry's number is not backed by the ledger),
    and that means the stale row has to be REPAIRED, not waived.  MEASURED
    2026-10-31: exactly one row was stale — ``m2_r2_per_cell`` — because the
    LEDGER was behind the registry, not the other way round: the registry row
    and the file on disk agree (``d7db0e00...``) and that sha IS in git
    history (``d7513c38`` / ``752e356d``).

    The appended row carries the CURRENT on-disk sha, the registry's value,
    ``git_sha = HEAD`` and ``seed = True`` — the marker
    :func:`seed_from_disk` already uses for "a baseline re-point, not a
    claimed run".

    IT REFUSES when the source file's sha is present in NO commit of this
    repository's OBJECT DATABASE.  That would append a row naming a file
    version this repository has never held — i.e. it would CREATE the very
    defect class :func:`uncommitted_provenance` exists to report.

    WHAT IT ACTUALLY CHECKED (2026-11-04, Wf-A269 wave-23): the predicate
    used to be "is the sha reproducible from NO **refs-reachable** commit?",
    and the refusal message said "present in NO commit of this repository" —
    a claim the check did not support.  A version that WAS committed and has
    since become unreachable (rewritten away) was refused as if the
    repository had never held it, and the message asserted a fact about the
    whole repository when only the reachable half had been examined.  The
    predicate is now the three-valued whole-ODB verdict, and the message
    states which class the sha falls into.
    """
    reg = _load_yaml(registry_metrics_path)
    rows = reg.get("metrics") or {}
    if metric_id not in rows:
        raise KeyError(metric_id)
    rec = rows[metric_id] or {}
    src = str(rec.get("source_json") or "")
    if not src:
        raise ValueError(f"metric {metric_id!r} has no source_json")
    src_path = _resolve_source_json(src, repo_root=repo_root)
    disk_sha = _sha256_file(src_path)
    if disk_sha is None:
        raise ValueError(f"source_json not found on disk: {src_path}")
    reachable_content, unreachable_content = _committed_content_shas_by_reachability(
        repo_root, [src]
    )
    status = commit_status(disk_sha, reachable_content, unreachable_content, src)
    committed: set = set(reachable_content.get(src, set())) | set(
        unreachable_content.get(src, set())
    )
    if status == COMMIT_STATUS_NEVER_COMMITTED:
        raise ValueError(
            f"refusing to repoint {metric_id!r}: the sha of {src} on disk "
            f"({disk_sha}) is held by NO commit in this repository's object "
            "database (no ref reaches a commit holding it and no unreachable "
            "commit holds it either), so the row would name a file version "
            "this repo has never held"
        )
    # ``committed`` keeps its LEGACY meaning (REFS-REACHABLE versions) so the
    # ``committed_versions`` field below keeps its value; the unreachable
    # class is reported separately.
    committed: set = set(reachable_content.get(src, set()))
    payload = _read_json(src_path)
    ts = (
        payload.get("timestamp")
        or rec.get("metric_timestamp")
        or rec.get("last_refreshed")
        or _now_utc()
    )
    protos = (_load_yaml(registry_metrics_path.parent / "protocols.yaml").get(
        "protocols"
    ) or {})
    trainer_cfg = (protos.get(metric_id) or {}).get("trainer_config")
    if dry_run:
        return {
            "metric_id": metric_id,
            "source_json": src,
            "source_json_sha256": disk_sha,
            "registry_value": rec.get("value"),
            "committed_versions": len(committed),
            "commit_status": status,
            "appended": False,
            "dry_run": True,
        }
    entry = record_run(
        metric_id,
        rec.get("value"),
        source_json=src,
        source_json_sha256=disk_sha,
        data_source=payload.get("data_source"),
        config_path=trainer_cfg,
        timestamp=ts,
        seed=True,
        ledger_path=ledger_path,
        repo_root=repo_root,
        extra={
            "value_key": rec.get("value_key"),
            "data_source": payload.get("data_source"),
            "repoint": True,
            "repoint_note": (
                "re-pointed at the file on disk; the registry row and the "
                "file agree, and this sha is present in git history "
                f"(whole-object-database commit status = {status})"
            ),
        },
    )
    return {
        "metric_id": metric_id,
        "source_json": src,
        "source_json_sha256": disk_sha,
        "registry_value": rec.get("value"),
        "committed_versions": len(committed),
        "appended": True,
        "dry_run": False,
        "entry": entry,
    }


def check_ledger(
    registry_metrics_path: Path = REGISTRY_METRICS_PATH,
    *,
    ledger_path: Path = LEDGER_PATH,
    repo_root: Path = REPO_ROOT,
) -> Dict[str, Any]:
    """Report-only staleness check.  Returns {status, counts, detail}."""
    info = staleness(registry_metrics_path, ledger_path=ledger_path, repo_root=repo_root)
    counts = {"current": 0, "reformatted": 0, "stale": 0, "no_run": 0}
    for v in info.values():
        counts[v["status"]] = counts.get(v["status"], 0) + 1
    return {
        # `reformatted` is not a provenance gap: the value the ledger saw is
        # still the value on disk. Only a changed value or a missing run is.
        "status": "PASS" if counts["stale"] == 0 and counts["no_run"] == 0 else "WARN",
        "counts": counts,
        "detail": info,
    }


def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    cmd = args[0] if args else "status"
    if cmd == "seed":
        written = seed_from_disk()
        print(f"seeded {len(written)} entr{'y' if len(written)==1 else 'ies'} into {LEDGER_PATH}")
        return 0
    if cmd == "status":
        rep = check_ledger()
        c = rep["counts"]
        print(
            f"ledger: current={c['current']} reformatted={c['reformatted']} "
            f"stale={c['stale']} no_run={c['no_run']}"
        )
        for mid, info in rep["detail"].items():
            if info["status"] != "current":
                print(f"    {info['status']:6s} {mid}: ledger_ts={info['ledger_timestamp']}")
        print(f"LEDGER: {rep['status']}")
        return 0
    if cmd == "latest":
        if len(args) < 2:
            print("usage: run_ledger latest <metric_id>", file=sys.stderr)
            return 2
        e = latest(args[1])
        print(json.dumps(e, indent=2, ensure_ascii=False) if e else "no entry")
        return 0 if e else 1
    if cmd == "history":
        if len(args) < 2:
            print("usage: run_ledger history <metric_id>", file=sys.stderr)
            return 2
        hs = history(args[1])
        print(json.dumps(hs, indent=2, ensure_ascii=False))
        return 0
    if cmd == "repoint":
        # Wf-A125 G11c — repair a STALE ledger row by re-pointing it at the
        # file on disk.  Refuses when the sha is in no commit.
        if len(args) < 2:
            print("usage: run_ledger repoint <metric_id> [--dry-run]", file=sys.stderr)
            return 2
        metric_id = args[1]
        dry = "--dry-run" in args[2:]
        try:
            rep = repoint_metric(metric_id, dry_run=dry)
        except (KeyError, ValueError) as exc:
            print(f"ledger repoint FAILED: {exc}", file=sys.stderr)
            return 1
        if rep["appended"]:
            print(
                f"ledger repoint [{metric_id}]: appended one row -> "
                f"{rep['source_json']} sha {rep['source_json_sha256'][:12]}... "
                f"({rep['committed_versions']} committed version(s) of that path)"
            )
        else:
            print(
                f"ledger repoint [{metric_id}]: dry run — would append one row "
                f"-> {rep['source_json']} sha {rep['source_json_sha256'][:12]}..."
            )
        return 0
    if cmd == "provenance":
        # Wf-A123 FIX 7 -- the ledger rows whose source_json_sha256 matches
        # NEITHER the file on disk NOR any committed version of it.
        include_superseded = "--all" in args[1:]
        rep = uncommitted_provenance(include_superseded=include_superseded)
        c = rep["counts"]
        # Wf-A198 (2026-11-02) -- the DEFAULT mode must not print a count it
        # did NOT measure.  `uncommitted_provenance()` puts a metric into
        # `detail` only when it has an OPEN finding or when
        # `include_superseded=True`; so in default mode the superseded-only
        # metrics are filtered out BEFORE the count is taken, and
        # `superseded_uncommitted_rows` is 0 BY CONSTRUCTION, not by
        # measurement.  The default mode therefore prints the token
        # NOT_MEASURED; `--all` still prints the measured number.
        #
        # PRESERVED VERBATIM (AGENTS.md 5.5 rule #6) -- the SUPERSEDED
        # default-mode rendering, MEASURED 2026-11-02 on this working tree
        # (432 ledger rows, 399 metrics, 401 distinct source paths):
        #   ledger provenance: rows=432 metrics=399 distinct_source_paths=401 OPEN_never_committed=2 superseded_never_committed=0
        # and the `--all` rendering, which this fix does NOT change:
        #   ledger provenance: rows=432 metrics=399 distinct_source_paths=401 OPEN_never_committed=2 superseded_never_committed=9
        superseded_token = (
            str(c["superseded_uncommitted_rows"])
            if include_superseded
            else NOT_MEASURED_TOKEN
        )
        # 2026-11-04 (Wf-A269 wave-22) -- the ORTHOGONAL axis is now printed
        # too, under a token that is NOT the legacy one.
        #
        # The legacy token ``OPEN_never_committed`` has meant
        # ``open_uncommitted_metrics`` (the "neither on disk NOR in any
        # commit" count) since Wf-A125, and it is quoted VERBATIM in
        # ``docs/HONEST_NEGATIVE_CATALOG.md`` and ``TODO/INDEX.md`` — so its
        # NAME and its VALUE are both preserved here.  But that name now
        # collides with a genuinely different quantity: ``sha_never_committed``
        # ("no commit reproduces this sha"), which is a SUPERSET of the
        # legacy count (a row whose content is currently in the working tree
        # but in NO commit is never-committed yet NOT uncommitted).  Printing
        # both under distinct tokens is the only rendering that is not
        # ambiguous; the legacy token keeps its meaning, the new one gets the
        # name of the quantity it actually measures.
        #
        # PRESERVED VERBATIM (AGENTS.md §5.5 rule #6) -- the 2026-11-02
        # rendering of this line (432 ledger rows, 399 metrics, 401 distinct
        # source paths):
        #   ledger provenance: rows=432 metrics=399 distinct_source_paths=401 OPEN_never_committed=2 superseded_never_committed=0
        # and the `--all` rendering, which this fix does NOT change:
        #   ledger provenance: rows=432 metrics=399 distinct_source_paths=401 OPEN_never_committed=2 superseded_never_committed=9
        # The two tokens appended below are ADDITIVE: every published
        # quotation of the line above remains an accurate prefix of it.
        print(
            f"ledger provenance: rows={c['ledger_rows']} "
            f"metrics={c['distinct_metrics']} "
            f"distinct_source_paths={c['distinct_source_paths']} "
            f"OPEN_never_committed={c['open_uncommitted_metrics']} "
            f"superseded_never_committed={superseded_token} "
            f"OPEN_sha_never_committed={c['open_never_committed_metrics']} "
            f"superseded_sha_never_committed={c['superseded_never_committed_rows']}"
        )
        if not include_superseded:
            print(
                "    NOTE: superseded_never_committed was NOT MEASURED in "
                "default mode; pass --all to measure it.",
                file=sys.stderr,
            )
        for d in rep["detail"]:
            row = d["latest_row"]
            print(f"    [{d['state']}] {d['metric_id']}  {row['source_json']}")
            print(f"        on-disk sha256 {d['on_disk_sha256']}")
            print(f"        committed versions of that path: {d['n_committed_versions']}")
            if d["state"] == "open":
                print(f"        ledger  sha256 {row['source_json_sha256']}")
                print(f"        ledger value: {row['value']!r}")
            else:
                # The OFFENDING row is the SUPERSEDED one.  The latest row is
                # the appended correction and IS reproducible, so printing it
                # here would hide the sha that exists in NO commit -- which is
                # the whole point of the report.
                for bad in d["superseded_rows"]:
                    print(f"        NEVER-COMMITTED sha256 {bad['source_json_sha256']}")
                    print(
                        f"            (row value {bad['value']!r}, "
                        f"git_sha {bad['git_sha']!r})"
                    )
                print(
                    f"        superseded uncommitted rows for this metric: "
                    f"{d['n_superseded_uncommitted_rows']}"
                )
        print(f"LEDGER PROVENANCE: {rep['status']}")
        return 0 if rep["status"] == "PASS" else 1
    print(
        f"unknown command {cmd!r}; use 'seed', 'status', 'latest', 'history', "
        "'provenance', 'repoint'",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
