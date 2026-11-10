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
"""Wf-A269 wave-22 — a MERGE COMMIT's own file versions must be visible.

THE DEFECT
----------
``molmetal/scripts/run_ledger.py::_committed_blob_shas`` answered "every
committed version of this path" with::

    git log --all --format= --raw --no-abbrev --no-renames -- <paths>

``git log --raw`` does **not** diff a MERGE commit by default — git's default
merge handling suppresses the merge's own diff.  A file whose version exists
ONLY in a merge commit (e.g. a file that the merge's conflict resolution
introduced, so that NO parent commit holds it) is therefore INVISIBLE to the
query.  The producer then under-collects the committed set, and every
consumer of it reports a FALSE "no commit reproduces this sha".

MEASURED on this repository (2026-11-04): the flag
``--diff-merges=first-parent`` adds **45** of the ledger's **401**
referenced paths (**397** vs **352** paths with history; **754** vs **700**
blobs).  The independent validator
``molmetal/scripts/wfa166_run_ledger_provenance.py`` has always carried the
flag, and the two implementations disagreed by **59 vs 14** on the same
question.

THE FLAG IS LOAD-BEARING; THE MODE IS NOT
------------------------------------------
There are two merge-diff modes: ``--diff-merges=first-parent`` and ``-m``
(``--diff-merges=separate``).  MEASURED 2026-11-04 on this repository, they
return the **IDENTICAL** committed-content sets — **397** paths / **754**
blobs over the ledger's 401 referenced paths, and **9835** paths / **17308**
DISTINCT blobs over the whole tree (the per-path SUM convention gives
**18228** for the same set; the two conventions are only equal for the
ledger-scoped set, which is why the failure message below prints both).
What differs is COMMIT **ATTRIBUTION**: the
merge ``8289a7f5`` is credited with **585** raw diff lines under
first-parent and **1351** under ``-m``.  ⇒ a mode is not a different
*query*, and comparing the two as if they were is how one derives a false
"one mode is strictly more complete" (see test 7, which pins this).

THE REGRESSION GUARD
--------------------
This file pins the behaviour, not a count:

1. a synthetic repository whose merge commit holds a file that NEITHER
   parent holds — the exact shape that was invisible;
2. the NEGATIVE CONTROL — the same query WITHOUT the flag finds nothing, so
   the flag is load-bearing and the test is not a no-op;
3. the producer finds that file and returns its CONTENT sha;
4. on the REAL repository, the producer and the independent validator agree
   on the set of paths that have committed history;
5. on the REAL repository, a path that a merge commit holds and neither of
   its parents holds is found by the producer;
6. ALL FIVE ledger-provenance implementations agree on WHICH paths have
   committed history — after normalising the one API-shape difference that
   otherwise manufactures a phantom 4-path "disagreement";
7. the two merge-diff MODES return the identical committed-content set and
   differ only in commit attribution;
8. the SECOND instance of the blind spot — the Wf-A198 auditor — also
   finds a merge-only file;
9. the five are anchored against an INDEPENDENT ORACLE — the committed set
   re-derived from the TREES, one ``ls-tree`` per reachable commit — so
   "they agree" cannot survive a common-mode regression: if all five lost
   the merge diff together they would all drop to 352 and the agreement
   test alone would stay GREEN;
10. each implementation KEEPS ITS DECLARED IDENTIFIER NAMESPACE — three
    return the git blob sha1 (40 hex), two return the sha256 of the
    blob's CONTENT (64 hex) — and all five agree once expressed in the
    LEDGER's own namespace, behind a non-vacuity floor.  ⇒ a refactor
    cannot move one implementation into the other's namespace silently.

CPU-ONLY
--------
stdlib + git only; no torch import, no CUDA context, no GPU allocation.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from molmetal.scripts import run_ledger as rl
from molmetal.scripts import wfa166_run_ledger_provenance as tool

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_LEDGER = REPO_ROOT / "molmetal" / "reports" / "run_ledger.jsonl"

_CPU_ENV = dict(os.environ)
_CPU_ENV["CUDA_VISIBLE_DEVICES"] = ""
_CPU_ENV["GIT_AUTHOR_NAME"] = "wfa269"
_CPU_ENV["GIT_AUTHOR_EMAIL"] = "wfa269@example.invalid"
_CPU_ENV["GIT_COMMITTER_NAME"] = "wfa269"
_CPU_ENV["GIT_COMMITTER_EMAIL"] = "wfa269@example.invalid"
_CPU_ENV["GIT_CONFIG_GLOBAL"] = "/dev/null"
_CPU_ENV["GIT_CONFIG_SYSTEM"] = "/dev/null"


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        env=_CPU_ENV,
        check=False,
    )


# ---------------------------------------------------------------------------
# 1. the synthetic repository
# ---------------------------------------------------------------------------
def _make_merge_only_repo(repo: Path) -> str:
    """Build a repo whose MERGE commit holds a file neither parent holds.

    Returns the repo-relative path of that file.
    """
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-q", "-b", "master")
    _git(repo, "config", "user.name", "wfa269")
    _git(repo, "config", "user.email", "wfa269@example.invalid")
    (repo / "base.json").write_text('{"base":1}\n', encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    _git(repo, "checkout", "-q", "-b", "feature")
    (repo / "feat.json").write_text('{"feat":1}\n', encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "feature")
    _git(repo, "checkout", "-q", "master")
    # merge WITHOUT committing, then add a file that NEITHER parent holds —
    # the shape a conflict resolution produces.
    _git(repo, "merge", "--no-commit", "--no-ff", "feature")
    (repo / "merge_only.json").write_text('{"merged":1}\n', encoding="utf-8")
    _git(repo, "add", "merge_only.json")
    _git(repo, "commit", "-q", "-m", "merge feature, adds a merge-only file")
    return "merge_only.json"


def _raw_query(repo: Path, path: str, *, first_parent: bool) -> List[str]:
    """The producer's own git query, with and without the flag."""
    args = [
        "log",
        "--all",
        "--format=",
        "--raw",
        "--no-abbrev",
        "--no-renames",
    ]
    if first_parent:
        args.append("--diff-merges=first-parent")
    args += ["--", path]
    proc = _git(repo, *args)
    out = []
    for line in proc.stdout.splitlines():
        if line.startswith(":"):
            out.append(line)
    return out


def test_the_flag_is_load_bearing(tmp_path: Path) -> None:
    """NEGATIVE CONTROL — without the flag the merge-only file is INVISIBLE.

    Without this test the positive control below would pass for the wrong
    reason (e.g. because the synthetic repo is wrong, or because git
    changed its default).  The flag must be the thing that makes the
    difference.
    """
    repo = tmp_path / "merge_repo"
    rel = _make_merge_only_repo(repo)
    without = _raw_query(repo, rel, first_parent=False)
    with_flag = _raw_query(repo, rel, first_parent=True)
    assert not without, (
        "the merge-only file WAS visible without --diff-merges=first-parent, "
        "so this test no longer proves the flag is load-bearing: %r" % without
    )
    assert with_flag, (
        "the merge-only file is invisible even WITH --diff-merges=first-parent: "
        "the synthetic repository no longer reproduces the defect shape"
    )


def test_the_producer_finds_the_merge_only_file(tmp_path: Path) -> None:
    """The producer must return the merge commit's version of the file."""
    repo = tmp_path / "merge_repo"
    rel = _make_merge_only_repo(repo)
    blobs = rl._committed_blob_shas(repo, [rel])
    assert rel in blobs, (
        "the producer did not find the merge-only file %r (found: %r)"
        % (rel, sorted(blobs))
    )
    assert blobs[rel], "the merge-only file resolved to no blob"
    content = rl._blob_content_shas(repo, sorted({b for v in blobs.values() for b in v}))
    got = {content.get(b) for b in blobs[rel] if content.get(b)}
    expected = hashlib.sha256((repo / rel).read_bytes()).hexdigest()
    assert expected in got, (
        "the producer returned blob(s) whose CONTENT does not match the "
        "merged file: expected %s, got %r" % (expected, sorted(got))
    )


def test_the_producer_agrees_with_the_independent_validator() -> None:
    """The two implementations must agree on the REAL ledger.

    ``wfa166_run_ledger_provenance.py`` has always carried
    ``--diff-merges=first-parent``; the producer did not.  The disagreement
    was MEASURED as **397 vs 352** paths with committed history (2026-11-04).

    **REPAIRED AT THE API (2026-11-04, wave-23).**  This comparison used to
    compare the two implementations' KEY SETS directly.  That was only
    possible because they disagreed on SHAPE: the producer returned no entry
    at all for a path with no committed version, the validator's internal
    walker did the same, and the phantom 4-path "disagreement" had to be
    compensated for.  Both sides now use their contract-honouring RESOLVER
    (an entry for every requested path), so the comparison is made on the
    DERIVED view "paths with a committed version" — and the contract itself
    is ASSERTED here rather than assumed.
    """
    rows = [
        json.loads(line)
        for line in REAL_LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    src_paths = sorted({str(o.get("source_json")) for o in rows if o.get("source_json")})
    mine = rl._committed_blob_shas(REPO_ROOT, src_paths)
    theirs = tool.committed_content_shas(REPO_ROOT, src_paths)
    for name, mapping in (
        ("run_ledger", mine),
        ("wfa166_run_ledger_provenance", theirs),
    ):
        missing = sorted(set(src_paths) - set(mapping))
        assert not missing, (
            "%s violates the API-shape contract: it returned NO entry for %d "
            "of the %d requested path(s) %r — every resolver must return an "
            "entry for every requested path (empty = 'no committed "
            "version')." % (name, len(missing), len(src_paths), missing[:5])
        )
    mine_paths = {p for p, ids in mine.items() if ids}
    their_paths = {p for p, ids in theirs.items() if ids}
    only_validator = their_paths - mine_paths
    only_producer = mine_paths - their_paths
    assert not only_validator, (
        "the producer is missing %d path(s) the independent validator sees: "
        "%r" % (len(only_validator), sorted(only_validator)[:5])
    )
    assert not only_producer, (
        "the producer sees %d path(s) the independent validator does not: %r"
        % (len(only_producer), sorted(only_producer)[:5])
    )


def _merge_only_path_on_disk(repo_root: Path) -> Optional[Tuple[str, str]]:
    """-> (merge_commit, path) for a path a merge holds and no parent holds.

    Found by comparing TREES (``ls-tree``), which is independent of the
    ``--diff-merges`` flag under test.
    """
    proc = _git(repo_root, "rev-list", "--all", "--merges", "--max-count=40")
    for merge in proc.stdout.split():
        parents = _git(repo_root, "rev-list", "--parents", "-n", "1", merge).stdout.split()[1:]
        if len(parents) < 2:
            continue
        tree = set(_git(repo_root, "ls-tree", "-r", "--name-only", merge).stdout.splitlines())
        parent_tree: set = set()
        for p in parents:
            parent_tree |= set(
                _git(repo_root, "ls-tree", "-r", "--name-only", p).stdout.splitlines()
            )
        only = sorted(tree - parent_tree)
        if only:
            return merge, only[0]
    return None


def test_a_real_merge_only_file_is_found_by_the_producer() -> None:
    """On the REAL repository the producer must see a merge-only file.

    MEASURED 2026-11-04: the merge ``8289a7f5`` ("Merge Wf-A17 Track A
    natural-charge 9-layer wire-in (resolved 3 conflicts)") holds **561**
    paths that NEITHER of its parents holds — every one of them invisible to
    the pre-fix query.
    """
    found = _merge_only_path_on_disk(REPO_ROOT)
    assert found is not None, (
        "no merge-only path found in this repository — the real-world shape "
        "the defect needs has disappeared (that is worth knowing, but this "
        "test must be updated rather than silently pass)"
    )
    merge, rel = found
    blobs = rl._committed_blob_shas(REPO_ROOT, [rel])
    assert rel in blobs, (
        "merge %s holds %r, but the producer cannot see it" % (merge[:12], rel)
    )
    assert blobs[rel], "merge %s: the path resolved to no blob" % merge[:12]


# ---------------------------------------------------------------------------
# 6. the FIVE implementations must agree on WHICH paths have history
# ---------------------------------------------------------------------------
def _ledger_source_paths() -> List[str]:
    rows = [
        json.loads(line)
        for line in REAL_LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return sorted({str(o.get("source_json")) for o in rows if o.get("source_json")})


def _paths_with_history_per_implementation(src_paths: List[str]) -> Dict[str, set]:
    """-> ``{implementation name: paths that have committed history}``.

    **THE API-SHAPE TRAP** (MEASURED 2026-11-04, wave-22).  The five
    implementations of "every committed content version of a path" do NOT
    return the same SHAPE.  ``wfa156`` returns an entry for **every**
    requested path — an EMPTY set when the path has no committed history —
    while the other four return only the paths that **have** history.
    Comparing them naively makes ``wfa156`` look like it sees **401** paths
    while the others see **397**: a 4-path "disagreement" that is pure API
    shape and says NOTHING about git.

    That phantom is not hypothetical: it is what produced a FALSE
    conclusion in this very wave ("``-m`` is strictly more complete than
    ``--diff-merges=first-parent``"), which the measurement then REFUTED —
    the two modes return the identical set (test 7).

    **REPAIRED AT THE API (2026-11-04, wave-23).**  The trap was normalised
    HERE, at the comparison site, which meant every future comparison had to
    remember to compensate.  It is now normalised AT THE API: all five
    resolvers return an entry for EVERY requested path (empty = "no committed
    version"), and this helper ASSERTS that contract instead of compensating
    for it.  The four resolvers that did not honour the contract
    (``run_ledger._committed_blob_shas``,
    ``wfa198_ledger_pin_consistency._committed_blob_shas``,
    ``wfa166_run_ledger_census.bulk_committed_content_shas`` and
    ``wfa166_run_ledger_provenance``'s committed-content map) now do.

    "Which paths HAVE history" is now a DERIVED view over a uniform map
    (the paths whose set is non-empty) — not a compensation for a shape
    mismatch, because the shape is identical for all five.
    """
    maps = _committed_blob_maps_per_implementation(src_paths)
    return {
        name: {p for p, ids in m.items() if ids}
        for name, m in maps.items()
    }


def _assert_api_shape_contract(
    src_paths: List[str], resolvers: Dict[str, object]
) -> None:
    """Every resolver must return an entry for EVERY requested path.

    This is the assertion the comparison site used to have to work around.
    A resolver that drops a path it was asked about makes the comparison
    meaningless: the missing key is indistinguishable from "this path has no
    committed version" unless the caller knows the contract.
    """
    requested = set(src_paths)
    for name, mapping in resolvers.items():
        keys = set(mapping)  # type: ignore[arg-type]
        missing = requested - keys
        assert not missing, (
            f"resolver {name!r} violates the API-shape contract: it returned "
            f"NO entry for {len(missing)} of {len(requested)} requested "
            f"path(s), e.g. {sorted(missing)[:3]}. Every resolver must "
            f"return an entry for every requested path (empty = 'no "
            f"committed version')."
        )


def _committed_blob_maps_per_implementation(src_paths: List[str]) -> Dict[str, Dict[str, Set[str]]]:
    """-> ``{implementation name: path -> {committed content shas}}``.

    The BLOB-level twin of :func:`_paths_with_history_per_implementation`.

    **REPAIRED AT THE API (2026-11-04, wave-23).**  This helper used to
    DROP every empty set before comparing, because four of the five
    resolvers returned no entry at all for a path with no committed history.
    The compensation is gone: all five now return an entry for EVERY
    requested path, so the maps are compared on the FULL requested
    population and the API-shape contract is ASSERTED (not assumed).

    **TWO IDENTIFIER NAMESPACES (MEASURED 2026-11-04).**  ``wfa156``, the
    census and ``wfa166_run_ledger_provenance``'s public resolver return the
    **sha256 of the blob's CONTENT** (they hash the bytes); ``run_ledger``
    and ``wfa198`` return the **git blob sha1** straight off the diff line.
    Both identify the same content, but they are different identifiers ⇒ a
    blob-level comparison of the five is meaningless until the namespaces
    are reconciled.  MEASURED: for
    ``metrics/by_metric/pa20_external_temporal_5metal.json`` the oracle's
    blob is ``763ce105…`` (sha1) and ``wfa156`` reports ``445fc861…`` —
    which is exactly ``sha256(content of 763ce105…)``.
    """
    from molmetal.scripts import wfa156_ledger_provenance_guard as wfa156
    from molmetal.scripts import wfa166_run_ledger_census as census
    from molmetal.scripts import wfa198_ledger_pin_consistency as wfa198

    def _as_sets(mapping: Dict[str, object]) -> Dict[str, Set[str]]:
        # NOTE: no ``if v`` filter — an EMPTY set is a MEANINGFUL answer
        # ("this path has no committed version"), not an absent answer.
        return {str(p): {str(s) for s in v} for p, v in mapping.items()}

    out: Dict[str, Dict[str, Set[str]]] = {}
    out["run_ledger"] = _as_sets(rl._committed_blob_shas(REPO_ROOT, src_paths))
    out["wfa198_ledger_pin_consistency"] = _as_sets(
        wfa198._committed_blob_shas(REPO_ROOT, src_paths)
    )
    out["wfa166_run_ledger_provenance"] = _as_sets(
        tool.committed_content_shas(REPO_ROOT, src_paths)
    )
    bulk, _n_blobs = census.bulk_committed_content_shas(src_paths)
    out["wfa166_run_ledger_census"] = _as_sets(bulk)
    out["wfa156_ledger_provenance_guard"] = _as_sets(
        wfa156.committed_content_shas_bulk(REPO_ROOT, src_paths)
    )

    _assert_api_shape_contract(src_paths, out)
    return out


def _exact_committed_blob_oracle(src_paths: List[str]) -> Dict[str, Set[str]]:
    """The INDEPENDENT ground truth, computed WITHOUT git's diff machinery.

    "A committed content version of a path" is, by definition, the blob at
    that path in some commit's TREE.  So the oracle reads the trees:
    ``git ls-tree -r --full-tree <commit> -- <paths>``, once per commit
    reachable from a ref.

    ONE TREE AT A TIME, deliberately.  A multi-tree ``git ls-tree`` invocation
    DEDUPES and silently UNDER-COUNTS — MEASURED 2026-11-04: it returns
    **601** blobs where the per-tree enumeration returns **754**.  A silent
    under-count here would make the oracle agree with a broken query, which
    is precisely the failure this file exists to prevent.
    """
    commits = _git(REPO_ROOT, "rev-list", "--all").stdout.split()

    def _one(commit: str) -> List[Tuple[str, str]]:
        proc = _git(
            REPO_ROOT, "ls-tree", "-r", "--full-tree", commit, "--", *src_paths
        )
        pairs: List[Tuple[str, str]] = []
        for line in proc.stdout.splitlines():
            if not line.strip():
                continue
            meta, _, path = line.partition("\t")
            parts = meta.split()
            if len(parts) != 3 or parts[1] != "blob" or not path:
                continue
            pairs.append((path, parts[2]))
        return pairs

    by_path: Dict[str, Set[str]] = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        for pairs in ex.map(_one, commits):
            for path, sha in pairs:
                by_path.setdefault(path, set()).add(sha)
    return by_path


def _content_sha256s(blob_shas: List[str]) -> Dict[str, str]:
    """-> ``{git blob sha: sha256 of the blob's CONTENT}``.

    Written here rather than borrowed from ``wfa156`` so that the oracle's
    second namespace is computed INDEPENDENTLY of the implementation under
    test — otherwise the comparison would be circular.
    """
    if not blob_shas:
        return {}
    proc = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "cat-file", "--batch"],
        input=("\n".join(blob_shas) + "\n").encode("utf-8"),
        capture_output=True,
        env=_CPU_ENV,
        check=False,
    )
    data = proc.stdout
    out: Dict[str, str] = {}
    pos = 0
    for sha in blob_shas:
        nl = data.find(b"\n", pos)
        if nl < 0:
            break
        parts = data[pos:nl].decode("utf-8", "replace").split()
        pos = nl + 1
        if len(parts) != 3:
            continue
        try:
            size = int(parts[2])
        except ValueError:
            continue
        out[sha] = hashlib.sha256(data[pos : pos + size]).hexdigest()
        pos += size + 1
    return out


def test_the_five_implementations_match_an_independent_ground_truth() -> None:
    """AGREEMENT ALONE IS NOT A PIN — the five must match an ORACLE.

    MEASURED 2026-11-04 (adversarial review of this file): the sibling test
    asserts the five implementations AGREE **with each other**.  That is
    VACUOUS against a COMMON-MODE regression: if all five lost the merge
    diff at once — the very failure this wave found — they would all drop
    to **352** paths and the agreement test would still be GREEN.

    ⇒ the five are pinned against an INDEPENDENT oracle: the committed
    content set is re-derived from the TREES themselves (one ``ls-tree`` per
    reachable commit), which uses **no diff machinery at all**.  MEASURED:
    the oracle is **397** paths / **754** blobs; the repaired query matches
    it EXACTLY (0 missed, 0 extra) while the un-repaired query misses
    **54** blobs.

    The oracle is itself checked for NON-VACUITY: it must contain paths that
    ONLY the merge diff reveals (the **45** merge-recovered paths), and the
    un-repaired query must miss every one of them.  Without that, an oracle
    that silently collapsed to the pre-fix set would pass this test.
    """
    src_paths = _ledger_source_paths()
    oracle = _exact_committed_blob_oracle(src_paths)
    oracle_paths = set(oracle)
    assert len(oracle_paths) >= 397, (
        "the independent tree oracle found only %d paths with committed "
        "history (MEASURED 2026-11-04: 397) — the oracle itself is broken, "
        "so it cannot anchor anything" % (len(oracle_paths),)
    )

    # NON-VACUITY: the merge-recovered paths must be in the oracle and must
    # be INVISIBLE to the same query without the merge-diff flag.
    un_repaired = _raw_blob_and_attribution(None, src_paths)
    un_repaired_paths = set(un_repaired[0])
    merge_recovered = sorted(oracle_paths - un_repaired_paths)
    assert len(merge_recovered) >= 45, (
        "expected the merge diff to reveal >=45 paths the un-repaired query "
        "cannot see (MEASURED 2026-11-04: 45); only %d such paths exist now "
        "— either git changed or the oracle collapsed to the pre-fix set"
        % (len(merge_recovered),)
    )
    assert not (set(merge_recovered) & un_repaired_paths), (
        "a merge-recovered path is visible to the un-repaired query — the "
        "oracle's non-vacuity check has lost its teeth"
    )

    maps = _committed_blob_maps_per_implementation(src_paths)

    # RECONCILE THE TWO IDENTIFIER NAMESPACES (see the helper's docstring):
    # wfa156 and the census report content-sha256 (they hash the bytes
    # internally); run_ledger, wfa198 and wfa166_provenance expose the RAW
    # git blob sha1.  The oracle is expressed in BOTH, so each
    # implementation is compared in the namespace it actually speaks.
    # ⇒ comparing all five naively at blob level manufactures a FALSE
    # "disagreement" for 2 of the 5 (MEASURED: the sha256 of the oracle's
    # blob 763ce105… IS the value the census and wfa156 report).
    content_sha = _content_sha256s(
        sorted({s for v in oracle.values() for s in v})
    )
    oracle_blobs = {p: set(v) for p, v in oracle.items()}
    oracle_as_content_sha = {
        p: {content_sha[s] for s in v if s in content_sha}
        for p, v in oracle.items()
    }
    namespace = {
        "wfa156_ledger_provenance_guard": ("content sha256", oracle_as_content_sha),
        "wfa166_run_ledger_census": ("content sha256", oracle_as_content_sha),
        "wfa166_run_ledger_provenance": ("content sha256", oracle_as_content_sha),
    }

    for name, got in sorted(maps.items()):
        # The maps now carry an entry for EVERY requested path (the
        # API-shape contract), so "paths with a committed version" is a
        # DERIVED view: the paths whose set is non-empty.  A path with an
        # EMPTY set means "this path has no committed version" — and the
        # oracle MUST agree that it has none.  That is now ASSERTED, not
        # assumed (before the repair it was a phantom "extra path").
        got_paths = {p for p, ids in got.items() if ids}
        empty_paths = {p for p, ids in got.items() if not ids}
        missing = sorted(oracle_paths - got_paths)
        extra = sorted(got_paths - oracle_paths)
        assert not missing and not extra, (
            "%s does not match the independent tree oracle: it MISSES %d "
            "path(s) %r and ADDS %d path(s) %r — the oracle has %d paths / "
            "%d blobs, this implementation has %d / %d"
            % (
                name, len(missing), missing[:5], len(extra), extra[:5],
                len(oracle_paths), sum(len(v) for v in oracle.values()),
                len(got_paths), sum(len(v) for v in got.values()),
            )
        )
        for path in sorted(empty_paths):
            assert path not in oracle_paths, (
                "%s reports NO committed version for %s but the independent "
                "tree oracle finds one — an EMPTY set is a claim, not an "
                "absence" % (name, path)
            )
        label, expected = namespace.get(name, ("git blob sha1", oracle_blobs))
        for path in sorted(got_paths):
            missing_blobs = expected[path] - got[path]
            extra_blobs = got[path] - expected[path]
            assert not missing_blobs and not extra_blobs, (
                "%s disagrees with the oracle on the CONTENT of %s "
                "(namespace: %s): it MISSES %d identifier(s) %r and ADDS %d "
                "identifier(s) %r"
                % (
                    name, path, label, len(missing_blobs),
                    sorted(missing_blobs)[:3], len(extra_blobs),
                    sorted(extra_blobs)[:3],
                )
            )


def test_each_implementation_keeps_its_declared_identifier_namespace() -> None:
    """**AN IDENTIFIER NAMESPACE IS PART OF A FUNCTION'S CONTRACT.**

    MEASURED 2026-11-04: the five implementations of "the committed content
    of a path" speak **TWO** identifier namespaces —

    * ``run_ledger``, ``wfa198`` return the **git blob sha1** (40 hex),
      straight off the diff line;
    * ``wfa156``, ``wfa166_run_ledger_census`` return the **sha256 of the
      blob's CONTENT** (64 hex), which they compute themselves.

    Both identify the same content, so a naive blob-level comparison of the
    five manufactures a FALSE "disagreement" for **2 of the 5** (MEASURED:
    ``sha256(content of 763ce105…)`` IS the value the census and ``wfa156``
    report).  **⇒ NO LIVE BUG** — every one of the five converts into the
    ledger's namespace *before* comparing, so no consumer is wrong.

    **CHANGED 2026-11-04 (wave-23) — WHY.**  The comparison site now reaches
    ``wfa166_run_ledger_provenance`` through its PUBLIC, contract-honouring
    resolver ``committed_content_shas`` instead of the internal walker
    ``_history_raw`` (which returns the git blob sha1 but has no
    API-shape contract).  The public resolver speaks **content sha256**, so
    ``wfa166_run_ledger_provenance`` moved into the sha256 group.  The
    internal walker's namespace is NOT lost — it is pinned separately by
    :func:`test_the_internal_history_walker_speaks_the_git_blob_namespace`.

    ⇒ this test pins the **contract** so that a future refactor cannot
    silently move one implementation into the other namespace: a consumer
    that compares early would then be comparing sha1 against sha256 and
    would report a defect that does not exist — or miss one that does.

    The second half of the test then asserts the property the contract
    exists to make possible: **all five agree once expressed in the
    ledger's own namespace** (the sha256 of the file's CONTENT, which is
    what ``run_ledger.jsonl``'s ``source_json_sha256`` column holds).
    """
    src_paths = _ledger_source_paths()
    maps = _committed_blob_maps_per_implementation(src_paths)

    #: the DOCUMENTED namespace of each implementation
    expected_namespace = {
        "run_ledger": "git blob sha1",
        "wfa198_ledger_pin_consistency": "git blob sha1",
        "wfa166_run_ledger_provenance": "content sha256",
        "wfa156_ledger_provenance_guard": "content sha256",
        "wfa166_run_ledger_census": "content sha256",
    }
    assert set(maps) == set(expected_namespace), (
        "the set of implementations changed: %r vs %r"
        % (sorted(maps), sorted(expected_namespace))
    )

    for name, m in sorted(maps.items()):
        label = expected_namespace[name]
        want_len = 40 if label == "git blob sha1" else 64
        wrong = sorted(
            {
                ident
                for ids in m.values()
                for ident in ids
                if len(ident) != want_len
            }
        )
        assert not wrong, (
            "%s declares the %s namespace (%d hex) but returns %d "
            "identifier(s) of another length %r — the identifier namespace "
            "IS part of the contract; a consumer that compares before "
            "converting would report a disagreement that is not there"
            % (name, label, want_len, len(wrong), wrong[:3])
        )

    # ---- and they agree in the LEDGER's namespace ------------------------
    sha_of_blob = _content_sha256s(
        sorted(
            {
                ident
                for name, m in maps.items()
                if expected_namespace[name] == "git blob sha1"
                for ids in m.values()
                for ident in ids
            }
        )
    )
    ledger_ns: Dict[str, Dict[str, Set[str]]] = {}
    for name, m in maps.items():
        converted: Dict[str, Set[str]] = {}
        for path, ids in m.items():
            bucket = converted.setdefault(path, set())
            for ident in ids:
                if len(ident) == 64:
                    bucket.add(ident)
                else:
                    c = sha_of_blob.get(ident)
                    assert c, (
                        "%s reported git blob %s for %s but its content "
                        "sha256 could not be computed" % (name, ident, path)
                    )
                    bucket.add(c)
        ledger_ns[name] = converted

    names = sorted(ledger_ns)
    reference = ledger_ns[names[0]]
    n_ref = sum(len(v) for v in reference.values())
    assert len(reference) >= 397 and n_ref >= 754, (
        "the ledger-namespace view has only %d paths / %d identifiers "
        "(MEASURED 2026-11-04: 397 / 754) — a degenerate view would make "
        "the agreement below vacuous" % (len(reference), n_ref)
    )

    for name in names[1:]:
        got = ledger_ns[name]
        missing_paths = sorted(set(reference) - set(got))
        extra_paths = sorted(set(got) - set(reference))
        assert not missing_paths and not extra_paths, (
            "%s disagrees with %s in the ledger's namespace: it MISSES %d "
            "path(s) %r and ADDS %d path(s) %r"
            % (name, names[0], len(missing_paths), missing_paths[:5],
               len(extra_paths), extra_paths[:5])
        )
        for path in sorted(set(reference) & set(got)):
            assert got[path] == reference[path], (
                "%s disagrees with %s on the committed CONTENT of %s in the "
                "ledger's namespace: %d identifier(s) differ"
                % (name, names[0], path,
                   len(got[path] ^ reference[path]))
            )


def test_the_internal_history_walker_speaks_the_git_blob_namespace() -> None:
    """The internal walker still speaks **git blob sha1** (40 hex).

    **CHANGED 2026-11-04 (wave-23).**  The comparison site now reaches
    ``wfa166_run_ledger_provenance`` through its public resolver
    ``committed_content_shas`` (content sha256).  The internal walker
    ``_history_raw`` still returns the git blob sha1 — and that claim is
    pinned HERE so the change of the comparison site does not silently drop
    it.  (The walker's SHAPE is deliberately NOT pinned: it is a low-level
    git walker, not a resolver, and it does not carry the API-shape
    contract.)
    """
    src_paths = _ledger_source_paths()[:40]
    _commit_paths, path_blobs = tool._history_raw(REPO_ROOT, src_paths)
    identifiers = {s for v in path_blobs.values() for s in v}
    assert identifiers, (
        "the internal walker returned no git blob identifier at all — the "
        "walker is broken"
    )
    wrong = sorted(s for s in identifiers if len(s) != 40)
    assert not wrong, (
        "the internal history walker returned %d identifier(s) that are not "
        "40-hex git blob shas %r — its namespace is part of its contract"
        % (len(wrong), wrong[:3])
    )
    # and the public resolver is the CONTENT sha256 of those very blobs
    resolved = tool.committed_content_shas(REPO_ROOT, src_paths)
    from molmetal.scripts import wfa166_run_ledger_provenance as _prov
    content = _prov._batch_content_shas(
        REPO_ROOT, sorted({s for v in path_blobs.values() for s in v})
    )
    sample = next(iter(path_blobs))
    assert content[next(iter(path_blobs[sample]))] in resolved[sample], (
        "the public resolver does not return the CONTENT sha256 of the blob "
        "the internal walker names — the two namespaces are not reconciled"
    )


def test_all_five_implementations_agree_on_the_paths_with_history() -> None:
    """Every implementation of the question must give the SAME answer.

    MEASURED 2026-11-04: all five agree on **397** of the ledger's **401**
    referenced paths (the other 4 have no committed version at all).  The
    question is not "do they agree" — it is "can a reader TELL when they
    stop agreeing", because the failure mode this wave found was two
    implementations answering the same question with different SQL.
    """
    sets = _paths_with_history_per_implementation(_ledger_source_paths())
    ref_name = "run_ledger"
    ref = sets[ref_name]
    for name, got in sorted(sets.items()):
        if name == ref_name:
            continue
        only_ref = sorted(ref - got)
        only_got = sorted(got - ref)
        assert not only_ref and not only_got, (
            "the ledger-provenance implementations DISAGREE on which paths "
            "have committed history: %s sees %d path(s) %s does not (%r); "
            "%s sees %d path(s) %s does not (%r)"
            % (
                ref_name, len(only_ref), name, only_ref[:5],
                name, len(only_got), ref_name, only_got[:5],
            )
        )


# ---------------------------------------------------------------------------
# 7. the merge-diff MODE is not a different QUERY
# ---------------------------------------------------------------------------
def _raw_blob_and_attribution(
    flag: Optional[str], src_paths: List[str]
) -> Tuple[Dict[str, set], Dict[str, int]]:
    """-> (path -> blob shas, commit -> number of raw diff lines).

    ``flag`` is the merge-diff mode: ``None`` (git's default, which
    SUPPRESSES a merge's own diff), ``"--diff-merges=first-parent"`` or
    ``"-m"``.
    """
    args = ["log", "--all", "--format=%H", "--raw", "--no-abbrev", "--no-renames"]
    if flag:
        args.append(flag)
    args += ["--", *src_paths]
    proc = _git(REPO_ROOT, *args)
    by_path: Dict[str, set] = {}
    per_commit: Dict[str, int] = {}
    current: Optional[str] = None
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if len(stripped) == 40 and all(c in "0123456789abcdef" for c in stripped):
            current = stripped
            per_commit.setdefault(current, 0)
            continue
        if not line.startswith(":"):
            continue
        meta, _, path = line.partition("\t")
        parts = meta.split()
        if len(parts) < 4 or not path:
            continue
        if set(parts[3]) == {"0"}:
            continue  # a deletion, not a version
        if current is not None:
            per_commit[current] += 1
        by_path.setdefault(path, set()).add(parts[3])
    return by_path, per_commit


def test_the_merge_diff_mode_changes_attribution_not_the_content_set() -> None:
    """THE FLAG IS LOAD-BEARING; THE MODE IS NOT.

    MEASURED 2026-11-04 on this repository: ``--diff-merges=first-parent``
    and ``-m`` return the IDENTICAL committed-content set over the ledger's
    401 referenced paths (**397** paths / **754** blobs each), and the
    identical set over the whole tree (**9835** paths / **17308** blobs
    each).  What differs is COMMIT ATTRIBUTION — the merge ``8289a7f5`` is
    credited with **585** raw diff lines under first-parent and **1351**
    under ``-m`` (git re-emits the merge's own sha before the second
    parent's diff).

    ⇒ A mode is not a different QUERY.  Comparing the two modes as if they
    were is how this wave produced a FALSE "``-m`` is strictly more
    complete" — which the measurement then refuted.  This test pins the
    fact so the claim cannot be re-derived wrong, and it FAILS LOUDLY if
    git ever changes the relationship (in which case the premise must be
    re-measured, not silently kept).
    """
    src_paths = _ledger_source_paths()
    fp_paths, fp_commits = _raw_blob_and_attribution("--diff-merges=first-parent", src_paths)
    m_paths, m_commits = _raw_blob_and_attribution("-m", src_paths)

    assert fp_paths == m_paths, (
        "the two merge-diff MODES now return DIFFERENT committed-content "
        "sets (%d vs %d paths; %d vs %d DISTINCT blobs; %d vs %d blobs "
        "summed per path) — if git changed, this wave's premise must be "
        "re-measured, not silently kept"
        % (
            len(fp_paths), len(m_paths),
            len({s for v in fp_paths.values() for s in v}),
            len({s for v in m_paths.values() for s in v}),
            sum(len(v) for v in fp_paths.values()),
            sum(len(v) for v in m_paths.values()),
        )
    )

    # the modes are NOT the same query — they differ in commit attribution
    assert sum(fp_commits.values()) != sum(m_commits.values()), (
        "the two merge-diff modes now attribute raw diff lines IDENTICALLY "
        "(%d each) — the attribution half of this test's premise has "
        "changed and must be re-measured"
        % sum(fp_commits.values())
    )
    differing = [
        c for c in set(fp_commits) | set(m_commits)
        if fp_commits.get(c, 0) != m_commits.get(c, 0)
    ]
    assert differing, (
        "no commit's attribution differs between the two modes — the "
        "attribution half of this test's premise has changed"
    )


# ---------------------------------------------------------------------------
# 8. the SECOND instance of the blind spot
# ---------------------------------------------------------------------------
def test_the_wfa198_auditor_also_finds_the_merge_only_file() -> None:
    """A read-only sweep found THREE implementations of the same query.

    MEASURED 2026-11-04: sweeping the repository for "every committed
    content version of a path" found five implementations.  Two already
    carried a merge-diff flag; the producer (``run_ledger.py``) and the
    Wf-A198 auditor (``wfa198_ledger_pin_consistency.py``) did not.  Both
    were repaired.  This test pins the SECOND one.

    THE CLASS WAS ALREADY DIAGNOSED IN THIS REPO — in the SIBLING module
    ``wfa156_ledger_provenance_guard.py``, whose docstring says verbatim:
    *"``-m`` is load-bearing: without it git prints no diff for a MERGE
    commit, so a content version introduced by a merge is invisible."*
    ⇒ THE FIX WAS NEVER PROPAGATED FROM THE SIBLING THAT DIAGNOSED IT.
    """
    from molmetal.scripts import wfa198_ledger_pin_consistency as wfa198

    found = _merge_only_path_on_disk(REPO_ROOT)
    assert found is not None, (
        "no merge-only path found in this repository — the real-world shape "
        "the defect needs has disappeared (that is worth knowing, but this "
        "test must be updated rather than silently pass)"
    )
    merge, rel = found
    blobs = wfa198._committed_blob_shas(REPO_ROOT, [rel])
    assert rel in blobs, (
        "merge %s holds %r, but the Wf-A198 auditor cannot see it"
        % (merge[:12], rel)
    )
    assert blobs[rel], "merge %s: the path resolved to no blob" % merge[:12]
