# SPDX-License-Identifier: Apache-2.0
"""Wf-A134 — census re-quote guard.

WHY THIS EXISTS
---------------
Wf-A134 rebuilt ``metrics/by_metric/cfm_run_census.json`` (the artifact had
drifted away from the run directories) and re-pointed ``registry/metrics.yaml``
at the rebuilt file, but the wave DIED before writing the regression guard
that pins the rebuild.  Without the guard:

  * nothing checks that the registry row still points at the artifact that is
    actually ON DISK (the sha is recomputed here, never hard-coded);
  * nothing stops a PER-RUN number (``fragmentation_summary.n_fail``) from
    being re-quoted as the POOL number (``fragmented_ratio``).  The two live
    in DIFFERENT artifacts — the pool measurement is in
    ``metrics/by_metric/cfm_fragmentation_audit.json`` — and the boundary
    between them is invisible in prose;
  * nothing checks the DOCUMENTS.  Every document that quotes the census was
    re-quoted by hand, and each one is a chance to publish a number the
    artifact no longer carries.

WHAT THIS GUARD ASSERTS
-----------------------
1. ``registry/metrics.yaml``'s ``cfm_run_census`` row carries the sha256 of
   the artifact on disk (computed here, compared — never a literal).
2. ``fragmentation_summary`` carries EXACTLY a pinned key set, and carries NO
   pool-level key (``fragmented_ratio`` / ``measured_rate`` /
   ``largest_fragment_frac_mean``).  The census also must not carry the
   pool's measured VALUES.
3. The census numbers the documents quote are the ones the artifact carries.
4. Every document that quotes the census's flip set or its headline rates
   quotes the artifact's value.  Stale sites are reported with file, line,
   quoted text and the correct value.
   **THIS TEST IS EXPECTED TO FAIL ON TODAY'S TREE**: the rebuild moved the
   numbers and the documents have not caught up.  It must fail LOUDLY, with a
   precise report — it must not pass silently, and the assertion must not be
   weakened to make it green.
5. The artifact carries parseable generation metadata, and the builder script
   is importable on CPU (so the rebuild is reproducible in principle).

HISTORICAL-TEXT CONVENTION (AGENTS.md §5.5 rule #6)
---------------------------------------------------
A superseded number may stay in a document — corrections are ADDITIVE and
DATED, never deletions.  The guard therefore does not ban the old string; it
bans the old string being presented as the CURRENT claim.  A quote is
treated as HISTORICAL (and not reported) when EITHER

  * it is struck through (``~~0.03125~~``), OR
  * the line it starts on carries a historical marker
    (``STALE`` / ``superseded`` / ``preserved verbatim`` / ``historical`` /
    ``PREVIOUS`` / ``formerly`` / ``no longer`` / ``struck``), OR
  * the text immediately before it is past-tense
    (``was 0.057291666666666664``), OR
  * (flip-set claims only) the enclosing paragraph says ``preserved verbatim``.

A dated correction that follows this convention passes.  A correction that
merely mentions the old value with no marker is reported again — that is the
point.

NON-VACUITY
-----------
``test_guard_detects_a_mutated_artifact`` and
``test_doc_scan_detects_an_injected_stale_quote`` mutate a COPY of the
artifact / inject a stale quote and assert that the SAME check functions
report problems.  If this guard were removed, these are the assertions that
would have caught the drift:

  * ``test_registry_row_points_at_the_artifact_on_disk``          (sha mismatch)
  * ``test_fragmentation_summary_carries_exactly_the_pinned_key_set``
  * ``test_fragmentation_summary_carries_no_pool_level_key``
  * ``test_census_artifact_carries_no_pool_level_value``
  * ``test_pinned_census_values_match_the_artifact``
  * ``test_documents_quote_the_artifacts_flip_set``
  * ``test_documents_quote_the_artifacts_headline_rates``

CPU ONLY.  This module never touches a GPU: it reads JSON/YAML/Markdown and
imports the builder with ``CUDA_VISIBLE_DEVICES=""``.
"""

from __future__ import annotations

import copy
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
ARTIFACT_REL = "metrics/by_metric/cfm_run_census.json"
ARTIFACT = REPO / ARTIFACT_REL
POOL_AUDIT_REL = "metrics/by_metric/cfm_fragmentation_audit.json"
POOL_AUDIT = REPO / POOL_AUDIT_REL
REGISTRY_REL = "registry/metrics.yaml"
REGISTRY = REPO / REGISTRY_REL
BUILDER_REL = "molmetal/scripts/wf_cfm_run_census.py"
BUILDER = REPO / BUILDER_REL
METRIC_ID = "cfm_run_census"
POOL_RUN = "cfm_c7_fullgen_2026_10_20"

# --------------------------------------------------------------------------
# PINS
# --------------------------------------------------------------------------

#: EXACT key set of ``fragmentation_summary`` in the rebuilt artifact.
PINNED_FS_KEYS = frozenset(
    {
        "n_runs_with_verdict",
        "n_pass",
        "n_fail",
        "n_skip",
        "failing_runs",
        "runs_with_a_recorded_number",
        "n_runs_with_an_end_to_end_rate",
        "end_to_end_single_molecule_rate",
        "best_epoch_single_molecule_rate",
        "end_to_end_single_molecule_rate_all_runs",
        "best_epoch_single_molecule_rate_all_runs",
        "degenerate_runs_excluded_from_the_headline",
        "degenerate_note",
        "verdict_counts_clean_only",
        "retracted_max",
        "runs_collapsed_at_their_last_epoch",
        "retracted_numbers",
        "collapsed_note",
    }
)

#: Keys that belong to the POOL measurement (cfm_fragmentation_audit.json).
#: ``fragmentation_summary`` is a per-RUN block and must never carry them.
POOL_LEVEL_KEYS = (
    "fragmented_ratio",
    "measured_rate",
    "largest_fragment_frac_mean",
    "single_fragment_ratio",
    "single_molecule_rate",
    "single_molecule_rate_over_pool_rows",
    "single_molecule_rate_over_declared_denominator",
    "n_scored",
    "n_rows_in_csv",
    "n_single",
)

#: The census values the documents quote (MEASURED 2026-10-31, Wf-A134 rebuild).
PINNED_CENSUS_VALUES = {
    "fragmentation_summary.n_runs_with_verdict": 59,
    "fragmentation_summary.n_pass": 5,
    "fragmentation_summary.n_fail": 25,
    "fragmentation_summary.n_skip": 29,
    "fragmentation_summary.n_runs_with_an_end_to_end_rate": 27,
    "fragmentation_summary.end_to_end_single_molecule_rate.n": 27,
    "fragmentation_summary.end_to_end_single_molecule_rate.mean": 0.17755582555110858,
    "fragmentation_summary.end_to_end_single_molecule_rate.median": 0.0625,
    "fragmentation_summary.end_to_end_single_molecule_rate_all_runs.n": 30,
    "budget_matched.flipped_budgets": [9, 11, 25],
}

#: Renderings the documents still carry from BEFORE the Wf-A134 rebuild.
#: (artifact path of the CURRENT value, regex of the STALE rendering, template
#: for the correct text).  These are the values the docs quote today; after
#: the main thread re-quotes the docs they must be gone (or marked historical).
STALE_RENDERINGS = (
    (
        "fragmentation_summary.end_to_end_single_molecule_rate.median",
        r"0\.057291666666666664",
        "{v}",
    ),
    (
        "fragmentation_summary.end_to_end_single_molecule_rate.mean",
        r"0\.16531059620275626",
        "{v}",
    ),
    (
        "fragmentation_summary.end_to_end_single_molecule_rate.n_below_0.50",
        r"\b25 of 29\b",
        "{v} of {n}",
    ),
    (
        "fragmentation_summary.end_to_end_single_molecule_rate.n_below_0.10",
        r"\b18 below 0\.10\b",
        "{v} below 0.10",
    ),
    (
        "fragmentation_summary.end_to_end_single_molecule_rate.n",
        r"\b29 of those are CLEAN\b",
        "{v} of those are CLEAN",
    ),
    (
        "fragmentation_summary.end_to_end_single_molecule_rate.n",
        r"\b29 CLEAN of 32\b",
        "{v} CLEAN of {n_all}",
    ),
    (
        "fragmentation_summary.end_to_end_single_molecule_rate_all_runs.median",
        r"0\.05989583333333333",
        "{v}",
    ),
    (
        "fragmentation_summary.end_to_end_single_molecule_rate_all_runs.mean",
        r"0\.17213911053246944",
        "{v}",
    ),
    (
        "fragmentation_summary.end_to_end_single_molecule_rate_all_runs.n_below_0.50",
        r"\b27 of 32\b",
        "{v} of {n}",
    ),
    (
        "fragmentation_summary.end_to_end_single_molecule_rate_all_runs.n_below_0.10",
        r"\b19 below 0\.10\b",
        "{v} below 0.10",
    ),
    (
        "fragmentation_summary.end_to_end_single_molecule_rate_all_runs.n",
        r"\b32 carry both columns\b",
        "{v} carry both columns",
    ),
    (
        "fragmentation_summary.n_fail",
        r"PASS/FAIL/SKIP \*\*5 / 27 / 27\*\*",
        "PASS/FAIL/SKIP {pass} / {fail} / {skip}",
    ),
    (
        "fragmentation_summary.verdict_counts_clean_only.n_fail",
        r"\b4 / 25 / 26\b",
        "{clean_pass} / {clean_fail} / {clean_skip} (over the CLEAN runs)",
    ),
    # --- the pre-Wf-A116/A121 generation of the same census numbers ---------
    # Each of these values is ABSENT from the current artifact (verified by
    # ``test_legacy_renderings_are_absent_from_the_artifact``), so a document
    # quoting one is quoting a superseded census value.  The rendering with a
    # decimal-dot tail (``0.03125``) is deliberately NOT listed: that value is
    # still in the artifact as a per-run number, so a document quoting it may
    # be quoting a run, not the median.
    (
        "fragmentation_summary.end_to_end_single_molecule_rate.mean",
        r"0\.1000971460292581",
        "{v}",
    ),
    (
        "fragmentation_summary.end_to_end_single_molecule_rate_all_runs.median",
        r"0\.03580729166666667",
        "{v}",
    ),
    (
        "fragmentation_summary.end_to_end_single_molecule_rate_all_runs.mean",
        r"0\.11303942131273674",
        "{v}",
    ),
    (
        "fragmentation_summary.best_epoch_single_molecule_rate_all_runs.median",
        r"0\.18229166666666669",
        "{v}",
    ),
    (
        "fragmentation_summary.best_epoch_single_molecule_rate.mean",
        r"0\.2990913989028213",
        "{v}",
    ),
    (
        "fragmentation_summary.best_epoch_single_molecule_rate_all_runs.mean",
        r"0\.2933779629794034",
        "{v}",
    ),
    (
        "fragmentation_summary.end_to_end_single_molecule_rate.median",
        r"0\.04166875(?!\d)",
        "{v}",
    ),
    (
        "fragmentation_summary.best_epoch_single_molecule_rate.median",
        r"0\.23827330729166665",
        "{v}",
    ),
    (
        "fragmentation_summary.best_epoch_single_molecule_rate.max",
        r"0\.9374541666666667",
        "{v}",
    ),
    # --- the by_n_train cohort medians --------------------------------------
    (
        "by_n_train.26715.clean_only.last_epoch.median",
        r"0\.09635416666666666",
        "{v} (n={n})",
    ),
    (
        "by_n_train.256.clean_only.last_epoch.median",
        r"0\.026041666666666664",
        "{v} (n={n})",
    ),
)

# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def load_artifact(path: Path = ARTIFACT) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_registry(path: Path = REGISTRY) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def registry_row(registry: dict, metric_id: str = METRIC_ID) -> dict:
    row = (registry.get("metrics") or {}).get(metric_id)
    assert isinstance(row, dict), f"registry/metrics.yaml has no row for {metric_id!r}"
    return row


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def census_value(artifact: dict, dotted: str):
    """Resolve a dotted path.

    Some artifact keys contain a dot of their own (``n_below_0.50``), so the
    path is resolved by longest-match rather than by splitting on every dot.
    """
    parts = dotted.split(".")
    node = artifact
    index = 0
    while index < len(parts):
        chosen = None
        for end in range(len(parts), index, -1):
            candidate = ".".join(parts[index:end])
            if isinstance(node, dict) and candidate in node:
                chosen = (candidate, end)
                break
        if chosen is None:
            raise KeyError(dotted)
        node = node[chosen[0]]
        index = chosen[1]
    return node


def _deepest_dict_path(artifact: dict, dotted: str) -> str:
    """The dotted path of the deepest dict ancestor of ``dotted``."""
    parts = dotted.split(".")
    node = artifact
    index = 0
    path = ""
    while index < len(parts):
        chosen = None
        for end in range(len(parts), index, -1):
            candidate = ".".join(parts[index:end])
            if isinstance(node, dict) and candidate in node:
                chosen = (candidate, end)
                break
        if chosen is None:
            break
        candidate, end = chosen
        node = node[candidate]
        index = end
        if isinstance(node, dict):
            path = candidate if not path else path + "." + candidate
        else:
            break
    return path


def _sibling_n(artifact: dict, dotted: str):
    """The ``n`` of the population the claim belongs to."""
    parent = _deepest_dict_path(artifact, dotted)
    if not parent:
        return None
    try:
        value = census_value(artifact, parent + ".n")
    except (KeyError, TypeError):
        return None
    return value if isinstance(value, int) else None


def _render_correct(artifact: dict, dotted: str, template: str) -> str:
    fs = artifact.get("fragmentation_summary") or {}
    clean = fs.get("verdict_counts_clean_only") or {}
    allruns = fs.get("end_to_end_single_molecule_rate_all_runs") or {}
    ctx = {
        "v": census_value(artifact, dotted),
        "n": _sibling_n(artifact, dotted),
        "n_all": allruns.get("n"),
        "pass": fs.get("n_pass"),
        "fail": fs.get("n_fail"),
        "skip": fs.get("n_skip"),
        "clean_pass": clean.get("n_pass"),
        "clean_fail": clean.get("n_fail"),
        "clean_skip": clean.get("n_skip"),
    }
    out = template
    for key, value in ctx.items():
        out = out.replace("{" + key + "}", str(value))
    return out


# --------------------------------------------------------------------------
# check functions (pure — they take the artifact so the mutation test can
# call them on a mutated COPY)
# --------------------------------------------------------------------------


def check_registry_sha(registry: dict, artifact_path: Path = ARTIFACT) -> list[str]:
    problems: list[str] = []
    row = registry_row(registry)
    on_disk = sha256_of(artifact_path)
    declared = row.get("source_json_sha256")
    if declared != on_disk:
        problems.append(
            f"registry row {METRIC_ID!r} declares source_json_sha256={declared!r} but the "
            f"artifact on disk hashes to {on_disk!r} (re-point the registry row)"
        )
    if row.get("source_json") != ARTIFACT_REL:
        problems.append(
            f"registry row {METRIC_ID!r} points at {row.get('source_json')!r}, "
            f"expected {ARTIFACT_REL!r}"
        )
    return problems


def check_fs_keys(artifact: dict) -> list[str]:
    problems: list[str] = []
    fs = artifact.get("fragmentation_summary")
    if not isinstance(fs, dict):
        return ["the artifact carries no `fragmentation_summary` block"]
    got = set(fs)
    missing = sorted(PINNED_FS_KEYS - got)
    extra = sorted(got - PINNED_FS_KEYS)
    if missing:
        problems.append(f"fragmentation_summary is MISSING pinned keys: {missing}")
    if extra:
        problems.append(f"fragmentation_summary carries UNEXPECTED keys: {extra}")
    return problems


def _walk(obj, path: str = ""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            here = f"{path}/{key}"
            yield here, str(key), value
            yield from _walk(value, here)
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            yield from _walk(value, f"{path}[{index}]")


def check_no_pool_level_keys(artifact: dict) -> list[str]:
    """The ATTRIBUTION boundary: the census is a per-RUN artifact."""
    problems: list[str] = []
    fs = artifact.get("fragmentation_summary")
    if not isinstance(fs, dict):
        return ["the artifact carries no `fragmentation_summary` block"]
    for path, key, _value in _walk(fs, "fragmentation_summary"):
        if key in POOL_LEVEL_KEYS:
            problems.append(
                f"fragmentation_summary carries the POOL-level key {key!r} at {path} — the pool "
                "measurement lives in metrics/by_metric/cfm_fragmentation_audit.json, not here"
            )
    return problems


def pool_measurement() -> dict:
    """The pool-level measurement, read from the artifact that owns it."""
    audit = load_artifact(POOL_AUDIT)
    for entry in audit.get("runs") or []:
        if isinstance(entry, dict) and entry.get("run") == POOL_RUN:
            return entry
    raise AssertionError(f"{POOL_AUDIT_REL} has no run {POOL_RUN!r}")


def check_no_pool_level_values(artifact: dict, pool: dict) -> list[str]:
    """The census must not carry the pool's measured VALUES either."""
    problems: list[str] = []
    forbidden = {
        key: pool[key]
        for key in (
            "fragmented_ratio",
            "largest_fragment_frac_mean",
            "single_fragment_ratio",
            "single_molecule_rate_over_declared_denominator",
            "n_scored",
            "n_rows_in_csv",
            "n_single",
        )
        if key in pool
    }
    numbers: dict[float, list[str]] = {}
    for path, _key, value in _walk(artifact, ""):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        numbers.setdefault(float(value), []).append(path)
    for key, value in forbidden.items():
        probe = float(value)
        if probe in numbers:
            problems.append(
                f"the census carries the POOL-level value {key}={value!r} at "
                f"{numbers[probe][0]} — that number is measured in "
                f"{POOL_AUDIT_REL}, not in the census"
            )
    return problems


def check_pinned_values(artifact: dict) -> list[str]:
    problems: list[str] = []
    for dotted, want in PINNED_CENSUS_VALUES.items():
        try:
            got = census_value(artifact, dotted)
        except (KeyError, TypeError):
            problems.append(f"{dotted} is missing from the artifact (pinned: {want!r})")
            continue
        if isinstance(want, list):
            got = list(got) if got is not None else None
        if got != want:
            problems.append(f"{dotted} = {got!r}, the pinned value is {want!r}")
    return problems


def check_generation_metadata(artifact: dict, registry: dict) -> list[str]:
    problems: list[str] = []
    changelog = artifact.get("changelog")
    if not isinstance(changelog, dict) or not changelog:
        problems.append("the artifact carries no `changelog` generation block")
        changelog = {}
    for wave, entries in sorted(changelog.items()):
        if not isinstance(entries, list) or not entries:
            problems.append(f"changelog[{wave!r}] is not a non-empty list of notes")
            continue
        for entry in entries:
            if not isinstance(entry, str) or not entry.strip():
                problems.append(f"changelog[{wave!r}] carries an empty note")
    corpus = artifact.get("corpus")
    if (
        not isinstance(corpus, dict)
        or not corpus.get("loader")
        or not corpus.get("series_source")
    ):
        problems.append(
            "the artifact carries no `corpus` generation metadata (loader / series_source)"
        )
    dates: set[str] = set()
    for entries in changelog.values():
        for entry in entries or []:
            if isinstance(entry, str):
                dates.update(re.findall(r"\d{4}-\d{2}-\d{2}", entry))
    if not dates:
        problems.append(
            "the artifact carries no parseable ISO-8601 date in its changelog notes "
            "(the artifact has no explicit timestamp field of its own)"
        )
    for value in sorted(dates):
        try:
            dt.date.fromisoformat(value)
        except ValueError:
            problems.append(f"changelog date {value!r} does not parse as ISO-8601")
    generated_at = registry.get("generated_at")
    try:
        dt.datetime.fromisoformat(str(generated_at))
    except Exception as exc:  # noqa: BLE001 - report, do not raise
        problems.append(f"registry generated_at {generated_at!r} does not parse as ISO-8601: {exc}")
    row = registry_row(registry)
    try:
        dt.datetime.fromisoformat(str(row.get("last_refreshed")))
    except Exception as exc:  # noqa: BLE001 - report, do not raise
        problems.append(
            f"registry row last_refreshed {row.get('last_refreshed')!r} does not parse as "
            f"ISO-8601: {exc}"
        )
    if not row.get("registry_origin"):
        problems.append("the registry row carries no `registry_origin`")
    return problems


# --------------------------------------------------------------------------
# document scanning
# --------------------------------------------------------------------------

#: A paragraph only counts as a census quote if it says so.
DOC_CONTEXT_RE = re.compile(
    r"cfm_run_census|fragmentation_summary|end_to_end_single_molecule_rate|budget_matched"
    r"|flipped_budgets|\bcensus\b|last[- ]epoch|all-runs|\bALL runs\b|\bCLEAN\b",
    re.I,
)

#: A paragraph that reads a SINGLE run's own metrics is not a census quote.
#: The Wf-A128 comparison report quotes ``e2e(B) = 0.057291666666666664`` —
#: a per-run value (22/192 x 0.5) that COINCIDES with the census's old
#: median.  A census-rate scan that flagged it would be reporting a
#: coincidence as a stale claim.
PER_RUN_CONTEXT_RE = re.compile(
    r"per_epoch_metrics|reports/cfm_[a-z0-9_]+/|e2e\([AB]\)|per[- ]run",
    re.I,
)

#: A flip-set claim: the artifact's ``budget_matched.flipped_budgets``.
FLIP_ANCHOR_RE = re.compile(
    r"flipped_budgets"
    r"|flips?\s+at\s+\**K\b"
    r"|flipping\s+at\b"
    r"|flip\s+(?:set|points?|list)"
    r"|budget-matched\s+flip",
    re.I,
)

#: A line carrying one of these is presenting the value as history.
HIST_LINE_RE = re.compile(
    r"(?:\bSTALE\b|\bstale\b|\bSUPERSEDED\b|\bsuperseded\b|preserved verbatim"
    r"|preserved in the dated note|\bhistorical\b|\bHISTORICAL\b|\bPREVIOUS\b|\bprevious\b"
    r"|\bpreviously\b|\bformerly\b|no longer|\boutdated\b|\bstruck\b)"
)

#: A value quoted as a MINIMUM is a per-cohort minimum, not the median claim
#: this guard polices (the artifact still carries the old median as a min).
MIN_CLAIM_RE = re.compile(r"\bmin(?:imum)?\b", re.I)

#: Past-tense right before the quote: ``was 0.057291666666666664``.
PRE_VALUE_HIST_RE = re.compile(
    r"(?:\bwas\b|\bformerly\b|\bpreviously\b|\bsuperseded\b|\bstale\b|\bSTALE\b)\s*[*_~`]*\s*$"
)

#: The paragraph itself says the quoted numbers are the OLD ones.  A dated
#: correction that quotes the superseded text ("This paragraph originally
#: read ...", "the table above originally read ...") is preserving history
#: under AGENTS.md §5.5 rule #6, not publishing a stale claim.
HIST_PARA_RE = re.compile(r"preserved verbatim|originally read|originally said", re.I)

#: A table COLUMN whose header marks the cell as historical.  A markdown table
#: states which cell is the OLD value and which is the CURRENT one in its
#: header — "stale claim | corrected (MEASURED)", "before | after".  Without
#: this, every "was" cell of a re-quote table would be reported as a live
#: stale claim.
HIST_HEADER_RE = re.compile(
    r"(?:\bSTALE\b|\bstale\b|\bwas\b|\bold\b|\bbefore\b|\bprevious\b|\bformerly\b"
    r"|\boriginal(?:ly)?\b|\bsuperseded\b|\bhistorical\b|\bpre\b|\bdeprecated\b"
    r"|\bat the time\b|\bthen-current\b)",
    re.I,
)

#: A K-list must not be the head of a "n collapses from 10 at K = 5 to 9 at
#: K = 9, 7 at K = 11, ..." profile — there the comma joins two DIFFERENT
#: K-claims.  The lookahead rejects a list followed by " at K".
_FLIP_LIST_RES = (
    re.compile(
        r"K\s*=\s*(\d+(?:\s*,\s*\d+)*(?:\s*(?:,|and|&)\s*\d+)?)(?!\s+at\s+K\b)"
    ),
    re.compile(r"K\s*=\s*(\d+)\s*/\s*(\d+)\s*/\s*(\d+)(?!\s+at\s+K\b)"),
    re.compile(r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]"),
)


def doc_files(root: Path = REPO) -> list[str]:
    """Every markdown file in the repo except ``paper/**`` (TODO-84).

    Tracked AND untracked-but-not-ignored: the census's own home documents
    (``docs/CFM_FRAGMENTATION_AUDIT_2026_10_30.md`` and the two Wf-A114 TODO
    files) are still untracked, so a tracked-only scan would miss them.
    ``git ls-files`` is used instead of a filesystem walk — the repo is 201 GB
    with huge ``.pt`` files and an ``os.walk`` once took out a whole tmux
    session.
    """
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "*.md"],
        cwd=str(root),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return sorted(f for f in out if not f.startswith("paper/"))


def _paragraphs(lines: list[str]):
    """Yield (first_line_no, last_line_no, [(line_no, text), ...])."""
    index = 0
    total = len(lines)
    while index < total:
        if not lines[index].strip():
            index += 1
            continue
        end = index
        while end < total and lines[end].strip():
            end += 1
        yield index + 1, end, [(k + 1, lines[k]) for k in range(index, end)]
        index = end


def _flatten(block: list[tuple[int, str]]) -> tuple[str, list[int]]:
    """Flatten a paragraph: drop blockquote / bullet markers, blank out
    ``~~struck~~`` spans, and keep a per-character line-number map so a match
    can be reported at the line it starts on."""
    chars: list[str] = []
    owners: list[int] = []
    for position, (lineno, raw) in enumerate(block):
        if position:
            chars.append(" ")
            owners.append(lineno)
        text = re.sub(r"^\s*(?:>\s*)?(?:[-*+]\s+|\d+[.)]\s+)?", "", raw)
        struck = list(text)
        for match in re.finditer(r"~~[^~]*~~", text):
            for position_in_line in range(match.start(), match.end()):
                struck[position_in_line] = " "
        text = "".join(struck)
        for char in text:
            chars.append(char)
            owners.append(lineno)
    return "".join(chars), owners


class _Offset:
    """Minimal stand-in for a regex match (only ``start()`` is needed)."""

    __slots__ = ("offset",)

    def __init__(self, offset: int) -> None:
        self.offset = offset

    def start(self) -> int:
        return self.offset


def _blank_struck(text: str) -> str:
    """Blank out ``~~struck~~`` spans, preserving the character offsets."""
    out = list(text)
    for match in re.finditer(r"~~[^~]*~~", text):
        for position in range(match.start(), match.end()):
            out[position] = " "
    return "".join(out)


def _table_rows(block: list[tuple[int, str]]):
    """Parse a contiguous markdown table into rows of cells.

    Returns ``None`` when the block is not a table.  The first row is the
    header; the second is the ``|---|---|`` separator.
    """
    rows = []
    for lineno, raw in block:
        stripped = raw.strip()
        if not stripped.startswith("|"):
            return None
        cells = stripped.split("|")
        if cells and not cells[0].strip():
            cells = cells[1:]
        if cells and not cells[-1].strip():
            cells = cells[:-1]
        rows.append((lineno, cells))
    if len(rows) < 2:
        return None
    return rows


def _units(block: list[tuple[int, str]]):
    """Yield ``(unit_text, owners, header)`` for each matchable unit.

    A table yields one unit per CELL (so a "before" column and a "now"
    column are judged separately); any other paragraph yields one unit for
    the whole block.
    """
    rows = _table_rows(block)
    if rows is None:
        text, owners = _flatten(block)
        yield text, owners, None
        return
    header = [cell.strip() for cell in rows[0][1]]
    for lineno, cells in rows[1:]:
        if cells and all(set(cell.strip()) <= set("-: ") for cell in cells):
            continue  # the |---|---| separator row
        for index, cell in enumerate(cells):
            head = header[index] if index < len(header) else ""
            yield _blank_struck(cell), [lineno] * (len(cell) + 1), head


def _raw_line(lines: list[str], owners: list[int], match: "re.Match[str]") -> str:
    """The raw source line a match starts on."""
    lineno = owners[match.start()]
    return lines[lineno - 1] if 0 < lineno <= len(lines) else ""


def _is_historical(
    unit_raw: str,
    unit_text: str,
    match: "re.Match[str]",
    *,
    scope_text: str = "",
) -> bool:
    """Is this quote presented as history rather than as the current claim?"""
    if HIST_LINE_RE.search(unit_raw):
        return True
    window = unit_text[max(0, match.start() - 60) : match.start()]
    if PRE_VALUE_HIST_RE.search(window):
        return True
    if HIST_PARA_RE.search(scope_text):
        return True
    return False


def _flip_lists(text: str) -> list[tuple[int, list[int]]]:
    """Every K-list in a paragraph: (offset, [k, ...])."""
    found: list[tuple[int, list[int]]] = []
    for pattern in _FLIP_LIST_RES:
        for match in pattern.finditer(text):
            if pattern.groups == 3:
                numbers = [int(match.group(1)), int(match.group(2)), int(match.group(3))]
            else:
                numbers = [int(part) for part in re.findall(r"\d+", match.group(1))]
            found.append((match.start(), numbers))
    return found


def scan_flip_sites(artifact: dict, root: Path = REPO, files=None) -> list[dict]:
    """Doc sites that quote a flip set other than the artifact's."""
    current = [int(k) for k in census_value(artifact, "budget_matched.flipped_budgets")]
    findings: list[dict] = []
    for rel in files if files is not None else doc_files(root):
        path = root / rel
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for _start, _end, block in _paragraphs(lines):
            para_text, _ = _flatten(block)
            if not FLIP_ANCHOR_RE.search(para_text):
                continue
            for unit_text, owners, header in _units(block):
                if header is not None and HIST_HEADER_RE.search(header):
                    continue  # the cell sits in a historical column
                for offset, numbers in _flip_lists(unit_text):
                    if len(numbers) < 2:
                        continue  # a single K mention ("K=5 leader 256") is not a flip set
                    if sorted(numbers) == sorted(current):
                        continue
                    scope = unit_text if header is not None else para_text
                    if _is_historical(
                        unit_text, unit_text, _Offset(offset), scope_text=scope
                    ):
                        continue
                    findings.append(
                        {
                            "file": rel,
                            "line": owners[offset],
                            "quoted": numbers,
                            "claim": "budget_matched.flipped_budgets",
                            "correct": current,
                            "text": unit_text.strip()[:400],
                        }
                    )
    return findings


def scan_stale_rate_sites(artifact: dict, root: Path = REPO, files=None) -> list[dict]:
    """Doc sites that quote a pre-rebuild census rate."""
    findings: list[dict] = []
    for rel in files if files is not None else doc_files(root):
        path = root / rel
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for _start, _end, block in _paragraphs(lines):
            para_text, _ = _flatten(block)
            if not DOC_CONTEXT_RE.search(para_text):
                continue
            for unit_text, owners, header in _units(block):
                if header is not None and HIST_HEADER_RE.search(header):
                    continue  # the cell sits in a historical column
                scope = unit_text if header is not None else para_text
                for dotted, pattern, template in STALE_RENDERINGS:
                    for match in re.finditer(pattern, unit_text):
                        if PER_RUN_CONTEXT_RE.search(
                            unit_text
                            if header is not None
                            else _raw_line(lines, owners, match)
                        ):
                            continue  # a single run's own metric, not an aggregate
                        if MIN_CLAIM_RE.search(unit_text):
                            continue  # quoted as a minimum, not as the headline
                        if _is_historical(
                            unit_text, unit_text, match, scope_text=scope
                        ):
                            continue
                        findings.append(
                            {
                                "file": rel,
                                "line": owners[match.start()],
                                "quoted": match.group(0),
                                "claim": dotted,
                                "correct": _render_correct(artifact, dotted, template),
                                "text": unit_text.strip()[:400],
                            }
                        )
    return findings


def scan_misattributed_pool_sites(root: Path = REPO, files=None) -> list[dict]:
    """Doc lines that attribute a POOL number to the census artifact."""
    pool = pool_measurement()
    values = {
        key: pool[key]
        for key in (
            "fragmented_ratio",
            "largest_fragment_frac_mean",
            "single_fragment_ratio",
            "single_molecule_rate_over_declared_denominator",
        )
        if key in pool
    }
    findings: list[dict] = []
    for rel in files if files is not None else doc_files(root):
        path = root / rel
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for lineno, raw in enumerate(lines, 1):
            if "cfm_run_census" not in raw:
                continue
            for key, value in values.items():
                if repr(value) in raw:
                    findings.append(
                        {
                            "file": rel,
                            "line": lineno,
                            "quoted": f"{key}={value!r}",
                            "claim": "attribution boundary",
                            "correct": f"{key} belongs to {POOL_AUDIT_REL}",
                            "text": raw.strip(),
                        }
                    )
    return findings


def _render_report(title: str, findings: list[dict]) -> str:
    lines = [f"STALE DOC SITES — {title} ({len(findings)} site(s))", ""]
    for finding in findings:
        lines.append(f"  {finding['file']}:{finding['line']}")
        lines.append(f"    claim   : {finding['claim']}")
        lines.append(f"    quoted  : {finding['quoted']}")
        lines.append(f"    correct : {finding['correct']}")
        lines.append(f"    line    : {finding['text'][:220]}")
        lines.append("")
    lines.append(
        "The main thread owns these documents.  Fix them ADDITIVELY and DATED "
        "(AGENTS.md §5.5 rule #6): keep the old number only if it is struck "
        "through or marked historical, and state the artifact's current value."
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------


def test_artifact_is_present_and_parseable():
    assert ARTIFACT.exists(), f"{ARTIFACT_REL} is missing"
    artifact = load_artifact()
    assert artifact.get("metric") == METRIC_ID
    assert isinstance(artifact.get("fragmentation_summary"), dict)
    assert isinstance(artifact.get("runs"), list)


def test_registry_row_points_at_the_artifact_on_disk():
    """The registry sha is COMPUTED from disk, never hard-coded."""
    problems = check_registry_sha(load_registry(), ARTIFACT)
    assert not problems, "\n".join(problems)


def test_fragmentation_summary_carries_exactly_the_pinned_key_set():
    problems = check_fs_keys(load_artifact())
    assert not problems, "\n".join(problems)


def test_fragmentation_summary_carries_no_pool_level_key():
    problems = check_no_pool_level_keys(load_artifact())
    assert not problems, "\n".join(problems)


def test_census_artifact_carries_no_pool_level_value():
    problems = check_no_pool_level_values(load_artifact(), pool_measurement())
    assert not problems, "\n".join(problems)


def test_pinned_census_values_match_the_artifact():
    problems = check_pinned_values(load_artifact())
    assert not problems, "\n".join(problems)


def test_stale_renderings_are_not_the_current_value():
    """A stale rendering must not BE the artifact's current value at the path
    the claim names.

    The check is PATH-SCOPED, not artifact-wide: the artifact still carries
    ``0.057291666666666664`` as a per-run and per-cohort MINIMUM (see
    ``runs[37].fragmentation_gate.measured_rate`` and
    ``by_cohort["26715/6714"].clean_only.last_epoch.min``).  A document
    quoting that number as a MINIMUM is right; a document quoting it as the
    CLEAN MEDIAN is stale.  The distinction is the claim's path, so that is
    what is compared.
    """
    artifact = load_artifact()
    offenders = []
    for dotted, pattern, _template in STALE_RENDERINGS:
        try:
            current = census_value(artifact, dotted)
        except (KeyError, TypeError):
            continue
        for rendered in (repr(current), str(current)):
            for match in re.finditer(pattern, rendered):
                offenders.append(
                    f"{dotted}: the artifact's current value {current!r} still renders as "
                    f"the stale form {match.group(0)!r}"
                )
    assert not offenders, "\n".join(offenders)


def test_documents_quote_the_artifacts_flip_set():
    artifact = load_artifact()
    findings = scan_flip_sites(artifact)
    assert not findings, _render_report("budget_matched.flipped_budgets", findings)


def test_documents_quote_the_artifacts_headline_rates():
    artifact = load_artifact()
    findings = scan_stale_rate_sites(artifact)
    assert not findings, _render_report("census headline rates", findings)


def test_no_document_attributes_a_pool_number_to_the_census():
    findings = scan_misattributed_pool_sites()
    assert not findings, _render_report("pool/census attribution", findings)


def test_artifact_generation_metadata_is_present_and_parseable():
    problems = check_generation_metadata(load_artifact(), load_registry())
    assert not problems, "\n".join(problems)


def test_builder_script_is_importable_on_cpu():
    """The rebuild is reproducible in principle: the builder imports on CPU.

    Importing only — a full census rebuild is expensive and is NOT run here.
    """
    assert BUILDER.exists(), f"{BUILDER_REL} is missing"
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('wf_cfm_run_census', {str(BUILDER)!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "assert 'torch' not in sys.modules, 'the builder pulled torch in'\n"
        "assert hasattr(mod, 'main'), 'the builder exposes no main()'\n"
        "print('IMPORT_OK')\n"
    )
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, f"builder import failed:\n{proc.stdout}\n{proc.stderr}"
    assert "IMPORT_OK" in proc.stdout, proc.stdout


# --------------------------------------------------------------------------
# non-vacuity
# --------------------------------------------------------------------------


def test_guard_detects_a_mutated_artifact(tmp_path):
    """NON-VACUITY: mutating the artifact must break the guard."""
    artifact = load_artifact()

    mutated = copy.deepcopy(artifact)
    mutated["fragmentation_summary"]["n_fail"] = 24
    mutated["fragmentation_summary"]["end_to_end_single_molecule_rate"]["median"] = 0.0626
    mutated["budget_matched"]["flipped_budgets"] = [9, 12, 14, 18, 20]
    problems = check_pinned_values(mutated)
    assert problems, "mutating n_fail / median / flipped_budgets did NOT break the value pin"

    smuggled = copy.deepcopy(artifact)
    smuggled["fragmentation_summary"]["fragmented_ratio"] = 0.9149104112400347
    problems = check_no_pool_level_keys(smuggled)
    assert problems, "smuggling `fragmented_ratio` in did NOT break the attribution boundary"

    value_smuggle = copy.deepcopy(artifact)
    value_smuggle["fragmentation_summary"]["measured_rate"] = 0.08508958875996527
    problems = check_no_pool_level_values(value_smuggle, pool_measurement())
    assert problems, "smuggling the pool VALUE in did NOT break the attribution boundary"

    extra = copy.deepcopy(artifact)
    extra["fragmentation_summary"]["an_extra_key"] = 1
    problems = check_fs_keys(extra)
    assert problems, "adding a key to fragmentation_summary did NOT break the key pin"

    mutated_path = tmp_path / "cfm_run_census.json"
    mutated_path.write_text(json.dumps(mutated, indent=2), encoding="utf-8")
    problems = check_registry_sha(load_registry(), mutated_path)
    assert problems, "mutating the artifact on disk did NOT break the registry sha check"


def test_doc_scan_detects_an_injected_stale_quote(tmp_path):
    """NON-VACUITY: the doc scanner must catch a stale quote."""
    artifact = load_artifact()
    injected = tmp_path / "injected.md"
    injected.write_text(
        "# injected\n\n"
        "The census's ordering flips at **K = 9, 12, 14, 18 and 20** "
        "(`budget_matched.flipped_budgets`).\n",
        encoding="utf-8",
    )
    findings = scan_flip_sites(artifact, root=tmp_path, files=["injected.md"])
    assert findings, "the doc scanner did NOT detect the injected stale flip set"
    assert findings[0]["quoted"] == [9, 12, 14, 18, 20]
    assert findings[0]["correct"] == [9, 11, 25]
