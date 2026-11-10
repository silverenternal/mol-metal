#!/usr/bin/env python3
"""Wf-A171 D3 — L5 protocol-validation hook threshold accounting.

WHAT THIS MODULE IS FOR
-----------------------
``molmetal/scripts/pre_flight_check.py`` implements the nine §4 "L5
per-baseline protocol validation" hooks (L5-Krasnov-1..3, L5-Rusanov-1..3,
L5-DrugOOD-1..3).  All nine funnel through ONE helper,
``_hook_passthrough_payload``, whose PASS branch keys ONLY on
``source_present`` (the source file exists) plus "the emitter wrote a
file".  It never compares the source against the threshold that
``TODO/03_plans/PRE_TRAINING_GATE_v2.md`` states for the hook, so a hook
can report PASS while the source is far below the spec threshold.

This module is the STANDALONE, IMPORTABLE, TORCH-FREE measurement half of
the repair: it reads the spec thresholds off disk, measures the real
record count of each source by JSON parsing, and returns a per-hook record
carrying ``threshold`` / ``measured count`` / ``meets_threshold`` /
``source_present`` / ``shortfall_n`` / ``shortfall_pp``.  A MISSING source
is reported as MISSING — never as 0.

WHERE THE NUMBERS COME FROM (READ THIS BEFORE QUOTING ANYTHING)
---------------------------------------------------------------
* THRESHOLDS ARE PARSED FROM DISK AT RUNTIME.  They are read out of the
  "Verdict gate" line of each §4 hook block in
  ``TODO/03_plans/PRE_TRAINING_GATE_v2.md`` by :func:`parse_plan_specs`.
  Nothing numeric is hard-coded here for the thresholds; if the plan doc
  changes, the thresholds move with it.
* THE COUNTING KEY + THE SOURCE PATH ARE **NOT** MACHINE-PARSEABLE FROM THE
  PLAN DOC.  The plan doc names the source FILES in prose ("current state:
  ...") but never names the JSON KEY that holds the records, and for two
  hooks it names a file that does not exist on disk.  Those two facts are
  therefore held in the clearly-marked constant block
  :data:`COUNTING_SOURCES`, where every entry cites the plan-doc line it
  came from.  THIS IS A CONSTANT BLOCK, NOT A PARSE — the docstring says so
  explicitly, as required by the Wf-A171 D3 brief.
* THE SPEC THRESHOLD IS NOT A UNIFORM RECORD COUNT.  Only THREE of the nine
  hooks carry a record-count threshold (125 / 45 / 12 records).  FOUR carry
  a VALUE threshold (``delta_imold > -0.02`` x3, ``delta > +0.03`` x1) and
  TWO carry a DOC-PRESENCE threshold ("a §6 paragraph exists for each
  Tier-1 LEAD metric x <baseline>").  The module models all four kinds
  (:class:`Threshold`) instead of pretending the threshold is always a
  count.

REPAIR SHAPE (honest, and it cannot turn a gate red today)
----------------------------------------------------------
The staged patch (``molmetal/reports/wfa171_d3_threshold/patch_d3_threshold.diff``)
leaves every VERDICT untouched by default and only enriches the evidence
string with ``threshold`` / ``measured`` / ``SHORTFALL``.  A separate
opt-in strict mode (env ``WFA171_L5_STRICT=1``) downgrades a shortfall to
**WARN** — never FAIL.  The nine L5 gates are not registered in any
pipeline today, so neither mode changes a live pre-flight number.

CPU-only.  No torch import, no GPU, no network.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "REPO_ROOT",
    "PLAN_DOC_REL",
    "STRICT_ENV",
    "Threshold",
    "CountingSource",
    "HookSpec",
    "HookRecord",
    "parse_plan_specs",
    "default_specs",
    "evaluate_hook",
    "evaluate_hooks",
    "evaluate_hook_by_id",
    "format_evidence",
    "strict_enabled",
    "load_json",
]

REPO_ROOT = Path(__file__).resolve().parents[2]

#: The spec.  Thresholds are parsed out of this file at runtime.
PLAN_DOC_REL = "TODO/03_plans/PRE_TRAINING_GATE_v2.md"

#: The §6 honest-negative catalog that carries the L5 doc-presence hooks.
CATALOG_REL = "docs/HONEST_NEGATIVE_CATALOG.md"

#: Canonical protocol-alignment matrix (the plan doc cites it as
#: "PROTOCOL_ALIGNMENT_MATRIX.md §1.4"; the canonical path is under
#: TODO/03_baselines/, and the file has NO §1.4 heading — see report.md).
PROTOCOL_MATRIX_REL = "TODO/03_baselines/PROTOCOL_ALIGNMENT_MATRIX.md"

#: Opt-in strict mode.  When set to "1" a threshold shortfall becomes a
#: WARN.  It NEVER becomes a FAIL.
STRICT_ENV = "WFA171_L5_STRICT"

# ---------------------------------------------------------------------------
# Plan-doc parsing
# ---------------------------------------------------------------------------

_HOOK_HEADER_RE = re.compile(r"^####\s+(L5-[A-Za-z0-9-]+):\s+`([A-Za-z0-9_]+)`")
_RE_SPEC_RE = re.compile(r"^-\s+\*\*Spec\*\*:\s*(.*)$")
_RE_SOURCE_RE = re.compile(r"^-\s+\*\*Source\*\*:\s*(.*)$")
_RE_VERDICT_RE = re.compile(r"^-\s+\*\*Verdict gate\*\*:\s*(.*)$")

# "5+ metals x 5 seeds x 5 folds (125+ records) present"  -> 125
_RE_RECORDS = re.compile(r"\((\d+)\+?\s+records\)")
# "at least 1 cell-line `delta > +0.03`"  -> (1, 0.03)
_RE_COUNT_ABOVE = re.compile(
    r"at least\s+(\d+)\s+cell-line[^`]*`delta\s*>\s*\+?([0-9]*\.?[0-9]+)`"
)
# "`delta_imold > -0.02`"  -> -0.02
_RE_DELTA_IMOLD = re.compile(r"`delta_imold\s*>\s*(-?[0-9]*\.?[0-9]+)`")
# "§6 paragraph exists for each Tier-1 LEAD metric x Krasnov"
_RE_DOC_PARAGRAPH = re.compile(r"§6 paragraph exists for each Tier-1 LEAD metric")
# "§6 entries for m1 + m21"  -> ("m1", "m21")
_RE_ENTRIES_FOR = re.compile(r"§6 entries for ([^`]+)")
_RE_METRIC_TOKEN = re.compile(r"\bm\d+\b")


@dataclass(frozen=True)
class Threshold:
    """One hook's spec threshold, parsed from the plan doc."""

    kind: str  # record_count | value_gt | count_above_value | doc_presence
    value: Any  # int (record_count / doc_presence) or float (value_gt)
    value_threshold: Optional[float] = None  # count_above_value: the per-item bar
    count_threshold: Optional[int] = None  # count_above_value: how many items
    required_metrics: Tuple[str, ...] = ()  # doc_presence: which metrics
    spec_line: int = 0  # line of the Verdict-gate line in the plan doc
    quote: str = ""  # the verbatim Verdict-gate line
    source_quote: str = ""  # the verbatim Source line
    source_line: int = 0

    def describe(self) -> str:
        if self.kind == "record_count":
            return f"{self.value} records"
        if self.kind == "value_gt":
            return f"value > {self.value}"
        if self.kind == "count_above_value":
            return f">= {self.count_threshold} item(s) with delta > {self.value_threshold}"
        if self.kind == "doc_presence":
            return f"§6 paragraph for each of {', '.join(self.required_metrics)}"
        return self.kind


@dataclass(frozen=True)
class CountingSource:
    """Where the records for one hook actually live, and under which key."""

    role: str  # primary | alternate
    path: str  # repo-relative
    key: Optional[str] = None  # dotted JSON key (None => whole document)
    note: str = ""


@dataclass(frozen=True)
class HookSpec:
    """A fully-specified hook: what the spec demands and where to look."""

    hook_id: str
    label: str
    threshold: Threshold
    source_path: str
    counting_key: Optional[str]
    alternates: Tuple[CountingSource, ...] = ()

    def as_tuple(self) -> Tuple[str, Any, str, Optional[str]]:
        """The (hook_id, spec_threshold, source_path, counting_key) 4-tuple."""
        return (self.hook_id, self.threshold, self.source_path, self.counting_key)


@dataclass
class HookRecord:
    """The measured outcome for one hook."""

    hook_id: str
    label: str
    threshold_kind: str
    threshold: Any
    value_threshold: Optional[float]
    count_threshold: Optional[int]
    required_metrics: Tuple[str, ...]
    spec_line: int
    verdict_quote: str
    source_quote: str
    # effective measurement (from whichever source supplied it)
    measured_source_path: Optional[str] = None
    measured_count: Optional[int] = None
    measured_value: Optional[float] = None
    meets_threshold: Optional[bool] = None
    shortfall_n: Optional[int] = None
    shortfall_pp: Optional[float] = None
    status: str = "UNMEASURED"  # MEETS | SHORTFALL | MISSING_SOURCE | MISSING_KEY
    # primary-source detail (what the plan doc / the gate actually points at)
    primary_source_path: Optional[str] = None
    primary_source_present: Optional[bool] = None
    primary_counting_key: Optional[str] = None
    primary_counting_key_present: Optional[bool] = None
    primary_status: Optional[str] = None
    primary_measured_count: Optional[int] = None
    primary_measured_value: Optional[float] = None
    alternates: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def source_present(self) -> Optional[bool]:
        """True when the source that produced the measurement exists."""
        if self.primary_source_present is None:
            return None
        return bool(self.primary_source_present)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "hook_id": self.hook_id,
            "label": self.label,
            "threshold_kind": self.threshold_kind,
            "threshold": self.threshold,
            "value_threshold": self.value_threshold,
            "count_threshold": self.count_threshold,
            "required_metrics": list(self.required_metrics),
            "spec_line": self.spec_line,
            "verdict_quote": self.verdict_quote,
            "source_quote": self.source_quote,
            "measured_source_path": self.measured_source_path,
            "measured_count": self.measured_count,
            "measured_value": self.measured_value,
            "meets_threshold": self.meets_threshold,
            "shortfall_n": self.shortfall_n,
            "shortfall_pp": self.shortfall_pp,
            "status": self.status,
            "primary_source_path": self.primary_source_path,
            "primary_source_present": self.primary_source_present,
            "primary_counting_key": self.primary_counting_key,
            "primary_counting_key_present": self.primary_counting_key_present,
            "primary_status": self.primary_status,
            "primary_measured_count": self.primary_measured_count,
            "primary_measured_value": self.primary_measured_value,
            "alternates": self.alternates,
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# THE CONSTANT BLOCK — counting sources + counting keys.
#
# The plan doc names the source FILES in prose but never names the JSON KEY
# that holds the records, and for two hooks it names a file that is NOT on
# disk.  These entries therefore live here, each one citing the plan-doc
# line it came from.  THIS IS A CONSTANT BLOCK, NOT A PARSE.  The
# THRESHOLDS are parsed from the plan doc at runtime (see
# :func:`parse_plan_specs`); only the key/path mapping is fixed here.
# ---------------------------------------------------------------------------
COUNTING_SOURCES: Dict[str, Tuple[CountingSource, ...]] = {
    # L5-Krasnov-1 (plan doc line 267-273).  Spec (270): "For each
    # (metal x seed x fold) tuple ... JSON record".  Verdict (272): 125+
    # records.  The source holds ONE scalar per (metal, seed) run.
    "krasnov_m1_per_seed_per_split_5fold_5metal": (
        CountingSource(
            "primary",
            "metrics/by_metric/metal_lomo_auc_wasserstein.json",
            "per_run_auc",
            "plan doc L271 names this file; n_folds=5 is recorded but the "
            "per-fold values are NOT stored per run (only per_run_auc "
            "scalars and per_fold_auc_mean over the 5 held-out metals).",
        ),
    ),
    # L5-Krasnov-2 (275-281).  Spec (278): "For each (cell x seed) tuple".
    # Verdict (280): 45+ records.  Source holds 9 cell lines, no per-seed.
    "krasnov_m21_per_cell_line_per_seed": (
        CountingSource(
            "primary",
            "metrics/by_metric/nci60_precious_9cell_auc.json",
            "per_cell_line",
            "plan doc L279 names this file; every cell record carries "
            "per_seed_available=false, so the per-seed records do not exist.",
        ),
    ),
    # L5-Krasnov-3 (283-289).  Verdict (288): "PASS if §6 paragraph exists
    # for each Tier-1 LEAD metric x Krasnov" -> m1, m21 (plan doc L287).
    "krasnov_protocol_alignment_diffs_documented": (
        CountingSource(
            "primary",
            CATALOG_REL,
            None,
            "plan doc L287: '§6 entries for m1 + m21'.",
        ),
        CountingSource(
            "alternate",
            PROTOCOL_MATRIX_REL,
            None,
            "plan doc L287 cites 'PROTOCOL_ALIGNMENT_MATRIX.md §1.4'; the "
            "canonical file exists but has NO §1.4 heading.",
        ),
    ),
    # L5-Rusanov-1 (293-299).  Spec (296): "For each (cell x cohort)".
    # Verdict (298): 12 records.  Source holds 3 cohorts, no cell axis.
    "rusanov_m2_per_cell_line_per_cohort": (
        CountingSource(
            "primary",
            "metrics/by_metric/m2_r2_per_cohort.json",
            "per_cohort",
            "the metric JSON the gate is wired to; holds 3 cohorts and no "
            "per-cell-line axis at all (plan doc L297 names "
            "molmetal/reports/wf388a1_m2_r2_real/result.json instead).",
        ),
        CountingSource(
            "alternate",
            "molmetal/reports/wf388a1_m2_r2_real/result.json",
            "per_cell_line",
            "plan doc L297 names this file; it is NOT on disk.",
        ),
    ),
    # L5-Rusanov-2 (301-307).  Verdict (306): "PASS if at least 1 cell-line
    # `delta > +0.03`".  Both named sources are absent from disk.
    "rusanov_m2_4cell_baseline_comparison": (
        CountingSource(
            "primary",
            "molmetal/reports/wf388a1_m2_r2_real/result.json",
            "per_cell_line",
            "plan doc L305 names this file; it is NOT on disk.",
        ),
        CountingSource(
            "alternate",
            "molmetal/reports/rusanov_2025_paper_table_s4_s5_baseline",
            None,
            "plan doc L305 names this directory; it is NOT on disk.",
        ),
    ),
    # L5-Rusanov-3 (309-315).  Verdict (314): "§6 paragraph ... x Rusanov"
    # -> m2 (plan doc L313).
    "rusanov_protocol_alignment_diffs_documented": (
        CountingSource(
            "primary",
            CATALOG_REL,
            None,
            "plan doc L313: '§6 entries for m2'.",
        ),
        CountingSource(
            "alternate",
            PROTOCOL_MATRIX_REL,
            None,
            "plan doc L313 cites 'PROTOCOL_ALIGNMENT_MATRIX.md §1.4'; the "
            "canonical file exists but has NO §1.4 heading.",
        ),
    ),
    # L5-DrugOOD-1/2/3 (319-341).  Verdict (324/332/340): "PASS if
    # `delta_imold > -0.02`".  Plan doc L323/331/339 name
    # molmetal/reports/wf265_collect/all_results.json, which carries NO
    # assay/scaffold/size split keys; the deltas live in the m17 metric
    # JSON that the L3 gate already reads.
    "drugood_m17_assay_split_auc": (
        CountingSource(
            "primary",
            "molmetal/reports/wf265_collect/all_results.json",
            "per_axis_auc_delta_vs_imold.assay",
            "plan doc L323 names this file; it has no split/imold keys.",
        ),
        CountingSource(
            "alternate",
            "metrics/by_metric/drugood_ood_auc.json",
            "per_axis_auc_delta_vs_imold.assay",
            "the m17 metric JSON (also read by the L3 gate "
            "p3_mock_to_real_m17_drugood_assay_REAL_3_seeds).",
        ),
    ),
    "drugood_m17_scaffold_split_auc": (
        CountingSource(
            "primary",
            "molmetal/reports/wf265_collect/all_results.json",
            "per_axis_auc_delta_vs_imold.scaffold",
            "plan doc L331 names this file; it has no split/imold keys.",
        ),
        CountingSource(
            "alternate",
            "metrics/by_metric/drugood_ood_auc.json",
            "per_axis_auc_delta_vs_imold.scaffold",
            "the m17 metric JSON (also read by the L3 gate).",
        ),
    ),
    "drugood_m17_size_split_auc": (
        CountingSource(
            "primary",
            "molmetal/reports/wf265_collect/all_results.json",
            "per_axis_auc_delta_vs_imold.size",
            "plan doc L339 names this file; it has no split/imold keys.",
        ),
        CountingSource(
            "alternate",
            "metrics/by_metric/drugood_ood_auc.json",
            "per_axis_auc_delta_vs_imold.size",
            "the m17 metric JSON (also read by the L3 gate).",
        ),
    ),
}


# ---------------------------------------------------------------------------
# Plan-doc parsing (thresholds come from here, from disk, at runtime)
# ---------------------------------------------------------------------------


def parse_plan_specs(plan_path: Optional[Path] = None) -> Dict[str, Threshold]:
    """Parse the nine L5 thresholds out of the plan doc.

    Returns a mapping ``hook_id -> Threshold``.  Raises ``FileNotFoundError``
    if the plan doc is missing and ``ValueError`` if a hook block has no
    parseable threshold — a silent default would be exactly the kind of
    fabricated number this repo forbids.
    """
    path = Path(plan_path) if plan_path else REPO_ROOT / PLAN_DOC_REL
    if not path.exists():
        raise FileNotFoundError(f"plan doc not found: {path}")
    lines = path.read_text(encoding="utf-8").split("\n")

    out: Dict[str, Threshold] = {}
    i = 0
    while i < len(lines):
        m = _HOOK_HEADER_RE.match(lines[i])
        if not m:
            i += 1
            continue
        label, hook_id = m.group(1), m.group(2)
        # collect the block until the next "####" or "## " heading
        j = i + 1
        spec_quote = source_quote = verdict_quote = ""
        spec_line = source_line = verdict_line = 0
        while j < len(lines):
            ln = lines[j]
            if ln.startswith("####") or ln.startswith("## "):
                break
            ms = _RE_SPEC_RE.match(ln)
            if ms:
                spec_quote, spec_line = ms.group(1), j + 1
            msrc = _RE_SOURCE_RE.match(ln)
            if msrc:
                source_quote, source_line = msrc.group(1), j + 1
            mv = _RE_VERDICT_RE.match(ln)
            if mv:
                verdict_quote, verdict_line = mv.group(1), j + 1
            j += 1

        thr = _threshold_from_verdict(
            verdict_quote,
            source_quote,
            verdict_line,
            source_line,
        )
        if not label:  # pragma: no cover - the header regex always yields one
            raise ValueError(f"hook {hook_id!r} has no label")
        out[hook_id] = thr
        i = j

    if not out:
        raise ValueError(f"no L5 hook blocks parsed from {path}")
    return out


def _threshold_from_verdict(
    verdict_quote: str,
    source_quote: str,
    verdict_line: int,
    source_line: int,
) -> Threshold:
    """Turn one Verdict-gate line into a :class:`Threshold`."""
    m = _RE_RECORDS.search(verdict_quote)
    if m:
        return Threshold(
            kind="record_count",
            value=int(m.group(1)),
            spec_line=verdict_line,
            quote=verdict_quote,
            source_quote=source_quote,
            source_line=source_line,
        )
    m = _RE_COUNT_ABOVE.search(verdict_quote)
    if m:
        return Threshold(
            kind="count_above_value",
            value=int(m.group(1)),
            value_threshold=float(m.group(2)),
            count_threshold=int(m.group(1)),
            spec_line=verdict_line,
            quote=verdict_quote,
            source_quote=source_quote,
            source_line=source_line,
        )
    m = _RE_DELTA_IMOLD.search(verdict_quote)
    if m:
        return Threshold(
            kind="value_gt",
            value=float(m.group(1)),
            spec_line=verdict_line,
            quote=verdict_quote,
            source_quote=source_quote,
            source_line=source_line,
        )
    if _RE_DOC_PARAGRAPH.search(verdict_quote):
        metrics: Tuple[str, ...] = ()
        mentries = _RE_ENTRIES_FOR.search(source_quote)
        if mentries:
            metrics = tuple(_RE_METRIC_TOKEN.findall(mentries.group(1)))
        if not metrics:
            raise ValueError(
                "doc-presence hook at line "
                f"{verdict_line} names no metric in its Source line: "
                f"{source_quote!r}"
            )
        return Threshold(
            kind="doc_presence",
            value=len(metrics),
            required_metrics=metrics,
            spec_line=verdict_line,
            quote=verdict_quote,
            source_quote=source_quote,
            source_line=source_line,
        )
    raise ValueError(
        "unparseable Verdict-gate line at line "
        f"{verdict_line}: {verdict_quote!r}"
    )


# ---------------------------------------------------------------------------
# The nine specs
# ---------------------------------------------------------------------------


def default_specs(
    plan_path: Optional[Path] = None,
    repo_root: Optional[Path] = None,
) -> List[HookSpec]:
    """The nine L5 hook specs: thresholds from disk, keys from the block."""
    root = Path(repo_root) if repo_root else REPO_ROOT
    thresholds = parse_plan_specs(plan_path or (root / PLAN_DOC_REL))
    labels = _hook_labels(plan_path or (root / PLAN_DOC_REL))
    specs: List[HookSpec] = []
    for hook_id, thr in thresholds.items():
        sources = COUNTING_SOURCES.get(hook_id)
        if not sources:
            raise KeyError(f"no counting source for hook {hook_id!r}")
        primary = sources[0]
        specs.append(
            HookSpec(
                hook_id=hook_id,
                label=labels.get(hook_id, ""),
                threshold=thr,
                source_path=primary.path,
                counting_key=primary.key,
                alternates=tuple(sources[1:]),
            )
        )
    # stable order: as they appear in the plan doc (dict preserves insertion)
    return specs


def _hook_labels(plan_path: Path) -> Dict[str, str]:
    """hook_id -> the 'L5-Krasnov-1' label used by the plan doc."""
    lines = Path(plan_path).read_text(encoding="utf-8").split("\n")
    out: Dict[str, str] = {}
    for ln in lines:
        m = _HOOK_HEADER_RE.match(ln)
        if m:
            out[m.group(2)] = m.group(1)
    return out


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _resolve_key(doc: Any, key: Optional[str]) -> Tuple[bool, Any]:
    """Resolve a dotted key.  Returns (found, value)."""
    if key is None:
        return True, doc
    cur = doc
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.isdigit() and int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return False, None
    return True, cur


#: A §6 entry qualifies for a metric when it mentions the metric AND a
#: split/protocol-difference token.  Documented here so the count is
#: reproducible rather than a matter of opinion.
PROTOCOL_SPLIT_TOKEN_RE = re.compile(
    r"(5-fold|scaffold[- ]split|per-cell-line|split[- ]protocol|protocol replication|cohort-split)"
)
_METRIC_RE = re.compile(r"(?<![a-z0-9])(m\d+)(?![0-9a-z])")
_CATALOG_ENTRY_RE = re.compile(r"^###\s+(§6\.[^\s]*)", re.MULTILINE)


def _catalog_entries(catalog_path: Path) -> List[Tuple[str, str]]:
    """Split the §6 catalog into (entry_id, body) pairs."""
    text = catalog_path.read_text(encoding="utf-8")
    marks = list(_CATALOG_ENTRY_RE.finditer(text))
    out: List[Tuple[str, str]] = []
    for idx, m in enumerate(marks):
        end = marks[idx + 1].start() if idx + 1 < len(marks) else len(text)
        out.append((m.group(1), text[m.start() : end]))
    return out


def _doc_presence_measure(
    catalog_path: Path, required_metrics: Sequence[str]
) -> Tuple[Optional[int], List[str], Dict[str, List[str]]]:
    """Count how many of ``required_metrics`` have a qualifying §6 entry.

    Entry boundary: a ``### §6.<id>`` heading (the catalog's dominant
    per-entry level, 138 of them).  A section qualifies for metric ``mX``
    when it mentions ``mX`` (token-boundary regex) AND mentions a
    split/protocol-difference token.  MEASURED 2026-11-01: the covered
    metric set is IDENTICAL when every ``##``/``###``/``####`` heading is
    used as a boundary instead, so the verdict does not hinge on this
    choice.
    """
    if not catalog_path.exists():
        return None, [], {}
    entries = _catalog_entries(catalog_path)
    per_metric: Dict[str, List[str]] = {m: [] for m in required_metrics}
    for entry_id, body in entries:
        if not PROTOCOL_SPLIT_TOKEN_RE.search(body):
            continue
        mentioned = set(_METRIC_RE.findall(body))
        for metric in required_metrics:
            if metric in mentioned and entry_id not in per_metric[metric]:
                per_metric[metric].append(entry_id)
    covered = sum(1 for m in required_metrics if per_metric.get(m))
    return covered, sorted({e for v in per_metric.values() for e in v}), per_metric


def _measure_source(
    source: CountingSource, thr: Threshold, root: Path
) -> Dict[str, Any]:
    """Measure one candidate source against one threshold."""
    path = root / source.path
    out: Dict[str, Any] = {
        "role": source.role,
        "path": source.path,
        "key": source.key,
        "source_present": path.exists(),
        "key_present": None,
        "measured_count": None,
        "measured_value": None,
        "meets_threshold": None,
        "shortfall_n": None,
        "shortfall_pp": None,
        "status": "MISSING_SOURCE",
        "note": source.note,
    }
    if not path.exists():
        return out

    if thr.kind == "doc_presence":
        covered, entries, per_metric = _doc_presence_measure(
            path, thr.required_metrics
        )
        out["key_present"] = covered is not None
        out["measured_count"] = covered
        out["measured_entries"] = entries
        out["per_metric_entries"] = per_metric
        if covered is None:
            out["status"] = "MISSING_SOURCE"
            return out
        if covered >= int(thr.value):
            out["status"] = "MEETS"
            out["meets_threshold"] = True
        else:
            out["status"] = "SHORTFALL"
            out["meets_threshold"] = False
            out["shortfall_n"] = int(thr.value) - covered
        return out

    if path.suffix.lower() != ".json":
        # A directory (e.g. the rusanov baseline table dir): presence only.
        out["key_present"] = True
        out["status"] = "MISSING_KEY"
        out["note"] = (out["note"] + " [not a JSON document]").strip()
        return out

    try:
        doc = load_json(path)
    except Exception as exc:  # noqa: BLE001 - a broken source is a MISSING source
        out["status"] = "MISSING_KEY"
        out["note"] = (out["note"] + f" [unreadable JSON: {exc}]").strip()
        return out

    found, val = _resolve_key(doc, source.key)
    out["key_present"] = found
    if not found:
        out["status"] = "MISSING_KEY"
        return out

    if thr.kind == "record_count":
        if isinstance(val, (list, dict)):
            count = len(val)
        else:
            count = 1
        out["measured_count"] = count
        if count >= int(thr.value):
            out["status"] = "MEETS"
            out["meets_threshold"] = True
        else:
            out["status"] = "SHORTFALL"
            out["meets_threshold"] = False
            out["shortfall_n"] = int(thr.value) - count
        return out

    if thr.kind == "value_gt":
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            out["status"] = "MISSING_KEY"
            out["note"] = (out["note"] + " [value is not numeric]").strip()
            return out
        out["measured_value"] = float(val)
        out["measured_count"] = 1
        if float(val) > float(thr.value):
            out["status"] = "MEETS"
            out["meets_threshold"] = True
        else:
            out["status"] = "SHORTFALL"
            out["meets_threshold"] = False
            out["shortfall_pp"] = (float(thr.value) - float(val)) * 100.0
        return out

    if thr.kind == "count_above_value":
        items = val if isinstance(val, list) else [val]
        bar = float(thr.value_threshold)
        n_above = 0
        for item in items:
            v = None
            if isinstance(item, dict):
                for cand in ("delta", "delta_imold", "delta_value"):
                    if cand in item and isinstance(item[cand], (int, float)):
                        v = float(item[cand])
                        break
            elif isinstance(item, (int, float)) and not isinstance(item, bool):
                v = float(item)
            if v is not None and v > bar:
                n_above += 1
        out["measured_count"] = n_above
        need = int(thr.count_threshold or thr.value)
        if n_above >= need:
            out["status"] = "MEETS"
            out["meets_threshold"] = True
        else:
            out["status"] = "SHORTFALL"
            out["meets_threshold"] = False
            out["shortfall_n"] = need - n_above
        return out

    out["status"] = "MISSING_KEY"
    return out


def evaluate_hook(
    spec: Any, repo_root: Optional[Path] = None
) -> HookRecord:
    """Evaluate one hook.

    ``spec`` may be a :class:`HookSpec` or the 4-tuple
    ``(hook_id, spec_threshold, source_path, counting_key)`` required by the
    Wf-A171 D3 brief.  A bare number as ``spec_threshold`` is treated as a
    record-count threshold.
    """
    root = Path(repo_root) if repo_root else REPO_ROOT
    if isinstance(spec, HookSpec):
        hook_id = spec.hook_id
        thr = spec.threshold
        source_path = spec.source_path
        counting_key = spec.counting_key
        alternates = spec.alternates
        label = spec.label
    else:
        hook_id, thr_raw, source_path, counting_key = spec
        alternates = ()
        label = ""
        if isinstance(thr_raw, Threshold):
            thr = thr_raw
        elif isinstance(thr_raw, str):
            thr = Threshold(kind="record_count", value=int(thr_raw))
        elif isinstance(thr_raw, float):
            thr = Threshold(kind="value_gt", value=thr_raw)
        else:
            thr = Threshold(kind="record_count", value=int(thr_raw))

    primary = CountingSource("primary", source_path, counting_key)
    primary_m = _measure_source(primary, thr, root)

    alt_measurements: List[Dict[str, Any]] = []
    for alt in alternates:
        alt_measurements.append(_measure_source(alt, thr, root))

    # Effective outcome: the primary source wins when it could measure;
    # otherwise the first alternate that could measure.
    effective = primary_m
    if primary_m["status"] in ("MISSING_SOURCE", "MISSING_KEY"):
        for alt in alt_measurements:
            if alt["status"] in ("MEETS", "SHORTFALL"):
                effective = alt
                break

    rec = HookRecord(
        hook_id=hook_id,
        label=label,
        threshold_kind=thr.kind,
        threshold=thr.value,
        value_threshold=thr.value_threshold,
        count_threshold=thr.count_threshold,
        required_metrics=tuple(thr.required_metrics),
        spec_line=thr.spec_line,
        verdict_quote=thr.quote,
        source_quote=thr.source_quote,
        measured_source_path=effective["path"],
        measured_count=effective["measured_count"],
        measured_value=effective["measured_value"],
        meets_threshold=effective["meets_threshold"],
        shortfall_n=effective["shortfall_n"],
        shortfall_pp=effective["shortfall_pp"],
        status=effective["status"],
        primary_source_path=primary.path,
        primary_source_present=primary_m["source_present"],
        primary_counting_key=primary.key,
        primary_counting_key_present=primary_m["key_present"],
        primary_status=primary_m["status"],
        primary_measured_count=primary_m["measured_count"],
        primary_measured_value=primary_m["measured_value"],
        alternates=alt_measurements,
    )
    if primary_m["status"] != effective["status"]:
        rec.notes.append(
            f"primary source {primary.path} -> {primary_m['status']}; "
            f"measurement taken from {effective['path']} -> {effective['status']}"
        )
    if thr.kind == "doc_presence":
        per_metric = effective.get("per_metric_entries") or {}
        rec.notes.append(
            "qualifying §6 entries: "
            + "; ".join(f"{k}={v}" for k, v in per_metric.items())
        )
    return rec


def evaluate_hooks(
    specs: Optional[Sequence[Any]] = None,
    repo_root: Optional[Path] = None,
) -> List[HookRecord]:
    """Evaluate all nine L5 hooks.  ``specs`` defaults to the nine on-disk specs."""
    if specs is None:
        specs = default_specs(repo_root=repo_root)
    return [evaluate_hook(s, repo_root=repo_root) for s in specs]


def evaluate_hook_by_id(
    hook_id: str, repo_root: Optional[Path] = None
) -> Optional[HookRecord]:
    """Evaluate one hook by id.  Returns ``None`` for an unknown id."""
    for spec in default_specs(repo_root=repo_root):
        if spec.hook_id == hook_id:
            return evaluate_hook(spec, repo_root=repo_root)
    return None


# ---------------------------------------------------------------------------
# Evidence string + strict mode
# ---------------------------------------------------------------------------


def format_evidence(record: HookRecord) -> str:
    """One-line evidence fragment: threshold + measurement + explicit marker."""
    if record.status == "MISSING_SOURCE":
        return (
            f"spec_threshold={record.threshold_kind}:"
            f"{record.threshold} MEASURED=MISSING_SOURCE"
            f"({record.primary_source_path})"
        )
    if record.status == "MISSING_KEY":
        return (
            f"spec_threshold={record.threshold_kind}:"
            f"{record.threshold} MEASURED=MISSING_KEY"
            f"({record.primary_counting_key} in {record.primary_source_path})"
        )
    if record.status == "MEETS":
        measured = (
            f"measured_count={record.measured_count}"
            if record.measured_count is not None
            else f"measured_value={record.measured_value}"
        )
        return f"spec_threshold={record.threshold_kind}:{record.threshold} {measured} MEETS"
    # SHORTFALL
    parts = []
    if record.measured_count is not None:
        parts.append(f"measured_count={record.measured_count}")
    if record.measured_value is not None:
        parts.append(f"measured_value={record.measured_value:+.6f}")
    if record.shortfall_n is not None:
        parts.append(f"shortfall_n={record.shortfall_n}")
    if record.shortfall_pp is not None:
        parts.append(f"shortfall_pp={record.shortfall_pp:.4f}")
    return (
        f"spec_threshold={record.threshold_kind}:{record.threshold} "
        + " ".join(parts)
        + " SHORTFALL"
    )


def strict_enabled(env: Optional[Dict[str, str]] = None) -> bool:
    """True when the opt-in strict mode is on.  It can only ever WARN."""
    environ = os.environ if env is None else env
    return str(environ.get(STRICT_ENV, "0")).strip() == "1"
