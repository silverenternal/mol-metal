#!/usr/bin/env python
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

"""Wf-A205 -- emit the evidence record for the citability gate.

Everything the report quotes comes out of THIS script, so every number in the
report has a record file behind it (AGENTS.md 1.1 / 5.2).

Writes (all under this directory):
    row_census.json            every registry row, with the four columns the
                               gate reads, plus the payload-side caveat flag
    payload_direction_proof.json  the code citations + the live reproduction
                               that a regenerate would NOT erase the caveat
    summary.json               the counts

CPU-only.  No GPU.  Writes NOTHING under registry/ or metrics/.

Usage:
    CUDA_VISIBLE_DEVICES="" .venv/bin/python \
        molmetal/reports/wfa205_annotation_reproducibility/gate/evidence.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib
import sys
from typing import Any, Dict, List, Optional

REPO = pathlib.Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

GATE_MODULE_PATH = REPO / "molmetal" / "tests" / "test_wfa205_citability_gate.py"
REGISTRY = REPO / "registry" / "metrics.yaml"
REGISTRY_PY = REPO / "molmetal" / "scripts" / "registry.py"
OUT_DIR = pathlib.Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# CITATION-LINE RESOLUTION (added 2026-11-02 by the read-only repair wave)
#
# WHY THIS EXISTS.  The 2026-11-02 rendering of
# ``payload_direction_proof.json`` cited 3 of its 8 code citations at the
# WRONG line of ``molmetal/scripts/registry.py`` -- 1361 -> 1362,
# 2061 -> 2056, 2077 -> 2072.  The numbers were hand-memorised, so nothing
# noticed they rotted.  The repo's locator auditor (Wf-A200,
# ``molmetal/scripts/wfa200_crosswave_locator_audit.py``) CANNOT see them:
# its ``LOCATOR_RE`` requires a path-then-colon-then-digits ADJACENCY, and
# a machine-readable ``{"file": ..., "line": ...}`` record has no adjacency
# (and ``line`` is classified as metadata) -- MEASURED by the main thread:
# 0 of the 3 mis-cited lines flagged, and its single ``live_drift`` on this
# tree was a FALSE POSITIVE (docs/HONEST_NEGATIVE_CATALOG.md 6.269.4e).
#
# WHAT CHANGED.  Every citation is now RESOLVED FROM THE FILE BY CONTENT:
# the anchor block below must appear, verbatim, at the cited line.  Drift
# that no dated correction covers is a HARD ERROR -- the generator refuses
# to write.  The 2026-11-02 rendering stays VERBATIM in the artifact's
# ``code_citations`` array (AGENTS.md 5.5 rule #6); the correction is
# additive (``corrections``).
# ---------------------------------------------------------------------------

#: ``scope`` is the enclosing top-level ``def``/``class`` the anchor must sit
#: in, or the literal ``"module"`` for module level.  ``block`` is the list
#: of STRIPPED source lines that must appear, in order, at the cited line.
CITATION_ANCHORS: List[Dict[str, Any]] = [
    {
        "citation_index": 0,
        "scope": "build_metrics",
        "block": ["out: Dict[str, Any] = {}"],
    },
    {
        "citation_index": 1,
        "scope": "assess_citability",
        "block": ['caveat = payload.get("headline_caveat")'],
    },
    {
        "citation_index": 2,
        "scope": "assess_citability",
        "block": [
            '"headline_caveat": (',
            "caveat.strip() if isinstance(caveat, str) and caveat.strip() else None",
            "),",
        ],
    },
    {
        "citation_index": 3,
        "scope": "build_metrics",
        "block": [
            "rec.update(",
            'assess_citability(payload, rec["value"], _rel(src_path), tracked)',
            ")",
        ],
    },
    {
        "citation_index": 4,
        "scope": "generate",
        "block": ["metrics_doc = {", '"schema": "molmetal-registry-metrics-v1",'],
    },
    {
        "citation_index": 5,
        "scope": "generate",
        "block": ["text = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)"],
    },
    {
        "citation_index": 6,
        "scope": "main",
        "block": ['if r.get("registry_role") != "paper_line":', "continue"],
    },
    {
        "citation_index": 7,
        "scope": "module",
        "block": ["LINE_METRICS: Dict[str, Dict[str, str]] = {"],
    },
]

#: The dated, additive correction for the 3 rotted locators.  The
#: ``code_citations`` array is NOT rewritten: the superseded rendering is
#: preserved verbatim (AGENTS.md 5.5 rule #6) and the move is recorded here.
CITATION_CORRECTIONS: List[Dict[str, Any]] = [
    {
        "dated": "2026-11-02",
        "wave": "Wf-A205 repair wave (read-only-fixable repairs)",
        "kind": "code_citation_line_number_correction",
        "where": [
            "molmetal/reports/wfa205_annotation_reproducibility/gate/payload_direction_proof.json -> code_citations[0]",
            "molmetal/reports/wfa205_annotation_reproducibility/gate/README.md -> section 3, proof table",
            "molmetal/tests/test_wfa205_citability_gate.py -> module docstring + test_bite_5 docstring",
        ],
        "superseded_rendering_verbatim": "molmetal/scripts/registry.py:1361",
        "corrected_to": "molmetal/scripts/registry.py:1362",
        "why": (
            "MEASURED 2026-11-02 against the working-tree "
            "molmetal/scripts/registry.py (sha256 "
            "18fae08ab25dfb866dd857dc1a92654e4647d63b9b75f37a36f33004a568a438, "
            "2346 lines): line 1361 reads '    \"\"\"' -- the CLOSING line of "
            "build_metrics' docstring -- and the statement it was cited for, "
            "'    out: Dict[str, Any] = {}', is at line 1362. The cited "
            "STATEMENT (build_metrics starts from an empty dict) is unchanged; "
            "only the locator moved."
        ),
        "entries": [
            {
                "citation_index": 0,
                "file": "molmetal/scripts/registry.py",
                "superseded_line": 1361,
                "corrected_line": 1362,
                "superseded_citation": {
                    "file": "molmetal/scripts/registry.py",
                    "line": 1361,
                    "text": "out: Dict[str, Any] = {}  # build_metrics starts from an EMPTY dict",
                    "meaning": (
                        "the metric layer is rebuilt from the payloads; there is no "
                        "preservation pass for metrics (the preservation helper "
                        "_preserve_underived_datasets is DATASETS-only, registry.py:2022-2024)"
                    ),
                },
                "corrected_citation": {
                    "file": "molmetal/scripts/registry.py",
                    "line": 1362,
                    "text": "out: Dict[str, Any] = {}  # build_metrics starts from an EMPTY dict",
                    "meaning": (
                        "the metric layer is rebuilt from the payloads; there is no "
                        "preservation pass for metrics (the preservation helper "
                        "_preserve_underived_datasets is DATASETS-only, registry.py:2022-2024)"
                    ),
                },
            }
        ],
    },
    {
        "dated": "2026-11-02",
        "wave": "Wf-A205 repair wave (read-only-fixable repairs)",
        "kind": "code_citation_line_number_correction",
        "where": [
            "molmetal/reports/wfa205_annotation_reproducibility/gate/payload_direction_proof.json -> code_citations[4]",
            "molmetal/reports/wfa205_annotation_reproducibility/gate/README.md -> section 3, proof table",
            "molmetal/tests/test_wfa205_citability_gate.py -> module docstring",
        ],
        "superseded_rendering_verbatim": "molmetal/scripts/registry.py:2061",
        "corrected_to": "molmetal/scripts/registry.py:2056",
        "why": (
            "MEASURED 2026-11-02 against the working-tree "
            "molmetal/scripts/registry.py (sha256 "
            "18fae08ab25dfb866dd857dc1a92654e4647d63b9b75f37a36f33004a568a438, "
            "2346 lines): line 2061 reads '        \"lines\": list(LINE_METRICS.keys()),' "
            "-- a MEMBER of the metrics_doc literal, not the literal itself -- and "
            "'    metrics_doc = {' is at line 2056. The cited STATEMENT (generate "
            "dumps those rows into the metrics document) is unchanged; only the "
            "locator moved."
        ),
        "entries": [
            {
                "citation_index": 4,
                "file": "molmetal/scripts/registry.py",
                "superseded_line": 2061,
                "corrected_line": 2056,
                "superseded_citation": {
                    "file": "molmetal/scripts/registry.py",
                    "line": 2061,
                    "text": "metrics_doc = { ... \"metrics\": metrics, ... }",
                    "meaning": "generate dumps those rows into the metrics document",
                },
                "corrected_citation": {
                    "file": "molmetal/scripts/registry.py",
                    "line": 2056,
                    "text": "metrics_doc = { ... \"metrics\": metrics, ... }",
                    "meaning": "generate dumps those rows into the metrics document",
                },
            }
        ],
    },
    {
        "dated": "2026-11-02",
        "wave": "Wf-A205 repair wave (read-only-fixable repairs)",
        "kind": "code_citation_line_number_correction",
        "where": [
            "molmetal/reports/wfa205_annotation_reproducibility/gate/payload_direction_proof.json -> code_citations[5]",
            "molmetal/reports/wfa205_annotation_reproducibility/gate/README.md -> section 3, proof table",
            "molmetal/tests/test_wfa205_citability_gate.py -> module docstring + test_bite_5 docstring",
        ],
        "superseded_rendering_verbatim": "molmetal/scripts/registry.py:2077",
        "corrected_to": "molmetal/scripts/registry.py:2072",
        "why": (
            "MEASURED 2026-11-02 against the working-tree "
            "molmetal/scripts/registry.py (sha256 "
            "18fae08ab25dfb866dd857dc1a92654e4647d63b9b75f37a36f33004a568a438, "
            "2346 lines): line 2077 reads '        if path.is_file() and "
            "_strip_generated_at(path.read_text(encoding=\"utf-8\")) == "
            "_strip_generated_at(text):' -- the no-op write guard -- and "
            "'        text = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)' "
            "is at line 2072. The companion 'atomic_write_file(path, text)' "
            "citation at 2079 was ALREADY correct. The cited STATEMENT is "
            "unchanged; only the locator moved."
        ),
        "entries": [
            {
                "citation_index": 5,
                "file": "molmetal/scripts/registry.py",
                "superseded_line": 2077,
                "corrected_line": 2072,
                "superseded_citation": {
                    "file": "molmetal/scripts/registry.py",
                    "line": 2077,
                    "text": "text = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)",
                    "meaning": "and writes it over registry/metrics.yaml with atomic_write_file (2079)",
                },
                "corrected_citation": {
                    "file": "molmetal/scripts/registry.py",
                    "line": 2072,
                    "text": "text = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)",
                    "meaning": "and writes it over registry/metrics.yaml with atomic_write_file (2079)",
                },
            }
        ],
    },
]


def _registry_py_lines() -> List[str]:
    return REGISTRY_PY.read_text(encoding="utf-8").splitlines()


def _enclosing_scope(lines: List[str], idx: int) -> Optional[str]:
    """The nearest enclosing top-level ``def``/``class`` name, or None."""
    for i in range(idx, -1, -1):
        line = lines[i]
        if line.startswith("def ") or line.startswith("class "):
            return line.split("(")[0].split()[-1]
    return None


def _resolve_anchor(
    lines: List[str], block: List[str], scope: Optional[str]
) -> Optional[int]:
    """The 1-based line where ``block`` starts, or None when not unique."""
    n = len(block)
    hits: List[int] = []
    for i in range(len(lines) - n + 1):
        if [lines[i + k].strip() for k in range(n)] != block:
            continue
        found = _enclosing_scope(lines, i)
        if scope == "module":
            if found is not None:
                continue
        elif found != scope:
            continue
        hits.append(i + 1)
    return hits[0] if len(hits) == 1 else None


def _citation_line_check(citations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Resolve every citation against the file on disk; report the drift."""
    lines = _registry_py_lines()
    corrections: Dict[int, Dict[str, Any]] = {}
    for corr in CITATION_CORRECTIONS:
        for entry in corr.get("entries", []):
            entry = dict(entry)
            entry["dated"] = corr.get("dated")
            corrections[entry["citation_index"]] = entry

    checked: List[Dict[str, Any]] = []
    uncovered: List[int] = []
    for anchor in CITATION_ANCHORS:
        idx = anchor["citation_index"]
        cited = citations[idx]["line"]
        resolved = _resolve_anchor(lines, anchor["block"], anchor["scope"])
        entry = corrections.get(idx)
        if resolved is None:
            status = "ANCHOR-NOT-FOUND"
        elif resolved == cited:
            status = "OK"
        elif entry is not None and entry["corrected_line"] == resolved:
            status = "SUPERSEDED-BY-CORRECTION"
        else:
            status = "DRIFT-UNCOVERED"
        if status in ("ANCHOR-NOT-FOUND", "DRIFT-UNCOVERED"):
            uncovered.append(idx)
        checked.append(
            {
                "citation_index": idx,
                "cited_line": cited,
                "resolved_line": resolved,
                "resolved_line_text": (
                    lines[resolved - 1] if resolved else None
                ),
                "status": status,
                "correction_dated": entry.get("dated") if entry else None,
            }
        )
    return {
        "registry_py_sha256": _sha256(REGISTRY_PY),
        "method": (
            "every citation's anchor block is searched for in the file on disk; "
            "the cited line must be the line where the block starts"
        ),
        "citations": checked,
        "uncorrected_drift": uncovered,
    }


def _load_gate_module():
    spec = importlib.util.spec_from_file_location("wfa205_gate", GATE_MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


GATE = _load_gate_module()


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    doc = GATE._load_registry(REGISTRY)
    rows = GATE._rows(doc)

    census: List[Dict[str, Any]] = []
    for metric_id, row in sorted(rows.items()):
        payload, why = GATE._load_payload(row.get(GATE.KEY_SOURCE_JSON))
        registry_caveat = GATE._caveat_state(row.get(GATE.KEY_CAVEAT))
        payload_caveat = (
            GATE._caveat_state(payload.get(GATE.KEY_CAVEAT)) if payload is not None else None
        )
        census.append(
            {
                "metric_id": metric_id,
                "registry_role": row.get(GATE.KEY_ROLE),
                "quotable": bool(row.get(GATE.KEY_QUOTABLE)),
                "headline_grade": bool(row.get(GATE.KEY_HEADLINE_GRADE)),
                "status": row.get("status"),
                "value": row.get("value"),
                "source_json": row.get(GATE.KEY_SOURCE_JSON),
                "registry_caveat_present": registry_caveat is not None,
                "payload_caveat_present": payload_caveat is not None,
                "payload_readable": payload is not None,
                "payload_read_error": None if payload is not None else why,
                "caveats_agree": registry_caveat == payload_caveat,
                "governed_by_gate": GATE._is_governed(row),
            }
        )

    governed = [r for r in census if r["governed_by_gate"]]
    headline = [r for r in census if r["quotable"] and r["headline_grade"]]
    soft = [r for r in census if r["quotable"] and not r["headline_grade"]]
    soft_paper_caveatless = [
        r for r in soft
        if r["registry_role"] == GATE.ROLE_PAPER_LINE and not r["registry_caveat_present"]
    ]
    out_of_scope_caveatless = [
        r for r in headline if not r["governed_by_gate"] and not r["registry_caveat_present"]
    ]

    # ---- the payload-direction proof -------------------------------------
    from molmetal.scripts.registry import LINE_METRICS, assess_citability

    proof_rows = []
    for r in governed:
        row = rows[r["metric_id"]]
        payload, why = GATE._load_payload(row[GATE.KEY_SOURCE_JSON])
        assert payload is not None, why
        rebuilt = assess_citability(payload, row.get("value"), row[GATE.KEY_SOURCE_JSON], None)
        payload_path = REPO / row[GATE.KEY_SOURCE_JSON]
        proof_rows.append(
            {
                "metric_id": r["metric_id"],
                "source_json": row[GATE.KEY_SOURCE_JSON],
                "source_json_sha256": _sha256(payload_path),
                "registry_caveat_sha256": hashlib.sha256(
                    (row.get(GATE.KEY_CAVEAT) or "").encode("utf-8")
                ).hexdigest(),
                "payload_caveat_sha256": hashlib.sha256(
                    (payload.get(GATE.KEY_CAVEAT) or "").encode("utf-8")
                ).hexdigest(),
                "registry_caveat_present": r["registry_caveat_present"],
                "payload_caveat_present": r["payload_caveat_present"],
                "assess_citability_rebuilds_the_same_caveat": (
                    rebuilt[GATE.KEY_CAVEAT] == row.get(GATE.KEY_CAVEAT)
                ),
                "assess_citability_quotable": rebuilt[GATE.KEY_QUOTABLE],
                "assess_citability_headline_grade": rebuilt[GATE.KEY_HEADLINE_GRADE],
            }
        )

    proof = {
        "claim": (
            "a headline_caveat that exists ONLY in the registry row is ERASED by "
            "the next `registry.py generate`; the payload is the durable direction"
        ),
        "code_citations": [
            {
                "file": "molmetal/scripts/registry.py",
                "line": 1361,
                "text": "out: Dict[str, Any] = {}  # build_metrics starts from an EMPTY dict",
                "meaning": "the metric layer is rebuilt from the payloads; there is no "
                           "preservation pass for metrics (the preservation helper "
                           "_preserve_underived_datasets is DATASETS-only, registry.py:2022-2024)",
            },
            {
                "file": "molmetal/scripts/registry.py",
                "line": 1239,
                "text": "caveat = payload.get(\"headline_caveat\")",
                "meaning": "the ONLY source of a row's headline_caveat is the payload",
            },
            {
                "file": "molmetal/scripts/registry.py",
                "line": 1245,
                "text": "\"headline_caveat\": (caveat.strip() if isinstance(caveat, str) and caveat.strip() else None)",
                "meaning": "a blank or non-string caveat becomes None -- the same rule the gate enforces",
            },
            {
                "file": "molmetal/scripts/registry.py",
                "line": 1426,
                "text": "rec.update(assess_citability(payload, rec[\"value\"], _rel(src_path), tracked))",
                "meaning": "build_metrics writes the row from assess_citability's return value",
            },
            {
                "file": "molmetal/scripts/registry.py",
                "line": 2061,
                "text": "metrics_doc = { ... \"metrics\": metrics, ... }",
                "meaning": "generate dumps those rows into the metrics document",
            },
            {
                "file": "molmetal/scripts/registry.py",
                "line": 2077,
                "text": "text = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)",
                "meaning": "and writes it over registry/metrics.yaml with atomic_write_file (2079)",
            },
            {
                "file": "molmetal/scripts/registry.py",
                "line": 2276,
                "text": "if r.get(\"registry_role\") != \"paper_line\": continue",
                "meaning": "`registry.py cite` -- the report the citability rule points at -- "
                           "prints ONLY paper-line rows; this is why the gate's scope is paper-facing",
            },
            {
                "file": "molmetal/scripts/registry.py",
                "line": 103,
                "text": "LINE_METRICS: Dict[str, Dict[str, str]] = { ... }",
                "meaning": "the paper-line class is declared in code, so the gate's scope is "
                           "re-derived on every run instead of being hand-listed",
            },
        ],
        "registry_sha256": _sha256(REGISTRY),
        "registry_parse": {
            "loader": "molmetal/scripts/registry.py:255-261  _load_yaml -> "
                      "yaml.safe_load(path.read_text(encoding=\"utf-8\")) or {}",
            "keys_read": ["quotable", "headline_grade", "headline_caveat",
                          "registry_role", "source_json"],
        },
        "paper_facing_ids_from_code": sorted(
            {mid for mapping in LINE_METRICS.values() for mid in mapping}
        ),
        "rows": proof_rows,
    }

    # ---- the citation-line check (added 2026-11-02) ------------------------
    # The 8 citations are resolved from the file BY CONTENT.  Drift that no
    # dated correction covers is a HARD ERROR: the generator refuses to
    # write, because a proof that cites the wrong line is not a proof.
    check = _citation_line_check(proof["code_citations"])
    proof["corrections"] = CITATION_CORRECTIONS
    proof["citation_line_check"] = check
    if check["uncorrected_drift"]:
        print(
            "evidence: REFUSING to write -- uncorrected citation drift at "
            f"code_citations{check['uncorrected_drift']}; "
            "resolve the anchor or record a dated correction",
            file=sys.stderr,
        )
        return 1

    summary = {
        "registry_sha256": _sha256(REGISTRY),
        "registry_rows_total": len(census),
        "quotable_rows": sum(1 for r in census if r["quotable"]),
        "quotable_and_headline_grade_rows": len(headline),
        "governed_rows": len(governed),
        "governed_metric_ids": [r["metric_id"] for r in governed],
        "quotable_not_headline_rows": len(soft),
        "quotable_not_headline_paper_rows_without_caveat": [
            r["metric_id"] for r in soft_paper_caveatless
        ],
        "out_of_scope_headline_rows_without_caveat": [
            r["metric_id"] for r in out_of_scope_caveatless
        ],
        "census_bound": {
            "corpus": "the 582 rows of registry/metrics.yaml (sha256 above)",
            "match_rule": "quotable is True AND headline_grade is True",
            "descent_rule": "the registry row is the row; the payload is metrics/by_metric/<file>.json "
                            "named by the row's source_json",
            "path_scoping": "source_json resolved against the repo root and rejected if it escapes it",
            "value_parity": "the registry row's headline_caveat must equal "
                            "assess_citability's rendering of the payload's headline_caveat",
            "note": "this is a COVERAGE census over one registry file, not a defect count "
                    "(docs/HONEST_NEGATIVE_CATALOG.md 6.244, 6.268 O12)",
        },
    }

    for name, payload in (
        ("row_census.json", census),
        ("payload_direction_proof.json", proof),
        ("summary.json", summary),
    ):
        text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n"
        atomic_write_file(OUT_DIR / name, text)
        print(f"wrote {OUT_DIR / name}")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
