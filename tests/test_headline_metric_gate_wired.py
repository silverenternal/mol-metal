"""Regression test for the headline-metric gate wire-in (Wf-A12, 2026-10-05).

The headline-metric gate ``molmetal.scripts.headline_metric_loader.
load_headline_metric`` was a decorative no-op: zero callers outside its
own module and its tests (the audit confirmed this on 2026-10-05). This
test suite asserts that:

  I1. ``pre_flight_check.py`` actually imports + invokes the gate
      (``gate_p3_honest_neg_closure_m1_re_fold`` reads
      ``metal_lomo_auc_wasserstein.json`` via ``load_headline_metric``,
      not via plain ``json.loads``).
  I2. Every ``CLAIM_CATALOG`` entry in ``paper_dashboard.py`` that
      references ``metal_lomo_auc_wasserstein.json`` in its ``json_path``
      or ``ref`` carries an explicit ``disclosure`` field — without this
      the paper dashboard could render the disqualified value as a
      clean headline.
  I3. The gate is actually invoked end-to-end against the real on-disk
      disqualified JSON: ``load_headline_metric(path, as_headline=True,
      allow_disqualified=True)`` returns the payload AND surfaces the
      disqualifying reasons (proves the gate is REAL, not a one-line
      import).

These three invariants close the "decorative gate" defect surfaced by
the Wf-A11 ultracode audit.

CPU-only, stdlib-only, no torch. Run:

    pytest tests/test_headline_metric_gate_wired.py -v
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PRE_FLIGHT_PY = REPO_ROOT / "molmetal" / "scripts" / "pre_flight_check.py"
PAPER_DASHBOARD_PY = REPO_ROOT / "molmetal" / "scripts" / "paper_dashboard.py"
LOADER_PY = REPO_ROOT / "molmetal" / "scripts" / "headline_metric_loader.py"
HEADLINE_JSON = (
    REPO_ROOT / "metrics" / "by_metric" / "metal_lomo_auc_wasserstein.json"
)


def _load_module(path: Path, modname: str):
    spec = importlib.util.spec_from_file_location(modname, path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO_ROOT))
    spec.loader.exec_module(module)
    return module


def _load_loader():
    return _load_module(LOADER_PY, "headline_metric_loader_for_test")


# ---------------------------------------------------------------------------
# I1 — pre_flight_check.py imports + invokes the gate
# ---------------------------------------------------------------------------


def test_I1_pre_flight_imports_load_headline_metric():
    """``pre_flight_check.py`` MUST import ``load_headline_metric``.

    Without this import, the gate is decorative even if the loader
    module exists on disk.
    """
    src = PRE_FLIGHT_PY.read_text(encoding="utf-8")
    assert "load_headline_metric" in src, (
        "pre_flight_check.py does NOT import load_headline_metric — "
        "the gate is decorative. Wf-A12 wire-in is missing."
    )
    assert "from molmetal.scripts.headline_metric_loader import" in src, (
        "pre_flight_check.py imports load_headline_metric from the wrong "
        "path — must be `from molmetal.scripts.headline_metric_loader "
        "import (load_headline_metric, HeadlineNotUsableError)`."
    )


def test_I1_pre_flight_invokes_gate_at_the_json_read_site():
    """``gate_p3_honest_neg_closure_m1_re_fold`` MUST NOT use plain
    ``json.loads`` to read the headline JSON. Must call
    ``load_headline_metric`` instead.

    Audit (2026-10-05): the prior implementation used
    ``json.loads(metal_lomo_auc_json.read_text(...))`` which bypassed
    the canonical gate.
    """
    src = PRE_FLIGHT_PY.read_text(encoding="utf-8")
    # Locate the gate function body using AST-light scan: count braces
    # from the def line until the function body's indent drops back.
    body = _extract_function_body(
        src, "gate_p3_honest_neg_closure_m1_re_fold"
    )
    assert body is not None, "gate_p3_honest_neg_closure_m1_re_fold not found"
    # Must NOT use plain json.loads / json.load on the headline JSON.
    assert "json.loads(metal_lomo_auc_json.read_text" not in body, (
        "gate_p3_honest_neg_closure_m1_re_fold still uses plain "
        "json.loads on the headline JSON — the canonical gate is "
        "bypassed. Fix: route through load_headline_metric."
    )
    # Must call the gate.
    assert "load_headline_metric(" in body, (
        "gate_p3_honest_neg_closure_m1_re_fold does NOT invoke "
        "load_headline_metric — gate is decorative. Fix: invoke the "
        "gate with as_headline=True, allow_disqualified=True."
    )


def test_I1_pre_flight_surfaces_disqualifying_reasons():
    """The gate function must surface the disqualifying_reasons (so
    the gate's disqualification cannot be silently bypassed).
    """
    src = PRE_FLIGHT_PY.read_text(encoding="utf-8")
    body = _extract_function_body(
        src, "gate_p3_honest_neg_closure_m1_re_fold"
    )
    assert body is not None, "gate_p3_honest_neg_closure_m1_re_fold not found"
    assert "disqualifying_reasons" in body, (
        "gate_p3_honest_neg_closure_m1_re_fold does NOT surface "
        "disqualifying_reasons — gate can be silently bypassed."
    )


def _extract_function_body(src: str, func_name: str) -> str | None:
    """Return the source lines belonging to the top-level def named
    ``func_name``. Returns None if not found. Tolerant of multi-line
    def signatures and triple-quoted docstrings.
    """
    lines = src.splitlines(keepends=True)
    sig_idx = None
    for i, line in enumerate(lines):
        if re.match(rf"^def {re.escape(func_name)}\(", line):
            sig_idx = i
            break
    if sig_idx is None:
        return None
    # Walk forward; the function body is the contiguous block whose
    # first non-blank line has a leading indent > 0 AND whose indent
    # matches the indent of the first body line. We accept the simpler
    # "until the next top-level def or end of file".
    body_lines: list[str] = []
    for line in lines[sig_idx + 1 :]:
        if re.match(r"^def |^class |^@", line):
            break
        body_lines.append(line)
    return "".join(body_lines)


# ---------------------------------------------------------------------------
# I2 — paper_dashboard.py CLAIM_CATALOG carries disclosure for headline refs
# ---------------------------------------------------------------------------


def _extract_claim_catalog_dict_literals(src: str) -> list[dict]:
    """Walk paper_dashboard.py source and pull every dict literal that
    starts with ``"id":`` and contains a ``"json_path"`` or ``"ref"`` key.

    Uses a brace-counting scan — paper_dashboard.py claim entries are
    hand-curated python literals, no JSON or yaml parsing needed.
    """
    entries: list[dict] = []
    # Locate every top-level CLAIM_CATALOG / HONEST_NEGATIVE_CATALOG list
    # assignment. (DO_NOT_CLAIM is a single-entry ID list-style constant
    # in this file but the same pattern works.)
    list_starts: list[int] = []
    for marker in ("CLAIM_CATALOG", "HONEST_NEGATIVE_CATALOG", "DO_NOT_CLAIM"):
        # Accept type-annotated assignments: ``MARKER: List[...] = [``
        for m in re.finditer(rf"{marker}\b[^\n]*\[", src):
            # find the matching ']' by counting brackets (string-aware)
            i = m.end() - 1  # at the '['
            depth = 0
            j = i
            in_str = False
            str_q = ""
            esc = False
            while j < len(src):
                c = src[j]
                if in_str:
                    if esc:
                        esc = False
                    elif c == "\\":
                        esc = True
                    elif c == str_q:
                        in_str = False
                else:
                    if c in ('"', "'"):
                        in_str = True
                        str_q = c
                    elif c == "[":
                        depth += 1
                    elif c == "]":
                        depth -= 1
                        if depth == 0:
                            list_starts.append((m.start(), j + 1))
                            break
                j += 1
    for _start, end in list_starts:
        body = src[_start:end]
        # Within body, find every { ... } block containing "id" key.
        i = 0
        while i < len(body):
            brace_idx = body.find("{", i)
            if brace_idx < 0:
                break
            depth = 0
            j = brace_idx
            in_str = False
            str_q = ""
            esc = False
            while j < len(body):
                c = body[j]
                if in_str:
                    if esc:
                        esc = False
                    elif c == "\\":
                        esc = True
                    elif c == str_q:
                        in_str = False
                else:
                    if c in ('"', "'"):
                        in_str = True
                        str_q = c
                    elif c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            block = body[brace_idx : j + 1]
                            if '"id"' in block and (
                                '"json_path"' in block or '"ref"' in block
                            ):
                                entries.append({"_raw": block})
                            break
                j += 1
            i = j + 1
    return entries


def _entry_references_headline(raw: str) -> bool:
    return "metal_lomo_auc_wasserstein.json" in raw


def _entry_has_disclosure(raw: str) -> bool:
    """An entry has disclosure if it carries a ``"disclosure"`` field
    OR its ``json_path``/``ref`` ends in ``::disqualifying_reasons``
    OR its ``json_path``/``ref`` literally references the gate
    (load_headline_metric).
    """
    if '"disclosure"' in raw:
        return True
    if "disqualifying_reasons" in raw and "USABLE_AS_HEADLINE" in raw:
        return True
    if "load_headline_metric" in raw:
        return True
    return False


def test_I2_paper_dashboard_claim_entries_carry_disclosure():
    """Every CLAIM_CATALOG / DO_NOT_CLAIM entry that references
    ``metal_lomo_auc_wasserstein.json`` MUST carry a disclosure field.

    Without this, the paper dashboard renders the disqualified value
    (USABLE_AS_HEADLINE=false) as a clean headline.
    """
    src = PAPER_DASHBOARD_PY.read_text(encoding="utf-8")
    entries = _extract_claim_catalog_dict_literals(src)
    headline_entries = [
        e for e in entries if _entry_references_headline(e["_raw"])
    ]
    assert len(headline_entries) > 0, (
        "Expected paper_dashboard.py CLAIM_CATALOG to reference "
        "metal_lomo_auc_wasserstein.json at least once — but the "
        "scan found zero. Either the audit's premise is wrong OR "
        "the entry was already removed (in which case the gate is "
        "still decorative because nothing is gated)."
    )
    missing = [
        idx
        for idx, e in enumerate(headline_entries)
        if not _entry_has_disclosure(e["_raw"])
    ]
    assert not missing, (
        f"{len(missing)}/{len(headline_entries)} paper_dashboard.py "
        f"CLAIM_CATALOG / DO_NOT_CLAIM entries reference "
        f"metal_lomo_auc_wasserstein.json without a `disclosure` "
        f"field. Indices: {missing}. The paper dashboard could render "
        f"the disqualified value as a clean headline."
    )


# ---------------------------------------------------------------------------
# I3 — gate runs end-to-end on the on-disk disqualified JSON
# ---------------------------------------------------------------------------


def test_I3_gate_invocation_returns_payload_with_disqualifying_reasons():
    """Smoke: ``load_headline_metric(path, as_headline=True,
    allow_disqualified=True)`` MUST return the payload AND carry the
    disqualifying_reasons so the caller can render them.
    """
    loader = _load_loader()
    payload = loader.load_headline_metric(
        HEADLINE_JSON, as_headline=True, allow_disqualified=True
    )
    assert payload["USABLE_AS_HEADLINE"] is False, (
        "Expected on-disk JSON to be disqualified "
        "(USABLE_AS_HEADLINE=false). If this assertion fails, a clean "
        "re-measurement happened and the gate should be dropped."
    )
    reasons = payload["disqualifying_reasons"]
    assert isinstance(reasons, list) and len(reasons) >= 1
    joined = " ".join(reasons).lower()
    assert "oversample" in joined or "synthetic" in joined
    assert "n_folds" in joined or "5-metal" in joined or "2-metal" in joined


def test_I3_pre_flight_gate_function_surfaces_reasons_in_stderr(capsys):
    """End-to-end smoke: invoking
    ``gate_p3_honest_neg_closure_m1_re_fold`` against the real
    disqualified JSON must emit the disqualifying reasons to stderr
    (proving the gate fired, not just that the import is present).
    """
    pf = _load_module(PRE_FLIGHT_PY, "pre_flight_check_for_test")
    Path("docs/HONEST_NEGATIVE_CATALOG.md").exists() or None
    Path("TODO/03_plans/PRE_TRAINING_GATE_v2.md").exists() or None
    Path("docs/PAPER_SECTION_6_v4_FINAL.md").exists() or None
    pf.gate_p3_honest_neg_closure_m1_re_fold(
        "p3_honest_neg_closure_m1_re_fold_smoke",
        "P3",
        "all",
        HEADLINE_JSON,
        Path("docs/HONEST_NEGATIVE_CATALOG.md"),
        Path("TODO/03_plans/PRE_TRAINING_GATE_v2.md"),
        Path("docs/PAPER_SECTION_6_v4_FINAL.md"),
    )
    captured = capsys.readouterr()
    # The gate emits "[Wf-A12 pre_flight_check] headline gate fired ...
    # disqualifying_reasons=..." to stderr.
    assert "headline gate fired" in captured.err, (
        "Expected gate to surface disqualifying reasons to stderr. "
        f"Got: {captured.err[:500]!r}"
    )
    assert "disqualifying_reasons" in captured.err
    assert "oversample" in captured.err.lower()