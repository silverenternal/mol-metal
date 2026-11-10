# SPDX-License-Identifier: Apache-2.0
"""Wf-A158 — a wall-clock TIMEOUT must never be reported as a FAIL.

THE DEFECT
----------
Wf-A135 (2026-10-31) established the rule: a pre-flight gate whose inner
subprocess ran out of wall time did NOT obtain its measurement, so the
honest verdict is a TIMEOUT — not a FAIL.  Reporting "the subprocess
timed out" as "the thing under test is broken" is a FALSE DEFECT CLAIM
(AGENTS.md §5.2 honest framing: a genuine defect is reported as FAIL, and
only a genuine defect may be reported as FAIL).

MEASURED 2026-11-01: the rule had been applied to 3 of the 6 gates that
carry a ``rc == 124`` branch.  The other three still mapped the timeout
onto FAIL:

  * ``gate_p2_data_leakage_diag``      — a timed-out leakage diagnostic
    was reported as "the data leaks".
  * ``gate_scaffold_zero_leak``        — a timed-out scaffold verifier
    was reported as "the scaffold split is not strict".
  * ``gate_r20_phase0_loaders_complete`` — a timed-out loader pytest was
    reported as "the R20 loader contract is broken".

THE FIX
-------
* ``_timeout_gate_result`` gained ``not_a_verdict_on`` (Wf-A158,
  2026-11-01) so the SAME helper serves gates that check something other
  than a trainer flag.  The evidence now names WHAT the timeout is not a
  verdict about.
* All three gates route ``rc == 124`` into that helper, so the status is
  ``TIMEOUT`` and the evidence says the measurement was NOT obtained.

WHAT THIS FILE GUARDS
---------------------
1. ``_timeout_gate_result`` returns status ``TIMEOUT`` — never ``FAIL``.
2. Each of the three repaired gates returns ``TIMEOUT`` (not ``FAIL``)
   when its inner subprocess returns ``rc == 124``.
3. The ``not_a_verdict_on`` phrase reaches the evidence string.
4. STATICALLY: EVERY ``if rc == 124:`` block in ``pre_flight_check.py``
   routes to a TIMEOUT verdict and contains no ``"FAIL"`` literal — so
   the next gate that is written cannot silently re-introduce the rule
   violation.

CPU-ONLY: every subprocess here is monkeypatched; nothing is spawned and
no GPU is touched.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _load(module_name: str, relpath: str):
    """Import a repo script by path (the scripts are not a package)."""
    spec = importlib.util.spec_from_file_location(module_name, REPO_ROOT / relpath)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pre_flight = _load("wfa158_pre_flight", "molmetal/scripts/pre_flight_check.py")

PREFLIGHT_REL = "molmetal/scripts/pre_flight_check.py"


# ---------------------------------------------------------------------------
# 1 — the helper itself
# ---------------------------------------------------------------------------
def test_timeout_status_constant() -> None:
    assert pre_flight.GATE_STATUS_TIMEOUT == "TIMEOUT"


def test_timeout_gate_result_never_returns_fail() -> None:
    res = pre_flight._timeout_gate_result(
        "wfa158_helper",
        "P0",
        "all",
        budget_sec=12.0,
        elapsed_sec=12.4,
        tail="(no output)",
        detail="a synthetic subprocess",
        not_a_verdict_on="a synthetic contract",
    )
    assert res.status == "TIMEOUT", f"expected TIMEOUT, got {res.status}"
    assert res.status != "FAIL"
    assert "FAIL" not in res.status


def test_timeout_gate_result_evidence_carries_not_a_verdict_on() -> None:
    res = pre_flight._timeout_gate_result(
        "wfa158_helper2",
        "P0",
        "all",
        budget_sec=12.0,
        elapsed_sec=12.4,
        tail="(no output)",
        detail="a synthetic subprocess",
        not_a_verdict_on="the widget strictness",
    )
    assert "the widget strictness" in res.evidence, (
        f"the not_a_verdict_on phrase must reach the evidence: {res.evidence}"
    )
    assert "NOT a verdict" in res.evidence
    assert "was NOT obtained" in res.evidence


def test_timeout_gate_result_default_not_a_verdict_on_is_the_wf_a135_wording() -> None:
    """The default must keep the original Wf-A135 phrase (back-compat)."""
    res = pre_flight._timeout_gate_result(
        "wfa158_helper3",
        "P0",
        "all",
        budget_sec=12.0,
        elapsed_sec=12.4,
        tail="(no output)",
        detail="a synthetic subprocess",
    )
    assert "the wire-in" in res.evidence


# ---------------------------------------------------------------------------
# 2 — the three repaired gates, driven with a fake subprocess
# ---------------------------------------------------------------------------
def _timeout_run(_cmd, **_kwargs):
    """A stand-in for ``_run`` that always reports a wall-clock timeout."""
    return 124, "", "[TIMEOUT after budget] the subprocess did not finish"


def test_p2_data_leakage_diag_reports_timeout_not_fail(tmp_path, monkeypatch) -> None:
    script = tmp_path / "check_data_leakage.py"
    script.write_text("# synthetic\n", encoding="utf-8")
    csv_a = tmp_path / "platinai.csv"
    csv_b = tmp_path / "metalcytotox.csv"
    csv_a.write_text("smiles\nC\n", encoding="utf-8")
    csv_b.write_text("smiles\nC\n", encoding="utf-8")
    monkeypatch.setattr(pre_flight, "_run", _timeout_run)

    res = pre_flight.gate_p2_data_leakage_diag(
        "wfa158_leak",
        "P2",
        "all",
        script,
        platinai_csv=csv_a,
        metalcytotox_csv=csv_b,
        out_path=tmp_path / "leak.json",
    )
    assert res.status == "TIMEOUT", f"expected TIMEOUT, got {res.status}: {res.evidence}"
    assert "the data-leakage split strictness" in res.evidence


def test_scaffold_zero_leak_reports_timeout_not_fail(tmp_path, monkeypatch) -> None:
    csv = tmp_path / "scaffold.csv"
    csv.write_text("smiles,split,scaffold_id\nC,train,s1\n", encoding="utf-8")
    monkeypatch.setattr(pre_flight, "_run", _timeout_run)

    res = pre_flight.gate_scaffold_zero_leak(
        "wfa158_scaffold",
        "P2",
        "all",
        [(csv, "synthetic")],
    )
    assert res.status == "TIMEOUT", f"expected TIMEOUT, got {res.status}: {res.evidence}"
    assert "the scaffold split strictness" in res.evidence


def test_r20_phase0_loaders_complete_reports_timeout_not_fail(tmp_path, monkeypatch) -> None:
    module = tmp_path / "loader.py"
    doc = tmp_path / "loader.md"
    test = tmp_path / "test_loader.py"
    for p in (module, doc, test):
        p.write_text("# non-empty\n", encoding="utf-8")
    monkeypatch.setattr(pre_flight, "_run", _timeout_run)

    res = pre_flight.gate_r20_phase0_loaders_complete(
        "wfa158_r20",
        "P1",
        "all",
        (module,),
        (doc,),
        (test,),
    )
    assert res.status == "TIMEOUT", f"expected TIMEOUT, got {res.status}: {res.evidence}"
    assert "the R20 loader contract" in res.evidence


# ---------------------------------------------------------------------------
# 3 — the STATIC guard: no rc==124 branch may map onto FAIL
# ---------------------------------------------------------------------------
def _iter_rc124_blocks(text: str):
    """Yield the source block of every ``if rc == 124:`` branch."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if not re.search(r"\brc\s*==\s*124\b", line):
            continue
        if not line.lstrip().startswith("if "):
            continue
        indent = len(line) - len(line.lstrip())
        body: list[str] = []
        for j in range(i + 1, len(lines)):
            nxt = lines[j]
            if not nxt.strip():
                body.append(nxt)
                continue
            nxt_indent = len(nxt) - len(nxt.lstrip())
            if nxt_indent <= indent:
                break
            body.append(nxt)
        yield i + 1, "\n".join(body)


def test_every_rc124_branch_routes_to_a_timeout_verdict() -> None:
    text = (REPO_ROOT / PREFLIGHT_REL).read_text(encoding="utf-8")
    blocks = list(_iter_rc124_blocks(text))
    assert len(blocks) >= 6, (
        f"expected >= 6 'rc == 124' branches, found {len(blocks)} — "
        f"a gate may have been added without the guard noticing"
    )
    for lineno, body in blocks:
        assert '"FAIL"' not in body and "'FAIL'" not in body, (
            f"{PREFLIGHT_REL}:{lineno} — an rc==124 branch contains a FAIL "
            f"status; a wall-clock timeout is NOT a defect verdict "
            f"(AGENTS.md §5.2):\n{body}"
        )
        assert (
            "_timeout_gate_result(" in body or "GATE_STATUS_TIMEOUT" in body
        ), (
            f"{PREFLIGHT_REL}:{lineno} — an rc==124 branch does not route to a "
            f"TIMEOUT verdict:\n{body}"
        )


def test_the_three_repaired_gates_are_among_the_rc124_branches() -> None:
    """The three Wf-A158 gates must actually carry an rc==124 branch."""
    text = (REPO_ROOT / PREFLIGHT_REL).read_text(encoding="utf-8")
    for fn in (
        "gate_p2_data_leakage_diag",
        "gate_scaffold_zero_leak",
        "gate_r20_phase0_loaders_complete",
    ):
        start = text.index(f"def {fn}(")
        # the function body runs until the next top-level def
        nxt = text.find("\ndef ", start + 1)
        body = text[start: nxt if nxt != -1 else len(text)]
        assert "rc == 124" in body, f"{fn} has no rc==124 branch"
        assert "_timeout_gate_result(" in body, (
            f"{fn} does not route its timeout into _timeout_gate_result"
        )
