"""Wf-A135 — a wall-clock TIMEOUT must be a DISTINCT verdict, and every
AGENTS.md §5.7 wire-in smoke budget must be NAMED, OVERRIDABLE, and
derived from a MEASUREMENT.

WHY THIS GUARD EXISTS
======================

MEASURED 2026-10-31 on this host::

    .venv/bin/python molmetal/scripts/pre_flight_check.py --pipeline all --strict
    -> {"pass": 113, "fail": 3, "skip": 11, "warn": 0, "total": 127,
        "total_wall_sec": 411.0377}

All THREE FAILs were wall-clock TIMEOUTS, not defects:
``p1_chemprop_smoke`` (90 s budget, timed out @ 94.8 s),
``p3_wf385_scaffold_loss_wire_in_verified`` (150 s inner budget) and
``p3_dmpnn_loss_wire_in_smoke`` (300 s budget, rc=124).

For these three gates the FAIL verdict LITERALLY means *"the trainer flag
is NOT wired in"*, so a timeout — a measurement that was NEVER OBTAINED —
was indistinguishable from a wire-in defect.  Per the Wf-A128 principle
("a gate that FAILs on missing data is worse than no gate"), Wf-A135
gives the timeout its own verdict.

NON-VACUITY
===========

Every test below names the assertion that fails if the feature is
removed — a test that would still pass without the TIMEOUT verdict (or
without the named budgets) is worthless:

* ``test_timeout_verdict_is_distinct_from_fail``   -> the FIRST assert
  (``res.status == "TIMEOUT"``) fails if ``_timeout_gate_result`` is
  deleted or downgraded to ``"FAIL"``.
* ``test_dmpnn_gate_reads_the_named_budget``       -> the status assert
  fails if the gate stops resolving its budget from
  ``wfa135_smoke_budget`` (a hard-coded ``timeout=300.0`` would let the
  smoke run to completion and return FAIL/PASS instead of TIMEOUT).
* ``test_chemprop_gate_reads_the_named_budget``    -> same, for the
  chemprop smoke.
* ``test_wf385_inner_timeout_produces_timeout_verdict`` -> the status
  assert fails if the inner-smoke timeout is not detected from the
  inner gate's JSON.
* ``test_completed_smoke_with_false_assertion_remains_fail`` -> the
  status assert fails if a completed-but-false smoke is re-labelled
  TIMEOUT.
* ``test_budget_table_matches_pre_flight_constants`` /
  ``test_fixture_mirrors_the_budget_table`` -> fail if a budget is
  edited in one place and not the other.

CPU-ONLY BY CONTRACT
====================

The Wf-A135 budgets were MEASURED CPU-only, so this module pins
``CUDA_VISIBLE_DEVICES=""`` for the whole session: the smokes must never
reach for a GPU (and the measurement that justifies the budget would not
be reproduced).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molmetal.scripts import pre_flight_check as pfc  # noqa: E402
from molmetal.scripts import wfa135_smoke_budget as budget_mod  # noqa: E402

DMPNN_TRAINER = (
    REPO_ROOT / "molmetal" / "scripts" / "train_dmpnn_multitask_wasserstein.py"
)
CHEMPROP_TRAINER = REPO_ROOT / "molmetal" / "scripts" / "train_chemprop_v2_v3.py"
WF385_GATE_SCRIPT = (
    REPO_ROOT / "molmetal" / "scripts" / "wf385a3_scaffold_wire_in_gate.py"
)


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin CPU-only for every test in this module (Wf-A135 measurement contract)."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_dmpnn_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the D-MPNN gate at throwaway CSVs so the test never depends on
    the gitignored ``.data/**`` cohort being present."""
    joint = tmp_path / "joint.csv"
    joint.write_text("metal,seed\nRe,0\n", encoding="utf-8")
    aux = tmp_path / "aux.csv"
    aux.write_text("id,smiles\nX,CCO\n", encoding="utf-8")
    monkeypatch.setattr(pfc, "_DMPNN_SMOKE_DATA_CSV_CANDIDATES", (aux,), raising=True)
    return joint


# ---------------------------------------------------------------------------
# 1. The verdict itself
# ---------------------------------------------------------------------------


def test_timeout_verdict_is_distinct_from_fail() -> None:
    """A timeout is NOT a FAIL verdict.

    FAILS IF THE FEATURE IS REMOVED: the first assert
    (``res.status == "TIMEOUT"``) — with the verdict folded back into
    FAIL, the status would be ``"FAIL"``.
    """
    res = pfc._timeout_gate_result(
        "p1_chemprop_smoke",
        "P1",
        "chemprop_v2_v3",
        budget_sec=90.0,
        elapsed_sec=94.8,
        tail="some-subprocess-output-tail",
        detail="chemprop --quick smoke",
    )
    assert res.status == "TIMEOUT", (
        f"a wall-clock timeout must carry its own verdict, got " f"{res.status!r}"
    )
    assert res.status != "FAIL"
    assert res.status not in (
        "PASS",
        "FAIL",
        "SKIP",
        "WARN",
    ), "TIMEOUT must be DISTINCT from every pre-existing status"
    # The evidence must name the budget, the elapsed time, the output
    # tail, and state that the measurement was NOT obtained.
    assert "90" in res.evidence, "evidence must name the budget"
    assert "94.8" in res.evidence, "evidence must name the elapsed time"
    assert (
        "some-subprocess-output-tail" in res.evidence
    ), "evidence must carry the subprocess output tail"
    assert (
        "NOT obtained" in res.evidence
    ), "evidence must state that the measurement was NOT obtained"


def test_timeout_status_constant_is_distinct() -> None:
    """FAILS IF THE FEATURE IS REMOVED: the constant would not exist."""
    assert pfc.GATE_STATUS_TIMEOUT == "TIMEOUT"
    assert pfc.GATE_STATUS_TIMEOUT != "FAIL"


# ---------------------------------------------------------------------------
# 2. The real gates, exercised on the real timeout path
# ---------------------------------------------------------------------------


def test_dmpnn_gate_reads_the_named_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The D-MPNN smoke must take its budget from ``wfa135_smoke_budget``.

    FAILS IF THE FEATURE IS REMOVED: with the old hard-coded 300 s the
    smoke would RUN TO COMPLETION and the status assert below would see
    FAIL/PASS instead of ``TIMEOUT``.
    """
    joint = _fake_dmpnn_inputs(tmp_path, monkeypatch)
    monkeypatch.setenv(budget_mod.ENV_DMPNN_SMOKE_BUDGET_SEC, "0.05")
    res = pfc._gate_dmpnn_quick_smoke(DMPNN_TRAINER, csv_path=joint)
    assert res.status == "TIMEOUT", (
        f"the D-MPNN smoke exceeded its (overridden) budget but reported "
        f"{res.status!r}: {res.evidence}"
    )
    assert "NOT obtained" in res.evidence
    assert "0.05" in res.evidence or "budget" in res.evidence


def test_chemprop_gate_reads_the_named_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The chemprop smoke must take its budget from ``wfa135_smoke_budget``.

    FAILS IF THE FEATURE IS REMOVED: with the old hard-coded 90 s the
    smoke would RUN TO COMPLETION (it takes ~2 min on this host) and the
    status assert below would see FAIL/PASS instead of ``TIMEOUT``.

    The gate backs the shipped checkpoint up and restores it byte-identically
    (its own contract), and the smoke is killed long before it could write
    one — the sha256 assertion below proves the shipped bytes did not move.
    """
    monkeypatch.setenv(budget_mod.ENV_CHEMPROP_SMOKE_BUDGET_SEC, "0.05")
    ckpt = pfc._CHEMPROP_SMOKE_CKPT_PATH
    pre_sha = _sha256(ckpt) if ckpt.is_file() else None
    res = pfc._gate_chemprop_quick_smoke(CHEMPROP_TRAINER)
    assert res.status == "TIMEOUT", (
        f"the chemprop --quick smoke exceeded its (overridden) budget but "
        f"reported {res.status!r}: {res.evidence}"
    )
    assert "NOT obtained" in res.evidence
    if pre_sha is not None:
        assert _sha256(ckpt) == pre_sha, (
            "the timed-out smoke must not have disturbed the shipped " "checkpoint"
        )


def test_wf385_inner_timeout_produces_timeout_verdict(tmp_path: Path) -> None:
    """The Wf-385-A3 gate's INNER smoke timeout must be a TIMEOUT verdict.

    The inner gate script (``wf385a3_scaffold_wire_in_gate.py``) shells out
    to a 50-step CFM smoke and reports its own timeout in the JSON it writes
    (``smoke.error == "timeout after <N>s"``).  Before Wf-A135 the outer gate
    never read that JSON when the inner script exited non-zero, so an inner
    timeout was reported as the generic ``FAIL (rc=1)``.

    FAILS IF THE FEATURE IS REMOVED: the status assert below would see
    ``FAIL`` instead of ``TIMEOUT``.
    """
    out_dir = tmp_path / "wf385a3"
    # outer budget 33 s -> the inner gate script is given 3 s for its
    # 50-step smoke, which cannot finish in 3 s (MEASURED standalone: the
    # inner trainer alone needs several seconds) -> an inner timeout.
    res = pfc.gate_wf385_scaffold_loss_wire_in_verified(
        "p3_wf385_scaffold_loss_wire_in_verified",
        "P3",
        "wf385a3",
        WF385_GATE_SCRIPT,
        out_dir=out_dir,
        timeout=33.0,
    )
    assert res.status == "TIMEOUT", (
        f"the inner Wf-385-A3 smoke timed out but the gate reported "
        f"{res.status!r}: {res.evidence}"
    )
    assert "NOT obtained" in res.evidence


def test_completed_smoke_with_false_assertion_remains_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A smoke that FINISHED and whose assertion was FALSE stays a FAIL.

    This is the counterpart of the timeout case: the new verdict must not
    swallow genuine defects.  The fake trainer exits 0 and writes NO loss
    record, so the gate's loss-decrease invariant is false — the exact
    "the smoke finished and asserted false" case.

    FAILS IF THE FEATURE IS REMOVED (or over-reaches): the status assert
    below would see ``TIMEOUT`` instead of ``FAIL``.
    """
    joint = _fake_dmpnn_inputs(tmp_path, monkeypatch)
    fake_trainer = tmp_path / "fake_trainer.py"
    fake_trainer.write_text(
        "import sys\nprint('fake trainer finished')\nsys.exit(0)\n",
        encoding="utf-8",
    )
    res = pfc._gate_dmpnn_quick_smoke(fake_trainer, csv_path=joint, timeout=120.0)
    assert res.status == "FAIL", (
        f"a completed smoke whose invariant is false must stay a FAIL, got "
        f"{res.status!r}: {res.evidence}"
    )
    # The assertion text must survive verbatim.
    assert (
        "loss invariant FAILED" in res.evidence
    ), f"the FAIL verdict must keep its assertion text; got: {res.evidence}"


def test_completed_smoke_with_nonzero_exit_remains_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A smoke that finished with a non-zero exit stays a FAIL.

    FAILS IF THE FEATURE IS REMOVED (or over-reaches): the status assert
    below would see ``TIMEOUT`` instead of ``FAIL``.
    """
    joint = _fake_dmpnn_inputs(tmp_path, monkeypatch)
    fake_trainer = tmp_path / "fake_trainer.py"
    fake_trainer.write_text(
        "import sys\nsys.stderr.write('boom\\n')\nsys.exit(3)\n",
        encoding="utf-8",
    )
    res = pfc._gate_dmpnn_quick_smoke(fake_trainer, csv_path=joint, timeout=120.0)
    assert res.status == "FAIL", (
        f"a smoke that exited 3 must stay a FAIL, got {res.status!r}: "
        f"{res.evidence}"
    )
    assert "exit=3" in res.evidence


# ---------------------------------------------------------------------------
# 3. The budget table — one source of truth, mirrored in the fixture
# ---------------------------------------------------------------------------


def test_budget_table_matches_pre_flight_constants() -> None:
    """The budget the pre-flight uses must be the budget the module holds.

    FAILS IF THE FEATURE IS REMOVED: the aliases in ``pre_flight_check.py``
    would not exist, or would disagree with the module.
    """
    assert pfc.CHEMPROP_SMOKE_BUDGET_SEC == budget_mod.CHEMPROP_SMOKE_BUDGET_SEC
    assert pfc.DMPNN_SMOKE_BUDGET_SEC == budget_mod.DMPNN_SMOKE_BUDGET_SEC
    assert pfc.WF385_SMOKE_BUDGET_SEC == budget_mod.WF385_SMOKE_BUDGET_SEC

    for gate_name, budget in budget_mod.SMOKE_BUDGETS.items():
        assert budget.gate_name == gate_name
        assert budget.env_var, "every budget needs a NAMED env var"
        assert budget.measured_baseline_sec > 0.0, (
            f"{gate_name}: the budget must be derived from a MEASURED "
            f"baseline, not from nothing"
        )
        assert budget.headroom_factor > 1.0, (
            f"{gate_name}: the headroom factor must leave room above the "
            f"measured baseline"
        )
        # The budget must be the baseline times the stated factor (the
        # ceil() in the derivation may add < 1 s).
        expected = budget.measured_baseline_sec * budget.headroom_factor
        assert budget.budget_sec == pytest.approx(expected, abs=1.0), (
            f"{gate_name}: budget_sec {budget.budget_sec} is not "
            f"baseline {budget.measured_baseline_sec} x factor "
            f"{budget.headroom_factor} = {expected}"
        )


def test_fixture_mirrors_the_budget_table() -> None:
    """The machine-readable fixture must agree with the module.

    FAILS IF THE FEATURE IS REMOVED: the fixture would not exist, or a
    budget would have been edited in one place only.
    """
    payload = budget_mod.load_measured_fixture()
    smokes = payload["smokes"]
    for gate_name, agrees in budget_mod.fixture_matches_budgets().items():
        assert agrees, (
            f"{gate_name}: the fixture at {budget_mod.FIXTURE_PATH} does not "
            f"mirror the module's budget"
        )
        row = smokes[gate_name]
        assert row["measured_baseline_sec"] > 0.0
        assert row["headroom_factor"] > 1.0
        assert row["budget_sec"] > 0.0
        assert row["env_var"] == budget_mod.SMOKE_BUDGETS[gate_name].env_var
        assert isinstance(row["standalone_walls_sec"], list)
        assert len(row["standalone_walls_sec"]) >= 1, (
            f"{gate_name}: the fixture must record the MEASURED standalone "
            f"wall times, not just a derived number"
        )


def test_every_budget_is_reachable_from_the_pre_flight() -> None:
    """Every smoke in the module's table must be reachable from the pre-flight.

    FAILS IF THE FEATURE IS REMOVED: the gate names would not be wired.
    """
    for gate_name in budget_mod.SMOKE_BUDGETS:
        assert gate_name in (
            pfc.SMOKE_GATE_CHEMPROP,
            pfc.SMOKE_GATE_DMPNN,
            pfc.SMOKE_GATE_WF385,
        )


# ---------------------------------------------------------------------------
# 4. The environment override
# ---------------------------------------------------------------------------


def test_env_override_changes_the_effective_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A NAMED env var must override the budget — in the module AND in the
    pre-flight's resolver.

    FAILS IF THE FEATURE IS REMOVED: the resolver would ignore the env
    var and return the hard-coded constant.
    """
    monkeypatch.setenv(budget_mod.ENV_DMPNN_SMOKE_BUDGET_SEC, "123.5")
    assert budget_mod.smoke_budget_sec(budget_mod.GATE_DMPNN_SMOKE) == 123.5
    assert pfc._smoke_budget_sec(pfc.SMOKE_GATE_DMPNN) == 123.5


def test_malformed_env_override_is_reported_and_ignored(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """A malformed override must NOT become an instant timeout.

    FAILS IF THE FEATURE IS REMOVED: the override would be honoured
    verbatim, and a typo would silently mean "budget = 0".
    """
    monkeypatch.setenv(budget_mod.ENV_DMPNN_SMOKE_BUDGET_SEC, "not-a-number")
    resolved = budget_mod.smoke_budget_sec(budget_mod.GATE_DMPNN_SMOKE)
    assert resolved == budget_mod.DMPNN_SMOKE_BUDGET_SEC
    assert "not-a-number" in capsys.readouterr().err

    monkeypatch.setenv(budget_mod.ENV_DMPNN_SMOKE_BUDGET_SEC, "-5")
    assert (
        budget_mod.smoke_budget_sec(budget_mod.GATE_DMPNN_SMOKE)
        == budget_mod.DMPNN_SMOKE_BUDGET_SEC
    )


def test_unknown_gate_name_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A typo must not silently fall back to another smoke's budget."""
    with pytest.raises(KeyError):
        budget_mod.smoke_budget_sec("p1_not_a_real_smoke")


# ---------------------------------------------------------------------------
# 5. The summary must count the new verdict
# ---------------------------------------------------------------------------


def test_summary_reports_the_timeout_bucket() -> None:
    """The pre-flight's JSON summary must carry a ``timeout`` bucket.

    FAILS IF THE FEATURE IS REMOVED: the key would be absent, and
    ``pass + fail + skip + warn`` would silently be less than ``total``.
    """
    source = (REPO_ROOT / "molmetal" / "scripts" / "pre_flight_check.py").read_text(
        encoding="utf-8"
    )
    assert '"timeout": n_timeout' in source, (
        "the pre-flight summary must count the TIMEOUT verdict in its own "
        "bucket (Wf-A125 G11a: every status must be counted)"
    )
    assert (
        "n_timeout = sum(1 for r in results if r.status == GATE_STATUS_TIMEOUT)"
        in source
    )
    # And the exit code must treat a timeout like a SKIP under --strict.
    assert (
        "args.strict and (n_skip + n_timeout) > 0" in source
    ), "under --strict a TIMEOUT must fail the run exactly like a SKIP"
    # And the JSON must remain parseable + the fixture must be valid JSON.
    payload = json.loads(budget_mod.FIXTURE_PATH.read_text(encoding="utf-8"))
    assert payload["schema_version"]
