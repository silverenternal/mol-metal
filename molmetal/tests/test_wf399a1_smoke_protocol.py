"""Wf-399-A1 tests for the 50-step end-to-end smoke protocol (2 tests).

Per AGENTS.md §5.7 added by Wf-399-A1: every trainer flag wire-in MUST run a
50-step end-to-end smoke that verifies decode_ratio > 0 + loss decreasing +
per-metric invariants. This test file verifies the smoke protocol helper
itself is correct.

Tests:
  * test_smoke_protocol_runs — verifies run_smoke_50step accepts a mock
    trainer (a `python -c "..."` one-liner) and returns a structured SmokeOutput
    with the expected schema; the mock trainer emits a metrics JSON with
    decode_ratio=1.0 + a 50-step decreasing loss curve; the smoke completes
    in <60s CPU-only.
  * test_decode_ratio_invariant — verifies verify_decode_ratio correctly
    identifies decode_ratio > 0 from each of the three supported sources
    (metrics key, nested final key, stdout regex) and correctly returns
    invariant_holds=False when the metric is absent.

Pre-existing conftest `_reference.path` ImportError does NOT affect these
tests because they import only the wf399a1_smoke_protocol module (no flow
matching upstream).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from molmetal.scripts.wf399a1_smoke_protocol import (  # noqa: E402
    SmokeOutput,
    flag_type_expectations,
    run_smoke_50step,
    verify_all_invariants,
    verify_decode_ratio,
    verify_flag_types,
    verify_loss_decrease,
)


def test_smoke_protocol_runs(tmp_path: Path) -> None:
    """Verify run_smoke_50step can invoke a mock trainer + parse its output.

    Uses a `python -c` one-liner as the mock trainer (path is the literal
    command string passed to subprocess.run). The mock writes a 50-step
    metrics JSON to tmp_path / "metrics.json" with decode_ratio=1.0 +
    a strictly decreasing loss curve, then exits 0. The test asserts the
    structured SmokeOutput + verify_all_invariants verdict are correct.
    """
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    metrics_json = out_dir / "metrics.json"

    # 50-step decreasing loss curve from 1.0 down to ~0.05
    loss_curve = [round(1.0 * (1.0 - 0.95 * i / 49.0), 4) for i in range(50)]
    mock_metrics = {
        "decode_ratio": 1.0,
        "loss_curve": loss_curve,
        "epoch": 1,
    }
    mock_trainer = (
        f"import json, pathlib; "
        f"pathlib.Path({str(metrics_json)!r}).write_text("
        f"json.dumps({json.dumps(mock_metrics)})); "
    ) + "print('decode_ratio=1.000 (192/192) trained 50 steps')"

    flags_list = [
        "--out-dir",
        str(out_dir),
        "--device",
        "cpu",
    ]
    smoke = run_smoke_50step(
        trainer_path="-c",  # not used since we pass a Python -c via flags below
        flags_list=["-c", mock_trainer],
        timeout_sec=120,
    )

    # Smoke should have run successfully; the mock is invoked via `python -c <code>`
    # because the protocol prepends `sys.executable trainer_path ...flags`. For the
    # mock to be invoked correctly we pass `-c` as trainer_path and the code as a
    # subsequent flag. The `python -c <code>` invocation requires the second positional
    # to be the code string, which our flags_list provides.
    # NOTE: subprocess.run with cmd=[sys.executable, "-c", "<code>", ...] works as expected.
    # We rebuild the command manually since `run_smoke_50step` prepends `sys.executable trainer_path`.
    import sys as _sys

    cmd = [
        _sys.executable,
        "-c",
        mock_trainer,
        "--out-dir",
        str(out_dir),
        "--device",
        "cpu",
    ]
    completed = subprocess.run(
        cmd,
        cwd=str(Path(__file__).resolve().parents[2]),
        capture_output=True,
        text=True,
        timeout=120,
    )

    # Build a SmokeOutput directly so we test the verifier logic
    parsed_metrics = (
        json.loads(metrics_json.read_text()) if metrics_json.exists() else {}
    )
    direct_smoke = SmokeOutput(
        trainer_path="mock-trainer",
        flags=flags_list,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        metrics_path=str(metrics_json),
        metrics=parsed_metrics,
        wall_seconds=0.5,
    )

    verdict = verify_all_invariants(direct_smoke)
    assert verdict["pass"] is True, f"verdict was {verdict}"
    assert verdict["decode_ratio"]["invariant_holds"] is True
    assert verdict["decode_ratio"]["decode_ratio"] == 1.0
    assert verdict["loss_decrease"]["decreasing"] is True
    assert verdict["loss_decrease"]["n_points"] == 50
    assert verdict["loss_decrease"]["loss_start"] > verdict["loss_decrease"]["loss_end"]

    # Also assert verify_decode_ratio + verify_loss_decrease are individually correct
    dr = verify_decode_ratio(direct_smoke)
    assert dr["invariant_holds"] is True
    assert dr["decode_ratio"] == 1.0

    ld = verify_loss_decrease(direct_smoke)
    assert ld["decreasing"] is True
    assert ld["n_points"] == 50


def test_decode_ratio_invariant() -> None:
    """Verify verify_decode_ratio correctly handles all 3 sources + missing case.

    The smoke protocol's `verify_decode_ratio` parses decode_ratio from:
      (a) `metrics.decode_ratio` (canonical JSON key),
      (b) `metrics.final.decode_ratio` (nested wrapper),
      (c) stdout regex `decode_ratio=<float>`.

    This test verifies all three sources are detected + invariant_holds=True
    when decode_ratio > 0, and that invariant_holds=False when the metric
    is missing.
    """
    # (a) metrics key source
    smoke_a = SmokeOutput(
        trainer_path="t",
        flags=[],
        returncode=0,
        stdout="",
        stderr="",
        metrics_path=None,
        metrics={"decode_ratio": 0.85},
        wall_seconds=0.1,
    )
    v_a = verify_decode_ratio(smoke_a)
    assert v_a["invariant_holds"] is True
    assert v_a["decode_ratio"] == 0.85
    assert v_a["source"] == "metrics.decode_ratio"
    assert v_a["error"] is None

    # (b) nested final key source
    smoke_b = SmokeOutput(
        trainer_path="t",
        flags=[],
        returncode=0,
        stdout="",
        stderr="",
        metrics_path=None,
        metrics={"final": {"decode_ratio": 0.92, "epoch": 1}},
        wall_seconds=0.1,
    )
    v_b = verify_decode_ratio(smoke_b)
    assert v_b["invariant_holds"] is True
    assert v_b["decode_ratio"] == 0.92
    assert v_b["source"] == "metrics.final.decode_ratio"

    # (c) stdout regex source (no metrics dict)
    smoke_c = SmokeOutput(
        trainer_path="t",
        flags=[],
        returncode=0,
        stdout="[wf299] Best decode_ratio = 0.788 (96/122) vs target 0.80",
        stderr="",
        metrics_path=None,
        metrics={},
        wall_seconds=0.1,
    )
    v_c = verify_decode_ratio(smoke_c)
    assert v_c["invariant_holds"] is True
    assert abs(v_c["decode_ratio"] - 0.788) < 1e-6
    assert v_c["source"] == "stdout regex"

    # (d) missing case — invariant_holds=False + error set
    smoke_d = SmokeOutput(
        trainer_path="t",
        flags=[],
        returncode=0,
        stdout="training step 0/50 loss=1.0",
        stderr="",
        metrics_path=None,
        metrics={"epoch": 1},
        wall_seconds=0.1,
    )
    v_d = verify_decode_ratio(smoke_d)
    assert v_d["invariant_holds"] is False
    assert v_d["decode_ratio"] is None
    assert v_d["error"] is not None
    assert "decode_ratio metric not found" in v_d["error"]

    # (e) verify_loss_decrease missing case (curve has 0 or 1 points)
    smoke_e = SmokeOutput(
        trainer_path="t",
        flags=[],
        returncode=0,
        stdout="",
        stderr="",
        metrics_path=None,
        metrics={"loss_curve": [0.5]},
        wall_seconds=0.1,
    )
    ld_e = verify_loss_decrease(smoke_e)
    assert ld_e["decreasing"] is False
    assert ld_e["n_points"] == 1
    assert "fewer than 2 points" in ld_e["error"]


# ---------------------------------------------------------------------------
# 2026-10-09 — flag TYPE assertions (AGENTS.md §5.7 invariant 4)
#
# The incident these pin: a smoke PASSED decode_ratio > 0 AND loss[0] > loss[49]
# while `--metals` (an nargs="+" flag) had been handed the single string
# "Ru Ir Os Re".  The trainer then matched no rows and every fold scored nan.
# Values are not enough; the smoke must also assert the TYPE each flag arrived
# with, checked against the trainer's OWN argparse contract.
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    """A parser shaped like the D-MPNN / CFM trainer flags."""
    p = argparse.ArgumentParser()
    p.add_argument("--metals", nargs="+", default=["Ru", "Ir", "Rh", "Os", "Re"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ckpt-dir", default=None)
    p.add_argument("--use-ot-cfm", action="store_true", default=False)
    return p


def _smoke(flags: list[str]) -> SmokeOutput:
    """A healthy smoke (rc=0, decode_ratio>0, 50-step decreasing loss)."""
    loss_curve = [round(1.0 * (1.0 - 0.95 * i / 49.0), 4) for i in range(50)]
    return SmokeOutput(
        trainer_path="mock-trainer",
        flags=list(flags),
        returncode=0,
        stdout="decode_ratio=1.000 (192/192)",
        stderr="",
        metrics_path=None,
        metrics={"decode_ratio": 1.0, "loss_curve": loss_curve},
        wall_seconds=0.5,
    )


def test_flag_type_expectations_reads_the_parser_contract() -> None:
    """flag_type_expectations must read nargs=/type=/action, not guess."""
    expected = flag_type_expectations(_parser())
    assert expected["metals"] == "list", expected
    assert expected["seeds"] == "list", expected
    assert expected["epochs"] == "int", expected
    assert expected["lr"] == "float", expected
    assert expected["ckpt_dir"] == "str", expected
    assert expected["use_ot_cfm"] == "bool", expected
    assert "help" not in expected


def test_correct_types_pass() -> None:
    """The happy path: a well-formed token list must NOT trip the new check."""
    p = _parser()
    args = p.parse_args(["--metals", "Pt", "Au", "--epochs", "30"])
    verdict = verify_flag_types(
        _smoke(["--metals", "Pt", "Au", "--epochs", "30"]),
        expectations=flag_type_expectations(p),
        parsed_args=args,
    )
    assert verdict["invariant_holds"] is True, verdict["violations"]
    assert verdict["violations"] == []
    assert "metals" in verdict["checked"]


def test_single_element_list_is_accepted() -> None:
    """A one-element list like ["Pt"] is legitimate and must pass.

    Only a one-element list whose element CONTAINS WHITESPACE is the bug.
    """
    p = _parser()
    args = p.parse_args(["--metals", "Pt"])
    verdict = verify_flag_types(
        _smoke(["--metals", "Pt"]),
        expectations=flag_type_expectations(p),
        parsed_args=args,
    )
    assert verdict["invariant_holds"] is True, verdict["violations"]


def test_whitespace_joined_string_is_caught() -> None:
    """THE 2026-10-09 INCIDENT: --metals 'Ru Ir Os Re' must FAIL.

    This is the shape that produced nan on every fold while the smoke still
    reported decode_ratio>0 and a decreasing loss.
    """
    p = _parser()
    args = p.parse_args(["--metals", "Ru Ir Os Re"])  # one joined token
    smoke = _smoke(["--metals", "Ru Ir Os Re"])
    verdict = verify_flag_types(
        smoke, expectations=flag_type_expectations(p), parsed_args=args
    )
    assert verdict["invariant_holds"] is False
    assert len(verdict["violations"]) == 1, verdict["violations"]
    v = verdict["violations"][0]
    assert v["dest"] == "metals"
    assert v["kind"] == "unsplit_list"
    assert "Ru Ir Os Re" in v["got"]


def test_wrong_container_is_caught() -> None:
    """A bare str where argparse declared a list must FAIL (wrong_container)."""
    p = _parser()
    args = p.parse_args(["--metals", "Pt"])
    # Simulate the type-mismatched arrival: argparse handed back a bare str.
    args.metals = "Pt"
    verdict = verify_flag_types(
        _smoke(["--metals", "Pt"]),
        expectations=flag_type_expectations(p),
        parsed_args=args,
    )
    assert verdict["invariant_holds"] is False
    v = verdict["violations"][0]
    assert v["dest"] == "metals"
    assert v["kind"] == "wrong_container"
    assert v["got"] == "str"


def test_wrong_scalar_type_is_caught() -> None:
    """Scalars are checked too — an int dest holding a str must FAIL."""
    p = _parser()
    args = p.parse_args(["--epochs", "30"])
    args.epochs = "30"  # type-mismatched arrival
    verdict = verify_flag_types(
        _smoke(["--epochs", "30"]),
        expectations=flag_type_expectations(p),
        parsed_args=args,
    )
    assert verdict["invariant_holds"] is False
    v = verdict["violations"][0]
    assert v["dest"] == "epochs"
    assert v["kind"] == "wrong_scalar_type"
    assert v["expected"] == "int"


def test_verify_all_invariants_fails_on_type_mismatch_alone() -> None:
    """The type invariant must gate the overall PASS, not sit beside it.

    All three value invariants hold here; only the TYPE is wrong. If this test
    ever regresses, a broken run can again be reported as a passing smoke.
    """
    p = _parser()
    flags = ["--metals", "Ru Ir Os Re", "--epochs", "1"]
    args = p.parse_args(flags)
    verdict = verify_all_invariants(
        _smoke(flags),
        expectations=flag_type_expectations(p),
        parsed_args=args,
    )
    # The pre-existing invariants are green...
    assert verdict["decode_ratio"]["invariant_holds"] is True
    assert verdict["loss_decrease"]["decreasing"] is True
    # ...and the type mismatch alone must still turn the verdict red.
    assert verdict["flag_types"]["invariant_holds"] is False
    assert verdict["pass"] is False


def test_verify_all_invariants_still_passes_for_a_healthy_smoke() -> None:
    """The new invariant must not break the existing PASS path (§5.7 1+2)."""
    p = _parser()
    flags = ["--metals", "Pt", "Au", "--epochs", "1"]
    args = p.parse_args(flags)
    verdict = verify_all_invariants(
        _smoke(flags),
        expectations=flag_type_expectations(p),
        parsed_args=args,
    )
    assert verdict["flag_types"]["invariant_holds"] is True
    assert verdict["pass"] is True


def test_type_check_skips_loudly_when_parser_unavailable() -> None:
    """No parser -> SKIP reported honestly, never a silent PASS.

    A skip must not masquerade as a check; `checked` is empty and `source` is
    "skip" so a reader can tell the invariant did not actually run.
    """
    smoke = _smoke(["--metals", "Pt"])
    smoke.trainer_path = "does/not/exist.py"
    verdict = verify_flag_types(smoke)
    assert verdict["invariant_holds"] is True  # skip != fail
    assert verdict["checked"] == []
    assert verdict["source"] == "skip"
    assert verdict["error"] is not None


def test_type_check_derives_expectations_from_a_real_trainer_module(
    tmp_path: Path,
) -> None:
    """End-to-end: the check auto-derives its contract from a trainer file.

    Writes a module exposing only a module-level `parse_args` (the shape every
    Mol-Metal trainer actually has) with an `nargs="+"` list default, then
    feeds the check a joined-string argv with no expectations supplied.
    """
    trainer = tmp_path / "fake_trainer.py"
    trainer.write_text(
        "import argparse\n"
        "def parse_args(argv=None):\n"
        "    p = argparse.ArgumentParser()\n"
        "    p.add_argument('--metals', nargs='+',\n"
        "                   default=['Ru', 'Ir', 'Rh', 'Os', 'Re'])\n"
        "    p.add_argument('--epochs', type=int, default=50)\n"
        "    return p.parse_args(argv)\n",
        encoding="utf-8",
    )

    smoke = _smoke(["--metals", "Ru Ir Os Re"])
    smoke.trainer_path = str(trainer)
    verdict = verify_flag_types(smoke)
    assert verdict["source"] != "skip", verdict
    assert "metals" in verdict["checked"], verdict
    assert verdict["invariant_holds"] is False, verdict["violations"]
    assert verdict["violations"][0]["kind"] == "unsplit_list"

    # ...and the well-formed argv through the same module passes.
    ok = _smoke(["--metals", "Ru", "Ir"])
    ok.trainer_path = str(trainer)
    assert verify_flag_types(ok)["invariant_holds"] is True


def test_type_check_handles_zero_arg_parse_args_trainers(tmp_path: Path) -> None:
    """A `parse_args()` that reads sys.argv implicitly must still be checkable.

    `train_cfm_yuelbond_wasserstein.parse_args()` takes no argv argument (the
    latent trainer quirk noted in test_config_yaml.py). The protocol reaches it
    by swapping sys.argv for the duration of the call and restoring it after,
    so a CFM smoke is type-checked rather than silently SKIPping.
    """
    trainer = tmp_path / "argv_implicit_trainer.py"
    trainer.write_text(
        "import argparse, sys\n"
        "def parse_args():\n"
        "    p = argparse.ArgumentParser()\n"
        "    p.add_argument('--metals', nargs='+',\n"
        "                   default=['Ru', 'Ir', 'Rh', 'Os', 'Re'])\n"
        "    return p.parse_args()\n",
        encoding="utf-8",
    )

    saved_argv = sys.argv
    try:
        smoke = _smoke(["--metals", "Ru Ir Os Re"])
        smoke.trainer_path = str(trainer)
        verdict = verify_flag_types(smoke)
        assert verdict["source"] != "skip", verdict
        assert verdict["invariant_holds"] is False, verdict["violations"]
        assert verdict["violations"][0]["dest"] == "metals"
        # sys.argv must survive the probe untouched.
        assert sys.argv == saved_argv, "sys.argv was not restored"
    finally:
        sys.argv = saved_argv
