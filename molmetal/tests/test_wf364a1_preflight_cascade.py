"""Wf-364-A1 — pre-flight cascade deep-dive guardrail tests.

Per Wf-358-D1 + Wf-362-B1: pre-flight had 1 FAIL on ``p2_tests_pass`` (root
cause: ``--wasserstein-w1`` argparse help text contained a literal ``%`` in
``PROJECTED +20-30%`` that crashed Python 3.12+ argparse's
``HelpFormatter._expand_help`` strict ``%``-formatting).  That single FAIL
cascaded to ``p3_wf343c2_per_wave_runtime_check`` because the Wf-343C2
companion script's ``preflight_no_fail_gates`` check (check #2) reads the
summary's ``fail`` count and FAILs when ``fail > 0``.

These tests pin two contracts that, if regressed, will silently re-trip
the cascade:

  1. ``train_dmpnn_multitask_wasserstein.py --help`` exits 0 (no Python
     3.12+ argparse ``ValueError: unsupported format character 'X'``).
     This is the Wf-364-A1 root-cause fix: every ``help=`` string in the
     script's ``add_argument`` calls must be free of literal ``%`` chars
     that aren't valid ``%(name)s`` substitution variables.

  2. ``wf343c2.check_preflight_no_fail_gates`` returns PASS when
     ``summary.fail == 0`` and FAIL when ``summary.fail > 0``.  The
     cascade contract is that the Wf-343C2 sub-check is fully driven by
     the pre-flight JSON summary; fixing the upstream ``p2_tests_pass``
     FAIL is what closes the cascade — the Wf-343C2 check itself is
     correct as-is.

All tests are CPU-only.  Wall budget: < 5 s on the test suite.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "molmetal" / "scripts"
VENV_PY = Path("/home/hugo/codes/mol-metal/.venv/bin/python")


# ===========================================================================
# Test 1: --help exits 0 (Wf-364-A1 root-cause regression guard)
# ===========================================================================
def test_train_dmpnn_multitask_wasserstein_help_exits_zero() -> None:
    """``train_dmpnn_multitask_wasserstein.py --help`` exits 0 (no crash).

    Wf-364-A1 root cause: Python 3.12+ argparse HelpFormatter does
    strict ``%``-formatting on help text; any literal ``%`` (e.g. a
    ``PROJECTED +20-30%`` annotation) crashes with ``ValueError:
    unsupported format character 'X' at index N``.  This regression
    silently breaks ``p2_tests_pass`` (which subprocess-invokes ``--help``
    to verify the ``--no-kendall`` / ``--no-dann`` flags are wired into
    the help formatter) and cascades to
    ``p3_wf343c2_per_wave_runtime_check`` via the
    ``preflight_no_fail_gates`` sub-check.
    """
    script = SCRIPTS_DIR / "train_dmpnn_multitask_wasserstein.py"
    assert script.is_file(), f"script not found at {script}"
    result = subprocess.run(
        [str(VENV_PY), str(script), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"--help crashed (rc={result.returncode}).\n"
        f"This is the Wf-364-A1 root cause: a literal '%' in some\n"
        f"argparse help= string breaks Python 3.12+ strict %-formatting.\n"
        f"stderr={result.stderr[-500:]}"
    )
    # The two flags pinned by the Wf-283B test should be visible in help
    assert "--no-kendall" in result.stdout
    assert "--no-dann" in result.stdout


# ===========================================================================
# Test 2: cascade status correctly propagates from pre-flight summary
# ===========================================================================
def test_preflight_no_fail_gates_cascade_status(tmp_path: Path) -> None:
    """``check_preflight_no_fail_gates`` correctly propagates FAIL cascade.

    Synthesise two pre-flight JSON stubs:

      * ``summary.fail == 0`` → ``check_preflight_no_fail_gates`` returns
        PASS  (cascade-resolved state — what Wf-364-A1 should leave us
        in).
      * ``summary.fail >  0`` → returns FAIL  (cascade-tripped state).

    The Wf-343C2 companion sub-check then aggregates the FAIL status
    into the per-wave verdict which, combined with ``--strict``, FAILs
    the parent pre-flight gate ``p3_wf343c2_per_wave_runtime_check``.
    This test pins the cascade contract so that future regressions on
    the Wf-343C2 status-propagation logic are caught early.
    """
    sys.path.insert(0, str(SCRIPTS_DIR))
    import wf343c2_per_wave_check as wf343c2  # noqa: E402

    # ---- Case A: 0 FAILs → PASS ----
    pf_clean = tmp_path / "preflight_clean.json"
    pf_clean.write_text(
        json.dumps(
            {
                "schema_version": "207.v1",
                "pipeline": "all",
                "summary": {"pass": 90, "fail": 0, "skip": 2, "total": 92},
                "gates": [],
            }
        )
    )
    status_a, evidence_a = wf343c2.check_preflight_no_fail_gates(
        preflight_path=pf_clean,
    )
    assert status_a == "PASS", (
        f"expected PASS when summary.fail=0; got {status_a!r} "
        f"(evidence={evidence_a!r})"
    )
    assert "upstream FAIL=0" in evidence_a, (
        f"evidence should mention upstream FAIL=0 to satisfy contract; "
        f"got {evidence_a!r}"
    )

    # ---- Case B: ≥ 1 upstream FAIL (e.g. p2_tests_pass) → FAIL (cascade-tripped)
    pf_fail = tmp_path / "preflight_fail.json"
    pf_fail.write_text(
        json.dumps(
            {
                "schema_version": "207.v1",
                "pipeline": "all",
                "summary": {"pass": 90, "fail": 1, "skip": 2, "total": 93},
                "gates": [{"name": "p2_tests_pass", "status": "FAIL"}],
            }
        )
    )
    status_b, evidence_b = wf343c2.check_preflight_no_fail_gates(
        preflight_path=pf_fail,
    )
    assert status_b == "FAIL", (
        f"expected FAIL when summary.fail=1 (upstream FAIL); got {status_b!r} "
        f"(evidence={evidence_b!r})"
    )
    assert "upstream FAIL=1" in evidence_b, (
        f"evidence should mention upstream FAIL=1 to satisfy contract; "
        f"got {evidence_b!r}"
    )

    # ---- Case C: Wf-364-A1 self-referential cycle fix.
    # When the only FAIL in pre-flight is this gate itself
    # (p3_wf343c2_per_wave_runtime_check), the cascade check should
    # self-exclude and PASS — breaking the cycle.  Without this fix,
    # the gate would FAIL forever once it ever FAILed.
    pf_self = tmp_path / "preflight_self_fail.json"
    pf_self.write_text(
        json.dumps(
            {
                "schema_version": "207.v1",
                "pipeline": "all",
                "summary": {"pass": 90, "fail": 1, "skip": 2, "total": 93},
                "gates": [
                    {"name": "p3_wf343c2_per_wave_runtime_check", "status": "FAIL"},
                ],
            }
        )
    )
    status_c, evidence_c = wf343c2.check_preflight_no_fail_gates(
        preflight_path=pf_self,
    )
    assert status_c == "PASS", (
        f"expected PASS when only FAIL is self (cycle fix); got {status_c!r} "
        f"(evidence={evidence_c!r})"
    )
    assert "self-excluded" in evidence_c, (
        f"evidence should mention self-exclusion per Wf-364-A1 cycle fix; "
        f"got {evidence_c!r}"
    )
    # Cascade propagation: parent gate p3_wf343c2_per_wave_runtime_check
    # uses --strict, so the FAIL here propagates 1:1 to the parent gate's
    # exit code 1 → FAIL on the parent pre-flight surface.  Pin this
    # contract with a tiny end-to-end smoke via subprocess.
    rc_b, _, _ = _run_wf343c2(
        preflight=pf_fail,
        expect_fail=True,
    )
    assert rc_b != 0, (
        f"wf343c2_per_wave_check.py --strict should FAIL when preflight has "
        f"any FAIL gate (cascade propagation); got rc={rc_b}"
    )


# ===========================================================================
# Helpers
# ===========================================================================
def _run_wf343c2(
    *,
    preflight: Path,
    expect_fail: bool,
) -> Tuple[int, str, str]:
    """Run ``wf343c2_per_wave_check.py --strict --skip-live`` subprocess."""
    script = SCRIPTS_DIR / "wf343c2_per_wave_check.py"
    out_dir = preflight.parent
    result = subprocess.run(
        [
            str(VENV_PY),
            str(script),
            "--wave",
            "Wf-364-A1 cascade regression test",
            "--pairs-metals",
            "Ru,Ir,Os,Rh,Re",
            "--pairs-seeds",
            "0,1,2",
            "--budget-hr",
            "10.0",
            "--preflight",
            str(preflight),
            "--out",
            str(out_dir / "wf343c2_cascade.json"),
            "--skip-live",
            "--strict",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(PROJECT_ROOT),
    )
    if expect_fail:
        assert result.returncode != 0, (
            f"expected non-zero rc (cascade FAIL); got rc={result.returncode}\n"
            f"stdout={result.stdout[-300:]}\nstderr={result.stderr[-300:]}"
        )
    return result.returncode, result.stdout, result.stderr
