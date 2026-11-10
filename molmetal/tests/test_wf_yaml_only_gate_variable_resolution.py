# Licensed under the Apache License, Version 2.0.
"""Regression tests for the yaml-only gate's VARIABLE-HELD trainer resolution.

The gate used to detect a trainer call with one literal regex,
``\\btrain_[a-z0-9_]+\\.py\\b``. A wrapper that holds its trainer in a shell
variable hid from it — so every flag that wrapper passed to the trainer was
unaudited, and the gate could not fail on that entire class of wrapper.

These tests pin the repaired behaviour:

* a variable-held trainer path IS detected, and its disallowed flags reported;
* the resolution is FAIL-CLOSED — a variable the gate cannot resolve is
  treated as a trainer call, not as clean;
* the resolution does not invent violations on lines that merely MENTION a
  flag (`echo "... --prereg"`, `grep -qE -- "--metals"`), which is the
  difference between a precise guard and a noisy one.

They live in their own file on purpose: the gate's existing regression file
belongs to another agent, and a repair must not edit somebody else's tests.
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_PATH = REPO_ROOT / "scripts" / "_yaml_only_gate.py"

_spec = importlib.util.spec_from_file_location("_yaml_only_gate", GATE_PATH)
gate = importlib.util.module_from_spec(_spec)
sys.modules["_yaml_only_gate"] = gate
_spec.loader.exec_module(gate)


def _write(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    return p


def _flags(found: list[dict]) -> set[str]:
    return {v["detail"].split()[0] for v in found if v["kind"] == "trainer_flag"}


class TestVariableHeldTrainerIsDetected:
    """The defect this repair closes: a trainer named by a shell variable."""

    def test_variable_held_trainer_flags_are_reported(self, tmp_path):
        w = _write(
            tmp_path,
            "launch_var.sh",
            """\
            #!/usr/bin/env bash
            SCRIPTS_DIR=/repo/molmetal/scripts
            TRAINER="${SCRIPTS_DIR}/train_dmpnn_multitask_wasserstein.py"
            PY=/repo/.venv/bin/python
            "${PY}" "${TRAINER}" \\
              --config configs/dmpnn.yaml \\
              --epochs 50 \\
              --hidden 128
            """,
        )
        found, _ = gate._audit_wrapper(w)
        assert _flags(found) == {"--epochs", "--hidden"}, found

    def test_variable_held_trainer_with_only_allowed_flags_is_clean(self, tmp_path):
        w = _write(
            tmp_path,
            "launch_var_ok.sh",
            """\
            #!/usr/bin/env bash
            TRAINER="${SCRIPTS_DIR}/train_metal_lomo_wasserstein.py"
            PY=/repo/.venv/bin/python
            "${PY}" "${TRAINER}" --config configs/x.yaml --out-dir "${OUT}"
            """,
        )
        found, _ = gate._audit_wrapper(w)
        assert found == [], found

    def test_transitive_variable_is_resolved(self, tmp_path):
        """`A="${B}"` where `B` holds the trainer — resolution follows the chain."""
        w = _write(
            tmp_path,
            "launch_chain.sh",
            """\
            #!/usr/bin/env bash
            BASE="${SCRIPTS_DIR}/train_cfm_yuelbond_wasserstein.py"
            TRAINER="${BASE}"
            "${PY}" "${TRAINER}" --lr 0.0001
            """,
        )
        found, _ = gate._audit_wrapper(w)
        assert _flags(found) == {"--lr"}, found

    def test_a_trainer_behind_setsid_and_a_conditional_is_still_detected(self, tmp_path):
        """`if ... setsid --wait "${PY}" "${TRAINER}"` — and `--wait` is setsid's."""
        w = _write(
            tmp_path,
            "launch_setsid.sh",
            """\
            #!/usr/bin/env bash
            TRAINER="${SCRIPTS_DIR}/train_dmpnn_multitask_wasserstein.py"
            PY=/repo/.venv/bin/python
            if CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}" setsid --wait "${PY}" "${TRAINER}" \\
                    --config configs/dmpnn.yaml \\
                    --w2-weight 0.05 \\
                    >> "${RUN_LOG}" 2>&1; then
                echo done
            fi
            """,
        )
        found, _ = gate._audit_wrapper(w)
        assert _flags(found) == {"--w2-weight"}, found


class TestResolutionIsFailClosed:
    """When the gate cannot resolve a variable it must flag, not shrug."""

    def test_unassigned_variable_in_script_position_is_treated_as_a_trainer(self, tmp_path):
        w = _write(
            tmp_path,
            "launch_unknown.sh",
            """\
            #!/usr/bin/env bash
            "${PY}" "${MYSTERY_SCRIPT}" --batch-size 64
            """,
        )
        found, _ = gate._audit_wrapper(w)
        assert _flags(found) == {"--batch-size"}, found

    def test_unresolvable_variable_reports_itself_as_fail_closed(self, tmp_path):
        w = _write(
            tmp_path,
            "launch_unknown2.sh",
            """\
            #!/usr/bin/env bash
            "${PY}" "${MYSTERY_SCRIPT}" --batch-size 64
            """,
        )
        env = gate._collect_assignments(w.read_text(encoding="utf-8"))
        line = next(
            line
            for _lineno, line in gate._logical_lines(w.read_text(encoding="utf-8"))
            if "MYSTERY_SCRIPT" in line
        )
        assert gate._trainer_call(line, env) == "variable-unresolved"

    def test_a_resolved_non_trainer_variable_is_not_a_trainer(self, tmp_path):
        """The fail-closed rule must not fire on a variable the gate CAN read."""
        w = _write(
            tmp_path,
            "launch_helper.sh",
            """\
            #!/usr/bin/env bash
            PY=/repo/.venv/bin/python
            HELPER="${SCRIPTS_DIR}/wf343c3_data_quality_check.py"
            "${PY}" "${HELPER}" --strict
            """,
        )
        found, _ = gate._audit_wrapper(w)
        assert found == [], found

    def test_an_unresolved_command_position_variable_is_fail_closed(self, tmp_path):
        """`${PY}` never assigned: the gate cannot rule the line out, so it flags."""
        w = _write(
            tmp_path,
            "launch_unresolved_cmd.sh",
            """\
            #!/usr/bin/env bash
            "${PY}" "${HELPER}" --strict
            """,
        )
        found, _ = gate._audit_wrapper(w)
        assert _flags(found) == {"--strict"}, found
        assert found[0]["via"] == "variable-unresolved", found


class TestNoPhantomViolations:
    """Mentioning a flag is not passing it."""

    def test_echo_of_a_flag_is_not_a_trainer_call(self, tmp_path):
        w = _write(
            tmp_path,
            "launch_echo.sh",
            """\
            #!/usr/bin/env bash
            echo "FAIL: --prereg file not found: ${PREREG}"
            """,
        )
        found, _ = gate._audit_wrapper(w)
        assert found == [], found

    def test_grep_pattern_is_not_a_trainer_call(self, tmp_path):
        w = _write(
            tmp_path,
            "launch_grep.sh",
            """\
            #!/usr/bin/env bash
            echo "${TRAINER_HELP}" | grep -qE -- "--metals\\b" || RC=$?
            """,
        )
        found, _ = gate._audit_wrapper(w)
        assert found == [], found

    def test_parameter_expansion_assignment_is_not_a_trainer_call(self, tmp_path):
        w = _write(
            tmp_path,
            "launch_param.sh",
            """\
            #!/usr/bin/env bash
            WAVE_TARGET="${1#--wave-}"
            """,
        )
        found, _ = gate._audit_wrapper(w)
        assert found == [], found

    def test_literal_trainer_is_still_detected(self, tmp_path):
        """The original literal rule must keep working — the repair is additive."""
        w = _write(
            tmp_path,
            "launch_literal.sh",
            """\
            #!/usr/bin/env bash
            .venv/bin/python molmetal/scripts/train_cfm_yuelbond_wasserstein.py \\
              --config configs/x.yaml --lr 0.0001
            """,
        )
        found, _ = gate._audit_wrapper(w)
        assert _flags(found) == {"--lr"}, found


class TestCoverageCensus:
    """A guard's own census is not a defect count, but it IS a coverage measure."""

    def test_report_carries_the_detection_census(self):
        report = gate.audit()
        census = report["trainer_call_detection"]
        assert set(census) == {
            "literal",
            "variable",
            "variable_unresolved_fail_closed",
        }, census
        assert census["literal"] > 0, census

    def test_shipped_variable_held_wrapper_is_detected(self):
        """The MEASURED instance this repair was written for.

        scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh names its
        trainer through `${TRAINER}`. The gate must SEE that invocation. The
        assertion is on the DETECTION, not on whether the wrapper still
        carries disallowed flags, so it stays true once the wrapper is
        repaired.
        """
        wrapper = (
            REPO_ROOT / "scripts" / "launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh"
        )
        text = wrapper.read_text(encoding="utf-8", errors="replace")
        env = gate._collect_assignments(text)
        detections = [
            gate._trainer_call(line, env)
            for _lineno, line in gate._logical_lines(text)
        ]
        assert any(d and d.startswith("variable:") for d in detections), detections
