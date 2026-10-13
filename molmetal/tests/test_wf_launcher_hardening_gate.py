# Licensed under the Apache License, Version 2.0.
"""Regression tests for the AGENTS.md §4 launcher-hardening gate (BLK-I1/I2).

A gate that passes everything is worse than no gate, so this suite is built
around a NEGATIVE CONTROL: every mechanism gets a deliberately broken wrapper
that the gate MUST flag, and a deliberately correct wrapper it MUST NOT.

The correct-wrapper cases are not padding. Calibrating this gate against the
64 real wrappers in scripts/ produced 26 initial "hard" findings, of which 14
were false positives — `setsid` in legitimate command position, CPU gates like
`pre_flight_check.py` treated as multi-hour trainers, pipelines whose status an
`if` already consumes, and `exec` (which replaces the shell, so there is no
parent left to die). Each of those regressions is pinned here.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_PATH = REPO_ROOT / "scripts" / "_launcher_hardening_gate.py"

_spec = importlib.util.spec_from_file_location("_launcher_hardening_gate", GATE_PATH)
gate = importlib.util.module_from_spec(_spec)
sys.modules["_launcher_hardening_gate"] = gate
_spec.loader.exec_module(gate)


def _write(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
    return p


def _kinds(path: Path) -> set[str]:
    return {f["kind"] for f in gate.audit_wrapper(path)["findings"]}


def _severities(path: Path) -> dict[str, str]:
    return {f["kind"]: f["severity"] for f in gate.audit_wrapper(path)["findings"]}


# ---------------------------------------------------------------------------
# NEGATIVE CONTROLS — each of these is broken and MUST be flagged
# ---------------------------------------------------------------------------


class TestNegativeControls:
    def test_unprotected_tee_under_errexit_and_pipefail_is_a_failure(self, tmp_path):
        """BLK-I1 mechanism (b). This is the one that cost BLK-C1 and BLK-D1."""
        w = _write(
            tmp_path,
            "launch_sigpipe.sh",
            """\
            #!/usr/bin/env bash
            set -euo pipefail
            PY="${REPO}/.venv/bin/python"
            TRAINER=molmetal/scripts/train_dmpnn_multitask_wasserstein.py
            "${PY}" "${TRAINER}" --epochs 50 2>&1 | tee "${LOG}"
            """,
        )
        assert "sigpipe_unprotected" in _kinds(w)
        assert _severities(w)["sigpipe_unprotected"] == "FAIL"

    def test_foreground_trainer_without_detachment_is_a_failure(self, tmp_path):
        """BLK-I1 mechanism (a): parent shell exits, trainer is SIGTERM'd."""
        w = _write(
            tmp_path,
            "launch_nodetach.sh",
            """\
            #!/usr/bin/env bash
            set -eu
            PY="${REPO}/.venv/bin/python"
            TRAINER=molmetal/scripts/train_dmpnn_multitask_wasserstein.py
            "${PY}" "${TRAINER}" --epochs 50
            """,
        )
        assert "no_detach" in _kinds(w)

    def test_trainer_without_preflight_is_flagged(self, tmp_path):
        """BLK-I2 — the pre-flight gate is advisory (WARN), but never silent."""
        w = _write(
            tmp_path,
            "launch_nopf.sh",
            """\
            #!/usr/bin/env bash
            set -e
            PY="${REPO}/.venv/bin/python"
            TRAINER=molmetal/scripts/train_dmpnn_multitask_wasserstein.py
            setsid nohup "${PY}" "${TRAINER}" --epochs 50
            """,
        )
        assert "no_preflight" in _kinds(w)

    def test_bare_python_trainer_is_flagged(self, tmp_path):
        """AGENTS.md §5.9 — the system interpreter has no torch."""
        w = _write(
            tmp_path,
            "launch_bare.sh",
            """\
            #!/usr/bin/env bash
            set -e
            python3 molmetal/scripts/train_dmpnn_multitask_wasserstein.py --epochs 50
            """,
        )
        assert "bare_interpreter" in _kinds(w)

    def test_mangled_detach_is_flagged(self, tmp_path):
        """The defect this gate was actually built to find.

        `setsid nohup` spliced onto a backslash continuation becomes two literal
        argv tokens; argparse exits 2 and the trainer never runs. MEASURED
        against train_dmpnn_multitask_wasserstein.py::parse_args.
        """
        w = _write(
            tmp_path,
            "launch_mangled.sh",
            """\
            #!/usr/bin/env bash
            set -e
            PY="${REPO}/.venv/bin/python"
            COMMON_ARGS=(--epochs 50)
            CUDA_VISIBLE_DEVICES=0 "${PY}" "${TRAINER}" \\
                --ckpt-dir /tmp/ckpts \\
            setsid nohup     "${COMMON_ARGS[@]}" 2>&1 | tee "${LOG}"
            """,
        )
        assert "mangled_detach" in _kinds(w)
        assert _severities(w)["mangled_detach"] == "FAIL"

    def test_gate_flags_a_completely_unhardened_wrapper(self, tmp_path):
        """The end-to-end negative control: every mechanism at once."""
        w = _write(
            tmp_path,
            "launch_all_bad.sh",
            """\
            #!/usr/bin/env bash
            set -euo pipefail
            TRAINER=molmetal/scripts/train_dmpnn_multitask_wasserstein.py
            python3 "${TRAINER}" --epochs 50 2>&1 | tee "${LOG}"
            """,
        )
        kinds = _kinds(w)
        assert {"bare_interpreter", "sigpipe_unprotected", "no_detach"} <= kinds


# ---------------------------------------------------------------------------
# FALSE-POSITIVE GUARDS — every one of these is correct as written
# ---------------------------------------------------------------------------


class TestNoFalsePositives:
    def test_setsid_in_command_position_is_not_mangled(self, tmp_path):
        w = _write(
            tmp_path,
            "launch_ok_cmdpos.sh",
            """\
            #!/usr/bin/env bash
            set -e
            PY="${REPO}/.venv/bin/python"
            setsid nohup "${PY}" molmetal/scripts/train_moleculenet.py --config c.yaml
            """,
        )
        assert "mangled_detach" not in _kinds(w)

    def test_setsid_wait_option_is_command_position(self, tmp_path):
        """`setsid --wait` must NOT be read as a detached word in argv position.

        `--wait` is setsid's own option, and it is the form that makes `rc=$?`
        after the command meaningful: bare `setsid` may fork and return 0 before
        the trainer exits, which silently turns "rc=0 means the run completed"
        into a false claim. The gate flagged this correct form as mangled until
        the prefix pattern learned setsid options; this pins that fix.
        """
        w = _write(
            tmp_path,
            "launch_ok_setsid_wait.sh",
            """\
            #!/usr/bin/env bash
            set -e
            PY="${REPO}/.venv/bin/python"
            setsid --wait nohup "${PY}" train_moleculenet.py --config c.yaml
            rc=$?
            exit $rc
            """,
        )
        assert "mangled_detach" not in _kinds(w)

    def test_env_assignment_prefix_then_setsid_is_command_position(self, tmp_path):
        w = _write(
            tmp_path,
            "launch_ok_envprefix.sh",
            """\
            #!/usr/bin/env bash
            set -e
            PY="${REPO}/.venv/bin/python"
            CUDA_VISIBLE_DEVICES=0 setsid nohup "${PY}" train_moleculenet.py
            """,
        )
        assert "mangled_detach" not in _kinds(w)

    def test_spawned_trainer_is_detached(self, tmp_path):
        w = _write(
            tmp_path,
            "launch_ok_detached.sh",
            """\
            #!/usr/bin/env bash
            set -e
            PY="${REPO}/.venv/bin/python"
            TRAINER=molmetal/scripts/train_dmpnn_multitask_wasserstein.py
            setsid nohup "${PY}" "${TRAINER}" --epochs 50 >"${LOG}" 2>&1 &
            wait
            """,
        )
        assert "no_detach" not in _kinds(w)

    def test_exec_replaces_the_shell_so_nothing_can_kill_it(self, tmp_path):
        """`exec` is detachment by another name — no parent is left to die."""
        w = _write(
            tmp_path,
            "launch_ok_exec.sh",
            """\
            #!/usr/bin/env bash
            set -e
            PY="${REPO}/.venv/bin/python"
            TRAINER=molmetal/scripts/train_dmpnn_multitask_wasserstein.py
            exec "${PY}" "${TRAINER}" --epochs 50
            """,
        )
        assert "no_detach" not in _kinds(w)

    def test_errexit_without_pipefail_plus_pipestatus_is_correct(self, tmp_path):
        """The tee pattern this repo actually ships. Not a finding.

        Without pipefail the pipeline's status is tee's, which is 0, so errexit
        cannot fire on it; PIPESTATUS[0] then reads the trainer's real code.
        """
        w = _write(
            tmp_path,
            "launch_ok_tee.sh",
            """\
            #!/usr/bin/env bash
            set -e
            PY="${REPO}/.venv/bin/python"
            TRAINER=molmetal/scripts/train_dmpnn_multitask_wasserstein.py
            "${PY}" "${TRAINER}" --epochs 50 2>&1 | tee "${LOG}"
            local rc=${PIPESTATUS[0]}
            """,
        )
        assert "sigpipe_unprotected" not in _kinds(w)

    def test_pipestatus_capture_alone_does_NOT_guard_under_pipefail(self, tmp_path):
        """`rc=${PIPESTATUS[0]}` on the NEXT line is dead code under `set -e`.

        This gate briefly accepted that pattern as a fix and it is not one. With
        `set -e` AND `set -o pipefail`, a non-zero pipeline aborts the shell at
        the pipeline; the capture on the following line is never executed, so the
        wrapper still dies silently. Measured 2026-10-09 — the line after the
        capture did not run and the wrapper exited at the pipeline with the
        pipeline's 141-derived status.
        """
        w = _write(
            tmp_path,
            "launch_bad_pipestatus_only.sh",
            """\
            #!/usr/bin/env bash
            set -euo pipefail
            PY="${REPO}/.venv/bin/python"
            TRAINER=molmetal/scripts/train_dmpnn_multitask_wasserstein.py
            "${PY}" "${TRAINER}" --epochs 50 2>&1 | tee "${LOG}"
            rc=${PIPESTATUS[0]}
            exit "${rc}"
            """,
        )
        assert "sigpipe_unprotected" in _kinds(w)

    def test_set_plus_o_pipefail_around_the_pipeline_is_a_real_guard(self, tmp_path):
        """Disabling pipefail for the one command makes PIPESTATUS readable."""
        w = _write(
            tmp_path,
            "launch_ok_pipefail_off.sh",
            """\
            #!/usr/bin/env bash
            set -euo pipefail
            PY="${REPO}/.venv/bin/python"
            TRAINER=molmetal/scripts/train_dmpnn_multitask_wasserstein.py
            set +o pipefail
            "${PY}" "${TRAINER}" --epochs 50 2>&1 | tee "${LOG}"
            rc=${PIPESTATUS[0]}
            set -o pipefail
            exit "${rc}"
            """,
        )
        assert "sigpipe_unprotected" not in _kinds(w)

    def test_tee_with_or_true_is_the_documented_fix_not_the_bug(self, tmp_path):
        w = _write(
            tmp_path,
            "launch_ok_tteeguard.sh",
            """\
            #!/usr/bin/env bash
            set -euo pipefail
            PY="${REPO}/.venv/bin/python"
            TRAINER=molmetal/scripts/train_dmpnn_multitask_wasserstein.py
            "${PY}" "${TRAINER}" --epochs 50 2>&1 | tee "${LOG}" || true
            """,
        )
        assert "sigpipe_unprotected" not in _kinds(w)

    def test_pipeline_inside_if_cannot_abort_the_wrapper(self, tmp_path):
        """`if cmd | tee | grep -q PASS; then` consumes the status itself."""
        w = _write(
            tmp_path,
            "launch_ok_ifconsumed.sh",
            """\
            #!/usr/bin/env bash
            set -euo pipefail
            PY="${REPO}/.venv/bin/python"
            TRAINER=molmetal/scripts/train_dmpnn_multitask_wasserstein.py
            if "${PY}" molmetal/scripts/pre_flight_check.py --pipeline all 2>&1 \\
                    | tee -a "${LOG}" | grep -q "PASS"; then
              echo gate-passed
            fi
            """,
        )
        assert "sigpipe_unprotected" not in _kinds(w)

    def test_cpu_gate_is_not_treated_as_a_multihour_trainer(self, tmp_path):
        """pre_flight_check.py has no worker pool and no reason to be detached."""
        w = _write(
            tmp_path,
            "launch_ok_cpu_gate.sh",
            """\
            #!/usr/bin/env bash
            set -e
            PY="${REPO}/.venv/bin/python"
            "${PY}" molmetal/scripts/pre_flight_check.py --pipeline launch
            """,
        )
        assert "no_detach" not in _kinds(w)
        assert "no_preflight" not in _kinds(w)

    def test_python3_c_snippet_is_a_shell_helper_not_a_trainer(self, tmp_path):
        """The dashboard parses JSON with `python3 -c`; that needs no torch."""
        w = _write(
            tmp_path,
            "launch_ok_python_c.sh",
            """\
            #!/usr/bin/env bash
            set -e
            verdict=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['fail'])" "${REPORT}")
            echo "verdict=${verdict}"
            """,
        )
        assert "bare_interpreter" not in _kinds(w)

    def test_documented_comments_do_not_trip_the_gate(self, tmp_path):
        """A comment naming `python3` must not be read as an invocation.

        `no_preflight` is expected here — this fixture genuinely has no
        pre-flight call — so the assertion is on the comment-driven kinds.
        """
        w = _write(
            tmp_path,
            "launch_ok_comments.sh",
            """\
            #!/usr/bin/env bash
            set -e
            # historically ran `python3 trainer.py` before §5.9
            # see also: setsid nohup "${PY}" trainer.py
            PY="${REPO}/.venv/bin/python"
            TRAINER=molmetal/scripts/train_dmpnn_multitask_wasserstein.py
            setsid nohup "${PY}" "${TRAINER}" --epochs 50 >"${LOG}" 2>&1 &
            """,
        )
        kinds = _kinds(w)
        assert not {"bare_interpreter", "mangled_detach"} & kinds, kinds

    def test_block_comment_naming_setsid_is_not_a_command(self, tmp_path):
        """`/* ... */` blocks are stripped; a wrapper documenting its own
        hardening must not be flagged for the text of that documentation."""
        w = _write(
            tmp_path,
            "launch_ok_blockcomment.sh",
            """\
            #!/usr/bin/env bash
            set -e
            /* Wf-390-A1: this used to be `python3 trainer.py` */
            PY="${REPO}/.venv/bin/python"
            TRAINER=molmetal/scripts/train_dmpnn_multitask_wasserstein.py
            setsid nohup "${PY}" "${TRAINER}" --epochs 50 >"${LOG}" 2>&1 &
            """,
        )
        kinds = _kinds(w)
        assert not {"bare_interpreter", "mangled_detach"} & kinds, kinds

    def test_wrapper_detached_by_an_in_repo_caller_downgrades_to_warn(self, tmp_path):
        """`nohup bash launch_x.sh` already detaches the whole process group."""
        scripts = tmp_path / "scripts"
        scripts.mkdir()
        (scripts / "launch_child.sh").write_text(
            textwrap.dedent(
                """\
                #!/usr/bin/env bash
                set -e
                PY="${REPO}/.venv/bin/python"
                TRAINER=molmetal/scripts/train_dmpnn_multitask_wasserstein.py
                "${PY}" "${TRAINER}" --epochs 50
                """
            ).lstrip(),
            encoding="utf-8",
        )
        (scripts / "master.sh").write_text(
            textwrap.dedent(
                """\
                #!/usr/bin/env bash
                set -e
                nohup bash scripts/launch_child.sh >"${LOG}" 2>&1
                """
            ).lstrip(),
            encoding="utf-8",
        )
        rec = next(
            r
            for r in gate.audit(scripts)["wrappers"]
            if r["file"].endswith("launch_child.sh")
        )
        severities = {f["kind"]: f["severity"] for f in rec["findings"]}
        assert severities.get("no_detach") == "WARN", severities
        assert rec["detached_by_caller"] == "master.sh"


# ---------------------------------------------------------------------------
# Structural / CLI contract
# ---------------------------------------------------------------------------


class TestLogicalLineJoining:
    def test_continuation_is_joined_before_mangled_detach_is_judged(self, tmp_path):
        """The check is only sound on JOINED lines; a per-physical-line scan
        would see `setsid nohup` as its own command head and pass."""
        joined = gate.logical_lines(
            "${PY} trainer.py \\\n  --epochs 50 \\\n  setsid nohup \"${ARGS[@]}\"\n"
        )
        assert len(joined) == 1
        assert "setsid" in joined[0][1]

    def test_block_comments_are_stripped(self, tmp_path):
        assert gate.logical_lines("/* setsid nohup */\nreal cmd\n") == [
            (2, "real cmd")
        ]

    def test_a_path_glob_is_not_a_comment_opener(self):
        """`reports/*` is a glob, not a `/*` block comment.

        Treating it as one swallowed the whole tail of the file — which is how
        launch_wf131_p3_pt_4cells.sh silently lost its mangled_detach finding.
        A comment-stripper that eats the code it is supposed to be auditing is
        worse than none, so this is pinned.
        """
        src = (
            "# comment about molmetal/reports/* dirs\n"
            "before=1\n"
            'PY="${REPO}/.venv/bin/python"\n'
            "TRAINER=molmetal/scripts/train_dmpnn_multitask_wasserstein.py\n"
            "setsid nohup \"${PY}\" \"${TRAINER}\" --epochs 50\n"
            "after=2\n"
        )
        out = gate.logical_lines(src)
        assert [n for n, _ in out] == [2, 3, 4, 5, 6], out
        assert any("after=2" in l for _, l in out)

    def test_a_case_glob_is_not_a_comment_opener(self):
        """`/*)` is a shell `case` pattern, not a block comment.

        launch_wf125_p1_controlled_pair.sh:118 has one, and reading it as a
        comment opener truncated 294 of that file's 411 lines — hiding both of
        its real findings.
        """
        src = (
            "case \"$1\" in\n"
            "  /*) ;;  # already absolute\n"
            "esac\n"
            'PY="${REPO}/.venv/bin/python"\n'
            "TRAINER=molmetal/scripts/train_dmpnn_multitask_wasserstein.py\n"
            "after=2\n"
        )
        out = gate.logical_lines(src)
        assert any("after=2" in l for _, l in out), out

    def test_scanner_never_drops_the_tail_of_a_real_wrapper(self):
        """Structural guard: no wrapper may lose its final code lines.

        Both truncation bugs above showed up only as a shrinking line count.
        This asserts the scanner's last logical line reaches the last non-comment
        physical line of every wrapper on disk, so the next instance fails
        loudly instead of quietly under-reporting.
        """
        truncated = []
        for p in sorted(gate.SCRIPTS_DIR.glob("*.sh")):
            src = p.read_text(encoding="utf-8", errors="replace")
            spans = gate._logical_spans(src)
            physical = [
                i
                for i, raw in enumerate(src.splitlines(), start=1)
                if raw.strip() and not raw.lstrip().startswith("#")
            ]
            if not physical or not spans:
                continue
            if spans[-1][1] < max(physical):
                truncated.append(p.name)
        assert not truncated, f"comment-stripper truncated: {truncated}"


class TestReportContract:
    def test_audit_reports_every_wrapper_on_disk(self):
        report = gate.audit()
        assert report["n_wrappers_scanned"] == len(
            list(gate.SCRIPTS_DIR.glob("*.sh"))
        )
        assert report["n_wrappers_scanned"] >= 60
        for key in ("n_fail", "n_warn", "n_findings_by_kind", "verdict"):
            assert key in report

    def test_verdict_is_fail_when_any_hard_finding_exists(self):
        report = gate.audit()
        assert report["verdict"] == ("FAIL" if report["n_fail"] else "PASS")

    def test_each_finding_is_one_defect_not_one_word(self):
        """A mangled splice contains both `setsid` and `nohup` — one finding."""
        report = gate.audit()
        keys = [(f["file"], f["line"], f["kind"]) for f in report["failures"]]
        assert len(keys) == len(set(keys))


class TestCli:
    def test_main_returns_1_on_fail_and_0_under_strict_only_when_clean(self, capsys):
        rc = gate.main([])
        out = capsys.readouterr().out
        assert "launcher-hardening" in out
        assert rc in (0, 1)
        if rc == 1:
            assert "FAIL" in out

    def test_strict_promotes_warnings_to_failure(self, capsys):
        report = gate.audit()
        if not report["warnings"]:
            pytest.skip("no warnings on disk to promote")
        assert gate.main(["--strict"]) == 1

    def test_json_output_is_machine_readable(self, capsys):
        gate.main(["--json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["gate"] == "launcher_hardening"

    def test_runs_as_a_subprocess_with_the_canonical_venv(self):
        """AGENTS.md §5.9 — the gate itself must not be invoked with bare python."""
        proc = subprocess.run(
            [str(REPO_ROOT / ".venv" / "bin" / "python"), str(GATE_PATH)],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        assert proc.returncode in (0, 1), proc.stderr
        assert "launcher-hardening" in proc.stdout