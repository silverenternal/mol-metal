# Licensed under the Apache License, Version 2.0.
"""Regression tests for the AGENTS.md §5.11 yaml-only launch gate.

The gate is the enforcement surface for "training and inference launch from
YAML only, no CLI parameters" (user directive 2026-10-09). A gate that itself
regresses is worse than no gate, so these cover the two ways it could silently
stop working: it stops flagging a real violation, or it starts flagging the
documented migration comments that necessarily name the old flags.
"""

from __future__ import annotations

import importlib.util
import sys
import textwrap
from pathlib import Path

import pytest

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


class TestWrapperAuditing:
    def test_extra_flags_is_a_violation(self, tmp_path):
        w = _write(
            tmp_path,
            "launch_bad.sh",
            """\
            #!/usr/bin/env bash
            bash scripts/start.sh configs/x.yaml \\
              --extra-flags "--lr 0.001 --batch-size 32"
            """,
        )
        found, _ = gate._audit_wrapper(w)
        assert [v["kind"] for v in found] == ["extra_flags"]

    def test_trainer_flag_outside_allowlist_is_a_violation(self, tmp_path):
        w = _write(
            tmp_path,
            "launch_bad.sh",
            """\
            #!/usr/bin/env bash
            .venv/bin/python molmetal/scripts/train_cfm_yuelbond_wasserstein.py \\
              --config configs/x.yaml --lr 0.0001
            """,
        )
        found, _ = gate._audit_wrapper(w)
        kinds = {(v["kind"], v["detail"].split()[0]) for v in found}
        assert ("trainer_flag", "--lr") in kinds

    @pytest.mark.parametrize("flag", sorted(gate.ALLOWED_TRAINER_FLAGS))
    def test_allowed_flags_do_not_trip_the_gate(self, tmp_path, flag):
        """--config / --out-dir / --ckpt-dir identify the run, not the model."""
        w = _write(
            tmp_path,
            "launch_ok.sh",
            f"""\
            #!/usr/bin/env bash
            .venv/bin/python molmetal/scripts/train_metal_lomo_wasserstein.py \\
              --config configs/x.yaml {flag} somewhere
            """,
        )
        found, _ = gate._audit_wrapper(w)
        assert found == [], f"{flag} should be allowed, got {found}"

    def test_config_only_wrapper_is_clean(self, tmp_path):
        w = _write(
            tmp_path,
            "launch_ok.sh",
            """\
            #!/usr/bin/env bash
            export CUDA_VISIBLE_DEVICES=0
            bash scripts/start.sh configs/cfm.yaml --fast
            """,
        )
        found, _ = gate._audit_wrapper(w)
        assert found == []

    def test_flags_named_only_in_comments_are_not_violations(self, tmp_path):
        """The migrated wrappers document the flags they used to pass.

        Scanning comments would pin every one of them as broken forever, which
        is how a gate gets disabled rather than fixed.
        """
        w = _write(
            tmp_path,
            "launch_migrated.sh",
            """\
            #!/usr/bin/env bash
            # 2026-10-09: the --extra-flags block is GONE. Its values moved to
            #   --gen-head-lr-scale -> model.gen_head_lr_scale
            #   --entropy-reg-weight -> train.entropy_reg_weight
            # --device cuda:0 also moved to compute.device
            bash scripts/start.sh configs/cfm.yaml
            """,
        )
        found, _ = gate._audit_wrapper(w)
        assert found == []


class TestStartShAudit:
    def test_current_start_sh_refuses_extra_flags(self):
        """The shipped surface must actively refuse, per the gate's rule 3."""
        found = gate._audit_start_sh(gate.SCRIPTS_DIR / "start.sh")
        assert found == [], f"scripts/start.sh must refuse --extra-flags: {found}"

    def test_a_permissive_start_sh_is_a_violation(self, tmp_path):
        p = _write(
            tmp_path,
            "start.sh",
            """\
            #!/usr/bin/env bash
            while [[ $# -gt 0 ]]; do
              case "$1" in
                --extra-flags) EXTRA_FLAGS="$2"; shift 2 ;;
              esac
            done
            """,
        )
        saved = gate.SCRIPTS_DIR
        try:
            gate.SCRIPTS_DIR = tmp_path
            found = gate._audit_start_sh(p)
        finally:
            gate.SCRIPTS_DIR = saved
        assert [v["kind"] for v in found] == ["start_sh_permits_extra_flags"]


class TestEndToEnd:
    def test_repo_wrappers_pass(self):
        """The real repo, today: no wrapper smuggles a hyperparameter."""
        report = gate.audit()
        assert report["verdict"] == "PASS", report["violations"]
        assert report["n_wrappers_scanned"] > 0

    def test_report_is_json_serialisable(self):
        import json

        json.dumps(gate.audit())

    def test_exit_codes_follow_the_stdout_invariant(self, capsys):
        """0=PASS, 1=FAIL — the §4 invariant every gate shares."""
        assert gate.main([]) == 0
        assert "PASS" in capsys.readouterr().out


class TestDetachHeadIsNotTrainerFlags:
    """A leading `setsid`/`nohup` option is not a smuggled hyperparameter.

    `setsid --wait nohup "$PY" trainer.py` puts `--wait` on the same logical
    line as the trainer call. The gate read that as a trainer flag and reported
    a §5.11 violation, which pushed authors back toward bare `setsid` — the form
    whose occasional forking makes `rc=$?` after the command meaningless. Both
    gates in this repo hit that same blind spot; this pins the yaml-only half.
    """

    def test_setsid_option_before_allowed_flags_is_clean(self):
        line = 'setsid --wait nohup .venv/bin/python train_x.py --config c.yaml --out-dir o'
        flags = gate._FLAG.findall(gate._strip_detach_head(line))
        assert flags == ["--config", "--out-dir"]

    def test_real_trainer_flag_after_a_detach_head_is_still_caught(self):
        """The loosening must not become a bypass: `--epochs` is still a violation."""
        line = "setsid --wait nohup .venv/bin/python train_x.py --epochs 50"
        flags = gate._FLAG.findall(gate._strip_detach_head(line))
        assert "--epochs" in flags
        assert "--epochs" not in gate.ALLOWED_TRAINER_FLAGS

    def test_env_prefix_before_a_violation_is_still_caught(self):
        line = "CUDA_VISIBLE_DEVICES=0 setsid nohup .venv/bin/python train_x.py --lr 0.01"
        flags = gate._FLAG.findall(gate._strip_detach_head(line))
        assert "--lr" in flags

    def test_detach_head_mid_line_is_not_stripped(self):
        """Only a LEADING detach head is stripped; a stray one stays visible.

        Stripping anywhere would hide exactly the BLK-I1 `mangled_detach` defect,
        where `setsid nohup` lands in argv position after a continuation.
        """
        line = '.venv/bin/python train_x.py --config c.yaml setsid --wait'
        assert gate._strip_detach_head(line) == line
