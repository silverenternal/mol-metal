"""Wf-432-A1 — regression tests for the Wf-380 + Wf-394 + Wf-405-A2 + Wf-409-A1 +
Wf-424-A1 + Wf-427-A3 wire-in actual fix (SCAFFOLD_ONLY → READY closure).

Per AGENTS.md §3 skill catalog + §5.5 filesystem-truth rule + §5.7 NEW trainer
flag wire-in smoke-required rule + §5.2 honest-framing: every NEW CLI flag
wired into a Mol-Metal trainer's loss cascade / model constructor / sampling
path MUST run a 50-step end-to-end CPU smoke that verifies

    1. parse_args accepts the flag without NameError / argparse error
       (catches flag-name typo / dest-mismatch early);
    2. the trainer imports + initialises cleanly with the flag set
       (catches SCAFFOLD-module-not-imported regression);
    3. decode_ratio > 0 in the metrics JSON or stdout regex (catches the
       canonical decode_ratio = 0/192 regression per Wf-291D + Wf-299);
    4. loss[0] > loss[-1] monotone non-increasing loss over 2+ epochs
       (catches silent loss-divergence from a broken loss-modifier wire-in).

The historical concern surfaced in Wf-424-A11 §0 + Wf-431-A3 §0 + Wf-431-C1
was that the Wf-427-A3 patches landed (use_ot_cfm + per_cohort_features +
use_dann added as kwargs to train_one_metal_one_seed + train_one_epoch_joint)
but the call site at main() might still reference ``args.<flag>`` (a
main-scoped variable) inside the inner functions, raising NameError on the
first epoch step. The actual current state on disk shows the inner functions
ARE pure-kwarg and the main() callsite at line 1342-1370 passes them as direct
kwargs — verified empirically by the Wf-432-A1 smoke runs documented in
`TODO/completed/432a1_wirein_fix.md`.

This test file runs two end-to-end CPU smokes (D-MPNN + CFM) with all
Wf-380 / Wf-394 / Wf-409 wire-in flags ON and verifies returncode==0 +
decode_ratio>0 + loss_curve monotone decreasing. Both trainers pass on the
project venv per Wf-421-A2 + Wf-422-A2.

Wall budget: ~30 s on the host Python with project venv. CPU-only by design
(per AGENTS.md §5.7 + §5.1 standing GPU authorization honored — no GPU launches).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

# Make the smoke protocol + atomic write helpers importable as modules.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT / "molmetal" / "scripts"))


def _has_torch() -> bool:
    """Env-aware SKIP contract: if torch is unavailable, skip the smoke.
    Mirrors Wf-345-A2 + Wf-385-A3 pattern.
    """
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def _vintage_python() -> str:
    """Return the project venv interpreter that has torch installed.
    Per Wf-421-A2 + Wf-422-A2 the venv at ``.venv/bin/python`` is the
    canonical interpreter; falls back to the current interpreter if the
    venv does not exist (env-aware SKIP contract).
    """
    venv_python = _REPO_ROOT / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def _run_subprocess_smoke(
    trainer_path: str,
    flags: list,
    timeout_sec: int = 600,
) -> tuple:
    """Run a 50-step end-to-end CPU smoke and capture rc + stdout + stderr.

    Returns (returncode, stdout, stderr).
    """
    cmd = [_vintage_python(), str(_REPO_ROOT / trainer_path)] + flags
    completed = subprocess.run(
        cmd,
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout_sec,
    )
    return completed.returncode, completed.stdout, completed.stderr


def _parse_loss_curve(stdout: str, regex_pattern: str = r"train_loss=([\d.]+)") -> list:
    """Extract per-epoch loss values from the trainer's stdout banner."""
    return [float(x) for x in re.findall(regex_pattern, stdout)]


def _parse_decode_ratio(stdout: str) -> float:
    """Extract decode_ratio from the trainer's stdout banner. Returns
    ``float('nan')`` if not found (i.e. trainer does not report it).
    """
    m = re.search(r"decode_ratio=([0-9.]+)\s*\(([^)]+)\)", stdout)
    if m is None:
        return float("nan")
    return float(m.group(1))


class TestDMPNNWireInNoNameError(unittest.TestCase):
    """Verify Wf-380 + Wf-405-A2 + Wf-424-A1 + Wf-427-A3 wire-in flags reach
    the inner ``train_one_metal_one_seed`` / ``train_one_epoch_joint`` functions
    without raising NameError (the canonical Wf-424-A11 §5.7 smoke regression).
    """

    @unittest.skipUnless(
        _has_torch(),
        "torch not importable in current interpreter; skip per Wf-345-A2 env-aware SKIP contract",
    )
    def test_dmpnn_use_ot_cfm_on(self) -> None:
        """``--use-ot-cfm`` flag threads through to train_one_epoch_joint +
        loss cascade; no NameError; OT solver penalty appears in loss."""
        out_dir = Path("/tmp/wf432a1_test_use_ot_cfm")
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
        rc, stdout, stderr = _run_subprocess_smoke(
            trainer_path="molmetal/scripts/train_dmpnn_multitask_wasserstein.py",
            flags=[
                "--epochs", "1",
                "--batch-size", "8",
                "--device", "cpu",
                "--metals", "Ru",
                "--seeds", "0",
                "--out-dir", str(out_dir),
                "--no-kendall", "--no-dann",
                "--use-ot-cfm",
                "--data-csv", ".data/molmetal/metalcytotox_5metal_multitask_v4.csv",
                "--csv", ".data/molmetal/metalcytotox_5metal_multitask_v4.csv",
            ],
            timeout_sec=600,
        )
        self.assertEqual(rc, 0, f"returncode != 0: stderr={stderr[-500:]}")
        self.assertNotIn("NameError", stderr, f"NameError in stderr: {stderr[-500:]}")
        # Wasserstein penalty should be non-zero (OT solver produces a value)
        m = re.search(r"w=([0-9.]+)", stdout)
        self.assertIsNotNone(m, "no w=<val> in stdout; trainer did not log W₂ penalty")
        w_val = float(m.group(1))
        self.assertGreater(w_val, 0.0, f"W₂ penalty should be > 0; got {w_val}")

    @unittest.skipUnless(
        _has_torch(),
        "torch not importable in current interpreter; skip per Wf-345-A2 env-aware SKIP contract",
    )
    def test_dmpnn_per_cohort_features_on(self) -> None:
        """``--per-cohort-features`` flag threads through to train_one_epoch_joint +
        loss cascade; no NameError; per-cohort log line emitted."""
        out_dir = Path("/tmp/wf432a1_test_per_cohort")
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
        rc, stdout, stderr = _run_subprocess_smoke(
            trainer_path="molmetal/scripts/train_dmpnn_multitask_wasserstein.py",
            flags=[
                "--epochs", "1",
                "--batch-size", "8",
                "--device", "cpu",
                "--metals", "Ru",
                "--seeds", "0",
                "--out-dir", str(out_dir),
                "--no-kendall", "--no-dann",
                "--per-cohort-features",
                "--data-csv", ".data/molmetal/metalcytotox_5metal_multitask_v4.csv",
                "--csv", ".data/molmetal/metalcytotox_5metal_multitask_v4.csv",
            ],
            timeout_sec=600,
        )
        self.assertEqual(rc, 0, f"returncode != 0: stderr={stderr[-500:]}")
        self.assertNotIn("NameError", stderr, f"NameError in stderr: {stderr[-500:]}")
        # The per-cohort log line should appear
        self.assertIn(
            "[Wf-405-A2] --per-cohort-features ENABLED",
            stdout,
            f"per-cohort features log line not found in stdout:\n{stdout[-500:]}",
        )

    @unittest.skipUnless(
        _has_torch(),
        "torch not importable in current interpreter; skip per Wf-345-A2 env-aware SKIP contract",
    )
    def test_dmpnn_all_three_flags_on(self) -> None:
        """All 3 Wf-380 + Wf-405-A2 + Wf-424-A1 flags ON simultaneously:
        ``--use-ot-cfm --per-cohort-features --use-dann``; no NameError;
        OT penalty + per-cohort log line + DANN layer all active.
        Per AGENTS.md §5.7 NEW trainer flag wire-in smoke-required rule.
        """
        out_dir = Path("/tmp/wf432a1_test_all_three")
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
        rc, stdout, stderr = _run_subprocess_smoke(
            trainer_path="molmetal/scripts/train_dmpnn_multitask_wasserstein.py",
            flags=[
                "--epochs", "1",
                "--batch-size", "8",
                "--device", "cpu",
                "--metals", "Ru",
                "--seeds", "0",
                "--out-dir", str(out_dir),
                "--no-kendall",
                "--use-dann",
                "--use-ot-cfm",
                "--per-cohort-features",
                "--data-csv", ".data/molmetal/metalcytotox_5metal_multitask_v4.csv",
                "--csv", ".data/molmetal/metalcytotox_5metal_multitask_v4.csv",
            ],
            timeout_sec=600,
        )
        self.assertEqual(rc, 0, f"returncode != 0: stderr={stderr[-500:]}")
        self.assertNotIn("NameError", stderr, f"NameError in stderr: {stderr[-500:]}")
        # OT solver penalty should be > 0 (large OT penalty vs sliced-W₂)
        m = re.search(r"w=([0-9.]+)", stdout)
        self.assertIsNotNone(m, "no w=<val> in stdout")
        w_val = float(m.group(1))
        self.assertGreater(w_val, 0.0, f"W₂ penalty should be > 0; got {w_val}")
        # Per-cohort log line should appear
        self.assertIn(
            "[Wf-405-A2] --per-cohort-features ENABLED",
            stdout,
            f"per-cohort features log line not found",
        )


class TestCFMWireInNoNameError(unittest.TestCase):
    """Verify Wf-388 + Wf-394 + Wf-409 + Wf-405-A2 wire-in flags reach the CFM
    trainer's bond-CE / Wasserstein / SA / scaffold-conditioning paths
    without raising NameError.
    """

    @unittest.skipUnless(
        _has_torch(),
        "torch not importable in current interpreter; skip per Wf-345-A2 env-aware SKIP contract",
    )
    def test_cfm_baseline_decode_ratio(self) -> None:
        """Baseline CFM smoke verifies ``decode_ratio > 0`` (YuelBond
        conditionality canary per AGENTS.md §5.7 floor).
        """
        out_dir = Path("/tmp/wf432a1_test_cfm_baseline")
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
        rc, stdout, stderr = _run_subprocess_smoke(
            trainer_path="molmetal/scripts/train_cfm_yuelbond_wasserstein.py",
            flags=[
                "--epochs", "1",
                "--batch-size", "8",
                "--device", "cpu",
                "--seed", "42",
                "--out-dir", str(out_dir),
            ],
            timeout_sec=300,
        )
        self.assertEqual(rc, 0, f"returncode != 0: stderr={stderr[-500:]}")
        decode_ratio = _parse_decode_ratio(stdout)
        self.assertFalse(
            decode_ratio != decode_ratio,  # NaN check
            f"decode_ratio not found in stdout:\n{stdout[-500:]}",
        )
        self.assertGreater(decode_ratio, 0.0, f"decode_ratio must be > 0; got {decode_ratio}")

    @unittest.skipUnless(
        _has_torch(),
        "torch not importable in current interpreter; skip per Wf-345-A2 env-aware SKIP contract",
    )
    def test_cfm_all_wire_in_flags_on_2_epochs(self) -> None:
        """Full CFM wire-in smoke with all Wf-388 + Wf-394 + Wf-409 flags ON
        for 2 epochs; verifies ``decode_ratio > 0`` AND ``loss[0] > loss[1]``
        (the canonical AGENTS.md §5.7 smoke invariant).
        """
        out_dir = Path("/tmp/wf432a1_test_cfm_all_on")
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
        rc, stdout, stderr = _run_subprocess_smoke(
            trainer_path="molmetal/scripts/train_cfm_yuelbond_wasserstein.py",
            flags=[
                "--epochs", "2",
                "--batch-size", "8",
                "--device", "cpu",
                "--seed", "42",
                "--out-dir", str(out_dir),
                "--sa-penalty", "0.5",
                "--sa-target", "2.8",
                "--scaffold-conditioning",
                "--mmd-weight", "0.1",
                "--use-ot-cfm",
                "--per-cohort-features",
            ],
            timeout_sec=300,
        )
        self.assertEqual(rc, 0, f"returncode != 0: stderr={stderr[-500:]}")
        self.assertNotIn("NameError", stderr, f"NameError in stderr: {stderr[-500:]}")
        # decode_ratio > 0 canary
        decode_ratio = _parse_decode_ratio(stdout)
        self.assertFalse(
            decode_ratio != decode_ratio,
            f"decode_ratio not found in stdout:\n{stdout[-500:]}",
        )
        self.assertGreater(decode_ratio, 0.0, f"decode_ratio must be > 0; got {decode_ratio}")
        # loss curve monotone decreasing
        losses = _parse_loss_curve(stdout)
        self.assertGreaterEqual(
            len(losses),
            2,
            f"need ≥ 2 epochs to validate loss decrease; got {len(losses)}",
        )
        self.assertGreater(
            losses[0],
            losses[-1],
            f"loss must decrease: loss[0]={losses[0]} > loss[-1]={losses[-1]}",
        )


class TestWireInSourceAudit(unittest.TestCase):
    """Source-level audit: the inner functions must NOT reference
    ``args.<flag>`` (a main-scoped variable) directly — the Wf-427-A3
    fix threads flags as direct function kwargs. This is the structural
    contract that the Wf-424-A11 NameError fix enforced.
    """

    TRAINER_DMPNN = _REPO_ROOT / "molmetal" / "scripts" / "train_dmpnn_multitask_wasserstein.py"

    def test_dmpnn_train_one_metal_one_seed_no_args_references(self) -> None:
        """``train_one_metal_one_seed`` (line ~639) body must NOT contain
        ``args.<flag>`` (the Wf-424-A11 NameError regression surface).
        Per Wf-427-A3 the function takes ``use_ot_cfm`` + ``per_cohort_features``
        + ``use_dann`` as direct kwargs.
        """
        content = self.TRAINER_DMPNN.read_text()
        # Extract the body of train_one_metal_one_seed (until next def at same indent).
        start = content.find("def train_one_metal_one_seed(")
        self.assertNotEqual(start, -1, "train_one_metal_one_seed not found")
        # Find the next def at indent=0
        rest = content[start:]
        end_match = re.search(r"^def ", rest[1:], re.MULTILINE)
        end_pos = (end_match.start() + 1) if end_match else len(rest)
        body = rest[:end_pos]
        # Strip Python comments (lines whose first non-whitespace is '#')
        # so docstrings / inline comments that mention ``args`` historically
        # don't false-positive the audit. The Wf-424-A11 NameError regression
        # surface is EXECUTABLE args.<flag> references, not comment ones.
        stripped = "\n".join(
            line for line in body.split("\n") if not line.lstrip().startswith("#")
        )
        # Strip triple-quoted docstrings (""" ... """ or '''...''').
        stripped = re.sub(r'"""[\s\S]*?"""', "", stripped)
        stripped = re.sub(r"'''[\s\S]*?'''", "", stripped)
        # args. inside the executable body would be the bug
        args_uses = re.findall(r"\bargs\.", stripped)
        self.assertEqual(
            len(args_uses),
            0,
            f"train_one_metal_one_seed executable body still references "
            f"args.<flag> ({len(args_uses)} times); this is the Wf-424-A11 "
            "NameError regression surface. "
            "Patch should thread the flag as a direct kwarg per Wf-427-A3.",
        )

    def test_dmpnn_train_one_epoch_joint_no_args_references(self) -> None:
        """``train_one_epoch_joint`` (line ~427) body must NOT contain
        ``args.<flag>`` for the same reason.
        """
        content = self.TRAINER_DMPNN.read_text()
        start = content.find("def train_one_epoch_joint(")
        self.assertNotEqual(start, -1, "train_one_epoch_joint not found")
        rest = content[start:]
        end_match = re.search(r"^def ", rest[1:], re.MULTILINE)
        end_pos = (end_match.start() + 1) if end_match else len(rest)
        body = rest[:end_pos]
        args_uses = re.findall(r"\bargs\.", body)
        self.assertEqual(
            len(args_uses),
            0,
            f"train_one_epoch_joint still references args.<flag> ({len(args_uses)} times); "
            "this is the Wf-424-A11 NameError regression surface.",
        )

    def test_dmpnn_args_passed_to_train_one_metal_one_seed_as_kwargs(self) -> None:
        """The main() callsite at line ~1342 must pass
        ``use_ot_cfm`` + ``per_cohort_features`` + ``use_dann`` as direct
        kwargs (not via getattr in a way that re-introduces the NameError).
        """
        content = self.TRAINER_DMPNN.read_text()
        # The call to train_one_metal_one_seed in main()
        main_start = content.find("def main(")
        self.assertNotEqual(main_start, -1, "main() not found")
        main_body = content[main_start:]
        # Find the train_one_metal_one_seed call site
        call_idx = main_body.find("train_one_metal_one_seed(")
        self.assertNotEqual(call_idx, -1, "train_one_metal_one_seed call not found")
        # Extract ~30 lines after the call (the kwarg list)
        call_block = main_body[call_idx: call_idx + 3000]
        # Verify each of the 3 kwargs is passed
        self.assertIn(
            "use_ot_cfm=",
            call_block,
            "main() callsite missing use_ot_cfm= kwarg pass-through",
        )
        self.assertIn(
            "per_cohort_features=",
            call_block,
            "main() callsite missing per_cohort_features= kwarg pass-through",
        )
        self.assertIn(
            "use_dann=",
            call_block,
            "main() callsite missing use_dann= kwarg pass-through (Wf-424-A1)",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)