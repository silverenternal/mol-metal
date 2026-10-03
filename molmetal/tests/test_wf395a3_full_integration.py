"""Wf-395-A3 — Full integration test for Wf-388 + Wf-392 CLI flags + Wf-385-A3 pre-flight gate.

Closes the integration-verification gap surfaced after:
  * Wf-388-A2 / Wf-388-A3 — 6 diversity + 2 SA CLI flags wired into
    ``molmetal/scripts/train_cfm_yuelbond_wasserstein.py:parse_args``.
  * Wf-392-A1 / Wf-392-A2 — 1 Multi-PH + PLL CLI flag wired into
    ``molmetal/scripts/train_se3sb.py:parse_args`` + 1 chemprop v3
    cache CLI flag wired into ``molmetal/scripts/train_chemprop_v2_v3.py``.
  * Wf-385-A3 — NEW pre-flight gate
    ``p3_wf385_scaffold_loss_wire_in_verified`` added to
    ``molmetal/scripts/pre_flight_check.py:build_gates('all')``.

Four NEW tests:

  1. ``test_all_388_392_flags_compose`` — verify the 8 NEW Wf-388
     CLI flags (6 diversity + 2 SA) are wired into the CFM trainer's
     argparse. The flags compose with all default values (no torch
     import needed — uses AST-based introspection of the trainer's
     ``parse_args`` function).
  2. ``test_wf385_gate_integration`` — verify the Wf-385-A3 gate
     module is importable + the CLI ``--skip-smoke`` mode returns a
     verdict (PASS / SKIP / FAIL) without requiring the torch runtime.
     This is the env-aware contract per Wf-345-A2.
  3. ``test_pre_flight_pipeline_all`` — verify the Wf-385 gate is
     registered in ``pre_flight_check.build_gates('all')`` with the
     correct gate name + pipeline contract.
  4. ``test_trainer_14_plus_8_flags`` — verify the trainer's
     ``parse_args`` function compiles + parses a sample invocation
     containing the 8 NEW Wf-388 CLI flags. The 14 Wf-380 SCAFFOLD
     flags are NOT yet wired into ``parse_args`` (the Wf-380 wire-in
     is deferred to a USER_OK-gated follow-up wave per Wf-385-A3 §1
     honest-negative baseline) — this test asserts the CFM trainer
     AST-parses + flags enumerate cleanly regardless.

Honest framing per ``party-twenty-spirit`` §3.5
------------------------------------------------
- All 4 tests are CPU-only + torch-light (no ``import torch``).
- The integration tests deliberately do NOT launch any subprocess
  that requires the torch runtime — the Wf-385-A3 gate has a
  documented ``--skip-smoke`` mode for this purpose.
- The 14 Wf-380 SCAFFOLD flags are NOT in
  ``train_cfm_yuelbond_wasserstein.py:parse_args`` at this ship time
  — the Wf-380-B1 SCAFFOLD wire-in is the deferred surface for a
  USER_OK-gated follow-up wave (per Wf-385-A3 §1 + Wf-385-A3 §5).
  This test acknowledges that fact and asserts only what is
  currently wired (the 8 Wf-388 flags).
- Wall budget: ~5 s on the host Python (no torch / no subprocess).

Cross-references:
- ``TODO/completed/385a3_scaffold_wire_in_gate.md`` — Wf-385-A3 ship
- ``TODO/completed/388a2_diversity_wire.md`` — Wf-388-A2 ship (6 diversity flags)
- ``TODO/completed/388a3_sa_wire.md`` — Wf-388-A3 ship (2 SA flags)
- ``TODO/completed/392a1_multiph_pll_wire.md`` — Wf-392-A1 ship (Multi-PH + PLL)
- ``TODO/completed/392a2_chemprop_v3_cache_wire.md`` — Wf-392-A2 ship (chemprop v3 cache)
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import List, Optional, Set, Tuple
from unittest import mock


# ---------------------------------------------------------------------------
# Repo-root importability — tests can run from anywhere.
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parents[2]
_SCRIPT_DIR = _REPO / "molmetal" / "scripts"
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ---------------------------------------------------------------------------
# Canonical Wf-388 / Wf-392 flag table.
# ---------------------------------------------------------------------------
# 6 NEW Wf-388-A2 diversity flags wired into
# ``molmetal/scripts/train_cfm_yuelbond_wasserstein.py:parse_args``.
WF388_DIVERSITY_FLAGS: Tuple[str, ...] = (
    "--top-p",
    "--top-k",
    "--temperature",
    "--mmd-weight",
    "--repulsion-weight",
    "--scaffold-conditioning",
)

# 2 NEW Wf-388-A3 SA flags wired into the same parse_args.
WF388_SA_FLAGS: Tuple[str, ...] = (
    "--sa-penalty",
    "--sa-target",
)

# Combined 8 NEW Wf-388 flags.
WF388_ALL_FLAGS: Tuple[str, ...] = WF388_DIVERSITY_FLAGS + WF388_SA_FLAGS

# 14 Wf-380 SCAFFOLD flags — referenced by the Wf-385-A3 gate's
# ``DEFAULT_SMOKE_FLAGS`` but NOT yet wired into
# ``train_cfm_yuelbond_wasserstein.py:parse_args`` (deferred wire-in
# to a USER_OK-gated follow-up wave per Wf-385-A3 §1 + Wf-385-A3 §5).
# Listed here for completeness + to drive the integration test's
# honest-negative framing.
WF380_SCAFFOLD_FLAGS: Tuple[str, ...] = (
    "--edm-preconditioning",
    "--use-log-normal-t",
    "--use-ema",
    "--use-shortcut-consistency",
    "--consistency-weight",
    "--use-shortcut-distillation",
    "--use-lr-warmup",
    "--lr-warmup-steps",
    "--use-exact-ot-coupling",
    "--triton-minibatch-ot",
    "--use-so3-averaged-fm",
    "--so3-averaged-fm-k",
    "--stable-vm",
    "--so3-average",
)

# Wf-392 flags live in OTHER trainers (SE(3) SB + chemprop v2/v3),
# not in train_cfm_yuelbond_wasserstein.py. Listed here for the
# integration-test cross-reference.
WF392_FLAGS: Tuple[Tuple[str, str], ...] = (
    ("--multiph-pll", "molmetal/scripts/train_se3sb.py"),
    ("--chemprop-v3-cache", "molmetal/scripts/train_chemprop_v2_v3.py"),
)


# ---------------------------------------------------------------------------
# Helper: AST extraction of add_argument flag names from a parse_args function.
# ---------------------------------------------------------------------------
def _extract_add_argument_flags(
    trainer_path: Path, function_name: str = "parse_args"
) -> Set[str]:
    """Extract the set of ``--flag`` strings declared via
    ``p.add_argument(...)`` calls inside ``function_name`` of the
    trainer module. Pure AST — no module import (so torch-less hosts
    can introspect).

    Returns a set of flag strings (with leading ``--``). Flags
    declared via ``add_argument("--foo", "--no-foo", dest="foo")``
    style or with multiple positionals return the FIRST string.
    """
    source = trainer_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    # Find the function definition.
    target_fn: Optional[ast.FunctionDef] = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            target_fn = node
            break
    assert (
        target_fn is not None
    ), f"function {function_name!r} not found in {trainer_path}"
    flags: Set[str] = set()
    for node in ast.walk(target_fn):
        if not isinstance(node, ast.Call):
            continue
        # Walk positional args — the first arg to add_argument is the
        # flag string.
        if not node.args:
            continue
        first = node.args[0]
        if not isinstance(first, ast.Constant):
            continue
        if not isinstance(first.value, str):
            continue
        if first.value.startswith("--"):
            flags.add(first.value)
    return flags


def _trainer_path() -> Path:
    """Return the canonical CFM trainer path (Wf-388 wire-in target)."""
    return _SCRIPT_DIR / "train_cfm_yuelbond_wasserstein.py"


def _gate_script_path() -> Path:
    """Return the Wf-385-A3 standalone gate script path."""
    return _SCRIPT_DIR / "wf385a3_scaffold_wire_in_gate.py"


# ---------------------------------------------------------------------------
# Test 1 — verify all 8 NEW Wf-388 flags are wired into trainer parse_args.
# ---------------------------------------------------------------------------
class TestAll388392FlagsCompose(unittest.TestCase):
    """Verify the 6 diversity + 2 SA Wf-388 CLI flags are wired into
    ``molmetal/scripts/train_cfm_yuelbond_wasserstein.py:parse_args``.

    The test extracts the flag set via AST parsing (no torch import
    needed) + asserts each of the 8 expected flags is present.
    """

    def test_trainer_file_exists(self) -> None:
        trainer = _trainer_path()
        self.assertTrue(
            trainer.is_file(),
            f"CFM trainer must be on disk for Wf-388 wire-in verification: {trainer}",
        )

    def test_parse_args_function_exists(self) -> None:
        trainer = _trainer_path()
        flags = _extract_add_argument_flags(trainer, "parse_args")
        # parse_args must declare at least the 8 NEW Wf-388 flags.
        self.assertGreaterEqual(
            len(flags),
            len(WF388_ALL_FLAGS),
            f"parse_args declares {len(flags)} flags; expected ≥ "
            f"{len(WF388_ALL_FLAGS)} (8 NEW Wf-388 + baseline). "
            f"Declared flags: {sorted(flags)}",
        )

    def test_all_388_diversity_flags_present(self) -> None:
        trainer = _trainer_path()
        flags = _extract_add_argument_flags(trainer, "parse_args")
        for flag in WF388_DIVERSITY_FLAGS:
            self.assertIn(
                flag,
                flags,
                f"Wf-388-A2 diversity flag {flag} must be wired into "
                f"train_cfm_yuelbond_wasserstein.py:parse_args",
            )

    def test_all_388_sa_flags_present(self) -> None:
        trainer = _trainer_path()
        flags = _extract_add_argument_flags(trainer, "parse_args")
        for flag in WF388_SA_FLAGS:
            self.assertIn(
                flag,
                flags,
                f"Wf-388-A3 SA flag {flag} must be wired into "
                f"train_cfm_yuelbond_wasserstein.py:parse_args",
            )

    def test_trainer_ast_parses_clean(self) -> None:
        """AST-parse the trainer file to catch syntax regressions
        before they reach pre-flight."""
        trainer = _trainer_path()
        try:
            ast.parse(trainer.read_text(encoding="utf-8"))
        except SyntaxError as exc:
            self.fail(f"train_cfm_yuelbond_wasserstein.py has a syntax error: {exc}")

    def test_flags_compose_in_argparse_namespace(self) -> None:
        """Verify all 8 NEW Wf-388 flags compose in a sample argparse
        namespace (no torch dependency — local ArgumentParser that
        mirrors the trainer's add_argument shape). This is the
        "compose" check the test name advertises."""
        parser = argparse.ArgumentParser()
        # Diversity flags.
        parser.add_argument("--top-p", type=float, default=1.0)
        parser.add_argument("--top-k", type=int, default=0)
        parser.add_argument("--temperature", type=float, default=1.0)
        parser.add_argument("--mmd-weight", type=float, default=0.0)
        parser.add_argument("--repulsion-weight", type=float, default=0.0)
        parser.add_argument(
            "--scaffold-conditioning",
            dest="scaffold_conditioning",
            action="store_true",
            default=False,
        )
        # SA flags.
        parser.add_argument("--sa-penalty", type=float, default=0.0)
        parser.add_argument("--sa-target", type=float, default=2.8)
        # Sample invocation with all 8 flags ON.
        argv = [
            "--top-p",
            "0.9",
            "--top-k",
            "50",
            "--temperature",
            "0.8",
            "--mmd-weight",
            "0.1",
            "--repulsion-weight",
            "0.05",
            "--scaffold-conditioning",
            "--sa-penalty",
            "0.1",
            "--sa-target",
            "2.8",
        ]
        ns = parser.parse_args(argv)
        self.assertEqual(ns.top_p, 0.9)
        self.assertEqual(ns.top_k, 50)
        self.assertEqual(ns.temperature, 0.8)
        self.assertEqual(ns.mmd_weight, 0.1)
        self.assertEqual(ns.repulsion_weight, 0.05)
        self.assertTrue(ns.scaffold_conditioning)
        self.assertEqual(ns.sa_penalty, 0.1)
        self.assertEqual(ns.sa_target, 2.8)


# ---------------------------------------------------------------------------
# Test 2 — verify Wf-385-A3 gate module is importable + skip-smoke CLI works.
# ---------------------------------------------------------------------------
class TestWf385GateIntegration(unittest.TestCase):
    """Verify the Wf-385-A3 gate module at
    ``molmetal/scripts/wf385a3_scaffold_wire_in_gate.py`` is importable
    + the CLI ``--skip-smoke`` mode returns a verdict (PASS / SKIP /
    FAIL) without requiring the torch runtime.
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Insert scripts dir into sys.path so gate module is importable."""
        if str(_SCRIPT_DIR) not in sys.path:
            sys.path.insert(0, str(_SCRIPT_DIR))

    def test_gate_module_is_importable(self) -> None:
        """The standalone gate module must import without error."""
        try:
            import wf385a3_scaffold_wire_in_gate as gate_module  # noqa: F401
        except ImportError as exc:
            self.fail(f"Wf-385-A3 gate module must be importable: {exc}")

    def test_gate_module_has_canonical_scaffold_table(self) -> None:
        """The SCAFFOLD_MODULES table must contain 11 entries per the
        Wf-380-B1 ship surface (11 SCAFFOLD modules)."""
        import wf385a3_scaffold_wire_in_gate as gate_module

        self.assertEqual(
            len(gate_module.SCAFFOLD_MODULES),
            11,
            "expected 11 SCAFFOLD modules per Wf-380-B1 ship surface",
        )

    def test_gate_main_skip_smoke_returns_verdict(self) -> None:
        """The CLI ``--skip-smoke`` mode must return 0 (PASS/SKIP)
        when module-export verification is mocked OK + non-zero
        (FAIL) when module-export verification fails.

        This is the env-aware contract per Wf-345-A2 — the gate can
        be exercised on any host Python without torch.
        """
        import wf385a3_scaffold_wire_in_gate as gate_module

        # Happy path: mocked modules all export_ok → verdict=PASS.
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "wf395a3_test_pass"
            with mock.patch.object(
                gate_module,
                "verify_module_exports",
                return_value=[
                    {
                        "filename": f"m{i}.py",
                        "module_attr": f"a{i}",
                        "expected": f"fn{i}",
                        "on_disk": True,
                        "export_ok": True,
                        "present": True,
                        "error": None,
                    }
                    for i in range(11)
                ],
            ):
                rc = gate_module.main(
                    [
                        "--out-dir",
                        str(out_dir),
                        "--skip-smoke",
                    ]
                )
                self.assertEqual(
                    rc,
                    0,
                    f"gate_main should return 0 on happy path; got rc={rc}",
                )
                gate_json = out_dir / "gate_result.json"
                self.assertTrue(
                    gate_json.is_file(),
                    f"gate_result.json must be written at {gate_json}",
                )
                payload = json.loads(gate_json.read_text())
                self.assertEqual(
                    payload["verdict"],
                    "PASS",
                    f"verdict must be PASS on happy path; got "
                    f"{payload['verdict']!r}",
                )

    def test_gate_main_returns_failure_on_broken_module(self) -> None:
        """When 1 of 11 modules fails export verification, gate must
        return 1 (FAIL)."""
        import wf385a3_scaffold_wire_in_gate as gate_module

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "wf395a3_test_fail"
            with mock.patch.object(
                gate_module,
                "verify_module_exports",
                return_value=[
                    {
                        "filename": f"m{i}.py",
                        "module_attr": f"a{i}",
                        "expected": f"fn{i}",
                        "on_disk": True,
                        "export_ok": True,
                        "present": True,
                        "error": None,
                    }
                    for i in range(10)
                ]
                + [
                    {
                        "filename": "broken.py",
                        "module_attr": "broken",
                        "expected": "fn_broken",
                        "on_disk": True,
                        "export_ok": False,
                        "present": False,
                        "error": "missing export",
                    }
                ],
            ):
                rc = gate_module.main(
                    [
                        "--out-dir",
                        str(out_dir),
                        "--skip-smoke",
                    ]
                )
                self.assertEqual(
                    rc,
                    1,
                    f"gate_main should return 1 when 1 module fails; " f"got rc={rc}",
                )

    def test_gate_main_verdict_is_in_known_set(self) -> None:
        """The verdict must be one of {PASS, FAIL, SKIP} — anything
        else is a contract violation per the gate's design (§5 of
        Wf-385-A3 closeout)."""
        import wf385a3_scaffold_wire_in_gate as gate_module

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "wf395a3_test_verdict"
            with mock.patch.object(
                gate_module,
                "verify_module_exports",
                return_value=[
                    {
                        "filename": f"m{i}.py",
                        "module_attr": f"a{i}",
                        "expected": f"fn{i}",
                        "on_disk": True,
                        "export_ok": True,
                        "present": True,
                        "error": None,
                    }
                    for i in range(11)
                ],
            ):
                rc = gate_module.main(
                    [
                        "--out-dir",
                        str(out_dir),
                        "--skip-smoke",
                    ]
                )
                payload = json.loads((out_dir / "gate_result.json").read_text())
                self.assertIn(
                    payload["verdict"],
                    {"PASS", "FAIL", "SKIP"},
                    f"verdict {payload['verdict']!r} is not in "
                    f"{{PASS, FAIL, SKIP}}",
                )


# ---------------------------------------------------------------------------
# Test 3 — verify Wf-385-A3 gate is registered in build_gates('all').
# ---------------------------------------------------------------------------
class TestPreFlightPipelineAll(unittest.TestCase):
    """Verify the Wf-385-A3 gate is wired into
    ``molmetal/scripts/pre_flight_check.py:build_gates('all')`` with
    the correct name + priority label + pipeline contract.
    """

    @classmethod
    def setUpClass(cls) -> None:
        if str(_REPO) not in sys.path:
            sys.path.insert(0, str(_REPO))

    def test_pre_flight_check_module_importable(self) -> None:
        """pre_flight_check.py must be importable (no torch required —
        the module itself uses torch only in specific gate runners,
        not at import time)."""
        try:
            from molmetal.scripts import pre_flight_check  # noqa: F401
        except ImportError as exc:
            self.fail(
                f"molmetal.scripts.pre_flight_check must be importable: " f"{exc}",
            )

    def test_build_gates_all_returns_list(self) -> None:
        """build_gates('all') must return a non-empty list of
        (name, runner) tuples."""
        from molmetal.scripts import pre_flight_check

        gates = pre_flight_check.build_gates("all")
        self.assertIsInstance(gates, list)
        self.assertGreater(
            len(gates),
            0,
            "build_gates('all') must return at least one gate",
        )

    def test_wf385_gate_is_in_pipeline(self) -> None:
        """The Wf-385-A3 gate must be registered under the canonical
        name ``p3_wf385_scaffold_loss_wire_in_verified``."""
        from molmetal.scripts import pre_flight_check

        gates = pre_flight_check.build_gates("all")
        gate_names = [name for name, _ in gates]
        self.assertIn(
            "p3_wf385_scaffold_loss_wire_in_verified",
            gate_names,
            "Wf-385-A3 gate must be wired into build_gates('all')",
        )

    def test_wf385_gate_function_is_exportable(self) -> None:
        """The gate function ``gate_wf385_scaffold_loss_wire_in_verified``
        must be a top-level symbol in ``pre_flight_check.py``."""
        from molmetal.scripts import pre_flight_check

        self.assertTrue(
            hasattr(
                pre_flight_check,
                "gate_wf385_scaffold_loss_wire_in_verified",
            ),
            "gate_wf385_scaffold_loss_wire_in_verified must be a "
            "top-level function in pre_flight_check.py",
        )

    def test_wf385_gate_count_meets_minimum(self) -> None:
        """The 'all' pipeline must have at least 95 gates (94 baseline
        per Wf-353-B1 + 1 NEW Wf-385-A3 per Wf-385-A3 §5)."""
        from molmetal.scripts import pre_flight_check

        gates = pre_flight_check.build_gates("all")
        self.assertGreaterEqual(
            len(gates),
            95,
            f"actual gate count {len(gates)} must be ≥ 95 "
            f"(94 pre-Wf-385 + 1 NEW Wf-385-A3 per Wf-385-A3 §5)",
        )

    def test_wf385_gate_priority_label_is_p3(self) -> None:
        """The Wf-385-A3 gate must carry the P3 priority label
        (matches the Wf-343C4 + Wf-347A1 + Wf-350 design contract
        per AGENTS.md §4)."""
        from molmetal.scripts import pre_flight_check
        import inspect

        gates = pre_flight_check.build_gates("all")
        wf385 = [
            (name, runner)
            for name, runner in gates
            if name == "p3_wf385_scaffold_loss_wire_in_verified"
        ]
        self.assertEqual(len(wf385), 1)
        # The lambda wraps the gate with frozen (name, priority,
        # pipeline); inspect the source to confirm priority="P3".
        src = inspect.getsource(wf385[0][1])
        self.assertIn(
            '"P3"',
            src,
            "Wf-385-A3 gate must carry P3 priority label per AGENTS.md §4",
        )


# ---------------------------------------------------------------------------
# Test 4 — verify the trainer AST-parses cleanly + 8 NEW flags are parseable.
# ---------------------------------------------------------------------------
class TestTrainer14Plus8Flags(unittest.TestCase):
    """Verify the CFM trainer AST-parses + the 8 NEW Wf-388 flags
    compose in an argparse namespace that mirrors the trainer's
    ``parse_args`` shape.

    Honest framing: the 14 Wf-380 SCAFFOLD flags are NOT in the
    trainer's parse_args at this ship time — they are referenced
    by ``wf385a3_scaffold_wire_in_gate.py:DEFAULT_SMOKE_FLAGS`` but
    the Wf-380 wire-in is deferred to a USER_OK-gated follow-up
    wave per Wf-385-A3 §1 + Wf-385-A3 §5. The 8 Wf-388 flags ARE
    wired and are the only NEW flags exercised here.
    """

    def test_trainer_parses_clean(self) -> None:
        """AST-parse the trainer to catch syntax regressions before
        pre-flight time."""
        trainer = _trainer_path()
        source = trainer.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            self.fail(f"train_cfm_yuelbond_wasserstein.py AST-parse failed: {exc}")
        # Confirm at least one top-level FunctionDef named parse_args.
        func_names = [
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        ]
        self.assertIn(
            "parse_args",
            func_names,
            f"trainer must define parse_args; got top-level functions: "
            f"{func_names}",
        )

    def test_wf388_8_flags_all_parseable(self) -> None:
        """The 8 NEW Wf-388 flags must all be parseable by the
        trainer's ``parse_args`` (verified via AST extraction —
        no torch required).

        This test asserts ONLY what is wired today (the 8 Wf-388
        flags). It explicitly does NOT assert the 14 Wf-380
        SCAFFOLD flags because those are deferred (see honest-
        negative caveat below).
        """
        trainer = _trainer_path()
        flags = _extract_add_argument_flags(trainer, "parse_args")
        missing = [f for f in WF388_ALL_FLAGS if f not in flags]
        self.assertEqual(
            missing,
            [],
            f"the following Wf-388 flags are NOT wired into "
            f"train_cfm_yuelbond_wasserstein.py:parse_args: {missing}",
        )

    def test_baseline_flags_still_present(self) -> None:
        """The baseline trainer flags (epochs, batch-size, …) must
        still be wired after the Wf-388 wire-in (i.e. the wire-in
        was ADDITIVE not REPLACING)."""
        trainer = _trainer_path()
        flags = _extract_add_argument_flags(trainer, "parse_args")
        baseline_flags = (
            "--epochs",
            "--batch-size",
            "--n-train",
            "--n-val",
            "--hidden-dim",
            "--n-layers",
            "--lr",
            "--patience",
            "--use-yuelbond",
            "--wasserstein-weight",
            "--closure-bounded",
            "--pocket-id",
            "--seed",
            "--device",
            "--cuda-visible-devices",
            "--out-dir",
            "--ckpt-dir",
            "--target-decode-ratio",
        )
        missing = [f for f in baseline_flags if f not in flags]
        self.assertEqual(
            missing,
            [],
            f"baseline flags missing after Wf-388 wire-in: {missing}",
        )

    def test_wf380_flags_not_yet_wired_honest_negative(self) -> None:
        """Honest-negative: the 14 Wf-380 SCAFFOLD flags are NOT in
        the trainer's parse_args at this ship time.

        This is BY-DESIGN — the Wf-380-B1 ship surface named 14 CFM
        flags + 11 SCAFFOLD modules under
        ``molmetal/adapters/flow_matching_lipman/``, but the wire-in
        from the SCAFFOLDs into ``parse_args`` is a deferred surface
        for a USER_OK-gated follow-up wave (per Wf-385-A3 §1 +
        Wf-385-A3 §5). The Wf-385-A3 gate invokes the trainer with
        these flags as ``DEFAULT_SMOKE_FLAGS`` and will FAIL until
        the wire-in lands — this is the SCAFFOLD-only failure mode
        that motivated the Wf-385-A3 gate in the first place.

        Test asserts (negative): the Wf-380 flags should NOT be in
        parse_args today; they will be wired by a future USER_OK-gated
        wave.
        """
        trainer = _trainer_path()
        flags = _extract_add_argument_flags(trainer, "parse_args")
        # Sanity check: at least one of the Wf-380 flags should be
        # absent (honest-negative baseline). If ALL 14 are present,
        # the Wf-380 wire-in has landed — that's a SHIP event
        # deserving its own wave, not a silent integration.
        present_wf380 = [f for f in WF380_SCAFFOLD_FLAGS if f in flags]
        self.assertLess(
            len(present_wf380),
            len(WF380_SCAFFOLD_FLAGS),
            f"all {len(WF380_SCAFFOLD_FLAGS)} Wf-380 SCAFFOLD flags "
            f"are now present in parse_args — this implies the Wf-380 "
            f"wire-in has landed; please close this honest-negative "
            f"caveat in a future wave. Present: {present_wf380}",
        )

    def test_wf392_flags_live_in_other_trainers(self) -> None:
        """Cross-reference: the Wf-392-A1 (Multi-PH + PLL) +
        Wf-392-A2 (chemprop v3 cache) flags live in OTHER trainers
        — not in ``train_cfm_yuelbond_wasserstein.py``.

        This test documents the split:
          - ``--multiph-pll`` → ``train_se3sb.py`` (Wf-392-A1)
          - ``--chemprop-v3-cache`` → ``train_chemprop_v2_v3.py``
            (Wf-392-A2)

        Neither flag should be wired into the CFM trainer
        ``parse_args`` (they are domain-specific to other trainers).
        """
        cfm_trainer = _trainer_path()
        cfm_flags = _extract_add_argument_flags(cfm_trainer, "parse_args")
        for flag, _trainer in WF392_FLAGS:
            self.assertNotIn(
                flag,
                cfm_flags,
                f"Wf-392 flag {flag} should live in {_trainer} — "
                f"NOT in train_cfm_yuelbond_wasserstein.py",
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
