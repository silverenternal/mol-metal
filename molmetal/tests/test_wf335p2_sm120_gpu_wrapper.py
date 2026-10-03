"""Wf-335-P2 — Tests for the sm_120 Triton retune GPU wrapper.

================================================================
Two test classes (CPU-only):

  * :class:`TestWrapperInitialization` — Verifies :func:`build_launch_plan`
    composes a :class:`LaunchPlan` from the Wf-326E audit + the Wf-333-T1
    per-kernel presets + the Wf-331-X3 generic grid; verifies the
    ~3x Wf-326E-B speedup projection; verifies unknown kernel names
    raise :class:`ValueError`.

  * :class:`TestWrapperCliArgs` — Verifies :func:`main` handles all
    4 documented CLI flag combinations (``--smoke``,
    ``--dump-launch-plan-only``, ``--per-kernel``,
    ``--kernels <csv>``) and refuses ``--launch`` without
    ``--launch-prereqs-confirmed``.

Honest framing (per AGENTS.md §5.2):

  * The wrapper itself is CPU-only — no GPU launch occurs in any
    test.
  * The launch-gate refusal is verified (gate_passed=False), NOT
    bypassed — the wrapper would only accept ``--launch`` AFTER the
    operator manually runs the 3 prereqs (VRAM + pre-flight +
    config-aligned).
  * Full GPU retune remains DEFERRED per AGENTS.md §5.1.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "molmetal" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import wf335p2_sm120_triton_retune_gpu as _w  # noqa: E402


# ---------------------------------------------------------------------------
# Test A: wrapper initialization (CPU-only)
# ---------------------------------------------------------------------------
class TestWrapperInitialization:
    """Verify ``build_launch_plan`` composes the plan from audit + presets."""

    def test_01_default_plan_uses_24_kernels_and_per_kernel_presets(self):
        """Default plan covers all 24 audit kernels + per-kernel grid."""
        plan = _w.build_launch_plan(use_per_kernel=True)
        assert plan.audit_n_kernels == 24
        assert len(plan.selected_kernels) == 24
        assert plan.use_per_kernel is True
        assert plan.gpu_retune_deferred is True
        assert len(plan.prereqs) == 3

    def test_02_per_kernel_total_configs_is_117(self):
        """Per-kernel plan sweeps 117 configs (5*3 + 13*6 + 3*6 + 3*2)."""
        plan = _w.build_launch_plan(use_per_kernel=True)
        assert plan.total_configs_to_sweep == 117
        assert plan.estimated_kernel_launches == 117 * 5

    def test_03_generic_plan_sweeps_24_times_12_configs(self):
        """Generic plan uses Wf-331-X3 grid for every kernel."""
        plan = _w.build_launch_plan(use_per_kernel=False)
        assert plan.total_configs_to_sweep == 24 * 12
        assert plan.estimated_kernel_launches == 288 * 5

    def test_04_per_kernel_strictly_smaller_than_generic_with_3x_band(self):
        """Per-kernel / generic ratio is in the Wf-326E-B [2.0, 3.5] band."""
        plan_per = _w.build_launch_plan(use_per_kernel=True)
        plan_gen = _w.build_launch_plan(use_per_kernel=False)
        ratio = plan_gen.total_configs_to_sweep / plan_per.total_configs_to_sweep
        assert plan_per.total_configs_to_sweep < plan_gen.total_configs_to_sweep
        assert 2.0 <= ratio <= 3.5

    def test_05_kernel_subset_preserves_audit_order(self):
        """Subset plan keeps audit order (not random)."""
        subset = ("matmul_fn", "fused_silu_mlp")  # out of alphabetical order
        plan = _w.build_launch_plan(use_per_kernel=True, kernels=subset)
        assert plan.selected_kernels == (
            "fused_silu_mlp",
            "matmul_fn",
        ), f"Subset plan should preserve audit order; got " f"{plan.selected_kernels}."

    def test_06_unknown_kernel_raises_value_error(self):
        """Unknown kernel name raises ValueError (not silent skip)."""
        with pytest.raises(ValueError, match="Unknown kernel"):
            _w.build_launch_plan(
                use_per_kernel=True, kernels=("not_a_real_kernel_xyz",)
            )

    def test_07_every_per_kernel_config_is_sm120_safe(self):
        """Every per-kernel (num_warps, num_stages) is in the Wf-331-X3 grid."""
        plan = _w.build_launch_plan(use_per_kernel=True)
        import wf331x3_sm120_retune as _x3

        for name, pairs in plan.per_kernel_configs.items():
            for nw, ns in pairs:
                cfg = {"num_warps": nw, "num_stages": ns}
                assert _x3.is_sm120_config_safe(cfg), (
                    f"Plan config ({nw}, {ns}) for kernel {name!r} "
                    f"is NOT sm_120-safe."
                )

    def test_08_plan_dump_emits_metadata_and_per_kernel_configs(self, tmp_path: Path):
        """dump_launch_plan_json writes a JSON file with expected fields."""
        plan = _w.build_launch_plan(use_per_kernel=True)
        out = tmp_path / "plan.json"
        result = _w.dump_launch_plan_json(plan, out)
        assert result["output_path"] == str(out)
        assert result["total_configs_to_sweep"] == 117
        assert result["n_selected_kernels"] == 24
        body = json.loads(out.read_text())
        assert body["plan"]["use_per_kernel"] is True
        assert body["plan"]["gpu_retune_deferred"] is True
        assert body["metadata"]["extends"] == [
            "wf331x3_sm120_retune.py",
            "wf333t1_sm120_autotune_extended.py",
        ]
        assert "fused_silu_mlp" in body["plan"]["per_kernel_configs"]

    def test_09_launch_gate_always_refuses_with_3_prereqs(self):
        """launch_gate_check returns gate_passed=False (CPU stub)."""
        gate = _w.launch_gate_check()
        assert gate["gate_passed"] is False
        assert len(gate["prereqs_required"]) == 3
        for prereq in gate["prereqs_required"]:
            assert isinstance(prereq, str)
            assert len(prereq) > 20  # non-trivial prereq description


# ---------------------------------------------------------------------------
# Test B: wrapper CLI args (CPU-only)
# ---------------------------------------------------------------------------
class TestWrapperCliArgs:
    """Verify ``main()`` handles all 4 documented CLI flag combinations."""

    def test_10_smoke_flag_returns_zero_on_pass(self, tmp_path: Path, capsys):
        """--smoke flag returns 0 and emits the smoke JSON."""
        rc = _w.main(["--smoke", "--output-dir", str(tmp_path)])
        assert rc == 0
        out = capsys.readouterr().out
        body = json.loads(out)
        assert body["smoke_passed"] is True
        assert body["per_kernel_total_configs"] == 117
        assert body["generic_total_configs"] == 288

    def test_11_dump_launch_plan_only_emits_per_kernel_plan(
        self, tmp_path: Path, capsys
    ):
        """--dump-launch-plan-only emits the per-kernel plan JSON."""
        rc = _w.main(
            [
                "--dump-launch-plan-only",
                "--output-dir",
                str(tmp_path),
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        body = json.loads(out)
        assert body["use_per_kernel"] is True
        assert body["gpu_retune_deferred"] is True
        assert body["total_configs_to_sweep"] == 117
        assert "launch_plan_per_kernel.json" in body["launch_plan_path"]
        # 3 prereqs documented.
        assert len(body["prereqs_required"]) == 3

    def test_12_per_kernel_flag_is_default(self, tmp_path: Path, capsys):
        """--per-kernel flag works; --kernels subset restricts the plan."""
        rc = _w.main(
            [
                "--per-kernel",
                "--kernels",
                "fused_silu_mlp,matmul_fn",
                "--dump-launch-plan-only",
                "--output-dir",
                str(tmp_path),
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        body = json.loads(out)
        assert body["use_per_kernel"] is True
        assert body["n_selected_kernels"] == 2
        # 6 (fused_silu_mlp compute_bound) + 6 (matmul_fn compute_bound) = 12.
        assert body["total_configs_to_sweep"] == 12

    def test_13_launch_without_prereq_confirmed_refuses_with_exit_2(
        self,
        tmp_path: Path,
        capsys,
    ):
        """--launch without --launch-prereqs-confirmed exits 2 + gate JSON."""
        rc = _w.main(
            [
                "--launch",
                "--output-dir",
                str(tmp_path),
            ]
        )
        assert rc == 2
        err = capsys.readouterr().err
        gate = json.loads(err)
        assert gate["gate_passed"] is False
        assert len(gate["prereqs_required"]) == 3

    def test_14_launch_with_prereq_confirmed_emits_launch_accepted(
        self,
        tmp_path: Path,
        capsys,
    ):
        """--launch + --launch-prereqs-confirmed exits 0 with launch_accepted=True."""
        rc = _w.main(
            [
                "--launch",
                "--launch-prereqs-confirmed",
                "--output-dir",
                str(tmp_path),
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        body = json.loads(out)
        assert body["launch_accepted"] is True
        assert body["prereqs_acknowledged"] is True
        # Note explicitly defers the actual retune execution to a follow-up.
        assert "deferred" in body["note"].lower() or "follow-up" in body["note"].lower()

    def test_15_unknown_kernel_via_cli_returns_exit_1(self, tmp_path: Path, capsys):
        """--kernels with unknown name returns 1 (CLI-side validation)."""
        rc = _w.main(
            [
                "--kernels",
                "not_a_real_kernel_xyz",
                "--dump-launch-plan-only",
                "--output-dir",
                str(tmp_path),
            ]
        )
        assert rc == 1
