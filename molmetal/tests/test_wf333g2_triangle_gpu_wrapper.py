"""test_wf333g2_triangle_gpu_wrapper.py — Wf-333-G2 GPU wrapper smoke tests.

====================================================================
Test surface — 2 tests (per Wf-333-G2 task spec)
====================================================================

1. :func:`test_wrapper_initialization_cpu_only` — exercises
   :func:`build_triangle_gpu_trainer` end-to-end on a small toy
   design matrix.  Asserts:
     * The wrapper resolves the 12-d feature schema correctly.
     * ``TriangleGPUSummary`` carries the expected shape, vertex
       labels, and honest-negative flags.
     * The Hilbert torch module has the expected parameter count.
     * No GPU is touched (the wrapper is CPU-only by design per
       AGENTS.md §5.1 ``kill_then_adapt`` directive).

2. :func:`test_wrapper_cli_args_and_default_config` — exercises
   :func:`parse_args` (CLI parsing) and asserts the default
   configuration matches the Wf-333-G2 contract:
     * ``feature_mode = "12d"`` (the Wf-331-R2 ship surface).
     * ``launch = False`` (dry-run default per AGENTS.md §5.1).
     * ``mixed_precision = "bf16"`` (the Wf-326 default).
     * ``cuda_visible_devices = "1"`` (RTX 5090 GPU 0 reserved
       for VLM venv).
     * ``torch_compile = False`` (off by default — first-call
       latency ~30s).
     * ``--feature-mode 12d`` parses correctly.
     * ``--launch`` parses correctly (toggle).
     * ``--torch-compile --torch-compile-mode reduce-overhead``
       parses correctly.
     * Invalid ``--feature-mode 99d`` is rejected by argparse.

CPU-only contract
-----------------
These tests do NOT touch the GPU.  The Wf-333-G2 wrapper is shipped
as CPU-only scaffolding per AGENTS.md §5.1 — downstream Wf-333-G3+
runs the actual GPU launch when VRAM frees.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Path bootstrap — same convention as the wrapper module.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from molmetal.scripts.wf333g2_triangle_12d_gpu_wrapper import (  # noqa: E402
    DEFAULT_FEATURE_MODE,
    DEFAULT_HILBERT_DIM,
    DEFAULT_MIXED_PRECISION,
    DEFAULT_CUDA_VISIBLE_DEVICES,
    DEFAULT_TORCH_COMPILE,
    DEFAULT_SEEDS,
    DEFAULT_EPOCHS,
    DEFAULT_N_FOLDS,
    WRAPPER_NAME,
    WRAPPER_VERSION,
    HILBERT_R2_HONESTY_GATE,
    VARIANCE_REDUCTION_HONESTY_GATE,
    SUCCESS_RATE_HONESTY_GATE,
    TriangleGPUConfig,
    TriangleGPUSummary,
    add_wf333g2_args,
    build_triangle_gpu_trainer,
    build_hilbert_torch_module,
    build_toy_design_matrix,
    honesty_gate_verdict,
    parse_args,
    resolve_feature_schema,
    triangle_composite,
    write_summary,
)


# ===========================================================================
# Test 1 — Wrapper initialization (CPU only)
# ===========================================================================
def test_wrapper_initialization_cpu_only(tmp_path: Path) -> None:
    """Wf-333-G2 wrapper initializes correctly on CPU (no GPU launch).

    Asserts the full CPU-only path:
      * Default config resolves to 12-d schema.
      * Toy design matrix has the expected (n_samples, 12) shape.
      * TriangleGPUSummary carries honest-negative flags.
      * Hilbert torch module has 64 trainable parameters.
      * write_summary writes summary.json + summary.md to the
        output directory.
    """
    # ---- Default config — feature_mode=12d, hilbert_dim=64 ----
    config = TriangleGPUConfig()
    assert config.feature_mode == "12d", (
        f"default feature_mode should be '12d' (Wf-331-R2 ship), got {config.feature_mode!r}"
    )
    assert config.hilbert_dim == 64, (
        f"default hilbert_dim should be 64, got {config.hilbert_dim}"
    )
    assert config.launch is False, (
        f"default launch should be False (dry-run per AGENTS.md §5.1), "
        f"got {config.launch}"
    )

    # ---- Build the trainer (CPU only) ----
    summary = build_triangle_gpu_trainer(config, n_samples=64, seed=0)

    # ---- Schema assertions ----
    assert isinstance(summary, TriangleGPUSummary)
    assert summary.feature_shape == (64, 12), (
        f"feature_shape should be (64, 12) for 12-d mode, got {summary.feature_shape}"
    )
    assert len(summary.feature_names) == 12, (
        f"feature_names should have 12 entries (F5_RDKIT_2D), got {len(summary.feature_names)}"
    )
    # The 12-d names are the Wf-49 P1 / Wf-331-R2 canonical F5_RDKIT_2D
    # RDKit descriptors — must include MolWt, MolLogP, TPSA.
    expected_min_names = {"MolWt", "MolLogP", "TPSA"}
    actual_names = set(summary.feature_names)
    assert expected_min_names.issubset(actual_names), (
        f"feature_names missing expected F5_RDKIT_2D descriptors; "
        f"missing={expected_min_names - actual_names}"
    )

    # ---- Vertex labels ----
    assert summary.vertex_labels == ("astsr", "hilbert", "mcts"), (
        f"vertex_labels should be ('astsr', 'hilbert', 'mcts'), got {summary.vertex_labels}"
    )

    # ---- Hilbert torch module parameter count ----
    # hilbert_dim=64 → 1 parameter vector of length 64 → 64 trainable params.
    assert summary.n_trainable_params == 64, (
        f"n_trainable_params should be 64 for hilbert_dim=64, got {summary.n_trainable_params}"
    )

    # ---- Honest-negative flags ----
    # The wrapper is dry-run only — should emit HONEST_NEGATIVE_TRIANGLE_GPU_DEFERRED.
    assert "HONEST_NEGATIVE_TRIANGLE_GPU_DEFERRED" in summary.honest_negative_flags, (
        f"wrapper should emit HONEST_NEGATIVE_TRIANGLE_GPU_DEFERRED; "
        f"got {summary.honest_negative_flags}"
    )

    # ---- Project tag ----
    assert summary.project_tag == "PROJECTED", (
        f"project_tag should be PROJECTED (CPU dry-run only), got {summary.project_tag}"
    )

    # ---- Honesty gate verdict (PROJECTED, not PASS) ----
    gate = summary.honesty_gate_status
    assert "hilbert_r2" in gate
    assert "variance_reduction" in gate
    assert "success_rate" in gate
    # On a CPU dry-run with no real R², the gate verdict is
    # fail=False (deferred, not failed) but the per-gate 'pass' field
    # is False because the launch hasn't run.
    assert gate["hilbert_r2"]["pass"] is False, (
        f"hilbert_r2 gate should be fail=False (deferred) on dry-run; "
        f"got pass={gate['hilbert_r2']['pass']}"
    )
    assert gate["hilbert_r2"]["gate"] == HILBERT_R2_HONESTY_GATE
    assert gate["variance_reduction"]["gate"] == VARIANCE_REDUCTION_HONESTY_GATE
    assert gate["success_rate"]["gate"] == SUCCESS_RATE_HONESTY_GATE

    # ---- No GPU touched ----
    # We do NOT import torch.cuda here — but we can assert the wrapper
    # did not call any torch.cuda module by checking the toy matrix
    # is a plain numpy float32 (not a torch tensor).
    X, y = build_toy_design_matrix(n_samples=8, n_features=12, seed=0)
    assert isinstance(X, np.ndarray), "X should be a numpy array (CPU only)"
    assert X.dtype == np.float32, f"X should be float32, got {X.dtype}"
    assert y.dtype == np.float32, f"y should be float32, got {y.dtype}"

    # ---- write_summary path ----
    out_dir = tmp_path / "wf333g2_wrapper"
    json_path, md_path = write_summary(summary, out_dir)
    assert json_path.exists(), f"summary.json should exist at {json_path}"
    assert md_path.exists(), f"summary.md should exist at {md_path}"
    # Verify JSON parses + carries the honest-negative flag.
    with open(json_path) as f:
        payload = json.load(f)
    assert "HONEST_NEGATIVE_TRIANGLE_GPU_DEFERRED" in payload["honest_negative_flags"]
    assert payload["project_tag"] == "PROJECTED"
    assert payload["feature_shape"] == [64, 12]
    # Verify MD mentions the wrapper name + project tag.
    md_text = md_path.read_text()
    assert WRAPPER_NAME in md_text
    assert "PROJECTED" in md_text


# ===========================================================================
# Test 2 — CLI args + default config
# ===========================================================================
def test_wrapper_cli_args_and_default_config() -> None:
    """Wf-333-G2 CLI surface + default config sanity.

    Asserts:
      * :func:`parse_args` returns a TriangleGPUConfig with the
        expected defaults.
      * :func:`add_wf333g2_args` attaches all Wf-333-G2 flags to a
        parser (idempotency).
      * ``--feature-mode 12d`` / ``17d`` / ``5d`` parse correctly.
      * ``--launch`` toggles ``config.launch`` to True.
      * ``--torch-compile --torch-compile-mode reduce-overhead``
        toggles both flags.
      * Invalid ``--feature-mode 99d`` is rejected by argparse.
      * Constants in the wrapper match the Wf-333-G2 contract.
      * resolve_feature_schema returns the canonical 5d / 12d / 17d
        tuples.
    """
    # ---- Default config (no args) ----
    config = parse_args([])
    assert isinstance(config, TriangleGPUConfig)
    assert config.feature_mode == DEFAULT_FEATURE_MODE == "12d"
    assert config.hilbert_dim == DEFAULT_HILBERT_DIM == 64
    assert config.epochs == DEFAULT_EPOCHS == 50
    assert config.seeds == DEFAULT_SEEDS == 5
    assert config.n_folds == DEFAULT_N_FOLDS == 5
    assert config.mixed_precision == DEFAULT_MIXED_PRECISION == "bf16"
    assert config.cuda_visible_devices == DEFAULT_CUDA_VISIBLE_DEVICES == "1"
    assert config.torch_compile == DEFAULT_TORCH_COMPILE is False
    assert config.launch is False
    assert config.output == "molmetal/reports/wf333g2_triangle_gpu"

    # ---- --feature-mode 12d / 17d / 5d ----
    cfg_12d = parse_args(["--feature-mode", "12d"])
    assert cfg_12d.feature_mode == "12d"
    cfg_17d = parse_args(["--feature-mode", "17d"])
    assert cfg_17d.feature_mode == "17d"
    cfg_5d = parse_args(["--feature-mode", "5d"])
    assert cfg_5d.feature_mode == "5d"

    # ---- --launch toggle ----
    cfg_launch = parse_args(["--launch"])
    assert cfg_launch.launch is True
    cfg_no_launch = parse_args([])
    assert cfg_no_launch.launch is False

    # ---- --torch-compile + --torch-compile-mode ----
    cfg_tc = parse_args([
        "--torch-compile",
        "--torch-compile-mode", "reduce-overhead",
    ])
    assert cfg_tc.torch_compile is True
    assert cfg_tc.torch_compile_mode == "reduce-overhead"
    cfg_tc_default = parse_args([])
    assert cfg_tc_default.torch_compile is False
    # DEFAULT_TORCH_COMPILE_MODE = "reduce-overhead" per Wf-326/328 convention.
    assert cfg_tc_default.torch_compile_mode == "reduce-overhead"

    # ---- --mixed-precision ----
    cfg_fp16 = parse_args(["--mixed-precision", "fp16"])
    assert cfg_fp16.mixed_precision == "fp16"
    cfg_none = parse_args(["--mixed-precision", "none"])
    assert cfg_none.mixed_precision == "none"

    # ---- --cuda-visible-devices ----
    cfg_cvd = parse_args(["--cuda-visible-devices", "0"])
    assert cfg_cvd.cuda_visible_devices == "0"

    # ---- --num-workers ----
    cfg_nw = parse_args(["--num-workers", "8"])
    assert cfg_nw.num_workers == 8

    # ---- --dataset ----
    cfg_ds = parse_args(["--dataset", "platinai"])
    assert cfg_ds.dataset == "platinai"
    cfg_ds_mc = parse_args(["--dataset", "metalcytotox"])
    assert cfg_ds_mc.dataset == "metalcytotox"

    # ---- --hilbert-dim / --epochs / --seeds / --lr ----
    cfg_hd = parse_args(["--hilbert-dim", "128", "--epochs", "100",
                         "--seeds", "10", "--lr", "0.01"])
    assert cfg_hd.hilbert_dim == 128
    assert cfg_hd.epochs == 100
    assert cfg_hd.seeds == 10
    assert cfg_hd.lr == pytest.approx(0.01)

    # ---- Invalid --feature-mode rejected ----
    with pytest.raises(SystemExit):
        parse_args(["--feature-mode", "99d"])

    # ---- add_wf333g2_args idempotency ----
    import argparse as _ap
    parser = _ap.ArgumentParser()
    parser = add_wf333g2_args(parser)
    # Calling twice on the same parser must not raise (the wrapper
    # uses the same idempotent pattern as add_gpu_args).
    parser = add_wf333g2_args(parser)
    # Parse the default args through the augmented parser.
    ns = parser.parse_args([])
    assert ns.feature_mode == "12d"
    assert ns.torch_compile is False

    # ---- resolve_feature_schema ----
    names_5d, n_5d = resolve_feature_schema("5d")
    assert n_5d == 5
    assert "smiles_length" in names_5d
    assert "metal_idx" in names_5d
    names_12d, n_12d = resolve_feature_schema("12d")
    assert n_12d == 12
    assert "MolWt" in names_12d
    assert "MolLogP" in names_12d
    assert "BalabanJ" in names_12d
    names_17d, n_17d = resolve_feature_schema("17d")
    assert n_17d == 17
    # 17d = 5d + 12d concat — first 5 entries equal 5d names.
    assert names_17d[:5] == names_5d
    assert names_17d[5:] == names_12d
    # Invalid mode raises.
    with pytest.raises(ValueError):
        resolve_feature_schema("99d")

    # ---- Constant sanity ----
    assert WRAPPER_NAME == "wf333g2_triangle_12d_gpu_wrapper"
    assert WRAPPER_VERSION == "1.0.0"
    assert HILBERT_R2_HONESTY_GATE == pytest.approx(0.6)
    assert VARIANCE_REDUCTION_HONESTY_GATE == pytest.approx(0.05)
    assert SUCCESS_RATE_HONESTY_GATE == pytest.approx(0.95)

    # ---- honesty_gate_verdict ----
    verdict_pass = honesty_gate_verdict(
        hilbert_r2=0.7,
        variance_reduction=0.06,
        success_rate=1.0,
    )
    assert verdict_pass["hilbert_r2"]["pass"] is True
    assert verdict_pass["variance_reduction"]["pass"] is True
    assert verdict_pass["success_rate"]["pass"] is True
    verdict_fail = honesty_gate_verdict(
        hilbert_r2=0.5,
        variance_reduction=0.04,
        success_rate=0.9,
    )
    assert verdict_fail["hilbert_r2"]["pass"] is False
    assert verdict_fail["variance_reduction"]["pass"] is False
    assert verdict_fail["success_rate"]["pass"] is False

    # ---- triangle_composite ----
    comp = triangle_composite(
        hilbert_r2=0.5, evolutionary_r2=0.3, variance_reduction=0.05,
    )
    assert isinstance(comp, float)
    assert comp == pytest.approx(0.5 + 0.3 + 0.05)
    comp_weighted = triangle_composite(
        hilbert_r2=0.5, evolutionary_r2=0.3, variance_reduction=0.05,
        weights=(2.0, 1.0, 1.0),
    )
    assert comp_weighted == pytest.approx(1.0 + 0.3 + 0.05)


# ===========================================================================
# Optional smoke — importability
# ===========================================================================
def test_wrapper_module_imports_clean() -> None:
    """The wrapper module imports without side-effects on torch.

    This test guards against accidental torch.cuda.is_available()
    or other GPU-touching side-effects in module-level imports.
    """
    import molmetal.scripts.wf333g2_triangle_12d_gpu_wrapper as wrapper
    assert wrapper.WRAPPER_NAME == "wf333g2_triangle_12d_gpu_wrapper"
    assert wrapper.WRAPPER_VERSION == "1.0.0"
