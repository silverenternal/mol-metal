"""Pytest: Wf-35 Phase 1 — ``_common_train_args.py`` contract.

This module pins the contract of the centralised helpers that the
eight R18 / R19 trainer scripts will pick up in Wf-35 Phase 2:

* :func:`add_common_args`     — recipe-level flags.
* :func:`add_gpu_args`        — GPU env flags.
* :func:`add_benchmark_args`  — CV / benchmark flags.
* :func:`setup_gpu_env`       — env side effects.
* :func:`tag_metrics_payload` / :func:`write_tagged_metrics` — JSON tagging.
* :func:`MEASURED_vs_PROJECTED_decorator` — auto-tagging ``main()``.

All tests run on CPU and finish in <30 s.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "molmetal" / "scripts"
TESTS_DIR = PROJECT_ROOT / "molmetal" / "tests"
VENV_PY = (
    PROJECT_ROOT / "molmetal" / ".venvs" / "mol-metal-venv" / "bin" / "python"
)

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import _common_train_args as cta  # noqa: E402  — sys.path tweak above


# ===========================================================================
# 1. Module imports + constants exposed
# ===========================================================================
def test_module_imports_clean() -> None:
    """The module imports without torch / GPU dependencies."""
    assert hasattr(cta, "MEASURED_TAG")
    assert hasattr(cta, "PROJECTED_TAG")
    assert cta.MEASURED_TAG == "MEASURED"
    assert cta.PROJECTED_TAG == "PROJECTED"


def test_recipe_constants_match_train_se3sb_defaults() -> None:
    """RECIPE_* mirror the W2.1 / W2.2 / W2.3 trainer defaults."""
    assert cta.RECIPE_EPOCHS == 50
    assert cta.RECIPE_BATCH == 128
    assert cta.RECIPE_LR == 1e-3
    assert cta.RECIPE_SEEDS == (0, 1, 2, 3, 4)
    assert cta.RECIPE_HIDDEN == 300
    assert cta.RECIPE_DEPTH == 3
    assert cta.RECIPE_EVAL_EVERY == 5
    assert cta.RECIPE_DROPOUT == 0.2


def test_init_choices_cover_all_r18_adapters() -> None:
    """--init choices include the R17 tmqm/random + R18 W2.1-W2.3 attaches."""
    for required in ("tmqm", "random", "se3sb", "transport", "multiph"):
        assert required in cta.INIT_CHOICES, f"missing init choice {required!r}"
    assert cta.INIT_DEFAULT == "tmqm"


def test_mixed_precision_and_cv_choices() -> None:
    """--mixed-precision and --cv-strategy choices are stable."""
    assert cta.MIXED_PRECISION_CHOICES == ("bf16", "fp16", "none")
    assert cta.MIXED_PRECISION_DEFAULT == "bf16"
    assert cta.CV_STRATEGY_CHOICES == ("scaffold", "random", "temporal", "lomo")
    assert cta.CV_STRATEGY_DEFAULT == "scaffold"


# ===========================================================================
# 2. add_common_args — recipe-level flags
# ===========================================================================
def _fresh_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p = cta.add_common_args(p)
    p = cta.add_gpu_args(p)
    p = cta.add_benchmark_args(p)
    return p


def test_add_common_args_exposes_all_recipe_flags() -> None:
    """All standard recipe flags land on the parser with correct types."""
    p = _fresh_parser()
    args = p.parse_args([])
    # I/O
    assert isinstance(args.data, Path)
    assert isinstance(args.encoder_ckpt, Path)
    assert isinstance(args.out_dir, Path)
    assert isinstance(args.metrics_path, Path)
    assert isinstance(args.summary_path, Path)
    # Training recipe
    assert args.seeds == [0, 1, 2, 3, 4]
    assert args.epochs == 50
    assert args.batch == 128
    assert args.lr == 1e-3
    assert args.hidden == 300
    assert args.depth == 3
    assert args.eval_every == 5
    assert args.dropout == 0.2
    assert args.train_max_per_metal == 4000
    assert args.init == "tmqm"


def test_add_common_args_accepts_overrides() -> None:
    """CLI overrides flow through to the namespace."""
    p = _fresh_parser()
    args = p.parse_args([
        "--data", "/tmp/x.csv",
        "--epochs", "10",
        "--lr", "5e-4",
        "--seeds", "42", "7",
        "--init", "random",
    ])
    assert args.data == Path("/tmp/x.csv")
    assert args.epochs == 10
    assert args.lr == 5e-4
    assert args.seeds == [42, 7]
    assert args.init == "random"


def test_add_common_args_idempotent() -> None:
    """Calling ``add_common_args`` twice does NOT duplicate flags."""
    p = argparse.ArgumentParser()
    cta.add_common_args(p)
    # Second call must be a no-op (no SystemExit).
    cta.add_common_args(p)
    args = p.parse_args([])
    # We should still have *one* ``--epochs`` action.
    epoch_actions = [a for a in p._actions if a.dest == "epochs"]
    assert len(epoch_actions) == 1


def test_add_common_args_optional_includes() -> None:
    """``include_*`` flags actually drop flags."""
    p = argparse.ArgumentParser()
    cta.add_common_args(
        p,
        include_init=False,
        include_train_max_per_metal=False,
        include_eval_every=False,
        include_dropout=False,
    )
    args = p.parse_args([])
    for missing in ("init", "train_max_per_metal", "eval_every", "dropout"):
        assert not hasattr(args, missing), f"{missing} should be omitted"


# ===========================================================================
# 3. add_gpu_args — GPU env flags
# ===========================================================================
def test_add_gpu_args_exposes_env_flags() -> None:
    """GPU env flags exist with the right defaults.

    Wf-130 — ``--cuda-visible-devices`` defaults to ``"1"`` (RTX 5090
    on kanzi; GPU 0 is the VLM venv).  Pass ``--cuda-visible-devices=``
    (empty) to inherit the shell env.
    """
    p = _fresh_parser()
    args = p.parse_args([])
    assert args.cuda_visible_devices == "1"
    assert args.mixed_precision == "bf16"
    assert args.num_workers == 4


def test_add_gpu_args_accepts_overrides() -> None:
    """GPU env flag overrides flow through."""
    p = _fresh_parser()
    args = p.parse_args([
        "--cuda-visible-devices", "0,1",
        "--mixed-precision", "fp16",
        "--num-workers", "8",
    ])
    assert args.cuda_visible_devices == "0,1"
    assert args.mixed_precision == "fp16"
    assert args.num_workers == 8


# ===========================================================================
# 4. add_benchmark_args — CV flags
# ===========================================================================
def test_add_benchmark_args_exposes_cv_flags() -> None:
    """Benchmark / CV flags exist with the right defaults."""
    p = _fresh_parser()
    args = p.parse_args([])
    assert args.n_folds == 5
    assert args.cv_strategy == "scaffold"
    assert args.folds_seed == 0


def test_add_benchmark_args_accepts_overrides() -> None:
    """Benchmark flag overrides flow through."""
    p = _fresh_parser()
    args = p.parse_args([
        "--n-folds", "10",
        "--cv-strategy", "lomo",
        "--folds-seed", "123",
    ])
    assert args.n_folds == 10
    assert args.cv_strategy == "lomo"
    assert args.folds_seed == 123


# ===========================================================================
# 5. setup_gpu_env — side effects
# ===========================================================================
def test_setup_gpu_env_sets_cuda_visible_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    """``setup_gpu_env`` writes CUDA_VISIBLE_DEVICES when set."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    p = _fresh_parser()
    args = p.parse_args(["--cuda-visible-devices", "2,3"])
    state = cta.setup_gpu_env(args)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "2,3"
    assert state.cuda_visible_devices == "2,3"


def test_setup_gpu_env_no_op_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Empty ``--cuda-visible-devices`` (explicit ``--cuda-visible-devices=""``)
    is a no-op (does not write the env var).

    Wf-130 — the *default* ``--cuda-visible-devices`` is ``"1"``, so this
    test must pass the empty string explicitly to assert no-op behaviour.
    """
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    p = _fresh_parser()
    args = p.parse_args(["--cuda-visible-devices", ""])
    cta.setup_gpu_env(args)
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_setup_gpu_env_default_is_gpu_one(monkeypatch: pytest.MonkeyPatch) -> None:
    """Wf-130 — when the user does NOT pass ``--cuda-visible-devices``,
    :func:`setup_gpu_env` defaults to ``"1"`` (RTX 5090 on kanzi).
    """
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    p = _fresh_parser()
    args = p.parse_args([])  # no override
    cta.setup_gpu_env(args)
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "1"


def test_setup_gpu_env_seeds_torch() -> None:
    """``setup_gpu_env`` calls ``torch.manual_seed`` with the right seed."""
    import torch
    p = _fresh_parser()
    args = p.parse_args(["--seeds", "42", "7"])
    state = cta.setup_gpu_env(args)
    # The state reports seed == first --seeds entry.
    assert state.seed == 42
    # torch's initial seed was bumped; the exact value is implementation-defined,
    # but a subsequent manual_seed call should produce a deterministic advance.
    s0 = torch.initial_seed()
    torch.manual_seed(42)
    assert torch.initial_seed() == s0


def test_setup_gpu_env_returns_state_dataclass() -> None:
    """The returned ``GpuEnvState`` carries the env snapshot."""
    p = _fresh_parser()
    args = p.parse_args(["--cuda-visible-devices", "0", "--mixed-precision", "fp16"])
    state = cta.setup_gpu_env(args)
    assert isinstance(state, cta.GpuEnvState)
    assert state.mixed_precision == "fp16"
    assert state.num_workers == 4
    assert state.cuda_visible_devices == "0"


# ===========================================================================
# 6. tag_metrics_payload — JSON tagging
# ===========================================================================
def test_tag_metrics_payload_adds_provenance_measured() -> None:
    """``MEASURED_TAG`` lands in ``_provenance`` and payload is a copy."""
    payload = {"mean_test_auc": 0.85, "per_seed": [0.83, 0.84, 0.85, 0.86, 0.87]}
    out = cta.tag_metrics_payload(payload, provenance=cta.MEASURED_TAG)
    assert out["_provenance"] == "MEASURED"
    assert out["mean_test_auc"] == 0.85
    # The original is NOT mutated.
    assert "_provenance" not in payload


def test_tag_metrics_payload_adds_provenance_projected() -> None:
    """``PROJECTED_TAG`` lands in ``_provenance``."""
    payload = {"mean_test_auc": 0.5}
    out = cta.tag_metrics_payload(payload, provenance=cta.PROJECTED_TAG)
    assert out["_provenance"] == "PROJECTED"


def test_tag_metrics_payload_includes_gpu_env_when_provided() -> None:
    """``gpu_env`` is serialised under ``_gpu_env``."""
    state = cta.GpuEnvState(
        cuda_visible_devices="0",
        mixed_precision="bf16",
        num_workers=4,
        seed=42,
        deterministic=True,
    )
    payload = {"x": 1}
    out = cta.tag_metrics_payload(payload, gpu_env=state)
    assert out["_gpu_env"]["cuda_visible_devices"] == "0"
    assert out["_gpu_env"]["seed"] == 42
    assert out["_gpu_env"]["deterministic"] is True


def test_tag_metrics_payload_includes_extra_meta() -> None:
    """``extra_meta`` is exposed under ``_meta``."""
    payload = {"y": 2}
    out = cta.tag_metrics_payload(payload, extra_meta={"script": "train_x", "n_seeds": 5})
    assert out["_meta"]["script"] == "train_x"
    assert out["_meta"]["n_seeds"] == 5


def test_write_tagged_metrics_writes_json(tmp_path: Path) -> None:
    """``write_tagged_metrics`` writes a tagged JSON and returns the payload."""
    payload = {"mean_test_auc": 0.9, "per_seed": [0.88, 0.89, 0.90, 0.91, 0.92]}
    out_path = tmp_path / "metrics.json"
    tagged = cta.write_tagged_metrics(out_path, payload, provenance=cta.MEASURED_TAG)
    assert out_path.exists()
    loaded = json.loads(out_path.read_text())
    assert loaded["_provenance"] == "MEASURED"
    assert loaded["mean_test_auc"] == 0.9
    assert tagged["_provenance"] == "MEASURED"


def test_write_tagged_metrics_handles_nan(tmp_path: Path) -> None:
    """NaN floats serialise as ``null`` (matches the trainer scripts)."""
    payload = {"rmse": float("nan"), "ok": 1.0}
    out_path = tmp_path / "nan.json"
    cta.write_tagged_metrics(out_path, payload)
    loaded = json.loads(out_path.read_text())
    assert loaded["rmse"] is None
    assert loaded["ok"] == 1.0


# ===========================================================================
# 7. MEASURED_vs_PROJECTED_decorator — auto-tagging
# ===========================================================================
def test_decorator_tags_measured_when_ckpt_exists(tmp_path: Path) -> None:
    """When a ckpt is written, the JSON is tagged MEASURED."""
    out_dir = tmp_path / "ckpt"
    out_dir.mkdir()
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"x": 1}))

    # Simulate a "training run" by writing a non-empty ckpt.
    (out_dir / "seed_0.pt").write_bytes(b"\x00\x01")

    argv = [
        "--out-dir", str(out_dir),
        "--metrics-path", str(metrics_path),
        "--seeds", "0",
    ]

    @cta.MEASURED_vs_PROJECTED_decorator()
    def main(argv=None):
        p = _fresh_parser()
        p.parse_args(argv)
        return 0

    main(argv)
    loaded = json.loads(metrics_path.read_text())
    assert loaded["_provenance"] == "MEASURED"


def test_decorator_tags_projected_when_no_ckpt(tmp_path: Path) -> None:
    """When no ckpt is written, the JSON is tagged PROJECTED."""
    out_dir = tmp_path / "ckpt_empty"
    out_dir.mkdir()  # exists but no .pt inside
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"y": 2}))

    argv = [
        "--out-dir", str(out_dir),
        "--metrics-path", str(metrics_path),
        "--seeds", "0",
    ]

    @cta.MEASURED_vs_PROJECTED_decorator()
    def main(argv=None):
        p = _fresh_parser()
        p.parse_args(argv)
        return 0

    main(argv)
    loaded = json.loads(metrics_path.read_text())
    assert loaded["_provenance"] == "PROJECTED"


def test_decorator_preserves_existing_payload_keys(tmp_path: Path) -> None:
    """Tagging must not destroy any pre-existing JSON keys."""
    out_dir = tmp_path / "ckpt"
    out_dir.mkdir()
    (out_dir / "seed_0.pt").write_bytes(b"\x00")
    metrics_path = tmp_path / "metrics.json"
    payload = {"mean_test_auc": 0.85, "per_seed": [0.83, 0.84]}
    metrics_path.write_text(json.dumps(payload))

    argv = [
        "--out-dir", str(out_dir),
        "--metrics-path", str(metrics_path),
    ]

    @cta.MEASURED_vs_PROJECTED_decorator()
    def main(argv=None):
        p = _fresh_parser()
        p.parse_args(argv)
        return 0

    main(argv)
    loaded = json.loads(metrics_path.read_text())
    assert loaded["mean_test_auc"] == 0.85
    assert loaded["per_seed"] == [0.83, 0.84]
    assert loaded["_provenance"] == "MEASURED"


def test_decorator_swallows_tagging_failure(tmp_path: Path) -> None:
    """If argparse re-parse fails, the wrapped main() still returns."""
    metrics_path = tmp_path / "nonexistent_dir" / "metrics.json"
    # Don't write metrics_path; the decorator should silently skip tagging.

    @cta.MEASURED_vs_PROJECTED_decorator()
    def main(argv=None):
        return 42  # the wrapped return value

    argv = [
        "--out-dir", str(tmp_path),
        "--metrics-path", str(metrics_path),
    ]
    # Should NOT raise — metrics_path doesn't exist, so tagging is a no-op.
    result = main(argv)
    assert result == 42


def test_decorator_accepts_explicit_args_namespace(tmp_path: Path) -> None:
    """``main()`` returning ``(exit_code, args)`` skips argv re-parse."""
    out_dir = tmp_path / "ckpt"
    out_dir.mkdir()
    (out_dir / "seed_0.pt").write_bytes(b"\x00")
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"w": 4}))

    @cta.MEASURED_vs_PROJECTED_decorator()
    def main(argv=None):
        p = _fresh_parser()
        ns = p.parse_args([
            "--out-dir", str(out_dir),
            "--metrics-path", str(metrics_path),
            "--seeds", "0",
        ])
        return (0, ns)

    exit_code = main(None)
    assert exit_code == 0
    loaded = json.loads(metrics_path.read_text())
    assert loaded["_provenance"] == "MEASURED"


def test_decorator_policy_disable_ckpt_check(tmp_path: Path) -> None:
    """``measured_when_ckpt_exists=False`` always → PROJECTED."""
    out_dir = tmp_path / "ckpt"
    out_dir.mkdir()
    (out_dir / "seed_0.pt").write_bytes(b"\x00")
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"z": 3}))

    policy = cta.DecoratorPolicy(measured_when_ckpt_exists=False)
    argv = [
        "--out-dir", str(out_dir),
        "--metrics-path", str(metrics_path),
    ]

    @cta.MEASURED_vs_PROJECTED_decorator(policy=policy)
    def main(argv=None):
        p = _fresh_parser()
        p.parse_args(argv)
        return 0

    main(argv)
    loaded = json.loads(metrics_path.read_text())
    assert loaded["_provenance"] == "PROJECTED"


# ===========================================================================
# 8. CLI integration — the module's argparse output is well-formed
# ===========================================================================
def test_help_exits_zero_subprocess() -> None:
    """A trivial script that uses the helpers exits 0 on --help."""
    if not VENV_PY.exists():
        pytest.skip(f"venv python not at {VENV_PY}")

    probe = SCRIPTS_DIR / "_cta_probe.py"
    if probe.exists():
        probe.unlink()
    probe.write_text(
        "import argparse, sys\n"
        "sys.path.insert(0, repr(" + repr(str(SCRIPTS_DIR)) + ")[1:-1])\n"
        "from _common_train_args import add_common_args, add_gpu_args, add_benchmark_args\n"
        "p = argparse.ArgumentParser()\n"
        "p = add_common_args(p)\n"
        "p = add_gpu_args(p)\n"
        "p = add_benchmark_args(p)\n"
        "p.parse_args()\n"
    )
    try:
        result = subprocess.run(
            [str(VENV_PY), str(probe), "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"--help exited {result.returncode}\n"
            f"stderr:\n{result.stderr}"
        )
        # Sanity: each arg group renders in the help output.
        for section in ("Common recipe", "GPU env", "Benchmark / CV"):
            assert section in result.stdout, f"missing group {section!r} in --help"
    finally:
        if probe.exists():
            probe.unlink()


def test_module_help_lists_documented_flags() -> None:
    """``--help`` lists every documented CLI flag by name."""
    if not VENV_PY.exists():
        pytest.skip(f"venv python not at {VENV_PY}")

    probe = SCRIPTS_DIR / "_cta_probe2.py"
    if probe.exists():
        probe.unlink()
    probe.write_text(
        "import argparse, sys\n"
        f"sys.path.insert(0, {str(SCRIPTS_DIR)!r})\n"
        "from _common_train_args import (\n"
        "    add_common_args, add_gpu_args, add_benchmark_args,\n"
        ")\n"
        "p = argparse.ArgumentParser()\n"
        "p = add_common_args(p)\n"
        "p = add_gpu_args(p)\n"
        "p = add_benchmark_args(p)\n"
        "p.print_help()\n"
    )
    try:
        result = subprocess.run(
            [str(VENV_PY), str(probe)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        for flag in (
            "--data", "--seeds", "--epochs", "--batch", "--lr",
            "--hidden", "--depth", "--init",
            "--cuda-visible-devices", "--mixed-precision", "--num-workers",
            "--n-folds", "--cv-strategy",
        ):
            assert flag in result.stdout, f"{flag} missing from --help output"
    finally:
        if probe.exists():
            probe.unlink()
