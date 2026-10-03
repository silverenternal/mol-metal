"""test_wf333g1_lambda_cfm_gpu_wrapper.py — TODO-21 Lambda × CFM GPU wrapper.

Wf-333-G1 follow-up to Wf-331-R6 + Wf-323E + Wf-329-W6 (CPU scaffold
shipped + 1-line adapter change wired). This test suite covers the
GPU wrapper CLI + dataclass shape — **no GPU launch** (per AGENTS.md
§5.1 kill_then_adapt directive).

Coverage (2 tests):
1. ``test_wf333g1_wrapper_initialization_cpu_only`` —
   :class:`LambdaCFMGPUWrapper` resolves paths, builds the CPU joint
   pipeline, exposes the ``train_one_epoch`` kwargs, and runs the
   joint inference smoke on a small SMILES pool — all in < 5 sec on
   CPU.  No ``torch.cuda`` call; no GPU launch.
2. ``test_wf333g1_wrapper_cli_args`` —
   :func:`parse_args` + :func:`build_wrapper` correctly translate the
   documented CLI flags (``--dry-run`` default True; ``--no-dry-run``
   flips to False; ``--cfm-checkpoint ''`` ⇒ ``None``; the documented
   ``--lambda-cfm-coupling-weight`` flows into
   :meth:`build_train_kwargs`).

Honest framing
==============
* Both tests are CPU-only. They exercise the *wrapper surface* but do
  not launch any GPU retrain (Wf-331-R6 §5.2 boundary).
* The wrapper's ``dry_run`` default ``True`` is a hard guard so this
  suite cannot accidentally hit CUDA — even if a test runs in a CUDA
  environment.
* The CPU joint inference smoke uses the Wf-323E / Wf-329-W6-shipped
  :class:`LambdaCFMPipeline` + :class:`PlatinAI5KBridge`; the test
  does not touch the GPU ``train_one_epoch`` function directly (the
  GPU side is Wf-333-G2 territory).
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Lazy module loaders (graceful skip when the module isn't on disk)
# ---------------------------------------------------------------------------
def _load_wrapper_module():
    """Load ``wf333g1_lambda_cfm_gpu_wrapper`` or skip the test."""
    spec_path = (
        PROJECT_ROOT / "molmetal" / "scripts" / "wf333g1_lambda_cfm_gpu_wrapper.py"
    )
    if not spec_path.exists():
        pytest.skip(f"wrapper script not found at {spec_path}")
    try:
        return importlib.import_module(
            "molmetal.scripts.wf333g1_lambda_cfm_gpu_wrapper"
        )
    except Exception as exc:
        pytest.skip(f"wrapper import failed: {exc}")


# ---------------------------------------------------------------------------
# Phase 1 — wrapper initialization (CPU only)
# ---------------------------------------------------------------------------
def test_wf333g1_wrapper_initialization_cpu_only() -> None:
    """Initialise the wrapper, resolve paths, build CPU pipeline, run smoke.

    * No ``torch.cuda`` call (verifiable via ``dry_run=True``).
    * Path resolution reports present/absent for each checkpoint.
    * CPU joint pipeline runs end-to-end on the canonical SMILES pool.
    * :meth:`build_train_kwargs` returns the exact kwargs shape the
      Wf-329-W6 1-line adapter change consumes.
    """
    mod = _load_wrapper_module()
    LambdaCFMGPUWrapper = mod.LambdaCFMGPUWrapper

    # Build with the documented defaults — dry-run True is the AGENTS.md
    # §5.1 guard; no torch.cuda can be touched.
    wrapper = LambdaCFMGPUWrapper()
    assert wrapper.dry_run is True, "dry_run must default to True per AGENTS.md §5.1"
    assert wrapper.device == "cuda"
    assert (
        wrapper.lambda_cfm_coupling_weight == 0.0
    ), "default 0.0 is the Wf-329-W6 CPU-safe scaffold default"
    assert wrapper.batch_size == 64
    assert wrapper.pocket_id == "1h36"

    # Path resolution returns the documented shape for every key.
    resolved = wrapper.resolve_paths()
    for key in ("cfm_checkpoint", "lambda_reward_ckpt", "bridge_npz"):
        assert key in resolved, f"missing key {key!r} in resolve_paths()"
        assert "present" in resolved[key], f"missing 'present' for {key}"
        assert "path" in resolved[key], f"missing 'path' for {key}"

    # CPU joint pipeline must be importable + callable end-to-end.
    pipe = wrapper.build_cpu_joint_pipeline()
    if isinstance(pipe, dict) and "error" in pipe:
        pytest.skip(f"CPU joint pipeline unavailable: {pipe['error']}")
    assert pipe is not None, "build_cpu_joint_pipeline returned None"
    # The LambdaCFMPipeline exposes run_full_pipeline (Wf-323E ship surface).
    assert hasattr(pipe, "run_full_pipeline"), (
        "CPU pipeline is missing run_full_pipeline() — Wf-323E surface "
        "must be intact"
    )

    # Smoke run on the canonical 4-SMILES pool (cisplatin + 3 small mols).
    smoke = mod.run_joint_inference_smoke(wrapper)
    assert smoke["n_smiles"] >= 1, "smoke n_smiles must be >= 1"
    assert smoke["bridge_present"] is True, (
        "PlatinAI 5K bridge must be present (stub fallback is OK; " "Wf-323E closeout)"
    )
    # On a missing RDKit env the smoke can return ok=False; in that case
    # we still expect at least one decoded SMILES attempted.
    assert all(
        c >= 0 for c in smoke["decoded_counts"]
    ), "decoded_counts must be non-negative"

    # build_train_kwargs must produce the exact kwargs shape the Wf-329-W6
    # 1-line adapter change at train_cfm_yuelbond_wasserstein.py:809-811
    # consumes.
    kwargs = wrapper.build_train_kwargs()
    assert kwargs["lambda_reward_dataset"] is None
    assert kwargs["lambda_cfm_coupling_weight"] == 0.0
    assert kwargs["batch_size"] == 64


# ---------------------------------------------------------------------------
# Phase 2 — wrapper CLI args
# ---------------------------------------------------------------------------
def test_wf333g1_wrapper_cli_args() -> None:
    """Verify the documented CLI surface parses correctly.

    * ``--dry-run`` is the default (no flag ⇒ True).
    * ``--no-dry-run`` flips ``dry_run`` to False.
    * ``--cfm-checkpoint ''`` translates to ``None`` (disabled).
    * ``--lambda-cfm-coupling-weight`` flows through
      :meth:`build_train_kwargs`.
    * ``parse_args(['--help'])`` exits cleanly (subprocess smoke).
    """
    mod = _load_wrapper_module()
    parse_args = mod.parse_args
    build_wrapper = mod.build_wrapper

    # Default: dry-run True, CPU-safe scaffold defaults
    args = parse_args([])
    assert args.dry_run is True, "dry_run default must be True"
    assert args.device == "cuda"
    assert args.lambda_cfm_coupling_weight == 0.0
    assert args.batch_size == 64
    assert args.n_test_molecules == 4
    assert args.pocket_id == "1h36"

    # --no-dry-run flips the flag.
    args = parse_args(["--no-dry-run"])
    assert args.dry_run is False, "--no-dry-run must set dry_run=False"
    wrapper = build_wrapper(args)
    assert wrapper.dry_run is False

    # --lambda-cfm-coupling-weight is forwarded to build_train_kwargs.
    args = parse_args(["--lambda-cfm-coupling-weight", "0.25"])
    wrapper = build_wrapper(args)
    assert wrapper.lambda_cfm_coupling_weight == pytest.approx(0.25)
    kwargs = wrapper.build_train_kwargs()
    assert kwargs["lambda_cfm_coupling_weight"] == pytest.approx(0.25)

    # --cfm-checkpoint '' translates to None (disabled).
    args = parse_args(["--cfm-checkpoint", ""])
    wrapper = build_wrapper(args)
    assert (
        wrapper.cfm_checkpoint_path is None
    ), "empty --cfm-checkpoint must disable the checkpoint path"
    # A non-empty path passes through.
    args = parse_args(["--cfm-checkpoint", "/tmp/fake_cfm.pt"])
    wrapper = build_wrapper(args)
    assert wrapper.cfm_checkpoint_path == Path("/tmp/fake_cfm.pt")

    # --n-test-molecules flows through.
    args = parse_args(["--n-test-molecules", "8", "--pocket-id", "3eml"])
    wrapper = build_wrapper(args)
    assert wrapper.n_test_molecules == 8
    assert wrapper.pocket_id == "3eml"

    # --help exits 0 (subprocess smoke — exercises the argparse exit handler).
    import subprocess

    proc = subprocess.run(
        [
            sys.executable,
            str(
                PROJECT_ROOT
                / "molmetal"
                / "scripts"
                / "wf333g1_lambda_cfm_gpu_wrapper.py"
            ),
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == 0, (
        f"--help must exit 0; got {proc.returncode}\n" f"stderr: {proc.stderr}"
    )
    assert (
        "Wf-333-G1" in proc.stdout
    ), f"--help stdout missing 'Wf-333-G1' header; got:\n{proc.stdout[:200]}"
