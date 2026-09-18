"""GPU execution must remain explicit, reproducible and fail closed."""
from pathlib import Path
import subprocess

import numpy as np
import pytest

from molmetal.ports import DockingConfig
from molmetal_lam.sbdd_env import vina_gpu_adapter as module


@pytest.fixture
def adapter(tmp_path):
    binary = tmp_path / "QuickVina2-GPU-2-1"
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o700)
    (tmp_path / "OpenCL").mkdir()
    (tmp_path / "OpenCL" / "kernel.cl").write_text("// mock OpenCL source\n")
    return module.QuickVinaGPUAdapter(binary, output_dir=tmp_path / "runs")


def invoke(adapter, tmp_path, config=None):
    receptor, ligand = tmp_path / "rec.pdbqt", tmp_path / "lig.pdbqt"
    receptor.write_text("receptor\n")
    ligand.write_text("ligand\n")
    return adapter._dock_cli_binary(receptor, ligand, np.zeros(3), np.ones(3) * 20,
                                    config or DockingConfig(seed=0))


def successful_process(command, *, stdout, device="gfx1101", returncode=0, write_pose=True, **kwargs):
    seed = command[command.index("--seed") + 1]
    stdout.write(f"GPU Platform: AMD Accelerated Parallel ProcessingGPU Platform 0 version: OpenCL 2.1\n"
                 f"GPU Device: {device}Platform 0 global memory size:17 GB\nUsing random seed: {seed}\n")
    if write_pose:
        Path(command[command.index("--out") + 1]).write_text(
            "MODEL 1\nREMARK VINA RESULT: -7.5 0 0\nENDMDL\n")
    return subprocess.CompletedProcess(command, returncode)


def test_gpu_budget_seed_and_isolated_cache(monkeypatch, adapter, tmp_path):
    commands = []
    def run(command, **kwargs):
        commands.append((command, kwargs))
        return successful_process(command, **kwargs)
    monkeypatch.setattr(module.subprocess, "run", run)
    first, _ = invoke(adapter, tmp_path, DockingConfig(seed=0, exhaustiveness=99))
    first_metadata = dict(adapter.last_run_metadata)
    invoke(adapter, tmp_path)
    assert first[0, 0] == -7.5
    cmd, kwargs = commands[0]
    assert "--exhaustiveness" not in cmd and "--cpu" not in cmd
    assert cmd[cmd.index("--thread") + 1] == "1000"
    assert cmd[cmd.index("--search_depth") + 1] == "1"
    assert int(cmd[cmd.index("--seed") + 1]) != 0
    assert kwargs["env"]["GPU_DEVICE_ORDINAL"] == "0"
    assert Path(kwargs["cwd"], "OpenCL").is_symlink()
    assert first_metadata["requested_native_exhaustiveness_ignored"] == 99
    assert first_metadata["gpu_budget_native_exhaustiveness_equivalent"] is None
    assert first_metadata["gpu_verified"] is True
    assert first_metadata["cwd"] != adapter.last_run_metadata["cwd"]
    assert first_metadata["seed"] == adapter.last_run_metadata["seed"]
    assert Path(first_metadata["output_pdbqt"]).is_file()


@pytest.mark.parametrize("options,error", [
    ({"device": "CPU"}, "Expected AMD OpenCL GPU"),
    ({"returncode": 1}, "exited 1"),
    ({"write_pose": False}, "produced no pose"),
])
def test_gpu_failure_does_not_fallback(monkeypatch, adapter, tmp_path, options, error):
    monkeypatch.setattr(module.subprocess, "run", lambda c, **kw: successful_process(c, **kw, **options))
    with pytest.raises(RuntimeError, match=error):
        invoke(adapter, tmp_path)
    assert adapter.last_run_metadata["status"] == "failed"
    assert adapter.last_run_metadata["gpu_verified"] is False
    assert Path(adapter.last_run_metadata["log_path"]).is_file()


def test_gpu_timeout_preserves_log(monkeypatch, adapter, tmp_path):
    def timeout(command, **kwargs):
        kwargs["stdout"].write("GPU stage started\n")
        raise subprocess.TimeoutExpired(command, 1)
    monkeypatch.setattr(module.subprocess, "run", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        invoke(adapter, tmp_path)
    assert adapter.last_run_metadata["status"] == "failed"
    assert "GPU stage started" in Path(adapter.last_run_metadata["log_path"]).read_text()


def test_explicit_device_and_prepared_receptor_required(adapter):
    from types import SimpleNamespace
    with pytest.raises(ValueError, match="opencl:0"):
        adapter.setup("cuda")
    with pytest.raises(ValueError, match="prepared receptor"):
        adapter._prepare_receptor(SimpleNamespace(pdb_id="unprepared"))
    with pytest.raises(ValueError, match="passed receptor"):
        adapter.set_prepared_receptor(SimpleNamespace(pdb_id="x"), {"passed": False})


def test_instrumented_run_requires_both_completed_kernels(monkeypatch, adapter, tmp_path):
    adapter._trace_library = tmp_path / "trace.so"
    adapter._trace_library.write_text("test")
    monkeypatch.setattr(module.subprocess, "run", successful_process)
    with pytest.raises(RuntimeError, match="No successful kernel1"):
        invoke(adapter, tmp_path)
    def traced(command, **kwargs):
        for kernel in ("kernel1", "kernel2"):
            kwargs["stdout"].write(f"\rPerform docking|=======|OPENCL_TRACE kernel={kernel} device=gfx1101 enqueue=0 wait=0 status=0 profile_valid=0\n")
        return successful_process(command, **kwargs)
    monkeypatch.setattr(module.subprocess, "run", traced)
    invoke(adapter, tmp_path)
    assert adapter.last_run_metadata["gpu_evidence"] == "kernel_trace"


def test_seed_log_mismatch_is_rejected(monkeypatch, adapter, tmp_path):
    def wrong_seed(command, **kwargs):
        kwargs["stdout"].write("Using random seed: -1\n")
        return successful_process(command, **kwargs)
    monkeypatch.setattr(module.subprocess, "run", wrong_seed)
    with pytest.raises(RuntimeError, match="seed log"):
        invoke(adapter, tmp_path)


@pytest.mark.parametrize("kwargs", [{"gpu_threads": 999}, {"search_depth": 0}, {"timeout_s": -1}])
def test_invalid_gpu_budgets_rejected_before_execution(adapter, kwargs):
    with pytest.raises(ValueError):
        module.QuickVinaGPUAdapter(adapter._engine_binary, **kwargs)
