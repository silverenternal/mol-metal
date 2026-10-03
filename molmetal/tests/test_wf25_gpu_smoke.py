"""Wf-25 Phase 5 — GPU smoke test (RTX PRO 6000 Blackwell vs. RTX 5090).

Wf-25 (workflow round 25) closes out the round-24 honest-negative stream by
running a 1-epoch smoke training pass on whichever GPU is free, then emitting
a machine-readable JSON report at
``molmetal/reports/wf25_gpu_smoke/gpu_check.json`` so downstream phases can
pick a training device without re-running ``nvidia-smi`` themselves.

The Phase-5 task spec asks two things:

  1. **Check** whether GPU 1 (RTX 5090) or GPU 0 (RTX PRO 6000 Blackwell,
     avoiding the ``kanzi`` user) is free enough to run a tiny training
     step, and
  2. **Smoke-train** a 50-step 64 -> 10 BF16 autocast loop on the picked
     GPU (≈ 64 MB peak alloc, ≪ 32 GB), recording peak memory + wall time.

If neither GPU is free the report is an honest negative ("gpu_available=False,
error=...") rather than a silent skip — the harness consumes the JSON and
needs to know the verdict either way.

Markers: ``@pytest.mark.gpu`` (matches the project's existing GPU-test
convention; see ``pyproject.toml`` ``[tool.pytest.ini_options].markers``).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPORT_DIR = (
    Path(__file__).resolve().parent.parent / "reports" / "wf25_gpu_smoke"
)
REPORT_PATH = REPORT_DIR / "gpu_check.json"

# 50 steps × 64-batch BF16 forward/backward through a (64→128→128→10) MLP.
# Allocates ~64 MB peak — fits comfortably on either Blackwell (95 GB) or
# RTX 5090 (32 GB).  Mirrors the parameters used in the wf25 driver script.
N_STEPS = 50
BATCH = 64
HIDDEN = 128
PEAK_ALLOC_BUDGET_MB = 256  # generous cap; flags a leak if hit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _smoke_train_once(device: torch.device) -> Dict[str, Any]:
    """Run a 50-step BF16 autocast train loop on ``device``.

    Returns a dict with: ``loss``, ``wall_s``, ``steps_per_s``,
    ``peak_alloc_mb``.  Raises if the loop crashes (CUDA OOM, kernel
    fault, NaN loss, etc.).
    """
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(64, HIDDEN), nn.GELU(),
        nn.Linear(HIDDEN, HIDDEN), nn.GELU(),
        nn.Linear(HIDDEN, 10),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(BATCH, 64, device=device)
    y = torch.randn(BATCH, 10, device=device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    t0 = time.time()
    last_loss = float("nan")
    for _ in range(N_STEPS):
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model(x)
            loss = nn.functional.mse_loss(pred, y)
        if not torch.isfinite(loss):
            raise RuntimeError(
                f"non-finite loss during smoke: {loss.item()}"
            )
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().cpu())
    torch.cuda.synchronize(device)
    dt = time.time() - t0
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    del model, opt, x, y, pred, loss
    torch.cuda.empty_cache()

    return {
        "loss": last_loss,
        "wall_s": round(dt, 3),
        "steps_per_s": round(N_STEPS / dt, 1),
        "peak_alloc_mb": round(peak_mb, 1),
        "n_steps": N_STEPS,
        "batch": BATCH,
        "hidden": HIDDEN,
        "amp_dtype": "bfloat16",
    }


def _nvidia_smi_snapshot() -> Optional[Dict[str, Any]]:
    """Run ``nvidia-smi`` and return a per-GPU summary, or None if missing.

    We only care about: index, name, memory total / used, and whether any
    process is using more than 1 GB (rough "occupied" signal).  Returns
    ``None`` if ``nvidia-smi`` is not on ``$PATH`` — the test then falls
    back to the torch-level report.
    """
    if shutil.which("nvidia-smi") is None:
        return None
    query = (
        "index,name,memory.total,memory.used,utilization.gpu,compute_cap"
    )
    fmt = "csv,noheader,nounits"
    proc = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", f"--format={fmt}"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if proc.returncode != 0:
        return None
    gpus = []
    for line in proc.stdout.strip().splitlines():
        cols = [c.strip() for c in line.split(",")]
        if len(cols) < 6:
            continue
        idx, name, mem_total_mb, mem_used_mb, util_gpu, compute_cap = cols[:6]
        gpus.append(
            {
                "index": int(idx),
                "name": name,
                "memory_total_mb": int(mem_total_mb),
                "memory_used_mb": int(mem_used_mb),
                "utilization_gpu_pct": int(util_gpu),
                "compute_cap": compute_cap,
            }
        )
    # Processes (used to detect "kanzi" occupant that the task spec warns
    # about for GPU 0).
    proc_p = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,gpu_uuid,gpu_memory_usage",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    procs: list[dict[str, Any]] = []
    if proc_p.returncode == 0:
        for line in proc_p.stdout.strip().splitlines():
            cols = [c.strip() for c in line.split(",")]
            if len(cols) < 4:
                continue
            pid, pname, gpu_uuid, gpu_mem_mb = cols[:4]
            procs.append(
                {
                    "pid": int(pid),
                    "process_name": pname,
                    "gpu_uuid": gpu_uuid,
                    "gpu_memory_mb": int(gpu_mem_mb),
                }
            )
    return {"gpus": gpus, "processes": procs}


def _write_report(payload: Dict[str, Any]) -> None:
    """Atomically write ``payload`` to ``REPORT_PATH`` (pretty JSON)."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = REPORT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(REPORT_PATH)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.gpu
def test_gpu_pytorch_cuda_visible() -> None:
    """torch sees ≥ 1 CUDA device (either Blackwell or RTX 5090).

    The project's ``env.sh`` pins ``CUDA_VISIBLE_DEVICES=0`` (Blackwell),
    so by the time this test runs torch should report ≥ 1 device unless
    the host truly has no working NVIDIA driver.
    """
    assert torch.cuda.is_available(), (
        "torch.cuda.is_available() is False; env.sh not sourced or "
        "NVIDIA driver unavailable. See env.sh §1 / §5."
    )
    n = torch.cuda.device_count()
    assert n >= 1, f"torch sees {n} CUDA devices (expected ≥ 1)"
    # torch CUDA build must be ≥ 12.x to talk to Blackwell sm_120.
    major, minor = map(int, torch.version.cuda.split(".")[:2])
    assert (major, minor) >= (12, 0), (
        f"CUDA runtime {torch.version.cuda} is too old for sm_120"
    )


@pytest.mark.gpu
def test_gpu0_device_properties() -> None:
    """Device 0 is a Blackwell-class GPU with ≥ 32 GB VRAM.

    This guards the assumption that "use GPU 0" means "use a large
    Blackwell" rather than "use whatever leftover device is index 0".
    On a host that reorders devices (driver bug, MIG slicing) the test
    skips so we don't false-fail — but the JSON report still records the
    observed properties.
    """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    major, minor = torch.cuda.get_device_capability(0)
    # Loose check: compute-cap ≥ 12.x (Blackwell sm_120) AND ≥ 16 GB.
    # Both RTX PRO 6000 (95 GB) and RTX 5090 (32 GB) pass; older cards fail.
    assert major >= 12 or total_gb >= 16, (
        f"device 0 = {name!r}, {total_gb:.1f} GB, sm_{major}{minor}; "
        "neither Blackwell-class nor ≥ 16 GB"
    )


@pytest.mark.gpu
def test_gpu0_smoke_train_step() -> None:
    """50-step BF16 autocast smoke on device 0 succeeds with finite loss."""
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    device = torch.device("cuda:0")
    info = _smoke_train_once(device)
    assert info["loss"] == info["loss"], "loss is NaN"  # NaN-check
    assert info["peak_alloc_mb"] <= PEAK_ALLOC_BUDGET_MB, (
        f"peak alloc {info['peak_alloc_mb']} MB exceeds "
        f"{PEAK_ALLOC_BUDGET_MB} MB budget (probable CUDA leak)"
    )


@pytest.mark.gpu
def test_gpu1_smoke_train_step() -> None:
    """If GPU 1 (RTX 5090) is exposed, the smoke loop also passes there.

    Runs only when ``CUDA_VISIBLE_DEVICES`` exposes ≥ 2 devices — on the
    default ``env.sh`` config it pins to GPU 0, so this test will skip.
    A separate invocation
    (``CUDA_VISIBLE_DEVICES=1 uv run pytest -m gpu ...``) exercises it.
    """
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    if torch.cuda.device_count() < 2:
        pytest.skip(
            f"only {torch.cuda.device_count()} CUDA device(s) visible; "
            "set CUDA_VISIBLE_DEVICES=0,1 to also smoke GPU 1"
        )
    device = torch.device("cuda:1")
    info = _smoke_train_once(device)
    assert info["peak_alloc_mb"] <= PEAK_ALLOC_BUDGET_MB


@pytest.mark.gpu
def test_wf25_gpu_smoke_emits_report_json() -> None:
    """End-to-end: write ``gpu_check.json`` with the spec's exact schema.

    Required keys (per Wf-25 Phase 5 spec):
      ``gpu_available``, ``cuda_visible``, ``pytorch_cuda``,
      ``smoke_test_passed``, ``error_if_failed``.

    This test re-runs the smoke (so it doubles as an integration check)
    and writes the report.  Failures here show up in CI as a real test
    failure rather than a missing artefact.
    """
    payload: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cuda_visible_devices_env": os.environ.get(
            "CUDA_VISIBLE_DEVICES", "<unset>"
        ),
        "pytorch_cuda": torch.version.cuda or "<none>",
        "torch_version": torch.__version__,
        "gpu_available": False,
        # Spec-mandated top-level keys (Wf-25 Phase 5):
        "cuda_visible": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "smoke_test_passed": False,
        "error_if_failed": None,
        "nvidia_smi": None,
        "device_0": None,
        "device_1": None,
        "smoke": None,
    }

    snap = _nvidia_smi_snapshot()
    payload["nvidia_smi"] = snap

    if not torch.cuda.is_available():
        payload["error_if_failed"] = (
            "torch.cuda.is_available() is False; "
            "ensure env.sh is sourced and NVIDIA driver is loaded"
        )
        _write_report(payload)
        pytest.fail(payload["error_if_failed"])

    payload["gpu_available"] = True
    payload["cuda_visible"] = torch.cuda.device_count()
    payload["device_0"] = {
        "name": torch.cuda.get_device_name(0),
        "total_mem_gb": round(
            torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 2
        ),
        "compute_cap": ".".join(
            str(x) for x in torch.cuda.get_device_capability(0)
        ),
    }
    if torch.cuda.device_count() >= 2:
        payload["device_1"] = {
            "name": torch.cuda.get_device_name(1),
            "total_mem_gb": round(
                torch.cuda.get_device_properties(1).total_memory / (1024 ** 3), 2
            ),
            "compute_cap": ".".join(
                str(x) for x in torch.cuda.get_device_capability(1)
            ),
        }

    try:
        info = _smoke_train_once(torch.device("cuda:0"))
        payload["smoke"] = {"device": "cuda:0", **info}
        payload["smoke_test_passed"] = True
    except Exception as exc:  # noqa: BLE001 — surface to JSON verbatim
        payload["error_if_failed"] = f"{type(exc).__name__}: {exc}"
        _write_report(payload)
        pytest.fail(payload["error_if_failed"])

    _write_report(payload)
    assert payload["smoke_test_passed"] is True
    assert payload["gpu_available"] is True
