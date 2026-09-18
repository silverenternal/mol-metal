"""Isolated, explicitly learned REINVENT prior NLL transport.

This is independent of the four-component multiproperty/proxy adapter. It
does not normalize NLL, combine rewards, or import REINVENT/PyTorch here.
"""
from __future__ import annotations

import json
import math
import os
import selectors
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PriorLikelihoodResult:
    index: int
    smiles: Any
    status: str
    prior_nll: float | None = None
    predicted_token_count: int | None = None
    error: str | None = None
    backend: str = "reinvent-prior"
    model_sha256: str | None = None
    device: str | None = None


class REINVENT4PriorAdapter:
    """Batch NLL with one total deadline including startup, locks and I/O.

    Timeout cleanup has a bounded additional 0.4-second reap allowance.
    Worker logs go to a file, avoiding unconsumed stderr pipe deadlocks.
    """

    def __init__(self, *, worker_python: str, prior_path: str, prior_sha256: str,
                 device: str = "cuda:0", timeout: float = 60.0,
                 expected_architecture: str = "gfx1101", worker_script: str | None = None,
                 log_path: str | None = None):
        if device not in ("cpu", "cuda:0"):
            raise ValueError("device must be explicitly cpu or cuda:0")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if len(prior_sha256) != 64 or any(c not in "0123456789abcdef" for c in prior_sha256):
            raise ValueError("prior_sha256 must be a lowercase SHA256")
        self.worker_python = str(worker_python)
        self.worker_script = worker_script or str(Path(__file__).with_name("reinvent_prior_jsonl_worker.py"))
        self.prior_path, self.prior_sha256 = str(prior_path), prior_sha256
        self.device, self.timeout, self.expected_architecture = device, timeout, expected_architecture
        self.log_path = log_path
        self.last_metadata: dict = {}
        self.last_error: str | None = None
        self._proc: subprocess.Popen | None = None
        self._log = None
        self._lock = threading.Lock()
        self._buffer = bytearray()
        self._request_id = 0

    @classmethod
    def from_config(cls, path: str | Path) -> "REINVENT4PriorAdapter":
        config = json.loads(Path(path).read_text())
        if config.pop("mode", None) != "prior_nll":
            raise ValueError("Expected explicit mode=prior_nll")
        return cls(**config)

    @staticmethod
    def _remaining(deadline: float) -> float:
        value = deadline - time.monotonic()
        if value <= 0:
            raise TimeoutError("total_request_timeout")
        return value

    def _stop(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                pass
            # Also kill descendants when the leader exited before them.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=0.2)
            except subprocess.TimeoutExpired:
                pass
            for stream in (proc.stdin, proc.stdout):
                if stream is not None:
                    stream.close()
        if self._log is not None:
            self._log.close()
            self._log = None
        self._buffer.clear()

    def close(self) -> None:
        with self._lock:
            self._stop()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _rpc(self, op: str, deadline: float, **payload) -> dict:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.stdout is None:
            raise RuntimeError("worker_unavailable")
        self._request_id += 1
        request_id = self._request_id
        data = json.dumps({"id": request_id, "op": op, **payload}, allow_nan=False).encode() + b"\n"
        with selectors.DefaultSelector() as selector:
            selector.register(proc.stdin, selectors.EVENT_WRITE)
            while data:
                if not selector.select(self._remaining(deadline)):
                    raise TimeoutError("total_request_timeout")
                try:
                    count = os.write(proc.stdin.fileno(), data)
                    data = data[count:]
                except BlockingIOError:
                    continue
            selector.unregister(proc.stdin)
            selector.register(proc.stdout, selectors.EVENT_READ)
            while b"\n" not in self._buffer:
                if not selector.select(self._remaining(deadline)):
                    raise TimeoutError("total_request_timeout")
                try:
                    chunk = os.read(proc.stdout.fileno(), 65536)
                except BlockingIOError:
                    continue
                if not chunk:
                    raise RuntimeError(f"worker_closed_stdout:{proc.poll()}")
                self._buffer.extend(chunk)
                if len(self._buffer) > 4 * 1024 * 1024:
                    raise ValueError("worker_response_too_large")
        line, _, remainder = self._buffer.partition(b"\n")
        self._buffer = bytearray(remainder)
        response = json.loads(line)
        if not isinstance(response, dict) or response.get("id") != request_id:
            raise ValueError("worker_response_id_mismatch")
        self._remaining(deadline)
        if response.get("error"):
            raise RuntimeError(f"worker_error:{response['error']}")
        return response

    def _validate_metadata(self, metadata: dict) -> None:
        if (metadata.get("backend") != "reinvent-prior"
                or metadata.get("protocol") != "reinvent-prior-jsonl-v1"
                or metadata.get("model_sha256") != self.prior_sha256
                or metadata.get("device") != self.device):
            raise ValueError("worker_backend_model_or_device_mismatch")
        if self.device.startswith("cuda") and (
                not metadata.get("torch_hip")
                or metadata.get("architecture") != self.expected_architecture):
            raise ValueError("worker_rocm_architecture_mismatch")

    def _start(self, deadline: float) -> None:
        self._remaining(deadline)
        env = os.environ.copy()
        if self.device.startswith("cuda"):
            env["HIP_VISIBLE_DEVICES"] = "0"
        if self.log_path:
            path = Path(self.log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._log = path.open("ab", buffering=0)
        else:
            self._log = tempfile.TemporaryFile()
        command = [self.worker_python, self.worker_script, "--prior", self.prior_path,
                   "--prior-sha256", self.prior_sha256, "--device", self.device]
        self._proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=self._log, start_new_session=True, bufsize=0, env=env)
        os.set_blocking(self._proc.stdin.fileno(), False)
        os.set_blocking(self._proc.stdout.fileno(), False)
        response = self._rpc("capabilities", deadline)
        self._validate_metadata(response.get("metadata", {}))
        self.last_metadata = response["metadata"]

    def likelihood(self, smiles_batch: list[Any]) -> list[PriorLikelihoodResult]:
        if not isinstance(smiles_batch, list):
            raise TypeError("smiles_batch must be a list")
        if not smiles_batch:
            return []
        deadline = time.monotonic() + self.timeout

        def failure(error):
            return [PriorLikelihoodResult(i, s, "error", error=error,
                                          model_sha256=self.prior_sha256, device=self.device)
                    for i, s in enumerate(smiles_batch)]

        if not self._lock.acquire(timeout=self.timeout):
            self.last_error = "total_request_timeout:lock"
            return failure(self.last_error)
        try:
            if self._proc is None or self._proc.poll() is not None:
                self._stop()
                self._start(deadline)
            response = self._rpc("likelihood", deadline, smiles=smiles_batch)
            metadata = response.get("metadata", {})
            self._validate_metadata(metadata)
            rows = response.get("results")
            if not isinstance(rows, list) or len(rows) != len(smiles_batch):
                raise ValueError("worker_result_count_mismatch")
            results = []
            for index, (smiles, row) in enumerate(zip(smiles_batch, rows)):
                if row.get("index") != index or row.get("smiles") != smiles:
                    raise ValueError("worker_result_identity_mismatch")
                if row.get("status") == "ok":
                    value, count = row.get("prior_nll"), row.get("predicted_token_count")
                    if (isinstance(value, bool) or not isinstance(value, (int, float))
                            or not math.isfinite(value) or value < 0
                            or isinstance(count, bool) or not isinstance(count, int) or count < 1):
                        raise ValueError("worker_invalid_nll_or_token_count")
                elif row.get("status") != "error" or not row.get("error"):
                    raise ValueError("worker_invalid_error_result")
                results.append(PriorLikelihoodResult(index, smiles, row["status"],
                    prior_nll=row.get("prior_nll") if row["status"] == "ok" else None,
                    predicted_token_count=row.get("predicted_token_count") if row["status"] == "ok" else None,
                    error=row.get("error"), model_sha256=metadata["model_sha256"], device=metadata["device"]))
            scored = sum(row.status == "ok" for row in results)
            forwards = metadata.get("neural_forwards", [])
            if (metadata.get("n_requested") != len(smiles_batch) or metadata.get("n_scored") != scored
                    or (scored and not forwards)
                    or any(entry.get("parameter_device") != self.device
                           or entry.get("input_device") != self.device
                           or not isinstance(entry.get("count"), int) or entry["count"] <= 0
                           for entry in forwards)):
                raise ValueError("worker_neural_execution_or_count_mismatch")
            self._remaining(deadline)
            self.last_metadata, self.last_error = metadata, None
            return results
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}:{exc}"
            self._stop()
            return failure(self.last_error)
        finally:
            self._lock.release()
