"""Explicit CPU learned-synthesis bridge into the isolated uv environment."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess

from .aizynth_adapter import RetrosynthesisReport


def _run_process_group(command, *, input, timeout):
    """A deadline owns the launcher and every CPU/GPU worker it starts."""
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        stdout, stderr = process.communicate(input=input, timeout=timeout)
    except BaseException as exc:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        finally:
            # Descendants can outlive the direct launcher; kill the group even
            # when communicate() has already reaped that launcher.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate()
        if isinstance(exc, subprocess.TimeoutExpired):
            raise TimeoutError(f"AiZynth worker exceeded {timeout}s; its process group was terminated") from exc
        raise
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


class IsolatedAiZynthChecker:
    """One process per batch, so policy/stock load only once for that batch."""

    def __init__(self, config_path, *, max_iterations=200, time_limit_s=30, seed=20260913):
        self.config_path = str(Path(config_path).resolve())
        self.max_iterations = max_iterations
        self.time_limit_s = time_limit_s
        self.seed = seed
        root = Path(__file__).resolve().parents[3]
        self.command = ["bash", str(root / "environments/aizynth/run.sh"),
                        "python", str(root / "molmetal/scripts/aizynth_worker.py")]
        self.metadata = None

    def _invoke(self, smiles):
        request = {"config_path": self.config_path, "smiles": list(smiles),
                   "max_iterations": self.max_iterations,
                   "time_limit_s": self.time_limit_s, "seed": self.seed}
        # A hard process bound includes startup/model load and iteration overrun.
        timeout = 120 + len(smiles) * (self.time_limit_s + 30)
        result = _run_process_group(self.command, input=json.dumps(request), timeout=timeout)
        try:
            response = json.loads(result.stdout)
        except (ValueError, TypeError) as exc:
            raise RuntimeError(f"AiZynth worker returned no valid JSON (exit {result.returncode}): {result.stderr[-1000:]}") from exc
        if response.get("schema_version") != 1:
            raise RuntimeError("Unsupported AiZynth worker schema")
        self.metadata = response.get("metadata", {})
        if result.returncode or not response.get("ready"):
            raise RuntimeError(self.metadata.get("load_error") or f"AiZynth worker unavailable (exit {result.returncode})")
        if len(response.get("reports", [])) != len(smiles):
            raise RuntimeError("AiZynth worker report count mismatch")
        return response["reports"]

    def probe(self):
        self._invoke([])
        return self.metadata

    def check_many(self, smiles):
        return [RetrosynthesisReport(
            smiles=row["smiles"], synthesizable=bool(row["synthesizable"]),
            depth=int(row["depth"]), route_smiles=tuple(row.get("route_smiles", [])),
            engine=row["engine"],
        ) for row in self._invoke(smiles)]

    def __call__(self, smiles):
        return self.check_many([smiles])[0]
