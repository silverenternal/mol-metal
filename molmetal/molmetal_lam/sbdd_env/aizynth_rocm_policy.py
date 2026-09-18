"""Official AiZynth template search with explicit remote ROCm neural inference.

Only import this optional extension inside the isolated AiZynth environment.
RDKit, templates, stock queries, MCTS and route analysis remain official CPU
implementations. A persistent root-environment worker performs the Dense net.
"""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import select
import subprocess
import tempfile

import numpy as np
from aizynthfinder.context.policy.expansion_strategies import TemplateBasedExpansionStrategy


class RemoteTorchPolicy:
    output_size = 46695

    def __init__(self, weights, expected_sha256):
        root = Path(__file__).resolve().parents[3]
        command = ["uv", "run", "--project", str(root), "--no-sync", "python",
                   str(root / "molmetal/scripts/aizynth_torch_worker.py"),
                   "--weights", str(weights), "--sha256", expected_sha256]
        env = dict(os.environ, HIP_VISIBLE_DEVICES="0")
        env.pop("VIRTUAL_ENV", None)
        self._stderr = tempfile.TemporaryFile()
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self._stderr, text=True, bufsize=1, env=env)
        try:
            greeting = self._read()
            if not greeting.get("ready"):
                raise RuntimeError("ROCm inference worker did not become ready")
            self.runtime_metadata = greeting["runtime"]
        except Exception:
            self.close()
            raise

    def _read(self):
        if not select.select([self.process.stdout], [], [], 120)[0]:
            raise TimeoutError("ROCm inference worker response exceeded 120 seconds")
        line = self.process.stdout.readline()
        if not line:
            self._stderr.seek(0)
            raise RuntimeError("ROCm worker exited: " + self._stderr.read().decode(errors="replace")[-1500:])
        return json.loads(line)

    def __len__(self):
        return 2048

    def predict(self, *args, **kwargs):
        if len(args) != 1:
            raise ValueError("Expected a single fingerprint array")
        x = np.asarray(args[0], dtype=np.float32)
        request = {"shape": list(x.shape), "input": base64.b64encode(x.tobytes()).decode("ascii")}
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        response = self._read()
        self.runtime_metadata = response["runtime"]
        output = np.frombuffer(base64.b64decode(response["output"], validate=True), dtype=np.float32).reshape(response["shape"])
        if output.shape != (len(x), self.output_size) or not np.isfinite(output).all():
            raise RuntimeError("Invalid ROCm policy probabilities")
        return output

    def close(self):
        process = getattr(self, "process", None)
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            for stream in [process.stdin, process.stdout]:
                if stream:
                    stream.close()
            self.process = None
        stderr = getattr(self, "_stderr", None)
        if stderr:
            stderr.close()
            self._stderr = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class TorchRemoteExpansionStrategy(TemplateBasedExpansionStrategy):
    def __init__(self, key, config, **kwargs):
        super().__init__(key, config, **kwargs)
        # Official initialization validates ONNX dimensions against templates;
        # replace only the predictor after these original checks pass.
        self.model = RemoteTorchPolicy(kwargs["weights"], kwargs["weights_sha256"])
