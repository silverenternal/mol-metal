#!/usr/bin/env python3
"""Persistent JSON/base64 policy inference in the root ROCm environment."""
import argparse
import base64
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "molmetal")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    with redirect_stdout(sys.stderr):
        import numpy as np
        from molmetal_lam.sbdd_env.aizynth_torch_model import TorchDensePolicy
        model = TorchDensePolicy(args.weights, expected_sha256=args.sha256, device="cuda:0")
    print(json.dumps({"ready": True, "runtime": model.runtime_metadata}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("close"):
            break
        x = np.frombuffer(base64.b64decode(request["input"], validate=True), dtype=np.float32).reshape(request["shape"]).copy()
        with redirect_stdout(sys.stderr):
            output = model.predict(x)
        print(json.dumps({"shape": list(output.shape), "output": base64.b64encode(output.tobytes()).decode("ascii"),
                          "runtime": model.runtime_metadata}), flush=True)


if __name__ == "__main__":
    main()
