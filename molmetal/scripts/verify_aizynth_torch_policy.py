#!/usr/bin/env python3
"""Compare real ROCm policy inference with saved ONNX inference outputs."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "molmetal")]


def main():
    import numpy as np
    import torch
    from molmetal_lam.sbdd_env.aizynth_torch_model import TorchDensePolicy
    directory = Path("/mnt/storage/data/molmetal/aizynth_public/legacy_v3")
    conversion = json.loads((directory / "conversion_validation.json").read_text())
    for name, key in [("validation_inputs.npy", "validation_inputs_sha256"),
                      ("validation_onnx_probabilities.npy", "validation_onnx_sha256")]:
        assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == conversion[key]
    x = np.load(directory / "validation_inputs.npy", allow_pickle=False)
    expected = np.load(directory / "validation_onnx_probabilities.npy", allow_pickle=False)
    model = TorchDensePolicy(directory / "uspto_legacy_weights.npz",
                             expected_sha256=conversion["weights_npz_sha256"], device="cuda:0")
    observed = model.predict(x)
    np.testing.assert_allclose(observed, expected, rtol=3e-4, atol=3e-7)
    np.testing.assert_array_equal(observed.argmax(1), expected.argmax(1))
    topk_agreement = []
    for k in [1, 10, 50]:
        matches = [set(a[-k:]) == set(b[-k:]) for a, b in zip(np.argsort(expected, axis=1), np.argsort(observed, axis=1))]
        topk_agreement.append({"k": k, "set_agreement_fraction": float(np.mean(matches))})
    timings = []
    for _ in range(5):
        torch.cuda.synchronize()
        start = time.perf_counter()
        model.predict(x)
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)
    report = {
        "status": "real_rocm_policy_matches_onnx", "seed": 20260913,
        "protocol": "same official legacy weights; CPU RDKit fingerprints; GPU Dense/ELU/Softmax; NumPy return",
        "n_inputs": len(x), "maximum_absolute_probability_error": float(np.abs(observed - expected).max()),
        "topk_agreement": topk_agreement, "gpu_predict_wall_seconds": timings,
        "timing_scope": "post-warmup batch of 28, including transfer and NumPy output; no CPU speedup claim",
        "runtime": model.runtime_metadata,
        "onnx_sha256": conversion["converted_sha256"], "original_hdf5_sha256": conversion["source_sha256"],
    }
    output = ROOT / "molmetal/reports/aizynth_real_backend_20260913/rocm_policy_comparison.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
