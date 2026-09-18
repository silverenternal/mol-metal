"""AiZynth-compatible inference for the exact audited legacy Dense policy."""
from __future__ import annotations

import hashlib
from pathlib import Path


class TorchDensePolicy:
    def __init__(self, weights_path, *, expected_sha256, device="cuda:0"):
        import numpy as np
        import torch

        with Path(weights_path).open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != expected_sha256:
                raise ValueError("Policy weights hash mismatch")
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested GPU policy is unavailable; no CPU fallback")
        with np.load(weights_path, allow_pickle=False) as arrays:
            shapes = {key: arrays[key].shape for key in ["w1", "b1", "w2", "b2"]}
            if shapes != {"w1": (2048, 512), "b1": (512,), "w2": (512, 46695), "b2": (46695,)}:
                raise ValueError("Unexpected legacy policy tensor shapes")
            self.parameters = [torch.as_tensor(arrays[key].copy(), dtype=torch.float32,
                               device=self.device) for key in ["w1", "b1", "w2", "b2"]]
        self.parameters[0] = self.parameters[0].T.contiguous()
        self.parameters[2] = self.parameters[2].T.contiguous()
        self.output_size = 46695
        props = torch.cuda.get_device_properties(self.device) if self.device.type == "cuda" else None
        self.runtime_metadata = {
            "backend": "torch_dense_legacy_policy", "device": str(self.device),
            "torch_version": torch.__version__, "hip_version": torch.version.hip,
            "gcn_arch": getattr(props, "gcnArchName", None),
            "device_name": getattr(props, "name", None),
            "device_memory_bytes": getattr(props, "total_memory", None),
            "parameter_devices": [str(p.device) for p in self.parameters],
            "weights_sha256": expected_sha256, "predict_calls": 0,
        }

    def __len__(self):
        return 2048

    def predict(self, *args, **kwargs):
        import torch
        from torch.nn import functional as F
        if len(args) != 1:
            raise ValueError("Legacy expansion policy expects exactly one fingerprint input")
        with torch.inference_mode():
            x = torch.as_tensor(args[0], dtype=torch.float32, device=self.device)
            if x.ndim != 2 or x.shape[1] != 2048:
                raise ValueError("Expected [batch, 2048] fingerprints")
            w1, b1, w2, b2 = self.parameters
            output = torch.softmax(F.linear(F.elu(F.linear(x, w1, b1)), w2, b2), dim=1)
            self.runtime_metadata["predict_calls"] += 1
            self.runtime_metadata["last_output_device"] = str(output.device)
            return output.cpu().numpy()
