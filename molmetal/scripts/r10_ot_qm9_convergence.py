"""Calibrate numerical OT residuals on the fixed real-QM9 training groups.

No model is trained and no validation outcome is used for hyperparameter
selection. The saved prior real-data manifest fixes data rows and hashes.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from data.qm9 import QM9Dataset
from flow_matching.optimal_transport import mini_batch_ot_coupling
from molmetal.scripts.r10_ot_qm9_3seed import compatible_batch, file_hash


def main():
    prior_path = ROOT / "molmetal/reports/r10_ot_qm9_3seed.json"
    prior = json.loads(prior_path.read_text())
    for source in prior["sources"]:
        if file_hash(source["path"]) != source["sha256"]:
            raise ValueError(f"Source hash changed: {source['path']}")
    dataset = QM9Dataset(split="train")
    if dataset.source != "real":
        raise ValueError("Requires verified real QM9")
    device = torch.device("cuda:0")
    ids = [row["split_row_index"] for row in prior["selected_rows"]["train"]]
    batches = [compatible_batch([dataset[i] for i in ids[start:start + 8]], device,
                                prior["protocol"]["position_scale"])
               for start in range(0, len(ids), 8)]
    records = []
    # More iterations preserve the objective; increasing reg deliberately
    # changes its entropy coefficient. No cost normalization is applied.
    for reg, max_iter in ((0.05, 200), (0.05, 2000), (1.0, 2000)):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        errors = []
        for seed in (42, 0, 1234):
            for step, batch in enumerate(batches):
                torch.manual_seed(seed + 10_000 + step)
                mask = batch["node_mask"].unsqueeze(-1)
                x0 = torch.randn_like(batch["positions"]) * mask
                x0 = (x0 - x0.sum(1, keepdim=True) / mask.sum(1, keepdim=True)) * mask
                diagnostics = []
                mini_batch_ot_coupling(batch["positions"], x0, torch.zeros(8, device=device),
                                      reg=reg, max_iter=max_iter, diagnostics=diagnostics)
                errors.append({"seed": seed, "group": step, **diagnostics[0]})
        torch.cuda.synchronize(device)
        result = {"reg": reg, "max_iter": max_iter,
                  "objective": "raw squared-coordinate cost, no normalization; entropy coefficient=reg",
                  "seconds": time.perf_counter() - started, "groups": errors,
                  "max_marginal_error": max(row["marginal_max_abs_error"] for row in errors),
                  "passed_1e_3": sum(row["marginal_max_abs_error"] <= 1e-3 for row in errors)}
        records.append(result)
        print({key: value for key, value in result.items() if key != "groups"}, flush=True)
    output = {"prior_manifest": str(prior_path), "manifest_sha256": file_hash(prior_path),
              "scope": "24 real-QM9 training noise/group draws; no model training or validation metric used",
              "calibration": records}
    (ROOT / "molmetal/reports/r10_ot_qm9_convergence.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
