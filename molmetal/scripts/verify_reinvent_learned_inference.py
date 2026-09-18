"""Observe official REINVENT sampling and learned prior NLL without proxies.

Run with the isolated REINVENT Python environment; see the report README.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, choices=["cpu", "cuda:0"])
    parser.add_argument("--prior", type=Path, default=Path("/mnt/storage/models/reinvent4/reinvent_v4.4.22.prior"))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {"created_utc": datetime.now(timezone.utc).isoformat(), "passed": False,
              "device_requested": args.device, "seed": 42,
              "scope": "Official learned RNN prior sampling and SMILES sequence NLL; not task-specific activity/desirability scoring"}
    try:
        import numpy as np
        import torch
        import reinvent
        from rdkit import Chem
        from reinvent.Reinvent import main_script
        from reinvent.runmodes import create_adapter

        data = args.prior.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        if blob != "0dee328238b3d413b5a34c80e3ddab49e4e0af28":
            raise RuntimeError("Prior differs from the pinned official Git blob")
        report.update(prior=str(args.prior), prior_bytes=len(data), prior_git_blob_sha1=blob,
                      prior_sha256=hashlib.sha256(data).hexdigest(), torch_version=torch.__version__,
                      torch_hip=torch.version.hip, torch_cuda=torch.version.cuda,
                      reinvent_module=reinvent.__file__, numpy_version=np.__version__)
        if args.device.startswith("cuda"):
            if not torch.cuda.is_available() or not torch.version.hip:
                raise RuntimeError("ROCm GPU required; no CPU fallback")
            prop = torch.cuda.get_device_properties(0)
            report["gpu"] = {"name": prop.name, "architecture": prop.gcnArchName,
                             "memory_bytes": prop.total_memory}
            if not prop.gcnArchName.startswith("gfx1101"):
                raise RuntimeError("Expected RX7800XT gfx1101")
        executions = Counter()

        def observe(module, inputs):
            if module.__class__.__name__ in ("RNN", "LSTM"):
                parameter_device = str(next(module.parameters()).device)
                input_device = str(inputs[0].device)
                executions[(module.__class__.__name__, parameter_device, input_device)] += 1

        handle = torch.nn.modules.module.register_module_forward_pre_hook(observe)
        config = out / "sampling.toml"
        config.write_text(f'''run_type = "sampling"
device = "{args.device}"
json_out_config = "{out / 'sampling.json'}"
[parameters]
model_file = "{args.prior}"
output_file = "{out / 'samples.csv'}"
num_smiles = 5
unique_molecules = false
randomize_smiles = true
''')
        command = ["reinvent", "-d", args.device, "-s", "42", "-l", str(out / "official_cli.log"), str(config)]
        report["official_entrypoint"] = "reinvent.Reinvent.main_script"
        report["official_cli_argv"] = command
        original_argv = sys.argv
        try:
            sys.argv = command
            main_script()
            with (out / "samples.csv").open() as stream:
                samples = list(csv.DictReader(stream))
            report["samples"] = samples
            report["n_generated"] = len(samples)
            report["n_valid_rdkit"] = sum(Chem.MolFromSmiles(row["SMILES"]) is not None for row in samples)
            report["sampling_neural_forwards"] = [
                {"module": key[0], "parameter_device": key[1], "input_device": key[2], "count": value}
                for key, value in sorted(executions.items())]
            executions.clear()
            adapter, state, model_type = create_adapter(str(args.prior), "inference", torch.device(args.device))
            report.update(model_type=model_type, network=str(adapter.model.network),
                          n_parameters=sum(p.numel() for p in adapter.model.network.parameters()),
                          prior_metadata=str(state.get("metadata")))
            score_smiles = ["CCO", "c1ccccc1", "CC(=O)Oc1ccccc1C(=O)O", "CCN(CC)CC", "Cn1c(=O)c2c(ncn2C)n(C)c1=O"]
            with torch.no_grad():
                nll = adapter.likelihood_smiles(score_smiles)
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            scores = nll.detach().cpu().tolist()
            report["learned_scores"] = [{"smiles": s, "prior_sequence_nll": n}
                                        for s, n in zip(score_smiles, scores)]
            report["scoring_neural_forwards"] = [
                {"module": key[0], "parameter_device": key[1], "input_device": key[2], "count": value}
                for key, value in sorted(executions.items())]
        finally:
            sys.argv = original_argv
            handle.remove()
        records = report["sampling_neural_forwards"] + report["scoring_neural_forwards"]
        report["all_observed_forwards_on_requested_device"] = bool(records) and all(
            entry["parameter_device"] == args.device and entry["input_device"] == args.device for entry in records)
        report["legacy_metadata_hash_warning"] = "has invalid hash" in (out / "official_cli.log").read_text()
        report["prior_file_unchanged"] = args.prior.read_bytes() == data
        report["passed"] = (len(samples) == 5 and report["n_valid_rdkit"] == 5
                            and len(scores) == 5 and all(math.isfinite(n) and n > 0 for n in scores)
                            and report["all_observed_forwards_on_requested_device"] and report["prior_file_unchanged"])
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (out / "seed.json").write_text(json.dumps({"seed": 42, "num_smiles": 5}, indent=2) + "\n")
    print(json.dumps({key: report.get(key) for key in ("passed", "error", "device_requested", "n_generated", "n_valid_rdkit", "sampling_neural_forwards", "scoring_neural_forwards")}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
