"""Official REINVENT prior likelihood worker; explicit CPU or ROCm, no proxy."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path


class PriorWorker:
    def __init__(self, args):
        with contextlib.redirect_stdout(sys.stderr):
            import torch
            from rdkit import Chem
            from reinvent.runmodes import create_adapter
            from reinvent.version import __version__
            self.torch, self.Chem = torch, Chem
            actual_sha = hashlib.sha256(args.prior.read_bytes()).hexdigest()
            if actual_sha != args.prior_sha256:
                raise ValueError("prior_sha256_mismatch")
            self.metadata = {"protocol": "reinvent-prior-jsonl-v1", "backend": "reinvent-prior",
                             "model_sha256": actual_sha, "model_path": str(args.prior.resolve()),
                             "device": args.device, "torch_version": torch.__version__,
                             "torch_hip": torch.version.hip, "reinvent_version": __version__,
                             "mode": "prior_nll", "sequence_policy": "input SMILES unchanged; length buckets avoid added EOS padding; count includes one EOS, excludes BOS"}
            if args.device.startswith("cuda"):
                if not torch.cuda.is_available() or not torch.version.hip:
                    raise RuntimeError("requested_rocm_gpu_unavailable")
                props = torch.cuda.get_device_properties(0)
                self.metadata.update(architecture=props.gcnArchName, gpu_name=props.name,
                                     gpu_memory_bytes=props.total_memory)
            self.adapter, state, model_type = create_adapter(str(args.prior), "inference", torch.device(args.device))
            if model_type != "Reinvent":
                raise ValueError(f"unsupported_prior_type:{model_type}")
            self.metadata.update(model_type=model_type,
                                 n_parameters=sum(p.numel() for p in self.adapter.model.network.parameters()))
            from reinvent.models import check_valid_hash
            self.metadata["legacy_internal_metadata_hash_valid"] = bool(check_valid_hash(state))
            self.metadata["parameter_devices"] = sorted({str(p.device) for p in self.adapter.model.network.parameters()})
            if self.metadata["parameter_devices"] != [args.device]:
                raise RuntimeError("parameter_device_mismatch")

    def likelihood(self, smiles):
        if not isinstance(smiles, list):
            raise ValueError("smiles_must_be_list")
        if len(smiles) > 256:
            raise ValueError("batch_size_exceeds_256")
        results, buckets = [], defaultdict(list)
        for index, value in enumerate(smiles):
            row = {"index": index, "smiles": value, "status": "error", "prior_nll": None,
                   "predicted_token_count": None, "error": None}
            results.append(row)
            try:
                if not isinstance(value, str) or not value.strip():
                    raise ValueError("empty_or_nonstring_smiles")
                if len(value) > 4096:
                    raise ValueError("smiles_exceeds_4096_characters")
                mol = self.Chem.MolFromSmiles(value)
                if mol is None or mol.GetNumAtoms() == 0:
                    raise ValueError("invalid_smiles")
                tokens = self.adapter.model.tokenizer.tokenize(value)
                try:
                    self.adapter.model.vocabulary.encode(tokens)
                except (KeyError, ValueError) as exc:
                    raise ValueError("unsupported_prior_token") from exc
                buckets[len(tokens)].append(index)
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}:{exc}"
        forwards = Counter()

        def observe(module, inputs):
            forwards[(str(next(module.parameters()).device), str(inputs[0].device))] += 1

        hook = self.adapter.model.network.register_forward_pre_hook(observe)
        try:
            for token_length, indices in buckets.items():
                try:
                    with self.torch.no_grad():
                        values = self.adapter.likelihood_smiles([smiles[i] for i in indices]).detach().cpu().tolist()
                    if len(values) != len(indices):
                        raise RuntimeError("model_output_count_mismatch")
                    for index, nll in zip(indices, values):
                        if not math.isfinite(nll) or nll < 0:
                            results[index]["error"] = "nonfinite_or_negative_model_nll"
                        else:
                            results[index].update(status="ok", prior_nll=nll,
                                                  predicted_token_count=token_length-1, error=None)
                except Exception as exc:
                    for index in indices:
                        results[index]["error"] = f"inference_error:{type(exc).__name__}:{exc}"
        finally:
            hook.remove()
        metadata = dict(self.metadata)
        metadata["neural_forwards"] = [{"parameter_device": p, "input_device": i, "count": n}
                                       for (p, i), n in sorted(forwards.items())]
        metadata["n_requested"] = len(smiles)
        metadata["n_scored"] = sum(row["status"] == "ok" for row in results)
        if any(p != self.metadata["device"] or i != self.metadata["device"] for p, i in forwards):
            raise RuntimeError("neural_execution_device_mismatch")
        return {"metadata": metadata, "results": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--prior-sha256", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda:0"], required=True)
    args = parser.parse_args()
    worker, startup_error = None, None
    try:
        worker = PriorWorker(args)
    except Exception as exc:
        startup_error = f"initialization_failed:{type(exc).__name__}:{exc}"
    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("request_must_be_object")
            if startup_error:
                raise RuntimeError(startup_error)
            with contextlib.redirect_stdout(sys.stderr):
                if request.get("op") == "capabilities":
                    response = {"metadata": worker.metadata}
                elif request.get("op") == "likelihood":
                    response = worker.likelihood(request.get("smiles"))
                else:
                    raise ValueError("unsupported_operation")
        except Exception as exc:
            response = {"error": f"{type(exc).__name__}:{exc}"}
        response["id"] = request.get("id") if isinstance(request, dict) else None
        sys.stdout.write(json.dumps(response, allow_nan=False) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
