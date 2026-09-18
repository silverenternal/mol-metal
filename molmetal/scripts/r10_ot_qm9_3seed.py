"""Small real-QM9 OT on/off experiment using the project's loader and models.

No downloads or synthetic fallback are allowed. Pairing only mixes molecules
with identical ordered atom types and complete graphs, so the existing CFM
coordinate permutation cannot disconnect coordinates from molecular metadata.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import platform
import shlex
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from data.qm9 import QM9Dataset
from flow_matching.loss import ConditionalFlowMatchingLoss
from flow_matching.optimal_transport import mini_batch_ot_coupling
from scripts.train import build_encoder, build_velocity_net, collate_qm9


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def sample_hash(sample):
    digest = hashlib.sha256()
    digest.update(sample.atom_types.cpu().numpy().astype("<i8").tobytes())
    digest.update(sample.coords.cpu().numpy().astype("<f4").tobytes())
    return digest.hexdigest()


def select_groups(train, val, n_groups=8, train_per_group=8, val_per_group=4):
    """Fixed subset of existing splits, grouped by ordered chemical identity.

    Ordered atomic numbers (not molecular graph identity) define compatible
    coordinate matching. Training and validation still contain different
    molecules; graph identities are checked against the original SDF later.
    """
    buckets = []
    for dataset in (train, val):
        grouped = defaultdict(list)
        for index, sample in enumerate(dataset):
            signature = tuple(sample.atom_types.tolist())
            if 4 <= len(signature) <= 16:
                grouped[signature].append(index)
        buckets.append(grouped)
    eligible = sorted((key for key in buckets[0]
                       if len(buckets[0][key]) >= train_per_group
                       and len(buckets[1][key]) >= val_per_group), key=lambda key: (len(key), key))
    if len(eligible) < n_groups:
        raise ValueError("Insufficient compatible real-QM9 groups for the requested protocol")
    keys = eligible[:n_groups]
    return ([buckets[0][key][:train_per_group] for key in keys],
            [buckets[1][key][:val_per_group] for key in keys])


def compatible_batch(items, device, scale):
    """Reuse QM9 centering/padding, with a fixed complete molecular graph."""
    first = items[0].atom_types
    if any(not torch.equal(item.atom_types, first) for item in items):
        raise ValueError("OT groups require identical ordered atom types")
    batch = collate_qm9(items, device, position_scale=scale, edge_cutoff=float("inf"))
    # The complete graph and identical atom types/masks are invariant under
    # a within-group coordinate permutation. Keep the existing padding.
    return batch


def validate_provenance(datasets, selections, sdf_path, check_budget):
    """Match every selected coordinate/atomic-number tensor to the raw SDF."""
    from rdkit import Chem, RDLogger
    from data._base import MoleculeSample

    RDLogger.DisableLog("rdApp.*")
    records = {split: [
        {"split_row_index": index, "sample_sha256": sample_hash(dataset[index])}
        for group in indices for index in group
    ] for split, dataset, indices in zip(("train", "val"), datasets, selections)}
    wanted = {row["sample_sha256"] for values in records.values() for row in values}
    found = {}
    for index, mol in enumerate(Chem.SDMolSupplier(str(sdf_path), removeHs=False)):
        if index % 1000 == 0:
            check_budget()
        if mol is None:
            continue
        sample = MoleculeSample(
            coords=torch.from_numpy(mol.GetConformer().GetPositions().astype(np.float32)),
            atom_types=torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()]),
        )
        key = sample_hash(sample)
        if key in wanted:
            found[key] = {"sdf_record_index": index, "sdf_mol_id": mol.GetProp("_Name"),
                          "canonical_smiles": Chem.MolToSmiles(Chem.RemoveHs(mol))}
        if len(found) == len(wanted):
            break
    if set(found) != wanted:
        raise ValueError(f"{len(wanted - set(found))} selected cache samples absent from original SDF")
    for values in records.values():
        for row in values:
            row.update(found[row["sample_sha256"]])
    train_smiles = {row["canonical_smiles"] for row in records["train"]}
    val_smiles = {row["canonical_smiles"] for row in records["val"]}
    if train_smiles & val_smiles:
        raise ValueError("Selected training and validation molecules have overlapping canonical SMILES")
    return records


def model_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def forward_loss(objective, batch, seed, device, diagnostics=None, ot_settings=None,
                 max_marginal_error=None):
    torch.manual_seed(seed)
    mask = batch["node_mask"].unsqueeze(-1)
    x0 = torch.randn_like(batch["positions"]) * mask
    center = x0.sum(1, keepdim=True) / mask.sum(1, keepdim=True).clamp(min=1)
    x0 = (x0 - center) * mask
    x1 = batch["positions"]
    if ot_settings is not None:
        records = []
        idx = mini_batch_ot_coupling(x1, x0, torch.zeros(len(x0), dtype=torch.long, device=device),
                                    diagnostics=records, **ot_settings)
        if diagnostics is not None:
            diagnostics.extend(records)
        if max_marginal_error is not None:
            for record in records:
                error = record.get("marginal_max_abs_error", float("inf"))
                if record["effective_backend"] != "pot_sinkhorn_log" or error > max_marginal_error:
                    raise RuntimeError(f"Coupling rejected before optimizer update: {record}; "
                                       f"required marginal error <= {max_marginal_error}")
        # Metadata is invariant under these compatible within-group permutations.
        x1 = x1[idx]
    return objective(
        x0, x1, atomic_numbers=batch["atomic_numbers"],
        edge_index=batch["edge_index"], edge_mask=batch["edge_mask"],
        node_mask=batch["node_mask"],
        batch_idx=torch.zeros(len(x0), dtype=torch.long, device=device),
        ot_diagnostics=diagnostics,
    ).loss


def run_branch(seed, enabled, train_batches, val_batches, args, device, check_budget):
    torch.manual_seed(seed)
    cfg = {"model": {"hidden_dim": args.hidden_dim, "n_layers": 2, "encoder_n_layers": 1}}
    model = build_velocity_net(cfg, device)
    encoder = build_encoder(cfg, device)
    # Solve explicitly so the research-only solver settings do not modify
    # global production defaults. forward_loss applies this exact setting
    # once per training step, and the CFM objective never re-couples it.
    objective = ConditionalFlowMatchingLoss(model, encoder=encoder, use_minibatch_ot=False)
    ot_settings = {"reg": args.ot_reg, "max_iter": args.ot_max_iter} if enabled else None
    initial_hash = model_hash(objective)
    optimizer = torch.optim.AdamW(objective.parameters(), lr=args.lr)
    diagnostics = []

    def evaluate():
        # Both treatments evaluated on the same independent-pairing objective.
        objective.eval()
        evaluation = ConditionalFlowMatchingLoss(model, encoder=encoder, use_minibatch_ot=False)
        losses = []
        with torch.no_grad():
            for i, batch in enumerate(val_batches):
                check_budget()
                losses.append(float(forward_loss(evaluation, batch, seed + 100_000 + i, device)))
        return losses

    before = evaluate()
    objective.train()
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    losses = []
    for step in range(args.steps):
        check_budget()
        batch = train_batches[step % len(train_batches)]
        optimizer.zero_grad(set_to_none=True)
        loss = forward_loss(objective, batch, seed + 10_000 + step, device, diagnostics,
                            ot_settings, args.max_marginal_error)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss at seed={seed}, OT={enabled}, step={step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(objective.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    after = evaluate()
    assert all(np.isfinite(before + after))
    if enabled and any(row["effective_backend"] != "pot_sinkhorn_log" for row in diagnostics):
        raise RuntimeError("Real-data protocol requires POT Sinkhorn; fallback observed")
    return {
        "seed": seed, "ot_enabled": enabled, "initial_state_sha256": initial_hash,
        "train_loss_curve": losses, "train_first": losses[0], "train_last": losses[-1],
        "validation_before_batches": before, "validation_after_batches": after,
        "validation_before": statistics.mean(before), "validation_after": statistics.mean(after),
        "training_seconds": seconds, "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "effective_backend_counts": dict(Counter(r["effective_backend"] for r in diagnostics)),
        "fallback_reasons": [r["fallback_reason"] for r in diagnostics if r.get("fallback_reason")],
        "solver_devices": sorted({r.get("plan_device") for r in diagnostics}),
        "cpu_boundaries": sorted({r["cpu_boundary"] for r in diagnostics}),
        "max_marginal_abs_error": max((r.get("marginal_max_abs_error", 0.0)
                                       for r in diagnostics), default=None),
        "solver_settings": ot_settings,
        "marginal_error_gate": args.max_marginal_error,
        "all_couplings_passed_gate": enabled and args.max_marginal_error is not None,
    }


def write_tables(report, prefix):
    """Derive human-readable evidence and raw loss CSV from the saved run."""
    with prefix.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "ot_enabled", "phase", "step_or_batch", "loss"])
        for run in report["runs"]:
            for phase, key in (("train", "train_loss_curve"),
                               ("validation_before", "validation_before_batches"),
                               ("validation_after", "validation_after_batches")):
                writer.writerows([run["seed"], run["ot_enabled"], phase, index, value]
                                 for index, value in enumerate(run[key]))
    lines = ["# Real QM9 three-seed OT training experiment", "",
             f"Execution status: **{report['status']}**. Total wall: {report['total_wall_seconds']:.2f}s.",
             "", "```sh", report["command"], "```", "",
             "This protocol uses real QM9 (DFT-optimized) SDF coordinates, separate from the Gaussian micro-benchmark. "
             "No Vina, molecular generation, or MMP13 experiment was performed.", ""]
    if "protocol" in report:
        lines += ["## Fixed protocol", "", "```json", json.dumps(report["protocol"], indent=2), "```", "",
                  "## Runtime", "", "```json", json.dumps(report["environment"], indent=2), "```", ""]
    lines += ["## Results", "", "| Seed | OT | First train loss | Last train loss | Common val before | Common val after | Train seconds |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for run in report["runs"]:
        lines.append(f"| {run['seed']} | {run['ot_enabled']} | {run['train_first']:.6f} | "
                     f"{run['train_last']:.6f} | {run['validation_before']:.6f} | "
                     f"{run['validation_after']:.6f} | {run['training_seconds']:.3f} |")
    if "aggregate" in report:
        agg = report["aggregate"]
        lines += ["", "Paired common-validation differences (OT minus off; negative is lower): "
                  + ", ".join(f"{d:+.6f}" for d in agg["paired_validation_delta_ot_minus_off"]),
                  f"Mean ± sample std across 3 seeds: **{agg['delta_mean']:+.6f} ± {agg['delta_sample_std']:.6f}** (ddof=1).",
                  "Training losses use different pairing objectives and are not an unbiased between-method evaluation. "
                  "The validation rows use identical held-out samples/noise/times within each seed, with OT disabled for both treatments.", ""]
    lines += ["## Effective backend and numerical residual", "",
              "| Seed | Solver counts | Device | Fallbacks | Maximum marginal absolute error | Peak allocated GPU bytes |",
              "|---|---|---|---|---:|---:|"]
    for run in report["runs"]:
        if run["ot_enabled"]:
            lines.append(f"| {run['seed']} | {run['effective_backend_counts']} | {run['solver_devices']} | "
                         f"{run['fallback_reasons']} | {run['max_marginal_abs_error']:.9f} | {run['peak_gpu_allocated_bytes']} |")
    lines += ["", "POT soft plans and the model execute on GPU. Dataset parsing, group bookkeeping, "
              "and greedy hard-permutation rounding run on CPU. The hard pairing algorithm is unchanged.", "",
              "**Finite soft plans alone do not certify convergence.** The fixed solver settings "
              "must be interpreted alongside the reported marginal residuals. These losses are descriptive "
              "micro-experiment results, not evidence for OT optimality, large-scale model convergence, general superiority, "
              "molecular quality, or better Vina scores. The subset is small and selected for compatible atom signatures.", "",
              "## Provenance and split", ""]
    if "selected_rows" in report:
        lines += ["Every selected cached atom/coordinate tensor was matched byte-for-byte to an original SDF record. "
                  "The JSON contains 64 training and 32 validation split-row indices, sample SHA256s, SDF IDs, "
                  "and canonical SMILES. Training/validation canonical SMILES overlap is zero. "
                  "The project loader's seed-42 80/10/10 split is not described as an official benchmark split.", ""]
    else:
        lines += ["Selected-sample provenance was not fully validated; see failures.", ""]
    lines += ["| Source | SHA256 | Bytes |", "|---|---|---:|"]
    for source in report.get("sources", []):
        lines.append(f"| {source['path']} | {source['sha256']} | {source['bytes']} |")
    lines += ["", "## Failures and budget", "", json.dumps(report["failures"], indent=2), "",
              "An empty failures list means no runtime failure; it does not certify scientific acceptance or solver convergence. "
              "The runner enforces its wall budget and saves each completed branch to JSON. "
              "Hardware/toolchain-dependent GPU reductions may produce small floating-point variation on reproduction."]
    gate = report.get("protocol", {}).get("marginal_error_gate")
    if gate is not None:
        lines += ["", f"This run rejects every soft plan with marginal error >{gate} before "
                  "evaluating its training loss or updating model weights. Completion of all OT branches "
                  "therefore certifies this marginal tolerance for all used training couplings. "
                  "Regularization and iteration budget are fixed throughout each branch; "
                  "validation before and after training uses the same common independent-pairing objective."]
    prefix.with_suffix(".md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seeds", type=int, nargs=3, default=[42, 0, 1234])
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ot-reg", type=float, default=0.05)
    parser.add_argument("--ot-max-iter", type=int, default=200)
    parser.add_argument("--max-marginal-error", type=float, default=None,
                        help="Reject every out-of-tolerance soft plan before its optimizer update")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--budget-seconds", type=float, default=240)
    parser.add_argument("--output-prefix", type=Path,
                        default=ROOT / "molmetal/reports/r10_ot_qm9_3seed")
    args = parser.parse_args(argv)
    if args.steps < 1 or len(set(args.seeds)) != 3 or args.budget_seconds <= 0:
        parser.error("positive steps/budget and three distinct seeds are required")
    if args.ot_reg <= 0 or args.ot_max_iter < 1 or (args.max_marginal_error is not None and args.max_marginal_error <= 0):
        parser.error("OT regularization/iterations and optional marginal tolerance must be positive")
    started = time.perf_counter()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "runs": [], "failures": [],
              "command": shlex.join(["uv", "run", "python", str(Path(__file__).relative_to(ROOT)),
                                     *(sys.argv[1:] if argv is None else argv)])}

    def check_budget():
        if time.perf_counter() - started > args.budget_seconds:
            raise TimeoutError(f"Exceeded {args.budget_seconds}s total experiment budget")

    try:
        if Path.cwd().resolve() != ROOT:
            raise RuntimeError("Run from repository root: existing QM9 loader uses relative data paths")
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("This protocol requires a visible CUDA/ROCm GPU")
        import ot
        from importlib.metadata import version

        paths = [ROOT / "data/qm9_processed.pt", ROOT / "data/qm9_raw/gdb9.sdf",
                 ROOT / "data/qm9_raw/gdb9.sdf.csv"]
        if not all(path.is_file() for path in paths):
            raise FileNotFoundError("Real QM9 cache/SDF/CSV must already be staged; downloads prohibited")
        report["sources"] = [{"path": str(path), "sha256": file_hash(path),
                              "bytes": path.stat().st_size} for path in paths]
        report["source_code_at_run"] = [{"path": str(path.relative_to(ROOT)), "sha256": file_hash(path)}
            for path in [Path(__file__), ROOT / "scripts/train.py", ROOT / "data/qm9.py",
                         ROOT / "models/encoder.py", ROOT / "models/velocity_net.py", ROOT / "models/_scatter.py",
                         ROOT / "flow_matching/loss.py", ROOT / "flow_matching/optimal_transport.py"]]
        report["environment"] = {
            "python": platform.python_version(), "torch": str(torch.__version__),
            "hip": torch.version.hip, "pot": ot.__version__, "scipy": version("scipy"),
            "triton_rocm": version("triton-rocm"), "device": str(device),
            "gpu": torch.cuda.get_device_name(device),
            "arch": getattr(torch.cuda.get_device_properties(device), "gcnArchName", None),
        }
        train, val = QM9Dataset(split="train"), QM9Dataset(split="val")
        if train.source != "real" or val.source != "real":
            raise ValueError("Dataset source is not real QM9; synthetic fallback prohibited")
        selections = select_groups(train, val)
        report["selected_rows"] = validate_provenance((train, val), selections, paths[1], check_budget)
        selected_train = [train[i] for group in selections[0] for i in group]
        scale = statistics.median(float((s.coords - s.coords.mean(0)).std()) for s in selected_train)
        train_batches, val_batches = [
            [compatible_batch([dataset[i] for i in group], device, scale) for group in groups]
            for dataset, groups in zip((train, val), selections)
        ]
        report["protocol"] = {
            "dataset": "QM9 real 3D SDF coordinates; no generated targets",
            "split": "project QM9Dataset 80/10/10 permutation using numpy RNG seed 42; not official benchmark split",
            "train_full_size": len(train), "val_full_size": len(val),
            "train_selected": len(selected_train), "val_selected": sum(map(len, selections[1])),
            "selection": "first 8 shared ordered-atom signatures (4..16 atoms), sorted by length then values; first 8 train/4 val split rows per signature",
            "canonical_smiles_train_val_overlap": 0,
            "preprocess": "train-only median coordinate std, per-molecule centering; complete graphs; 29-atom padding",
            "position_scale": scale, "steps_per_run": args.steps, "seeds": args.seeds,
            "model": "VelocityNet 2 layers + MolEncoder 1 layer, scalar-property conditioning disabled",
            "hidden_dim": args.hidden_dim, "optimizer": "AdamW", "lr": args.lr,
            "gradient_clip_norm": 1.0, "dtype": "float32",
            "ot": f"POT sinkhorn_log, reg={args.ot_reg},max_iter={args.ot_max_iter}, greedy hard projection; one compatible group of 8 per training batch",
            "marginal_error_gate": args.max_marginal_error,
            "ot_objective_note": "raw sum-of-squared-coordinate cost, without normalization; changing reg changes entropic objective; changing iteration count only changes numerical budget",
            "eval": "common independent-pairing CFM objective, fixed noise/times per seed; 8 compatible batches of 4; equal batch weighting",
            "initialization": "matched model/encoder state hash within each seed; fresh training noise seeded identically across branches",
            "budget_seconds": args.budget_seconds,
        }
        for seed in args.seeds:
            for enabled in (False, True):
                print(f"QM9 seed={seed} OT={enabled}", flush=True)
                run = run_branch(seed, enabled, train_batches, val_batches, args, device, check_budget)
                report["runs"].append(run)
                args.output_prefix.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
                print(f"  train={run['train_first']:.4f}->{run['train_last']:.4f}; "
                      f"common val={run['validation_before']:.4f}->{run['validation_after']:.4f}; "
                      f"seconds={run['training_seconds']:.2f}", flush=True)
            off, on = report["runs"][-2:]
            assert off["initial_state_sha256"] == on["initial_state_sha256"]
        delta = [report["runs"][i + 1]["validation_after"] - report["runs"][i]["validation_after"]
                 for i in range(0, 6, 2)]
        report["aggregate"] = {"paired_validation_delta_ot_minus_off": delta,
                               "delta_mean": statistics.mean(delta), "delta_sample_std": statistics.stdev(delta),
                               "seeds": 3, "std_ddof": 1}
        report["status"] = "complete"
    except Exception as exc:
        report["status"] = "failed"
        report["failures"].append({"type": type(exc).__name__, "message": str(exc),
                                   "traceback": traceback.format_exc()})
        print(report["failures"][-1]["traceback"], file=sys.stderr)
    report["total_wall_seconds"] = time.perf_counter() - started
    args.output_prefix.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    write_tables(report, args.output_prefix)
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
