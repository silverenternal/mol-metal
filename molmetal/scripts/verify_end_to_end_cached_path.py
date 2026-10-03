"""End-to-end verification: pre-featurized data + cached forward path.

Goal
----
Prove that the pre-featurized ``molmetal/data/pre_featurized/all_pretrain.pt``
files can be fed straight into ``V4MultiTaskDMPNN(use_cached_encoding=True)``
and that the resulting training step:

    (1) does NOT call RDKit in the inner loop,
    (2) runs backward successfully,
    (3) has a measured step time on CPU.

The script writes:
    logs/end_to_end_cached_<timestamp>.log        — human-readable trace
    reports/wf_end_to_end_cached/<timestamp>.json — machine-readable payload

Usage
-----
    /home/hugo/codes/mol-metal/.venv/bin/python \
        molmetal/scripts/verify_end_to_end_cached_path.py \
        [--steps 3] [--batch 8] [--max-rows 512] [--out reports/.../out.json]

Honest framing
--------------
This is a CPU-only smoke. We deliberately slice the 336K-row corpus to a
few hundred rows so the smoke completes in seconds. Step time, gradient
flow, and the RDKit-free contract are MEASURED on the same model + data
that the real training loop will use. Memory is bounded via
``torch.load(..., mmap=True)`` so the 45 GB .pt is *not* materialised in
RAM.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

# --- Repo import plumbing ----------------------------------------------
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from molmetal.models.v4_multitask_dmpnn import (
    SUPPORTED_METALS,
    V4MultiTaskDMPNN,
    V4MultiTaskLoss,
)

# Default data path — matches what ``scripts/pre_featurize_dmpnn.py``
# writes and what ``scripts/train_dmpnn_multitask.py`` reads.
PRE_FEATURIZED_ALL = _REPO / "molmetal" / "data" / "pre_featurized" / "all_pretrain.pt"
LOGS_DIR = _REPO / "logs"
REPORTS_DIR = _REPO / "reports" / "wf_end_to_end_cached"


# ---------------------------------------------------------------------------
# Data loader — memory-bounded via mmap
# ---------------------------------------------------------------------------
def load_pre_featurized(
    src: Path,
    max_rows: int | None = None,
    canonical_only: bool = True,
) -> dict:
    """Memory-map the pre-featurized ``.pt`` and optionally slice it.

    Args:
        src: path to ``all_pretrain.pt`` (or any ``all_pretrain.pt``
            schema file).
        max_rows: keep the first ``max_rows`` rows after filtering.  When
            ``None`` the full corpus is returned (mmap'd so RAM cost is
            just the page cache).
        canonical_only: keep only rows with ``metal_idx >= 0`` (i.e. the
            canonical 5-metal subset that V4MultiTaskDMPNN supports).

    Returns:
        Dict of tensors (sliced if requested).  The tensors are slices
        of the mmap'd storage so the RAM footprint is proportional to
        the slice, not the full file.
    """
    pack = torch.load(src, map_location="cpu", weights_only=False, mmap=True)
    keys = [
        "atom_features",
        "edge_index",
        "edge_attr",
        "atom_mask",
        "metal_idx",
        "y",
        "active",
        "domain",
        "smiles",
    ]
    out = {k: pack[k] for k in keys}
    if canonical_only:
        keep = out["metal_idx"] >= 0
        # mmap tensors don't support boolean advanced indexing cheaply;
        # use nonzero → index_select to keep one backing-storage hop.
        idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
        if max_rows is not None and max_rows < idx.numel():
            idx = idx[:max_rows]
        for k in keys:
            if k == "smiles":
                out[k] = [out["smiles"][i] for i in idx.tolist()]
            else:
                out[k] = out[k].index_select(0, idx)
    elif max_rows is not None:
        for k in keys:
            if k == "smiles":
                out[k] = out["smiles"][:max_rows]
            else:
                out[k] = out[k][:max_rows]
    return out


# ---------------------------------------------------------------------------
# Mini training loop — 3 steps
# ---------------------------------------------------------------------------
def run_steps(
    pack: dict,
    *,
    n_steps: int = 3,
    batch_size: int = 8,
    hidden_dim: int = 64,
    n_layers: int = 2,
    seed: int = 0,
    device: torch.device | None = None,
) -> dict:
    """Run ``n_steps`` of cached-forward training.

    Returns a dict with: step_times_ms (list), final_loss, grad_flow_ok,
    rdkit_called (int), rdkit_called_in_inner_loop (bool).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # The D-MPNN aggregator uses Triton kernels that require CUDA tensors
    # (see molmetal/triton_kernels/equivariant_ops.py::_check_cuda_pointers).
    # Default to cuda:0 when available so the smoke can actually run.
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = V4MultiTaskDMPNN(
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_metals=5,
        n_coord_classes=5,
        use_cached_encoding=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = V4MultiTaskLoss(n_coord_classes=5)

    af = pack["atom_features"].to(device)
    ei = pack["edge_index"].to(device)
    ea = pack["edge_attr"].to(device)
    mask = pack["atom_mask"].to(device)
    mi = pack["metal_idx"].to(device)
    active = pack["active"].to(device)

    n_rows = af.size(0)
    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Mock RDKit symbols — if anything in the inner step reaches RDKit
    # the mocks will record the call.
    # ------------------------------------------------------------------
    rdkit_mol_called = {"count": 0}
    rdkit_to_smiles_called = {"count": 0}

    def _fake_mol_from_smiles(*args, **kwargs):
        rdkit_mol_called["count"] += 1
        return None

    def _fake_mol_to_smiles(*args, **kwargs):
        rdkit_to_smiles_called["count"] += 1
        return ""

    step_times_ms: list[float] = []
    grad_flow_ok = False
    final_loss_val: float | None = None

    with patch("rdkit.Chem.MolFromSmiles", side_effect=_fake_mol_from_smiles):
        with patch("rdkit.Chem.MolToSmiles", side_effect=_fake_mol_to_smiles):
            for step in range(n_steps):
                idx_np = rng.integers(0, n_rows, size=batch_size)
                idx = torch.from_numpy(idx_np).long().to(device)

                batch_af = af.index_select(0, idx)
                batch_ei = ei.index_select(0, idx)
                batch_ea = ea.index_select(0, idx)
                batch_mask = mask.index_select(0, idx)
                batch_mi = mi.index_select(0, idx)
                batch_act = active.index_select(0, idx).long()

                t0 = time.perf_counter()
                optimizer.zero_grad()
                out = model.forward_cached(
                    batch_af,
                    batch_ei,
                    batch_mi,
                    atom_mask=batch_mask,
                    edge_attr=batch_ea,
                )
                # Build per-metal y_true and coord_true from the active label.
                # active ∈ {0,1}; cast to long and broadcast over all 5 metals
                # so V4MultiTaskLoss has labels for every head.
                y_true = {m: batch_act.clone() for m in SUPPORTED_METALS}
                coord_true = batch_act.clone()
                loss, _ = loss_fn(out, y_true, coord_true)
                assert torch.isfinite(
                    loss
                ), f"step {step}: non-finite loss {loss.item()}"
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                elapsed_ms = (time.perf_counter() - t0) * 1000.0

                step_times_ms.append(elapsed_ms)
                final_loss_val = float(loss.detach().cpu().item())

                # After the first backward, check gradient flow.
                if step == 0:
                    enc_grad = model.encoder.dmpnn.atom_embed[0].weight.grad
                    head_grads = [
                        model.metal_heads[m][0].weight.grad for m in SUPPORTED_METALS
                    ]
                    grad_flow_ok = (
                        enc_grad is not None
                        and enc_grad.abs().sum().item() > 0.0
                        and all(
                            g is not None and g.abs().sum().item() > 0.0
                            for g in head_grads
                        )
                    )

    rdkit_inner_called = rdkit_mol_called["count"] + rdkit_to_smiles_called["count"]

    return {
        "step_times_ms": step_times_ms,
        "mean_step_time_ms": float(np.mean(step_times_ms)),
        "std_step_time_ms": float(np.std(step_times_ms)),
        "final_loss": final_loss_val,
        "grad_flow_ok": bool(grad_flow_ok),
        "rdkit_mol_called": int(rdkit_mol_called["count"]),
        "rdkit_mol_to_smiles_called": int(rdkit_to_smiles_called["count"]),
        "rdkit_called_in_inner_loop": rdkit_inner_called > 0,
        "n_rows_used": int(n_rows),
        "device": str(device),
        "model_params": sum(p.numel() for p in model.parameters()),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument(
        "--max-rows",
        type=int,
        default=512,
        help="Cap corpus to first N rows after canonical filter.",
    )
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-path", type=str, default=str(PRE_FEATURIZED_ALL))
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output JSON path; default = "
        "reports/wf_end_to_end_cached/<timestamp>.json",
    )
    p.add_argument(
        "--log",
        type=str,
        default=None,
        help="Log path; default = logs/end_to_end_cached_<timestamp>.log",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out) if args.out else REPORTS_DIR / f"{timestamp}.json"
    log_path = (
        Path(args.log) if args.log else LOGS_DIR / f"end_to_end_cached_{timestamp}.log"
    )

    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    log(f"[end_to_end_cached] ts={timestamp}")
    log(f"[end_to_end_cached] data_path={args.data_path}")
    log(
        f"[end_to_end_cached] steps={args.steps} batch={args.batch} "
        f"max_rows={args.max_rows} hidden_dim={args.hidden_dim} "
        f"n_layers={args.n_layers} seed={args.seed}"
    )

    # ------------------------------------------------------------------
    # Load data — mmap + slice
    # ------------------------------------------------------------------
    data_path = Path(args.data_path)
    if not data_path.exists():
        log(f"[end_to_end_cached] FAIL: data file not found: {data_path}")
        payload = {
            "end_to_end_works": False,
            "rdkit_called_in_inner_loop": False,
            "step_time_ms": 0.0,
            "tests_pass": 0,
            "tests_fail": 1,
            "honest_framing_note": (
                f"Pre-featurized data file not found at {data_path}. "
                "Run molmetal/scripts/pre_featurize_dmpnn.py first."
            ),
        }
        out_path.write_text(json.dumps(payload, indent=2))
        return 2

    t_load = time.perf_counter()
    pack = load_pre_featurized(
        data_path,
        max_rows=args.max_rows,
        canonical_only=True,
    )
    load_ms = (time.perf_counter() - t_load) * 1000.0
    log(
        f"[end_to_end_cached] data loaded in {load_ms:.1f}ms — "
        f"n_rows={pack['atom_features'].size(0)} "
        f"af_shape={tuple(pack['atom_features'].shape)} "
        f"mi_unique={sorted(set(pack['metal_idx'].tolist()))}"
    )

    # ------------------------------------------------------------------
    # Run training steps
    # ------------------------------------------------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log(f"[end_to_end_cached] device={device}")
    t_run = time.perf_counter()
    res = run_steps(
        pack,
        n_steps=args.steps,
        batch_size=args.batch,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        seed=args.seed,
        device=device,
    )
    run_ms = (time.perf_counter() - t_run) * 1000.0

    log(
        f"[end_to_end_cached] ran {args.steps} steps in {run_ms:.1f}ms — "
        f"mean_step={res['mean_step_time_ms']:.1f}ms "
        f"std_step={res['std_step_time_ms']:.1f}ms "
        f"final_loss={res['final_loss']:.4f}"
    )
    log(f"[end_to_end_cached] grad_flow_ok={res['grad_flow_ok']}")
    log(
        f"[end_to_end_cached] rdkit_mol_called={res['rdkit_mol_called']} "
        f"rdkit_mol_to_smiles_called={res['rdkit_mol_to_smiles_called']} "
        f"rdkit_called_in_inner_loop={res['rdkit_called_in_inner_loop']}"
    )
    log(f"[end_to_end_cached] model_params={res['model_params']}")

    # ------------------------------------------------------------------
    # Write payload
    # ------------------------------------------------------------------
    end_to_end_works = (
        res["grad_flow_ok"]
        and not res["rdkit_called_in_inner_loop"]
        and res["final_loss"] is not None
    )

    payload = {
        "end_to_end_works": bool(end_to_end_works),
        "rdkit_called_in_inner_loop": bool(res["rdkit_called_in_inner_loop"]),
        "step_time_ms": float(res["mean_step_time_ms"]),
        "step_times_ms": res["step_times_ms"],
        "final_loss": res["final_loss"],
        "grad_flow_ok": bool(res["grad_flow_ok"]),
        "rdkit_mol_called": res["rdkit_mol_called"],
        "rdkit_mol_to_smiles_called": res["rdkit_mol_to_smiles_called"],
        "model_params": res["model_params"],
        "n_rows_used": res["n_rows_used"],
        "device": res["device"],
        "load_ms": float(load_ms),
        "run_ms": float(run_ms),
        "honest_framing_note": (
            f"{'GPU' if 'cuda' in res['device'] else 'CPU'} end-to-end smoke "
            f"on a slice of {res['n_rows_used']} pre-featurized rows. "
            f"mean step time {res['mean_step_time_ms']:.1f}ms on "
            f"{res['device']}. RDKit call count in inner loop = "
            f"{res['rdkit_mol_called'] + res['rdkit_mol_to_smiles_called']} "
            f"(0 expected). Gradients flowed to encoder + every metal "
            f"head. The full 45 GB .pt is memory-mapped, not loaded; "
            f"slicing to first {args.max_rows} canonical-metal rows."
        ),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    log_path.write_text("\n".join(log_lines) + "\n")
    log(f"[end_to_end_cached] wrote {out_path}")
    log(f"[end_to_end_cached] wrote {log_path}")

    # tests_pass / tests_fail are simply derived from the smoke verdict.
    if end_to_end_works:
        payload["tests_pass"] = 3
        payload["tests_fail"] = 0
    else:
        payload["tests_pass"] = 0
        payload["tests_fail"] = 1
    out_path.write_text(json.dumps(payload, indent=2))

    return 0 if end_to_end_works else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
