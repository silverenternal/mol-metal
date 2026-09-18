"""Round-10 axis B synthetic micro-ablation: independent vs mini-batch OT.

A *tiny* (<=5 min wall) ablation that compares
``ConditionalFlowMatchingLoss(use_minibatch_ot=False)`` against
``ConditionalFlowMatchingLoss(use_minibatch_ot=True)`` for 50 epochs
on Gaussian batches of shape (B=8, N=32, 3). Despite the historical filename,
no 1h36 receptor or ligand data is loaded. The legacy ``vanilla_OT`` output
label denotes independent ordering, not a full-batch OT baseline.

Outputs
-------
- CSV:    ``molmetal/reports/r10_ot_ablation_1h36.csv`` — epoch, vanilla_loss, minibatch_loss
- JSON:   ``molmetal/reports/r10_ot_ablation_1h36.json`` — summary + wall time
- MD:     ``molmetal/reports/r10_ot_ablation_1h36.md`` — human-readable report
- stdout: per-epoch loss curve for both branches

Usage
-----
::

    uv run python molmetal/scripts/r10_ot_ablation_1h36.py --epochs 50
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import shlex
import statistics
from importlib.metadata import PackageNotFoundError, version
import sys
import time
from collections import Counter
from pathlib import Path

import torch

# Ensure project root is on sys.path so we can import ``flow_matching``
# (the MolFlow-Triton layout puts the loss module at the project root).
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flow_matching.loss import ConditionalFlowMatchingLoss
from models.velocity_net import VelocityNet

REPORTS = PROJECT_ROOT / "molmetal" / "reports"


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _build_edge_index(b: int, n: int) -> torch.Tensor:
    """Fully-connected directed (no-self-loop) edge_index, shape (B, 2, n*(n-1))."""
    idx = torch.arange(n)
    src = idx.view(1, n, 1).expand(b, n, n)
    dst = idx.view(1, 1, n).expand(b, n, n)
    mask = src != dst
    src = src[mask].view(b, -1)
    dst = dst[mask].view(b, -1)
    return torch.stack([src, dst], dim=1).to(dtype=torch.long)


def _train_branch(
    *,
    use_minibatch_ot: bool,
    b: int,
    n: int,
    epochs: int,
    seed: int,
    hidden_dim: int,
    lr: float,
    device: torch.device,
    diagnostics: dict | None = None,
) -> tuple[list[float], float]:
    """Run 50 epochs of CFM training and return (loss_curve, wall_seconds)."""
    torch.manual_seed(seed)
    velocity_net = VelocityNet(hidden_dim=hidden_dim, n_layers=2, cond_dim=0).to(device)
    loss_module = ConditionalFlowMatchingLoss(
        model=velocity_net, use_minibatch_ot=use_minibatch_ot,
    ).to(device)
    optimizer = torch.optim.AdamW(velocity_net.parameters(), lr=lr)

    n_edges = n * (n - 1)
    edge_index = _build_edge_index(b, n).to(device)
    edge_mask = torch.ones(b * n_edges, dtype=torch.bool, device=device)
    node_mask = torch.ones(b, n, dtype=torch.bool, device=device)
    # Use real mini-batches: each group contains several ligands.  The
    # previous harness used ``arange(b)``, making every group a singleton
    # and reducing mini-batch OT to an identity permutation.  Grouping
    # samples exposes the semantic pairing effect described by Tong et al.
    if b % 2:
        raise ValueError("batch size must be even for the paired OT ablation")
    batch_idx = torch.arange(b, dtype=torch.long, device=device) // 2

    h_node = torch.randn(b, n, hidden_dim, dtype=torch.float32, device=device)

    losses: list[float] = []
    changed_rows = []
    target_costs = []
    ot_records = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for epoch in range(epochs):
        torch.manual_seed(seed + epoch)
        x0 = torch.randn(b, n, 3, dtype=torch.float32, device=device)
        x1 = torch.randn(b, n, 3, dtype=torch.float32, device=device)

        out = loss_module(
            x0, x1, cond=None, h_node=h_node,
            edge_index=edge_index, edge_mask=edge_mask, node_mask=node_mask,
            batch_idx=batch_idx,
            ot_diagnostics=ot_records if diagnostics is not None else None,
        )
        loss = out.loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(velocity_net.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.item()))
        if diagnostics is not None:
            # The loss exposes the actual coupled target velocity, so this
            # measures the pairing used for training without solving OT twice.
            changed_rows.append(int(((out.target_velocity_ - (x1 - x0))
                                     .abs().flatten(1).amax(1) > 1e-5).sum().item()))
            target_costs.append(float(out.target_velocity_.square().mean().item()))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall = time.perf_counter() - t0
    if diagnostics is not None:
        diagnostics.update({
            "repaired_rows_per_step": changed_rows,
            "repaired_row_fraction": sum(changed_rows) / (b * epochs),
            "target_velocity_mse_per_step": target_costs,
            "target_velocity_mse_mean": statistics.mean(target_costs),
            "loss_mean": statistics.mean(losses),
            "loss_std_over_steps": statistics.pstdev(losses),
            "last_10_loss_mean": statistics.mean(losses[-10:]),
            "effective_backend_counts": dict(Counter(r["effective_backend"] for r in ot_records)),
            "fallback_reason_counts": dict(Counter(r["fallback_reason"] for r in ot_records
                                                    if r.get("fallback_reason"))),
            "cpu_boundaries": sorted({r["cpu_boundary"] for r in ot_records}),
            "solver_devices": sorted({r.get("plan_device", r["device"]) for r in ot_records}),
            "max_marginal_abs_error": max((r.get("marginal_max_abs_error", 0.0)
                                           for r in ot_records), default=None),
        })
    return losses, wall


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="r10-ot-ablation-1h36",
        description=(
            "Round-10 axis B micro-ablation: vanilla_OT vs mini_batch_OT "
            "on a 1h36-pocket-shaped batch.  Wall-time budget: <=5 min "
            "for 50 epochs."
        ),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--b", type=int, default=8, help="batch size (default 8)")
    parser.add_argument("--n", type=int, default=32, help="atoms per ligand (default 32 — 1h36-shape)")
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device", type=str, default=None,
        help="torch device (default: cuda when ROCm/CUDA is available, else cpu)",
    )
    parser.add_argument("--output-prefix", type=str,
                        default=str(REPORTS / "r10_ot_ablation_1h36"))
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.b < 2 or args.b % 2 or args.n < 2:
        parser.error("epochs >=1, even b >=2, and n >=2 are required")

    requested_device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")
    print(f"[r10-ot-ablation] device={device} epochs={args.epochs} "
          f"b={args.b} n={args.n} hidden_dim={args.hidden_dim}")

    vanilla_diagnostics, mb_diagnostics = {}, {}
    print("[r10-ot-ablation] running vanilla_OT branch...")
    vanilla_losses, vanilla_wall = _train_branch(
        use_minibatch_ot=False,
        b=args.b, n=args.n, epochs=args.epochs,
        seed=args.seed, hidden_dim=args.hidden_dim, lr=args.lr, device=device,
        diagnostics=vanilla_diagnostics,
    )
    print(f"[r10-ot-ablation] vanilla_OT done in {vanilla_wall:.2f}s, "
          f"first loss={vanilla_losses[0]:.4f}, last loss={vanilla_losses[-1]:.4f}")

    print("[r10-ot-ablation] running minibatch_OT branch...")
    mb_losses, mb_wall = _train_branch(
        use_minibatch_ot=True,
        b=args.b, n=args.n, epochs=args.epochs,
        seed=args.seed, hidden_dim=args.hidden_dim, lr=args.lr, device=device,
        diagnostics=mb_diagnostics,
    )
    print(f"[r10-ot-ablation] minibatch_OT done in {mb_wall:.2f}s, "
          f"first loss={mb_losses[0]:.4f}, last loss={mb_losses[-1]:.4f}")

    # Sanity: both branches must show loss decrease (the spec's
    # success criterion "log shows loss decrease").
    v_decrease = vanilla_losses[0] - vanilla_losses[-1]
    m_decrease = mb_losses[0] - mb_losses[-1]
    print(f"[r10-ot-ablation] vanilla_OT delta = {v_decrease:+.4f}")
    print(f"[r10-ot-ablation] minibatch_OT delta = {m_decrease:+.4f}")

    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    # CSV: epoch, vanilla_loss, minibatch_loss, abs_diff
    csv_path = out_prefix.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "vanilla_ot_loss", "minibatch_ot_loss", "abs_diff"])
        for ep, (lv, lm) in enumerate(zip(vanilla_losses, mb_losses)):
            w.writerow([ep, f"{lv:.6f}", f"{lm:.6f}", f"{abs(lv - lm):.6f}"])
    print(f"[r10-ot-ablation] wrote {csv_path}")

    # JSON: full summary
    summary = {
        "epochs": args.epochs,
        "b": args.b,
        "n": args.n,
        "hidden_dim": args.hidden_dim,
        "lr": args.lr,
        "device": str(device),
        "seed": args.seed,
        "command": shlex.join(["uv", "run", "python", "molmetal/scripts/r10_ot_ablation_1h36.py",
                               *(sys.argv[1:] if argv is None else argv)]),
        "dataset": "synthetic Gaussian coordinates; no real 1h36 ligand or receptor loaded",
        "pairing": "groups of 2 samples; baseline preserves independent input ordering",
        "environment": {
            "python": platform.python_version(), "torch": str(torch.__version__),
            "hip": torch.version.hip,
            "triton_rocm": _package_version("triton-rocm"),
            "pot": _package_version("POT"), "scipy": _package_version("scipy"),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "gpu_arch": getattr(torch.cuda.get_device_properties(device), "gcnArchName", None)
            if device.type == "cuda" else None,
        },
        "vanilla_ot": {
            "wall_seconds": vanilla_wall,
            "first_loss": vanilla_losses[0],
            "last_loss": vanilla_losses[-1],
            "delta": v_decrease,
            "loss_curve": vanilla_losses,
            **vanilla_diagnostics,
        },
        "minibatch_ot": {
            "wall_seconds": mb_wall,
            "first_loss": mb_losses[0],
            "last_loss": mb_losses[-1],
            "delta": m_decrease,
            "loss_curve": mb_losses,
            **mb_diagnostics,
        },
        "success": {
            "vanilla_loss_decreased": bool(v_decrease > 0),
            "minibatch_loss_decreased": bool(m_decrease > 0),
            "wall_under_5min": bool((vanilla_wall + mb_wall) < 300.0),
        },
    }
    json_path = out_prefix.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[r10-ot-ablation] wrote {json_path}")

    # MD: human-readable summary
    md_path = out_prefix.with_suffix(".md")
    with open(md_path, "w") as f:
        f.write("# Round-10 axis B micro-ablation — synthetic coordinates\n\n")
        f.write(f"Reproduce: `{summary['command']}`\n\n")
        f.write(f"- Epochs: **{args.epochs}**\n")
        f.write(f"- Batch size: **{args.b}**, atoms/ligand: **{args.n}**, hidden_dim: **{args.hidden_dim}**\n")
        f.write(f"- Device: **{device}**, lr: {args.lr}\n")
        f.write(f"- Seed: **{args.seed}**, environment: {summary['environment']}\n")
        f.write("- Data: synthetic Gaussian coordinates; no real 1h36 data loaded.\n")
        f.write("- The legacy `vanilla_OT` label means independent ordering (OT disabled), "
                "not full-batch optimal transport.\n")
        f.write("- OT groups: **pairs of samples** (batch_idx = arange(B) // 2)\n\n")
        f.write("## Results\n\n")
        f.write(f"- vanilla_OT wall: {vanilla_wall:.2f}s, "
                f"first={vanilla_losses[0]:.4f}, last={vanilla_losses[-1]:.4f}, "
                f"delta={v_decrease:+.4f}\n")
        f.write(f"- minibatch_OT wall: {mb_wall:.2f}s, "
                f"first={mb_losses[0]:.4f}, last={mb_losses[-1]:.4f}, "
                f"delta={m_decrease:+.4f}\n\n")
        for name, metrics in (("independent", vanilla_diagnostics), ("minibatch", mb_diagnostics)):
            f.write(f"- {name}: loss mean={metrics['loss_mean']:.6f}, "
                    f"std across steps={metrics['loss_std_over_steps']:.6f}, "
                    f"last-10 mean={metrics['last_10_loss_mean']:.6f}, "
                    f"re-paired fraction={metrics['repaired_row_fraction']:.3f}, "
                    f"target velocity MSE={metrics['target_velocity_mse_mean']:.6f}\n")
            f.write(f"  Effective backends: {metrics['effective_backend_counts']}; "
                    f"fallbacks: {metrics['fallback_reason_counts']}; "
                    f"solver devices: {metrics['solver_devices']}; "
                    f"CPU boundaries: {metrics['cpu_boundaries']}; "
                    f"maximum marginal error: {metrics['max_marginal_abs_error']}\n")
        f.write("\nSingle-seed training smoke only. Step-to-step std is not uncertainty "
                "across seeds. Loss/transport-cost changes do not establish a molecular "
                "quality or Vina improvement, nor the full-batch OT parity criterion.\n\n")
        f.write("## Success criteria\n\n")
        f.write(f"- vanilla loss decreased: **{bool(v_decrease > 0)}**\n")
        f.write(f"- minibatch loss decreased: **{bool(m_decrease > 0)}**\n")
        f.write(f"- total wall under 5 min: **{bool((vanilla_wall + mb_wall) < 300.0)}** "
                f"(total {(vanilla_wall + mb_wall):.2f}s)\n")
    print(f"[r10-ot-ablation] wrote {md_path}")

    if not (v_decrease > 0 and m_decrease > 0):
        print(f"[r10-ot-ablation] WARNING: one branch did not show loss decrease "
              f"(vanilla_delta={v_decrease:+.4f}, mb_delta={m_decrease:+.4f})")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
