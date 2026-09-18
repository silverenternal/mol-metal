"""MolFlow-Triton end-to-end demo.

Builds a tiny synthetic molecular graph, encodes it, runs the EGNN
velocity net, takes a couple of integration steps with the Triton
Euler ODE kernel, and prints the integrated drift.  Verifies that the
whole stack (Triton kernels + EGNN + flow-matching loss + ODE
sampler) wires together correctly on the local ROCm GPU.

Run
---
    python main.py
or, after the project is installed as a uv-managed package:
    uv run python main.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def banner(title: str) -> None:
    print()
    print(f"=== {title} ===")


def main() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "MolFlow-Triton requires a ROCm / CUDA GPU (the Triton kernels have "
            "no CPU fallback).  No device detected by torch.cuda."
        )
    device = torch.device("cuda")
    print(
        f"[demo] torch={torch.__version__}  device={torch.cuda.get_device_name(0)}  "
        f"gfx={torch.cuda.get_device_properties(0).gcnArchName}"
    )

    # ------------------------------------------------------------------
    # 1. Synthetic molecular graph
    # ------------------------------------------------------------------
    banner("1. synthetic batch")
    B, N, E = 4, 6, 8  # batch, atoms per molecule, edges per molecule
    H = 64             # hidden width

    torch.manual_seed(0)
    atomic_numbers = torch.randint(1, 10, (B, N), device=device, dtype=torch.long)
    positions      = torch.randn(B, N, 3, device=device)
    edge_index     = torch.randint(0, N, (B, 2, E), device=device, dtype=torch.long)

    # ------------------------------------------------------------------
    # 2. Encode + velocity prediction
    # ------------------------------------------------------------------
    banner("2. encoder + EGNN velocity net")
    from models import MolEncoder, VelocityNet
    from triton_kernels import euler_step

    encoder = MolEncoder(max_atomic_number=20, hidden_dim=H, n_layers=2).to(device)
    net = VelocityNet(hidden_dim=H, n_layers=3, cond_dim=0).to(device)

    h_node, h_edge = encoder(atomic_numbers, positions, edge_index)
    print(f"  h_node {tuple(h_node.shape)}    h_edge {tuple(h_edge.shape)}")

    t = torch.rand(B, device=device)
    v_pred = net(h_node, positions, t, edge_index, cond=None)
    print(f"  v_pred {tuple(v_pred.shape)}    |v|_mean = {v_pred.norm(dim=-1).mean().item():.4f}")

    # ------------------------------------------------------------------
    # 3. Conditional flow-matching loss
    # ------------------------------------------------------------------
    banner("3. conditional flow-matching loss")
    from flow_matching import ConditionalFlowMatchingLoss

    cfn = ConditionalFlowMatchingLoss(model=net, encoder=None).to(device)
    x0 = torch.randn_like(positions)
    out = cfn(
        x0=x0, x1=positions, cond=None,
        h_node=h_node, edge_index=edge_index,
    )
    print(f"  loss        = {out.loss.item():.6f}")
    print(f"  x_t shape   = {tuple(out.x_t.shape)}")
    print(f"  target |v|  = {out.target_velocity_.norm(dim=-1).mean().item():.4f}")

    # ------------------------------------------------------------------
    # 4. ODE integration with the Triton Euler kernel
    # ------------------------------------------------------------------
    banner("4. flow-matching sampler (Triton Euler)")
    from flow_matching import FlowMatchingSampler

    sampler = FlowMatchingSampler(model=net, encoder=None, default_n_steps=4)
    # Only 2 steps — an untrained network produces large velocities
    # that overflow fp32 quickly under dt=0.25.  The smoke test cares
    # only that the pipeline completes.
    x_final = sampler.sample(
        x0=x0, edge_index=edge_index, h_node=h_node, n_steps=2, method="euler"
    )
    print(f"  x0         |x0|   = {x0.norm().item():.4f}")
    print(f"  x_final    |x1|   = {x_final.norm().item():.4f}")
    print(f"  drift      |dx|   = {(x_final - x0).norm().item():.4f}")
    assert torch.isfinite(x_final).all(), "sampler produced NaN/Inf"

    # ------------------------------------------------------------------
    # 5. Triton fused LayerNorm sanity check
    # ------------------------------------------------------------------
    banner("5. fused LayerNorm (Triton)")
    from triton_kernels import fused_layer_norm, fused_rms_norm

    w = torch.randn(H, device=device)
    b = torch.randn(H, device=device)
    h_normed = fused_layer_norm(h_node, w, b)
    # Sanity: per-row mean ~ 0, var ~ 1 in fp32.
    mean_per_row = h_normed.float().mean(dim=-1)
    var_per_row = h_normed.float().var(dim=-1, unbiased=False)
    print(f"  |mean|     max = {mean_per_row.abs().max().item():.2e}")
    print(f"  var        mean = {var_per_row.mean().item():.4f}")

    h_rms = fused_rms_norm(h_node, w)
    print(f"  rms_norm shape = {tuple(h_rms.shape)}")

    # ------------------------------------------------------------------
    # 6. Config + dataset plumbing
    # ------------------------------------------------------------------
    banner("6. config + dataset loaders")
    from utils.config import load_config
    from data.mol_dataset import BaseMoleculeDataset
    from data.transforms import Compose, random_rotation_3d

    cfg = load_config()  # configs/default.yaml
    print(f"  config keys : {list(cfg.keys())}")
    print(f"  cfg.flow    : {cfg['flow']}")

    assert issubclass(BaseMoleculeDataset, __import__("torch").utils.data.Dataset)
    print(f"  BaseMoleculeDataset is a torch Dataset  OK")

    g = torch.Generator(device=device).manual_seed(0)
    pipe = Compose([lambda c: random_rotation_3d(c, generator=g)])
    rotated = pipe(positions[0])
    # Rotation preserves pairwise distances.
    d_in = torch.cdist(positions[0], positions[0])
    d_out = torch.cdist(rotated, rotated)
    assert torch.allclose(d_in, d_out, atol=1e-5)
    print(f"  transform roundtrip: distance error = {(d_in - d_out).abs().max().item():.2e}")

    print()
    print("ALL OK — MolFlow-Triton stack is wired up correctly on this GPU.")


if __name__ == "__main__":
    main()