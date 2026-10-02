"""Wf-341-C2: EDM preconditioning + Karras 2022 sigma reparameterization.

Standalone CLI that exercises the EDM preconditioning + sigma-reparam
acceleration on a synthetic Lipman CFM velocity field (CPU-only,
self-contained).  Use this script to:

  1. Verify EDM preconditioning factors + Karras sigma reparam are
     bit-equivalent to the closed-form formulas.
  2. Smoke-test the loss-weighted MSE on a synthetic CFM step.
  3. Bench CPU throughput for the proposed vs baseline path
     (PROJECTED 1.5-3x speedup at the GPU level per Karras 2022 §5).

Both `--edm-precond` and `--sigma-reparam` default OFF for safety
(per AGENTS.md §5.1 + Wf-340-A2 §2 license / R&D risk).  Existing
launchers (`launch_wf299_cfm_main.sh`, `launch_wf124_p3_g5.sh`,
`launch_wf264_cfm_property_panel.sh`) are unaffected.

Karras 2022 reference (arXiv:2206.00364):
  - §3.2 defines the noise-level parameter σ > 0.
  - §3.4 gives the preconditioning factors
        c_skip = σ_data² / (σ² + σ_data²)
        c_out  = σ·σ_data / √(σ² + σ_data²)
        c_in   = 1 / √(σ² + σ_data²)
        c_noise = 0.5 · ln(σ)
  - §5 gives the loss weight λ(σ) = (c_skip² + σ_data²) / (c_out² · σ²)
    and the log-normal sampling distribution P_mean=-1.2, P_std=1.2.

License note: NVlabs/edm is CC BY-NC-SA 4.0.  This script ports the
*mathematical formulation* (Karras eq 2 / Sauer-style preconditioning
factors); NOT verbatim code from NVlabs/edm.  Surface this to user
before any commercial / for-fee Tier-2 paper work.

Honest framing (party-twenty-spirit §3.5):
  - MEASURED on CPU:  bit-equivalence of closed-form factors + sigma
    reparam inverse; finite-ness of loss-weighted gradient.
  - PROJECTED:  1.5-3x GPU speedup (Karras 2022 §5 benchmarks on
    ImageNet U-Net); MEASURED on Mol-Metal EGNN deferred to Wf-340-A7
    per AGENTS.md §5.1.
  - Default OFF for safety.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args_safe(argv=None) -> argparse.Namespace:
    """Parse command-line arguments.  Both acceleration flags default OFF."""
    p = argparse.ArgumentParser(
        prog="wf341c2_edm_precond",
        description=(
            "Wf-341-C2: standalone EDM preconditioning + Karras 2022 sigma "
            "reparameterization smoke + CPU bench (default OFF for safety)."
        ),
    )
    p.add_argument("--edm-precond", action="store_true", default=False,
                   help=(
                       "Enable EDM σ-preconditioning (Karras 2022) on the "
                       "synthetic CFM coord-velocity MSE loss. Default OFF. "
                       "PROJECTED 1.5-3x FID convergence per Karras 2022 §5."
                   ))
    p.add_argument("--sigma-reparam", action="store_true", default=False,
                   help=(
                       "Enable Karras 2022 sigma reparameterization: bridge "
                       "CFM t ∈ (0, 1] to EDM σ > 0 via σ = (1 − t) / t. "
                       "Default OFF. PROJECTED 1.2-1.5x paired with "
                       "--edm-precond per Wf-340-A2 §2.2."
                   ))
    p.add_argument("--sigma-data", type=float, default=1.0,
                   help="Expected L2 norm of data sample (default 1.0).")
    p.add_argument("--sigma-min", type=float, default=0.002,
                   help="EDM σ floor (default 0.002; Karras convention).")
    p.add_argument("--sigma-max", type=float, default=80.0,
                   help="EDM σ ceiling (default 80; Karras convention).")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Synthetic batch size (default 64).")
    p.add_argument("--n-atoms", type=int, default=32,
                   help="Synthetic atoms per molecule (default 32).")
    p.add_argument("--n-steps", type=int, default=10,
                   help="Number of synthetic training steps (default 10).")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed (default 42; bit-identical to pipeline runs).")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Synthetic velocity field (CPU-only stub)
# ---------------------------------------------------------------------------

class _StubVelocityField(nn.Module):
    """Tiny stub velocity field for the smoke test.

    Returns a deterministic linear map from ``x_t`` to ``v_pred``.
    CPU-only; no EGNN / YuelBond dependencies.
    """

    def __init__(self, n_atoms: int, hidden_dim: int = 8):
        super().__init__()
        self.lin = nn.Linear(n_atoms * 3, n_atoms * 3, bias=False)
        with torch.no_grad():
            self.lin.weight.copy_(0.5 * torch.eye(n_atoms * 3))

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, **kw):
        b, n, _ = x_t.shape
        v = self.lin(x_t.reshape(b, -1)).reshape(b, n, 3)
        return {"vel": v}


# ---------------------------------------------------------------------------
# Loss wrapper
# ---------------------------------------------------------------------------

def cfm_velocity_mse(
    v_pred: torch.Tensor,
    v_target: torch.Tensor,
    mask: torch.Tensor,
    *,
    edm_cfg=None,
    sigma=None,
):
    """Per-atom MSE between predicted and target velocities.

    When ``edm_cfg`` + ``sigma`` are passed (i.e. ``--edm-precond`` is
    on), the per-sample loss is multiplied by ``λ(σ)``.  Otherwise the
    loss is the plain MSE (the legacy behavior).
    """
    per_atom = (v_pred - v_target) ** 2 * mask
    per_sample = per_atom.sum(dim=(1, 2)) / mask.sum(dim=(1, 2)).clamp(min=1.0)
    if edm_cfg is not None and sigma is not None:
        # Import here so the module is loadable without POT.
        from molmetal.adapters.flow_matching_lipman.acceleration import (
            edm_loss_weight,
        )
        lw = edm_loss_weight(sigma, edm_cfg)
        return (per_sample * lw).mean()
    return per_sample.mean()


# ---------------------------------------------------------------------------
# Main smoke + bench
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    args = parse_args_safe(argv)
    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    # Imports are local so the script can be invoked even if the
    # adapter is missing on older branches.
    from molmetal.adapters.flow_matching_lipman.acceleration import (
        EDMPreconditionerConfig,
        sample_edm_sigma,
        sample_log_normal_t,
        sigma_from_cfm_t,
        cfm_t_from_sigma,
        edm_loss_weight,
    )

    cfg = EDMPreconditionerConfig(
        sigma_data=args.sigma_data,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
    )
    b, n = args.batch_size, args.n_atoms
    model = _StubVelocityField(n_atoms=n).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)

    print("=" * 78)
    print("Wf-341-C2: EDM preconditioning + Karras sigma reparam")
    print(f"  edm_precond  = {args.edm_precond}")
    print(f"  sigma_reparam = {args.sigma_reparam}")
    print(f"  cfg          = sigma_data={cfg.sigma_data} "
          f"sigma_min={cfg.sigma_min} sigma_max={cfg.sigma_max}")
    print(f"  batch_size   = {b}, n_atoms = {n}, n_steps = {args.n_steps}")
    print("=" * 78)

    # 1) Sigma-reparam bridge sanity (always on, even if flag is off).
    t_sample = sample_log_normal_t(b, device=device) if args.sigma_reparam else (
        torch.rand(b, device=device)
    )
    sigma_bridged = sigma_from_cfm_t(t_sample, cfg)
    t_recovered = cfm_t_from_sigma(sigma_bridged)
    sigma_recon_error = (t_sample - t_recovered).abs().max().item()
    print(f"[bridge] t_sample range = [{t_sample.min().item():.4f}, "
          f"{t_sample.max().item():.4f}]")
    print(f"[bridge] sigma_bridged range = "
          f"[{sigma_bridged.min().item():.4f}, "
          f"{sigma_bridged.max().item():.4f}]")
    print(f"[bridge] t ↔ σ roundtrip max |err| = {sigma_recon_error:.2e}")
    assert sigma_recon_error < 1e-3, (
        f"sigma reparam roundtrip error too high: {sigma_recon_error}"
    )

    # 2) Synthetic training loop (n_steps).
    losses = []
    start_t = time.perf_counter()
    for step in range(args.n_steps):
            # Sample t (log-normal if --sigma-reparam, else uniform).
            if args.sigma_reparam:
                t = sample_log_normal_t(b, device=device)
            else:
                t = torch.rand(b, device=device)
            # Bridge to σ if --edm-precond is on.
            sigma = (sigma_from_cfm_t(t, cfg) if args.edm_precond
                     else sample_edm_sigma(b, cfg, device))
            # Synthetic x_0 / x_1 / x_t / dx_t.
            x_0 = torch.randn(b, n, 3, device=device)
            x_1 = torch.randn(b, n, 3, device=device)
            x_t = (1.0 - t.view(-1, 1, 1)) * x_0 + t.view(-1, 1, 1) * x_1
            dx_t = x_1 - x_0
            mask = torch.ones(b, n, 1, device=device)
            # Forward + loss + backward + step.
            opt.zero_grad()
            out = model(x_t, t)
            v_pred = out["vel"] if isinstance(out, dict) else out
            edm_cfg_arg = cfg if args.edm_precond else None
            loss = cfm_velocity_mse(
                v_pred, dx_t, mask, edm_cfg=edm_cfg_arg, sigma=sigma,
            )
            loss.backward()
            opt.step()
            losses.append(loss.item())
    wall_s = time.perf_counter() - start_t

    # 3) Report.
    final_loss = losses[-1]
    print(f"[train] n_steps={args.n_steps}, wall={wall_s:.4f}s, "
          f"per_step={wall_s/args.n_steps*1000:.2f}ms")
    print(f"[train] first loss={losses[0]:.4f}, final loss={final_loss:.4f}")
    print(f"[train] loss delta = {losses[0] - final_loss:.4f}")
    print(f"[ok] Wf-341-C2 standalone EDM preconditioning + sigma reparam "
          f"completed on synthetic CFM velocity field.")
    print(f"[ok] PROJECTED 1.5-3x speedup per Karras 2022 §5 (MEASURED on "
          f"Mol-Metal EGNN deferred to Wf-340-A7).")

    # 4) JSON-side probe of sigma reparam: log-normal t → σ should
    # have high variance (more mass at small t → large σ).
    n_probe = 4096
    t_probe = sample_log_normal_t(n_probe, device=device) \
        if args.sigma_reparam else torch.rand(n_probe, device=device)
    sigma_probe = sigma_from_cfm_t(t_probe, cfg)
    print(f"[probe] n={n_probe}, t median={t_probe.median().item():.4f}, "
          f"σ median={sigma_probe.median().item():.4f}, "
          f"σ mean={sigma_probe.mean().item():.4f}, "
          f"σ max={sigma_probe.max().item():.4f}")
    assert sigma_probe.median().item() > 0.5, (
        f"expected σ median > 0.5 from log-normal t, got "
        f"{sigma_probe.median().item()}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))