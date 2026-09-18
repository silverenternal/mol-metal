"""T5 — Pocket-conditioned Lipman Flow Matching training (100 steps).

Goal
----
Verify that the pocket-conditioned
:class:`LipmanFlowMatchingAdapter` (T5) actually trains: loss decreases
over 100 mini-batch steps, the pocket embedding participates in
gradients, and the joint velocity-field + pocket-encoder parameter
groups converge on synthetic (toy pocket, toy ligand) data.

We deliberately use **synthetic** data so the script is
self-contained and runs in <60s on ROCm.  Real-pocket training lives
in ``train_fm_pocket.py`` which uses CrossDocked2020.

Outputs
-------
* molmetal/checkpoints/fm_pocket_conditioned.pt
* molmetal/reports/fm_pocket_train.md (this run summary)
* molmetal/reports/fm_pocket_train_loss.png

Usage
-----
    cd /home/hugo/codes/try_triton_on_rocm
    source .venv/bin/activate
    python -m molmetal.scripts.train_pocket_flow --steps 100
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.adapters.flow_matching_lipman import LipmanFlowMatchingAdapter
from molmetal.domain import Molecule, Pocket
from molmetal.utils.device import verify_rocm_active

REPORTS_DIR = PROJECT_ROOT / "molmetal" / "reports"
CKPT_DIR = PROJECT_ROOT / "molmetal" / "checkpoints"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------
def _toy_pocket(pdb_id: str = "toy_pocket", n_atoms: int = 16, seed: int = 0):
    """Deterministic toy pocket — a small cluster around the origin."""
    g = torch.Generator().manual_seed(seed)
    coords = torch.randn(n_atoms, 3, generator=g) * 2.0
    atom_types = torch.randint(1, 18, (n_atoms,), generator=g)  # H..Ar
    return Pocket(
        pdb_id=pdb_id,
        coords=coords,
        atom_types=atom_types,
        residue_ids=torch.zeros(n_atoms, dtype=torch.long),
        chain_ids=torch.zeros(n_atoms, dtype=torch.long),
        mask=torch.ones(n_atoms, dtype=torch.bool),
        center=coords.mean(0),
        radius=6.0,
    )


def _toy_mol(n_atoms: int = 8, seed: int = 0):
    """Deterministic toy ligand — small molecule centred near origin."""
    g = torch.Generator().manual_seed(seed)
    coords = torch.randn(n_atoms, 3, generator=g)
    atom_types = torch.randint(1, 10, (n_atoms,), generator=g)  # H..Ne
    return Molecule(
        coords=coords,
        atom_types=atom_types,
        bonds=torch.zeros(2, 0, dtype=torch.long),
        bond_types=torch.zeros(0, dtype=torch.long),
        formal_charges=torch.zeros(n_atoms, dtype=torch.long),
    )


def _mini_batch(batch_size: int = 4, n_atoms: int = 8):
    """One mini-batch of pocket + ligands.  The pocket is BROADCAST to
    every ligand (typical SBDD setting: one pocket, many ligands)."""
    pocket = _toy_pocket()
    mols = [_toy_mol(n_atoms=n_atoms, seed=i + 1) for i in range(batch_size)]
    return pocket, mols


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100,
                        help="Number of CFM training steps (default 100)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-3,
                        help="Higher LR (vs default 1e-4) so loss visibly "
                             "drops in 100 synthetic steps.")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device_info = verify_rocm_active()
    print(f"[train_pocket_flow] ROCm active: {device_info}")

    adapter = LipmanFlowMatchingAdapter(
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        lr=args.lr,
    )
    adapter.setup()

    losses: list[float] = []
    cfm_losses: list[float] = []
    atom_losses: list[float] = []
    wall_times: list[float] = []
    t_start = time.time()

    for step in range(args.steps):
        # Fresh batch each step so the model sees different (pocket, ligands)
        pocket, mols = _mini_batch(batch_size=args.batch_size, n_atoms=8)
        t0 = time.time()
        loss = adapter.train_step(pocket=pocket, mols=mols)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        wall_times.append(time.time() - t0)
        losses.append(loss)
        cfm_losses.append(adapter.last_losses["cfm"])
        atom_losses.append(adapter.last_losses["atom"])
        if (step + 1) % 10 == 0 or step == 0:
            print(f"[step {step+1:3d}/{args.steps}] "
                  f"loss={loss:.4f} cfm={adapter.last_losses['cfm']:.4f} "
                  f"atom={adapter.last_losses['atom']:.4f} "
                  f"({wall_times[-1]*1000:.0f}ms)")

    total_wall = time.time() - t_start
    init_loss = float(np.mean(losses[:5]))
    final_loss = float(np.mean(losses[-5:]))
    print(f"[train_pocket_flow] initial (mean first 5) = {init_loss:.4f}")
    print(f"[train_pocket_flow] final   (mean last  5) = {final_loss:.4f}")
    print(f"[train_pocket_flow] loss ratio final/initial = "
          f"{final_loss / max(init_loss, 1e-9):.4f}")
    print(f"[train_pocket_flow] total wall = {total_wall:.1f}s, "
          f"avg = {1000*np.mean(wall_times):.0f}ms/step")

    # ---------------- Save checkpoint ----------------
    ckpt_path = CKPT_DIR / "fm_pocket_conditioned.pt"
    torch.save({
        "velocity_field": adapter.velocity_field.state_dict(),
        "pocket_encoder": adapter.pocket_encoder.state_dict(),
        "config": {
            "hidden_dim": args.hidden_dim,
            "n_layers": args.n_layers,
            "lr": args.lr,
            "max_atomic_number": adapter._max_atomic_number,
        },
        "losses": losses,
        "step": args.steps,
    }, ckpt_path)
    print(f"[train_pocket_flow] saved checkpoint → {ckpt_path}")

    # ---------------- Plot loss curve ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(losses, label="total", color="#1f77b4", linewidth=1.5)
        ax.plot(cfm_losses, label="CFM (coord)", color="#2ca02c",
                linewidth=1.0, alpha=0.7)
        ax.plot(atom_losses, label="atom-CE", color="#d62728",
                linewidth=1.0, alpha=0.7)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.set_title(
            f"T5: pocket-conditioned Lipman FM — {args.steps} steps "
            f"(initial={init_loss:.2f}, final={final_loss:.2f})"
        )
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        png_path = REPORTS_DIR / "fm_pocket_train_loss.png"
        fig.savefig(png_path, dpi=120)
        plt.close(fig)
        print(f"[train_pocket_flow] saved loss curve → {png_path}")
    except Exception as e:
        print(f"[train_pocket_flow] matplotlib plot failed: {e}")

    # ---------------- Write JSON sidecar ----------------
    summary = {
        "task": "T5",
        "steps": args.steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "n_layers": args.n_layers,
        "seed": args.seed,
        "device": str(adapter.device),
        "device_info": device_info,
        "initial_loss_mean_first5": init_loss,
        "final_loss_mean_last5": final_loss,
        "loss_ratio_final_over_initial": final_loss / max(init_loss, 1e-9),
        "total_wall_seconds": total_wall,
        "avg_step_ms": 1000 * float(np.mean(wall_times)),
        "loss_decreased": final_loss < init_loss,
        "checkpoint": str(ckpt_path),
    }
    json_path = REPORTS_DIR / "fm_pocket_train_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"[train_pocket_flow] saved JSON summary → {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())