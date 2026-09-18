"""Hyperparameter sweep over the QM9 edge-cutoff radius.

For each cutoff in ``[4.0, 5.0, 6.0, 7.0, 8.0]`` this script:

1. Snapshots ``checkpoints/model.pt`` (the user's pre-trained model).
2. Trains for 2 epochs at batch size 32 on QM9 with
   ``train.edge_cutoff = c`` by calling the building blocks exported by
   :mod:`scripts.train` as a library.  The cutoff is plumbed through
   ``cfg["train"]["edge_cutoff"]``; ``collate_qm9`` reads it natively.
3. Generates 32 samples at 100 Euler steps and reports validity /
   uniqueness via :mod:`utils.metrics`.
4. Restores the original checkpoint so the next cutoff starts from the
   same pre-trained weights (no contamination between runs).
5. Prints and saves a summary Markdown table at the project root.

The training body mirrors :func:`scripts.train.train` because that
function unpacks ``build_dataset`` incorrectly (it returns a tuple
``(dataset, target_index)`` and the existing code treats it as a bare
dataset).  Rather than edit ``scripts/train.py`` — the task explicitly
forbids it — we drive the same primitives in the same order with the
tuple unpacked at the caller side.  No model or loss code is
duplicated.

Each run uses ``seed=42`` so the comparison is deterministic.

Usage
-----

    source .venv/bin/activate
    python scripts/sweep_cutoff.py
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from copy import deepcopy
from pathlib import Path

# Make the project root importable so ``scripts.train`` /
# ``scripts.generate`` resolve their package-internal imports
# (models, flow_matching, data, ...) when we import them here.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from flow_matching import ConditionalFlowMatchingLoss
from utils.config import load_config
from utils.metrics import uniqueness, validity

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CUTOFFS = (4.0, 5.0, 6.0, 7.0, 8.0)
DEFAULT_EPOCHS = 2
DEFAULT_BATCH_SIZE = 32
DEFAULT_N_SAMPLES = 32
DEFAULT_N_STEPS = 100
DEFAULT_METHOD = "euler"
DEFAULT_SEED = 42
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "model.pt"
SWEEP_REPORT_PATH = PROJECT_ROOT / "cutoff_sweep.md"


# ---------------------------------------------------------------------------
# CLI (mostly for overwriting defaults from shell)
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Sweep the QM9 edge-cutoff radius and report "
        "validity/uniqueness."
    )
    p.add_argument(
        "--cutoffs",
        type=float,
        nargs="+",
        default=list(DEFAULT_CUTOFFS),
        help="Edge-cutoff values to sweep (default: 4 5 6 7 8).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Epochs per cutoff (default: 2).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Batch size per cutoff (default: 32).",
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help="Samples to generate per cutoff (default: 32).",
    )
    p.add_argument(
        "--n-steps",
        type=int,
        default=DEFAULT_N_STEPS,
        help="ODE integration steps for generation (default: 100).",
    )
    p.add_argument(
        "--method",
        type=str,
        choices=("euler", "rk4"),
        default=DEFAULT_METHOD,
        help="Integration scheme (default: euler).",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="'cuda' or 'cpu'.  Default: auto-detect.",
    )
    p.add_argument(
        "--report",
        type=str,
        default=str(SWEEP_REPORT_PATH),
        help="Where to save the Markdown report.",
    )
    p.add_argument(
        "--early-stop",
        action="store_true",
        help="Stop after the first cutoff that strictly beats every prior "
        "cutoff on both validity and uniqueness.",
    )
    return p


# ---------------------------------------------------------------------------
# Checkpoint snapshot / restore so each run is independent
# ---------------------------------------------------------------------------
def snapshot_checkpoint(path: Path) -> bytes:
    """Return the raw bytes of ``path`` and ensure the file is on disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}. Run scripts/train.py first."
        )
    return path.read_bytes()


def restore_checkpoint(path: Path, blob: bytes) -> None:
    """Rewrite the checkpoint from ``blob`` so subsequent runs start fresh."""
    path.write_bytes(blob)


# ---------------------------------------------------------------------------
# Training driver: call into scripts.train as a library without modifying
# the buggy ``build_dataset`` return contract.
# ---------------------------------------------------------------------------
def train_with_cutoff(
    cutoff: float,
    epochs: int,
    batch_size: int,
    base_cfg: dict,
    device: torch.device,
) -> dict:
    """Train ``epochs`` epochs at the given cutoff; return per-run metrics.

    ``cutoff`` is forwarded both via ``cfg["train"]["edge_cutoff"]`` and
    the module attribute ``scripts.train._QM9_EDGE_CUTOFF`` so any code
    path that reads either picks up the override.
    """
    import scripts.train as train_mod

    cfg = deepcopy(base_cfg)
    cfg.setdefault("train", {})
    cfg["train"]["edge_cutoff"] = float(cutoff)
    cfg["train"]["epochs"] = int(epochs)
    cfg["train"]["batch_size"] = int(batch_size)
    cfg["seed"] = DEFAULT_SEED
    if device is not None:
        cfg["device"] = str(device)

    original_cutoff = train_mod._QM9_EDGE_CUTOFF
    train_mod._QM9_EDGE_CUTOFF = float(cutoff)
    try:
        # Set deterministic seeds before building any tensors.
        torch.manual_seed(DEFAULT_SEED)

        # 1. dataset --------------------------------------------------------
        # scripts.train.build_dataset returns (dataset, target_index).
        # The bundled train() function unpacks this incorrectly; we use
        # the library primitives directly so the bug does not block us.
        built = train_mod.build_dataset(cfg)
        if isinstance(built, tuple):
            dataset, _effective_target_index = built
        else:
            dataset = built

        from data.mol_dataset import QM9Dataset as _QM9Dataset
        position_scale = (
            train_mod.compute_qm9_position_scale(dataset)
            if isinstance(dataset, _QM9Dataset)
            else 1.0
        )

        edge_cutoff = float(cfg["train"]["edge_cutoff"])

        def _collate(items):
            if isinstance(dataset, _QM9Dataset):
                return train_mod.collate_qm9(
                    items,
                    device=device,
                    position_scale=position_scale,
                    edge_cutoff=edge_cutoff,
                )
            feats = torch.stack(
                [s.features for s in items if s.features is not None], dim=0
            )
            feats = feats / 1000.0
            return train_mod.collate_tabular_items(feats, device=device)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=_collate,
        )

        # 2. model + encoder + loss + optimizer ------------------------------
        velocity_net = train_mod.build_velocity_net(cfg, device)
        encoder = train_mod.build_encoder(cfg, device)
        # Warm-start each cutoff from the snapshotted pre-trained weights
        # so the comparison is apples-to-apples (no cutoff starts from a
        # random init).  The snapshot is restored before each cutoff in
        # main() so contamination is impossible.
        blob = torch.load(CHECKPOINT_PATH, map_location=device)
        velocity_net.load_state_dict(blob["model_state_dict"])
        encoder.load_state_dict(blob["encoder_state_dict"])
        loss_fn = ConditionalFlowMatchingLoss(
            model=velocity_net, encoder=encoder
        ).to(device)
        optimizer = torch.optim.Adam(
            list(velocity_net.parameters()) + list(encoder.parameters()),
            lr=float(cfg["train"]["lr"]),
        )
        warmup_steps = int(cfg["train"]["warmup_steps"])
        grad_clip_norm = float(cfg["train"]["grad_clip_norm"])
        ema_decay = float(cfg["train"].get("ema_decay", 0.999))
        velocity_net_ema = {
            name: param.detach().clone()
            for name, param in velocity_net.named_parameters()
        }
        n_batches = len(loader)
        total_steps = max(1, epochs * n_batches)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # 3. epoch loop ------------------------------------------------------
        t0 = time.time()
        initial_loss = None
        final_loss = None
        global_step = 0
        for epoch in range(1, epochs + 1):
            velocity_net.train()
            encoder.train()
            epoch_loss = 0.0
            n_batches_epoch = 0
            for batch in loader:
                x1 = batch["positions"]
                x0 = x1 + 0.1 * torch.randn_like(x1) * x1.std()
                out = loss_fn(
                    x0=x0,
                    x1=x1,
                    cond=None,
                    atomic_numbers=batch["atomic_numbers"],
                    edge_index=batch["edge_index"],
                    edge_mask=batch["edge_mask"],
                    node_mask=batch["node_mask"],
                )
                loss_val = float(out.loss.item())
                if initial_loss is None:
                    initial_loss = loss_val

                optimizer.zero_grad()
                out.loss.backward()

                has_nan_inf = any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in list(velocity_net.parameters()) + list(encoder.parameters())
                )
                if has_nan_inf:
                    optimizer.zero_grad()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        list(velocity_net.parameters()) + list(encoder.parameters()),
                        grad_clip_norm,
                    )
                    optimizer.step()

                    with torch.no_grad():
                        for name, param in velocity_net.named_parameters():
                            ema_t = velocity_net_ema[name]
                            ema_t.mul_(ema_decay).add_(
                                param.detach(), alpha=1.0 - ema_decay
                            )

                scheduler.step()
                epoch_loss += loss_val
                n_batches_epoch += 1
                global_step += 1
                final_loss = loss_val

            avg = epoch_loss / max(n_batches_epoch, 1)
            print(
                f"[sweep-train] cutoff={edge_cutoff} "
                f"epoch {epoch}/{epochs}  avg_loss={avg:.6f}"
            )

            # Save the EMA-smoothed weights under the same keys
            # scripts.generate.py expects.
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": velocity_net.state_dict(),
                    "velocity_net_ema_state_dict": {
                        name: t.detach().clone()
                        for name, t in velocity_net_ema.items()
                    },
                    "encoder_state_dict": encoder.state_dict(),
                    "encoder_config": train_mod.encoder_config_from(cfg),
                    "position_scale": float(position_scale),
                    "config": cfg,
                },
                CHECKPOINT_PATH,
            )
        elapsed = time.time() - t0
    finally:
        train_mod._QM9_EDGE_CUTOFF = original_cutoff

    # Stable eval loss under EMA shadow weights, evaluated BEFORE we
    # return so the freshly-trained checkpoint is still on disk.
    eval_loss = compute_eval_loss(
        cfg, velocity_net_ema=velocity_net_ema, device=device,
        position_scale=position_scale,
    )

    return {
        "cutoff": float(cutoff),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "initial_loss": float(initial_loss) if initial_loss is not None else float("nan"),
        "final_loss": float(final_loss) if final_loss is not None else float("nan"),
        "train_loss": float(final_loss) if final_loss is not None else float("nan"),
        "eval_loss": float(eval_loss),
        "wall_time_s": float(elapsed),
    }


def compute_eval_loss(
    cfg: dict,
    velocity_net_ema: dict | None = None,
    device: torch.device = None,
    max_batches: int = 4,
    position_scale: float = 1.0,
) -> float:
    """Evaluate loss on the freshly-saved checkpoint.

    When ``velocity_net_ema`` is supplied we use the EMA shadow
    weights instead of the live parameters for a stable signal.
    """
    import scripts.train as train_mod

    built = train_mod.build_dataset(cfg)
    dataset = built[0] if isinstance(built, tuple) else built
    from data.mol_dataset import QM9Dataset as _QM9Dataset
    edge_cutoff = float(cfg["train"].get("edge_cutoff", train_mod._QM9_EDGE_CUTOFF))

    def _collate(items):
        if isinstance(dataset, _QM9Dataset):
            return train_mod.collate_qm9(
                items, device=device, position_scale=position_scale,
                edge_cutoff=edge_cutoff,
            )
        feats = torch.stack(
            [s.features for s in items if s.features is not None], dim=0
        )
        feats = feats / 1000.0
        return train_mod.collate_tabular_items(feats, device=device)

    loader = DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True, drop_last=True, collate_fn=_collate,
    )
    velocity_net = train_mod.build_velocity_net(cfg, device)
    encoder = train_mod.build_encoder(cfg, device)
    blob = torch.load(CHECKPOINT_PATH, map_location=device)
    velocity_net.load_state_dict(blob["model_state_dict"])
    encoder.load_state_dict(blob["encoder_state_dict"])
    velocity_net.eval(); encoder.eval()
    if velocity_net_ema is None and "velocity_net_ema_state_dict" in blob:
        velocity_net_ema = blob["velocity_net_ema_state_dict"]
    loss_fn = ConditionalFlowMatchingLoss(
        model=velocity_net, encoder=encoder
    ).to(device)
    return train_mod.evaluate(
        loss_fn, loader, device,
        max_batches=max_batches,
        ema_state_dict=velocity_net_ema,
    )


# ---------------------------------------------------------------------------
# Generation wrapper that forwards the cutoff to the noise-time edge graph
# ---------------------------------------------------------------------------
def generate_with_cutoff(
    cutoff: float,
    n_samples: int,
    n_steps: int,
    method: str,
    device: torch.device,
) -> dict:
    """Sample ``n_samples`` molecules and compute validity/uniqueness."""
    import scripts.generate as gen_mod

    original_gen_cutoff = gen_mod.GENERATE_EDGE_CUTOFF
    gen_mod.GENERATE_EDGE_CUTOFF = float(cutoff)
    try:
        blob = torch.load(CHECKPOINT_PATH, map_location="cpu")
        cfg = blob.get("config") or {}

        net, blob_dev = gen_mod.load_velocity_net(CHECKPOINT_PATH, device)
        encoder = gen_mod.load_encoder(blob_dev, device, cfg)
        # Prefer EMA weights for generation (smoother than the live ones).
        if isinstance(blob, dict) and "velocity_net_ema_state_dict" in blob:
            net.load_state_dict(blob["velocity_net_ema_state_dict"])
        position_scale = float(blob.get("position_scale", 1.0))

        from flow_matching import FlowMatchingSampler
        sampler = FlowMatchingSampler(
            model=net, encoder=encoder,
            default_method=method, default_n_steps=n_steps,
        )

        from data.mol_dataset import QM9Dataset
        try:
            seed_ds = QM9Dataset(split="test")
        except Exception:
            seed_ds = None

        x0, atomic_numbers, edge_index, edge_mask, node_mask = (
            gen_mod.build_qm9_inputs(n_samples, device, seed_dataset=seed_ds)
        )
        x_final = sampler.sample(
            x0=x0, cond=None,
            n_steps=n_steps, method=method,
            atomic_numbers=atomic_numbers,
            edge_index=edge_index, edge_mask=edge_mask, node_mask=node_mask,
        )
        x_real = gen_mod.denormalize_positions(x_final, position_scale)
        smiles = gen_mod.positions_to_smiles(x_real, atomic_numbers)
        val = float(validity(smiles))
        uniq = float(uniqueness(smiles))
    finally:
        gen_mod.GENERATE_EDGE_CUTOFF = original_gen_cutoff

    return {
        "validity": val,
        "uniqueness": uniq,
        "n_valid": int(round(val * len(smiles))),
        "n_samples": int(len(smiles)),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def render_markdown_table(rows: list[dict]) -> str:
    """Render the sweep results as a Markdown table."""
    lines = [
        "# Edge-cutoff sweep",
        "",
        "Each row is a fresh 2-epoch retrain (batch size 32, QM9) at the "
        "given edge cutoff, followed by 32 Euler samples at 100 steps.  "
        "The pre-trained checkpoint is restored before every cutoff so "
        "runs are independent.",
        "",
        "| cutoff | train_loss | eval_loss | validity | uniqueness |",
        "|-------:|-----------:|----------:|---------:|-----------:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['cutoff']:.1f} "
            f"| {r['train_loss']:.4f} "
            f"| {r['eval_loss']:.4f} "
            f"| {r['validity']:.4f} "
            f"| {r['uniqueness']:.4f} |"
        )
    lines.append("")
    return "\n".join(lines)


def pick_winner(rows: list[dict]) -> dict:
    """Pick the row that maximises eval_loss / 32-sample validity.

    Primary key: validity (descending).  Secondary key: uniqueness
    (descending).  Tertiary key: eval_loss (ascending, used as a
    tie-breaker only).
    """
    return max(
        rows,
        key=lambda r: (
            r["validity"],
            r["uniqueness"],
            -r["eval_loss"],
        ),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    cutoffs: list[float] = list(args.cutoffs)

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[sweep] device = {device}")
    print(f"[sweep] cutoffs = {cutoffs}")
    print(f"[sweep] epochs={args.epochs} batch_size={args.batch_size} "
          f"n_samples={args.n_samples} n_steps={args.n_steps} method={args.method}")

    base_cfg = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    base_cfg["train"]["epochs"] = int(args.epochs)
    base_cfg["train"]["batch_size"] = int(args.batch_size)

    snapshot = snapshot_checkpoint(CHECKPOINT_PATH)
    print(f"[sweep] snapshotted {CHECKPOINT_PATH} "
          f"({len(snapshot):,} bytes)")

    rows: list[dict] = []
    best_so_far: dict | None = None
    try:
        for c in cutoffs:
            print(f"\n========== cutoff = {c} ==========")
            # Start each cutoff from the original pre-trained weights.
            restore_checkpoint(CHECKPOINT_PATH, snapshot)

            train_metrics = train_with_cutoff(
                cutoff=c,
                epochs=args.epochs,
                batch_size=args.batch_size,
                base_cfg=base_cfg,
                device=device,
            )
            print(
                f"[sweep] cutoff={c} train_loss={train_metrics['train_loss']:.4f} "
                f"eval_loss={train_metrics['eval_loss']:.4f} "
                f"({train_metrics['wall_time_s']:.1f}s)"
            )

            gen_metrics = generate_with_cutoff(
                cutoff=c,
                n_samples=args.n_samples,
                n_steps=args.n_steps,
                method=args.method,
                device=device,
            )
            print(
                f"[sweep] cutoff={c} validity={gen_metrics['validity']:.4f} "
                f"uniqueness={gen_metrics['uniqueness']:.4f} "
                f"({gen_metrics['n_valid']}/{gen_metrics['n_samples']} valid)"
            )

            row = {**train_metrics, **gen_metrics}
            rows.append(row)

            # Always restore the snapshot before the next cutoff.
            restore_checkpoint(CHECKPOINT_PATH, snapshot)
            print(f"[sweep] restored {CHECKPOINT_PATH} for independence")

            if (
                args.early_stop
                and best_so_far is not None
                and row["validity"] > best_so_far["validity"]
                and row["uniqueness"] > best_so_far["uniqueness"]
            ):
                print("[sweep] early-stop: clear winner found")
                break
            if best_so_far is None or (
                row["validity"], row["uniqueness"]
            ) > (best_so_far["validity"], best_so_far["uniqueness"]):
                best_so_far = row
    finally:
        # Always restore the original checkpoint, even on exception.
        restore_checkpoint(CHECKPOINT_PATH, snapshot)
        print(f"[sweep] restored {CHECKPOINT_PATH} after sweep")

    # ---- Report ---------------------------------------------------------
    print("\n========== sweep summary ==========")
    md = render_markdown_table(rows)
    print(md)
    Path(args.report).write_text(md + "\n", encoding="utf-8")
    print(f"[sweep] wrote {args.report}")

    winner = pick_winner(rows)
    print(
        f"\n[sweep] winner cutoff={winner['cutoff']:.1f} "
        f"validity={winner['validity']:.4f} "
        f"uniqueness={winner['uniqueness']:.4f} "
        f"eval_loss={winner['eval_loss']:.4f}"
    )
    # Machine-readable summary for the orchestrator that spawned us.
    summary_path = Path(args.report).with_suffix(".json")
    summary_path.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "cutoff": r["cutoff"],
                        "train_loss": r["train_loss"],
                        "eval_loss": r["eval_loss"],
                        "validity": r["validity"],
                        "uniqueness": r["uniqueness"],
                    }
                    for r in rows
                ],
                "winner": {
                    "cutoff": winner["cutoff"],
                    "validity": winner["validity"],
                    "uniqueness": winner["uniqueness"],
                    "eval_loss": winner["eval_loss"],
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[sweep] wrote {summary_path}")


if __name__ == "__main__":
    main()