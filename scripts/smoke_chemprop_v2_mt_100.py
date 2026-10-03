"""1-epoch smoke test on 100 MetalCytoToxDB samples for ChempropV2MTModel.

Wf-148 ultracode / TODO-148 §3.  Verifies fwd+bwd+AdamW works on a tiny
CPU-friendly slice of the data using the chemprop v2 multi-task
backbone.  Not a training run — just a sanity gate that gradients flow
and the loss is finite.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Ensure the project root is importable when the script is run from anywhere.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from molmetal.data.metal_cytotox import load_metal_cytotox
from molmetal.models.chemprop_v2_mt_model import (
    SUPPORTED_METALS,
    ChempropV2MTLoss,
    ChempropV2MTModel,
)


def _set_seed(seed: int = 0) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _coord_label_for(metal_name: str) -> int:
    """Map metal → canonical coordination-geometry class id (default 3 = square_planar).

    Heuristic: d^8 Pt/Pd/Au(III) → square_planar (3); d^6 Ru/Ir/Os/Rh → octahedral (4);
    everything else → square_planar.  We only need a deterministic integer
    target here (the smoke test does not require a calibrated coord head).
    """
    table = {
        "Pt": 3, "Pd": 3, "Au": 3,                       # square_planar
        "Ru": 4, "Ir": 4, "Os": 4, "Rh": 4, "Re": 4,   # octahedral
    }
    return table.get(metal_name, 3)


def _slice_100(df):
    """Return 100 balanced samples covering all 5 supported metals."""
    df = df[df["metal"].isin(SUPPORTED_METALS)].reset_index(drop=True)
    # Round-robin sample 20 per metal.
    pieces = []
    per_metal = 20
    for m in SUPPORTED_METALS:
        sub = df[df["metal"] == m]
        if len(sub) == 0:
            continue
        idx = np.linspace(0, len(sub) - 1, per_metal).round().astype(int)
        pieces.append(sub.iloc[idx])
    out = pd.concat(pieces, ignore_index=True)
    out = out.sample(frac=1.0, random_state=0).reset_index(drop=True)
    return out.head(100)


def main() -> dict:
    _set_seed(0)
    print("[smoke] loading MetalCytoToxDB …", flush=True)
    df = load_metal_cytotox()
    print(f"[smoke] full DB rows = {len(df)}", flush=True)

    sub = _slice_100(df)
    assert len(sub) == 100, f"expected 100 samples, got {len(sub)}"

    smiles_list = sub["smiles"].tolist()
    metals_list = sub["metal"].tolist()
    actives_list = sub["active"].astype(float).tolist()

    # Map metal names → ids inside the model's supported tuple.
    metal_to_id = {m: i for i, m in enumerate(SUPPORTED_METALS)}
    metal_ids = torch.tensor(
        [metal_to_id[m] for m in metals_list], dtype=torch.long
    )

    # Build y_true dict (one tensor per metal, all rows; missing metals get 0s).
    y_true: dict[str, torch.Tensor] = {}
    for m in SUPPORTED_METALS:
        mask = torch.tensor(
            [1.0 if mn == m else 0.0 for mn in metals_list], dtype=torch.float32
        )
        # Per-metal target: 1.0 where the row's metal matches, else 0.0.
        # For rows where the metal is m, use the row's active label (0/1).
        per_row = torch.tensor(
            [
                float(actives_list[i]) if mn == m else 0.0
                for i, mn in enumerate(metals_list)
            ],
            dtype=torch.float32,
        )
        y_true[m] = per_row

    coord_true = torch.tensor(
        [_coord_label_for(m) for m in metals_list], dtype=torch.long
    )

    print(
        f"[smoke] smoke batch: B={len(smiles_list)} "
        f"metals_in_batch={set(metals_list)}",
        flush=True,
    )

    # Tiny backbone so the smoke runs in seconds on CPU.
    print("[smoke] instantiating ChempropV2MTModel (hidden_dim=64, depth=2) …",
          flush=True)
    model = ChempropV2MTModel(
        hidden_dim=64,
        depth=2,
        n_metals=5,
        n_coord_classes=5,
    )
    loss_fn = ChempropV2MTLoss(n_coord_classes=5)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[smoke] model params = {n_params}", flush=True)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # 1-step forward + backward + optimizer.
    t0 = time.time()
    model.train()
    out = model(smiles_list, metal_ids)
    loss, info = loss_fn(out, y_true, coord_true)
    optim.zero_grad()
    loss.backward()
    optim.step()
    t1 = time.time()
    step_time_ms = (t1 - t0) * 1000.0

    loss_value = float(loss.detach().item())
    loss_finite = bool(np.isfinite(loss_value))
    smoke_works = bool(loss_finite and not np.isnan(loss_value))

    print(f"[smoke] loss = {loss_value:.6f}  finite={loss_finite}", flush=True)
    print(f"[smoke] per-metal loss: " +
          ", ".join(f"{k}={v.item():.4f}" for k, v in info.items()
                    if k.startswith("loss_") and k != "loss_total"),
          flush=True)
    print(f"[smoke] step_time = {step_time_ms:.1f} ms", flush=True)

    # Checkpoint.
    ckpt_dir = PROJECT_ROOT / "molmetal" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "chemprop_v2_mt_smoke_100.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optim.state_dict(),
            "loss": loss_value,
            "n_samples": len(smiles_list),
            "metals": list(SUPPORTED_METALS),
            "hidden_dim": 64,
            "depth": 2,
        },
        ckpt_path,
    )
    print(f"[smoke] saved checkpoint → {ckpt_path}", flush=True)

    # Honest framing — this is a 100-sample 1-step CPU smoke, not training.
    honest_framing_note = (
        "CPU 1-step smoke on 100 MetalCytoToxDB samples (hidden_dim=64, "
        "depth=2 — far smaller than the production 300/3). loss_finite + "
        "non-NaN only proves fwd+bwd+AdamW wire-up; it is NOT a "
        "training-run claim, NOT a metric, and NOT comparable to "
        "5-seed × 50-epoch R18 GPU retrain.  MEASURED on this CPU run, "
        "not projected."
    )

    return {
        "smoke_works": smoke_works,
        "loss_finite": loss_finite,
        "loss_value": loss_value,
        "step_time_ms": step_time_ms,
        "ckpt_path": str(ckpt_path),
        "honest_framing_note": honest_framing_note,
    }


if __name__ == "__main__":
    out = main()
    print("\n=== SMOKE RESULT (JSON) ===", flush=True)
    print(json.dumps(out, indent=2), flush=True)
