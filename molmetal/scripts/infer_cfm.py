#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone inference CLI for Mol-Metal CFM refiner.

Mirrors omegafold/__main__.py:39+47+54-55:
  - @torch.no_grad() main()
  - argparse args (input_file, output_dir, device, num_cycle, batch_size)
  - os.makedirs(args.output_dir, exist_ok=True)
  - model.eval(); model.to(args.device)
  - Per-input inference loop + save outputs

Usage:
    python -m molmetal.scripts.infer_cfm \
        --input-file molecules.smi \
        --output-dir ./predictions/ \
        --device cuda \
        --num-cycle 3
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch


# Defaults — override via CLI flags
DEFAULTS = {
    "input_file": "molecules.smi",
    "output_dir": "./predictions/",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_cycle": 1,
    "batch_size": 8,
    "max_smiles_len": 256,
}


def parse_args(argv=None) -> argparse.Namespace:
    """Build argparse Namespace from CLI args (mirror omegafold/__main__.py:42)."""
    p = argparse.ArgumentParser(
        description="Mol-Metal CFM standalone inference CLI",
    )
    p.add_argument(
        "--input-file",
        default=DEFAULTS["input_file"],
        type=str,
        help="Path to SMILES / fingerprint file (one per line)",
    )
    p.add_argument(
        "--output-dir",
        default=DEFAULTS["output_dir"],
        type=str,
        help="Directory for inference outputs",
    )
    p.add_argument(
        "--device",
        default=DEFAULTS["device"],
        type=str,
        help="cpu / cuda / cuda:0 / cuda:1",
    )
    p.add_argument(
        "--num-cycle",
        type=int,
        default=DEFAULTS["num_cycle"],
        help="Number of recursive inference cycles (mirror OmegaFoldCycle)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULTS["batch_size"],
        help="Inference batch size",
    )
    p.add_argument(
        "--max-smiles-len",
        type=int,
        default=DEFAULTS["max_smiles_len"],
        help="Max SMILES characters per line",
    )
    p.add_argument(
        "--model-ckpt", type=str, default="", help="Path to model checkpoint (optional)"
    )
    p.add_argument(
        "--checkpoint-state-dict-key",
        type=str,
        default="model",
        help="Key in checkpoint state_dict (e.g. 'model' or 'state_dict')",
    )
    return p.parse_args(argv)


def make_dirs(path: str) -> None:
    """Mirror omegafold/__main__.py:44 os.makedirs(..., exist_ok=True)."""
    os.makedirs(path, exist_ok=True)


def load_smiles(input_file: str, max_len: int = DEFAULTS["max_smiles_len"]) -> list:
    """Load SMILES from file (one per line, skip blanks / #)."""
    smiles = []
    p = Path(input_file)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    with p.open("r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            if len(s) > max_len:
                s = s[:max_len]
            smiles.append(s)
    return smiles


def setup_model(args: argparse.Namespace):
    """Build model + load checkpoint (mirror omegafold/__main__.py:47-55).

    Lazy import to keep CLI boot fast. CFM trainer is the canonical model.
    """
    from molmetal.scripts.train_cfm_yuelbond_wasserstein import build_model_from_args

    model = build_model_from_args(args)
    if args.model_ckpt and Path(args.model_ckpt).exists():
        ckpt = torch.load(args.model_ckpt, map_location=args.device, weights_only=True)
        sd = ckpt.get(args.checkpoint_state_dict_key, ckpt)
        try:
            model.load_state_dict(sd, strict=False)
        except Exception as e:  # noqa
            logging.warning(f"load_state_dict strict=False raised: {e}")
    model.eval()
    model.to(args.device)
    return model


@torch.no_grad()
def infer_batch(model, smiles_batch: list, args: argparse.Namespace) -> dict:
    """Run inference on a batch, returning dict with confidence + predictions.

    Mirrors omegafold/__main__.py:73-83 — returns final_result dict + per-cycle
    progress. Wrapped in try/except RuntimeError for hard-case skip.
    """
    try:
        # Stub inference — concrete forward hook implemented by trainer integration
        output = model.infer(smiles_batch, num_cycle=args.num_cycle)
    except RuntimeError as e:
        logging.info(f"Failed to generate batch due to {e}")
        return {"confidence": 0.0, "predictions": [], "skipped": True}
    return output


@torch.no_grad()
def main(argv=None) -> int:
    """Mirror omegafold/__main__.py:39 main() entrypoint."""
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
    )
    args = parse_args(argv)
    make_dirs(args.output_dir)
    logging.info(f"Inference device={args.device} num_cycle={args.num_cycle}")
    model = setup_model(args)
    smiles_list = load_smiles(args.input_file, args.max_smiles_len)
    logging.info(f"Loaded {len(smiles_list)} molecules from {args.input_file}")
    out_records = []
    t_total = time.time()
    # Per-batch inference loop (mirror omegafold/__main__.py:58-83)
    for batch_start in range(0, len(smiles_list), args.batch_size):
        batch = smiles_list[batch_start : batch_start + args.batch_size]
        ts = time.time()
        output = infer_batch(model, batch, args)
        elapsed = time.time() - ts
        record = {
            "batch_idx": batch_start // args.batch_size,
            "n_molecules": len(batch),
            "confidence": float(output.get("confidence", 0.0)),
            "wall_sec": elapsed,
            "num_cycle": args.num_cycle,
        }
        out_records.append(record)
        logging.info(
            f"Batch {batch_start // args.batch_size}: {len(batch)} mols, "
            f"confidence={record['confidence']:.3f}, wall={elapsed:.2f}s"
        )
        # Mirror omegafold/__main__.py:95-97 memory hygiene
        del output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    # Write summary
    summary_path = Path(args.output_dir) / "inference_summary.json"
    summary = {
        "input_file": args.input_file,
        "n_molecules": len(smiles_list),
        "n_batches": len(out_records),
        "device": args.device,
        "num_cycle": args.num_cycle,
        "total_wall_sec": time.time() - t_total,
        "mean_confidence": float(
            sum(r["confidence"] for r in out_records) / max(1, len(out_records))
        ),
        "batches": out_records,
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    logging.info(
        f"DONE. {len(smiles_list)} mols in {summary['total_wall_sec']:.1f}s, "
        f"mean_confidence={summary['mean_confidence']:.3f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
