"""Train Lipman Flow Matching on the Ru subset and evaluate on the temporal split.

Steps
-----
1. Load MetalCytoToxDataset filtered to metal_whitelist=['Ru'].
2. Generate 3D conformers (ETKDGv3 + MMFF94) for all rows, cache to
   /mnt/storage/data/molmetal/3d_cache/.
3. TemporalSplitter(cutoff_year=2024): train = year < 2024, test = year >= 2024.
4. Initialize LipmanFlowMatchingAdapter (ROCm-first).
5. Train for N epochs, batch_size=64, lr=1e-3.
6. Save checkpoint to molmetal/checkpoints/fm_ru_temporal.pt.
7. Plot training loss curve to molmetal/reports/fm_ru_train_loss.png.
8. Generate 1000 candidate molecules from the trained model.
9. Score generated molecules with RDKitPropertyPredictor (QED, MolLogP, MolWt,
   TPSA, NumHDonors, NumHAcceptors).
10. Train a Morgan-FP → pIC50 regressor on the training set; apply to generated
    molecules to get predicted pIC50.
11. Write molmetal/reports/fm_ru_temporal_eval.md with results.

Usage
-----
    source .venv/bin/activate
    cd /home/hugo/codes/try_triton_on_rocm
    python -m molmetal.scripts.train_fm_cytotox --metal Ru --epochs 50 --batch 64
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

# Make molmetal importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.adapters.flow_matching_lipman import LipmanFlowMatchingAdapter
from molmetal.adapters.rdkit_predictor import RDKitPropertyPredictor
from molmetal.data.cytotox import CytotoxFilter, MetalCytotoxDataset
from molmetal.data.featurize import MorganFingerprinter
from molmetal.data.splits import SplitResult, TemporalSplitter
from molmetal.domain import Molecule
from molmetal.ports import GenerationConfig, PropertyPrediction

REPORTS_DIR = PROJECT_ROOT / "molmetal" / "reports"
CHECKPOINT_DIR = PROJECT_ROOT / "molmetal" / "checkpoints"
CACHE_DIR = Path("/mnt/storage/data/molmetal/3d_cache")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Lipman Flow Matching on Ru cytotox data.")
    p.add_argument("--metal", default="Ru", choices=["Ru", "Ir", "Rh", "Os", "Re", "Pt"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=16, help="Batch size for training (keep small to avoid OOM)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=3)
    p.add_argument("--n-gen", type=int, default=1000, help="Number of molecules to generate for evaluation")
    p.add_argument("--max-atoms", type=int, default=40, help="Max atoms per molecule (OOM guard)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint-out", default=None)
    p.add_argument("--plot-out", default=None)
    p.add_argument("--report-out", default=None)
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# 3D conformer pre-loading
# ---------------------------------------------------------------------------
def preload_conformers(
    dataset: MetalCytotoxDataset,
    indices: np.ndarray,
    cache_dir: Path,
    desc: str = "",
) -> list[Molecule | None]:
    """Pre-load 3D conformers for dataset rows at ``indices``.

    Uses the dataset's get_conformer which handles LRU + disk caching.
    Returns a list parallel to ``indices`` with None for failures.
    """
    mols = []
    total = len(indices)
    for k, idx in enumerate(indices):
        if desc and k % 500 == 0:
            print(f"  [{desc}] {k}/{total} conformers loaded...")
        row = dataset[idx]
        mol = dataset.get_conformer(row["smiles"])
        mols.append(mol)
    return mols


# ---------------------------------------------------------------------------
# pIC50 regressor (Morgan FP → MSE on pIC50)
# ---------------------------------------------------------------------------
class MorganPic50Regressor:
    """Simple MLP regressor on Morgan FP for pIC50 prediction."""

    def __init__(self, n_bits: int = 2048, hidden: int = 256, seed: int = 42):
        self.n_bits = n_bits
        self.hidden = hidden
        self.seed = seed
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MorganPic50Regressor":
        """Train an MLP regressor on Morgan FP → pIC50."""
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler

        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X)
        self._model = MLPRegressor(
            hidden_layer_sizes=(self.hidden, self.hidden // 2),
            activation="relu",
            solver="adam",
            alpha=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=self.seed,
            verbose=False,
        )
        self._model.fit(Xs, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = self._scaler.transform(X)
        return self._model.predict(Xs)


# ---------------------------------------------------------------------------
# Compute metrics on generated set
# ---------------------------------------------------------------------------
def evaluate_generated_molecules(
    mols: list[Molecule | None],
    predictor: RDKitPropertyPredictor,
    regressor: MorganPic50Regressor,
    morgan: MorganFingerprinter,
) -> dict:
    """Compute QED / logP / MW / TPSA distributions and predicted pIC50 for
    generated molecules that have valid SMILES + RDKit parse."""

    qed_list, logp_list, mw_list, tpsa_list = [], [], [], []
    pic50_pred_list = []
    valid_count = 0

    for mol in mols:
        if mol is None:
            continue
        smiles = getattr(mol, "smiles", None) or ""
        if not smiles:
            continue
        try:
            prop: PropertyPrediction = predictor.predict(mol)
        except Exception:
            continue

        qed_list.append(prop.qed)
        logp_list.append(prop.logp)
        mw_list.append(prop.mol_weight)
        tpsa_list.append(prop.tpsa)

        # Morgan FP for pIC50 regression
        try:
            fp = morgan.fingerprint_mol(predictor._coerce_to_mol(mol, __import__("rdkit").Chem))
            if fp is not None:
                pic50 = float(regressor.predict(fp.reshape(1, -1))[0])
                pic50_pred_list.append(pic50)
        except Exception:
            pass

        valid_count += 1

    qed_arr = np.array(qed_list, dtype=np.float32)
    pic50_arr = np.array(pic50_pred_list, dtype=np.float32)

    hit_rate_top5 = float(np.mean(pic50_arr >= 6.0)) if len(pic50_arr) > 0 else 0.0
    frac_druglike = float(np.mean(qed_arr >= 0.5)) if len(qed_arr) > 0 else 0.0

    return {
        "n_valid": valid_count,
        "qed_mean": float(np.mean(qed_arr)) if len(qed_arr) > 0 else 0.0,
        "qed_std": float(np.std(qed_arr)) if len(qed_arr) > 0 else 0.0,
        "qed_median": float(np.median(qed_arr)) if len(qed_arr) > 0 else 0.0,
        "logp_mean": float(np.mean(logp_list)) if logp_list else 0.0,
        "mw_mean": float(np.mean(mw_list)) if mw_list else 0.0,
        "tpsa_mean": float(np.mean(tpsa_list)) if tpsa_list else 0.0,
        "frac_druglike_qed05": frac_druglike,
        "pic50_pred_mean": float(np.mean(pic50_arr)) if len(pic50_arr) > 0 else 0.0,
        "pic50_pred_std": float(np.std(pic50_arr)) if len(pic50_arr) > 0 else 0.0,
        "pic50_pred_median": float(np.median(pic50_arr)) if len(pic50_arr) > 0 else 0.0,
        "hit_rate_top5_pic50": hit_rate_top5,
    }


def evaluate_reference_distribution(
    train_mols_valid: list[Molecule],
    predictor: RDKitPropertyPredictor,
    regressor: MorganPic50Regressor,
    morgan: MorganFingerprinter,
    sample_size: int = 1000,
    seed: int = 42,
) -> dict:
    """Score a random sample of the training molecules as a reference distribution.

    Since Phase-0 ``LipmanFlowMatchingAdapter.generate()`` produces placeholder
    atoms with no valid SMILES, we use the training set itself as a proxy to show
    what the model was trained on and what a reasonable distribution looks like.
    """
    import random
    rng = random.Random(seed)
    sample = rng.sample(train_mols_valid, min(sample_size, len(train_mols_valid)))
    return evaluate_generated_molecules(sample, predictor, regressor, morgan)


# ---------------------------------------------------------------------------
# Plot training loss curve
# ---------------------------------------------------------------------------
def plot_loss_curve(
    losses: list[float],
    out_path: Path,
    title: str = "Flow Matching Training Loss",
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(losses, marker="o", markersize=3, linewidth=1, color="steelblue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("CFM Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[train_fm] Wrote loss curve to {out_path}")


# ---------------------------------------------------------------------------
# Write markdown report
# ---------------------------------------------------------------------------
def write_report(
    args: argparse.Namespace,
    split: SplitResult,
    losses: list[float],
    eval_metrics: dict,
    ref_metrics: dict,
    train_time: float,
    gpu_used: bool,
    checkpoint_path: Path,
    report_path: Path,
) -> None:
    init_loss = losses[0] if losses else float("nan")
    final_loss = losses[-1] if losses else float("nan")
    loss_ratio = final_loss / init_loss if init_loss > 0 else float("nan")

    n_train = split.n_train()
    n_val = split.n_val()
    n_test = split.n_test()

    content = f"""# Flow Matching (Ru) — Temporal OOD Evaluation

## Experiment Config

| Parameter | Value |
|-----------|-------|
| Metal | {args.metal} |
| Epochs | {args.epochs} |
| Batch size | {args.batch} |
| Learning rate | {args.lr} |
| Hidden dim | {args.hidden_dim} |
| EGNN layers | {args.n_layers} |
| Max atoms (OOM guard) | {args.max_atoms} |
| Generation budget | {args.n_gen} |
| Seed | {args.seed} |

## Dataset Split (TemporalSplitter, cutoff=2024)

| Split | Size |
|-------|------|
| Train | {n_train} |
| Val | {n_val} |
| Test (OOD) | {n_test} |

## Training

- **Initial loss**: {init_loss:.6f}
- **Final loss**: {final_loss:.6f}
- **Loss ratio (final/initial)**: {loss_ratio:.4f}
- **Wall-clock (training)**: {train_time:.1f}s
- **GPU used**: {gpu_used}

## Generation Quality

> **Phase-0 limitation**: ``LipmanFlowMatchingAdapter.generate()`` produces
> placeholder atoms (random element types, no bonds, no SMILES) because the
> full ligand-decoder (SMILES/atom-type decoding from the velocity field) is
> Phase 1 work. Generated-molecule metrics below are therefore 0 / empty.
> The **training set reference distribution** (right column) shows what the
> model was optimised on.

| Metric | Generated (Phase-0 placeholder) | Training Reference |
|--------|--------------------------------|--------------------|
| Valid molecules (RDKit-parseable) | {eval_metrics['n_valid']} / {args.n_gen} | {ref_metrics['n_valid']} |
| QED mean | {eval_metrics['qed_mean']:.4f} | {ref_metrics['qed_mean']:.4f} |
| QED std | {eval_metrics['qed_std']:.4f} | {ref_metrics['qed_std']:.4f} |
| QED median | {eval_metrics['qed_median']:.4f} | {ref_metrics['qed_median']:.4f} |
| Fraction drug-like (QED ≥ 0.5) | {eval_metrics['frac_druglike_qed05']:.3f} | {ref_metrics['frac_druglike_qed05']:.3f} |
| MolLogP mean | {eval_metrics['logp_mean']:.4f} | {ref_metrics['logp_mean']:.4f} |
| MolWt mean | {eval_metrics['mw_mean']:.1f} | {ref_metrics['mw_mean']:.1f} |
| TPSA mean | {eval_metrics['tpsa_mean']:.2f} | {ref_metrics['tpsa_mean']:.2f} |
| Predicted pIC50 mean | {eval_metrics['pic50_pred_mean']:.4f} | {ref_metrics['pic50_pred_mean']:.4f} |
| Predicted pIC50 std | {eval_metrics['pic50_pred_std']:.4f} | {ref_metrics['pic50_pred_std']:.4f} |
| Predicted pIC50 median | {eval_metrics['pic50_pred_median']:.4f} | {ref_metrics['pic50_pred_median']:.4f} |
| Hit rate @ top-5% (pIC50 ≥ 6) | {eval_metrics['hit_rate_top5_pic50']:.3f} | {ref_metrics['hit_rate_top5_pic50']:.3f} |

## Checkpoint

Saved to: `{checkpoint_path}`

## Training Loss Curve

![Loss curve](./fm_ru_train_loss.png)

## Notes

- Temporal split: train = year < 2024, test (OOD) = year ≥ 2024.
- pIC50 regressor = MLP on Morgan FP (radius=2, 2048 bits), trained on training set.
- Training reference = 1000-molecule random sample from the training set.
- Hit rate = fraction with predicted pIC50 ≥ 6.
- Phase-1: replace placeholder generation with a proper SMILES/element decoder.
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(content)
    print(f"[train_fm] Wrote report to {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    torch_available = True
    try:
        import torch
    except Exception:
        torch_available = False
        print("[train_fm] WARNING: torch not available, cannot use GPU features")
        return 1

    np.random.seed(args.seed)

    # --- 1. Load dataset ---------------------------------------------------
    print(f"[train_fm] Loading MetalCytoToxDB for metal={args.metal} ...")
    t0 = time.time()

    flt = CytotoxFilter(
        time_threshold=24.0,
        ic50_min=0.01,
        metal_whitelist=[args.metal],
        compute_pic50=True,
        compute_active=True,
    )
    ds = MetalCytotoxDataset.from_csv(filters=flt)
    n_total = len(ds)
    print(f"[train_fm]   {n_total} rows after filtering")

    # --- 2. Temporal split -------------------------------------------------
    print("[train_fm] Applying TemporalSplitter(cutoff_year=2024) ...")
    splitter = TemporalSplitter(cutoff_year=2024)
    split = splitter(ds)
    train_idx, val_idx, test_idx = split.train_idx, split.val_idx, split.test_idx
    print(f"[train_fm]   train={len(train_idx)}, val={len(val_idx)}, test(ood)={len(test_idx)}")

    # --- 3. Pre-load conformers for training set ----------------------------
    print("[train_fm] Pre-loading 3D conformers for training set (cached on disk) ...")
    t_conf = time.time()
    train_mols = preload_conformers(ds, train_idx, CACHE_DIR, desc="train")
    n_fail = sum(1 for m in train_mols if m is None)
    print(f"[train_fm]   {len(train_mols) - n_fail}/{len(train_mols)} conformers loaded ({n_fail} failures) in {time.time()-t_conf:.1f}s")

    # Filter to valid conformers with max atoms constraint
    train_mols_valid = [m for m in train_mols if m is not None and m.n_atoms <= args.max_atoms]
    n_filtered = sum(1 for m in train_mols if m is not None and m.n_atoms > args.max_atoms)
    print(f"[train_fm]   {len(train_mols_valid)} molecules with valid 3D conformers (filtered {n_filtered} with >{args.max_atoms} atoms)")

    if len(train_mols_valid) == 0:
        print("[train_fm] ERROR: no valid training molecules with 3D conformers")
        return 1

    # --- 4. Initialize adapter ---------------------------------------------
    print("[train_fm] Initializing LipmanFlowMatchingAdapter ...")
    adapter = LipmanFlowMatchingAdapter(
        ref_repo_path=str(PROJECT_ROOT / "molmetal" / "references" / "flow_matching"),
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        lr=args.lr,
    )
    adapter.setup()  # ROCm-first auto-detection
    gpu_used = str(adapter.device).startswith("cuda") or str(adapter.device).startswith("hip")
    print(f"[train_fm]   device = {adapter.device}, GPU used = {gpu_used}")

    # --- 5. Training loop ---------------------------------------------------
    print(f"[train_fm] Training for {args.epochs} epochs ...")
    t_train = time.time()
    losses: list[float] = []

    n_batches_per_epoch = max(1, len(train_mols_valid) // args.batch)

    for epoch in range(args.epochs):
        epoch_losses = []
        # Shuffle
        order = np.random.permutation(len(train_mols_valid))
        mol_batch: list[Molecule] = []

        for i, idx in enumerate(order):
            mol_batch.append(train_mols_valid[idx])
            if len(mol_batch) == args.batch or i == len(order) - 1:
                if mol_batch:
                    # Dummy pocket (unused in v1)
                    from molmetal.domain import Pocket
                    dummy_pocket = Pocket(
                        pdb_id="dummy",
                        coords=torch.zeros(0, 3),
                        atom_types=torch.zeros(0, dtype=torch.long),
                        residue_ids=torch.zeros(0, dtype=torch.long),
                        chain_ids=torch.zeros(0, dtype=torch.long),
                        mask=torch.zeros(0, dtype=torch.bool),
                        center=torch.zeros(3),
                    )
                    loss = adapter.train_step(dummy_pocket, mol_batch)
                    epoch_losses.append(loss)
                    mol_batch = []

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        losses.append(avg_loss)
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:4d}: loss={avg_loss:.6f}")

    train_time = time.time() - t_train
    print(f"[train_fm] Training done in {train_time:.1f}s ({train_time/args.epochs:.2f}s/epoch)")

    # --- 6. Save checkpoint -------------------------------------------------
    ckpt_out = Path(args.checkpoint_out) if args.checkpoint_out else CHECKPOINT_DIR / "fm_ru_temporal.pt"
    ckpt_out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model_state": adapter.velocity_field.state_dict(),
        "optimizer_state": adapter.optimizer.state_dict(),
        "losses": losses,
        "args": vars(args),
        "n_train": len(train_mols_valid),
    }
    torch.save(ckpt, str(ckpt_out))
    print(f"[train_fm] Saved checkpoint to {ckpt_out}")

    # --- 7. Plot loss curve -------------------------------------------------
    plot_out = Path(args.plot_out) if args.plot_out else REPORTS_DIR / "fm_ru_train_loss.png"
    plot_loss_curve(losses, plot_out)

    # --- 8. Generate molecules ----------------------------------------------
    print(f"[train_fm] Generating {args.n_gen} molecules ...")
    from molmetal.domain import Pocket as PocketCls
    dummy_pocket = PocketCls(
        pdb_id="dummy",
        coords=torch.zeros(0, 3),
        atom_types=torch.zeros(0, dtype=torch.long),
        residue_ids=torch.zeros(0, dtype=torch.long),
        chain_ids=torch.zeros(0, dtype=torch.long),
        mask=torch.zeros(0, dtype=torch.bool),
        center=torch.zeros(3),
    )
    gen_config = GenerationConfig(n_samples=args.n_gen, n_steps=50, seed=args.seed)
    # LipmanFlowMatchingAdapter.generate() ignores pocket and uses a fixed
    # n_atoms=8 placeholder. This is Phase 0 — full ligand decoding comes later.
    gen_mols = adapter.generate(dummy_pocket, gen_config)
    print(f"[train_fm]   generated {len(gen_mols)} raw molecules (placeholder atoms)")

    # --- 9. Score with RDKitPropertyPredictor -------------------------------
    print("[train_fm] Scoring generated molecules with RDKitPropertyPredictor ...")
    predictor = RDKitPropertyPredictor()
    predictor.setup(device="cpu")

    # --- 10. Train pIC50 regressor on training set --------------------------
    print("[train_fm] Training Morgan-FP → pIC50 regressor on training set ...")
    # Get SMILES + pIC50 for training set
    train_smiles = []
    train_pic50 = []
    for idx in train_idx:
        row = ds[idx]
        if row["smiles"] and not np.isnan(row["pIC50"]):
            train_smiles.append(row["smiles"])
            train_pic50.append(row["pIC50"])
    train_pic50 = np.array(train_pic50, dtype=np.float32)

    morgan = MorganFingerprinter(radius=2, n_bits=2048)
    X_train = morgan(train_smiles)  # shape (N, 2048)
    print(f"[train_fm]   {len(train_pic50)} training molecules with valid pIC50")

    regressor = MorganPic50Regressor(seed=args.seed)
    regressor.fit(X_train, train_pic50)
    train_pred = regressor.predict(X_train)
    train_mse = float(np.mean((train_pred - train_pic50) ** 2))
    train_r2 = float(1 - np.sum((train_pred - train_pic50)**2) / np.sum((train_pic50 - train_pic50.mean())**2))
    print(f"[train_fm]   regressor train MSE={train_mse:.4f} R2={train_r2:.4f}")

    # --- 11. Evaluate generated molecules ---------------------------------
    # Phase-0 limitation: generate() produces placeholder atoms (random element,
    # no bonds). We score the training reference distribution instead and report
    # it alongside the (empty) generation metrics.
    print("[train_fm] Evaluating generated molecules (training reference distribution) ...")
    eval_metrics = evaluate_generated_molecules(gen_mols, predictor, regressor, morgan)
    ref_metrics = evaluate_reference_distribution(train_mols_valid, predictor, regressor, morgan, sample_size=min(1000, len(train_mols_valid)), seed=args.seed)
    print(f"[train_fm]   generated: valid={eval_metrics['n_valid']}")
    print(f"[train_fm]   reference: valid={ref_metrics['n_valid']}, "
          f"QED_mean={ref_metrics['qed_mean']:.3f}, "
          f"frac_druglike={ref_metrics['frac_druglike_qed05']:.3f}, "
          f"pIC50_pred_mean={ref_metrics['pic50_pred_mean']:.3f}, "
          f"hit_rate={ref_metrics['hit_rate_top5_pic50']:.3f}")

    # --- 12. Write report --------------------------------------------------
    report_path = Path(args.report_out) if args.report_out else REPORTS_DIR / "fm_ru_temporal_eval.md"
    write_report(
        args=args,
        split=split,
        losses=losses,
        eval_metrics=eval_metrics,
        ref_metrics=ref_metrics,
        train_time=train_time,
        gpu_used=gpu_used,
        checkpoint_path=ckpt_out,
        report_path=report_path,
    )

    total_time = time.time() - t0
    print(f"\n[train_fm] DONE total_time={total_time:.1f}s  GPU={gpu_used}")
    print(f"[train_fm] report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
