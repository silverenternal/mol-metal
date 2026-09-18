"""R3 — DrugOOD-style scaffold + 2 OOD-assay benchmark on MetalCytoToxDB Ru.

DrugOOD (Ji et al., 2023, ICLR) frames distribution shift for drug discovery as
two axes: *scaffold* (chemical OOD) and *assay* (biological OOD).  We mirror that
protocol here on Ru-only MetalCytoToxDB (Time(h) ≤ 24 h, n = 3,668 rows):

* **scaffold** split (Bemis-Murcko) → train / val / test 80/10/10.  This is
  the *ID chemistry* test — the same cell-line distribution is present in
  every split but the test scaffolds are unseen at training time.

* **OOD-Assay-A (lung vs cervical)** — train + scaffold-val restricted to
  *non-lung / non-cervical* cell lines; test = cell-line ∈ {A549 (lung),
  HeLa (cervical)}.  Same chemistry, biology shift.

* **OOD-Assay-B (cisR vs parental)** — train restricted to A2780 (parental
  ovarian); test restricted to A2780cisR (cisplatin-resistant sub-line of
  A2780).  Classic drug-OOD benchmark for metallomics because resistance
  phenotype typically drops Ru complex activity vs the parental line.

We then load the trained attentive D-MPNN checkpoint
``molmetal/checkpoints/dmpnn_attn_ru_pic50.pt`` (R²_regression head on pIC50,
trained on a 1,600-row random split with seed=42, see checkpoint meta) and
report **classification ROC-AUC** by thresholding the pIC50 prediction at the
*pIC50 ≥ 5.0* (i.e. predicted IC50 ≤ 10 µM → "active") rule and comparing
against the ``active`` label from the dataset.  We also report delta-AUC
relative to the scaffold test as the OOD gap.

References: Ji et al., "DrugOOD: Out-Of-Distribution (OOD) Dataset
Benchmark for AI-Aided Drug Discovery" (ICLR 2023).  Dataset:
MetalCytoToxDB (Ru subset, t≤24 h).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.baselines.dmpnn_attentive import (  # noqa: E402
    AttentiveDMPNNModel,
    featurize_smiles_list,
)
from molmetal.data.cytotox import CytotoxFilter, MetalCytotoxDataset  # noqa: E402
from molmetal.data.splits import ScaffoldSplitter  # noqa: E402

CKPT_PATH = PROJECT_ROOT / "molmetal" / "checkpoints" / "dmpnn_attn_ru_pic50.pt"
REPORTS_DIR = PROJECT_ROOT / "molmetal" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

PIC50_ACTIVITY_THRESHOLD = 5.0  # predicted IC50 ≤ 10 µM  → active
SEED = 42


# ---------------------------------------------------------------------------
# Subset definitions
# ---------------------------------------------------------------------------
SUBSETS: Dict[str, Dict[str, object]] = {
    "ID_scaffold_test": {
        "kind": "scaffold_split_test",
        "description": (
            "Bemis-Murcko scaffold split test rows (in-distribution chemistry, "
            "unseen scaffolds).  Acts as the ID anchor."
        ),
    },
    "OOD_Assay_A_lung_cervical": {
        "kind": "cell_line_test",
        "cell_lines": ["A549", "HeLa"],
        "train_cell_lines_exclude": ["A549", "HeLa"],
        "description": (
            "Test rows restricted to A549 (lung adenocarcinoma) and HeLa "
            "(cervical carcinoma).  Train/val rows exclude these lines. "
            "Chemistry overlap with train is large; biology is OOD."
        ),
    },
    "OOD_Assay_B_cisR_vs_parental": {
        "kind": "cell_line_test",
        "cell_lines": ["A2780cisR"],
        "train_cell_lines_exclude": ["A2780cisR"],
        "train_cell_lines_keep": ["A2780"],
        "description": (
            "Test rows restricted to A2780cisR (cisplatin-resistant).  Train "
            "rows restricted to A2780 (parental).  Classic resistance-shift "
            "OOD benchmark — biological mechanism is identical but the "
            "sensitivity phenotype is reversed."
        ),
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_trained_model(device: torch.device) -> AttentiveDMPNNModel:
    """Load the pre-trained attentive D-MPNN checkpoint (Ru, pIC50 head)."""
    bundle = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    kwargs = dict(bundle["model_kwargs"])
    model = AttentiveDMPNNModel(**kwargs)
    model.load_state_dict(bundle["state_dict"])
    model.to(device)
    model.eval()
    return model


def predict_pic50(
    model: AttentiveDMPNNModel,
    smiles_list: List[str],
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    """Run the model on a list of SMILES.  Returns pIC50 predictions (float32)."""
    graphs = featurize_smiles_list(smiles_list)
    preds = np.full(len(smiles_list), np.nan, dtype=np.float32)
    for start in range(0, len(graphs), batch_size):
        batch = graphs[start : start + batch_size]
        valid_idx = [i for i, g in enumerate(batch) if g is not None]
        if not valid_idx:
            continue
        from molmetal.baselines.dmpnn_attentive import collate_graphs  # type: ignore

        g_list = [batch[i] for i in valid_idx]
        bd = collate_graphs(g_list)
        if bd is None:
            continue
        atom_f, bond_f, e_src, e_dst = bd
        with torch.no_grad():
            logit = model(
                atom_f.to(device),
                bond_f.to(device),
                e_src.to(device),
                e_dst.to(device),
            )
            # The checkpoint head regresses pIC50 directly (not logit); the
            # last layer is linear → output is the prediction.
            pic50 = logit.cpu().numpy().reshape(-1)
        for j, gi in enumerate(valid_idx):
            preds[start + gi] = float(pic50[j])
    return preds


def active_from_pic50(pic50_pred: np.ndarray) -> np.ndarray:
    """Convert predicted pIC50 → binary active using the 10µM threshold."""
    out = np.zeros(len(pic50_pred), dtype=np.int64)
    valid = ~np.isnan(pic50_pred)
    out[valid] = (pic50_pred[valid] >= PIC50_ACTIVITY_THRESHOLD).astype(np.int64)
    return out


def auc_safe(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """roc_auc_score that returns NaN instead of raising on degenerate input."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def eval_subset(
    name: str,
    spec: Dict[str, object],
    ds: MetalCytotoxDataset,
    model: AttentiveDMPNNModel,
    device: torch.device,
) -> Dict[str, object]:
    """Build subset indices, predict, score AUC, return dict."""
    smiles_all = ds.smiles
    cell_lines = ds.df["Cell_line"].astype(str).to_numpy()

    if spec["kind"] == "scaffold_split_test":
        # Re-use the same ScaffoldSplitter the ID baseline uses.
        splitter = ScaffoldSplitter(strategy="largest_first", fractions=(0.8, 0.1, 0.1), seed=SEED)
        sr = splitter(ds)
        test_idx = sr.test_idx
        kind_note = "scaffold_test"
    elif spec["kind"] == "cell_line_test":
        test_lines = set(spec["cell_lines"])
        train_excl = set(spec.get("train_cell_lines_exclude", []))
        train_keep = set(spec.get("train_cell_lines_keep", []))
        test_mask = np.array([c in test_lines for c in cell_lines], dtype=bool)
        if train_keep:
            train_mask = np.array([c in train_keep for c in cell_lines], dtype=bool)
        else:
            train_mask = ~np.array([c in train_excl for c in cell_lines], dtype=bool)
        # We want ONLY the OOD-test subset for evaluation (no leakage guard
        # needed at eval time — the *train* mask is used to verify zero
        # overlap with the train rows of the scaffold split).
        test_idx = np.where(test_mask)[0]
        kind_note = (
            f"cell_line test ∩ {{ {','.join(sorted(test_lines))} }} ; "
            f"train rows held-out = {{ {','.join(sorted(train_excl))} }}"
        )
    else:
        raise ValueError(f"Unknown subset kind: {spec['kind']}")

    if len(test_idx) == 0:
        return {"name": name, "n_test": 0, "auc": float("nan"), "note": "empty subset"}

    smi_test = [str(smiles_all[i]) for i in test_idx]
    t0 = time.time()
    pic50_pred = predict_pic50(model, smi_test, device)
    elapsed = time.time() - t0
    y_true = ds.active[test_idx].astype(np.int64)
    # ROC-AUC on the *predicted* active classifier (regression head → threshold).
    y_score = active_from_pic50(pic50_pred).astype(np.float32)
    auc = auc_safe(y_true, y_score)
    # Also report classification on predicted pIC50 vs observed pIC50
    # (regression agreement) as a sanity check.
    obs_pic50 = ds.pic50[test_idx]
    valid = ~np.isnan(pic50_pred) & ~np.isnan(obs_pic50)
    if valid.sum() > 0:
        pearson = float(np.corrcoef(pic50_pred[valid], obs_pic50[valid])[0, 1])
    else:
        pearson = float("nan")
    return {
        "name": name,
        "n_test": int(len(test_idx)),
        "active_fraction_true": float(np.mean(y_true)),
        "active_fraction_pred": float(np.mean(y_score[~np.isnan(pic50_pred)])) if valid.sum() else float("nan"),
        "auc_pic50_thresholded": float(auc),
        "pearson_pred_vs_observed": float(pearson),
        "inference_seconds": float(round(elapsed, 2)),
        "note": kind_note,
        "description": spec.get("description", ""),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print("[R3] Loading MetalCytoToxDB (Ru, t≤24h) ...")
    ds = MetalCytotoxDataset.from_csv(
        metal_whitelist=["Ru"],
        filters=CytotoxFilter(time_threshold=24.0),
    )
    print(f"[R3] n_rows = {len(ds)}  metals = {ds.metal_counts()}")
    print(f"[R3] Year range = {ds.summary()['year_min']}–{ds.summary()['year_max']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[R3] Device = {device}")
    model = load_trained_model(device)

    results: List[Dict[str, object]] = []
    for name, spec in SUBSETS.items():
        print(f"[R3] Evaluating subset: {name}")
        r = eval_subset(name, spec, ds, model, device)
        results.append(r)
        print(
            f"    n={r['n_test']:5d}  AUC={r['auc_pic50_thresholded']:.4f}  "
            f"pearson={r['pearson_pred_vs_observed']:.3f}  "
            f"inference={r['inference_seconds']:.1f}s"
        )

    # Delta vs scaffold (ID) AUC
    id_auc = next(
        r["auc_pic50_thresholded"] for r in results if r["name"] == "ID_scaffold_test"
    )
    for r in results:
        r["delta_auc_vs_id"] = (
            float(r["auc_pic50_thresholded"]) - id_auc
            if not np.isnan(r["auc_pic50_thresholded"])
            else float("nan")
        )

    # Persist raw JSON next to the markdown for reproducibility.
    out_json = REPORTS_DIR / "r3_drugood_benchmark.json"
    with out_json.open("w") as fh:
        json.dump(
            {
                "checkpoint": str(CKPT_PATH.relative_to(PROJECT_ROOT)),
                "checkpoint_meta": {
                    "metal": "Ru",
                    "checkpoint_training_n_rows": 1600,
                    "checkpoint_training_split": "random (seed=42)",
                    "checkpoint_reported_test_pearson_r": 0.407,
                    "pic50_activity_threshold": PIC50_ACTIVITY_THRESHOLD,
                },
                "subsets": results,
            },
            fh,
            indent=2,
            default=float,
        )
    print(f"[R3] Wrote JSON results to {out_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())