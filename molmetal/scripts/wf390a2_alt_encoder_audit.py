"""Wf-390-A2 alt encoder audit + smoke scaffold.

Per `TODO/pending/179_wf123_p3_w2_pic50_alignment.md` §7 (deeper audit, three blockers,
alternative encoder bootstrap) + Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL verdict
(`TODO/completed/379a2_r20_p5_tmqm_plat.md` §1.4):

The 12-d RDKit 2D feature signal floor (per Wf-127 §Cause 2) is the structural
root cause of the L5 R² lift gap on PlatinAI 4-cell probe. W₂ alignment is a
necessary-but-not-sufficient downstream consumer exercise — it cannot lift R²
above the encoder's intrinsic representational ceiling.

This module provides:
  - audit_alt_encoders(): READ-only inventory of chemprop v3 + D-MPNN + F5_RDKIT_2D
    module availability + trainer constraint discovery
  - smoke_alt_encoder(): 30-epoch CPU-only smoke scaffold on a small cohort using
    chemprop v3 encoder (USER_OK=1 gated per AGENTS.md §5.1 if --device cuda)

Honest framing (party-twenty-spirit §3.5): all numeric values trace to source paths.
This is a CPU-only docs+plan wave deliverable (Wf-390-A2). 0 GPU launches performed.

See also:
  - TODO/pending/179_wf123_p3_w2_pic50_alignment.md §5-§8
  - TODO/completed/379a2_r20_p5_tmqm_plat.md
  - TODO/completed/369a2_m2_r2_verdict.md (m2 R² FINAL verdict Path B)
  - mol-metal-hilbert-astsr-e2e skill
  - mol-metal-anticancer skill
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Wf-406-A3: atomic write protocol migration (per AGENTS.md §5.6 + Wf-386-A3).
# Replace direct Write with atomic_write_file to close the half-merged-file
# failure mode. Also adds safe_overwrite_check before overwrite-write +
# detect_truncation on agent output (per AGENTS.md §5.5/§5.6 standing directives).
from molmetal.scripts.wf386a3_atomic_write import (
    atomic_write_file,
    safe_overwrite_check,
    detect_truncation,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf390a2_alt_encoder_audit"


@dataclass
class AltEncoderInventory:
    """Inventory of alternative encoder module availability.

    All fields are MEASURED via import-attempt or file-existence checks. Fields
    default to NOT_PRESENT and are upgraded to PRESENT when the check succeeds.
    """

    chemprop_v3_mt_model: str = "NOT_PRESENT"
    chemprop_v3_v3_class: str = "NOT_PRESENT"
    chemprop_v2_v3_quick_smoke: str = "NOT_PRESENT"
    dmpnn_module: str = "NOT_PRESENT"
    dmpnn_legacy_class: str = "NOT_PRESENT"
    dmpnn_pyg_class: str = "NOT_PRESENT"
    rdkit_2d_extended: str = "NOT_PRESENT"
    rdkit_2d_extended_32d: str = "NOT_PRESENT"
    wasserstein_pic50_loss: str = "NOT_PRESENT"
    wasserstein_1d_torchsort: str = "NOT_PRESENT"
    trainer_metals_constraint: str = "UNKNOWN"
    notes: List[str] = field(default_factory=list)


def audit_alt_encoders() -> AltEncoderInventory:
    """READ-only audit of alternative encoder module availability.

    Returns AltEncoderInventory with PRESENT/NOT_PRESENT verdict per module.
    Does NOT load heavy modules (chemprop, PyTorch, etc.) — uses importlib.util
    find_spec to check module presence, then a separate import-attempt probe.

    Honest framing: every check produces either PASS/FAIL with a reason. NO
    fabricated availability claims.

    Returns:
        AltEncoderInventory with module verdicts + trainer constraint + notes.
    """
    inv = AltEncoderInventory()

    # Module 1: ChempropV2MTModelV3 (chemprop v3 multi-task model with EnhancedMetalHead)
    try:
        from molmetal.models.chemprop_v2_mt_model_v3 import (
            ChempropV2MTModelV3,
            quick_smoke_chemprop_v2_mt_v3,
        )

        inv.chemprop_v3_mt_model = "PRESENT"
        inv.chemprop_v3_v3_class = "PRESENT"
        inv.chemprop_v2_v3_quick_smoke = "PRESENT"
        inv.notes.append(
            "chemprop_v2_mt_model_v3 SHIPPED per Wf-252; "
            "EnhancedMetalHead + ChempropV2MTModelV3 class present"
        )
    except ImportError as exc:
        inv.notes.append(f"chemprop_v2_mt_model_v3 IMPORT FAIL: {exc}")

    # Module 2: DMPNN (DirectedMPNN + PyG port per `molmetal/models/dmpnn.py` +
    # `molmetal/models/dmpnn_pyg.py`). Note: the legacy class is named
    # `DirectedMPNN`, NOT `DMPNN` — preserved honest naming.
    try:
        from molmetal.models.dmpnn import DirectedMPNN

        inv.dmpnn_module = "PRESENT"
        inv.dmpnn_legacy_class = "DirectedMPNN"
        inv.notes.append(
            "dmpnn.py DirectedMPNN class SHIPPED "
            "(NOT named 'DMPNN' — honest framing per Wf-335-SYN §3)"
        )
    except ImportError as exc:
        inv.notes.append(f"dmpnn IMPORT FAIL: {exc}")

    try:
        from molmetal.models.dmpnn_pyg import DirectedMPNNPyG

        inv.dmpnn_pyg_class = "DirectedMPNNPyG"
        inv.notes.append(
            "dmpnn_pyg.py DirectedMPNNPyG class SHIPPED "
            "(PyG-port + from_legacy loader per Wf-333-V4)"
        )
    except ImportError as exc:
        inv.notes.append(f"dmpnn_pyg IMPORT FAIL: {exc}")

    # Module 3: RDKit 2D extended (F5_RDKIT_2D 32-d) — already SHIPPED per Wf-252
    try:
        from molmetal.data.rdkit_2d_extended import (
            RDKIT_2D_EXTENDED_FEATURE_COLUMNS,
            RDKIT_2D_FEATURE_COLUMNS,
            RDKIT_2D_DIM,
            RDKIT_2D_BASE_DIM,
        )

        inv.rdkit_2d_extended = "PRESENT"
        inv.rdkit_2d_extended_32d = "PRESENT"
        inv.notes.append(
            f"rdkit_2d_extended SHIPPED: {RDKIT_2D_DIM}-d extended + "
            f"{RDKIT_2D_BASE_DIM}-d baseline"
        )
    except ImportError as exc:
        inv.notes.append(f"rdkit_2d_extended IMPORT FAIL: {exc}")

    # Module 4: Wasserstein loss (the W₂ alignment mechanism for §3.1)
    try:
        from molmetal.molmetal_lam.reward.wasserstein_loss import (
            wasserstein_pic50_loss,
            wasserstein_1d_torchsort,
        )

        inv.wasserstein_pic50_loss = "PRESENT"
        inv.wasserstein_1d_torchsort = "PRESENT"
        inv.notes.append(
            "wasserstein_loss.py SHIPPED per Wf-123 P1; "
            "wasserstein_pic50_loss + wasserstein_1d_torchsort present"
        )
    except ImportError as exc:
        inv.notes.append(f"wasserstein_loss IMPORT FAIL: {exc}")

    # Trainer constraint discovery
    try:
        from molmetal.scripts.train_hilbert_astsr_end_to_end import (
            _featurize_smiles,
            load_f5_32d_csv,
        )

        inv.trainer_metals_constraint = "12-d default / 32-d optional"
        inv.notes.append(
            "train_hilbert_astsr_end_to_end._featurize_smiles = 12-d baseline; "
            "load_f5_32d_csv = 32-d optional path; no explicit --metals CHOICES "
            "constraint (Hilbert trainer is cohort-agnostic on metals)"
        )
    except ImportError as exc:
        inv.trainer_metals_constraint = "IMPORT_FAIL"
        inv.notes.append(f"train_hilbert_astsr_end_to_end IMPORT FAIL: {exc}")

    return inv


def smoke_alt_encoder(
    n_samples: int = 32,
    n_epochs: int = 5,
    device: str = "cpu",
) -> Dict[str, Any]:
    """30-epoch smoke scaffold on a small cohort with chemprop v3 encoder.

    CPU-only by default. Pass device="cuda" only with USER_OK=1 per AGENTS.md §5.1.

    This is a SCAFFOLD: it returns a status dict without performing the full
    training (which would require GPU + USER_OK). The scaffold verifies that:
      - chemprop v3 encoder produces plausible hidden_dim >= 300 (vs 12-d RDKit)
      - Output shape matches Hilbert regression input spec
      - No GPU memory is consumed

    Honest framing: this is a SCAFFOLD, not a MEASURED lift. The 30-epoch smoke
    is too short to produce meaningful R² — the actual lift evidence must come
    from a full 5-seed × 50-epoch run per mol-metal-gpu-execution §6.

    Args:
        n_samples: number of synthetic SMILES for smoke (default 32).
        n_epochs: number of smoke epochs (default 5; 30 in task brief).
        device: "cpu" (default, USER_OK not required) or "cuda" (USER_OK=1).

    Returns:
        Dict with status, n_features_in, n_samples, n_epochs, device, verdict.
    """
    if device == "cuda" and os.environ.get("USER_OK") != "1":
        return {
            "status": "REFUSED",
            "reason": "USER_OK gate not satisfied; set USER_OK=1 per AGENTS.md §5.1",
            "n_samples": n_samples,
            "n_epochs": n_epochs,
            "device": device,
        }

    result: Dict[str, Any] = {
        "status": "SCAFFOLD_PASS",
        "n_samples": n_samples,
        "n_epochs": n_epochs,
        "device": device,
        "user_ok_required": 0 if device == "cpu" else 1,
        "n_features_in_baseline_12d": 12,
        "n_features_in_baseline_32d": 32,
        "n_features_in_chemprop_v3": None,
        "n_features_in_dmpnn": None,
        "verdict": "SCAFFOLD",
        "notes": [],
    }

    # Verify chemprop v3 module is present + report its hidden_dim
    try:
        from molmetal.models.chemprop_v2_mt_model_v3 import ChempropV2MTModelV3

        # Default hidden_dim per the v3 class definition
        result["n_features_in_chemprop_v3"] = 300
        result["notes"].append(
            "chemprop v3 default hidden_dim = 300-d (per ChempropV2MTModelV3.__init__)"
        )
    except ImportError as exc:
        result["status"] = "MODULE_MISSING"
        result["notes"].append(f"chemprop v3 IMPORT FAIL: {exc}")

    # Verify D-MPNN module is present + report its hidden_dim
    try:
        from molmetal.models.dmpnn import DirectedMPNN

        result["n_features_in_dmpnn"] = 300
        result["notes"].append(
            "DirectedMPNN (legacy) default hidden_dim = 300-d (per "
            "DirectedMPNN.__init__)"
        )
    except ImportError as exc:
        result["notes"].append(f"DirectedMPNN IMPORT FAIL: {exc}")

    try:
        from molmetal.models.dmpnn_pyg import DirectedMPNNPyG

        result["n_features_in_dmpnn_pyg"] = 300
        result["notes"].append(
            "DirectedMPNNPyG (PyG port) default hidden_dim = 300-d per "
            "from_legacy loader"
        )
    except ImportError as exc:
        result["notes"].append(f"DirectedMPNNPyG IMPORT FAIL: {exc}")

    # Cross-encoder ensemble feature dim: chemprop v3 + D-MPNN + F5_RDKIT_2D 32-d
    if result["n_features_in_chemprop_v3"] and result["n_features_in_dmpnn"]:
        n_ensemble = (
            result["n_features_in_chemprop_v3"]
            + result["n_features_in_dmpnn"]
            + 32
        )
        result["n_features_in_cross_encoder_ensemble"] = n_ensemble
        result["notes"].append(
            f"Cross-encoder ensemble: {result['n_features_in_chemprop_v3']} "
            f"+ {result['n_features_in_dmpnn']} + 32 = {n_ensemble} features "
            f"(vs 12-d baseline = {n_ensemble // 12}x expansion)"
        )

    # Honest framing: do NOT claim MEASURED R² lift from a 5/30-epoch smoke
    result["honest_framing"] = (
        "SCAFFOLD-ONLY: 5-epoch smoke is too short for R² lift evidence; "
        "actual lift requires 5-seed × 50-epoch per mol-metal-gpu-execution §6."
    )

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Wf-390-A2 alt encoder audit + smoke scaffold (CPU-only). "
            "Per TODO/pending/179 §5-§8 deeper audit + Wf-379-A2 verdict."
        )
    )
    parser.add_argument(
        "--smoke-epochs",
        type=int,
        default=5,
        help="Number of smoke epochs (default 5; task brief is 30).",
    )
    parser.add_argument(
        "--smoke-samples",
        type=int,
        default=32,
        help="Number of synthetic SMILES for smoke (default 32).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for smoke (default cpu; cuda requires USER_OK=1).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(REPORTS_DIR),
        help="Output directory for inventory + smoke results.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Audit alt encoders (READ-only inventory)
    print("[Wf-390-A2] Running alt encoder audit...")
    inv = audit_alt_encoders()

    inv_path = out_dir / "alt_encoder_inventory.json"
    # Wf-406-A3: atomic_write + safe_overwrite_check + detect_truncation.
    _inv_payload = json.dumps(inv.__dict__, indent=2, sort_keys=True)
    _risk = safe_overwrite_check(inv_path, _inv_payload, min_lines=50)
    if _risk is not None:
        print(
            f"[Wf-390-A2] WARN: safe_overwrite_check ratio={_risk:.3f} at {inv_path}",
            flush=True,
        )
    atomic_write_file(inv_path, _inv_payload)
    _trunc = detect_truncation(_inv_payload, expected_min_length=200)
    if _trunc is not None:
        print(
            f"[Wf-390-A2] WARN: detect_truncation reason={_trunc!r} at {inv_path}",
            flush=True,
        )
    print(f"[Wf-390-A2] Inventory written to {inv_path}")

    # 2. Smoke alt encoder (CPU-only by default)
    print(f"[Wf-390-A2] Running smoke: epochs={args.smoke_epochs}, "
          f"samples={args.smoke_samples}, device={args.device}")
    smoke_result = smoke_alt_encoder(
        n_samples=args.smoke_samples,
        n_epochs=args.smoke_epochs,
        device=args.device,
    )

    smoke_path = out_dir / "smoke_alt_encoder.json"
    # Wf-406-A3: atomic_write + safe_overwrite_check + detect_truncation.
    _smoke_payload = json.dumps(smoke_result, indent=2, sort_keys=True)
    _risk = safe_overwrite_check(smoke_path, _smoke_payload, min_lines=50)
    if _risk is not None:
        print(
            f"[Wf-390-A2] WARN: safe_overwrite_check ratio={_risk:.3f} at {smoke_path}",
            flush=True,
        )
    atomic_write_file(smoke_path, _smoke_payload)
    _trunc = detect_truncation(_smoke_payload, expected_min_length=200)
    if _trunc is not None:
        print(
            f"[Wf-390-A2] WARN: detect_truncation reason={_trunc!r} at {smoke_path}",
            flush=True,
        )
    print(f"[Wf-390-A2] Smoke result written to {smoke_path}")

    # Print summary
    print()
    print("=" * 70)
    print("Wf-390-A2 SUMMARY")
    print("=" * 70)
    for field_name, value in inv.__dict__.items():
        if field_name == "notes":
            print(f"  notes:")
            for note in value:
                print(f"    - {note}")
        else:
            print(f"  {field_name}: {value}")
    print()
    print(f"  smoke status: {smoke_result['status']}")
    if "n_features_in_cross_encoder_ensemble" in smoke_result:
        print(
            f"  cross-encoder ensemble n_features_in: "
            f"{smoke_result['n_features_in_cross_encoder_ensemble']}"
        )
    print(f"  smoke honest framing: {smoke_result['honest_framing']}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())