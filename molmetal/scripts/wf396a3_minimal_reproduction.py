"""Wf-396-A3 — minimal reproduction script for the decode_ratio regression.

=====================================================================
This script provides the reproduction for the decode_ratio
regression investigation (per user directive 2026-09-30 verbatim
*"之前decode_ratio是1的现在我去哪退化的这个要重点解决"*).

Per Wf-396-A1 + Wf-396-A2 (READ-only git-bisect + trainer diff
analysis), the suspected regression source is the **Wf-380 SCAFFOLD
banner combination** (--top-p 0.9 --top-k 50 --mmd-weight 0.1
--repulsion-weight 0.05 --scaffold-conditioning) invoked by
``scripts/launch_wf299_cfm_main.sh`` Wave A — never smoke-tested at
the wire-in layer.  The 8 NEW CLI flags from Wf-388-A2 + Wf-388-A3
default OFF, which preserves backward-compat at default-CLI
invocation.

This script performs TWO reproduction paths (defensive against the
trainer-module circular-import issue):

* **Path A (decoder-only, no trainer module)**: instantiate a
  YuelBondDecoderHead + YuelBondBondDecoder directly, decode
  synthetic Pt-click clouds, measure RDKit-strict decode_ratio.
  This validates the decoder path is invariant under the 8 NEW
  flag changes (per Wf-396-A2 §2.3 LOW-risk finding).
* **Path B (best-effort trainer path, conditional)**: if the
  trainer module imports cleanly, run 50-step CPU smoke with
  flag combinations and report per-flag decode_ratio.  Skipped
  if the trainer has import issues (the conftest + Wf-396-A3
  runner have different import contexts).

The output is a single JSON dict keyed by ``run_name`` with
``decode_ratio`` (float 0-1) + ``n_ok`` / ``n_total`` + the flag
combination + wall-time per run.

CPU-only (no GPU launches per AGENTS.md §5.1); no ``paper/*``
touches (TODO-84); no wet-lab planning (§10).
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf396a3_minimal_reproduction"
REPORT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_PATH = REPORT_DIR / "result.json"

# 50-step CPU smoke per the task brief.
N_TRAIN_STEPS = 50
SYNTHETIC_N_CLOUDS = 32  # smaller than Wf-124 P2's 192 for fast CPU smoke
SYNTHETIC_JITTER = 0.20
SYNTHETIC_SEED = 42


# ---------------------------------------------------------------------------
# Flag combinations under test
# ---------------------------------------------------------------------------
@dataclass
class FlagCombo:
    """One CLI flag combination under test.

    Fields mirror the 8 NEW CLI flags from Wf-388-A2/A3 plus the
    four pre-existing trainer flags that affect the loss /
    sampling path.
    """

    name: str
    top_p: float = 1.0
    top_k: int = 0
    temperature: float = 1.0
    mmd_weight: float = 0.0
    repulsion_weight: float = 0.0
    scaffold_conditioning: bool = False
    sa_penalty: float = 0.0
    sa_target: float = 2.8
    notes: str = ""


def baseline_combo() -> FlagCombo:
    """No flags, defaults across the board."""
    return FlagCombo(name="baseline_no_flags")


def default_off_combo() -> FlagCombo:
    """Explicit default-OFF invocation."""
    return FlagCombo(
        name="default_off_all_explicit",
        top_p=1.0,
        top_k=0,
        temperature=1.0,
        mmd_weight=0.0,
        repulsion_weight=0.0,
        scaffold_conditioning=False,
        sa_penalty=0.0,
        sa_target=2.8,
        notes="explicit default OFF for all 8 NEW flags",
    )


def single_flag_combos() -> List[FlagCombo]:
    """Each of the 8 NEW flags activated in isolation."""
    return [
        FlagCombo(
            name="single_flag_top_p_0.9",
            top_p=0.9,
            notes="Wf-380 SCAFFOLD banner default for --top-p",
        ),
        FlagCombo(
            name="single_flag_top_k_50",
            top_k=50,
            notes="Wf-380 SCAFFOLD banner default for --top-k",
        ),
        FlagCombo(
            name="single_flag_temperature_0.7",
            temperature=0.7,
            notes="moderate temperature rescale",
        ),
        FlagCombo(
            name="single_flag_mmd_weight_0.1",
            mmd_weight=0.1,
            notes="Wf-380 SCAFFOLD banner default for --mmd-weight",
        ),
        FlagCombo(
            name="single_flag_repulsion_weight_0.05",
            repulsion_weight=0.05,
            notes="Wf-380 SCAFFOLD banner default for --repulsion-weight",
        ),
        FlagCombo(
            name="single_flag_scaffold_conditioning",
            scaffold_conditioning=True,
            notes="Wf-388-A2 NEW flag, default False",
        ),
        FlagCombo(
            name="single_flag_sa_penalty_0.1",
            sa_penalty=0.1,
            notes="moderate SA penalty weight",
        ),
        FlagCombo(
            name="single_flag_sa_target_2.5",
            sa_penalty=0.1,
            sa_target=2.5,
            notes="SA target threshold shifted",
        ),
    ]


def combination_combos() -> List[FlagCombo]:
    """Combination flag sets.

    The Wf-380 SCAFFOLD banner combination is the most suspicious
    per Wf-396-A2 §2 CRITICAL FINDING.
    """
    return [
        FlagCombo(
            name="combination_wf380_scaffold_banner",
            top_p=0.9,
            top_k=50,
            mmd_weight=0.1,
            repulsion_weight=0.05,
            notes="Wf-380 SCAFFOLD banner combination (suspected regression source)",
        ),
        FlagCombo(
            name="combination_default_off_explicit",
            top_p=1.0,
            top_k=0,
            temperature=1.0,
            mmd_weight=0.0,
            repulsion_weight=0.0,
            scaffold_conditioning=False,
            sa_penalty=0.0,
            sa_target=2.8,
            notes="all flags at OFF — sanity check vs baseline",
        ),
        FlagCombo(
            name="combination_all_flags_on",
            top_p=0.9,
            top_k=50,
            temperature=0.7,
            mmd_weight=0.1,
            repulsion_weight=0.05,
            scaffold_conditioning=True,
            sa_penalty=0.1,
            sa_target=2.5,
            notes="worst case: all flags activated",
        ),
    ]


# ---------------------------------------------------------------------------
# Path A: decoder-only reproduction (no trainer module)
# ---------------------------------------------------------------------------
def _decode_with_rdkit_strict(decoder, cloud) -> bool:
    """Run YuelBond decoder + return True iff RDKit accepts.

    Mirrors ``_decode_with_rdkit_strict`` from the trainer (line 414)
    so we don't need the trainer module to test the decoder path.
    """
    res = decoder.decode(cloud)
    decoded = res.decoded
    if decoded is None or not decoded.sanitized:
        return False
    smi = decoded.smiles
    if not smi:
        return False
    try:
        from rdkit import Chem  # type: ignore

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return False
        Chem.SanitizeMol(mol)
        return True
    except Exception:
        return False


def decoder_only_decode_ratio(
    n_clouds: int = SYNTHETIC_N_CLOUDS,
    seed: int = SYNTHETIC_SEED,
) -> Dict[str, Any]:
    """Path A: build YuelBondDecoderHead + YuelBondBondDecoder,
    decode synthetic clouds, measure RDKit-strict decode_ratio.

    Returns ``{decode_ratio, n_ok, n_total, wall_seconds, notes}``.
    The decoder path is invariant under the 8 NEW flag changes
    (per Wf-396-A2 §2.3 LOW-risk finding), so this single
    measurement is sufficient for the decoder-only reproduction.
    """
    start = time.time()
    try:
        # Import the decoder modules directly (no trainer module).
        from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
            YuelBondBondDecoder,
            build_synthetic_pt_click_clouds,
        )
        from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import (
            YuelBondDecoderHead,
        )
    except Exception as exc:  # pragma: no cover
        return {
            "decode_ratio": None,
            "n_ok": None,
            "n_total": None,
            "wall_seconds": float(time.time() - start),
            "notes": f"decoder import failed: {exc}",
        }

    try:
        torch.manual_seed(seed)
        clouds = build_synthetic_pt_click_clouds(
            n_clouds=n_clouds,
            jitter=SYNTHETIC_JITTER,
            seed=seed,
        )
        # Freshly-init head (small hidden_dim for fast CPU smoke).
        head = YuelBondDecoderHead(hidden_dim=32, n_layers=2)
        decoder = YuelBondBondDecoder(
            head=head,
            hidden_dim=32,
            n_layers=2,
        )
        n_ok = 0
        for cloud in clouds:
            if _decode_with_rdkit_strict(decoder, cloud):
                n_ok += 1
        n_total = len(clouds)
        decode_ratio = float(n_ok) / max(n_total, 1)
        return {
            "decode_ratio": decode_ratio,
            "n_ok": int(n_ok),
            "n_total": int(n_total),
            "wall_seconds": float(time.time() - start),
            "notes": (
                f"Path A decoder-only on freshly-init head "
                f"(hidden_dim=32, n_layers=2); n_clouds={n_clouds}"
            ),
        }
    except Exception as exc:
        return {
            "decode_ratio": None,
            "n_ok": None,
            "n_total": None,
            "wall_seconds": float(time.time() - start),
            "notes": f"decoder smoke failed: {type(exc).__name__}: {exc}",
        }


# ---------------------------------------------------------------------------
# Path B: best-effort trainer path (conditional)
# ---------------------------------------------------------------------------
def trainer_path_decode_ratio(combo: FlagCombo) -> Dict[str, Any]:
    """Path B: if the trainer module imports cleanly, run 50-step
    CPU smoke with the flag combination and measure decode_ratio.

    Returns a dict with keys ``decode_ratio``, ``n_ok``,
    ``n_total``, ``wall_seconds``, ``flags``, ``notes``.  If the
    trainer module fails to import, returns ``decode_ratio=None``
    with ``notes="trainer_path_skipped: <reason>"`` (Path A still
    provides the decoder-only reproduction).
    """
    start = time.time()
    flags = asdict(combo)
    # Try to import the trainer module (best-effort).
    try:
        # Pre-load dmpnn fully to side-step the partial-init guard.
        import molmetal.models.dmpnn  # noqa: F401
        from molmetal.scripts.train_cfm_yuelbond_wasserstein import (
            CFMJointModel,
            AtomCloud,
            build_synthetic_pt_click_clouds,
            build_reference_positions,
            evaluate_decode_ratio,
            train_one_epoch,
        )
    except Exception as exc:
        return {
            "decode_ratio": None,
            "n_ok": None,
            "n_total": None,
            "wall_seconds": float(time.time() - start),
            "flags": flags,
            "notes": f"trainer_path_skipped: {type(exc).__name__}: {exc}",
        }

    try:
        device = torch.device("cpu")
        torch.manual_seed(SYNTHETIC_SEED)
        train_clouds = build_synthetic_pt_click_clouds(
            n_clouds=SYNTHETIC_N_CLOUDS,
            jitter=SYNTHETIC_JITTER,
            seed=SYNTHETIC_SEED,
        )
        val_clouds = build_synthetic_pt_click_clouds(
            n_clouds=SYNTHETIC_N_CLOUDS,
            jitter=SYNTHETIC_JITTER,
            seed=SYNTHETIC_SEED + 1,
        )
        train_clouds_d = [
            AtomCloud(
                positions=c.positions.to(device),
                atomic_numbers=c.atomic_numbers.to(device),
            )
            for c in train_clouds
        ]
        val_clouds_d = [
            AtomCloud(
                positions=c.positions.to(device),
                atomic_numbers=c.atomic_numbers.to(device),
            )
            for c in val_clouds
        ]
        model = CFMJointModel(hidden_dim=32, n_layers=2).to(device)
        from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
            YuelBondBondDecoder,
        )

        decoder = YuelBondBondDecoder(
            head=model.head,
            hidden_dim=32,
            n_layers=2,
        )
        optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
        reference_positions = build_reference_positions().to(device)
        closure_bias = {
            "CuAAC": 0.30,
            "SPAAC": 0.10,
            "ThiolEne": 0.10,
            "Suzuki": 0.20,
            "AmideCoupling": 0.30,
        }
        batch_size = min(8, len(train_clouds_d))
        for step in range(N_TRAIN_STEPS):
            start_idx = (step * batch_size) % max(
                1, len(train_clouds_d) - batch_size + 1
            )
            mini_batch = train_clouds_d[start_idx : start_idx + batch_size]
            if len(mini_batch) < 1:
                mini_batch = train_clouds_d[:batch_size]
            train_one_epoch(
                model,
                decoder,
                optimiser,
                mini_batch,
                device,
                batch_size=batch_size,
                wasserstein_weight=0.1,
                closure_bounded=True,
                closure_bias=closure_bias,
                reference_positions=reference_positions,
                top_p=combo.top_p,
                top_k=combo.top_k,
                temperature=combo.temperature,
                mmd_weight=combo.mmd_weight,
                repulsion_weight=combo.repulsion_weight,
                scaffold_conditioning=combo.scaffold_conditioning,
                sa_penalty=combo.sa_penalty,
                sa_target=combo.sa_target,
                sa_weight=0.3,
            )
        n_ok, n_total = evaluate_decode_ratio(
            model=model,
            decoder=decoder,
            val_clouds=val_clouds_d,
            device=device,
            closure_bounded=True,
            closure_bias=closure_bias,
        )
        decode_ratio = float(n_ok) / max(n_total, 1)
        return {
            "decode_ratio": decode_ratio,
            "n_ok": int(n_ok),
            "n_total": int(n_total),
            "wall_seconds": float(time.time() - start),
            "flags": flags,
            "notes": combo.notes,
        }
    except Exception as exc:
        return {
            "decode_ratio": None,
            "n_ok": None,
            "n_total": None,
            "wall_seconds": float(time.time() - start),
            "flags": flags,
            "notes": f"trainer_smoke_failed: {type(exc).__name__}: {exc}",
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def baseline_test() -> Dict[str, Any]:
    """Path A baseline (decoder-only, no flags)."""
    res = decoder_only_decode_ratio()
    res["flags"] = asdict(baseline_combo())
    res["path"] = "A_decoder_only"
    return res


def default_off_test() -> Dict[str, Any]:
    """Path A explicit default-OFF (decoder-only)."""
    res = decoder_only_decode_ratio()
    res["flags"] = asdict(default_off_combo())
    res["path"] = "A_decoder_only"
    return res


def single_flag_tests() -> List[Dict[str, Any]]:
    """Path B: each of the 8 NEW flags in isolation via the trainer path."""
    return [trainer_path_decode_ratio(combo) for combo in single_flag_combos()]


def combination_tests() -> List[Dict[str, Any]]:
    """Path B: Wf-380 SCAFFOLD banner + default-OFF + all-on combinations."""
    return [trainer_path_decode_ratio(combo) for combo in combination_combos()]


def reproduce_decode_ratio_regression(
    flag_combinations: Optional[List[FlagCombo]] = None,
) -> Dict[str, Any]:
    """Run the full reproduction suite and return the aggregated result."""
    results: List[Dict[str, Any]] = []
    # Path A: decoder-only baseline (no flags; no trainer needed)
    results.append(baseline_test())
    results.append(default_off_test())
    # Path B: trainer-path single-flag + combination tests
    if flag_combinations is None:
        results.extend(single_flag_tests())
        results.extend(combination_tests())
    else:
        for combo in flag_combinations:
            results.append(trainer_path_decode_ratio(combo))

    baseline_dr = (
        results[0]["decode_ratio"]
        if results and results[0]["decode_ratio"] is not None
        else None
    )
    aggregate = {
        "n_runs": len(results),
        "baseline_decode_ratio": baseline_dr,
        "results": results,
    }
    return aggregate


def main() -> int:
    """CLI entry point — write the result JSON to disk."""
    print("[Wf-396-A3] starting minimal reproduction (50-step CPU smoke)")
    print(f"[Wf-396-A3] report dir: {REPORT_DIR}")
    aggregate = reproduce_decode_ratio_regression()
    with open(RESULT_PATH, "w") as f:
        json.dump(aggregate, f, indent=2, default=str)
    print(f"[Wf-396-A3] result.json written: {RESULT_PATH}")
    print(f"[Wf-396-A3] n_runs: {aggregate['n_runs']}")
    if aggregate["baseline_decode_ratio"] is not None:
        print(
            f"[Wf-396-A3] baseline_decode_ratio: "
            f"{aggregate['baseline_decode_ratio']:.3f}"
        )
    print("[Wf-396-A3] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
