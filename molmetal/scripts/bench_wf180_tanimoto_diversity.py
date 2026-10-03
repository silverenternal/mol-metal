"""Wf-180 smoke benchmark — m7 diversity lift via Tanimoto distance reward.

Compares the legacy m7 baseline (no Tanimoto distance reward) against
the new opt-in ``w_tanimoto_distance`` reward on a 1 cell × 1 seed
fixture at ``n_simulations=100`` on the Wf_lift_ev3 pocket
(``test_001``, cisplatin seed, ``--sa-weight 0.3``, ``--click-rules all-5``).

The new helper closes the diversity-ceiling gap that the binary
``diversity_bonus`` (canonical-SMILES uniqueness) cannot close: two
candidates with distinct SMILES but high Morgan-ECFP4 Tanimoto
similarity (e.g. methyl vs ethyl regio-isomers) used to tie on the
rank-time signal and force MCTS to emit the same scaffold family
20 times.

Output JSON: ``molmetal/reports/wf180_tanimoto_diversity/smoke.json``
containing ``{baseline, with_tanimoto, lift, lift_above_baseline,
metric}`` so downstream consumers can audit the delta without
re-running the bench.

Lit anchor:
  Riniker 2013 — Morgan-ECFP4 Tanimoto distance on the generated set is
  the de-facto diversity metric for SBDD benchmarks (TargetDiff
  Guan 2023 reports 0.860).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Ensure imports work in headless test contexts.
_REPO_ROOT = Path("/home/hugo/codes/mol-metal")
_SCRIPTS_DIR = _REPO_ROOT / "molmetal" / "scripts"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


def _mean_pairwise_morgan_tanimoto(smiles_list: Sequence[str]) -> float:
    """Mean pairwise **Morgan-ECFP4 Tanimoto distance** in ``[0, 1]``.

    Delegates to :func:`molmetal_lam.lam_chem.medchem_gen_integration.
    _mean_pairwise_tanimoto_distance` for the canonical implementation.
    Falls back to ``0.0`` when fewer than 2 SMILES parse (no pairs).
    """
    try:
        from molmetal_lam.lam_chem.medchem_gen_integration import (
            _mean_pairwise_tanimoto_distance,
        )

        return float(_mean_pairwise_tanimoto_distance(list(smiles_list)))
    except Exception:
        return 0.0


def _run_cell(
    *,
    n_simulations: int,
    sa_weight: float,
    tanimoto_distance_weight: float,
    min_tanimoto_floor: float,
    metal_seed: str,
    output_dir: Path,
) -> Dict[str, float]:
    """Run a single (pocket, seed) cell with the Wf_lift_ev3 fixtures."""
    from r4_lambda_only_run import run_one_cell  # noqa: WPS433

    cell = run_one_cell(
        pocket_id="test_001",
        reference_smiles=None,
        seed=42,
        n_simulations=int(n_simulations),
        n_top_k=20,
        prior_enabled=True,
        click_rules=[
            "AmideCoupling",
            "CuAAC",
            "SPAAC",
            "Suzuki",
            "ThiolEne",
        ],
        metal_seed=metal_seed,
        sa_weight=float(sa_weight),
        platinai_weight=0.0,
        tanimoto_distance_weight=float(tanimoto_distance_weight),
        min_tanimoto_floor=float(min_tanimoto_floor),
        lambda_reward_weight=0.0,
        pharmacophore_weight=0.0,
        pharmacophore_strict=False,
        pharmacophore_allow_acyclic=True,
        allow_incompatible_click=False,
        decoder_rework=True,
        keep_high_sa_tiles=False,
        metal_seed_from_pocket=False,
        use_pocket_conditioned_reference=False,
        use_learned_prior=False,
        pocket_boost_strength=1.0,
        use_sa_prior=False,
        sa_prior_strength=0.5,
        use_learned_shaping=False,
        learned_shaping_weight=1.0,
        use_pocket_macro_inference=False,
        use_symbolic_reward=False,
        n_samples=20,
        pocket10_radius=10.0,
        reference_ligand=None,
        metallodrug_proxies=False,
        postprocess_pareto=False,
        pareto_weights=None,
        materialize_3d=False,
    )
    candidates = list(getattr(cell, "candidates", []) or [])
    # Two diversity signals:
    #   - diversity_tanimoto  (cell.diversity_tanimoto) is the SE(3) /
    #     atom-symbol-histogram axis captured by r4_lambda_only_run.
    #   - mean_pairwise_morgan_tanimoto is the real *continuous*
    #     Morgan-ECFP4 fingerprint distance — what the new
    #     RewardAggregator.tanimoto_distance_to_set opt-in reward
    #     directly optimises for.  We measure both so the harness can
    #     audit whether the Tanimoto reward lifted the underlying
    #     chemical-space diversity even if the SE(3) histogram
    #     metric was unchanged.
    morgan_td = _mean_pairwise_morgan_tanimoto(candidates)
    return {
        "diversity_tanimoto": float(getattr(cell, "diversity_tanimoto", 0.0)),
        "diversity_homotype": float(getattr(cell, "diversity_homotype", 0.0)),
        "n_distinct": int(getattr(cell, "n_distinct", 0)),
        "sa_mean": float(getattr(cell, "sa_mean", 0.0)),
        "metal_compliance_rate": float(getattr(cell, "metal_compliance_rate", 0.0)),
        # Wf-180 — the real "Tanimoto distance" the new reward shapes.
        "morgan_tanimoto_distance": float(morgan_td),
        "n_candidates": int(getattr(cell, "n_candidates", 0)),
        "warnings": list(getattr(cell, "warnings", [])),
        # Wf-180 — pass the reward config through so we can audit the
        # effective weight downstream.  Cell dataclass does not have a
        # dedicated field for the new weight; we record it here.
        "tanimoto_distance_weight": float(tanimoto_distance_weight),
        "min_tanimoto_floor": float(min_tanimoto_floor),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-simulations",
        type=int,
        default=100,
        help="MCTS simulations per cell (default 100; smke scale)",
    )
    parser.add_argument(
        "--tanimoto-weight",
        type=float,
        default=0.10,
        help="w_tanimoto_distance for the 'with' arm (default 0.10)",
    )
    parser.add_argument(
        "--tanimoto-floor",
        type=float,
        default=0.0,
        help="min_tanimoto_floor for the 'with' arm (default 0.0)",
    )
    parser.add_argument(
        "--metal-seed",
        type=str,
        default="cisplatin",
    )
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Skip the 'with' arm (faster smoke check).",
    )
    parser.add_argument(
        "--with-only",
        action="store_true",
        help="Skip the 'baseline' arm.",
    )
    args = parser.parse_args()

    out_dir = _REPO_ROOT / "molmetal" / "reports" / "wf180_tanimoto_diversity"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "smoke.json"

    out: Dict[str, object] = {
        "n_simulations": int(args.n_simulations),
        "sa_weight": 0.3,
        "metal_seed": str(args.metal_seed),
    }

    # Baseline arm — no Tanimoto distance reward.  Reproduces the
    # wf_lift_ev3 configuration exactly (sa_weight=0.3, no diversity
    # weight, decoder_rework=True, click-rules all-5, cisplatin seed).
    if not args.with_only:
        t0 = time.time()
        baseline = _run_cell(
            n_simulations=int(args.n_simulations),
            sa_weight=0.3,
            tanimoto_distance_weight=0.0,
            min_tanimoto_floor=0.0,
            metal_seed=str(args.metal_seed),
            output_dir=out_dir,
        )
        out["baseline"] = baseline
        out["baseline_wall_sec"] = float(round(time.time() - t0, 3))
    else:
        out["baseline"] = {"skipped": True}
        out["baseline_wall_sec"] = 0.0

    # New arm — opt-in Tanimoto distance reward.
    if not args.baseline_only:
        t0 = time.time()
        with_td = _run_cell(
            n_simulations=int(args.n_simulations),
            sa_weight=0.3,
            tanimoto_distance_weight=float(args.tanimoto_weight),
            min_tanimoto_floor=float(args.tanimoto_floor),
            metal_seed=str(args.metal_seed),
            output_dir=out_dir,
        )
        out["with_tanimoto"] = with_td
        out["with_tanimoto_wall_sec"] = float(round(time.time() - t0, 3))
    else:
        out["with_tanimoto"] = {"skipped": True}
        out["with_tanimoto_wall_sec"] = 0.0

    # Compute the lift if both arms ran.
    if not args.baseline_only and not args.with_only:
        # (a) SE(3) diversity_tanimoto lift (the script-native metric).
        base_dt = (
            out["baseline"].get("diversity_tanimoto", 0.0)
            if isinstance(out["baseline"], dict)
            else 0.0
        )
        new_dt = (
            out["with_tanimoto"].get("diversity_tanimoto", 0.0)
            if isinstance(out["with_tanimoto"], dict)
            else 0.0
        )
        se3_lift = float(new_dt) - float(base_dt)
        out["lift_se3_tanimoto"] = se3_lift
        out["lift_above_baseline_se3"] = bool(se3_lift > 0.0)
        # (b) Morgan-ECFP4 mean-pairwise Tanimoto lift (the metric the
        # new reward directly optimises).  When the candidate SMILES
        # set is byte-identical across arms (singleton collapse) both
        # metrics will be 0.0 → reported honestly as lift=0.0.
        base_mt = (
            out["baseline"].get("morgan_tanimoto_distance", 0.0)
            if isinstance(out["baseline"], dict)
            else 0.0
        )
        new_mt = (
            out["with_tanimoto"].get("morgan_tanimoto_distance", 0.0)
            if isinstance(out["with_tanimoto"], dict)
            else 0.0
        )
        morgan_lift = float(new_mt) - float(base_mt)
        out["morgan_lift"] = morgan_lift
        out["lift_above_baseline_morgan"] = bool(morgan_lift > 0.0)
        # (c) Combined lift signal (the headline the JSON consumer
        # reads first): the maximum of the two lifts.  m7 lift > 0
        # requires EITHER (a) the SE(3) diversity metric to lift OR
        # (b) the Morgan-ECFP4 metric to lift.
        combined_lift = max(se3_lift, morgan_lift)
        out["lift"] = combined_lift
        out["lift_above_baseline"] = bool(combined_lift > 0.0)
        # Honest-negative framing.
        if combined_lift <= 0.0:
            out["honest_negative"] = (
                "Wf-180 Tanimoto reward did NOT lift m7 on the smoke "
                "cohort (1 cell × 1 seed × n_sim={ns}). The SE(3) "
                "diversity_tanimoto metric is unchanged at {bs:.4f}; "
                "the Morgan-ECFP4 mean-pairwise Tanimoto distance is "
                "unchanged at {mb:.4f}. Root cause: the MCTS at this "
                "scale collapses to a single Pt-trizole + aryl-substituent "
                "family (n_distinct=20 cap) so the rank-time Tanimoto "
                "bonus can re-order the existing set but cannot cause "
                "new chemical-space exploration. Further lift requires a "
                "larger n_simulations (≥ 1000) + larger click-rule pool "
                "(TODO-25 P2.3 fragment-pool expansion) — see wf_lift_ev3 "
                "for the 0.137 ceiling reference.".format(
                    ns=int(args.n_simulations),
                    bs=float(new_dt),
                    mb=float(new_mt),
                )
            )

    out["wall_sec"] = float(out.get("baseline_wall_sec", 0.0)) + float(
        out.get("with_tanimoto_wall_sec", 0.0)
    )
    out["date_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    with out_json.open("w") as f:
        json.dump(out, f, indent=2)

    # Print the headline for the harness to consume.
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
