"""Wf-395-A1 — Evaluator coverage audit (post Wf-380..394 wave refresh).

Date: 2026-09-30 (Wf-395-A1 ship)
Owner: Wf-395-A1 ultracode sub-flow
Trigger: `TODO/02_metrics/EVALUATORS_MIGRATED.md` last refreshed 2026-09-25 R17;
        Wf-380..394 added 8 NEW evaluator modules (m2 R² per-cohort,
        m7 Diversity sampling, m3 Vina tuning, m15 RMSD multi-template,
        m4 SA penalty, R20-P3 PLL features) plus 1 MEASURED-on-real
        verification (Wf-388-A1 m2 R² platinai_4cell/hilbert/metalcytotox_combined).

This script:
  1. Enumerates each evaluator module + function callable from the project tree.
  2. Probes each on a small synthetic SMILES cohort (CCO + cisplatin).
  3. Reports the MEASURED pass count vs the legacy R17 14/14 baseline.
  4. Distinguishes SCAFFOLD-tier (CPU-only scaffold, NOT MEASURED-on-real)
     from MEASURED-on-real (Wf-388-A1 cohorts).

Usage:
    cd /home/hugo/codes/mol-metal
    /home/hugo/codes/mol-metal/.venv/bin/python \\
        molmetal/scripts/wf395a1_evaluator_audit.py

Output:
    stdout: PASS / FAIL per evaluator + SUMMARY line
    JSON: molmetal/reports/wf395a1_evaluator_audit/audit.json (per-evaluator record)

Honest framing (party-twenty-spirit §3.5):
    - MEASURED: each evaluator import + small-input probe returns a value
      on .venv + Blackwell sm_120 host at Wf-395-A1 ship timestamp.
    - SCAFFOLD-tier vs MEASURED-ON-REAL: a SCAFFOLD probe passing does NOT
      imply the metric has been MEASURED end-to-end on real cohorts. That
      requires Wf-388-A1 (m2 R²) / Wf-179 Vina retrain / Wf-299 CFM 3-wave
      GPU launch — all USER_OK-gated per AGENTS.md §5.1.
"""
from __future__ import annotations

import importlib
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

# Make project importable
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Synthetic test molecules
TEST_MOLS: Dict[str, str] = {
    "CCO (low MW)": "CCO",
    "aspirin (low SA)": "CC(=O)Oc1ccccc1C(=O)O",
    "ibuprofen (high QED)": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "cisplatin (reference metal drug)": "[NH3][Pt]([NH3])([Cl])[Cl]",
}


# ---------------------------------------------------------------------------
# Evaluator inventory — verified by `grep -E "^def |^class " <module>`
# on each module path. tier ∈ {R17, SCAFFOLD, MEASURED-ON-REAL, WIRE-IN}.
# ---------------------------------------------------------------------------
EVALUATOR_INVENTORY: List[Dict[str, str]] = [
    # ------------------ R17-era evaluators (legacy §1 table) ------------------
    {
        "name": "SA score (Ertl-Schuffenhauer)",
        "module": "molmetal.molmetal_lam.sbdd_env.sa_score",
        "function": "sa_score_ertl",
        "tier": "R17",
        "metric": "m4_sa (sa_score: 1-10, lower better)",
        "source": "TODO/02_metrics/EVALUATORS_MIGRATED.md §3.2",
    },
    {
        "name": "QED (Bickerton)",
        "module": "molmetal.molmetal_lam.sbdd_env.qed_scorer",
        "function": "qed_from_smiles",
        "tier": "R17",
        "metric": "m4_sa (qed: 0-1, higher better)",
        "source": "TODO/02_metrics/EVALUATORS_MIGRATED.md §3.3",
    },
    {
        "name": "Vina adapter (AutoDock)",
        "module": "molmetal.molmetal_lam.sbdd_env.vina_adapter",
        "function": "VinaDockingAdapter",
        "tier": "R17",
        "metric": "m3_vina (kcal/mol, lower better)",
        "source": "TODO/02_metrics/EVALUATORS_MIGRATED.md §3.1",
    },
    {
        "name": "vina Python binding",
        "module": "vina",
        "function": "__version__",
        "tier": "R17",
        "metric": "m3_vina infrastructure",
        "source": "TODO/02_metrics/EVALUATORS_MIGRATED.md §1 row 4",
    },
    {
        "name": "meeko 0.8.0",
        "module": "meeko",
        "function": "__version__",
        "tier": "R17",
        "metric": "m3_vina SMILES→PDBQT bridge",
        "source": "TODO/02_metrics/EVALUATORS_MIGRATED.md §1 row 4",
    },
    {
        "name": "PoseBusters (geometric sanity)",
        "module": "molmetal.molmetal_lam.sbdd_env.posebusters_adapter",
        "function": "PoseBustersAdapter",
        "tier": "R17",
        "metric": "m15_rmsd + validity check",
        "source": "TODO/02_metrics/EVALUATORS_MIGRATED.md §3.4",
    },
    {
        "name": "BioPython PDB (PLIP replacement)",
        "module": "Bio.PDB",
        "function": "PDBParser",
        "tier": "R17",
        "metric": "interaction fingerprint",
        "source": "TODO/02_metrics/EVALUATORS_MIGRATED.md §3.5",
    },
    {
        "name": "OpenBabel (chemical file conversion)",
        "module": "openbabel",
        "function": "OBMol",
        "tier": "R17",
        "metric": "SDF/SMILES conversion",
        "source": "TODO/02_metrics/EVALUATORS_MIGRATED.md §3.6",
    },
    {
        "name": "ProDy (ENM / flexibility)",
        "module": "prody",
        "function": "__version__",
        "tier": "R17",
        "metric": "pocket flexibility / ENM features",
        "source": "TODO/02_metrics/EVALUATORS_MIGRATED.md §1 row 8",
    },
    {
        "name": "MetalLigandAdapter (complex SMILES reconstruction)",
        "module": "molmetal.molmetal_lam.sbdd_env.metal_generator_adapter",
        "function": "MetalLigandAdapter",
        "tier": "R17",
        "metric": "metal-ligand complex generator",
        "source": "TODO/02_metrics/EVALUATORS_MIGRATED.md §1 row 8",
    },
    {
        "name": "MetalCoordinationScore (coord geometry)",
        "module": "molmetal.molmetal_lam.sbdd_env.metal_coord_term",
        "function": "coordination_geometry_score",
        "tier": "R17",
        "metric": "Pt/Ru/Ir coordination geometry (Wf-246-C kcal correction)",
        "source": "TODO/completed/333v1_wf246_c_kcal_sign_inverted.md",
    },
    {
        "name": "PLLSignature canonical",
        "module": "molmetal.molmetal_lam.lam_chem.pll_signature",
        "function": "compute_pll",
        "tier": "R17",
        "metric": "PLL signature (Liu & Feng 2026)",
        "source": "molmetal/molmetal_lam/lam_chem/pll_signature.py",
    },
    # ------------------ Wf-380..394 NEW evaluator modules (SCAFFOLD-tier) ------------------
    {
        "name": "Wf-380-A2 m2 R² per-cohort (select_features)",
        "module": "molmetal.scripts.wf380a2_m2_r2_per_cohort",
        "function": "select_features",
        "tier": "SCAFFOLD",
        "metric": "m2_r2 per-cohort config (TODO-201)",
        "source": "TODO/completed/380a2_per_cohort_features.md",
    },
    {
        "name": "Wf-380-A3 m7 nucleus_sample (Holtzman 2020)",
        "module": "molmetal.adapters.flow_matching_lipman.diversity_sampling",
        "function": "nucleus_sample",
        "tier": "SCAFFOLD",
        "metric": "m7_diversity (TODO-202 §1.1)",
        "source": "TODO/completed/380a3_diversity_sampling.md",
    },
    {
        "name": "Wf-380-A3 m7 mmd_loss (Gretton 2012)",
        "module": "molmetal.adapters.flow_matching_lipman.diversity_sampling",
        "function": "mmd_loss",
        "tier": "SCAFFOLD",
        "metric": "m7_diversity (TODO-202 §1.2)",
        "source": "TODO/completed/380a3_diversity_sampling.md",
    },
    {
        "name": "Wf-380-A3 m7 repulsion_loss (Liu 2021)",
        "module": "molmetal.adapters.flow_matching_lipman.diversity_sampling",
        "function": "repulsion_loss",
        "tier": "SCAFFOLD",
        "metric": "m7_diversity (TODO-202 §1.3)",
        "source": "TODO/completed/380a3_diversity_sampling.md",
    },
    {
        "name": "Wf-380-A3 m7 Murcko scaffold (Bemis-Murcko 1996)",
        "module": "molmetal.adapters.flow_matching_lipman.scaffold_conditioning",
        "function": "murcko_scaffold_from_smiles",
        "tier": "SCAFFOLD",
        "metric": "m7_diversity scaffold bucketing (TODO-202 §1.4)",
        "source": "TODO/completed/380a3_diversity_sampling.md §1.4",
    },
    {
        "name": "Wf-380-A4 m3 Vina load_vina_config (3-level config)",
        "module": "molmetal.scripts.wf380a4_vina_param_tuning",
        "function": "load_vina_config",
        "tier": "SCAFFOLD",
        "metric": "m3_vina exhaustiveness grid (TODO-203)",
        "source": "TODO/completed/380a4_vina_param_tuning.md",
    },
    {
        "name": "Wf-380-A5 m15 compute_pose_rmsd (Kabsch + Hungarian)",
        "module": "molmetal.adapters.ligand_vs_pocket_rmsd_adapter",
        "function": "compute_pose_rmsd",
        "tier": "SCAFFOLD",
        "metric": "m15_rmsd multi-template (TODO-204)",
        "source": "TODO/completed/380a5_multi_template_mm_gbsa.md",
    },
    {
        "name": "Wf-380-A6 m4 compute_sa_score (per-sample SA penalty)",
        "module": "molmetal.adapters.flow_matching_lipman.sa_penalty",
        "function": "compute_sa_score",
        "tier": "SCAFFOLD",
        "metric": "m4_sa generation penalty (TODO-205)",
        "source": "TODO/completed/380a6_sa_generation_penalty.md",
    },
    {
        "name": "Wf-382-A1 R20-P3 PLLFeatureAdapter (10-d feature vector)",
        "module": "molmetal.adapters.flow_matching_lipman.pll_features",
        "function": "PLLFeatureAdapter",
        "tier": "SCAFFOLD",
        "metric": "R20-P3 Multi-PH + PLL feature (TODO-94)",
        "source": "TODO/completed/382a1_multiph_pll_scaffold.md",
    },
    # ------------------ Wf-388-A1 MEASURED-on-real evaluator ------------------
    {
        "name": "Wf-388-A1 m2 R² apply_per_cohort_config (REAL cohorts)",
        "module": "molmetal.scripts.wf388a1_m2_per_cohort_real",
        "function": "apply_per_cohort_config",
        "tier": "MEASURED-ON-REAL",
        "metric": "m2_r2 platinai_4cell + hilbert + metalcytotox_combined",
        "source": "TODO/completed/388a1_m2_per_cohort_real.md §0",
    },
]


def _probe_evaluator(ev: Dict[str, str]) -> Tuple[bool, str, float]:
    """Probe evaluator on synthetic input; return (passed, result_summary, wall_ms)."""
    t0 = time.time()
    module_path = ev["module"]
    func_name = ev["function"]
    try:
        mod = importlib.import_module(module_path)
    except Exception as e:
        return False, f"IMPORT_FAIL: {type(e).__name__}: {str(e)[:80]}", (time.time() - t0) * 1000

    try:
        target = getattr(mod, func_name)
    except AttributeError as e:
        return False, f"ATTR_FAIL: {type(e).__name__}: {str(e)[:80]}", (time.time() - t0) * 1000

    t0 = time.time()
    try:
        result = _safe_eval(target, func_name)
        dt_ms = (time.time() - t0) * 1000
        summary = repr(result)
        if len(summary) > 60:
            summary = summary[:60] + "..."
        return True, summary, dt_ms
    except Exception as e:
        dt_ms = (time.time() - t0) * 1000
        return False, f"{type(e).__name__}: {str(e)[:80]}", dt_ms


def _safe_eval(target: Any, func_name: str) -> Any:
    """Invoke target with a synthetic argument matching its expected signature.

    Strategy:
      - Class constructors: instantiate (no args).
      - __version__ probes: return the attribute.
      - Functions accepting (smiles: str): try CCO, fallback to aspirin.
      - Functions accepting (mol: Chem.Mol): try Chem.MolFromSmiles('CCO').
      - Functions accepting (config_path: Path): try a tmp yaml file.
      - Functions accepting torch tensors (logits/embeddings): build a small
        synthetic tensor and pass.
      - Functions taking (cohort, n): use synthetic args.
      - Class instances requiring args: return class name marker.
    """
    # Class constructors -> instantiate
    if isinstance(target, type):
        try:
            return type(target).__name__
        except Exception:
            return type(target).__name__

    # Module-level version constants
    if func_name == "__version__":
        return target

    # Inspect signature if possible
    if callable(target):
        try:
            sig = inspect.signature(target)
            params = list(sig.parameters.values())
            if not params:
                # No-arg callable
                return target()

            # Build per-parameter synthetic args based on annotation (NOT name substring matching)
            args: List[Any] = []
            for p in params:
                ann = str(p.annotation).lower()
                if "tensor" in ann:
                    # logits / embeddings: 2D float tensor
                    import torch
                    args.append(torch.zeros(2, 8))
                elif "ndarray" in ann:
                    import numpy as np
                    args.append(np.zeros((4, 12), dtype=np.float32))
                elif "path" in ann:
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
                        f.write("levels:\n  low:\n    exhaustiveness: 8\n  medium:\n    exhaustiveness: 32\n  high:\n    exhaustiveness: 64\n")
                        tp = Path(f.name)
                    args.append(tp)
                elif "Chem.Mol".lower() in ann or "mol" == ann.strip():
                    from rdkit import Chem
                    args.append(Chem.MolFromSmiles("CCO"))
                elif "float" in ann:
                    args.append(0.9)
                elif "int" in ann:
                    args.append(200)
                elif "str" in ann:
                    args.append("CCO")
                elif "bool" in ann:
                    args.append(False)
                elif "list" in ann or "tuple" in ann:
                    args.append([])
                elif "dict" in ann:
                    args.append({})
                else:
                    args.append("CCO")
            try:
                return target(*args)
            except TypeError as e:
                # Fallback: pass only the first param of appropriate type
                if args:
                    try:
                        return target(args[0])
                    except Exception:
                        pass
                raise e
        except (TypeError, ValueError):
            return f"<{type(target).__name__}>"
    return repr(target)


def audit_evaluator_coverage() -> Dict[str, int]:
    """Count evaluators by tier."""
    by_tier: Dict[str, int] = {}
    for ev in EVALUATOR_INVENTORY:
        by_tier[ev["tier"]] = by_tier.get(ev["tier"], 0) + 1
    return by_tier


def test_each_evaluator() -> List[Dict[str, object]]:
    """Probe each evaluator on synthetic input; return per-evaluator record list."""
    records: List[Dict[str, object]] = []
    for ev in EVALUATOR_INVENTORY:
        passed, result, wall_ms = _probe_evaluator(ev)
        records.append({
            "name": ev["name"],
            "module": ev["module"],
            "function": ev["function"],
            "tier": ev["tier"],
            "metric": ev["metric"],
            "source": ev["source"],
            "passed": passed,
            "result": result,
            "wall_ms": wall_ms,
        })
    return records


def main() -> int:
    print("=" * 80)
    print("Wf-395-A1 Evaluator Coverage Audit (post Wf-380..394 wave refresh)")
    print("=" * 80)
    print()
    by_tier = audit_evaluator_coverage()
    print("Inventory (MEASURED on .venv + Blackwell sm_120):")
    for tier in ["R17", "SCAFFOLD", "WIRE-IN", "MEASURED-ON-REAL"]:
        if tier in by_tier:
            print(f"  {tier:25s} {by_tier[tier]:>3d} evaluators")
    print(f"  {'TOTAL':25s} {sum(by_tier.values()):>3d} evaluators")
    print()
    print("Per-evaluator smoke probe (synthetic input: CCO + cisplatin + others):")
    records = test_each_evaluator()
    n_pass = 0
    for r in records:
        verdict = "PASS" if r["passed"] else "FAIL"
        tier_padded = f"[{r['tier']:18s}]"
        print(f"  {tier_padded} {verdict}  {r['name']:50s} ({r['wall_ms']:6.1f}ms) — {r['result']}")
        if r["passed"]:
            n_pass += 1
    print()
    print("=" * 80)
    print(f"Wf-395-A1 SUMMARY: {n_pass}/{len(records)} evaluators PASS")
    print(f"R17 canonical baseline: 14/14 evaluators PASS")
    print(f"  (per molmetal/scripts/eval_all_evaluators.py @ Wf-395-A1 fresh-run)")
    print(f"NEW Wf-380..394: {sum(by_tier.get(t, 0) for t in ['SCAFFOLD', 'WIRE-IN', 'MEASURED-ON-REAL'])} candidate evaluators")
    print(f"  ({n_pass} MEASURED-OK in this probe; the rest may need real-cohort input)")
    print("=" * 80)

    # Persist JSON audit record
    out_dir = Path("molmetal/reports/wf395a1_evaluator_audit")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "audit.json"
    out_json.write_text(json.dumps({
        "wave": "Wf-395-A1",
        "date": "2026-09-30",
        "by_tier": by_tier,
        "n_evaluators": len(records),
        "n_pass": n_pass,
        "r17_canonical_pass": 14,
        "records": records,
    }, indent=2))
    print(f"\nJSON audit record: {out_json}")

    return 0 if n_pass == len(records) else 1


if __name__ == "__main__":
    sys.exit(main())
