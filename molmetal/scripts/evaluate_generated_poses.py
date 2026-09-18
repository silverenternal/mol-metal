"""Physical evaluation of generated products against exact manifest receptors.

Search ranking remains separate. This module redocks the paired reference for
an explicitly labelled threshold, saves actual poses, and never reembeds them
for PoseBusters. A failed preparation/dock is retained in the denominator.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path


def summarize(rows: list[dict], total_generated: int) -> dict:
    values = [r["score_kcal_mol"] for r in rows
              if isinstance(r.get("score_kcal_mol"), (int, float))
              and math.isfinite(r["score_kcal_mol"])]
    passed = sum(r.get("posebusters", {}).get("pb_valid") is True for r in rows)
    better = sum(r.get("beats_redocked_reference") is True for r in rows)
    triple = sum(r.get("triple_threshold_redocked_reference") is True for r in rows)
    return {
        "n_generated": total_generated, "n_selected": len(rows),
        "n_docked": len(values), "n_pb_pass": passed,
        "pb_pass_rate_all_generated": passed / total_generated if total_generated else None,
        "pb_pass_rate_selected": passed / len(rows) if rows else None,
        "n_beats_redocked_reference": better,
        "n_triple_threshold_redocked_reference": triple,
        "triple_threshold_rate_all_generated": triple / total_generated if total_generated else None,
        "n_vina_below_minus8": sum(v < -8. for v in values),
        "better_reference_rate_all_generated": better / total_generated if total_generated else None,
        "vina_mean_docked_kcal_mol": sum(values) / len(values) if values else None,
        "vina_best_kcal_mol": min(values) if values else None,
    }


def evaluate_candidates(candidates: list[dict], receptor_path: str, ligand_path: str,
                        output_dir: str, *, seed: int = 42, engine: str = "vina",
                        exhaustiveness: int = 8, n_poses: int = 9,
                        top_k: int = 10, gpu_config: dict | None = None, progress=None,
                        total_generated: int | None = None,
                        relax_mmff94: bool = False,
                        relax_max_iters: int = 200) -> dict:
    import numpy as np
    import torch
    from rdkit import Chem
    from rdkit.Chem import Descriptors, QED, Crippen, rdMolDescriptors
    from rdkit.Contrib.SA_Score import sascorer
    from molmetal.domain import Molecule, Pocket
    from molmetal.ports import DockingConfig
    from molmetal_lam.sbdd_env.vina_adapter import VinaDockingAdapter
    from molmetal.scripts.receptor_preparation_for_evaluation import prepare_receptor_for_evaluation
    from molmetal.validation.posebusters_runner import check_docked_pose
    from molmetal_lam.sbdd_env.posebusters_adapter import mmff94s_relax_pose

    if min(exhaustiveness, n_poses, top_k) < 1:
        raise ValueError("Physical evaluation budgets must be positive")
    generated = [c for c in candidates if c.get("is_generated") is True]
    if total_generated is None:
        total_generated = len(generated)
    if isinstance(total_generated, bool) or not isinstance(total_generated, int) or total_generated < len(generated):
        raise ValueError("total_generated must include all supplied generated candidates")
    selected = generated[:top_k]
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running", "scope": "generated-product physical evaluation; not a SOTA comparison",
        "protocol": {"engine": engine, "device": "OpenCL GPU + CPU chemistry/refinement" if engine == "quickvina2-gpu" else "cpu", "cpu": None if engine == "quickvina2-gpu" else 1,
                     "seed": seed, "exhaustiveness": exhaustiveness, "n_poses": n_poses,
                     "top_k": top_k, "selection": "first retained generated products in search rank order",
                     "coverage_denominator": "all search-returned generated candidates before synthesis filtering and selection",
                     "posebusters_config": "dock", "pose_selection": "lowest finite docking energy",
                     "reference_threshold": "same-engine reference-ligand redocked score; not crystal-pose scoring",
                     # WF-PB-MMFF94-Relax — record the MMFF94s intra-ligand
                     # relaxation flag in the per-cell protocol so the
                     # paper audit can grep for it.
                     "mmff94s_relax": bool(relax_mmff94),
                     "mmff94s_relax_max_iters": int(relax_max_iters) if relax_mmff94 else None,
                     "dock_pipeline": ("dock -> MMFF94s relax -> PoseBusters "
                                       "(WF-PB-MMFF9494s-Relax, Halgren 1996)"
                                       if relax_mmff94 else "dock -> PoseBusters"),
                     },
        "source_sha256": {name: hashlib.sha256(Path(p).read_bytes()).hexdigest()
                          for name, p in (("receptor", receptor_path), ("reference_ligand", ligand_path))},
        "candidates": [], "reference": None,
    }
    def emit():
        report["summary"] = summarize(report["candidates"], total_generated)
        if progress:
            progress(report)
    emit()
    if not selected:
        report["status"] = "no_selected_candidates" if total_generated else "no_generated_candidates"
        emit()
        return report
    prep = prepare_receptor_for_evaluation(Path(receptor_path), out / "receptor")
    report["receptor_preparation"] = prep
    if not prep["passed"]:
        report["status"] = "receptor_preparation_failed"
        report["candidates"] = [{"smiles": c["smiles"], "status": "not_docked_receptor_failure"} for c in selected]
        emit()
        return report
    effective_receptor = prep["effective_pdb"]
    reference = next((m for m in Chem.SDMolSupplier(ligand_path, removeHs=False) if m is not None), None)
    if reference is None or reference.GetNumConformers() != 1:
        raise ValueError("Paired ligand must have exactly one valid conformer")
    heavy = [a.GetIdx() for a in reference.GetAtoms() if a.GetAtomicNum() > 1]
    xyz = reference.GetConformer().GetPositions()[heavy]
    if not len(heavy) or not np.isfinite(xyz).all():
        raise ValueError("Invalid reference heavy-atom coordinates")
    center = xyz.mean(axis=0)
    side = max(12., 2. * float(np.abs(xyz - center).max()) + 8.)
    report["protocol"].update(center_A=center.tolist(), box_size_A=[side] * 3)
    pocket = Pocket.from_pdb_file(effective_receptor, torch.tensor(center), radius=side / 2.)
    # D7: --engine {vina, qvina, quickvina2, both, all}
    # 'both'  -> Vina + QVina (two columns)
    # 'all'   -> Vina + QVina + QuickVina2 (three columns)
    # Single names -> only that engine's column.
    if engine in ("both", "all"):
        engine_set = ["vina", "qvina"] if engine == "both" else ["vina", "qvina", "quickvina2"]
        adapters = {name: VinaDockingAdapter(engine=name, cpu_count=1, default_box_padding=0)
                    for name in engine_set}
        primary_name = engine_set[0]
    elif engine == "quickvina2-gpu":
        from molmetal_lam.sbdd_env.vina_gpu_adapter import QuickVinaGPUAdapter
        if not gpu_config:
            raise ValueError("GPU docking requires an explicit validated GPU configuration")
        adapter = QuickVinaGPUAdapter(**gpu_config, default_box_padding=0, output_dir=out / "gpu_runs")
        report["protocol"].update(native_exhaustiveness_applied=False,
            gpu_threads=gpu_config.get("gpu_threads", 1000), gpu_search_depth=gpu_config.get("search_depth", 1),
            gpu_budget_note="OpenCL lanes/search depth are not equivalent to native Vina exhaustiveness")
        adapters = {None: adapter}
        primary_name = None
    else:
        adapter = VinaDockingAdapter(engine=engine, cpu_count=1, default_box_padding=0)
        adapters = {engine: adapter}
        primary_name = engine
    for adp in adapters.values():
        adp._receptor_pdbqt[pocket.pdb_id] = Path(prep["pdbqt"])
    # Primary adapter drives the headline score_kcal_mol column; the others
    # contribute per-engine columns (vina_score, qvina_score, ...).
    primary_adapter = adapters[primary_name] if primary_name is not None else adapter
    report["engine_metadata"] = primary_adapter.get_metadata()
    report["protocol"]["engine_set"] = list(adapters.keys())
    config = DockingConfig(seed=seed, exhaustiveness=exhaustiveness, n_poses=n_poses)

    def _score_column(name):
        return {"vina": "vina_score",
                "qvina": "qvina_score",
                "quickvina2": "quickvina2_score",
                None: "score_kcal_mol"}[name]

    def dock_one(smiles, name):
        row = {"smiles": smiles, "status": "docking_failed", "score_kcal_mol": None}
        try:
            parsed = Chem.MolFromSmiles(smiles)
            if parsed is None or parsed.GetNumAtoms() == 0:
                raise ValueError("Invalid candidate SMILES")
            row["descriptors"] = {"sa": float(sascorer.calculateScore(parsed)),
                                  "qed": float(QED.qed(parsed)), "logp": float(Crippen.MolLogP(parsed)),
                                  "tpsa": float(rdMolDescriptors.CalcTPSA(parsed)),
                                  "rotatable_bonds": int(rdMolDescriptors.CalcNumRotatableBonds(parsed)),
                                  "molecular_weight": float(Descriptors.MolWt(parsed))}
            # Run every requested engine and record each score in its own column.
            # The primary engine's column also lands in the legacy
            # ``score_kcal_mol`` field so single-engine callers still see it.
            last_seed = None
            for eng_name, adp in adapters.items():
                complexes = adp.dock(Molecule.from_smiles(smiles), pocket, config)
                last_seed = getattr(adp, "last_docking_seed", None)
                if not complexes:
                    row[_score_column(eng_name)] = None
                    continue
                best = min(range(len(complexes)), key=lambda i: complexes[i].vina_score)
                score = float(complexes[best].vina_score)
                if not math.isfinite(score):
                    row[_score_column(eng_name)] = None
                    continue
                row[_score_column(eng_name)] = score
            row["docking_seed"] = last_seed or {"requested": seed, "effective": None}
            # Pose + SDF + PoseBusters come from the PRIMARY engine only (the
            # other engines share the same receptor but not necessarily the
            # same pose orientation; PoseBusters is per-pose).
            primary_complexes = primary_adapter.dock(Molecule.from_smiles(smiles), pocket, config)
            if engine == "quickvina2-gpu":
                row["gpu_execution"] = dict(adapter.last_run_metadata)
            if not primary_complexes:
                raise ValueError("Docking returned no valid poses")
            best = min(range(len(primary_complexes)), key=lambda i: primary_complexes[i].vina_score)
            pose = primary_adapter.last_pose_mols[best]
            score = float(primary_complexes[best].vina_score)
            if not math.isfinite(score):
                raise ValueError("Non-finite docking score")
            pose_path = out / f"{name}.sdf"
            with Chem.SDWriter(str(pose_path)) as writer:
                pose.SetDoubleProp("docking_score_kcal_mol", score)
                writer.write(pose)
            row.update(status="docked", score_kcal_mol=score, pose_sdf=str(pose_path),
                       returned_poses=len(primary_complexes), selected_pose_index=best,
                       pose_sha256=hashlib.sha256(pose_path.read_bytes()).hexdigest())
            # WF-PB-MMFF94-Relax — Stage 1 of the dock -> MMFF94s relax
            # -> PB check pipeline.  Halgren 1996 *J. Comput. Chem.* 17,
            # 490-512 reports MMFF94s reaches 0.014 Å bond-length and
            # 1.2° angle RMS against MP2/6-31G*; Tosco 2014 RDKit MMFF
            # docs cite it as the production variant.  Vina's scoring
            # function does NOT optimise bonded terms, so un-relaxed
            # docked geometries frequently fail PoseBusters' tight
            # bond/angle windows (we observed this in the
            # WF-PB-Pass-10x3 baseline, which produced 0/30 PB-eligible
            # molecules because the search was bound at
            # n_simulations=100).  The relaxation is performed on a
            # CLONE so the original Vina pose remains intact on disk
            # (the SDF at ``pose_path`` is the un-relaxed pose).
            if relax_mmff94:
                pose_for_relax = Chem.Mol(pose)
                relax_report = mmff94s_relax_pose(
                    pose_for_relax, max_iters=int(relax_max_iters))
                # Capture a coarse pre/post relaxation RMSD so the
                # "preserves Vina pose" invariant is testable
                # downstream (the spec calls for within 0.5 A
                # co-ordinate deviation on average; we capture heavy-
                # atom RMS here, not the Vina energy delta which
                # would require an extra re-dock call).
                abs_rmsd_pre_post = None
                try:
                    if relax_report.get("ok"):
                        from rdkit import RDLogger  # type: ignore
                        RDLogger.DisableLog("rdApp.*")
                        from rdkit.Chem import AllChem  # type: ignore
                        # AllChem.GetBestRMS handles conformer
                        # alignment between identical molecules.
                        rms = AllChem.GetBestRMS(pose, pose_for_relax, prbId=0, refId=0)
                        abs_rmsd_pre_post = float(rms)
                except Exception:
                    abs_rmsd_pre_post = None
                row["mmff94s_relax"] = {
                    "ok": bool(relax_report.get("ok")),
                    "status": int(relax_report.get("status", -1)),
                    "variant": relax_report.get("variant", "MMFF94s"),
                    "max_iters": int(relax_max_iters),
                    "error": relax_report.get("error"),
                    "pre_relax_heavy_atom_rmsd_to_post_A": abs_rmsd_pre_post,
                }
                # If relaxation succeeded, run PB on the RELAXED pose
                # so the geometry checks land inside the MMFF94s
                # window.  Falls back to the original pose if
                # relaxation failed (no harm done).
                pose_for_pb = pose_for_relax if relax_report.get("ok") else pose
            else:
                row["mmff94s_relax"] = {"enabled": False}
                pose_for_pb = pose
            row["posebusters"] = check_docked_pose(pose_for_pb, effective_receptor)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            if engine == "quickvina2-gpu":
                row["gpu_execution"] = dict(getattr(adapter, "last_run_metadata", {}))
        return row

    report["reference"] = dock_one(Chem.MolToSmiles(Chem.RemoveHs(reference)), "reference_redocked")
    # When multi-engine mode is on, every per-engine reference threshold is
    # available; downstream code can use whichever matches the column it is
    # comparing against. ``threshold`` (legacy) keeps the primary engine.
    threshold = report["reference"].get("score_kcal_mol")
    report["reference_posebusters_status"] = report["reference"].get("posebusters", {}).get("status", "unavailable")
    emit()
    for i, candidate in enumerate(selected):
        row = dock_one(candidate["smiles"], f"candidate_{i:04d}")
        score = row.get("score_kcal_mol")
        row["beats_redocked_reference"] = score < threshold if score is not None and threshold is not None else None
        # Per-engine beats/threshold for multi-engine mode (D7).
        for eng_name in adapters.keys():
            col = _score_column(eng_name)
            ref_col = col  # same column in reference row
            cand_score = row.get(col)
            ref_score = report["reference"].get(ref_col)
            if cand_score is not None and ref_score is not None:
                row[f"beats_redocked_reference_{col}"] = cand_score < ref_score
        descriptors = row.get("descriptors", {})
        row["triple_threshold_redocked_reference"] = (
            bool(score < threshold and descriptors["sa"] < 4. and descriptors["qed"] > .5)
            if score is not None and threshold is not None and "sa" in descriptors and "qed" in descriptors else None)
        report["candidates"].append(row)
        emit()
    pb_complete = all(row.get("posebusters", {}).get("status") in ("passed", "failed") for row in report["candidates"])
    report["status"] = "completed" if (all(r["status"] == "docked" for r in report["candidates"])
                                                   and threshold is not None and pb_complete) else "partial_failure"
    emit()
    return report
