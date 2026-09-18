"""Actual pocket-conditioned docking energy for search, with a bounded cache.

Only reference ligand coordinates define the box; reference chemistry and
reference docking scores do not enter this reward. No score is fabricated
for invalid inputs, exhausted budgets or failed/unverified docking.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time


class PocketDockingRewardError(RuntimeError):
    pass


def _hash(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


class PocketDockingReward:
    """Callable raw kcal/mol provider; caller decides reward scaling/sign.

    ``max_evaluations`` counts actual docking attempts, including failed
    attempts. Cached results and invalid inputs consume no further docking
    budget. Every call is retained in report(), including cached failures.
    This provider never connects itself to a search runner or aggregator.
    """

    def __init__(self, receptor_path, ligand_path, output_dir, *, seed=42,
                 engine="quickvina2-gpu", gpu_config=None, max_evaluations=16,
                 exhaustiveness=1, n_poses=1):
        for name, value, minimum in (("max_evaluations", max_evaluations, 0),
                                     ("exhaustiveness", exhaustiveness, 1), ("n_poses", n_poses, 1)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        if engine not in ("quickvina2-gpu", "vina", "vina-cli", "qvina", "quickvina2"):
            raise ValueError("Unsupported measured docking engine")
        if engine == "quickvina2-gpu" and not gpu_config:
            raise ValueError("GPU reward requires explicit GPU configuration")
        if isinstance(gpu_config, (str, Path)):
            gpu_config = json.loads(Path(gpu_config).read_text())
        self.receptor_path = Path(receptor_path).expanduser().resolve()
        self.ligand_path = Path(ligand_path).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed, self.engine = seed, engine
        self.max_evaluations, self.exhaustiveness, self.n_poses = max_evaluations, exhaustiveness, n_poses
        self.gpu_config = copy.deepcopy(gpu_config)
        self._lock = threading.RLock()
        self._adapter = self._pocket = self._docking_config = None
        self._setup_attempted, self._setup_error = False, None
        self._cache: dict[str, int] = {}
        self._attempts = 0
        config = {"receptor_path": str(self.receptor_path), "ligand_path": str(self.ligand_path),
                  "seed": seed, "engine": engine, "gpu_config": self.gpu_config,
                  "max_evaluations": max_evaluations, "exhaustiveness": exhaustiveness, "n_poses": n_poses}
        self._record = {"created_utc": datetime.now(timezone.utc).isoformat(),
                        "provider": "PocketDockingReward", "config": config,
                        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
                        "source_sha256": {"receptor": _hash(self.receptor_path), "reference_geometry": _hash(self.ligand_path)},
                        "protocol": {"output": "raw best finite docking score in kcal/mol; lower is better",
                                     "reference_usage": "heavy-atom coordinates only define box; no reference chemical reward or reference score threshold",
                                     "box_rule": "same as evaluate_generated_poses: heavy-atom mean center; cube side=max(12,2*max_abs_deviation+8) Angstrom",
                                     "failure_policy": "raise; score_kcal_mol stays null; no proxy or CPU fallback",
                                     "cache_key": "RDKit canonical isomeric SMILES within one fixed receptor/config provider",
                                     "budget_unit": "actual docking attempts, including failures",
                                     "native_exhaustiveness_applied": engine != "quickvina2-gpu",
                                     "posebusters": "not a search reward component; saved actual poses are available for separate physical evaluation"},
                        "setup_status": "not_started", "evaluations": [], "calls": []}
        self._checkpoint()

    def report(self) -> dict:
        with self._lock:
            result = copy.deepcopy(self._record)
            calls, evaluations = result["calls"], result["evaluations"]
            result["summary"] = {"n_calls": len(calls), "n_cache_hits": sum(c.get("cache_hit", False) for c in calls),
                                 "n_unique_evaluation_records": len(evaluations),
                                 "n_docking_attempts": self._attempts, "max_evaluations": self.max_evaluations,
                                 "remaining_evaluations": max(0, self.max_evaluations-self._attempts),
                                 "n_docked": sum(e["status"] == "docked" for e in evaluations),
                                 "n_failed_calls": sum(c["status"] == "error" for c in calls),
                                 "n_budget_exhausted_calls": sum(c.get("error_code") == "budget_exhausted" for c in calls)}
            result.update(n_attempted=self._attempts, n_docked=result["summary"]["n_docked"],
                          n_cache_hits=result["summary"]["n_cache_hits"],
                          n_budget_exhausted=result["summary"]["n_budget_exhausted_calls"])
            result["candidates"] = result.pop("evaluations")
            return result

    def _checkpoint(self):
        target = self.output_dir / "search_docking_reward.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.report(), indent=2, allow_nan=False) + "\n")
        os.replace(temporary, target)

    def _ensure_ready(self):
        if self._setup_attempted:
            if self._setup_error:
                raise PocketDockingRewardError(self._setup_error)
            return
        self._setup_attempted = True
        self._record["setup_status"] = "running"
        self._checkpoint()
        try:
            import numpy as np
            import torch
            from rdkit import Chem
            from molmetal.domain import Pocket
            from molmetal.ports import DockingConfig
            from molmetal.scripts.receptor_preparation_for_evaluation import prepare_receptor_for_evaluation
            prep = prepare_receptor_for_evaluation(self.receptor_path, self.output_dir / "receptor")
            self._record["receptor_preparation"] = prep
            if prep.get("passed") is not True:
                raise PocketDockingRewardError(f"receptor_preparation_failed:{prep.get('error')}")
            reference = next((m for m in Chem.SDMolSupplier(str(self.ligand_path), removeHs=False) if m is not None), None)
            if reference is None or reference.GetNumConformers() != 1:
                raise ValueError("Paired ligand requires one valid conformer for box geometry")
            heavy = [a.GetIdx() for a in reference.GetAtoms() if a.GetAtomicNum() > 1]
            xyz = reference.GetConformer().GetPositions()[heavy]
            if not heavy or not np.isfinite(xyz).all():
                raise ValueError("Invalid paired-ligand heavy-atom coordinates")
            center = xyz.mean(axis=0)
            side = max(12.0, 2.0*float(np.abs(xyz-center).max())+8.0)
            self._record["protocol"].update(center_A=center.tolist(), box_size_A=[side]*3,
                                             reference_geometry_heavy_atoms=len(heavy))
            self._pocket = Pocket.from_pdb_file(prep["effective_pdb"], torch.tensor(center), radius=side/2)
            if self.engine == "quickvina2-gpu":
                from .vina_gpu_adapter import QuickVinaGPUAdapter
                options = copy.deepcopy(self.gpu_config)
                if not options.get("trace_library"):
                    raise ValueError("Search GPU reward requires a real kernel trace library")
                options["default_box_padding"] = 0
                options["output_dir"] = self.output_dir / "gpu_runs"
                self._adapter = QuickVinaGPUAdapter(**options)
                self._adapter.set_prepared_receptor(self._pocket, prep)
            else:
                from .vina_adapter import VinaDockingAdapter
                self._adapter = VinaDockingAdapter(engine=self.engine, cpu_count=1, default_box_padding=0)
                self._adapter._receptor_pdbqt[self._pocket.pdb_id] = Path(prep["pdbqt"])
            self._record["engine_metadata"] = self._adapter.get_metadata()
            self._docking_config = DockingConfig(seed=self.seed, exhaustiveness=self.exhaustiveness, n_poses=self.n_poses)
            self._record["setup_status"] = "ready"
        except Exception as exc:
            self._setup_error = f"{type(exc).__name__}:{exc}"
            self._record.update(setup_status="failed", setup_error=self._setup_error)
            raise
        finally:
            self._checkpoint()

    @staticmethod
    def _canonical(state):
        from rdkit import Chem
        if isinstance(state, str):
            value = state
        elif callable(getattr(state, "canonical_smiles", None)):
            value = state.canonical_smiles()
        else:
            value = getattr(state, "smiles", None)
        if not isinstance(value, str) or not value.strip():
            raise ValueError("missing_or_empty_smiles")
        mol = Chem.MolFromSmiles(value)
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError("invalid_smiles")
        return value, Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

    def _run_docking(self, canonical, evaluation):
        import numpy as np
        from rdkit import Chem
        from molmetal.domain import Molecule
        self._ensure_ready()
        molecule = Molecule.from_smiles(canonical)
        self._attempts += 1
        evaluation.update(docking_attempt=self._attempts, status="running")
        self._checkpoint()
        try:
            complexes = self._adapter.dock(molecule, self._pocket, self._docking_config)
            evaluation["docking_seed"] = copy.deepcopy(getattr(self._adapter, "last_docking_seed", {}))
            if self.engine == "quickvina2-gpu":
                metadata = copy.deepcopy(self._adapter.last_run_metadata)
                evaluation["gpu_execution"] = metadata
                if not metadata.get("gpu_verified") or metadata.get("gpu_evidence") != "kernel_trace":
                    raise ValueError("GPU search reward requires verified completed kernel traces")
            poses = self._adapter.last_pose_mols
            if not complexes or len(complexes) != len(poses):
                raise ValueError("Docking returned no unambiguous score/pose pairs")
            directory = self.output_dir / f"evaluation_{evaluation['evaluation_id']:04d}"
            directory.mkdir(exist_ok=True)
            records = []
            for index, (complex_, pose) in enumerate(zip(complexes, poses)):
                score = float(complex_.vina_score)
                if not math.isfinite(score) or pose.GetNumConformers() != 1 or not np.isfinite(pose.GetConformer().GetPositions()).all():
                    raise ValueError("Docking returned a nonfinite score or invalid pose")
                if Chem.MolToSmiles(Chem.RemoveHs(pose), canonical=True, isomericSmiles=True) != canonical:
                    raise ValueError("Docked pose chemistry differs from canonical candidate")
                path = directory / f"pose_{index:03d}.sdf"
                saved = Chem.Mol(pose)
                saved.SetDoubleProp("docking_score_kcal_mol", score)
                with Chem.SDWriter(str(path)) as writer:
                    writer.write(saved)
                records.append({"pose_index": index, "score_kcal_mol": score,
                                "pose_sdf": str(path), "pose_sha256": _hash(path)})
            best = min(records, key=lambda r: r["score_kcal_mol"])
            evaluation.update(status="docked", score_kcal_mol=best["score_kcal_mol"],
                              selected_pose_index=best["pose_index"], pose_sdf=best["pose_sdf"], poses=records)
            return best["score_kcal_mol"]
        finally:
            if self.engine == "quickvina2-gpu":
                evaluation["gpu_execution"] = copy.deepcopy(getattr(self._adapter, "last_run_metadata", {}))

    def __call__(self, state) -> float:
        started = time.monotonic()
        with self._lock:
            call = {"call_id": len(self._record["calls"]), "status": "running", "cache_hit": False,
                    "score_kcal_mol": None, "input_smiles": state if isinstance(state, str) else None,
                    "input_state_type": type(state).__name__}
            self._record["calls"].append(call)
            evaluation = None
            try:
                value, canonical = self._canonical(state)
                call.update(input_smiles=value, canonical_smiles=canonical)
                hashes = {"receptor": _hash(self.receptor_path), "reference_geometry": _hash(self.ligand_path)}
                if hashes != self._record["source_sha256"]:
                    raise PocketDockingRewardError("source_files_changed; create a new provider/protocol")
                if canonical in self._cache:
                    cached = self._record["evaluations"][self._cache[canonical]]
                    call.update(cache_hit=True, evaluation_id=cached["evaluation_id"])
                    if cached["status"] != "docked":
                        raise PocketDockingRewardError(f"cached_failure:{cached.get('error')}")
                    value = cached["score_kcal_mol"]
                else:
                    if self._attempts >= self.max_evaluations:
                        call["error_code"] = "budget_exhausted"
                        raise PocketDockingRewardError("docking_budget_exhausted")
                    evaluation = {"evaluation_id": len(self._record["evaluations"]),
                                  "canonical_smiles": canonical, "status": "running", "score_kcal_mol": None}
                    self._record["evaluations"].append(evaluation)
                    self._cache[canonical] = evaluation["evaluation_id"]
                    call["evaluation_id"] = evaluation["evaluation_id"]
                    self._checkpoint()
                    value = self._run_docking(canonical, evaluation)
                call.update(status="scored", score_kcal_mol=value)
                return value
            except Exception as exc:
                message = f"{type(exc).__name__}:{exc}"
                call.update(status="error", error=message)
                if evaluation is not None:
                    evaluation.update(status="failed", error=message, score_kcal_mol=None)
                raise PocketDockingRewardError(message) from exc
            finally:
                call["elapsed_wall_s"] = time.monotonic()-started
                self._checkpoint()
