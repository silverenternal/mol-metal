"""Lambda search on explicit CrossDocked test receptor/ligand pairs.

Uses the staged test manifest by default. Records exact input files, seeds,
requested settings, executed search settings and per-candidate descriptors.
Optional physical evaluation records actual docking and PoseBusters separately
from search proxies. No SOTA-equivalence is claimed. --dry-run validates inputs and the
resolved search budget without generating molecules.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import signal
import subprocess
import tempfile
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional

# Ensure PROJECT_ROOT is on sys.path so ``from molmetal_lam...`` works
# when this file is invoked via ``python molmetal/scripts/r4_c_full_sweep.py``
# (which otherwise puts the script's directory on sys.path[0] and
# shadows the top-level ``molmetal_lam`` shim).
_PROJ_ROOT_HINT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
if _PROJ_ROOT_HINT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT_HINT)
del _PROJ_ROOT_HINT

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
LAMBDA_SWEEP = os.path.join(HERE, "lambda_100pocket_sweep.py")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("r4_c_full_sweep")


# ---------------------------------------------------------------------------
# Result dataclass — one row per pocket
# ---------------------------------------------------------------------------
@dataclass
class PocketResult:
    pocket_id: str
    status: str = "ok"
    n_candidates: int = 0
    top1_smiles: str = ""
    top1_sa: Optional[float] = None
    top1_qed: Optional[float] = None
    top1_lipinski: Optional[bool] = None
    top1_vina_proxy: Optional[float] = None
    mean_sa: Optional[float] = None
    mean_qed: Optional[float] = None
    lipinski_pass_count: int = 0
    n_sims: int = 0
    wall_seconds: float = 0.0
    notes: str = ""
    seed: int = 42
    receptor_path: str = ""
    ligand_path: str = ""
    search_config: dict = field(default_factory=dict)
    candidates: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)
    metric_backend: dict = field(default_factory=dict)
    n_generated_candidates: int = 0
    n_seed_candidates: int = 0
    physical: dict = field(default_factory=dict)
    n_pb_pass: int = 0
    pb_pass_rate: Optional[float] = None
    pb_status: str = "disabled"
    pb_check: dict = field(default_factory=dict)
    diffdock_score_mean: Optional[float] = None
    diffdock_score_std: Optional[float] = None
    diffdock_status: str = "disabled"
    diffdock_n_invoked: int = 0
    diffdock_n_scored: int = 0
    diffdock_per_smiles: dict = field(default_factory=dict)
    flowdock_score_mean: Optional[float] = None
    flowdock_score_std: Optional[float] = None
    flowdock_status: str = "disabled"
    flowdock_n_invoked: int = 0
    flowdock_n_scored: int = 0
    flowdock_per_smiles: dict = field(default_factory=dict)
    # SOTA scoring column: BioLM-Score learned protein-ligand binding
    # affinity.  Recorded independently of any docking oracle; uses
    # upstream ``molmetal/references/BioLM-Score`` checkpoint + ESM-C
    # and Chemformer embeddings when available, otherwise records
    # ``status="unavailable"``.
    biomlm_score_mean: Optional[float] = None
    biomlm_score_std: Optional[float] = None
    biomlm_status: str = "disabled"
    biomlm_n_invoked: int = 0
    biomlm_n_scored: int = 0
    biomlm_encoder: str = "gatedgcn"
    biomlm_model_type: str = "biolm"
    biomlm_per_smiles: dict = field(default_factory=dict)
    # SOTA scoring column: AiZynthFinder retrosynthesis oracle.
    # Recorded independently of the search-time synthesis_oracle:
    # the search may use SMARTS for cheap pruning while the column
    # records an honest SMARTS / real-AiZynth measurement after the
    # search completes.
    aizynth_status: str = "disabled"
    aizynth_mode: str = "smarts"
    aizynth_n_invoked: int = 0
    aizynth_n_scored: int = 0
    aizynth_n_synthesis_route: int = 0
    aizynth_synthesis_success_rate: Optional[float] = None
    aizynth_per_smiles: dict = field(default_factory=dict)

    def to_csv_row(self) -> Dict[str, object]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Pocket discovery + per-pocket runner (Lambda side)
# ---------------------------------------------------------------------------
def load_manifest(path: str, offset: int, count: int) -> list[dict]:
    """Validate pair identity and files before any expensive computation."""
    base = Path(path).resolve().parent
    with open(path, newline="") as handle:
        rows = list(csv.DictReader(handle))
    seen = set()
    pairs_seen = set()
    for row in rows:
        if not row.get("pocket_id") or row["pocket_id"] in seen:
            raise ValueError("Manifest contains an empty or duplicate pocket_id")
        seen.add(row["pocket_id"])
        for key in ("receptor_path", "ligand_path"):
            if not row.get(key):
                raise ValueError(f"Missing {key} for {row['pocket_id']}")
            file = Path(row[key])
            file = file if file.is_absolute() else base / file
            if not file.is_file():
                raise FileNotFoundError(file)
            row[key] = str(file.resolve())
        identity = (row["receptor_path"], row["ligand_path"])
        if identity in pairs_seen:
            raise ValueError("Manifest contains a duplicate receptor/ligand pair")
        pairs_seen.add(identity)
    return rows[offset:offset + count]


def file_digest(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def runtime_fingerprints() -> dict:
    """Fingerprint Lambda runtime code, excluding tests and unrelated experiments."""
    root = Path(PROJECT_ROOT)
    directories = ("molmetal/molmetal_lam", "molmetal/domain", "molmetal/validation")
    files = {p for directory in directories for p in (root / directory).rglob("*.py")
             if "tests" not in p.relative_to(root).parts}
    files.update(Path(HERE) / name for name in ("r4_c_full_sweep.py", "lambda_100pocket_sweep.py",
                 "evaluate_generated_poses.py", "prepare_crossdocked_receptor.py",
                 "receptor_preparation_for_evaluation.py"))
    return {str(p.relative_to(root)): file_digest(str(p)) for p in sorted(files)}


def load_runner():
    from molmetal.scripts import lambda_100pocket_sweep
    return lambda_100pocket_sweep


def run_one_pocket(
    pocket_dir: str,
    n_simulations: int,
    max_depth: int,
    branching_target: int = 1020,
    *, seed: int = 42, pocket_id: str | None = None,
    receptor_path: str = "", ligand_path: str = "",
    top_k: int = 100, early_stop: bool = True, patience: int = 50,
    tile_library: str = "extended_204", click_rules: str = "all_5",
    symbolic_prior=None, synthesis_oracle=None, symbolic_prior_refit_every=None,
    seed_strategy: str = "reference",
    prior_state=None, prior_mode: str = "frozen", prior_data_split: str = "test",
    synthesis_config_path: str | None = None,
    docking_reward_config: dict | None = None,
) -> PocketResult:
    pocket_id = pocket_id or os.path.basename(pocket_dir)
    t0 = time.monotonic()
    base = dict(pocket_id=pocket_id, seed=seed, receptor_path=receptor_path,
                ligand_path=ligand_path, n_sims=n_simulations)
    try:
        if docking_reward_config is not None:
            docking_reward_config = dict(docking_reward_config)
            base_dir = docking_reward_config.pop("output_dir")
            job_id = hashlib.sha256(f"{pocket_id}:{seed}".encode()).hexdigest()[:16]
            docking_reward_config.update(receptor_path=receptor_path, ligand_path=ligand_path,
                                         output_dir=str(Path(base_dir) / job_id))
        r = load_runner().run_one_pocket(
            pocket_dir, n_simulations=n_simulations, max_depth=max_depth,
            seed_offset=0, seed=seed, include_fragments=True,
            branching_target=branching_target, top_k=top_k,
            early_stop=early_stop, patience=patience,
            tile_library=tile_library, click_rules=click_rules,
            ligand_path=ligand_path or None,
            symbolic_prior=symbolic_prior, synthesis_oracle=synthesis_oracle,
            symbolic_prior_refit_every=symbolic_prior_refit_every,
            seed_strategy=seed_strategy,
            prior_state=prior_state, prior_mode=prior_mode,
            prior_data_split=prior_data_split, synthesis_config_path=synthesis_config_path,
            docking_reward_config=docking_reward_config,
        )
        cands = r.get("candidates") or []
        status = r.get("status", "ok")
        reward_report = r.get("docking_reward_report", {})
        if status == "ok" and reward_report.get("requested") and reward_report.get("status") != "executed":
            status = "docking_reward_unavailable"
        if not cands and status == "ok":
            status = "no_candidates"
        if status == "ok" and r.get("generation_status") in ("seed_only", "invalid_candidates"):
            status = r["generation_status"]
        top1 = cands[0] if cands else {}
        def avg(key):
            vals = [c[key] for c in cands if isinstance(c.get(key), (int, float))
                    and math.isfinite(c[key])]
            return sum(vals) / len(vals) if vals else None
        return PocketResult(
            **base, status=status, n_candidates=len(cands),
            top1_smiles=top1.get("smiles", ""), top1_sa=top1.get("sa"),
            top1_qed=top1.get("qed"), top1_lipinski=top1.get("lipinski"),
            top1_vina_proxy=top1.get("vina_proxy"), mean_sa=avg("sa"),
            mean_qed=avg("qed"), lipinski_pass_count=sum(bool(c.get("lipinski")) for c in cands),
            wall_seconds=time.monotonic() - t0,
            search_config=r.get("search_config", {}), candidates=cands,
            metric_backend=r.get("metric_backend", {}),
            n_generated_candidates=r.get("n_generated_candidates", 0),
            n_seed_candidates=r.get("n_seed_candidates", 0),
            diagnostics={"seed_smiles": r.get("seed_smiles"),
                         "init_strategy": r.get("init_strategy", seed_strategy),
                         "canonical_seed": r.get("canonical_seed"), "chosen_tile": r.get("chosen_tile"),
                         "initialization_seconds": r.get("initialization_seconds"),
                         "empty_candidates_reason": r.get("empty_candidates_reason"),
                         "search": r.get("search_diagnostics", {}), "error": r.get("error"),
                         "prior_state": r.get("prior_state"),
                         "prior": r.get("prior_report", {}),
                         "synthesis": r.get("synthesis_report", {}),
                         "docking_reward": r.get("docking_reward_report", {}),
                         "all_candidates": r.get("all_candidates", cands),
                         "candidate_accounting": r.get("candidate_accounting", {}),
                         "selection_counts": {key: r[key] for key in (
                             "n_pre_synthesis_candidates", "n_retained_candidates", "n_selected_candidates",
                             "n_synthesis_rejected_candidates", "n_generated_retained_candidates",
                             "n_generated_selected_candidates", "n_generated_synthesis_rejected_candidates") if key in r},
                         "generation": r.get("generation_diagnostics", {})},
            notes=f"{seed_strategy} initialized search; physical measurements recorded separately when enabled",
        )
    except Exception as e:
        return PocketResult(**base, status=f"error: {type(e).__name__}: {e}",
                            wall_seconds=time.monotonic() - t0)


# ---------------------------------------------------------------------------
# Aggregator + IO
# ---------------------------------------------------------------------------
def aggregate(results: List[PocketResult]) -> Dict[str, object]:
    ok = [r for r in results if r.status == "ok"]
    n_ok = len(ok)
    n_total = len(results)
    sa_vals = [r.mean_sa for r in ok if isinstance(r.mean_sa, float)]
    qed_vals = [r.mean_qed for r in ok if isinstance(r.mean_qed, float)]
    vina_vals = [r.top1_vina_proxy for r in ok if isinstance(r.top1_vina_proxy, float)]
    lip_pass = sum(r.lipinski_pass_count for r in ok)
    n_total_cands = sum(r.n_candidates for r in ok)
    mean_wall = sum(r.wall_seconds for r in results) / n_total if n_total else 0
    return {
        "n_jobs_total": n_total,
        "n_unique_pockets": len({r.pocket_id for r in results}),
        "n_generated_candidates": sum(r.n_generated_candidates for r in results),
        "n_seed_candidates": sum(r.n_seed_candidates for r in results),
        "n_pockets_total": n_total,
        "n_pockets_ok": n_ok,
        "n_pockets_fail": n_total - n_ok,
        "n_candidates_total": n_total_cands,
        "mean_candidates_per_pocket": n_total_cands / n_ok if n_ok else 0,
        "lipinski_pass_count": lip_pass,
        "lipinski_pass_rate": lip_pass / n_total_cands if n_total_cands else None,
        "sa_mean": sum(sa_vals) / len(sa_vals) if sa_vals else None,
        "qed_mean": sum(qed_vals) / len(qed_vals) if qed_vals else None,
        "vina_proxy_top1_mean": sum(vina_vals) / len(vina_vals) if vina_vals else None,
        "vina_proxy_top1_min": min(vina_vals) if vina_vals else None,
        "vina_proxy_top1_max": max(vina_vals) if vina_vals else None,
        "wall_seconds_mean_per_pocket": mean_wall,
        "physical_jobs_completed": sum(r.physical.get("status") == "completed" for r in results),
        "physical_n_docked": sum(r.physical.get("summary", {}).get("n_docked", 0) for r in results),
        "physical_n_pb_pass": sum(r.physical.get("summary", {}).get("n_pb_pass", 0) for r in results),
        "n_pb_pass": sum(r.n_pb_pass for r in results),
        "pb_pass_rate": _safe_mean(r.pb_pass_rate for r in results),
        "pb_n_jobs_with_check": sum(1 for r in results if r.pb_status not in ("disabled", "")),
        "diffdock_score_mean": _safe_mean(r.diffdock_score_mean for r in results),
        "diffdock_score_std": _safe_std([r.diffdock_score_mean for r in results
                                          if isinstance(r.diffdock_score_mean, (int, float))
                                          and math.isfinite(r.diffdock_score_mean)]),
        "diffdock_n_pockets_scored": sum(
            1 for r in results
            if isinstance(r.diffdock_score_mean, (int, float)) and math.isfinite(r.diffdock_score_mean)),
        "diffdock_n_invoked_total": sum(r.diffdock_n_invoked for r in results),
        "diffdock_n_scored_total": sum(r.diffdock_n_scored for r in results),
        "diffdock_status_counts": _status_counts(r.diffdock_status for r in results),
        "flowdock_score_mean": _safe_mean(r.flowdock_score_mean for r in results),
        "flowdock_score_std": _safe_std([r.flowdock_score_mean for r in results
                                         if isinstance(r.flowdock_score_mean, (int, float))
                                         and math.isfinite(r.flowdock_score_mean)]),
        "flowdock_n_pockets_scored": sum(
            1 for r in results
            if isinstance(r.flowdock_score_mean, (int, float)) and math.isfinite(r.flowdock_score_mean)),
        "flowdock_n_invoked_total": sum(r.flowdock_n_invoked for r in results),
        "flowdock_n_scored_total": sum(r.flowdock_n_scored for r in results),
        "flowdock_status_counts": _status_counts(r.flowdock_status for r in results),
        "biomlm_score_mean": _safe_mean(r.biomlm_score_mean for r in results),
        "biomlm_score_std": _safe_std([r.biomlm_score_mean for r in results
                                        if isinstance(r.biomlm_score_mean, (int, float))
                                        and math.isfinite(r.biomlm_score_mean)]),
        "biomlm_n_pockets_scored": sum(
            1 for r in results
            if isinstance(r.biomlm_score_mean, (int, float)) and math.isfinite(r.biomlm_score_mean)),
        "biomlm_n_invoked_total": sum(r.biomlm_n_invoked for r in results),
        "biomlm_n_scored_total": sum(r.biomlm_n_scored for r in results),
        "biomlm_status_counts": _status_counts(r.biomlm_status for r in results),
        "aizynth_n_invoked_total": sum(r.aizynth_n_invoked for r in results),
        "aizynth_n_scored_total": sum(r.aizynth_n_scored for r in results),
        "aizynth_n_synthesis_route_total": sum(r.aizynth_n_synthesis_route for r in results),
        "aizynth_synthesis_success_rate_mean": _safe_mean(
            r.aizynth_synthesis_success_rate for r in results),
        "aizynth_status_counts": _status_counts(r.aizynth_status for r in results),
        "aizynth_mode_counts": {mode: sum(1 for r in results if r.aizynth_mode == mode)
                                 for mode in ("smarts", "aizynthfinder", "aizynthfinder_isolated")},
        "aizynth_n_pockets_scored": sum(
            1 for r in results
            if isinstance(r.aizynth_synthesis_success_rate, (int, float))
            and math.isfinite(r.aizynth_synthesis_success_rate)),
    }


def _safe_mean(values):
    finite = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return sum(finite) / len(finite) if finite else None


def _safe_std(values):
    finite = [v for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    if len(finite) < 2:
        return None
    mean = sum(finite) / len(finite)
    return math.sqrt(sum((v - mean) ** 2 for v in finite) / len(finite))


def _status_counts(statuses):
    out: dict = {}
    for s in statuses:
        out[s] = out.get(s, 0) + 1
    return out


def write_csv(results: List[PocketResult], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fields = list(PocketResult.__dataclass_fields__.keys())  # type: ignore
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in results:
            row = r.to_csv_row()
            for key in ("search_config", "candidates", "diagnostics", "metric_backend",
                        "physical", "pb_check"):
                row[key] = json.dumps(clean_json(row[key]), allow_nan=False)
            w.writerow(row)


def clean_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    return value


def write_json(results, summary, path, metadata=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temp = Path(path).with_suffix(".json.tmp")
    payload = {"metadata": metadata or {}, "summary": summary,
               "per_pocket": [asdict(r) for r in results]}
    # WF-T25 Cite-Only SOTA Comparator (TODO-22 §4c + TODO-25 (d)).
    # When the comparator is enabled, attach per-pocket comparison
    # rows to the JSON.  We compute gap_vina from each pocket's
    # measured vina_proxy (search-time proxy, NOT physical-docked Vina)
    # so the rows are honest: comparison vs literature only, no
    # head-to-head claim.  The metadata block records the full
    # comparator + user_protocol_flags for reproducibility.
    cmp_meta = (metadata or {}).get("cite_only_sota_comparator") if metadata else None
    if cmp_meta and results:
        try:
            from molmetal_lam.sbdd_env.cite_only_sota_comparator import (
                CiteOnlySOTAComparator,
            )
            cmp = CiteOnlySOTAComparator()
            user_flags = cmp_meta.get("user_protocol_flags") or {}
            per_pocket_rows = []
            for r in results:
                # Use top1_vina_proxy as the per-pocket value.  Note:
                # top1_vina_proxy is the search-time docking proxy,
                # NOT the physical-docked Vina; we surface this honestly
                # in the comparator notes field.
                our_value = (r.get("top1_vina_proxy") if isinstance(r, dict)
                             else r.top1_vina_proxy)
                if our_value is None:
                    our_value = float("nan")
                rows = cmp.compare_per_pocket(
                    pocket_id=r.pocket_id if isinstance(r, dict) else r.pocket_id,
                    our_value=our_value,
                    num_samples=(r.get("n_candidates") if isinstance(r, dict)
                                 else r.n_candidates) or 0,
                    protocol_flags=user_flags,
                    our_sa=(r.get("mean_sa") if isinstance(r, dict)
                            else r.mean_sa),
                    our_qed=(r.get("mean_qed") if isinstance(r, dict)
                             else r.mean_qed),
                )
                per_pocket_rows.append({
                    "pocket_id": r.pocket_id if isinstance(r, dict) else r.pocket_id,
                    "seed": r.seed if isinstance(r, dict) else r.seed,
                    "comparison_rows": rows,
                    "value_source": "top1_vina_proxy (search-time proxy; not physical-docked)",
                })
            payload["cite_only_sota_per_pocket"] = per_pocket_rows
        except Exception as exc:
            log.warning("Cite-only SOTA comparator integration failed: %s", exc)
    temp.write_text(json.dumps(clean_json(payload), indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def write_markdown(results, summary, path, n_pockets_requested, n_simulations, max_depth):
    def fmt(value):
        return f"{value:.3f}" if isinstance(value, (int, float)) and math.isfinite(value) else "—"
    lines = [
        "# Lambda paired-input search diagnostic", "",
        f"Requested pockets: {n_pockets_requested}; completed pocket/seed jobs: {len(results)}.",
        f"Jobs returning generated candidates: {summary['n_pockets_ok']}; failed, empty or seed-only: {summary['n_pockets_fail']}.",
        f"Search budget: {n_simulations} simulations, depth {max_depth}.", "",
        "Initialization strategies are recorded per job; reference initialization uses the paired ligand chemistry.",
        "Click-tile initialization uses the declared tile library. Reference coordinates define physical docking boxes.",
        "Search docking proxies have no kcal/mol interpretation. Actual docking/PoseBusters, when enabled, are in the physical records.",
        "Reference thresholds are redocked-ligand scores, not measured co-crystal affinity. This is not a SOTA comparison.",
        "Lipinski pass rate is a descriptor statistic, not the published SBDD success rate.",
        "Requested configuration and executed search settings are recorded separately in JSON.", "",
        "| Metric | Value |", "|---|---:|",
        f"| Candidates | {summary['n_candidates_total']} |",
        f"| Newly generated candidates | {summary['n_generated_candidates']} |",
        f"| Returned input-seed candidates | {summary['n_seed_candidates']} |",
        f"| Physical jobs completed | {summary['physical_jobs_completed']} |",
        f"| Generated products physically docked | {summary['physical_n_docked']} |",
        f"| Docked products passing PoseBusters | {summary['physical_n_pb_pass']} |",
        f"| Candidates passing PoseBusters (chemistry, post-dock) | {summary['n_pb_pass']} |",
        f"| PoseBusters (chemistry) pass rate | {fmt(summary['pb_pass_rate'])} |",
        f"| Lipinski pass rate | {fmt(summary['lipinski_pass_rate'])} |",
        f"| AiZynth synthesis success rate (mean, mode-aware) | {fmt(summary['aizynth_synthesis_success_rate_mean'])} |",
        f"| AiZynth n_pockets_scored | {summary['aizynth_n_pockets_scored']} |",
        f"| AiZynth n_synthesis_route_total | {summary['aizynth_n_synthesis_route_total']} |",
        f"| BioLM-Score affinity score (mean across pockets) | {fmt(summary['biomlm_score_mean'])} |",
        f"| BioLM-Score n_pockets_scored | {summary['biomlm_n_pockets_scored']} |",
        f"| BioLM-Score status counts | {summary['biomlm_status_counts']} |",
        f"| SA mean (backend recorded in search_config) | {fmt(summary['sa_mean'])} |",
        f"| QED mean | {fmt(summary['qed_mean'])} |",
        f"| Descriptor docking proxy, top-1 mean | {fmt(summary['vina_proxy_top1_mean'])} |", "",
        "| pocket | seed | status | candidates | generated | SMILES | seconds |",
        "|---|---:|---|---:|---:|---|---:|",
    ]
    for r in results:
        status = r.status.replace("|", "\\|").replace("\n", " ")
        lines.append(f"| {r.pocket_id} | {r.seed} | {status} | {r.n_candidates} | {r.n_generated_candidates} | `{r.top1_smiles}` | {r.wall_seconds:.2f} |")
    lines += ["", "Published baseline context is maintained in `lambda_vs_sbdd_protocol_aligned.md`.",
              "These proxy values are not compared numerically or statistically to cited docking energies."]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines) + "\n")



def execute_job(job, search, timeout, log_path):
    """Bound each job in its own process and terminate its subprocess group."""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    kwargs = dict(search, pocket_dir=str(Path(job["ligand_path"]).parent),
                  pocket_id=job["pocket_id"], receptor_path=job["receptor_path"],
                  ligand_path=job["ligand_path"], seed=job["seed"])
    start = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="r4_job_") as tmp:
        input_path, output_path = Path(tmp) / "input.json", Path(tmp) / "output.json"
        input_path.write_text(json.dumps(kwargs))
        with open(log_path, "w") as log_file:
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                                     "--worker-input", str(input_path), "--worker-output", str(output_path)],
                                    stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = proc.wait(timeout=timeout)
                if code == 0 and output_path.is_file():
                    return PocketResult(**json.loads(output_path.read_text()))
                status = f"worker_failed: exit={code}; see {log_path}"
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                status = f"timeout: {timeout:g}s; see {log_path}"
            if output_path.is_file():
                partial = PocketResult(**json.loads(output_path.read_text()))
                partial.physical.update(status="interrupted", error=status)
                partial.notes += "; physical evaluation interrupted; completed search/pose records retained"
                partial.wall_seconds = time.monotonic() - start
                return partial
    return PocketResult(pocket_id=job["pocket_id"], seed=job["seed"],
                        receptor_path=job["receptor_path"], ligand_path=job["ligand_path"],
                        status=status, n_sims=search["n_simulations"],
                        wall_seconds=time.monotonic() - start)


def main() -> int:
    if "--worker-input" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--worker-input", required=True)
        parser.add_argument("--worker-output", required=True)
        args = parser.parse_args()
        from rdkit import rdBase
        kwargs = json.loads(Path(args.worker_input).read_text())
        physical_config = kwargs.pop("physical_config", None)
        diffdock_config = kwargs.pop("diffdock_config", None)
        flowdock_config = kwargs.pop("flowdock_config", None)
        aizynth_config = kwargs.pop("aizynth_config", None)
        pb_config = kwargs.pop("pb_config", None)
        biomlm_config = kwargs.pop("biomlm_config", None)
        def checkpoint(result):
            output = Path(args.worker_output)
            temp = output.with_suffix(".tmp")
            temp.write_text(json.dumps(clean_json(asdict(result)), allow_nan=False))
            temp.replace(output)
        with rdBase.BlockLogs():
            result = run_one_pocket(**kwargs)
        checkpoint(result)
        if physical_config:
            from molmetal.scripts.evaluate_generated_poses import evaluate_candidates
            def progress(report):
                result.physical = report
                checkpoint(result)
            started = time.monotonic()
            try:
                physical_config = dict(physical_config)
                base_dir = physical_config.pop("output_dir")
                job_id = hashlib.sha256(f"{result.pocket_id}:{result.seed}".encode()).hexdigest()[:16]
                result.physical = evaluate_candidates(result.candidates, result.receptor_path,
                    result.ligand_path, str(Path(base_dir) / job_id), seed=result.seed,
                    total_generated=result.n_generated_candidates,
                    progress=progress, **physical_config)
            except Exception as exc:
                result.physical.update(status="error", error=f"{type(exc).__name__}: {exc}")
            result.wall_seconds += time.monotonic() - started
            checkpoint(result)
        if pb_config:
            pb_started = time.monotonic()
            try:
                from molmetal_lam.sbdd_env.posebusters_adapter import PoseBustersAdapter
                pb_mode = pb_config.get("mode", "mol")
                # WF-PB-Dock-Mode-Wire: pass the receptor PDB through to
                # ``validate_docked`` so the protein-aware clash checks
                # (minimum_distance_to_protein, volume_overlap_with_protein,
                # protein-ligand_maximum_distance, cofactor checks) actually
                # run instead of being skipped because we have no reference.
                receptor_pdb_for_pb = pb_config.get("receptor_pdb") or result.receptor_path
                adapter = PoseBustersAdapter(mode=pb_mode)
                report = {"status": "running", "scope": "PoseBusters chemistry validity on docked candidates",
                          "mode": adapter.name, "engine_version": adapter.get_metadata().get("engine_version"),
                          "per_smiles": {}}
                smiles_list = [c.get("smiles", "") for c in result.candidates
                               if c.get("is_generated") is True and c.get("smiles")]
                if not smiles_list:
                    report.update(status="no_candidates", n=0, n_pb_pass=0, pb_pass_rate=None)
                    result.n_pb_pass = 0
                    result.pb_pass_rate = None
                    result.pb_status = "no_candidates"
                elif pb_mode == "dock":
                    if not receptor_pdb_for_pb or not Path(receptor_pdb_for_pb).is_file():
                        # Honest-framing: do NOT silently fall back to mol-mode.
                        # Record a clear reason so the downstream report can
                        # distinguish "no receptor" from "no candidates".
                        report.update(status="receptor_missing",
                                      reason=f"receptor_pdb not found: {receptor_pdb_for_pb!r}",
                                      n=len(smiles_list), n_pb_pass=0, pb_pass_rate=None)
                        result.n_pb_pass = 0
                        result.pb_pass_rate = None
                        result.pb_status = "receptor_missing"
                    else:
                        per_smiles = {}
                        n_pass = 0
                        n_total = 0
                        extras_seen: set = set()
                        for smi in smiles_list:
                            r = adapter.validate_docked(smi, receptor_pdb_for_pb)
                            n_total += 1
                            if r.passed:
                                n_pass += 1
                            per_smiles[smi] = {
                                "passed": r.passed, "n_checks": r.n_checks,
                                "n_passed": r.n_passed, "pass_rate": r.pass_rate,
                                "failed_checks": list(r.failed_checks),
                                "mode": r.mode, "receptor_pdb": r.receptor_pdb,
                                "extra_checks": list(r.extra_checks),
                            }
                            extras_seen.update(r.extra_checks)
                        report.update(status="completed", n=n_total, n_pb_pass=n_pass,
                                      pb_pass_rate=(n_pass / n_total) if n_total else None,
                                      receptor_pdb=receptor_pdb_for_pb,
                                      extra_checks=sorted(extras_seen),
                                      per_smiles=per_smiles)
                        result.n_pb_pass = n_pass
                        result.pb_pass_rate = (n_pass / n_total) if n_total else None
                        result.pb_status = "completed"
                else:
                    reports = adapter.validate_list(smiles_list)
                    n_pass = sum(1 for r in reports if r.passed)
                    n_total = len(reports)
                    for r in reports:
                        report["per_smiles"][r.smiles] = {
                            "passed": r.passed, "n_checks": r.n_checks,
                            "n_passed": r.n_passed, "pass_rate": r.pass_rate,
                            "failed_checks": list(r.failed_checks)}
                    report.update(status="completed", n=n_total, n_pb_pass=n_pass,
                                  pb_pass_rate=(n_pass / n_total) if n_total else None)
                    result.n_pb_pass = n_pass
                    result.pb_pass_rate = (n_pass / n_total) if n_total else None
                    result.pb_status = "completed"
                result.pb_check = report
            except ImportError as exc:
                # posebusters not installed; degrade gracefully
                result.pb_status = "unavailable"
                result.pb_check = {"status": "unavailable",
                                   "reason": str(exc),
                                   "n": 0, "n_pb_pass": 0, "pb_pass_rate": None,
                                   "per_smiles": {}}
            except Exception as exc:
                result.pb_status = "error"
                result.pb_check = {"status": "error",
                                   "error": f"{type(exc).__name__}: {exc}",
                                   "n": 0, "n_pb_pass": 0, "pb_pass_rate": None,
                                   "per_smiles": {}}
            result.wall_seconds += time.monotonic() - pb_started
            checkpoint(result)
        if diffdock_config:
            dd_started = time.monotonic()
            try:
                from molmetal_lam.sbdd_env.diffdock_sota_scoring import score_candidates
                col = score_candidates(
                    result.candidates, result.receptor_path,
                    repo_root=diffdock_config.get("repo_root",
                        str(Path(PROJECT_ROOT) / "molmetal/references/DiffDock")),
                    samples_per_complex=int(diffdock_config.get("samples_per_complex", 4)),
                    timeout_sec=float(diffdock_config.get("timeout_sec", 600.0)),
                    config_yaml=diffdock_config.get("config_yaml"),
                )
                result.diffdock_score_mean = col.diffdock_score_mean
                result.diffdock_score_std = col.diffdock_score_std
                result.diffdock_status = col.status
                result.diffdock_n_invoked = col.n_invoked
                result.diffdock_n_scored = col.n_scored
                result.diffdock_per_smiles = col.per_smiles
            except Exception as exc:
                result.diffdock_status = "error"
                result.diffdock_per_smiles = {"error": f"{type(exc).__name__}: {exc}"}
            result.wall_seconds += time.monotonic() - dd_started
            checkpoint(result)
        if flowdock_config:
            fd_started = time.monotonic()
            try:
                from molmetal_lam.sbdd_env.flowdock_sota_scoring import score_candidates as fd_score
                col = fd_score(
                    result.candidates, result.receptor_path,
                    repo_root=flowdock_config.get("repo_root",
                        str(Path(PROJECT_ROOT) / "molmetal/references/FlowDock")),
                    n_samples=int(flowdock_config.get("n_samples", 4)),
                    num_steps=int(flowdock_config.get("num_steps", 40)),
                    ckpt_path=flowdock_config.get("ckpt_path"),
                    device=flowdock_config.get("device", "cpu"),
                    timeout_sec=float(flowdock_config.get("timeout_sec", 600.0)),
                )
                result.flowdock_score_mean = col.flowdock_score_mean
                result.flowdock_score_std = col.flowdock_score_std
                result.flowdock_status = col.status
                result.flowdock_n_invoked = col.n_invoked
                result.flowdock_n_scored = col.n_scored
                result.flowdock_per_smiles = col.per_smiles
            except Exception as exc:
                result.flowdock_status = "error"
                result.flowdock_per_smiles = {"error": f"{type(exc).__name__}: {exc}"}
            result.wall_seconds += time.monotonic() - fd_started
            checkpoint(result)
        if biomlm_config:
            biomlm_started = time.monotonic()
            try:
                from molmetal_lam.sbdd_env.biomlm_sota_scoring import score_candidates as biomlm_score
                col = biomlm_score(
                    result.candidates, result.receptor_path,
                    repo_root=biomlm_config.get("repo_root",
                        str(Path(PROJECT_ROOT) / "molmetal/references/BioLM-Score")),
                    encoder=biomlm_config.get("encoder", "gatedgcn"),
                    model_type=biomlm_config.get("model_type", "biolm"),
                    ckpt_path=biomlm_config.get("ckpt_path"),
                    timeout_sec=float(biomlm_config.get("timeout_sec", 600.0)),
                )
                result.biomlm_score_mean = col.biomlm_score_mean
                result.biomlm_score_std = col.biomlm_score_std
                result.biomlm_status = col.status
                result.biomlm_n_invoked = col.n_invoked
                result.biomlm_n_scored = col.n_scored
                result.biomlm_encoder = col.encoder
                result.biomlm_model_type = col.model_type
                result.biomlm_per_smiles = col.per_smiles
            except Exception as exc:
                result.biomlm_status = "error"
                result.biomlm_per_smiles = {"error": f"{type(exc).__name__}: {exc}"}
            result.wall_seconds += time.monotonic() - biomlm_started
            checkpoint(result)
        if aizynth_config:
            az_started = time.monotonic()
            try:
                from molmetal_lam.sbdd_env.aizynth_sota_scoring import score_candidates as az_score
                col = az_score(
                    result.candidates,
                    mode=aizynth_config.get("mode", "smarts"),
                    config_path=aizynth_config.get("config_path"),
                    isolated_run_sh=aizynth_config.get(
                        "isolated_run_sh",
                        str(Path(PROJECT_ROOT) / "environments/aizynth/run.sh")),
                    timeout_sec=float(aizynth_config.get("timeout_sec", 600.0)),
                    max_iterations=int(aizynth_config.get("max_iterations", 200)),
                    time_limit_s=int(aizynth_config.get("time_limit_s", 30)),
                    seed=result.seed,
                )
                result.aizynth_status = col.status
                result.aizynth_mode = col.mode
                result.aizynth_n_invoked = col.n_invoked
                result.aizynth_n_scored = col.n_scored
                result.aizynth_n_synthesis_route = col.n_synthesis_route
                result.aizynth_synthesis_success_rate = col.synthesis_success_rate
                result.aizynth_per_smiles = col.per_smiles
            except Exception as exc:
                result.aizynth_status = "error"
                result.aizynth_per_smiles = {"error": f"{type(exc).__name__}: {exc}"}
            result.wall_seconds += time.monotonic() - az_started
            checkpoint(result)
        return 0

    from molmetal_lam.configs.sota_aligned import load_sota_aligned_config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pockets", help="Extraction root hint (selection always uses explicit manifest pairs)")
    parser.add_argument("--manifest", default=str(Path(PROJECT_ROOT) / "molmetal/data/crossdocked100_manifest.csv"))
    parser.add_argument("--config", default=str(Path(PROJECT_ROOT) / "molmetal/configs/sota_aligned_targetdiff.yaml"))
    parser.add_argument("--output-prefix", default="molmetal/reports/r4_c_full_sweep")
    parser.add_argument("--n-pockets", type=int, default=5)
    parser.add_argument("--pocket-offset", type=int, default=0)
    parser.add_argument("--n-simulations", type=int)
    parser.add_argument("--branching-target", type=int)
    parser.add_argument("--seed-strategy", choices=("reference", "click_tile"), default="reference")
    parser.add_argument("--symbolic-prior", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable the labelled linear descriptor prior (not PySR); overrides YAML")
    parser.add_argument("--prior-state", help="Development-trained JSON prior; enables frozen test inference")
    parser.add_argument("--synthesis-oracle", choices=("none", "smarts", "aizynthfinder", "aizynthfinder_isolated"),
                        help="Override YAML; SMARTS is a heuristic, AiZynth requires real assets")
    parser.add_argument("--synthesis-config", help="AiZynth configuration file; recorded with SHA256")
    parser.add_argument("--search-docking-reward", action="store_true",
                        help="Feed real pocket docking energies into MCTS during search")
    parser.add_argument("--search-docking-engine", choices=("auto", "vina", "quickvina2", "quickvina2-gpu"), default="auto")
    parser.add_argument("--search-docking-max-evals", type=int, default=100,
                        help="Maximum distinct candidate docking attempts per pocket/seed; cache repeats")
    parser.add_argument("--search-docking-weight", type=float, default=0.4,
                        help="Nonnegative energy weight; zero measures/caches the same oracle without rewarding it")
    parser.add_argument("--physical-docking", action="store_true")
    parser.add_argument("--physical-engine", choices=("auto", "vina", "quickvina2", "quickvina2-gpu"), default="auto",
                        help="auto prefers configured AMD GPU docking; actual backend/budget is recorded")
    parser.add_argument("--engine", choices=("vina", "qvina", "quickvina2", "both", "all"), default="both",
                        help=("D7 default: both engines emit per-pocket vina_score AND qvina_score columns "
                              "for headline-table parity. Single-engine values emit only that engine's column. "
                              "'both' = Vina + QVina; 'all' = Vina + QVina + QuickVina2. "
                              "Used by --physical-docking; ignored otherwise."))
    parser.add_argument("--gpu-config", default=str(Path(PROJECT_ROOT) / "molmetal/configs/amd_gpu_docking.json"))
    parser.add_argument("--physical-exhaustiveness", type=int, default=8,
                        help="Exhaustiveness for Vina / QuickVina2 docking (default 8 = "
                             "TargetDiff / CrossDocked2020 production standard; set 1 to "
                             "recover the legacy smoke bit-exact behaviour).  "
                             "WF-Metallodrug-Vertical Phase 3 protocol-align bumped from 1 to 8 "
                             "to match the TargetDiff per-cell protocol (Guan ICLR 2023 §4.4 "
                             "Table 4).")
    # WF-Metallodrug-Vertical Phase 3 protocol-align — pocket-10 Å
    # radius flag.  Default 10.0 Å matches the CrossDocked2020
    # standard pocket extraction radius (Francoeur et al. 2020, also
    # adopted by DiffDock / TargetDiff / Pocket2Mol).  Backward
    # compatible: setting --pocket10-radius 8.0 recovers the legacy
    # DiffDock-Pocket 8 Å crop radius.
    parser.add_argument("--pocket10-radius", type=float, default=10.0,
                        help="Pocket extraction radius in Angstroms (default 10.0 = "
                             "CrossDocked2020 / TargetDiff standard).  Set 8.0 to recover "
                             "the legacy DiffDock-Pocket 8 Å crop.")
    # WF-Metallodrug-Vertical Phase 3 — pocket-conditioned reference
    # ligand warm-start.  When set the generator uses the pocket's
    # reference ligand SDF (under
    # /mnt/storage/data/molmetal/crossdocked/extracted/crossdocked_pocket10/<pocket_id>/<pocket_id>_ligand.sdf)
    # as the warm-start for pocket-conditioned generation instead of
    # the unconditional base.  Default off (legacy unconditional).
    parser.add_argument("--reference-ligand", action="store_true",
                        help="Use the pocket-specific reference ligand SDF "
                             "(/mnt/storage/data/molmetal/crossdocked/extracted/"
                             "crossdocked_pocket10/<pocket_id>/<pocket_id>_ligand.sdf) "
                             "as a warm-start for pocket-conditioned generation.  "
                             "Default off (legacy unconditional sampling).")
    parser.add_argument("--physical-n-poses", type=int, default=9)
    parser.add_argument("--physical-top-k", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 1234])
    parser.add_argument("--sota-diffdock", action="store_true",
                        help="Subprocess-call DiffDock-L for every Lambda candidate and record confidence")
    parser.add_argument("--sota-diffdock-samples", type=int, default=4,
                        help="DiffDock samples_per_complex budget (default 4 = cheap)")
    parser.add_argument("--sota-diffdock-timeout", type=float, default=600.0,
                        help="Per-SMILES DiffDock subprocess timeout (seconds)")
    parser.add_argument("--sota-diffdock-repo",
                        default=str(Path(PROJECT_ROOT) / "molmetal/references/DiffDock"),
                        help="Path to the cloned DiffDock checkout")
    parser.add_argument("--sota-diffdock-config", default=None,
                        help="Optional path to default_inference_args.yaml override")
    parser.add_argument("--sota-flowdock", action="store_true",
                        help="Subprocess-call FlowDock for every Lambda candidate and record confidence")
    parser.add_argument("--sota-flowdock-samples", type=int, default=4,
                        help="FlowDock n_samples budget (default 4 = cheap)")
    parser.add_argument("--sota-flowdock-steps", type=int, default=40,
                        help="FlowDock ODE num_steps budget (default 40 = upstream default)")
    parser.add_argument("--sota-flowdock-timeout", type=float, default=600.0,
                        help="Per-SMILES FlowDock subprocess timeout (seconds)")
    parser.add_argument("--sota-flowdock-repo",
                        default=str(Path(PROJECT_ROOT) / "molmetal/references/FlowDock"),
                        help="Path to the cloned FlowDock checkout")
    parser.add_argument("--sota-flowdock-ckpt", default=None,
                        help="Optional path to a FlowDock checkpoint (.ckpt)")
    parser.add_argument("--sota-flowdock-device", choices=("cpu", "cuda"), default="cpu",
                        help="FlowDock inference device (default cpu; ROCm has no CUDA backend)")
    parser.add_argument("--sota-biomlm", action="store_true",
                        help="Subprocess-call BioLM-Score (CASF-2016 scoring) for every Lambda "
                             "candidate and record affinity scores.  Requires the vendored "
                             "BioLM-Score repo + a Zenodo mm*.pth checkpoint; gracefully "
                             "degrades to status='unavailable' when missing.")
    parser.add_argument("--sota-biomlm-encoder", choices=("gatedgcn", "gt"), default="gatedgcn",
                        help="BioLM-Score graph encoder (default 'gatedgcn' = upstream default)")
    parser.add_argument("--sota-biomlm-model-type", choices=("biolm", "genscore"), default="biolm",
                        help="BioLM-Score model variant (default 'biolm' = uses protein+ligand LMs)")
    parser.add_argument("--sota-biomlm-ckpt", default=None,
                        help="Path to BioLM-Score mm*.pth checkpoint (Zenodo DOI 10.5281/zenodo.21878818)")
    parser.add_argument("--sota-biomlm-timeout", type=float, default=600.0,
                        help="BioLM-Score subprocess timeout (seconds)")
    parser.add_argument("--sota-biomlm-repo",
                        default=str(Path(PROJECT_ROOT) / "molmetal/references/BioLM-Score"),
                        help="Path to the cloned BioLM-Score checkout")
    parser.add_argument("--sota-aizynth", action="store_true",
                        help="Record per-pocket AiZynthFinder retrosynthesis statistics "
                             "(n_synthesis_route / synthesis_success_rate) as a SOTA column. "
                             "Defaults to SMARTS fallback; pass --sota-aizynth-mode to upgrade.")
    parser.add_argument("--sota-aizynth-mode", choices=("smarts", "aizynthfinder", "aizynthfinder_isolated"),
                        default="smarts",
                        help="AiZynth execution mode (default 'smarts' = RDKit reverse-templates; "
                             "the two real-AiZynth modes require assets and degrade gracefully)")
    parser.add_argument("--sota-aizynth-config", default=None,
                        help="AiZynthFinder config.yml (required for the two real-AiZynth modes)")
    parser.add_argument("--sota-aizynth-isolated-run-sh",
                        default=str(Path(PROJECT_ROOT) / "environments/aizynth/run.sh"),
                        help="Entry-point for the isolated AiZynth env (default environments/aizynth/run.sh)")
    parser.add_argument("--sota-aizynth-timeout", type=float, default=600.0,
                        help="Per-batch AiZynth subprocess timeout (seconds)")
    parser.add_argument("--sota-aizynth-max-iterations", type=int, default=200)
    parser.add_argument("--sota-aizynth-time-limit", type=int, default=30)
    # WF-T25 Cite-Only SOTA Comparator (TODO-22 §4c + TODO-25 (d)).
    # When set, the run metadata records the cite-only SOTA comparison
    # data for every pocket, including the 9 SOTA rows (cited-only) +
    # the 7 protocol-mismatch flags (M1)-(M7) and per-row gap
    # information.  This does NOT add any SOTA scoring columns or
    # re-run any SOTA baseline — the comparator is comparison vs
    # literature, NOT head-to-head.
    parser.add_argument("--cite-only-sota-comparator", action="store_true",
                        help="Record cite-only SOTA comparison rows in the "
                             "run JSON metadata (9 SOTA rows + 7 protocol-mismatch "
                             "flags + per-pocket gap).  Citation-only, no live re-run.")
    parser.add_argument("--cite-only-sota-tex",
                        default=str(Path(PROJECT_ROOT) / "molmetal/reports/wf_3_citeonly_sota.tex"),
                        help="Path to the cite-only SOTA .tex source (used for "
                             "parser-based consistency check; hardcoded inventory "
                             "is authoritative).")
    parser.add_argument("--pb-check", action="store_true",
                        help="Run PoseBustersAdapter chemistry validity check on each generated candidate "
                             "after Vina docking (requires --physical-docking; CPU-only)")
    parser.add_argument("--pb-mode", choices=("mol", "dock", "redock"), default="mol",
                        help="PoseBusters adapter mode (default 'mol' = chemistry + geometry, no protein)")
    # WF-PB-MMFF94-Relax — Stage 1 of the dock -> MMFF94s relax -> PB
    # pipeline.  When --pb-check is set and this flag is set, the
    # docked pose is MMFF94s-relaxed (Halgren 1996, RDKit's
    # AllChem.MMFFOptimizeMolecule with mmffVariant='MMFF94s') BEFORE
    # the PoseBusters check, so the bonded-geometry checks land inside
    # the MMFF94s reference window (PoseBusters' reference minimum).
    parser.add_argument("--pb-relax-mmff94", dest="pb_relax_mmff94",
                        action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Apply MMFF94s intra-ligand relaxation to each "
                             "docked pose BEFORE the PoseBusters check "
                             "(WF-PB-MMFF94-Relax).  Requires --pb-check "
                             "+ --physical-docking.  Default False "
                             "(preserves pre-relax bit-exact behaviour).")
    parser.add_argument("--pb-relax-max-iters", type=int, default=200,
                        help="Max MMFF94s iterations (default 200; "
                             "convergence is reached well below this for "
                             "drug-like organics)")
    parser.add_argument("--job-timeout", type=float, default=600.0, help="Seconds per pocket/seed, capped at 600")
    parser.add_argument("--append", action="store_true", help="Resume matching JSON checkpoint; preserve complete records")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    for name in ("n_pockets", "n_simulations", "branching_target", "physical_exhaustiveness", "physical_n_poses", "physical_top_k", "search_docking_max_evals"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.pocket_offset < 0 or not 0 < args.job_timeout <= 600:
        parser.error("offset must be nonnegative and timeout must be in (0, 600]")
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 or s >= 2**32 for s in args.seeds):
        parser.error("seeds must be unique integers in [0, 2**32)")
    if not math.isfinite(args.search_docking_weight) or args.search_docking_weight < 0:
        parser.error("--search-docking-weight must be finite and nonnegative")
    try:
        config = load_sota_aligned_config(args.config)
        projected = config.to_mcts_kwargs()
        call = dict(projected["__mcts_call_kwargs__"])
        if args.n_simulations is not None:
            call["n_simulations"] = args.n_simulations
        if args.branching_target is not None:
            call["branching_target"] = args.branching_target
        lam = projected["__lambda_kwargs__"]
        search = {**call, **{k: projected[k] for k in ("top_k", "early_stop", "patience")},
                  **lam, "seed_strategy": args.seed_strategy}
        guidance_inputs = {}
        if args.symbolic_prior is not None:
            search["symbolic_prior"] = args.symbolic_prior
        if args.prior_state:
            if args.symbolic_prior is False:
                raise ValueError("--prior-state conflicts with --no-symbolic-prior")
            state_path = Path(args.prior_state).resolve()
            state = json.loads(state_path.read_text())
            if not isinstance(state, dict):
                raise ValueError("Prior state must be a JSON object")
            from molmetal_lam.search_alg.sweep_guidance import prepare_prior
            state, fitted = prepare_prior(state, seed=args.seeds[0], mode="frozen", data_split="test")
            if fitted is None:
                raise ValueError("--prior-state requires a fitted prior with training provenance")
            search.update(symbolic_prior=True, prior_state=state)
            guidance_inputs["prior_state"] = {"path": str(state_path), "sha256": file_digest(str(state_path))}
        # This driver selects held-out test pairs. Mutable training/replay belongs
        # in a separately labelled development or transductive experiment.
        search.update(prior_mode="frozen", prior_data_split="test")
        if args.synthesis_oracle is not None:
            search["synthesis_oracle"] = False if args.synthesis_oracle == "none" else args.synthesis_oracle
        if args.synthesis_config:
            config_path = Path(args.synthesis_config).resolve()
            search["synthesis_config_path"] = str(config_path)
            guidance_inputs["synthesis_config"] = {"path": str(config_path), "sha256": file_digest(str(config_path))}
        if args.search_docking_reward:
            search_gpu_config = None
            if args.search_docking_engine in ("auto", "quickvina2-gpu") and Path(args.gpu_config).is_file():
                search_gpu_config = json.loads(Path(args.gpu_config).read_text())
                if not Path(search_gpu_config["binary_path"]).is_file():
                    search_gpu_config = None
            reward_engine = args.search_docking_engine
            if reward_engine == "auto":
                reward_engine = "quickvina2-gpu" if search_gpu_config else "vina"
            if reward_engine == "quickvina2-gpu" and search_gpu_config is None:
                raise ValueError("Explicit GPU search reward requested but configured binary is unavailable")
            search["docking_reward_config"] = {
                "engine": reward_engine, "max_evaluations": args.search_docking_max_evals,
                "weight": args.search_docking_weight,
                "exhaustiveness": args.physical_exhaustiveness, "n_poses": args.physical_n_poses,
                "output_dir": str(Path(args.output_prefix + "_search_docking").resolve())}
            if reward_engine == "quickvina2-gpu":
                search["docking_reward_config"]["gpu_config"] = search_gpu_config
        if args.physical_docking:
            gpu_config = None
            if args.physical_engine in ("auto", "quickvina2-gpu") and Path(args.gpu_config).is_file():
                gpu_config = json.loads(Path(args.gpu_config).read_text())
                if not Path(gpu_config["binary_path"]).is_file():
                    gpu_config = None
            if args.physical_engine == "auto":
                args.physical_engine = "quickvina2-gpu" if gpu_config else "vina"
            if args.physical_engine == "quickvina2-gpu" and not gpu_config:
                raise ValueError("Explicit GPU docking requested but configured binary is unavailable")
            # --engine (D7 default 'both') selects the headline-table engine
            # set; --physical-engine still selects GPU-vs-CPU dispatch.
            # GPU path emits a single column, so when both engines are
            # requested we force CPU by overwriting --engine to 'vina' and
            # logging the conflict. The user can re-enable GPU single-engine
            # by passing --engine vina explicitly with --physical-engine
            # quickvina2-gpu.
            if args.engine in ("both", "all") and args.physical_engine == "quickvina2-gpu":
                log.warning(
                    "--engine %s is incompatible with GPU docking; "
                    "downgrading to --engine vina (single column).",
                    args.engine,
                )
                engine_arg = "vina"
            else:
                engine_arg = args.engine
            search["physical_config"] = dict(engine=engine_arg,
                exhaustiveness=args.physical_exhaustiveness, n_poses=args.physical_n_poses,
                top_k=args.physical_top_k, output_dir=str(Path(args.output_prefix + "_poses").resolve()),
                # WF-PB-MMFF94-Relax — Stage 1 of the
                # dock -> MMFF94s relax -> PB pipeline.  Both flags
                # live on physical_config (consumed by
                # evaluate_generated_poses); pb_config keeps the
                # PoseBusters mode.
                relax_mmff94=bool(args.pb_relax_mmff94),
                relax_max_iters=int(args.pb_relax_max_iters))
            if args.physical_engine == "quickvina2-gpu":
                search["physical_config"]["gpu_config"] = gpu_config
        if args.sota_diffdock:
            search["diffdock_config"] = dict(
                repo_root=str(Path(args.sota_diffdock_repo).resolve()),
                samples_per_complex=args.sota_diffdock_samples,
                timeout_sec=args.sota_diffdock_timeout,
                config_yaml=(str(Path(args.sota_diffdock_config).resolve())
                             if args.sota_diffdock_config else None),
            )
        if args.sota_flowdock:
            search["flowdock_config"] = dict(
                repo_root=str(Path(args.sota_flowdock_repo).resolve()),
                n_samples=args.sota_flowdock_samples,
                num_steps=args.sota_flowdock_steps,
                ckpt_path=(str(Path(args.sota_flowdock_ckpt).resolve())
                           if args.sota_flowdock_ckpt else None),
                device=args.sota_flowdock_device,
                timeout_sec=args.sota_flowdock_timeout,
            )
        if args.sota_biomlm:
            search["biomlm_config"] = dict(
                repo_root=str(Path(args.sota_biomlm_repo).resolve()),
                encoder=args.sota_biomlm_encoder,
                model_type=args.sota_biomlm_model_type,
                ckpt_path=(str(Path(args.sota_biomlm_ckpt).resolve())
                           if args.sota_biomlm_ckpt else None),
                timeout_sec=args.sota_biomlm_timeout,
            )
        if args.sota_aizynth:
            search["aizynth_config"] = dict(
                mode=args.sota_aizynth_mode,
                config_path=(str(Path(args.sota_aizynth_config).resolve())
                             if args.sota_aizynth_config else None),
                isolated_run_sh=str(Path(args.sota_aizynth_isolated_run_sh).resolve()),
                timeout_sec=args.sota_aizynth_timeout,
                max_iterations=args.sota_aizynth_max_iterations,
                time_limit_s=args.sota_aizynth_time_limit,
            )
        if args.pb_check:
            if not args.physical_docking:
                raise ValueError("--pb-check requires --physical-docking (PB runs after Vina docking)")
            search["pb_config"] = dict(mode=args.pb_mode,
                                       relax_mmff94=bool(args.pb_relax_mmff94),
                                       relax_max_iters=int(args.pb_relax_max_iters))
        # Initialise the cite-only comparator placeholder before the
        # manifest is loaded.  The actual instantiation happens after
        # ``pairs`` is defined so we can capture ``len(pairs)`` in the
        # user protocol_flags.
        cite_only_comparator = None
        if args.cite_only_sota_comparator:
            from molmetal_lam.sbdd_env.cite_only_sota_comparator import (
                CiteOnlySOTAComparator,
                DEFAULT_PROTOCOL_FLAGS,
            )
            cite_only_comparator = CiteOnlySOTAComparator(
                tex_path=str(Path(args.cite_only_sota_tex).resolve()),
            )
        args.n_pockets = min(args.n_pockets, config.n_test_pockets)
        pairs = load_manifest(args.manifest, args.pocket_offset, args.n_pockets)
        if args.pockets:
            root = Path(args.pockets).resolve()
            if not root.is_dir() or any(not Path(row[key]).is_relative_to(root)
                                       for row in pairs for key in ("receptor_path", "ligand_path")):
                raise ValueError("Manifest pair paths must be inside the --pockets extraction root")
        if len(pairs) != args.n_pockets:
            raise ValueError(f"Requested {args.n_pockets} pairs but manifest selection contains {len(pairs)}")
        # WF-T25 Cite-Only SOTA Comparator (TODO-22 §4c + TODO-25 (d)).
        # After manifest load, populate the user protocol_flags with
        # the actual n_pockets / n_seeds for the M6 / M7 mismatch check.
        if cite_only_comparator is not None:
            from molmetal_lam.sbdd_env.cite_only_sota_comparator import (
                DEFAULT_PROTOCOL_FLAGS,
            )
            user_protocol_flags = dict(DEFAULT_PROTOCOL_FLAGS)
            engine_for_comparator = args.engine
            if engine_for_comparator in ("both", "all"):
                # "both" emits both Vina and QVina columns; cite-only
                # comparator matches whichever single engine the user's
                # downstream analysis pins (default: vina).
                engine_for_comparator = "vina"
            user_protocol_flags["docking_engine"] = engine_for_comparator
            user_protocol_flags["validity_def"] = "RDKit + PB (graceful)"
            user_protocol_flags["n_seeds"] = len(set(args.seeds))
            user_protocol_flags["n_pockets"] = len(pairs)
            cite_only_comparator.user_protocol_flags = user_protocol_flags
            log.info("Cite-only SOTA comparator enabled (n_rows=%d, n_flags=%d, "
                     "discrepancies=%d)", len(cite_only_comparator.rows),
                     len(cite_only_comparator.flags),
                     len(cite_only_comparator.parser_discrepancies))
        jobs = [dict(pair, seed=seed) for pair in pairs for seed in args.seeds]
        inputs = [{**p, "receptor_sha256": file_digest(p["receptor_path"]),
                   "ligand_sha256": file_digest(p["ligand_path"])} for p in pairs]
        metadata = {"schema_version": 4, "mode": f"{args.seed_strategy}_initialized_search",
                    "manifest": str(Path(args.manifest).resolve()), "manifest_sha256": file_digest(args.manifest),
                    "config_sha256": file_digest(args.config), "search": search, "seeds": args.seeds,
                    "driver_sha256": file_digest(__file__), "runner_sha256": file_digest(LAMBDA_SWEEP),
                    "requested_protocol": projected["__protocol_fingerprint__"],
                    "requested_lambda_settings": lam,
                    "physical_evaluation_requested": args.physical_docking,
                    "physical_evaluator_sha256": file_digest(str(Path(HERE) / "evaluate_generated_poses.py")),
                    "physical_implementation_sha256": {
                        name: file_digest(str(Path(PROJECT_ROOT) / name)) for name in (
                            "molmetal/molmetal_lam/sbdd_env/vina_adapter.py",
                            "molmetal/validation/posebusters_runner.py",
                            "molmetal/scripts/prepare_crossdocked_receptor.py",
                            "molmetal/scripts/receptor_preparation_for_evaluation.py",
                            "molmetal/molmetal_lam/sbdd_env/diffdock_sota_scoring.py",
                            "molmetal/molmetal_lam/sbdd_env/flowdock_sota_scoring.py",
                            "molmetal/molmetal_lam/sbdd_env/aizynth_sota_scoring.py",
                            "molmetal/molmetal_lam/sbdd_env/biomlm_sota_scoring.py")},
                    "runtime_sha256": runtime_fingerprints(),
                    "not_executed": ((["physical_docking", "posebusters"] if not args.physical_docking else [])
                                     + (["posebusters_adapter_extra_check"] if not args.pb_check else [])
                                     + ["symbolic_prior_refit"]),
                    # WF-T25 Cite-Only SOTA Comparator (TODO-22 §4c + TODO-25 (d)).
                    # Cached here so write_json can attach per-pocket comparison
                    # rows at end-of-run without re-importing the module.
                    "cite_only_sota_comparator": (None if cite_only_comparator is None else {
                        "rows": [r.to_dict() for r in cite_only_comparator.rows],
                        "flags": [f.to_dict() for f in cite_only_comparator.flags],
                        "user_protocol_flags": getattr(cite_only_comparator,
                                                       "user_protocol_flags", {}),
                        "parser_discrepancies": cite_only_comparator.parser_discrepancies,
                        "tex_path": str(Path(args.cite_only_sota_tex).resolve()),
                        "n_rows": len(cite_only_comparator.rows),
                        "n_flags": len(cite_only_comparator.flags),
                        "comparison_mode": "cited-only (no head-to-head)",
                    }),
                    "guidance_inputs": guidance_inputs,
                    "guidance_execution": "See each job's diagnostics and search_config.guidance; requested is not executed",
                    "inputs": inputs, "job_timeout_seconds": args.job_timeout}
    except (OSError, ValueError, KeyError) as exc:
        log.error("Preflight failed: %s", exc)
        return 2
    # TODO-30 / P6.2 — checkpoint_sha256 emission.  Hash the input
    # SMILES (per pocket), the receptor PDB, and the active CLI flags
    # into a deterministic fingerprint and persist it next to the run
    # as ``runs/{run_id}/fingerprint.json``.  The fingerprint is
    # consumed by test_reproducibility_split.py::test_fingerprint_*
    # to verify stability across runs.
    try:
        from dataclasses import asdict as _asdict
        # Collect active CLI flags from the parsed namespace.  We
        # exclude the worker-only flags (--worker-input/output) so a
        # parent + child invocation share the same fingerprint.
        excluded = {"worker_input", "worker_output"}
        active_flags = {
            k: (v if not isinstance(v, Path) else str(v))
            for k, v in vars(args).items()
            if k not in excluded
        }
        # Active CLI flags are serialised with sorted keys for
        # stability; their SHA-256 anchors the CLI surface.
        active_flags_str = json.dumps(active_flags, sort_keys=True, default=str)
        smiles_concat = "\n".join(
            sorted({r.top1_smiles for r in results} if results else {
                # Fallback to manifest pair ids when no cells have run yet
                p["pocket_id"] for p in pairs
            })
        )
        fingerprint_payload = {
            "schema_version": 1,
            "tool": "r4_c_full_sweep.py",
            "run_id": Path(args.output_prefix).name,
            "cli_sha256": hashlib.sha256(active_flags_str.encode("utf-8")).hexdigest(),
            "active_flags": active_flags,
            "smiles_concat_sha256": hashlib.sha256(smiles_concat.encode("utf-8")).hexdigest(),
            "receptor_sha256": {p["pocket_id"]: p["receptor_sha256"] for p in inputs},
            "ligand_sha256": {p["pocket_id"]: p["ligand_sha256"] for p in inputs},
            "manifest_sha256": metadata.get("manifest_sha256"),
            "driver_sha256": metadata.get("driver_sha256"),
            "runner_sha256": metadata.get("runner_sha256"),
            "runtime_sha256": metadata.get("runtime_sha256"),
            "seeds": list(args.seeds),
            "n_pockets": int(metadata.get("requested_protocol", {}).get("n_pockets", len(pairs))),
        }
        run_dir = Path(args.output_prefix).parent / "runs" / Path(args.output_prefix).name
        run_dir.mkdir(parents=True, exist_ok=True)
        fingerprint_path = run_dir / "fingerprint.json"
        with fingerprint_path.open("w") as fh:
            json.dump(fingerprint_payload, fh, indent=2, sort_keys=True)
        log.info("Wrote checkpoint fingerprint to %s", fingerprint_path)
    except Exception as exc:
        log.warning("Could not emit checkpoint fingerprint: %s", exc)
    log.info("Effective run plan: n_pockets=%d (config cap = %d), seeds=%s, jobs=%d",
             args.n_pockets, config.n_test_pockets, args.seeds, len(jobs))
    if args.dry_run:
        print("[dry-run] SOTA-aligned config OK:")
        print(f"  config_path       : {args.config}")
        print(f"  test_set          : {config.test_set}")
        print(f"  n_test_pockets    : {config.n_test_pockets}")
        print(f"  mcts (direct)     : top_k={search['top_k']}, early_stop={search['early_stop']}, patience={search['patience']}")
        print(f"  mcts (call-time)  : {call}")
        print(f"  lambda_kwargs     : {lam}")
        print(f"  effective_guidance: prior={search.get('symbolic_prior')}, mode=frozen, synthesis={search.get('synthesis_oracle')}")
        print(f"  guidance_inputs   : {guidance_inputs}")
        print(f"  docking_reward    : {search.get('docking_reward_config')}")
        print(f"  pb_check          : {search.get('pb_config')}")
        print(f"  n_pockets_request : {args.n_pockets}")
        print(f"  n_pockets_found   : {len(pairs)}")
        print(f"  pair_ids          : {[p['pocket_id'] for p in pairs]}")
        print(f"  seeds             : {args.seeds}")
        print("[dry-run] OK — sweep NOT executed.")
        return 0
    csv_path, json_path, md_path = [args.output_prefix + ext for ext in (".csv", ".json", ".md")]
    results = []
    if args.append and Path(json_path).exists():
        try:
            previous = json.loads(Path(json_path).read_text())
            if previous.get("metadata") != metadata:
                raise ValueError("Checkpoint inputs/configuration differ; choose a new output prefix")
            results = [PocketResult(**row) for row in previous["per_pocket"]]
            if len({(r.pocket_id, r.seed) for r in results}) != len(results):
                raise ValueError("Checkpoint has duplicate pocket/seed records")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            log.error("Resume failed: %s", exc)
            return 2
    completed = {(r.pocket_id, r.seed) for r in results}
    for i, job in enumerate(jobs):
        if (job["pocket_id"], job["seed"]) in completed:
            continue
        if runtime_fingerprints() != metadata["runtime_sha256"]:
            log.error("Runtime code changed during experiment; existing records preserved, use a new output prefix")
            return 2
        if any(not Path(entry["path"]).is_file() or file_digest(entry["path"]) != entry["sha256"]
               for entry in guidance_inputs.values()):
            log.error("Guidance input changed during experiment; existing records preserved, use a new output prefix")
            return 2
        log.info("[%d/%d] %s seed=%d search=%s", i + 1, len(jobs), job["pocket_id"], job["seed"], search)
        log_id = hashlib.sha256(f"{job['pocket_id']}:{job['seed']}".encode()).hexdigest()[:16]
        result = execute_job(job, search, args.job_timeout, f"{args.output_prefix}_logs/{log_id}.log")
        results.append(result)
        log.info("status=%s candidates=%d seconds=%.2f", result.status, result.n_candidates, result.wall_seconds)
        write_json(results, aggregate(results), json_path, metadata)
        write_csv(results, csv_path)
        write_markdown(results, aggregate(results), md_path, args.n_pockets, call["n_simulations"], call["max_depth"])
    summary = aggregate(results)
    log.info("Completed jobs=%d successful=%d failed/empty=%d", len(results), summary["n_pockets_ok"], summary["n_pockets_fail"])
    physical_ok = not args.physical_docking or summary["physical_jobs_completed"] == len(jobs)
    return 0 if summary["n_pockets_ok"] == len(jobs) and physical_ok else 1


if __name__ == "__main__":
    sys.exit(main())
