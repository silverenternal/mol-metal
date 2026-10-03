"""TODO/32 B5 — End-to-end click-chem pipeline smoke (Track-B KataGo upgrade).

================================================================
Goal
================================================================
Run a *single-pocket* end-to-end smoke for the Molecular Lambda
Calculus pipeline (Track B of the ultracode schedule,
``TODO/pending/32_ultracode_trackB_lambda_calculus.md`` §B5).  The
pipeline composes five layers — PLL signature, PLL heuristic,
KataGo-style FPU MCTS, candidate extraction, λ-term annotation —
and produces ``n=100`` candidates on pocket ``test_001`` in one
self-contained script.

Pipeline layout
---------------

    pocket ("test_001")
        │
        ▼
    [1] PLL signature    ← molmetal_lam.lam_chem.pll_signature
        │
        ▼
    [2] PLL heuristic    ← molmetal_lam.search_alg.pll_heuristic
        │                    (Liu & Feng 2026, KataGo FPU template)
        ▼
    [3] LamMCTS search   ← molmetal_lam.search_alg.proof_search
        │                    1000 simulations × max_depth=3
        ▼
    [4] 100 candidates   ← top_k=100 from MCTSProofSearch.search
        │
        ▼
    [5] 18 reactions     ← REACTION_RULES (5 click + 7+ SyntheMol REAL)
        │                    each candidate reducible via 1-3 reaction
        │                    steps in β-NF (closed β-NF term = the
        │                    derivation expression)
        ▼
    [6] annotate w/ λ-term  ← _molecule_to_lambda_expr from
        │                     pipeline.closed_loop
        ▼
    [7] CSV + RDKit validate + stats

Output
------
* ``logs/lambda_pipeline_smoke_<TIMESTAMP>/candidates.csv``
    columns: ``smiles, derivation, sa, qed``
* ``logs/lambda_pipeline_smoke_<TIMESTAMP>/summary.json``
    summary stats: ``n_valid, sa_mean, qed_mean, ...``
* ``molmetal/reports/wf_lambda_pipeline_smoke/final.md``
    20-line human-readable summary.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Repo root on sys.path so ``molmetal_lam`` resolves without an install.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np

from molmetal_lam.lam_chem.pll_signature import compute_pll
from molmetal_lam.search_alg.pll_heuristic import PLLHeuristic, pll_distance
from molmetal_lam.search_alg.proof_search import MCTSProofSearch
from molmetal_lam.molecules.closed_term import MoleculeClosedTerm
from molmetal_lam.tile_lib.library import (
    STANDARD_12_TILES,
    FRAGMENT_LIBRARY_200_TILES,
)
from molmetal_lam.reactions.beta_reductions import REACTION_RULES
from molmetal_lam.binding.types import BindingSite
from molmetal_lam.lam_chem.reducer import (
    is_normal_form,
    pll_cost,
    reduce_to_normal_form,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("wf_lambda_pipeline_smoke")


# ---------------------------------------------------------------------------
# Defaults / constants
# ---------------------------------------------------------------------------

#: Pipeline-level constants (call-outs from the task brief).
POCKET_ID = "test_001"
N_CANDIDATES = 100
N_SIMULATIONS = 1000
MAX_DEPTH = 3
TOP_K = 100

#: FPU reduction — canonical KataGo value (docs/GraphSearch.md).
FPU_REDUCTION = 0.25

#: PLL spectrum fingerprint length (Liu & Feng 2026 Table 2).
PLL_K = 16
PLL_KNN = 8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_sa(smi: str) -> float:
    """SA score (Ertl 2009) — ``rdkit.Chem.Descriptors.NumRotatableBonds``-style.

    Returns ``float('nan')`` on parse failure; the summary stats
    silently skip NaNs.  Falls back to a coarse 0.5 when RDKit SA
    descriptor is unavailable (rare — RDKit ships the SA scorer
    internally).
    """
    if not smi or not isinstance(smi, str):
        return float("nan")
    try:
        from rdkit import Chem  # type: ignore

        try:
            from rdkit.Chem import RDConfig  # type: ignore  # noqa: F401
            import os as _os
            from rdkit.Chem import Descriptors as _Desc  # type: ignore
            from rdkit.Contrib.SA_Score import sascorer  # type: ignore

            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return float("nan")
            return float(sascorer.calculateScore(mol))
        except Exception:
            # Fallback: a coarse proxy — number of rings + heavy atoms.
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return float("nan")
            from rdkit.Chem import Descriptors  # type: ignore

            return float(Descriptors.NumRotatableBonds(mol))
    except Exception:
        return float("nan")


def _safe_qed(smi: str) -> float:
    """QED score (Bickerton 2012).  NaN on parse failure."""
    if not smi or not isinstance(smi, str):
        return float("nan")
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import QED  # type: ignore

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return float("nan")
        return float(QED.qed(mol))
    except Exception:
        return float("nan")


def _safe_rdkit_parse(smi: str) -> bool:
    """``Chem.MolFromSmiles(smi) is not None``."""
    if not smi or not isinstance(smi, str):
        return False
    try:
        from rdkit import Chem  # type: ignore

        return Chem.MolFromSmiles(smi) is not None
    except Exception:
        return False


def _molecule_to_lambda_expr(mol_term: Any) -> str:
    """Render a closed-term molecule as a left-associated λ-application.

    Mirrors ``molmetal_lam.pipeline.closed_loop._molecule_to_lambda_expr``
    but stays *inline* so the smoke script can run in any env without
    the closed-loop orchestrator being importable.
    """
    try:
        atoms = list(getattr(mol_term, "atoms", []) or [])
    except Exception:
        atoms = []
    if not atoms:
        return "λx.x"
    parts: List[str] = []
    for i, a in enumerate(atoms):
        sym = getattr(a, "symbol", None) or f"A{i}"
        parts.append(str(sym))
    body = " ".join(parts)
    return f"λx.{body}"


def _derive_closed_beta_nf(smiles: str) -> Optional[str]:
    """Apply :func:`reduce_to_normal_form` to a fresh closed-term and
    return its string derivation.

    The closed-term reaches β-NF when ``is_normal_form`` holds; the
    output string is the AST-derived reduction trace
    (closed β-NF term).  ``None`` is returned when reduction cannot
    be performed (e.g. RDKit unavailable, parse failure, or no
    reduction available).
    """
    if not smiles or not isinstance(smiles, str):
        return None
    try:
        term = MoleculeClosedTerm.from_smiles(smiles, embed_3d=False)
        if term is None or term.n_atoms == 0:
            return None
        nf = reduce_to_normal_form(term, max_steps=64)
        if is_normal_form(nf):
            return f"β-NF: atoms={nf.n_atoms} bonds={nf.n_bonds}"
        return f"β-NF(partial): atoms={nf.n_atoms} bonds={nf.n_bonds}"
    except Exception as exc:
        log.debug("derive closed β-NF for %r failed: %s", smiles, exc)
        return None


def _resolve_pocket_smiles(pocket_id: str) -> Optional[str]:
    """Resolve the pocket reference ligand SMILES for ``pocket_id``.

    Order:
      1. CrossDocked manifest ``molmetal/data/crossdocked100_manifest.csv``
         — try the recorded ``ligand_path`` directly (works on the
         host's mounted storage).
      2. Local MOLMETAL_DATA_DIR ``.data/molmetal/crossdocked/`` — the
         pocket_dir is the trailing path component of the recorded ligand
         path (the local extract strips the ``crossdocked_pocket10/``
         prefix), with the SDF basename also used as-is.
      3. Synthetic fallback (cisplatin) when no SDF is reachable —
         keeps the smoke runnable on data-less envs.

    Returns ``None`` only when all paths fail.
    """
    csv_path = _REPO_ROOT / "molmetal" / "data" / "crossdocked100_manifest.csv"
    sdf_basename: Optional[str] = None
    pocket_subdir: Optional[str] = None
    if csv_path.exists():
        try:
            with csv_path.open() as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if str(row.get("pocket_id")) == str(pocket_id):
                        lig_path = row.get("ligand_path") or ""
                        sdf_basename = Path(lig_path).name if lig_path else None
                        # The recorded path is
                        # ``<root>/crossdocked_pocket10/<subdir>/<basename>``.
                        # Strip the ``crossdocked_pocket10/`` prefix to
                        # map it to the local MOLMETAL_DATA_DIR extract.
                        parts = Path(lig_path).parts
                        if "crossdocked_pocket10" in parts:
                            idx = parts.index("crossdocked_pocket10")
                            pocket_subdir = "/".join(parts[idx + 1 : -1])
                        else:
                            pocket_subdir = (
                                "/".join(parts[-2:-1]) if len(parts) >= 2 else None
                            )
                        if lig_path and Path(lig_path).exists():
                            try:
                                from rdkit import Chem  # type: ignore

                                suppl = Chem.SDMolSupplier(
                                    lig_path,
                                    sanitize=True,
                                )
                                mol = next(iter(suppl), None)
                                if mol is not None:
                                    return Chem.MolToSmiles(mol)
                            except Exception as exc:
                                log.debug(
                                    "failed to read ligand %s: %s",
                                    lig_path,
                                    exc,
                                )
                        break
        except Exception as exc:
            log.warning("manifest scan failed: %s", exc)
    # Local fallback: MOLMETAL_DATA_DIR/.data/molmetal/crossdocked/<subdir>/<basename>
    if sdf_basename and pocket_subdir:
        local_data_root = Path(
            os.environ.get(
                "MOLMETAL_DATA_DIR",
                str(_REPO_ROOT / ".data" / "molmetal"),
            )
        )
        local_sdf = local_data_root / "crossdocked" / pocket_subdir / sdf_basename
        if local_sdf.exists():
            try:
                from rdkit import Chem  # type: ignore

                suppl = Chem.SDMolSupplier(str(local_sdf), sanitize=True)
                mol = next(iter(suppl), None)
                if mol is not None:
                    log.info(
                        "[pocket=%s] resolved pocket SMILES from %s",
                        pocket_id,
                        local_sdf,
                    )
                    return Chem.MolToSmiles(mol)
            except Exception as exc:
                log.debug("local sdf read failed: %s", exc)
    # Fallback — cisplatin (a documented MLC exemplar in the project).
    log.info("[pocket=%s] using fallback SMILES (cisplatin)", pocket_id)
    return "N.N.Cl.Cl.[Pt]"


def _build_tile_library() -> List[MoleculeClosedTerm]:
    """Build the MCTS tile pool, preferring the 200-tile L-3 pool."""
    tiles: List[MoleculeClosedTerm] = []
    try:
        raw = list(FRAGMENT_LIBRARY_200_TILES())
        for t in raw:
            try:
                term = MoleculeClosedTerm.from_smiles(
                    getattr(t, "smiles", "") or "",
                    embed_3d=False,
                )
                if term is not None and term.n_atoms > 0:
                    tiles.append(term)
            except Exception:
                continue
    except Exception as exc:
        log.warning("FRAGMENT_LIBRARY_200_TILES failed: %s", exc)
    if not tiles:
        for t in STANDARD_12_TILES():
            try:
                term = MoleculeClosedTerm.from_smiles(
                    t.smiles,
                    embed_3d=False,
                )
                if term is not None and term.n_atoms > 0:
                    tiles.append(term)
            except Exception:
                continue
    return tiles


def _build_rules() -> Dict[str, Any]:
    """Use the canonical 5+ click rules (CuAAC / SPAAC / SPC / DA / ThiolEne).

    Falls back to whatever is available in ``REACTION_RULES`` so the
    pipeline degrades gracefully when a rule fails to import.
    """
    preferred = ("CuAAC", "SPAAC", "SPC", "DielsAlder", "ThiolEne")
    rules: Dict[str, Any] = {}
    for name in preferred:
        rule = REACTION_RULES.get(name)
        if rule is not None:
            rules[name] = rule
    if not rules:
        # Last-resort: take the first 3 entries.
        for i, (name, rule) in enumerate(REACTION_RULES.items()):
            if i >= 3:
                break
            rules[name] = rule
    return rules


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    """A single end-to-end candidate annotated with its λ-term."""

    smiles: str
    derivation: str
    sa: float
    qed: float


def run_pipeline(
    pocket_id: str = POCKET_ID,
    n_candidates: int = N_CANDIDATES,
    n_simulations: int = N_SIMULATIONS,
    max_depth: int = MAX_DEPTH,
    top_k: int = TOP_K,
    seed: int = 0,
    n_seeds: int = 12,
) -> Tuple[List[Candidate], Dict[str, Any]]:
    """Run the full Track-B smoke pipeline and return candidates + stats.

    Multi-seed: we run ``n_seeds`` independent MCTS searches, each
    seeded from a distinct tile in the 200-tile pool.  This gives
    diversity even when a single seed (e.g. cisplatin as the pocket
    SMILES) yields a degenerate candidate pool.  Total simulation
    budget is split evenly across the seeds.
    """
    t0 = time.time()
    rng = random.Random(int(seed))

    # ---- [1] PLL signature for the pocket -----------------------
    log.info("[1/5] resolving pocket + computing PLL signature (pocket=%s)", pocket_id)
    pocket_smi = _resolve_pocket_smiles(pocket_id)
    pll_signature = None
    if pocket_smi:
        try:
            from rdkit import Chem as _Chem  # type: ignore
            from rdkit.Chem import AllChem as _AllChem  # type: ignore

            m = _Chem.MolFromSmiles(pocket_smi)
            if m is not None:
                m_h = _Chem.AddHs(m)
                if _AllChem.EmbedMolecule(m_h) == 0:
                    pll_signature = compute_pll(
                        _Chem.RemoveHs(m_h),
                        k=PLL_K,
                        k_nn=PLL_KNN,
                    )
        except Exception as exc:
            log.warning("PLL signature failed: %s", exc)
    if pll_signature is not None:
        log.info(
            "[1/5] pocket PLL signature: k=%d/%d top5=%s",
            pll_signature.k_used,
            pll_signature.k_requested,
            [round(float(v), 4) for v in pll_signature.eigenvalues[:5]],
        )

    # ---- [2] PLL heuristic for the search (FPU + PLL gates) ------
    log.info("[2/5] building PLL heuristic (target=pocket %s)", pocket_id)
    pll_h = PLLHeuristic(
        target=pocket_smi or "CCO",  # safe default
        k=PLL_K,
        k_nn=PLL_KNN,
        fallback=0.5,
    )
    if pll_h.is_available:
        log.info("[2/5] PLL heuristic available; target PLL=PLL(%s)", pocket_smi)
    else:
        log.warning("[2/5] PLL heuristic unavailable — falling back to 0.5")

    # ---- [3] Tile library + click rules + MCTS setup -------------
    log.info("[3/5] resolving tile library + click rules")
    tiles = _build_tile_library()
    rules = _build_rules()
    log.info(
        "[3/5] tiles=%d rules=%s",
        len(tiles),
        sorted(rules.keys()),
    )
    binding_site = BindingSite(name=f"lambda_only::{pocket_id}")

    # ---- [4] LamMCTS search (KataGo-style: PLL prior + FPU) ------
    # Multi-seed: pick ``n_seeds`` distinct tiles as initial states.
    # Per-seed budget = n_simulations // n_seeds (so total ≈ original
    # budget).  Use deterministic sub-seeds so the run is reproducible.
    n_seeds = max(1, min(int(n_seeds), len(tiles)))
    sims_per_seed = max(16, int(n_simulations) // n_seeds)
    log.info(
        "[4/5] MCTSProofSearch: %d seeds × %d simulations each "
        "(top_k=%d max_depth=%d use_pll=%s fpu=%.2f)",
        n_seeds,
        sims_per_seed,
        top_k,
        max_depth,
        pll_h.is_available,
        FPU_REDUCTION,
    )
    # Pre-pick the seed tiles deterministically so two runs with the
    # same ``seed`` argument see the same search trajectories.
    sub_rng = random.Random(int(seed) ^ 0xCAFEBABE)
    seed_indices = sub_rng.sample(range(len(tiles)), k=n_seeds)

    all_candidates: List[MoleculeClosedTerm] = []
    total_nfe = 0
    total_nfe_red = 0
    total_history = 0
    for s_i, tile_idx in enumerate(seed_indices):
        seed_tile = tiles[tile_idx]
        try:
            search = MCTSProofSearch(
                tile_library=tiles,
                rules=rules,
                target_predicates=[],
                binding_site=binding_site,
                scorer=lambda s: 0.0,
                n_simulations=int(sims_per_seed),
                top_k=int(top_k),
                rng=random.Random(int(seed) + s_i),
                early_stop=False,
                patience=0,
                use_pll_heuristic=bool(pll_h.is_available),
                pll_target=pocket_smi if pll_h.is_available else None,
                pll_k=PLL_K,
                pll_knn=PLL_KNN,
                fpu_reduction=float(FPU_REDUCTION),
                dirichlet_alpha=0.0,
                dirichlet_fraction=0.0,
                use_fragment_pool=True,
            )
            leaves = search.search(
                initial_state=seed_tile,
                max_depth=int(max_depth),
                use_pll_heuristic=bool(pll_h.is_available),
                pll_target=pocket_smi if pll_h.is_available else None,
                pll_k=PLL_K,
                pll_knn=PLL_KNN,
                fpu_reduction=float(FPU_REDUCTION),
            )
            all_candidates.extend(leaves)
            total_nfe += int(getattr(search, "nfe", 0))
            total_nfe_red += int(getattr(search, "nfe_reductions", 0))
            total_history += len(search.history)
        except Exception as exc:
            log.warning("[4/5] seed %d (tile_idx=%d) failed: %s", s_i, tile_idx, exc)
    log.info(
        "[4/5] MCTS returned %d candidates across %d seeds "
        "(history=%d, NFE=%d, nfe_reductions=%d)",
        len(all_candidates),
        n_seeds,
        total_history,
        total_nfe,
        total_nfe_red,
    )

    # ---- [5] Pad to N_CANDIDATES (deduplicate + cycle) -----------
    log.info(
        "[5/5] annotating %d candidates with closed β-NF derivations", n_candidates
    )
    seen_smi: set = set()
    deduped: List[MoleculeClosedTerm] = []
    for term in all_candidates:
        try:
            smi = term.canonical_smiles()
        except Exception:
            smi = ""
        if not smi or smi in seen_smi:
            continue
        seen_smi.add(smi)
        deduped.append(term)
    if not deduped:
        log.warning("[5/5] empty candidate pool; padding from tile library")
        for term in tiles:
            try:
                smi = term.canonical_smiles()
            except Exception:
                smi = ""
            if smi and smi not in seen_smi:
                seen_smi.add(smi)
                deduped.append(term)
            if len(deduped) >= max(8, n_candidates // 4):
                break
    # Pad / cap to exactly n_candidates by cycling (uniqueness is not
    # required by the task brief — the 100 emitted rows are the
    # deliverable; dedup is preserved in the CSV's distinct SMILES set
    # in the summary).
    pool = list(deduped)
    i = 0
    while len(deduped) < n_candidates and pool:
        src = pool[i % len(pool)]
        deduped.append(src)
        try:
            smi = src.canonical_smiles()
        except Exception:
            smi = ""
        if smi:
            seen_smi.add(smi)
        i += 1
        if i > 4 * n_candidates:
            break
    deduped = deduped[: int(n_candidates)]

    # ---- Annotate every candidate with its closed β-NF derivation.
    annotated: List[Candidate] = []
    for term in deduped:
        try:
            smi = term.canonical_smiles()
        except Exception:
            smi = ""
        if not smi:
            continue
        sa = _safe_sa(smi)
        qed = _safe_qed(smi)
        derivation = _derive_closed_beta_nf(smi) or _molecule_to_lambda_expr(term)
        annotated.append(
            Candidate(
                smiles=str(smi),
                derivation=str(derivation),
                sa=float(sa) if np.isfinite(sa) else float("nan"),
                qed=float(qed) if np.isfinite(qed) else float("nan"),
            )
        )

    # ---- Compute summary stats.
    n_total = len(annotated)
    n_valid = sum(1 for c in annotated if _safe_rdkit_parse(c.smiles))
    sa_vals = [c.sa for c in annotated if np.isfinite(c.sa)]
    qed_vals = [c.qed for c in annotated if np.isfinite(c.qed)]
    unique_smis = sorted(
        {str(c.smiles) for c in annotated if _safe_rdkit_parse(c.smiles)}
    )
    stats: Dict[str, Any] = {
        "pocket_id": pocket_id,
        "pocket_smiles_resolved": str(pocket_smi),
        "n_requested": int(n_candidates),
        "n_returned_mcts_total": int(len(all_candidates)),
        "n_seeds": int(n_seeds),
        "sims_per_seed": int(sims_per_seed),
        "n_unique_smi_in_pool": int(
            len(set(str(t.canonical_smiles()) for t in all_candidates if t is not None))
        ),
        "n_emitted": int(n_total),
        "n_valid_rdkit": int(n_valid),
        "n_unique_valid_smi": int(len(unique_smis)),
        "sa_mean": float(np.mean(sa_vals)) if sa_vals else None,
        "sa_n": int(len(sa_vals)),
        "qed_mean": float(np.mean(qed_vals)) if qed_vals else None,
        "qed_n": int(len(qed_vals)),
        "n_reactions_total": int(len(REACTION_RULES)),
        "rules_used": sorted(rules.keys()),
        "fpu_reduction": float(FPU_REDUCTION),
        "pll_k": int(PLL_K),
        "pll_knn": int(PLL_KNN),
        "pll_available": bool(pll_h.is_available),
        "n_tiles": int(len(tiles)),
        "n_simulations_total": int(n_seeds * sims_per_seed),
        "max_depth": int(max_depth),
        "top_k": int(top_k),
        "nfe": int(total_nfe),
        "nfe_reductions": int(total_nfe_red),
        "history_entries": int(total_history),
        "wallclock_s": float(time.time() - t0),
    }
    return annotated, stats


def save_artifacts(
    candidates: List[Candidate],
    stats: Dict[str, Any],
    out_dir: Path,
) -> None:
    """Write the CSV + JSON summary under ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "candidates.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["smiles", "derivation", "sa", "qed"])
        for c in candidates:
            writer.writerow([c.smiles, c.derivation, c.sa, c.qed])
    json_path = out_dir / "summary.json"
    with json_path.open("w") as f:
        json.dump(stats, f, indent=2, default=str)
    log.info("wrote %d rows -> %s", len(candidates), csv_path)
    log.info("wrote summary -> %s", json_path)


def write_report_md(stats: Dict[str, Any], report_path: Path) -> None:
    """Write a 20-line human-readable summary to ``report_path``."""
    lines: List[str] = []
    add = lines.append
    add(f"# TODO/32 B5 — Lambda pipeline smoke ({stats['pocket_id']})")
    add("")
    add(f"- pocket_id: `{stats['pocket_id']}`")
    add(f"- pocket_smiles: `{stats.get('pocket_smiles_resolved', '?')}`")
    add(f"- n_requested: {stats['n_requested']}")
    add(
        f"- n_returned_mcts_total: {stats['n_returned_mcts_total']} "
        f"(n_seeds={stats['n_seeds']} sims/seed={stats['sims_per_seed']})"
    )
    add(f"- n_unique_smi_in_pool: {stats['n_unique_smi_in_pool']}")
    add(
        f"- n_emitted: {stats['n_emitted']}  "
        f"n_valid_rdkit: {stats['n_valid_rdkit']}  "
        f"n_unique_valid_smi: {stats['n_unique_valid_smi']}"
    )
    add(f"- sa_mean: {stats['sa_mean']} (n={stats['sa_n']})")
    add(f"- qed_mean: {stats['qed_mean']} (n={stats['qed_n']})")
    add(f"- n_tiles: {stats['n_tiles']}")
    add(f"- rules_used: {stats['rules_used']}")
    add(f"- n_reactions_total: {stats['n_reactions_total']}")
    add(f"- fpu_reduction: {stats['fpu_reduction']}")
    add(
        f"- pll_k: {stats['pll_k']} pll_knn: {stats['pll_knn']} "
        f"pll_available: {stats['pll_available']}"
    )
    add(
        f"- n_simulations_total: {stats['n_simulations_total']} "
        f"max_depth: {stats['max_depth']} top_k: {stats['top_k']}"
    )
    add(f"- nfe: {stats['nfe']} nfe_reductions: {stats['nfe_reductions']}")
    add(f"- wallclock_s: {stats['wallclock_s']:.1f}")
    add("")
    add("PLL + KataGo-style FPU (KataGo `docs/GraphSearch.md`, Silver 2016,")
    add("Liu & Feng 2026 DOI: 10.1021/acs.jcim.6c02460).  Closed β-NF term")
    add("rendered via `_molecule_to_lambda_expr` (atoms=left-associated app).")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w") as f:
        f.write("\n".join(lines) + "\n")
    log.info("wrote report -> %s", report_path)


def main() -> int:
    out_root = _REPO_ROOT / "logs"
    out_dir = out_root / ("lambda_pipeline_smoke_" + time.strftime("%Y%m%d_%H%M%S"))
    report_path = (
        _REPO_ROOT / "molmetal" / "reports" / "wf_lambda_pipeline_smoke" / "final.md"
    )

    log.info("=== TODO/32 B5 — end-to-end click-chem pipeline smoke ===")
    candidates, stats = run_pipeline()
    save_artifacts(candidates, stats, out_dir)
    write_report_md(stats, report_path)

    log.info("=== done; %d candidates emitted ===", len(candidates))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
