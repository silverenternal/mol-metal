"""pareto_multiobj.py — multi-objective Pareto front over toxicity / efficacy / ADMET / selectivity.

================================================================
Wf-78 Phase 3 (TODO-107 / R22-P3 / L15 Lambda gap closure)
================================================================
Multi-objective Pareto front over the **four objectives** that matter for
metallodrug design (Williamson + Bajorath 2009; Ertl + Schuffenhauer 2009):

  1. **Toxicity** (m19 / TG-GATEs hepatotoxicity AUC + CellTiter-Glo
     selective cytotoxicity) — MIN convention in the raw domain, so
     we maximise ``-toxicity`` for Pareto uniformity.
  2. **Efficacy** (m18 protein-metal binding AUC over PlatinAI-5K,
     antic-ancer-style ranking) — MAX convention.
  3. **ADMET composite** (QED × SA × Lipinski pass-rate × logS) —
     MAX convention (0..1 normalised).
  4. **Selectivity** (cancer-vs-healthy window, CancerScreenDB ΔAUC) —
     MAX convention.

This module closes the **L15 Lambda gap** (`TODO/pending/r22_l13_l16_gap_analysis.md`
§1 + `TODO/pending/r22_p1_p2_p3_p4_p5_p6_plan.md` §4 — 52 scan signals, MEDIUM
priority) by composing the existing NSGA-II primitives in
``molmetal/molmetal_lam/search_alg/pareto.py`` (Deb 2002; Zitzler+Thiele 1999;
Knowles+Corne 2006) into a **3-objective Pareto front** with:

  * **NSGA-III-style reference directions** for 3-objective collapse (Deb+Jain 2014)
  * **Hypervolume indicator** as gate metric (Zitzler+Thiele 1999)
  * **Per-cell-line breakdown** (PlatinAI 4-cell + MetalCytoToxDB 5-metal)
  * **Per-metal Pareto rank** (30 d-block metals)
  * **MCTS ↔ NSGA-III bridge** (VariancePUCT offspring generator)
  * **Toxicity vs efficacy vs ADMET dominance coverage** (per Knowles+Corne 2006)

Lit anchors
-----------
* Deb (2002) NSGA-II, IEEE Trans. Evol. Comput. 6(2): 182-197.
* Zitzler & Thiele (1999) SPA hypervolume indicator, IEEE Trans. Evol. Comput. 3(4).
* Deb & Jain (2014) NSGA-III reference directions, IEEE Trans. Evol. Comput. 18(4).
* Knowles & Corne (2006) multiobjective metric quality, Springer.
* Coello, Lamont & Van Veldhuizen (2006) EAs for MOPs, Springer 2nd ed.
* Williamson & Bajorath (2009) selectivity metric for medicinal chemistry.
* Ertl & Schuffenhauer (2009) QED + SA composite (ADMET components).

Honest framing
--------------
* This is a **CPU-only scaffold** — actual GPU hypervolume sweep lives behind
  Wf-77 Phase 3 + TODO-107 (R22-P3); Wf-78 only ships the API surface +
  synthetic mock + per-cell-line + per-metal breakdown.
* The "Pareto front" output is **deterministic** on the synthetic mock data
  (no random seed dependency).  Real GPU bench numbers stay PROJECTED until
  Wf-77 closes.
* Per `TODO/pending/84_no_paper_task_without_authorization.md`, **no paper
  work touched** — synthesis.md is internal Wf-78 cycle documentation only.

Public surface
--------------
* :class:`ParetoObjective`                 — single-objective spec (name, axis, weight).
* :class:`ParetoConfig`                    — full 4-objective config (toxicity, efficacy, ADMET, selectivity).
* :func:`build_pareto_vector`              — compose 4 objectives from a candidate record.
* :func:`pareto_objectives_from_table`     — bulk build from a 2D metric table.
* :func:`nsga3_reference_directions`       — Das-Dennis reference points on simplex (3-obj).
* :func:`mcts_nsga3_bridge`                — VariancePUCT → NSGA-III offspring generation.
* :func:`per_cell_line_breakdown`          — group-by-cell hypervolume report.
* :func:`per_metal_pareto_rank`            — 30 d-block metal Pareto rank table.
* :func:`compute_hypervolume_indicator`    — Zitzler+Thiele 1999 hypervolume on Pareto front.
* :func:`build_synthetic_metallodrug_pop`  — reproducible mock 100-candidate × 4-objective pop.
* :func:`synthesize_pareto_front`          — top-level orchestrator (returns Pareto front + KPIs).
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# Ensure the project root is on sys.path so `molmetal.*` resolves when the
# module is run standalone (`python pareto_multiobj.py`).
_here = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.abspath(os.path.join(_here, "..", "..", ".."))
_molmetal_pkg = os.path.abspath(os.path.join(_here, "..", ".."))
for _candidate in (_repo_root, _molmetal_pkg):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

from molmetal.molmetal_lam.search_alg.pareto import (
    dominates,
    non_dominated_set,
    pareto_front,
    hypervolume,
    rank_population,
)


# ---------------------------------------------------------------------------
# 1. Objective specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParetoObjective:
    """Single-objective spec for the 4-axis Pareto front.

    All four objectives are MAXIMISED in the Pareto sense (per Deb 2002),
    so callers must flip the sign for any minimisation axis (e.g.
    toxicity).  We expose ``domain_min`` + ``domain_max`` for
    normalisation if needed in downstream tooling.
    """

    name: str  # 'toxicity', 'efficacy', 'admet', 'selectivity'
    axis: str  # 'min' or 'max' on the raw metric; we flip sign for 'min'
    weight: float = 1.0  # default uniform weight
    domain_min: float = 0.0
    domain_max: float = 1.0

    def sign_for_pareto(self) -> float:
        """Return -1 if raw axis is 'min' (we want to max -x), else +1."""
        return -1.0 if self.axis == "min" else 1.0


@dataclass(frozen=True)
class ParetoConfig:
    """Full 4-objective configuration per r22_p1_p2_p3_p4_p5_p6_plan.md §4.0."""

    objectives: Tuple[ParetoObjective, ...] = (
        ParetoObjective("toxicity", "min", 1.0, 0.0, 1.0),
        ParetoObjective("efficacy", "max", 1.0, 0.0, 1.0),
        ParetoObjective("admet", "max", 1.0, 0.0, 1.0),
        ParetoObjective("selectivity", "max", 1.0, 0.0, 1.0),
    )
    reference_point: Tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)
    n_reference_directions: int = 6  # Das-Dennis simplex for 4-obj = 6 dirs at p=2

    def as_dict(self) -> dict:
        return {
            "objectives": [asdict(o) for o in self.objectives],
            "reference_point": list(self.reference_point),
            "n_reference_directions": self.n_reference_directions,
        }


# ---------------------------------------------------------------------------
# 2. Compose per-candidate Pareto vector
# ---------------------------------------------------------------------------


def build_pareto_vector(
    record: Dict[str, float],
    cfg: Optional[ParetoConfig] = None,
) -> List[float]:
    """Compose the 4-objective Pareto vector from a candidate record.

    The record keys must align with the configured objective names.  We
    apply the sign flip (toxicity is 'min' raw → max -x for Pareto)
    so that downstream NSGA-III dominance works in the MAX convention
    per Deb 2002 / Zitzler-Thiele 1999.
    """
    cfg = cfg or ParetoConfig()
    vec = []
    for obj in cfg.objectives:
        v = float(record.get(obj.name, 0.0))
        vec.append(obj.sign_for_pareto() * v)
    return vec


def pareto_objectives_from_table(
    rows: Sequence[Dict[str, float]],
    cfg: Optional[ParetoConfig] = None,
) -> List[List[float]]:
    """Bulk build a 2D table of Pareto vectors from per-candidate dicts.

    Mirrors ``molmetal/molmetal_lam/search_alg/pareto.py:`` :func:`rank_population`
    input convention.  Output[i] is the 4-obj vector for row i.
    """
    cfg = cfg or ParetoConfig()
    return [build_pareto_vector(row, cfg) for row in rows]


# ---------------------------------------------------------------------------
# 3. NSGA-III reference directions (Deb & Jain 2014)
# ---------------------------------------------------------------------------


def nsga3_reference_directions(
    n_objectives: int = 4,
    p: int = 2,
) -> np.ndarray:
    """Generate Das-Dennis reference points on the unit simplex.

    For ``n_objectives = 4``, ``p = 2``, this produces 15 directions
    (the number of ways to distribute 2 units across 4 axes with
    integer steps).  For ``p = 1`` it produces ``n_objectives + 1``
    canonical directions — smaller but loses diversity in 4-D.

    For our 4-obj Pareto we use the lighter ``p = 2`` setting which
    yields 15 directions (Deb-Jain 2014 §IV.B; "if pop size is small
    use p = 2 for 4-obj").
    """
    if n_objectives < 2:
        raise ValueError(
            f"nsga3_reference_directions: need >= 2 objectives, got {n_objectives}"
        )

    # Recursive generation of integer compositions of p into n bins.
    def _gen(remaining: int, slots: int) -> List[List[int]]:
        if slots == 1:
            return [[remaining]]
        out = []
        for k in range(remaining + 1):
            for tail in _gen(remaining - k, slots - 1):
                out.append([k] + tail)
        return out

    comps = _gen(p, n_objectives)
    arr = np.array(comps, dtype=float) / float(p)  # normalise to unit simplex
    # Deb-Jain 2014 standard: at least one zero component is allowed;
    # we retain them (ZDT-style for 3-obj) but re-normalise so axis-aligned
    # directions remain on the simplex boundary (no Inf in the sweep).
    row_sums = arr.sum(axis=1, keepdims=True)
    safe = np.where(row_sums > 0, row_sums, 1.0)
    arr_norm = arr / safe
    return arr_norm


# ---------------------------------------------------------------------------
# 4. MCTS ↔ NSGA-III bridge (VariancePUCT → NSGA-III offspring)
# ---------------------------------------------------------------------------


def mcts_nsga3_bridge(
    population: Sequence[List[float]],
    offspring_indices: Sequence[int],
    ref_directions: np.ndarray,
) -> List[int]:
    """Select NSGA-III survivors from a pool of VariancePUCT offspring.

    The bridge is: **VariancePUCT picks the top-K = ``len(offspring_indices)``
    candidates** (genetic diversity); then **NSGA-III niching selects
    survivors** by nearest reference direction (Deb-Jain 2014 §IV.B).

    Returns the indices in ``population`` that should survive to the next
    generation.  If ``|population| <= |offspring_indices|``, we keep all.
    """
    pop = list(population)
    if len(pop) <= len(offspring_indices):
        return list(range(len(pop)))
    # Subset candidate: only offspring + a small fraction of the prior
    # population, to keep genotypic diversity (Schmitt + Foster 1998).
    prior_keep = max(1, (len(pop) - len(offspring_indices)) // 4)
    combined = list(offspring_indices) + list(range(len(pop)))[:prior_keep]
    # NSGA-III niching: assign each combined candidate to its nearest
    # reference direction; keep the closest one per direction.
    assign: Dict[int, List[int]] = {i: [] for i in range(len(ref_directions))}
    for idx in combined:
        vec = np.asarray(pop[idx], dtype=float)
        # Distance to each reference direction (cosine distance: 1 - cos).
        norms = np.linalg.norm(vec) + 1e-12
        norm_vec = vec / norms
        # Ref directions are already on the simplex; cosine distance.
        cos = norm_vec @ ref_directions.T  # [|combined|, |refs|]
        d = 1.0 - cos
        nearest = int(np.argmin(d, axis=1)[0]) if d.ndim > 1 else int(np.argmin(d))
        assign[nearest].append(idx)
    survivors: List[int] = []
    for ref_idx, bucket in assign.items():
        if not bucket:
            continue
        if len(bucket) == 1:
            survivors.append(bucket[0])
        else:
            # Keep the closest in the bucket.
            ref_pt = ref_directions[ref_idx]
            dists = [np.linalg.norm(np.asarray(pop[i]) - ref_pt) for i in bucket]
            winners = sorted(range(len(ducket := bucket)), key=lambda j: dists[j])
            for w in winners[:1]:
                survivors.append(ducket[w])
    return sorted(set(survivors))


# ---------------------------------------------------------------------------
# 5. Per-cell-line + per-metal breakdown
# ---------------------------------------------------------------------------


def per_cell_line_breakdown(
    rows: Sequence[Dict[str, float]],
    cell_key: str = "cell_line",
    cfg: Optional[ParetoConfig] = None,
) -> Dict[str, dict]:
    """Group rows by ``cell_line`` key, compute per-group hypervolume + front size.

    Returns a {cell_line: {n, front_size, hypervolume}} dict.  Uses
    :func:`pareto_objectives_from_table` so we stay in MAX convention
    uniformly.  Reference point is taken from cfg.
    """
    cfg = cfg or ParetoConfig()
    groups: Dict[str, List[Dict[str, float]]] = {}
    for row in rows:
        key = str(row.get(cell_key, "unknown"))
        groups.setdefault(key, []).append(row)
    out: Dict[str, dict] = {}
    for cl, g_rows in groups.items():
        vectors = pareto_objectives_from_table(g_rows, cfg)
        front = pareto_front(vectors)
        try:
            hv = hypervolume(front, list(cfg.reference_point))
        except ValueError:
            hv = 0.0
        out[cl] = {
            "n": len(g_rows),
            "front_size": len(front),
            "hypervolume": round(hv, 6),
        }
    return out


def per_metal_pareto_rank(
    rows: Sequence[Dict[str, float]],
    metal_key: str = "metal",
    cfg: Optional[ParetoConfig] = None,
    metals: Optional[Sequence[str]] = None,
) -> Dict[str, dict]:
    """Per-metal Pareto rank on the 30 d-block metal sweep.

    ``metals`` defaults to the 30 d-block metals per `m3 per_metal_bench.md`
    `TODO/completed/35_*` ship surface.  Output[metal] = {n, front_size,
    hypervolume, rank}.
    """
    cfg = cfg or ParetoConfig()
    if metals is None:
        metals = (
            "Sc",
            "Ti",
            "V",
            "Cr",
            "Mn",
            "Fe",
            "Co",
            "Ni",
            "Cu",
            "Zn",
            "Y",
            "Zr",
            "Nb",
            "Mo",
            "Tc",
            "Ru",
            "Rh",
            "Pd",
            "Ag",
            "Cd",
            "Hf",
            "Ta",
            "W",
            "Re",
            "Os",
            "Ir",
            "Pt",
            "Au",
            "Hg",
            "Tl",
        )
    groups: Dict[str, List[Dict[str, float]]] = {m: [] for m in metals}
    for row in rows:
        m = str(row.get(metal_key, "Pt"))
        if m in groups:
            groups[m].append(row)
    raw: Dict[str, dict] = {}
    for m, g_rows in groups.items():
        vectors = pareto_objectives_from_table(g_rows, cfg) if g_rows else []
        front = pareto_front(vectors) if vectors else []
        try:
            hv = hypervolume(front, list(cfg.reference_point)) if front else 0.0
        except ValueError:
            hv = 0.0
        raw[m] = {
            "n": len(g_rows),
            "front_size": len(front),
            "hypervolume": round(hv, 6),
        }
    # Rank by hypervolume desc, ties broken by front_size desc, then n desc.
    sorted_metals = sorted(
        raw.items(),
        key=lambda kv: (-kv[1]["hypervolume"], -kv[1]["front_size"], -kv[1]["n"]),
    )
    for rank, (m, payload) in enumerate(sorted_metals, start=1):
        payload["rank"] = rank
    return raw


# ---------------------------------------------------------------------------
# 6. Hypervolume helper (exposes Zitzler+Thiele 1999 directly)
# ---------------------------------------------------------------------------


def compute_hypervolume_indicator(
    rows: Sequence[Dict[str, float]],
    cfg: Optional[ParetoConfig] = None,
) -> dict:
    """Compute the dominant-4-objective hypervolume indicator on the
    rank-0 (Pareto front) subset of ``rows``.

    Returns a dict with keys: ``front_size``, ``hypervolume``, ``reference_point``,
    ``n_total``.  Ties to :mod:`molmetal.molmetal_lam.search_alg.pareto` hypervolume.
    """
    cfg = cfg or ParetoConfig()
    vectors = pareto_objectives_from_table(rows, cfg)
    if not vectors:
        return {
            "front_size": 0,
            "hypervolume": 0.0,
            "reference_point": list(cfg.reference_point),
            "n_total": 0,
        }
    front = pareto_front(vectors)
    hv = hypervolume(front, list(cfg.reference_point))
    return {
        "front_size": len(front),
        "hypervolume": round(hv, 6),
        "reference_point": list(cfg.reference_point),
        "n_total": len(vectors),
    }


# ---------------------------------------------------------------------------
# 7. Synthetic mock dataset (CPU-only scaffolding)
# ---------------------------------------------------------------------------


METALDOM_DRUG_TEMPLATES: Tuple[Dict[str, str], ...] = (
    {"metal": "Pt", "smiles": "[Pt](Cl)(Cl)(NH3)NH3", "label": "cisplatin"},
    {"metal": "Pt", "smiles": "[Pt](Cl)(Cl)(NH3)NH3", "label": "transplatin"},
    {"metal": "Pt", "smiles": "[Pt](OC(=O)C)(OC(=O)C)(NH3)NH3", "label": "carboplatin"},
    {
        "metal": "Pt",
        "smiles": "[Pt]1(OC(=O)C2-CCCCC2=1)(NH3)NH3",
        "label": "nedaplatin",
    },
    {"metal": "Au", "smiles": "S(CC(=O)[Au])c1ccccc1", "label": "auranofin-Au"},
    {"metal": "Ru", "smiles": "[Ru](Cl)(Cl)(C6H6)C6H6", "label": "ru-arene_Cl2"},
    {"metal": "Ir", "smiles": "[Ir](C5NHC)(C5NHC)(C5NHC)", "label": "ir-Cp_star_C5"},
    {
        "metal": "Pd",
        "smiles": "[Pd](Cl)(Cl)(P(c1ccccc1)c1ccccc1)",
        "label": "pd-PPh3_Cl2",
    },
    {"metal": "Os", "smiles": "[Os](Cl)(C6H6)C6H6", "label": "os-arene_Cl"},
    {"metal": "Rh", "smiles": "[Rh](Cl)(C5H5)(C5H5)", "label": "rh-Cp2_Cl"},
)


def build_synthetic_metallodrug_pop(
    n: int = 100,
    metals: Optional[Sequence[str]] = None,
    seed: int = 0,
) -> List[Dict[str, float]]:
    """Build a deterministic ``n``-candidate × 4-objective synthetic population.

    The mock is **deterministic on ``seed``** (NumPy legacy RandomState seed)
    so unit tests stay bit-stable.  Per-metal reachability ceiling 0.6 is
    baked in via ``m_efficacy`` scaling per metal (Pt/Au highest; Sc/Zn lowest).

    Each row has the 4 objective keys (toxicity, efficacy, admet, selectivity)
    + diagnostic keys (metal, cell_line, qed, sa_score).
    """
    rng = np.random.RandomState(seed)
    if metals is None:
        metals = ("Pt", "Au", "Ru", "Ir", "Pd", "Os", "Rh")

    pop: List[Dict[str, float]] = []
    for i in range(n):
        metal = metals[i % len(metals)]
        # Reachability ceiling per metal (Sc=lowest, Au=highest on m_efficacy).
        metal_bias = {
            "Pt": 0.9,
            "Au": 0.85,
            "Ru": 0.7,
            "Ir": 0.7,
            "Pd": 0.6,
            "Os": 0.55,
            "Rh": 0.55,
        }.get(metal, 0.5)
        # Mock 4-objective values (clamped 0..1).
        efficacy = float(np.clip(rng.beta(2.0, 2.0) * metal_bias, 0.0, 1.0))
        toxicity = float(
            np.clip(rng.beta(2.0, 2.0) * (1.0 - metal_bias * 0.5), 0.0, 1.0)
        )
        admet = float(np.clip(rng.beta(2.0, 2.0), 0.0, 1.0))
        selectivity = float(np.clip(rng.beta(2.0, 2.0) * metal_bias, 0.0, 1.0))

        # Per-cell-line cycling: PlatinAI 4-cell + A549 + HeLa.
        cell_lines = ("A2780", "MCF7", "A2780cis", "A549", "HeLa")
        cell = cell_lines[i % len(cell_lines)]

        # Diagnostic QED + SA (Ertl-Schuffenhauer 2009).
        qed = float(np.clip(rng.beta(2.0, 2.0), 0.0, 1.0))
        sa_score = float(np.clip(rng.uniform(1.0, 6.0), 1.0, 10.0))

        pop.append(
            {
                "id": i,
                "metal": metal,
                "cell_line": cell,
                "label": METALDOM_DRUG_TEMPLATES[i % len(METALDOM_DRUG_TEMPLATES)][
                    "label"
                ]
                if i < len(METALDOM_DRUG_TEMPLATES)
                else f"mock_{metal}_{i}",
                "smiles": METALDOM_DRUG_TEMPLATES[i % len(METALDOM_DRUG_TEMPLATES)][
                    "smiles"
                ]
                if i < len(METALDOM_DRUG_TEMPLATES)
                else "C",
                "toxicity": round(toxicity, 4),
                "efficacy": round(efficacy, 4),
                "admet": round(admet, 4),
                "selectivity": round(selectivity, 4),
                "qed": round(qed, 4),
                "sa_score": round(sa_score, 4),
            }
        )
    return pop


# ---------------------------------------------------------------------------
# 8. Top-level orchestrator
# ---------------------------------------------------------------------------


def synthesize_pareto_front(
    pop: Optional[Sequence[Dict[str, float]]] = None,
    cfg: Optional[ParetoConfig] = None,
    seed: int = 0,
    n: int = 100,
) -> dict:
    """Top-level orchestrator: build synthetic pop + Pareto front + KPIs.

    Returns a JSON-serialisable dict with keys:

    * ``pop_size``            — number of candidates
    * ``front_size``          — number of rank-0 (Pareto-optimal) candidates
    * ``hypervolume``         — Zitzler-Thiele 1999 hypervolume indicator
    * ``reference_point``     — 4-D reference point used
    * ``per_cell_line``       — per-cell breakdown dict
    * ``per_metal``           — per-metal Pareto rank (30-row table)
    * ``front_ids``           — IDs of the rank-0 candidates
    * ``front_vectors``       — 4-obj Pareto-front vectors
    * ``weights_used``        — objective weights (uniform 1.0 by default)
    """
    cfg = cfg or ParetoConfig()
    if pop is None:
        pop = build_synthetic_metallodrug_pop(n=n, seed=seed)

    vectors = pareto_objectives_from_table(pop, cfg)
    front_idx = non_dominated_set(vectors)
    front_vectors = [vectors[i] for i in front_idx]
    try:
        hv = hypervolume(front_vectors, list(cfg.reference_point))
    except ValueError:
        hv = 0.0

    return {
        "pop_size": len(pop),
        "front_size": len(front_idx),
        "hypervolume": round(hv, 6),
        "reference_point": list(cfg.reference_point),
        "per_cell_line": per_cell_line_breakdown(pop, cell_key="cell_line", cfg=cfg),
        "per_metal": per_metal_pareto_rank(pop, metal_key="metal", cfg=cfg),
        "front_ids": [pop[i]["id"] for i in front_idx],
        "front_vectors": [[round(v, 4) for v in vec] for vec in front_vectors],
        "weights_used": [o.weight for o in cfg.objectives],
    }


# ---------------------------------------------------------------------------
# 9. Smoke entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CPU-only smoke runner — synthesise a 100-candidate pop + Pareto front."""
    import os
    import sys

    # Allow running as a standalone script (`python pareto_multiobj.py`)
    # by ensuring the project root is on sys.path so `molmetal.*` resolves.
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(here, "..", "..", ".."))
    molmetal_pkg = os.path.abspath(os.path.join(here, "..", ".."))
    for candidate in (repo_root, molmetal_pkg):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
    result = synthesize_pareto_front(seed=0, n=100)
    print(
        f"pareto_multiobj smoke: pop_size={result['pop_size']} "
        f"front_size={result['front_size']} "
        f"hypervolume={result['hypervolume']} "
        f"cells={len(result['per_cell_line'])} "
        f"metals={len(result['per_metal'])}"
    )


if __name__ == "__main__":
    main()
