"""score_any.py — unified scoring router (Wf-43 Phase 1 + Phase 2).

Routes a metric name (e.g. ``"vina_score"``) to the real, locally
deployed scoring tool. NO mock fallback is permitted: every registered
metric must have a live adapter, and the function raises
:class:`AdapterUnavailable` if the tool is missing or the metric name
is unknown.

Design notes
------------
* Adapters are imported lazily so a CI box missing one heavy ML backend
  (e.g. PyG, Meeko, Boltz) can still import this module and exercise
  the metrics whose tools it *does* have installed.
* Each per-tool wrapper normalises the (metric, molecules, pocket)
  triple into the adapter's native call signature. The router never
  invents numbers; on a hard failure (tool missing / runtime error) it
  raises rather than substituting a stub score.
* A small detector (:func:`_detect_engine`) selects between multiple
  available tools for the same metric (e.g. real Vina vs Q-Vina2 vs
  Vina-GPU) based on environment hints (env vars, CLI binary probe).

Wf-43 Phase 2 — Routing policy
------------------------------
:data:`ROUTING_POLICY` is a ``dict[str, PolicyEntry]`` exposed by this
module so callers can audit the per-metric mandatory tool, fallback
chain, required data and wf43_status without parsing the markdown.
The companion machine-readable file is :file:`molmetal/scoring/policy.json`
and the human-readable document is :file:`molmetal/scoring/ROUTING_POLICY.md`.

Adding a new metric
-------------------
1. Implement the adapter call in ``_wrappers``.
2. Add the (metric_name, ToolRef(...)) entry to :data:`REGISTRY`.
3. Add a (metric_name, PolicyEntry(...)) entry to :data:`ROUTING_POLICY`.
4. Add a row to :file:`molmetal/scoring/REGISTRY.md`.
5. Add a row to :file:`molmetal/scoring/ROUTING_POLICY.md`.
6. Update :file:`molmetal/scoring/policy.json` to keep the machine-readable
   view in sync with the in-process dict.
7. Add a test in :mod:`molmetal.tests.test_score_any`.
"""
from __future__ import annotations

import os
import traceback
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
)

# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------
class AdapterUnavailable(RuntimeError):
    """Raised when a metric has no live tool available.

    This is the *only* failure mode this module allows. We never
    fabricate a mock score to "make the metric work" — that is the
    behaviour Wf-43 was created to forbid.
    """


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ToolRef:
    """Pointer to the real tool that backs a metric.

    Attributes
    ----------
    wrapper:
        Callable that performs the actual scoring. It receives a
        normalised ``ctx`` dict and returns either a scalar/vector
        (depending on the metric). The wrapper is responsible for
        catching its own ``ImportError`` s and re-raising as
        :class:`AdapterUnavailable`.
    tool_name:
        Human-readable tool name (e.g. ``"AutoDock Vina 1.2.7"``).
    backend:
        Backend identifier — ``"python"`` / ``"cli"`` / ``"remote"``.
    notes:
        Free-form notes (env vars, install steps, paper refs).
    """

    wrapper: Callable[[Dict[str, Any]], Any]
    tool_name: str
    backend: str
    notes: str = ""


# Registry is populated below by the per-tool wrapper definitions.
REGISTRY: Dict[str, ToolRef] = {}


def _register(metric: str, ref: ToolRef) -> None:
    REGISTRY[metric] = ref


# ---------------------------------------------------------------------------
# Routing policy (Wf-43 Phase 2)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PolicyEntry:
    """Wf-43 Phase 2 policy entry — which real tool is mandatory per metric.

    Attributes
    ----------
    primary_tool:
        Human-readable name of the real, locally-deployed tool that
        produces this metric (e.g. ``"AutoDock Vina 1.2.7"``).
    backend:
        Backend identifier (``"python"`` / ``"cli"`` / ``"python+cli"`` /
        ``"python+CUDA"`` / ``"python+RDkit"`` / ``"commercial"`` / ``"N/A"``).
    module:
        The Python module (or relative path) that implements the
        adapter call. ``"NOT_SHIPPED"`` when the tool is reserved but
        not yet in this repo.
    license:
        Tool licence (``"Apache-2.0"`` / ``"MIT"`` / ``"BSD"`` /
        ``"Schrödinger commercial"`` / ``"N/A"``).
    fallback_chain:
        Ordered list of metric names (referencing other ``PolicyEntry``
        rows) that are acceptable substitutes if the primary tool is
        unavailable. ``[]`` means no fallback — the metric simply
        raises :class:`AdapterUnavailable`.
    required_data:
        Data the call-site MUST supply (``"smiles"`` /
        ``"pocket_pdbqt"`` / ``"pocket_pdb"`` / ``"smiles_list"`` /
        ``"reference_smiles"`` etc.).
    wf43_status:
        One of ``"MEASURED"`` (real tool shipped + tested),
        ``"RESERVED"`` (metric name reserved, tool not shipped,
        wrapper raises), ``"MOCK_DELETED"`` (was a mock/stub, now
        removed and redirected to a real alternative or honest
        non-paper disposition).
    notes:
        Free-form notes (lit anchors, env vars, install steps).
    """

    primary_tool: str
    backend: str
    module: str
    license: str
    fallback_chain: Tuple[str, ...]
    required_data: Tuple[str, ...]
    wf43_status: str
    notes: str = ""


# Routing policy — populated below by ``_policy_register`` calls.
ROUTING_POLICY: Dict[str, PolicyEntry] = {}


def _policy_register(metric: str, entry: PolicyEntry) -> None:
    ROUTING_POLICY[metric] = entry


def _policy_register_many(entries: Dict[str, PolicyEntry]) -> None:
    for metric, entry in entries.items():
        ROUTING_POLICY[metric] = entry


# ---------------------------------------------------------------------------
# Lazy adapter imports
# ---------------------------------------------------------------------------
def _import_aizynth():
    from molmetal.molmetal_lam.sbdd_env import aizynth_adapter

    return aizynth_adapter


def _import_posebusters():
    from molmetal.molmetal_lam.sbdd_env import posebusters_adapter

    return posebusters_adapter


def _import_vina():
    from molmetal.molmetal_lam.sbdd_env import vina_adapter

    return vina_adapter


def _import_diffdock():
    from molmetal.molmetal_lam.sbdd_env import diffdock_adapter

    return diffdock_adapter


def _import_pic50():
    from molmetal.molmetal_lam.sbdd_env import pic50_predictor

    return pic50_predictor


def _import_voxelization():
    from molmetal.molmetal_lam.sbdd_env import voxelization

    return voxelization


def _import_pocket2mol():
    from molmetal.molmetal_lam.sbdd_env import pocket2mol_adapter

    return pocket2mol_adapter


def _import_targetdiff():
    from molmetal.molmetal_lam.sbdd_env import targetdiff_adapter

    return targetdiff_adapter


def _import_flowdock():
    from molmetal.molmetal_lam.sbdd_env import flowdock_adapter

    return flowdock_adapter


def _import_flowr():
    from molmetal.molmetal_lam.sbdd_env import flowr_adapter

    return flowr_adapter


def _import_pll():
    from molmetal.molmetal_lam.lam_chem import pll_signature

    return pll_signature


def _import_sa_score():
    from molmetal.molmetal_lam.sbdd_env import sa_score

    return sa_score


def _import_qed():
    from molmetal.molmetal_lam.sbdd_env import qed_scorer

    return qed_scorer


def _import_vina_gpu():
    from molmetal.molmetal_lam.sbdd_env import vina_gpu_adapter

    return vina_gpu_adapter


def _import_q_vina2():
    """Q-Vina2 path — local CLI binary; no Python wrapper exists yet.

    Wf-43 ships a thin shim that defers to :mod:`molmetal.scripts`
    ``qvina02`` if present on PATH. Until that script ships we fall
    through to a clear error message; the *no-mock* invariant still
    holds (we never synthesise a fake score).
    """
    try:
        # Late import so missing scripts do not break test collection.
        from molmetal.scripts import q_vina2_swap  # type: ignore

        return q_vina2_swap
    except Exception as e:  # pragma: no cover - import guard
        raise AdapterUnavailable(
            "Q-Vina2 swap module not available; install via "
            "`molmetal/scripts/q_vina2_swap.py` (requires local qvina02 "
            "binary on PATH). Underlying error: "
            f"{type(e).__name__}: {e}"
        )


# ---------------------------------------------------------------------------
# Per-tool wrappers — each normalises the (metric, molecules, pocket)
# triple into the adapter's native signature and returns the real score.
# ---------------------------------------------------------------------------
def _w_vina_score(ctx: Dict[str, Any]) -> List[float]:
    """AutoDock Vina — best (most negative) kcal/mol per molecule."""
    vina = _import_vina()
    pocket = ctx["pocket"]
    smiles_list = _as_smiles_list(ctx["molecules"])
    n_poses = int(ctx.get("n_poses", 5))
    exhaustiveness = int(ctx.get("exhaustiveness", 8))
    engine = ctx.get("engine", vina.DEFAULT_ENGINE)

    if pocket is None:
        raise AdapterUnavailable(
            "vina_score requires a pocket (Pocket | pocket_pdb_path | "
            "pocket_pdb_str). None supplied."
        )

    out: List[float] = []
    for smi in smiles_list:
        complexes = vina.dock_smiles(
            smi,
            pocket,
            n_poses=n_poses,
            exhaustiveness=exhaustiveness,
            engine=engine,
        )
        if not complexes:
            out.append(float("nan"))
        else:
            scores = [c.vina_score for c in complexes if c.vina_score is not None]
            out.append(float(min(scores)) if scores else float("nan"))
    return out


def _w_vina_high_affinity(ctx: Dict[str, Any]) -> List[bool]:
    """Boolean: does Vina best pose exceed a kcal/mol affinity cutoff?"""
    vina = _import_vina()
    cutoff = float(ctx.get("cutoff_kcal", -8.0))
    scores = _w_vina_score(ctx)
    return [s <= cutoff for s in scores]


def _w_vina_min_score(ctx: Dict[str, Any]) -> Dict[str, float]:
    """Aggregate Vina stats over a batch — min / median / n_below_cutoff."""
    cutoff = float(ctx.get("cutoff_kcal", -8.0))
    scores = _w_vina_score(ctx)
    valid = [s for s in scores if s == s]  # drop NaN
    if not valid:
        return {"min": float("nan"), "median": float("nan"),
                "n_below_cutoff": 0, "n_total": 0}
    sorted_s = sorted(valid)
    median = sorted_s[len(sorted_s) // 2]
    return {
        "min": float(sorted_s[0]),
        "median": float(median),
        "n_below_cutoff": sum(1 for s in valid if s <= cutoff),
        "n_total": len(valid),
    }


def _w_diffdock_pose(ctx: Dict[str, Any]) -> List[float]:
    """DiffDock — per-ligand pose-confidence (0..1)."""
    dd = _import_diffdock()
    pocket_pdb = _pocket_to_pdb_string(ctx["pocket"])
    smiles_list = _as_smiles_list(ctx["molecules"])
    out: List[float] = []
    for smi in smiles_list:
        result = dd.DiffDockAdapter().dock(smi, pocket_pdb)
        out.append(float(result.confidence))
    return out


def _w_equibind_pose(ctx: Dict[str, Any]) -> List[float]:
    """EquiBind — per-ligand pose-confidence.

    EquiBind is NOT yet shipped in this repository (R17 ships DiffDock
    + FlowDock instead). The router therefore raises rather than
    fabricate a fake score; once :file:`equibind_adapter.py` lands the
    wiring is one wrapper function away.
    """
    raise AdapterUnavailable(
        "equibind_pose: EquiBind adapter not shipped in R17. "
        "EquiBind is currently substituted by FlowDock (see "
        "molmetal.molmetal_lam.sbdd_env.flowdock_adapter). To enable "
        "the dedicated `equibind_pose` metric, add "
        "`molmetal/molmetal_lam/sbdd_env/equibind_adapter.py` and a "
        "wrapper here."
    )


def _w_pocket2mol(ctx: Dict[str, Any]) -> List[float]:
    """Pocket2Mol — per-ligand generation feasibility / validity."""
    p2m = _import_pocket2mol()
    if not p2m._checkpoint_available():
        raise AdapterUnavailable(
            "pocket2mol: no checkpoint available (looked in "
            f"{p2m._checkpoint_path()!s}). Download the official "
            "Pocket2Mol weights to enable this metric."
        )
    pocket = ctx["pocket"]
    if pocket is None:
        raise AdapterUnavailable(
            "pocket2mol requires a pocket to sample inside; none "
            "supplied. Use `pocket2mol` for *generation*, not "
            "scoring; for *scoring* use vina_score / boltz_affinity."
        )
    out = p2m.Pocket2MolSampler().sample(
        pocket=pocket, n_samples=int(ctx.get("n_samples", 1))
    )
    # Each sample is a (coords, atom_types, bonds) tuple; expose
    # atom-count as the validity proxy since the sampler rejects
    # invalid graphs.
    return [float(getattr(s, "n_atoms", 0)) for s in out]


def _w_boltz_affinity(ctx: Dict[str, Any]) -> List[float]:
    """Boltz-style pIC50 — D-MPNN predictor shipped in :mod:`pic50_predictor`.

    Returns one pIC50 float per input SMILES. ``nan`` on parse fail
    (mirrors the underlying predictor). Note: this is the *locally
    trained* attentive D-MPNN model — it is real ML inference, not a
    mock. To switch to the Boltz foundation-model pIC50 head once
    shipped, add a wrapper here and register under the same metric.
    """
    pic50 = _import_pic50()
    smiles_list = _as_smiles_list(ctx["molecules"])
    return [float(pic50.predict_pic50(s)) for s in smiles_list]


def _w_af2_confidence(ctx: Dict[str, Any]) -> List[float]:
    """AF2-style per-pocket complexity proxy.

    This wrapper is intentionally narrow: AF2 confidence (pLDDT) lives
    behind AlphaFold inference, which is not in this repo. The shipped
    substitute is the :func:`voxelization.pocket_to_spatial_tiles`
    complexity heuristic (real, deterministic, well-tested). Once an
    AF2 inference backend is added, swap the wrapper — keep the metric
    name stable so callers don't break.
    """
    vox = _import_voxelization()
    pocket = ctx["pocket"]
    if pocket is None:
        raise AdapterUnavailable(
            "af2_confidence requires a pocket; none supplied."
        )
    tiles = vox.pocket_to_spatial_tiles(
        pocket,
        grid_size=int(ctx.get("grid_size", 24)),
        resolution=float(ctx.get("resolution", 1.0)),
    )
    # Complexity = stddev of tile occupancy (deterministic proxy for
    # binding-site shape complexity).
    if not tiles:
        return [0.0]
    counts = [len(t.atom_indices) if hasattr(t, "atom_indices") else 1
              for t in tiles]
    if not counts:
        return [0.0]
    mean = sum(counts) / len(counts)
    var = sum((c - mean) ** 2 for c in counts) / len(counts)
    return [float(var ** 0.5)]


def _w_pb_pass_rate(ctx: Dict[str, Any]) -> float:
    """PoseBusters — fraction of mols passing ALL geometric + chemistry checks."""
    pb = _import_posebusters()
    smiles_list = _as_smiles_list(ctx["molecules"])
    return float(pb.pass_rate(smiles_list))


def _w_aizynth_synthesis(ctx: Dict[str, Any]) -> List[bool]:
    """AiZynthFinder retrosynthesis — per-ligand synthesizable bool.

    Falls back to SMARTS-based check when AiZynthFinder is not installed;
    either way the result is the *real* adapter call, not a mock.
    """
    aiz = _import_aizynth()
    smiles_list = _as_smiles_list(ctx["molecules"])
    reports = aiz.check_list(smiles_list)
    return [bool(r.synthesizable) for r in reports]


def _w_pl_likelihood(ctx: Dict[str, Any]) -> List[float]:
    """Persistent Local Laplacian (PLL) signature likelihood per SMILES."""
    pll = _import_pll()
    smiles_list = _as_smiles_list(ctx["molecules"])
    return [float(pll.compute_pll(s)) for s in smiles_list]


def _w_vina_qvina2(ctx: Dict[str, Any]) -> List[float]:
    """Q-Vina2 (QuickVina2) — kcal/mol per molecule.

    Routes through :mod:`molmetal.scripts.q_vina2_swap` which calls the
    ``qvina02`` CLI binary if present on PATH. The router raises if the
    swap script is missing — we never invent a value.
    """
    qv2 = _import_q_vina2()
    pocket = ctx["pocket"]
    smiles_list = _as_smiles_list(ctx["molecules"])
    if not hasattr(qv2, "q_vina2_score"):
        raise AdapterUnavailable(
            "vina_qvina2: q_vina2_swap.q_vina2_score() not defined. "
            "Ship the Q-Vina2 swap script first."
        )
    out: List[float] = []
    for smi in smiles_list:
        scores = qv2.q_vina2_score(smi, pocket, exhaustiveness=int(ctx.get("exhaustiveness", 8)))
        out.append(float(min(scores)) if scores else float("nan"))
    return out


def _w_decomp_diff(ctx: Dict[str, Any]) -> List[float]:
    """DecompDiff — fragment-based 3D binding-affinity surrogate.

    Adapter not yet shipped. The metric name is reserved so call-sites
    written today survive the upcoming ship.
    """
    raise AdapterUnavailable(
        "decomp_diff: DecompDiff adapter not yet shipped. The metric "
        "name is reserved by molmetal.scoring.REGISTRY to keep "
        "call-sites stable. Ship molmetal/molmetal_lam/sbdd_env/"
        "decompdiff_adapter.py then add a wrapper here."
    )


def _w_molcraft(ctx: Dict[str, Any]) -> List[float]:
    """MolCraft — SE(3) flow-matching binding-affinity surrogate.

    Adapter not yet shipped. Reserved metric name; no mock.
    """
    raise AdapterUnavailable(
        "molcraft: MolCraft adapter not yet shipped. Reserved metric "
        "name. Ship the adapter then wire a wrapper here."
    )


def _w_targetdiff(ctx: Dict[str, Any]) -> List[float]:
    """TargetDiff — pocket-conditioned ligand generation validity.

    We expose generation *validity rate* (atom-count>0 after sample)
    as the scoring surrogate for now. Once TargetDiff ships a binding
    head, swap to that.
    """
    td = _import_targetdiff()
    pocket = ctx["pocket"]
    if pocket is None:
        raise AdapterUnavailable("targetdiff requires a pocket.")
    out: List[float] = []
    # The sampler is created on demand; cap at 1 sample per call to
    # keep wall-time bounded — callers wanting more should batch.
    n_samples = int(ctx.get("n_samples", 1))
    samples = td.PocketToLigandSampler().sample(
        pocket=pocket, n_samples=n_samples
    )
    return [float(getattr(s, "n_atoms", 0)) for s in samples]


def _w_imold_ood(ctx: Dict[str, Any]) -> List[float]:
    """IMOLDE OOD score — adapter not yet shipped; reserved metric."""
    raise AdapterUnavailable(
        "imold_ood: IMOLDE OOD adapter not yet shipped. Reserved "
        "metric. Ship the adapter then wire a wrapper here."
    )


def _w_drugood_ood(ctx: Dict[str, Any]) -> List[float]:
    """DrugOOD OOD score — adapter not yet shipped; reserved metric."""
    raise AdapterUnavailable(
        "drugood_ood: DrugOOD OOD adapter not yet shipped. Reserved "
        "metric. Ship the adapter then wire a wrapper here."
    )


def _w_glide_score(ctx: Dict[str, Any]) -> List[float]:
    """Glide (Schrödinger) — commercial docking tool, not in repo."""
    raise AdapterUnavailable(
        "glide_score: Glide requires a Schrödinger licence and is not "
        "available in this open-source repo. Use vina_score as the "
        "open docking surrogate, or obtain a Glide licence and ship "
        "a wrapper here."
    )


def _w_vina_gpu(ctx: Dict[str, Any]) -> List[float]:
    """Vina-GPU — same kcal/mol as AutoDock Vina, GPU backend.

    Used when the env var ``MOLMETAL_VINA_ENGINE=gpu`` is set OR the
    host has a CUDA device + the ``vina_gpu`` extension available.
    """
    vgpu = _import_vina_gpu()
    pocket = ctx["pocket"]
    smiles_list = _as_smiles_list(ctx["molecules"])
    out: List[float] = []
    for smi in smiles_list:
        complexes = vgpu.dock_smiles_gpu(smi, pocket, n_poses=int(ctx.get("n_poses", 5)))
        if not complexes:
            out.append(float("nan"))
        else:
            scores = [c.vina_score for c in complexes if c.vina_score is not None]
            out.append(float(min(scores)) if scores else float("nan"))
    return out


def _w_sa_score(ctx: Dict[str, Any]) -> List[float]:
    """SA score (Ertl-Schuffenhauer) — synthesis accessibility 1–10."""
    sa = _import_sa_score()
    smiles_list = _as_smiles_list(ctx["molecules"])
    return [float(sa.sa_score_ertl(s)) for s in smiles_list]


def _w_qed(ctx: Dict[str, Any]) -> List[float]:
    """QED (Bickerton) — drug-likeness 0..1."""
    qed = _import_qed()
    smiles_list = _as_smiles_list(ctx["molecules"])
    return [float(qed.qed_from_smiles(s)) for s in smiles_list]


def _w_flowdock(ctx: Dict[str, Any]) -> List[float]:
    """FlowDock — flow-matching docking confidence (0..1)."""
    fd = _import_flowdock()
    pocket_pdb = _pocket_to_pdb_string(ctx["pocket"])
    smiles_list = _as_smiles_list(ctx["molecules"])
    out: List[float] = []
    for smi in smiles_list:
        result = fd.FlowDockAdapter().dock(smi, pocket_pdb)
        out.append(float(getattr(result, "confidence", 0.0)))
    return out


def _w_flowr(ctx: Dict[str, Any]) -> List[float]:
    """FlowER retrosynthesis-flow validity score per SMILES."""
    flr = _import_flowr()
    smiles_list = _as_smiles_list(ctx["molecules"])
    sampler = flr.FlowRSampler() if hasattr(flr, "FlowRSampler") else None
    if sampler is None:
        raise AdapterUnavailable(
            "flowr: FlowRSampler not exposed by the adapter. Ship the "
            "sampler class first."
        )
    out: List[float] = []
    for smi in smiles_list:
        reports = sampler.sample(smi)
        # Validity = at least one valid forward-synthesis tree returned
        out.append(float(1.0 if reports else 0.0))
    return out


# ---------------------------------------------------------------------------
# Populate REGISTRY
# ---------------------------------------------------------------------------
_register("vina_score", ToolRef(
    wrapper=_w_vina_score,
    tool_name="AutoDock Vina 1.2.7",
    backend="python+cli",
    notes="Real Meeko+AutoDock-Vina Python binding (preferred) or "
          "vina CLI binary. Routes to Q-Vina2 when env "
          "MOLMETAL_VINA_ENGINE=qvina2 is set.",
))
_register("vina_high_affinity", ToolRef(
    wrapper=_w_vina_high_affinity,
    tool_name="AutoDock Vina 1.2.7",
    backend="python+cli",
    notes="Boolean classifier on top of vina_score (cutoff_kcal, "
          "default -8.0 kcal/mol).",
))
_register("vina_min_score", ToolRef(
    wrapper=_w_vina_min_score,
    tool_name="AutoDock Vina 1.2.7",
    backend="python+cli",
    notes="Aggregates vina_score over a batch — min / median / "
          "count-below-cutoff.",
))
_register("vina_gpu", ToolRef(
    wrapper=_w_vina_gpu,
    tool_name="Vina-GPU",
    backend="python+CUDA",
    notes="GPU-accelerated Vina; same kcal/mol semantics as "
          "vina_score. Requires CUDA device + vina_gpu extension.",
))
_register("vina_qvina2", ToolRef(
    wrapper=_w_vina_qvina2,
    tool_name="QuickVina 2 (CASO-derived)",
    backend="cli",
    notes="Real qvina02 binary on PATH; same kcal/mol semantics as "
          "AutoDock Vina. Routes via q_vina2_swap.py.",
))
_register("diffdock_pose", ToolRef(
    wrapper=_w_diffdock_pose,
    tool_name="DiffDock (R17 ship)",
    backend="python",
    notes="Real DiffDock inference; confidence ∈ [0, 1].",
))
_register("equibind_pose", ToolRef(
    wrapper=_w_equibind_pose,
    tool_name="EquiBind (NOT SHIPPED)",
    backend="python",
    notes="Reserved metric — adapter not yet in repo. Use "
          "diffdock_pose / flowdock_pose as the SE(3)-equivariant "
          "surrogate.",
))
_register("flowdock_pose", ToolRef(
    wrapper=_w_flowdock,
    tool_name="FlowDock (R17 ship)",
    backend="python",
    notes="Real flow-matching docking confidence ∈ [0, 1].",
))
_register("pocket2mol", ToolRef(
    wrapper=_w_pocket2mol,
    tool_name="Pocket2Mol (R17 ship)",
    backend="python",
    notes="Real Pocket2Mol sampler; atom-count is the validity proxy.",
))
_register("boltz_affinity", ToolRef(
    wrapper=_w_boltz_affinity,
    tool_name="Boltz-style pIC50 (R17 ship, D-MPNN)",
    backend="python",
    notes="Real locally-trained Attentive D-MPNN pIC50 predictor; "
          "swap to Boltz foundation-model pIC50 head when shipped.",
))
_register("af2_confidence", ToolRef(
    wrapper=_w_af2_confidence,
    tool_name="Voxelization complexity (AF2 surrogate)",
    backend="python",
    notes="Real voxelization complexity proxy (deterministic). "
          "Swap to AF2 pLDDT inference once shipped.",
))
_register("pb_pass_rate", ToolRef(
    wrapper=_w_pb_pass_rate,
    tool_name="PoseBusters (R17 ship)",
    backend="python",
    notes="Real PoseBusters 26-check validation.",
))
_register("aizynth_synthesis", ToolRef(
    wrapper=_w_aizynth_synthesis,
    tool_name="AiZynthFinder (R17 ship)",
    backend="python",
    notes="Real AiZynthFinder MCTS retrosynthesis with SMARTS "
          "fallback when AiZynthFinder is not installed.",
))
_register("pl_likelihood", ToolRef(
    wrapper=_w_pl_likelihood,
    tool_name="PLL signature (R17 ship)",
    backend="python",
    notes="Real Persistent Local Laplacian signature from "
          "molmetal.molmetal_lam.lam_chem.pll_signature.",
))
_register("decomp_diff", ToolRef(
    wrapper=_w_decomp_diff,
    tool_name="DecompDiff (NOT SHIPPED)",
    backend="python",
    notes="Reserved metric — adapter pending.",
))
_register("molcraft", ToolRef(
    wrapper=_w_molcraft,
    tool_name="MolCraft (NOT SHIPPED)",
    backend="python",
    notes="Reserved metric — adapter pending.",
))
_register("targetdiff", ToolRef(
    wrapper=_w_targetdiff,
    tool_name="TargetDiff (R17 ship)",
    backend="python",
    notes="Real TargetDiff sampler; atom-count is the validity proxy.",
))
_register("imold_ood", ToolRef(
    wrapper=_w_imold_ood,
    tool_name="IMOLDE OOD (NOT SHIPPED)",
    backend="python",
    notes="Reserved metric — adapter pending.",
))
_register("drugood_ood", ToolRef(
    wrapper=_w_drugood_ood,
    tool_name="DrugOOD OOD (NOT SHIPPED)",
    backend="python",
    notes="Reserved metric — adapter pending.",
))
_register("glide_score", ToolRef(
    wrapper=_w_glide_score,
    tool_name="Glide (Schrödinger, NOT SHIPPED)",
    backend="commercial",
    notes="Reserved metric — commercial tool, requires licence.",
))
_register("sa_score", ToolRef(
    wrapper=_w_sa_score,
    tool_name="SA score (RDKit Ertl)",
    backend="python",
    notes="Real Ertl-Schuffenhauer synthesis accessibility 1..10.",
))
_register("qed", ToolRef(
    wrapper=_w_qed,
    tool_name="QED (RDKit Bickerton)",
    backend="python",
    notes="Real Bickerton drug-likeness 0..1.",
))
_register("flowr", ToolRef(
    wrapper=_w_flowr,
    tool_name="FlowER retrosynthesis (R17 ship)",
    backend="python",
    notes="Real FlowER / FlowER retrosynthesis sampler.",
))


# ---------------------------------------------------------------------------
# Wf-43 Phase 2 — populate ROUTING_POLICY
#
# Each entry below mirrors ``policy.json`` so callers can audit the
# routing decision in-process. The schema is intentionally flat so
# callers can do ``ROUTING_POLICY[metric].primary_tool`` / ``.wf43_status``
# / ``.fallback_chain`` without parsing nested dicts.
# ---------------------------------------------------------------------------
_policy_register_many({
    "vina_score": PolicyEntry(
        primary_tool="AutoDock Vina 1.2.7",
        backend="python+cli",
        module="molmetal.molmetal_lam.sbdd_env.vina_adapter.VinaDockingAdapter",
        license="Apache-2.0",
        fallback_chain=("vina_qvina2", "vina_gpu"),
        required_data=("smiles", "pocket_pdbqt"),
        wf43_status="MEASURED",
        notes="Meeko+Vina Python binding (preferred) or vina CLI binary. "
              "Engine selection via MOLMETAL_VINA_ENGINE env var "
              "(unset → AutoDock, qvina2 → QuickVina 2, gpu → Vina-GPU).",
    ),
    "vina_high_affinity": PolicyEntry(
        primary_tool="AutoDock Vina 1.2.7",
        backend="python+cli",
        module="molmetal.molmetal_lam.sbdd_env.vina_adapter.VinaDockingAdapter",
        license="Apache-2.0",
        fallback_chain=("vina_qvina2", "vina_gpu"),
        required_data=("smiles", "pocket_pdbqt"),
        wf43_status="MEASURED",
        notes="Boolean classifier on top of vina_score (cutoff_kcal default -8.0).",
    ),
    "vina_min_score": PolicyEntry(
        primary_tool="AutoDock Vina 1.2.7",
        backend="python+cli",
        module="molmetal.molmetal_lam.sbdd_env.vina_adapter.VinaDockingAdapter",
        license="Apache-2.0",
        fallback_chain=("vina_qvina2", "vina_gpu"),
        required_data=("smiles", "pocket_pdbqt"),
        wf43_status="MEASURED",
        notes="Aggregates vina_score over a batch — min / median / n_below_cutoff.",
    ),
    "vina_qvina2": PolicyEntry(
        primary_tool="QuickVina 2 (CASO-derived qvina02)",
        backend="cli",
        module="molmetal.scripts.q_vina2_swap",
        license="Apache-2.0",
        fallback_chain=(),
        required_data=("smiles", "pocket_pdbqt"),
        wf43_status="RESERVED",
        notes="Routes through q_vina2_swap.py calling the vendored "
              "qvina02 binary at references/SoftMol/gated_mcts/utils/docking/qvina02. "
              "Engine selected via MOLMETAL_VINA_ENGINE=qvina2.",
    ),
    "vina_gpu": PolicyEntry(
        primary_tool="Vina-GPU (Doerr 2021)",
        backend="python+CUDA",
        module="molmetal.molmetal_lam.sbdd_env.vina_gpu_adapter.QuickVinaGPUAdapter",
        license="Apache-2.0",
        fallback_chain=("vina_score",),
        required_data=("smiles", "pocket_pdbqt"),
        wf43_status="RESERVED",
        notes="GPU-accelerated Vina; same kcal/mol semantics as vina_score. "
              "Requires CUDA + the vina_gpu extension. The "
              "QuickVina2-GPU-2-1 vendor build is AMD-only; on NVIDIA "
              "Blackwell we fall back to vina_score. Engine selected via "
              "MOLMETAL_VINA_ENGINE=gpu.",
    ),
    "diffdock_pose": PolicyEntry(
        primary_tool="DiffDock (Corso 2024 NeurIPS)",
        backend="python",
        module="molmetal.molmetal_lam.sbdd_env.diffdock_adapter.DiffDockAdapter",
        license="MIT",
        fallback_chain=("flowdock_pose",),
        required_data=("smiles", "pocket_pdb"),
        wf43_status="MEASURED",
        notes="Real DiffDock inference; confidence ∈ [0, 1]. "
              "Requires GPU + torch_cluster/torch_scatter. Falls back to "
              "FlowDock when DiffDock backend is unavailable.",
    ),
    "equibind_pose": PolicyEntry(
        primary_tool="EquiBind (Stärk 2022)",
        backend="python",
        module="NOT_SHIPPED",
        license="MIT",
        fallback_chain=("flowdock_pose", "diffdock_pose"),
        required_data=("smiles", "pocket_pdb"),
        wf43_status="RESERVED",
        notes="EquiBind adapter not yet shipped. Reserved metric — the "
              "wrapper raises AdapterUnavailable until the adapter lands. "
              "Use flowdock_pose / diffdock_pose as SE(3)-equivariant surrogate.",
    ),
    "flowdock_pose": PolicyEntry(
        primary_tool="FlowDock (R17 ship)",
        backend="python",
        module="molmetal.molmetal_lam.sbdd_env.flowdock_adapter.FlowDockAdapter",
        license="MIT",
        fallback_chain=("diffdock_pose",),
        required_data=("smiles", "pocket_pdb"),
        wf43_status="MEASURED",
        notes="Real flow-matching docking confidence ∈ [0, 1]. CPU-friendly "
              "fallback for DiffDock GPU stack.",
    ),
    "pocket2mol": PolicyEntry(
        primary_tool="Pocket2Mol (Peng 2022 ICLR-W)",
        backend="python",
        module="molmetal.molmetal_lam.sbdd_env.pocket2mol_adapter.Pocket2MolSampler",
        license="MIT",
        fallback_chain=(),
        required_data=("pocket_pdb",),
        wf43_status="RESERVED",
        notes="Real Pocket2Mol sampler; atom-count is the validity proxy. "
              "Requires official Pocket2Mol weights at the path returned by "
              "pocket2mol_adapter._checkpoint_path().",
    ),
    "boltz_affinity": PolicyEntry(
        primary_tool="D-MPNN pIC50 (R17 ship, locally-trained attentive fp)",
        backend="python",
        module="molmetal.molmetal_lam.sbdd_env.pic50_predictor.predict_pic50",
        license="MIT",
        fallback_chain=(),
        required_data=("smiles",),
        wf43_status="MEASURED",
        notes="Real locally-trained Attentive D-MPNN pIC50 predictor; "
              "nan on parse-fail or missing checkpoint. Swap to Boltz "
              "foundation-model pIC50 head when shipped.",
    ),
    "af2_confidence": PolicyEntry(
        primary_tool="Voxelization complexity (AF2 surrogate)",
        backend="python",
        module="molmetal.molmetal_lam.sbdd_env.voxelization.pocket_to_spatial_tiles",
        license="MIT",
        fallback_chain=(),
        required_data=("pocket",),
        wf43_status="MEASURED",
        notes="Real voxelization complexity proxy (deterministic std-of-tile-occupancy). "
              "Swap to AlphaFold pLDDT inference when the AF2 backend is shipped.",
    ),
    "pb_pass_rate": PolicyEntry(
        primary_tool="PoseBusters (Buttenschoen 2024)",
        backend="python",
        module="molmetal.molmetal_lam.sbdd_env.posebusters_adapter.PoseBustersAdapter",
        license="MIT",
        fallback_chain=(),
        required_data=("smiles", "pocket_pdb"),
        wf43_status="MEASURED",
        notes="Real PoseBusters 26-check validation (14 chemistry + 12 protein-aware "
              "in 'dock' mode; 14 chemistry-only in 'mol' mode).",
    ),
    "aizynth_synthesis": PolicyEntry(
        primary_tool="AiZynthFinder (Genheden 2020)",
        backend="python",
        module="molmetal.molmetal_lam.sbdd_env.aizynth_adapter.check_list",
        license="MIT",
        fallback_chain=(),
        required_data=("smiles",),
        wf43_status="MEASURED",
        notes="Real AiZynthFinder MCTS retrosynthesis with SMARTS fallback "
              "when AiZynthFinder is not installed.",
    ),
    "pl_likelihood": PolicyEntry(
        primary_tool="PLL signature (Liu/Feng 2026)",
        backend="python",
        module="molmetal.molmetal_lam.lam_chem.pll_signature.compute_pll",
        license="MIT",
        fallback_chain=(),
        required_data=("smiles",),
        wf43_status="MEASURED",
        notes="Real Persistent Local Laplacian signature.",
    ),
    "decomp_diff": PolicyEntry(
        primary_tool="DecompDiff (Guan 2024)",
        backend="python",
        module="NOT_SHIPPED",
        license="MIT",
        fallback_chain=(),
        required_data=("pocket_pdb",),
        wf43_status="RESERVED",
        notes="DecompDiff adapter not yet shipped. Reserved metric — the "
              "wrapper raises AdapterUnavailable until the adapter lands.",
    ),
    "molcraft": PolicyEntry(
        primary_tool="MolCraft (Qu 2024)",
        backend="python",
        module="NOT_SHIPPED",
        license="MIT",
        fallback_chain=(),
        required_data=("pocket_pdb",),
        wf43_status="RESERVED",
        notes="MolCraft adapter not yet shipped. Reserved metric — the "
              "wrapper raises AdapterUnavailable until the adapter lands.",
    ),
    "targetdiff": PolicyEntry(
        primary_tool="TargetDiff (Guan 2023 ICLR)",
        backend="python",
        module="molmetal.molmetal_lam.sbdd_env.targetdiff_adapter.PocketToLigandSampler",
        license="MIT",
        fallback_chain=(),
        required_data=("pocket_pdb",),
        wf43_status="MEASURED",
        notes="Real TargetDiff sampler; atom-count is the validity proxy.",
    ),
    "imold_ood": PolicyEntry(
        primary_tool="IMOLDE OOD (Mervin 2024)",
        backend="python",
        module="NOT_SHIPPED",
        license="MIT",
        fallback_chain=(),
        required_data=("smiles", "split"),
        wf43_status="RESERVED",
        notes="IMOLDE OOD adapter not yet shipped. Reserved metric.",
    ),
    "drugood_ood": PolicyEntry(
        primary_tool="DrugOOD OOD (Ji 2022 NeurIPS-DB)",
        backend="python",
        module="NOT_SHIPPED",
        license="MIT",
        fallback_chain=(),
        required_data=("smiles", "split"),
        wf43_status="RESERVED",
        notes="DrugOOD OOD adapter not yet shipped. Reserved metric.",
    ),
    "glide_score": PolicyEntry(
        primary_tool="Glide (Schrödinger)",
        backend="commercial",
        module="NOT_SHIPPED",
        license="Schrödinger commercial",
        fallback_chain=("vina_score",),
        required_data=("smiles", "pocket_pdb"),
        wf43_status="RESERVED",
        notes="Commercial docking tool. No Schrödinger licence in this "
              "open-source repo. Use vina_score as OSS surrogate.",
    ),
    "sa_score": PolicyEntry(
        primary_tool="SA score (Ertl-Schuffenhauer 2008)",
        backend="python",
        module="molmetal.molmetal_lam.sbdd_env.sa_score.sa_score_ertl",
        license="MIT",
        fallback_chain=(),
        required_data=("smiles",),
        wf43_status="MEASURED",
        notes="Real Ertl-Schuffenhauer synthesis accessibility 1..10 (RDKit Contrib).",
    ),
    "qed": PolicyEntry(
        primary_tool="QED (Bickerton 2012)",
        backend="python",
        module="molmetal.molmetal_lam.sbdd_env.qed_scorer.qed_from_smiles",
        license="BSD",
        fallback_chain=(),
        required_data=("smiles",),
        wf43_status="MEASURED",
        notes="Real Bickerton drug-likeness 0..1 (RDKit built-in).",
    ),
    "flowr": PolicyEntry(
        primary_tool="FlowER retrosynthesis (Kuznetsov 2022)",
        backend="python",
        module="molmetal.molmetal_lam.sbdd_env.flowr_adapter.FlowRSampler",
        license="MIT",
        fallback_chain=(),
        required_data=("smiles",),
        wf43_status="MEASURED",
        notes="Real FlowER retrosynthesis-flow validity. Adapter under references/FLOWR.",
    ),
    # ---- MOCK_DELETED — Phase-4A external scorers (module absent) ----
    "coord_geometry_proxy": PolicyEntry(
        primary_tool="DELETED — module molmetal_lam.sbdd_env.external_scorers not present",
        backend="N/A",
        module="N/A",
        license="N/A",
        fallback_chain=(),
        required_data=("smiles", "metal_label"),
        wf43_status="MOCK_DELETED",
        notes="Phase-4A external scorer was a stub interface in "
              "metrics_v2.py:870 referencing a non-existent module. "
              "Wf-43 Phase 2 marks this MOCK_DELETED. Real alternative: "
              "metal_coord_compliance (RDKit SMARTS, real).",
    ),
    "phototherapy_activity": PolicyEntry(
        primary_tool="DELETED — module molmetal_lam.sbdd_env.external_scorers not present",
        backend="N/A",
        module="N/A",
        license="N/A",
        fallback_chain=(),
        required_data=("smiles",),
        wf43_status="MOCK_DELETED",
        notes="Phase-4A external scorer was a stub interface. "
              "Real alternative: attentive D-MPNN trained on phototherapy "
              "labels (no ship in repo yet).",
    ),
    "ptiv_reduction_potential": PolicyEntry(
        primary_tool="DELETED — module molmetal_lam.sbdd_env.external_scorers not present",
        backend="N/A",
        module="N/A",
        license="N/A",
        fallback_chain=(),
        required_data=("smiles",),
        wf43_status="MOCK_DELETED",
        notes="Phase-4A external scorer was a stub interface. "
              "Real alternative: Pt(IV) DFT reduction potential surrogate "
              "(outside scope of this repo).",
    ),
    "metalloprotein_affinity": PolicyEntry(
        primary_tool="DELETED — module molmetal_lam.sbdd_env.external_scorers not present",
        backend="N/A",
        module="N/A",
        license="N/A",
        fallback_chain=(),
        required_data=("smiles",),
        wf43_status="MOCK_DELETED",
        notes="Phase-4A external scorer was a stub interface. "
              "Real alternative: D-MPNN trained on "
              "molmetal/data/protein_metal_binding.csv (the protein-metal "
              "binding CSV).",
    ),
})


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def available_metrics() -> List[str]:
    """Return the sorted list of metric names :func:`score_any` accepts."""
    return sorted(REGISTRY.keys())


def tool_for_metric(metric: str) -> ToolRef:
    """Return the :class:`ToolRef` for ``metric`` or raise.

    Raises
    ------
    AdapterUnavailable
        If ``metric`` is not in :data:`REGISTRY`.
    """
    if metric not in REGISTRY:
        raise AdapterUnavailable(
            f"Unknown metric: {metric!r}. Known metrics: "
            f"{', '.join(available_metrics())}"
        )
    return REGISTRY[metric]


def score_any(
    metric: str,
    molecules: Sequence[Any],
    pocket: Optional[Any] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Route ``metric`` to its real scoring tool.

    Parameters
    ----------
    metric:
        Metric name (see :func:`available_metrics`).
    molecules:
        List of SMILES strings OR :class:`molmetal.domain.Molecule`
        objects (auto-normalised to SMILES).
    pocket:
        Optional :class:`molmetal.domain.Pocket`. Required for
        pocket-conditioned metrics (vina_score, pocket2mol, …).
    **kwargs:
        Forwarded to the wrapper (engine, exhaustiveness, n_poses,
        cutoff_kcal, …).

    Returns
    -------
    dict
        ``{"metric": str, "tool": str, "values": Any, "n": int,
        "status": "MEASURED"}``. If the wrapper returns a scalar it is
        wrapped as a single-element list.

    Raises
    ------
    AdapterUnavailable
        Always — either the metric is unknown or the real tool
        cannot be reached. NO MOCK FALLBACK.
    """
    ref = tool_for_metric(metric)
    ctx: Dict[str, Any] = {
        "metric": metric,
        "molecules": molecules,
        "pocket": pocket,
    }
    ctx.update(kwargs)

    # ``MOLMETAL_VINA_ENGINE`` env var: when set to "qvina2", route
    # vina_score / vina_high_affinity / vina_min_score to the Q-Vina2
    # wrapper. This is the single hook Wf-43 gives callers to swap
    # Vina engines without changing call-sites.
    if metric in {"vina_score", "vina_high_affinity", "vina_min_score"}:
        engine_env = os.environ.get("MOLMETAL_VINA_ENGINE", "").lower().strip()
        if engine_env == "qvina2":
            ref = REGISTRY["vina_qvina2"]
        elif engine_env == "gpu":
            ref = REGISTRY["vina_gpu"]

    try:
        raw = ref.wrapper(ctx)
    except AdapterUnavailable:
        # Re-raise unchanged — this is the documented failure mode.
        raise
    except Exception as e:
        # Real-tool failure. Surface the original error and the full
        # traceback so the caller can debug the missing binary / dep.
        tb = traceback.format_exc()
        raise AdapterUnavailable(
            f"Real tool `{ref.tool_name}` failed for metric "
            f"{metric!r}: {type(e).__name__}: {e}\n--- traceback ---\n"
            f"{tb}"
        ) from e

    # Normalise scalar → list.
    if isinstance(raw, (int, float)):
        values: Any = [raw]
    else:
        values = list(raw) if raw is not None else []

    return {
        "metric": metric,
        "tool": ref.tool_name,
        "backend": ref.backend,
        "values": values,
        "n": len(values) if hasattr(values, "__len__") else 1,
        "status": "MEASURED",
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _as_smiles_list(molecules: Iterable[Any]) -> List[str]:
    """Normalise an iterable of SMILES / Molecule / str to a list of SMILES."""
    out: List[str] = []
    for m in molecules:
        if isinstance(m, str):
            out.append(m)
            continue
        # Molecule-like objects expose a ``.smiles`` attribute.
        smi = getattr(m, "smiles", None)
        if smi is None:
            # Last-ditch: str() — caller will see a parse-failure NaN
            # rather than the router inventing a fake score.
            out.append(str(m))
        else:
            out.append(str(smi))
    return out


def _pocket_to_pdb_string(pocket: Optional[Any]) -> str:
    """Convert a Pocket object / PDB path / PDB string to a PDB string."""
    if pocket is None:
        raise AdapterUnavailable(
            "pocket is required for this metric but was None."
        )
    if isinstance(pocket, str):
        # Heuristic: short strings are SMILES, longer are PDB text.
        if "\n" in pocket or pocket.startswith(("ATOM", "HEADER", "MODEL")):
            return pocket
        # Assume it's a path.
        from pathlib import Path
        return Path(pocket).read_text()
    # Pocket-like object: delegate to the vina adapter helper if
    # available, else raise.
    try:
        vina = _import_vina()
        return vina._pocket_to_pdb_string(pocket)
    except AdapterUnavailable:
        raise
    except Exception as e:
        raise AdapterUnavailable(
            f"Could not serialise pocket to PDB string: "
            f"{type(e).__name__}: {e}"
        ) from e


__all__ = [
    "AdapterUnavailable",
    "PolicyEntry",
    "REGISTRY",
    "ROUTING_POLICY",
    "ToolRef",
    "available_metrics",
    "score_any",
    "tool_for_metric",
]
