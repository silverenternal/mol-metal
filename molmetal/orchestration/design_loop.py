"""DesignLoop orchestrator — wires generator + docker + predictor + scorer.

This module is the **canonical entry point** for the W1 (Phase 0) end-to-end
pipeline described in ``TODO/11_design_loop/closed_loop_design.md``.  It is
distinct from :class:`molmetal.orchestration.ClosedLoop` in two ways:

* it accepts a **flat signature** (``run(pocket, n_generate, n_dock, n_top)``)
  that matches the spec — no ``DesignLoopConfig`` object needed;
* it returns a single :class:`LoopResult` dataclass that carries the top
  candidates together with their full provenance (generator metadata,
  pose coordinates, per-objective score breakdown) instead of just a
  ``List[ScoredCandidate]``.

Five-step pipeline
------------------

1. ``generator.generate(pocket, GenerationConfig(n_samples=n_generate))``
2. rank generated molecules by the predictor (RDKit QED + MW + TPSA),
   keep the top ``n_dock``
3. ``docker.dock(top_mols, pocket, DockingConfig(n_poses=1))`` — get one
   pose per molecule
4. ``scorer.score(complexes)`` — combine Vina + QED + SA into a scalar
5. return the top ``n_top`` with full provenance

Optional refinement
-------------------

When ``refinement_iters > 0`` the orchestrator takes the top-10 SMILES
from the previous iteration as **seeds** for the next round.  The actual
seed-to-molecule mapping is the responsibility of the generator
(generators with latent-space seed injection can use the SMILES to seed
their latent; text-conditioned generators can use them as prompts;
random samplers will simply dedup duplicates).

This implementation is **adapter-agnostic**: any duck-typed
:class:`MoleculeGenerator`, :class:`DockingEngine`,
:class:`PropertyPredictor`, :class:`ScoringFunction` works.

Reference
---------
* TODO/11_design_loop/closed_loop_design.md — Phase 0 / Phase 1 / Phase 2 spec
* TODO/04_architecture/model_design.md  — model contracts
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from molmetal.domain import Complex, Molecule, Pocket
from molmetal.ports import (
    DockingConfig,
    DockingEngine,
    GenerationConfig,
    MoleculeGenerator,
    PropertyPrediction,
    PropertyPredictor,
    ScoredCandidate,
    ScoringFunction,
)


__all__ = [
    "LoopResult",
    "CandidateProvenance",
    "DesignLoop",
]


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CandidateProvenance:
    """Full provenance of one top candidate.

    Every field is serialisable (``torch.Tensor`` round-trips through
    ``torch.save``/``torch.load``; SMILES is a plain string).  This
    object is what callers should persist when they want to reproduce
    a design round later.
    """

    rank: int
    smiles: str
    combined_score: float
    qed: float
    sa_score: float
    mol_weight: float
    logp: float
    tpsa: float
    vina_score: Optional[float]
    pose_confidence: Optional[float]
    binding_affinity_pic50: Optional[float]
    # Pose coordinates after docking (or original coords if no docker).
    pose_coords: Optional[torch.Tensor] = None
    # Score breakdown — per-objective contributions (only present when the
    # scorer exposes a ``breakdown`` method).
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    # Generator provenance (model name + adapter metadata snapshot).
    generator_metadata: Dict[str, Any] = field(default_factory=dict)
    # Docker provenance (model name + adapter metadata snapshot).
    docker_metadata: Dict[str, Any] = field(default_factory=dict)
    # Predictor metadata snapshot.
    predictor_metadata: Dict[str, Any] = field(default_factory=dict)
    # Per-objective raw values used by the scorer (filled by ``scorer.breakdown``
    # when available; otherwise copied from the prediction).
    raw_values: Dict[str, Optional[float]] = field(default_factory=dict)


@dataclass(frozen=True)
class LoopResult:
    """Aggregate result of one :meth:`DesignLoop.run` call."""

    pocket_id: str
    n_generate: int
    n_dock: int
    n_top: int
    refinement_iters: int
    candidates: List[CandidateProvenance]
    wall_clock_s: float
    per_iteration_summary: List[Dict[str, Any]]
    loop_metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_qed(mol: Molecule) -> float:
    """Return ``Molecule.qed`` (or 0.0 if None).  Used for the predictor
    pre-filter where QED is the cheap 2D descriptor."""
    v = getattr(mol, "qed", 0.0)
    return float(v) if v is not None else 0.0


def _safe_complex_vina(complex: Optional[Complex]) -> Optional[float]:
    if complex is None:
        return None
    v = getattr(complex, "vina_score", None)
    return float(v) if v is not None else None


def _safe_complex_confidence(complex: Optional[Complex]) -> Optional[float]:
    if complex is None:
        return None
    v = getattr(complex, "pose_confidence", None)
    return float(v) if v is not None else None


def _predictor_score(
    predictor: PropertyPredictor,
    mol: Molecule,
    cmplx: Optional[Complex],
) -> PropertyPrediction:
    """Call ``predictor.predict`` with one catch-all for adapter failure.

    A prediction failure must not kill a multi-hour design round; we
    fall back to a default :class:`PropertyPrediction` instead.
    """
    try:
        return predictor.predict(mol, cmplx)
    except Exception:  # noqa: BLE001
        return PropertyPrediction()


def _dock_one(
    docker: DockingEngine,
    mol: Molecule,
    pocket: Pocket,
    cfg: DockingConfig,
) -> Optional[Complex]:
    """Best-pose extractor for a single (mol, pocket) pair.

    Returns the highest-confidence pose, breaking ties on the more
    negative (better) Vina score.  ``None`` is returned when docking
    fails — the caller will fall back to ``mol`` with no pose.
    """
    try:
        poses = docker.dock(mol, pocket, cfg)
    except Exception:  # noqa: BLE001
        return None
    if not poses:
        return None
    return max(
        poses,
        key=lambda c: (
            float(getattr(c, "pose_confidence", 0.0) or 0.0),
            -float(_safe_complex_vina(c) if _safe_complex_vina(c) is not None else 0.0),
        ),
    )


# ---------------------------------------------------------------------------
# DesignLoop
# ---------------------------------------------------------------------------
class DesignLoop:
    """End-to-end orchestrator: generate → pre-filter → dock → score → top-K.

    Parameters
    ----------
    generator:
        Anything implementing :class:`MoleculeGenerator`.
    docker:
        Anything implementing :class:`DockingEngine`.
    predictor:
        Anything implementing :class:`PropertyPredictor`.
    scorer:
        Anything implementing :class:`ScoringFunction`.

    The four components are kept as attributes so callers can introspect
    the wiring (and so unit tests can monkey-patch any of them).

    Notes
    -----
    * ``scorer.setup()`` is called once at construction time (it has no
      device to set).  All other components are set up lazily on the
      first :meth:`run` call (or via :meth:`setup`) so importing this
      module never triggers model downloads.
    * Adapter failures (docking crash, predictor crash) are caught per
      candidate so a single bad sample cannot abort a multi-hour round.
    """

    def __init__(
        self,
        generator: MoleculeGenerator,
        docker: DockingEngine,
        predictor: PropertyPredictor,
        scorer: ScoringFunction,
    ) -> None:
        self.generator = generator
        self.docker = docker
        self.predictor = predictor
        self.scorer = scorer

        # Setup the scorer once (no device argument).
        scorer_setup = getattr(self.scorer, "setup", None)
        if scorer_setup is not None:
            scorer_setup()

        self._device: str = "cpu"
        self._is_setup: bool = False

    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return "DesignLoop_v1"

    # ------------------------------------------------------------------
    def setup(self, device: str = "cpu") -> None:
        """Set up the heavy adapters (generator, docker, predictor).

        The scorer was already set up in ``__init__`` because it has no
        device to bind.  Repeated calls are idempotent — the adapters
        themselves decide what to re-load.
        """
        self._device = device
        for component in (self.generator, self.docker, self.predictor):
            setup = getattr(component, "setup", None)
            if setup is not None:
                setup(device=device)
        self._is_setup = True

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def run(
        self,
        pocket: Pocket,
        n_generate: int = 1000,
        n_dock: int = 100,
        n_top: int = 10,
        refinement_iters: int = 0,
        docking_config: Optional[DockingConfig] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> LoopResult:
        """Run the end-to-end pipeline and return a :class:`LoopResult`.

        Parameters
        ----------
        pocket:
            The binding site — passed to ``generator.generate`` and
            ``docker.dock``.
        n_generate:
            Number of molecules to ask the generator for.
        n_dock:
            How many of the generated molecules to dock (selected by
            predictor-QED descending).
        n_top:
            Number of top-scoring candidates to return.
        refinement_iters:
            Extra closed-loop iterations using the previous round's
            top-10 SMILES as seeds.  ``0`` = single pass (Phase 0
            default).
        docking_config:
            Override the default ``DockingConfig(n_poses=1)``.
        generation_kwargs:
            Extra keyword arguments merged into every
            ``GenerationConfig`` (e.g. ``temperature=1.5``).
        seed:
            Base RNG seed — each iteration adds its index.
        """
        if not self._is_setup:
            # Lazy auto-setup with CPU so the first run() doesn't require
            # an explicit setup() call.  Real deployments will pre-call
            # setup() with ``device="cuda"`` / ``"hip"``.
            self.setup(device=self._device)

        t_total = time.time()
        dock_cfg = docking_config or DockingConfig(n_poses=1)
        gen_kwargs = dict(generation_kwargs or {})

        per_iter_summary: List[Dict[str, Any]] = []
        candidates_pool: List[CandidateProvenance] = []
        seed_smiles: List[str] = []

        total_iters = 1 + max(0, int(refinement_iters))
        for it in range(total_iters):
            t0 = time.time()
            iter_record: Dict[str, Any] = {"iteration": it}

            # --- 1. Generate -------------------------------------------------
            gen_cfg = GenerationConfig(
                n_samples=int(n_generate),
                seed=int(seed) + it,
                **gen_kwargs,
            )
            molecules = list(self.generator.generate(pocket, gen_cfg))
            iter_record["n_generated"] = len(molecules)

            # --- Inject seed SMILES as conditioning when we have any -------
            # The generator may (a) ignore them, (b) use them as latent
            # seeds, or (c) use them as prompts.  We just attach them
            # via the ``conditioning`` dict — generators that recognise
            # the key "seed_smiles" will consume it; others silently
            # drop it.
            if seed_smiles and "seed_smiles" not in gen_cfg.conditioning:
                # rebuild with updated conditioning
                conditioning = dict(gen_cfg.conditioning)
                conditioning["seed_smiles"] = list(seed_smiles)
                gen_cfg = GenerationConfig(
                    n_samples=gen_cfg.n_samples,
                    n_steps=gen_cfg.n_steps,
                    temperature=gen_cfg.temperature,
                    seed=gen_cfg.seed,
                    conditioning=conditioning,
                )
                # If the generator supports seed_smiles natively it may
                # produce different molecules; for generators that
                # don't, this is a no-op.
                molecules = list(self.generator.generate(pocket, gen_cfg))
                iter_record["n_generated"] = len(molecules)
                iter_record["seed_smiles_count"] = len(seed_smiles)

            # --- 2. Pre-filter by predictor-QED -----------------------------
            n_dock_eff = max(0, min(int(n_dock), len(molecules)))
            order = sorted(
                range(len(molecules)),
                key=lambda i: _safe_qed(molecules[i]),
                reverse=True,
            )
            dock_indices = order[:n_dock_eff]

            # --- 3. Dock the top-K -----------------------------------------
            complexes: List[Optional[Complex]] = [None] * len(molecules)
            n_dock_failures = 0
            for i in dock_indices:
                c = _dock_one(self.docker, molecules[i], pocket, dock_cfg)
                if c is None:
                    n_dock_failures += 1
                else:
                    complexes[i] = c
            iter_record["n_docked"] = len(dock_indices) - n_dock_failures
            iter_record["n_dock_failures"] = n_dock_failures

            # --- 4. Predict properties for ALL molecules -------------------
            triples: List[Tuple[Molecule, Optional[Complex], PropertyPrediction]] = []
            for mol, c in zip(molecules, complexes):
                triples.append((mol, c, _predictor_score(self.predictor, mol, c)))
            iter_record["n_triples"] = len(triples)

            # --- 5. Score --------------------------------------------------
            scored = list(self.scorer.score(triples))

            # Take top-N from this iteration (top-N == n_top keeps the
            # pool bounded; the refinement loop will overwrite).
            this_iter_top = scored[: max(0, int(n_top))]
            iter_record["n_kept_this_iter"] = len(this_iter_top)
            iter_record["best_score"] = (
                float(this_iter_top[0].combined_score) if this_iter_top else float("-inf")
            )
            iter_record["mean_score"] = (
                float(sum(c.combined_score for c in scored) / len(scored))
                if scored
                else float("nan")
            )
            iter_record["elapsed_s"] = time.time() - t0

            # Append the iteration's top-K to the global pool.
            for cand in this_iter_top:
                candidates_pool.append(self._to_provenance(cand))
            per_iter_summary.append(iter_record)

            # --- 6. Seed next refinement round with this round's top-10 ---
            seed_smiles = [
                c.molecule.smiles
                for c in scored[:10]
                if getattr(c.molecule, "smiles", "")
            ]

        # Final ranking across all iterations (refinement is meant to
        # *find better* candidates, not to forget previous winners).
        candidates_pool.sort(key=lambda c: c.combined_score, reverse=True)
        top = candidates_pool[: max(0, int(n_top))]
        # Re-number ranks after the cross-iteration sort.
        top = [
            CandidateProvenance(
                rank=i + 1,
                smiles=c.smiles,
                combined_score=c.combined_score,
                qed=c.qed,
                sa_score=c.sa_score,
                mol_weight=c.mol_weight,
                logp=c.logp,
                tpsa=c.tpsa,
                vina_score=c.vina_score,
                pose_confidence=c.pose_confidence,
                binding_affinity_pic50=c.binding_affinity_pic50,
                pose_coords=c.pose_coords,
                score_breakdown=c.score_breakdown,
                generator_metadata=c.generator_metadata,
                docker_metadata=c.docker_metadata,
                predictor_metadata=c.predictor_metadata,
                raw_values=c.raw_values,
            )
            for i, c in enumerate(top)
        ]

        wall = time.time() - t_total
        meta = self.get_metadata()
        return LoopResult(
            pocket_id=getattr(pocket, "pdb_id", "?"),
            n_generate=int(n_generate),
            n_dock=int(n_dock),
            n_top=int(n_top),
            refinement_iters=int(refinement_iters),
            candidates=top,
            wall_clock_s=wall,
            per_iteration_summary=per_iter_summary,
            loop_metadata=meta,
        )

    # ------------------------------------------------------------------
    # ScoredCandidate -> CandidateProvenance
    # ------------------------------------------------------------------
    def _to_provenance(self, cand: ScoredCandidate) -> CandidateProvenance:
        mol = cand.molecule
        cmplx = cand.complex
        pred = cand.property_pred

        pose_coords = (
            cmplx.molecule.coords.detach().cpu()
            if (cmplx is not None and cmplx.molecule is not None)
            else None
        )

        # Pull score breakdown if the scorer exposes it.
        breakdown: Dict[str, float] = {}
        raw_values: Dict[str, Optional[float]] = {}
        breakdown_fn = getattr(self.scorer, "breakdown", None)
        if breakdown_fn is not None:
            try:
                breakdown = breakdown_fn(mol, cmplx, pred)
            except Exception:  # noqa: BLE001
                breakdown = {}
        # Raw values for traceability.
        raw_values["qed"] = float(pred.qed) if pred.qed is not None else None
        raw_values["sa_score"] = (
            float(pred.sa_score) if pred.sa_score is not None else None
        )
        raw_values["mol_weight"] = (
            float(pred.mol_weight) if pred.mol_weight is not None else None
        )
        raw_values["tpsa"] = float(pred.tpsa) if pred.tpsa is not None else None
        raw_values["logp"] = float(pred.logp) if pred.logp is not None else None
        raw_values["binding_pic50"] = pred.binding_affinity_pic50
        raw_values["vina_score"] = _safe_complex_vina(cmplx)

        # Snapshot adapter metadata (best-effort).
        gen_meta = self._safe_meta(self.generator)
        dock_meta = self._safe_meta(self.docker)
        pred_meta = self._safe_meta(self.predictor)

        return CandidateProvenance(
            rank=int(cand.rank),
            smiles=str(getattr(mol, "smiles", "") or ""),
            combined_score=float(cand.combined_score),
            qed=float(getattr(pred, "qed", 0.0) or 0.0),
            sa_score=float(getattr(pred, "sa_score", 0.0) or 0.0),
            mol_weight=float(getattr(pred, "mol_weight", 0.0) or 0.0),
            logp=float(getattr(pred, "logp", 0.0) or 0.0),
            tpsa=float(getattr(pred, "tpsa", 0.0) or 0.0),
            vina_score=_safe_complex_vina(cmplx),
            pose_confidence=_safe_complex_confidence(cmplx),
            binding_affinity_pic50=(
                float(pred.binding_affinity_pic50)
                if getattr(pred, "binding_affinity_pic50", None) is not None
                else None
            ),
            pose_coords=pose_coords,
            score_breakdown=breakdown,
            generator_metadata=gen_meta,
            docker_metadata=dock_meta,
            predictor_metadata=pred_meta,
            raw_values=raw_values,
        )

    @staticmethod
    def _safe_meta(component: Any) -> Dict[str, Any]:
        get = getattr(component, "get_metadata", None)
        if get is None:
            return {}
        try:
            m = get()
            return dict(m) if isinstance(m, dict) else {}
        except Exception:  # noqa: BLE001
            return {}

    # ------------------------------------------------------------------
    def get_metadata(self) -> Dict[str, Any]:
        """Pipeline-level provenance — wraps each adapter's metadata."""
        return {
            "loop": self.name,
            "device": self._device,
            "generator": self._safe_meta(self.generator),
            "docker": self._safe_meta(self.docker),
            "predictor": self._safe_meta(self.predictor),
            "scorer": self._safe_meta(self.scorer),
        }
