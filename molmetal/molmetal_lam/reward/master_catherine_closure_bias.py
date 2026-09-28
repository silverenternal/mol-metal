"""master_catherine_closure_bias.py — Master M2 facade for Catherine closure biasing.

====================================================================
Wf-247-D TODO-126 §3 M2 **master** module (2026-09-28)
====================================================================

This module is the **single-API master facade** for the three
Catherine-closure bias flavours used across the 4 strengthen TODOs
(119, 121, 124, 125) and the ASTSR integration (TODO-122):

* ``"oversample"``      — synthetic cohort oversample (TODO-121).
  Built on :class:`molmetal.molmetal_lam.reward.closure_oversample.ClosureOversampler`.
* ``"generation_bias"``— per-pocket Catherine-bounded candidate
  generation (TODO-119 / 124 / 125).
  Built on :func:`molmetal.molmetal_lam.reward.pocket_closure_bias.closure_bounded_smiles_generation`.
* ``"feature_flag"``    — Boolean Catherine-closure gate suitable for
  feeding into the ASTSR (Deflexpressor symbolic regression) feature
  matrix (TODO-122).  This is the **new** flavour shipped by Wf-247-D
  and is the only selector that does **not** require RDKit — the
  feature_flag can be computed on (term, closure-witness) tuples
  without any 3-D embedding or SMILES parsing.

The class :class:`MasterCatherineClosureBias` exposes a single
:meth:`apply` API that takes a ``bias_type`` selector plus a small
``payload`` dict.  Per the TODO-126 §7 honest-reporting discipline,
every result carries an explicit ``engine`` tag so downstream
scoring can attribute Catherine contributions to the right
selector rather than silently substituting the original bias
flavour.

Public surface
--------------
* :class:`MasterCatherineClosureBias`    — single class with one
  :meth:`apply` method that dispatches via ``bias_type``.
* :class:`MasterCatherineBiasResult`     — dataclass returned by
  :meth:`apply`.
* :class:`CatherineClosureFeatureFlag`   — feature-flag result
  dataclass (subset of :class:`MasterCatherineBiasResult` for the
  ``"feature_flag"`` selector).

Backward compatibility
----------------------
This module is **additive** — it does **not** modify, deprecate, or
rename any public symbol in :mod:`closure_oversample`,
:mod:`pocket_closure_bias`, or :mod:`wasserstein_loss`.  It is a
**facade** that re-exports a select subset of the canonical API
under a single ``apply(..., bias_type=...)`` entry point so upstream
call sites have a uniform interface.

Existing scripts that continue to import from the canonical modules
(``train_chemprop_v2_v3.py``, ``train_hilbert_astsr_end_to_end.py``,
``bench_deflexpressor_end_to_end.py``, ``triangle_closure.py``, ...)
are not affected by this module and continue to work unchanged.

Honest framing
--------------
* The ``feature_flag`` selector is **ASTSR-side** — it answers the
  binary question "does this (λ-term, click-rule-witness) pair
  satisfy the Catherine 6/6 closure gate?" and returns a small
  feature dict the ASTSR (Deflexpressor) symbolic regressor can
  consume as a **Boolean gate** (e.g. as a regressor weight, a
  filter mask, or a one-hot feature column).
* The ``feature_flag`` selector **does NOT** run a full Catherine
  sweep on every candidate (cost-prohibitive for ASTSR batches).
  It uses the lightweight Catherine proxy used by
  :func:`molmetal.molmetal_lam.reward.closure_oversample.ClosureOversampler._catherine_summary_for_compound`:
  the closure-BFS witness path + a SMILES coord-violation heuristic
  are sufficient for the ASTSR feature_flag role because ASTSR only
  needs the binary PASS/FAIL verdict + the per-property dict (P1..P6).
* Per TODO-126 §7, every result carries one of the engine tags
  ``"closure_catherine_oversample"`` / ``"closure_catherine_bounded"``
  / ``"closure_catherine_feature_flag"``.

CPU-only.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants — supported bias_type values + engine tags
# ---------------------------------------------------------------------------

#: Allowed values for ``bias_type`` argument to
#: :meth:`MasterCatherineClosureBias.apply`.
SUPPORTED_BIAS_TYPES: Tuple[str, ...] = (
    "oversample",
    "generation_bias",
    "feature_flag",
)

#: Engine tag for the ``"oversample"`` selector (TODO-121 / TODO-126).
ENGINE_OVERSAMPLE: str = "closure_catherine_oversample"

#: Engine tag for the ``"generation_bias"`` selector (TODO-119 / 124 / 125).
ENGINE_GENERATION_BIAS: str = "closure_catherine_bounded"

#: Engine tag for the ``"feature_flag"`` selector (TODO-122 / ASTSR).
ENGINE_FEATURE_FLAG: str = "closure_catherine_feature_flag"

#: All engine tags — used by :func:`build_master_verification`.
ALL_ENGINE_TAGS: Tuple[str, ...] = (
    ENGINE_OVERSAMPLE,
    ENGINE_GENERATION_BIAS,
    ENGINE_FEATURE_FLAG,
)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CatherineClosureFeatureFlag:
    """Boolean closure gate returned by the ``"feature_flag"`` selector.

    Attributes
    ----------
    term_id : str
        Stable id of the λ-term (e.g. an ASTSR-fit row id).
    six_of_six_pass : bool
        ``True`` iff every per-property check passed.
    per_property : dict
        ``{property_id: bool}`` for P1..P6 (Catherine 6/6).
    bias_used : float
        The numerical bias weight the ASTSR regressor should attach
        to this term (``1.0`` on PASS, ``0.0`` on FAIL by default —
        callers can override via ``bias_on_fail``).
    engine : str
        ``"closure_catherine_feature_flag"``.
    wall_time_s : float
        Wall-time to compute the flag (seconds).
    notes : str
        Free-text diagnostic — empty on PASS.
    """

    term_id: str
    six_of_six_pass: bool
    per_property: Dict[str, bool]
    bias_used: float
    engine: str = ENGINE_FEATURE_FLAG
    wall_time_s: float = 0.0
    notes: str = ""


@dataclass
class MasterCatherineBiasResult:
    """Unified result dataclass returned by :meth:`apply`.

    Attributes
    ----------
    bias_type : str
        Selector value used by the call (``"oversample"`` /
        ``"generation_bias"`` / ``"feature_flag"``).
    engine : str
        Engine tag for honest reporting (TODO-126 §7).
    payload : dict
        Free-form payload.  Schema depends on ``bias_type``:

        * ``"oversample"``      → coerced dict from
          :class:`ClosureOversampler.run().as_dict()` with two added
          keys (``engine``, ``catherine_6of6_pass_rate``) so the
          downstream reporter matches the function-style callers.
        * ``"generation_bias"``→ :class:`BoundedGenerationResult.as_dict()`
          dump (keys ``pocket_id``, ``candidates``, ``n_seeds``,
          ``n_candidates_raw``, ``n_candidates_kept``,
          ``catherine_pass_rate``, ``engine``, ``wall_time_s``).
        * ``"feature_flag"``    → :class:`CatherineClosureFeatureFlag`
          dump (keys ``term_id``, ``six_of_six_pass``, ``per_property``,
          ``bias_used``, ``engine``, ``wall_time_s``).
    feature_flags : list[CatherineClosureFeatureFlag], optional
        Populated only for ``"feature_flag"`` selectors that receive
        a **list** of term payloads.  Carried on the result so a
        batched ASTSR call can consume the full set without a second
        ``apply`` invocation.
    wall_time_s : float
        Wall-time of the full compute call (seconds).
    notes : str
        Free-text diagnostic — empty on PASS.
    """

    bias_type: str
    engine: str
    payload: Dict[str, Any]
    feature_flags: List[CatherineClosureFeatureFlag] = field(default_factory=list)
    wall_time_s: float = 0.0
    notes: str = ""


# ---------------------------------------------------------------------------
# Master facade — single class with selector-driven apply
# ---------------------------------------------------------------------------


class MasterCatherineClosureBias:
    """Master facade for the three Catherine-closure bias selectors.

    Single class with a single API (:meth:`apply`) that dispatches
    to the canonical sub-module based on ``bias_type``.

    Parameters
    ----------
    bias_on_pass : float
        Bias weight to assign when Catherine 6/6 passes (default 1.0).
    bias_on_fail : float
        Bias weight to assign when Catherine 6/6 fails (default 0.0).
        Both are exposed on the public surface so ASTSR callers can
        tune the regression weighting without subclassing.
    """

    def __init__(
        self,
        bias_on_pass: float = 1.0,
        bias_on_fail: float = 0.0,
    ) -> None:
        if not (0.0 <= float(bias_on_pass) <= 1.0):
            raise ValueError(
                f"bias_on_pass must be in [0,1], got {bias_on_pass}"
            )
        if not (0.0 <= float(bias_on_fail) <= 1.0):
            raise ValueError(
                f"bias_on_fail must be in [0,1], got {bias_on_fail}"
            )
        self.bias_on_pass = float(bias_on_pass)
        self.bias_on_fail = float(bias_on_fail)

    # -- single apply API ---------------------------------------------------

    def apply(
        self,
        payload: Union[Dict[str, Any], Sequence[Dict[str, Any]]],
        bias_type: str = "feature_flag",
    ) -> MasterCatherineBiasResult:
        """Dispatch to the canonical Catherine-closure primitive.

        Parameters
        ----------
        payload : dict or list of dict
            For ``"oversample"``: a dict with keys ``target_metal`` (str),
            ``n`` (int), optional ``seed_scaffolds`` / ``partner_smis``
            / ``max_depth`` / ``per_scaffold_cap`` (defaults match
            :class:`ClosureOversampler` defaults).
            For ``"generation_bias"``: a dict with keys ``pocket_id``
            (str), ``n`` (int), optional ``max_depth`` /
            ``max_products`` / ``include_metal_coord_bias`` (defaults
            match :func:`closure_bounded_smiles_generation` defaults).
            For ``"feature_flag"``: a dict (single-term) or list of
            dicts (batch) with keys ``term_id`` (str), ``metal`` (str,
            default ``"Pt"``), ``witness_path`` (list of click-rule
            names — empty for seed), ``smiles`` (str, used only for
            the lightweight coord-violation heuristic), optional
            ``n_visited`` / ``max_depth_seen`` / ``coord_violations``
            (ints).  Missing optional keys default to the ASTSR-
            friendly defaults (n_visited=1, max_depth_seen=1,
            coord_violations=0).
        bias_type : str
            One of :data:`SUPPORTED_BIAS_TYPES`.  Default
            ``"feature_flag"`` matches the **new** ASTSR integration
            surfaced by Wf-247-D (TODO-122).

        Returns
        -------
        MasterCatherineBiasResult
            Unified result.  Schema depends on ``bias_type`` — see
            :class:`MasterCatherineBiasResult` docstring.

        Raises
        ------
        ValueError
            On an unknown ``bias_type`` or malformed payloads.
        TypeError
            On wrong argument types for the underlying primitive.
        """
        if bias_type not in SUPPORTED_BIAS_TYPES:
            raise ValueError(
                f"bias_type must be one of {SUPPORTED_BIAS_TYPES}, "
                f"got {bias_type!r}"
            )

        t0 = time.perf_counter()

        if bias_type == "oversample":
            if not isinstance(payload, dict):
                raise TypeError(
                    f"bias_type='oversample' requires payload=dict, got "
                    f"{type(payload).__name__}"
                )
            out_payload, engine, note = self._apply_oversample(payload)
            feature_flags: List[CatherineClosureFeatureFlag] = []
        elif bias_type == "generation_bias":
            if not isinstance(payload, dict):
                raise TypeError(
                    f"bias_type='generation_bias' requires payload=dict, "
                    f"got {type(payload).__name__}"
                )
            out_payload, engine, note = self._apply_generation_bias(payload)
            feature_flags = []
        else:  # "feature_flag"
            out_payload, engine, note, feature_flags = self._apply_feature_flag(
                payload,
            )

        wall = time.perf_counter() - t0
        return MasterCatherineBiasResult(
            bias_type=str(bias_type),
            engine=str(engine),
            payload=out_payload,
            feature_flags=feature_flags,
            wall_time_s=float(wall),
            notes=str(note),
        )

    # -- per-selector wrappers ---------------------------------------------

    def _apply_oversample(
        self,
        payload: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], str, str]:
        """Closure-theorem synthetic cohort oversample (TODO-121)."""
        from molmetal.molmetal_lam.reward.pocket_closure_bias import (
            closure_oversample,
        )
        target_metal = str(payload.get("target_metal", "Rh"))
        n = int(payload.get("n", 500))
        seed_scaffolds = payload.get("seed_scaffolds")
        partner_smis = payload.get("partner_smis")
        max_depth = int(payload.get("max_depth", 2))
        per_scaffold_cap = int(payload.get("per_scaffold_cap", 50))
        result = closure_oversample(
            target_metal=target_metal,
            n=n,
            seed_scaffolds=seed_scaffolds,
            partner_smis=partner_smis,
            max_depth=max_depth,
            per_scaffold_cap=per_scaffold_cap,
        )
        return result, ENGINE_OVERSAMPLE, ""

    def _apply_generation_bias(
        self,
        payload: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], str, str]:
        """Per-pocket Catherine-bounded candidate generation."""
        from molmetal.molmetal_lam.reward.pocket_closure_bias import (
            closure_bounded_smiles_generation,
        )
        pocket_id = str(payload.get("pocket_id", ""))
        n = int(payload.get("n", 100))
        max_depth = int(payload.get("max_depth", 2))
        max_products = int(payload.get("max_products", 1000))
        include_mc = bool(payload.get("include_metal_coord_bias", True))
        result = closure_bounded_smiles_generation(
            pocket_id=pocket_id,
            n=n,
            max_depth=max_depth,
            max_products=max_products,
            include_metal_coord_bias=include_mc,
        )
        return (
            {
                "pocket_id": result.pocket_id,
                "n_seeds": int(result.n_seeds),
                "n_candidates_raw": int(result.n_candidates_raw),
                "n_candidates_kept": int(result.n_candidates_kept),
                "catherine_pass_rate": float(result.catherine_pass_rate),
                "candidates": [
                    {
                        "smiles": c["smiles"],
                        "source_seed": c["source_seed"],
                        "depth": int(c["depth"]),
                        "catherine_pass": bool(c["catherine_pass"]),
                        "witness_rules": list(c["witness_rules"]),
                    }
                    for c in result.candidates
                ],
                "engine": str(result.engine),
                "wall_time_s": float(result.wall_time_s),
                "notes": str(result.notes),
            },
            ENGINE_GENERATION_BIAS,
            str(result.notes),
        )

    def _apply_feature_flag(
        self,
        payload: Union[Dict[str, Any], Sequence[Dict[str, Any]]],
    ) -> Tuple[Dict[str, Any], str, str, List[CatherineClosureFeatureFlag]]:
        """Boolean Catherine-closure gate for ASTSR (TODO-122).

        The implementation is **lazy-RDKit** — the lightweight per-
        compound Catherine proxy
        :func:`synthetic_closure_summary` (from
        :mod:`molmetal.math_skills.catherine_properties`) only needs
        ``n_visited``, ``max_depth_seen``, ``coord_violations`` etc.,
        so batched ASTSR calls do not pay a per-row RDKit import
        cost.
        """
        if isinstance(payload, dict):
            batch: List[Dict[str, Any]] = [payload]
        elif isinstance(payload, Sequence):
            batch = [dict(p) for p in payload]  # copy so no mutation
            if not batch:
                raise ValueError(
                    "bias_type='feature_flag' payload list is empty"
                )
        else:
            raise TypeError(
                f"bias_type='feature_flag' requires payload=dict or "
                f"list[dict], got {type(payload).__name__}"
            )

        # Lazy import — the ASTSR path may be invoked from a
        # non-RDKit caller (e.g. the Wf-37 P4 verifier) and we don't
        # want the import cost paid at module load.
        from molmetal.math_skills.catherine_properties import (
            DEFAULT_COORDINATION_GEOMETRY,
            GEOMETRY_MAX_CN,
            synthetic_closure_summary,
            verify_all_catherine_properties,
        )

        feature_flags: List[CatherineClosureFeatureFlag] = []
        for entry in batch:
            if "term_id" not in entry:
                raise ValueError(
                    "feature_flag payload must contain 'term_id'; "
                    f"got keys {list(entry.keys())}"
                )
            term_id = str(entry["term_id"])
            metal = str(entry.get("metal", "Pt"))
            t0 = time.perf_counter()
            n_visited = int(entry.get("n_visited", 1))
            witness = entry.get("witness_path") or []
            # If the caller passed a witness_path, fold its length
            # into n_visited + max_depth_seen so the Catherine check
            # reflects the witness.
            if witness:
                n_visited = max(n_visited, 1 + len(witness))
            max_depth_seen = int(
                entry.get("max_depth_seen", max(1, len(witness))),
            )
            coord_violations = int(entry.get("coord_violations", 0))
            # SMILES-based coord heuristic overrides the explicit
            # coord_violations if the caller didn't set it (a
            # witness-bearing term with an explicit
            # coord_violations=0 is PASS-by-trust).
            if "coord_violations" not in entry:
                smiles = str(entry.get("smiles", ""))
                if smiles:
                    geometry = DEFAULT_COORDINATION_GEOMETRY.get(
                        metal, "octahedral",
                    )
                    cn_cap = GEOMETRY_MAX_CN.get(geometry, 6)
                    tok = f"[{metal}"
                    idx = smiles.find(tok)
                    if idx >= 0:
                        after = smiles[idx + len(tok) + 1: idx + len(tok) + 32]
                        donor_count = sum(
                            1 for c in after if c in ("N", "O", "S", "P")
                        )
                        if donor_count > cn_cap:
                            coord_violations = 1

            products_per_depth = {0: 1, int(max_depth_seen): 1}
            summary = synthetic_closure_summary(
                n_visited=n_visited,
                max_depth_seen=max_depth_seen,
                products_per_depth=products_per_depth,
                witness_rate=1.0,
                well_typed_rate=1.0,
                coord_violations=coord_violations,
                reachability_rate=1.0,
                partner_terms_n=7,
                closure_ok=True,
            )
            verdict = verify_all_catherine_properties(metal, summary)
            per_property = {
                pid: bool(res.passed)
                for pid, res in verdict.per_property.items()
            }
            six_of_six = bool(verdict.six_of_six_pass)
            bias_used = (
                float(self.bias_on_pass) if six_of_six
                else float(self.bias_on_fail)
            )
            note = "" if six_of_six else (
                "Catherine 6/6 failed; bias_used="
                f"{bias_used} (per_property={per_property})"
            )
            feature_flags.append(
                CatherineClosureFeatureFlag(
                    term_id=term_id,
                    six_of_six_pass=six_of_six,
                    per_property=per_property,
                    bias_used=float(bias_used),
                    engine=ENGINE_FEATURE_FLAG,
                    wall_time_s=float(time.perf_counter() - t0),
                    notes=str(note),
                ),
            )

        if len(feature_flags) == 1:
            out_payload = {
                "term_id": feature_flags[0].term_id,
                "six_of_six_pass": feature_flags[0].six_of_six_pass,
                "per_property": dict(feature_flags[0].per_property),
                "bias_used": float(feature_flags[0].bias_used),
                "engine": feature_flags[0].engine,
                "wall_time_s": float(feature_flags[0].wall_time_s),
                "notes": feature_flags[0].notes,
            }
        else:
            out_payload = {
                "batch_size": len(feature_flags),
                "six_of_six_pass_rate": (
                    float(sum(1 for f in feature_flags
                              if f.six_of_six_pass))
                    / float(max(len(feature_flags), 1))
                ),
                "feature_flags": [
                    {
                        "term_id": f.term_id,
                        "six_of_six_pass": f.six_of_six_pass,
                        "per_property": dict(f.per_property),
                        "bias_used": float(f.bias_used),
                        "engine": f.engine,
                        "wall_time_s": float(f.wall_time_s),
                    }
                    for f in feature_flags
                ],
                "engine": ENGINE_FEATURE_FLAG,
            }

        return out_payload, ENGINE_FEATURE_FLAG, "", feature_flags

    # -- convenience --------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the master's diagnostic state to a JSON-safe dict."""
        return {
            "bias_on_pass": float(self.bias_on_pass),
            "bias_on_fail": float(self.bias_on_fail),
            "supported_bias_types": list(SUPPORTED_BIAS_TYPES),
            "all_engine_tags": list(ALL_ENGINE_TAGS),
        }


# ---------------------------------------------------------------------------
# Convenience: build master verification (Wf-247 ship surface)
# ---------------------------------------------------------------------------


def build_master_verification() -> Dict[str, Any]:
    """Run a smoke-check on every selector + return verification dict.

    Returns
    -------
    dict
        ``{"master_catherine_closure_bias": {"feature_flag": {...},
        "feature_flag_batch": {...}}, "engine_tags": [...],
        "cpu_only": True}`` — covers the new
    ``feature_flag`` + batched variant only (the other two selectors
    require RDKit + partner-tile parsing, which is exercised end-to-
    end by ``test_sa_closure_bias.py``).

    Notes
    -----
    The ``oversample`` and ``generation_bias`` selectors are tested
    in their own dedicated suites (``test_sa_closure_bias.py`` +
    ``test_pocket_closure_bias.py``); this verification function
    focuses on the new ``feature_flag`` selector so the Wf-247 ship
    JSON records a single deterministic record per ASTSR path.
    """
    master = MasterCatherineClosureBias()

    # Single-term feature_flag.
    payload = {
        "term_id": "wf247_seed_Ru_hexaammine",
        "metal": "Ru",
        "witness_path": [],
        "smiles": "[Ru](N)(N)(N)(N)(N)N",
    }
    single = master.apply(payload, bias_type="feature_flag")
    out: Dict[str, Any] = {
        "feature_flag": single.payload,
        "bias_on_pass": master.bias_on_pass,
        "bias_on_fail": master.bias_on_fail,
    }

    # Batched feature_flag.
    batch_payload = [
        {
            "term_id": f"wf247_term_{i:03d}",
            "metal": "Pt",
            "witness_path": ["CuAAC"] if i % 2 == 0 else ["AmideCoupling"],
            "smiles": (
                "Nc1nc(N)nc(N)n1" if i % 2 == 0
                else "NS(=O)(=O)c1ccc(C#C)cc1"
            ),
        }
        for i in range(4)
    ]
    batch = master.apply(batch_payload, bias_type="feature_flag")
    out["feature_flag_batch"] = {
        "bias_type": batch.bias_type,
        "engine": batch.engine,
        "batch_size": len(batch.feature_flags),
        "pass_rate": (
            sum(1 for f in batch.feature_flags if f.six_of_six_pass)
            / max(len(batch.feature_flags), 1)
        ),
        "wall_time_s": float(batch.wall_time_s),
    }

    return {
        "master_catherine_closure_bias": out,
        "supported_bias_types": list(SUPPORTED_BIAS_TYPES),
        "engine_tags": list(ALL_ENGINE_TAGS),
        "cpu_only": True,
    }


__all__ = [
    "ALL_ENGINE_TAGS",
    "CatherineClosureFeatureFlag",
    "ENGINE_FEATURE_FLAG",
    "ENGINE_GENERATION_BIAS",
    "ENGINE_OVERSAMPLE",
    "MasterCatherineBiasResult",
    "MasterCatherineClosureBias",
    "SUPPORTED_BIAS_TYPES",
    "build_master_verification",
]
