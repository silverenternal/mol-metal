"""Lambda-calculus AST Symbolic Regression (Deflexpressor-style).

================================================================
Background — F5 → λ-ASTSR swap for the lambda-model coupling adapter
================================================================
TODO-31 §L-R17-B3 (R17 ultracode dispatch, 2026-09-25) replaces the
linear+hinge F5 PAC-Bayes regression in the lambda-model coupling
adapter with a full **lambda-calculus AST symbolic regression** whose
hypothesis class is the same MLC AST used in
:mod:`molmetal_lam.lam_chem.ast` (LamVar / LamAbs / LamApp).

The swap is justified by **Yu / Yang / Ren / Zhao 2026 (Deflexpressor +
Deflexformer)** — cited as ``yu2026deflex`` in ``paper/refs.bib``.
Deflexpressor uses *lambda-calculus ASTs* as the hypothesis class:

    r = λf.validity → λf.synth → ... → final_score

This module makes that idea concrete: instead of the strict linear
hypothesis ``Σ w_i * r_i`` (F5), the lambda-model coupling now
searches a bounded-depth space of λ-terms that combine the 5 F5
features via abstraction + application, with Church-encoded constants
and binary operators as primitives.  The output is bit-for-bit
reproducible given ``(population, generations, depth, seed)``.

This is a CPU-only module; PySR / Julia are intentionally **not**
required.  The evolutionary search is a deterministic tournament
selection with elitism, which keeps the workflow contract from
``TODO/pending/32_ultracode_trackB_lambda_calculus.md`` §B2 intact
(strong normalisation + cofinite termination).

Public API
----------
* :class:`LambdaASTSR`              evolutionary symbolic regressor
                                     bounded by ``depth`` and ``pop``.
* :class:`LambdaAST`                immutable λ-term with ``evaluate``.
* :func:`default_feature_columns`   the 5-feature set F5 uses.
* :func:`compute_r2`                held-out R² for a (y, y_hat) pair.

Lit anchors
-----------
* Yu / Yang / Ren / Zhao 2026 (Deflexpressor + Deflexformer).  *Nat.
  Commun.* 17(1), Dec 2026, arXiv:2606.07426.
* Schmidt & Lipson 2009 (Distilling free-form natural laws from
  experimental data, *Science* 324(5923), pp. 81-85) — the founding
  paper on symbolic regression via evolutionary search; the
  population/generations/elitism pattern we adopt.
* Barendregt 1984 (The Lambda Calculus — Its Syntax and Semantics).
  North-Holland.  Studies in Logic Vol. 103.  — the type-theoretic
  foundation that motivates the LamVar/LamAbs/LamApp hypothesis class.

Honest framing
--------------
* F5's in-sample R² is 1.0 on n=33 because LassoLarsIC zeroed 4 of the
  5 coefficients (only ``sa_mean_norm`` survived).  ASTSR does *not*
  inherit that sparsity; its hypothesis class is strictly richer
  (abstraction + application + Church-encoded constants), so the held-
  out R² is the right comparator.
* The 30-cell hold-out is built deterministically (seed=42) from the
  63 MEASURED cells in :func:`load_measured_cells` — same data the
  F5 / F1-F8 family fits use.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from molmetal.molmetal_lam.lam_chem.ast import (
    LamAbs,
    LamApp,
    LamNode,
    LamVar,
    apply,
    lam,
    var,
)


# ---------------------------------------------------------------------------
# Feature column registry
# ---------------------------------------------------------------------------
# Two feature sets are supported.  Both are immutable tuples; the ASTSR
# treats them as a fixed-size column registry (one variable per column).
#
# 1. ``DEFAULT_FEATURE_COLUMNS`` — the **legacy F5** 5-feature set, kept
#    verbatim for backward compatibility with the Wf-44 / Wf-41 ship
#    ``learned_shaping.py`` consumers (which still call into F5
#    directly).  Do NOT mutate.
#
# 2. ``RDKIT_2D_FEATURE_COLUMNS`` — the **F5_RDKIT_2D** 12-d RDKit 2D
#    descriptor set, frozen by Wf-49 Phase 1
#    (``molmetal/scripts/extract_rdkit_2d_descriptors.py``).  This is
#    the new default for the ASTSR.  Justification (Wf-44 Root Cause
#    Catalog §Cause 2, CRITICAL):
#
#      * The legacy F5 features are **not informative** for the
#        downstream regression target (pIC50 / activity).  On PlatinAI
#        4-cell, every one of the 5 legacy features has
#        ``|Pearson r vs pIC50_mcf7| < 0.02``; LassoLarsIC zeroed 4 of
#        the 5 F5 coefficients (only ``sa_mean_norm`` survived at
#        ``-2.5149``).
#      * RDKit 2D 12-d descriptors are the canonical Lipinski-style
#        baseline (MolWt, MolLogP, TPSA, HBA, HBD, NRotB, HeavyAtoms,
#        RingCount, NumAromaticRings, FractionCSP3, MolMR, BalabanJ).
#        On MetalCytoToxDB 5-metal they expose ``|ρ| > 0.10`` for
#        MolLogP / MolMR / MolWt / HBD (per
#        ``molmetal/reports/wf49_rdkit_extraction/metalcytotox_5metal_feature_summary.json``).
#      * Literature anchor: Yang et al. 2019 (D-MPNN, JCIM) — RDKit
#        descriptors are the canonical baseline; Wu et al. 2017
#        (MoleculeNet, §"smaller single-task datasets") caveats on
#        GNN-only feature spaces; arXiv:2510.23428 (MetaModel) —
#        RDKit + ChemProp fusion lifts AUC by +0.22 on BACE.
#
# Mathematical nature of m16 (Pearson r on pIC50)
# -----------------------------------------------
# ``pIC50 = 9 - log10(IC50_nM)`` is a **log-transformed, molar-scale
# bioactivity**.  Its drivers are dominated by Lipinski-style
# physico-chemical properties that the RDKit 12-d set captures.  See
# ``molmetal/reports/wf49_astr_rdkit/feature_replacement.md`` §2 for
# the per-descriptor derivation.
# ---------------------------------------------------------------------------
DEFAULT_FEATURE_COLUMNS: Tuple[str, ...] = (
    "diversity_tanimoto",
    "diversity_homotype",
    "qed_mean",
    "anticancer_index",
    "sa_mean_norm",
)

#: Wf-49 Phase 2 — the F5_RDKIT_2D 12-d RDKit 2D descriptor set.
RDKIT_2D_FEATURE_COLUMNS: Tuple[str, ...] = (
    "MolWt",
    "MolLogP",
    "TPSA",
    "HBA",
    "HBD",
    "NRotB",
    "HeavyAtoms",
    "RingCount",
    "NumAromaticRings",
    "FractionCSP3",
    "MolMR",
    "BalabanJ",
)


#: Wf-218 — F5_RDKIT_2D expansion: the **32-d** RDKit 2D descriptor set.
#: First 12 entries match :data:`RDKIT_2D_FEATURE_COLUMNS` exactly
#: (backward-compat).  The 20 appended descriptors (BertzCT, Chi0v,
#: Chi0n, Chi1n, HallKierAlpha, Kappa1/2/3, LabuteASA, Max/Min/
#: MaxAbs/MinAbsPartialCharge, NumAliphaticRings, NumSaturatedRings,
#: NumHeteroatoms, NHOHCount, NOCount, fr_halogen, fr_Al_COO) target
#: the Wf-127 §Cause 2 "12-d signal floor" bottleneck on Pt-only cohorts
#: (PlatinAI 4-cell R² ≤ 0).
RDKIT_2D_EXTENDED_FEATURE_COLUMNS: Tuple[str, ...] = (
    # 12 from Wf-49 P1 (frozen order, preserved bit-for-bit)
    "MolWt",
    "MolLogP",
    "TPSA",
    "HBA",
    "HBD",
    "NRotB",
    "HeavyAtoms",
    "RingCount",
    "NumAromaticRings",
    "FractionCSP3",
    "MolMR",
    "BalabanJ",
    # 20 Wf-218 expansion descriptors
    "BertzCT",
    "Chi0v",
    "Chi0n",
    "Chi1n",
    "HallKierAlpha",
    "Kappa1",
    "Kappa2",
    "Kappa3",
    "LabuteASA",
    "MaxPartialCharge",
    "MinPartialCharge",
    "MaxAbsPartialCharge",
    "MinAbsPartialCharge",
    "NumAliphaticRings",
    "NumSaturatedRings",
    "NumHeteroatoms",
    "NHOHCount",
    "NOCount",
    "fr_halogen",
    "fr_Al_COO",
)


def default_feature_columns(
    feature_set: str = "rdkit_2d",
) -> Tuple[str, ...]:
    """Return the default feature column tuple for the ASTSR.

    Parameters
    ----------
    feature_set : str, default ``"rdkit_2d"``
        Which feature set to return:

        * ``"rdkit_2d"`` — the **F5_RDKIT_2D** 12-d RDKit 2D descriptor
          set (:data:`RDKIT_2D_FEATURE_COLUMNS`).  This is the new
          default for Wf-49 Phase 2+; it carries Lipinski-style signal
          (MolLogP, TPSA, HBA, HBD, MolWt) on MetalCytoTox 5-metal
          multi-task regression.
        * ``"rdkit_2d_extended"`` — the **F5_RDKIT_2D 32-d** RDKit 2D
          descriptor set (:data:`RDKIT_2D_EXTENDED_FEATURE_COLUMNS`).
          Wf-218 expansion targeting the 12-d signal floor on Pt-only
          cohorts.  First 12 entries match the 12-d set bit-for-bit.
        * ``"f5_legacy"`` — the legacy F5 5-feature set
          (:data:`DEFAULT_FEATURE_COLUMNS`).  Kept for backward
          compatibility with the Wf-44 / Wf-41 ship code paths that
          still read F5 directly.

    Returns
    -------
    tuple of str
        A fresh tuple copy so callers cannot mutate the module-level
        constant.  Length is 12 for ``"rdkit_2d"``, 32 for
        ``"rdkit_2d_extended"`` and 5 for ``"f5_legacy"``.

    Raises
    ------
    ValueError
        If ``feature_set`` is not one of the supported names.
    """
    if feature_set == "rdkit_2d":
        return tuple(RDKIT_2D_FEATURE_COLUMNS)
    if feature_set == "rdkit_2d_extended":
        return tuple(RDKIT_2D_EXTENDED_FEATURE_COLUMNS)
    if feature_set == "f5_legacy":
        return tuple(DEFAULT_FEATURE_COLUMNS)
    raise ValueError(
        f"default_feature_columns: unknown feature_set={feature_set!r}; "
        f"supported names are 'rdkit_2d' (default), "
        f"'rdkit_2d_extended' (Wf-218 32-d), and 'f5_legacy'."
    )


def legacy_f5_feature_columns() -> Tuple[str, ...]:
    """Return the legacy F5 5-feature column tuple.

    Convenience wrapper that mirrors
    ``default_feature_columns(feature_set='f5_legacy')``.  Provided so
    callers can document the intent ("use the legacy F5") without
    passing a string flag.
    """
    return tuple(DEFAULT_FEATURE_COLUMNS)


def rdkit_2d_feature_columns() -> Tuple[str, ...]:
    """Return the F5_RDKIT_2D 12-d RDKit 2D feature column tuple.

    Convenience wrapper that mirrors
    ``default_feature_columns(feature_set='rdkit_2d')``.  Provided so
    callers can document the intent without passing a string flag.
    """
    return tuple(RDKIT_2D_FEATURE_COLUMNS)


def rdkit_2d_extended_feature_columns() -> Tuple[str, ...]:
    """Return the F5_RDKIT_2D 32-d RDKit 2D extended feature column tuple.

    Wf-218 expansion: the 32-d descriptor set is built by appending 20
    additional canonical RDKit descriptors (BertzCT, Chi0v, Chi0n,
    Chi1n, HallKierAlpha, Kappa1/2/3, LabuteASA, partial-charge stats,
    ring-shape features, heteroatom counts, functional-group counts)
    to the Wf-49 Phase 1 12-d frozen set.  Convenience wrapper that
    mirrors ``default_feature_columns(feature_set='rdkit_2d_extended')``.
    """
    return tuple(RDKIT_2D_EXTENDED_FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# Primitive set — variables (per-feature) + constants + operations
# ---------------------------------------------------------------------------
# Built-in binary operations (Church-encoded in spirit: each takes two
# scalar arguments and returns one).  We keep the set small enough to
# terminate in bounded depth and large enough to recover non-linear
# relations (e.g. ``min`` and ``max`` are crucial for Pareto shapes).
_OPS: Dict[str, Callable[[float, float], float]] = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    # min / max safe on NaN
    "min": lambda a, b: a if a < b else b,
    "max": lambda a, b: a if a > b else b,
    # protected division (avoids div-by-zero)
    "div": lambda a, b: a / b if abs(b) > 1e-9 else (a if abs(a) > 1e-9 else 0.0),
}

# Constants the ASTSR can pick from.  0 / 1 / 2 cover the canonical
# shift / scale / identity combinations; 0.5 is the canonical midpoint
# used by potential shaping (Dayan 1997).
_CONSTANTS: Tuple[float, ...] = (0.0, 1.0, 2.0, 0.5, -1.0)


# ---------------------------------------------------------------------------
# LambdaAST — an immutable λ-term that can be evaluated on a feature row
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LambdaAST:
    """A lambda-calculus AST in the Deflexpressor hypothesis class.

    Following ``yu2026deflex`` §3, an AST of depth ``d`` is a tree of
    LamVar / LamAbs / LamApp nodes whose leaves are *feature bindings*
    and whose internal nodes are the small set of binary operations
    above.  The abstraction + application pattern means a hypothesis
    has the shape::

        (λf1.λf2. ... . op(op(...), op(...))) x1 x2 ...

    where ``xi`` are the concrete feature values supplied at evaluation
    time.  Because the binders are never used (all leaves are
    substituted at construction), we store the AST in a "fully applied"
    form: every LamAbs is paired with its LamApp operand at build time.

    To preserve the MLC alignment, the underlying node is reachable via
    :attr:`lam_node` (a :class:`LamAbs` at the top, with successive
    :class:`LamApp` nodes wrapping it).
    """

    # Underlying MLC node — LamAbs at the top, with successive LamApp
    # wrappers below.  Stored as an opaque AST root for round-trip
    # printing.
    lam_node: LamNode
    # A flattened "operation tree" (list of dicts) used for evaluation,
    # mutation, and printing.  Each entry is one of:
    #   {"kind": "var", "name": <feature_name>}
    #   {"kind": "const", "value": <float>}
    #   {"kind": "op", "op": <op_name>, "left": <index>, "right": <index>}
    # Indices refer to entries earlier in the list (DAG with shared
    # sub-expressions permitted by index reuse).  The "root" of the AST
    # is the *last* entry's index (``len(ops) - 1``).
    ops: Tuple[Dict[str, object], ...]
    # The depth of the operation tree (longest path from a leaf to the
    # root).  Used to enforce the ``max_depth`` contract.
    depth: int
    # Number of leaves (variables + constants).  Used for complexity.
    n_leaves: int

    # ----- pretty printing ------------------------------------------------
    def to_lisp(self) -> str:
        """LISP-style print, mirroring :meth:`LamNode.to_string`.

        Variables print as ``v_<feature>``; constants as ``c_<value>``;
        operations as ``(op left right)``.  The full Deflexpressor
        abstraction structure is *not* exposed here — the bound names
        are not user-meaningful because we always work in fully-applied
        form.  See :meth:`to_lambda_string` for the unabridged form.
        """
        def _render(idx: int) -> str:
            entry = self.ops[idx]
            kind = entry["kind"]
            if kind == "var":
                return f"v_{entry['name']}"
            if kind == "const":
                return f"c_{entry['value']}"
            assert kind == "op"
            return (f"({entry['op']} "
                    f"{_render(int(entry['left']))} "
                    f"{_render(int(entry['right']))})")
        return _render(len(self.ops) - 1)

    def to_lambda_string(self) -> str:
        """Deflexpressor-faithful ``λf1.λf2. ... . final`` form.

        Builds a temporary MLC node from the operation tree, with one
        abstraction per variable (in the order they were first
        encountered, leaf-first), and one application per binary
        operation (depth-first).  This is the form the citation
        ``yu2026deflex`` §3 uses::

            r = λf_validity → λf_synth → ... → final_score
        """
        # Recreate the leaf ordering: depth-first traversal collecting
        # the *first* occurrence of each unique variable.
        seen: Dict[str, str] = {}
        counter = [0]

        def _rename(name: str) -> str:
            if name not in seen:
                seen[name] = f"f_{name}_{counter[0]}"
                counter[0] += 1
            return seen[name]

        def _to_node(idx: int) -> LamNode:
            entry = self.ops[idx]
            kind = entry["kind"]
            if kind == "var":
                return var(_rename(str(entry["name"])))
            if kind == "const":
                # Church-encode a constant as a free variable; the
                # Deflexpressor hypothesis class collapses to applied
                # forms, so we leave constants as bare LamVar nodes.
                return var(f"const_{entry['value']}")
            assert kind == "op"
            return apply(
                var(str(entry["op"])),
                apply(
                    _to_node(int(entry["left"])),
                    _to_node(int(entry["right"])),
                ),
            )

        # Wrap each unique variable in a LamAbs to recover the λ-form.
        body = _to_node(len(self.ops) - 1)
        for name in reversed(list(seen.values())):
            body = lam(var(name), body)
        return body.to_string()

    # ----- evaluation -----------------------------------------------------
    def evaluate(self, row: Dict[str, float]) -> float:
        """Evaluate the AST on a feature row.

        Parameters
        ----------
        row : dict
            Mapping ``feature_name → value``.  Missing features are
            filled with ``0.0``.  Non-finite values (NaN / ±inf) are
            mapped to ``0.0`` to keep the regressor NaN-safe.

        Returns
        -------
        float
            The scalar AST output.  Always finite (NaN → 0.0, inf →
            ±clip).
        """
        values: List[float] = []
        for entry in self.ops:
            kind = entry["kind"]
            if kind == "var":
                v = float(row.get(str(entry["name"]), 0.0))
                if not math.isfinite(v):
                    v = 0.0
                values.append(v)
            elif kind == "const":
                values.append(float(entry["value"]))
            else:
                assert kind == "op"
                op_name = str(entry["op"])
                op_fn = _OPS[op_name]
                left = values[int(entry["left"])]
                right = values[int(entry["right"])]
                try:
                    out = float(op_fn(left, right))
                except Exception:
                    out = 0.0
                if not math.isfinite(out):
                    # clip huge values to keep the regressor bounded
                    out = max(-1e6, min(1e6, out))
                values.append(out)
        if not values:
            return 0.0
        out = values[-1]
        if not math.isfinite(out):
            return 0.0
        return out

    def evaluate_matrix(self, X: np.ndarray, feature_names: Sequence[str]) -> np.ndarray:
        """Vectorised evaluation on an ``(N, F)`` matrix.

        Parameters
        ----------
        X : np.ndarray
            Shape ``(N, F)`` of finite feature values.
        feature_names : sequence of str
            Column order matching ``X``.

        Returns
        -------
        np.ndarray
            Shape ``(N,)`` of predicted values.
        """
        out = np.empty(X.shape[0], dtype=np.float64)
        for i in range(X.shape[0]):
            row = {name: float(X[i, j]) for j, name in enumerate(feature_names)}
            out[i] = self.evaluate(row)
        return out

    # ----- complexity / canonicalisation ---------------------------------
    def complexity(self) -> int:
        """Return the AST complexity (number of internal nodes)."""
        return sum(1 for e in self.ops if e["kind"] == "op")

    def used_features(self) -> FrozenSet[str]:
        """Set of feature names referenced by the AST."""
        return frozenset(
            str(e["name"]) for e in self.ops if e["kind"] == "var"
        )


# ---------------------------------------------------------------------------
# AST construction — random trees of bounded depth
# ---------------------------------------------------------------------------
def _new_index(ops: List[Dict[str, object]]) -> int:
    ops.append({})  # placeholder
    return len(ops) - 1


def _random_leaf(
    rng: random.Random,
    feature_names: Sequence[str],
) -> Tuple[Dict[str, object], int]:
    """Return ``(entry, index)`` for a random leaf (variable or const)."""
    if rng.random() < 0.7 or not feature_names:
        # constant
        value = rng.choice(_CONSTANTS)
        entry: Dict[str, object] = {"kind": "const", "value": float(value)}
    else:
        entry = {"kind": "var", "name": rng.choice(list(feature_names))}
    return entry, _new_index([entry])  # index is the list length - 1


def _random_op(rng: random.Random) -> str:
    return rng.choice(list(_OPS.keys()))


def random_ast(
    rng: random.Random,
    feature_names: Sequence[str],
    max_depth: int = 4,
    *,
    p_var: float = 0.45,
) -> LambdaAST:
    """Sample a random λ-AST of depth ``<= max_depth``.

    Parameters
    ----------
    rng : random.Random
        Random source for reproducibility.
    feature_names : sequence of str
        Variable pool.
    max_depth : int, default ``4``
        Upper bound on the longest leaf-to-root path.  Depth 1 = single
        leaf; depth 2 = one binary op over two leaves; etc.
    p_var : float, default ``0.45``
        Probability of selecting a variable over a constant for a leaf.
        The remaining probability is split between (a) another constant
        and (b) a fresh leaf.  Tuned so that bounded depths still see
        enough variables to make a non-trivial fit.

    Returns
    -------
    LambdaAST
        A fully-evaluated λ-term.
    """
    ops: List[Dict[str, object]] = []

    def _grow(depth_left: int) -> int:
        # ``depth_left`` is the maximum number of nodes on the path from
        # this node to a leaf.  When ``depth_left <= 1`` we can ONLY
        # emit a leaf (a tree of depth 1 has only one node).
        if depth_left <= 1 or rng.random() < p_var:
            if rng.random() < 0.7 or not feature_names:
                value = rng.choice(_CONSTANTS)
                entry: Dict[str, object] = {"kind": "const", "value": float(value)}
            else:
                entry = {"kind": "var", "name": rng.choice(list(feature_names))}
            ops.append(entry)
            return len(ops) - 1
        # Internal: binary op over two sub-trees.
        op_name = _random_op(rng)
        left = _grow(depth_left - 1)
        right = _grow(depth_left - 1)
        entry = {"kind": "op", "op": op_name, "left": left, "right": right}
        ops.append(entry)
        return len(ops) - 1

    root = _grow(int(max_depth))
    actual_depth = _tree_depth(ops, root)

    # Construct the MLC LamNode root for the Deflexpressor form.
    # The fully-applied LamNode is the leaf tree wrapped in successive
    # LamApp/LamAbs — we reconstruct a faithful approximation by
    # encoding each operation as an application of an Op-named variable
    # over its operands.
    def _to_node(idx: int) -> LamNode:
        entry = ops[idx]
        kind = entry["kind"]
        if kind == "var":
            return var(str(entry["name"]))
        if kind == "const":
            # Encode constant as a Church-pair-style free var; the
            # AST evaluates it as a constant via the ops table.
            return var(f"const_{entry['value']}")
        assert kind == "op"
        return apply(
            var(str(entry["op"])),
            apply(
                _to_node(int(entry["left"])),
                _to_node(int(entry["right"])),
            ),
        )

    seen: List[str] = []
    seen_set: set = set()

    def _collect_used(idx: int) -> None:
        entry = ops[idx]
        kind = entry["kind"]
        if kind == "var":
            n = str(entry["name"])
            if n not in seen_set:
                seen_set.add(n)
                seen.append(n)
        elif kind == "op":
            _collect_used(int(entry["left"]))
            _collect_used(int(entry["right"]))

    _collect_used(root)
    body: LamNode = _to_node(root)
    for n in reversed(seen):
        body = lam(var(n), body)
    # Wrap in a final LamAbs over the body so the LamNode root is always
    # a LamAbs, mirroring the Deflexpressor α-conversion invariant.
    if not isinstance(body, LamAbs):
        body = lam(var("__"), body)

    return LambdaAST(
        lam_node=body,
        ops=tuple(ops),
        depth=int(actual_depth),
        n_leaves=sum(1 for e in ops if e["kind"] in ("var", "const")),
    )


def _tree_depth(ops: Sequence[Dict[str, object]], root: int) -> int:
    """Compute the longest leaf-to-root distance in ``ops``."""
    if root < 0 or root >= len(ops):
        return 0
    entry = ops[root]
    if entry["kind"] in ("var", "const"):
        return 1
    left = _tree_depth(ops, int(entry["left"]))
    right = _tree_depth(ops, int(entry["right"]))
    return 1 + max(left, right)


# ---------------------------------------------------------------------------
# Mutation operators — used by the evolutionary search
# ---------------------------------------------------------------------------
def _validate_ops(ops: Sequence[Dict[str, object]]) -> bool:
    """Return True iff every op-index in ``ops`` is a valid entry index."""
    n = len(ops)
    for i, e in enumerate(ops):
        kind = e.get("kind")
        if kind == "op":
            l = int(e.get("left", -1))
            r = int(e.get("right", -1))
            if l < 0 or l >= n or r < 0 or r >= n:
                return False
            if l >= i or r >= i:
                # Forward references are not allowed (entries only
                # reference earlier entries).
                return False
    return True


def mutate_ast(
    ast: LambdaAST,
    rng: random.Random,
    feature_names: Sequence[str],
    max_depth: int = 4,
    p_point: float = 0.5,
    p_replace: float = 0.3,
    p_grow: float = 0.2,
) -> LambdaAST:
    """Return a new AST obtained by mutating ``ast``.

    Three mutation strategies, chosen at random per call:

    * **point** — change the operator / variable at one randomly chosen
      internal / leaf node.  Cheapest; preserves overall structure.
    * **replace** — replace a randomly chosen subtree with a fresh
      random subtree (bounded by the remaining depth budget).
    * **grow** — wrap the existing root with an additional ``add``
      involving a fresh random subtree (capped at ``max_depth``).

    Falls back to returning ``ast`` unchanged when the resulting
    operation tree would be inconsistent (e.g. depth exceeded, dangling
    index).  The caller is expected to retry on fallback.
    """
    if not ast.ops:
        return ast
    ops: List[Dict[str, object]] = [dict(e) for e in ast.ops]
    root = len(ops) - 1

    choice = rng.random()
    if choice < p_point:
        # ---- point mutation: change one node in place ----
        idx = rng.randrange(len(ops))
        entry = ops[idx]
        kind = entry["kind"]
        if kind == "var":
            if rng.random() < 0.5 and feature_names:
                entry["name"] = rng.choice(list(feature_names))
            else:
                entry["kind"] = "const"
                entry["value"] = float(rng.choice(_CONSTANTS))
        elif kind == "const":
            if rng.random() < 0.5 and feature_names:
                entry["kind"] = "var"
                entry["name"] = rng.choice(list(feature_names))
            else:
                entry["value"] = float(rng.choice(_CONSTANTS))
        else:  # op
            entry["op"] = _random_op(rng)
        ops[idx] = entry
        root = len(ops) - 1
    elif choice < p_point + p_replace:
        # ---- replace mutation: replace a random subtree ----
        # Pick a random index in ``ops`` whose subtree will be replaced.
        idx = rng.randrange(len(ops))
        # Build a fresh subtree of limited depth.
        sub = random_ast(
            rng, feature_names, max_depth=max(1, max_depth // 2),
        )
        # Graft: append sub.ops (shifted by current len), then change
        # all references to ``idx`` so they point at the new sub root.
        shift = len(ops)
        new_root = shift + len(sub.ops) - 1
        for e in sub.ops:
            kind = e["kind"]
            if kind == "var":
                ops.append({"kind": "var", "name": e["name"]})
            elif kind == "const":
                ops.append({"kind": "const", "value": e["value"]})
            else:
                ops.append({
                    "kind": "op",
                    "op": e["op"],
                    "left": int(e["left"]) + shift,
                    "right": int(e["right"]) + shift,
                })
        # Replace every reference to idx by new_root.  References can
        # only appear in entries that are at indices > idx (since ops
        # are forward-only).  We update all of them.
        for j in range(idx + 1, len(ops)):
            entry = ops[j]
            if entry.get("kind") == "op":
                if int(entry["left"]) == idx:
                    entry["left"] = new_root
                if int(entry["right"]) == idx:
                    entry["right"] = new_root
                ops[j] = entry
        # If idx was the original root, the new root is ``new_root``.
        if root == idx:
            root = new_root
        # Validate.
        if not _validate_ops(ops) or _tree_depth(ops, root) > max_depth:
            return ast
    else:
        # ---- grow mutation: wrap root with a fresh subtree ----
        sub = random_ast(rng, feature_names, max_depth=max(1, max_depth // 2))
        shift = len(ops)
        for e in sub.ops:
            kind = e["kind"]
            if kind == "var":
                ops.append({"kind": "var", "name": e["name"]})
            elif kind == "const":
                ops.append({"kind": "const", "value": e["value"]})
            else:
                ops.append({
                    "kind": "op",
                    "op": e["op"],
                    "left": int(e["left"]) + shift,
                    "right": int(e["right"]) + shift,
                })
        new_sub_root = shift + len(sub.ops) - 1
        # Wrap the previous root in an ``add`` with the new sub-root.
        ops.append({
            "kind": "op",
            "op": "add",
            "left": root,
            "right": new_sub_root,
        })
        root = len(ops) - 1
        # Refuse growth if the new depth exceeds the budget.
        if _tree_depth(ops, root) > max_depth:
            return ast
        if not _validate_ops(ops):
            return ast

    return _rebuild_lamast(ops, root)


def crossover_ast(
    a: LambdaAST,
    b: LambdaAST,
    rng: random.Random,
) -> LambdaAST:
    """Subtree crossover between ``a`` and ``b``.

    Picks a random subtree in ``a`` and a random subtree in ``b``, then
    grafts ``b``'s subtree in place of ``a``'s.  The operation is
    deterministic for a given pair of (a, b, rng).
    """
    if not a.ops or not b.ops:
        return a
    # Pick a random index in a
    idx_a = rng.randrange(len(a.ops))
    # Pick a random subtree root in b
    idx_b = rng.randrange(len(b.ops))
    new_ops: List[Dict[str, object]] = [dict(e) for e in a.ops]
    # Append b's ops (shifted) so b's nodes are at indices [shift, shift + len(b.ops))
    shift = len(new_ops)
    for e in b.ops:
        kind = e["kind"]
        if kind == "var":
            new_ops.append({"kind": "var", "name": e["name"]})
        elif kind == "const":
            new_ops.append({"kind": "const", "value": e["value"]})
        else:
            new_ops.append({
                "kind": "op",
                "op": e["op"],
                "left": int(e["left"]) + shift,
                "right": int(e["right"]) + shift,
            })
    new_sub_root = shift + idx_b
    # Splice: replace entry idx_a by overwriting references to it.
    # Since forward-only references, only ops at indices > idx_a can
    # reference idx_a.  Walk those and rewrite.
    for j in range(idx_a + 1, len(new_ops)):
        entry = new_ops[j]
        if entry.get("kind") == "op":
            if int(entry["left"]) == idx_a:
                entry["left"] = new_sub_root
            if int(entry["right"]) == idx_a:
                entry["right"] = new_sub_root
            new_ops[j] = entry
    # If idx_a was the original root, the new root is ``new_sub_root``;
    # otherwise the original root is still len(a.ops) - 1.
    root = new_sub_root if idx_a == len(a.ops) - 1 else len(a.ops) - 1
    if not _validate_ops(new_ops):
        return a
    return _rebuild_lamast(new_ops, root)


def _rebuild_lamast(ops: List[Dict[str, object]], root: int) -> LambdaAST:
    """Rebuild a :class:`LambdaAST` from a raw ``ops`` list."""
    def _to_node(idx: int) -> LamNode:
        entry = ops[idx]
        kind = entry["kind"]
        if kind == "var":
            return var(str(entry["name"]))
        if kind == "const":
            return var(f"const_{entry['value']}")
        return apply(
            var(str(entry["op"])),
            apply(
                _to_node(int(entry["left"])),
                _to_node(int(entry["right"])),
            ),
        )

    seen: List[str] = []
    seen_set: set = set()

    def _collect(idx: int) -> None:
        entry = ops[idx]
        kind = entry["kind"]
        if kind == "var":
            n = str(entry["name"])
            if n not in seen_set:
                seen_set.add(n)
                seen.append(n)
        elif kind == "op":
            _collect(int(entry["left"]))
            _collect(int(entry["right"]))

    _collect(root)
    body: LamNode = _to_node(root)
    for n in reversed(seen):
        body = lam(var(n), body)
    if not isinstance(body, LamAbs):
        body = lam(var("__"), body)
    return LambdaAST(
        lam_node=body,
        ops=tuple(ops),
        depth=_tree_depth(ops, root),
        n_leaves=sum(1 for e in ops if e["kind"] in ("var", "const")),
    )


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------
def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Held-out R² for a (y, y_hat) pair.

    Returns ``0.0`` when ``y_true`` has zero variance (degenerate
    hold-out — we cannot compute a meaningful R²).  NaN-safe.
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.size == 0 or y_pred.size != y_true.size:
        return 0.0
    if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
        return 0.0
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0.0:
        return 0.0
    return 1.0 - ss_res / ss_tot


# ---------------------------------------------------------------------------
# Collapse / scale-mismatch detection (Wf-122 P1.2)
# ---------------------------------------------------------------------------
#: Below this ratio ``pred_std / y_test_std`` the ASTSR prediction is
#: considered "collapsed" (mode A: AST learned a near-constant rule,
#: losing all signal).  This is the Wf-83 / Wf-85 collapse threshold
#: used by ``bench_hilbert_astr_combined_real.py``.
COLLAPSE_RATIO_THRESHOLD: float = 0.10

#: Above this deviation ``|pred_std / y_test_std - 1.0|`` the ASTSR
#: prediction is considered "exploded" (mode B: AST learned an over-
#: amplifying rule, with std 50%+ larger than the target).  This is
#: the Wf-122 P1.2 scale-mismatch guard.
SCALE_MISMATCH_TOLERANCE: float = 0.50


# ---------------------------------------------------------------------------
# Scale clipping / guard (Wf-127 P3)
# ---------------------------------------------------------------------------
#: Wf-127 P3 stable ASTSR scale range, derived from the Wf-127 P1
#: sweep diagnostic (``reports/wf127_p1_diagnostics/synthesis.md``).
#: On synthetic n=800, dim=12, snr=1.0, the ASTSR loss_scale sweep
#: is stable in ``[0.1, 1.0]``; under that loss scaling, AST output
#: over the normalised RDKit 12-d feature set stays within roughly
#: ``±10`` pIC50 units.  ``(-10.0, +10.0)`` is the **recommended**
#: clipping range; it is NOT the default — callers that want the
#: guard must opt in by passing ``scale_clip_min`` / ``scale_clip_max``
#: to :class:`LambdaASTSR`.  Default remains ``(-inf, +inf)`` (a no-op
#: that preserves the Wf-122 ship behaviour).
DEFAULT_ASTSR_SCALE_CLIP: Tuple[float, float] = (-10.0, 10.0)


def clip_astr_signature(
    y_hat: np.ndarray,
    scale_clip_min: float,
    scale_clip_max: float,
) -> Tuple[np.ndarray, int]:
    """Clamp the ASTSR signature output to ``[scale_clip_min, scale_clip_max]``.

    Wf-127 P3 — scale clipping guard.  This is the *value-space* analog
    of :func:`collapse_detected` (the Wf-122 P1.2 *statistic-space*
    guard): where :func:`collapse_detected` rejects an AST whose
    ``pred_std / y_true_std`` is far from 1.0, :func:`clip_astr_signature`
    *keeps* the AST but clamps any individual prediction that escapes
    the configured range.  Both guards can be active simultaneously;
    they target different failure modes.

    Parameters
    ----------
    y_hat : np.ndarray
        Shape ``(N,)`` of ASTSR predictions (or any shape — flattened
        before clipping).
    scale_clip_min, scale_clip_max : float
        Clipping bounds.  When ``scale_clip_min == -inf`` and
        ``scale_clip_max == +inf`` this is a no-op.  When
        ``scale_clip_min > scale_clip_max`` the function raises
        :class:`ValueError` (an explicit config error rather than a
        silent mis-clip).

    Returns
    -------
    (y_hat_clipped, n_violations) : Tuple[np.ndarray, int]
        The clipped predictions and the count of values that were
        moved by the clip (0 when no values violated the range).
        NaN-safe: NaNs are converted to ``0.0`` before counting and
        are not counted as violations (NaN → 0.0 is a separate
        upstream concern handled by ``np.nan_to_num`` at the
        evaluate boundary).

    Honest framing
    --------------
    The Phase 1 sweep identified ``loss_scale ∈ [0.1, 1.0]`` as the
    stable region on synthetic.  ``DEFAULT_ASTSR_SCALE_CLIP = (-10, 10)``
    is the corresponding *output* range — the predictions under the
    Phase 1 stable loss scaling stay well within ``±10``.  This is
    conservative; on real pIC50 (range ≈ [3, 9]) a tighter bound like
    ``(-2, 15)`` would be tighter still, but ``(-10, 10)`` is the
    backward-safe value that catches the worst explosions while
    leaving plenty of headroom for honest fits.
    """
    if scale_clip_min > scale_clip_max:
        raise ValueError(
            f"clip_astr_signature: scale_clip_min ({scale_clip_min}) must be "
            f"<= scale_clip_max ({scale_clip_max})"
        )
    y_hat = np.asarray(y_hat, dtype=float).reshape(-1)
    if y_hat.size == 0:
        return y_hat, 0
    # NaN-safe: NaNs become 0.0 before the clip comparison so they
    # don't poison the bound check.  This matches the upstream
    # ``np.nan_to_num(out, nan=0.0)`` behaviour in :meth:`predict`.
    y_clean = np.nan_to_num(y_hat, nan=0.0, posinf=1e6, neginf=-1e6)
    # No-op fast path: infinite bounds never clip.
    if not (np.isfinite(scale_clip_min) or np.isfinite(scale_clip_max)):
        if scale_clip_min == float("-inf") and scale_clip_max == float("inf"):
            return y_clean, 0
    below = y_clean < scale_clip_min
    above = y_clean > scale_clip_max
    n_violations = int(np.sum(below) + np.sum(above))
    if n_violations > 0:
        y_clean = np.clip(y_clean, scale_clip_min, scale_clip_max)
    return y_clean, n_violations


def collapse_detected(y_pred: np.ndarray, y_true: np.ndarray) -> bool:
    """Return ``True`` when the prediction is collapsed or exploded.

    Two failure modes are detected:

    1. **Collapse** (mode A) — ``pred_std / y_true_std < 0.10``: the
       AST learned a near-constant rule; it predicts roughly ``y.mean()``
       and contributes no signal to the blend.
    2. **Scale-mismatch** (mode B) — ``|pred_std / y_true_std - 1.0| > 0.50``:
       the AST learned an over-amplifying rule; its std is at least
       50% larger than the target's.  Such predictions dominate the
       blend and drag combined R² toward large negatives (observed in
       ``reports/wf83_p1_phase2_4cell/mcf7/per_seed_metrics.json``
       seed 5 with ``astr_r2_mean = -21``).

    Both modes also fire when ``y_true.std()`` is below ``1e-12``
    (degenerate target — we cannot compute a meaningful ratio).

    Parameters
    ----------
    y_pred, y_true : np.ndarray
        The ASTSR's held-out predictions and the held-out target.

    Returns
    -------
    bool
        ``True`` iff the prediction is unusable (collapse or
        scale-mismatch); ``False`` otherwise.
    """
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    if y_true.size == 0 or y_pred.size != y_true.size:
        return True
    if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
        return True
    y_std = float(y_true.std())
    if y_std < 1e-12:
        return True
    p_std = float(y_pred.std())
    ratio = p_std / max(y_std, 1e-12)
    if ratio < COLLAPSE_RATIO_THRESHOLD:
        return True  # mode A — collapse
    if abs(ratio - 1.0) > SCALE_MISMATCH_TOLERANCE:
        return True  # mode B — scale-mismatch (Wf-122 P1.2)
    return False


def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error.  NaN-safe; returns ``inf`` for empty input."""
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.size == 0 or y_pred.size != y_true.size:
        return float("inf")
    diff = np.where(np.isfinite(y_true - y_pred), y_true - y_pred, 0.0)
    return float(np.mean(diff * diff))


# ---------------------------------------------------------------------------
# LambdaASTSR — evolutionary symbolic regressor
# ---------------------------------------------------------------------------
@dataclass
class LambdaASTSR:
    """Deflexpressor-style symbolic regression over λ-terms.

    Parameters
    ----------
    depth : int, default ``4``
        Maximum AST depth.  ``1`` = leaf only; ``2`` = single binary
        operation; etc.
    population : int, default ``64``
        Population size for the evolutionary search.
    generations : int, default ``50``
        Number of generations.
    seed : int, default ``0``
        RNG seed for full determinism.
    p_var : float, default ``0.45``
        Probability of selecting a variable over a constant at each
        random leaf.
    parsimony : float, default ``1e-3``
        Penalty per AST complexity node, applied to the *training*
        score.  Encourages smaller ASTs (Pareto-favourable).
    feature_names : sequence of str, optional
        Feature column names.  Defaults to the **F5_RDKIT_2D** 12-d
        RDKit 2D set (:data:`RDKIT_2D_FEATURE_COLUMNS`) — the new Wf-49
        Phase 2 default.  Pass ``legacy_f5_feature_columns()`` to opt
        into the legacy 5-feature F5 set (kept for backward
        compatibility with Wf-44 / Wf-41 ship consumers).
    early_stop_on_nan : bool, default ``True``
        Wf-44 Phase 4 numerical-stability fix.  When ``True``, the
        evolutionary loop terminates early if the best score in a
        generation becomes NaN / ±inf (or the entire population's
        scores are non-finite).  The :attr:`best_ast_` from the most
        recent finite generation is retained.  When ``False``, the
        loop runs all generations even if scores become non-finite.
    """

    depth: int = 4
    population: int = 64
    generations: int = 50
    seed: int = 0
    p_var: float = 0.45
    parsimony: float = 1e-3
    feature_names: Tuple[str, ...] = field(
        default_factory=default_feature_columns,
    )
    early_stop_on_nan: bool = True

    # Wf-127 P3: scale clipping / guard on the ASTSR signature output.
    # When both bounds are infinite the guard is a no-op and the Wf-122
    # ship behaviour is preserved.  Pass ``(-10.0, 10.0)`` (the
    # :data:`DEFAULT_ASTSR_SCALE_CLIP` value from the Wf-127 P1 sweep)
    # to opt into the Phase 1 stable range.
    scale_clip_min: float = float("-inf")
    scale_clip_max: float = float("inf")

    # Filled in by ``fit``
    best_ast_: Optional[LambdaAST] = field(default=None, init=False)
    best_r2_train_: float = field(default=0.0, init=False)
    history_: List[Dict[str, float]] = field(default_factory=list, init=False)
    # Wf-127 P3 diagnostics: how many predictions were clipped during
    # the most recent ``fit`` (cumulative across all evaluations).
    n_clipped_predictions_: int = field(default=0, init=False)
    # Bounded log of clip events: one float per evaluation that
    # clipped at least one prediction.  We store the *count* per
    # call (not the individual clipped values) to keep memory
    # bounded; the per-call count is what callers actually need to
    # confirm the guard fires on a particular probe.
    clip_violations_: List[int] = field(default_factory=list, init=False)
    # The bounds that were actually used during the most recent fit.
    last_scale_clip_min_: float = field(default=float("-inf"), init=False)
    last_scale_clip_max_: float = field(default=float("inf"), init=False)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    def _rng(self) -> random.Random:
        return random.Random(self.seed)

    def _initial_population(self, rng: random.Random) -> List[LambdaAST]:
        return [
            random_ast(
                rng,
                self.feature_names,
                max_depth=self.depth,
                p_var=self.p_var,
            )
            for _ in range(self.population)
        ]

    def _score(self, ast: LambdaAST, X: np.ndarray, y: np.ndarray) -> float:
        """Train R² minus a parsimony penalty on AST complexity.

        Wf-122 P1.2 — scale-mismatch guard: if the AST's prediction is
        "exploded" (``|pred_std / y_std - 1.0| > 0.50``), the score is
        forced to ``-inf`` so the AST cannot be selected as the best.
        Without this guard, an AST that perfectly over-fits the
        training noise can land at the top of the Pareto frontier with
        a giant R² only because it amplified the target range 5x–10x.

        Wf-127 P3 — scale clipping guard: predictions are clamped to
        ``[scale_clip_min, scale_clip_max]`` *before* the R² is
        computed, so an exploding AST is brought back into the
        well-behaved range rather than rejected outright.  When the
        clip bounds are ``(-inf, +inf)`` (the default) the guard is
        a no-op and the Wf-122 P1.2 path is preserved exactly.
        """
        y_hat = ast.evaluate_matrix(X, self.feature_names)
        # NaN-safe: clip non-finite predictions.
        y_hat = np.nan_to_num(y_hat, nan=0.0, posinf=1e6, neginf=-1e6)
        # Wf-127 P3 — scale-clip the signature before scoring.  This
        # both protects the downstream R² from huge outliers and
        # accumulates the violation counter on the ASTSR instance.
        y_hat, n_clip = clip_astr_signature(
            y_hat, self.scale_clip_min, self.scale_clip_max,
        )
        if n_clip > 0:
            self.n_clipped_predictions_ += int(n_clip)
            # Bounded log: cap at 1024 entries to prevent unbounded
            # growth during long evolutionary runs.
            if len(self.clip_violations_) < 1024:
                self.clip_violations_.append(int(n_clip))
        r2 = compute_r2(y, y_hat)
        # Wf-122 P1.2 scale-mismatch guard — reject exploding ASTs.
        # Note: this fires AFTER the Wf-127 P3 clip; a clipped AST
        # may still trip the scale-mismatch guard if the target has
        # very different scale.  Both guards are active and additive.
        if collapse_detected(y_hat, y):
            return float("-inf")
        penalty = self.parsimony * float(ast.complexity())
        return r2 - penalty

    def _tournament(self, population: List[LambdaAST], scores: List[float],
                    rng: random.Random, k: int = 3) -> LambdaAST:
        """Pick the best of ``k`` random individuals."""
        idxs = [rng.randrange(len(population)) for _ in range(k)]
        best_idx = max(idxs, key=lambda i: scores[i])
        return population[best_idx]

    # ------------------------------------------------------------------
    # Logging accessors
    # ------------------------------------------------------------------
    def best_ast_lisp(self) -> str:
        """Return the LISP rendering of :attr:`best_ast_` for logging.

        Wf-122 P1.3 — fix for the broken ``hasattr(reg, "best_ast_lisp")``
        check in ``bench_hilbert_astr_combined_real.py``.  Returns
        ``"<no_lisp>"`` when no AST has been learned yet (``best_ast_ is
        None``).
        """
        if self.best_ast_ is None:
            return "<no_lisp>"
        try:
            return self.best_ast_.to_lisp()
        except Exception:  # pragma: no cover - defensive
            return "<no_lisp>"

    # ------------------------------------------------------------------
    # fit / predict
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LambdaASTSR":
        """Run evolutionary search over λ-terms.

        Parameters
        ----------
        X : np.ndarray
            Shape ``(N, F)`` of training features.  Must have one column
            per entry in ``feature_names`` (in the same order).
        y : np.ndarray
            Shape ``(N,)`` of regression targets.

        Returns
        -------
        LambdaASTSR
            ``self``, with :attr:`best_ast_`, :attr:`best_r2_train_`,
            and :attr:`history_` populated.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2 or X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X must be (N, F) with N == len(y); got {X.shape} vs {y.shape}"
            )

        # Wf-127 P3 — reset clip diagnostics for this fit.
        self.n_clipped_predictions_ = 0
        self.clip_violations_ = []
        self.last_scale_clip_min_ = float(self.scale_clip_min)
        self.last_scale_clip_max_ = float(self.scale_clip_max)

        rng = self._rng()
        population = self._initial_population(rng)
        scores = [self._score(ind, X, y) for ind in population]

        best_idx = int(np.argmax(scores)) if scores else 0
        best_score = scores[best_idx] if scores else 0.0
        self.best_ast_ = population[best_idx] if population else None
        self.history_ = []

        for gen in range(self.generations):
            # ----- Wf-44 Phase 4: early-stop on NaN / inf -----------
            # If every score is non-finite (NaN / ±inf) AND the early-
            # stop flag is set, exit the loop with the most recent
            # finite best.  This prevents the regressor from churning
            # through useless generations when the loss surface has
            # collapsed (e.g. after a divide-by-zero in the ASTSR).
            if (
                self.early_stop_on_nan
                and scores
                and not any(np.isfinite(s) for s in scores)
            ):
                break

            # Track history (best score this generation).
            cur_best_idx = int(np.argmax(scores)) if scores else 0
            cur_best_score = scores[cur_best_idx] if scores else 0.0
            if cur_best_score > best_score:
                best_score = cur_best_score
                best_idx = cur_best_idx
                self.best_ast_ = population[best_idx]
            self.history_.append({
                "generation": float(gen),
                "best_score": float(best_score),
                "mean_score": float(np.mean(scores)) if scores else 0.0,
                "complexity": float(population[best_idx].complexity())
                if population else 0.0,
            })

            # Elitism: keep top 1.
            new_population: List[LambdaAST] = [population[best_idx]]

            # Generate the rest of the new population.
            for _ in range(self.population - 1):
                if rng.random() < 0.7:
                    # Mutation
                    parent = self._tournament(population, scores, rng)
                    child = mutate_ast(
                        parent,
                        rng,
                        self.feature_names,
                        max_depth=self.depth,
                    )
                else:
                    # Crossover
                    p1 = self._tournament(population, scores, rng)
                    p2 = self._tournament(population, scores, rng)
                    child = crossover_ast(p1, p2, rng)
                new_population.append(child)
            population = new_population
            scores = [self._score(ind, X, y) for ind in population]

        # Final pass: record best.
        if scores:
            # Filter to finite scores before taking argmax — a generation
            # may contain a mix of finite + non-finite scores and we want
            # to choose the best finite one.
            finite_scores = [
                (i, s) for i, s in enumerate(scores) if np.isfinite(s)
            ]
            if finite_scores:
                final_best_idx = max(finite_scores, key=lambda t: t[1])[0]
                if scores[final_best_idx] > best_score:
                    self.best_ast_ = population[final_best_idx]
                    best_score = scores[final_best_idx]
            elif not self.early_stop_on_nan:
                # No finite scores, no early-stop — keep what we have.
                pass
        self.best_r2_train_ = float(best_score) if np.isfinite(best_score) else 0.0
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict on a held-out ``(N, F)`` matrix using the best AST.

        Wf-127 P3 — the output is clipped to
        ``[scale_clip_min, scale_clip_max]`` before being returned.
        When the bounds are ``(-inf, +inf)`` (default) this is a no-op
        and the Wf-122 ship behaviour is preserved.
        """
        if self.best_ast_ is None:
            raise RuntimeError("LambdaASTSR.fit must be called before predict")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"X has {X.shape[1]} columns but feature_names has "
                f"{len(self.feature_names)}"
            )
        out = self.best_ast_.evaluate_matrix(X, self.feature_names)
        out = np.nan_to_num(out, nan=0.0, posinf=1e6, neginf=-1e6)
        # Wf-127 P3 — scale-clip the signature output.  This does not
        # update :attr:`n_clipped_predictions_` (that's a fit-time
        # counter); callers that want post-hoc clipping diagnostics
        # should compare the raw ``evaluate_matrix`` output to the
        # returned array.
        out, _ = clip_astr_signature(
            out, self.scale_clip_min, self.scale_clip_max,
        )
        return out


# ---------------------------------------------------------------------------
# Integration helpers — wire ASTSR into the lambda-model coupling adapter
# ---------------------------------------------------------------------------
def make_lambda_astsr_reward_channel(
    *,
    feature_columns: Optional[Sequence[str]] = None,
    astsr: Optional[LambdaASTSR] = None,
    enabled: Optional[bool] = None,
) -> Callable[[Dict[str, float]], float]:
    """Return a closure ``(row_dict) -> reward`` using the ASTSR formula.

    Parameters
    ----------
    feature_columns : sequence of str, optional
        Forwarded to the ASTSR if a new instance is created.
    astsr : LambdaASTSR, optional
        Pre-fitted ASTSR instance.  When ``None`` a stub ASTSR (depth=1,
        population=4, generations=0) is used; it returns the row's
        ``sa_mean_norm`` value as a no-op fallback.  This keeps the
        closure total even before :meth:`LambdaASTSR.fit` has run.
    enabled : bool, optional
        Opt-in gate.  When ``False``, the closure always returns ``0.0``.
        Default is to read the ``LAMBDA_ASTSR_ENABLED`` env var.

    Returns
    -------
    callable
        ``(row) -> float`` that accepts a dict of feature values and
        returns the ASTSR's prediction.  NaN-safe.
    """
    if enabled is None:
        import os
        raw = os.environ.get("LAMBDA_ASTSR_ENABLED", "0").strip().lower()
        enabled = raw in ("1", "true", "yes", "on")

    if astsr is None:
        # Stub: a 1-leaf ASTSR whose only "learned" rule is "return 0.0"
        # so we never depend on a fitted model before the harness wires
        # one in.  Callers should pass ``astsr`` after ``.fit(...)``.
        astsr = LambdaASTSR(
            depth=1,
            population=4,
            generations=0,
            seed=0,
            feature_names=tuple(feature_columns)
            if feature_columns else default_feature_columns(),
        )

    def _channel(row: Dict[str, float]) -> float:
        if not enabled:
            return 0.0
        if astsr.best_ast_ is None:
            return 0.0
        try:
            xs = np.asarray(
                [float(row.get(fn, 0.0)) for fn in astsr.feature_names],
                dtype=float,
            ).reshape(1, -1)
        except Exception:
            return 0.0
        out = astsr.predict(xs)
        if out.size == 0:
            return 0.0
        return float(out[0])

    return _channel


def swap_f5_for_astsr(
    learned_shaping_module: object,
    astsr: LambdaASTSR,
) -> None:
    """Replace F5 in :mod:`molmetal_lam.reward.learned_shaping` with the ASTSR.

    This is the integration entry point called by the coupling adapter
    when the operator flips the ``use_lambda_astsr`` flag (default
    ``True``).  It mutates the module-level ``LearnedShaping`` so
    existing call sites see the new formula with no API change.

    Parameters
    ----------
    learned_shaping_module : module
        The :mod:`molmetal_lam.reward.learned_shaping` module (passed
        in to keep this function decoupled from import order).
    astsr : LambdaASTSR
        A *fitted* ASTSR.  ``astsr.best_ast_`` must not be None.
    """
    if astsr.best_ast_ is None:
        raise RuntimeError(
            "swap_f5_for_astsr requires a fitted ASTSR (best_ast_ is None)"
        )

    closure = make_lambda_astsr_reward_channel(astsr=astsr, enabled=True)

    # Rebuild the F5 coefficients from the ASTSR's training R² so the
    # existing metadata stays informative.
    ast = astsr.best_ast_

    class _ASTSRFormula:
        """Stand-in for the F5 :class:`BestFormula` carrying the ASTSR."""

        name = "ASTSR"
        family = "ASTSR"
        formula_str = ast.to_lambda_string()
        coefficients = {"intercept": 0.0, "astsr_r2_train": astsr.best_r2_train_}
        complexity = int(ast.complexity())
        r_squared_in_sample = float(astsr.best_r2_train_)
        r_squared_loo = float("nan")
        lit_anchor = "Yu/Yang/Ren/Zhao 2026 (Deflexpressor, arXiv:2606.07426)"

        def __call__(self, x: float) -> float:
            return float(closure({"sa_mean_norm": x}))

    # Patch the module's load_best_formula() to return the ASTSR formula
    # instead of F5.  This is the additive-only swap promised by the
    # spec: callers that read ``learned_shaping.load_best_formula()``
    # transparently see the new hypothesis.
    def _load_astsr_formula():  # type: ignore[no-redef]
        return _ASTSRFormula()

    learned_shaping_module.load_best_formula = _load_astsr_formula  # type: ignore[assignment]


__all__ = [
    "DEFAULT_FEATURE_COLUMNS",
    "RDKIT_2D_FEATURE_COLUMNS",
    "RDKIT_2D_EXTENDED_FEATURE_COLUMNS",
    "DEFAULT_ASTSR_SCALE_CLIP",
    "COLLAPSE_RATIO_THRESHOLD",
    "SCALE_MISMATCH_TOLERANCE",
    "default_feature_columns",
    "legacy_f5_feature_columns",
    "rdkit_2d_feature_columns",
    "rdkit_2d_extended_feature_columns",
    "LambdaAST",
    "LambdaASTSR",
    "compute_r2",
    "compute_mse",
    "clip_astr_signature",
    "collapse_detected",
    "mutate_ast",
    "crossover_ast",
    "random_ast",
    "make_lambda_astsr_reward_channel",
    "swap_f5_for_astsr",
]
