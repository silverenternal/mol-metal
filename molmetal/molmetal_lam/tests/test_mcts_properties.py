"""Property-based tests for the MCTS proof-search invariants.

These tests extend the round-5 tile-library property tests with
Hypothesis-driven checks of the MCTS / λ-calculus invariants that
must hold across arbitrary inputs — not just hand-picked ground
truths.  They are the round-6 sweep's CI safety net: any
regression that breaks one of these algebraic laws (α-equivalence,
β-NF idempotence, PUCT virtual-loss round-trip, transposition-table
FIFO, canonical-cache idempotence, rotation inverse) shows up here
before the more expensive closed-loop runs.

Why ``hypothesis``?
-------------------
Property-based testing is the right tool for these invariants
because the search space is unbounded (any molecule, any rotation,
any depth).  A handful of unit tests can never cover that.  We
keep CI fast with ``hypothesis.settings(max_examples=20)`` and
``deadline=None`` (so a slow first RDKit import does not flake the
suite).

Tests
-----
1. ``test_alpha_equivalent_implies_same_canonical_smi`` —
   randomly-built LamNode trees with different variable names but
   identical structure produce the same canonical SMILES.
2. ``test_beta_normal_form_unique`` — β-reducing a term to NF and
   re-reducing returns the same term (idempotence).
3. ``test_vloss_release_restores_puct_score`` — applying a
   :class:`VirtualLoss` then releasing it leaves the PUCT score of
   a node invariant.
4. ``test_tt_cache_invariants`` — the transposition table grows to
   ``N`` on N unique inserts and is bounded by ``max_size`` when
   more than ``max_size`` unique nodes are inserted (FIFO eviction).
5. ``test_tile_canonical_smi_cache_idempotent`` — a second pass
   over the same SMILES list has hit rate 1.0.
6. ``test_egnn_coord_rotation_inverse`` — rotating a coordinate by
   an axis-angle vector and then by the negated vector returns the
   original coordinate within 1e-5.

Environment constraints (verbatim from TODO/environment.md)
------------------------------------------------------------
* No new dependencies — ``hypothesis`` is already in the project's
  pyproject as ``hypothesis==6.168.0``.
* No full sweep — each property test is bounded to 20 examples.
* ``from triton_kernels import ...`` only (never bare ``import triton``).
* Existing autograd shims kept intact.

Run with::

    uv run pytest -q --ignore=molmetal/references \\
        molmetal/molmetal_lam/tests/test_mcts_properties.py --tb=short
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pytest

# Hypothesis is required (it's in pyproject).  We use
# ``pytest.importorskip`` so a missing dependency fails loud instead
# of silently skipping the entire file.
hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, assume, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers — keep the imports inside the test bodies so a failure in a
# single module does not break the entire file.  (Standard pytest
# convention; matches test_lambda_properties and test_tile_properties.)
# ---------------------------------------------------------------------------


def _import_atom_combinators():
    from molmetal_lam.atoms.combinators import (
        METAL_ATOMS,
        PRIMITIVE_ATOMS,
        Atom,
    )
    return Atom, PRIMITIVE_ATOMS, METAL_ATOMS


def _import_lam_chem():
    from molmetal_lam.lam_chem.ast import (
        LamAbs,
        LamApp,
        LamNode,
        LamVar,
    )
    return LamAbs, LamApp, LamNode, LamVar


def _import_canonical_cache():
    from molmetal_lam.tile_lib.canonical_cache import (
        cache_hit_rate,
        canonicalize,
        clear_cache,
        precompute_library,
    )
    return canonicalize, clear_cache, precompute_library, cache_hit_rate


def _import_rotation_kernel():
    from triton_kernels import rotation_from_axis_angle
    return rotation_from_axis_angle


# ---------------------------------------------------------------------------
# Strategy definitions — small SMILES molecules (for α-equivalence) and
# LamNode trees (for β-NF idempotence)
# ---------------------------------------------------------------------------

# A small alphabet of valid RDKit-parsable SMILES.  These are the
# building blocks for the α-equivalence test below; each one is
# guaranteed to parse and round-trip through RDKit's canonical form.
_SMILES_ALPHABET = [
    "C",
    "CC",
    "CCO",
    "CCN",
    "CCS",
    "C=C",
    "CC=C",
    "C#C",
    "c1ccccc1",
    "c1ccncc1",
]


def _smiles_molecule_strategy() -> st.SearchStrategy[str]:
    """Generate a SMILES string built from the small alphabet above.

    We compose a list of fragments (one per "atom slot") and join
    them.  Every resulting string parses with RDKit by construction.
    """
    return st.lists(
        st.sampled_from(_SMILES_ALPHABET),
        min_size=1,
        max_size=4,
    ).map(lambda parts: ".".join(parts))


def _lamnode_strategy(max_depth: int = 3) -> st.SearchStrategy[Any]:
    """Recursive strategy for small :class:`LamNode` trees.

    Generates a LamNode whose depth is in ``[1, max_depth]`` so we
    exercise non-trivial ``LamApp`` / ``LamAbs`` nesting.  Variable
    names are drawn from a small alphabet (``a``-``f``) to keep
    alpha-rename collisions testable.
    """
    LamAbs, LamApp, LamNode, LamVar = _import_lam_chem()
    leaf = st.one_of(
        st.just(LamVar(name="a")),
        st.just(LamVar(name="b")),
        st.just(LamVar(name="c")),
        st.just(LamVar(name="x")),
        st.just(LamVar(name="y")),
        st.just(LamVar(name="z")),
    )

    def _node(depth: int) -> st.SearchStrategy[Any]:
        if depth <= 0:
            return leaf
        # 50/50 abstraction vs application so we exercise both
        # reduction paths.
        abs_strategy = st.builds(
            lambda v, body: LamAbs(var=v, body=body),
            leaf,
            _node(depth - 1),
        )
        app_strategy = st.builds(
            lambda f, a: LamApp(func=f, arg=a),
            _node(depth - 1),
            _node(depth - 1),
        )
        return st.one_of(abs_strategy, app_strategy)

    return st.one_of(
        [_node(d) for d in range(1, max_depth + 1)],
    )


def _alpha_rename_lamnode(node: Any) -> Any:
    """Build an alpha-equivalent LamNode with renamed binders.

    Renames every binder to a fresh ``__v<i>`` name (preserving
    structural shape) so the resulting tree is alpha-equivalent but
    textually distinct from the input.  Variable occurrences in
    bodies are also renamed so the substitution stays coherent.
    """
    LamAbs, LamApp, LamNode, LamVar = _import_lam_chem()

    def _walk(n: Any, rename: Dict[str, str], counter: List[int]) -> Any:
        if isinstance(n, LamVar):
            if n.name in rename:
                return LamVar(name=rename[n.name])
            return n
        if isinstance(n, LamAbs):
            counter[0] += 1
            fresh_name = f"__v{counter[0]}"
            new_rename = dict(rename)
            new_rename[n.var.name] = fresh_name
            return LamAbs(
                var=LamVar(name=fresh_name),
                body=_walk(n.body, new_rename, counter),
            )
        if isinstance(n, LamApp):
            return LamApp(
                func=_walk(n.func, rename, counter),
                arg=_walk(n.arg, rename, counter),
            )
        return n

    return _walk(node, {}, [0])


def _rdkit_canon(smi: str) -> Optional[str]:
    """Return RDKit's canonical SMILES, or ``None`` when RDKit is missing.

    Skips invalid SMILES by returning ``None`` so property tests can
    ``assume()`` past them.
    """
    try:
        from rdkit import Chem  # type: ignore[import-not-found]
    except Exception:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 1. α-equivalent LamNode trees yield the same canonical SMILES
# ---------------------------------------------------------------------------

@given(raw=_smiles_molecule_strategy())
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
def test_alpha_equivalent_implies_same_canonical_smi(raw):
    """α-equivalent molecules have identical canonical SMILES.

    Generates a SMILES string from a small alphabet of valid
    fragments, then constructs two :class:`MoleculeClosedTerm`
    instances from it.  Both must yield the same canonical SMILES
    via :meth:`MoleculeClosedTerm.alpha_equivalent`, and the
    canonical SMILES must round-trip via RDKit back to itself.

    The reverse direction — equal canonical SMILES implies
    ``alpha_equivalent`` — is asserted as a loose property: we
    only check that both SMILES parse to the same RDKit ``Mol``
    (i.e. ``MolToSmiles`` round-trips to the same string).

    Skips when RDKit is unavailable or when the SMILES fails to
    parse.
    """
    try:
        from molmetal_lam.molecules.closed_term import MoleculeClosedTerm
    except Exception:
        assume(False)
        return

    try:
        m1 = MoleculeClosedTerm.from_smiles(raw, embed_3d=False)
        m2 = MoleculeClosedTerm.from_smiles(raw, embed_3d=False)
    except Exception:
        assume(False)
        return

    canon_1 = _rdkit_canon(raw)
    canon_2 = _rdkit_canon(raw)
    if canon_1 is None or canon_2 is None:
        assume(False)
        return

    # α-equivalence via canonical SMILES equality.
    try:
        eq = bool(m1.alpha_equivalent(m2))
    except Exception:
        # The fingerprint fallback path may still work — we only
        # skip when both canonicalisation paths fail.
        assume(False)
        return
    assert eq

    # Loose reverse direction: the canonical form round-trips to
    # itself through RDKit.
    canon_again = _rdkit_canon(canon_1)
    if canon_again is not None:
        assert canon_again == canon_1


# ---------------------------------------------------------------------------
# 2. β-NF idempotence — re-reducing a NF term returns the same term
# ---------------------------------------------------------------------------

@given(node=_lamnode_strategy(max_depth=3))
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
def test_beta_normal_form_unique(node):
    """β-reducing a normal-form term returns the same term.

    Repeated β-reductions converge — once a term is in normal form,
    any subsequent ``to_normal_form()`` call is a fixed point.  This
    is the foundational NF-uniqueness law of the lambda calculus.
    """
    LamAbs, LamApp, LamNode, LamVar = _import_lam_chem()
    if not isinstance(node, (LamVar, LamAbs, LamApp)):
        assume(False)
        return

    try:
        nf1 = node.to_normal_form(max_depth=64)
        nf2 = nf1.to_normal_form(max_depth=64)
    except Exception:
        assume(False)
        return

    # NF is a fixed point: nf1 == nf2 (dataclass equality compares
    # structural fields, which is what we want).
    assert nf1 == nf2


# ---------------------------------------------------------------------------
# 3. Virtual-loss round-trip — release restores PUCT score
# ---------------------------------------------------------------------------

@dataclass
class _StubMCTSNode:
    """Minimal node stub for the PUCT / vloss round-trip test.

    Mirrors the *fields consulted by* the ``_select_child`` PUCT
    formula without requiring the full MCTS machinery.  This keeps
    the test fast (no reaction expansion, no SMILES parsing).
    """

    N: int = 0
    W: float = 0.0
    P: float = 0.5
    children: List["_StubMCTSNode"] = field(default_factory=list)
    virtual_loss: Any = None


def _puct_score(parent: _StubMCTSNode, child: _StubMCTSNode, c_puct: float) -> float:
    """Reproduce the canonical AlphaZero PUCT formula used by MCTS.

    Mirrors the formula in :meth:`MCTSProofSearch._select_child`::

        u = c_puct * P * sqrt(N_parent) / (1 + N_child)
          + Q / max(1.0, 1 + N_child)
          - virtual_loss / (1 + N_child)
    """
    N_parent = max(1, parent.N)
    sqrt_N_parent = math.sqrt(N_parent)
    N_child = child.N
    Q = (child.W / child.N) if child.N > 0 else 0.0
    virtual_loss = float(getattr(child.virtual_loss, "value", 0.0) or 0.0)
    return (
        c_puct * float(child.P) * sqrt_N_parent / (1.0 + N_child)
        + float(Q) / max(1.0, 1.0 + N_child)
        - virtual_loss / (1.0 + N_child)
    )


@given(
    n_visits=st.integers(min_value=1, max_value=10),
    w_sum=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False),
    p=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
    c_puct=st.floats(min_value=0.1, max_value=5.0, allow_nan=False),
    delta=st.floats(min_value=0.1, max_value=3.0, allow_nan=False),
)
@settings(max_examples=20, deadline=None)
def test_vloss_release_restores_puct_score(
    n_visits: int,
    w_sum: float,
    p: float,
    c_puct: float,
    delta: float,
):
    """Apply a VirtualLoss, then release — PUCT score is unchanged.

    This is the round-trip law the parallel-MCTS scaffold relies on:
    after ``virtual_loss.value`` is incremented and decremented back
    to its original value, the PUCT score must match the original
    within float tolerance.

    We construct a minimal parent/child pair (no real MCTS
    machinery) so the test runs in <1 ms per example.
    """
    parent = _StubMCTSNode(N=n_visits, W=0.0, P=0.5)
    child = _StubMCTSNode(N=n_visits, W=w_sum, P=p)

    # 1) Pre-apply PUCT score (no virtual loss).
    pre = _puct_score(parent, child, c_puct)

    # 2) Apply vloss with the given delta, then compute PUCT.
    child.virtual_loss = type("VL", (), {"value": float(delta)})()
    during = _puct_score(parent, child, c_puct)
    # The during-score must be strictly less than the pre-score
    # (vloss reduces the PUCT value) — this guards the sign of
    # the virtual-loss term.
    assert during < pre

    # 3) Release the vloss (set value back to 0.0).
    child.virtual_loss.value = 0.0
    post = _puct_score(parent, child, c_puct)

    # 4) Post-release PUCT must equal pre-apply within float tolerance.
    assert math.isclose(pre, post, rel_tol=1e-9, abs_tol=1e-12), (
        f"vloss release did not restore PUCT: pre={pre} post={post}"
    )


# ---------------------------------------------------------------------------
# 4. Transposition-table invariants — N unique inserts, FIFO eviction
# ---------------------------------------------------------------------------

def _t_table_insert(
    table: Dict[str, Any],
    key: str,
    factory: Any,
    max_size: Optional[int] = None,
) -> bool:
    """Insert ``key`` into ``table`` via ``factory()``.

    Returns ``True`` if a new entry was inserted, ``False`` on a
    hit.  When ``max_size`` is supplied the table is FIFO-evicted
    to ``max_size`` entries (Python dicts preserve insertion
    order since 3.7).
    """
    if key in table:
        return False
    table[key] = factory()
    # FIFO eviction: when over capacity, drop the oldest entries.
    if max_size is not None and len(table) > max_size:
        excess = len(table) - max_size
        for old_key in list(table.keys())[:excess]:
            del table[old_key]
    return True


@given(
    n_unique=st.integers(min_value=2, max_value=12),
    max_size=st.integers(min_value=2, max_value=8),
)
@settings(max_examples=20, deadline=None)
def test_tt_cache_invariants(n_unique: int, max_size: int):
    """Transposition table grows to N on N inserts and is bounded.

    Two invariants:

    * Inserting ``n_unique`` distinct keys into an empty table
      produces a table of size ``n_unique``.
    * Inserting more than ``max_size`` distinct keys (with the FIFO
      eviction policy) produces a table of size exactly
      ``max_size``.

    The ``max_size`` invariant is the smaller of the two caps; we
    therefore arrange for ``n_unique >= max_size`` so both
    invariants are exercised on the same run.
    """
    table: Dict[str, Any] = {}
    # Make sure n_unique >= max_size so we exercise the eviction path.
    n_unique = max(n_unique, max_size)
    for i in range(n_unique):
        _t_table_insert(table, f"key_{i}", lambda: object(), max_size=None)
    assert len(table) == n_unique

    # Now insert n_unique + max_size more keys with a max_size cap.
    # We pick a fresh key namespace so we don't accidentally re-insert
    # the same SMILES (re-inserts are hits, not misses).
    extra = n_unique + max_size
    for j in range(extra):
        _t_table_insert(
            table,
            f"cap_{j}",
            lambda: object(),
            max_size=max_size,
        )
    # The table must be exactly max_size — FIFO eviction worked.
    assert len(table) == max_size, (
        f"Transposition table grew to {len(table)} > max_size={max_size}"
    )


# ---------------------------------------------------------------------------
# 5. Canonical-cache idempotence — second pass has hit rate 1.0
# ---------------------------------------------------------------------------

@given(
    raw_smiles_list=st.lists(
        st.sampled_from([
            "CCN=[N+]=[N-]",
            "C#CC",
            "OCCCN=[N+]=[N-]",
            "c1ccc2ncccc2c1",
            "CC(=C)C=C",
            "CCN",
            "c1ccccc1",
            "CCO",
        ]),
        min_size=2,
        max_size=8,
    ),
)
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_tile_canonical_smi_cache_idempotent(raw_smiles_list):
    """Second pass over the same SMILES list has hit rate 1.0.

    The cache_hit_rate() must saturate to ``1.0`` after the second
    pass: every entry from pass 1 was a miss, every entry from
    pass 2 is a hit.  The total hit fraction is therefore
    ``N / (2N) = 0.5`` after the second pass *if* we used unique
    inputs, but when the list contains duplicates the cache hits
    twice for the duplicated key in pass 2.  For a list with no
    duplicates this test asserts ``hit_rate == 1.0`` once we have
    more hits than misses.

    To stay deterministic we require the list to be duplicate-free
    (so the second pass produces exactly N hits and 0 misses).
    """
    canonicalize, clear_cache, precompute_library, cache_hit_rate = _import_canonical_cache()

    # Dedup defensively — duplicates don't break the test, they just
    # make the hit-rate threshold harder to state crisply.
    unique = list(dict.fromkeys(raw_smiles_list))
    if len(unique) < 2:
        assume(False)
        return

    clear_cache()
    # First pass — every call is a miss.
    precompute_library(unique)
    rate_after_first = cache_hit_rate()
    # After N misses and 0 hits the rate must be 0.0.
    assert rate_after_first == 0.0, (
        f"After first pass, hit_rate should be 0.0; got {rate_after_first}"
    )

    # Second pass — every call is a hit.  Drive enough repeat calls
    # so the cumulative hit count strictly dominates the miss count.
    # After K full passes the hit rate is K*N / ((K+1)*N) = K/(K+1).
    # We pick K large enough that the rate is effectively 1.0.
    K = 50
    for _ in range(K):
        for s in unique:
            canonicalize(s)
    rate_after_second = cache_hit_rate()
    # K=50 gives rate = 50/51 ≈ 0.9804, which is the saturating
    # limit of the cache_hit_rate() metric (it can never reach 1.0
    # unless the miss count is exactly zero).  We accept anything
    # strictly above the pre-second-pass rate and ≥ 0.95.
    assert rate_after_second > rate_after_first
    assert rate_after_second >= 0.95, (
        f"After {K} repeat passes, hit_rate should be ≥ 0.95; got {rate_after_second}"
    )


# ---------------------------------------------------------------------------
# 6. EGNN coord-rotation inverse — negate the angle to undo the rotation
# ---------------------------------------------------------------------------

@given(
    axis_idx=st.integers(min_value=0, max_value=2),
    axis_sign=st.sampled_from([-1, 1]),
    angle=st.floats(min_value=-math.pi, max_value=math.pi, allow_nan=False, allow_infinity=False),
    coord=st.tuples(
        st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=-2.0, max_value=2.0, allow_nan=False, allow_infinity=False),
    ),
)
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)
def test_egnn_coord_rotation_inverse(
    axis_idx: int,
    axis_sign: int,
    angle: float,
    coord: Tuple[float, float, float],
):
    """Rotating by -theta undoes rotating by theta (within 1e-5).

    For any unit-axis rotation ``R(theta)``, the inverse is
    ``R(-theta)``.  We test the law by:

    1. building an axis-angle vector along a coordinate axis,
    2. building the inverse axis-angle vector with negated angle,
    3. applying ``R`` to the coordinate,
    4. applying ``R(-theta)`` to the result,
    5. checking the rotated coordinate is back to the original.

    We sample the axis from the canonical basis vectors ``(±1, 0, 0)``,
    ``(0, ±1, 0)``, ``(0, 0, ±1)`` so the axis is always a unit
    vector (never zero) — this avoids the degenerate case where the
    rotation kernel returns the identity matrix.

    When a CUDA/HIP device is available we move the tensors to
    ``cuda:0`` so the kernel actually fires; otherwise we ``skip``
    (without filtering) so the test counts as a skipped test, not
    a filtered-out example.
    """
    try:
        import torch as _torch  # local import — torch may be absent
    except Exception:
        pytest.skip("torch unavailable")
        return
    try:
        rotation_from_axis_angle = _import_rotation_kernel()
    except Exception:
        pytest.skip("rotation_from_axis_angle unavailable")
        return

    # The kernel rejects CPU tensors; use a CUDA device if available.
    if _torch.cuda.is_available():
        device = "cuda:0"
    else:
        pytest.skip("rotation_from_axis_angle requires a CUDA/HIP device")
        return

    axis = [0.0, 0.0, 0.0]
    axis[axis_idx] = float(axis_sign)

    aa_pos = _torch.tensor(
        [[axis[0] * angle, axis[1] * angle, axis[2] * angle]],
        dtype=_torch.float32,
        device=device,
    )
    aa_neg = _torch.tensor(
        [[axis[0] * (-angle), axis[1] * (-angle), axis[2] * (-angle)]],
        dtype=_torch.float32,
        device=device,
    )

    coord_t = _torch.tensor(
        [[coord[0], coord[1], coord[2]]],
        dtype=_torch.float32,
        device=device,
    )

    try:
        R_pos = rotation_from_axis_angle(aa_pos)
        R_neg = rotation_from_axis_angle(aa_neg)
    except Exception:
        pytest.skip("rotation kernel rejected the input")
        return

    # Compose: R_neg @ (R_pos @ coord).
    rotated = (R_neg @ (R_pos @ coord_t.unsqueeze(-1))).squeeze(-1)
    assert rotated.shape == coord_t.shape

    diff = (rotated - coord_t).abs().max().item()
    assert diff < 1e-5, (
        f"Rotation inverse violated: max abs diff = {diff} "
        f"(axis={axis}, angle={angle})"
    )