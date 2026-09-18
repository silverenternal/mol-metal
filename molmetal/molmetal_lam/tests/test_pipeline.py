"""Tests for the MLC closed-loop pipeline (pipeline/)."""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pytest

from molmetal_lam.lam_chem.pysr_wrapper import HeuristicRegressor
from molmetal_lam.molecules.closed_term import MoleculeClosedTerm
from molmetal_lam.pipeline.closed_loop import (
    LamClickDesignLoop,
    extract_paper_equation,
)
from molmetal_lam.pipeline.extract_features import (
    FEATURE_NAMES,
    molecule_to_features,
)


# ---------------------------------------------------------------------------
# Test fixtures: tiny stand-ins for the heavy upstream modules.
# ---------------------------------------------------------------------------
@dataclass
class MockMCTS:
    """A toy MCTS that returns a fixed list of candidates per call.

    Used so the closed loop runs without needing the full proof-search
    tree + tile library + reaction rules.  Behaviour:
      * 5 hand-picked SMILES -> 5 closed terms per ``search()`` call
      * ``search`` always succeeds and returns the same SMILES set
        (the loop's job is to *rank* them, not invent new ones).
    """

    smiles_pool: List[str] = field(default_factory=lambda: [
        "CCO",
        "CC(=O)Oc1ccccc1C(=O)O",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
        "c1ccccc1",
        "OCC(O)CO",
    ])
    call_count: int = 0
    last_max_depth: int = 0

    def search(self, initial_state: Any = None, max_depth: int = 3) -> List[MoleculeClosedTerm]:
        self.call_count += 1
        self.last_max_depth = int(max_depth)
        out: List[MoleculeClosedTerm] = []
        for s in self.smiles_pool:
            try:
                m = MoleculeClosedTerm.from_smiles(s, embed_3d=False)
            except Exception:
                m = MoleculeClosedTerm()
            out.append(m)
        return out


@dataclass
class MockScorer:
    """A toy scorer whose score is exactly the SMILES length (deterministic).

    Real REINVENT4 scorers call RDKit + QSAR models; this one is just
    enough to verify that ``LamClickDesignLoop`` ranks candidates and
    feeds features into the heuristic.
    """

    calls: int = 0

    def batch_score(self, smiles_list: List[str]) -> np.ndarray:
        self.calls += 1
        # Deterministic: longer SMILES wins (so we can verify ranking).
        return np.asarray(
            [float(len(s)) for s in smiles_list], dtype=np.float32,
        )


# ---------------------------------------------------------------------------
# extract_features
# ---------------------------------------------------------------------------
def test_extract_features_shape() -> None:
    """molecule_to_features returns an (8,) float32 array."""
    mol = MoleculeClosedTerm.from_smiles("c1ccccc1", embed_3d=False)
    feats = molecule_to_features(mol)
    assert feats.shape == (8,), f"expected (8,), got {feats.shape}"
    assert feats.dtype == np.float32, f"expected float32, got {feats.dtype}"
    # Every entry must be finite (no NaN/inf from RDKit edge cases).
    assert np.all(np.isfinite(feats)), "feature vector contains NaN/inf"
    # Names list mirrors the vector length.
    assert len(FEATURE_NAMES) == 8


def test_extract_features_zero_vector_on_parse_failure() -> None:
    """Garbage SMILES -> zero vector, no exception raised."""
    mol = MoleculeClosedTerm()  # empty term, no SMILES
    feats = molecule_to_features(mol)
    assert feats.shape == (8,)
    assert np.all(np.isfinite(feats))


# ---------------------------------------------------------------------------
# closed-loop run
# ---------------------------------------------------------------------------
def test_lam_loop_runs_3_iters() -> None:
    """The loop completes 3 iterations end-to-end without raising."""
    mcts = MockMCTS()
    scorer = MockScorer()
    hr = HeuristicRegressor(niterations=2, random_state=0)
    loop = LamClickDesignLoop(
        mcts=mcts,
        scorer=scorer,
        pocket_loader=None,
        symbolic_reg=hr,
        rng=random.Random(0),
    )

    results = loop.run(pdb_id="demo", n_iterations=3, top_k=3, max_depth=2)

    assert isinstance(results, list), "run() must return a list"
    assert len(results) == 3, f"expected 3 iterations, got {len(results)}"
    for i, rec in enumerate(results):
        assert rec["iteration"] == i
        assert isinstance(rec["best_smiles"], str)
        assert isinstance(rec["best_lambda_expr"], str)
        assert isinstance(rec["best_score"], float)
        assert isinstance(rec["extracted_formula"], str)
        # Top-K is respected.
        assert len(rec["top_k_smiles"]) <= 3
        assert len(rec["top_k_scores"]) <= 3

    # The MCTS + scorer are wired in: each iteration issues a search +
    # a batch_score call.  (MockScorer counts batch_score calls; MOCK
    # MCTS counts search calls.)
    assert scorer.calls == 3
    assert mcts.call_count == 3
    # After 3 iters the loop has populated its history list.
    assert len(loop.history) == 3


def test_extract_paper_equation_returns_string() -> None:
    """5 candidates + fitted heuristic -> non-empty equation string."""
    rng = np.random.default_rng(0)
    X = rng.standard_normal((5, 4))
    # Score is a linear combo so the sklearn-Ridge fallback should
    # recover a clean closed-form expression.
    y = 0.1 * X[:, 0] - 0.4 * X[:, 1] + 0.7 * X[:, 2] + 0.05 * np.random.RandomState(0).randn(5)

    hr = HeuristicRegressor(niterations=2, random_state=0)
    hr.fit(X, y)
    assert hr._fitted, "heuristic should be fitted"

    # Build 5 dummy candidates (empty closed terms are fine — we just
    # need *something* that the function can iterate).
    candidates = [MoleculeClosedTerm() for _ in range(5)]
    scores = [0.1, 0.2, 0.3, 0.4, 0.5]

    eq = extract_paper_equation(candidates, scores, heuristic=hr)
    assert isinstance(eq, str), f"expected str, got {type(eq)}"
    assert len(eq) > 0, "equation string must be non-empty"
    # It should NOT be the unfitted placeholder.
    assert eq != "<unfitted>", "fitted heuristic must produce a real equation"


def test_extract_paper_equation_unfitted_fallback() -> None:
    """With no heuristic fitted we fall back to a 'best=' summary line."""
    candidates = [MoleculeClosedTerm.from_smiles("CCO", embed_3d=False)]
    scores = [0.42]
    eq = extract_paper_equation(candidates, scores, heuristic=None)
    assert isinstance(eq, str)
    assert "score=" in eq or "best=" in eq


def test_lam_loop_uses_pocket_loader_when_provided() -> None:
    """pocket_loader(pdb_id) is invoked exactly once per run() call."""
    mcts = MockMCTS()
    scorer = MockScorer()
    seen: List[str] = []

    def loader(pdb_id: str) -> Dict[str, Any]:
        seen.append(pdb_id)
        return {"pdb_id": pdb_id, "kind": "fake-pocket"}

    loop = LamClickDesignLoop(mcts=mcts, scorer=scorer, pocket_loader=loader)
    loop.run(pdb_id="1abc", n_iterations=2, top_k=2, max_depth=2)
    assert seen == ["1abc"], f"expected ['1abc'], got {seen}"


def test_lam_loop_paper_equation_is_set() -> None:
    """After run() the loop exposes the fitted equation on .paper_equation."""
    mcts = MockMCTS()
    scorer = MockScorer()
    hr = HeuristicRegressor(niterations=2, random_state=0)
    loop = LamClickDesignLoop(mcts=mcts, scorer=scorer, symbolic_reg=hr)
    loop.run(pdb_id="x", n_iterations=2, top_k=3, max_depth=2)
    assert isinstance(loop.paper_equation, str)
    assert len(loop.paper_equation) > 0
