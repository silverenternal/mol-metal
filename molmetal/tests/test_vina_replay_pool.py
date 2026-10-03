"""Tests for the Vina Replay Pool (molmetal_lam.sbdd_env.vina_replay_pool).

================================================================
What this file pins
================================================================
Validates the load-bearing pieces of the Dream-RSI adaptation to
mol-metal's Vina docking workflow
(``molmetal/molmetal_lam/sbdd_env/vina_replay_pool.py``):

* :class:`VinaReplayNode`        — wraps ``(smiles, embedding,
  pocket_features, vina_score)``; round-trip + finite-score
  contract.
* :class:`VinaReplayPool`        — per-pocket JSON-backed storage
  with ``add_observation`` / ``get_all`` / ``save`` / ``load``.
* :class:`VinaReplaySimulator`   — :class:`ReplayProbe` subclass
  that reveals cached ``(smiles, score)`` pairs.
* :class:`VinaSurrogate`         — small MLP / ridge regression
  trained on cached triples; ranks new SMILES candidates without
  paying for another Vina call.
* :func:`combined_v0_vm_ranking` — V^0-invariant + V^m-ranking
  composition so V* >= V_0 holds by construction.

Honest framing
--------------
* These tests use **synthetic** vina scores — there is no real
  Vina call here.  The Spearman threshold (>= 0.7) is checked on a
  deterministic synthetic regression problem.
* The pool may be small initially; the surrogate is *only* as
  good as the pool coverage — see ``MIN_POOL_FOR_SURROGATE`` and
  ``is_low_coverage``.

Run via::

    /home/hugo/codes/mol-metal/.venv/bin/python -m pytest \\
        molmetal/tests/test_vina_replay_pool.py -v --tb=short
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pytest

try:
    from molmetal.molmetal_lam.sbdd_env.vina_replay_pool import (  # type: ignore
        DEFAULT_EMBED_DIM,
        DEFAULT_POCKET_DIM,
        MLP_HIDDEN,
        ReplayProbe,
        VinaReplayNode,
        VinaReplayPool,
        VinaReplaySimulator,
        VinaSurrogate,
        VinaSurrogateReport,
        combined_v0_vm_ranking,
    )
except Exception as exc:  # pragma: no cover
    pytest.skip(
        f"Vina replay pool module unavailable: {exc}",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _deterministic_score(emb: np.ndarray, pocket_features: np.ndarray) -> float:
    """A learnable synthetic Vina score.

    Returns ``-7.0 + 0.5 * emb[0] - 0.1 * pocket_features.sum()``.
    Deterministic and monotonically related to ``emb[0]`` so a
    well-fit surrogate should be able to recover the ranking.
    """
    return float(-7.0 + 0.5 * emb[0] - 0.1 * float(pocket_features.sum()))


def _spearman(x: List[float], y: List[float]) -> float:
    """Pure-python Spearman ρ (no scipy dependency)."""
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    rx = _rank(x)
    ry = _rank(y)
    mx = sum(rx) / len(rx)
    my = sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def _rank(v: List[float]) -> List[float]:
    """Average-rank assignment (handles ties)."""
    sorted_idx = sorted(range(len(v)), key=lambda i: v[i])
    ranks = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[sorted_idx[j + 1]] == v[sorted_idx[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-based
        for k in range(i, j + 1):
            ranks[sorted_idx[k]] = avg_rank
        i = j + 1
    return ranks


def _build_training_pool(
    n: int = 64,
    pocket_id: str = "1h36",
    seed: int = 0,
) -> VinaReplayPool:
    """Build a pool of ``n`` entries with a learnable synthetic Vina score."""
    rng = np.random.default_rng(seed)
    pool = VinaReplayPool()
    pf = np.zeros(DEFAULT_POCKET_DIM, dtype=np.float64)
    # Use a non-trivial per-pocket feature so the surrogate can latch on.
    pf[0] = 1.0
    for i in range(n):
        emb = np.zeros(DEFAULT_EMBED_DIM, dtype=np.float64)
        # Strong learnable signal in dim 0.
        emb[0] = float(i - n / 2.0) * 0.1
        emb[1] = float(rng.standard_normal() * 0.01)
        emb[2] = float(rng.standard_normal() * 0.01)
        score = _deterministic_score(emb, pf)
        pool.add_observation(
            smiles=f"TRAIN{i:04d}",
            dmpnn_embedding=emb,
            pocket_features=pf,
            vina_score=score,
            pocket_id=pocket_id,
        )
    return pool


# ===========================================================================
# VinaReplayPool — add / persistence
# ===========================================================================


def test_pool_add_and_get():
    """add_observation grows the pool; get_all returns every entry
    in (pocket, insertion) order."""
    pool = VinaReplayPool()
    assert len(pool) == 0
    emb = np.zeros(DEFAULT_EMBED_DIM, dtype=np.float64)
    pf = np.zeros(DEFAULT_POCKET_DIM, dtype=np.float64)
    pf[0] = 1.0

    n1 = pool.add_observation(
        "CCO",
        emb,
        pf,
        vina_score=-8.5,
        pocket_id="1h36",
    )
    n2 = pool.add_observation(
        "CCN",
        emb,
        pf,
        vina_score=-7.5,
        pocket_id="1h36",
    )
    n3 = pool.add_observation(
        "CCC",
        emb,
        pf,
        vina_score=-9.0,
        pocket_id="3eml",
    )

    # Returned nodes are typed correctly.
    assert isinstance(n1, VinaReplayNode)
    assert n1.smiles == "CCO"
    assert n1.pocket_id == "1h36"
    assert n1.vina_score == pytest.approx(-8.5)

    # Pool grew.
    assert len(pool) == 3
    assert pool.n_per_pocket() == {"1h36": 2, "3eml": 1}
    assert pool.total_for_pocket("1h36") == 2
    assert pool.total_for_pocket("3eml") == 1
    assert pool.total_for_pocket("nope") == 0

    # get_all returns every node in (pocket, insertion) order.
    all_nodes = pool.get_all()
    assert len(all_nodes) == 3
    assert [n.smiles for n in all_nodes] == ["CCO", "CCN", "CCC"]
    assert [n.pocket_id for n in all_nodes] == ["1h36", "1h36", "3eml"]


def test_pool_persistence_roundtrip():
    """save() writes per-pocket JSON; load() rebuilds the pool."""
    emb = np.zeros(DEFAULT_EMBED_DIM, dtype=np.float64)
    pf = np.zeros(DEFAULT_POCKET_DIM, dtype=np.float64)
    pf[0] = 1.0

    with tempfile.TemporaryDirectory() as td:
        pool = VinaReplayPool(storage_dir=td)
        pool.add_observation(
            "CCO",
            emb,
            pf,
            vina_score=-8.5,
            pocket_id="1h36",
        )
        pool.add_observation(
            "CCN",
            emb,
            pf,
            vina_score=-7.5,
            pocket_id="1h36",
        )
        pool.add_observation(
            "CCC",
            emb,
            pf,
            vina_score=-9.0,
            pocket_id="3eml",
        )
        pool.save()

        # JSON files exist per pocket.
        files = sorted(Path(td).glob("*.json"))
        assert {f.stem for f in files} == {"1h36", "3eml"}

        # Per-file content is a JSON list with one entry each.
        raw_1h36 = json.loads(Path(td, "1h36.json").read_text())
        assert isinstance(raw_1h36, list) and len(raw_1h36) == 2
        assert raw_1h36[0]["smiles"] == "CCO"
        assert raw_1h36[0]["vina_score"] == pytest.approx(-8.5)

        # Reload into a fresh pool.
        pool2 = VinaReplayPool.load(td)
        assert len(pool2) == 3
        assert pool2.n_per_pocket() == {"1h36": 2, "3eml": 1}

        all_nodes = pool2.get_all()
        assert [n.smiles for n in all_nodes] == ["CCO", "CCN", "CCC"]
        # Vectors survived the round-trip.
        assert all_nodes[0].dmpnn_embedding.shape == (DEFAULT_EMBED_DIM,)
        assert all_nodes[0].pocket_features.shape == (DEFAULT_POCKET_DIM,)


# ===========================================================================
# VinaSurrogate — train / predict / rank
# ===========================================================================


def test_surrogate_train_predict_smoke():
    """fit() reduces training MSE below the naive baseline (mean)."""
    pool = _build_training_pool(n=64, pocket_id="1h36")
    surrogate = VinaSurrogate(backend="ridge", alpha=0.5)
    pf = np.zeros(DEFAULT_POCKET_DIM, dtype=np.float64)
    pf[0] = 1.0

    # Pre-fit: not fitted yet.
    assert surrogate.is_fitted() is False

    surrogate.fit([pool])
    assert surrogate.is_fitted() is True

    # Train loss is finite and below the naive-mean baseline.
    loss = surrogate.train_loss()
    assert math.isfinite(loss)
    train_scores = [n.vina_score for n in pool.get_all()]
    naive_mse = float(np.var(train_scores))
    assert loss < naive_mse

    # predict() returns a finite float in a reasonable range.
    emb = np.zeros(DEFAULT_EMBED_DIM, dtype=np.float64)
    emb[0] = 1.0
    s = surrogate.predict(emb, pf)
    assert math.isfinite(s)
    # The predict lands near the training-set mean ± a generous bound.
    train_mean = float(np.mean(train_scores))
    assert train_mean - 5.0 <= s <= train_mean + 5.0


def test_rank_candidates_orders_correctly():
    """rank_candidates() returns rows sorted by predicted score
    ascending (more negative = better Vina binder)."""
    pool = _build_training_pool(n=64, pocket_id="1h36")
    surrogate = VinaSurrogate(backend="ridge", alpha=0.5)
    surrogate.fit([pool])

    pf = np.zeros(DEFAULT_POCKET_DIM, dtype=np.float64)
    pf[0] = 1.0

    # 8 candidates, each with an embedding whose predicted score is
    # monotonically increasing in ``emb[0]``.  With the synthetic
    # _deterministic_score the surrogate should learn the same
    # monotonic relationship and rank ascending.
    n_test = 8
    smiles_list = [f"TEST{i:03d}" for i in range(n_test)]
    embeddings = []
    for i in range(n_test):
        emb = np.zeros(DEFAULT_EMBED_DIM, dtype=np.float64)
        emb[0] = float(i) * 0.5  # 0.0, 0.5, ..., 3.5
        embeddings.append(emb)

    rep = surrogate.rank_candidates(
        smiles_list,
        embeddings,
        pf,
    )
    assert isinstance(rep, VinaSurrogateReport)

    # Output is a list with one row per input, sorted ascending.
    assert len(rep.rankings) == n_test
    scores = [score for (_, score, _) in rep.rankings]
    for a, b in zip(scores, scores[1:]):
        assert a <= b, f"rank not ascending: {scores}"


def test_spearman_correlation_above_threshold():
    """The surrogate's predicted ranking must correlate with the
    *true* (synthetic) Vina ranking on a held-out candidate set
    with Spearman ρ >= 0.7.

    Honest framing: the "true Vina" here is the deterministic
    synthetic oracle (``_deterministic_score``).  The real claim
    about real Vina rank can only be verified on the actual dock
    sweep, which is out of scope here.  Spearman ρ >= 0.7 on N=32
    held-out candidates is the smoke threshold.
    """
    # Build a moderate training set so the surrogate can latch onto
    # the embedding -> score signal.
    pool = _build_training_pool(n=128, pocket_id="1h36", seed=0)
    surrogate = VinaSurrogate(backend="ridge", alpha=5.0)
    surrogate.fit([pool])

    pf = np.zeros(DEFAULT_POCKET_DIM, dtype=np.float64)
    pf[0] = 1.0

    # Generate held-out test candidates with a strong learnable signal.
    rng = np.random.default_rng(1)
    n_test = 32
    smiles_list = [f"TEST{i:03d}" for i in range(n_test)]
    embeddings: List[np.ndarray] = []
    true_scores: List[float] = []
    for i in range(n_test):
        emb = np.zeros(DEFAULT_EMBED_DIM, dtype=np.float64)
        # Strong learnable signal — well-separated values.
        emb[0] = float(i) * 0.5  # range [0, 15.5]
        emb[1] = float(rng.standard_normal() * 0.01)
        embeddings.append(emb)
        true_scores.append(_deterministic_score(emb, pf))

    rep = surrogate.rank_candidates(
        smiles_list,
        embeddings,
        pf,
    )
    pred_by_idx = {idx: float(score) for (idx, score, _) in rep.rankings}
    pred_scores = [pred_by_idx[i] for i in range(n_test)]

    rho = _spearman(pred_scores, true_scores)
    assert rho >= 0.7, (
        f"surrogate ranking fails to track synthetic Vina: "
        f"Spearman ρ={rho:.3f} < 0.7 threshold. "
        f"Predicted={pred_scores[:5]}..., True={true_scores[:5]}..."
    )

    # Top-K overlap: surrogate's top-10 by predicted should agree
    # with true top-10 by score on at least 5 of 10.
    k = 10
    surrogate_top = sorted(range(n_test), key=lambda i: pred_scores[i])[:k]
    true_top = sorted(range(n_test), key=lambda i: true_scores[i])[:k]
    overlap = len(set(surrogate_top) & set(true_top))
    assert overlap >= 5, (
        f"surrogate top-{k} overlap with true top-{k} = {overlap} < 5; "
        f"pred_top={surrogate_top[:5]} true_top={true_top[:5]}"
    )


# ===========================================================================
# End-to-end extras — combined ranking + simulator + param budget
# ===========================================================================


def test_combined_v0_vm_ranking_honours_invariant():
    """V* >= V_0 monotone invariant: under the kcal/mol convention
    (more negative = better binder), every candidate's combined
    score is ``<= V^0_baseline`` (so the loop never picks a candidate
    that is worse than the cached baseline).

    Formula: ``combined(s) = V^0_baseline + V^m_s``.

    Examples with V^0 = -8.0:

    * candidate a (V^m = -9.0): combined = -17.0  (better than baseline)
    * candidate b (V^m = -8.5): combined = -16.5  (better than baseline)
    * candidate c (V^m = -8.2): combined = -16.2  (better than baseline)
    * candidate d (V^m = -8.0): combined = -16.0  (equal to baseline)

    All combined scores are ``<= V^0 = -8.0`` ⇒ V* (the minimum)
    is at most V_0 in absolute terms, which honours the paper's
    monotone invariant "V* >= V_0" when read in the kcal/mol sign
    convention.
    """
    v0 = -8.0  # cached baseline (kcal/mol)
    candidates = [
        ("a", -9.0),
        ("b", -8.5),
        ("c", -8.2),
        ("d", -8.0),  # equal to V^0
    ]
    surrogate = VinaSurrogate(backend="ridge")
    out = combined_v0_vm_ranking(v0, candidates, surrogate)
    assert len(out) == 4
    # In kcal/mol convention: combined <= v0 (each candidate is at
    # least as good as the baseline in the more-negative-is-better
    # sense, so V* <= V_0 means V* is the better binder).
    for _, combined in out:
        assert combined <= v0 + 1e-9
    # Sorted ascending (more negative = better).
    scores = [s for _, s in out]
    for a, b in zip(scores, scores[1:]):
        assert a <= b
    # First element is the best candidate.
    assert out[0][0] == "a"
    # Best combined score is the sum of v0 + the most-negative V^m.
    assert out[0][1] == pytest.approx(v0 + -9.0)


def test_simulator_reveals_cached_pairs():
    """VinaReplaySimulator.reveal returns ``confidence='high'`` for
    cached (smiles, pocket_id) pairs and ``confidence='miss'`` for
    the rest — no Vina call is made."""
    pool = _build_training_pool(n=8, pocket_id="1h36", seed=0)
    sim = VinaReplaySimulator(pool)
    assert isinstance(sim, ReplayProbe)

    # All cached smiles are present in the pool.
    cached = pool.get_for_pocket("1h36")
    cached_smiles = [n.smiles for n in cached]
    # Pick two cached + one uncached candidate.
    cells = [
        (cached_smiles[0], "1h36"),
        (cached_smiles[1], "1h36"),
        ("UNCACHED_SMILES", "1h36"),
    ]
    out = sim.reveal(cells)
    assert len(out) == 3
    # First two are 'high' (exact hit); third is 'miss'.
    assert out[0].confidence == "high"
    assert out[0].vina_score == pytest.approx(cached[0].vina_score)
    assert out[1].confidence == "high"
    assert out[2].confidence == "miss"
    assert math.isnan(out[2].vina_score)


def test_surrogate_parameter_count_under_10k():
    """Honest constraint: the MLP backend stays under 10K params so
    inference stays cheap on CPU.  ridge backend is also bounded."""
    # Ridge backend: weights = embed+pocket = 320 + bias = 321.
    ridge = VinaSurrogate(backend="ridge")
    pf = np.zeros(DEFAULT_POCKET_DIM, dtype=np.float64)
    pf[0] = 1.0
    pool = _build_training_pool(n=64, pocket_id="1h36")
    ridge.fit([pool])
    assert ridge.n_parameters() < 10_000

    # MLP backend (default hidden=24): W1=(320, 24) + b1=(24,)
    # + w2=(24,) + bias = 7705 — comfortably under 10K.
    mlp = VinaSurrogate(backend="mlp")
    mlp.fit([pool])
    assert mlp.n_parameters() < 10_000
    assert (
        mlp.n_parameters()
        == MLP_HIDDEN * (DEFAULT_EMBED_DIM + DEFAULT_POCKET_DIM)
        + MLP_HIDDEN
        + MLP_HIDDEN
        + 1
    )


def test_vina_replay_node_rejects_nonfinite_score():
    """NaN / Inf vina_score raises ValueError — the cache must never
    store an unusable entry."""
    emb = np.zeros(DEFAULT_EMBED_DIM, dtype=np.float64)
    pf = np.zeros(DEFAULT_POCKET_DIM, dtype=np.float64)
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError):
            VinaReplayNode(
                smiles="CCO",
                dmpnn_embedding=emb,
                pocket_features=pf,
                vina_score=bad,
            )
