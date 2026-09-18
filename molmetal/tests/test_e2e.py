"""End-to-end smoke test for the molmetal orchestration loop.

Wires the four mock adapters (Generator / Docker / Predictor / Scorer) into
:class:`~molmetal.orchestration.ClosedLoop` and runs a tiny design round.

Asserts:
- non-empty result list
- every score is finite
- every rank is in [1, n]
- the result count never exceeds n_top_k
"""

from __future__ import annotations

import math

import pytest
import torch

from molmetal.adapters.mock import (
    MockDocker,
    MockGenerator,
    MockPredictor,
    MockScorer,
)
from molmetal.domain import Pocket
from molmetal.orchestration import ClosedLoop
from molmetal.ports import DesignLoopConfig


def _tiny_pocket(n_atoms: int = 10) -> Pocket:
    """A 10-atom pocket centred at the origin."""
    coords = torch.zeros(n_atoms, 3, dtype=torch.float32)
    atom_types = torch.tensor(
        [6, 7, 8, 16, 6, 7, 8, 16, 6, 7], dtype=torch.long
    )[:n_atoms]
    residue_ids = torch.arange(n_atoms, dtype=torch.long)
    chain_ids = torch.zeros(n_atoms, dtype=torch.long)
    mask = torch.ones(n_atoms, dtype=torch.bool)
    center = torch.zeros(3, dtype=torch.float32)
    return Pocket(
        pdb_id="e2e",
        coords=coords,
        atom_types=atom_types,
        residue_ids=residue_ids,
        chain_ids=chain_ids,
        mask=mask,
        center=center,
        radius=6.0,
    )


class TestE2EClosedLoop:
    def test_closed_loop_one_iteration(self):
        pocket = _tiny_pocket(n_atoms=10)

        gen = MockGenerator(seed=123)
        docker = MockDocker(seed=456)
        predictor = MockPredictor(seed=789)
        scorer = MockScorer()

        loop = ClosedLoop(
            generator=gen,
            docker=docker,
            predictor=predictor,
            scorer=scorer,
            verbose=False,  # keep test output clean
        )
        loop.setup(device="cpu")

        cfg = DesignLoopConfig(
            n_iterations=1,
            n_samples_per_round=4,
            n_top_k=2,
        )
        result = loop.run(pocket, cfg)

        # 1. non-empty result
        assert len(result) > 0
        # 2. every rank is in [1, n]
        n = len(result)
        ranks = [c.rank for c in result]
        assert all(1 <= r <= n for r in ranks), f"bad ranks: {ranks}"
        # 3. every combined_score is finite
        for c in result:
            assert math.isfinite(c.combined_score), (
                f"non-finite combined_score: {c.combined_score}"
            )
        # 4. capped by n_top_k
        assert len(result) <= cfg.n_top_k

    def test_closed_loop_history_nonempty(self):
        pocket = _tiny_pocket(n_atoms=10)
        loop = ClosedLoop(
            generator=MockGenerator(seed=1),
            docker=MockDocker(seed=2),
            predictor=MockPredictor(seed=3),
            scorer=MockScorer(),
            verbose=False,
        )
        loop.setup(device="cpu")
        cfg = DesignLoopConfig(n_iterations=1, n_samples_per_round=2, n_top_k=1)
        result = loop.run(pocket, cfg)
        history = loop.get_history()
        assert len(history) == 1
        rec = history[0]
        assert rec["n_generated"] == cfg.n_samples_per_round
        assert rec["n_scored"] == cfg.n_samples_per_round
        # With n_top_k=1, we expect exactly 1 candidate in the pool.
        assert len(result) == 1
        assert result[0].rank == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))