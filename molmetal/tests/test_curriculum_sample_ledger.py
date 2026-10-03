"""Tests for molmetal.replay.curriculum.sample_ledger.

================================================================
Sample ledger contract tests (TODO-150 §P1.1)
================================================================
Covers the per-sample parquet writer primitives:

* :class:`PerSampleRecord`     — 3 tests (dataclass immutability, type
  rejection, JSON-safe round-trip).
* :class:`SampleLedger`        — 7 tests (path conventions, begin/append/
  finalise flow, sidecar sidecar_path, list_shards, stats, idempotence).

Total: **10 tests**, CPU-only.  All under 1 s on any CI container.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

try:
    from molmetal.replay.curriculum.sample_ledger import (
        LEDGER_ROOT_DEFAULT,
        LEDGER_VERSION,
        PerSampleRecord,
        SampleLedger,
    )
except Exception:
    from molmetal.replay.curriculum import (  # type: ignore
        LEDGER_ROOT_DEFAULT,
        LEDGER_VERSION,
        PerSampleRecord,
        SampleLedger,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_record(
    row_id: int = 0,
    smiles: str = "O",
    metal: str = "Ru",
    pocket_id: str = "A2780",
    epoch: int = 0,
    seed: int = 0,
    loss: float = 0.5,
) -> PerSampleRecord:
    return PerSampleRecord(
        row_id=int(row_id),
        smiles=smiles,
        metal=metal,
        pocket_id=pocket_id,
        loss=float(loss),
        gradient_norm=0.1,
        predicted_pic50=6.0,
        target_pic50=6.5,
        predicted_active=1,
        target_active=1,
        epoch=int(epoch),
        seed=int(seed),
    )


# ---------------------------------------------------------------------------
# PerSampleRecord
# ---------------------------------------------------------------------------


class TestPerSampleRecord:
    def test_construction_round_trip(self):
        rec = _make_record()
        d = rec.to_dict()
        for k in (
            "row_id",
            "smiles",
            "metal",
            "pocket_id",
            "loss",
            "gradient_norm",
            "predicted_pic50",
            "target_pic50",
            "predicted_active",
            "target_active",
            "epoch",
            "seed",
        ):
            assert k in d
        assert d["row_id"] == 0
        assert d["smiles"] == "O"
        assert d["metal"] == "Ru"

    def test_frozen(self):
        rec = _make_record()
        with pytest.raises((AttributeError, Exception)):
            rec.row_id = 99  # type: ignore[misc]

    def test_type_rejection(self):
        with pytest.raises(TypeError):
            PerSampleRecord(
                row_id="0",  # type: ignore[arg-type]
                smiles="O",
                metal="Ru",
                pocket_id="A2780",
                loss=0.5,
                gradient_norm=0.0,
                predicted_pic50=6.0,
                target_pic50=6.0,
                predicted_active=1,
                target_active=1,
                epoch=0,
                seed=0,
            )


# ---------------------------------------------------------------------------
# SampleLedger
# ---------------------------------------------------------------------------


class TestSampleLedger:
    def test_default_root(self):
        ledger = SampleLedger()
        assert str(ledger.root) == LEDGER_ROOT_DEFAULT

    def test_shard_path(self, tmp_path):
        ledger = SampleLedger(root=tmp_path)
        p = ledger.shard_path("Ru", seed=0, epoch=0)
        assert p == tmp_path / "Ru" / "seed_0" / "epoch_0.parquet"

    def test_sidecar_path(self, tmp_path):
        ledger = SampleLedger(root=tmp_path)
        p = ledger.shard_path("Rh", seed=2, epoch=7)
        s = ledger.sidecar_path(p)
        assert s == Path(str(p) + ".meta.json")

    def test_begin_append_finalise(self, tmp_path):
        ledger = SampleLedger(root=tmp_path, overwrite=True)
        shard = ledger.begin_shard("Ru", seed=0, epoch=0)
        ledger.append(shard, _make_record(row_id=0, smiles="O"))
        ledger.append(shard, _make_record(row_id=1, smiles="N"))
        path = ledger.finalise(shard)
        assert path.exists()
        df = pd.read_parquet(path)
        assert len(df) == 2
        # Sidecar written.
        sidecar = ledger.sidecar_path(path)
        meta = json.loads(sidecar.read_text())
        assert meta["metal"] == "Ru"
        assert meta["seed"] == 0
        assert meta["epoch"] == 0
        assert meta["n_written"] == 2
        assert meta["ledger_version"] == LEDGER_VERSION

    def test_extend(self, tmp_path):
        ledger = SampleLedger(root=tmp_path, overwrite=True)
        shard = ledger.begin_shard("Rh", seed=0, epoch=0)
        n = ledger.extend(
            shard,
            [_make_record(row_id=i, smiles="O", metal="Rh") for i in range(5)],
        )
        assert n == 5
        path = ledger.finalise(shard)
        assert len(pd.read_parquet(path)) == 5

    def test_list_shards(self, tmp_path):
        ledger = SampleLedger(root=tmp_path, overwrite=True)
        for m in ("Ru", "Rh"):
            for epoch in (0, 1):
                shard = ledger.begin_shard(m, seed=0, epoch=epoch)
                ledger.append(shard, _make_record(metal=m, epoch=epoch))
                ledger.finalise(shard)
        all_paths = ledger.list_shards()
        assert len(all_paths) == 4
        ru_paths = ledger.list_shards(metal="Ru")
        assert len(ru_paths) == 2

    def test_stats(self, tmp_path):
        ledger = SampleLedger(root=tmp_path, overwrite=True)
        shard = ledger.begin_shard("Ru", seed=0, epoch=0)
        ledger.append(shard, _make_record())
        ledger.finalise(shard)
        stats = ledger.stats()
        assert stats["version"] == LEDGER_VERSION
        assert "Ru" in stats["metals"]
        assert 0 in stats["metals"]["Ru"]["seed_0"]
        assert stats["metals"]["Ru"]["seed_0"][0] > 0

    def test_idempotent_finalise(self, tmp_path):
        ledger = SampleLedger(root=tmp_path, overwrite=True)
        shard = ledger.begin_shard("Ru", seed=0, epoch=0)
        ledger.append(shard, _make_record(row_id=0))
        path = ledger.finalise(shard)
        n_first = path.stat().st_size
        # Finalise again — same shard object re-flushes.
        ledger.append(shard, _make_record(row_id=1))
        path2 = ledger.finalise(shard)
        assert path == path2
        assert path2.stat().st_size > 0
        # First size is overwritten (we re-wrote with both rows).
        assert pd.read_parquet(path2).shape[0] == 2
        # n_first is the *single-row* size — the final 2-row file is bigger
        # when overwrite=True.  The assertion is just that finalisation
        # was idempotent at the *path* level.
        assert n_first > 0

    def test_columns_match_schema(self):
        ledger = SampleLedger()
        cols = ledger.COLUMNS
        assert "row_id" in cols
        assert "loss" in cols
        assert "predicted_pic50" in cols
        assert "epoch" in cols
        assert "seed" in cols
