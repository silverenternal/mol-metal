"""Sample ledger — append-only per-sample (loss, gradient, pIC50) writer.

================================================================
Per-sample ledger for RSI curriculum learning (TODO-150 §P1.1)
================================================================
Every minibatch that flows through the multi-task D-MPNN trainer emits
one row per sample carrying:

* :attr:`PerSampleRecord.row_id`            — stable dataset index
* :attr:`PerSampleRecord.smiles`            — input SMILES
* :attr:`PerSampleRecord.metal`             — metal tag (``"Ru"``, ``"Rh"`` …)
* :attr:`PerSampleRecord.pocket_id`         — pocket tag (``"A2780"``,
                                               ``"MCF7"``, …)
* :attr:`PerSampleRecord.loss`              — sample-level loss scalar
* :attr:`PerSampleRecord.gradient_norm`     — ||∂L/∂h_pool||_2 (per-sample,
                                               measured via the loss
                                               ``backward(retain_graph=True)``
                                               trick on the pooled
                                               representation)
* :attr:`PerSampleRecord.predicted_pic50`   — model output
* :attr:`PerSampleRecord.target_pic50`      — ground-truth label
* :attr:`PerSampleRecord.predicted_active`  — softmax-argmax activity class
* :attr:`PerSampleRecord.target_active`     — ground-truth activity class
* :attr:`PerSampleRecord.epoch`             — training epoch (0-indexed)
* :attr:`PerSampleRecord.seed`              — RNG seed

Shards are written as **parquet** to::

    metrics/by_metric/rsi_sample_ledger/<metal>/seed_<n>/epoch_<k>.parquet

Parquet is the chosen on-disk format because:

* columnar — cheap to scan / filter by ``metal`` / ``pocket_id``;
* self-describing — schema evolution (e.g. adding
  ``hardness_pred``) is forward-compatible;
* native to the existing ``pyarrow`` / ``pandas`` stack.

The ledger is **append-only** at the *shard* level (one file per
epoch); it never overwrites prior epochs.  This matches the Wf-124 P3
training protocol where each epoch is a frozen checkpoint.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

#: On-disk schema version.  Bump when adding a non-optional column so
#: readers can detect stale shards.
LEDGER_VERSION: int = 1

#: Default root directory for shards.  ``metrics/by_metric/`` is the
#: canonical mol-metal metrics root (matches ``ReplayPool.save``
#: convention in :mod:`molmetal.replay.pool`).
LEDGER_ROOT_DEFAULT: str = "metrics/by_metric/rsi_sample_ledger"


@dataclass(frozen=True)
class PerSampleRecord:
    """One row in the sample ledger.

    Fields are intentionally scalar + JSON-safe so the entire record
    can be serialised to a parquet row without nested types (parquet
    nesting costs a column of struct / repeated types).
    """

    row_id: int
    smiles: str
    metal: str
    pocket_id: str
    loss: float
    gradient_norm: float
    predicted_pic50: float
    target_pic50: float
    predicted_active: int
    target_active: int
    epoch: int
    seed: int

    def __post_init__(self) -> None:
        if not isinstance(self.row_id, int):
            raise TypeError(f"row_id must be int; got {type(self.row_id).__name__}")
        if not isinstance(self.smiles, str):
            raise TypeError(f"smiles must be str; got {type(self.smiles).__name__}")
        if not isinstance(self.metal, str):
            raise TypeError(f"metal must be str; got {type(self.metal).__name__}")
        if not isinstance(self.pocket_id, str):
            raise TypeError(
                f"pocket_id must be str; got {type(self.pocket_id).__name__}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict (useful for round-trip tests)."""
        return {
            "row_id": int(self.row_id),
            "smiles": str(self.smiles),
            "metal": str(self.metal),
            "pocket_id": str(self.pocket_id),
            "loss": float(self.loss),
            "gradient_norm": float(self.gradient_norm),
            "predicted_pic50": float(self.predicted_pic50),
            "target_pic50": float(self.target_pic50),
            "predicted_active": int(self.predicted_active),
            "target_active": int(self.target_active),
            "epoch": int(self.epoch),
            "seed": int(self.seed),
        }


@dataclass
class SampleLedgerShard:
    """A single parquet shard (one epoch of one seed of one metal)."""

    path: Path
    metal: str
    seed: int
    epoch: int
    n_rows: int
    n_written: int = 0
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": str(self.path),
            "metal": str(self.metal),
            "seed": int(self.seed),
            "epoch": int(self.epoch),
            "n_rows": int(self.n_rows),
            "n_written": int(self.n_written),
            "timestamp": float(self.timestamp),
            "ledger_version": LEDGER_VERSION,
        }


class SampleLedger:
    """Append-only ledger writer.

    Use::

        ledger = SampleLedger()
        shard = ledger.begin_shard(metal="Rh", seed=0, epoch=0)
        for rec in records:
            ledger.append(shard, rec)
        ledger.finalise(shard)

    The shard's path is computed lazily — :meth:`begin_shard` does not
    touch the filesystem.  :meth:`finalise` flushes any rows buffered
    in :attr:`_buffers` to disk and writes a sidecar ``.meta.json``.

    Backward compatibility
    ----------------------
    Earlier versions of the protocol did not write a sidecar; readers
    should tolerate missing ``.meta.json`` and fall back to the schema
    in the parquet file itself.
    """

    # ---- column order is part of the public schema ----
    COLUMNS: Tuple[str, ...] = (
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
    )

    def __init__(
        self,
        root: Optional[str | os.PathLike[str]] = None,
        overwrite: bool = False,
    ) -> None:
        self.root = Path(root) if root is not None else Path(LEDGER_ROOT_DEFAULT)
        self.overwrite = bool(overwrite)
        self._buffers: Dict[Tuple[str, int, int], List[Dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Path conventions
    # ------------------------------------------------------------------
    def shard_path(self, metal: str, seed: int, epoch: int) -> Path:
        """Return the canonical path for a (metal, seed, epoch) shard."""
        metal_safe = str(metal).replace(os.sep, "_")
        return (
            self.root
            / metal_safe
            / f"seed_{int(seed)}"
            / f"epoch_{int(epoch)}.parquet"
        )

    def sidecar_path(self, shard_path: Path) -> Path:
        """Return the ``.meta.json`` sidecar for ``shard_path``."""
        return Path(str(shard_path) + ".meta.json")

    # ------------------------------------------------------------------
    # Write API
    # ------------------------------------------------------------------
    def begin_shard(
        self, metal: str, seed: int, epoch: int
    ) -> SampleLedgerShard:
        """Allocate a shard slot.  No filesystem writes happen here."""
        metal = str(metal)
        seed = int(seed)
        epoch = int(epoch)
        path = self.shard_path(metal, seed, epoch)
        key = (metal, seed, epoch)
        if key not in self._buffers:
            self._buffers[key] = []
        return SampleLedgerShard(
            path=path,
            metal=metal,
            seed=seed,
            epoch=epoch,
            n_rows=len(self._buffers[key]),
        )

    def append(self, shard: SampleLedgerShard, record: PerSampleRecord) -> None:
        """Append a single record to the in-memory buffer."""
        key = (shard.metal, shard.seed, shard.epoch)
        self._buffers.setdefault(key, []).append(record.to_dict())
        shard.n_rows = len(self._buffers[key])

    def extend(
        self,
        shard: SampleLedgerShard,
        records: Iterable[PerSampleRecord],
    ) -> int:
        """Append a batch of records.  Returns the count actually added."""
        n_before = shard.n_rows
        for rec in records:
            self.append(shard, rec)
        return shard.n_rows - n_before

    def finalise(self, shard: SampleLedgerShard) -> Path:
        """Flush the buffered rows to parquet + write ``.meta.json`` sidecar.

        Idempotent — calling ``finalise`` on an already-flushed shard
        re-writes the same file (matches ``ReplayPool.save``).
        """
        key = (shard.metal, shard.seed, shard.epoch)
        rows = self._buffers.get(key, [])
        shard.path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows, columns=self.COLUMNS)
        # Cast categorical-friendly columns to plain str to avoid pyarrow
        # mixed-type inference complaints.
        df["smiles"] = df["smiles"].astype(str)
        df["metal"] = df["metal"].astype(str)
        df["pocket_id"] = df["pocket_id"].astype(str)
        if self.overwrite and shard.path.exists():
            shard.path.unlink()
        df.to_parquet(shard.path, index=False)
        shard.n_written = len(rows)
        shard.timestamp = time.time()
        # Sidecar meta — JSON-friendly, no schema duplication.
        sidecar = self.sidecar_path(shard.path)
        with open(sidecar, "w", encoding="utf-8") as fh:
            json.dump(shard.to_dict(), fh, sort_keys=True, indent=2)
        return shard.path

    # ------------------------------------------------------------------
    # Read API
    # ------------------------------------------------------------------
    def read_shard(self, path: str | os.PathLike[str]) -> pd.DataFrame:
        """Read a single shard back as a DataFrame.

        Convenience for the A/B harness — the trainer only ever writes.
        """
        return pd.read_parquet(path)

    def list_shards(
        self,
        metal: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> List[Path]:
        """Discover shards on disk; optionally filter by metal / seed."""
        if not self.root.exists():
            return []
        out: List[Path] = []
        metals = [metal] if metal is not None else [p.name for p in self.root.iterdir() if p.is_dir()]
        for m in metals:
            m_dir = self.root / m
            if not m_dir.is_dir():
                continue
            seed_dirs = (
                [m_dir / f"seed_{int(seed)}"]
                if seed is not None
                else [p for p in m_dir.iterdir() if p.is_dir()]
            )
            for s_dir in seed_dirs:
                if not s_dir.is_dir():
                    continue
                for p in sorted(s_dir.glob("epoch_*.parquet")):
                    out.append(p)
        return out

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        """Per-metal × per-seed × per-epoch row counts."""
        out: Dict[str, Any] = {"version": LEDGER_VERSION, "root": str(self.root), "metals": {}}
        if not self.root.exists():
            return out
        for m_dir in sorted(self.root.iterdir()):
            if not m_dir.is_dir():
                continue
            metal_dict: Dict[str, Any] = {}
            for s_dir in sorted(m_dir.iterdir()):
                if not s_dir.is_dir():
                    continue
                seed_dict: Dict[str, Any] = {}
                for p in sorted(s_dir.glob("epoch_*.parquet")):
                    epoch = int(p.stem.split("_")[-1])
                    seed_dict[epoch] = int(p.stat().st_size)
                metal_dict[s_dir.name] = seed_dict
            out["metals"][m_dir.name] = metal_dict
        return out


__all__ = [
    "LEDGER_VERSION",
    "LEDGER_ROOT_DEFAULT",
    "PerSampleRecord",
    "SampleLedgerShard",
    "SampleLedger",
]