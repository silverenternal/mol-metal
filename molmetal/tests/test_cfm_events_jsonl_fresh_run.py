"""A fresh run must not inherit the previous run's rows.

MEASURED 2026-10-25.  A Wf-A108j arm was killed after epoch 0 and relaunched
with ``resume_from: none``.  ``events.jsonl`` is opened in append mode, so the
new run wrote its own epoch 0 directly under the dead run's epoch 0 — two rows
labelled 0, with train losses 109.68 and 96.31, from two different objectives
(``energy_conservation_weight`` had been changed in between).  Every consumer
of the file — the TUI, the per-epoch summary, and any script that takes the
first ``train_epoch`` row — then reads a history that never existed.

The fix is a rotation, not a truncation: the old rows stay on disk as
``<name>.prev`` so the killed run is still inspectable, and the file the new
run writes describes only the new run.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def trainer():
    path = PROJECT_ROOT / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"
    spec = importlib.util.spec_from_file_location("_t_cfm_events_fresh", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_t_cfm_events_fresh"] = mod
    spec.loader.exec_module(mod)
    return mod


def _write(path: Path, epoch: int, loss: float) -> None:
    path.write_text(
        json.dumps({"event": "train_epoch", "epoch": epoch, "train_loss": loss}) + "\n"
    )


def test_fresh_run_rotates_the_previous_file(trainer, tmp_path):
    events = tmp_path / "events.jsonl"
    _write(events, 0, 109.68)

    trainer._install_event_emitter(True, events, fresh_run=True)
    trainer.emit_event("train_epoch", epoch=0, train_loss=96.31)
    trainer._close_event_emitter()

    rows = [json.loads(line) for line in events.read_text().splitlines() if line]
    assert [r["train_loss"] for r in rows] == [96.31], (
        "the new run's file carries rows from the run before it"
    )
    prev = tmp_path / "events.jsonl.prev"
    assert prev.is_file(), "the previous rows were deleted instead of kept"
    assert json.loads(prev.read_text().splitlines()[0])["train_loss"] == 109.68


def test_resumed_run_keeps_appending(trainer, tmp_path):
    """A resumed run's file is the SAME run's history, so it must survive."""
    events = tmp_path / "events.jsonl"
    _write(events, 0, 109.68)

    trainer._install_event_emitter(True, events, fresh_run=False)
    trainer.emit_event("train_epoch", epoch=1, train_loss=96.31)
    trainer._close_event_emitter()

    rows = [json.loads(line) for line in events.read_text().splitlines() if line]
    assert [r["epoch"] for r in rows] == [0, 1]
    assert not (tmp_path / "events.jsonl.prev").exists()


def test_a_fresh_run_with_no_previous_file_is_a_no_op(trainer, tmp_path):
    events = tmp_path / "events.jsonl"
    trainer._install_event_emitter(True, events, fresh_run=True)
    trainer.emit_event("train_epoch", epoch=0, train_loss=1.0)
    trainer._close_event_emitter()

    assert len(events.read_text().splitlines()) == 1
    assert not (tmp_path / "events.jsonl.prev").exists()
