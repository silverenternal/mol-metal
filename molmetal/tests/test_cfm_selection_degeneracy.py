# SPDX-License-Identifier: Apache-2.0
"""A criterion that never moved must not be allowed to pick epoch 0.

THE DEFECT THIS GUARDS
----------------------
``_best_gate`` starts at ``-inf``, so the first epoch improves whatever its
gate value is.  When the gate is constant — and on the c8 arm it was, 0.000 at
all 18 epochs — every later epoch TIES rather than improving, and the
checkpoint the run keeps is epoch 0: the least-trained weights it ever had.

MEASURED (metrics/by_metric/wfa108j_module_gates_gcv36.json, 2026-10-25):
``best_epoch`` 0 while the run reached epoch 17, and gate M10 compared the
sha256 of ``_last.pt`` (9fffb0eb) against the sha256 the artifact recorded for
``seed0.pt`` (d5b22bcb).  The mismatch was reported as a broken metric chain;
the chain was fine, the SELECTION was empty.

The rule under test is in ``resolve_selection``: one distinct gate value means
the argmax ranked nothing, so the final epoch is selected and the artifact says
so.  The second half of the file guards the other end of the same chain — the
gates must load the checkpoint the artifact names, not the one an operator
typed on the command line.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _load(rel: str, name: str):
    path = REPO_ROOT / rel
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _trainer():
    return _load("molmetal/scripts/train_cfm_yuelbond_wasserstein.py", "cfm_trainer_sel")


def _gates():
    return _load("molmetal/scripts/cfm_module_gates.py", "cfm_gates_sel")


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------
def test_a_constant_criterion_selects_the_last_epoch():
    sel = _trainer().resolve_selection(
        "combined", [(e, 0.0) for e in range(18)], best_epoch=0
    )
    assert sel["degenerate"] is True
    assert sel["distinct_values"] == 1
    assert sel["best_epoch_by_gate"] == 0, (
        "the gate's own arithmetic choice must stay visible; overriding it "
        "silently would hide why the selection changed"
    )
    assert sel["selected_epoch"] == 17
    assert sel["selected_checkpoint"] == "last"
    assert "0" in sel["reason"]


def test_a_moving_criterion_keeps_the_gates_choice():
    history = [(0, 0.10), (1, 0.22), (2, 0.31), (3, 0.28)]
    sel = _trainer().resolve_selection("bond_type_f1", history, best_epoch=2)
    assert sel["degenerate"] is False
    assert sel["selected_epoch"] == 2
    assert sel["selected_checkpoint"] == "selected"
    assert sel["reason"] == ""


def test_a_run_with_no_epochs_is_not_called_degenerate():
    """0 epochs is 'nothing was measured', not 'the gate is flat'."""
    sel = _trainer().resolve_selection("combined", [], best_epoch=-1)
    assert sel["degenerate"] is False
    assert sel["n_epochs_recorded"] == 0
    assert sel["selected_checkpoint"] == "selected"


def test_eval_only_never_reselects():
    """An eval pass must not claim its weights are the run's best."""
    sel = _trainer().resolve_selection(
        "combined", [(e, 0.0) for e in range(4)], best_epoch=0, eval_only=True
    )
    assert sel["degenerate"] is True
    assert sel["selected_checkpoint"] == "selected"
    assert "eval-only" in sel["reason"]


def test_two_distinct_values_are_enough_to_select():
    """Near-flat is still a ranking; only a single value is no ranking."""
    history = [(0, 0.0), (1, 0.0), (2, 1e-9)]
    sel = _trainer().resolve_selection("combined", history, best_epoch=2)
    assert sel["degenerate"] is False
    assert sel["selected_epoch"] == 2


# ---------------------------------------------------------------------------
# the other end of the chain: which file the gates load
# ---------------------------------------------------------------------------
def _run_dir(tmp_path: Path, named: str) -> Path:
    d = tmp_path / "run"
    d.mkdir()
    (d / "per_epoch_metrics.json").write_text(
        json.dumps({"provenance": {"checkpoint": named}}), encoding="utf-8"
    )
    return d


def test_the_artifact_names_the_checkpoint(tmp_path):
    ckpt = tmp_path / "seed0.pt"
    ckpt.write_bytes(b"weights")
    run = _run_dir(tmp_path, str(ckpt))
    named, why = _gates()._artifact_named_ckpt(str(run))
    assert named == ckpt
    assert "provenance" in why


def test_a_named_checkpoint_that_is_gone_is_refused(tmp_path):
    """The c7 failure mode: the artifact outlived its weights."""
    run = _run_dir(tmp_path, str(tmp_path / "gone.pt"))
    named, why = _gates()._artifact_named_ckpt(str(run))
    assert named is None
    assert "not on disk" in why


def test_no_run_dir_falls_back_to_the_cli(tmp_path):
    named, why = _gates()._artifact_named_ckpt(None)
    assert named is None
    assert "no --run-dir" in why


def test_an_artifact_without_a_checkpoint_falls_back(tmp_path):
    d = tmp_path / "run2"
    d.mkdir()
    (d / "per_epoch_metrics.json").write_text("{}", encoding="utf-8")
    named, why = _gates()._artifact_named_ckpt(str(d))
    assert named is None
    assert "names no checkpoint" in why
