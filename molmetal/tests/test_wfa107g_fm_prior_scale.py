# SPDX-License-Identifier: Apache-2.0
"""Wf-A107g — the FM prior and the data must share a scale, and it must be checked.

WHY THIS EXISTS
---------------
``x0 ~ N(0, fm_noise_sigma**2 I)`` and ``x1`` are the RAW cloud coordinates
(the tmQM loader does not scale them).  ``--fm-noise-sigma`` defaults to 1.0,
so a config that turns the FM objective on and forgets the key trains a flow
whose two ends sit at different scales: the velocity target is then dominated
by a constant radial expansion instead of structure, and a short run looks
exactly like "the sampler never moves the cloud".

MEASURED 2026-10-22: 11 of the 48 configs with ``use_fm_objective: true`` do
not set the key, including the two Wf-A107e arms whose "5 epochs moved the
cloud by 0.08 A" reading is therefore confounded.  Wf-R28 had already measured
the same defect on the c7 cohort (data std 2.485 vs prior 1.0).

The check is pure measurement: it reads ``x1`` and returns numbers, so every
run that already sets a matched sigma stays bit-exact.

RED CHECK: return ``ok = 1.0`` unconditionally and
``test_mismatched_sigma_is_flagged`` fails; compare against ``x1.mean()``
instead of ``x1.std()`` and ``test_matched_sigma_passes`` fails.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parent.parent.parent
TRAINER = REPO / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"


def _trainer():
    if "_wfa107g_trainer" in sys.modules:
        return sys.modules["_wfa107g_trainer"]
    spec = importlib.util.spec_from_file_location("_wfa107g_trainer", str(TRAINER))
    m = importlib.util.module_from_spec(spec)
    sys.modules["_wfa107g_trainer"] = m
    spec.loader.exec_module(m)
    return m


def _clouds(std: float, n_atoms: int = 400, seed: int = 0) -> "torch.Tensor":
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n_atoms, 3, generator=g) * float(std)


def test_matched_sigma_passes():
    t = _trainer()
    x1 = _clouds(2.485)
    out = t.fm_prior_scale_check(x1, 2.485)
    assert out["ok"] == 1.0
    assert out["ratio"] == pytest.approx(1.0, abs=0.02)
    assert out["data_std"] == pytest.approx(2.485, abs=0.05)


def test_mismatched_sigma_is_flagged():
    """The Wf-A107e case: prior 1.0 against data at 2.485."""
    t = _trainer()
    out = t.fm_prior_scale_check(_clouds(2.485), 1.0)
    assert out["ok"] == 0.0
    assert out["ratio"] > 2.0
    assert out["noise_sigma"] == 1.0


def test_too_large_prior_is_flagged_too():
    """A prior larger than the data is the same defect with the sign flipped."""
    t = _trainer()
    out = t.fm_prior_scale_check(_clouds(1.0), 2.485)
    assert out["ok"] == 0.0
    assert out["ratio"] < 0.5


def test_tolerance_edges_are_inclusive():
    t = _trainer()
    x1 = _clouds(2.0)
    r = float(x1.std())
    assert t.fm_prior_scale_check(
        x1, 1.0, tol_low=r, tol_high=r
    )["ok"] == 1.0
    assert t.fm_prior_scale_check(
        x1, 1.0, tol_low=r * 1.0001, tol_high=99.0
    )["ok"] == 0.0


def test_check_does_not_modify_the_input_or_consume_rng():
    t = _trainer()
    x1 = _clouds(2.485)
    before = x1.clone()
    torch.manual_seed(1234)
    _ = t.fm_prior_scale_check(x1, 2.485)
    after_state = torch.randn(4)
    torch.manual_seed(1234)
    expect_state = torch.randn(4)
    assert torch.equal(x1, before)
    assert torch.equal(after_state, expect_state)


def test_zero_sigma_is_infinite_ratio_not_a_crash():
    t = _trainer()
    out = t.fm_prior_scale_check(_clouds(2.0), 0.0)
    assert out["ok"] == 0.0
    assert out["ratio"] == float("inf")


def test_run_artifact_declares_the_check_and_the_sigma():
    """The artifact must carry both, so a reader can see which sigma trained."""
    src = TRAINER.read_text(encoding="utf-8")
    assert '"fm_prior_scale"' in src
    assert '"fm_noise_sigma": float(getattr(args, "fm_noise_sigma", 1.0))' in src
