# SPDX-License-Identifier: Apache-2.0
"""Step 1 DMPNN wire-in integration test (Wf-R24-A, AGENTS.md §5.7 gate).

Two layers:

* **smoke** — the ``RewardAggregator`` channel must change the MCTS leaf
  value when wired and leave it byte-identical when not.
* **bench A/B** — the bench harness must accept ``--dmpnn-value`` /
  ``--no-dmpnn-value`` and emit the ``mean_dmpnn_value`` /
  ``max_dmpnn_value`` / ``dmpnn_value_var`` telemetry fields.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "molmetal") not in sys.path:
    sys.path.insert(0, str(REPO / "molmetal"))

BENCH = REPO / "molmetal/scripts/bench_mcts_search_quality.py"


def _load_bench(name="_r24_bench"):
    """Import the bench script as a module.

    The module MUST be registered in ``sys.modules`` before exec: the file
    uses ``from __future__ import annotations``, so every dataclass field is
    a *string* annotation that ``dataclasses`` resolves via
    ``sys.modules[cls.__module__].__dict__``.  Without registration that
    lookup is ``None`` and ``TargetRecord`` blows up at class-creation.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, BENCH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------
# layer 1 — aggregator smoke (§5.7: the flag must change real behaviour)
# ----------------------------------------------------------------------
def test_wired_channel_changes_leaf_value():
    """§5.7 smoke: wiring the value fn must actually change the reward.

    Unit tests on the SCAFFOLD module alone do not catch trainer-integration
    regressions — only exercising the aggregator's ``__call__`` with the
    real DMPNN fn does.
    """
    from molmetal_lam.search_alg.dmpnn_value import make_value_fn
    from molmetal_lam.search_alg.proof_search import RewardAggregator

    class _State:
        def __init__(self, smi):
            self._smi = smi

        def canonical_smiles(self):
            return self._smi

    dmpnn = make_value_fn()
    if not dmpnn.dmpnn._load():
        pytest.skip("DMPNN checkpoint not loadable")

    st = _State("C#CCN")
    control = RewardAggregator()(st)
    wired = RewardAggregator(dmpnn_value_fn=dmpnn)(st)
    assert wired != control, "wiring the DMPNN channel must change the leaf value"
    assert wired > control, "a positive pIC50 must increase the reward"
    assert dmpnn.dmpnn.n_forward > 0, "the DMPNN forward never actually ran"


def test_channel_off_is_unchanged():
    """The control arm must add exactly 0.0 — A/B integrity precondition."""
    from molmetal_lam.search_alg.proof_search import RewardAggregator

    class _State:
        def canonical_smiles(self):
            return "C#CCN"

    st = _State()
    assert RewardAggregator()(st) == RewardAggregator(dmpnn_value_fn=None)(st)
    assert RewardAggregator()(st) == RewardAggregator(w_dmpnn_value=0.0)(st)


# ----------------------------------------------------------------------
# layer 2 — bench CLI surface
# ----------------------------------------------------------------------
def test_bench_accepts_dmpnn_flags():
    """--dmpnn-value / --no-dmpnn-value / --dmpnn-metal / --dmpnn-pocket parse."""
    import subprocess
    import sys as _sys

    # ``--help`` exercises the full argparse construction, so a bad flag
    # name or a dest mismatch fails here rather than mid-bench.
    out = subprocess.run(
        [_sys.executable, str(BENCH), "--help"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    for flag in ("--dmpnn-value", "--no-dmpnn-value", "--dmpnn-metal",
                 "--dmpnn-pocket", "--dmpnn-ckpt"):
        assert flag in out.stdout, f"{flag} missing from --help"


def test_bench_aggregate_reports_dmpnn_fields():
    """_aggregate_metrics must emit the 4 new DMPNN telemetry fields."""
    mod = _load_bench("_r24_bench_agg")

    recs = [
        mod.TargetRecord(target_idx=0, smiles_in="C#CCN", seed=0, n_dmpnn_eval=10,
                         mean_dmpnn_value=2.0, max_dmpnn_value=3.0),
        mod.TargetRecord(target_idx=1, smiles_in="C#CCO", seed=0, n_dmpnn_eval=10,
                         mean_dmpnn_value=4.0, max_dmpnn_value=5.0),
    ]
    agg = mod._aggregate_metrics(recs, 100)
    assert agg["dmpnn_enabled"] is True
    assert agg["n_dmpnn_eval"] == 20
    assert agg["mean_dmpnn_value"] == pytest.approx(3.0)
    assert agg["max_dmpnn_value"] == pytest.approx(5.0)
    assert agg["dmpnn_value_var"] == pytest.approx(1.0)


def test_bench_aggregate_channel_off_is_zero_not_none():
    """Channel off => dmpnn_enabled False and 0.0 stats (honest sentinel).

    A `None` here would break downstream JSON consumers; a nonzero value
    would falsely imply DMPNN ran.
    """
    mod = _load_bench("_r24_bench_off")

    agg = mod._aggregate_metrics(
        [mod.TargetRecord(target_idx=0, smiles_in="C#CCN", seed=0)], 100
    )
    assert agg["dmpnn_enabled"] is False
    assert agg["n_dmpnn_eval"] == 0
    for k in ("mean_dmpnn_value", "max_dmpnn_value", "dmpnn_value_var"):
        assert agg[k] == 0.0


@pytest.mark.skipif(
    not (
        (REPO / "molmetal/reports/r24_step1/armA_control/search_quality_bench.json").exists()
        or (REPO / "molmetal/reports/r24_step1/armB_dmpnn/search_quality_bench.json").exists()
    ),
    reason="Step 1 bench A/B JSON not on disk yet",
)
def test_bench_ab_json_has_dmpnn_arm_identity():
    """The on-disk A/B JSONs must record their arm in `config` (§5.5 truth).

    Skipped (not failed) when the bench has not been run, so the suite stays
    green on a fresh checkout while still asserting arm identity the moment
    a real JSON lands.
    """
    base = REPO / "molmetal/reports/r24_step1"
    found = []
    for arm in ("armA_control", "armB_dmpnn"):
        p = base / arm / "search_quality_bench.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        assert "dmpnn_value" in d["config"], f"{arm}: missing dmpnn_value in config"
        for k in ("mean_dmpnn_value", "max_dmpnn_value", "dmpnn_value_var"):
            assert k in d["agg"], f"{arm}: missing {k} in agg"
        found.append(arm)
    assert found, "neither arm JSON is on disk"