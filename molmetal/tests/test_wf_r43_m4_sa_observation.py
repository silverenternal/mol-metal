# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Wf-R43 m4 — the SA observation must be REAL and must not be silently 0.

Why this test exists
--------------------
m4 (`metrics/by_metric/sa_score.json`, `m4_success_rate.json`) sat in the
🔴 MOCK_NEEDS_REMEASURE column because "SA SCAFFOLD-only". The mechanism was
not a missing module: `molmetal/adapters/flow_matching_lipman/sa_penalty.py`
exists, exports `compute_sa_score`, and `--sa-penalty` was threaded all the
way into `train_one_epoch`. What was missing was a *consumer of real
molecules*. The flag's only consumer computed SA from the DATA clouds via
`_synthesize_sa_scores_from_clouds` — a function of the batch the trainer was
GIVEN, not of anything the model PREDICTED — and that constant was correctly
removed from the `loss = (...)` sum on 2026-10-04 (see
`molmetal/tests/test_cfm_loss_gradient_liveness.py`).

So "the flag is parsed and the arithmetic runs" was true and "SA is measured
on generated chemistry" was false, and nothing in the repo could tell those
apart. Per AGENTS.md §5.7 that is FLAG_ONLY, not SHIPPED.

These tests pin the Wf-R43 m4 correction, which has two halves that must
both hold and are easy to confuse:

1. **MEASUREMENT** — every decoded molecule gets its real Ertl-2009 score,
   and the score is what sascorer says for that molecule (not a constant, not
   a surrogate).
2. **HONESTY OF ABSENCE** — when scoring was not requested, or sascorer is
   unavailable, `sa_observed` is False and the mean must not be readable as a
   measurement. A run that scored nothing and a run whose molecules were all
   perfectly synthesizable both mean `sa_mean == 0.0`; only `sa_observed`
   separates them, so the summary has to carry it.

The test deliberately does NOT assert that SA can influence training. It
cannot: sascorer is a fragment-frequency lookup with no gradient. Asserting a
gradient here would be the exact mistake the 2026-10-04 audit corrected, so
the observable framing is pinned instead.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINER_PATH = (
    REPO_ROOT / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"
)


def _load_trainer():
    """Import the trainer as a module without triggering its __main__."""
    spec = importlib.util.spec_from_file_location(
        "wf_r43_m4_trainer", TRAINER_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def trainer():
    return _load_trainer()


# ---------------------------------------------------------------------------
# 1. The score is REAL — it is sascorer's number for THIS molecule
# ---------------------------------------------------------------------------


def test_sa_score_is_sascorer_value_not_a_constant(trainer) -> None:
    """Two structurally different molecules get two different real scores.

    The failure this catches is the surrogate: `_synthesize_sa_scores_from_clouds`
    maps n_atoms/n_distinct into [1, 10], so a flat pair of molecules scores
    by atom count and nothing else. Benzene and a long alkane differ in
    fragment complexity, so sascorer separates them.
    """
    if not trainer.RDKIT_AVAILABLE:
        pytest.skip("RDKit unavailable")
    if not trainer._SA_PENALTY_AVAILABLE:
        pytest.skip("sa_penalty module unavailable")

    simple = trainer.decode_chemistry("c1ccccc1", None, 2.8)
    complex_ = trainer.decode_chemistry("CCCCCCCCCC(=O)O", None, 2.8)

    assert simple is not None, "benzene must parse"
    assert complex_ is not None, "the second probe must parse"
    assert simple["sa_score"] is not None, (
        "sa_score must be a real number when sa_target was supplied; "
        "None here means the m4 wire-in is not live"
    )
    assert complex_["sa_score"] is not None
    # The canonical SA range is [1, 10] (Ertl 2009 §3).
    for row in (simple, complex_):
        assert 1.0 <= row["sa_score"] <= 10.0, (
            f"SA outside the canonical [1, 10] band: {row['sa_score']}"
        )
    assert simple["sa_score"] != complex_["sa_score"], (
        "distinct molecules received an identical SA — that is the "
        "surrogate's signature, not a measurement"
    )


def test_sa_score_is_not_nan(trainer) -> None:
    """sascorer signals failure with NaN; a NaN must not reach the summary.

    A NaN that entered the epoch mean would poison every downstream read
    silently, so `_score_decoded_sa` converts it to `None` and this pins it.
    """
    if not trainer.RDKIT_AVAILABLE or not trainer._SA_PENALTY_AVAILABLE:
        pytest.skip("RDKit / sa_penalty unavailable")
    from rdkit import Chem

    val = trainer._score_decoded_sa(Chem.MolFromSmiles("c1ccccc1"))
    assert val is not None
    assert val == val  # not NaN


def test_score_decoded_sa_returns_none_for_no_mol(trainer) -> None:
    """``None`` in, ``None`` out — never an exception, never 0.0."""
    assert trainer._score_decoded_sa(None) is None


# ---------------------------------------------------------------------------
# 2. HONESTY OF ABSENCE — the failure that made m4 look worse than it was
# ---------------------------------------------------------------------------


def test_sa_absent_by_default_so_pre_m4_callers_are_unchanged(trainer) -> None:
    """``sa_target`` defaults to None, so a caller who never asked pays nothing.

    This is the backward-compat contract: `test_cfm_chemistry_blind_spot.py`
    and `test_cfm_d4_metal_presence_gate.py` call `decode_chemistry(smiles)`
    and must keep getting exactly the pre-m4 row shape with no SA key
    reading as a number.
    """
    if not trainer.RDKIT_AVAILABLE:
        pytest.skip("RDKit unavailable")
    row = trainer.decode_chemistry("c1ccccc1")
    assert row is not None
    assert row["sa_score"] is None
    assert row["sa_target"] is None


def test_sa_observed_false_when_nothing_was_scored(trainer) -> None:
    """An unscored run says so. `sa_mean == 0.0` alone must not be read as data.

    This is the whole point of `sa_observed`: without it, "sascorer was not
    available" and "every generated molecule was perfectly synthesizable"
    produce the same file.
    """
    rows = [trainer.decode_chemistry("c1ccccc1")] if trainer.RDKIT_AVAILABLE else []
    summary = trainer.summarise_decode_chemistry(rows)
    assert summary["sa_observed"] is False
    assert summary["sa_scored"] == 0
    assert summary["sa_target"] is None
    assert summary["sa_synthesizable_rate"] == 0.0
    # Explicitly NOT a claim about the chemistry.
    assert summary["sa_subcriterion_met"] is None, (
        "an unobserved run must not report a pass/fail on the m4 "
        "sub-criterion — None is the only honest value here"
    )


def test_sa_observed_true_and_mean_over_scored_rows_only(trainer) -> None:
    """The mean is over SCORED rows, never over all rows.

    Mixing in unscored rows as zeros would pull the mean toward 0.0 and
    report a cohort that looks far more synthesizable than it is — the
    mirror image of the surrogate's error.
    """
    if not trainer.RDKIT_AVAILABLE or not trainer._SA_PENALTY_AVAILABLE:
        pytest.skip("RDKit / sa_penalty unavailable")
    scored = [
        trainer.decode_chemistry("c1ccccc1", None, 2.8),
        trainer.decode_chemistry("c1ccc2ccccc2c1", None, 2.8),
    ]
    unscored = trainer.decode_chemistry("CCO")  # no sa_target → None

    summary = trainer.summarise_decode_chemistry([*scored, unscored])
    assert summary["sa_observed"] is True
    assert summary["sa_scored"] == 2
    assert summary["n_chemistry_scored"] == 3
    expected = sum(r["sa_score"] for r in scored) / 2.0
    assert summary["sa_mean"] == pytest.approx(expected, abs=1e-9)
    assert 1.0 <= summary["sa_mean"] <= 10.0


def test_synthesizable_rate_counts_only_below_target(trainer) -> None:
    """`sa_synthesizable_rate` is the SA half of the m4 sub-criterion.

    A target of 1.0 makes every real molecule (SA >= 1) fall outside it, so
    the rate must be exactly 0.0 — pinning that the comparison is strict
    `<` against the supplied target and not against a hardcoded 2.8.
    """
    if not trainer.RDKIT_AVAILABLE or not trainer._SA_PENALTY_AVAILABLE:
        pytest.skip("RDKit / sa_penalty unavailable")
    rows = [trainer.decode_chemistry("c1ccccc1", None, 1.0)]
    summary = trainer.summarise_decode_chemistry(rows)
    assert summary["sa_target"] == 1.0
    assert summary["sa_synthesizable_rate"] == 0.0
    assert summary["sa_subcriterion_met"] is False


def test_empty_rows_branch_carries_the_same_keys(trainer) -> None:
    """A consumer must never have to branch on row count.

    The empty branch predates the SA keys; if it did not grow them, every
    reader would need `if rows:` around a plain key lookup, and one of the
    two branches would rot.
    """
    summary = trainer.summarise_decode_chemistry([])
    for key in (
        "sa_observed",
        "sa_scored",
        "sa_mean",
        "sa_target",
        "sa_synthesizable_rate",
        "sa_subcriterion_met",
    ):
        assert key in summary, f"empty-rows branch is missing {key!r}"


def test_unparseable_smiles_still_returns_none(trainer) -> None:
    """Pre-existing contract, re-checked: bad input is None, not a raise."""
    assert trainer.decode_chemistry("this is not a smiles", None, 2.8) is None


# ---------------------------------------------------------------------------
# 3. The flag reaches the generation eval (the FLAG_ONLY check, §5.7)
# ---------------------------------------------------------------------------


def test_sa_penalty_flag_reaches_the_generation_eval(trainer) -> None:
    """`--sa-penalty > 0` must reach `evaluate_generation_validity`.

    §5.7's rule: a flag that parses and whose value is read somewhere is not
    evidence that anything consumed it. The getattr-aware check is that the
    main loop passes a target that the eval forwards into
    `decode_chemistry`. A source assertion is the blunt instrument here, but
    it is the one that catches a rename or a dropped kwarg, which is the
    failure that leaves a flag silently inert across a big refactor.
    """
    src = TRAINER_PATH.read_text(encoding="utf-8")
    assert 'sa_target=float(getattr(args, "sa_target"' in src, (
        "the main loop must read args.sa_target (getattr-aware per the "
        "FLAG_ONLY lesson: a bare `args.sa_target` grep misses a "
        "getattr call)"
    )
    assert 'float(getattr(args, "sa_penalty", 0.0)) > 0.0' in src, (
        "the SA observation must be gated on --sa-penalty > 0 so a "
        "default run pays no sascorer cost"
    )
    assert (
        "decode_chemistry(smi, seed_metal_z(cloud), sa_target)" in src
    ), "the eval must forward sa_target into decode_chemistry"


def test_epoch_metrics_declares_the_sa_fields(trainer) -> None:
    """The observation must be a first-class metric, not a stdout string."""
    fields = trainer.EpochMetrics.__dataclass_fields__
    for name in (
        "gen_sa_observed",
        "gen_sa_mean",
        "gen_sa_target",
        "gen_sa_synthesizable_rate",
        "gen_sa_scored",
    ):
        assert name in fields, f"EpochMetrics is missing {name}"
    # The observed flag must default False so an un-run epoch serialises
    # honestly rather than claiming SA was measured at 0.0.
    default = trainer.EpochMetrics.__dataclass_fields__["gen_sa_observed"]
    assert default.default is False