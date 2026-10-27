"""Wf-A113 (T34/D31+D32) — M9's weight table, and reading a run with no PEM.

Two defects, both of which made M9's output mean less than it looked like it
meant:

**D31 — a string in a tuple-valued table.**  ``_WEIGHT_TO_TERM`` maps a weight
dest to the term name(s) it scales.  ``wasserstein_pocket_weight`` was written
as a bare string while every other entry was a tuple, so the inverse build
``for t in terms`` iterated characters: ``_TERM_TO_WEIGHTS`` gained keys ``"w"``,
``"a"``, ``"s"``, ... and ``wasserstein_pocket_term`` mapped to ``()``.  M9 then
reported that term as contributing without a weight whenever it fired.  Three
more entries named a dest that does not exist on the trainer's parser
(``diversity_weight``, ``natural_charge_weight``, ``edm_weight``), with the same
effect.

**D32 — "unmeasurable" read as "fine".**  M9 required
``per_epoch_metrics.json`` and returned ``pass: False`` without it.  Every run
archived before that file existed — c8 included — was in that state, so the
gate that is supposed to answer "does one term own the objective" could not be
run on the run the paper quotes.

The tables are checked against the trainer's REAL dests, not against a copy of
the list, because a hand-maintained copy is how the three phantom keys survived.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from molmetal.scripts import cfm_module_gates as G

REPO_ROOT = Path(__file__).resolve().parents[2]
C8 = REPO_ROOT / "molmetal" / "reports" / "cfm_c8_fullgen_2026_10_24"


# --------------------------------------------------------------------------
# The table itself
# --------------------------------------------------------------------------


def test_a_string_value_is_rejected() -> None:
    """The exact shape that shipped: a bare string instead of a 1-tuple."""
    with pytest.raises(TypeError, match="not a tuple"):
        G._build_term_to_weights({"wasserstein_pocket_weight": "wasserstein_pocket_term"})


def test_a_non_string_element_is_rejected() -> None:
    with pytest.raises(TypeError):
        G._build_term_to_weights({"x_weight": (1,)})


def test_every_value_in_the_shipped_table_is_a_tuple() -> None:
    """Guards against "fixing" the builder instead of the table."""
    bad = {k: v for k, v in G._WEIGHT_TO_TERM.items() if not isinstance(v, tuple)}
    assert not bad, f"non-tuple values in _WEIGHT_TO_TERM: {bad}"


def _trainer_dests() -> set:
    import importlib.util
    import sys

    p = REPO_ROOT / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"
    spec = importlib.util.spec_from_file_location("_t_dests", p)
    m = importlib.util.module_from_spec(spec)
    sys.modules["_t_dests"] = m
    spec.loader.exec_module(m)
    old = sys.argv
    try:
        sys.argv = ["train_cfm_yuelbond_wasserstein.py", "--device", "cpu"]
        return set(m._objective_weights(m.parse_args()))
    finally:
        sys.argv = old


def test_every_table_key_is_a_real_dest_on_the_trainer() -> None:
    """A key that names no dest is a mapping that can never fire.

    ``diversity_weight``, ``natural_charge_weight`` and ``edm_weight`` were
    exactly that: the term they mapped was reported weightless no matter what
    the config said.
    """
    dests = _trainer_dests()
    phantom = sorted(set(G._WEIGHT_TO_TERM) - dests)
    assert not phantom, (
        f"_WEIGHT_TO_TERM names dests the trainer does not have: {phantom}. "
        "Each one makes its term permanently unobservable."
    )


def test_every_unmapped_dest_is_declared() -> None:
    """Every weight the trainer can set is either mapped or explicitly excused."""
    dests = _trainer_dests()
    unexplained = sorted(dests - set(G._WEIGHT_TO_TERM) - G._UNMAPPED_DESTS)
    assert not unexplained, (
        f"these weights map to no term and are not in _UNMAPPED_DESTS: "
        f"{unexplained}"
    )


def test_the_inverse_map_has_no_single_letter_keys() -> None:
    """The fingerprint of the string-iteration bug."""
    assert not [k for k in G._TERM_TO_WEIGHTS if len(k) == 1]
    assert G._TERM_TO_WEIGHTS["wasserstein_pocket_term"] == (
        "wasserstein_pocket_weight",
    )
    assert G._TERM_TO_WEIGHTS["diversity_term"] == ("mmd_weight", "repulsion_weight")


# --------------------------------------------------------------------------
# Reading events.jsonl
# --------------------------------------------------------------------------


def test_events_jsonl_rows_from_c8() -> None:
    rows, source = G._events_jsonl_rows(C8)
    assert source == "events_jsonl", source
    assert [r["epoch"] for r in rows] == [0, 1, 2]
    assert "shortcut_distill_loss" in rows[0]["loss_terms"]


def test_missing_events_jsonl_is_a_reason_not_a_crash(tmp_path: Path) -> None:
    rows, why = G._events_jsonl_rows(tmp_path)
    assert rows == []
    assert why == "no_events_jsonl"


def test_m9_reads_c8_from_events_jsonl() -> None:
    """The gate that decides objective balance must run on c8."""
    ns = argparse.Namespace(fm_weight=1.0, bond_loss_weight=1.0, wasserstein_weight=0.1)
    out = G.gate_m9({"run_dir": str(C8), "ns": ns})
    assert out["measured"]["source"] == "events_jsonl"
    assert out["measured"]["n_epochs"] == 3
    assert out["measured"]["worst_term"]
    # c8's own args had no shortcut_distill_weight dest at all: the flag did
    # not exist on 2026-10-24.  With that namespace the term is weightless,
    # which is what the run actually was.
    assert "shortcut_distill_loss" in out["measured"]["terms_contributing_without_a_weight"]
    # fm_weight and bond_loss_weight are declared and do export terms.
    assert "fm_weight" not in out["measured"]["unobservable_weights"]


def test_m9_still_prefers_per_epoch_metrics(tmp_path: Path) -> None:
    """The fallback must not shadow the primary source."""
    import json

    (tmp_path / "per_epoch_metrics.json").write_text(
        json.dumps({"per_epoch_metrics": [{"epoch": 0, "loss_terms": {"fm_term": 1.0}}]})
    )
    (tmp_path / "events.jsonl").write_text(
        json.dumps({"event": "train_epoch", "epoch": 9, "loss_terms": {"fm_term": 2.0}})
        + "\n"
    )
    out = G.gate_m9(
        {"run_dir": str(tmp_path), "ns": argparse.Namespace(fm_weight=1.0)}
    )
    assert out["measured"]["source"] == "loss_terms"
    assert out["measured"]["n_epochs"] == 1
    assert out["measured"]["per_epoch"][0]["epoch"] == 0
