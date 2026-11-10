"""Wf-A114 (Cluster A) — the generation metric columns must be honest.

WHY this file exists
--------------------
The CFM trainer used to write ``gen_validity`` as
``_gen_ok / max(_gen_tot, 1)``.  When the generation eval drew NO samples
that expression is a MEASURED ``0.0000`` — indistinguishable from "the model
generated chemistry that RDKit rejected".  MEASURED on disk (Wf-A114,
2026-10-30): 119 of 496 generation rows across 25 runs carried that
fabricated ``0.0000``, and two of them were LAST-epoch rows, i.e. the
headline number of those runs was invented by the writer.

The fix is a sentinel plus a self-describing column set:

* the rate columns are ``None`` (JSON ``null``) / the literal ``"nan"``
  (CSV) when the epoch drew no samples — never ``0.0000``, never ``-1``
  (a real number that would survive a median filter);
* the COUNT columns stay ``int``: 0 there means "no sample was drawn", which
  is a fact, not a rate;
* ``gen_measured`` (0/1) makes the CSV self-explaining;
* ``gen_end_to_end_rate`` publishes the honest end-to-end number —
  ``gen_validity * gen_single_fragment_ratio`` — the rate that can fail,
  because ``gen_validity`` is an RDKit PARSE rate and ``C.C.C`` parses.

THE PREDICATE (N3)
------------------
"this epoch drew no samples" is ``gen_validity_n == 0`` and NOTHING else.
``gen_eval_n == 0`` is NOT the predicate: 11 epoch rows on disk carry
``gen_eval_n == 0`` with ``gen_validity_n > 0`` (the eval ran; only the
per-seed record was lost).

CPU-only.  No checkpoint, no sampler, no training, no CUDA.
"""

from __future__ import annotations

from dataclasses import asdict

import pytest


@pytest.fixture(scope="module")
def cfm():
    try:
        from molmetal.scripts import train_cfm_yuelbond_wasserstein as _m
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"CFM trainer unavailable: {exc}")
    return _m


#: A chemistry block shaped like the trainer's `_gen_chem` after the
#: generation eval ran and parsed everything.
CHEM_MEASURED = {
    "molecule_validity": 0.5,
    "molecule_validity_n": 8,
    "molecule_validity_total": 16,
    "largest_fragment_frac_mean": 0.75,
    "single_fragment_ratio": 0.5,
    "metal_ratio": 1.0,
    "mean_heavy_atoms": 12.0,
    "rings_per_mol_mean": 1.5,
    "ring_5_6_share": 0.8,
    "ring_3_share": 0.1,
    "raw_rings_per_mol_mean": 0.9,
    "raw_ring_5_6_share": 0.6,
    "raw_ring_3_share": 0.2,
    "conditional_success": 0.25,
    "sa_observed": True,
    "sa_mean": 3.1,
    "sa_target": 2.8,
    "sa_synthesizable_rate": 0.4,
    "sa_scored": 8,
}


#: The fields `EpochMetrics` requires positionally (everything else has a
#: default).  They are irrelevant to the generation columns under test.
BASE_EPOCH = {
    "epoch": 0,
    "wall_time_s": 1.0,
    "train_loss": 0.5,
    "bond_ce_loss": 0.25,
    "wasserstein_loss": 0.125,
    "catherine_audit_pass": True,
    "decode_ratio_rdkit_strict": 0.5,
    "decode_ratio_val": 8,
    "decode_ratio_n": 16,
    "decoding_attempted": True,
    "lr": 1e-3,
    "grad_norm": 1.0,
}


def _unmeasured_epoch(cfm, **overrides):
    """One epoch whose generation eval drew no samples."""
    kwargs = dict(BASE_EPOCH)
    kwargs.update(cfm._gen_metric_fields(0, 0, {}))
    kwargs.update(overrides)
    return cfm.EpochMetrics(**kwargs)


def _measured_epoch(cfm, **overrides):
    """One epoch whose generation eval ran and parsed 12 of 16 samples."""
    kwargs = dict(BASE_EPOCH)
    kwargs.update(cfm._gen_metric_fields(12, 16, CHEM_MEASURED))
    kwargs.update(overrides)
    return cfm.EpochMetrics(**kwargs)


# ---------------------------------------------------------------------------
# 1 — the header carries the new columns
# ---------------------------------------------------------------------------
def test_csv_header_carries_the_molecule_columns(cfm):
    """The five Wf-A114 columns are in the CSV header, in order."""
    header = list(cfm.CSV_HEADER)
    for name in (
        "gen_molecule_validity",
        "gen_molecule_validity_n",
        "gen_molecule_validity_total",
        "gen_end_to_end_rate",
        "gen_measured",
    ):
        assert name in header, f"CSV_HEADER is missing {name!r}"
    # The strict rate sits next to the parse rate and the end-to-end rate
    # next to the single-fragment rate, so the two can be read side by side.
    assert header.index("gen_molecule_validity") > header.index(
        "gen_single_fragment_ratio"
    )
    assert header.index("gen_end_to_end_rate") < header.index("gen_metal_ratio")


# ---------------------------------------------------------------------------
# 2 — the row builder cannot drift from the header
# ---------------------------------------------------------------------------
def test_csv_row_length_matches_header(cfm):
    """One-sided column edits used to shift every cell after the change.

    `csv_row` now ASSERTS the length; this test is the belt to that
    braces: a header-only or row-only edit fails here, loudly.

    It ALSO pins the WRITER to the two lists.  A module-level `CSV_HEADER`
    that the actual writer does not call is worse than no header at all:
    the CSV on disk keeps the old column set, the row builder keeps the new
    one, and nothing complains.  (The writer used to carry its OWN header
    literal and its OWN row literal, and formatted the rate columns with
    `f"{m.gen_validity:.4f}"` — which raises TypeError the moment the value
    is None, i.e. on the first epoch whose generation eval drew no samples.)
    """
    row = cfm.csv_row(_measured_epoch(cfm))
    assert len(row) == len(cfm.CSV_HEADER)
    assert len(row) == len(list(cfm.CSV_HEADER))

    import inspect

    src = inspect.getsource(cfm)
    assert "w.writerow(list(CSV_HEADER))" in src, (
        "the per_epoch_metrics.csv writer does not use the shared CSV_HEADER"
    )
    assert "w.writerow(csv_row(m))" in src, (
        "the per_epoch_metrics.csv writer does not use the shared csv_row"
    )


# ---------------------------------------------------------------------------
# 3 — the JSON carries the same keys as the CSV
# ---------------------------------------------------------------------------
def test_json_carries_the_same_keys(cfm):
    """`asdict(EpochMetrics)` is the JSON side of the artifact."""
    payload = asdict(_measured_epoch(cfm))
    for name in (
        "gen_molecule_validity",
        "gen_molecule_validity_n",
        "gen_molecule_validity_total",
        "gen_end_to_end_rate",
        "gen_measured",
    ):
        assert name in payload, f"per_epoch_metrics.json is missing {name!r}"
        assert name in cfm.CSV_HEADER, (
            f"{name!r} is in the JSON but not in the CSV header — the two "
            "sides of the artifact must carry the same column set"
        )
    assert payload["gen_molecule_validity_total"] == 16
    assert payload["gen_measured"] == 1


# ---------------------------------------------------------------------------
# 4 — an unmeasured epoch writes nan, never 0.0000  (N1, the本体)
# ---------------------------------------------------------------------------
def test_unmeasured_epoch_writes_nan_not_zero(cfm):
    """`gen_validity_n == 0` ⇒ "nan" in the CSV and null in the JSON."""
    fields = cfm._gen_metric_fields(0, 0, {})
    kwargs = dict(BASE_EPOCH)
    kwargs.update(fields)
    m = cfm.EpochMetrics(**kwargs)

    row = cfm.csv_row(m)
    header = list(cfm.CSV_HEADER)

    # The rate cells are the literal sentinel.
    for name in (
        "gen_validity",
        "gen_largest_fragment_frac",
        "gen_single_fragment_ratio",
        "gen_molecule_validity",
        "gen_end_to_end_rate",
        "gen_metal_ratio",
        "gen_sa_mean",
    ):
        assert row[header.index(name)] == "nan", (
            f"{name} rendered {row[header.index(name)]!r} for an UNMEASURED "
            "epoch — it must be the literal 'nan', never '0.0000' and never "
            "'-1' (a real number that would survive a median filter)"
        )

    # The COUNT cells stay bare ints: 0 there is a fact, not a rate.
    assert row[header.index("gen_validity_n")] == "0"
    assert row[header.index("gen_molecule_validity_n")] == "0"
    assert row[header.index("gen_molecule_validity_total")] == "0"
    assert row[header.index("gen_measured")] == "0"

    # The JSON side is null, never 0.0.
    payload = asdict(m)
    assert payload["gen_validity"] is None
    assert payload["gen_end_to_end_rate"] is None
    assert payload["gen_molecule_validity"] is None
    assert payload["gen_measured"] == 0


# ---------------------------------------------------------------------------
# 5 — a measured epoch renders numbers
# ---------------------------------------------------------------------------
def test_measured_epoch_renders_numbers(cfm):
    """The sentinel must not swallow REAL measurements."""
    row = cfm.csv_row(_measured_epoch(cfm))
    header = list(cfm.CSV_HEADER)
    assert row[header.index("gen_validity")] == "0.7500"  # 12 of 16
    assert row[header.index("gen_single_fragment_ratio")] == "0.5000"
    assert row[header.index("gen_molecule_validity")] == "0.5000"
    assert row[header.index("gen_end_to_end_rate")] == "0.3750"  # 0.75*0.5
    assert row[header.index("gen_measured")] == "1"
    assert "nan" not in row[header.index("gen_validity")]


# ---------------------------------------------------------------------------
# 6 — the end-to-end rate is the product
# ---------------------------------------------------------------------------
def test_end_to_end_rate_is_the_product(cfm):
    """`gen_end_to_end_rate == gen_validity * gen_single_fragment_ratio`.

    Either input missing ⇒ None.  A product with a missing factor is not a
    measurement, it is a number wearing a chemistry label.
    """
    assert cfm._end_to_end_rate(0.75, 0.5) == pytest.approx(0.375)
    assert cfm._end_to_end_rate(1.0, 1.0) == pytest.approx(1.0)
    assert cfm._end_to_end_rate(0.0, 0.0) == pytest.approx(0.0)
    assert cfm._end_to_end_rate(None, 0.5) is None
    assert cfm._end_to_end_rate(0.75, None) is None
    assert cfm._end_to_end_rate(None, None) is None

    fields = cfm._gen_metric_fields(12, 16, CHEM_MEASURED)
    assert fields["gen_end_to_end_rate"] == pytest.approx(0.375)

    # An unmeasured epoch has no end-to-end rate at all.
    unmeasured = cfm._gen_metric_fields(0, 0, {})
    assert unmeasured["gen_end_to_end_rate"] is None


# ---------------------------------------------------------------------------
# 7 — `gen_measured` follows the N3 predicate
# ---------------------------------------------------------------------------
def test_gen_measured_follows_the_n3_predicate(cfm):
    """`gen_measured == 1` iff `gen_validity_n > 0` — nothing else.

    The case that matters: `gen_eval_n == 0` with `gen_validity_n > 0` is a
    REAL measurement (the eval ran; only the per-seed record was lost), so
    it must read as measured.
    """
    assert cfm._gen_metric_fields(0, 0, {})["gen_measured"] == 0
    assert cfm._gen_metric_fields(0, 0, CHEM_MEASURED)["gen_measured"] == 0
    assert cfm._gen_metric_fields(12, 16, CHEM_MEASURED)["gen_measured"] == 1

    # The predicate constant names itself, so a reader does not have to guess.
    assert cfm.GEN_UNMEASURED_PREDICATE == "gen_validity_n == 0"

    # `gen_eval_n` is NOT the predicate — the trainer's own bookkeeping
    # count.  A row with no per-seed record but a real parse count is a
    # measurement.
    fields = cfm._gen_metric_fields(12, 16, CHEM_MEASURED)
    assert fields["gen_measured"] == 1
    assert fields["gen_validity"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# 8 — the census median must not count a "nan" cell
# ---------------------------------------------------------------------------
def test_census_col_drops_the_nan_cells(cfm):
    """`wf_cfm_run_census._col` must ignore the sentinel cells.

    The census computes per-run medians over a CSV column.  A "nan" cell is
    NOT MEASURED and must not enter a median; a "0.0000" cell WOULD enter it
    and drag the median down — which is exactly the bug this change removes.
    """
    try:
        from molmetal.scripts import wf_cfm_run_census as _c
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"census module unavailable: {exc}")

    rows = [
        {"gen_single_fragment_ratio": "0.5000"},
        {"gen_single_fragment_ratio": "nan"},
        {"gen_single_fragment_ratio": "0.3000"},
        {"gen_single_fragment_ratio": ""},
    ]
    vals = _c._col(rows, "gen_single_fragment_ratio")
    assert 0.5 in vals and 0.3 in vals
    assert len(vals) == 2, (
        f"_col returned {vals!r} — the 'nan' / empty cells must be dropped, "
        "not counted as 0.0"
    )


# ---------------------------------------------------------------------------
# 9 — the early-stop degradation must NAME the predicate
# ---------------------------------------------------------------------------
def test_early_stop_degrades_and_names_the_predicate(cfm):
    """`molecule_validity` on an unmeasured epoch degrades and says why.

    The note must name `gen_validity_n == 0`, so a reader does not have to
    guess which count was zero (the bond side has a DIFFERENT predicate).
    """
    name, value, note = cfm.resolve_early_stop_criterion(
        "molecule_validity",
        decode_ratio=0.5,
        gen_validity=None,
        gen_n=0,
        bond_exact_type_rate=0.0,
        bond_n_scored=0,
    )
    assert name != "molecule_validity", (
        "an unmeasured epoch must NOT gate on molecule_validity"
    )
    assert "gen_validity_n == 0" in note, (
        f"the degradation note does not name the predicate: {note!r}"
    )

    # The same holds for the parse-rate criterion.
    name2, _value2, note2 = cfm.resolve_early_stop_criterion(
        "gen_validity",
        decode_ratio=0.5,
        gen_validity=None,
        gen_n=0,
        bond_exact_type_rate=0.0,
        bond_n_scored=0,
    )
    assert name2 != "gen_validity"
    assert "gen_validity_n == 0" in note2

    # A MEASURED epoch still gates on the requested criterion.
    name3, value3, note3 = cfm.resolve_early_stop_criterion(
        "molecule_validity",
        decode_ratio=0.5,
        gen_validity=0.75,
        gen_n=16,
        bond_exact_type_rate=0.0,
        bond_n_scored=0,
        molecule_validity=0.5,
    )
    assert name3 == "molecule_validity"
    assert value3 == pytest.approx(0.5)
    assert note3 == ""
