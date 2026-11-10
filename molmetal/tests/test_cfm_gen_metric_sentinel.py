"""Wf-A114/T85 (Cluster T) — the LAST fabricated-zero hole in the trainer.

WHY this file exists
--------------------
Wf-A114 fixed the "the generation eval drew NO samples" hole: a rate whose
denominator is zero was published as a MEASURED ``0.0000``.  One hole was
left, and it is the harder one:

    the eval RAN, drew N samples, and NONE of them parsed.

There the parse rate (``gen_validity`` = 0 of N) IS a real measurement —
its denominator is N, the number of samples DRAWN.  But every CHEMISTRY
rate has a different denominator: the number of rows the decode actually
SCORED (``n_chemistry_scored``).  When nothing parsed, that count is 0 and
the rate is 0/0 — UNDEFINED, not zero.

``summarise_decode_chemistry``'s zero-row branch publishes ``0.0`` for
those keys (it has to: a reader must never have to branch on the row
count), and the writer's key-presence check turned that placeholder into a
published number.  MEASURED on disk (2026-10-30): 66 epoch rows across
the CFM corpus carry ``gen_validity_n > 0`` with ``gen_validity = 0.0000``,
and 2 of 59 runs END on such a row — so their census
``gen_single_fragment_ratio_end`` read 0.0 for a rate that was never
defined.

THE FIX (three parts, one owner)
--------------------------------
1. ``_chem_val`` reads the chemistry block's OWN denominator
   (``n_chemistry_scored``).  Zero scored rows ⇒ the rate is not measured
   ⇒ ``None``.  ``summarise_decode_chemistry`` is NOT changed: its
   zero-row branch keeps its keys and its placeholders, because five other
   consumers read those placeholders directly (see the audit test below).
2. ``molecule_validity`` / ``decode_ratio`` / ``rdkit_parse_rate`` are
   rates over the samples DRAWN, so they are ``None`` exactly when the
   eval drew nothing — never ``0.0``.  A measured 0 of N stays ``0.0``.
3. The early-stop criterion reads ``molecule_validity``; when it is
   ``None`` the existing degradation ladder must fire and must NAME the
   reason.

CPU-only.  No checkpoint, no sampler, no training, no CUDA.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def cfm():
    try:
        from molmetal.scripts import train_cfm_yuelbond_wasserstein as _m
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"CFM trainer unavailable: {exc}")
    return _m


#: The chemistry block the trainer builds when the eval RAN, drew 16
#: samples, and NONE of them parsed.  Built from the trainer's own
#: ``summarise_decode_chemistry`` so the test cannot drift from the
#: production shape, then completed with the fields the generation block
#: adds (the parse rate and the strict rate, both over DRAWN samples).
def _zero_scored_block(cfm, n_drawn: int = 16) -> dict:
    block = dict(cfm.summarise_decode_chemistry([None] * n_drawn))
    block.update(
        {
            "n_samples": int(n_drawn),
            "decode_ratio": 0.0,
            "rdkit_parse_rate": 0.0,
            "molecule_validity": 0.0,
            "molecule_validity_n": 0,
            "molecule_validity_total": int(n_drawn),
        }
    )
    return block


#: The chemistry block of an epoch whose generation eval never ran.  The
#: trainer leaves it empty; the ``max(denominator, 1)`` idiom used to fill
#: it with a measured 0.0 anyway.
def _never_ran_block() -> dict:
    return {
        "n_samples": 0,
        "decode_ratio": 0.0,
        "rdkit_parse_rate": 0.0,
        "molecule_validity": 0.0,
        "molecule_validity_n": 0,
        "molecule_validity_total": 0,
    }


#: Every chemistry rate whose denominator is the SCORED-row count.
CHEM_RATE_FIELDS = (
    "gen_largest_fragment_frac",
    "gen_single_fragment_ratio",
    "gen_metal_ratio",
    "gen_mean_heavy_atoms",
    "gen_rings_per_mol",
    "gen_ring_5_6_share",
    "gen_ring_3_share",
    "gen_raw_rings_per_mol",
    "gen_raw_ring_5_6_share",
    "gen_raw_ring_3_share",
    # the end-to-end rate is the product of the parse rate and the
    # single-fragment rate, so it inherits the undefined factor
    "gen_end_to_end_rate",
)


# ---------------------------------------------------------------------------
# 1 — "the eval ran and drew 0 parseable" publishes None, not 0.0
# ---------------------------------------------------------------------------
def test_zero_scored_rows_publish_none_not_zero(cfm):
    """0 scored rows ⇒ every chemistry rate is 0/0 ⇒ NOT MEASURED."""
    chem = _zero_scored_block(cfm)
    assert chem["n_chemistry_scored"] == 0, (
        "the fixture must be the zero-row case, or the test proves nothing"
    )
    fields = cfm._gen_metric_fields(0, 16, chem)
    for name in CHEM_RATE_FIELDS:
        assert fields[name] is None, (
            f"{name} published {fields[name]!r} for an epoch that scored ZERO "
            "chemistry rows — 0/0 is UNDEFINED, never a measured 0.0"
        )


def test_zero_scored_rows_csv_renders_nan(cfm):
    """The CSV cell must be the sentinel, never ``0.0000``."""
    fields = cfm._gen_metric_fields(0, 16, _zero_scored_block(cfm))
    header = list(cfm.CSV_HEADER)
    row = cfm.csv_row(cfm.EpochMetrics(**_epoch_kwargs(cfm, fields)))
    for name in CHEM_RATE_FIELDS:
        assert row[header.index(name)] == "nan", (
            f"{name} rendered {row[header.index(name)]!r} — an unmeasured rate "
            "must be the literal 'nan', never '0.0000'"
        )


# ---------------------------------------------------------------------------
# 2 — a genuinely measured 0.0 stays numeric
# ---------------------------------------------------------------------------
def test_measured_zero_is_still_numeric(cfm):
    """0 of 16 parsed IS a measurement: its denominators are 16, not 0."""
    fields = cfm._gen_metric_fields(0, 16, _zero_scored_block(cfm))

    # The parse rate: 0 of 16 drawn samples parsed.
    assert fields["gen_validity"] == pytest.approx(0.0)
    assert fields["gen_validity_val"] == 0
    assert fields["gen_validity_n"] == 16
    assert fields["gen_measured"] == 1, (
        "the eval ran and drew 16 samples — that IS a measurement, so the "
        "epoch must not be filed as 'not measured'"
    )

    # The strict rate: 0 of 16 drawn samples were a single-fragment,
    # metal-bearing molecule.  Its denominator is the samples DRAWN, so it
    # is a measured 0.0 — and it must stay distinguishable from the
    # undefined per-scored-row rates above.
    assert fields["gen_molecule_validity"] == pytest.approx(0.0)
    assert fields["gen_molecule_validity_n"] == 0
    assert fields["gen_molecule_validity_total"] == 16

    # ... and the CSV renders it as a number, not as the sentinel.
    header = list(cfm.CSV_HEADER)
    row = cfm.csv_row(cfm.EpochMetrics(**_epoch_kwargs(cfm, fields)))
    assert row[header.index("gen_validity")] == "0.0000"
    assert row[header.index("gen_molecule_validity")] == "0.0000"
    assert row[header.index("gen_measured")] == "1"


def test_a_fully_scored_epoch_keeps_every_number(cfm):
    """The sentinel must not swallow a REAL chemistry measurement."""
    row_chem = {
        "n_chemistry_scored": 2,
        "single_fragment_ratio": 0.5,
        "largest_fragment_frac_mean": 0.75,
        "metal_ratio": 1.0,
        "mean_heavy_atoms": 12.0,
        "rings_per_mol_mean": 1.5,
        "ring_5_6_share": 0.8,
        "ring_3_share": 0.1,
        "raw_rings_per_mol_mean": 0.9,
        "raw_ring_5_6_share": 0.6,
        "raw_ring_3_share": 0.2,
        "molecule_validity": 0.5,
        "molecule_validity_n": 8,
        "molecule_validity_total": 16,
    }
    fields = cfm._gen_metric_fields(12, 16, row_chem)
    assert fields["gen_validity"] == pytest.approx(0.75)
    assert fields["gen_single_fragment_ratio"] == pytest.approx(0.5)
    assert fields["gen_largest_fragment_frac"] == pytest.approx(0.75)
    assert fields["gen_metal_ratio"] == pytest.approx(1.0)
    assert fields["gen_mean_heavy_atoms"] == pytest.approx(12.0)
    assert fields["gen_rings_per_mol"] == pytest.approx(1.5)
    assert fields["gen_end_to_end_rate"] == pytest.approx(0.75 * 0.5)
    assert fields["gen_molecule_validity"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 3 — molecule_validity: None when the eval never ran, 0.0 when it found none
# ---------------------------------------------------------------------------
def test_molecule_validity_is_none_when_the_eval_never_ran(cfm):
    """`gen_validity_n == 0` ⇒ the strict rate is 0/0 ⇒ None."""
    fields = cfm._gen_metric_fields(0, 0, _never_ran_block())
    assert fields["gen_validity"] is None
    assert fields["gen_molecule_validity"] is None, (
        "the strict rate published a number for an epoch whose generation "
        "eval never ran — the parse rate and the strict rate must agree on "
        "what 'no measurement' means"
    )
    assert fields["gen_measured"] == 0
    assert fields["gen_molecule_validity_n"] == 0
    assert fields["gen_molecule_validity_total"] == 0


def test_molecule_validity_is_zero_when_the_eval_found_none(cfm):
    """The eval ran and found no strict molecule: 0 of N is a measurement."""
    fields = cfm._gen_metric_fields(0, 16, _zero_scored_block(cfm))
    assert fields["gen_molecule_validity"] == pytest.approx(0.0)
    assert fields["gen_molecule_validity_total"] == 16


def test_measured_rate_helper_is_none_only_for_an_empty_denominator(cfm):
    """The replacement for ``max(denominator, 1)``."""
    assert cfm._measured_rate(0, 16) == pytest.approx(0.0)
    assert cfm._measured_rate(12, 16) == pytest.approx(0.75)
    assert cfm._measured_rate(3, 4) == pytest.approx(0.75)
    assert cfm._measured_rate(0, 0) is None
    assert cfm._measured_rate(5, 0) is None
    assert cfm._measured_rate(0, -1) is None


# ---------------------------------------------------------------------------
# 4 — the early-stop criterion degrades on a None molecule_validity
# ---------------------------------------------------------------------------
def test_early_stop_degrades_on_a_none_molecule_validity(cfm):
    """A None strict rate must DEGRADE, never raise and never gate."""
    name, value, note = cfm.resolve_early_stop_criterion(
        "molecule_validity",
        decode_ratio=0.5,
        gen_validity=0.0,
        gen_n=16,
        bond_exact_type_rate=0.0,
        bond_n_scored=0,
        molecule_validity=None,
    )
    assert name != "molecule_validity", (
        "an unmeasured strict rate must NOT gate on molecule_validity"
    )
    assert note, "the degradation note must be non-empty — it names the reason"
    assert "molecule_validity" in note
    assert isinstance(value, float)

    # The same ladder must fire for the parse-rate criterion and for the
    # combined criterion — None must not raise anywhere.
    for requested in ("gen_validity", "combined", "bond_exact_type"):
        _n, _v, _note = cfm.resolve_early_stop_criterion(
            requested,
            decode_ratio=0.5,
            gen_validity=0.0,
            gen_n=16,
            bond_exact_type_rate=0.0,
            bond_n_scored=0,
            molecule_validity=None,
        )


def test_early_stop_names_the_predicate_when_the_eval_never_ran(cfm):
    """The note must name WHICH count was zero (the bond side differs)."""
    _name, _value, note = cfm.resolve_early_stop_criterion(
        "molecule_validity",
        decode_ratio=0.5,
        gen_validity=None,
        gen_n=0,
        bond_exact_type_rate=0.0,
        bond_n_scored=0,
        molecule_validity=None,
    )
    assert "gen_validity_n == 0" in note, (
        f"the degradation note does not name the predicate: {note!r}"
    )


def test_early_stop_still_gates_on_a_measured_strict_rate(cfm):
    """The sentinel must not swallow a REAL strict rate."""
    name, value, note = cfm.resolve_early_stop_criterion(
        "molecule_validity",
        decode_ratio=0.5,
        gen_validity=0.0,
        gen_n=16,
        bond_exact_type_rate=0.0,
        bond_n_scored=0,
        molecule_validity=0.0,
    )
    assert name == "molecule_validity"
    assert value == pytest.approx(0.0)
    # A measured 0.0 IS the collapse this criterion exists to name, so the
    # note records the saturation rather than swallowing the number.
    assert "saturated" in note


# ---------------------------------------------------------------------------
# 5 — the summarise zero-row branch is UNCHANGED (and its consumers are safe)
# ---------------------------------------------------------------------------
def test_summarise_zero_row_branch_keeps_its_keys_and_placeholders(cfm):
    """The undefined reading happens at the ARTIFACT boundary, not here.

    ``summarise_decode_chemistry`` still publishes ``0.0`` for an empty row
    set, because five consumers read those placeholders directly and a
    reader must never have to branch on the row count:

    * ``molmetal/scripts/wfa111_ring_prior_ab.py`` — ``single_fragment_ratio``
      / ``largest_fragment_frac_mean`` / ``metal_ratio`` (it publishes
      ``n_chemistry_scored`` beside them);
    * ``molmetal/scripts/wfa107d_decode_head_swap.py`` — ``ring_3_share`` /
      ``ring_5_6_share`` / ``rings_per_mol_mean`` / ``single_fragment_ratio``;
    * ``molmetal/scripts/cfm_fragmentation_audit.py`` — via the CSV/JSON;
    * the four test modules that call the summary directly.

    So the branch is NOT changed here: the trainer's own writer reads the
    denominator and decides that the rate is undefined.
    """
    summary = cfm.summarise_decode_chemistry([])
    assert summary["n_chemistry_scored"] == 0
    for key in (
        "largest_fragment_frac_mean",
        "single_fragment_ratio",
        "metal_ratio",
        "mean_heavy_atoms",
        "rings_per_mol_mean",
        "ring_5_6_share",
        "ring_3_share",
        "raw_rings_per_mol_mean",
        "raw_ring_5_6_share",
        "raw_ring_3_share",
    ):
        assert key in summary, f"{key} must exist so a reader never branches"
        assert summary[key] == 0.0, (
            f"{key} changed in the zero-row branch — the OTHER consumers of "
            "this summary read it as a placeholder and must not be broken"
        )


# ---------------------------------------------------------------------------
# 6 — the `max(denominator, 1)` idiom is gone from the generation path
# ---------------------------------------------------------------------------
def test_max_denominator_idiom_is_gone(cfm):
    """A source guard, so the hole cannot be re-opened by a later edit."""
    import inspect

    src = inspect.getsource(cfm)
    for forbidden in (
        '_gen_chem["decode_ratio"] = _gen_ok / max(_gen_tot, 1)',
        '_gen_chem["rdkit_parse_rate"] = _gen_ok / max(_gen_tot, 1)',
        '_gen_chem["molecule_validity"] = _n_mol / max(_gen_tot, 1)',
        'chemistry_out["decode_ratio"] = n_ok / max(len(clouds), 1)',
        'chemistry_out["molecule_validity"] = _n_mol / max(len(clouds), 1)',
        "{_gen_ok / max(_gen_tot, 1):.3f}",
    ):
        assert forbidden not in src, (
            f"the trainer still contains {forbidden!r} — a rate over an empty "
            "denominator is 0/0, so it must be None (not measured)"
        )


# ---------------------------------------------------------------------------
# 7 — the Wf-A114 column suite still passes
# ---------------------------------------------------------------------------
def test_existing_gen_metric_column_suite_still_passes():
    """The already-landed Wf-A114 suite must not regress."""
    root = Path(__file__).resolve().parents[2]
    target = root / "molmetal" / "tests" / "test_cfm_gen_metric_columns.py"
    assert target.is_file(), f"{target} is missing"
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(target)],
        cwd=str(root),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (
        "the Wf-A114 generation-column suite regressed:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _epoch_kwargs(cfm, fields: dict) -> dict:
    """The fields ``EpochMetrics`` needs positionally; the rest defaults."""
    kwargs = {
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
    kwargs.update(fields)
    return kwargs


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
