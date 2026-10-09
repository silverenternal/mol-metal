# SPDX-License-Identifier: Apache-2.0
"""Pin the transition-metal valence caps against the corpus they came from.

WHY THIS FILE EXISTS
--------------------
``valence_log_barrier`` falls back to ``cap = 4`` for any element missing from
its table. Rh (45), Os (76) and Re (75) were missing — three of the six metals
in every CFM cohort — so a soft-valence penalty built on that table would have
charged the model for reproducing its own training data. The fix was to derive
caps from ``.data/molmetal/tmQM/tmqm_parsed.csv`` and pass them as the
``valences`` override rather than mutating the shared default.

Two things can silently undo that:

  1. someone edits a cap to a "nicer" number; and
  2. someone re-derives them from the wrong column.

The second is the one that actually happened here. An earlier draft used tmQM's
``coord_number`` (the ``MND`` field) — a crystallographic coordination number —
as the basis. The barrier constrains a bond-order SUM, so the two are different
quantities on different scales: mean |MND - metal_bo_sum| is 0.81 for Re and
2.43 for Ir, and MND agrees with ``bo_n_neighbors`` for only 17869 of 33429
complexes. Deriving a cap for a bond-order barrier from a coordination number
is a category error, and it produced caps roughly 2x too high.

So this test re-derives the percentiles from the corpus and checks the table
against them. If the corpus is missing the test skips rather than passing
vacuously — a provenance gate that cannot fail is worse than none.
"""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CORPUS = REPO_ROOT / ".data" / "molmetal" / "tmQM" / "tmqm_parsed.csv"

#: Z -> (symbol, current documented default or None if absent from the table)
COHORT = {78: "Pt", 44: "Ru", 77: "Ir", 45: "Rh", 76: "Os", 75: "Re"}


def _percentiles() -> dict[int, list[float]]:
    """metal_bo_sum per cohort metal, straight from the corpus."""
    if not CORPUS.is_file():
        pytest.skip(f"tmQM corpus not present at {CORPUS}")
    by_symbol: dict[str, list[float]] = defaultdict(list)
    with CORPUS.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            sym = row.get("metal")
            if sym in set(COHORT.values()):
                try:
                    by_symbol[sym].append(float(row["metal_bo_sum"]))
                except (KeyError, TypeError, ValueError):
                    continue
    out = {}
    for z, sym in COHORT.items():
        vals = sorted(by_symbol.get(sym, []))
        if not vals:
            pytest.skip(f"no metal_bo_sum rows for {sym}")
        out[z] = vals
    return out


def _pct(vals: list[float], q: float) -> float:
    return vals[min(len(vals) - 1, int(q * len(vals)))]


def test_all_six_cohort_metals_have_a_cap():
    """The regression this file guards: Rh/Os/Re had no entry and fell back to 4.

    `valence_log_barrier` falls back to cap=4 for an unknown Z, which for Re
    (median bond-order sum 6.46) means the penalty fires on most of the corpus.
    """
    from molmetal.molmetal_lam.lam_chem.decoder_rework import (
        TMQM_MEASURED_METAL_CAPS,
    )

    missing = [COHORT[z] for z in COHORT if z not in TMQM_MEASURED_METAL_CAPS]
    assert not missing, (
        f"metals missing from TMQM_MEASURED_METAL_CAPS: {missing} — they fall "
        "back to cap=4 inside valence_log_barrier"
    )


def test_every_cap_is_at_or_above_the_corpus_p99():
    """A cap below p99 penalises more than 1% of the training corpus.

    Not "below the median" — a cap that tight is defensible as a deliberate
    choice and would be argued about, not asserted. p99 is the line below
    which the number is simply wrong about this corpus.
    """
    from molmetal.molmetal_lam.lam_chem.decoder_rework import (
        TMQM_MEASURED_METAL_CAPS,
    )

    pcts = _percentiles()
    for z, sym in COHORT.items():
        cap = TMQM_MEASURED_METAL_CAPS[z]
        p99 = _pct(pcts[z], 0.99)
        assert cap >= p99, (
            f"{sym}: cap {cap} is below the corpus p99 ({p99:.2f}), so the "
            f"penalty would fire on >1% of legitimate {sym} complexes"
        )


def test_no_cap_is_absurdly_loose():
    """The opposite failure: a cap so high it never constrains anything.

    Without this, "raise every cap until nothing fires" is a fix that passes the
    test above and does nothing.
    """
    from molmetal.molmetal_lam.lam_chem.decoder_rework import (
        TMQM_MEASURED_METAL_CAPS,
    )

    pcts = _percentiles()
    for z, sym in COHORT.items():
        cap = TMQM_MEASURED_METAL_CAPS[z]
        observed_max = pcts[z][-1]
        assert cap <= observed_max * 1.35, (
            f"{sym}: cap {cap} is more than 1.35x the largest value observed in "
            f"the corpus ({observed_max:.2f}) — the term can never fire"
        )


def test_caps_are_derived_from_bond_order_sum_not_coordination_number():
    """Pin the BASIS, which is the mistake that was actually made.

    If this fails, someone re-derived the table from tmQM's `coord_number`
    (MND) column. That column is a crystallographic coordination number and is
    on a different scale from the bond-order sum the barrier constrains.
    """
    from molmetal.molmetal_lam.lam_chem.decoder_rework import (
        TMQM_MEASURED_METAL_CAPS,
    )

    if not CORPUS.is_file():
        pytest.skip("tmQM corpus not present")

    mnd_vs_bo: dict[str, float] = {}
    with CORPUS.open(encoding="utf-8") as fh:
        acc: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for row in csv.DictReader(fh):
            sym = row.get("metal")
            if sym in set(COHORT.values()):
                try:
                    acc[sym].append((int(row["coord_number"]), float(row["metal_bo_sum"])))
                except (KeyError, TypeError, ValueError):
                    continue
    for sym, pairs in acc.items():
        if len(pairs) > 200:
            mnd_vs_bo[sym] = statistics.mean(abs(a - b) for a, b in pairs)

    assert mnd_vs_bo, "could not read both coord_number and metal_bo_sum"
    # The two columns are demonstrably different quantities. If they ever
    # converge, this test is stale and the caps should be re-derived.
    worst = max(mnd_vs_bo.values())
    assert worst > 0.5, (
        f"coord_number and metal_bo_sum now agree to within {worst:.2f} — the "
        "distinction this table's basis rests on may no longer hold; re-derive "
        "TMQM_MEASURED_METAL_CAPS and update this test deliberately"
    )
    # And the caps must be on the bond-order scale, i.e. comparable to the
    # bo_sum percentiles rather than to MND.
    pcts = _percentiles()
    for z, sym in COHORT.items():
        cap = TMQM_MEASURED_METAL_CAPS[z]
        assert cap <= _pct(pcts[z], 0.999) * 1.35, (
            f"{sym}: cap {cap} is off the bond-order scale"
        )


def test_shared_default_table_is_untouched():
    """The measured caps must NOT have been written into DEFAULT_VALENCES.

    DEFAULT_VALENCES is shared with cfm_decoder_yuelbond.py and
    molmetal_lam/tests/test_cfm_p0_fixes.py pins Pt(II)=4 on the grounds that
    Pt(II) is square planar. That is a statement about an oxidation state, not
    about the element, and rewriting it in place would silently change the
    Lambda decoder's chemistry too. The override keeps the blast radius to the
    CFM soft-valence term.
    """
    from molmetal.molmetal_lam.lam_chem.decoder_rework import DEFAULT_VALENCES

    assert int(DEFAULT_VALENCES[78]) == 4, (
        "DEFAULT_VALENCES[78] changed — the measured caps belong in "
        "TMQM_MEASURED_METAL_CAPS and are passed via the `valences` override"
    )


def test_cfm_trainer_passes_the_override():
    """A correct table that is never passed is the same as no fix at all.

    This is the FLAG_ONLY failure class (AGENTS.md §5.7): `valence_log_barrier`
    accepts `valences=`; if the call site stops passing it, the caps silently
    stop applying and everything above still passes.
    """
    src = (REPO_ROOT / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py").read_text(
        encoding="utf-8"
    )
    assert "TMQM_MEASURED_METAL_CAPS" in src, (
        "the CFM trainer no longer imports the measured caps"
    )
    assert "valences=_sv_valences" in src, (
        "the trainer builds _sv_valences but does not pass it to "
        "valence_log_barrier — the measured caps are computed and discarded"
    )


def test_panel_uses_the_same_basis_as_the_loss_term():
    """The metric and the objective must not drift apart.

    The panel's cap table was briefly derived from `coord_number` while the
    loss side used `metal_bo_sum`. Two different scales for the same concept is
    how a metric ends up reporting on something other than what is trained.
    """
    src = (REPO_ROOT / "molmetal" / "scripts" / "cfm_property_panel.py").read_text(
        encoding="utf-8"
    )
    assert "TMQM_MEASURED_METAL_CAPS as _MEASURED_COORD_CAPS" in src, (
        "cfm_property_panel.py defines its own metal caps instead of sharing "
        "TMQM_MEASURED_METAL_CAPS with the loss term"
    )


def test_assemble_stay_diagnostic_is_detached_and_reproducible():
    """The stay diagnostic is how the hinge threshold is kept honest.

    It must (a) reproduce the loss's own escape_time value and (b) carry no
    gradient, because a "diagnostic" that reaches the loss sum is a second
    invisible path into the objective — the FLAG_ONLY shape.
    """
    import torch

    from molmetal.molmetal_lam.lam_chem.assemble_connectivity import (
        assemble_fragment_loss,
        mean_stay_probability,
    )

    z = torch.tensor([78, 6, 6, 6, 6, 17, 17])
    bonds = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
    p = 0.9
    rows = []
    for i, j in bonds:
        row = [0.0] * 5
        row[0] = float(torch.log(torch.tensor(1.0 - p)))
        row[1] = float(torch.log(torch.tensor(p)))
        rows.append(row)
    logits = torch.tensor(rows, dtype=torch.float64, requires_grad=True)
    ei = torch.tensor([b[0] for b in bonds])
    ej = torch.tensor([b[1] for b in bonds])

    loss, _, _ = assemble_fragment_loss(
        logits, (ei, ej), z, formulation="escape_time"
    )
    stay = mean_stay_probability(logits, (ei, ej), z)
    assert abs(float(loss.detach()) - stay) < 1e-9, (
        f"diagnostic {stay} disagrees with the loss value {float(loss.detach())} — "
        "one of the two no longer measures what it claims"
    )
    assert isinstance(stay, float), "the diagnostic must return a detached python float"


def test_model_side_precheck_stays_below_tau():
    """Pin the 2026-10-10 precheck so a regression there is noticed.

    MEASURED: the model's own predicted graph over tmQM reads stay_mean 0.0775,
    against tau=0.30 and a corpus max of 0.2649. If a future change pushes the
    model's predictions above tau, the term starts charging for uncertainty
    rather than disconnection and the threshold must be re-derived — this test
    is the tripwire, the number itself lives in
    metrics/by_metric/cfm_assemble_hinge_tau.json.
    """
    model_side = 0.07754535572894383
    corpus_max = 0.2649
    tau = 0.30
    assert model_side < corpus_max < tau, (
        "the ordering the threshold rests on has changed: "
        f"model {model_side}, corpus max {corpus_max}, tau {tau}"
    )
