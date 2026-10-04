"""Guards for the metrics that ``decode_ratio`` cannot see.

2026-10-04 — ``decode_ratio`` counts a decode as good when RDKit parses it,
and MEASURED it read 0.977 on a bond head that predicted BOND_NO_BOND for
nearly every pair.  ``C.C.C.C`` is a perfectly sanitizable SMILES, so a
decoder emitting every atom as its own fragment scores 1.000 on a model that
has learned nothing.  The metric is anti-correlated with correctness over the
range that matters, so it cannot gate a retrain and must not be read as a
chemistry result.

:func:`decode_chemistry` supplies the facts that separate the two cases:
fragment count, largest-fragment fraction, and metal presence.  These tests
pin the separation, because the failure mode is silent — a high
``decode_ratio`` looks like success.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def trainer():
    return _load(
        PROJECT_ROOT
        / "molmetal"
        / "scripts"
        / "train_cfm_yuelbond_wasserstein.py",
        "_t_chem_blind_spot",
    )


def _rdkit_ok() -> bool:
    try:
        import rdkit  # noqa: F401

        return True
    except ImportError:
        return False


needs_rdkit = pytest.mark.skipif(not _rdkit_ok(), reason="rdkit not installed")


# ---------------------------------------------------------------------------
# The separation decode_ratio cannot make
# ---------------------------------------------------------------------------
@needs_rdkit
def test_loose_atoms_parse_but_are_not_a_molecule(trainer):
    """THE regression guard: 'C.C.C.C' passes RDKit and is not a molecule.

    This is the exact input that made decode_ratio read 0.977 on a head that
    had learned nothing.  Both strings below are RDKit-strict valid, so
    nothing in the decode path can tell them apart — only the chemistry
    report can.
    """
    from rdkit import Chem

    loose = "C.C.C.C.C.C"
    real = "c1ccccc1CCN"

    assert Chem.MolFromSmiles(loose) is not None, "premise: loose atoms parse"
    assert Chem.MolFromSmiles(real) is not None, "premise: the real one parses"

    loose_c = trainer.decode_chemistry(loose)
    real_c = trainer.decode_chemistry(real)
    assert loose_c is not None and real_c is not None

    assert loose_c["n_fragments"] == 6
    assert loose_c["largest_fragment_frac"] < 0.2, (
        "six loose carbons must show a dominant fragment of 1/6 — this is the "
        "signal decode_ratio is blind to"
    )
    assert real_c["n_fragments"] == 1
    assert real_c["largest_fragment_frac"] == 1.0

    # The aggregate must separate a batch of each, not just one string.
    bad = trainer.summarise_decode_chemistry([loose_c] * 8)
    good = trainer.summarise_decode_chemistry([real_c] * 8)
    assert bad["single_fragment_ratio"] == 0.0
    assert good["single_fragment_ratio"] == 1.0
    assert bad["largest_fragment_frac_mean"] < good["largest_fragment_frac_mean"]


@needs_rdkit
def test_metal_presence_is_reported(trainer):
    """The corpus is metal complexes; a decode with no metal is not one."""
    pt = trainer.decode_chemistry("[Pt](Cl)(Cl)(N)N")
    organic = trainer.decode_chemistry("CCO")
    assert pt is not None and organic is not None
    assert pt["has_metal"] is True
    assert organic["has_metal"] is False


@needs_rdkit
def test_summary_reports_metal_ratio(trainer):
    rows = [
        trainer.decode_chemistry("[Pt](Cl)(Cl)(N)N"),
        trainer.decode_chemistry("CCO"),
        trainer.decode_chemistry("CCO"),
        trainer.decode_chemistry("CCO"),
    ]
    s = trainer.summarise_decode_chemistry(rows)
    assert s["metal_ratio"] == pytest.approx(0.25)
    assert s["n_chemistry_scored"] == 4


def test_unparseable_smiles_scores_none_not_zero(trainer):
    """A parse failure must be absent from the chemistry stats, not a zero.

    Counting it as ``largest_fragment_frac = 0.0`` would drag the mean down
    for a reason that has nothing to do with loose atoms.
    """
    rows = [None, None]
    if _rdkit_ok():
        rows.append(trainer.decode_chemistry("this is not a smiles"))
    s = trainer.summarise_decode_chemistry(rows)
    assert s["n_chemistry_scored"] == 0
    assert s["largest_fragment_frac_mean"] == 0.0


def test_empty_row_set_does_not_divide_by_zero(trainer):
    s = trainer.summarise_decode_chemistry([])
    assert s["n_chemistry_scored"] == 0
    assert s["single_fragment_ratio"] == 0.0


# ---------------------------------------------------------------------------
# The threshold-free ranking report
# ---------------------------------------------------------------------------
def test_pr_auc_is_one_for_a_perfect_ranking(trainer):
    """Separable scores must give AP 1.0 — else the sweep is miscomputed."""
    scores = [0.9, 0.8, 0.2, 0.1]
    labels = [1, 1, 0, 0]
    r = trainer._bond_ranking_report(scores, labels)
    assert r["bond_pr_auc"] == pytest.approx(1.0)
    assert r["bond_best_f1_at_threshold"] == pytest.approx(1.0)


def test_pr_auc_is_chance_for_an_inverted_ranking(trainer):
    """A perfectly WRONG ranking must not score well.

    MEASURED, the argmax F1 sits at a fixed operating point that class
    weighting moves; AP is what tells the two apart.  If this returned a high
    value the whole diagnostic would be worthless.
    """
    scores = [0.1, 0.2, 0.8, 0.9]
    labels = [1, 1, 0, 0]
    r = trainer._bond_ranking_report(scores, labels)
    assert r["bond_pr_auc"] < 0.5


def test_pr_auc_matches_the_base_rate_when_scores_are_constant(trainer):
    """No ranking information ⇒ AP equals the positive rate."""
    scores = [0.5] * 10
    labels = [1] * 3 + [0] * 7
    r = trainer._bond_ranking_report(scores, labels)
    assert r["bond_pr_auc"] == pytest.approx(0.3, abs=0.02)
    assert r["bond_positive_rate"] == pytest.approx(0.3)


def test_no_positives_is_reported_not_faked(trainer):
    r = trainer._bond_ranking_report([0.5, 0.6], [0, 0])
    assert r["bond_pr_auc"] == 0.0
    assert "no positives" in r["bond_pr_auc_note"]


def test_best_f1_is_labelled_optimistic(trainer):
    """The sweep selects its threshold on the pairs it scores.

    That is a ceiling on what calibration could buy, not a result, and the
    note must say so — otherwise a later reader quotes it as achieved.
    """
    r = trainer._bond_ranking_report([0.9, 0.1], [1, 0])
    assert "OPTIMISTIC" in r["bond_pr_auc_note"]


# ---------------------------------------------------------------------------
# Class-weight softening
# ---------------------------------------------------------------------------
def test_weight_power_zero_is_exactly_unweighted():
    """power 0 must reproduce the unweighted CE, so it can serve as baseline.

    This is what makes the three-point sweep (0.0 / 0.5 / 1.0) a controlled
    comparison rather than three unrelated runs.
    """
    import torch

    counts = torch.tensor([24378, 4460, 421, 28, 5661], dtype=torch.long)
    k = 5
    p = counts.float() / counts.sum().float()
    raw = counts.sum().float() / (k * counts.float().clamp(min=1.0))
    w = raw.pow(0.0)
    w = w / (p * w).sum()
    assert torch.allclose(w, torch.ones_like(w))


def test_weights_are_scale_neutral_at_every_power():
    """``sum_c p_c w_c == 1`` must hold, or bond_ce changes magnitude.

    Normalising on ``w.mean() == 1`` instead divides by ~54 on these counts
    and silently shrinks bond_ce by that factor, which removes the term from
    ``total = loss + ...`` — the opposite of the intent.
    """
    import torch

    counts = torch.tensor([24378, 4460, 421, 28, 5661], dtype=torch.long)
    k = 5
    p = counts.float() / counts.sum().float()
    raw = counts.sum().float() / (k * counts.float().clamp(min=1.0))
    for power in (0.0, 0.25, 0.5, 1.0):
        w = raw.pow(power)
        w = w / (p * w).sum()
        assert float((p * w).sum()) == pytest.approx(1.0, abs=1e-6), power


def test_softening_pulls_in_the_extreme_class():
    """sqrt must tame BOND_TRIPLE without discarding the ordering.

    MEASURED counts give BOND_TRIPLE 0.08% of pairs and a 250x textbook
    weight on 28 examples — enough to dominate the gradient.  The ordering
    must survive the softening.
    """
    import torch

    counts = torch.tensor([24378, 4460, 421, 28, 5661], dtype=torch.long)
    k = 5
    p = counts.float() / counts.sum().float()
    raw = counts.sum().float() / (k * counts.float().clamp(min=1.0))
    full = raw.pow(1.0)
    full = full / (p * full).sum()
    soft = raw.pow(0.5)
    soft = soft / (p * soft).sum()
    assert float(full[3]) > 200.0, "premise: the textbook triple weight is extreme"
    assert float(soft[3]) < 30.0, "sqrt must pull the extreme in"
    # Ordering preserved: no-bond stays the smallest, triple the largest.
    assert float(soft[0]) == min(soft.tolist())
    assert float(soft[3]) == max(soft.tolist())


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
