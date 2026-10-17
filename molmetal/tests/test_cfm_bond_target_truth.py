"""Guards for the CFM bond target being READ rather than INVENTED.

2026-10-04 — the trainer's `_bond_target_for_clean_cloud` synthesised its
target from this rule:

    pt_z = 78                                   # hardcoded
    is_pt_donor = (zs == pt_z) or (zd == pt_z)
    targets.append(BOND_SINGLE if is_pt_donor else BOND_NO_BOND)

It was written for the synthetic 5-atom Pt-click cloud and never updated when
the corpus became tmQM.  MEASURED on a Ru complex: 1331/1331 targets were
BOND_NO_BOND.  The head therefore learned "no bond" for every pair, decoding
emitted loose atoms, and `decode_ratio` reported 1.000 because `C.C.C.C...`
parses.  Nothing anywhere failed — a perfect score on a target with no
chemistry in it.

The failure has a distinctive signature (zero positive targets), so it is
directly testable, which is what this module does.
"""

from __future__ import annotations

import collections
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TMQM_DIR = PROJECT_ROOT / ".data" / "molmetal" / "tmQM"

_HAVE_TMQM = TMQM_DIR.is_dir()


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
        "_t_bond_target",
    )


# ---------------------------------------------------------------------------
# AtomCloud contract
# ---------------------------------------------------------------------------
def test_atomcloud_still_constructs_without_bonds():
    """63 call sites build AtomCloud with no bond information; they must work."""
    from molmetal.models.bond_head import AtomCloud

    c = AtomCloud(positions=torch.randn(4, 3), atomic_numbers=torch.tensor([6] * 4))
    assert c.bonds is None and c.bond_types is None


def test_atomcloud_rejects_bonds_without_types():
    """A pair list with no classes cannot be a training target."""
    from molmetal.models.bond_head import AtomCloud

    with pytest.raises(ValueError, match="bond_types"):
        AtomCloud(
            positions=torch.randn(4, 3),
            atomic_numbers=torch.tensor([6] * 4),
            bonds=torch.tensor([[0], [1]]),
        )


def test_atomcloud_rejects_mismatched_lengths():
    from molmetal.models.bond_head import AtomCloud

    with pytest.raises(ValueError, match="entries"):
        AtomCloud(
            positions=torch.randn(4, 3),
            atomic_numbers=torch.tensor([6] * 4),
            bonds=torch.tensor([[0, 1], [1, 2]]),
            bond_types=torch.tensor([1]),  # one class for two pairs
        )


# ---------------------------------------------------------------------------
# The target must contain chemistry
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not _HAVE_TMQM, reason="tmQM corpus not present")
def test_tmqm_clouds_carry_bond_truth():
    """The loader must attach the Wiberg BO graph, not leave it None."""
    from molmetal.data.tmqm_clouds import load_tmqm_clouds

    clouds = load_tmqm_clouds(
        str(TMQM_DIR), metals=["Ru"], coord_number=None, n_clouds=4, seed=0
    )
    assert clouds, "no clouds returned"
    with_bonds = [c for c in clouds if c.bonds is not None]
    assert with_bonds, (
        "load_tmqm_clouds produced clouds with no bond truth — the trainer "
        "would fall back to the Pt-only rule and target all BOND_NO_BOND"
    )
    for c in with_bonds:
        assert c.bond_types is not None
        assert int(c.bonds.shape[1]) == int(c.bond_types.shape[0])
        # A real complex has bonds; an all-NO_BOND target would be all zeros.
        assert int((c.bond_types != 0).sum()) > 0, (
            "bond truth is present but every class is BOND_NO_BOND"
        )


@pytest.mark.skipif(not _HAVE_TMQM, reason="tmQM corpus not present")
def test_bond_targets_are_not_all_no_bond(trainer):
    """THE regression guard: a non-Pt metal must yield positive targets.

    Under the old rule this was exactly zero for Ru — the signature of the
    bug — so the assertion is on the positive count, not on a loss value.
    """
    from molmetal.data.tmqm_clouds import load_tmqm_clouds

    clouds = load_tmqm_clouds(
        str(TMQM_DIR), metals=["Ru"], coord_number=None, n_clouds=4, seed=0
    )
    counts: collections.Counter = collections.Counter()
    for cloud in clouds:
        out = trainer._bond_target_for_clean_cloud(cloud)
        y = out[1]
        if y is None:
            continue
        counts.update(torch.as_tensor(y).flatten().tolist())

    total = sum(counts.values())
    assert total > 0, "no bond targets were produced at all"
    n_no_bond = counts.get(0, 0)
    assert n_no_bond < total, (
        f"every one of {total} bond targets is BOND_NO_BOND.  This is the "
        "2026-10-04 bug: the target is being synthesised by the Pt-only rule "
        "instead of read from the corpus, the head will learn to predict 'no "
        "bond' everywhere, and decode_ratio will report 1.000 on loose atoms."
    )
    # And it must not be a token positive either — the real corpus is ~29%.
    positives = total - n_no_bond
    assert positives / total > 0.05, (
        f"only {positives}/{total} targets are positive; the bond graph looks "
        "truncated rather than absent"
    )


def test_training_loop_carries_bonds_through_the_device_move():
    """The training loop rebuilds each cloud to move it to the device.

    That rebuild is the second half of the 2026-10-04 bug.  Fixing
    `_bond_target_for_clean_cloud` alone changed nothing, because the loop
    constructed a fresh ``AtomCloud(positions=..., atomic_numbers=...)`` and
    dropped ``bonds``/``bond_types`` before calling it — the target function
    then silently fell back to the Pt-only rule, and `bond_ce` came back as
    0.0001 against an all-zero target while the head's argmax was class 0 for
    every pair.

    The previous test cannot see this: it passes the ORIGINAL cloud, which
    still has its bonds.  Asserted on the source because the rebuild is inline
    in the training loop with no unit boundary to call.

    Source-grep update (Wf-A66 housekeeping, 2026-10-17)
    -----------------------------------------
    The original assertion was ``assert "bonds=(" in src and "bond_types=(" in src``
    (positional keyword-arg form).  When ``_to_device_cloud`` was refactored
    to use a ``kwargs`` dict + ``AtomCloud(**kwargs)`` (per the Wf-A11
    helper-routing refactor in 2026-10-13), the literal ``bonds=(`` pattern
    disappeared but the bond-carrying contract was preserved as
    ``kwargs["bonds"]`` and ``kwargs["bond_types"]``.  This test pins the
    CURRENT pattern (dict-style kwarg build) and the contract (both keys
    must be present in the same device-move helper, since stripping either
    one alone re-triggers the 2026-10-04 bug).
    """
    src = (
        PROJECT_ROOT
        / "molmetal"
        / "scripts"
        / "train_cfm_yuelbond_wasserstein.py"
    ).read_text()
    # Slice the source to the body of `_to_device_cloud` so we don't pick up
    # incidental "bonds" or "bond_types" mentions elsewhere (e.g. metric
    # names, log strings, the dataset parser).
    import re as _re
    m = _re.search(
        r"def _to_device_cloud\([\s\S]+?\n    return AtomCloud\(\*\*kwargs\)\n",
        src,
    )
    assert m is not None, (
        "_to_device_cloud helper is missing from the trainer source — "
        "device-move would not route through the bond-carrying chokepoint"
    )
    body = m.group(0)
    assert 'kwargs["bonds"]' in body, (
        "the training loop's device-move rebuild no longer carries "
        "bonds in the kwargs dict — the bond target will silently fall "
        "back to the Pt-only rule and train on an all-BOND_NO_BOND target"
    )
    assert 'kwargs["bond_types"]' in body, (
        "the training loop's device-move rebuild no longer carries "
        "bond_types in the kwargs dict — the bond head loses the per-pair "
        "class label and degenerates to BOND_NO_BOND everywhere"
    )


def test_the_outer_device_move_carries_bonds_too():
    """There are TWO rebuilds; the OUTER one is the one that matters.

    Fixing only the rebuild inside `train_one_epoch`'s batch loop changes
    nothing: `train_clouds_d` is built BEFORE that call and stripped the bonds
    first, so the inner rebuild receives clouds that already have none.  This
    cost a full retrain to discover — the probe at the training loop still read
    ``tgt_dist={0: 1759}`` after the inner fix.  `_to_device_cloud` is the
    single helper both now go through; this pins that the outer list uses it.
    """
    src = (
        PROJECT_ROOT
        / "molmetal"
        / "scripts"
        / "train_cfm_yuelbond_wasserstein.py"
    ).read_text()
    assert "_to_device_cloud" in src, (
        "the outer device-move helper is gone — bonds would be stripped from "
        "train_clouds_d/val_clouds_d before train_one_epoch ever sees them"
    )
    for var in ("train_clouds_d", "val_clouds_d"):
        assert f"{var} = [_to_device_cloud(c, device) for c in" in src, (
            f"{var} is no longer built through _to_device_cloud — check that "
            "it still carries bonds/bond_types"
        )


def test_fallback_warns_rather_than_passing_silently(trainer, capsys):
    """A cloud with no truth must announce that targets are invented."""
    from molmetal.models.bond_head import AtomCloud

    trainer._BOND_TARGET_FALLBACK_WARNED = False
    cloud = AtomCloud(
        positions=torch.randn(5, 3),
        atomic_numbers=torch.tensor([78, 7, 7, 17, 17]),
    )
    trainer._bond_target_for_clean_cloud(cloud)
    out = capsys.readouterr().out
    assert "no true bond graph" in out, (
        "the geometric fallback ran without warning — a silent fallback is how "
        "the all-NO_BOND target stayed invisible"
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
