"""PA3 regression: the chemprop cross-metal run must use the paper's cohort.

BLK-PA3's whole point is that the chemprop D-MPNN is scored on the SAME cohort
and the SAME ligand-disjoint LOMO split as the paper's Morgan-FP + XGBoost
probe (``metal_activity_transfer_probe.py``), so the comparison isolates the
scorer rather than the cohort.  That guarantee is easy to lose silently: the
obvious-looking loader (``molmetal.data.metal_cytotox.load_metal_cytotox``) is
the *full* 26 801-row release, whose Ru slice is 19 135 rows — nothing like the
probe's 1 255.

These tests pin the cohort to the probe's frozen fold sizes as recorded in
``metrics/by_metric/metal_activity_transfer.json::cross_metal_lomo_dedup``, and
pin the model defaults to chemprop v2 production (d_h=300 / depth=3) rather
than the hidden=64/depth=2 smoke size the older LOMO scripts shipped.

CPU-only.  SKIPs when the MetalCytoToxDB CSV is not on this host.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent

#: Per-metal deduplicated ligand counts + LOMO fold sizes, frozen from the
#: probe's own artifact (metrics/by_metric/metal_activity_transfer.json).
#: (n_unique, n_train_unique, n_test_unique, n_train_rows_dropped_as_leaked)
FROZEN = {
    "Ru": (1255, 678, 1255, 17),
    "Ir": (534, 1391, 534, 25),
    "Rh": (51, 1874, 51, 25),
    "Os": (36, 1897, 36, 17),
    "Re": (74, 1876, 74, 0),
}


def _probe_artifact_frozen():
    """Read the same numbers off the probe artifact so the test can't drift."""
    p = REPO / "metrics" / "by_metric" / "metal_activity_transfer.json"
    if not p.is_file():
        return None
    d = json.loads(p.read_text())
    out = {}
    for fold in d.get("cross_metal_lomo_dedup", {}).get("per_fold", []):
        m = fold["metal"]
        out[m] = (
            fold["n_test_unique"],  # a metal's own dedup count == its test size
            fold["n_train_unique"],
            fold["n_test_unique"],
            fold["n_train_rows_dropped_as_leaked"],
        )
    return out or None


@pytest.fixture(scope="module")
def probe_cohort():
    from molmetal.scripts.train_chemprop_lomo_paper_cohort import (
        _load_env_default,
        build_probe_cohort,
    )

    _load_env_default()
    import os

    data_dir = os.environ.get("MOLMETAL_DATA_DIR")
    if not data_dir or not (Path(data_dir) / "MetalCytoToxDB.csv").is_file():
        pytest.skip("MetalCytoToxDB.csv not on this host")
    return build_probe_cohort(list(FROZEN), 24.0, 0.01)


def test_frozen_table_matches_the_probe_artifact():
    """The hard-coded FROZEN table must equal the probe's own recorded values.

    If the probe artifact is regenerated with a different cohort, this fails
    before the cohort tests below can silently keep passing against stale
    numbers.
    """
    frozen = _probe_artifact_frozen()
    if frozen is None:
        pytest.skip("probe artifact missing")
    assert frozen == FROZEN


def test_cohort_matches_probe_dedup(probe_cohort):
    per_metal, dup = probe_cohort
    for metal, (n_unique, _ntr, _nte, _leak) in FROZEN.items():
        assert dup[metal]["n_unique"] == n_unique, (
            f"{metal}: deduplicated cohort is {dup[metal]['n_unique']} ligands, "
            f"the probe froze {n_unique}.  If this is a deliberate cohort change, "
            f"re-freeze FROZEN and the probe artifact together."
        )
        assert len(per_metal[metal][0]) == n_unique


def test_lomo_split_matches_probe_folds(probe_cohort):
    from molmetal.scripts.train_chemprop_lomo_paper_cohort import lomo_split

    per_metal, _ = probe_cohort
    for metal, (_nu, n_train, n_test, n_leak) in FROZEN.items():
        # lomo_split returns (tr_s, tr_y, tr_metal, te_s, te_y, te_metal, n_leaked)
        # since the metal-conditioning tensors were added (2026-10); the test
        # predated that and unpacked 5 values.
        tr_s, tr_y, tr_metal, te_s, te_y, te_metal, n_leaked = lomo_split(
            per_metal, metal
        )
        assert len(te_s) == n_test, f"{metal}: test {len(te_s)} != {n_test}"
        assert len(tr_s) == n_train, f"{metal}: train {len(tr_s)} != {n_train}"
        assert n_leaked == n_leak, f"{metal}: leaked {n_leaked} != {n_leak}"
        # The leakage rule must actually hold, not just have the right count.
        assert not (set(tr_s) & set(te_s)), f"{metal}: ligand straddles the split"
        assert len(set(te_y.tolist())) >= 2, f"{metal}: single-class test fold"


def test_model_defaults_are_production_not_smoke():
    """PA3 is the 'official chemprop D-MPNN' claim — not the 64/2 smoke size."""
    from molmetal.scripts import train_chemprop_lomo_paper_cohort as M

    assert M.DEFAULT_HIDDEN == 300
    assert M.DEFAULT_DEPTH == 3
    assert M.DEFAULT_FFN_LAYERS == 1
    assert tuple(M.METALS) == ("Ru", "Ir", "Rh", "Os", "Re")


def test_config_roundtrip_is_typed():
    """§5.11: every config key must land on its dest with the right TYPE."""
    cfg = REPO / "configs" / "pa3_chemprop_lomo_v1.yaml"
    if not cfg.is_file():
        pytest.skip("pa3 config missing")
    from molmetal.scripts.train_chemprop_lomo_paper_cohort import parse_args

    a = parse_args(["--config", str(cfg)])
    assert isinstance(a.metals, list) and a.metals == ["Ru", "Ir", "Rh", "Os", "Re"]
    assert isinstance(a.seeds, list) and a.seeds == [42, 43, 44, 49, 123]
    assert isinstance(a.mode, str) and a.mode == "both"
    assert a.hidden == 300 and a.depth == 3
    assert a.n_splits == 5 and a.n_permutations == 1000 and a.perm_seed == 42


def _toy_sampler_inputs():
    import numpy as np

    rng = np.random.default_rng(0)
    n = 60
    y = (rng.random(n) < 0.4).astype(int)
    # 20 distinct scaffolds, 3 rows each, so scaffold coverage is measurable.
    scaffolds = [f"S{i // 3}" for i in range(n)]
    return y, scaffolds, n


def test_c3_support_samplers_are_distinct():
    """C3 regression (2026-10-13): the four support samplers must be four
    DISTINCT arms.  Before the fix ``scaffold_diverse`` and
    ``activity_scaffold`` shared one code path (both label-stratified), so the
    ablation had only three arms and the scaffold-coverage hypothesis could
    not be tested.
    """
    from molmetal.scripts.train_chemprop_lomo_paper_cohort import _draw_support

    y, scaffolds, n = _toy_sampler_inputs()
    k = 20
    arms = {
        s: set(_draw_support(y, scaffolds, k, 42, s).tolist())
        for s in ("random", "stratified", "scaffold_diverse", "activity_scaffold")
    }
    pairs = [
        ("random", "stratified"),
        ("random", "scaffold_diverse"),
        ("random", "activity_scaffold"),
        ("stratified", "scaffold_diverse"),
        ("stratified", "activity_scaffold"),
        ("scaffold_diverse", "activity_scaffold"),
    ]
    for a, b in pairs:
        assert arms[a] != arms[b], f"{a} and {b} produced the identical support set"


def test_scaffold_diverse_maximises_scaffold_coverage():
    """scaffold_diverse must cover strictly more distinct scaffolds than the
    label-stratified arms on a cohort where that is achievable."""
    from molmetal.scripts.train_chemprop_lomo_paper_cohort import _draw_support

    y, scaffolds, _ = _toy_sampler_inputs()
    k = 20

    def _coverage(idx):
        return len({scaffolds[i] for i in idx})

    cov_div = _coverage(_draw_support(y, scaffolds, k, 42, "scaffold_diverse"))
    cov_strat = _coverage(_draw_support(y, scaffolds, k, 42, "stratified"))
    assert cov_div > cov_strat, (cov_div, cov_strat)
    assert cov_div == 20, f"expected full scaffold coverage, got {cov_div}"


def test_c4_adapt_scope_freezes_the_encoder():
    """C4 regression: head-only adaptation must leave ONLY the head trainable;
    'full' must leave everything trainable."""
    import torch
    from molmetal.scripts.train_chemprop_lomo_paper_cohort import _set_adapt_scope

    class _Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Linear(4, 4)
            self.head = torch.nn.Linear(4, 1)
            self.film = torch.nn.Linear(4, 8)

    m = _Toy()
    _set_adapt_scope(m, "head")
    assert all(not p.requires_grad for p in m.encoder.parameters())
    assert all(not p.requires_grad for p in m.film.parameters())
    assert all(p.requires_grad for p in m.head.parameters())

    _set_adapt_scope(m, "head_metal")
    assert all(not p.requires_grad for p in m.encoder.parameters())
    assert all(p.requires_grad for p in m.film.parameters())

    _set_adapt_scope(m, "full")
    assert all(p.requires_grad for p in m.parameters())
