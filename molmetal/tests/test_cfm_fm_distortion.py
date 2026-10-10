"""TRACK 4 — FlowMol3 train-time geometry distortion of the CFM interpolant.

2026-10-11.  Mechanism + citation: `apply_fm_distortion` in
`molmetal/scripts/train_cfm_yuelbond_wasserstein.py`; config
`configs/cfm_fm_distort_2026_10_11.yaml`.

WHAT IS UNDER TEST.  `gen_validity` / `gen_single_fragment_ratio` /
`decode_ratio` have no live loss term, and the review verdict on this repo is
that an auxiliary penalty is the wrong instrument — the one term built for
validity (`soft_valence_term`) is a one-sided squared hinge with zero gradient
in the feasible region, so it buys validity by pushing bond orders down.  The
published mechanism instead changes the TRAINING DISTRIBUTION: Vost et al.,
Digital Discovery 2025 (DOI 10.1039/D4DD00331D) train with distorted
molecules; FlowMol3 (arXiv:2508.12629) does the same and reports 96%
PoseBusters against 93% on its own clean training data.

    X_t  <-  X_t + 1[t >= t_distort] * Bernoulli(p_distort) * (M * eps)

HONEST FRAMING (AGENTS.md 5.2).  What is MEASURED here is the DISTRIBUTION
SHIFT: with the seed fixed, the interpolant handed to the velocity field is
bit-identical when the mechanism is off, and provably perturbed when it is on
with the right probability and the right t gate.  What is NOT measured is any
`gen_validity` uplift — that needs a GPU run of
`configs/cfm_fm_distort_2026_10_11.yaml` against
`configs/cfm_fullcorpus_2026_10_10.yaml`, and no such run is on disk.  Anyone
claiming a validity gain from this file is quoting a paper, not this repo.
"""

from __future__ import annotations

import copy
import inspect
from pathlib import Path

import pytest
import torch

from molmetal.scripts import train_cfm_yuelbond_wasserstein as mod

_REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Unit tests on the helper itself.
# ---------------------------------------------------------------------------
def _toy(n_atoms: int = 6):
    torch.manual_seed(0)
    return torch.randn(n_atoms, 3), torch.full((n_atoms,), 0.5)


def test_off_is_the_default_and_is_bit_for_bit():
    """p_distort <= 0 returns the SAME tensor and consumes no RNG."""
    x_t, t = _toy()

    # Same tensor object — not merely an equal-valued copy.
    torch.manual_seed(7)
    out = mod.apply_fm_distortion(x_t, t, [3, 3], p_distort=0.0)
    assert out is x_t, "p_distort=0 must return the input tensor unchanged"

    # No RNG consumed: the next draw matches a run that never called it.
    torch.manual_seed(7)
    ref = torch.randn(4)
    torch.manual_seed(7)
    mod.apply_fm_distortion(x_t, t, [3, 3], p_distort=0.0)
    got = torch.randn(4)
    assert torch.equal(ref, got), (
        "the OFF path consumed RNG — a later draw in the epoch would shift, "
        "which is exactly the bit-for-bit break this rule forbids"
    )

    # sigma 0 is also a no-op, not a divide-by-zero path.
    assert (
        mod.apply_fm_distortion(x_t, t, [3, 3], p_distort=1.0, sigma_distort=0.0)
        is x_t
    )


def test_defaults_are_off_in_parser_and_signature():
    """Nothing predating 2026-10-11 may start distorting by accident."""
    sig = inspect.signature(mod.train_one_epoch)
    assert sig.parameters["fm_p_distort"].default == 0.0
    assert sig.parameters["fm_t_distort"].default == 0.25
    assert sig.parameters["fm_sigma_distort"].default == 0.5

    import sys

    old = sys.argv[:]
    try:
        sys.argv = ["trainer"]
        args = mod.parse_args()
    finally:
        sys.argv = old
    assert args.fm_p_distort == 0.0
    assert args.fm_t_distort == 0.25
    assert args.fm_sigma_distort == 0.5


def test_t_gate_excludes_clouds_below_t_distort():
    """A cloud with t < t_distort must not be perturbed, even at p = 1.0."""
    torch.manual_seed(0)
    x_t = torch.zeros(8, 3)
    sizes = [4, 4]
    # cloud 0 at t = 0.1 (below the gate), cloud 1 at t = 0.9 (above it).
    t = torch.tensor([0.1] * 4 + [0.9] * 4)

    out = mod.apply_fm_distortion(
        x_t, t, sizes, p_distort=1.0, t_distort=0.25, sigma_distort=0.5
    )
    assert torch.equal(out[:4], x_t[:4]), "t below the gate was distorted"
    assert not torch.equal(out[4:], x_t[4:]), "t above the gate was NOT distorted"


def test_probability_is_per_cloud_and_calibrated():
    """Across many clouds the hit rate matches p_distort."""
    n_c, n_atoms = 400, 4
    torch.manual_seed(1)
    x_t = torch.zeros(n_c * n_atoms, 3)
    t = torch.ones(n_c * n_atoms)  # gate always open
    sizes = [n_atoms] * n_c

    out = mod.apply_fm_distortion(
        x_t, t, sizes, p_distort=0.5, t_distort=0.25, sigma_distort=1.0
    )
    changed = [not torch.equal(out[i * n_atoms], x_t[i * n_atoms]) for i in range(n_c)]
    rate = sum(changed) / n_c
    # 400 Bernoulli(0.5) draws: 4 sigma ~ 0.05.  Generous but still falsifying
    # a p that was wired up as 0, as 1, or as a per-atom draw.
    assert 0.42 < rate < 0.58, f"hit rate {rate} is not consistent with p=0.5"


def test_the_coin_is_flipped_per_molecule_not_per_atom():
    """The hit RATE must not depend on how many atoms a molecule has.

    This is the sharp discriminator between a per-cloud draw and a per-atom
    one.  With p = 0.5 and 24 atoms per cloud, a per-atom coin would leave a
    cloud untouched with probability 0.5**24 ~ 6e-8 — over 200 clouds you
    would see essentially zero untouched clouds.  A per-cloud coin leaves
    ~half of them untouched, and the untouched fraction must not move when
    the cloud grows from 4 atoms to 24.
    """
    rates = {}
    for n_atoms in (4, 24):
        n_c = 200
        torch.manual_seed(3)
        x_t = torch.zeros(n_c * n_atoms, 3)
        t = torch.ones(n_c * n_atoms)
        out = mod.apply_fm_distortion(
            x_t, t, [n_atoms] * n_c, p_distort=0.5, t_distort=0.0, sigma_distort=1.0
        )
        untouched = sum(
            torch.equal(out[i * n_atoms], x_t[i * n_atoms]) for i in range(n_c)
        )
        rates[n_atoms] = untouched / n_c
    # 200 Bernoulli(0.5) draws -> 3.5 sigma ~ 0.09.
    assert 0.35 < rates[4] < 0.65, rates
    assert 0.35 < rates[24] < 0.65, (
        f"untouched-cloud fraction moved with atom count {rates} — the coin "
        "is being flipped per ATOM, not per molecule"
    )


def test_cloud_sizes_mismatch_raises():
    x_t = torch.zeros(5, 3)
    with pytest.raises(ValueError):
        mod.apply_fm_distortion(
            x_t, torch.zeros(5), [3, 3], p_distort=1.0
        )


# ---------------------------------------------------------------------------
# Wiring: the interpolant in train_one_epoch.
# ---------------------------------------------------------------------------
_SETUP = None


def _tiny_setup():
    """Built ONCE and deep-copied per arm.

    Rebuilding the cohort and the model inside every arm lets any global RNG
    the cohort builder touches advance between arms, which shows up as the
    two arms disagreeing about x1 and therefore about x_t — a difference that
    has nothing to do with the mechanism under test.
    """
    global _SETUP
    if _SETUP is not None:
        return _SETUP
    train_clouds = mod.build_training_cohort(n_train=6, seed=0, jitter=0.10)
    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
        YuelBondBondDecoder,
    )
    import torch.optim as optim

    torch.manual_seed(0)
    model = mod.CFMJointModel(
        hidden_dim=16, n_layers=1, fm_objective=True, use_egnn_velocity=True
    )
    decoder = YuelBondBondDecoder(head=model.head, hidden_dim=16, n_layers=1)
    optimiser = optim.Adam(model.parameters(), lr=1e-4)
    ref = mod.build_reference_positions()
    _SETUP = (model, decoder, optimiser, train_clouds, ref)
    return _SETUP


def _capture_first_interpolant(monkeypatch, seed: int, **fm_kwargs):
    """Run one epoch of train_one_epoch with fm_p_distort and return x_t.

    The velocity field is the component that RECEIVES the interpolant, so
    hooking it is the honest observation point: we record what the model was
    actually shown, not a reconstruction.

    `monkeypatch` pins the epoch's shuffle.  Without it this test cannot exist:
    see the note on `_pin_epoch_shuffle`.
    """
    _pin_epoch_shuffle(monkeypatch)
    model, decoder, optimiser, clouds, ref = _tiny_setup()
    model = copy.deepcopy(model)
    decoder = copy.deepcopy(decoder)
    # train_one_epoch mutates the cloud objects in place (it attaches cached
    # bond targets), so a shared cohort makes the SECOND arm train on a
    # different x1 than the first.  Hand each arm its own copy.
    clouds = copy.deepcopy(clouds)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-4)
    seen: list = []
    original = mod.CFMJointModel.fm_velocity

    def _spy(self, x_t, t, z, *a, **kw):
        if not seen:
            seen.append(x_t.detach().clone())
        return original(self, x_t, t, z, *a, **kw)

    mod.CFMJointModel.fm_velocity = _spy
    try:
        # Seed every global generator the epoch can reach. torch.manual_seed
        # alone left `random`/`numpy` advancing between the two arms, which is
        # how the OFF arm looked nondeterministic when it is not.
        import random as _random

        import numpy as _np

        torch.manual_seed(seed)
        _random.seed(seed)
        _np.random.seed(seed)
        mod.train_one_epoch(
            model=model,
            decoder=decoder,
            optimiser=optimiser,
            train_clouds=clouds,
            device=torch.device("cpu"),
            batch_size=6,
            reference_positions=ref,
            wasserstein_weight=0.0,
            fm_objective=True,
            **fm_kwargs,
        )
    finally:
        mod.CFMJointModel.fm_velocity = original
    assert seen, "the FM path never ran — this test proves nothing"
    return _canonical(seen[0])


def _canonical(x: torch.Tensor) -> torch.Tensor:
    """Flatten to a 1-D vector so two captures compare element-for-element."""
    return x.detach().reshape(-1).clone()


def _pin_epoch_shuffle(monkeypatch) -> None:
    """Freeze the epoch's cloud ordering.

    PRE-EXISTING, NOT MINE, and it is load-bearing for any A/B on this epoch:

        molmetal/scripts/train_cfm_yuelbond_wasserstein.py:2837
            rng = np.random.default_rng(seed=int(time.time()) % (2**31 - 1))
            order = rng.permutation(n_train).tolist()

    The epoch shuffle is seeded from the WALL CLOCK, so two runs at the same
    torch seed walk the cohort in a different order.  Because `x0` is one
    `randn_like` over the CONCATENATED batch, a different order does not
    permute `x_t` — it re-pairs noise rows with clouds, so `x_t` differs
    numerically.  Measured while writing this file: with `torch.manual_seed`,
    `random.seed` and `np.random.seed` all pinned, two back-to-back OFF runs
    still disagreed, and recovering `x1 = (x_t - (1-t)x0)/t` showed x0 and t
    bit-identical while x1 was not.

    This patches `default_rng` for the duration of the run only (monkeypatch
    restores it).  The trainer line itself is reported, NOT changed: a live
    run is reading this file, and re-seeding an epoch's shuffle is a
    reproducibility change that belongs to whoever owns that run, not to a
    track that only adds an interpolant transform.
    """
    import numpy as _np

    _orig = _np.random.default_rng  # capture BEFORE patching, else we recurse
    monkeypatch.setattr(_np.random, "default_rng", lambda *a, **kw: _orig(0))


def test_wiring_off_reproduces_the_clean_interpolant(monkeypatch):
    """With the mechanism off the field sees the plain (1-t)x0 + t x1 draw."""
    off = _capture_first_interpolant(monkeypatch, 1234, fm_p_distort=0.0)
    again = _capture_first_interpolant(monkeypatch, 1234, fm_p_distort=0.0)
    assert torch.equal(off, again), "the OFF path is not deterministic"


def test_wiring_on_perturbs_the_interpolant_the_field_is_shown(monkeypatch):
    """The distortion must reach fm_velocity, not be computed and dropped."""
    clean = _capture_first_interpolant(monkeypatch, 1234, fm_p_distort=0.0)
    dirty = _capture_first_interpolant(
        monkeypatch, 1234, fm_p_distort=1.0, fm_t_distort=0.0, fm_sigma_distort=0.5
    )
    assert clean.shape == dirty.shape
    delta = (dirty - clean).abs().max().item()
    assert delta > 0.0, (
        "fm_p_distort=1.0 / t_distort=0 produced an IDENTICAL interpolant — "
        "the call is not wired into the training path"
    )


def test_wiring_on_respects_the_t_gate_end_to_end(monkeypatch):
    """t_distort above every attainable t must be equivalent to OFF."""
    # t ~ U(0,1), so a gate at 2.0 is never open.
    clean = _capture_first_interpolant(monkeypatch, 1234, fm_p_distort=0.0)
    gated = _capture_first_interpolant(
        monkeypatch, 1234, fm_p_distort=1.0, fm_t_distort=2.0, fm_sigma_distort=0.5
    )
    assert torch.equal(clean, gated), (
        "t_distort=2.0 (gate never open) still changed the interpolant — "
        "the gate is not wired"
    )


def test_the_training_call_site_exists():
    """The helper must be CALLED from the epoch, not merely defined.

    Goes red if someone deletes the call line while leaving the helper, the
    flags and the config in place — which is the silent-degradation shape
    AGENTS.md 5.11 exists to prevent.
    """
    src = Path(mod.__file__).read_text()
    assert src.count("x_t = apply_fm_distortion(") == 1, (
        "the interpolant no longer passes through apply_fm_distortion"
    )
    start = src.index("def train_one_epoch(")
    end = src.index("\ndef ", start + 10)
    body = src[start:end]
    assert body.count("apply_fm_distortion(") == 1
    assert "x_t = (1.0 - t_col) * x0 + t_col * x1" in body, (
        "the plain interpolant is gone — check what this mechanism wraps"
    )


def test_target_dx_is_not_distorted():
    """The velocity target stays x1 - x0; the model denoises FROM distortion."""
    src = Path(mod.__file__).read_text()
    assert "target_dx = x1 - x0" in src
    # The distortion must not be reassigned into target_dx anywhere.
    assert "target_dx = apply_fm_distortion" not in src
    # The call is made on x_t, and target_dx is assigned after it, unchanged.
    call = src.index("x_t = apply_fm_distortion(")
    assert src.index("target_dx = x1 - x0", call) > call


# ---------------------------------------------------------------------------
# The sampler must be untouched — this is the load-bearing safety property.
# ---------------------------------------------------------------------------
def test_sampler_never_calls_the_distortion():
    """Distorting the sampler's initial state would corrupt the PRIOR."""
    src = Path(mod.__file__).read_text()
    start = src.index("def generate_clouds_from_noise(")
    end = src.index("\ndef ", start + 10)
    body = src[start:end]
    assert "apply_fm_distortion" not in body
    assert "fm_p_distort" not in body


def test_sampler_runs_even_when_the_helper_raises(monkeypatch):
    """Belt-and-braces: booby-trap the helper and integrate the sampler."""
    from molmetal.models.bond_head import AtomCloud
    from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import get_z_table

    def _boom(*a, **kw):  # pragma: no cover - must never run
        raise AssertionError("the decode-time sampler touched the distortion")

    monkeypatch.setattr(mod, "apply_fm_distortion", _boom)

    tpl = [
        AtomCloud(
            positions=torch.randn(5, 3),
            atomic_numbers=torch.tensor([78, 7, 7, 17, 17]),
        )
        for _ in range(2)
    ]
    model = mod.CFMJointModel(
        hidden_dim=32,
        n_layers=2,
        cutoff=3.5,
        z_table=get_z_table("pt"),
        use_egnn_velocity=True,
        max_atomic_number=100,
    )
    clouds = mod.generate_clouds_from_noise(
        model, tpl, torch.device("cpu"), method="euler", n_steps=3, seed=0
    )
    assert len(clouds) == 2
    assert clouds[0].positions.shape == (5, 3)


# ---------------------------------------------------------------------------
# Config round trip (AGENTS.md 5.11).
# ---------------------------------------------------------------------------
def test_new_config_round_trips_by_type():
    """0.7 / 0.25 / 0.5 must reach argparse as FLOATS, and resolve at all."""
    import sys

    cfg = "configs/cfm_fm_distort_2026_10_11.yaml"
    old = sys.argv[:]
    try:
        sys.argv = ["trainer", "--config", cfg]
        args = mod.parse_args()
    finally:
        sys.argv = old
    assert isinstance(args.fm_p_distort, float) and args.fm_p_distort == 0.7
    assert isinstance(args.fm_t_distort, float) and args.fm_t_distort == 0.25
    assert isinstance(args.fm_sigma_distort, float) and args.fm_sigma_distort == 0.5


def test_the_flowmol3_operating_point_is_in_the_config_not_the_default():
    """The p=1.0 trap is documented where someone will edit it."""
    text = (_REPO / "configs" / "cfm_fm_distort_2026_10_11.yaml").read_text()
    assert "fm_p_distort: 0.7" in text
    assert "COLLAPSES functional-group diversity" in text
    assert "DO NOT RAISE fm_p_distort TO 1.0" in text


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
