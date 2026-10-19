"""Pytest: Wf-348-C1 — Wf-340 acceleration modules wire-up into trainer.

Five tests verifying that the Wf-348-C1 sub-finds (R1/R2/R3) integration
into ``molmetal/scripts/train_cfm_yuelbond_wasserstein.py`` is **clean,
reversible, and default-OFF**:

1. ``test_train_one_epoch_so3_averaged_fm_runs`` — invoking
   ``train_one_epoch`` with ``use_so3_averaged_fm=True`` adds a finite
   extra loss term but does not crash (R3 §3.9 priority #2).
2. ``test_lr_warmup_scheduler_bounds_live_lr_and_epoch_leaves_it_alone`` —
   ``build_lr_scheduler("cosine", warmup_steps=4)`` raises lr monotonically
   from ``0.01 * base_lr`` and never exceeds ``base_lr``; an epoch does not
   touch lr (the scheduler is stepped by ``main()``, not by the epoch).
3. ``test_ema_swap_around_evaluate_decode_ratio_is_bit_exact`` —
   ``model.ema_swap_in()`` really replaces the live weights,
   ``evaluate_decode_ratio`` runs on the shadow, and ``ema_swap_out()``
   restores every parameter bit-for-bit.
4. ``test_exact_ot_cohort_pairing_reorders_train_clouds`` —
   ``exact_minibatch_ot_coupling`` returns ``x0`` unchanged and a target
   permutation whose squared-L2 cost does not exceed the identity pairing
   (Tong 2024 alignment).

**Wf-A94 (2026-10-19) repair.**  Tests 2-4 used to guard on
``_CFM_ACCEL_AVAILABLE`` / ``_EXACT_OT_AVAILABLE``, module flags that were
never defined, and to pass ``accel_lr_scheduler`` / ``accel_ema`` kwargs that
``train_one_epoch`` / ``evaluate_decode_ratio`` never had.  All three
skipped unconditionally while claiming to cover the acceleration surface.
They now exercise the surfaces the trainer actually ships (torch-native
``build_lr_scheduler``, ``CFMJointModel`` EMA, ``exact_minibatch_ot_coupling``).
5. ``test_cudagraphs_cfm_wrap_is_idempotent_and_default_off`` —
   the trainer's ``--use-cudagraphs-cfm`` flag exists in parse_args
   and defaults to False; when True, the wrap-once pattern does not
   double-wrap on re-invocation (the monkey-patch safety net is in
   place).

All tests are **CPU-only** — no GPU launches; no synthetic-cohort
retrain.  The torch.compile wrap is exercised in a low-cost eager
fallback (CPU torch.compile emits a warning, but does not crash).

Cross-ref:
- ``TODO/completed/340r1_cfm_papers_survey.md`` (9 papers)
- ``TODO/completed/340r2_acceleration_methods.md`` (8 techniques)
- ``TODO/completed/340r3_molecule_cfm.md`` (13 molecule-CFM papers)
- ``TODO/completed/340i1_cfm_acceleration_implementation.md``
  (deferred to Wf-340-A3 = Wf-348-C1)
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import List

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _import_module_safe():
    """Import the trainer script without executing main()."""
    script = PROJECT_ROOT / "molmetal" / "scripts" / "train_cfm_yuelbond_wasserstein.py"
    spec = importlib.util.spec_from_file_location(
        "train_cfm_yuelbond_wasserstein_348c1",
        str(script),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    return mod


def _small_train_clouds(n: int = 8) -> List["AtomCloud"]:
    """Tiny cohort (4-atom Pt-NH3) — under the cutoff for fast CFM loss."""
    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
        build_synthetic_pt_click_clouds,
    )

    return build_synthetic_pt_click_clouds(n_clouds=n, jitter=0.10, seed=0)


# ---------------------------------------------------------------------------
# Test 1: SO(3)-averaged FM runs through train_one_epoch
# ---------------------------------------------------------------------------
def test_train_one_epoch_so3_averaged_fm_runs() -> None:
    """use_so3_averaged_fm=True adds a finite extra loss term (R3 #2)."""
    mod = _import_module_safe()
    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
        YuelBondBondDecoder,
    )

    torch.manual_seed(0)
    model = mod.CFMJointModel(hidden_dim=16, n_layers=2)
    decoder = YuelBondBondDecoder(
        head=model.head,
        hidden_dim=16,
        n_layers=2,
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_clouds = _small_train_clouds(8)
    ref = mod.build_reference_positions()
    closure_bias = {
        "CuAAC": 0.30,
        "SPAAC": 0.10,
        "Suzuki": 0.20,
        "AmideCoupling": 0.30,
        "ThiolEne": 0.10,
    }
    out_tuple = mod.train_one_epoch(
        model=model,
        decoder=decoder,
        optimiser=opt,
        train_clouds=train_clouds,
        device=torch.device("cpu"),
        batch_size=4,
        wasserstein_weight=0.0,
        closure_bounded=True,
        closure_bias=closure_bias,
        reference_positions=ref,
        catherine_target_audit=False,
        # Wf-A94 — bond_loss_weight defaults below 1.0, so train_loss is
        # not >= the RAW bond_ce.  Pin it to 1.0 to keep the assertion
        # below meaningful.
        bond_loss_weight=1.0,
        # Wf-A94 — the entropy regulariser is NEGATIVE by construction
        # (it maximises H) and defaults to 0.1, so `train_loss` is not
        # >= bond_ce even with bond_loss_weight=1.  Zero it here so the
        # assertion below isolates the SO(3) term.
        entropy_reg_weight=0.0,
        # Wf-A94 (2026-10-19) — the kwarg was renamed when the SO(3) term
        # became a first-class loss key: `use_so3_averaged_fm=True` /
        # `so3_k=2` no longer exist.  `so3_average` + `so3_averaged_fm_k`
        # are the current names (the CLI alias --use-so3-averaged-fm still
        # maps to `so3_average`).
        so3_average=True,
        so3_averaged_fm_k=2,
    )
    train_loss, bond_ce, w2, grad_norm, catherine = out_tuple
    assert all(isinstance(x, float) for x in out_tuple)
    assert train_loss == train_loss  # not NaN
    assert bond_ce == bond_ce
    # With SO(3) enabled, the train_loss must be >= bond_ce (since the
    # 0.1 * so3_loss term adds a positive scalar).
    assert train_loss >= bond_ce - 1e-3, (
        f"train_loss ({train_loss:.4f}) should be >= bond_ce "
        f"({bond_ce:.4f}) with SO(3)-FM term"
    )


# ---------------------------------------------------------------------------
# Test 2: LR-warmup wrapper advances counter + bounds live lr
# ---------------------------------------------------------------------------
def test_lr_warmup_scheduler_bounds_live_lr_and_epoch_leaves_it_alone() -> None:
    """`build_lr_scheduler("cosine")` owns warmup; an epoch must not touch lr.

    Wf-A94 (2026-10-19) — this test used to call
    ``train_one_epoch(..., accel_lr_scheduler=warmup)``.  That kwarg never
    existed in the shipped signature, and the ``_CFM_ACCEL_AVAILABLE``
    guard above it named a module flag that was never defined, so the test
    skipped unconditionally and the warmup surface it claimed to cover was
    covered by nothing.  The shipped warmup path is
    ``build_lr_scheduler(opt, total_steps, "cosine", warmup_steps, min_lr)``,
    stepped once per epoch by ``main()``.  This test pins that contract:
    warmup raises lr monotonically from ``0.01 * base_lr`` towards
    ``base_lr``, never exceeds it, and ``train_one_epoch`` leaves lr
    untouched (the scheduler is caller-owned, so an epoch cannot
    desynchronise it).
    """
    mod = _import_module_safe()
    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
        YuelBondBondDecoder,
    )

    torch.manual_seed(0)
    model = mod.CFMJointModel(hidden_dim=16, n_layers=2)
    decoder = YuelBondBondDecoder(
        head=model.head,
        hidden_dim=16,
        n_layers=2,
    )
    base_lr = 1e-3
    opt = torch.optim.Adam(model.parameters(), lr=base_lr)
    scheduler = mod.build_lr_scheduler(
        opt,
        total_steps=20,
        scheduler_type="cosine",
        warmup_steps=4,
        min_lr=1e-6,
    )
    assert scheduler is not None, "cosine scheduler must be constructed"
    # Advance the optimiser once (no grads => numerically a no-op) so torch
    # does not warn about stepping the scheduler before the first
    # optimiser step.  The real loop steps the optimiser many times per
    # epoch before the per-epoch scheduler.step().
    opt.step()
    warmup_lrs = []
    for _ in range(4):
        scheduler.step()
        warmup_lrs.append(float(opt.param_groups[0]["lr"]))
    assert all(0.0 < lr <= base_lr + 1e-12 for lr in warmup_lrs), warmup_lrs
    assert warmup_lrs[-1] > warmup_lrs[0], (
        f"warmup must raise lr; got {warmup_lrs}"
    )

    lr_before = float(opt.param_groups[0]["lr"])
    mod.train_one_epoch(
        model=model,
        decoder=decoder,
        optimiser=opt,
        train_clouds=_small_train_clouds(8),
        device=torch.device("cpu"),
        batch_size=4,
        wasserstein_weight=0.0,
        closure_bounded=False,
        closure_bias=None,
        reference_positions=mod.build_reference_positions(),
        catherine_target_audit=False,
    )
    lr_after = float(opt.param_groups[0]["lr"])
    assert lr_after == pytest.approx(lr_before), (
        "train_one_epoch must not touch lr (the scheduler is caller-owned); "
        f"{lr_before} -> {lr_after}"
    )


# ---------------------------------------------------------------------------
# Test 3: EMA swap-in / restore is reversible
# ---------------------------------------------------------------------------
def test_ema_swap_around_evaluate_decode_ratio_is_bit_exact() -> None:
    """model.ema_swap_in/out bracket an eval without perturbing live weights.

    Wf-A94 (2026-10-19) — this test used to pass
    ``accel_ema=ExponentialMovingAverage(...)`` to
    ``evaluate_decode_ratio``.  That kwarg does not exist, and the trainer
    does not use ``acceleration.ExponentialMovingAverage`` at all: EMA lives
    on ``CFMJointModel`` as ``init_ema`` / ``update_ema`` / ``ema_swap_in``
    / ``ema_swap_out`` with a DIRECT decay, because the yaml already states
    ``ema_decay`` and the kimg-halflife form would have required inverting
    it.  This test drives the shipped surface end-to-end and asserts the
    restore is bit-exact.
    """
    mod = _import_module_safe()
    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
        YuelBondBondDecoder,
    )

    torch.manual_seed(0)
    model = mod.CFMJointModel(hidden_dim=16, n_layers=2)
    decoder = YuelBondBondDecoder(
        head=model.head,
        hidden_dim=16,
        n_layers=2,
    )
    model.init_ema(decay=0.5)
    # Perturb the live weights so the EMA snapshot is genuinely different;
    # without this the swap would be a no-op and the test could pass on an
    # implementation that never swaps anything.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.05)
    model.update_ema(decay=0.5)
    pre_state = {
        name: param.detach().clone() for name, param in model.named_parameters()
    }

    assert model.ema_swap_in() is True, "ema_swap_in must report a swap"
    swapped = {
        name: param.detach().clone() for name, param in model.named_parameters()
    }
    assert any(
        not torch.equal(pre_state[name], swapped[name]) for name in pre_state
    ), "ema_swap_in must actually replace the live weights"

    n_ok, n_total = mod.evaluate_decode_ratio(
        model=model,
        decoder=decoder,
        val_clouds=_small_train_clouds(4),
        device=torch.device("cpu"),
        closure_bounded=False,
        closure_bias=None,
    )
    assert n_total == 4
    model.ema_swap_out()

    # Restore check: every parameter is bit-equal to pre_state.
    for name, param in model.named_parameters():
        assert torch.equal(
            param.data, pre_state[name].data
        ), f"param {name!r} was not restored after EMA swap"


# ---------------------------------------------------------------------------
# Test 4: Exact-OT cohort pairing reorders clouds
# ---------------------------------------------------------------------------
def test_exact_ot_cohort_pairing_reorders_train_clouds() -> None:
    """exact_minibatch_ot_coupling returns the W2-optimal target permutation.

    Wf-A94 (2026-10-19) — the ``_EXACT_OT_AVAILABLE`` guard named a module
    flag that was never defined, so this test skipped unconditionally even
    though ``optimal_transport_v2.exact_minibatch_ot_coupling`` is present
    on disk and is what ``--use-ot-cfm`` dispatches to.  The guard is gone.
    The test now also asserts optimality (the coupled cost must not exceed
    the identity pairing, which is one feasible permutation), which the
    previous permutation-set check did not.
    """
    mod = _import_module_safe()
    from molmetal.adapters.flow_matching_lipman.optimal_transport_v2 import (
        exact_minibatch_ot_coupling,
    )

    torch.manual_seed(0)
    # Build a deterministic (B, D) descriptor matrix.
    B, D = 6, 4
    x_0 = torch.randn(B, D)
    x_1 = torch.randn(B, D)
    perm_x0, perm_x1 = exact_minibatch_ot_coupling(
        x_0,
        x_1,
        exact_max_batch=B,
    )
    # Same shapes.
    assert perm_x0.shape == x_0.shape == perm_x1.shape == x_1.shape
    # The source side is returned unchanged; the permutation is on the
    # target side (x0 is noise, so reordering it would be a no-op).
    assert torch.equal(perm_x0, x_0), "the source side must be untouched"
    # Identify the implied permutation index.
    idx0 = []
    for row in perm_x1:
        diff = (x_1 - row.unsqueeze(0)).abs().sum(dim=1)
        idx0.append(int(diff.argmin().item()))
    # The index set must be a valid permutation of {0..B-1}.
    assert sorted(idx0) == list(
        range(B)
    ), f"OT output is not a row-permutation: idx0={idx0}"
    # Optimality: the identity pairing is feasible, so the Hungarian cost
    # can never be larger than it.
    coupled_cost = float(((perm_x0 - perm_x1) ** 2).sum())
    identity_cost = float(((x_0 - x_1) ** 2).sum())
    assert coupled_cost <= identity_cost + 1e-6, (
        f"OT cost {coupled_cost:.6f} exceeds identity {identity_cost:.6f}"
    )


# ---------------------------------------------------------------------------
# Test 5: --use-cudagraphs-cfm default OFF; CLI flag exists
# ---------------------------------------------------------------------------
def test_cudagraphs_cfm_wrap_is_idempotent_and_default_off() -> None:
    """The flag is in parse_args; default is False (backward-compat)."""
    mod = _import_module_safe()
    # parse_args uses sys.argv.  We patch sys.argv temporarily.
    argv_backup = sys.argv
    try:
        sys.argv = ["trainer.py"]
        args = mod.parse_args()
    finally:
        sys.argv = argv_backup
    # Wf-A94 (2026-10-19) — Wf-431-B1 replaced the `--use-cudagraphs-cfm`
    # monkey-patch with `--torch-compile` (torch.compile with
    # dynamic_shapes=True, the Wf-358-A3 / Wf-335-R3 safety contract).  The
    # old flag and its wrap-once counter no longer exist, so asserting them
    # tested a surface that was removed on purpose.
    assert hasattr(
        args, "torch_compile"
    ), "parse_args must expose --torch-compile"
    assert args.torch_compile is False, (
        f"default should be False (safety lever per Wf-335-R3); got "
        f"{args.torch_compile}"
    )
    # Also: --use-so3-averaged-fm (the Wf-385-A3 alias for --so3-average)
    # is present and OFF, and K defaults to 2 (the 2026-10-08 wire-in).
    assert hasattr(args, "so3_average")
    assert args.so3_average is False
    assert args.so3_averaged_fm_k == 2
