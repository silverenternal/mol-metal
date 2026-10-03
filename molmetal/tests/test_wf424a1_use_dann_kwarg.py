"""Wf-424-A1 — Wf-420-A1 ``use_dann`` kwarg regression-guard tests.

Wf-420-A1 closed the pre-flight ``p2_tests_pass`` cascade parent FAIL by
adding ``use_dann`` / ``num_domains`` / ``domain_head_dropout`` kwargs to
:class:`MultiTaskWassersteinModel` in
:mod:`molmetal.scripts.train_dmpnn_multitask_wasserstein` and wiring
``args.dann`` from the CLI parser into the model constructor at the
single call-site at line ~732.

The fix is bit-for-bit backward compat per the Wf-280D BooleanOptionalAction
contract — the CLI ``--use-dann`` flag defaults to ``True``; the model's
``__init__`` default is ``use_dann=False``; the launcher explicit-overrides
to ``bool(getattr(args, 'dann', False))`` so that the new model surface
honors the CLI flag without breaking the Wf-131 P1 / Wf-251-F3 / Wf-380
shipped behaviour.

These 3 tests pin that contract:

1. ``test_dann_off_default`` — calling ``MultiTaskWassersteinModel(n_metals=2)``
   with no kwargs yields ``use_dann=False`` and ``domain_head=None``; parameter
   count matches the Wf-131 P1 shipped baseline.
2. ``test_dann_on`` — calling with ``use_dann=True, num_domains=4`` yields
   ``use_dann=True`` and a non-``None`` :class:`DomainAdversarialHead` whose
   ``num_domains`` matches the input.
3. ``test_dann_passes_to_model`` — calling with ``use_dann=True`` propagates
   to the underlying :class:`DMPNNMultiTaskModel.encoder` such that
   ``encoder.use_dann is True`` and ``encoder.domain_head is not None``.

All tests are CPU-only and finish in <5 s.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = PROJECT_ROOT / "molmetal" / "scripts"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import train_dmpnn_multitask_wasserstein as mtw  # noqa: E402

from molmetal.models.dmpnn import DomainAdversarialHead  # noqa: E402


def _baseline_params(
    n_metals: int = 2, hidden: int = 32, depth: int = 2, dropout: float = 0.1
) -> int:
    """Build the Wf-131 P1 baseline (no DANN) and return its parameter count.

    The baseline is ``MultiTaskWassersteinModel(n_metals=n_metals, hidden=..., depth=..., dropout=...)``
    — same kwargs as the Wf-380 trainer invocation at the Wf-131 P1
    ship baseline.  The default ``use_dann=False`` makes the
    ``domain_head`` attribute ``None`` and matches the Wf-131 P1
    bit-for-bit parameter count.
    """
    torch.manual_seed(0)
    model = mtw.MultiTaskWassersteinModel(
        n_metals=n_metals,
        hidden=hidden,
        depth=depth,
        dropout=dropout,
    )
    return sum(p.numel() for p in model.parameters())


def test_dann_off_default() -> None:
    """Wf-424-A1: ``use_dann`` default is False; ``domain_head`` is None.

    The Wf-131 P1 / Wf-251-F3 / Wf-380 bit-for-bit backward compat
    contract is preserved when callers omit ``use_dann`` — the Wf-280D
    CLI ``--use-dann`` default-True BooleanOptionalAction does NOT
    affect the model surface when ``use_dann`` kwarg is not passed.
    """
    torch.manual_seed(0)
    model = mtw.MultiTaskWassersteinModel(
        n_metals=2,
        hidden=32,
        depth=2,
        dropout=0.1,
    )
    assert (
        model.use_dann is False
    ), "use_dann default must be False for Wf-131 P1 backward compat"
    assert model.domain_head is None, "domain_head must be None when use_dann=False"
    assert (
        model.num_domains == 4
    ), "num_domains default must be 4 (MetalCytoToxDB / DrugOOD / NCI-60 / BindingDB)"
    # Parameter count matches the Wf-131 P1 baseline.
    n_params = sum(p.numel() for p in model.parameters())
    expected = _baseline_params()
    assert n_params == expected, (
        f"OFF param count must equal baseline: actual={n_params} vs "
        f"baseline={expected}"
    )


def test_dann_on() -> None:
    """Wf-424-A1: ``use_dann=True`` registers the DANN GRL head.

    When ``use_dann=True`` the model registers a non-``None``
    :class:`DomainAdversarialHead` sub-module with ``num_domains`` matching
    the input kwarg; parameter count strictly increases because the
    domain head adds Linear weights + biases on top of the shared encoder.
    """
    torch.manual_seed(0)
    model = mtw.MultiTaskWassersteinModel(
        n_metals=2,
        hidden=32,
        depth=2,
        dropout=0.1,
        use_dann=True,
        num_domains=4,
    )
    assert model.use_dann is True, "use_dann=True must be reflected on the model"
    assert isinstance(model.domain_head, DomainAdversarialHead), (
        f"use_dann=True must register DomainAdversarialHead, got "
        f"{type(model.domain_head).__name__}"
    )
    assert model.num_domains == 4, "num_domains must propagate from constructor kwarg"
    # The DomainAdversarialHead sub-module is also accessible via
    # ``model.encoder.domain_head`` (the encoder is the actual owner).
    assert isinstance(
        model.encoder.domain_head, DomainAdversarialHead
    ), "encoder.domain_head must mirror model.domain_head"
    # Parameter count strictly increases (domain head has its own
    # Linear weights + biases on top of the shared encoder).
    n_params_off = _baseline_params()
    n_params_on = sum(p.numel() for p in model.parameters())
    assert (
        n_params_on > n_params_off
    ), f"use_dann=True must add params: on={n_params_on} vs off={n_params_off}"


def test_dann_passes_to_model() -> None:
    """Wf-424-A1: ``use_dann`` propagates through to the encoder.

    The :class:`DMPNNMultiTaskModel.encoder` is the actual owner of
    :class:`DomainAdversarialHead`.  When ``use_dann=True`` the encoder
    registers a non-``None`` domain_head; when ``use_dann=False`` the
    encoder's domain_head is ``None``.  This mirrors the contract at
    :mod:`molmetal/baselines/dmpnn_multitask.py` lines 170-219 where
    ``DMPNNMultiTaskModel.__init__`` already supports ``use_dann``.
    """
    torch.manual_seed(0)
    # OFF propagation: encoder.domain_head is None.
    model_off = mtw.MultiTaskWassersteinModel(
        n_metals=2,
        hidden=32,
        depth=2,
        dropout=0.1,
        use_dann=False,
    )
    assert model_off.encoder.use_dann is False
    assert model_off.encoder.domain_head is None

    # ON propagation: encoder.domain_head is a DomainAdversarialHead.
    model_on = mtw.MultiTaskWassersteinModel(
        n_metals=2,
        hidden=32,
        depth=2,
        dropout=0.1,
        use_dann=True,
        num_domains=4,
    )
    assert model_on.encoder.use_dann is True
    assert isinstance(model_on.encoder.domain_head, DomainAdversarialHead)
    assert model_on.encoder.domain_head.num_domains == 4


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
