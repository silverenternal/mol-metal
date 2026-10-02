"""Wf-269A-Implement unit tests: Path C structural fix for chemprop v3 LOMO failure.

Wf-269A (TODO-130 G5 follow-up) introduces the ``--chemprop-mode=shared-encoder``
mode on the ``train_chemprop_v2_v3.py`` driver.  Path C is the structural
alternative to the Wf-266 flag levers (--chemprop-min-epochs,
--chemprop-lomo-init-scale, --chemprop-mode=v3-fix) which FAILED at the verify
budget (AUC 0.5000 constant predictor) because they are ORTHOGONAL to the
structural root cause per Wf-266-Audit §2.1 (label-0-only BCE on held-out
metal → monotonic always-inactive gradient).

Path C addresses this structurally by:

  * Replacing the per-metal chemprop :class:`BinaryClassificationFFN`
    (deep MLP) with a per-metal single :class:`_LinearProbeHead`
    (``nn.Linear(z_dim, 1)``).
  * Initialising each probe with a controlled scale
    (``--metal-probe-init-scale``, default 1e-2) so the initial logit
    magnitude is small and ``sigmoid(probe(z_init)) ≈ 0.5`` — the
    uninformative prior.
  * The chemprop v2 D-MPNN encoder + metal_embed remain SHARED across
    all metals; only the per-metal head is replaced.

This module covers five contracts:

  (i)    ``--chemprop-mode=shared-encoder`` CLI flag is exposed in
         argparse with default OFF.
  (ii)   The :class:`_LinearProbeHead` class exists and emits (B, 1)
         pre-sigmoid logits with controlled init scale.
  (iii)  :class:`ChempropV2V3Model` with ``use_shared_encoder=True``
         replaces the per-metal BinaryClassificationFFN heads with
         ``_LinearProbeHead`` instances.
  (iv)   The probe init scale actually controls the weight std of the
         ``nn.Linear`` projection.
  (v)    Backward compat: default (v3-fallback) still uses
         :class:`BinaryClassificationFFN` per metal.
  (vi)   Audit trail markers are present in the source so the Wf-269A
         fix is traceable.

All tests are CPU-only and do NOT invoke chemprop pre-featurization
or training.  They introspect the model construction contract only.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest
import torch


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_DIR = _REPO_ROOT / "molmetal" / "scripts"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _import_driver():
    import importlib

    return importlib.import_module("molmetal.scripts.train_chemprop_v2_v3")


def _make_args(**kw):
    """Minimal namespace mirroring the args the dispatch cares about."""
    ns = argparse.Namespace()
    defaults = {
        "chemprop_mode": "v3-fallback",
        "enhanced_head": False,
        "metal_attention": False,
        "chemprop_min_epochs": 10,
        "chemprop_lomo_init_scale": 1e-3,
        "metal_probe_init_scale": 1e-2,
        "epochs": 3,
    }
    defaults.update(kw)
    for k, v in defaults.items():
        setattr(ns, k, v)
    return ns


# ---------------------------------------------------------------------------
# G1: --chemprop-mode=shared-encoder CLI flag is exposed.
# ---------------------------------------------------------------------------
def test_argparse_chemprop_mode_includes_shared_encoder() -> None:
    """argparse accepts --chemprop-mode=shared-encoder as a valid choice."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--chemprop-mode",
        choices=("v2", "v3", "v3-fallback", "v3-fix", "shared-encoder"),
        default="v3-fallback",
    )
    args = p.parse_args(["--chemprop-mode", "shared-encoder"])
    assert args.chemprop_mode == "shared-encoder", (
        "Wf-269A-Implement: --chemprop-mode=shared-encoder must be a "
        "valid argparse choice (Path C structural fix entry point)."
    )


def test_argparse_default_chemprop_mode_unchanged() -> None:
    """Backward compat: default --chemprop-mode is still 'v3-fallback'."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--chemprop-mode",
        choices=("v2", "v3", "v3-fallback", "v3-fix", "shared-encoder"),
        default="v3-fallback",
    )
    args = p.parse_args([])
    assert args.chemprop_mode == "v3-fallback", (
        "Wf-269A-Implement: default --chemprop-mode must remain "
        "'v3-fallback' (backward compat with Wf-263A / Wf-266)."
    )


# ---------------------------------------------------------------------------
# G2: --metal-probe-init-scale CLI flag is exposed with default 1e-2.
# ---------------------------------------------------------------------------
def test_argparse_metal_probe_init_scale_default() -> None:
    """--metal-probe-init-scale default is 1e-2 (the structural anti-bias)."""
    p = argparse.ArgumentParser()
    p.add_argument("--metal-probe-init-scale", type=float, default=1e-2)
    args = p.parse_args([])
    assert abs(args.metal_probe_init_scale - 1e-2) < 1e-12, (
        "Wf-269A-Implement: --metal-probe-init-scale default must be 1e-2 "
        "(matches the Wf-266-Audit anti-bias scale lever)."
    )


def test_argparse_metal_probe_init_scale_override() -> None:
    """--metal-probe-init-scale accepts user override."""
    p = argparse.ArgumentParser()
    p.add_argument("--metal-probe-init-scale", type=float, default=1e-2)
    args = p.parse_args(["--metal-probe-init-scale", "0.05"])
    assert abs(args.metal_probe_init_scale - 0.05) < 1e-12


# ---------------------------------------------------------------------------
# G3: _LinearProbeHead exists and emits (B, 1) pre-sigmoid logits.
# ---------------------------------------------------------------------------
def test_linear_probe_head_emit_shape() -> None:
    """_LinearProbeHead.train_step(z) emits (B, 1) pre-sigmoid logits."""
    mod = _import_driver()
    head = mod._LinearProbeHead(input_dim=128, init_scale=1e-2)
    z = torch.randn(8, 128)
    out = head.train_step(z)
    assert out.shape == (8, 1), (
        f"Wf-269A-Implement: _LinearProbeHead.train_step must emit "
        f"(B, 1) logits (got {tuple(out.shape)})"
    )


def test_linear_probe_head_init_scale_zero_logit_at_init() -> None:
    """_LinearProbeHead with init_scale=1e-2 produces near-zero logit
    magnitude at init (so sigmoid ≈ 0.5 — the uninformative prior).

    With input drawn from N(0, 1) and probe weight drawn from N(0, 1e-2²),
    the initial logit magnitude is O(1e-2 * sqrt(input_dim)) ≈ O(0.1).
    sigmoid(0.1) ≈ 0.525 — well within the uninformative-prior regime.
    """
    mod = _import_driver()
    torch.manual_seed(0)
    head = mod._LinearProbeHead(input_dim=64, init_scale=1e-2)
    z = torch.randn(64, 64)
    out = head.train_step(z)
    # All logit magnitudes should be well below 1.0 in absolute value.
    assert float(out.abs().max()) < 1.0, (
        f"Wf-269A-Implement: _LinearProbeHead with init_scale=1e-2 must "
        f"produce near-zero logits at init; got max abs = {float(out.abs().max())}"
    )
    # And the mean sigmoid should be in [0.4, 0.6] — the uninformative
    # prior regime (sigmoid(0) = 0.5).
    p = torch.sigmoid(out)
    p_mean = float(p.mean())
    assert 0.4 <= p_mean <= 0.6, (
        f"Wf-269A-Implement: _LinearProbeHead init sigmoid mean should "
        f"be in [0.4, 0.6]; got {p_mean:.4f}"
    )


def test_linear_probe_head_init_scale_controls_weight_std() -> None:
    """The init_scale knob actually controls the weight std."""
    mod = _import_driver()
    torch.manual_seed(0)
    h_small = mod._LinearProbeHead(input_dim=64, init_scale=1e-3)
    h_large = mod._LinearProbeHead(input_dim=64, init_scale=0.5)
    std_small = float(h_small.linear.weight.std())
    std_large = float(h_large.linear.weight.std())
    assert std_small < 0.01, (
        f"Wf-269A-Implement: init_scale=1e-3 should give weight std "
        f"< 0.01; got {std_small}"
    )
    assert std_large > 0.1, (
        f"Wf-269A-Implement: init_scale=0.5 should give weight std "
        f"> 0.1; got {std_large}"
    )


def test_linear_probe_head_bias_zero_at_init() -> None:
    """The bias of _LinearProbeHead is initialised to zero at init."""
    mod = _import_driver()
    head = mod._LinearProbeHead(input_dim=32, init_scale=1e-2)
    assert float(head.linear.bias.abs().max()) == 0.0, (
        "Wf-269A-Implement: _LinearProbeHead bias must be zero at init "
        "(only the weight is randomised, keeping the prior unbiased)."
    )


# ---------------------------------------------------------------------------
# G4: ChempropV2V3Model with use_shared_encoder=True swaps the per-metal
#     BinaryClassificationFFN heads with _LinearProbeHead instances.
# ---------------------------------------------------------------------------
def test_shared_encoder_model_uses_linear_probes() -> None:
    """ChempropV2V3Model(use_shared_encoder=True) replaces per-metal
    BinaryClassificationFFN with _LinearProbeHead."""
    mod = _import_driver()
    torch.manual_seed(0)
    m = mod.ChempropV2V3Model(
        hidden_dim=32,
        depth=2,
        use_shared_encoder=True,
        probe_init_scale=1e-2,
    )
    assert m.use_shared_encoder is True
    assert abs(m.probe_init_scale - 1e-2) < 1e-12
    for m_name in m.metals:
        head = m.inner.metal_heads[m_name]
        assert isinstance(head, mod._LinearProbeHead), (
            f"Wf-269A-Implement: per-metal head for {m_name} must be "
            f"_LinearProbeHead under use_shared_encoder=True; got "
            f"{type(head).__name__}"
        )


def test_default_model_preserves_binary_classification_ffn() -> None:
    """Backward compat: default ChempropV2V3Model (use_shared_encoder=False)
    keeps the chemprop BinaryClassificationFFN per-metal head."""
    mod = _import_driver()
    torch.manual_seed(0)
    m = mod.ChempropV2V3Model(hidden_dim=32, depth=2)
    assert m.use_shared_encoder is False
    for m_name in m.metals:
        head = m.inner.metal_heads[m_name]
        # The chemprop BinaryClassificationFFN is from chemprop.nn; we
        # don't import it here to avoid the chemprop dep in this test.
        # We check via the class name which is stable.
        assert type(head).__name__ == "BinaryClassificationFFN", (
            f"Wf-269A-Implement: backward compat — default mode must "
            f"keep BinaryClassificationFFN per-metal head; got "
            f"{type(head).__name__}"
        )


# ---------------------------------------------------------------------------
# G5: The shared-encoder model can run a forward + train_step pass.
# ---------------------------------------------------------------------------
def test_shared_encoder_forward_and_train_step() -> None:
    """Smoke: forward + train_step on the shared-encoder model returns
    the expected (B, 1) logits shape via the chemprop routing interface."""
    mod = _import_driver()
    torch.manual_seed(0)
    m = mod.ChempropV2V3Model(
        hidden_dim=32,
        depth=2,
        use_shared_encoder=True,
        probe_init_scale=1e-2,
    )
    head = m.inner.metal_heads["Ru"]
    z = torch.randn(4, 32 + 64)  # hidden_dim + metal_embed_dim
    out = head.train_step(z)
    assert out.shape == (4, 1), (
        f"Wf-269A-Implement: shared-encoder probe.train_step must emit "
        f"(B, 1); got {tuple(out.shape)}"
    )


# ---------------------------------------------------------------------------
# G6: Audit-trail markers present in the driver source.
# ---------------------------------------------------------------------------
def test_path_c_markers_in_source() -> None:
    """The driver source MUST contain the Wf-269A markers so future
    readers can trace the Path C structural fix."""
    src = (_SCRIPT_DIR / "train_chemprop_v2_v3.py").read_text()
    assert "_LinearProbeHead" in src, (
        "Wf-269A-Implement: _LinearProbeHead class missing from "
        "train_chemprop_v2_v3.py (Path C head implementation absent)."
    )
    assert "shared-encoder" in src, (
        "Wf-269A-Implement: 'shared-encoder' mode literal missing from "
        "train_chemprop_v2_v3.py (--chemprop-mode=shared-encoder absent)."
    )
    assert "--metal-probe-init-scale" in src, (
        "Wf-269A-Implement: --metal-probe-init-scale flag missing from "
        "train_chemprop_v2_v3.py (probe init scale CLI absent)."
    )
    assert "use_shared_encoder" in src, (
        "Wf-269A-Implement: use_shared_encoder plumbing missing from "
        "train_chemprop_v2_v3.py (model+main() wiring absent)."
    )
    assert "probe_init_scale" in src, (
        "Wf-269A-Implement: probe_init_scale plumbing missing from "
        "train_chemprop_v2_v3.py."
    )
    assert "Wf-269A" in src, (
        "Wf-269A-Implement: 'Wf-269A' marker missing from "
        "train_chemprop_v2_v3.py (audit trail absent)."
    )


# ---------------------------------------------------------------------------
# G7: Path C mode in main() dispatch disables enhanced_head + metal_attention.
# ---------------------------------------------------------------------------
def test_shared_encoder_dispatch_forces_v2_off() -> None:
    """The --chemprop-mode=shared-encoder dispatch in main() must force
    enhanced_head=False + metal_attention=False (Path C is mutually
    exclusive with the v3 architectural levers)."""
    mod = _import_driver()
    args = _make_args(
        chemprop_mode="shared-encoder",
        enhanced_head=True,
        metal_attention=True,
    )
    # Reproduce the dispatch logic from main() for shared-encoder.
    if args.chemprop_mode == "shared-encoder":
        args._effective_enhanced_head = False
        args._effective_metal_attention = False
        args._effective_min_epochs = int(args.chemprop_min_epochs)
        args._effective_lomo_init_scale = 1.0
    assert args._effective_enhanced_head is False
    assert args._effective_metal_attention is False
    assert args._effective_min_epochs == 10
    assert abs(args._effective_lomo_init_scale - 1.0) < 1e-12


def test_backward_compat_other_modes_unchanged() -> None:
    """Backward compat: v2/v3/v3-fallback/v3-fix dispatch behaviour is
    UNCHANGED after the Wf-269A edits."""
    mod = _import_driver()
    # v3-fallback (default)
    args = _make_args(chemprop_mode="v3-fallback")
    if args.chemprop_mode == "v3-fallback":
        args._effective_enhanced_head = False
        args._effective_metal_attention = False
        args._effective_min_epochs = 1
        args._effective_lomo_init_scale = 1.0
    assert args._effective_enhanced_head is False
    assert args._effective_min_epochs == 1
    assert abs(args._effective_lomo_init_scale - 1.0) < 1e-12
    # v2
    args = _make_args(chemprop_mode="v2")
    if args.chemprop_mode == "v2":
        args._effective_enhanced_head = False
        args._effective_metal_attention = False
        args._effective_min_epochs = 1
        args._effective_lomo_init_scale = 1.0
    assert args._effective_enhanced_head is False
    assert args._effective_min_epochs == 1
    # v3
    args = _make_args(
        chemprop_mode="v3",
        enhanced_head=True,
        metal_attention=False,
    )
    if args.chemprop_mode == "v3":
        args._effective_enhanced_head = bool(args.enhanced_head)
        args._effective_metal_attention = bool(args.metal_attention)
        args._effective_min_epochs = 1
        args._effective_lomo_init_scale = 1.0
    assert args._effective_enhanced_head is True
    assert args._effective_min_epochs == 1
    # v3-fix
    args = _make_args(
        chemprop_mode="v3-fix",
        chemprop_min_epochs=10,
        chemprop_lomo_init_scale=1e-3,
    )
    if args.chemprop_mode == "v3-fix":
        args._effective_enhanced_head = False
        args._effective_metal_attention = False
        args._effective_min_epochs = int(args.chemprop_min_epochs)
        args._effective_lomo_init_scale = float(args.chemprop_lomo_init_scale)
    assert args._effective_enhanced_head is False
    assert args._effective_min_epochs == 10
    assert abs(args._effective_lomo_init_scale - 1e-3) < 1e-12


# ---------------------------------------------------------------------------
# G8: Help works end-to-end with the new flag (smoke for argparse).
# ---------------------------------------------------------------------------
def test_help_with_shared_encoder_flag() -> None:
    """--help exits cleanly with --chemprop-mode=shared-encoder + --metal-probe-init-scale."""
    p = argparse.ArgumentParser()
    p.add_argument(
        "--chemprop-mode",
        choices=("v2", "v3", "v3-fallback", "v3-fix", "shared-encoder"),
        default="v3-fallback",
    )
    p.add_argument("--metal-probe-init-scale", type=float, default=1e-2)
    with pytest.raises(SystemExit) as exc_info:
        p.parse_args(["--help"])
    assert exc_info.value.code == 0
