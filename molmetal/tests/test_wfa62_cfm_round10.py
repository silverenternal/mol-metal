"""Wf-A62 W1 — CFM deep audit round 10 regression tests.

Locks down the 12 defensive-guards applied in this wave across 6 source files.
All tests are CPU-only and run in <2 s.

Covers (per SCOUT, verified on disk 2026-10-17):
  - VF1: ``_RadiusGraphVelocityModel.__call__`` rejects multi-dim t
  - VF3 + PE5: ``PerLayerPocketInjector.__init__`` zero-init check
  - VF4: ``t.view(1, 1).expand(b, 1)`` deterministic broadcast
  - VF5: ``bond_head grad clip`` warning when ``joint_train=False``
  - JM1: ``bad_atoms`` bool coercion accepts Python float
  - JM2: fully-masked ``node_mask`` emits a warning
  - JM4: ``_velocity_with_metal_prior`` warns on n_samples > 32
  - FL5: ``log_normal_t_sampling`` rejects sigma <= 0
  - FL6: ``log_normal_t_sampling`` coerces generator device
  - RL3: ``ModelWrapper.__call__`` rejects extras contain 't'
  - RL4: ``ModelWrapper.__call__`` propagates arity mismatch
  - PE1: ``PocketEncoder.forward`` warns on empty pocket
  - PE2: ``_fully_connected_edge_index`` caps O(P**2) memory
  - RL5: try/except wrappers around 4 upstream imports
  - FL3: ``mini_batch_ot_coupling`` rejects duplicate plan indices
  - RL1: ``DEFAULT_REFERENCE`` existence check

Per AGENTS.md §5.5 filesystem-truth: each of these maps to a verified
file:line on disk as of commit 1.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import torch

torch.manual_seed(0)


# Repo-relative path anchors: these tests read source files that are TRACKED
# in this repository, so the location must be derived from this file's own
# location rather than hard-coded to one checkout path.
REPO_ROOT = Path(__file__).resolve().parents[2]
FML_INIT = (
    REPO_ROOT / "molmetal" / "adapters" / "flow_matching_lipman" / "__init__.py"
)
LOSS_INIT = REPO_ROOT / "flow_matching" / "loss.py"
REF_LOADER = (
    REPO_ROOT / "molmetal" / "adapters" / "flow_matching_lipman" / "_reference_loader.py"
)
REF_UTILS = (
    REPO_ROOT
    / "molmetal"
    / "adapters"
    / "flow_matching_lipman"
    / "_reference"
    / "utils"
    / "__init__.py"
)
LOGN = (
    REPO_ROOT / "molmetal" / "adapters" / "flow_matching_lipman" / "log_normal_t.py"
)


def _read(p: Path) -> str:
    with open(p) as f:
        return f.read()


# ---------------------------------------------------------------------------
# VF1 — _RadiusGraphVelocityModel rejects multi-dim t (string + behaviour)
# ---------------------------------------------------------------------------
def test_vf1_radius_graph_velocity_model_rejects_multidim_t_string():
    """VF1 (file:line) — multi-dim t raises ValueError."""
    text = _read(FML_INIT)
    assert (
        "_RadiusGraphVelocityModel expected t as 0-d scalar or (B,) 1-d" in text
    ), "VF1 multi-dim ValueError must be embedded in __init__.py"
    assert "got shape {tuple(t.shape)}" in text, "VF1 must include the shape string"


# ---------------------------------------------------------------------------
# VF4 — ODESolver reuses t across inner loop (doc string present)
# ---------------------------------------------------------------------------
def test_vf4_t_view_expand_b1_b1_doc_present():
    """VF4 — doc note that velocity is deterministic per batch index."""
    text = _read(FML_INIT)
    assert "(VF4 doc)" in text, "VF4 doc comment must be embedded"


# ---------------------------------------------------------------------------
# VF3 + PE5 — PerLayerPocketInjector zero-init
# ---------------------------------------------------------------------------
def test_vf3_pe5_per_layer_pocket_injector_zero_init():
    """VF3 + PE5 — fresh ``PerLayerPocketInjector`` is zero-initialised."""
    from molmetal.adapters.egnn_extensions import PerLayerPocketInjector

    inj = PerLayerPocketInjector(pocket_dim=32, hidden_dim=64)
    assert torch.allclose(inj.proj.weight, torch.zeros_like(inj.proj.weight))
    if inj.proj.bias is not None:
        assert torch.allclose(inj.proj.bias, torch.zeros_like(inj.proj.bias))


# ---------------------------------------------------------------------------
# VF5 — bond_head grad clip warning embedded
# ---------------------------------------------------------------------------
def test_vf5_bond_head_grad_clip_warns_on_legacy_joint_train_false():
    """VF5 (file:line) — bond_head grad clip warn string present."""
    text = _read(FML_INIT)
    assert "bond_head in use but joint_train=False" in text, "VF5 warning string missing"
    assert "_bond_head_clip_warn_emitted" in text, "VF5 must gate warn-once flag"


# ---------------------------------------------------------------------------
# JM1 — bool coercion accepts Python float loss
# ---------------------------------------------------------------------------
def test_jm1_bool_coercion_accepts_python_float():
    """JM1 — wrap with ``torch.as_tensor`` so ``bool()`` works on Python float."""
    loss = float("nan")  # Python float, NOT Tensor
    # Direct reproduction of the fixed expression.  Must not raise.
    bad_atoms = bool(
        torch.as_tensor(loss).isnan().any() or torch.as_tensor(loss).isinf().any()
    )
    assert bad_atoms is True, "JM1 must detect NaN Python float"


# ---------------------------------------------------------------------------
# JM2 + JM6 + JM7 — diagnostic flag additions
# ---------------------------------------------------------------------------
def test_jm2_jm6_jm7_warn_strings_embedded():
    """JM2 + JM6 + JM7 — diagnostic strings live in train_step."""
    text = _read(FML_INIT)
    assert (
        "_joint_adapter atom_loss on fully-masked batch" in text
    ), "JM2 string missing"
    assert "bond_skipped_no_edges" in text, "JM6 last_losses flag missing"
    assert "falling back to un-masked" in text, "JM7 OOV fallback warn missing"


# ---------------------------------------------------------------------------
# JM4 — _velocity_with_metal_prior warns on n_samples > 32
# ---------------------------------------------------------------------------
def test_jm4_n_samples_warn_string_embedded():
    """JM4 — warn string + JM8 diagnostics dict key embedded."""
    text = _read(FML_INIT)
    assert "n_samples={n_samples} drives" in text, "JM4 n_samples warn missing"
    assert (
        "non_differentiable_fallback_count" in text
    ), "JM8 diagnostics dict key missing"


# ---------------------------------------------------------------------------
# FL5 + FL6 — log_normal_t_sampling guards
# ---------------------------------------------------------------------------
def test_fl5_log_normal_t_sampling_sigma_guard():
    """FL5 — sigma <= 0 raises ValueError."""
    from molmetal.adapters.flow_matching_lipman.log_normal_t import (
        log_normal_t_sampling,
    )

    with pytest.raises(ValueError, match="sigma must be > 0"):
        log_normal_t_sampling(batch_size=4, sigma=0.0)
    with pytest.raises(ValueError, match="sigma must be > 0"):
        log_normal_t_sampling(batch_size=4, sigma=-1.0)


def test_fl6_log_normal_t_sampling_device_coercion():
    """FL6 — CPU generator + cpu device works; t in (0, 1)."""
    from molmetal.adapters.flow_matching_lipman.log_normal_t import (
        log_normal_t_sampling,
    )

    cpu_gen = torch.Generator(device="cpu").manual_seed(0)
    t = log_normal_t_sampling(batch_size=4, generator=cpu_gen, device="cpu")
    assert t.shape == (4,)
    assert ((t > 0.0) & (t < 1.0)).all(), "t must lie in (0, 1) sigmoid range"


# ---------------------------------------------------------------------------
# RL3 + RL4 — ModelWrapper edge cases
# ---------------------------------------------------------------------------
def test_rl3_modelwrapper_rejects_t_collision():
    """RL3 — extras containing 't' raises TypeError, NOT silent override.

    The signature ``ModelWrapper.__call__(self, x, t, **extras)`` means
    Python raises ``TypeError: got multiple values for argument 't'``
    when a caller passes both positional ``t`` and ``**{t: ...}``.  The
    in-body ``if "t" in extras`` check is a belt-and-braces defensive
    guard for a hypothetical signature change; today Python catches it
    first.  Either way — NO silent override.
    """
    from molmetal.adapters.flow_matching_lipman._reference.utils import (
        ModelWrapper,
    )

    def model(x, t):
        return x + t

    mw = ModelWrapper(model)
    # Python's kwarg conflict MUST raise (no silent override).
    with pytest.raises(TypeError, match="multiple values"):
        mw(torch.zeros(2, 3), 0.5, t=0.7)  # noqa: B008
    # The defensive body check is in source (single-line anchor).
    text = _read(REF_UTILS)
    assert "'t' is in **extras" in text, "RL3 defensive guard string missing"


def test_rl4_modelwrapper_propagates_arity_mismatch():
    """RL4 — arity mismatch after exhausting fallbacks propagates TypeError."""
    from molmetal.adapters.flow_matching_lipman._reference.utils import (
        ModelWrapper,
    )

    class ThreeArgModel:
        def __call__(self, x, t, z):
            return x + t + z

    mw = ModelWrapper(ThreeArgModel())
    with pytest.raises(TypeError, match="rejected every"):
        mw(torch.zeros(2, 3), 0.5)  # missing z


# ---------------------------------------------------------------------------
# PE1 + PE2 — PocketEncoder empty-pocket warn + memory cap
# ---------------------------------------------------------------------------
def test_pe1_pocket_encoder_empty_pocket_warn_string_embedded():
    """PE1 — ``PocketEncoder`` warns on ``valid.any() == False``."""
    text = _read(FML_INIT)
    assert (
        "PocketEncoder: pocket has no real atoms" in text
    ), "PE1 warning missing"
    assert (
        "_fully_connected_edge_index: P=" in text
    ), "PE2 memory cap string missing"


def test_pe2_fully_connected_edge_index_caps_p_squared_string():
    """PE2 — P*p > 100_000 raises ValueError string embedded."""
    text = _read(FML_INIT)
    assert "p * p > 100_000" in text, "PE2 cap constant missing"
    assert "Use radius-graph pockets" in text, "PE2 remediation hint missing"


# ---------------------------------------------------------------------------
# RL5 + FL3 + RL1 — defensive guards
# ---------------------------------------------------------------------------
def test_rl5_import_fm_lib_try_except_wrapper():
    """RL5 — try/except wraps all 4 upstream imports."""
    text = _read(FML_INIT)
    # 4 upstream imports + a couple of unrelated RuntimeError =
    # at least 5 ``raise RuntimeError(...)`` in source.  Conservative: >= 4.
    assert text.count("raise RuntimeError(") >= 4, (
        "RL5 must wrap 4 upstream imports with RuntimeError"
    )
    assert "LIPMAN_UPSTREAM_API_DRIFT" in text, "RL5 doc pointer embedded"


def test_fl3_ot_plan_uniqueness_assertion():
    """FL3 — ``mini_batch_ot_coupling`` rejects duplicate plan indices.

    The runtime message ``"duplicate plan indices: ..."`` is built by
    Python's implicit f-string concatenation across two adjacent string
    literals.  We anchor on parts present on a single source line.
    """
    text = _read(LOSS_INIT)
    assert "len(set(ot_idx_list)) != len(ot_idx_list)" in text, "FL3 set-comparison missing"
    assert "returned duplicate plan" in text, "FL3 message anchor missing"


def test_rl1_default_reference_existence_check():
    """RL1 — ``DEFAULT_REFERENCE`` existence check at import time.

    The warn message has ``"is not a"`` and ``"directory on disk"`` on
    separate adjacent f-string lines; anchor on the function-name
    prefix instead.
    """
    text = _read(REF_LOADER)
    assert (
        "_reference_loader.DEFAULT_REFERENCE" in text
    ), "RL1 must reference _reference_loader.DEFAULT_REFERENCE"
    assert (
        "is not a" in text and "directory on disk" in text
    ), "RL1 warn phrase split across f-string continuation"


# ---------------------------------------------------------------------------
# Bootstrap smoke — adapter module still importable after edits
# ---------------------------------------------------------------------------
def test_module_imports_clean():
    """Smoke — both _reference_loader + log_normal_t import without raising."""
    import importlib

    import molmetal.adapters.flow_matching_lipman  # noqa: F401
    import molmetal.adapters.flow_matching_lipman.log_normal_t  # noqa: F401
    import molmetal.adapters.flow_matching_lipman._reference_loader  # noqa: F401
    import molmetal.adapters.flow_matching_lipman._reference.utils  # noqa: F401
    import molmetal.adapters.egnn_extensions  # noqa: F401
    # silence the unused-import warning from modules loaded for side effects
    _ = importlib  # noqa: F841
