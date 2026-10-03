"""R18 Wf-23 Phase 2 — train-script contract tests.

=====================================================================
CPU-only smoke + contract tests for the R18 training scripts:

* :mod:`molmetal.scripts.train_se3sb` — R18 W2.1 SE(3) Schrödinger-Bridge
  5-metal multitask trainer.
* :mod:`molmetal.scripts.train_transport` — R18 W2.2 Transport-Relaxation
  5-metal multitask trainer.
* :mod:`molmetal.scripts.train_multiph` — R18 W2.3 Multi-Parameter PH
  5-metal multitask trainer.
* :mod:`molmetal.scripts.train_ru_multitask` — R17 baseline with the
  additive ``--init`` choices ``se3sb`` / ``transportrelax`` / ``multiph``.

The tests focus on:

* Module imports (no syntax errors, no missing dependencies).
* ``argparse`` ``--init`` choices include all R18 inits (R17 backward
  compatibility preserved).
* ``--help`` output is non-empty and contains the expected CLI flags.
* Default arg values match the W2.1/W2.2/W2.3 spec.
* Adapter-side constructions are exercised on tiny CPU tensors so we
  catch regressions in the wiring without running the full GPU path.

All tests are CPU-only and skip gracefully when optional deps are
unavailable.  They DO NOT spawn the actual training loop (that's
deferred to Phase 4).
"""

from __future__ import annotations

import importlib
import io
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_DIR = _REPO_ROOT / "molmetal" / "scripts"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# 1. Module imports
# ---------------------------------------------------------------------------
def _import_module(name: str):
    """Lazy-import a script as a module so ``parse_args`` is testable."""
    sys.path.insert(0, str(_SCRIPT_DIR))
    try:
        return importlib.import_module(name)
    finally:
        # Don't pop — the path is harmless and idempotent.
        pass


def test_train_se3sb_imports() -> None:
    """``molmetal.scripts.train_se3sb`` imports without raising."""
    mod = _import_module("train_se3sb")
    assert mod is not None
    assert hasattr(mod, "parse_args")
    assert hasattr(mod, "main")
    assert hasattr(mod, "train_one_seed")


def test_train_transport_imports() -> None:
    """``molmetal.scripts.train_transport`` imports without raising."""
    mod = _import_module("train_transport")
    assert mod is not None
    assert hasattr(mod, "parse_args")
    assert hasattr(mod, "main")
    assert hasattr(mod, "train_one_seed")


def test_train_multiph_imports() -> None:
    """``molmetal.scripts.train_multiph`` imports without raising."""
    mod = _import_module("train_multiph")
    assert mod is not None
    assert hasattr(mod, "parse_args")
    assert hasattr(mod, "main")
    assert hasattr(mod, "train_one_seed")


def test_train_ru_multitask_imports() -> None:
    """``molmetal.scripts.train_ru_multitask`` imports without raising
    and the new ``_SE3SBVelocityField`` / ``_RelaxedEndpointTerm`` /
    ``_MultiPHPocketFeatures`` lazy handles are present.
    """
    mod = _import_module("train_ru_multitask")
    assert mod is not None
    assert hasattr(mod, "parse_args")
    assert hasattr(mod, "main")


def test_train_ru_multitask_has_r18_init_lazy_imports() -> None:
    """The R18 init wiring is wired in (even if the adapter is missing)."""
    mod = _import_module("train_ru_multitask")
    # The three lazy import handles are always present (the adapters
    # may be None if the import fails, but the names must exist).
    assert hasattr(mod, "_SE3SBVelocityField")
    assert hasattr(mod, "_RelaxedEndpointTerm")
    assert hasattr(mod, "_MultiPHPocketFeatures")


# ---------------------------------------------------------------------------
# 2. argparse --init choices
# ---------------------------------------------------------------------------
R18_INIT_CHOICES = ["tmqm", "random", "se3sb", "transportrelax", "multiph"]


def test_argparse_init_choices_in_ru_multitask() -> None:
    """``train_ru_multitask --init`` exposes all R17+R18 choices."""
    mod = _import_module("train_ru_multitask")
    # Probe the parser for the --init choices.
    args = mod.parse_args(["--init", "se3sb"])
    assert args.init == "se3sb"
    args = mod.parse_args(["--init", "transportrelax"])
    assert args.init == "transportrelax"
    args = mod.parse_args(["--init", "multiph"])
    assert args.init == "multiph"
    # Backward compatibility — the R17 choices still work.
    args = mod.parse_args(["--init", "tmqm"])
    assert args.init == "tmqm"
    args = mod.parse_args(["--init", "random"])
    assert args.init == "random"


def test_argparse_init_choices_via_help_text() -> None:
    """The ``--init`` choices appear in the ``--help`` text."""
    mod = _import_module("train_ru_multitask")
    buf = io.StringIO()
    with redirect_stdout(buf):
        try:
            mod.parse_args(["--help"])
        except SystemExit:
            pass
    help_text = buf.getvalue()
    for choice in R18_INIT_CHOICES:
        assert choice in help_text, (
            f"--init choice {choice!r} missing from --help text:\n" f"{help_text}"
        )


def test_argparse_rejects_unknown_init() -> None:
    """Unknown ``--init`` values are rejected (SystemExit on bad choice)."""
    mod = _import_module("train_ru_multitask")
    with pytest.raises(SystemExit):
        mod.parse_args(["--init", "bogus"])


# ---------------------------------------------------------------------------
# 3. argparse --help
# ---------------------------------------------------------------------------
def test_argparse_help_se3sb() -> None:
    """``train_se3sb --help`` exits cleanly and emits expected flags."""
    mod = _import_module("train_se3sb")
    buf = io.StringIO()
    with redirect_stdout(buf):
        with pytest.raises(SystemExit) as exc:
            mod.parse_args(["--help"])
    # argparse --help exits 0
    assert exc.value.code == 0
    text = buf.getvalue()
    assert "--epochs" in text
    assert "--batch" in text
    assert "--lr" in text
    assert "--se3-n-steps" in text
    assert "--se3-use-sinkhorn" in text
    assert "--se3-sinkhorn-eps" in text


def test_argparse_help_transport() -> None:
    """``train_transport --help`` exits cleanly and emits expected flags."""
    mod = _import_module("train_transport")
    buf = io.StringIO()
    with redirect_stdout(buf):
        with pytest.raises(SystemExit) as exc:
            mod.parse_args(["--help"])
    assert exc.value.code == 0
    text = buf.getvalue()
    assert "--epochs" in text
    assert "--batch" in text
    assert "--lambda-relax" in text
    assert "--eps" in text


def test_argparse_help_multiph() -> None:
    """``train_multiph --help`` exits cleanly and emits expected flags."""
    mod = _import_module("train_multiph")
    buf = io.StringIO()
    with redirect_stdout(buf):
        with pytest.raises(SystemExit) as exc:
            mod.parse_args(["--help"])
    assert exc.value.code == 0
    text = buf.getvalue()
    assert "--epochs" in text
    assert "--batch" in text
    assert "--n-bins" in text
    assert "--mphp-max-radius" in text
    assert "--mphp-project-dim" in text


def test_argparse_help_ru_multitask() -> None:
    """``train_ru_multitask --help`` exits cleanly and emits expected flags."""
    mod = _import_module("train_ru_multitask")
    buf = io.StringIO()
    with redirect_stdout(buf):
        with pytest.raises(SystemExit) as exc:
            mod.parse_args(["--help"])
    assert exc.value.code == 0
    text = buf.getvalue()
    assert "--epochs" in text
    assert "--batch" in text
    assert "--init" in text


# ---------------------------------------------------------------------------
# 4. default arg values
# ---------------------------------------------------------------------------
def test_default_args_se3sb() -> None:
    """``train_se3sb`` defaults match the R18 W2.1 spec."""
    mod = _import_module("train_se3sb")
    args = mod.parse_args([])
    assert args.epochs == mod.RECIPE_EPOCHS == 50
    assert args.batch == mod.RECIPE_BATCH == 128
    assert args.lr == mod.RECIPE_LR == 1e-3
    assert tuple(args.seeds) == mod.RECIPE_SEEDS
    assert args.hidden == mod.RECIPE_HIDDEN == 300
    assert args.depth == mod.RECIPE_DEPTH == 3
    assert args.se3_n_steps == mod.RECIPE_SE3_INTEGRATION_STEPS
    assert args.se3_use_sinkhorn == mod.RECIPE_SE3_USE_SINKHORN
    assert args.se3_sinkhorn_eps == mod.RECIPE_SE3_SINKHORN_EPS


def test_default_args_transport() -> None:
    """``train_transport`` defaults match the R18 W2.2 spec."""
    mod = _import_module("train_transport")
    args = mod.parse_args([])
    assert args.epochs == mod.RECIPE_EPOCHS == 50
    assert args.batch == mod.RECIPE_BATCH == 128
    assert args.lr == mod.RECIPE_LR == 1e-3
    assert tuple(args.seeds) == mod.RECIPE_SEEDS
    assert args.lambda_relax == mod.RECIPE_LAMBDA_RELAX == 1e-3
    assert args.eps == mod.RECIPE_EPS == 0.1


def test_default_args_multiph() -> None:
    """``train_multiph`` defaults match the R18 W2.3 spec."""
    mod = _import_module("train_multiph")
    args = mod.parse_args([])
    assert args.epochs == mod.RECIPE_EPOCHS == 50
    assert args.batch == mod.RECIPE_BATCH == 128
    assert args.lr == mod.RECIPE_LR == 1e-3
    assert tuple(args.seeds) == mod.RECIPE_SEEDS
    assert args.n_bins == mod.RECIPE_MPHP_N_BINS
    assert args.mphp_max_radius == mod.RECIPE_MPHP_MAX_RADIUS
    assert args.mphp_max_coord == mod.RECIPE_MPHP_MAX_COORD
    assert args.mphp_project_dim == mod.RECIPE_MPHP_PROJECT_DIM


def test_default_args_ru_multitask_preserves_r17() -> None:
    """``train_ru_multitask`` defaults are unchanged for the R17 path."""
    mod = _import_module("train_ru_multitask")
    args = mod.parse_args([])
    assert args.epochs == mod.RECIPE_EPOCHS == 100
    assert args.batch == mod.RECIPE_BATCH == 128
    assert args.lr == mod.RECIPE_LR == 1e-3
    assert tuple(args.seeds) == mod.RECIPE_SEEDS
    assert args.init == "tmqm"  # R17 default preserved


def test_default_args_ru_multitask_accepts_r18_init() -> None:
    """``train_ru_multitask`` accepts the R18 init choices in CLI form."""
    mod = _import_module("train_ru_multitask")
    for init in ("se3sb", "transportrelax", "multiph"):
        args = mod.parse_args(["--init", init])
        assert args.init == init


# ---------------------------------------------------------------------------
# 5. CLI override smoke (no actual training, just arg wiring)
# ---------------------------------------------------------------------------
def test_cli_override_se3sb() -> None:
    """``train_se3sb`` parses user-provided overrides correctly."""
    mod = _import_module("train_se3sb")
    args = mod.parse_args(
        [
            "--epochs",
            "10",
            "--batch",
            "32",
            "--lr",
            "5e-4",
            "--seeds",
            "0",
            "1",
            "--se3-n-steps",
            "16",
            "--se3-sinkhorn-eps",
            "0.05",
        ]
    )
    assert args.epochs == 10
    assert args.batch == 32
    assert args.lr == 5e-4
    assert args.seeds == [0, 1]
    assert args.se3_n_steps == 16
    assert args.se3_sinkhorn_eps == 0.05


def test_cli_override_transport() -> None:
    """``train_transport`` parses user-provided overrides correctly."""
    mod = _import_module("train_transport")
    args = mod.parse_args(
        [
            "--epochs",
            "10",
            "--batch",
            "32",
            "--lambda-relax",
            "1e-2",
            "--eps",
            "0.2",
        ]
    )
    assert args.epochs == 10
    assert args.batch == 32
    assert args.lambda_relax == 1e-2
    assert args.eps == 0.2


def test_cli_override_multiph() -> None:
    """``train_multiph`` parses user-provided overrides correctly."""
    mod = _import_module("train_multiph")
    args = mod.parse_args(
        [
            "--epochs",
            "10",
            "--batch",
            "32",
            "--n-bins",
            "16",
            "--mphp-project-dim",
            "32",
        ]
    )
    assert args.epochs == 10
    assert args.batch == 32
    assert args.n_bins == 16
    assert args.mphp_project_dim == 32


# ---------------------------------------------------------------------------
# 6. Adapter-side constructions
# ---------------------------------------------------------------------------
def test_se3_velocity_field_construction() -> None:
    """The W2.1 SE(3) velocity field can be built and run on CPU."""
    pytest.importorskip("torch")
    import torch
    import torch.nn as nn
    from molmetal.adapters.se3_sb_integration import SE3SBVelocityField

    class _Stub(nn.Module):
        def forward(self, x, t=None, pocket=None):
            return torch.zeros_like(x)

    vf = SE3SBVelocityField(base_velocity_net=_Stub(), manifold="SE3")
    x = torch.randn(2, 4, 3)
    t = torch.tensor([0.5, 0.5])
    xi = vf(x, t)
    assert xi.shape == (2, 6)


def test_relax_endpoint_term_construction() -> None:
    """The W2.2 RelaxedEndpointTerm can be built and queried on CPU."""
    pytest.importorskip("torch")
    from molmetal.adapters.transport_relaxation_integration import (
        RelaxedEndpointTerm,
    )

    term = RelaxedEndpointTerm(lambda_relax=1e-3, eps=0.1)
    import torch

    traj = torch.randn(4, 3)
    e = torch.randn(4)
    out = term.forward(traj, e)
    assert out.dim() == 0  # scalar


def test_relax_zero_lambda_returns_zero() -> None:
    """W2.2 contract: ``lambda_relax=0`` makes the penalty identically zero."""
    pytest.importorskip("torch")
    from molmetal.adapters.transport_relaxation_integration import (
        RelaxedEndpointTerm,
    )
    import torch

    term = RelaxedEndpointTerm(lambda_relax=0.0, eps=0.1)
    out = term.forward(torch.randn(4, 3), torch.randn(4))
    assert float(out.item()) == 0.0


def test_multiph_extractor_construction() -> None:
    """The W2.3 MultiPHPocketFeatures can be built and called on CPU."""
    pytest.importorskip("torch")
    from molmetal.adapters.multi_ph_integration import MultiPHPocketFeatures
    import torch

    mphp = MultiPHPocketFeatures(max_radius=5.0, max_coord=4.0, n_bins=8)
    pocket = torch.randn(2, 12, 3)
    feats = mphp(pocket)
    assert feats.shape == (2, 8, 8)


def test_se3_integration_smoke() -> None:
    """``integrate_with_cfm`` runs end-to-end on CPU with the stub twist head."""
    pytest.importorskip("torch")
    from molmetal.adapters.se3_sb_integration import (
        SE3SBVelocityField,
        integrate_with_cfm,
    )
    import torch

    class _Stub(torch.nn.Module):
        def forward(self, x, t=None, pocket=None):
            return torch.zeros_like(x)

    vf = SE3SBVelocityField(base_velocity_net=_Stub(), manifold="SE3")
    samples = torch.randn(2, 8, 3)
    x_final = integrate_with_cfm(
        samples,
        vf,
        n_steps=4,
        method="midpoint",
    )
    assert x_final.shape == samples.shape
    assert torch.isfinite(x_final).all().item()


def test_synthetic_pocket_shape() -> None:
    """``smiles_to_synthetic_pocket`` returns a (B, n_atoms, 3) tensor."""
    pytest.importorskip("torch")
    mod = _import_module("train_multiph")
    pocket = mod.smiles_to_synthetic_pocket(
        ["CCO", "c1ccccc1", "CCN"],
        n_pocket_atoms=12,
    )
    assert pocket.shape == (3, 12, 3)
    import torch

    assert torch.isfinite(pocket).all().item()


def test_multiph_projector_forward() -> None:
    """``MultiPHProjector`` flattens + projects to the requested dim."""
    pytest.importorskip("torch")
    mod = _import_module("train_multiph")
    import torch

    proj = mod.MultiPHProjector(n_bins=8, mphp_dim=16)
    feats = torch.randn(2, 8, 8)
    out = proj(feats)
    assert out.shape == (2, 16)


def test_per_trajectory_energies_shape() -> None:
    """``compute_per_trajectory_energies`` returns a 1-D energy tensor."""
    pytest.importorskip("torch")
    mod = _import_module("train_transport")
    import torch

    logits = {
        "Ru": torch.randn(4, 2),
        "Ir": torch.randn(3, 2),
    }
    ys = {
        "Ru": torch.tensor([0, 1, 0, 1]),
        "Ir": torch.tensor([1, 0, 1]),
    }
    e = mod.compute_per_trajectory_energies(logits, ys)
    assert e.dim() == 1
    assert e.shape[0] == 7  # 4 + 3


# ---------------------------------------------------------------------------
# 7. Smoke JSON round-trip — train_one_seed contract
# ---------------------------------------------------------------------------
def test_train_se3sb_se3_integration_smoke_helper() -> None:
    """``se3_integration_smoke`` returns a finite-flag dict."""
    pytest.importorskip("torch")
    mod = _import_module("train_se3sb")
    from molmetal.adapters.se3_sb_integration import SE3SBVelocityField
    import torch

    class _Stub(torch.nn.Module):
        def forward(self, x, t=None, pocket=None):
            return torch.zeros_like(x)

    vf = SE3SBVelocityField(base_velocity_net=_Stub(), manifold="SE3")
    res = mod.se3_integration_smoke(vf, n_atoms=4, n_steps=2)
    assert "finite" in res
    assert "out_norm" in res
    assert res["finite"] is True


def test_train_multiph_mphp_smoke_helper() -> None:
    """``mphp_smoke`` returns a finite-flag dict."""
    pytest.importorskip("torch")
    mod = _import_module("train_multiph")
    from molmetal.adapters.multi_ph_integration import MultiPHPocketFeatures

    mphp = MultiPHPocketFeatures(max_radius=5.0, max_coord=4.0, n_bins=8)
    res = mod.mphp_smoke(mphp, ["CCO", "c1ccccc1", "CCN"], n_pocket_atoms=12)
    assert "finite" in res
    assert res["finite"] is True
    assert res.get("feature_shape") == [3, 8, 8]


# ---------------------------------------------------------------------------
# 8. R17 backward compatibility — train_ru_multitask default --init tmqm
# ---------------------------------------------------------------------------
def test_train_ru_multitask_default_tmqm_path_unaffected() -> None:
    """The R17 default ``--init tmqm`` is unchanged — no R18 wiring fires.

    We probe ``parse_args`` and the script source to make sure the
    default value is still ``"tmqm"`` and the R18 lazy imports are
    guarded by ``args.init in {se3sb, transportrelax, multiph}``.
    """
    mod = _import_module("train_ru_multitask")
    args = mod.parse_args([])
    assert args.init == "tmqm"
    # The argparse choices include the R18 inits.
    src_path = _SCRIPT_DIR / "train_ru_multitask.py"
    src = src_path.read_text()
    assert 'choices=["tmqm", "random", "se3sb", "transportrelax", "multiph"]' in src
    assert 'args.init == "se3sb"' in src
    assert 'args.init == "transportrelax"' in src
    assert 'args.init == "multiph"' in src


# ---------------------------------------------------------------------------
# 9. Output paths
# ---------------------------------------------------------------------------
def test_se3sb_default_metrics_path() -> None:
    """``train_se3sb`` writes ``metrics/by_metric/se3sb_5metal_multitask.json``."""
    mod = _import_module("train_se3sb")
    args = mod.parse_args([])
    assert args.metrics_path.name == "se3sb_5metal_multitask.json"
    assert args.summary_path.name == "se3sb_final_summary.json"


def test_transport_default_metrics_path() -> None:
    """``train_transport`` writes ``metrics/by_metric/transport_5metal_multitask.json``."""
    mod = _import_module("train_transport")
    args = mod.parse_args([])
    assert args.metrics_path.name == "transport_5metal_multitask.json"
    assert args.summary_path.name == "transport_final_summary.json"


def test_multiph_default_metrics_path() -> None:
    """``train_multiph`` writes ``metrics/by_metric/multiph_5metal_multitask.json``."""
    mod = _import_module("train_multiph")
    args = mod.parse_args([])
    assert args.metrics_path.name == "multiph_5metal_multitask.json"
    assert args.summary_path.name == "multiph_final_summary.json"


# ---------------------------------------------------------------------------
# 10. Recipe constants
# ---------------------------------------------------------------------------
def test_recipe_constants_se3sb() -> None:
    """``train_se3sb`` recipe constants match the W2.1 spec."""
    mod = _import_module("train_se3sb")
    assert mod.RECIPE_EPOCHS == 50
    assert mod.RECIPE_BATCH == 128
    assert mod.RECIPE_LR == 1e-3
    assert mod.RECIPE_SEEDS == (0, 1, 2, 3, 4)
    assert mod.RECIPE_HIDDEN == 300
    assert mod.RECIPE_DEPTH == 3


def test_recipe_constants_transport() -> None:
    """``train_transport`` recipe constants match the W2.2 spec."""
    mod = _import_module("train_transport")
    assert mod.RECIPE_EPOCHS == 50
    assert mod.RECIPE_BATCH == 128
    assert mod.RECIPE_LR == 1e-3
    assert mod.RECIPE_SEEDS == (0, 1, 2, 3, 4)
    assert mod.RECIPE_LAMBDA_RELAX == 1e-3
    assert mod.RECIPE_EPS == 0.1


def test_recipe_constants_multiph() -> None:
    """``train_multiph`` recipe constants match the W2.3 spec."""
    mod = _import_module("train_multiph")
    assert mod.RECIPE_EPOCHS == 50
    assert mod.RECIPE_BATCH == 128
    assert mod.RECIPE_LR == 1e-3
    assert mod.RECIPE_SEEDS == (0, 1, 2, 3, 4)
    assert mod.RECIPE_MPHP_N_BINS == 8
    assert mod.RECIPE_MPHP_PROJECT_DIM == 16


# ---------------------------------------------------------------------------
# 11. End-to-end CLI invocation (argparse only — no GPU)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "script_name,init_flag",
    [
        ("train_se3sb.py", None),
        ("train_transport.py", None),
        ("train_multiph.py", None),
        ("train_ru_multitask.py", "se3sb"),
        ("train_ru_multitask.py", "transportrelax"),
        ("train_ru_multitask.py", "multiph"),
        ("train_ru_multitask.py", "tmqm"),
        ("train_ru_multitask.py", "random"),
    ],
)
def test_script_cli_help_succeeds(script_name: str, init_flag) -> None:
    """``python <script> --help`` returns exit-code 0 across all R18 inits."""
    cmd = [sys.executable, str(_SCRIPT_DIR / script_name), "--help"]
    if init_flag is not None:
        # Insert the init flag in front of --help so argparse sees it.
        cmd = [
            sys.executable,
            str(_SCRIPT_DIR / script_name),
            "--init",
            init_flag,
            "--help",
        ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert (
        result.returncode == 0
    ), f"{script_name} --help failed: stderr={result.stderr}"
    assert "--help" in result.stdout or "options:" in result.stdout


# ---------------------------------------------------------------------------
# 12. JSON serialisation helper round-trip
# ---------------------------------------------------------------------------
def test_to_jsonable_handles_nan() -> None:
    """``_to_jsonable`` converts NaN → None and numpy scalars → Python scalars."""
    mod_se3sb = _import_module("train_se3sb")
    import numpy as np

    obj = {
        "nan": float("nan"),
        "inf": float("inf"),
        "np_float": np.float32(1.5),
        "np_int": np.int64(7),
        "nested": [np.array([1.0, 2.0]), {"k": float("nan")}],
    }
    out = mod_se3sb._to_jsonable(obj)
    assert out["nan"] is None
    assert out["np_float"] == 1.5
    assert out["np_int"] == 7
    assert out["nested"][0] == [1.0, 2.0]
    assert out["nested"][1]["k"] is None


# ---------------------------------------------------------------------------
# 13. Final integration — full module import sanity
# ---------------------------------------------------------------------------
def test_all_r18_train_scripts_importable() -> None:
    """All three R18 train scripts import in sequence without leaking state."""
    for name in ("train_se3sb", "train_transport", "train_multiph"):
        mod = _import_module(name)
        assert mod is not None
        # Each must expose the trio of public surface functions.
        assert callable(getattr(mod, "main"))
        assert callable(getattr(mod, "parse_args"))
        assert callable(getattr(mod, "train_one_seed"))
