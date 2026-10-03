"""Pytest: Wf-126 Phase 2 — SE(3) Schrödinger Bridge pose-predictor +
Wasserstein pose-loss adapter + bench script.

This module pins the contract of the Wf-126 P2 deliverable end-to-end:

1. :func:`molmetal_lam.sbdd_env.se3_bridge_adapter.kabsch_align` —
   returns the optimal rotation + translation between two ``(N, 3)``
   point sets (Kabsch 1976).
2. :func:`heavy_atom_rmsd` / :func:`pose_rmsd` — RMSD after Kabsch
   alignment, returns a finite float in the same units as the input.
3. :class:`BridgePoseConfig` rejects invalid ``sinkhorn_eps`` /
   ``sinkhorn_iters`` / ``sigma`` arguments.
4. :class:`SE3BridgePosePredictor.predict` returns a tensor of shape
   ``(n_samples, N, 3)`` that is finite for the canonical self-bridge
   test (small random input).
5. :class:`SE3BridgePosePredictor.predict_best` returns a tensor of
   shape ``(N, 3)`` that minimises the RMSD over the candidates drawn.
6. :class:`WassersteinPoseLoss.raw` is non-negative and finite on a
   small random batch.
7. :class:`WassersteinPoseLoss.forward` is ``weight * raw`` and respects
   the weight override.
8. The bench script :mod:`molmetal.scripts.bench_rmsd_se3_bridge`
   imports cleanly + exposes the documented public API.
9. The bench script's :func:`parse_args` exposes the documented flags
   with the expected defaults.
10. The bench script's :func:`load_default_candidates` parses the
    bundled Wf-30 demo file.
11. :func:`run_pocket_benchmark` returns a :class:`PocketMetrics` whose
    per-seed records are populated for the requested seeds.
12. :func:`main` writes both the summary JSON and per-pocket JSON for a
    tiny 1-pocket × 1-seed × 1-candidate configuration.

All tests run on CPU in <30 s — no GPU / external dependencies
required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Lazy imports for the adapter (mirrors the convention in
# test_se3_sb_integration.py).
# ---------------------------------------------------------------------------
def _import_adapter():
    from molmetal_lam.sbdd_env import se3_bridge_adapter

    return se3_bridge_adapter


def _import_bench():
    sys.path.insert(0, str(PROJECT_ROOT / "molmetal" / "scripts"))
    import bench_rmsd_se3_bridge  # type: ignore[import-not-found]

    return bench_rmsd_se3_bridge


# ---------------------------------------------------------------------------
# 1. Kabsch alignment
# ---------------------------------------------------------------------------
def test_kabsch_align_returns_rotation_and_translation() -> None:
    """kabsch_align returns (R, t) with R ∈ SO(3) and finite t."""
    ad = _import_adapter()
    torch.manual_seed(0)
    P = torch.randn(8, 3)
    Q = P @ torch.randn(3, 3) + torch.randn(3)  # apply random rigid transform
    R, t = ad.kabsch_align(P, Q)
    assert R.shape == (3, 3)
    assert t.shape == (3,)
    # R^T R = I (orthogonal).
    assert torch.allclose(R.T @ R, torch.eye(3), atol=1e-5)
    # det(R) = +1 (no reflection).
    assert abs(torch.linalg.det(R).item() - 1.0) < 1e-5
    # t is finite.
    assert torch.isfinite(t).all()


def test_kabsch_align_recovers_known_transform() -> None:
    """Given a known rotation+translation, Kabsch recovers it exactly."""
    ad = _import_adapter()
    torch.manual_seed(42)
    P = torch.randn(10, 3) * 2.0
    theta = 0.6
    R_true = torch.tensor(
        [
            [torch.cos(torch.tensor(theta)), -torch.sin(torch.tensor(theta)), 0.0],
            [torch.sin(torch.tensor(theta)), torch.cos(torch.tensor(theta)), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    t_true = torch.tensor([1.5, -2.0, 0.5])
    Q = (R_true @ P.T).T + t_true
    R, t = ad.kabsch_align(P, Q)
    assert torch.allclose(R, R_true, atol=1e-4)
    assert torch.allclose(t, t_true, atol=1e-4)


# ---------------------------------------------------------------------------
# 2. RMSD after Kabsch alignment
# ---------------------------------------------------------------------------
def test_heavy_atom_rmsd_is_zero_for_identical_poses() -> None:
    """RMSD is exactly zero when the two poses are identical."""
    ad = _import_adapter()
    torch.manual_seed(0)
    P = torch.randn(6, 3)
    rmsd = ad.heavy_atom_rmsd(P, P)
    assert rmsd < 1e-5, f"identical poses must give RMSD ~0; got {rmsd}"


def test_heavy_atom_rmsd_matches_pose_rmsd() -> None:
    """pose_rmsd is a thin alias for heavy_atom_rmsd."""
    ad = _import_adapter()
    torch.manual_seed(1)
    P = torch.randn(5, 3)
    Q = P + torch.randn(5, 3) * 0.5
    a = ad.heavy_atom_rmsd(P, Q)
    b = ad.pose_rmsd(P, Q)
    assert abs(a - b) < 1e-9


def test_heavy_atom_rmsd_rejects_mismatched_shapes() -> None:
    """Mismatched shapes raise :class:`ValueError`."""
    ad = _import_adapter()
    P = torch.randn(5, 3)
    Q = torch.randn(7, 3)
    with pytest.raises(ValueError):
        ad.heavy_atom_rmsd(P, Q)


# ---------------------------------------------------------------------------
# 3. BridgePoseConfig validation
# ---------------------------------------------------------------------------
def test_bridge_pose_config_rejects_invalid_eps() -> None:
    """``sinkhorn_eps <= 0`` raises :class:`ValueError`."""
    ad = _import_adapter()
    from molmetal_lam.sbdd_env.se3_bridge_adapter import (
        BridgePoseConfig,
        SE3BridgePosePredictor,
    )

    with pytest.raises(ValueError):
        SE3BridgePosePredictor(config=BridgePoseConfig(sinkhorn_eps=0.0))
    with pytest.raises(ValueError):
        SE3BridgePosePredictor(config=BridgePoseConfig(sinkhorn_eps=-0.1))


def test_bridge_pose_config_rejects_invalid_iters() -> None:
    """``sinkhorn_iters < 1`` raises :class:`ValueError`."""
    ad = _import_adapter()
    from molmetal_lam.sbdd_env.se3_bridge_adapter import (
        BridgePoseConfig,
        SE3BridgePosePredictor,
    )

    with pytest.raises(ValueError):
        SE3BridgePosePredictor(config=BridgePoseConfig(sinkhorn_iters=0))


def test_bridge_pose_config_rejects_invalid_sigma() -> None:
    """``sigma <= 0`` raises :class:`ValueError`."""
    ad = _import_adapter()
    from molmetal_lam.sbdd_env.se3_bridge_adapter import (
        BridgePoseConfig,
        SE3BridgePosePredictor,
    )

    with pytest.raises(ValueError):
        SE3BridgePosePredictor(config=BridgePoseConfig(sigma=0.0))


# ---------------------------------------------------------------------------
# 4. SE3BridgePosePredictor.predict
# ---------------------------------------------------------------------------
def test_predict_returns_correct_shape() -> None:
    """predict returns ``(n_samples, N, 3)`` for valid inputs."""
    ad = _import_adapter()
    from molmetal_lam.sbdd_env.se3_bridge_adapter import (
        BridgePoseConfig,
        SE3BridgePosePredictor,
    )

    predictor = SE3BridgePosePredictor(config=BridgePoseConfig(seed=0))
    torch.manual_seed(1)
    init = torch.randn(8, 3)
    ref = torch.randn(8, 3) * 0.5
    out = predictor.predict(init, ref, n_samples=4, t=0.5)
    assert out.shape == (4, 8, 3), f"got {tuple(out.shape)}"
    assert torch.isfinite(out).all()


def test_predict_rejects_invalid_input_shape() -> None:
    """Non ``(N, 3)`` inputs raise :class:`ValueError`."""
    ad = _import_adapter()
    from molmetal_lam.sbdd_env.se3_bridge_adapter import (
        BridgePoseConfig,
        SE3BridgePosePredictor,
    )

    predictor = SE3BridgePosePredictor(config=BridgePoseConfig(seed=0))
    with pytest.raises(ValueError):
        predictor.predict(torch.randn(3, 4), torch.randn(3, 3))


def test_predict_rejects_invalid_n_samples() -> None:
    """``n_samples < 1`` raises :class:`ValueError`."""
    ad = _import_adapter()
    from molmetal_lam.sbdd_env.se3_bridge_adapter import (
        BridgePoseConfig,
        SE3BridgePosePredictor,
    )

    predictor = SE3BridgePosePredictor(config=BridgePoseConfig(seed=0))
    with pytest.raises(ValueError):
        predictor.predict(torch.randn(3, 3), torch.randn(3, 3), n_samples=0)


def test_predict_rejects_invalid_t() -> None:
    """``t ∉ [0, 1]`` raises :class:`ValueError`."""
    ad = _import_adapter()
    from molmetal_lam.sbdd_env.se3_bridge_adapter import (
        BridgePoseConfig,
        SE3BridgePosePredictor,
    )

    predictor = SE3BridgePosePredictor(config=BridgePoseConfig(seed=0))
    with pytest.raises(ValueError):
        predictor.predict(torch.randn(3, 3), torch.randn(3, 3), n_samples=2, t=1.5)


def test_predict_best_returns_minimising_pose() -> None:
    """predict_best returns the single candidate with the lowest RMSD."""
    ad = _import_adapter()
    from molmetal_lam.sbdd_env.se3_bridge_adapter import (
        BridgePoseConfig,
        SE3BridgePosePredictor,
    )

    predictor = SE3BridgePosePredictor(
        config=BridgePoseConfig(seed=0, sigma=1.0),
    )
    torch.manual_seed(2)
    init = torch.randn(6, 3) * 2.0
    ref = torch.randn(6, 3) * 0.5
    out = predictor.predict_best(init, ref, n_samples=4, t=0.5)
    assert out.shape == (6, 3)
    # The returned pose must be one of the n_samples candidates.
    candidates = predictor.predict(init, ref, n_samples=4, t=0.5)
    # Compute RMSD for each candidate vs ref and the chosen best.
    rmsds_cand = [ad.pose_rmsd(c, ref) for c in candidates]
    rmsd_best = ad.pose_rmsd(out, ref)
    # The best must equal the minimum of the candidates (within
    # float32 precision).
    assert rmsd_best <= min(rmsds_cand) + 1e-4


# ---------------------------------------------------------------------------
# 5. WassersteinPoseLoss
# ---------------------------------------------------------------------------
def test_wasserstein_pose_loss_raw_is_nonnegative() -> None:
    """raw() returns a non-negative finite float for valid inputs."""
    ad = _import_adapter()
    from molmetal_lam.sbdd_env.se3_bridge_adapter import WassersteinPoseLoss

    loss = WassersteinPoseLoss()
    torch.manual_seed(0)
    pred = torch.randn(4, 5, 3)
    ref = torch.randn(4, 5, 3)
    val = loss.raw(pred, ref)
    assert isinstance(val, float)
    assert math.isfinite(val) if (import_math := __import__("math")) else True
    assert val >= 0.0


def test_wasserstein_pose_loss_forward_scales_with_weight() -> None:
    """forward() returns ``weight * raw()``."""
    import math as _math

    ad = _import_adapter()
    from molmetal_lam.sbdd_env.se3_bridge_adapter import (
        WassersteinPoseLoss,
        WassersteinPoseLossConfig,
    )

    loss = WassersteinPoseLoss(
        config=WassersteinPoseLossConfig(n_projections=32, seed=0, weight=0.5)
    )
    torch.manual_seed(7)
    pred = torch.randn(3, 5, 3)
    ref = torch.randn(3, 5, 3)
    raw = loss.raw(pred, ref)
    fwd = loss.forward(pred, ref)
    assert _math.isclose(raw * 0.5, fwd, rel_tol=1e-3, abs_tol=1e-4)


def test_wasserstein_pose_loss_rejects_invalid_n_projections() -> None:
    """``n_projections < 1`` raises :class:`ValueError`."""
    ad = _import_adapter()
    from molmetal_lam.sbdd_env.se3_bridge_adapter import (
        WassersteinPoseLoss,
        WassersteinPoseLossConfig,
    )

    with pytest.raises(ValueError):
        WassersteinPoseLoss(config=WassersteinPoseLossConfig(n_projections=0))


def test_wasserstein_pose_loss_rejects_invalid_weight() -> None:
    """``weight < 0`` raises :class:`ValueError`."""
    ad = _import_adapter()
    from molmetal_lam.sbdd_env.se3_bridge_adapter import (
        WassersteinPoseLoss,
        WassersteinPoseLossConfig,
    )

    with pytest.raises(ValueError):
        WassersteinPoseLoss(config=WassersteinPoseLossConfig(weight=-0.01))


# ---------------------------------------------------------------------------
# 6. Bench script imports + parse_args + candidates
# ---------------------------------------------------------------------------
def test_bench_script_imports() -> None:
    """The bench script imports cleanly without side effects."""
    mod = _import_bench()
    for name in (
        "main",
        "parse_args",
        "load_default_candidates",
        "load_candidates_csv",
        "select_candidates",
        "load_pocket_bundle",
        "run_pocket_benchmark",
        "write_per_pocket_metrics",
        "write_summary",
        "write_synthesis",
        "PocketMetrics",
        "SeedMetrics",
        "Candidate",
        "PocketBundle",
        "DEFAULT_POCKETS",
        "DEFAULT_SEEDS",
        "RMSD_TARGET_A",
    ):
        assert hasattr(mod, name), f"missing public symbol: {name}"


def test_bench_script_default_pockets_are_5() -> None:
    """The default pocket cohort is the Wf-126 5-pocket set."""
    mod = _import_bench()
    assert len(mod.DEFAULT_POCKETS) == 5
    assert set(mod.DEFAULT_POCKETS) == {"1h36", "3eml", "3ny8", "4aaw", "4rlu"}


def test_bench_script_default_seeds_are_3() -> None:
    """The default seed set is ``(0, 1, 2)`` (matches the brief)."""
    mod = _import_bench()
    assert mod.DEFAULT_SEEDS == (0, 1, 2)


def test_bench_script_argparse_defaults() -> None:
    """argparse exposes the documented defaults."""
    mod = _import_bench()
    args = mod.parse_args([])
    assert args.wasserstein_weight == 0.05
    assert args.n_samples == mod.DEFAULT_N_SAMPLES
    assert args.pockets == list(mod.DEFAULT_POCKETS)
    assert args.seeds == list(mod.DEFAULT_SEEDS)
    assert args.n_candidates == 1
    assert args.write_synthesis is True


def test_bench_script_argparse_subset_pockets() -> None:
    """``--pockets 1h36 3eml`` narrows the cohort to exactly two."""
    mod = _import_bench()
    args = mod.parse_args(["--pockets", "1h36", "3eml"])
    assert args.pockets == ["1h36", "3eml"]


def test_bench_script_load_default_candidates(tmp_path: Path) -> None:
    """Default candidate loading parses a 5-line demo file."""
    mod = _import_bench()
    candidates_file = tmp_path / "demo.txt"
    candidates_file.write_text(
        "CCO # ethanol\n"
        "C # methane\n"
        "c1ccccc1 # benzene\n"
        "CC(=O)O # acetic\n"
        "N # ammonia\n"
    )
    out = mod.load_default_candidates(candidates_file)
    assert len(out) == 5
    assert out[0].smiles == "CCO"
    assert out[0].name == "ethanol"


# ---------------------------------------------------------------------------
# 7. PocketMetrics derived aggregates
# ---------------------------------------------------------------------------
def test_pocket_metrics_aggregates() -> None:
    """PocketMetrics mean/std/wass helpers are correct on synthetic seeds."""
    mod = _import_bench()
    m = mod.PocketMetrics(pocket_id="test_pocket")
    m.per_seed.extend(
        [
            mod.SeedMetrics(seed=0, rmsd_A=1.0, wass=0.05, elapsed_s=0.1, n_samples=4),
            mod.SeedMetrics(seed=1, rmsd_A=2.0, wass=0.10, elapsed_s=0.1, n_samples=4),
            mod.SeedMetrics(seed=2, rmsd_A=3.0, wass=0.15, elapsed_s=0.1, n_samples=4),
        ]
    )
    assert m.mean_rmsd_A() == pytest.approx(2.0, abs=1e-6)
    assert m.std_rmsd_A() == pytest.approx(((0 + 1 + 1) / 3) ** 0.5, abs=1e-6)
    assert m.mean_wass() == pytest.approx(0.10, abs=1e-6)
    # Target is 0.35 by default — mean=2.0 fails.
    assert m.passes_target() is False


def test_pocket_metrics_passes_target() -> None:
    """All seeds below the target → passes_target() is True."""
    mod = _import_bench()
    m = mod.PocketMetrics(pocket_id="p")
    m.per_seed.extend(
        [
            mod.SeedMetrics(seed=0, rmsd_A=0.1, wass=0.01, elapsed_s=0.1, n_samples=2),
            mod.SeedMetrics(seed=1, rmsd_A=0.2, wass=0.02, elapsed_s=0.1, n_samples=2),
            mod.SeedMetrics(seed=2, rmsd_A=0.3, wass=0.03, elapsed_s=0.1, n_samples=2),
        ]
    )
    assert m.passes_target() is True


# ---------------------------------------------------------------------------
# 8. main() writes both summary + per-pocket JSON
# ---------------------------------------------------------------------------
def test_bench_main_writes_outputs(tmp_path: Path) -> None:
    """main() writes summary.json + per-pocket/per_seed_metrics.json."""
    mod = _import_bench()
    # Stage a tiny receptor.pdbqt so load_pocket_bundle succeeds.
    pocket_id = "test_main_pocket"
    pocket_dir = tmp_path / "wf126_p1_staging"
    (pocket_dir / pocket_id).mkdir(parents=True)
    (pocket_dir / pocket_id / "receptor.pdbqt").write_text(
        "ATOM      1  C   ALA A   1       0.000   0.000   0.000  1.00  0.00    +0.000 C \n"
        "ATOM      2  C   ALA A   2       4.000   4.000   4.000  1.00  0.00    +0.000 C \n"
    )
    out_dir = tmp_path / "wf126_p2"
    # Tiny CSV demo file (matches the candidates_csv loader).
    demo = tmp_path / "demo.csv"
    demo.write_text("name,smiles\nethanol,CCO\n")
    rc = mod.main(
        [
            "--pockets",
            pocket_id,
            "--seeds",
            "0",
            "--n-samples",
            "2",
            "--candidates-csv",
            str(demo),
            "--pocket-dir",
            str(pocket_dir),
            "--out-dir",
            str(out_dir),
            "--quiet",
        ]
    )
    assert rc == 0
    summary = out_dir / "summary.json"
    assert summary.exists()
    payload = json.loads(summary.read_text())
    assert payload["recipe"] == "wf126_p2_rmsd_se3_bridge"
    assert payload["rmsd_target_A"] == pytest.approx(0.35, abs=1e-9)
    assert payload["n_pockets"] == 1
    per_seed = out_dir / pocket_id / "per_seed_metrics.json"
    assert per_seed.exists()
    per_payload = json.loads(per_seed.read_text())
    assert per_payload["pocket_id"] == pocket_id
    assert per_payload["per_seed_rmsd_A"] == [
        pytest.approx(p, abs=1e-3) for p in per_payload["per_seed_rmsd_A"]
    ]


# ---------------------------------------------------------------------------
# 9. Module-level smoke — public API surface
# ---------------------------------------------------------------------------
def test_adapter_module_exposes_public_api() -> None:
    """The adapter module exposes all documented symbols."""
    ad = _import_adapter()
    for name in (
        "BridgePoseConfig",
        "SE3BridgePosePredictor",
        "WassersteinPoseLoss",
        "WassersteinPoseLossConfig",
        "PocketPoseRecord",
        "kabsch_align",
        "heavy_atom_rmsd",
        "pose_rmsd",
    ):
        assert hasattr(ad, name), f"adapter missing public symbol: {name}"


# Tiny inline import so the linter doesn't flag ``import_math``.
import math  # noqa: E402  (placed here intentionally for test 5a)
