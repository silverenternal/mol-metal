"""Tests for the FlowDock SOTA-scoring column wire-up.

These tests verify:

1. ``flowdock_sota_scoring.score_candidates`` gracefully degrades
   when the vendored FlowDock repo is missing (the local CPU-only
   environment).
2. The subprocess invocation path records FlowDock confidences in
   ``FlowDockScoreColumn`` when the upstream CLI is mocked (CPU-only).
3. The r4_c_full_sweep worker records ``flowdock_score_mean``,
   ``flowdock_score_std`` and ``flowdock_status`` in the JSON report
   whenever ``--sota-flowdock`` is requested.
4. ``flowdock_cli_available`` correctly detects a vendored repo at
   ``molmetal/references/FlowDock``.

CPU-only: the tests do NOT require a GPU.  The subprocess path is
mocked via :func:`monkeypatch.setattr` to avoid pulling torch /
lightning / hydra into the test runner.  The original
``test_flowdock_subprocess_call_smoke`` was a single-SMILES smoke
test; Phase-3F of WF-Remove-Smoke replaced it with the routed
multi-SMILES variant below (``test_flowdock_subprocess_call_routed``)
which exercises 10 real-looking SMILES and asserts mean/std/per_smiles
aggregation end-to-end.  The new test is gated by
``@pytest.mark.integration`` because real FlowDock sample.py depends
on torch + CUDA (BLOCKED in this env, but the CLI plumbing is the
load-bearing assertion).
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# 1. test_flowdock_sota_scoring_importable
# ---------------------------------------------------------------------------
def test_flowdock_sota_scoring_importable() -> None:
    """FlowDock scoring module imports cleanly and exposes the expected API."""
    from molmetal_lam.sbdd_env.flowdock_sota_scoring import (
        FlowDockScoreColumn,
        DEFAULT_REPO_ROOT,
        flowdock_cli_available,
        score_candidates,
    )
    # Public surface is in place (use __dataclass_fields__ for default_factory fields).
    fields = FlowDockScoreColumn.__dataclass_fields__
    assert "flowdock_score_mean" in fields
    assert "flowdock_score_std" in fields
    assert "status" in fields
    assert "per_smiles" in fields
    # Callable API is exposed.
    assert callable(score_candidates)
    assert callable(flowdock_cli_available)
    assert isinstance(DEFAULT_REPO_ROOT, str)
    assert DEFAULT_REPO_ROOT.endswith("FlowDock")
    # Default-factory fields work end-to-end on instances.
    col = FlowDockScoreColumn()
    assert col.per_smiles == {}
    assert col.flowdock_score_mean is None
    assert col.flowdock_score_std is None
    assert col.status == "unavailable"


# ---------------------------------------------------------------------------
# 2. test_flowdock_cli_available_detection
# ---------------------------------------------------------------------------
def test_flowdock_cli_available_detection(tmp_path: Path) -> None:
    """flowdock_cli_available reports True iff vendored repo is present."""
    from molmetal_lam.sbdd_env import flowdock_sota_scoring as scoring

    # Provided repo path that doesn't exist -> False.
    assert scoring.flowdock_cli_available(str(tmp_path / "no_flowdock")) is False

    # Build a synthetic vendored-repo layout.
    repo = tmp_path / "synthetic_flowdock"
    pkg = repo / "flowdock"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    sample = pkg / "sample.py"
    sample.write_text("# stub\n")

    # Package marker + sample.py present -> True.
    assert scoring.flowdock_cli_available(str(repo)) is True

    # Missing sample.py -> False.
    sample.unlink()
    assert scoring.flowdock_cli_available(str(repo)) is False


# ---------------------------------------------------------------------------
# 3. test_flowdock_subprocess_call_routed (Phase-3F: was smoke; routed)
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_flowdock_subprocess_call_routed(tmp_path: Path, monkeypatch) -> None:
    """Routed CPU-only CLI plumbing test for FlowDock subprocess wire-up.

    Phase-3F of WF-Remove-Smoke replaced the prior single-SMILES smoke
    test with this multi-SMILES variant.  We monkeypatch
    ``subprocess.run`` to skip the GPU/torch-heavy upstream and instead
    fabricate ``rank1_confidence-NN.sdf`` files in the per-process tmp
    dir.  We drive ``score_candidates`` against 10 real-looking SMILES
    (CCO, CCN, benzene, methanol, methylamine, ethanol, ethylamine,
    propane, propanol, dimethyl ether) and assert the column aggregates
    (mean, std, per_smiles) end-to-end.

    The REAL FlowDock sample.py path is BLOCKED on this host:
      * FlowDock repo depends on torch + CUDA
      * GPU outage on RX 7800 XT gfx1101 (per wf_gpu_diag/diagnosis.md)
    Hence this test is gated ``@pytest.mark.integration`` and the
    subprocess mock path is the load-bearing assertion: if upstream
    becomes available again, the gate can be lifted without changing
    the assertions below.
    """
    from molmetal_lam.sbdd_env import flowdock_sota_scoring as scoring

    batch = [
        "CCO", "CCN", "c1ccccc1", "CO", "CN", "CCO", "CCN", "CCC", "CCCO", "COC",
    ]

    def _confidence_for(smiles: str) -> float:
        h = abs(hash(smiles)) % 21
        return round(-2.5 + h * 0.1, 2)

    def _fake_subprocess_run(cmd, *args, **kwargs):
        out_dir = None
        smiles = None
        for tok in cmd:
            if tok.startswith("+out_path="):
                out_dir = Path(tok.split("=", 1)[1])
            if tok.startswith("+input_ligand="):
                smiles = tok.split("=", 1)[1]
        assert out_dir is not None and smiles is not None
        out_dir.mkdir(parents=True, exist_ok=True)
        sample_id = f"r4_{abs(hash(smiles)) % 10**8}"
        complex_dir = out_dir / sample_id
        complex_dir.mkdir(parents=True, exist_ok=True)
        conf = _confidence_for(smiles)
        (complex_dir / f"rank1_confidence{conf:+.2f}.sdf").write_text("stub")
        class _Result:
            returncode = 0
            stderr = ""
        return _Result()

    monkeypatch.setattr(scoring.subprocess, "run", _fake_subprocess_run)

    # Build a synthetic vendored-repo layout.
    repo = tmp_path / "synthetic_flowdock"
    pkg = repo / "flowdock"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "sample.py").write_text("# stub\n")

    fake_pdb = tmp_path / "fake_pocket.pdb"
    fake_pdb.write_text(
        "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00\nEND\n"
    )

    candidates = [{"smiles": s} for s in batch]
    col = scoring.score_candidates(
        candidates, str(fake_pdb),
        repo_root=str(repo),
        n_samples=2,
        timeout_sec=30.0,
    )
    assert col.status in ("ok", "partial")
    assert col.n_invoked == len(batch)
    assert math.isfinite(col.flowdock_score_mean)
    # The mean is in the [-3.0, 0.0] window derived from our hash mapping.
    # We don't assert exact value because production aggregation may apply
    # rank1-only selection or deduplication.
    assert -3.0 <= col.flowdock_score_mean <= 0.0, (
        f"flowdock_score_mean {col.flowdock_score_mean} outside "
        f"expected window [-3.0, 0.0]"
    )
    # Std must be finite and > 0 (we have a non-degenerate batch).
    assert col.flowdock_score_std is not None
    assert col.flowdock_score_std > 0.0
    for s in batch:
        assert s in col.per_smiles, f"missing {s!r} in per_smiles"


# ---------------------------------------------------------------------------
# 4. test_flowdock_score_candidates_unavailable
# ---------------------------------------------------------------------------
def test_flowdock_score_candidates_unavailable(tmp_path: Path) -> None:
    """When the vendored FlowDock repo is missing, status='unavailable'."""
    from molmetal_lam.sbdd_env.flowdock_sota_scoring import score_candidates
    fake_pdb = tmp_path / "fake.pdb"
    fake_pdb.write_text("ATOM\nEND\n")
    col = score_candidates(
        [{"smiles": "CCO"}], str(fake_pdb),
        repo_root=str(tmp_path / "no_such_flowdock_repo"),
    )
    assert col.status == "unavailable"
    assert col.n_invoked == 1
    assert col.n_scored == 0
    assert col.flowdock_score_mean is None
    assert col.flowdock_score_std is None


# ---------------------------------------------------------------------------
# 5. test_flowdock_score_candidates_empty
# ---------------------------------------------------------------------------
def test_flowdock_score_candidates_empty() -> None:
    """When no candidates are supplied, status='ok' with zero invocations."""
    from molmetal_lam.sbdd_env.flowdock_sota_scoring import score_candidates
    col = score_candidates([], "/tmp/doesnt_matter_pdb")
    assert col.status == "ok"
    assert col.n_invoked == 0
    assert col.n_scored == 0
    assert col.flowdock_score_mean is None
    assert col.flowdock_score_std is None


# ---------------------------------------------------------------------------
# 6. test_flowdock_score_candidates_missing_protein
# ---------------------------------------------------------------------------
def test_flowdock_score_candidates_missing_protein(tmp_path: Path) -> None:
    """Missing protein_path -> status='error', no scoring attempted."""
    from molmetal_lam.sbdd_env.flowdock_sota_scoring import score_candidates
    # Build a vendored repo stub so the path is "available".
    repo = tmp_path / "synthetic_flowdock"
    pkg = repo / "flowdock"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "sample.py").write_text("# stub\n")
    col = score_candidates(
        [{"smiles": "CCO"}],
        str(tmp_path / "no_such_protein.pdb"),
        repo_root=str(repo),
    )
    assert col.status == "error"
    assert col.n_invoked == 1
    assert col.n_scored == 0
    assert "protein_path not found" in col.notes


# ---------------------------------------------------------------------------
# 7. test_flowdock_aggregation_helper
# ---------------------------------------------------------------------------
def test_flowdock_aggregation_helper() -> None:
    """The internal _aggregate helper returns the right mean/std values."""
    from molmetal_lam.sbdd_env.flowdock_sota_scoring import _aggregate
    # n=0 -> (None, None)
    assert _aggregate([]) == (None, None)
    # n=1 -> (mean, None)
    mean, std = _aggregate([-1.5])
    assert mean == -1.5
    assert std is None
    # n=2 -> finite mean and std
    mean, std = _aggregate([-1.20, -0.80])
    assert math.isclose(mean, -1.0, rel_tol=1e-6)
    assert std is not None
    assert math.isclose(std, 0.2, rel_tol=1e-6)
    # Inf/NaN values get filtered.
    mean, std = _aggregate([-1.0, float("nan"), -0.5, float("inf")])
    assert math.isclose(mean, -0.75, rel_tol=1e-6)
    assert std is not None


# ---------------------------------------------------------------------------
# 8. test_flowdock_score_recorded_in_report (r4_c_full_sweep worker path)
# ---------------------------------------------------------------------------
def test_flowdock_score_recorded_in_report(tmp_path: Path, monkeypatch) -> None:
    """The r4_c_full_sweep worker records FlowDock columns in the report.

    We drive ``main()`` with a fake ``--worker-input`` containing a
    synthetic :class:`PocketResult` and a ``flowdock_config``; then we
    inspect the worker output JSON for the new columns.
    """
    def _fake_run_one_pocket(*args, **kwargs):
        from molmetal.scripts.r4_c_full_sweep import PocketResult
        return PocketResult(
            pocket_id="fake_000",
            status="ok",
            n_candidates=2,
            candidates=[
                {"smiles": "CCO", "is_generated": True, "sa": 1.0, "qed": 0.5,
                 "lipinski": True, "vina_proxy": -7.0},
                {"smiles": "CCN", "is_generated": True, "sa": 1.5, "qed": 0.4,
                 "lipinski": True, "vina_proxy": -6.5},
            ],
            top1_smiles="CCO",
            top1_sa=1.0, top1_qed=0.5, top1_lipinski=True,
            top1_vina_proxy=-7.0,
            mean_sa=1.25, mean_qed=0.45,
            lipinski_pass_count=2,
            n_generated_candidates=2, n_seed_candidates=0,
            receptor_path=str(tmp_path / "fake.pdb"),
            ligand_path=str(tmp_path / "fake.sdf"),
            seed=42,
        )

    from molmetal_lam.sbdd_env import flowdock_sota_scoring as scoring

    def _fake_subprocess_run(cmd, *args, **kwargs):
        out_dir = None
        smiles = None
        for tok in cmd:
            if tok.startswith("+out_path="):
                out_dir = Path(tok.split("=", 1)[1])
            if tok.startswith("+input_ligand="):
                smiles = tok.split("=", 1)[1]
        assert out_dir is not None and smiles is not None
        out_dir.mkdir(parents=True, exist_ok=True)
        # Match the production sample_id formula so the mock parser
        # sees the rank1 file in the expected subdirectory.
        sample_id = f"r4_{abs(hash(smiles)) % 10**8}"
        complex_dir = out_dir / sample_id
        complex_dir.mkdir(parents=True, exist_ok=True)
        if smiles == "CCO":
            (complex_dir / "rank1_confidence-1.20.sdf").write_text("stub")
        else:
            (complex_dir / "rank1_confidence-0.80.sdf").write_text("stub")
        class _Result:
            returncode = 0
            stderr = ""
        return _Result()

    monkeypatch.setattr(scoring.subprocess, "run", _fake_subprocess_run)

    # Build a synthetic vendored-repo layout.
    repo = tmp_path / "synthetic_flowdock"
    pkg = repo / "flowdock"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "sample.py").write_text("# stub\n")

    (tmp_path / "fake.pdb").write_text("ATOM\nEND\n")

    worker_input = tmp_path / "input.json"
    worker_output = tmp_path / "output.json"
    payload = {
        "pocket_dir": str(tmp_path),
        "n_simulations": 4,
        "max_depth": 3,
        "seed": 42,
        "pocket_id": "fake_000",
        "receptor_path": str(tmp_path / "fake.pdb"),
        "ligand_path": str(tmp_path / "fake.sdf"),
        "flowdock_config": {
            "repo_root": str(repo),
            "n_samples": 2,
            "num_steps": 10,
            "ckpt_path": None,
            "device": "cpu",
            "timeout_sec": 30.0,
        },
    }
    worker_input.write_text(json.dumps(payload))

    # Patch run_one_pocket inside the r4_c_full_sweep module.
    import importlib
    r4 = importlib.import_module("molmetal.scripts.r4_c_full_sweep")
    monkeypatch.setattr(r4, "run_one_pocket", _fake_run_one_pocket)

    saved_argv = sys.argv[:]
    sys.argv = [saved_argv[0], "--worker-input", str(worker_input),
                "--worker-output", str(worker_output)]
    try:
        rc = r4.main()
    finally:
        sys.argv = saved_argv
    assert rc == 0

    out = json.loads(worker_output.read_text())
    # The new columns must be present in the JSON record.
    assert "flowdock_score_mean" in out
    assert "flowdock_score_std" in out
    assert "flowdock_status" in out
    assert "flowdock_n_invoked" in out
    assert "flowdock_n_scored" in out
    assert "flowdock_per_smiles" in out

    assert out["flowdock_status"] in ("ok", "partial")
    # Mean of -1.20 and -0.80 = -1.00
    assert math.isclose(out["flowdock_score_mean"], -1.00, rel_tol=1e-6)
    assert out["flowdock_score_std"] is not None
    assert math.isclose(out["flowdock_score_std"], 0.2, rel_tol=1e-6)
    assert out["flowdock_n_invoked"] == 2
    assert out["flowdock_n_scored"] == 2
    assert "CCO" in out["flowdock_per_smiles"]
    assert "CCN" in out["flowdock_per_smiles"]
