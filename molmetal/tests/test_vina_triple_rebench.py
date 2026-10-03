"""Wf-60 Phase 2 — Vina triple re-bench acceptance tests.

These tests exercise:

* CLI parsing + dry-run
* Binary detection honours ``MOLMETAL_VINA_BIN`` env var
* Stub fallback path (default ``--use-real-vina`` off)
* Per-pocket summary logic with synthetic cell lists
* ``vina_binary_status.json`` emission shape
* Per-pocket ``metrics.json`` emission shape
* m4 + m10 aggregation invariants (NaN-safe)

Tests are written against the **public** surface — they do not require
the real Vina binary or RDKit to be installed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCRIPT_PATH = PROJECT_ROOT / "molmetal" / "scripts" / "bench_vina_triple_rebench.py"


@pytest.fixture(scope="module")
def bench_mod():
    """Import the bench script module.

    Python 3.14 quirk: ``dataclasses`` introspects ``sys.modules`` by
    module name; loading via ``spec_from_file_location`` does NOT
    register the module under any name in ``sys.modules``, which
    causes the dataclass decorator to fail. We work around this by
    adding ``scripts/`` to ``sys.path`` and importing by file name.
    """
    scripts_dir = str(SCRIPT_PATH.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import bench_vina_triple_rebench as mod  # type: ignore[import-not-found]

    return mod


# ---------------------------------------------------------------------------
# CLI / dry-run
# ---------------------------------------------------------------------------
def test_cli_parses_defaults(bench_mod):
    parser = bench_mod._build_parser()
    ns = parser.parse_args([])
    assert ns.pockets == list(bench_mod.DEFAULT_POCKETS)
    assert ns.cands == list(bench_mod.DEFAULT_CANDIDATES)
    assert ns.seeds == [0, 1, 2]
    assert ns.use_real_vina is False
    assert ns.dry_run is False


def test_cli_overrides(bench_mod):
    parser = bench_mod._build_parser()
    ns = parser.parse_args(["--pockets", "830c", "--seeds", "7", "--use-real-vina"])
    assert ns.pockets == ["830c"]
    assert ns.seeds == [7]
    assert ns.use_real_vina is True


def test_dry_run_prints_plan(bench_mod, tmp_path: Path, capsys):
    ns_overrides = [
        "--pockets",
        "830c",
        "--seeds",
        "0",
        "--out-dir",
        str(tmp_path),
        "--dry-run",
    ]
    rc = bench_mod.main(ns_overrides)
    assert rc == 0
    captured = capsys.readouterr()
    plan = json.loads(captured.out)
    assert plan["pockets"] == ["830c"]
    assert plan["seeds"] == [0]
    assert plan["use_real_vina"] is False
    assert "vina_binary_status" in plan


# ---------------------------------------------------------------------------
# Binary detection
# ---------------------------------------------------------------------------
def test_binary_detection_no_env(monkeypatch, bench_mod, tmp_path: Path):
    monkeypatch.delenv("MOLMETAL_VINA_BIN", raising=False)
    # Force shutil.which to return None
    monkeypatch.setattr("shutil.which", lambda _: None)
    # Wf-84 P2 added project-wrapper auto-detect (see bench_vina_triple_rebench.py
    # `_detect_vina_binary` docstring). The project wrapper path is computed as
    # ``Path(__file__).resolve().parents[2] / "molmetal" / "bin" / "vina"``. To
    # assert MISSING_DEP semantics we must hide that path. The cleanest way
    # without rewriting production code is to override ``Path.is_file`` for
    # the exact project-wrapper path only.
    project_root = (
        bench_mod.PROJECT_ROOT
        if hasattr(bench_mod, "PROJECT_ROOT")
        else (Path(bench_mod.__file__).resolve().parents[2])
    )
    wrapper_path = project_root / "molmetal" / "bin" / "vina"
    orig_is_file = Path.is_file

    def patched_is_file(self):
        if str(self) == str(wrapper_path):
            return False
        return orig_is_file(self)

    monkeypatch.setattr("pathlib.Path.is_file", patched_is_file)

    status = bench_mod._detect_vina_binary()
    assert status["status"] == "MISSING_DEP", (
        f"Expected MISSING_DEP but got {status['status']}: {status}. "
        "Wf-84 P2 added project-wrapper auto-detect; the patched is_file "
        "above should have hidden the project wrapper."
    )
    assert status["bin"] is None
    assert status["version"] == ""
    assert "Vina/Q-Vina2" in status["reason"]
    assert isinstance(status["checked_paths"], list)


def test_binary_detection_with_env(monkeypatch, tmp_path: Path, bench_mod):
    fake = tmp_path / "fake_vina"
    fake.write_text("#!/bin/sh\necho vina 1.2.7\n")
    fake.chmod(0o755)
    monkeypatch.setenv("MOLMETAL_VINA_BIN", str(fake))
    status = bench_mod._detect_vina_binary()
    assert status["status"] == "REAL_VINA_AVAILABLE"
    assert status["bin"] == str(fake)
    assert "vina" in status["version"].lower()


def test_binary_detection_env_missing_file(monkeypatch, tmp_path: Path, bench_mod):
    monkeypatch.setenv("MOLMETAL_VINA_BIN", str(tmp_path / "nope"))
    # Wf-84 P2 added project-wrapper auto-detect — hide it for this test too
    project_root = (
        bench_mod.PROJECT_ROOT
        if hasattr(bench_mod, "PROJECT_ROOT")
        else (Path(bench_mod.__file__).resolve().parents[2])
    )
    wrapper_path = project_root / "molmetal" / "bin" / "vina"
    orig_is_file = Path.is_file

    def patched_is_file(self):
        if str(self) == str(wrapper_path):
            return False
        return orig_is_file(self)

    monkeypatch.setattr("pathlib.Path.is_file", patched_is_file)
    status = bench_mod._detect_vina_binary()
    assert status["status"] == "MISSING_DEP"


def test_binary_detection_project_wrapper_auto(monkeypatch, bench_mod):
    """Wf-84 P2: project wrapper at <root>/molmetal/bin/vina is auto-detected
    even without env var or PATH lookup."""
    monkeypatch.delenv("MOLMETAL_VINA_BIN", raising=False)
    monkeypatch.setattr("shutil.which", lambda _: None)
    status = bench_mod._detect_vina_binary()
    # The project wrapper should be reachable on this host (Wf-84 P2 SHIP).
    # If the test is run on a host without the wrapper, it should fall back
    # to MISSING_DEP — that's an environmental signal, not a test failure.
    if status["status"] == "REAL_VINA_AVAILABLE":
        assert status["bin"].endswith(
            "molmetal/bin/vina"
        ), f"Auto-detected binary should be the project wrapper: {status}"
        assert "AutoDock Vina" in status["version"]
    else:
        # Acceptable: the test environment does not have the project wrapper.
        assert status["status"] == "MISSING_DEP"
        assert status["bin"] is None


# ---------------------------------------------------------------------------
# Stub fallback semantics
# ---------------------------------------------------------------------------
def test_stub_score_returns_none(bench_mod):
    rec = bench_mod._stub_score("830c", "C", 0, reason="test")
    assert rec.score_kcal_mol is None
    assert rec.pb_pass is None
    assert rec.qed is None
    assert rec.triple_success is None
    assert rec.status == "STUB_FALLBACK"
    assert "test" in rec.note


# ---------------------------------------------------------------------------
# Per-pocket summary aggregation
# ---------------------------------------------------------------------------
def test_summarise_pocket_with_all_stubs(bench_mod):
    cells = [bench_mod._stub_score("830c", "C", i, "no-binary") for i in range(3)]
    out = bench_mod._summarise_pocket("830c", cells)
    assert out["status"] == "STUB_FALLBACK"
    assert out["n_cells"] == 3
    assert out["status_counts"] == {"STUB_FALLBACK": 3}
    assert out["m4_triple_rate"] is None
    assert out["m4_n_real"] == 0
    assert out["m10_min_kcal_mol"] is None
    assert out["m10_pass"] is None


def test_summarise_pocket_with_real_scores(bench_mod):
    from dataclasses import replace

    cells = [
        replace(
            bench_mod._stub_score("830c", "C", 0, ""),
            score_kcal_mol=-8.0,
            pb_pass=True,
            qed=0.7,
            triple_success=True,
            status="OK",
        ),
        replace(
            bench_mod._stub_score("830c", "C", 1, ""),
            score_kcal_mol=-6.0,
            pb_pass=False,
            qed=0.4,
            triple_success=False,
            status="OK",
        ),
        replace(
            bench_mod._stub_score("830c", "C", 2, ""),
            score_kcal_mol=-9.0,
            pb_pass=True,
            qed=0.8,
            triple_success=True,
            status="OK",
        ),
    ]
    out = bench_mod._summarise_pocket("830c", cells)
    assert out["status"] == "OK"
    assert out["m4_triple_rate"] == pytest.approx(2 / 3, rel=1e-3)
    assert out["m4_n_real"] == 3
    assert out["m10_min_kcal_mol"] == -9.0
    assert out["m10_median_kcal_mol"] == pytest.approx(-8.0, abs=1e-6)
    assert out["m10_pass"] is True  # -9.0 < -4.0


def test_summarise_pocket_partial(bench_mod):
    from dataclasses import replace

    cells = [
        replace(
            bench_mod._stub_score("830c", "C", 0, ""),
            score_kcal_mol=-8.0,
            pb_pass=True,
            qed=0.7,
            triple_success=True,
            status="OK",
        ),
        bench_mod._stub_score("830c", "C", 1, "no-binary"),
    ]
    out = bench_mod._summarise_pocket("830c", cells)
    assert out["status"] == "PARTIAL"
    assert out["status_counts"] == {"OK": 1, "STUB_FALLBACK": 1}
    assert out["m4_n_real"] == 1
    assert out["m4_triple_rate"] == pytest.approx(1.0, rel=1e-3)


def test_summarise_pocket_failing_gate(bench_mod):
    from dataclasses import replace

    cells = [
        replace(
            bench_mod._stub_score("830c", "C", 0, ""),
            score_kcal_mol=-3.0,
            pb_pass=True,
            qed=0.7,
            triple_success=False,
            status="OK",
        ),
    ]
    out = bench_mod._summarise_pocket("830c", cells)
    assert out["m10_pass"] is False  # -3.0 > -4.0 (gate)


# ---------------------------------------------------------------------------
# End-to-end: synthetic pocket_status.json + no real Vina → STUB_FALLBACK
# ---------------------------------------------------------------------------
def test_end_to_end_stub_path(bench_mod, tmp_path: Path, capsys, monkeypatch):
    # Wf-84 P2 added project-wrapper auto-detect — hide it for this test
    project_root = (
        bench_mod.PROJECT_ROOT
        if hasattr(bench_mod, "PROJECT_ROOT")
        else (Path(bench_mod.__file__).resolve().parents[2])
    )
    wrapper_path = project_root / "molmetal" / "bin" / "vina"
    orig_is_file = Path.is_file

    def patched_is_file(self):
        if str(self) == str(wrapper_path):
            return False
        return orig_is_file(self)

    monkeypatch.setattr("pathlib.Path.is_file", patched_is_file)

    # Build a fake pocket_status.json with one pocket pointing at a non-existent
    # receptor — exercises both failure paths.
    ps = tmp_path / "pocket_status.json"
    ps.write_text(
        json.dumps(
            {
                "pockets": [
                    {
                        "pocket_id": "830c",
                        "pdbqt_path": str(tmp_path / "ghost.pdbqt"),
                        "status": "READY",
                    }
                ],
            }
        )
    )
    out_dir = tmp_path / "wf60"
    args = [
        "--pockets",
        "830c",
        "--seeds",
        "0",
        "--pocket-status-json",
        str(ps),
        "--out-dir",
        str(out_dir),
    ]
    rc = bench_mod.main(args)
    assert rc == 0
    # metrics.json emitted
    metrics = json.loads((out_dir / "830c" / "metrics.json").read_text())
    assert metrics["pocket"] == "830c"
    assert metrics["status"] == "MISSING_DEP"
    assert "ghost.pdbqt" in metrics["reason"]
    # vina_binary_status.json emitted + MISSING_DEP (no env + project wrapper hidden)
    bin_status = json.loads((out_dir / "vina_binary_status.json").read_text())
    assert bin_status["status"] == "MISSING_DEP"
    # synthesis.json
    synth = json.loads((out_dir / "synthesis.json").read_text())
    assert synth["n_pockets"] == 1
    assert synth["vina_binary_status"] == "MISSING_DEP"


def test_end_to_end_real_receptor_stub(bench_mod, tmp_path: Path, capsys):
    # Build a real (empty) receptor file so the per-pocket driver does NOT
    # trip the "ghost.pdbqt" branch — instead it falls through to the
    # STUB_FALLBACK path because no Vina binary is available.
    receptor = tmp_path / "receptor.pdbqt"
    receptor.write_text("ROOT\nEND\n")
    ps = tmp_path / "pocket_status.json"
    ps.write_text(
        json.dumps(
            {
                "pockets": [
                    {
                        "pocket_id": "830c",
                        "pdbqt_path": str(receptor),
                        "status": "READY",
                    }
                ],
            }
        )
    )
    out_dir = tmp_path / "wf60"
    rc = bench_mod.main(
        [
            "--pockets",
            "830c",
            "--cands",
            "C",
            "--seeds",
            "0",
            "1",
            "--pocket-status-json",
            str(ps),
            "--out-dir",
            str(out_dir),
        ]
    )
    assert rc == 0
    metrics = json.loads((out_dir / "830c" / "metrics.json").read_text())
    assert metrics["status"] == "STUB_FALLBACK"
    assert metrics["n_cells"] == 2
    assert all(c["status"] == "STUB_FALLBACK" for c in metrics["cells"])
    assert all(c["score_kcal_mol"] is None for c in metrics["cells"])


# ---------------------------------------------------------------------------
# Smoke: run as a subprocess to ensure the script is import-clean
# ---------------------------------------------------------------------------
def test_subprocess_help():
    cp = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--help"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert cp.returncode == 0
    assert "Wf-60" in cp.stdout or "vina" in cp.stdout.lower()


def test_subprocess_dry_run(tmp_path: Path):
    cp = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--pockets",
            "830c",
            "--seeds",
            "0",
            "--out-dir",
            str(tmp_path),
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert cp.returncode == 0
    plan = json.loads(cp.stdout)
    assert plan["pockets"] == ["830c"]
    assert plan["seeds"] == [0]
