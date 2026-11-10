"""Wf-262 / TODO-208 — unit tests for ``per_wave_checklist.py``.

Covers the §9 per-wave sub-checklist contract documented in
``TODO/completed/207_unified_pre_flight_checklist_v2.md``:

  1. CLI: argparse recognises all 4 contract flags
     (``--wave``, ``--config``, ``--preflight``, ``--out``).
  2. Backward-compat: ``--out`` defaults to
     ``molmetal/reports/wf262_checklist/per_wave.json``;
     ``--preflight`` defaults to ``wf259_run/pre_flight.json``.
  3. JSON schema: per-wave JSON has all required top-level keys
     (``schema_version``, ``wave``, ``generated_at``, ``preflight_pass_count``,
     ``preflight_fail_count``, ``p0_blocker_gates``, ``p1_verify_gates``,
     ``p2_documented_gates``, ``config_summary``, ``wave_budget_hr``,
     ``wave_remaining_hr``, ``abort_criteria_active``, ``per_pair_sub_checklist``)
     with correct types and the per-pair rows carry ``metal``, ``seed``,
     ``expected_ckpt`` contract fields.
  4. Markdown emitted: human-readable Markdown is written next to the JSON.
  5. CLI runs end-to-end against the real ``wf259_run/pre_flight.json``,
     produces a valid JSON with ``schema_valid=True``, and parses to a
     ``per_pair_sub_checklist`` row for Ru/seed=0 with ``expected_auc``
     extracted from the on-disk ``per_seed_metrics_seed0.json``.
  6. yaml schema check: when ``--config`` points at the real
     ``cfm_full_retrain.yaml`` (``data``, ``model``, ``flow``, ``train``,
     ``protocol``, ``eval``, ``output``, ``compute``, ``honest_framing``
     all present), ``parse_ok`` is True.
  7. abort-criteria detection: a fully-pending (no ckpt, no metrics) wave
     triggers ``7.2.2_failure_pattern_3consec`` for any 3 consecutive
     seeds on the same metal.
  8. validate schema: a stub object that omits a required key is flagged
     ``schema_valid=False``.

All tests are CPU-only; they construct ephemeral JSON under ``tmp_path``
so the real ``molmetal/reports/wf259_run/pre_flight.json`` and the real
``molmetal/checkpoints/`` tree are NOT modified.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Make ``per_wave_checklist`` importable as a module (no torch needed).
SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

import per_wave_checklist  # noqa: E402

VENV_PY = sys.executable

# Required top-level contract fields per 207 v2 §9.1 (per-wave JSON).
REQUIRED_TOP_KEYS = (
    "schema_version",
    "wave",
    "generated_at",
    "preflight_pass_count",
    "preflight_fail_count",
    "p0_blocker_gates",
    "p1_verify_gates",
    "p2_documented_gates",
    "config_summary",
    "wave_budget_hr",
    "wave_remaining_hr",
    "abort_criteria_active",
    "per_pair_sub_checklist",
)

REQUIRED_TOP_TYPES: dict = {
    "schema_version": str,
    "wave": str,
    "generated_at": str,
    "preflight_pass_count": int,
    "preflight_fail_count": int,
    "p0_blocker_gates": list,
    "p1_verify_gates": list,
    "p2_documented_gates": list,
    "config_summary": dict,
    "wave_budget_hr": float,
    "wave_remaining_hr": float,
    "abort_criteria_active": list,
    "per_pair_sub_checklist": list,
}

REQUIRED_PER_PAIR_KEYS = ("metal", "seed", "expected_ckpt")
REQUIRED_GATE_KEYS = ("gate_name", "status", "evidence")


# ---------------------------------------------------------------------------
# Test 1 — CLI exposes the 4 contract flags (G2 ≥ 4)
# ---------------------------------------------------------------------------


def test_cli_exposes_contract_flags():
    """The 4 contract flags (--wave, --config, --preflight, --out) must
    all be present in the argparse parser; bonus flags above the G2
    contract minimum are also recognised."""
    ns = per_wave_checklist.parse_argv(
        [
            "--wave",
            "Wave-1 chemprop v2 retrain",
            "--config",
            "/tmp/foo.yaml",
            "--preflight",
            "/tmp/preflight.json",
            "--out",
            "/tmp/wave1.json",
        ]
    )
    assert ns.wave == "Wave-1 chemprop v2 retrain"
    assert str(ns.config) == "/tmp/foo.yaml"
    assert str(ns.preflight) == "/tmp/preflight.json"
    assert str(ns.out) == "/tmp/wave1.json"
    # Bonus flags beyond the contract minimum (G2 ≥ 4):
    raw = per_wave_checklist.parse_argv(["--wave", "Wave-1 chemprop v2 retrain"])
    assert hasattr(raw, "out_md")
    assert hasattr(raw, "pairs_metals")
    assert hasattr(raw, "pairs_seeds")
    assert hasattr(raw, "time_budget_hr")
    assert hasattr(raw, "no_md")
    assert hasattr(raw, "verbose")
    # And the parser exposes NO unrecognised defaults:
    assert isinstance(ns.wave, str)
    assert ns.no_md is False
    assert ns.verbose is False


# ---------------------------------------------------------------------------
# Test 2 — Defaults match the wf262_checklist / wf259_run contract paths
# ---------------------------------------------------------------------------


def test_default_paths_match_contract():
    """``--preflight`` default = ``wf259_run/pre_flight.json``;
    ``--out`` default = ``wf262_checklist/per_wave.json``.
    See 207 v2 §10.3."""
    ns = per_wave_checklist.parse_argv(
        ["--wave", "Wave-test"],
    )
    assert ns.preflight == per_wave_checklist.DEFAULT_PREFLIGHT_JSON
    assert ns.out == per_wave_checklist.DEFAULT_OUT_JSON
    assert "wf259_run" in str(ns.preflight)
    assert "wf262_checklist" in str(ns.out)
    # out_md also defaults to wf262_checklist
    assert "wf262_checklist" in str(ns.out_md)
    # cross-check expected DEFAULT_PATH anchors are absolute
    for p in (ns.preflight, ns.out, ns.out_md):
        assert Path(str(p)).is_absolute()


# ---------------------------------------------------------------------------
# Test 3 — JSON schema (top-level keys + per-pair/gate sub-shapes)
# ---------------------------------------------------------------------------


def test_build_emits_valid_schema(tmp_path: Path):
    """Build the per-wave JSON against a small synthetic preflight +
    dummy config and verify the schema."""
    # Synthetic pre-flight JSON
    pf = {
        "timestamp": "2026-09-28T12:00:00+0000",
        "schema_version": "207.v1",
        "pipeline": "all",
        "summary": {"pass": 3, "fail": 1, "skip": 0, "total": 4, "total_wall_sec": 1.0},
        "gates": [
            {"name": "p0_a", "priority": "P0", "status": "PASS", "evidence": "ok"},
            {"name": "p1_a", "priority": "P1", "status": "PASS", "evidence": "ok"},
            {"name": "p1_b", "priority": "P1", "status": "FAIL", "evidence": "boom"},
            {"name": "p2_a", "priority": "P2", "status": "PASS", "evidence": "ok"},
        ],
    }
    pf_path = tmp_path / "preflight.json"
    pf_path.write_text(json.dumps(pf), encoding="utf-8")

    # Synthetic yaml (must contain the REQUIRED_CONFIG_KEYS contract)
    yaml_text = (
        "data: {csv: foo.csv}\n"
        "model: {backbone: x}\n"
        "flow: {n_steps: 100}\n"
        "train: {lr: 1e-4}\n"
        "protocol: {epochs: 50}\n"
        "eval: {decode_smoke_every: 200}\n"
        "output: {dir: out/}\n"
        "compute: {gpu: cuda:0}\n"
        "honest_framing: MEASURED-vs-PROJECTED\n"
    )
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")

    out_json = tmp_path / "out.json"
    out_md = tmp_path / "out.md"
    obj = per_wave_checklist.build_per_wave_checklist(
        wave="Wave-test",
        config_path=yaml_path,
        preflight_path=pf_path,
        pairs_metals=["Ru", "Ir"],
        pairs_seeds=[0, 1],
        time_budget_hr=10.0,
        out_json=out_json,
        out_md=out_md,
    )

    # Top-level required keys + types
    for k in REQUIRED_TOP_KEYS:
        assert k in obj, f"missing top-level key {k!r}"
        assert isinstance(obj[k], REQUIRED_TOP_TYPES[k]), (
            f"key {k!r}: expected {REQUIRED_TOP_TYPES[k].__name__}, "
            f"got {type(obj[k]).__name__}"
        )
    # Counts populated from summary
    assert obj["preflight_pass_count"] == 3
    assert obj["preflight_fail_count"] == 1
    # p0/p1/p2 partition matches the synthetic input
    assert len(obj["p0_blocker_gates"]) == 1
    assert len(obj["p1_verify_gates"]) == 2
    assert len(obj["p2_documented_gates"]) == 1
    # Each gate entry has the contract sub-fields
    for bucket in ("p0_blocker_gates", "p1_verify_gates", "p2_documented_gates"):
        for g in obj[bucket]:
            for k in REQUIRED_GATE_KEYS:
                assert k in g, f"{bucket} entry missing {k!r}: {g}"
    # Config schema parse OK with all 9 required keys present
    cs = obj["config_summary"]
    assert cs["parse_ok"] is True
    assert cs["n_required_keys_present"] == len(per_wave_checklist.REQUIRED_CONFIG_KEYS)
    assert cs["missing_keys"] == []
    # schema_valid True
    assert obj["schema_valid"] is True
    # per-pair rows
    pp = obj["per_pair_sub_checklist"]
    assert len(pp) == 4  # 2 metals * 2 seeds
    for r in pp:
        for k in REQUIRED_PER_PAIR_KEYS:
            assert k in r, f"per-pair row missing {k!r}: {r}"
        assert isinstance(r["seed"], int)
    # Markdown + JSON files exist
    assert out_json.exists()
    assert out_md.exists()


# ---------------------------------------------------------------------------
# Test 4 — Markdown emitted is human-readable and non-empty
# ---------------------------------------------------------------------------


def test_markdown_contains_summary_and_pairs(tmp_path: Path):
    """The Markdown file must include the wave name, the per-pair
    sub-checklist table headers, and the abort-criteria section."""
    pf = {
        "summary": {"pass": 2, "fail": 0, "skip": 0, "total": 2},
        "gates": [
            {"name": "g1", "priority": "P1", "status": "PASS", "evidence": "ok"},
            {"name": "g2", "priority": "P2", "status": "PASS", "evidence": "ok"},
        ],
    }
    pf_path = tmp_path / "preflight.json"
    pf_path.write_text(json.dumps(pf), encoding="utf-8")
    yaml_text = (
        "data: {}\nmodel: {}\nflow: {}\ntrain: {}\nprotocol: {}\n"
        "eval: {}\noutput: {}\ncompute: {}\nhonest_framing: 'ok'\n"
    )
    yaml_path = tmp_path / "c.yaml"
    yaml_path.write_text(yaml_text, encoding="utf-8")
    out_json = tmp_path / "o.json"
    out_md = tmp_path / "o.md"
    per_wave_checklist.build_per_wave_checklist(
        wave="Wave-XYZ sentinel string",
        config_path=yaml_path,
        preflight_path=pf_path,
        pairs_metals=["Ru"],
        pairs_seeds=[0, 1],
        time_budget_hr=10.0,
        out_json=out_json,
        out_md=out_md,
    )
    md_text = out_md.read_text(encoding="utf-8")
    # Wave name appears in header
    assert "Wave-XYZ sentinel string" in md_text
    # Required markdown sections
    assert "## Summary" in md_text
    assert "## P1 VERIFY gates" in md_text
    assert "## P2 DOCUMENTED gates" in md_text
    assert "## Config summary" in md_text
    assert "## Per-pair sub-checklist" in md_text
    # Per-pair table header
    assert "| Metal | Seed |" in md_text
    assert "| Ru | 0 |" in md_text
    assert "| Ru | 1 |" in md_text
    # Honest framing footer
    assert "Honest framing" in md_text
    assert len(md_text) > 300


# ---------------------------------------------------------------------------
# Test 5 — End-to-end CLI invocation with the real preflight + Ru/seed=0
# ---------------------------------------------------------------------------


def test_e2e_against_real_preflight_and_ru_seed0(tmp_path: Path):
    """Run the script end-to-end against the real
    ``molmetal/reports/wf259_run/pre_flight.json`` + the real
    ``configs/cfm_full_retrain.yaml``. Verify the emitted JSON parses,
    schema-validates, and includes a ``Ru/seed=0`` row with AUC
    extracted from the on-disk ``per_seed_metrics_seed0.json``."""
    real_pf = per_wave_checklist.REPORTS_DIR / "wf259_run" / "pre_flight.json"
    real_cfg = per_wave_checklist.DEFAULT_CONFIG_YAML
    if not real_pf.exists():
        pytest.skip(f"real preflight not present at {real_pf}")
    if not real_cfg.exists():
        pytest.skip(f"real config not present at {real_cfg}")
    out_json = tmp_path / "e2e.json"
    out_md = tmp_path / "e2e.md"
    rc = per_wave_checklist.main(
        [
            "--wave",
            "Wave-1 chemprop v2 retrain (e2e)",
            "--preflight",
            str(real_pf),
            "--config",
            str(real_cfg),
            "--out",
            str(out_json),
            "--out-md",
            str(out_md),
            "--pairs-metals",
            "Ru,Ir",
            "--pairs-seeds",
            "0",
        ]
    )
    assert rc == 0, f"main() returned {rc}"
    obj = json.loads(out_json.read_text(encoding="utf-8"))
    assert obj["schema_valid"] is True
    # Ru/seed=0 row should have a real expected_auc (per Wf-124 P3 ship)
    ru0 = next(
        (
            r
            for r in obj["per_pair_sub_checklist"]
            if r["metal"] == "Ru" and r["seed"] == 0
        ),
        None,
    )
    assert ru0 is not None
    assert (
        ru0["expected_auc"] is not None
    ), "expected_auc should be populated from on-disk per_seed_metrics_seed0.json"
    assert 0.5 < float(ru0["expected_auc"]) < 1.0
    # Markdown exists too
    assert out_md.exists()
    md = out_md.read_text(encoding="utf-8")
    assert "Wave-1 chemprop v2 retrain (e2e)" in md


# ---------------------------------------------------------------------------
# Test 6 — yaml schema check (parse_ok False when keys missing)
# ---------------------------------------------------------------------------


def test_yaml_schema_missing_keys(tmp_path: Path):
    """A yaml WITHOUT one of the required keys (e.g. ``flow``) reports
    ``parse_ok=False`` with the missing key listed."""
    pf = {
        "summary": {"pass": 1, "fail": 0, "skip": 0, "total": 1},
        "gates": [{"name": "g", "priority": "P1", "status": "PASS", "evidence": "ok"}],
    }
    (tmp_path / "pf.json").write_text(json.dumps(pf), encoding="utf-8")
    yaml_text = (
        "data: {csv: foo.csv}\n"
        "model: {backbone: x}\n"
        # NOTE: 'flow' missing — should be flagged
        "train: {lr: 1e-4}\n"
        "protocol: {epochs: 50}\n"
        "eval: {decode_smoke_every: 200}\n"
        "output: {dir: out/}\n"
        "compute: {gpu: cuda:0}\n"
        "honest_framing: 'ok'\n"
    )
    (tmp_path / "bad.yaml").write_text(yaml_text, encoding="utf-8")
    out_json = tmp_path / "out.json"
    out_md = tmp_path / "out.md"
    obj = per_wave_checklist.build_per_wave_checklist(
        wave="Wave-bad-yaml",
        config_path=tmp_path / "bad.yaml",
        preflight_path=tmp_path / "pf.json",
        pairs_metals=["Ru"],
        pairs_seeds=[0],
        time_budget_hr=10.0,
        out_json=out_json,
        out_md=out_md,
    )
    cs = obj["config_summary"]
    assert cs["parse_ok"] is False
    assert "flow" in cs["missing_keys"]
    assert (
        cs["n_required_keys_present"]
        == len(
            per_wave_checklist.REQUIRED_CONFIG_KEYS,
        )
        - 1
    )


# ---------------------------------------------------------------------------
# Test 7 — abort criteria: 3 consecutive missing seeds on same metal
# ---------------------------------------------------------------------------


def test_abort_criteria_3_consecutive(tmp_path: Path):
    """When 3 consecutive seeds for the same metal have no ckpt + no
    metrics, ``7.2.2_failure_pattern_3consec`` fires."""
    pf = {
        "summary": {"pass": 1, "fail": 0, "skip": 0, "total": 1},
        "gates": [{"name": "g", "priority": "P1", "status": "PASS", "evidence": "ok"}],
    }
    (tmp_path / "pf.json").write_text(json.dumps(pf), encoding="utf-8")
    pf_path = tmp_path / "pf.json"
    yaml_text = (
        "data: {}\nmodel: {}\nflow: {}\ntrain: {}\nprotocol: {}\n"
        "eval: {}\noutput: {}\ncompute: {}\nhonest_framing: 'ok'\n"
    )
    (tmp_path / "c.yaml").write_text(yaml_text, encoding="utf-8")

    # Use a temp checkpoints dir so no real Ru ckpt is found; pair it
    # with a temp per-seed metrics dir that is also empty.
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        fake_ckpt = Path(d) / "checkpoints"
        fake_ckpt.mkdir()
        fake_reports = Path(d) / "reports" / "wf124_p3_multitask"
        # don't create any per-metal subdir -> all pairs will be PENDING
        # but that's expected: we want 7.2.2 to fire.
        # Monkey-patch the constants for this test only.
        saved_ckpt = per_wave_checklist.CHECKPOINTS_DIR
        saved_reports = per_wave_checklist.REPORTS_DIR
        per_wave_checklist.CHECKPOINTS_DIR = fake_ckpt  # type: ignore[attr-defined]
        per_wave_checklist.REPORTS_DIR = fake_reports  # type: ignore[attr-defined]
        # Per-pair metrics helper uses REPORTS_DIR — so we patch both.
        try:
            out_json = tmp_path / "ab.json"
            out_md = tmp_path / "ab.md"
            obj = per_wave_checklist.build_per_wave_checklist(
                wave="Wave-abort",
                config_path=tmp_path / "c.yaml",
                preflight_path=pf_path,
                pairs_metals=["Ru"],
                pairs_seeds=[0, 1, 2, 3],  # 4 missing
                time_budget_hr=10.0,
                out_json=out_json,
                out_md=out_md,
            )
            assert "7.2.2_failure_pattern_3consec" in obj["abort_criteria_active"], (
                f"expected 7.2.2 to fire when 4 consecutive Ru seeds have no "
                f"ckpt + no metrics; got {obj['abort_criteria_active']}"
            )
        finally:
            per_wave_checklist.CHECKPOINTS_DIR = saved_ckpt  # type: ignore[attr-defined]
            per_wave_checklist.REPORTS_DIR = saved_reports  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Test 8 — validate_schema flags missing keys
# ---------------------------------------------------------------------------


def test_validate_schema_strict():
    """The internal ``_validate_schema`` helper must reject stub objects
    missing any required top-level field."""
    base = dict.fromkeys(REQUIRED_TOP_KEYS, None)
    base["schema_version"] = "262.v1"
    base["wave"] = "x"
    base["generated_at"] = "2026-09-28T00:00:00+00:00"
    base["preflight_pass_count"] = 0
    base["preflight_fail_count"] = 0
    base["p0_blocker_gates"] = []
    base["p1_verify_gates"] = []
    base["p2_documented_gates"] = []
    base["config_summary"] = {
        "yaml_path": "x",
        "parse_ok": True,
        "n_required_keys": 0,
        "missing_keys": [],
        "n_required_keys_present": 0,
        "error": "",
    }
    base["wave_budget_hr"] = 10.0
    base["wave_remaining_hr"] = 10.0
    base["abort_criteria_active"] = []
    base["per_pair_sub_checklist"] = []
    assert per_wave_checklist._validate_schema(base) is True
    # Now drop one required key
    broken = dict(base)
    del broken["per_pair_sub_checklist"]
    assert per_wave_checklist._validate_schema(broken) is False


# ---------------------------------------------------------------------------
# Test 9 — Subprocess invocation: --help + a real run
# ---------------------------------------------------------------------------


def test_subprocess_via_venv(tmp_path: Path):
    """The project-venv interpreter must be able to run --help and a real
    checklist run; covered for backward-compat smoke."""
    if not Path(VENV_PY).exists():
        pytest.skip(f"project-venv interpreter not present at {VENV_PY}")
    # --help (rc=0)
    rc = subprocess.run(
        [VENV_PY, str(per_wave_checklist.__file__), "--help"],
        capture_output=True,
        text=True,
        timeout=30,
    ).returncode
    assert rc == 0, f"--help failed: {rc}"
    # Real checklist run against the pre-flight + cfm_full_retrain.yaml
    real_pf = per_wave_checklist.REPORTS_DIR / "wf259_run" / "pre_flight.json"
    real_cfg = per_wave_checklist.DEFAULT_CONFIG_YAML
    if not (real_pf.exists() and real_cfg.exists()):
        pytest.skip("real preflight/config not present; skipping subproc run")
    out_json = tmp_path / "subproc_out.json"
    out_md = tmp_path / "subproc_out.md"
    rc2 = subprocess.run(
        [
            VENV_PY,
            str(per_wave_checklist.__file__),
            "--wave",
            "Wave-subproc",
            "--preflight",
            str(real_pf),
            "--config",
            str(real_cfg),
            "--out",
            str(out_json),
            "--out-md",
            str(out_md),
            "--pairs-metals",
            "Ru",
            "--pairs-seeds",
            "0,1",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    ).returncode
    assert rc2 == 0, f"real run failed: rc={rc2} stderr={rc2 and ''}"
    assert out_json.exists()
    out = json.loads(out_json.read_text(encoding="utf-8"))
    assert out["schema_valid"] is True
    assert "Wave-subproc" in out_md.read_text(encoding="utf-8")
