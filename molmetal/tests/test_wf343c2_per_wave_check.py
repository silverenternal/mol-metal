"""Wf-343C2 — unit tests for ``wf343c2_per_wave_check.py``.

Covers the per-wave internal pre-retrain check contract:

  1. CLI: argparse recognises all 8 contract flags
     (``--wave``, ``--config``, ``--preflight``, ``--out``,
     ``--pairs-metals``, ``--pairs-seeds``, ``--min-vram-mib``,
     ``--min-disk-gib``).
  2. Backward-compat defaults: ``--preflight`` defaults to
     ``wf259_run/pre_flight.json``; ``--out`` defaults to
     ``wf343c2_per_wave_check/per_wave_check.json``; ``--skip-live`` and
     ``--strict`` defaults are False.
  3. JSON schema: 7-check shape — every check has ``check_id`` / ``status``
     / ``evidence``; ``n_checks == 7``; the union of ``check_id`` values
     exactly matches the 7 expected IDs.
  4. Pass path: synthesise a fresh preflight + full keys YAML +
     ``--skip-live``; all 7 checks should PASS.
  5. Fail path: missing preflight + missing config + missing YuelBond
     adapter → multiple FAILs with evidence strings non-empty.
  6. Schema validator: ``_validate_schema`` rejects a stub missing
     required keys / wrong types / unknown check IDs; accepts a stub
     built by ``run_per_wave_checks`` with --skip-live.

All tests are CPU-only; they construct ephemeral JSON/YAML under
``tmp_path`` so the real ``molmetal/reports/wf259_run/pre_flight.json`` and
the canonical adapters are NOT modified.  Tests that need to monkey-patch
the YuelBond adapter path use ``tmp_path`` symlinks so the real
``molmetal/adapters/flow_matching_lipman`` is preserved.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make ``wf343c2_per_wave_check`` importable as a module (no torch needed).
SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

import wf343c2_per_wave_check as wf343c2  # noqa: E402

VENV_PY = "/home/hugo/codes/mol-metal/.venv/bin/python"

EXPECTED_CHECK_IDS = frozenset(
    {
        "preflight_freshness_24h",
        "preflight_no_fail_gates",
        "hyperparams_config_complete",
        "yuelbond_decoder_registered",
        "wave_budget_remaining_nonneg",
        "gpu_vram_available",
        "disk_free_space",
    }
)

REQUIRED_TOP_KEYS = (
    "schema_version",
    "wave",
    "generated_at",
    "n_checks",
    "n_pass",
    "n_fail",
    "n_skip",
    "overall_status",
    "checks",
    "honest_framing",
    "schema_valid",
    "config_path",
    "preflight_path",
    "n_pairs",
    "budget_hr",
    "skip_live",
    "error",
)

REQUIRED_TOP_TYPES: dict = {
    "schema_version": str,
    "wave": str,
    "generated_at": str,
    "n_checks": int,
    "n_pass": int,
    "n_fail": int,
    "n_skip": int,
    "overall_status": str,
    "checks": list,
    "honest_framing": str,
    "schema_valid": bool,
    "config_path": str,
    "preflight_path": str,
    "n_pairs": int,
    "budget_hr": float,
    "skip_live": bool,
    "error": str,
}


# ---------------------------------------------------------------------------
# Synthetic preflight + YAML builders
# ---------------------------------------------------------------------------


def _write_fresh_preflight(path: Path, *, n_pass: int = 22, n_fail: int = 0) -> None:
    """Write a fresh-looking pre-flight JSON (mtime = now)."""
    obj = {
        "timestamp": "2026-09-29T15:00:00+0000",
        "schema_version": "207.v1",
        "pipeline": "all",
        "summary": {
            "pass": n_pass,
            "fail": n_fail,
            "skip": 0,
            "total": n_pass + n_fail,
            "total_wall_sec": 1.0,
        },
        "gates": [],
    }
    path.write_text(json.dumps(obj), encoding="utf-8")


def _write_stale_preflight(path: Path, *, age_sec: int) -> None:
    """Write a pre-flight JSON whose mtime is `age_sec` seconds old."""
    import time as _time

    _write_fresh_preflight(path)
    old = _time.time() - age_sec
    import os as _os

    _os.utime(str(path), (old, old))


def _write_full_yaml(path: Path) -> None:
    """Write a YAML with all REQUIRED_CONFIG_KEYS."""
    text = (
        "data: {csv: foo.csv}\n"
        "model: {backbone: x}\n"
        "flow: {n_steps: 100}\n"
        "train: {lr: 1e-4}\n"
        "protocol: {epochs: 50, time_budget_hr: 10}\n"
        "eval: {decode_smoke_every: 200}\n"
        "output: {dir: out/}\n"
        "compute: {gpu: cuda:0, yuelbond: true}\n"
        "honest_framing: 'ok'\n"
    )
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Test 1 — CLI exposes contract flags
# ---------------------------------------------------------------------------


def test_cli_exposes_contract_flags():
    """The 8 contract flags + skip-live + strict + verbose + budget-hr +
    max-age-sec must all be present in the argparse parser."""
    ns = wf343c2.parse_argv(
        [
            "--wave",
            "Wave-1 chemprop v2 retrain",
            "--config",
            "/tmp/foo.yaml",
            "--preflight",
            "/tmp/preflight.json",
            "--out",
            "/tmp/wave_check.json",
            "--pairs-metals",
            "Ru,Ir",
            "--pairs-seeds",
            "0,1,2",
            "--min-vram-mib",
            "16384",
            "--min-disk-gib",
            "100",
        ]
    )
    assert ns.wave == "Wave-1 chemprop v2 retrain"
    assert str(ns.config) == "/tmp/foo.yaml"
    assert str(ns.preflight) == "/tmp/preflight.json"
    assert str(ns.out) == "/tmp/wave_check.json"
    assert ns.pairs_metals == "Ru,Ir"
    assert ns.pairs_seeds == "0,1,2"
    assert ns.min_vram_mib == 16384
    assert ns.min_disk_gib == 100
    # Bonus flags
    assert hasattr(ns, "budget_hr")
    assert hasattr(ns, "max_age_sec")
    assert hasattr(ns, "skip_live")
    assert hasattr(ns, "strict")
    assert hasattr(ns, "verbose")
    # Defaults — skip_live + strict + verbose off
    assert ns.skip_live is False
    assert ns.strict is False
    assert ns.verbose is False


# ---------------------------------------------------------------------------
# Test 2 — Default paths + flags match contract
# ---------------------------------------------------------------------------


def test_default_paths_match_contract():
    """``--preflight`` default = ``wf259_run/pre_flight.json``;
    ``--out`` default = ``wf343c2_per_wave_check/per_wave_check.json``;
    budget default = 10.0h; max_age default = 86400 (= 24h)."""
    ns = wf343c2.parse_argv(["--wave", "Wave-test"])
    assert ns.preflight == wf343c2.DEFAULT_PREFLIGHT_JSON
    assert ns.out == wf343c2.DEFAULT_OUT_JSON
    assert "wf259_run" in str(ns.preflight)
    assert "wf343c2_per_wave_check" in str(ns.out)
    assert ns.budget_hr == wf343c2.DEFAULT_BUDGET_HR == 10.0
    assert ns.max_age_sec == wf343c2.DEFAULT_PREFLIGHT_MAX_AGE_SEC == 86400
    assert ns.min_vram_mib == wf343c2.DEFAULT_MIN_VRAM_MIB
    assert ns.min_disk_gib == wf343c2.DEFAULT_MIN_DISK_GIB
    # Anchor guard anchor
    assert Path(str(ns.preflight)).is_absolute()
    assert Path(str(ns.out)).is_absolute()


# ---------------------------------------------------------------------------
# Test 3 — JSON schema: 7 checks + correct shape + correct types
# ---------------------------------------------------------------------------


def test_schema_seven_checks_with_contract_ids(tmp_path: Path):
    """Build a stub via ``run_per_wave_checks`` and verify the schema:
    exactly 7 checks, each with the 3 contract fields, and the union of
    check_id values equals the expected set."""
    pf = tmp_path / "preflight.json"
    _write_fresh_preflight(pf)
    cfg = tmp_path / "config.yaml"
    _write_full_yaml(cfg)
    obj = wf343c2.run_per_wave_checks(
        wave="Wave-schema-test",
        config_path=cfg,
        preflight_path=pf,
        pairs_metals=["Ru"],
        pairs_seeds=[0],
        min_vram_mib=1024,
        min_disk_gib=10,
        budget_hr=10.0,
        max_age_sec=86400,
        skip_live=True,
    )
    # Top-level keys + types
    for k in REQUIRED_TOP_KEYS:
        assert k in obj, f"missing top-level key {k!r}"
        assert isinstance(obj[k], REQUIRED_TOP_TYPES[k]), (
            f"key {k!r}: expected {REQUIRED_TOP_TYPES[k].__name__}, "
            f"got {type(obj[k]).__name__}"
        )
    # Exactly 7 checks
    assert obj["n_checks"] == wf343c2.N_CHECKS == 7
    assert len(obj["checks"]) == 7
    seen = set()
    for c in obj["checks"]:
        assert isinstance(c, dict)
        for sub_k in ("check_id", "status", "evidence"):
            assert sub_k in c, f"check missing {sub_k!r}: {c}"
        assert c["status"] in {"PASS", "FAIL", "SKIP"}
        assert isinstance(c["evidence"], str)
        seen.add(c["check_id"])
    assert seen == EXPECTED_CHECK_IDS
    # Counts sum to 7
    assert obj["n_pass"] + obj["n_fail"] + obj["n_skip"] == 7
    # overall_status enum
    assert obj["overall_status"] in {"PASS", "FAIL", "PASS_WITH_SKIP"}
    # schema_valid True (this stub was built via run_per_wave_checks)
    assert obj["schema_valid"] is True


# ---------------------------------------------------------------------------
# Test 4 — Pass path: fresh preflight + full YAML + skip_live → all PASS
# ---------------------------------------------------------------------------


def test_all_seven_checks_pass_with_skip_live(tmp_path: Path):
    """A clean environment (fresh preflight, full-keys YAML, --skip-live
    so GPU/disk probes are SKIP) should give n_fail=0 + at least 5 PASS."""
    pf = tmp_path / "preflight.json"
    _write_fresh_preflight(pf)
    cfg = tmp_path / "config.yaml"
    _write_full_yaml(cfg)
    obj = wf343c2.run_per_wave_checks(
        wave="Wave-pass",
        config_path=cfg,
        preflight_path=pf,
        pairs_metals=["Ru", "Ir"],
        pairs_seeds=[0, 1],
        min_vram_mib=1024,
        min_disk_gib=10,
        budget_hr=10.0,
        max_age_sec=86400,
        skip_live=True,
    )
    assert obj["n_fail"] == 0
    # Live probes are SKIP under --skip-live
    skip_ids = {c["check_id"] for c in obj["checks"] if c["status"] == "SKIP"}
    assert "gpu_vram_available" in skip_ids
    assert "disk_free_space" in skip_ids
    # Non-live checks should be PASS
    pass_ids = {c["check_id"] for c in obj["checks"] if c["status"] == "PASS"}
    assert "preflight_freshness_24h" in pass_ids
    assert "preflight_no_fail_gates" in pass_ids
    assert "hyperparams_config_complete" in pass_ids
    assert "yuelbond_decoder_registered" in pass_ids
    assert "wave_budget_remaining_nonneg" in pass_ids
    # overall_status reflects the mixed PASS+SKIP
    assert obj["overall_status"] == "PASS_WITH_SKIP"
    # Honest framing cites MEASURED live probes
    assert "MEASURED live probes" in obj["honest_framing"]
    assert obj["schema_valid"] is True


# ---------------------------------------------------------------------------
# Test 5 — Fail path: stale preflight + missing keys + missing YuelBond
# ---------------------------------------------------------------------------


def test_multiple_checks_fail_with_evidence(tmp_path: Path, monkeypatch):
    """A wave with a STALE preflight (age > 24h) + missing flow key +
    YuelBond adapter absent → at least 3 FAILs, each with non-empty
    evidence.  Also tests the budget-remaining check with too many pairs."""
    pf = tmp_path / "preflight.json"
    _write_stale_preflight(pf, age_sec=90000)  # 25h old
    # YAML missing the 'flow' key (and no 'yuelbond' token)
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(
        "data: {}\nmodel: {}\n"
        # 'flow' intentionally missing
        "train: {}\nprotocol: {}\neval: {}\noutput: {}\ncompute: {}\n"
        "honest_framing: 'ok'\n",
        encoding="utf-8",
    )

    # Monkey-patch the canonical YuelBond adapter path to a tmp dir
    # so the script's check_yuelbond_decoder_registered sees it as
    # absent.  This protects the real molmetal/adapters/ tree.
    monkeypatch.setattr(
        wf343c2,
        "MOLMETAL_ROOT",
        tmp_path / "fake_molmetal",
    )

    obj = wf343c2.run_per_wave_checks(
        wave="Wave-fail",
        config_path=bad_yaml,
        preflight_path=pf,
        pairs_metals=["Ru", "Ir", "Os", "Rh", "Re"],  # 5 metals
        pairs_seeds=[0, 1, 2, 3, 4],  # 5 seeds → 25 pairs
        min_vram_mib=1024,
        min_disk_gib=10,
        # 1.0h budget × 25 pairs × 0.7/25 = 0.7h/ea → 25 × 0.028 = 0.7h
        # to fit; we deliberately starve it
        budget_hr=0.5,
        max_age_sec=86400,
        skip_live=True,
    )
    assert obj["n_fail"] >= 3, (
        f"expected ≥ 3 FAILs (preflight stale, missing flow key, "
        f"yuelbond missing, budget overshoot), got {obj['n_fail']} "
        f"with statuses: {[(c['check_id'], c['status']) for c in obj['checks']]}"
    )
    # overall_status must be FAIL
    assert obj["overall_status"] == "FAIL"
    fail_ids = {c["check_id"] for c in obj["checks"] if c["status"] == "FAIL"}
    assert "preflight_freshness_24h" in fail_ids  # stale
    assert "hyperparams_config_complete" in fail_ids  # missing flow
    assert "yuelbond_decoder_registered" in fail_ids  # adapter absent
    assert "wave_budget_remaining_nonneg" in fail_ids  # 25 pairs × 0.014h > 0.5h
    # Every FAIL has a non-empty evidence string
    for c in obj["checks"]:
        if c["status"] == "FAIL":
            assert c["evidence"], f"FAIL {c['check_id']} has empty evidence"
            assert len(c["evidence"]) > 5


# ---------------------------------------------------------------------------
# Test 6 — Schema validator: rejects stubs / accepts run_per_wave_checks
# ---------------------------------------------------------------------------


def test_validate_schema_rejects_stubs_and_accepts_real(tmp_path: Path):
    """The internal ``_validate_schema`` must reject stub objects missing
    any required top-level field or with the wrong checks[] shape, and
    must accept a fully-built object from ``run_per_wave_checks``."""
    # Build a real obj first (acceptance anchor).
    pf = tmp_path / "preflight.json"
    _write_fresh_preflight(pf)
    cfg = tmp_path / "config.yaml"
    _write_full_yaml(cfg)
    real_obj = wf343c2.run_per_wave_checks(
        wave="Wave-validate",
        config_path=cfg,
        preflight_path=pf,
        pairs_metals=["Ru"],
        pairs_seeds=[0],
        min_vram_mib=1024,
        min_disk_gib=10,
        budget_hr=10.0,
        max_age_sec=86400,
        skip_live=True,
    )
    assert wf343c2._validate_schema(real_obj) is True

    # Now stub: drop 'checks' → invalid
    stub = dict(real_obj)
    del stub["checks"]
    assert wf343c2._validate_schema(stub) is False

    # Stub: wrong type for n_checks
    stub2 = dict(real_obj)
    stub2["n_checks"] = "7"  # str instead of int
    assert wf343c2._validate_schema(stub2) is False

    # Stub: wrong number of checks
    stub3 = dict(real_obj)
    stub3["checks"] = real_obj["checks"][:3]  # only 3 of 7
    assert wf343c2._validate_schema(stub3) is False

    # Stub: unknown check_id
    stub4 = dict(real_obj)
    stub4["checks"] = list(real_obj["checks"])
    stub4["checks"][0] = dict(stub4["checks"][0])
    stub4["checks"][0]["check_id"] = "unknown_check_id"
    assert wf343c2._validate_schema(stub4) is False

    # Stub: missing check_id field
    stub5 = dict(real_obj)
    stub5["checks"] = [{"status": "PASS", "evidence": "ok"} for _ in range(7)]
    assert wf343c2._validate_schema(stub5) is False

    # Stub: illegal status
    stub6 = dict(real_obj)
    stub6["checks"] = list(real_obj["checks"])
    stub6["checks"][0] = dict(stub6["checks"][0])
    stub6["checks"][0]["status"] = "MAYBE"
    assert wf343c2._validate_schema(stub6) is False
