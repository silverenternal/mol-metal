"""Wf-A13 — hyperparam resolution allow-list tests (HOLE 7).

================================================================
Wf-A13 (2026-10-03) split the Wf-A12 hyperparam RESOLUTION report
into buckets so the strict switch (``PREFLIGHT_HYPERPARAMS_
RESOLUTION_STRICT`` / ``HYPERPARAMS_RESOLUTION_STRICT_DEFAULT``) has
a reachable ON state:

  * loader-exempt — the canonical wfA2 loader already documents the
    skip as by-design (its own predicate is reused);
  * reader-exempt — a named non-trainer consumer reads the key
    (scripts/start.sh; see HP_RESOLUTION_EXEMPT_KEYS);
  * spec-exempt   — spec / provenance namespace with no executable
    consumer (HP_RESOLUTION_EXEMPT_PREFIXES, grep-verified);
  * RESIDUAL      — no argparse dest on the entry trainer and no
    documented non-trainer consumer.  The ONLY bucket the strict
    switch fails on (plus REQUIRED-HYPERPARAM UNRESOLVED).

These tests are the regression surface for that split:

  1. test_classify_buckets_unit — the classifier's bucket per key,
     including the never-exempt override and the dot-boundary rule.
  2. test_gate_strict_passes_on_documented_exempt_only — the NEW
     reachable ON state: strict mode PASSes a config whose only
     unresolved keys are documented-exempt (it FAILed pre-Wf-A13).
  3. test_gate_strict_fails_on_residual — PROOF the switch can
     still FAIL: one residual key makes strict mode FAIL and the
     evidence names it.
  4. test_gate_default_mode_reports_buckets_on_live_configs — the
     healthy tree still PASSes (non-blocking) with the split
     report in the evidence.

Honest framing: fixtures are synthetic YAMLs in tmp_path plus the
live ``configs/`` tree; no network, no GPU, no torch import.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "molmetal" / "scripts"
PREFLIGHT_SCRIPT = SCRIPTS_DIR / "pre_flight_check.py"
CONFIGS_DIR = REPO_ROOT / "configs"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

STRICT_ENV = "PREFLIGHT_HYPERPARAMS_RESOLUTION_STRICT"


def _load_preflight_module():
    spec = importlib.util.spec_from_file_location(
        "pre_flight_check_wfa13_under_test", PREFLIGHT_SCRIPT,
    )
    assert spec is not None and spec.loader is not None, (
        f"pre_flight_check.py has no import spec ({PREFLIGHT_SCRIPT})"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pre_flight_check_wfa13_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def preflight():
    return _load_preflight_module()


def _write_yaml(path: Path, body: dict) -> Path:
    import yaml as _yaml
    path.write_text(_yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return path


def _base_body(**extra) -> dict:
    """Minimal body: 4 required hyperparams (via D-MPNN aliases) + entry.

    ``model.hidden`` / ``model.depth`` are the D-MPNN dests, so the
    required-hyperparam presence check resolves via KEY_ALIASES AND the
    loader resolves both keys (no REQUIRED-HYPERPARAM UNRESOLVED noise).
    """
    body = {
        "model": {"hidden": 128, "depth": 3},
        "train": {"batch_size": 16, "lr": 1.0e-4},
        "entry": {
            "command_template": (
                "python molmetal/scripts/train_dmpnn_multitask_wasserstein.py "
                "--config {yaml_path}"
            ),
        },
    }
    for section, values in extra.items():
        body.setdefault(section, {}).update(values)
    return body


# ---------------------------------------------------------------------------
# 1 — classifier unit contract
# ---------------------------------------------------------------------------


def test_classify_buckets_unit(preflight) -> None:
    """Bucket per key: reader / spec / residual + never-exempt override."""
    classify = preflight._classify_unresolved_key

    bucket, why = classify("protocol.scheduler.type")
    assert bucket == "reader" and "start.sh" in why

    bucket, why = classify("eval.target_test_auc")
    assert bucket == "spec" and why

    bucket, _ = classify("data.split.train_frac")
    assert bucket == "spec"
    bucket, _ = classify("data.split")  # exact prefix key
    assert bucket == "spec"

    # Never-exempt override: names a real CFM dest, must stay visible.
    bucket, why = classify("eval.target_decode_ratio")
    assert bucket == "residual" and "never exempted" in why

    # Dot-boundary: a near-miss key must NOT be swallowed by a prefix.
    bucket, _ = classify("data.split_extra")
    assert bucket == "residual"
    bucket, _ = classify("train.definitely_not_a_dest")
    assert bucket == "residual"

    # Documented-dead key: no reader, no dest, no spec namespace.
    bucket, why = classify("output.events_jsonl")
    assert bucket == "residual" and "no documented non-trainer consumer" in why


# ---------------------------------------------------------------------------
# 2 — strict mode now has a reachable ON state (exempt-only config PASSes)
# ---------------------------------------------------------------------------


def test_gate_strict_passes_on_documented_exempt_only(
    preflight, tmp_path, monkeypatch,
) -> None:
    """Strict mode PASSes when every unresolved key is documented-exempt."""
    yaml_path = _write_yaml(
        tmp_path / "exempt_only.yaml",
        _base_body(
            protocol={"scheduler": {"type": "cosine"}},  # reader-exempt
            eval={"target_test_auc": 0.82},              # spec-exempt
        ),
    )
    monkeypatch.setenv(STRICT_ENV, "1")
    result = preflight.gate_wf343c4_hyperparams_config_complete(
        "wfa13_strict_exempt", "P3", "wf343c4", (yaml_path,),
    )
    assert result.status == "PASS", result.evidence
    assert "0 RESIDUAL" in result.evidence
    assert "1 reader-exempt + 1 spec-exempt" in result.evidence


# ---------------------------------------------------------------------------
# 3 — strict mode still FAILs on a residual key (the switch can block)
# ---------------------------------------------------------------------------


def test_gate_strict_fails_on_residual(preflight, tmp_path, monkeypatch) -> None:
    """One residual key => strict mode FAILs and names the key."""
    yaml_path = _write_yaml(
        tmp_path / "residual.yaml",
        _base_body(
            protocol={"scheduler": {"type": "cosine"}},
            eval={"target_test_auc": 0.82},
            train={"definitely_not_a_dest": 1},
        ),
    )
    monkeypatch.setenv(STRICT_ENV, "1")
    result = preflight.gate_wf343c4_hyperparams_config_complete(
        "wfa13_strict_residual", "P3", "wf343c4", (yaml_path,),
    )
    assert result.status == "FAIL", result.evidence
    assert "RESIDUAL" in result.evidence
    assert "train.definitely_not_a_dest" in result.evidence
    # The documented-exempt keys are NOT the reason for the FAIL.
    assert "documented loader/reader/spec exemptions are NOT counted" in result.evidence

    # Same fixture WITHOUT strict => PASS (non-blocking default preserved).
    monkeypatch.delenv(STRICT_ENV, raising=False)
    relaxed = preflight.gate_wf343c4_hyperparams_config_complete(
        "wfa13_relaxed_residual", "P3", "wf343c4", (yaml_path,),
    )
    assert relaxed.status == "PASS", relaxed.evidence
    assert "1 RESIDUAL" in relaxed.evidence


# ---------------------------------------------------------------------------
# 4 — healthy tree regression: PASS + split report present
# ---------------------------------------------------------------------------


def test_gate_default_mode_reports_buckets_on_live_configs(
    preflight, monkeypatch,
) -> None:
    """9 active configs: PASS (default mode) with the Wf-A13 bucket report."""
    monkeypatch.delenv(STRICT_ENV, raising=False)
    yaml_paths = tuple(
        p for p in (
            CONFIGS_DIR / "cfm_full_retrain.yaml",
            CONFIGS_DIR / "cfm_per_innovation_ablation_p1.4.yaml",
            CONFIGS_DIR / "cfm_real_decode_lift_p1.3.yaml",
            CONFIGS_DIR / "default.yaml",
            CONFIGS_DIR / "wf124_p3_50epoch_full_sweep_g5.yaml",
            CONFIGS_DIR / "wf124_p3_per_innovation_ablation_g6.yaml",
            CONFIGS_DIR / "wf124_p3_remaining_metals_g4.yaml",
            CONFIGS_DIR / "training" / "nci60_9cell_5seed_50ep_v1.yaml",
            CONFIGS_DIR / "training" / "platinai_pt_4cell_5seed_50ep_v1.yaml",
    )
    )
    missing = [str(p) for p in yaml_paths if not p.is_file()]
    assert not missing, f"config fixtures missing on disk: {missing}"

    result = preflight.gate_wf343c4_hyperparams_config_complete(
        "wfa13_live", "P3", "wf343c4", yaml_paths,
    )
    assert result.status == "PASS", result.evidence
    assert "Wf-A13 RESOLUTION REPORT" in result.evidence
    assert "RESIDUAL" in result.evidence
    # Bucket totals must be present and arithmetically plausible.
    assert "loader-exempt" in result.evidence
    assert "reader-exempt" in result.evidence
    assert "spec-exempt" in result.evidence
