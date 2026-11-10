# Licensed under the Apache License, Version 2.0.
"""The config↔trainer binding gate's DECLARED-SKIP class (Wf-A269, 2026-11-04).

The mechanism under test: a config with no ``entry.script`` (and no ``.py``
token in ``entry.command_template``) cannot be bound to a trainer parser —
there is no parser to bind it to.  The gate therefore has to SKIP it.  Before
2026-11-04 it skipped it **silently**, kept a PASS verdict, and printed a
closing line that read *"every config key reaches a dest of its own
entry.script parser"* — a sentence that was FALSE for every skipped config.

That is the §6.269.17 DEAD-GATE class in its "skipped" costume: the check
could PASS without checking.  MEASURED 2026-11-04: **2 of 498** configs were
skipped while the gate said PASS.

The repair gives the gate the loader's own THREE-DISPOSITION shape:

  1. bound to a trainer      → checked, normally;
  2. skipped **and DECLARED** in ``DECLARED_SKIPPED_CONFIGS`` with a reason
     → reported as ``DECLARED-SKIP`` every run, reason on file;
  3. skipped **and NOT declared** → a **VIOLATION**, always.

So the tests that matter are the MUTATION ones — they prove the check can
still FAIL:

  * an UNDECLARED skip must produce a violation and a FAIL verdict;
  * removing a declaration must flip the verdict from PASS to FAIL — the
    declaration is load-bearing, not decoration;
  * a declared skip must NOT be counted among the unchecked configs, and must
    be reported with its reason on every run.

CPU-only.  No config is written into the repository: every fixture is a
``tmp_path`` file, and the gate is audited over an explicit config list.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_PATH = REPO_ROOT / "scripts" / "_config_binding_gate.py"

_spec = importlib.util.spec_from_file_location("_config_binding_gate_decl", GATE_PATH)
gate = importlib.util.module_from_spec(_spec)
sys.modules["_config_binding_gate_decl"] = gate
_spec.loader.exec_module(gate)


def _entryless_config(tmp_path: Path, name: str) -> Path:
    """A config with NO ``entry`` section — the shape that cannot be bound."""
    p = tmp_path / name
    p.write_text(
        yaml.safe_dump({"protocol": {"epochs": 3}}, sort_keys=False),
        encoding="utf-8",
    )
    return p


# --------------------------------------------------------------------- tests
def test_an_undeclared_skip_is_a_violation_and_the_gate_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MUTATION: the check must be able to FAIL on a config it cannot bind.

    This is the anti-dead-gate test.  If an undeclared skip could still yield
    PASS, the whole repair would be cosmetic.
    """
    cfg = _entryless_config(tmp_path, "not_declared.yaml")
    report = gate.audit(configs=[cfg])
    assert report["verdict"] == "FAIL", (
        "a config the gate could not bind, and which nobody declared, still "
        "produced a PASS verdict"
    )
    assert report["n_configs_skipped"] == 1
    assert any("not_declared.yaml" in v.get("config", "") for v in report["violations"]), (
        f"the violation does not name the offending config: {report['violations']}"
    )


def test_a_declared_skip_is_reported_with_its_reason_and_is_not_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A DECLARED skip is disclosed every run, and is not an unchecked config."""
    cfg = _entryless_config(tmp_path, "declared.yaml")
    monkeypatch.setitem(
        gate.DECLARED_SKIPPED_CONFIGS,
        "declared.yaml",
        "a fixture that names no trainer on purpose",
    )
    report = gate.audit(configs=[cfg])
    assert report["verdict"] == "PASS", report["violations"]
    assert report["n_configs_skipped"] == 0, (
        "a DECLARED skip was counted among the unchecked configs"
    )
    assert report["n_configs_declared_skipped"] == 1
    assert "declared.yaml" in report["declared_skipped"][0]
    assert "on purpose" in report["declared_skipped"][0], (
        "the declaration's reason was not carried into the report"
    )


def test_removing_a_declaration_flips_the_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MUTATION: the declaration is load-bearing.

    The same config, the same fixture, the same gate — only the declaration
    differs.  If the declaration did nothing, this test could not exist.
    """
    cfg = _entryless_config(tmp_path, "flip.yaml")
    monkeypatch.setitem(gate.DECLARED_SKIPPED_CONFIGS, "flip.yaml", "declared")
    assert gate.audit(configs=[cfg])["verdict"] == "PASS"

    monkeypatch.delitem(gate.DECLARED_SKIPPED_CONFIGS, "flip.yaml")
    assert gate.audit(configs=[cfg])["verdict"] == "FAIL", (
        "removing the declaration did not change the verdict — the "
        "DECLARED_SKIPPED_CONFIGS registry is not load-bearing"
    )


def test_the_two_repository_fixtures_are_declared() -> None:
    """The repo's own entry:-less fixtures must be DECLARED, not skipped.

    They are entry:-less BY CONSTRUCTION (they exercise ``scripts/start.sh``'s
    legacy branch), so an undeclared skip would put the gate permanently in
    FAIL — which is the honest state, and the reason the registry exists.
    """
    for name in ("wfa269_c9_start_sh.yaml", "wfa269_c9_start_sh_dmpnn.yaml"):
        assert name in gate.DECLARED_SKIPPED_CONFIGS, (
            f"{name} is an entry:-less fixture and is not declared — the gate "
            "will FAIL on it, which is the honest state"
        )
        assert gate.DECLARED_SKIPPED_CONFIGS[name].strip(), (
            f"{name} is declared with no reason"
        )


def test_the_live_repo_audit_has_no_undeclared_skip() -> None:
    """The live population must have ZERO undeclared skips."""
    report = gate.audit()
    assert report["n_configs_skipped"] == 0, (
        "these configs are skipped by the gate and nobody declared them: "
        f"{report['skipped']}"
    )
    assert report["verdict"] == "PASS", report["violations"]
