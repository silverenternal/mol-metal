# Licensed under the Apache License, Version 2.0.
"""Wf-A269 (2026-11-04) — the ledger's CONFIG provenance.

The mechanism under test: a ledger row that carries a config HASH must be
able to resolve that hash back to a FILE.  A digest alone cannot be
dereferenced — NOMAD's DAG and SLSA's ``resolvedDependencies`` both carry
the artifact identifier alongside the digest for exactly this reason.

The tests that matter most are the MUTATION ones.  A provenance check that
cannot FAIL is the DEAD-GATE class (honest-negative catalog §6.269.17), and
a check that only ever sees matching inputs proves nothing.  So:

  * a row whose hash matches NOTHING must FAIL the gate;
  * a row with no pointer at all, outside the dated baseline, must FAIL;
  * the producer must persist the config PATH next to the hash — the bug
    this wave found (``run_ledger.py`` accepted ``config_path`` and dropped
    it from the written row).

CPU-only: no GPU, no training, no network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from molmetal.scripts.pre_flight_check import (  # noqa: E402
    LEDGER_CONFIG_UNPOINTERED_BASELINE,
    EXPECTED_GATE_COUNTS,
    gate_run_ledger_config_provenance,
    build_gates,
)
from molmetal.scripts.run_ledger import (  # noqa: E402
    LEDGER_PATH,
    config_provenance,
    record_run,
)


def _write_ledger(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8"
    )
    return path


def _row(**kw) -> dict:
    base = {
        "metric_id": "m1_ru_auc",
        "value": 0.7,
        "source_json": "metrics/by_metric/metalcytotox_ru_auc.json",
        "source_json_sha256": "0" * 64,
        "git_sha": None,
        "data_sha": None,
        "config_sha": None,
        "protocol_id": None,
        "host": "test",
        "timestamp": "2026-11-04T00:00:00+00:00",
        "seed": False,
    }
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# 1. the producer must persist the config PATH
# ---------------------------------------------------------------------------


def test_the_producer_persists_the_config_path(tmp_path: Path) -> None:
    """Wf-A269: the row used to carry a bare hash with no pointer."""
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("epochs: 3\n", encoding="utf-8")
    ledger = tmp_path / "ledger.jsonl"
    entry = record_run(
        metric_id="m_probe",
        value=1.0,
        source_json="metrics/by_metric/probe.json",
        config_path=str(cfg),
        ledger_path=ledger,
        repo_root=REPO_ROOT,
    )
    assert entry.get("config_path"), "the config path was dropped from the row"
    written = json.loads(ledger.read_text(encoding="utf-8").splitlines()[0])
    assert written.get("config_path") == str(cfg)
    assert written.get("config_sha"), "the config hash must still be recorded"


def test_the_producer_stores_the_path_repo_relative(tmp_path: Path) -> None:
    """An absolute path in a ledger row is a HOST-BOUND locator (§6.269.13)."""
    rel_cfg = REPO_ROOT / "configs" / "dmpnn.yaml"
    if not rel_cfg.is_file():
        pytest.skip("configs/dmpnn.yaml is not present in this checkout")
    ledger = tmp_path / "ledger.jsonl"
    entry = record_run(
        metric_id="m_probe",
        value=1.0,
        source_json="metrics/by_metric/probe.json",
        config_path=str(rel_cfg),
        ledger_path=ledger,
        repo_root=REPO_ROOT,
    )
    assert entry["config_path"] == "configs/dmpnn.yaml"
    assert not Path(entry["config_path"]).is_absolute()


# ---------------------------------------------------------------------------
# 2. the check must FAIL when it can
# ---------------------------------------------------------------------------


def test_the_gate_fails_on_a_hash_that_resolves_to_nothing(tmp_path: Path) -> None:
    """MUTATION: the row names a file version this repository never held."""
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("epochs: 3\n", encoding="utf-8")
    ledger = _write_ledger(
        tmp_path / "ledger.jsonl",
        [_row(config_sha="f" * 64, config_path=str(cfg))],
    )
    rep = config_provenance(ledger_path=ledger, repo_root=REPO_ROOT)
    assert rep["counts"]["unresolvable"] == 1
    res = gate_run_ledger_config_provenance(
        "p3_run_ledger_config_provenance", "P3", "run_ledger", ledger_path=ledger
    )
    assert res.status == "FAIL", "a hash that resolves to nothing did not FAIL"
    assert "UNRESOLVABLE" in res.evidence


def test_the_gate_fails_on_an_unpointed_row_outside_the_baseline(
    tmp_path: Path,
) -> None:
    """MUTATION: a hash with no pointer cannot be dereferenced at all."""
    ledger = _write_ledger(
        tmp_path / "ledger.jsonl",
        [_row(metric_id="m_brand_new", config_sha="a" * 64)],
    )
    res = gate_run_ledger_config_provenance(
        "p3_run_ledger_config_provenance", "P3", "run_ledger", ledger_path=ledger
    )
    assert res.status == "FAIL", "an unpointed row outside the baseline did not FAIL"
    assert "UNPOINTERED" in res.evidence


def test_the_gate_passes_on_the_documented_baseline(tmp_path: Path) -> None:
    """The dated baseline row is KNOWN debt, not a new finding."""
    mid, sha = sorted(LEDGER_CONFIG_UNPOINTERED_BASELINE)[0]
    ledger = _write_ledger(
        tmp_path / "ledger.jsonl",
        [_row(metric_id=mid, config_sha=sha)],
    )
    res = gate_run_ledger_config_provenance(
        "p3_run_ledger_config_provenance", "P3", "run_ledger", ledger_path=ledger
    )
    assert res.status == "PASS", "the documented baseline row became a new finding"


# ---------------------------------------------------------------------------
# 3. the real ledger — the measurement this wave is based on
# ---------------------------------------------------------------------------


def test_the_real_ledger_measurement_is_stable() -> None:
    """MEASURED 2026-11-04: 434 rows, 6 with a config hash, 0 with a path.

    The counts are asserted as a RANGE, not as exact numbers: the ledger is
    append-only, so it grows.  What must not change is the SHAPE — every row
    with a hash resolves to a committed version of a real file, and no row
    claims a file version the repository has never held.
    """
    rep = config_provenance(ledger_path=LEDGER_PATH, repo_root=REPO_ROOT)
    c = rep["counts"]
    assert c["unresolvable"] == 0, (
        "a ledger row names a file version this repository has never held"
    )
    assert c["rows_with_config_sha"] >= 6
    assert c["rows_with_own_config_path"] == 0, (
        "the producer fix should make NEW rows carry a path; the 6 legacy rows "
        "cannot be rewritten (append-only, AGENTS.md §5.5 rule #6)"
    )
    assert c["historical"] + c["current"] + c["unpointed"] == c["rows_with_config_sha"]
    res = gate_run_ledger_config_provenance(
        "p3_run_ledger_config_provenance", "P3", "run_ledger"
    )
    assert res.status == "PASS", res.evidence


# ---------------------------------------------------------------------------
# 4. the gate is REGISTERED (a gate that exists but never runs is a zombie)
# ---------------------------------------------------------------------------


def test_the_gate_is_registered_in_both_pipelines() -> None:
    for pipeline in ("all", "launch"):
        names = [n for n, _ in build_gates(pipeline)]
        assert "p3_run_ledger_config_provenance" in names
        assert len(names) == EXPECTED_GATE_COUNTS[pipeline], (
            f"{pipeline}: registered {len(names)} gates, "
            f"EXPECTED_GATE_COUNTS says {EXPECTED_GATE_COUNTS[pipeline]}"
        )
