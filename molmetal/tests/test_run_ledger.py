# Copyright 2026 Mol-Metal contributors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the append-only run ledger (molmetal/scripts/run_ledger.py).

CPU-only, no training, no network.  Every test writes to a pytest ``tmp_path``
ledger so the real ``molmetal/reports/run_ledger.jsonl`` is never mutated by
the suite.

Coverage:
  1. record_run + latest roundtrip.
  2. history returns every entry in append order.
  3. seed_from_disk records one baseline entry per registry metric and is
     idempotent-safe (a second seed appends nothing).
  4. staleness returns ``current`` for a fresh seed against the registry.
  5. staleness returns ``stale`` when the registry row's sha differs from the
     ledger entry, and ``no_run`` when a metric has no ledger entry.
  6. the ledger file is valid JSONL (one parseable JSON object per line).
  7. record_run appends (never rewrites): a second call yields two lines.
  8. the pre-flight gate emits SKIP by default and PASS under WF_RUN_LEDGER=1.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in str(sys.path):
    sys.path.insert(0, str(REPO_ROOT))

from molmetal.scripts import run_ledger as rl  # noqa: E402


def _mk_ledger(tmp_path: Path) -> Path:
    return tmp_path / "run_ledger.jsonl"


def test_record_then_latest_roundtrip(tmp_path):
    led = _mk_ledger(tmp_path)
    src = "metrics/by_metric/decode_ratio.json"
    entry = rl.record_run(
        "m14_decode_ratio", 0.998, source_json=src, ledger_path=led
    )
    assert entry["metric_id"] == "m14_decode_ratio"
    assert entry["value"] == 0.998
    got = rl.latest("m14_decode_ratio", ledger_path=led)
    assert got is not None
    assert got["value"] == 0.998
    assert got["source_json"] == src
    # a real sha256 was measured, not fabricated
    assert got["source_json_sha256"] and len(got["source_json_sha256"]) == 64


def test_history_returns_all_in_order(tmp_path):
    led = _mk_ledger(tmp_path)
    for v in (0.1, 0.2, 0.3):
        rl.record_run("m", v, source_json="metrics/by_metric/decode_ratio.json", ledger_path=led)
    h = rl.history("m", ledger_path=led)
    assert [e["value"] for e in h] == [0.1, 0.2, 0.3]
    # latest is the last appended
    assert rl.latest("m", ledger_path=led)["value"] == 0.3


def test_seed_from_disk_records_each_registry_metric(tmp_path):
    led = _mk_ledger(tmp_path)
    import yaml

    reg = yaml.safe_load(rl.REGISTRY_METRICS_PATH.read_text(encoding="utf-8"))
    n_metrics = len(reg["metrics"])
    written = rl.seed_from_disk(ledger_path=led)
    assert len(written) == n_metrics
    # every registry metric now has a ledger entry
    for mid in reg["metrics"]:
        assert rl.latest(mid, ledger_path=led) is not None
    assert sum(1 for _ in led.read_text().splitlines() if _.strip()) == n_metrics


def test_seed_is_idempotent_safe(tmp_path):
    led = _mk_ledger(tmp_path)
    first = rl.seed_from_disk(ledger_path=led)
    assert len(first) > 0
    before = led.read_text(encoding="utf-8")
    second = rl.seed_from_disk(ledger_path=led)  # skip_existing defaults True
    assert second == []
    assert led.read_text(encoding="utf-8") == before  # not rewritten


def test_staleness_current_for_fresh_seed(tmp_path):
    led = _mk_ledger(tmp_path)
    rl.seed_from_disk(ledger_path=led)
    info = rl.staleness(ledger_path=led)
    assert info, "registry metrics should be present"
    assert all(v["status"] == "current" for v in info.values()), info
    counts = rl.staleness_summary(ledger_path=led)
    assert counts["stale"] == 0
    assert counts["no_run"] == 0
    assert counts["current"] == len(info)


def test_staleness_stale_and_no_run(tmp_path):
    led = _mk_ledger(tmp_path)
    # record an entry whose sha deliberately differs from the registry row
    rl.record_run(
        "m1_ru_auc", 0.123, source_json="metrics/by_metric/metalcytotox_ru_auc.json",
        source_json_sha256="0" * 64, ledger_path=led,
    )
    info = rl.staleness(ledger_path=led)
    assert info["m1_ru_auc"]["status"] == "stale"
    # a registry metric with no ledger entry is no_run
    other = next(m for m in info if m != "m1_ru_auc")
    assert info[other]["status"] == "no_run"


def test_ledger_file_is_valid_jsonl(tmp_path):
    led = _mk_ledger(tmp_path)
    rl.seed_from_disk(ledger_path=led)
    lines = [ln for ln in led.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert lines, "seed wrote nothing"
    for ln in lines:
        obj = json.loads(ln)
        assert isinstance(obj, dict)
        assert "metric_id" in obj and "timestamp" in obj


def test_record_appends_never_rewrites(tmp_path):
    led = _mk_ledger(tmp_path)
    rl.record_run("m", 1, source_json="metrics/by_metric/decode_ratio.json", ledger_path=led)
    n1 = len(led.read_text(encoding="utf-8").splitlines())
    rl.record_run("m", 2, source_json="metrics/by_metric/decode_ratio.json", ledger_path=led)
    n2 = len(led.read_text(encoding="utf-8").splitlines())
    assert n1 == 1 and n2 == 2  # append-only, first line untouched
    assert rl.history("m", ledger_path=led)[0]["value"] == 1


def test_corrupt_ledger_line_raises(tmp_path):
    led = _mk_ledger(tmp_path)
    rl.record_run("m", 1, source_json="metrics/by_metric/decode_ratio.json", ledger_path=led)
    with led.open("a", encoding="utf-8") as fh:
        fh.write("{not json}\n")
    with pytest.raises(ValueError):
        rl.latest("m", ledger_path=led)


def test_preflight_gate_enforced_by_default(monkeypatch):
    """The gate enforces by default since 2026-10-08; WF_RUN_LEDGER=0 is the
    downgrade hatch, not an opt-in switch."""
    from molmetal.scripts.pre_flight_check import gate_run_ledger_staleness

    monkeypatch.delenv("WF_RUN_LEDGER", raising=False)
    r = gate_run_ledger_staleness("p3_run_ledger_staleness", "P3", "run_ledger")
    assert r.status in ("PASS", "FAIL"), r.evidence

    monkeypatch.setenv("WF_RUN_LEDGER", "0")
    r2 = gate_run_ledger_staleness("p3_run_ledger_staleness", "P3", "run_ledger")
    assert r2.status == "SKIP"


def test_preflight_gate_registered():
    from molmetal.scripts.pre_flight_check import build_gates

    names = [g[0] for g in build_gates("all")]
    assert "p3_run_ledger_staleness" in names


def _mk_registry(tmp_path: Path, metrics: dict) -> Path:
    import yaml

    path = tmp_path / "metrics.yaml"
    path.write_text(yaml.safe_dump({"metrics": metrics}, sort_keys=False), encoding="utf-8")
    return path


def test_a_reformat_with_the_same_value_is_not_stale(tmp_path):
    """MEASURED 2026-10-21: three of four "stale" rows had identical values.

    m7_diversity, m2_r2_per_cell and m21_nci60_9cell had only had their JSON
    rewritten; the number the ledger saw was still the number on disk. Calling
    that stale is what makes a provenance gate stop being read — and it hid the
    one row that had really changed (l5_hilbert_r2).
    """
    led = _mk_ledger(tmp_path)
    reg = _mk_registry(
        tmp_path,
        {
            "m7_diversity": {"value": 0.869, "source_json": "x.json", "source_json_sha256": "a" * 64},
        },
    )
    rl.record_run(
        "m7_diversity", 0.869, source_json="metrics/by_metric/m7_diversity.json",
        source_json_sha256="b" * 64, ledger_path=led,
    )
    info = rl.staleness(reg, ledger_path=led)
    assert info["m7_diversity"]["status"] == "reformatted"
    assert rl.check_ledger(reg, ledger_path=led)["status"] == "PASS"


def test_a_changed_value_with_a_changed_hash_is_still_stale(tmp_path):
    """The one row that really moved must not be reclassified.

    l5_hilbert_r2 went -5.6625 -> -6.7126 in 7e30d6ea with no ledger entry.
    """
    led = _mk_ledger(tmp_path)
    reg = _mk_registry(
        tmp_path,
        {
            "l5_hilbert_r2": {"value": -6.7126, "source_json": "x.json", "source_json_sha256": "a" * 64},
        },
    )
    rl.record_run(
        "l5_hilbert_r2", -5.6625, source_json="metrics/by_metric/hilbert_4cell_r2.json",
        source_json_sha256="b" * 64, ledger_path=led,
    )
    info = rl.staleness(reg, ledger_path=led)
    assert info["l5_hilbert_r2"]["status"] == "stale"
    assert rl.check_ledger(reg, ledger_path=led)["status"] == "WARN"


def test_the_summary_counts_reformatted_separately(tmp_path):
    led = _mk_ledger(tmp_path)
    reg = _mk_registry(
        tmp_path,
        {
            "a": {"value": 1.0, "source_json": "x.json", "source_json_sha256": "a" * 64},
            "b": {"value": 2.0, "source_json": "x.json", "source_json_sha256": "a" * 64},
        },
    )
    rl.record_run("a", 1.0, source_json="metrics/by_metric/decode_ratio.json",
                  source_json_sha256="b" * 64, ledger_path=led)
    counts = rl.staleness_summary(reg, ledger_path=led)
    assert counts == {"current": 0, "reformatted": 1, "stale": 0, "no_run": 1}
