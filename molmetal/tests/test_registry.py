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

"""Tests for the registry subsystem (molmetal/scripts/registry.py).

CPU-only, no training.  The registry is the machine-readable single source of
truth; these tests prove the three layers (datasets / protocols / metrics)
really verify against disk by content hash — and that they FAIL on drift, so
the guarantee is not vacuous.

Coverage:
  1. generate() writes all three registry files on disk.
  2. check_all() PASSes on the freshly-generated registry (hash match).
  3. a drifted dataset sha256 FAILs the datasets layer.
  4. a drifted metric source_json is reported STALE, not PASS.
  5. a missing metric source_json FAILs the metrics layer.
  6. check_datasets SKIPs when the registry file is absent.
  7. the LINE_METRICS mapping covers all three paper lines.
  8. the pre-flight gate emits SKIP by default (report-only rollout).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in str(sys.path):
    sys.path.insert(0, str(REPO_ROOT))

from molmetal.scripts import registry as reg  # noqa: E402


def test_generate_writes_three_files(tmp_path):
    rep = reg.generate(REPO_ROOT)
    assert rep["datasets"] > 0
    assert rep["protocols"] > 0
    assert rep["metrics"] > 0
    assert reg.DATASETS_PATH.is_file()
    assert reg.PROTOCOLS_PATH.is_file()
    assert reg.METRICS_PATH.is_file()


def test_check_all_passes_on_fresh_registry():
    reg.generate(REPO_ROOT)
    results = reg.check_all(REPO_ROOT)
    assert results["datasets"].status == "PASS", results["datasets"].evidence
    assert results["protocols"].status == "PASS", results["protocols"].evidence
    assert results["metrics"].status == "PASS", results["metrics"].evidence


def test_drifted_dataset_sha_fails(tmp_path):
    import yaml

    doc = yaml.safe_load(reg.DATASETS_PATH.read_text(encoding="utf-8"))
    # corrupt one recorded sha256 -> the live hash can't match
    first = next(iter(doc["datasets"]))
    doc["datasets"][first]["sha256"] = "0" * 64
    reg.DATASETS_PATH.write_text(yaml.safe_dump(doc), encoding="utf-8")
    try:
        r = reg.check_datasets(REPO_ROOT)
        assert r.status == "FAIL"
        assert any(c["status"] == "FAIL" for c in r.checks)
    finally:
        reg.generate(REPO_ROOT)  # restore a consistent registry


def test_drifted_metric_source_is_stale(tmp_path):
    import yaml

    doc = yaml.safe_load(reg.METRICS_PATH.read_text(encoding="utf-8"))
    first = next(iter(doc["metrics"]))
    doc["metrics"][first]["source_json_sha256"] = "0" * 64
    reg.METRICS_PATH.write_text(yaml.safe_dump(doc), encoding="utf-8")
    try:
        r = reg.check_metrics(REPO_ROOT)
        assert r.status == "STALE"
        assert any(c["status"] == "STALE" for c in r.checks)
    finally:
        reg.generate(REPO_ROOT)


def test_missing_metric_source_fails(tmp_path):
    import yaml

    doc = yaml.safe_load(reg.METRICS_PATH.read_text(encoding="utf-8"))
    first = next(iter(doc["metrics"]))
    doc["metrics"][first]["source_json"] = "metrics/by_metric/__does_not_exist__.json"
    doc["metrics"][first]["source_json_missing"] = True
    reg.METRICS_PATH.write_text(yaml.safe_dump(doc), encoding="utf-8")
    try:
        r = reg.check_metrics(REPO_ROOT)
        assert r.status == "FAIL"
    finally:
        reg.generate(REPO_ROOT)


def test_absent_registry_skips(tmp_path):
    missing = tmp_path / "nope.yaml"

    orig = reg.DATASETS_PATH
    try:
        reg.DATASETS_PATH = missing
        r = reg.check_datasets(REPO_ROOT)
        assert r.status == "SKIP"
    finally:
        reg.DATASETS_PATH = orig


def test_line_metrics_cover_three_lines():
    assert set(reg.LINE_METRICS) == {"line1_cfm", "line2_dmpnn", "line3_lambda_mcts"}
    for line, mapping in reg.LINE_METRICS.items():
        assert mapping, f"{line} has no metrics"


def test_preflight_gate_skips_by_default(monkeypatch):
    from molmetal.scripts.pre_flight_check import gate_registry_consistency

    monkeypatch.delenv("WF_REGISTRY_ENFORCE", raising=False)
    r = gate_registry_consistency("p3_registry_consistency", "P3", "registry")
    assert r.status == "SKIP"


def test_preflight_gate_registered():
    from molmetal.scripts.pre_flight_check import build_gates

    names = [g[0] for g in build_gates("all")]
    assert "p3_registry_consistency" in names
