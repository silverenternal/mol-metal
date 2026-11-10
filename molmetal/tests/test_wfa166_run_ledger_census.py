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

"""Tests for the Wf-A166 run-ledger provenance census.

CPU-only, no torch, no GPU, no network.  READ-ONLY on the real ledger: every
test that runs the census builds its own ledger under ``tmp_path`` and the
shipped artifact is only READ (and checked for internal consistency).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in str(sys.path):
    sys.path.insert(0, str(REPO_ROOT))

from molmetal.scripts import wfa166_run_ledger_census as census_mod  # noqa: E402

LEDGER = REPO_ROOT / "molmetal" / "reports" / "run_ledger.jsonl"
SHIPPED_JSON = (
    REPO_ROOT / "molmetal" / "reports" / "wfa166_run_ledger_census" / "census.json"
)
SHIPPED_MD = (
    REPO_ROOT / "molmetal" / "reports" / "wfa166_run_ledger_census" / "census.md"
)


def _write_ledger(path: Path, rows) -> Path:
    path.write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8"
    )
    return path


def test_census_is_read_only_on_the_real_ledger(tmp_path):
    """Running the census must not append to, rewrite or truncate the ledger."""
    before = LEDGER.read_bytes()
    row = json.loads(before.decode("utf-8").splitlines()[0])
    led = _write_ledger(tmp_path / "l.jsonl", [row])
    census_mod.build_census(ledger_path=led, repo_root=REPO_ROOT)
    assert LEDGER.read_bytes() == before


def test_axes_on_a_synthetic_row(tmp_path):
    """A row naming a real, committed file: A and B hold, C and E hold."""
    src = "metrics/by_metric/decode_ratio.json"
    p = REPO_ROOT / src
    assert p.is_file()
    disk_sha = census_mod.sha256_file(p)
    assert disk_sha
    head = census_mod.git("rev-parse", "HEAD")[1].strip()
    led = _write_ledger(
        tmp_path / "l.jsonl",
        [
            {
                "metric_id": "m_probe",
                "value": 1.0,
                "source_json": src,
                "source_json_sha256": disk_sha,
                "git_sha": head,
                "data_sha": None,
                "config_sha": None,
                "protocol_id": None,
                "host": "probe",
                "timestamp": "2026-11-01T00:00:00+00:00",
                "seed": True,
                "extra": {"value_key": "x"},
            }
        ],
    )
    c = census_mod.build_census(ledger_path=led, repo_root=REPO_ROOT)
    row = c["rows"][0]
    assert row["axis_A_sha_matches_disk"] is True
    assert row["axis_B_sha_matches_a_committed_version"] is True
    assert row["axis_C_commit_exists"] is True
    assert row["axis_E_commit_reachable"] is True
    assert row["axis_F_git_sha_is_a_bulk_constant"] is True  # 1 of 1
    assert c["denominator"]["ledger_rows"] == 1
    assert c["resolver_agreement"]["status"] == "AGREE"


def test_a_never_committed_sha_is_flagged(tmp_path):
    """A sha that is neither the disk content nor any committed version."""
    src = "metrics/by_metric/decode_ratio.json"
    led = _write_ledger(
        tmp_path / "l.jsonl",
        [
            {
                "metric_id": "m_probe",
                "value": 1.0,
                "source_json": src,
                "source_json_sha256": "b" * 64,
                "git_sha": "0" * 40,
                "data_sha": None,
                "config_sha": None,
                "protocol_id": None,
                "host": "probe",
                "timestamp": "2026-11-01T00:00:00+00:00",
                "seed": True,
                "extra": {},
            }
        ],
    )
    c = census_mod.build_census(ledger_path=led, repo_root=REPO_ROOT)
    row = c["rows"][0]
    assert row["axis_A_sha_matches_disk"] is False
    assert row["axis_B_sha_matches_a_committed_version"] is False
    assert row["axis_C_commit_exists"] is False
    assert row["axis_E_commit_reachable"] is False
    assert c["recoverability"]["rows_with_NO_recoverable_producing_commit"] == 1


def test_shipped_census_is_internally_consistent():
    """The shipped artifact: every axis count <= the denominator."""
    if not SHIPPED_JSON.is_file():
        pytest.skip("census artifact not built yet")
    c = json.loads(SHIPPED_JSON.read_text(encoding="utf-8"))
    n = c["denominator"]["ledger_rows"]
    assert n > 0
    for key, count in c["axis_counts"].items():
        assert 0 <= count <= n, (key, count, n)
    assert c["resolver_agreement"]["status"] == "AGREE"
    assert len(c["rows"]) == n
    assert c["bulk_constant"]["rows"] <= n
    assert SHIPPED_MD.is_file()


def test_column_contract_covers_every_written_column():
    """Every column the writer emits has a contract entry."""
    from molmetal.scripts import run_ledger as rl

    import inspect

    src = inspect.getsource(rl.record_run)
    written = {
        "metric_id",
        "value",
        "source_json",
        "source_json_sha256",
        "git_sha",
        "data_sha",
        "config_sha",
        "protocol_id",
        "host",
        "timestamp",
        "seed",
        "extra",
    }
    for col in written:
        assert f'"{col}"' in src, col
    contract_cols = {c["column"] for c in census_mod.COLUMN_CONTRACT}
    assert written <= contract_cols
    # the writer's own entry dict carries exactly these keys
    for col in written:
        assert any(c["column"] == col for c in census_mod.COLUMN_CONTRACT)
