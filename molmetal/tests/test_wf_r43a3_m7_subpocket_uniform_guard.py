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

"""Wf-R43 m7 — diversity_subpocket uniform-placeholder guard tests.

Contracts under test:
  1. Uniform ``[1] * n_residues`` in the live path FAILs loud.
  2. Both docstring-cited "real weight source" blank lines are detected.
  3. A 30-cell sweep with n_unique=1 is reported as zero-variance.
  4. Non-uniform weights + between-cell variance PASSes.
  5. Absent metric file is SKIP, not FAIL.
  6. Malformed metric JSON FAILs loud, never raises.
  7. Opt-in gate emits SKIP unless the env var is set; emits the real
     verdict when set.
  8. The gate is registered in build_gates("all") and the expected
     count matches the actual count.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from molmetal.scripts.wf_r43a3_m7_subpocket_uniform_guard import (
    GATE_NAME,
    IMPL_RELPATHS,
    OPT_IN_ENV,
    R16_SWEEP_RELPATH,
    audit_m7_subpocket_uniform,
    gate_m7_subpocket_uniform,
)

_UNIFORM_IMPL = "weights = [1] * int(n_residues)\n"
_NONUNIFORM_IMPL = "weights = [1.0, 2.0, 0.5, 3.0]\n"


def _write(root: Path, relpath: str, text: str) -> Path:
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _make_repo(tmp_path: Path, *, uniform: bool, cells=None) -> Path:
    """Build a minimal repo tree the guard can audit."""
    root = tmp_path / "repo"
    body = _UNIFORM_IMPL if uniform else _NONUNIFORM_IMPL
    for rel in IMPL_RELPATHS:
        _write(root, rel, "def metric(c, n_residues=4):\n    " + body)
    if cells is not None:
        sweep = root / R16_SWEEP_RELPATH
        sweep.parent.mkdir(parents=True, exist_ok=True)
        sweep.write_text(json.dumps({"per_cell": cells}), encoding="utf-8")
    return root


def _metric_file(tmp_path: Path, payload=None) -> Path:
    path = tmp_path / "diversity_subpocket.json"
    if payload is None:
        payload = {"metric": "diversity_subpocket", "verdict": "DESIGN_ONLY — x"}
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 1 — uniform weights FAIL loud
# ---------------------------------------------------------------------------
def test_uniform_weights_fail_loud(tmp_path):
    root = _make_repo(tmp_path, uniform=True)
    result = audit_m7_subpocket_uniform(
        _metric_file(tmp_path), repo_root=root
    )
    assert result["status"] == "FAIL"
    assert result["weights"]["uniform"] is True
    assert sorted(result["weights"]["offenders"]) == sorted(IMPL_RELPATHS)
    assert "uniform" in result["evidence"]


def test_nonuniform_weights_clear_defect_one(tmp_path):
    root = _make_repo(tmp_path, uniform=False)
    result = audit_m7_subpocket_uniform(
        _metric_file(tmp_path), repo_root=root
    )
    assert result["weights"]["uniform"] is False
    # defect 1 cleared, so the guard must no longer blame the weights
    assert "uniform placeholder" not in result["evidence"]


# ---------------------------------------------------------------------------
# 2 — blank cited lines detected
# ---------------------------------------------------------------------------
def test_blank_cited_weight_lines_detected(tmp_path):
    root = tmp_path / "repo"
    # voxelization.py:208 blank, 255 = voxel origin; vina_adapter.py:961 blank
    vox = [""] * 260
    vox[254] = "origin = center - half_extent"
    _write(root, "molmetal/molmetal_lam/sbdd_env/voxelization.py", "\n".join(vox) + "\n")
    vina = [""] * 962
    _write(root, "molmetal/molmetal_lam/sbdd_env/vina_adapter.py", "\n".join(vina) + "\n")

    from molmetal.scripts.wf_r43a3_m7_subpocket_uniform_guard import (
        _audit_uniform_weights,
    )

    out = _audit_uniform_weights(root)
    cited = out["cited_sources"]
    assert cited["molmetal/molmetal_lam/sbdd_env/voxelization.py"]["lines"]["208"]["blank"] is True
    assert cited["molmetal/molmetal_lam/sbdd_env/vina_adapter.py"]["lines"]["961"]["blank"] is True
    # the non-blank line still does not look like a weight vector
    l255 = cited["molmetal/molmetal_lam/sbdd_env/voxelization.py"]["lines"]["255"]
    assert l255["blank"] is False
    assert l255["looks_like_weights"] is False


# ---------------------------------------------------------------------------
# 3 — zero-variance sweep detected
# ---------------------------------------------------------------------------
def test_constant_sweep_reports_zero_variance(tmp_path):
    cells = [{"div_subpocket": 0.6538729654113289} for _ in range(30)]
    root = _make_repo(tmp_path, uniform=False, cells=cells)
    result = audit_m7_subpocket_uniform(
        _metric_file(tmp_path), repo_root=root
    )
    sweep = result["sweep"]
    assert sweep["present"] is True
    assert sweep["n_cells"] == 30
    assert sweep["n_unique"] == 1
    assert sweep["has_variance"] is False
    assert sweep["value"] == pytest.approx(0.6538729654113289)
    assert result["status"] == "FAIL"
    assert "ZERO between-cell variance" in result["evidence"]


def test_varying_sweep_passes(tmp_path):
    cells = [{"div_subpocket": 0.1 * i} for i in range(30)]
    root = _make_repo(tmp_path, uniform=False, cells=cells)
    result = audit_m7_subpocket_uniform(
        _metric_file(tmp_path), repo_root=root
    )
    assert result["status"] == "PASS"


# ---------------------------------------------------------------------------
# 4/5/6 — absent sweep, absent metric, malformed metric
# ---------------------------------------------------------------------------
def test_absent_sweep_does_not_fail_on_variance(tmp_path):
    root = _make_repo(tmp_path, uniform=False, cells=None)
    result = audit_m7_subpocket_uniform(
        _metric_file(tmp_path), repo_root=root
    )
    assert result["sweep"]["present"] is False
    assert result["status"] == "PASS"


def test_absent_metric_is_skip_not_fail(tmp_path):
    root = _make_repo(tmp_path, uniform=False)
    result = audit_m7_subpocket_uniform(
        tmp_path / "does_not_exist.json", repo_root=root
    )
    assert result["status"] == "SKIP"
    assert "nothing to audit" in result["evidence"]


def test_malformed_metric_fails_loud_without_raising(tmp_path):
    root = _make_repo(tmp_path, uniform=False)
    bad = tmp_path / "diversity_subpocket.json"
    bad.write_text("{not json", encoding="utf-8")
    result = audit_m7_subpocket_uniform(bad, repo_root=root)
    assert result["status"] == "FAIL"
    assert "unreadable" in result["evidence"]


# ---------------------------------------------------------------------------
# 7 — opt-in gate
# ---------------------------------------------------------------------------
def test_gate_skips_without_env(tmp_path, monkeypatch):
    monkeypatch.delenv(OPT_IN_ENV, raising=False)
    root = _make_repo(tmp_path, uniform=True)
    res = gate_m7_subpocket_uniform(
        GATE_NAME, "P3", "wfr43m7", _metric_file(tmp_path), repo_root=root
    )
    assert res.status == "SKIP"
    assert OPT_IN_ENV in res.evidence


def test_gate_enforces_with_env(tmp_path, monkeypatch):
    monkeypatch.setenv(OPT_IN_ENV, "1")
    root = _make_repo(tmp_path, uniform=True)
    res = gate_m7_subpocket_uniform(
        GATE_NAME, "P3", "wfr43m7", _metric_file(tmp_path), repo_root=root
    )
    assert res.status == "FAIL"
    assert "uniform" in res.evidence


# ---------------------------------------------------------------------------
# 8 — registration + expected count
# ---------------------------------------------------------------------------
def test_gate_registered_and_count_matches():
    from molmetal.scripts.pre_flight_check import EXPECTED_GATE_COUNTS, build_gates

    gates = build_gates("all")
    assert len(gates) == EXPECTED_GATE_COUNTS["all"]
    # build_gates yields (name, thunk) pairs; the name is the first element.
    names = [pair[0] for pair in gates]
    assert GATE_NAME in names
