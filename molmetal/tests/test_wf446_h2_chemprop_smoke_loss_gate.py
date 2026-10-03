"""Wf-446-H2 (HOLE 2) — chemprop quick-smoke loss-decrease invariant tests.

Fix (2026-10-03) in ``molmetal/scripts/pre_flight_check.py``:
``_gate_chemprop_quick_smoke`` used to PASS on ``rc == 0`` alone, but
``rc == 0`` only means every run's status is "ok", and status "ok" merely
means "not timed out" (``train_chemprop_v2_v3.py`` ~L1232) — a diverged /
NaN loss still exits 0.  The gate now runs the smoke with an explicit
``--out-dir`` (private temp dir; previously it polluted
``molmetal/reports/wf200_chemprop_v2_v3/ru/``) and enforces, via
``_check_chemprop_loss_decrease``, that ``initial_loss`` / ``final_loss``
are finite and the loss decreased by >=0.3% relative to ``|initial|``.

These tests are CPU-only, network-free, GPU-free, and do NOT run the real
trainer (the end-to-end live PASS was MEASURED at fix time: status=PASS,
wall 31.3 s, initial=-0.2149 -> final=-1.2632).  The stub-based gate tests
prove the wiring FAILs loudly on a diverged / missing-metrics smoke.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "molmetal" / "scripts" / "train_chemprop_v2_v3.py"


def _pf():
    from molmetal.scripts import pre_flight_check
    return pre_flight_check


def _write_metrics(path: Path, initial, final, status="ok") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"status": status, "initial_loss": initial,
                    "final_loss": final}),
        encoding="utf-8",
    )
    return path


class TestLossDecreaseHelper:
    """Unit tests for ``_check_chemprop_loss_decrease``."""

    def test_decreasing_loss_ok(self, tmp_path):
        pf = _pf()
        ok, detail = pf._check_chemprop_loss_decrease(
            _write_metrics(tmp_path / "m.json", -0.2134, -1.2563))
        assert ok is True, detail
        assert "loss decreased" in detail

    def test_non_decreasing_fails(self, tmp_path):
        pf = _pf()
        ok, detail = pf._check_chemprop_loss_decrease(
            _write_metrics(tmp_path / "m.json", -1.0, 0.5))
        assert ok is False
        assert "did NOT decrease" in detail

    def test_negative_direction_fails(self, tmp_path):
        """Loss got worse (final > initial, both negative)."""
        pf = _pf()
        ok, _ = pf._check_chemprop_loss_decrease(
            _write_metrics(tmp_path / "m.json", -1.0, -0.5))
        assert ok is False

    def test_flat_loss_fails(self, tmp_path):
        pf = _pf()
        ok, _ = pf._check_chemprop_loss_decrease(
            _write_metrics(tmp_path / "m.json", -1.0, -1.0))
        assert ok is False

    def test_tiny_drop_below_floor_fails(self, tmp_path):
        """0.01% drop < 0.3% floor must FAIL (2-epoch noise guard)."""
        pf = _pf()
        ok, detail = pf._check_chemprop_loss_decrease(
            _write_metrics(tmp_path / "m.json", -1.0, -1.0001))
        assert ok is False
        assert "did NOT decrease" in detail

    def test_drop_above_floor_passes(self, tmp_path):
        pf = _pf()
        ok, _ = pf._check_chemprop_loss_decrease(
            _write_metrics(tmp_path / "m.json", -1.0, -1.0031))
        assert ok is True

    def test_nan_final_fails(self, tmp_path):
        pf = _pf()
        ok, detail = pf._check_chemprop_loss_decrease(
            _write_metrics(tmp_path / "m.json", -1.0, float("nan")))
        assert ok is False
        assert "non-finite" in detail

    def test_nan_initial_fails(self, tmp_path):
        pf = _pf()
        ok, detail = pf._check_chemprop_loss_decrease(
            _write_metrics(tmp_path / "m.json", float("nan"), -1.0))
        assert ok is False
        assert "non-finite" in detail

    def test_missing_file_fails(self, tmp_path):
        pf = _pf()
        ok, detail = pf._check_chemprop_loss_decrease(tmp_path / "nope.json")
        assert ok is False
        assert "missing" in detail

    def test_missing_keys_fails(self, tmp_path):
        pf = _pf()
        p = tmp_path / "m.json"
        p.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
        ok, detail = pf._check_chemprop_loss_decrease(p)
        assert ok is False
        assert "lack numeric" in detail

    def test_unreadable_json_fails(self, tmp_path):
        pf = _pf()
        p = tmp_path / "m.json"
        p.write_text("{not json", encoding="utf-8")
        ok, detail = pf._check_chemprop_loss_decrease(p)
        assert ok is False
        assert "unreadable" in detail


class TestGateWiring:
    """End-to-end gate wiring with a stubbed smoke (no trainer spawned)."""

    @staticmethod
    def _fake_run_factory(payload=None, rc=0):
        def fake_run(cmd, **kw):
            out_dir = Path(cmd[cmd.index("--out-dir") + 1])
            if payload is not None:
                _write_metrics(
                    out_dir / "ru" / "per_seed_metrics_seed0.json",
                    payload[0], payload[1],
                )
            return rc, "epoch=0/2 loss=0.0", ""
        return fake_run

    def test_gate_fails_on_diverged_smoke(self, tmp_path, monkeypatch):
        pf = _pf()
        monkeypatch.setattr(
            pf, "_run", self._fake_run_factory(payload=(-1.0, 0.5)))
        res = pf._gate_chemprop_quick_smoke(SCRIPT)
        assert res.status == "FAIL", res.evidence
        assert "loss invariant FAILED" in res.evidence
        assert res.name == "p1_chemprop_smoke"
        assert res.priority == "P1" and res.pipeline == "chemprop_v2_v3"

    def test_gate_fails_on_no_metrics(self, monkeypatch):
        pf = _pf()
        monkeypatch.setattr(pf, "_run", self._fake_run_factory(payload=None))
        res = pf._gate_chemprop_quick_smoke(SCRIPT)
        assert res.status == "FAIL", res.evidence
        assert "wrote no per-seed metrics" in res.evidence

    def test_gate_fails_on_rc_nonzero(self, monkeypatch):
        pf = _pf()
        monkeypatch.setattr(
            pf, "_run", self._fake_run_factory(payload=None, rc=1))
        res = pf._gate_chemprop_quick_smoke(SCRIPT)
        assert res.status == "FAIL", res.evidence
        assert "exit=1" in res.evidence

    def test_gate_passes_on_healthy_smoke(self, monkeypatch):
        pf = _pf()
        monkeypatch.setattr(
            pf, "_run", self._fake_run_factory(payload=(-0.2134, -1.2563)))
        res = pf._gate_chemprop_quick_smoke(SCRIPT)
        assert res.status == "PASS", res.evidence
        assert "loss decreased" in res.evidence

    def test_gate_uses_private_out_dir(self, monkeypatch):
        """The smoke must pass an explicit --out-dir (side-effect fix)."""
        pf = _pf()
        seen = {}

        def fake_run(cmd, **kw):
            seen["out_dir"] = cmd[cmd.index("--out-dir") + 1]
            return 1, "", "stop early"

        monkeypatch.setattr(pf, "_run", fake_run)
        pf._gate_chemprop_quick_smoke(SCRIPT)
        assert "wf446_h2_chemprop_smoke_" in seen["out_dir"]
        assert seen["out_dir"].startswith("/tmp/")
