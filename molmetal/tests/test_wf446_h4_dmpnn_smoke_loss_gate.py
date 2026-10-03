"""Wf-446-H4 (HOLE 4) — D-MPNN loss-descent smoke gate tests.

Fix (2026-10-03) in ``molmetal/scripts/pre_flight_check.py``:
``train_dmpnn_multitask_wasserstein.py`` was executed by NO pre-flight
gate.  The new ``_gate_dmpnn_quick_smoke`` runs a 2-epoch CPU smoke on
the smallest v4 joint cohort (Re) and requires >=2 finite per-epoch
``train_loss`` points (parsed from ``events.jsonl`` ``train_epoch``
events, stdout regex fallback) with a >=0.3% relative decrease — the
classifier-side equivalent of the CFM §5.7 wire-in smoke (decode_ratio
is N/A for D-MPNN: no decoder exists).

These tests are CPU-only, network-free, GPU-free, and do NOT run the
real trainer (the end-to-end live PASS was MEASURED at fix time: status
PASS, wall 4.8 s, n_train=148, 3.0017 -> 0.9772).  The stub-based gate
tests prove the wiring FAILs loudly on diverged / missing-metric /
non-zero-exit smokes and SKIPs when the gitignored data CSVs are absent.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "molmetal" / "scripts" / "train_dmpnn_multitask_wasserstein.py"


def _pf():
    from molmetal.scripts import pre_flight_check

    return pre_flight_check


def _write_events(path: Path, losses) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for i, loss in enumerate(losses, start=1):
            fh.write(
                json.dumps(
                    {
                        "event": "train_epoch",
                        "epoch": i,
                        "train_loss": loss,
                        "metal": "Re",
                        "seed": 0,
                    },
                )
                + "\n"
            )
    return path


class TestLossDecreaseHelper:
    """Unit tests for ``_check_dmpnn_loss_decrease``."""

    def test_decreasing_loss_ok(self, tmp_path):
        pf = _pf()
        ej = _write_events(tmp_path / "events.jsonl", [3.0017, 0.9772])
        ok, detail = pf._check_dmpnn_loss_decrease(ej, "")
        assert ok is True, detail
        assert "loss decreased" in detail
        assert "events.jsonl" in detail

    def test_non_decreasing_fails(self, tmp_path):
        pf = _pf()
        ej = _write_events(tmp_path / "events.jsonl", [1.0, 1.1])
        ok, detail = pf._check_dmpnn_loss_decrease(ej, "")
        assert ok is False
        assert "did NOT decrease" in detail

    def test_negative_direction_fails(self, tmp_path):
        """Loss got worse (final > initial, both negative)."""
        pf = _pf()
        ej = _write_events(tmp_path / "events.jsonl", [-1.0, -0.5])
        ok, _ = pf._check_dmpnn_loss_decrease(ej, "")
        assert ok is False

    def test_flat_loss_fails(self, tmp_path):
        pf = _pf()
        ej = _write_events(tmp_path / "events.jsonl", [1.0, 1.0])
        ok, _ = pf._check_dmpnn_loss_decrease(ej, "")
        assert ok is False

    def test_tiny_drop_below_floor_fails(self, tmp_path):
        """0.01% drop < 0.3% floor must FAIL (2-epoch noise guard)."""
        pf = _pf()
        ej = _write_events(tmp_path / "events.jsonl", [1.0, 0.9999])
        ok, detail = pf._check_dmpnn_loss_decrease(ej, "")
        assert ok is False
        assert "did NOT decrease" in detail

    def test_drop_above_floor_passes(self, tmp_path):
        pf = _pf()
        ej = _write_events(tmp_path / "events.jsonl", [1.0, 0.9969])
        ok, _ = pf._check_dmpnn_loss_decrease(ej, "")
        assert ok is True

    def test_nan_final_fails(self, tmp_path):
        pf = _pf()
        ej = _write_events(tmp_path / "events.jsonl", [1.0, float("nan")])
        ok, detail = pf._check_dmpnn_loss_decrease(ej, "")
        assert ok is False
        assert "non-finite" in detail

    def test_single_point_fails(self, tmp_path):
        pf = _pf()
        ej = _write_events(tmp_path / "events.jsonl", [1.0])
        ok, detail = pf._check_dmpnn_loss_decrease(ej, "")
        assert ok is False
        assert "only 1 train_loss point" in detail

    def test_stdout_fallback_two_points(self, tmp_path):
        """No events.jsonl -> parse the legacy stdout lines."""
        pf = _pf()
        stdout = (
            "  [Re seed=0] epoch  1/2 loss=3.0017 cls=0.6639 reg=5.3395\n"
            "  [Re seed=0] epoch  2/2 loss=0.9772 cls=0.4605 reg=1.4940\n"
        )
        ok, detail = pf._check_dmpnn_loss_decrease(
            tmp_path / "missing.jsonl",
            stdout,
        )
        assert ok is True, detail
        assert "stdout regex fallback" in detail

    def test_missing_events_and_empty_stdout_fails(self, tmp_path):
        pf = _pf()
        ok, detail = pf._check_dmpnn_loss_decrease(
            tmp_path / "missing.jsonl",
            "",
        )
        assert ok is False
        assert "only 0 train_loss point" in detail

    def test_malformed_lines_skipped(self, tmp_path):
        pf = _pf()
        p = tmp_path / "events.jsonl"
        p.write_text(
            "{not json\n"
            + json.dumps({"event": "train_epoch", "epoch": 1, "train_loss": 3.0})
            + "\n"
            + json.dumps({"event": "eval_epoch", "loss": 99.0})
            + "\n"
            + json.dumps({"event": "train_epoch", "epoch": 2, "train_loss": 0.9})
            + "\n",
            encoding="utf-8",
        )
        ok, detail = pf._check_dmpnn_loss_decrease(p, "")
        assert ok is True, detail


class TestGateWiring:
    """End-to-end gate wiring with a stubbed smoke (no trainer spawned)."""

    @staticmethod
    def _fake_run_factory(payload=None, rc=0, emit_stdout=True):
        def fake_run(cmd, **kw):
            out_dir = Path(cmd[cmd.index("--out-dir") + 1])
            if payload is not None:
                _write_events(out_dir / "events.jsonl", payload)
            out = (
                "  [Re seed=0] epoch  1/2 loss=3.0017\n"
                "  [Re seed=0] epoch  2/2 loss=0.9772\n"
                "  Re: n_train=148 n_val=5 n_test=13 test_auc=0.2619\n"
                if emit_stdout
                else ""
            )
            return rc, out, ""

        return fake_run

    def _csv(self, tmp_path):
        p = tmp_path / "joint.csv"
        p.write_text("smiles,metal\n", encoding="utf-8")
        return p

    def test_gate_passes_on_healthy_smoke(self, tmp_path, monkeypatch):
        pf = _pf()
        monkeypatch.setattr(
            pf, "_run", self._fake_run_factory(payload=[3.0017, 0.9772])
        )
        monkeypatch.setattr(
            pf, "_DMPNN_SMOKE_DATA_CSV_CANDIDATES", (self._csv(tmp_path),)
        )
        res = pf._gate_dmpnn_quick_smoke(SCRIPT, csv_path=self._csv(tmp_path))
        assert res.status == "PASS", res.evidence
        assert "loss decreased" in res.evidence
        assert "n_train=148" in res.evidence
        assert res.name == "p3_dmpnn_loss_wire_in_smoke"
        assert res.priority == "P3" and res.pipeline == "dmpnn"

    def test_gate_fails_on_diverged_smoke(self, tmp_path, monkeypatch):
        pf = _pf()
        monkeypatch.setattr(pf, "_run", self._fake_run_factory(payload=[1.0, 1.1]))
        monkeypatch.setattr(
            pf, "_DMPNN_SMOKE_DATA_CSV_CANDIDATES", (self._csv(tmp_path),)
        )
        res = pf._gate_dmpnn_quick_smoke(SCRIPT, csv_path=self._csv(tmp_path))
        assert res.status == "FAIL", res.evidence
        assert "loss invariant FAILED" in res.evidence

    def test_gate_fails_on_no_events(self, tmp_path, monkeypatch):
        pf = _pf()
        monkeypatch.setattr(
            pf, "_run", self._fake_run_factory(payload=None, emit_stdout=False)
        )
        monkeypatch.setattr(
            pf, "_DMPNN_SMOKE_DATA_CSV_CANDIDATES", (self._csv(tmp_path),)
        )
        res = pf._gate_dmpnn_quick_smoke(SCRIPT, csv_path=self._csv(tmp_path))
        assert res.status == "FAIL", res.evidence
        assert "loss invariant FAILED" in res.evidence

    def test_gate_fails_on_rc_nonzero(self, tmp_path, monkeypatch):
        pf = _pf()
        monkeypatch.setattr(
            pf, "_run", self._fake_run_factory(payload=None, rc=1, emit_stdout=False)
        )
        monkeypatch.setattr(
            pf, "_DMPNN_SMOKE_DATA_CSV_CANDIDATES", (self._csv(tmp_path),)
        )
        res = pf._gate_dmpnn_quick_smoke(SCRIPT, csv_path=self._csv(tmp_path))
        assert res.status == "FAIL", res.evidence
        assert "exit=1" in res.evidence

    def test_gate_fails_on_timeout(self, tmp_path, monkeypatch):
        pf = _pf()
        monkeypatch.setattr(
            pf, "_run", self._fake_run_factory(payload=None, rc=124, emit_stdout=False)
        )
        monkeypatch.setattr(
            pf, "_DMPNN_SMOKE_DATA_CSV_CANDIDATES", (self._csv(tmp_path),)
        )
        res = pf._gate_dmpnn_quick_smoke(SCRIPT, csv_path=self._csv(tmp_path))
        assert res.status == "FAIL", res.evidence
        assert "TIMED OUT" in res.evidence

    def test_gate_skips_when_joint_csv_absent(self, tmp_path, monkeypatch):
        pf = _pf()
        monkeypatch.setattr(pf, "_run", self._fake_run_factory(payload=[1.0, 0.5]))
        res = pf._gate_dmpnn_quick_smoke(SCRIPT, csv_path=tmp_path / "nope.csv")
        assert res.status == "SKIP", res.evidence
        assert "gitignored" in res.evidence

    def test_gate_uses_private_out_dir_and_no_flag_only_max_joint_rows(
        self,
        tmp_path,
        monkeypatch,
    ):
        """Explicit --out-dir/--ckpt-dir (no repo side effects) and the
        FLAG_ONLY ``--max-joint-rows`` must NOT be relied on."""
        pf = _pf()
        seen = {}

        def fake_run(cmd, **kw):
            seen["out_dir"] = cmd[cmd.index("--out-dir") + 1]
            seen["ckpt_dir"] = cmd[cmd.index("--ckpt-dir") + 1]
            seen["cmd"] = list(cmd)
            return 1, "", "stop early"

        monkeypatch.setattr(pf, "_run", fake_run)
        monkeypatch.setattr(
            pf, "_DMPNN_SMOKE_DATA_CSV_CANDIDATES", (self._csv(tmp_path),)
        )
        pf._gate_dmpnn_quick_smoke(SCRIPT, csv_path=self._csv(tmp_path))
        assert "wf446_h4_dmpnn_smoke_" in seen["out_dir"]
        assert seen["out_dir"].startswith("/tmp/")
        assert "wf446_h4_dmpnn_smoke_" in seen["ckpt_dir"]
        assert "--max-joint-rows" not in seen["cmd"]
