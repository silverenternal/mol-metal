"""Wf-446-H2/L3 — shipped-checkpoint preservation for the chemprop smoke.

The trainer has no ckpt-dir knob (``CKPT_DIR`` is a module constant at
``train_chemprop_v2_v3.py:214``), so ``_gate_chemprop_quick_smoke`` would
silently overwrite the shipped checkpoint
``molmetal/checkpoints/wf200_chemprop_v2_v3/dmpnn_mtw_v3_ru_seed0.pt`` on
every run.  The gate now backs the file up (sha256) before the smoke and
restores it byte-identical afterwards; a failed/unverifiable restore FAILs
the gate, and a failed pre-run backup refuses to run the smoke at all.

CPU-only, network-free, GPU-free; the real trainer is never spawned (the
end-to-end live PASS is MEASURED separately: status=PASS with the byte-
identical restore line in evidence).
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


def _write_metrics(out_dir: Path, initial=-0.2134, final=-1.2563) -> Path:
    p = out_dir / "ru" / "per_seed_metrics_seed0.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps({"status": "ok", "initial_loss": initial, "final_loss": final}),
        encoding="utf-8",
    )
    return p


def _fake_run_overwriting_ckpt(ckpt_path: Path, payload=(-0.2134, -1.2563)):
    """Stub ``_run``: simulate the trainer overwriting the ckpt + metrics."""

    def fake_run(cmd, **kw):
        out_dir = Path(cmd[cmd.index("--out-dir") + 1])
        _write_metrics(out_dir, *payload)
        if ckpt_path.is_file():  # mirror trainer torch.save (in place)
            ckpt_path.write_bytes(b"SMOKE-OVERWROTE-THE-SHIPPED-CKPT" * 64)
        return 0, "epoch=0/2 loss=0.0", ""

    return fake_run


class TestBackupHelper:
    def test_backup_reads_sha_and_copies(self, tmp_path):
        pf = _pf()
        ckpt = tmp_path / "ckpt.pt"
        ckpt.write_bytes(b"shipped-bytes")
        bdir = tmp_path / "backup"
        bdir.mkdir()
        ok, pre_sha, detail = pf._prepare_chemprop_ckpt_backup(ckpt, bdir)
        assert ok is True, detail
        assert pre_sha == pf._sha256_file(ckpt)
        assert (bdir / "ckpt.pt").read_bytes() == b"shipped-bytes"
        assert "backed up" in detail and pre_sha[:16] in detail

    def test_absent_ckpt_is_ok_with_none_sha(self, tmp_path):
        pf = _pf()
        bdir = tmp_path / "backup"
        bdir.mkdir()
        ok, pre_sha, detail = pf._prepare_chemprop_ckpt_backup(
            tmp_path / "missing.pt", bdir
        )
        assert ok is True, detail
        assert pre_sha is None
        assert "fresh tree" in detail

    def test_unreadable_ckpt_refuses(self, tmp_path):
        pf = _pf()
        ckpt = tmp_path / "ckpt.pt"
        ckpt.write_bytes(b"x")
        bdir = tmp_path / "backup"
        bdir.mkdir()
        ckpt.chmod(0o000)
        try:
            ok, pre_sha, detail = pf._prepare_chemprop_ckpt_backup(ckpt, bdir)
        finally:
            ckpt.chmod(0o644)
        assert ok is False
        assert pre_sha is None
        assert "cannot back up" in detail and "refusing to run" in detail


class TestRestoreHelper:
    def test_overwritten_ckpt_is_restored_byte_identical(self, tmp_path):
        pf = _pf()
        ckpt = tmp_path / "ckpt.pt"
        ckpt.write_bytes(b"shipped")
        bdir = tmp_path / "backup"
        bdir.mkdir()
        ok, pre_sha, _ = pf._prepare_chemprop_ckpt_backup(ckpt, bdir)
        assert ok
        ckpt.write_bytes(b"smoke-output")
        ok, detail = pf._restore_chemprop_ckpt(ckpt, bdir, pre_sha)
        assert ok is True, detail
        assert ckpt.read_bytes() == b"shipped"
        assert pf._sha256_file(ckpt) == pre_sha
        assert "restored byte-identical" in detail and pre_sha[:16] in detail

    def test_unchanged_ckpt_reports_unchanged(self, tmp_path):
        pf = _pf()
        ckpt = tmp_path / "ckpt.pt"
        ckpt.write_bytes(b"shipped")
        bdir = tmp_path / "backup"
        bdir.mkdir()
        ok, pre_sha, _ = pf._prepare_chemprop_ckpt_backup(ckpt, bdir)
        assert ok
        ok, detail = pf._restore_chemprop_ckpt(ckpt, bdir, pre_sha)
        assert ok is True and "unchanged by smoke" in detail

    def test_vanish_after_smoke_fails(self, tmp_path):
        pf = _pf()
        ckpt = tmp_path / "ckpt.pt"
        ckpt.write_bytes(b"shipped")
        bdir = tmp_path / "backup"
        bdir.mkdir()
        ok, pre_sha, _ = pf._prepare_chemprop_ckpt_backup(ckpt, bdir)
        assert ok
        ckpt.unlink()
        ok, detail = pf._restore_chemprop_ckpt(ckpt, bdir, pre_sha)
        assert ok is False and "is gone after smoke" in detail

    def test_copy_back_failure_fails(self, tmp_path, monkeypatch):
        pf = _pf()
        ckpt = tmp_path / "ckpt.pt"
        ckpt.write_bytes(b"shipped")
        bdir = tmp_path / "backup"
        bdir.mkdir()
        ok, pre_sha, _ = pf._prepare_chemprop_ckpt_backup(ckpt, bdir)
        assert ok
        ckpt.write_bytes(b"smoke-output")
        real_copy2 = pf.shutil.copy2

        def exploding_copy2(src, dst, *a, **kw):
            if Path(dst) == ckpt:  # only the restore direction
                raise PermissionError("EACCES: restore denied (constructed)")
            return real_copy2(src, dst, *a, **kw)

        monkeypatch.setattr(pf.shutil, "copy2", exploding_copy2)
        ok, detail = pf._restore_chemprop_ckpt(ckpt, bdir, pre_sha)
        assert ok is False
        assert "restore of" in detail and "restore denied" in detail

    def test_fresh_tree_smoke_created_ckpt_is_removed(self, tmp_path):
        pf = _pf()
        ckpt = tmp_path / "ckpt.pt"
        ok, detail = pf._restore_chemprop_ckpt(ckpt, tmp_path, None)
        assert ok is True and "wrote no ckpt" in detail
        ckpt.write_bytes(b"smoke-created")
        ok, detail = pf._restore_chemprop_ckpt(ckpt, tmp_path, None)
        assert ok is True and "removed smoke-created" in detail
        assert not ckpt.exists()


class TestGateWiring:
    """Drive the real gate function with a stubbed smoke process."""

    def test_gate_restores_overwritten_shipped_ckpt(self, tmp_path, monkeypatch):
        pf = _pf()
        ckpt = tmp_path / "dmpnn_mtw_v3_ru_seed0.pt"
        ckpt.write_bytes(b"SHIPPED-CHECKPOINT-BYTES")
        pre_sha = pf._sha256_file(ckpt)
        monkeypatch.setattr(pf, "_CHEMPROP_SMOKE_CKPT_PATH", ckpt)
        monkeypatch.setattr(pf, "_run", _fake_run_overwriting_ckpt(ckpt))
        res = pf._gate_chemprop_quick_smoke(SCRIPT)
        assert res.status == "PASS", res.evidence
        assert pf._sha256_file(ckpt) == pre_sha
        assert ckpt.read_bytes() == b"SHIPPED-CHECKPOINT-BYTES"
        assert f"pre_sha256={pre_sha[:16]}" in res.evidence
        assert "restored byte-identical" in res.evidence

    def test_gate_fails_when_restore_fails(self, tmp_path, monkeypatch):
        pf = _pf()
        ckpt = tmp_path / "dmpnn_mtw_v3_ru_seed0.pt"
        ckpt.write_bytes(b"SHIPPED-CHECKPOINT-BYTES")
        monkeypatch.setattr(pf, "_CHEMPROP_SMOKE_CKPT_PATH", ckpt)
        monkeypatch.setattr(pf, "_run", _fake_run_overwriting_ckpt(ckpt))
        real_copy2 = pf.shutil.copy2

        def exploding_copy2(src, dst, *a, **kw):
            if Path(dst) == ckpt:
                raise PermissionError("EACCES: restore denied (constructed)")
            return real_copy2(src, dst, *a, **kw)

        monkeypatch.setattr(pf.shutil, "copy2", exploding_copy2)
        res = pf._gate_chemprop_quick_smoke(SCRIPT)
        assert res.status == "FAIL", res.evidence
        assert "checkpoint restore FAILED" in res.evidence
        assert "restore denied" in res.evidence

    def test_gate_refuses_to_run_when_backup_impossible(self, tmp_path, monkeypatch):
        pf = _pf()
        ckpt = tmp_path / "dmpnn_mtw_v3_ru_seed0.pt"
        ckpt.write_bytes(b"SHIPPED-CHECKPOINT-BYTES")
        ckpt.chmod(0o000)
        monkeypatch.setattr(pf, "_CHEMPROP_SMOKE_CKPT_PATH", ckpt)

        def must_not_run(cmd, **kw):
            raise AssertionError("smoke must NOT run without a ckpt backup")

        monkeypatch.setattr(pf, "_run", must_not_run)
        try:
            res = pf._gate_chemprop_quick_smoke(SCRIPT)
        finally:
            ckpt.chmod(0o644)
        assert res.status == "FAIL", res.evidence
        assert "cannot back up" in res.evidence
        assert "smoke NOT run" in res.evidence

    def test_gate_pins_metal_and_seed(self, tmp_path, monkeypatch):
        """The overwritten ckpt path is known only if Ru/seed0 is pinned."""
        pf = _pf()
        ckpt = tmp_path / "dmpnn_mtw_v3_ru_seed0.pt"
        ckpt.write_bytes(b"shipped")
        monkeypatch.setattr(pf, "_CHEMPROP_SMOKE_CKPT_PATH", ckpt)
        seen = {}

        def fake_run(cmd, **kw):
            seen["metals"] = cmd[cmd.index("--metals") + 1]
            seen["seeds"] = cmd[cmd.index("--seeds") + 1]
            return 1, "", "stop"

        monkeypatch.setattr(pf, "_run", fake_run)
        pf._gate_chemprop_quick_smoke(SCRIPT)
        assert seen["metals"] == "Ru"
        assert seen["seeds"] == "0"

    def test_gate_unchanged_ckpt_still_passes(self, tmp_path, monkeypatch):
        pf = _pf()
        ckpt = tmp_path / "dmpnn_mtw_v3_ru_seed0.pt"
        ckpt.write_bytes(b"shipped")
        pre_sha = pf._sha256_file(ckpt)
        monkeypatch.setattr(pf, "_CHEMPROP_SMOKE_CKPT_PATH", ckpt)

        def fake_run(cmd, **kw):
            _write_metrics(Path(cmd[cmd.index("--out-dir") + 1]))
            return 0, "ok", ""  # does not touch the ckpt

        monkeypatch.setattr(pf, "_run", fake_run)
        res = pf._gate_chemprop_quick_smoke(SCRIPT)
        assert res.status == "PASS", res.evidence
        assert "unchanged by smoke" in res.evidence
        assert pf._sha256_file(ckpt) == pre_sha


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
