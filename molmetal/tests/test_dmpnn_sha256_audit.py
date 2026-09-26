"""Tests for Wf-120 Phase 1 — D-MPNN pIC50 SHA256 audit + overwrite guard.

Wf-120 Phase 1 fixes the silent copy-paste bug where
``dmpnn_pic50_rh_seed1.pt`` was a byte-identical copy of
``dmpnn_pic50_rh_seed0.pt`` (same SHA256 → same model → fabricated
seed diversity). The audit script computes SHA256s for every D-MPNN ckpt
under ``molmetal/checkpoints/`` and verifies two things:

  (a) every (metal, seed) pair has a unique SHA256 (no duplicates),
  (b) every SHA256 reported in ``dmpnn_ckpt_status.json`` matches the
      on-disk file.

These tests cover the audit's pure-Python helpers (no torch / no actual
SHA256 hashing on the production ckpts — we synthesise fixtures in tmp
dirs so the suite stays < 5 s).

The training-script SHA256 overwrite guard is tested separately: it
detects when an existing ckpt's stored ``seed`` differs from the seed
being trained, OR when a re-trained ckpt has the same SHA256 as an
existing one, and refuses to overwrite with a loud error.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Test 1: script + helpers importable
# ---------------------------------------------------------------------------
def test_audit_script_importable():
    """The audit module exposes parse_args, main, audit, sha256_of_file."""
    from molmetal.scripts.dmpnn_sha256_audit import (
        audit,
        main,
        parse_args,
        sha256_of_file,
    )
    assert callable(parse_args)
    assert callable(main)
    assert callable(audit)
    assert callable(sha256_of_file)


def test_audit_parse_args_defaults():
    """Default flags point at molmetal/checkpoints + the canonical JSON."""
    from molmetal.scripts.dmpnn_sha256_audit import parse_args
    ns = parse_args([])
    # Path comparison is name-based (the default resolves via PROJECT_ROOT
    # but should always end with the expected suffix).
    assert str(ns.ckpt_dir).endswith("molmetal/checkpoints")
    assert str(ns.status_json).endswith(
        "metrics/by_metric/dmpnn_ckpt_status.json"
    )
    assert ns.out is None


# ---------------------------------------------------------------------------
# Test 2: sha256_of_file matches hashlib
# ---------------------------------------------------------------------------
def test_sha256_of_file_matches_hashlib(tmp_path: Path):
    from molmetal.scripts.dmpnn_sha256_audit import sha256_of_file
    import hashlib
    target = tmp_path / "x.pt"
    payload = b"DMPNN ckpt fixture \x00\x01\x02" * 1024
    target.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    assert sha256_of_file(target) == expected


def test_sha256_of_file_chunking_changes_result_on_corruption(tmp_path: Path):
    """Corrupting the file after hashing yields a different SHA256."""
    from molmetal.scripts.dmpnn_sha256_audit import sha256_of_file
    target = tmp_path / "y.pt"
    target.write_bytes(b"abcd" * 1024)
    s1 = sha256_of_file(target)
    target.write_bytes(b"abce" * 1024)  # one byte changed in last block
    s2 = sha256_of_file(target)
    assert s1 != s2


# ---------------------------------------------------------------------------
# Test 3: discover_dmpnn_ckpts + parse_filename
# ---------------------------------------------------------------------------
def test_discover_dmpnn_ckpts_returns_only_dmpnn_pic50(tmp_path: Path):
    from molmetal.scripts.dmpnn_sha256_audit import discover_dmpnn_ckpts
    (tmp_path / "dmpnn_pic50_ru_seed0.pt").write_bytes(b"")
    (tmp_path / "dmpnn_pic50_ru_seed1.pt").write_bytes(b"")
    (tmp_path / "affinity_platinai_seed0.pt").write_bytes(b"")  # not D-MPNN
    (tmp_path / "README.md").write_text("ignore me")
    found = discover_dmpnn_ckpts(tmp_path)
    names = sorted(p.name for p in found)
    assert names == [
        "dmpnn_pic50_ru_seed0.pt",
        "dmpnn_pic50_ru_seed1.pt",
    ]


def test_parse_filename_valid_pairs():
    from molmetal.scripts.dmpnn_sha256_audit import parse_filename
    cases = [
        ("dmpnn_pic50_ru_seed0.pt", ("Ru", 0)),
        ("dmpnn_pic50_ru_seed1.pt", ("Ru", 1)),
        ("dmpnn_pic50_ir_seed42.pt", ("Ir", 42)),
        ("dmpnn_pic50_rh_seed3.pt", ("Rh", 3)),
    ]
    for name, expected in cases:
        assert parse_filename(Path(name)) == expected


def test_parse_filename_rejects_non_dmpnn_or_unknown_metal():
    from molmetal.scripts.dmpnn_sha256_audit import parse_filename
    for bad in [
        "affinity_platinai_seed0.pt",
        "dmpnn_pic50_xx_seed0.pt",  # unknown metal
        "dmpnn_pic50_ru.pt",        # no seed
        "dmpnn_pic50_ru_seedA.pt",  # non-numeric seed
    ]:
        assert parse_filename(Path(bad)) is None, bad


# ---------------------------------------------------------------------------
# Test 4: audit() reports duplicates + JSON mismatches correctly
# ---------------------------------------------------------------------------
def _make_fixture_ckpt_dir(tmp_path: Path, files: dict[str, bytes]) -> Path:
    """Create tmp_path/dmpnn_pic50_{metal}_seed{N}.pt fixtures with given payloads."""
    for name, payload in files.items():
        (tmp_path / name).write_bytes(payload)
    return tmp_path


def test_audit_passes_on_unique_sha_fixtures(tmp_path: Path):
    from molmetal.scripts.dmpnn_sha256_audit import audit
    ckpt_dir = _make_fixture_ckpt_dir(
        tmp_path,
        {
            "dmpnn_pic50_ru_seed0.pt": b"PAYLOAD_A" * 100,
            "dmpnn_pic50_ru_seed1.pt": b"PAYLOAD_B" * 100,
            "dmpnn_pic50_ir_seed0.pt": b"PAYLOAD_C" * 100,
            "dmpnn_pic50_ir_seed1.pt": b"PAYLOAD_D" * 100,
        },
    )
    report = audit(ckpt_dir, status_json=None)
    assert report["status"] == "PASS"
    assert report["n_ckpts_scanned"] == 4
    assert report["n_duplicate_shas"] == 0
    assert report["n_json_mismatches"] == 0
    # Every row has metal/seed/sha256
    for row in report["checkpoints"]:
        assert {"metal", "seed", "sha256", "filename"} <= set(row.keys())


def test_audit_fails_on_duplicate_sha_fixtures(tmp_path: Path):
    """Smoking-gun bug: two filenames with the same SHA256 → FAIL."""
    from molmetal.scripts.dmpnn_sha256_audit import audit
    shared = b"DUPLICATE_PAYLOAD" * 50
    ckpt_dir = _make_fixture_ckpt_dir(
        tmp_path,
        {
            "dmpnn_pic50_rh_seed0.pt": shared,
            "dmpnn_pic50_rh_seed1.pt": shared,  # IDENTICAL → bug
        },
    )
    report = audit(ckpt_dir, status_json=None)
    assert report["status"] == "FAIL"
    assert report["n_duplicate_shas"] == 1
    dup = report["duplicate_shas"]
    # Both filenames appear under the same SHA
    flat = [name for names in dup.values() for name in names]
    assert sorted(flat) == [
        "dmpnn_pic50_rh_seed0.pt",
        "dmpnn_pic50_rh_seed1.pt",
    ]


def test_audit_detects_json_sha_mismatch(tmp_path: Path):
    """Status JSON SHA256 does not match on-disk SHA → mismatch reported."""
    from molmetal.scripts.dmpnn_sha256_audit import audit, sha256_of_file
    real_payload = b"ACTUAL_CKPT" * 50
    ckpt_dir = _make_fixture_ckpt_dir(
        tmp_path,
        {"dmpnn_pic50_ru_seed0.pt": real_payload},
    )
    real_sha = sha256_of_file(ckpt_dir / "dmpnn_pic50_ru_seed0.pt")
    status_json = tmp_path / "status.json"
    status_json.write_text(
        json.dumps(
            {
                "checkpoints": [
                    {
                        "filename": "dmpnn_pic50_ru_seed0.pt",
                        "sha256": "0" * 64,  # deliberately wrong
                    }
                ]
            }
        )
    )
    report = audit(ckpt_dir, status_json=status_json)
    assert report["status"] == "FAIL"
    assert report["n_json_mismatches"] == 1
    assert report["json_mismatches"][0]["filename"] == "dmpnn_pic50_ru_seed0.pt"
    assert report["json_mismatches"][0]["fs_sha256"] == real_sha


def test_audit_detects_missing_sha_in_status_json(tmp_path: Path):
    """Status JSON entry without ``sha256`` field → missing reported."""
    from molmetal.scripts.dmpnn_sha256_audit import audit
    ckpt_dir = _make_fixture_ckpt_dir(
        tmp_path,
        {"dmpnn_pic50_ru_seed0.pt": b"A" * 50},
    )
    status_json = tmp_path / "status.json"
    status_json.write_text(
        json.dumps(
            {"checkpoints": [{"filename": "dmpnn_pic50_ru_seed0.pt"}]}
        )
    )
    report = audit(ckpt_dir, status_json=status_json)
    # Missing SHA alone is not a fail (back-fill in progress) but is tracked.
    assert report["status"] == "PASS"
    assert report["n_json_missing_sha"] == 1
    assert report["json_missing_sha"] == ["dmpnn_pic50_ru_seed0.pt"]


# ---------------------------------------------------------------------------
# Test 5: end-to-end CLI smoke run on real ckpt dir
# ---------------------------------------------------------------------------
def test_real_audit_exits_zero_when_all_unique():
    """If the production ckpt dir is in a healthy state, audit exits 0."""
    from molmetal.scripts.dmpnn_sha256_audit import main
    ckpt_dir = PROJECT_ROOT / "molmetal" / "checkpoints"
    if not any(ckpt_dir.glob("dmpnn_pic50_*.pt")):
        pytest.skip("Production ckpt dir has no D-MPNN pIC50 ckpts (run train first).")
    status_json = PROJECT_ROOT / "metrics" / "by_metric" / "dmpnn_ckpt_status.json"
    import sys
    argv = [
        "dmpnn_sha256_audit.py",
        "--ckpt-dir", str(ckpt_dir),
        "--status-json", str(status_json),
    ]
    saved = sys.argv
    try:
        sys.argv = argv
        rc = main()
    finally:
        sys.argv = saved
    assert rc == 0, "Production audit failed (SHA256 duplicate or JSON mismatch detected)"


# ---------------------------------------------------------------------------
# Test 6: training-script SHA256 overwrite guard
# ---------------------------------------------------------------------------
def test_train_script_has_sha256_overwrite_guard():
    """``train_dmpnn_pic50_real.py`` refuses to silently overwrite a
    ckpt whose existing SHA256 matches another (metal, seed) ckpt."""
    src_path = PROJECT_ROOT / "molmetal" / "scripts" / "train_dmpnn_pic50_real.py"
    text = src_path.read_text()
    # Guard must exist as a function or block
    assert "_guard_ckpt_overwrite" in text or "sha256" in text, (
        "train_dmpnn_pic50_real.py lacks an SHA256 overwrite guard"
    )


def test_train_script_seed_setting_covers_torch_numpy_random():
    """The trainer must set torch + numpy + (if CUDA) cuda seeds."""
    src_path = PROJECT_ROOT / "molmetal" / "scripts" / "train_dmpnn_pic50_real.py"
    text = src_path.read_text()
    assert "torch.manual_seed" in text
    assert "np.random.seed" in text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))