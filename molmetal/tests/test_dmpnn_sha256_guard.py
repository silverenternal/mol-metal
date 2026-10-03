"""Tests for Wf-120 Phase 3 — D-MPNN pIC50 SHA256 overwrite guard.

Wf-120 Phase 3 adds a *post-save* SHA256 overwrite guard to
``molmetal/scripts/train_dmpnn_pic50_real.py`` (the script that produced
the silent copy-paste bug for Rh seed0/seed1 in Wf-120 P0). The guard:

  1. Computes the SHA256 of the just-written ``dmpnn_pic50_*.pt`` file.
  2. Logs the SHA256 to stdout for the audit trail.
  3. Verifies the SHA256 is unique among every other D-MPNN pIC50 ckpt
     already on disk. If a sibling ckpt has the same SHA256, the just-
     written file is removed (rollback) and a ``RuntimeError`` is raised
     with a loud, descriptive message.

The guards live in :mod:`molmetal.scripts._dmpnn_sha256_guard` — a
deliberately dependency-free module (hashlib + pathlib only) so the
tests run on CPU in <1 s without pandas / torch / rdkit.

**Hardware**: CPU only.

**Test design** (10 tests)
--------------------------
1.  ``test_guard_module_imports`` — module surface is importable.
2.  ``test_sha256_of_file_matches_hashlib`` — chunked SHA matches stdlib.
3.  ``test_parse_dmpnn_filename_valid_and_invalid`` — accept/reject cases.
4.  ``test_pre_train_guard_silent_when_target_absent`` — first-time
    training is not blocked.
5.  ``test_pre_train_guard_silent_when_sha_unique`` — pre-existing but
    unique target passes.
6.  ``test_pre_train_guard_raises_on_duplicate_target`` — pre-existing
    target SHA matches a sibling → ``RuntimeError``.
7.  ``test_post_save_guard_returns_sha_on_unique_save`` — happy path:
    new SHA logged, function returns the hex digest.
8.  ``test_post_save_guard_raises_and_removes_on_collision`` — smoke
    simulation of the silent-copy bug: write a file whose SHA matches
    an existing sibling → guard raises AND removes the just-written
    file (rollback).
9.  ``test_post_save_guard_raises_without_remove_when_flag_false`` —
    ``remove_on_fail=False`` keeps the file for forensic inspection.
10. ``test_post_save_guard_raises_when_target_missing`` — caller forgot
    to torch.save first → ``FileNotFoundError``.
11. ``test_scan_all_ckpts_finds_no_duplicates_on_production`` — the
    real ``molmetal/checkpoints/`` directory has 10 unique SHAs (sanity
    check; skipped if no ckpts are present).
12. ``test_find_duplicate_shas_groups_identical_files`` — helper API.
"""

from __future__ import annotations

import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Helper: make fixture ckpts with named payloads
# ---------------------------------------------------------------------------
def _make_fixtures(ckpt_dir: Path, files: dict[str, bytes]) -> None:
    """Write ``{name: payload}`` as ``dmpnn_pic50_*.pt`` files under ``ckpt_dir``."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in files.items():
        (ckpt_dir / name).write_bytes(payload)


# ---------------------------------------------------------------------------
# 1. Module surface
# ---------------------------------------------------------------------------
def test_guard_module_imports() -> None:
    """Every guard helper is importable from the canonical location."""
    from molmetal.scripts._dmpnn_sha256_guard import (
        GUARD_TAG,
        find_duplicate_shas,
        parse_dmpnn_filename,
        post_save_sha256_overwrite_guard,
        pre_train_sha256_overwrite_guard,
        scan_all_ckpts,
        sha256_of_file,
        target_path,
    )

    assert GUARD_TAG == "[wf120-dmpnn-sha256-guard]"
    assert callable(sha256_of_file)
    assert callable(pre_train_sha256_overwrite_guard)
    assert callable(post_save_sha256_overwrite_guard)
    assert callable(find_duplicate_shas)
    assert callable(scan_all_ckpts)
    assert callable(parse_dmpnn_filename)
    assert callable(target_path)


# ---------------------------------------------------------------------------
# 2. sha256_of_file matches stdlib
# ---------------------------------------------------------------------------
def test_sha256_of_file_matches_hashlib(tmp_path: Path) -> None:
    """Chunked SHA256 must equal hashlib's single-shot SHA256."""
    import hashlib
    from molmetal.scripts._dmpnn_sha256_guard import sha256_of_file

    p = tmp_path / "ck.pt"
    # 5 MiB of repeating pattern — spans multiple 1 MiB chunks.
    payload = (b"DMPNN_PIC50_FIXTURE_PAYLOAD" * 200_000)[: 5 * 1024 * 1024]
    p.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    assert sha256_of_file(p) == expected


# ---------------------------------------------------------------------------
# 3. parse_dmpnn_filename valid + invalid cases
# ---------------------------------------------------------------------------
def test_parse_dmpnn_filename_valid_and_invalid() -> None:
    from molmetal.scripts._dmpnn_sha256_guard import parse_dmpnn_filename

    valid = [
        ("dmpnn_pic50_ru_seed0.pt", ("Ru", 0)),
        ("dmpnn_pic50_ir_seed1.pt", ("Ir", 1)),
        ("dmpnn_pic50_rh_seed42.pt", ("Rh", 42)),
        ("dmpnn_pic50_re_seed4.pt", ("Re", 4)),
    ]
    for name, expected in valid:
        assert parse_dmpnn_filename(Path(name)) == expected

    invalid = [
        "affinity_platinai_seed0.pt",
        "dmpnn_pic50_xx_seed0.pt",  # unknown metal
        "dmpnn_pic50_ru.pt",  # no seed suffix
        "dmpnn_pic50_ru_seedA.pt",  # non-numeric seed
        "dmpnn_pic50_ru_seed.pt",  # empty seed
    ]
    for bad in invalid:
        assert parse_dmpnn_filename(Path(bad)) is None, bad


# ---------------------------------------------------------------------------
# 4. Pre-train guard: first-time training is not blocked
# ---------------------------------------------------------------------------
def test_pre_train_guard_silent_when_target_absent(tmp_path: Path) -> None:
    """If no target ckpt exists yet, the guard returns silently."""
    from molmetal.scripts._dmpnn_sha256_guard import (
        pre_train_sha256_overwrite_guard,
    )

    # Empty dir — no target, no siblings
    pre_train_sha256_overwrite_guard(metal="Ru", seed=0, ckpt_dir=tmp_path)
    pre_train_sha256_overwrite_guard(metal="Rh", seed=1, ckpt_dir=tmp_path)


# ---------------------------------------------------------------------------
# 5. Pre-train guard: pre-existing unique target passes
# ---------------------------------------------------------------------------
def test_pre_train_guard_silent_when_sha_unique(tmp_path: Path) -> None:
    """If target exists and its SHA is unique, the guard returns silently."""
    from molmetal.scripts._dmpnn_sha256_guard import (
        pre_train_sha256_overwrite_guard,
    )

    _make_fixtures(
        tmp_path,
        {
            "dmpnn_pic50_ru_seed0.pt": b"RU_SEED0" * 100,
            "dmpnn_pic50_ru_seed1.pt": b"RU_SEED1_DIFFERENT" * 100,
            "dmpnn_pic50_ir_seed0.pt": b"IR_SEED0" * 100,
        },
    )
    # Target is rh_seed0 (does not exist), no pre-train block
    pre_train_sha256_overwrite_guard(metal="Rh", seed=0, ckpt_dir=tmp_path)
    # Target is ru_seed0 (exists, SHA unique vs siblings) → pass
    pre_train_sha256_overwrite_guard(metal="Ru", seed=0, ckpt_dir=tmp_path)


# ---------------------------------------------------------------------------
# 6. Pre-train guard raises on duplicate target SHA
# ---------------------------------------------------------------------------
def test_pre_train_guard_raises_on_duplicate_target(tmp_path: Path) -> None:
    """The original Wf-120 P0 bug: rh_seed0.pt == rh_seed1.pt bytes."""
    from molmetal.scripts._dmpnn_sha256_guard import (
        pre_train_sha256_overwrite_guard,
    )

    shared = b"DUPLICATE_PAYLOAD_THAT_REPRESENTS_THE_SILENT_COPY_BUG" * 50
    _make_fixtures(
        tmp_path,
        {
            "dmpnn_pic50_rh_seed0.pt": shared,
            "dmpnn_pic50_rh_seed1.pt": shared,  # byte-identical to seed0
        },
    )
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        pre_train_sha256_overwrite_guard(metal="Rh", seed=1, ckpt_dir=tmp_path)


# ---------------------------------------------------------------------------
# 7. Post-save guard: happy path returns the SHA + logs to stdout
# ---------------------------------------------------------------------------
def test_post_save_guard_returns_sha_on_unique_save(tmp_path: Path) -> None:
    """Saving a brand-new unique ckpt returns the SHA and logs it."""
    from molmetal.scripts._dmpnn_sha256_guard import (
        post_save_sha256_overwrite_guard,
        sha256_of_file,
        GUARD_TAG,
    )

    _make_fixtures(
        tmp_path,
        {
            "dmpnn_pic50_ru_seed0.pt": b"RU0",
            "dmpnn_pic50_ru_seed1.pt": b"RU1",
        },
    )
    # Now simulate training rh_seed0 — write a new unique ckpt
    new_ckpt = tmp_path / "dmpnn_pic50_rh_seed0.pt"
    payload = b"RH0_FRESH_PAYLOAD" * 100
    new_ckpt.write_bytes(payload)

    # Capture stdout so we can assert the SHA is logged
    buf = io.StringIO()
    with redirect_stdout(buf):
        new_sha = post_save_sha256_overwrite_guard(
            metal="Rh",
            seed=0,
            ckpt_dir=tmp_path,
        )

    expected_sha = sha256_of_file(new_ckpt)
    assert new_sha == expected_sha

    out = buf.getvalue()
    assert GUARD_TAG in out
    assert "dmpnn_pic50_rh_seed0.pt" in out
    assert expected_sha in out


# ---------------------------------------------------------------------------
# 8. Post-save guard: raises + removes the just-written file on collision
# ---------------------------------------------------------------------------
def test_post_save_guard_raises_and_removes_on_collision(tmp_path: Path) -> None:
    """Simulate the silent-copy bug: re-write a file whose SHA matches
    an existing sibling. The guard must raise AND remove the just-
    written file (default ``remove_on_fail=True``)."""
    from molmetal.scripts._dmpnn_sha256_guard import (
        GUARD_TAG,
        post_save_sha256_overwrite_guard,
    )

    sibling_payload = b"EXISTING_SIBLING_PAYLOAD" * 200
    _make_fixtures(
        tmp_path,
        {
            "dmpnn_pic50_rh_seed0.pt": sibling_payload,
            # rh_seed1.pt is the *just-written* file that should collide
        },
    )

    target = tmp_path / "dmpnn_pic50_rh_seed1.pt"
    target.write_bytes(sibling_payload)  # byte-identical to rh_seed0
    assert target.exists()

    buf = io.StringIO()
    with redirect_stdout(buf):
        with pytest.raises(RuntimeError, match="refusing to keep"):
            post_save_sha256_overwrite_guard(
                metal="Rh",
                seed=1,
                ckpt_dir=tmp_path,
                # default remove_on_fail=True
            )

    # Rollback: the just-written file is removed
    assert not target.exists(), "post-save guard removed the colliding file (rollback)"

    # Sibling must still be intact
    sibling = tmp_path / "dmpnn_pic50_rh_seed0.pt"
    assert sibling.exists()
    assert sibling.read_bytes() == sibling_payload

    # Audit log mentions both filenames + the SHA collision
    out = buf.getvalue()
    assert GUARD_TAG in out
    assert "dmpnn_pic50_rh_seed1.pt" in out
    assert "dmpnn_pic50_rh_seed0.pt" in out
    assert "rollback" in out


# ---------------------------------------------------------------------------
# 9. Post-save guard: leave the file in place when remove_on_fail=False
# ---------------------------------------------------------------------------
def test_post_save_guard_raises_without_remove_when_flag_false(tmp_path: Path) -> None:
    """With ``remove_on_fail=False`` the just-written file is preserved
    so a forensic inspection can examine it."""
    from molmetal.scripts._dmpnn_sha256_guard import (
        post_save_sha256_overwrite_guard,
    )

    shared = b"COLLIDING_PAYLOAD_FOR_FORENSIC_TEST" * 50
    _make_fixtures(
        tmp_path,
        {
            "dmpnn_pic50_rh_seed0.pt": shared,
        },
    )
    target = tmp_path / "dmpnn_pic50_rh_seed1.pt"
    target.write_bytes(shared)

    with pytest.raises(RuntimeError, match="refusing to keep"):
        post_save_sha256_overwrite_guard(
            metal="Rh",
            seed=1,
            ckpt_dir=tmp_path,
            remove_on_fail=False,
        )

    # File is preserved for forensic inspection
    assert target.exists()
    assert target.read_bytes() == shared


# ---------------------------------------------------------------------------
# 10. Post-save guard: caller forgot to torch.save first
# ---------------------------------------------------------------------------
def test_post_save_guard_raises_when_target_missing(tmp_path: Path) -> None:
    """If the target ckpt does not exist on disk, the guard raises
    ``FileNotFoundError`` (defensive — caller bug, not a silent bug)."""
    from molmetal.scripts._dmpnn_sha256_guard import (
        post_save_sha256_overwrite_guard,
    )

    # Empty dir → no target on disk
    with pytest.raises(FileNotFoundError, match="does not exist"):
        post_save_sha256_overwrite_guard(
            metal="Rh",
            seed=0,
            ckpt_dir=tmp_path,
        )


# ---------------------------------------------------------------------------
# 11. scan_all_ckpts on the production ckpt dir (sanity)
# ---------------------------------------------------------------------------
def test_scan_all_ckpts_finds_no_duplicates_on_production() -> None:
    """End-to-end sanity check: the production ``molmetal/checkpoints/``
    has 10 (or 25) byte-unique D-MPNN pIC50 ckpts (Wf-120 P1+P2)."""
    from molmetal.scripts._dmpnn_sha256_guard import scan_all_ckpts

    prod_ckpt_dir = PROJECT_ROOT / "molmetal" / "checkpoints"
    if not any(prod_ckpt_dir.glob("dmpnn_pic50_*.pt")):
        pytest.skip(
            "Production ckpt dir has no D-MPNN pIC50 ckpts "
            "(run Wf-120 P1 / P2 train first)."
        )

    report = scan_all_ckpts(prod_ckpt_dir)
    assert report["status"] == "PASS"
    assert report["n_duplicates"] == 0
    assert report["guard_would_pass"] is True
    assert report["n_ckpts"] >= 1  # at least one ckpt present
    # Every row has the schema we expect
    for row in report["rows"]:
        assert {"filename", "metal", "seed", "sha256", "size_bytes"} <= set(row.keys())


# ---------------------------------------------------------------------------
# 12. find_duplicate_shas helper
# ---------------------------------------------------------------------------
def test_find_duplicate_shas_groups_identical_files(tmp_path: Path) -> None:
    """Two files with identical bytes → one duplicate group."""
    from molmetal.scripts._dmpnn_sha256_guard import find_duplicate_shas

    shared = b"SAME_PAYLOAD_TWICE" * 30
    _make_fixtures(
        tmp_path,
        {
            "dmpnn_pic50_ru_seed0.pt": shared,
            "dmpnn_pic50_ru_seed1.pt": shared,
            "dmpnn_pic50_ir_seed0.pt": b"DIFFERENT" * 30,
            # Non-D-MPNN file — must be ignored
            "astra_platinai_seed0.pt": shared,
            "README.md": b"hello",
        },
    )

    dupes = find_duplicate_shas(sorted(tmp_path.iterdir()))
    assert len(dupes) == 1
    sha, files = dupes[0]
    assert sorted(files) == [
        "dmpnn_pic50_ru_seed0.pt",
        "dmpnn_pic50_ru_seed1.pt",
    ]
    # SHA is a valid 64-hex digest
    assert len(sha) == 64 and all(c in "0123456789abcdef" for c in sha)


def test_find_duplicate_shas_empty_when_all_unique(tmp_path: Path) -> None:
    """All-unique files → no duplicate groups."""
    from molmetal.scripts._dmpnn_sha256_guard import find_duplicate_shas

    _make_fixtures(
        tmp_path,
        {
            "dmpnn_pic50_ru_seed0.pt": b"AAA",
            "dmpnn_pic50_ru_seed1.pt": b"BBB",
            "dmpnn_pic50_ir_seed0.pt": b"CCC",
        },
    )
    assert find_duplicate_shas(sorted(tmp_path.iterdir())) == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
