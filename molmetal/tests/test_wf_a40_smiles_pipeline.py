"""test_wf_a40_smiles_pipeline.py — regression tests for Wf-A40 fixup-1 (F1
SMILES fetcher) + fixup-2 (F2 integrator). Closes the completeness-critic
NEW-3 gap: ``grep -rn "wf_a40" molmetal/tests/`` was empty after fixup-2,
so a regression in fetcher/integrator could only be caught by re-running
the scripts end-to-end against the PlatinAI server. These tests verify
the pure helpers + §5.6 atomic-write wiring without touching network,
GPU, Java, or the live API.

CPU-only.  No fixtures beyond ``tmp_path``.  Lives in
``molmetal/tests/`` so ``pytest`` discovers it as
``molmetal/tests/test_wf_a40_smiles_pipeline.py`` per
``pyproject.toml`` defaults.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Module-level smoke — both modules must import cleanly
# ---------------------------------------------------------------------------
class TestImportsClean:
    def test_fetcher_imports_without_network(self):
        """F1 (fetcher) must import + set up ``no_proxy`` *=" without
        hitting the network. The module sets
        ``os.environ["no_proxy"] = "*"`` BEFORE importing ``requests``
        per the dead-local-proxy Wf-A37 fix.
        """
        from molmetal.scripts import wf_a40_smiles_fetcher as f

        assert hasattr(f, "load_target_imageids")
        assert hasattr(f, "load_seed_cache")
        assert hasattr(f, "load_existing_cache")
        assert hasattr(f, "write_cache")
        assert hasattr(f, "iter_pages")
        # Proxy bypass shipped (Wf-A37 verification).
        assert f._PROXIES == {"http": None, "https": None}

    def test_integrator_imports_without_disk_state(self):
        """F2 (integrator) must import even when the live cache/pool
        CSV are absent — ``main()`` is what requires the on-disk state,
        the import path does not.
        """
        from molmetal.scripts import wf_a40_smiles_integrator as i

        assert hasattr(i, "_write_atomic")
        assert hasattr(i, "MISSING_MARKER")
        assert i.MISSING_MARKER == "__no_smiles__"
        # §5.6 helper must be importable in the test venv.
        assert i._HAS_ATOMIC is True


# ---------------------------------------------------------------------------
# F1 — fetcher pure-function helpers
# ---------------------------------------------------------------------------
class TestFetcherHelpers:
    def test_load_target_imageids_strips_and_dedupes(self, tmp_path: Path):
        from molmetal.scripts.wf_a40_smiles_fetcher import load_target_imageids

        csv = tmp_path / "pool.csv"
        csv.write_text(
            "compound_code,smiles,label\n"
            "MB-Pt-001,O,active\n"
            "MB-Pt-002,,active\n"
            ",,junk-row\n"  # empty code → ignored
            "MB-Pt-001,O,active\n"  # duplicate → dedup
            "MB-Pt-003,Cl,active\n",
            encoding="utf-8",
        )
        out = load_target_imageids(csv)
        assert out == {"MB-Pt-001", "MB-Pt-002", "MB-Pt-003"}

    def test_load_seed_cache_pair_only(self, tmp_path: Path):
        from molmetal.scripts.wf_a40_smiles_fetcher import load_seed_cache

        csv = tmp_path / "seed.csv"
        csv.write_text(
            "imageId,smiles\n"
            "MB-Pt-001,O\n"
            "MB-Pt-002,\n"  # empty smiles → skipped
            ",N\n"  # empty imageId → skipped
            "MB-Pt-003,Cl\n",
            encoding="utf-8",
        )
        out = load_seed_cache(csv)
        assert out == {"MB-Pt-001": "O", "MB-Pt-003": "Cl"}

    def test_load_existing_cache_handles_missing_corrupt_valid(
        self, tmp_path: Path
    ):
        from molmetal.scripts.wf_a40_smiles_fetcher import load_existing_cache

        # Missing → empty.
        assert load_existing_cache(tmp_path / "absent.json") == {}

        # Corrupt → empty (no exception).
        bad = tmp_path / "bad.json"
        bad.write_text("{not json", encoding="utf-8")
        assert load_existing_cache(bad) == {}

        # Valid → coerced to {str: str}. The implementation drops
        # entries whose value is falsy (empty string); non-empty
        # whitespace is preserved as-is.
        good = tmp_path / "good.json"
        good.write_text(
            json.dumps({"MB-Pt-001": "O", "MB-Pt-002": "Cl", "MB-Pt-003": ""}),
            encoding="utf-8",
        )
        out = load_existing_cache(good)
        assert out == {"MB-Pt-001": "O", "MB-Pt-002": "Cl"}

    def test_iter_pages_is_exclusive_end(self):
        from molmetal.scripts.wf_a40_smiles_fetcher import iter_pages

        assert list(iter_pages(start_page=850, max_pages=3)) == [850, 851, 852]
        # max_pages=0 → empty (no scan).
        assert list(iter_pages(start_page=850, max_pages=0)) == []

    def test_write_cache_lands_atomically_and_round_trips(
        self, tmp_path: Path
    ):
        """§5.6 atomic-write: cache JSON must appear on disk with
        matching contents after :func:`write_cache`.
        """
        from molmetal.scripts.wf_a40_smiles_fetcher import (
            load_existing_cache,
            write_cache,
        )

        out = tmp_path / "cache.json"
        cache = {"MB-Pt-001": "O", "MB-Pt-002": "Cl"}
        write_cache(cache, out)
        assert out.exists(), "§5.6: cache must land on disk"
        assert json.loads(out.read_text(encoding="utf-8")) == cache
        # Round-trip through the resume helper preserves the values.
        assert load_existing_cache(out) == cache


# ---------------------------------------------------------------------------
# F2 — integrator atomic-write helper (§5.6)
# ---------------------------------------------------------------------------
class TestIntegratorAtomicWrite:
    def test_write_atomic_creates_file(self, tmp_path: Path):
        from molmetal.scripts.wf_a40_smiles_integrator import _write_atomic

        out = tmp_path / "atomic.txt"
        _write_atomic(out, "hello\nworld\n")
        assert out.read_text(encoding="utf-8") == "hello\nworld\n"

    def test_write_atomic_overwrites_existing(self, tmp_path: Path):
        from molmetal.scripts.wf_a40_smiles_integrator import _write_atomic

        out = tmp_path / "atomic.txt"
        out.write_text("stale", encoding="utf-8")
        _write_atomic(out, "fresh")
        assert out.read_text(encoding="utf-8") == "fresh"

    def test_integrator_main_uses_atomic_helper(self):
        """Source-grep: F2 ``main()`` must route both writes through the
        §5.6 atomic helper, not :meth:`Path.write_text` directly. Skips
        actually running ``main()`` (which needs the live CSVs on disk).
        """
        from pathlib import Path as P

        from molmetal.scripts import wf_a40_smiles_integrator as i

        src = P(i.__file__).read_text(encoding="utf-8")
        assert "_write_atomic(OUT_CSV, buf.getvalue())" in src
        assert "_write_atomic(META_JSON, json.dumps(meta, indent=2))" in src
        # No direct .write_text on the integrated output paths.
        assert "OUT_CSV.write_text" not in src
        assert "META_JSON.write_text" not in src


# ---------------------------------------------------------------------------
# CLI surface — fetcher --help (proves parse_args does not crash)
# ---------------------------------------------------------------------------
class TestFetcherCLI:
    def test_fetcher_help_exits_clean(self, capsys: pytest.CaptureFixture):
        """``--help`` exercises the parser without doing network IO.
        argparse calls ``sys.exit(0)`` on ``--help`` — that's the
        success path; the test asserts that exit. A regression in
        flag plumbing (typo / missing default) would surface as a
        non-zero exit or an argparse error.
        """
        from molmetal.scripts import wf_a40_smiles_fetcher as f

        with mock.patch.object(f, "_post_get_substances") as _:
            with pytest.raises(SystemExit) as excinfo:
                f.main(["--help"])
        assert excinfo.value.code == 0
        captured = capsys.readouterr().out + capsys.readouterr().err
        # Description must show in --help so users know what they run.
        assert "Wf-A40" in captured
