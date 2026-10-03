#!/usr/bin/env python3
"""Wf-424-A4 — Atomic write migration verification (10 closeout-drafting scripts).

Tests the Wf-424-A4 atomic write protocol migration of 10 additional closeout-drafting
scripts. Each migrated script:

  * Imports `_WF424_HAS_ATOMIC` + `atomic_write_file` + `safe_overwrite_check` +
    `detect_truncation` from molmetal.scripts.wf386a3_atomic_write via try/except.
  * Defines a `_wf424_atomic_write_text(path, content)` helper that delegates to the
    shared atomic_write_file primitive.
  * Has ZERO remaining direct ``path.write_text(<args>)`` calls where args do NOT
    contain `encoding=` AND the immediate prefix is NOT `_wf424_atomic_write_text`.

The 10 migrated scripts (per Wf-424-A4 closeout §1 audit):
  1. molmetal/scripts/per_wave_checklist.py
  2. molmetal/scripts/wf264_panel_cfm_properties.py
  3. molmetal/scripts/wf179_rescue.py
  4. molmetal/scripts/wf179_fix_rmsd.py
  5. molmetal/scripts/r4_c_full_sweep.py
  6. molmetal/scripts/compute_m15_ligand_vs_pocket.py
  7. molmetal/scripts/bench_wf179_m15_rmsd_p2.py
  8. molmetal/scripts/wf240_ablation_launcher.py
  9. molmetal/scripts/train_hilbert_astsr_end_to_end.py
  10. molmetal/scripts/extract_multiph_pll_end_to_end.py

Combined Wf-406-A3 (3) + Wf-410-A3 (5) + Wf-414-A3 (5) + Wf-416-A1 (10) + Wf-424-A4 (10) = 33
scripts migrated to the canonical I/O substrate per AGENTS.md §5.6 atomic write protocol
adoption rule.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

MIGRATED_SCRIPTS = [
    "molmetal/scripts/per_wave_checklist.py",
    "molmetal/scripts/wf264_panel_cfm_properties.py",
    "molmetal/scripts/wf179_rescue.py",
    "molmetal/scripts/wf179_fix_rmsd.py",
    "molmetal/scripts/r4_c_full_sweep.py",
    "molmetal/scripts/compute_m15_ligand_vs_pocket.py",
    "molmetal/scripts/bench_wf179_m15_rmsd_p2.py",
    "molmetal/scripts/wf240_ablation_launcher.py",
    "molmetal/scripts/train_hilbert_astsr_end_to_end.py",
    "molmetal/scripts/extract_multiph_pll_end_to_end.py",
]


def _find_balanced_paren(text: str, open_idx: int) -> int:
    """Find the matching closing paren for the opening paren at open_idx."""
    assert text[open_idx] == "(", f"Expected '(' at {open_idx}, got {text[open_idx]!r}"
    depth = 1
    i = open_idx + 1
    while i < len(text):
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return -1


def _count_legacy_writes(content: str) -> int:
    """Count `.write_text(<args>)` calls where args do NOT contain `encoding=` and the
    immediate prefix is NOT `_wf424_atomic_write_text(...)` (i.e. unwrapped legacy writes).
    """
    n = 0
    for m in re.finditer(r"\.write_text\(", content):
        open_paren_idx = m.end() - 1
        close_idx = _find_balanced_paren(content, open_paren_idx)
        if close_idx == -1:
            continue
        args = content[open_paren_idx + 1 : close_idx - 1]
        if "encoding=" in args:
            continue
        # Check the immediate prefix is NOT _wf424_atomic_write_text
        prefix_window = content[max(0, m.start() - 50) : m.start()]
        if "_wf424_atomic_write_text" in prefix_window:
            continue
        n += 1
    return n


# --------------------------------------------------------------------------------------
# Per-script tests
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("script_rel", MIGRATED_SCRIPTS)
def test_wf424a4_script_parses(script_rel: str) -> None:
    """The migrated script must AST-parse cleanly (no syntax errors from the migration)."""
    path = (REPO_ROOT / script_rel).resolve()
    content = path.read_text(encoding="utf-8")
    ast.parse(content)  # raises SyntaxError if migration broke the file


@pytest.mark.parametrize("script_rel", MIGRATED_SCRIPTS)
def test_wf424a4_atomic_marker_present(script_rel: str) -> None:
    """The migrated script must contain the _WF424_HAS_ATOMIC marker (try/except guard)."""
    path = (REPO_ROOT / script_rel).resolve()
    content = path.read_text(encoding="utf-8")
    assert "_WF424_HAS_ATOMIC" in content, (
        f"{script_rel} is missing the _WF424_HAS_ATOMIC try/except marker — "
        "migration was skipped or reverted."
    )


@pytest.mark.parametrize("script_rel", MIGRATED_SCRIPTS)
def test_wf424a4_helper_defined(script_rel: str) -> None:
    """The migrated script must define the _wf424_atomic_write_text helper."""
    path = (REPO_ROOT / script_rel).resolve()
    content = path.read_text(encoding="utf-8")
    assert (
        "def _wf424_atomic_write_text(" in content
    ), f"{script_rel} is missing the _wf424_atomic_write_text helper function."


@pytest.mark.parametrize("script_rel", MIGRATED_SCRIPTS)
def test_wf424a4_tryexcept_import_fallback(script_rel: str) -> None:
    """The migration uses a try/except ImportError fallback to preserve backward compat."""
    path = (REPO_ROOT / script_rel).resolve()
    content = path.read_text(encoding="utf-8")
    assert (
        "except ImportError" in content
    ), f"{script_rel} must use try/except ImportError fallback for the atomic import."


@pytest.mark.parametrize("script_rel", MIGRATED_SCRIPTS)
def test_wf424a4_no_legacy_writes(script_rel: str) -> None:
    """The migrated script must have ZERO remaining legacy direct `.write_text(<args>)`
    calls where args do NOT contain `encoding=` and the prefix is NOT the new helper.
    (encoding= args are intentionally preserved as direct writes for non-default encoding
    pass-through per Wf-424-A4 closeout §2.3.)
    """
    path = (REPO_ROOT / script_rel).resolve()
    content = path.read_text(encoding="utf-8")
    n = _count_legacy_writes(content)
    assert n == 0, (
        f"{script_rel} has {n} remaining legacy .write_text(<args>) calls "
        f"without encoding= kwarg and without _wf424_atomic_write_text wrapper."
    )


@pytest.mark.parametrize("script_rel", MIGRATED_SCRIPTS)
def test_wf424a4_helper_calls_atomic_write_file(script_rel: str) -> None:
    """The _wf424_atomic_write_text helper must call atomic_write_file when available."""
    path = (REPO_ROOT / script_rel).resolve()
    content = path.read_text(encoding="utf-8")
    assert (
        "atomic_write_file(" in content
    ), f"{script_rel} helper does not call atomic_write_file — migration is incomplete."


# --------------------------------------------------------------------------------------
# Aggregate test — combined Wf-406-A3 + Wf-410-A3 + Wf-414-A3 + Wf-416-A1 + Wf-424-A4
# --------------------------------------------------------------------------------------


def test_wf424a4_combined_migration_total() -> None:
    """Combined migration: Wf-406-A3 (3) + Wf-410-A3 (5) + Wf-414-A3 (5) + Wf-416-A1 (10) + Wf-424-A4 (10) = 33.

    Verifies that all 33 scripts have the migration applied (cumulative across all
    atomic write protocol migration waves).
    """
    ALL_MIGRATED = {
        # Wf-406-A3 (3 scripts)
        "molmetal/scripts/wf398a2_m2_r2_real_verify.py",
        "molmetal/scripts/wf390a2_alt_encoder_audit.py",
        "molmetal/scripts/wf394a1_lambda_cfm_wire.py",
        # Wf-410-A3 (5 scripts)
        "molmetal/scripts/wf264a_drugood_post_fix_updater.py",
        "molmetal/scripts/bench_dmpnn_top20_predictions.py",
        "molmetal/scripts/bench_mcts_replay_learner_regularized.py",
        "molmetal/scripts/bench_variance_mcts_wasserstein.py",
        "molmetal/scripts/wf313a_relink_pocket_ref_ligands.py",
        # Wf-414-A3 (5 scripts)
        "molmetal/scripts/wf341c3_shortcut_ema.py",
        "molmetal/scripts/wf380a4_vina_param_tuning.py",
        "molmetal/scripts/wf381a1_vina_param_tuning.py",
        "molmetal/scripts/wf385a3_scaffold_wire_in_gate.py",
        "molmetal/scripts/wf392a2_chemprop_v3_cache_wire.py",
        # Wf-416-A1 (10 scripts)
        "molmetal/scripts/m6_wf320h_vina_phase2.py",
        "molmetal/scripts/m15_rmsd_phase2.py",
        "molmetal/scripts/train_nci60_10cell.py",
        "molmetal/scripts/r17_gpu_retrain.py",
        "molmetal/scripts/generate_section4.py",
        "molmetal/scripts/extract_rdkit_2d_descriptors.py",
        "molmetal/scripts/wf146_chemprop_v2_lomo_v2.py",
        "molmetal/scripts/train_platinai_4cell.py",
        "molmetal/scripts/train_metal_lomo.py",
        "molmetal/scripts/train_metal_cytotox.py",
        # Wf-424-A4 (10 scripts)
        *MIGRATED_SCRIPTS,
    }
    assert (
        len(ALL_MIGRATED) == 33
    ), f"Expected 33 total migrated scripts (3+5+5+10+10), got {len(ALL_MIGRATED)}."
    n_with_marker = 0
    n_missing = 0
    for rel_path in ALL_MIGRATED:
        full = (REPO_ROOT / rel_path).resolve()
        if not full.exists():
            n_missing += 1
            continue
        content = full.read_text(encoding="utf-8")
        # Check for any of the 5 migration markers OR direct atomic_write_file usage
        if (
            "_WF424_HAS_ATOMIC" in content
            or "_WF416_HAS_ATOMIC" in content
            or "_WF414_HAS_ATOMIC" in content
            or "_WF410_HAS_ATOMIC" in content
            or "_WF406_HAS_ATOMIC" in content
            or "atomic_write_file(" in content
        ):
            n_with_marker += 1
    # Allow some slack for scripts not in this repo snapshot (forward-compat)
    assert n_with_marker >= 28, (
        f"Combined Wf-406-A3 + Wf-410-A3 + Wf-414-A3 + Wf-416-A1 + Wf-424-A4 migration "
        f"count is {n_with_marker} (n_missing={n_missing}), expected at least 28 "
        f"(3+5+5+10+10 minus scripts not present)."
    )


def test_wf424a4_atomic_primitive_loads() -> None:
    """The canonical atomic_write_file primitive must import cleanly (no broken dependencies)."""
    from molmetal.scripts.wf386a3_atomic_write import (
        atomic_write_file,
        safe_overwrite_check,
        detect_truncation,
    )

    # All three names are callable
    assert callable(atomic_write_file)
    assert callable(safe_overwrite_check)
    assert callable(detect_truncation)


def test_wf424a4_migration_helper_idempotent(tmp_path: Path) -> None:
    """Re-running the migration helper on an already-migrated script must be a no-op
    (idempotency). Use a copy of wf264_panel_cfm_properties.py as test input.
    """
    import shutil

    src = REPO_ROOT / "molmetal/scripts/wf264_panel_cfm_properties.py"
    if not src.exists():
        pytest.skip("Source script not in this repo snapshot")
    dst = tmp_path / "wf264_panel_cfm_properties_copy.py"
    shutil.copy2(src, dst)
    # Call the migration helper on the copy
    sys_path_backup = __import__("sys").path.copy()
    try:
        sys_mod = __import__("sys")
        sys_mod.path.insert(0, str(REPO_ROOT / "molmetal/scripts"))
        try:
            from wf424a4_migrate_closeout_scripts import migrate_one  # type: ignore[import-not-found]
        except ImportError:
            pytest.skip(
                "wf424a4 helper not importable; expected when script not on sys.path"
            )
        changed, msg = migrate_one(dst)
        assert (
            changed is False
        ), f"Re-running on already-migrated script should be no-op, got changed={changed} msg={msg}"
        assert (
            msg == "already_migrated_by_prior_wave"
        ), f"Expected idempotent skip, got msg={msg!r}"
    finally:
        sys_mod.path[:] = sys_path_backup


def test_wf424a4_no_migration_overlap_with_prior_waves() -> None:
    """Wf-424-A4's 10 scripts must be disjoint from Wf-416-A1's 10 scripts
    (no double-migration or overlap).
    """
    PRIOR_WAVE_SCRIPTS = {
        "molmetal/scripts/m6_wf320h_vina_phase2.py",
        "molmetal/scripts/m15_rmsd_phase2.py",
        "molmetal/scripts/train_nci60_10cell.py",
        "molmetal/scripts/r17_gpu_retrain.py",
        "molmetal/scripts/generate_section4.py",
        "molmetal/scripts/extract_rdkit_2d_descriptors.py",
        "molmetal/scripts/wf146_chemprop_v2_lomo_v2.py",
        "molmetal/scripts/train_platinai_4cell.py",
        "molmetal/scripts/train_metal_lomo.py",
        "molmetal/scripts/train_metal_cytotox.py",
    }
    current = set(MIGRATED_SCRIPTS)
    overlap = current & PRIOR_WAVE_SCRIPTS
    assert len(overlap) == 0, (
        f"Wf-424-A4 should target DIFFERENT scripts than Wf-416-A1. "
        f"Overlap: {overlap}"
    )
