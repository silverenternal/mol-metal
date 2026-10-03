"""Tests for Wf-410-A3: atomic write protocol migration of top-5 closeout-drafting scripts.

================================================================
Wf-410-A3 closes the AGENTS.md §5.6 adoption sequence step 3 (per
Wf-386-A3 §6.1): migrates 5 additional closeout-drafting scripts from
direct Write to ``atomic_write_file`` + ``safe_overwrite_check`` +
``detect_truncation``. The 3 previously-migrated scripts (Wf-398-A2 +
Wf-390-A2 + Wf-394-A1) live in `test_wf406a3_atomic_migration.py`;
Wf-410-A3 extends the migration to the next top-5 most-used scripts.

The 5 migrated scripts in this wave are:

  1. ``molmetal.scripts.wf264a_drugood_post_fix_updater`` — 2 Write sites
     (post_fix_update_report.json + updated_207_v2.md / updated_188_paper_4.md).
     The script already had its own ``_atomic_write_text`` helper that
     duplicated the POSIX atomic rename; Wf-410-A3 drops the local helper
     and routes through the shared ``atomic_write_file`` primitive.
  2. ``molmetal.scripts.bench_dmpnn_top20_predictions`` — 4 Write sites
     (synthesis_md + per_compound_json + per_cell_json + auc_per_seed_json).
  3. ``molmetal.scripts.bench_mcts_replay_learner_regularized`` — 3 Write
     sites (per_seed_path × N seeds + root_path + synthesis_md).
  4. ``molmetal.scripts.bench_variance_mcts_wasserstein`` — 3 Write sites
     (per_seed_path × N seeds + root_path + synthesis_md).
  5. ``molmetal.scripts.wf313a_relink_pocket_ref_ligands`` — 3 Write
     sites (ref_ligand.smi + ref_ligand_inchi.json + extraction_summary.json).

Test cases (5 NEW tests, 1 per migrated script):

  * Per-script: ``test_<script>_uses_atomic_write`` —
    verify the migrated script imports ``atomic_write_file`` AND has no
    remaining direct ``Path.write_text(...)`` or
    ``with open(..., "w")`` Write call sites writing to JSON / MD /
    SMI outputs without going through ``atomic_write_file``.

Honest framing
==============

These tests are CPU-only, stdlib-only, no third-party deps beyond the
existing ``pytest`` framework + ``molmetal.scripts.wf386a3_atomic_write``
helper module. They do NOT execute the migrated scripts end-to-end
(those scripts may depend on cohort data + trainer imports that are
not safe to load in a unit-test context); they verify the migration
surface via AST inspection + module-attribute checks.

Per Wf-386-A3 §6.1 adoption sequence: migration is opt-in per script,
auto-migrating all existing scripts in one sweep is too risky. Wf-410-A3
migrates 5 more closeout-drafting scripts after Wf-406-A3's 3; combined
Wf-406+410 = 8 of N existing scripts migrated.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "molmetal" / "scripts"

# Five migrated scripts per Wf-410-A3 closeout (next top-5 by write-site
# count + recency + completed_refs from the §6.2 audit). The 3 Wf-406-A3
# predecessors are intentionally excluded here — they are covered by
# `test_wf406a3_atomic_migration.py`.
MIGRATED_SCRIPTS = [
    "wf264a_drugood_post_fix_updater.py",
    "bench_dmpnn_top20_predictions.py",
    "bench_mcts_replay_learner_regularized.py",
    "bench_variance_mcts_wasserstein.py",
    "wf313a_relink_pocket_ref_ligands.py",
]


def _ast_has_atomic_write_import(tree: ast.AST) -> bool:
    """Return True iff `tree` imports `atomic_write_file` from
    `molmetal.scripts.wf386a3_atomic_write`.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names = {alias.name for alias in node.names}
            if module.endswith("wf386a3_atomic_write") and "atomic_write_file" in names:
                return True
    return False


def _ast_legacy_with_open_writes(tree: ast.AST) -> list[str]:
    """Return a list of `with open(..., "w")` block descriptions writing
    to .json / .md outputs (these would be legacy direct-Write sites
    that should have been migrated to ``atomic_write_file``).
    """
    legacy_writes: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.With):
            for item in node.items:
                ctx = item.context_expr
                if isinstance(ctx, ast.Call) and getattr(ctx.func, "id", "") == "open":
                    mode_arg = None
                    if len(ctx.args) >= 2:
                        arg2 = ctx.args[1]
                        if isinstance(arg2, ast.Constant):
                            mode_arg = arg2.value
                    if mode_arg is None:
                        for kw in ctx.keywords:
                            if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
                                mode_arg = kw.value.value
                    if mode_arg == "w":
                        path_arg = ctx.args[0] if ctx.args else None
                        path_str = None
                        if isinstance(path_arg, ast.Constant) and isinstance(
                            path_arg.value, str
                        ):
                            path_str = path_arg.value
                        elif path_arg is not None:
                            try:
                                path_str = ast.unparse(path_arg)
                            except Exception:
                                path_str = None
                        if path_str and (
                            path_str.endswith(".json")
                            or path_str.endswith(".md")
                            or path_str.endswith(".smi")
                        ):
                            legacy_writes.append(f"line {node.lineno}: {path_str}")
    return legacy_writes


def _ast_legacy_path_write_text_calls(src: str) -> list[str]:
    """Heuristic text-scan for remaining `.write_text(...)` calls on
    paths that look like .json / .md / .smi outputs. The migrated code
    may legitimately construct `Path(...)` arguments for
    ``atomic_write_file`` (those are fine), so we flag any `.write_text(`
    line whose adjacent context suggests a JSON/MD/SMI target.
    """
    flagged: list[str] = []
    for ln, line in enumerate(src.splitlines(), 1):
        if ".write_text(" not in line:
            continue
        # Filter out the benign pattern where the SAME line has both
        # .write_text( AND atomic_write_file( — those are wrapper helpers
        # that delegate to atomic_write_file under the hood.
        if "atomic_write_file(" in line:
            continue
        # Strip comments + docstring residue
        clean = line.split("#", 1)[0].strip()
        if any(
            suffix in clean
            for suffix in ('.json"', ".json'", '.md"', ".md'", '.smi"', ".smi'")
        ):
            flagged.append(f"line {ln}: {clean[:80]}")
    return flagged


# ---------------------------------------------------------------------------
# Per-script: atomic_write_used check (1 per migrated script = 5 tests)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("script_name", MIGRATED_SCRIPTS)
def test_script_uses_atomic_write(script_name: str) -> None:
    """Verify the migrated Wf-410-A3 script imports ``atomic_write_file``
    AND has no remaining direct Write call sites writing to JSON / MD /
    SMI outputs without going through ``atomic_write_file``.

    This closes the Wf-378-A3 + Wf-386-A1+A2 root cause: a verify agent
    may report "no remaining direct Write calls" even when a half-merged
    file state exists, because the direct Write happened in a different
    process / wave. AST-level inspection here catches the regression at
    pre-flight time (per AGENTS.md §5.6).
    """
    script_path = SCRIPTS_DIR / script_name
    assert script_path.exists(), f"script missing: {script_path}"

    src = script_path.read_text(encoding="utf-8")
    tree = ast.parse(src)

    # 1. atomic_write_file must be imported from wf386a3_atomic_write.
    assert _ast_has_atomic_write_import(tree), (
        f"{script_name}: must import atomic_write_file from "
        f"molmetal.scripts.wf386a3_atomic_write"
    )

    # 2. Every `with open(..., "w") as f:` block writing to .json / .md /
    #    .smi outputs should have been replaced. Legacy `with open(..., "r")`
    #    and csv/text reads are fine.
    legacy_writes = _ast_legacy_with_open_writes(tree)
    assert not legacy_writes, (
        f"{script_name}: legacy `with open(path, 'w')` Write sites "
        f"writing to .json/.md/.smi outputs must be migrated to "
        f"atomic_write_file:\n" + "\n".join(legacy_writes)
    )

    # 3. Heuristic text-scan: no remaining `.write_text(...)` on
    #    JSON/MD/SMI target paths. The helper wrapper functions are
    #    allowed to call atomic_write_file internally (skipped above).
    legacy_path_writes = _ast_legacy_path_write_text_calls(src)
    assert not legacy_path_writes, (
        f"{script_name}: legacy `.write_text(...)` calls on .json/.md/.smi "
        f"outputs must be migrated to atomic_write_file:\n"
        + "\n".join(legacy_path_writes)
    )
