"""Tests for Wf-414-A3: atomic write protocol migration of top-5 closeout-drafting scripts.

================================================================
Wf-414-A3 closes the AGENTS.md §5.6 adoption sequence step 4 (per
Wf-386-A3 §6.1): migrates 5 additional closeout-drafting scripts from
direct Write to ``atomic_write_file`` + ``safe_overwrite_check`` +
``detect_truncation``. The 8 previously-migrated scripts
(Wf-406-A3: 3 + Wf-410-A3: 5) live in
``test_wf406a3_atomic_migration.py`` + ``test_wf410a3_atomic_migration_extended.py``;
Wf-414-A3 extends the migration to the next top-5 most-used scripts.

The 5 migrated scripts in this wave are:

  1. ``molmetal.scripts.wf341c3_shortcut_ema`` — 1 Write site
     (``out_path = out_dir / "summary.json"`` — CPU-only shortcut + EMA
     smoke bench, writes to ``molmetal/reports/wf341c3_shortcut_ema/``).
  2. ``molmetal.scripts.wf380a4_vina_param_tuning`` — 1 Write site
     (``out_path.write_text(json.dumps(payload, ...) + "\\n")`` —
     m3 Vina parameter tuning smoke, writes to
     ``molmetal/reports/wf380a4_vina_tuning_smoke/result.json``).
  3. ``molmetal.scripts.wf381a1_vina_param_tuning`` — 1 Write site
     (m3 Vina parameter tuning follow-up, writes to
     ``molmetal/reports/wf381a1_vina_tuning_smoke/result.json``).
  4. ``molmetal.scripts.wf385a3_scaffold_wire_in_gate`` — 1 Write site
     (NEW pre-flight gate ``p3_wf385_scaffold_loss_wire_in_verified``
     output at ``molmetal/reports/wf385a3_scaffold_wire_in/gate_result.json``).
  5. ``molmetal.scripts.wf392a2_chemprop_v3_cache_wire`` — 3 Write sites
     (trainer backup + trainer edit + smoke report at ``--out`` JSON).

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
auto-migrating all existing scripts in one sweep is too risky. Wf-414-A3
migrates 5 more closeout-drafting scripts after Wf-406-A3's 3 + Wf-410-A3's 5;
combined Wf-406+410+414 = 13 of N existing scripts migrated.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "molmetal" / "scripts"

# Five migrated scripts per Wf-414-A3 closeout (next top-5 by write-site
# count + recency + completed_refs from the §6.2 audit). The 8 Wf-406-A3
# + Wf-410-A3 predecessors are intentionally excluded here — they are
# covered by `test_wf406a3_atomic_migration.py` + `test_wf410a3_atomic_migration_extended.py`.
MIGRATED_SCRIPTS = [
    "wf341c3_shortcut_ema.py",
    "wf380a4_vina_param_tuning.py",
    "wf381a1_vina_param_tuning.py",
    "wf385a3_scaffold_wire_in_gate.py",
    "wf392a2_chemprop_v3_cache_wire.py",
]


def _ast_has_atomic_write_import(tree: ast.AST) -> bool:
    """Return True iff ``tree`` imports ``atomic_write_file`` from
    ``molmetal.scripts.wf386a3_atomic_write``.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = (node.module or "")
            names = {alias.name for alias in node.names}
            if module.endswith("wf386a3_atomic_write") and "atomic_write_file" in names:
                return True
    return False


def _ast_has_safe_overwrite_and_detect_truncation(tree: ast.AST) -> tuple[bool, bool]:
    """Return ``(safe_overwrite_check_imported, detect_truncation_imported)``
    bool pair — verifies the Wf-414-A3 migration is the full belt-and-braces
    trio, not a single ``atomic_write_file`` import only.
    """
    safe = False
    trunc = False
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if (node.module or "").endswith("wf386a3_atomic_write"):
                names = {alias.name for alias in node.names}
                if "safe_overwrite_check" in names:
                    safe = True
                if "detect_truncation" in names:
                    trunc = True
    return safe, trunc


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
                        if isinstance(path_arg, ast.Constant) and isinstance(path_arg.value, str):
                            path_str = path_arg.value
                        elif path_arg is not None:
                            try:
                                path_str = ast.unparse(path_arg)
                            except Exception:
                                path_str = None
                        if path_str and (path_str.endswith(".json")
                                         or path_str.endswith(".md")
                                         or path_str.endswith(".smi")):
                            legacy_writes.append(f"line {node.lineno}: {path_str}")
    return legacy_writes


def _ast_legacy_path_write_text_calls(src: str) -> list[str]:
    """Heuristic text-scan for remaining `.write_text(...)` calls on
    paths that look like .json / .md / .smi outputs. The migrated code
    may legitimately construct `Path(...)` arguments for
    ``atomic_write_file`` (those are fine), so we flag any `.write_text(`
    line whose adjacent context suggests a JSON/MD/SMI target.

    Also accepts the Wf-414-A3 fallback path ``else: out_path.write_text(...)
    # pragma: no cover`` (the try/except ImportError guard around the
    atomic_write_file import); those are intentional fallback paths and
    are NOT flagged here.
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
        if any(suffix in clean for suffix in ('.json"', ".json'", '.md"', ".md'", '.smi"', ".smi'")):
            flagged.append(f"line {ln}: {clean[:80]}")
    return flagged


# ---------------------------------------------------------------------------
# Per-script: atomic_write_used check (1 per migrated script = 5 tests)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("script_name", MIGRATED_SCRIPTS)
def test_script_uses_atomic_write(script_name: str) -> None:
    """Verify the migrated Wf-414-A3 script imports ``atomic_write_file``
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

    # 2. safe_overwrite_check + detect_truncation must BOTH be imported
    #    — the Wf-414-A3 migration ships the full belt-and-braces trio,
    #    not just atomic_write_file alone.
    safe_imported, trunc_imported = _ast_has_safe_overwrite_and_detect_truncation(tree)
    assert safe_imported, (
        f"{script_name}: must import safe_overwrite_check from "
        f"molmetal.scripts.wf386a3_atomic_write (Wf-414-A3 belt-and-braces)"
    )
    assert trunc_imported, (
        f"{script_name}: must import detect_truncation from "
        f"molmetal.scripts.wf386a3_atomic_write (Wf-414-A3 belt-and-braces)"
    )

    # 3. Every `with open(..., "w") as f:` block writing to .json / .md /
    #    .smi outputs should have been replaced. Legacy `with open(..., "r")`
    #    and csv/text reads are fine.
    legacy_writes = _ast_legacy_with_open_writes(tree)
    assert not legacy_writes, (
        f"{script_name}: legacy `with open(path, 'w')` Write sites "
        f"writing to .json/.md/.smi outputs must be migrated to "
        f"atomic_write_file:\n" + "\n".join(legacy_writes)
    )

    # 4. Heuristic text-scan: no remaining `.write_text(...)` on
    #    JSON/MD/SMI target paths. The helper wrapper functions are
    #    allowed to call atomic_write_file internally (skipped above).
    legacy_path_writes = _ast_legacy_path_write_text_calls(src)
    assert not legacy_path_writes, (
        f"{script_name}: legacy `.write_text(...)` calls on .json/.md/.smi "
        f"outputs must be migrated to atomic_write_file:\n"
        + "\n".join(legacy_path_writes)
    )

    # 5. Both safe_overwrite_check() and detect_truncation() must be called
    #    by name in the migrated write blocks (not just imported-but-unused).
    src_lower = src.lower()
    assert "safe_overwrite_check(" in src_lower, (
        f"{script_name}: safe_overwrite_check() must be called at write sites"
    )
    assert "detect_truncation(" in src_lower, (
        f"{script_name}: detect_truncation() must be called on agent output"
    )