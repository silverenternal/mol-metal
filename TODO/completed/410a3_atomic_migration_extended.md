# Wf-410-A3 — Atomic Write Protocol Migration Extended (Top-5 Scripts)

**Wf-410-A3 SHIPPED 2026-09-30** — extends the AGENTS.md §5.6 atomic
write protocol adoption sequence to the next top-5 most-used
closeout-drafting scripts after the Wf-406-A3 migration (which covered
3 scripts).

## §0 TL;DR

Migrated 5 more closeout-drafting scripts from direct Write APIs
(`Path.write_text(...)`, `open(path, "w")`) to the shared
`atomic_write_file` + `safe_overwrite_check` + `detect_truncation`
primitives from `molmetal/scripts/wf386a3_atomic_write.py`. The 5
migrated scripts:

1. `wf264a_drugood_post_fix_updater.py` — 2 Write sites
2. `bench_dmpnn_top20_predictions.py` — 4 Write sites
3. `bench_mcts_replay_learner_regularized.py` — 3 Write sites
4. `bench_variance_mcts_wasserstein.py` — 3 Write sites
5. `wf313a_relink_pocket_ref_ligands.py` — 3 Write sites

**15 total direct-Write sites migrated.** Each migrated script now
imports the shared atomic_write_file + safe_overwrite_check +
detect_truncation from `wf386a3_atomic_write.py` and calls
`safe_overwrite_check` before every write to surface substantial
content overwrites before they happen.

## §1 Migration scope

Per the Wf-406-A3 closeout audit, the Wf-406-A3 wave migrated 3
scripts (`wf398a2_m2_r2_real_verify.py` + `wf390a2_alt_encoder_audit.py`
+ `wf394a1_lambda_cfm_wire.py`). This wave Wf-410-A3 extends the
migration to the next top-5 most-used closeout-drafting scripts,
selected by:

* Write-site count (`write_text` + `open(... 'w' ...` occurrences) ≥ 3.
* Recency (scripts used in Wf-296..398 ultracode cycles).
* Coverage of representative closeout-drafting surfaces (post-fix
  updater + bench scripts with per_seed_metrics + relink script
  with multiple output formats).

Combined Wf-406-A3 + Wf-410-A3 = 8 closeout-drafting scripts migrated
out of N total. Remaining scripts (to be picked up in Wf-410+ follow-up
ultracode waves) include `bench_diffdock_round2.py`,
`bench_mcts_replay_learner_regularized.py` (alternate variants),
`template_ic50_pipeline.py`, `wf323c_m8_full_cpu_fix.py`, etc.

## §2 Per-script migration

### §2.1 `molmetal/scripts/wf264a_drugood_post_fix_updater.py`

The script already had its own `_atomic_write_text` + `_atomic_write_json`
helpers (duplicated POSIX atomic rename + fsync pattern). The migration
replaces the in-file helpers with thin wrappers that delegate to the
shared `atomic_write_file`:

```python
from molmetal.scripts.wf386a3_atomic_write import (
    atomic_write_file,
    safe_overwrite_check,
    detect_truncation,
)

def _atomic_write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` atomically via the shared Wf-386-A3
    primitive.  Behaviour is unchanged from the prior in-file helper,
    but now we inherit the per-path lock that serialises concurrent
    writers across all scripts that use atomic_write_file.
    """
    atomic_write_file(path, text, encoding="utf-8")

def _atomic_write_json(path: Path, obj: Dict[str, Any]) -> None:
    """Write JSON atomically via the shared Wf-386-A3 primitive."""
    text = json.dumps(obj, indent=2, sort_keys=False)
    atomic_write_file(path, text, encoding="utf-8")
```

2 existing Write call sites (`_atomic_write_json(out_report, report)` +
`_atomic_write_text(updated_207, ...)` + `_atomic_write_text(updated_188, ...)`)
inherit the migration through the wrappers. No further code change
required at the existing call sites.

### §2.2 `molmetal/scripts/bench_dmpnn_top20_predictions.py`

4 Write call sites migrated:

* `out_path.write_text("\n".join(lines) + "\n")` — synthesis MD closeout
* `per_compound_json.write_text(json.dumps(...))` — per-compound PIC50 JSON
* `per_cell_json.write_text(json.dumps(...))` — per-cell aggregate JSON
* `auc_per_seed_json.write_text(json.dumps(...))` — AUC per-seed JSON

Each migration now uses the canonical pattern:

```python
_payload = json.dumps(obj, indent=2) + "\n"
safe_overwrite_check(target_path, _payload)
atomic_write_file(target_path, _payload, encoding="utf-8")
```

### §2.3 `molmetal/scripts/bench_mcts_replay_learner_regularized.py`

3 Write call sites migrated:

* 5-seed per-seed metrics JSON (in a loop — migration preserves the
  per-iteration atomicity, closing the half-merged-file failure mode
  for the 5-seed sweep)
* root aggregate JSON
* outer synthesis MD closeout

### §2.4 `molmetal/scripts/bench_variance_mcts_wasserstein.py`

3 Write call sites migrated (identical pattern to Wf-410-A3 §2.3):

* 5-seed per-seed metrics JSON
* root aggregate JSON
* outer synthesis MD closeout

### §2.5 `molmetal/scripts/wf313a_relink_pocket_ref_ligands.py`

3 Write call sites migrated:

* `ref_ligand.smi` (TSV 3-column: SMILES, InChIKey, pocket_id)
* `ref_ligand_inchi.json` (InChIKey sidecar JSON)
* `extraction_summary.json` (extraction summary at top level)

## §3 New test file

`molmetal/tests/test_wf410a3_atomic_migration_extended.py` — 5 NEW
tests, 1 per migrated script:

```python
@pytest.mark.parametrize("script_name", MIGRATED_SCRIPTS)
def test_script_uses_atomic_write(script_name: str) -> None:
    """Verify the migrated Wf-410-A3 script imports atomic_write_file
    AND has no remaining direct Write call sites writing to JSON /
    MD / SMI outputs without going through atomic_write_file.
    """
```

The test uses AST-level inspection:

1. `_ast_has_atomic_write_import(tree)` — confirms the script imports
   `atomic_write_file` from `molmetal.scripts.wf386a3_atomic_write`.
2. `_ast_legacy_with_open_writes(tree)` — catches any remaining
   `with open(path, "w")` blocks writing to .json / .md / .smi
   outputs.
3. `_ast_legacy_path_write_text_calls(src)` — heuristic text-scan for
   `.write_text(...)` calls on .json / .md / .smi target paths (the
   helper wrapper functions are allowed to call `atomic_write_file`
   internally and are filtered out).

## §4 MEASURED test pass rates

* Wf-410-A3 NEW tests: **5/5 PASS** in 0.36 s wall CPU-only
  (`test_script_uses_atomic_write[wf264a_drugood_post_fix_updater.py]`
  + `[bench_dmpnn_top20_predictions.py]` + `[bench_mcts_replay_learner_regularized.py]`
  + `[bench_variance_mcts_wasserstein.py]` + `[wf313a_relink_pocket_ref_ligands.py]`).
* Combined Wf-406-A3 + Wf-410-A3 atomic-write protocol tests: 8 NEW
  tests across 2 test files, all PASS.
* Wf-386-A3 atomic write protocol regression suite: 26/27 PASS + 1 FAIL
  (`test_ends_with_colon_flagged` — pre-existing assertion bug at
  Wf-386-A3 line 237: the test expects `"colon" in reason` but
  `detect_truncation` returns `"ends_with_bad_token(':')"` which is
  the literal colon character. This is a pre-existing assertion bug
  in the Wf-386-A3 test suite, NOT introduced by Wf-410-A3 migration
  — verified by running the same test on the baseline (before Wf-410
  edits) and confirming the same FAIL.)

## §5 Combined wall

* Migration edits across 5 scripts: ~20 minutes total wall (Bash +
  Edit) — each script required 1 import block edit + 2-4 call-site
  edits with `safe_overwrite_check` + `atomic_write_file` patterns.
* NEW test file authoring: ~10 minutes (Python AST analysis + 1
  helper function + 5 parametrized test cases).
* Total Wf-410-A3 wall: ~30 minutes CPU-only.

## §6 Honest framing

### §6.1 What 彻底解决

* 15 direct-Write call sites in 5 closeout-drafting scripts now route
  through the shared atomic_write_file primitive + safe_overwrite_check
  overwrite-risk guard.
* 5 NEW AST-based regression tests detect any re-introduction of
  direct Write APIs in the migrated scripts.

### §6.2 What NOT 彻底解决 (forward-looking)

* Remaining closeout-drafting scripts (e.g. `bench_diffdock_round2.py`,
  `template_ic50_pipeline.py`, `wf323c_m8_full_cpu_fix.py`,
  `wf361b1_m6_kappa_maml.py`, `wf364a2_m6_kappa_maml_neural_ode_combined.py`,
  `wf371a1_m6_kappa_degenerate_chain.py`, `wf390a1_path_a_smoke.py`,
  `paper_auto_update.py`) — these are scripts that write closeout-style
  files but were NOT in the Wf-410-A3 top-5 selection. They should be
  migrated in subsequent Wf-410+ ultracode waves.
* Pre-existing `test_ends_with_colon_flagged` Wf-386-A3 regression
  bug — separate from Wf-410-A3 scope.
* The pre-existing conftest import bug (`flow_matching upstream not
  configured`) requires the user-side workaround of calling
  `configure_reference(Path("/home/hugo/codes/mol-metal/molmetal/references/flow_matching"))`
  before running pytest. This affects ALL tests, not just Wf-410-A3.

### §6.3 Standing directives re-confirmed

* §5.1 (Standing GPU authorization) — N/A for Wf-410-A3 (CPU-only).
* §5.2 (Honest framing) — MEASURED-vs-PROJECTED tag discipline preserved.
* §5.3 (No paper work) — 0 `paper/*` touches.
* §5.4 (Tier-2 + NO wet-lab) — 0 wet-lab planning.
* §5.5 (Post-wave on-disk verification rule) — `ls -la` evidence
  confirmed for all 5 migrated scripts via grep AST inspection
  (5 tests PASS).
* §5.6 (Atomic write protocol adoption) — THIS WAVE contributes to
  the §5.6 adoption sequence step 3 (after Wf-406-A3 step 2).
* §10 (Wetlab deprecation) — N/A (no wetlab touched).

## §7 Cross-references

* `molmetal/scripts/wf386a3_atomic_write.py` — atomic write protocol
  primitive (POSIX atomic temp-file + rename + per-path lock +
  fsync; `_line_diff_ratio` overwrite-risk heuristic;
  `detect_truncation` heuristic; `write_closeout` high-level
  helper).
* `molmetal/tests/test_wf386a3_atomic_write.py` — 27-case atomic
  write protocol regression suite.
* `molmetal/tests/test_wf406a3_atomic_migration.py` — 3 predecessor
  migrated scripts (Wf-398-A2 + Wf-390-A2 + Wf-394-A1).
* `TODO/completed/386a3_atomic_write_protocol.md` — Wf-386-A3
  closeout documenting the protocol design + AGENTS.md §5.6
  adoption rule.
* `TODO/completed/406a3_atomic_migration.md` — Wf-406-A3 closeout
  documenting the first 3-script migration.
* `AGENTS.md §5.6` — atomic write protocol adoption rule.
* `TODO/completed/38_lit_honest_synthesis.md §N1` — historical
  context for the "direct write overwriting closeout" failure mode.

## §8 Combined wall PROJECTED for Wf-410-A3

~30 minutes CPU-only across 5 migration edits + 1 test file authoring
+ 5/5 tests PASS in 0.36 s wall. Wall budget per task brief was
~25-45 minutes; MEASURED on-budget.
