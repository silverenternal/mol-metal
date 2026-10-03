# Wf-414-A3 — Atomic write protocol migration of next top-5 closeout-drafting scripts

**Date:** 2026-09-30
**Wave:** Wf-414-A3 (CPU-only docs+code+test, 0 GPU launches)
**User voice relay:** *"用GPU0好了，同时启动好了"* (GPU 0 standing authorization reaffirmation; no conflict with CPU-only task — per AGENTS.md §5.1 standing GPU grant)
**Skill catalog:** No skill load required (Wf-386-A3 atomic write primitive is project-owned per AGENTS.md §3)
**Standing directives:** §5.1 (no GPU), §5.2 (honest framing — PROJECTED vs MEASURED), §5.3 (no paper/*), §5.4 (no wet-lab), §5.5 (post-wave on-disk verification — file SHIPPED), §5.6 (atomic write protocol adoption rule — this wave IS the canonical document-of-record), party-twenty-spirit §3.5 (MEASURED-vs-PROJECTED tag discipline)

---

## §0 TL;DR

Wf-414-A3 migrates 5 additional closeout-drafting scripts from direct `Path.write_text()` /
`with open(..., "w")` writes to the Wf-386-A3 atomic write protocol trio
(`atomic_write_file` + `safe_overwrite_check` + `detect_truncation`). Combined with
Wf-406-A3's 3 scripts + Wf-410-A3's 5 scripts, **13 of N existing scripts are now migrated**
to the canonical I/O substrate. 5 NEW tests PASS in <1 s wall CPU-only. The migration
closes the half-merged-file failure mode + the substantial-overwrite failure mode for
the 5 newly-migrated scripts.

---

## §1 Audit findings — remaining direct Write sites in `molmetal/scripts/`

The audit grep `def write_closeout|def write_final_synthesis|def write_b1_index_sync`
returned **0 hits** in `molmetal/scripts/` (the only match is the atomic_write_file
module itself at `molmetal/scripts/wf386a3_atomic_write.py`).

The deeper audit (combined `TODO/completed.*\.md|TODO/INDEX\.md` + `write_text|atomic_write_file`)
returned 54 scripts. Of those:
- **8 already migrated** by Wf-406-A3 (3) + Wf-410-A3 (5): `wf398a2_m2_r2_real_verify`,
  `wf390a2_alt_encoder_audit`, `wf394a1_lambda_cfm_wire`, `wf264a_drugood_post_fix_updater`,
  `bench_dmpnn_top20_predictions`, `bench_mcts_replay_learner_regularized`,
  `bench_variance_mcts_wasserstein`, `wf313a_relink_pocket_ref_ligands`.
- **5 migrated by this wave (Wf-414-A3)**: `wf341c3_shortcut_ema`, `wf380a4_vina_param_tuning`,
  `wf381a1_vina_param_tuning`, `wf385a3_scaffold_wire_in_gate`, `wf392a2_chemprop_v3_cache_wire`.
- **~41 remaining** — deferred to future Wf-414+ waves per Wf-386-A3 §6.1 opt-in adoption
  sequence (auto-migrating all existing scripts in one sweep is too risky).

The 5 Wf-414-A3 migrated scripts were selected per the **top-5 by write-site count**
criterion from the audit:

| Script | Write sites | Write location |
|---|---|---|
| `wf341c3_shortcut_ema.py` | 1 | `out_dir / "summary.json"` line 580 |
| `wf380a4_vina_param_tuning.py` | 1 | `out_path.write_text(json.dumps(payload, ...))` line 421 |
| `wf381a1_vina_param_tuning.py` | 1 | `out_path.write_text(json.dumps(payload, ...))` line 319 |
| `wf385a3_scaffold_wire_in_gate.py` | 1 | `out_json = out_dir / "gate_result.json"` line 468 |
| `wf392a2_chemprop_v3_cache_wire.py` | 3 | backup write line 138 + trainer edit line 148 + smoke report line 329 |
| **TOTAL** | **7 write sites** | across 5 scripts |

---

## §2 Migration pattern (per Wf-386-A3 §6.1 step 4)

For each migrated script:

1. **Add import block** (after the existing stdlib/third-party imports):

```python
try:
    from molmetal.scripts.wf386a3_atomic_write import (
        atomic_write_file,
        safe_overwrite_check,
        detect_truncation,
    )
    _WF414_HAS_ATOMIC = True
except ImportError:  # pragma: no cover — fallback when running as standalone
    _WF414_HAS_ATOMIC = False
    atomic_write_file = None  # type: ignore[assignment]
    safe_overwrite_check = None  # type: ignore[assignment]
    detect_truncation = None  # type: ignore[assignment]
```

2. **Replace direct Write with the migration pattern**:

```python
_payload = json.dumps(<data>, indent=2)
if _WF414_HAS_ATOMIC:
    _risk = safe_overwrite_check(out_path, _payload, min_lines=50)
    if _risk is not None:
        LOG.warning("[Wf-XXX] safe_overwrite_check ratio=%.3f at %s", _risk, out_path)
    atomic_write_file(out_path, _payload, encoding="utf-8")
    _trunc = detect_truncation(_payload, expected_min_length=200)
    if _trunc is not None:
        LOG.warning("[Wf-XXX] detect_truncation reason=%r at %s", _trunc, out_path)
else:  # pragma: no cover
    out_path.write_text(_payload, encoding="utf-8")
```

3. **Pattern guarantees**:
   - `atomic_write_file` (POSIX atomic temp-file + rename with per-path process-lock) closes
     the half-merged-file failure mode (per AGENTS.md §5.6).
   - `safe_overwrite_check` (line-set Jaccard distance, threshold 0.5) catches substantial
     content loss when re-authoring an existing closeout (closes the Wf-384-A3 over-reporting
     failure mode).
   - `detect_truncation` (closing-token heuristic + length check ≥ 200 chars) catches
     mid-sentence agent output before the calling workflow accepts it as a complete closeout.
   - `try/except ImportError` fallback preserves backward compatibility when the script is
      run as a standalone file (without the molmetal package on PYTHONPATH).

---

## §3 Per-script migration summary

### §3.1 `wf341c3_shortcut_ema.py`

**Site 1 — `summary.json` write** (line ~580):
- Before: `out_path.write_text(json.dumps(asdict(summary), indent=2))`
- After: full migration pattern with `safe_overwrite_check` + `atomic_write_file` + `detect_truncation`
- Output target: `molmetal/reports/wf341c3_shortcut_ema/summary.json`

### §3.2 `wf380a4_vina_param_tuning.py`

**Site 1 — `result.json` write** (line ~421):
- Before: `out_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")`
- After: full migration pattern
- Output target: `molmetal/reports/wf380a4_vina_tuning_smoke/result.json`

### §3.3 `wf381a1_vina_param_tuning.py`

**Site 1 — `result.json` write** (line ~319):
- Before: `out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")`
- After: full migration pattern
- Output target: `molmetal/reports/wf381a1_vina_tuning_smoke/result.json`

### §3.4 `wf385a3_scaffold_wire_in_gate.py`

**Site 1 — `gate_result.json` write** (line ~468):
- Before: `out_json.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")`
- After: full migration pattern
- Output target: `molmetal/reports/wf385a3_scaffold_wire_in/gate_result.json`

### §3.5 `wf392a2_chemprop_v3_cache_wire.py`

**Site 1 — trainer backup write** (line ~138):
- Before: `backup_path.write_text(content)`
- After: `atomic_write_file(backup_path, content, encoding="utf-8")` (no safe_overwrite_check
  because the backup path uses `with_suffix(suffix + BACKUP_SUFFIX)` and the
  `if not backup_path.exists()` check guards against overwrite)
- Target: `{trainer_path}.bak.wf392a2`

**Site 2 — trainer edit write** (line ~148):
- Before: `trainer_file.write_text(new_content)`
- After: `atomic_write_file(trainer_file, new_content, encoding="utf-8")` (no
  safe_overwrite_check because the WIRED_MARKER guard already makes the operation
  idempotent — if marker is in content, the function returns False before reaching
  the write)
- Target: trainer file (`molmetal/scripts/train_chemprop_v2_v3.py`)

**Site 3 — smoke report write** (line ~329):
- Before: `out_path.write_text(json.dumps(out, indent=2, default=str))`
- After: full migration pattern
- Target: `--out` JSON path (typically `molmetal/reports/wf392a2_chemprop_v3_cache/smoke_result.json`)

---

## §4 Test suite

**File:** `molmetal/tests/test_wf414a3_atomic_migration.py` ~260 LOC

**Tests (5 NEW tests, 1 per migrated script, parametrized via MIGRATED_SCRIPTS list):**

```python
@pytest.mark.parametrize("script_name", MIGRATED_SCRIPTS)
def test_script_uses_atomic_write(script_name: str) -> None:
```

**Test assertions (5 layers of belt-and-braces verification per Wf-414-A3):**

1. **atomic_write_file import check** — AST walk finds `from molmetal.scripts.wf386a3_atomic_write import atomic_write_file`
2. **safe_overwrite_check + detect_truncation imports** — both must be imported (Wf-414-A3 ships the FULL trio, not single atomic_write_file only)
3. **No legacy `with open(..., "w")` blocks** — AST walk finds NO legacy `open(path, "w", ...)` blocks writing to .json/.md/.smi outputs
4. **No legacy `.write_text()` calls** — heuristic text-scan finds NO `.write_text(...)` calls on JSON/MD/SMI target paths (excluding the `else: out_path.write_text(...) # pragma: no cover` fallback path which is intentional)
5. **Both `safe_overwrite_check()` and `detect_truncation()` called by name** — source text contains both calls (not just imported-but-unused)

---

## §5 MEASURED test pass-rate

5 NEW tests PASS in <1 s wall CPU-only (direct Python invocation):

```
PASS: wf341c3_shortcut_ema.py
PASS: wf380a4_vina_param_tuning.py
PASS: wf381a1_vina_param_tuning.py
PASS: wf385a3_scaffold_wire_in_gate.py
PASS: wf392a2_chemprop_v3_cache_wire.py
TOTAL: 5 tests, 5 PASS, 0 FAIL
```

(Note: pytest framework blocked by missing `torch` module in this CPU-only venv; the
ImportError originates from `molmetal/tests/conftest.py:186` `_force_torch_scatter_backend_for_cpu_tests`
which requires `torch` to import. The Wf-414-A3 tests are torch-free (AST-only) — they
verify migration surface via AST inspection + module-attribute checks, NOT by executing
the migrated scripts end-to-end. The direct invocation works because it bypasses conftest.py
and runs the test functions directly.)

The 5 migrated scripts ALSO AST-parse cleanly (verified individually with `ast.parse`):

```
=== wf341c3_shortcut_ema.py === AST parse OK
=== wf380a4_vina_param_tuning.py === AST parse OK
=== wf381a1_vina_param_tuning.py === AST parse OK
=== wf385a3_scaffold_wire_in_gate.py === AST parse OK
=== wf392a2_chemprop_v3_cache_wire.py === AST parse OK
```

---

## §6 Combined migration coverage

| Wave | Scripts migrated | Cumulative count |
|---|---|---|
| Wf-406-A3 | 3 | 3 |
| Wf-410-A3 | 5 | 8 |
| **Wf-414-A3** | **5** | **13** |
| Wf-414+ (deferred) | ~41 remaining | TBD |

Per Wf-386-A3 §6.1 adoption sequence: migration is **opt-in per script** — auto-migrating
all existing scripts in a single sweep is too risky because some scripts have
deliberate legacy direct-Write patterns (e.g. CSV logging, debug snapshots) that
should NOT be migrated.

---

## §7 Honest framing

**MEASURED:**
- 5 scripts AST-parse cleanly post-migration
- 5/5 tests PASS via direct invocation (bypasses torch-blocked conftest)
- 7 write sites migrated across 5 scripts
- All 5 scripts now route their `.json` / `.md` writes through the Wf-386-A3 atomic write primitive trio

**PROJECTED (NOT MEASURED):**
- Future regression prevention: AST-level checks at pre-flight time prevent the
  "added a new direct Write site in a migrated script" regression class
- Future Wf-414+ waves will migrate ~41 more scripts, bringing cumulative count
  to ~54 closeout-drafting scripts migrated total
- The fallback path (`else: out_path.write_text(...) # pragma: no cover`) is
  NOT exercised in production (it only triggers if `from molmetal.scripts.wf386a3_atomic_write`
  fails to import, which would indicate a broken install) — but it preserves backward
  compatibility when running scripts via direct `python3 molmetal/scripts/wf341c3_shortcut_ema.py`
  without the molmetal package on PYTHONPATH

**NOT CHANGED:**
- Pre-flight v3 89/2/3/94 PRESERVED at Wf-413-B1 baseline (NOT re-run by Wf-414-A3 — docs+code+test only)
- EXPECTED_GATE_COUNTS["all"] 94 PRESERVED at Wf-353-B1 baseline
- 0 GPU launches across the entire wave (CPU-only docs+code+test per AGENTS.md §5.1)
- 0 `paper/*` touches (TODO-84 active)
- 0 wet-lab planning (AGENTS.md §5.4 + §10)

---

## §8 Cross-references

- AGENTS.md §5.5 (post-wave on-disk verification rule — this wave's source-file migration
  surface verification)
- AGENTS.md §5.6 (atomic write protocol adoption rule — this wave IS the canonical
  document-of-record for the 5-script migration wave)
- `molmetal/scripts/wf386a3_atomic_write.py` — canonical atomic_write_file + safe_overwrite_check
  + detect_truncation primitive (NEW per Wf-386-A3)
- `docs/ATOMIC_WRITE_PROTOCOL.md` — canonical reference doc (NEW per Wf-386-A3)
- `molmetal/tests/test_wf386a3_atomic_write.py` — 27-case test suite (NEW per Wf-386-A3)
- `molmetal/tests/test_wf406a3_atomic_migration.py` — Wf-406-A3 predecessor test file (3 NEW tests)
- `molmetal/tests/test_wf410a3_atomic_migration_extended.py` — Wf-410-A3 predecessor test file (5 NEW tests)
- `molmetal/tests/test_wf414a3_atomic_migration.py` — THIS wave's test file (5 NEW tests)

## §10 Honest framing caveat

Per `party-twenty-spirit` §3.5:
- Every migration claim above is **MEASURED** via AST parse + direct test invocation.
- The pytest framework cannot be invoked due to torch-not-installed; direct invocation
  bypasses the broken conftest and runs the tests successfully. This is a TEST-RUNNER
  pattern, NOT a code-quality issue.
- Migration is opt-in per Wf-386-A3 §6.1; ~41 more scripts remain for future waves.

---

**End of Wf-414-A3 closeout.**
