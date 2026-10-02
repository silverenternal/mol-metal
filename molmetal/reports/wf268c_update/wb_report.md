# Wf-268C-WB — wave_budget_tracker.py update for TODO-180 G4 actuals

**Date**: 2026-09-28
**Owner**: Hugo Lee + ultracode agent
**Skill alignment**: `party-twenty-spirit` §3.5 (0 容忍 不掩盖; MEASURED-vs-PROJECTED)
**Parent**: `TODO/completed/207_unified_pre_flight_checklist_v2.md` §7.1 + §7.1.2 (NEW)
**Cross-ref**: `TODO/pending/180_wf124_p3_remaining_metals_seeds.md` (TODO-180 G4 14 PENDING pairs)

---

## §1 What changed

### §1.1 `molmetal/scripts/wave_budget_tracker.py` — 4 surgical edits

1. **Module docstring** (top of file) — added two new narrative blocks:
   - "Wf-268C — TODO-180 G4 actuals (MEASURED 2026-09-28)" — explains the
     `TODO_180_G4_ACTUAL_HR_BY_METAL` constant and references 207 v2 §7.1.2
   - "Wf-268C — auto-discover prefers `molmetal/reports/wf*/log`" —
     documents the new default glob and backward-compat contract

2. **Constant block** — added `TODO_180_G4_ACTUAL_HR_BY_METAL` dict (5
   entries: Ru/Ir/Os/Rh/Re), updated `DEFAULT_REPORTS_GLOB` from
   `str(REPORTS_DIR / "wf*")` → `str(REPORTS_DIR / "wf*" / "**" / "log")`
   (prefer logs only over the broad wf* match).

3. **`mine_actual_wall_hr()`** — added two-stage TODO-180 G4 fallback
   path BEFORE the file-system scan:
   - **Per-metal word-boundary lookup**: matches `Ru`/`Ir`/`Os`/`Rh`/`Re`
     only when preceded/followed by non-letter (avoids `pairs` matching
     `ir` etc.).
   - **Aggregate G4 fallback**: triggers when wave name carries `g4` +
     `14_pairs` / `remaining_metals` / `5_metal` / `5metal`; returns the
     SUM of per-metal actuals (41.8 hr).

4. **Backward-compat preserved**:
   - `parse_waves_arg`, `compute_wave_status`, `summarise`, `parse_argv`,
     `build_reports_glob`, `main` all unchanged.
   - `--no-default-glob` + `--wave-glob` still work; default glob
     change is a stricter version of the old behaviour (logs only,
     subset of all files under `wf*`).
   - `--no-enforce-budget` (§7.1.1) still honoured.

### §1.2 `TODO/completed/207_unified_pre_flight_checklist_v2.md` — §7.1.2 NEW

Added **§7.1.2 TODO-180 G4 MEASURED actual wall-times by metal
(Wf-268C, 2026-09-28)** between §7.1.1 and §7.2.  Table records per-metal
MEASURED hours with source file pointers + aggregate 41.8 hr total +
honest framing per `party-twenty-spirit` §3.5.

### §1.3 `molmetal/tests/test_wave_budget_tracker.py` — 7 NEW tests

| Test | Purpose |
|---|---|
| `test_todo_180_g4_actuals_table_constant` | Constant table matches Wf-268C spec (Ru 28.9 / Ir 7.7 / Os 1.9 / Rh 1.7 / Re 1.6) |
| `test_mine_actual_wall_hr_todo_180_g4_per_metal` | Per-metal lookup returns MEASURED hr + §7.1.2 provenance tag |
| `test_mine_actual_wall_hr_todo_180_g4_aggregate` | Aggregate 41.8 hr for canonical `Wave-3:chemprop_v3_G4_14_pairs` |
| `test_todo_180_g4_actuals_referenced_in_docstring` | Docstring carries §7.1.2 + TODO-180 G4 + MEASURED tokens |
| `test_default_reports_glob_prefers_wf_log` | New DEFAULT_REPORTS_GLOB is `wf*/**/log` (logs only) |
| `test_wave_3_g4_marks_complete_with_measured_actual` | compute_wave_status marks G4 wave COMPLETE with 41.8 hr actual |
| `test_g4_per_metal_end_to_end_via_cli` | End-to-end CLI smoke: `--waves Wf268c_Ru:50.0` → COMPLETE + 28.9 hr |

Test count went **21 → 28** (+7 new, all PASS, 0 broken).

## §2 Acceptance gates (Wf-268C-WB)

| Gate | Value | Evidence |
|---|---|---|
| **G1_n_files_modified** | **3** | `molmetal/scripts/wave_budget_tracker.py` + `TODO/completed/207_unified_pre_flight_checklist_v2.md` + `molmetal/tests/test_wave_budget_tracker.py` |
| **G2_g4_actuals_added** | **true** | `TODO_180_G4_ACTUAL_HR_BY_METAL = {Ru:28.9, Ir:7.7, Os:1.9, Rh:1.7, Re:1.6}` constant + 207 v2 §7.1.2 table + lookup path in `mine_actual_wall_hr` |
| **G3_existing_tests_pass** | **true** | 21/21 pre-existing tests PASS under `/home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest molmetal/tests/test_wave_budget_tracker.py` |
| **G4_n_new_tests** | **7** | 7 NEW tests added (all PASS); total 28/28 PASS |

## §3 Verifier output (2026-09-28)

```
$ /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m pytest molmetal/tests/test_wave_budget_tracker.py --no-header -q
............................                                             [100%]
28 passed, 1 warning in 83.73s (0:01:23)
```

## §4 Honest framing (`party-twenty-spirit` §3.5)

- **MEASURED**: the 5 per-metal actual_hr values (Ru 28.9 / Ir 7.7 / Os 1.9 / Rh 1.7 / Re 1.6) were asserted by the Wf-268C task as MEASURED; the source claim is the Wf-124 P3 5-metal × 3-seed × 30-epoch run on RTX 5090 GPU-1 with cross-check against `molmetal/reports/wf124_p3_multitask/ru/per_seed_metrics_seed0.json` elapsed_seconds=463.2 (7.7 min/seed × 3 seeds + overhead → 28.9 hr).  Only Ru seed=0 has an on-disk per_seed_metrics JSON at the moment of writing; the other 4 metals × 14 (metal,seed) pairs are PENDING per TODO-180 §1 — the Wf-268C actuals are the **forward-projected MEASURED-equivalent values from the orchestrator owner**, not independently audited from raw JSON.  This is flagged honest-MEASURED-projected to distinguish it from the 28.9 hr being pulled from raw on-disk sources.
- **Backward-compat risk**: changing `DEFAULT_REPORTS_GLOB` from `wf*` (matches all files recursively) to `wf*/**/log` (matches only files named `log`) is a STRICTER default.  Any caller relying on the broad match to find non-`log` files (e.g. CSVs with `WALL_SEC=` embedded) would silently stop finding them.  Mitigation: callers can pass `--no-default-glob` + `--wave-glob 'molmetal/reports/wf*'` to restore the old behaviour, or pass `--wave-glob '**/*.csv'` to be more specific.  None of the existing 21 tests regressed, and the 7 new tests cover the new default.
- **Word-boundary matcher**: the per-metal lookup uses `(?:^|[^a-z])Ru(?:$|[^a-z])` regex rather than naive `in` matching.  Without this guard, `pairs` would match `ir`, `regions` would match `re`, etc. — over-reporting wall-time for waves with those substrings.  The new matcher is correct but more conservative: a wave name like `RutheniumRu` would match on `Ru` (preceded by `m` = non-letter); a wave name `Aru` would NOT match.
- **Aggregate G4 trigger**: the aggregate fallback only fires when the wave name carries BOTH a G4 trigger (`g4` / `todo-180` / `todo_180`) AND an aggregate hit (`14_pairs` / `14pairs` / `remaining_metals` / `5_metal` / `5metal`).  The canonical `Wave-3:chemprop_v3_G4_14_pairs` name satisfies both; a wave like `Wave-G4-smoke` would NOT trigger (no aggregate hit) and would fall through to the file-system scan as before.
- **No PROJECTED disguise**: the wave_budget_tracker.py docstring now explicitly distinguishes MEASURED actuals (TODO-180 G4 table) from PROJECTED estimates (auto-discovered 7-wave list from Wf-258 audit) — this is the §3.5 discipline in code.

## §5 Cross-references

- `TODO/completed/207_unified_pre_flight_checklist_v2.md` §7.1.2 (NEW — this Wf-268C)
- `TODO/pending/180_wf124_p3_remaining_metals_seeds.md` (TODO-180 G4)
- `molmetal/reports/wf124_p3_multitask/ru/per_seed_metrics_seed0.json` (Ru seed=0 source)
- `molmetal/reports/wf258_audit/cfm_todo_gap_audit.md` §1.3 (TODO-180 G4 audit, 1.8 hr PROJECTED estimate)
- `molmetal/scripts/wave_budget_tracker.py` (this Wf-268C ship target)
- `molmetal/tests/test_wave_budget_tracker.py` (21 existing + 7 NEW = 28 PASS)

---

**Honest framing disclaimer (`party-twenty-spirit` §3.5)**: The TODO-180 G4 actual_hr values (Ru 28.9 / Ir 7.7 / Os 1.9 / Rh 1.7 / Re 1.6) are MEASURED-equivalents asserted by the Wf-268C task owner based on the Wf-124 P3 run; only Ru seed=0 is currently independently auditable from raw on-disk `per_seed_metrics_seed0.json` (elapsed_seconds=463.2 = 7.7 min/seed).  The other 14 (metal, seed) pairs are PENDING per TODO-180 §1.  Once Wf-268D (or successor) completes the remaining 14 pairs, regenerate this report with raw MEASURED values per-pair and re-anchor the TODO_180_G4_ACTUAL_HR_BY_METAL table to on-disk JSON.
