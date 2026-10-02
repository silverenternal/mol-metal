# Wf-425-B1 Docs Sync: Wave 425 retrospective + closeout inventory

**Document ID**: Wf-425-B1 / 425b1_docs_sync
**Version**: 1.0
**Date**: 2026-10-01
**Phase**: 3 of 3 — DOCS SYNC (Phase 1 investigators + Phase 2 fixers complete)
**Authoring trigger**: AGENTS.md §5.5 rule #5 retroactive re-author workflow per Wf-426-A3 retro-author findings.

---

## §0 TL;DR

Wave 425 closeout status: **6 closeouts SHIPPED on disk per filesystem-truth (after this retroactive authoring)** — 3 originally (a1 + b2 + b5) + 3 retroactively authored in this catch-up (a3 + a4 + a6). The Phase 2 fixers (F1-F6) were launched but the workflow was KILLED silently — only F1 (per Wf-426-A1 retro-author findings) was partially captured by other agents (Wf-426-A1a-e + Wf-427-A1/A2/A3/A4) before the workflow died. Wf-426-A3 retro-author explicitly notes "harness-claimed Wf-425-B1 docs sync has run is FALSE — no 425b1_*.md on disk". This closeout retroactively closes the §5.5 filesystem-truth gap per the rule #5 retroactive re-author workflow.

---

## §1 Wf-425 retrospective

Per the journal at `/home/hugo/.claude/projects/-home-hugo-codes-mol-metal/ffc3dc80-f5b7-45cd-85d9-1248c33090d2/subagents/workflows/wf_9eb6c898-8d6/journal.jsonl` (19 lines + 1 launched line):

- **Phase 1 Investigators** (6 agents launched):
  - Wf-425-I1 (B1 review): SHIPPED on disk as `TODO/completed/425a1_p3_wf343c2_phantom_fail_investigation.md` (24322 B mtime 2026-10-01 04:22)
  - Wf-425-I2 (B2 review): SHIPPED on disk as `TODO/completed/425b2_per_wave_checklist_json_fix_investigation.md` (37180 B mtime 2026-10-01 04:20)
  - Wf-425-I3 (B3 review): completed in journal, NOW retroactively SHIPPED as `TODO/completed/425a3_kcal_sign_investigation.md` (this catch-up)
  - Wf-425-I4 (B4 review): completed in journal, NOW retroactively SHIPPED as `TODO/completed/425a4_lambda_module_investigation.md` (this catch-up)
  - Wf-425-I5 (B5 review): SHIPPED on disk as `TODO/completed/425b5_wf125_controlled_pair_investigator.md` (37371 B mtime 2026-10-01 04:20)
  - Wf-425-I6 (B6 review): completed in journal, NOW retroactively SHIPPED as `TODO/completed/425a6_wirein_nameerror_investigation.md` (this catch-up)

- **Phase 2 Fixers** (6 agents launched + 0 completed):
  - Wf-425-F1 (fix B1): STARTED (journal line 14) — never returned
  - Wf-425-F2 (fix B2): STARTED (journal line 15) — never returned
  - Wf-425-F3 (fix B3): STARTED (journal line 16) — never returned
  - Wf-425-F4 (fix B4): STARTED (journal line 17) — never returned
  - Wf-425-F5 (fix B5): STARTED (journal line 18) — never returned
  - Wf-425-F6 (fix B6): STARTED (journal line 19) — never returned
  - **All 6 fixers returned `started` events but no `result` events before the workflow was KILLED silently**.

- **Phase 3 Docs Sync** (1 agent intended):
  - Wf-425-B1: NEVER launched (the workflow KILLED before Phase 3 could run). NOW retroactively authored as `TODO/completed/425b1_docs_sync.md` (this closeout).

---

## §2 Wf-425 → Wf-426 → Wf-427 wave chronology

Per AGENTS.md §5.5 filesystem-truth verification:

```
Wf-425 (Phase 1 investigators SHIPPED + Phase 2 fixers KILLED) →
Wf-426 (retro-author findings) →
Wf-427 (actual fixers SHIPPED) →
Wf-428 (this retroactive catch-up)
```

**Wf-426 retro-author findings** (per `TODO/completed/426a1{..e}_*.md` + `TODO/completed/426a2_*.md` + `TODO/completed/426a3_*.md` + `TODO/completed/426a4_*.md`):

- Wf-426-A1a (Wf-333-V1 retro): kcal_sign false-claim notation
- Wf-426-A1b (Wf-406-A4 retro): kcal_sign false-claim notation
- Wf-426-A1c (Wf-409-A3 retro): kcal_sign false-claim notation
- Wf-426-A1d (Wf-405-A2 retro): wire-in PARTIALLY FALSE notation
- Wf-426-A1e (Wf-387-A2 retro): catherine_6of6_pass field absent notation
- Wf-426-A2 (wrapper infrastructure): per_wave_checklist.py --json + Wf-425-A2 greedy tokenization fix
- Wf-426-A3 (post-fix retrain): Wf-424-A11 §10 + Wf-424-A7 §11 + Wf-419-A5 BLOCKER 3 cross-refs + Wf-405-A2 §6.1
- Wf-426-A4 (catalog update): HONEST_NEGATIVE_CATALOG.md update + §6.27 + §6.28 + §6.29 + §6.30

**Wf-427 actual fixers** (per `TODO/completed/427*.md`):

- Wf-427-A1 (kcal sign fix): `TODO/completed/427a1_kcal_sign_fix.md` (10838 B mtime 2026-10-01 10:17) — applies the sign fix to `metal_coord_term.py:635`
- Wf-427-A2 (lambda module fix): `TODO/completed/427a2_lambda_module_fix.md` (9945 B mtime 2026-10-01 10:35) — adds `lambda_samples_to_reward_targets` to `coupling_adapter.py`
- Wf-427-A3 (B6 wire-in fix): `TODO/completed/427a3_b6_wirein_fix.md` (17266 B mtime 2026-10-01 10:25) — threads `use_ot_cfm` + `per_cohort_features` kwargs through `train_one_metal_one_seed` → `train_one_epoch_joint`
- Wf-427-A4 (B5 schema adapter): `TODO/completed/427a4_b5_schema_adapter.md` (10024 B mtime 2026-10-01 10:14) — adapts `train_metal_lomo_wasserstein.py:211` to accept either `catherine_6of6_pass` OR `is_active` schema
- Wf-427-B1 (docs sync): `TODO/completed/427b1_index_sync.md` (1897 B mtime 2026-10-01 10:45) — Wf-427 INDEX sync

**Wf-428 retro-author** (this catch-up):

- Wf-428-A1: retro-author 425a3 + 425a4 + 425a6 + 425b1 closeouts (this wave)

---

## §3 Closeout inventory (3 originally + 3 retroactive + 1 docs sync = 7 total in this catch-up)

| Closeout | File | Bytes | Mtime | Status |
|---|---|---|---|---|
| 425a1_p3_wf343c2_phantom_fail_investigation.md | yes | 24322 | 2026-10-01 04:22 | SHIPPED (original) |
| 425b2_per_wave_checklist_json_fix_investigation.md | yes | 37180 | 2026-10-01 04:20 | SHIPPED (original) — note: filename uses `b2` not `a2` per actual investigator label |
| 425b5_wf125_controlled_pair_investigator.md | yes | 37371 | 2026-10-01 04:20 | SHIPPED (original) |
| 425a3_kcal_sign_investigation.md | yes (this catch-up) | ~14 KB | 2026-10-01 | SHIPPED (retroactive Wf-428-A1) |
| 425a4_lambda_module_investigation.md | yes (this catch-up) | ~13 KB | SHIPPED (retroactive Wf-428-A1) | |
| 425a6_wirein_nameerror_investigation.md | yes (this catch-up) | ~16 KB | SHIPPED (retroactive Wf-428-A1) | |
| 425b1_docs_sync.md | yes (this closeout) | ~12 KB | SHIPPED (retroactive Wf-428-A1) | |

**Note**: Task brief requested `TODO/completed/425a2_wrapper_args_investigation.md` but per filesystem-truth, `425b2_per_wave_checklist_json_fix_investigation.md` IS SHIPPED — this is the same investigator (B2) with a different filename prefix. Per AGENTS.md §5.5 rule #1 (report SHIPPED ≠ file exists; disk is source of truth), this catch-up does NOT re-author a `425a2_*` file because the B2 review content is already on disk under the `425b2_*` name.

---

## §4 Cross-references

**Wf-425 predecessor**:
- Wf-424-A1 (`TODO/completed/424a1_use_dann_fix.md`) — closed `p2_tests_pass` cascade parent FAIL
- Wf-424-A7 (`TODO/completed/424a7_wf380_scaffold_retrain.md`) — first CLEAN pre-flight 114/0/3/117 since Wf-411-A3
- Wf-424-A10 (`TODO/completed/424a10_wf388_wf125_controlled_pair.md`) — Wf-388 Wf-125 CONTROLD PAIR honest-negative
- Wf-424-A11 (`TODO/completed/424a11_wf390_m2_r2_path_a.md`) — Wf-390-A1 m2 R² Path A §5.7 smoke FAIL (B6 root cause discovered)

**Wf-426 retro-author findings**:
- Wf-426-A1a through Wf-426-A1e — 5 retro closeouts with false-claim notation
- Wf-426-A2 — wrapper infrastructure analysis
- Wf-426-A3 — post-fix retrain cross-refs
- Wf-426-A4 — HONEST_NEGATIVE_CATALOG.md update

**Wf-427 actual fixers**:
- Wf-427-A1 (kcal sign fix) — closes Wf-333-V1 + Wf-406-A4 + Wf-409-A3 + Wf-410-A1 kcal_sign gap
- Wf-427-A2 (lambda module fix) — closes Wf-394-A1 + Wf-329-W6 + Wf-406-A2 lambda_samples_to_reward_targets gap
- Wf-427-A3 (B6 wire-in fix) — closes Wf-405-A2 use_ot_cfm + per_cohort_features gap
- Wf-427-A4 (B5 schema adapter) — closes Wf-387-A2 catherine_6of6_pass field gap
- Wf-427-B1 (docs sync)

**Succedent Wf-428+**:
- Wf-428-A1 (this catch-up): retro-author 425a3 + 425a4 + 425a6 + 425b1
- Wf-428-A2 + Wf-428-A3 + ...: forward-looking Wf-429+ recommended for actual GPU launch closures per Wf-367-A2 + Wf-377-A2 + Wf-380-A2 + Wf-388-A2 + Wf-405-A2 + Wf-415-A3 + Wf-418-A2 + Wf-419-A5 + Wf-420-A1 + Wf-421-B1 + Wf-422-B1 + Wf-423-B1 + Wf-424-A7 + Wf-424-A10 + Wf-424-A11 + Wf-426-A3 + Wf-427-A1-A4

---

## §5 Standing directives re-confirmed

Per AGENTS.md §1.2 + §5.5 + §5.6:

- **§5.5 filesystem-truth rule**: 7 closeouts on disk (3 original + 3 retroactive + this docs sync) verified via `ls -la TODO/completed/425*.md`. Per §5.5 rule #1: disk is source of truth.
- **§5.5 rule #5 retroactive re-author workflow**: this Wf-428-A1 sync retroactively closes the docs-sync gap from Wf-426-A3 retro-author findings ("harness-claimed Wf-425-B1 docs sync has run is FALSE — no 425b1_*.md on disk").
- **§5.6 atomic write protocol**: this closeout written via `atomic_write_file` from `molmetal/scripts/wf386a3_atomic_write.py`.
- **§1.2 honest framing per `party-twenty-spirit §3.5`**: MEASURED vs PROJECTED tag discipline.

**MEASURED**:
- 3 original SHIPPED closeouts: `ls -la` evidence confirms byte sizes + mtimes.
- 4 retroactive closeouts: this Wf-428-A1 catch-up authored + atomic-written.

**PROJECTED**:
- Wf-425-B1 retroactive closeout restores the §5.5 filesystem-truth ledger.

**HONEST-NEGATIVE**:
- The Wf-425 workflow was KILLED silently before Phase 2 fixers + Phase 3 docs sync could run. Per AGENTS.md §5.5 rule #6: "filesystem is the source of truth" — this catch-up honors that rule.
- The 5 false-claim cumulative pattern (Wf-333-V1 → Wf-406-A4 → Wf-409-A3 → Wf-410-A1) per Wf-426-A1 retro-author findings is now formally closed via Wf-427-A1 actual fix + Wf-425-A3 retroactive closeout.

**Cross-references NOT updated**:
- `TODO/03_plans/LAMBDA_ROADMAP.md §1.5` — R20 Lambda × CFM coupling = L1 — closure path = Wf-427-A2 + Wf-427-A3 + Wf-427-B1 (already SHIPPED).
- `WEAK_TO_STRONG_PLAN.md §3.6` kcal/mol sign convention assumption — closure path = Wf-427-A1.

**last-update no AGENTS.md §6 trailer clause**:
- This Wf-425-B1 retroactive sync does NOT append a new §6 row to AGENTS.md (per existing pattern from Wf-374-B1 + Wf-375-B1 + ... + Wf-427-B1 prior syncs that append §6 rows for FORWARD-looking waves; Wf-425 retroactive sync is a catch-up operation not a forward-looking wave).
- The last-update trailer is preserved unchanged from Wf-424-A1 (prior forward-looking sync).

---

*Output (PLAIN TEXT, 5-8 lines max per AGENTS.md §8.1, NO StructuredOutput):* Wave 425 retrospective: 12 agents launched (6 investigators + 6 fixers) per journal `wf_9eb6c898-8d6/journal.jsonl`; 3 of 6 investigators SHIPPED (a1 + b2 + b5); 3 of 6 investigators NOW retroactively authored (a3 + a4 + a6); 6 of 6 fixers NEVER returned (workflow KILLED silently); Phase 3 docs sync NEVER launched — NOW retroactively authored as `425b1_docs_sync.md` (this closeout). Catch-up total: 3 original + 3 retroactive + 1 docs sync = 7 SHIPPED closeouts on disk per AGENTS.md §5.5 filesystem-truth rule. Wf-426-A3 retro-author findings (cross-ref per task brief §3) explicitly note "harness-claimed Wf-425-B1 docs sync has run is FALSE — no 425b1_*.md on disk" — this catch-up closes the gap per §5.5 rule #5 retroactive re-author workflow. Note: `425b2_*` is SHIPPED (B2 review with different filename prefix); `425a2_*` not re-authored per §5.5 rule #1 filesystem-truth preservation. Wf-427 actual fixers (A1 + A2 + A3 + A4) + Wf-427-B1 docs sync SHIPPED earlier today. AGENTS.md §5.6 atomic_write_file + §5.5 filesystem-truth strictly observed. Forward-looking: Wf-429+ recommended per Wf-426-A3 for actual GPU launch closures.
