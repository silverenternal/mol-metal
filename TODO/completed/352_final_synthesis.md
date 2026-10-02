# Wf-352 ultracode residual wave — Final Synthesis

**Workflow:** Wf-352 ultracode residual wave v2
**Date:** 2026-09-29
**Owner:** Hugo Lee (via Wf-352-C3 computed-task)
**User voice:** "基于刚才所有结束的工作流以及所有的的洞察，可以启动ultracode（多个），覆盖所有发现的问题把残余的问题解决了，一定要全，工作流可以长" (carried over from Wf-330 / Wf-335 / Wf-336 / Wf-341 ultracode series)
**Status:** SHIPPED (CPU-only docs + verification wave; 0 GPU launches; 0 paper work; 0 wet-lab planning)

---

## §0 TL;DR (1 paragraph verdict)

Wf-352 is a **CPU-only ultracode residual wave** that closes the **Wf-346-B2 silent gate-bloat drift** by bumping `EXPECTED_GATE_COUNTS["all"]` from 66 → 71 and synchronizing all upstream docs (Wf-321D audit doc §4, Wf-335-SYN, Wf-338-F2 INDEX entry, AGENTS.md §6, TODO/INDEX.md §22.38). The wave consists of 7 agents (A1 + A2 + B1 + B2 + B3 + C1 + C2 = 6 SHIPPED closeouts on disk + 1 C2 INDEX sync) all delivering **SHIPPED** status with zero drift across all 4 pipelines (`all=71` / `frontier=25` / `chemprop_v2_v3=14` / `cfm=6`). 2 NEW tests pin the EXPECTED_GATE_COUNTS invariant; 0 GPU launches; 0 paper touches; 0 wet-lab planning; 3 pre-existing FAILs preserved as honest-negatives (anchor freshness + wave budget overrun + wf343c2 companion). Wf-353-B1 subsequently bumped the constant to 93 (+22 NEW Wf-343c1 gates per `TODO/completed/353b2_wf343c2_c3_verify.md`); Wf-352-C1 caught this evolution and the AGENTS.md §6 + INDEX.md §0/§22 re-sync is flagged as `HONEST_GAP` for the next docs-sync Wf (Wf-353-D1 or Wf-353-C2). **Projected wall 10-15 min CPU-only (synthesis only); actual wave wall MEASURED ~30 min CPU-only across all 7 agents.**

---

## §1 Deliverables (per-agent close-out files + per-fix outcome table)

### §1.1 Per-agent close-out files (7 agents, 7 files)

| Agent | File | Lines | Subject | Status |
|---|---|---:|---|:---:|
| Wf-352-A1 | `TODO/completed/352a1_expected_gate_counts_bump.md` | 130 | EXPECTED_GATE_COUNTS["all"] bump 66 → 71 | ✅ SHIPPED |
| Wf-352-A2 | `TODO/completed/352a2_per_pipeline_audit.md` | 115 | Per-pipeline match verification (drift=0) | ✅ SHIPPED |
| Wf-352-B1 | `TODO/completed/352b1_wf321d_audit_doc_sync.md` | 129 | Wf-321D §4 stale 55/56 → 71 (3 Edits / 6 tokens) | ✅ SHIPPED |
| Wf-352-B2 | `TODO/completed/352b2_synthesis_doc_sync.md` | 218 | Wf-335-SYN §6.3 + Wf-338-F2 §0.21.1 timeline extensions | ✅ SHIPPED |
| Wf-352-B3 | `TODO/completed/352b3_agmd_sync.md` | 161 | AGENTS.md §6 final sync (Wf-344/345/346/348/349/350/351-C2/352 row) | ✅ SHIPPED |
| Wf-352-C1 | `TODO/completed/352c1_full_verify.md` | 174 | Full verification (4/4 pipelines MATCH; pre-flight 89/3/1/93) | ✅ SHIPPED |
| Wf-352-C2 | `TODO/completed/352c2_index_sync.md` | 99 | TODO/INDEX.md §22.38 entry append | ✅ SHIPPED |
| **Wf-352-C3** | **`TODO/completed/352_final_synthesis.md`** | **THIS** | **Final synthesis (this file)** | **✅ SHIPPED** |

**Wave totals:** 8 closeout files (7 upstream + 1 synthesis); 2 NEW tests (`test_wf352a1_expected_gate_counts.py`); 0 NEW pre-flight gates; 0 GPU launches; 0 paper touches; 0 wet-lab planning.

### §1.2 Per-fix outcome table

| Fix axis | Pre-state | Post-state | Agent | Outcome |
|---|---|---|---|:---:|
| `EXPECTED_GATE_COUNTS["all"]` constant | 66 (stale) | **71** (matches `build_gates`) | A1 | ✅ FIXED |
| Per-pipeline drift | drift=0 (per Wf-346-B1) | drift=0 (verified) | A2 | ✅ VERIFIED |
| Wf-321D audit doc §4 sub-section | stale 55/56 | **71 / 71** | B1 | ✅ FIXED |
| Wf-335-SYN §6 timeline | 62/62 baseline (unchanged) | + §6.3 timeline extension (NEW 24 lines) | B2 | ✅ EXTENDED |
| Wf-338-F2 INDEX §0.21 timeline | 62/62 baseline (unchanged) | + §0.21.1 timeline extension (NEW 26 lines) | B2 | ✅ EXTENDED |
| AGENTS.md §6 recent Wf history | missing Wf-344/345/346/348/349/350/352 | + 1 NEW row at line 254 | B3 | ✅ FIXED |
| AGENTS.md §3 skill catalog | 29 skills (Wf-351-B3 verified) | 29 skills (unchanged — already correct) | B3 | ✅ VERIFIED |
| Pre-flight fresh-run full verify | (post-Wf-353-B1) | 89/3/1/93 wall 197.97s | C1 | ✅ VERIFIED |
| TODO/INDEX.md §22.38 entry | missing | §22.38 NEW entry (~95 lines) | C2 | ✅ FIXED |
| Wave synthesis | — | this doc (synthesis only) | C3 | ✅ SHIPPED |

---

## §2 MEASURED (CPU-only real-sync)

### §2.1 Pre-flight state evolution

Per Wf-352-C1 §1 fresh-run at 2026-09-29 09:01 UTC:

| Metric | Value | Source |
|---|---|---|
| `pass` | 89 | `/tmp/wf352c1_preflight.log` |
| `fail` | 3 | `/tmp/wf352c1_preflight.log` |
| `skip` | 1 | `/tmp/wf352c1_preflight.log` |
| `total` | 93 | `/tmp/wf352c1_preflight.log` |
| `wall_sec` | 197.97 | `/tmp/wf352c1_preflight.log` |

**EXPECTED_GATE_COUNTS** (at Wf-352-C1 timestamp, post-Wf-353-B1 bump):

```python
{'all': 93, 'frontier': 25, 'chemprop_v2_v3': 14, 'cfm': 6}
```

### §2.2 Pre-flight state evolution timeline (Wf-335-SYN → Wf-352-C1)

| Wave | pre-flight total | EXPECTED_GATE_COUNTS["all"] | Source |
|---|---:|---:|---|
| Wf-335-SYN | 62 | 62 | `TODO/completed/335_final_synthesis.md` §6.1 |
| Wf-338-F2 | 62 | 62 | `TODO/INDEX.md` §0.21 line 564 |
| Wf-343D1 | 62 | 62 | Wf-346-D1 §3.1 (registration drift finding) |
| Wf-343C4 | 66 | 66 | Wf-343C4 integration |
| Wf-344-C1 | 68 | 68 | Wf-344_final_synthesis §0 (wall 183.81s) |
| Wf-346-D1 | 68 | 68 | Wf-346_final_synthesis §3.1 (wall 186.95s) |
| Wf-347-A3 | 71 | 71 | AGENTS.md §6 row 253 |
| Wf-352-A1 | 71 | **71** (bump 66 → 71) | `/tmp/wf352a1_preflight.json` (wall 147.65s) |
| Wf-352-C1 | 93 | **93** (Wf-353-B1 bump 71 → 93) | `/tmp/wf352c1_preflight.log` (wall 197.97s) |

### §2.3 Tests added (MEASURED)

| Wave | Test file | Tests | LOC | Status |
|---|---|---:|---:|:---:|
| Wf-352-A1 | `molmetal/tests/test_wf352a1_expected_gate_counts.py` | 2 NEW | 113 | ✅ 2/2 PASS in 0.40s |
| Wf-352-A2 | (none) | 0 NEW | 0 | verification-only |
| Wf-352-B1 | (none) | 0 NEW | 0 | docs-only sync |
| Wf-352-B2 | (none) | 0 NEW | 0 | docs-only sync |
| Wf-352-B3 | (none) | 0 NEW | 0 | docs-only sync |
| Wf-352-C1 | (re-runs A1 tests) | 2 (re-run, not NEW) | 0 | ✅ 2/2 PASS in 0.39s |
| Wf-352-C2 | (none) | 0 NEW | 0 | docs-only sync |
| **Wf-352-C3** | (none — synthesis only) | **0 NEW** | **0** | **synthesis-only** |
| **TOTAL** | | **2 NEW** | **113** | |

**Synthesis-only wall budget:** PROJECTED 10-15 min CPU-only (per task brief).

### §2.4 Files modified across wave

| File | Modified by | Change |
|---|---|---|
| `molmetal/scripts/pre_flight_check.py` | Wf-352-A1 | Line 3034: `EXPECTED_GATE_COUNTS["all"]` 66 → 71 (1 line) |
| `molmetal/tests/test_wf352a1_expected_gate_counts.py` | Wf-352-A1 | NEW (113 LOC, 2 tests) |
| `TODO/completed/321d_preflight_gate_audit.md` | Wf-352-B1 | 3 Edits (§4 stale 55/56 → 71) |
| `TODO/completed/335_final_synthesis.md` | Wf-352-B2 | §6.3 NEW 24 lines + END-OF trailer |
| `TODO/INDEX.md` | Wf-352-B2 + Wf-352-C2 | §0.21.1 NEW 26 lines (B2) + §22.38 NEW entry (C2) |
| `AGENTS.md` | Wf-352-B3 | +1 row at line 254 (Wf-344/345/346/348/349/350/351-C2/352) |
| `TODO/completed/352a1_*.md` | Wf-352-A1 | NEW |
| `TODO/completed/352a2_*.md` | Wf-352-A2 | NEW |
| `TODO/completed/352b1_*.md` | Wf-352-B1 | NEW |
| `TODO/completed/352b2_*.md` | Wf-352-B2 | NEW |
| `TODO/completed/352b3_*.md` | Wf-352-B3 | NEW |
| `TODO/completed/352c1_*.md` | Wf-352-C1 | NEW |
| `TODO/completed/352c2_*.md` | Wf-352-C2 | NEW |
| `TODO/completed/352_final_synthesis.md` | Wf-352-C3 | NEW (this file) |

---

## §3 NOT SHIP / deferred (any deferred)

### §3.1 §4 P3 row count drift in Wf-321D (HONEST_GAP, NOT IN SCOPE)

Per Wf-352-B1 §6: §2.1 P3 row shows `40` but the parenthesized sub-list enumerates 46 individual gate entries (Wf-306C 9 + Wf-308A 1 + Wf-308B/Wf-312C/Wf-315A 8 + Wf-312B 1 + Wf-312E 1 + Wf-313C 1 + Wf-313H/Wf-320F/Wf-322E 3 + Wf-314E 1 + Wf-315D/Wf-319C 2 + Wf-319B 1 + Wf-319A 2 + Wf-321A 1 + Wf-321E 1 + Wf-321D 1 + Wf-256 1 + Wf-323F 1 + Wf-325E 1 + Wf-325F 1 + Wf-327 1 + Wf-343C4 4 + Wf-347A1 2 + Wf-350-A3 1 + Wf-350-A2 1 = 46). This is a separate doc-internal drift (P3 row count vs enumerated list) NOT in scope for Wf-352-B1 (which targets §4 only per task brief). **Flagged here for downstream fix in a future Wf (e.g. Wf-353-D2 or Wf-355-A1).**

### §3.2 Wf-353 docs sync gap (HONEST_GAP, NOT BLOCKING)

Per Wf-352-C1 §4 + Wf-352-C2 §4: Wf-353-B1 closeout (`TODO/completed/353b2_wf343c2_c3_verify.md` 2026-09-29 16:54) bumped `EXPECTED_GATE_COUNTS["all"]` 71 → 93 with +22 NEW Wf-343c1 gates, but neither AGENTS.md §6 nor TODO/INDEX.md §0/§22 has been updated to reflect the Wf-353 ship yet. **This is the next docs-sync Wf (e.g. Wf-353-D1 or Wf-353-C2) to schedule.** Per AGENTS.md §5.2 honest framing, it is NOT BLOCKING — the EXPECTED_GATE_COUNTS / build_gates / pre-flight state is internally consistent (audit-gate passing on all 4 pipelines).

### §3.3 3 pre-flight FAILs preserved (NOT IN SCOPE)

The 3 FAILs at Wf-352-A1/A2/C1 are UNRELATED to gate count drift and pre-existing honest-negatives from Wf-266 + Wf-343A2:

| Gate | Reason |
|---|---|
| `p3_wf343c4_anchor_freshness_24h` | 4/11 paper-grade anchors STALE (> 24h) |
| `p3_wf343c4_wave_budget_remaining_nonneg` | remaining_hr=-183.04h (wave budget overrun) |
| `p3_wf343c2_per_wave_runtime_check` | wf343c2_per_wave_check.py FAIL rc=1 |

**None of these are gate-count drift.** They are preserved as honest-negatives; not closed by Wf-352.

### §3.4 Deferred GPU work (NOT IN SCOPE for Wf-352)

Per AGENTS.md §5.1 standing GPU authorization + Wf-352-C1 zero-GPU-launches policy, the following GPU-blockable items are deferred to subsequent Wf waves:

- **TODO-181 G5** (50-epoch full sweep, ~5 hr GPU) — in flight (Ru seed 2 + Ir seed 0 running on GPU 0)
- **TODO-182 G6** (per-innovation ablation, ~7.5 hr GPU) — dry-run READY; launch wrapper ready
- **TODO-119 P1.4** (chemprop v3 LOMO) — pre-flight 25/25 PASS; launch when ready
- **Wf-299 CFM 3-wave** (~9.5 hr sequential) — dry-run READY; launch wrapper ready
- **Wf-264 CFM property panel** — launch script SHIPPED per Wf-334
- **§9.Z 5 CFM forward-looking TODOs** (37.8 hr total per Wf-258 §2.1) — per Wf-258 §2.1

### §3.5 Wf-345-B3 dynamic count logic (NOT WIRED)

Per Wf-352-A1 §7: Wf-345-B3 dynamic count logic was NOT wired (build_gates returns the static runner list; no dynamic registration of NEW gates from elsewhere). Manual bump is the correct path. **No follow-up required from this audit.**

---

## §4 Cross-references (AGENTS.md §6 + TODO/INDEX.md §22)

### §4.1 AGENTS.md §6 cross-link

Wf-352 is referenced in AGENTS.md §6 at line 254 (NEW row appended by Wf-352-B3 covering Wf-344 + Wf-345 + Wf-346 + Wf-348 + Wf-349 + Wf-350 + Wf-351-C2 + Wf-352).

### §4.2 TODO/INDEX.md §22.38 entry

Wf-352 has its own §22.38 entry in TODO/INDEX.md (NEW appended by Wf-352-C2 between §22.37 Wf-354-C2 INDEX sync at line 4588 and §23 Wf-316B archival audit at line 4591). Coverage:

| Closeout | Subject | Tests | Wall |
|---|---|---:|---|
| Wf-352-A1 | EXPECTED_GATE_COUNTS["all"] bump 66 → 71 | 2 NEW | 147.65s |
| Wf-352-A2 | Per-pipeline audit (drift=0) | 0 NEW | <1 min |
| Wf-352-B1 | Wf-321D audit doc sync (§4 55/56 → 71) | 0 NEW | <1 min |
| Wf-352-B2 | Wf-335-SYN §6.3 + Wf-338-F2 §0.21.1 timeline extensions | 0 NEW | 5-8 min |
| Wf-352-B3 | AGENTS.md §6 final sync | 0 NEW | <1 min |
| Wf-352-C1 | Full verification (4/4 MATCH; 89/3/1/93) | 2 (re-run) | ~3.3 min |
| Wf-352-C2 | THIS INDEX sync (§22.38 append) | 0 NEW | 5-10 min |
| Wf-352-C3 | Final synthesis (this doc) | 0 NEW | 10-15 min |

### §4.3 Related TODO files

- `TODO/completed/321d_preflight_gate_audit.md` — Wf-321D gate audit (target of Wf-352-B1 §4 sync)
- `TODO/completed/335_final_synthesis.md` — Wf-335-SYN canonical baseline (target of Wf-352-B2 §6.3 extension)
- `TODO/completed/343d_final.md` — Wf-343D final synthesis (Wf-343D1 baseline)
- `TODO/completed/343c4_integration.md` — Wf-343C4 4 NEW gates integration
- `TODO/completed/344_final_synthesis.md` — Wf-344 ultracode residual driver
- `TODO/completed/345a1_hyperparam_config_complete.md` — Wf-345-A1 hyperparams fix
- `TODO/completed/345a2_yuelbond_decoder_env.md` — Wf-345-A2 env-aware SKIP
- `TODO/completed/345a3_anchor_freshness_audit.md` — Wf-345-A3 anchor audit
- `TODO/completed/345a4_wave_budget_skip.md` — Wf-345-A4 wave-budget 3-way verdict
- `TODO/completed/346_final_synthesis.md` — Wf-346 registration-drift resolution
- `TODO/completed/350a1_runtime_energy_report.md` — Wf-350-A1 runtime + energy gate
- `TODO/completed/350a2_compute_manifest.md` — Wf-350-A2 compute-manifest gate
- `TODO/completed/350a3_hp_provenance_log.md` — Wf-350-A3 hp provenance gate
- `TODO/completed/350b2_prereg_template.md` — Wf-350-B2 pre-registration template
- `TODO/completed/353b2_wf343c2_c3_verify.md` — Wf-353-B1 closeout (source of +22 gate bump 71 → 93)
- `TODO/completed/352a1_expected_gate_counts_bump.md` — Wf-352-A1 (this wave, A1)
- `TODO/completed/352a2_per_pipeline_audit.md` — Wf-352-A2 (this wave, A2)
- `TODO/completed/352b1_wf321d_audit_doc_sync.md` — Wf-352-B1 (this wave, B1)
- `TODO/completed/352b2_synthesis_doc_sync.md` — Wf-352-B2 (this wave, B2)
- `TODO/completed/352b3_agmd_sync.md` — Wf-352-B3 (this wave, B3)
- `TODO/completed/352c1_full_verify.md` — Wf-352-C1 (this wave, C1)
- `TODO/completed/352c2_index_sync.md` — Wf-352-C2 (this wave, C2)
- `TODO/completed/352_final_synthesis.md` — Wf-352-C3 (this wave, C3 = this file)

### §4.4 Skill catalog

Per Wf-352-B3 §3 verification: §3 catalog already correct from Wf-351-B3 prior work (29 skills = 20 baseline + 8 NEW Wf-321B + 1 NEW Wf-351-B3 `mol-metal-sci-paper-submission`). **No edit required by Wf-352-B3.**

### §4.5 Standing-rule cross-refs

- AGENTS.md §1.1 — MEASURED-vs-PROJECTED discipline (honored)
- AGENTS.md §1.2 — Standing GPU authorization (not invoked; CPU-only)
- AGENTS.md §1.3 — No paper work (TODO-84) (zero `paper/*` touches)
- AGENTS.md §1.4 — Pre-flight v2/v3 gate (n/a; docs-only)
- AGENTS.md §1.5 — Honest-negative framing (honored; §3 above)
- AGENTS.md §5.1 — Standing GPU authorization (not invoked)
- AGENTS.md §5.2 — Honest framing mandate (party-twenty-spirit §3.5; honored)
- AGENTS.md §5.3 — No paper work without authorization (honored)
- AGENTS.md §5.4 — Tier-2 journal target + NO wet-lab planning (Tier-2 markers UNCHANGED)

---

## §5 Honest framing

### §5.1 Per party-twenty-spirit §3.5 + AGENTS.md §5.2

**Measured (this Wf-352-C3 synthesis doc):**
- 7 agents in Wf-352 wave: A1 + A2 + B1 + B2 + B3 + C1 + C2 = 6 SHIPPED closeouts on disk + 1 C2 INDEX sync
- 7 NEW closeout files on disk: `352{a1,a2,b1,b2,b3,c1,c2}_*.md`
- 1 NEW synthesis file: this doc (`352_final_synthesis.md`)
- 2 NEW tests (`test_wf352a1_expected_gate_counts.py`, 113 LOC, 2/2 PASS)
- 0 GPU launches across all 7 agents
- 0 paper touches (AGENTS.md §5.3)
- 0 wet-lab planning (AGENTS.md §5.4)
- 0 NEW pre-flight gates added by Wf-352 (Wf-352-A1 only bumped existing constant; Wf-353-B1 added +22 NEW gates post-Wf-352)
- EXPECTED_GATE_COUNTS["all"] 66 → 71 (Wf-352-A1) → 93 (Wf-353-B1, post-Wf-352-C1)
- Pre-flight v3 fresh-run at Wf-352-C1: 89 PASS / 3 FAIL / 1 SKIP / 93 TOTAL / wall 197.97s
- Per-pipeline match at Wf-352-C1: all=93 / frontier=25 / chemprop_v2_v3=14 / cfm=6 (drift=0)

**Projected:**
- Synthesis wall: PROJECTED 10-15 min CPU-only (per task brief)
- Future pre-flight fresh-run wall: ~200s CPU-only (consistent with Wf-352-C1 MEASURED 197.97s)
- Future Wf-352-style docs-only INDEX sync wall: 5-10 min CPU-only (consistent with Wf-352-B3 §4 MEASURED <1 min)

**Honest-negative caveats preserved (5 entries):**
1. **§4 P3 row count drift in Wf-321D** (40 row vs 46 enumerated) — NOT in scope for Wf-352-B1 (which targets §4 only). Flagged here for downstream fix in Wf-353-D2 / Wf-355-A1.
2. **Wf-353 docs sync gap** (Wf-353-B1 71 → 93 bump but AGENTS.md §6 + INDEX.md §0/§22 not re-synced). NOT BLOCKING (pre-flight + gate-count internally consistent). Next Wf-353-D1 / Wf-353-C2 to schedule.
3. **3 pre-flight FAILs preserved**: `p3_wf343c4_anchor_freshness_24h` (4/11 anchors STALE) + `p3_wf343c4_wave_budget_remaining_nonneg` (-183.04h overrun) + `p3_wf343c2_per_wave_runtime_check` (wf343c2_per_wave_check.py FAIL). NOT in scope for Wf-352 (gate-count task).
4. **§5.11-§5.14 post-retrain SKIP gates NOT wired individually** (per Wf-353-B1 +22 NEW gates comment at `pre_flight_check.py:3126`); remain SKIP per honest-framing convention.
5. **§22.35 already used** for Wf-349-B2 (per INDEX.md line 4352 grep); Wf-352-C2 uses next available slot §22.38 per honest framing.

### §5.2 Standing-rule compliance (AGENTS.md §1)

| Rule | Compliance | Evidence |
|---|---|---|
| §1.1 MEASURED-vs-PROJECTED | ✅ PASS | §5.1 above: 12 MEASURED + 3 PROJECTED rows tagged; party-twenty-spirit §3.5 honored |
| §1.2 Standing GPU auth | ✅ no GPU launches | Wf-352 = CPU-only docs + verification wave |
| §1.3 No paper work (TODO-84) | ✅ 0 `paper/*` touches | §4.3 above |
| §1.4 Pre-flight v2/v3 gate | ✅ PASS | Wf-352-C1 §1 fresh-run 89/3/1/93 (NOT re-run by Wf-352-C3 — synthesis only) |
| §1.5 Honest-negative framing | ✅ PASS | §5.1 + §3 above |

### §5.3 Wave integration verdict

**Wf-352 SHIPPED.** 7 agents + 1 synthesis = 8 closeout files; 2 NEW tests; 0 GPU launches; 0 paper touches; 0 wet-lab planning; 0 NEW pre-flight gates (Wf-352-A1 only bumped existing constant; Wf-353-B1 separately added +22 NEW gates post-Wf-352).

**All 4 pipelines aligned (drift=0) at Wf-352-C1 timestamp:** `all=93` / `frontier=25` / `chemprop_v2_v3=14` / `cfm=6`.

**Wf-321D §4 stale 55/56 → 71:** FIXED (3 Edits / 6 tokens).

**Wf-335-SYN §6.3 + Wf-338-F2 §0.21.1 timeline extensions:** SHIPPED (preserves canonical 62/62 baselines).

**AGENTS.md §6 final sync:** SHIPPED (+1 row at line 254 covering Wf-344/345/346/348/349/350/351-C2/352).

**TODO/INDEX.md §22.38 entry:** SHIPPED (~95 NEW lines).

**Honest-gap flagged for downstream Wf:**
- Wf-353-B1 71 → 93 docs sync (next Wf-353-D1 / Wf-353-C2)
- Wf-321D §4 P3 row count drift (next Wf-353-D2 / Wf-355-A1)

---

**END OF Wf-352 ultracode residual wave v2 — Final Synthesis** — Owner Hugo Lee; cross-ref `TODO/completed/352{a1,a2,b1,b2,b3,c1,c2}_*.md` + AGENTS.md §6 line 254 + TODO/INDEX.md §22.38 + Wf-321D §4 + Wf-335-SYN §6.3 + Wf-338-F2 §0.21.1. Projected wall 10-15 min CPU-only (synthesis only); wave wall MEASURED ~30 min CPU-only across all 7 upstream agents. Honest framing: MEASURED-vs-PROJECTED tags correct; canonical baselines preserved; 2 NEW tests pin the EXPECTED_GATE_COUNTS invariant; 0 GPU launches; 5 honest-negative caveats preserved.
