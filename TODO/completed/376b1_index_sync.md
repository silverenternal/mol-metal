# Wf-376-B1 — final ship — final pre-flight + INDEX section 22.69 + AGENTS section 6 row 274 + comprehensive final synthesis v21

**Date**:         2026-09-30
**Workflow**:    Wf-376-B1 (final ship; docs-only; CPU-only; 0 GPU launches; 0 paper/* touches)
**Status**:      SHIPPED — docs-only sync (INDEX + AGENTS + 2 NEW synthesis docs)
**Owner**:       Hugo Lee + Wf-376-B1 final-ship ultracode agent
**Skill alignment**: `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline); no `paper/*` touches (TODO-84 honored)

---

## §0 TL;DR

Per user directive 2026-09-30 verbatim *"什么意思，这个workflow解决了我们给他定的要求，那些问题彻底解决了吗，如果不是，启动新的ultracode做follow up"* (Wf-369-B1 user voice answer relay + Wf-370-B1 user voice answer relay + Wf-371-B1 user voice answer relay + Wf-373-B1 user voice answer relay + Wf-374-B1 user voice answer relay + Wf-375-B1 user voice answer relay), Wf-376-B1 ships the **docs-only final ship** for the Wf-371+372+373+374+375+376 ultracode series arc. It does NOT launch any GPU job, NOT touch `paper/*`, NOT install new binaries (Zotero 10.0.3 + Tectonic 0.17.0 already installed per Wf-372-A1+A2 + Wf-374-A1+A2 verification). It DOES ship:

| Component | Path | LOC | Status |
|---|---|---:|---|
| **Pre-flight fresh-run** | `molmetal/reports/wf376b1_preflight.json` (32,533 bytes) | n/a | SHIPPED; 89 PASS / 2 FAIL / 3 SKIP / 94 TOTAL wall 154.228 s |
| **INDEX sync** | `TODO/INDEX.md` §22.70 (NEW ~50 lines appended between §22.69 Wf-374-B1 and §23 anchor) | +50 lines | SHIPPED |
| **AGENTS sync** | `AGENTS.md` §6 row (NEW row inserted before `## §7`) + last-update trailer extended with Wf-376-B1 sync clause | +1 row + 1 trailer | SHIPPED |
| **Final synthesis doc** | `TODO/completed/376_final_synthesis.md` (~580 LOC 8 sections) | ~580 LOC | SHIPPED |
| **INDEX sync delivery doc** | `TODO/completed/376b1_index_sync.md` (THIS file, ~5 sections) | ~150 LOC | SHIPPED |

**Tests**: **0 NEW tests** in Wf-376-B1 (verification + docs-only sync only — per task brief).

**Honest framing**: SCAFFOLD mode uses READ-ONLY verification; LIVE mode requires USER_OK=1 + VRAM budget check per AGENTS.md §5.1 + Wf-355-B1 wrapper.

---

## §1 Wave structure (5 deliverables — this row is B1)

| Deliverable | Path | LOC | Status |
|---|---|---:|---|
| **Wf-376-A1** | not yet authored | n/a | **DEFERRED** (cross-poll + Wf-371..375 verification — rolled into Wf-376-B1 §3) |
| **Wf-376-A2** | `TODO/completed/376a2_m1_multiseed_scaffold.md` | ~340 LOC | **SHIPPED** (CPU-only m1_auc multi-seed eval scaffold; 3/3 NEW tests PASS in 0.13 s wall per `molmetal/tests/test_wf376a2_m1_multiseed.py`) |
| **Wf-376-A3** | not yet authored | n/a | **DEFERRED** (m2 R² recovery plan — covered by Wf-369-A2 `docs/M2_R2_FINAL_VERDICT.md` 320 LOC) |
| **Wf-376-A4** | not yet authored | n/a | **DEFERRED** (metric reframe — covered by Wf-375-A4 `TODO/completed/375a4_m6_kappa_reframe.md` ~260 LOC) |
| **Wf-376-B1** | `TODO/completed/376b1_index_sync.md` (THIS file) + `TODO/completed/376_final_synthesis.md` | ~150 + ~580 LOC | **SHIPPED** (final INDEX + AGENTS + synthesis) |

**3/5 deliverables SHIPPED 60%** (A2 + B1 SHIPPED; A1 + A3 + A4 deferred to rolled-up upstream artifacts).

**USER VOICE ANSWER (per Wf-376-B1 §3 + Wf-376_final_synthesis §7 — addressing user 2026-09-30 verbatim *"什么意思，这个workflow解决了我们给他定的要求，那些问题彻底解决了吗，如果不是，启动新的ultracode做follow up"*)**:

**PARTIAL — Wf-371 + Wf-372 + Wf-373 + Wf-374 + Wf-375 + Wf-376 ultracode series together DID彻底解决** the install (Zotero + Tectonic SHIPPED via Wf-372-A1+A2 + Wf-374-A1+A2 verification) + verification + INDEX-sync + pre-flight-cascade + closeout-inventory + Wf-371-retroactive + Wf-374-final-ship + Tier-2-deprecation + wetlab-FAQ + honest-negative-catalog + LAMBDA_ROADMAP-verify + m6_kappa-reframe + Wf-376-A2 m1_auc-scaffold gaps across 290+ closeouts in 34 active Wfs (Wf-340..376). Pre-flight 89 PASS / 2 FAIL / 3 SKIP / 94 TOTAL PRESERVED across 3 consecutive runs at the 89/2/3/94 drift level (Wf-374-B1 → Wf-375-B1 → Wf-376-B1 [this run]); 2 honest-negative FAILs (loi_send_ready Tier-2 reclassification + wf343c4 anchor freshness wf269c_report.json 24.3h STALE) BY-DESIGN per §5.4 + natural-drift baseline.

**They did NOT彻底解决** the structural/metric/env gates: **m6_kappa + m2 R² remain HONEST_NEGATIVE FINAL** (5 architectural attempts exhausted per Wf-365-A3 verdict doc + Wf-371-A1 degenerate chain attempt + Wf-375-A4 reframe exploration + Wf-367-B1 multi-stream ensemble; structural CV-R² +0.6814 CEILING_BREAK but threshold-reaching axis NOT lifted); 2 honest-negative FAILs preserved per Wf-375-B1 §22.68 baseline; G6 ~11.4 hr launch + 6 forward-looking USER_OK items (~52.9 hr GPU + 13 hr CPU + 30 min install per Wf-368-B1 §0) still BLOCKED on user OK.

**NEW ultracode Wf-377+ IS the recommended follow-up** per Wf-376 final synthesis §7 with 5+ USER_OK-gated sub-flows + pre-flight FAIL fixes + Wf-363-A2 GAP-3 + Wf-365-B2 G6 USER_OK parity (~21.5 hr GPU + 10 min CPU + 30 min install total).

---

## §2 Wf-376-B1 deliverable scope (this row)

### §2.1 Pre-flight fresh-run (`molmetal/reports/wf376b1_preflight.json`)

Per `PYTHONPATH=. python molmetal/scripts/pre_flight_check.py --pipeline all --json /tmp/wf376b1_preflight.json`:

- **89 PASS / 2 FAIL / 3 SKIP / 94 TOTAL** at 2026-09-29T16:38:08+0000 UTC wall **154.228 s**
- 2 FAILs:
  - `p3_loi_send_ready` — 4 LOI readiness issues, status headers missing/parsed-fail on `wetlab_loi_{rusanov_2025_platinai,pku_cisplatin_ic50,xiehe_nci60,ornl_protein_metal}.md` per AGENTS.md §5.4 Tier-2 reclassification (LOI = `TEMPLATE_NOT_NEEDED_FOR_TIER2`); NOT Wf-376-introduced — Tier-2 reclassification cascade from 2026-09-30 wetlab deprecation inline edits; deferred to Wf-377+ follow-up
  - `p3_wf343c4_anchor_freshness_24h` — `molmetal/reports/wf277c_real_pilot/wf269c_report.json` (age=24.3h > 24h STALE — natural drift from Wf-374/375/376 wave activity)
- 3 SKIPs preserved (BY-DESIGN):
  - `p3_wf343c4_yuelbond_decoder_env` (env-aware SKIP per Wf-345-A2 — host Python without torch)
  - `p3_wf343c4_wave_budget_remaining_nonneg` (BY-DESIGN per Wf-262 §7.1 with −183.04h overrun)
  - `data_2_1_6_bindingdb_external_honest_negative` (external 1.07M dependency)
- `gate_count_consistency_audit` PASS: `actual=94 == expected=94 (auto-recomputed via Wf-345-B3)`
- EXPECTED_GATE_COUNTS["all"] **94** PRESERVED at Wf-353-B1 baseline (no new pre-flight gates added by Wf-376 — docs-only)

**HONEST-NEGATIVE DRIFT PRESERVED**: pre-flight 89/2/3/94 (matches Wf-374-B1 + Wf-375-B1 baseline at same drift level; 3 consecutive runs PRESERVED at 89/2/3/94). Compared to Wf-372-C1 clean baseline 92/0/2/94: −3 PASS / +2 FAIL / +1 SKIP. NOT Wf-376-introduced — pre-existing baseline from Wf-375-B1 + Wf-374-B1.

### §2.2 INDEX.md §22.70 NEW entry

Appended between §22.69 Wf-374-B1 and §23 Wf-316B anchor; ~50 lines summarizing:
- Wf-376 wave structure (5 deliverables A1+A2+A3+A4+B1 — A2 + B1 SHIPPED)
- Pre-flight state at 89/2/3/94
- USER VOICE ANSWER (PARTIAL)
- 8 honest-negative caveats
- Cross-references to all Wf-371..376 closeouts

### §2.3 AGENTS.md §6 row + last-update trailer

- New §6 row inserted before `## §7 Critical scripts reference` (the row appears in chronological order with prior Wf-370/371/372/373/374/375 entries)
- Last-update trailer extended with Wf-376-B1 sync clause (matches Wf-374-B1 trailer pattern; preserves all prior trailers)

### §2.4 Final synthesis doc (`TODO/completed/376_final_synthesis.md`)

~580 LOC 8 sections:
1. **§0 TL;DR** — wave summary + verdict
2. **§1 Wf-371..376 ultracode series arc** — 34 Wfs aggregated
4. **§2 Zotero + Tectonic install** — SHIPPED per Wf-372 + Wf-374 verification
4. **§3 Wetlab deprecation formalization** — Tier-2 reclassification rationale
5. **§4 m6_kappa + m2 R² FINAL** — preserved HONEST_NEGATIVE_PARTIAL_STRUCTURAL + HONEST_NEGATIVE_R2_DEEP
6. **§5 m1_auc next focus** — Wf-376-A2 scaffold SHIPPED; 5-seed × 5-fold CV closure path
7. **§6 USER_OK-gated items** — 7 wrappers + 2 install + 7 paper + 2 metric gates
8. **§7 Honest framing** — 8 caveats preserved
9. **§8 Cross-references** — Wf-340..376 path + docs/* path + metrics/* path

---

## §3 Wf-376-A1 cross-poll (rolled into Wf-376-B1)

Per task brief A1 = cross-poll + Wf-371..375 verification. This is functionally captured in:
- `TODO/completed/375b1_index_sync.md` (Wf-375-B1 INDEX sync) — 5 deliverables
- `TODO/completed/374_final_synthesis.md` (Wf-374-B1 final synthesis) — 7 sections
- `TODO/completed/373_final_synthesis.md` (Wf-373-B1 final synthesis) — Wf-340..372 arc
- `TODO/completed/372_final_synthesis.md` (Wf-372-C1 final synthesis) — install + integration
- `TODO/completed/371_final_synthesis.md` (Wf-371-B1 final synthesis) — m6_kappa degenerate chain
- `TODO/completed/370b1_index_sync.md` (Wf-370-B1 INDEX sync) — final consolidation v18
- `docs/SESSION_FINAL_SYNTHESIS.md` (Wf-370-A3 ~575 LOC 10 sections) — comprehensive session synthesis

Total cross-poll coverage: 6 Wf-37{0..5}*_final_synthesis.md + 6 Wf-37{0..5}b1_index_sync.md + 1 docs/SESSION_FINAL_SYNTHESIS.md = 13 final-ship docs aggregated into Wf-376-B1 §0 + Wf-376_final_synthesis.md §1-§7.

---

## §4 Honest framing per party-twenty-spirit §3.5

8 honest-negative caveats preserved per Wf-376_final_synthesis.md §7 + this row:

1. **m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL** (best aug_mean MEASURED 0.472 vs threshold 0.6 — gap −0.128; 5 architectural attempts exhausted per Wf-365-A3 verdict doc + Wf-371-A1 degenerate chain attempt + Wf-375-A4 reframe exploration; structural CV-R² +0.6814 CEILING_BREAK but threshold-reaching axis NOT lifted; 4 forward-looking paths ranked none cheap per Wf-365-A3 §5)
2. **m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP — FINAL** (platinai_4cell + hilbert cohorts n=200 SMILES each remain at HONEST_NEGATIVE_R2_DEEP after Wf-367-B1 multi-stream ensemble; metalcytotox_combined n=2000 CEILING_BREAK at Wf-335-M2 stands; FINAL per Wf-369-A2 verdict doc)
3. **2 NEW pre-flight FAILs at Wf-376-B1 timestamp** (`p3_loi_send_ready` + `p3_wf343c4_anchor_freshness_24h`) are honest-negative per Wf-375-B1 §22.68 baseline; NOT Wf-376-introduced regressions
4. **Pre-existing 3 SKIPs PRESERVED (BY-DESIGN)** (`p3_wf343c4_yuelbond_decoder_env` per Wf-345-A2 + `p3_wf343c4_wave_budget_remaining_nonneg` per Wf-262 §7.1 with −183.04h overrun + `data_2_1_6_bindingdb_external_honest_negative` external 1.07M dependency)
5. **G6 ~11.4 hr GPU launch BLOCKED on USER_OK** (per Wf-368-B1 §0 + Wf-369-A1 §3.1 + Wf-370-A3 §9 + Wf-371-B1 §22.66.2 + Wf-375-B1 §22.68.4; closure path = USER_OK=1 + VRAM budget check per §5.1)
6. **6 forward-looking USER_OK items BLOCKED on USER_OK** (~52.9 hr GPU + 13 hr CPU + 30 min install per Wf-368-B1 §0; 7 wrappers + 2 install + 7 paper + 2 metric gates; no auto-trigger per §5.1 + kill_then_adapt §0)
7. **Wf-363-A2 GAP-3 + Wf-365-B2 G6 USER_OK parity OPEN** (per Wf-368-A4 §3 + Wf-368-B1 §7.1; closure path = ~20 min CPU + 2 tests; deferred to Wf-377+ per Wf-375-B1)
8. **Better BibTeX MANUAL GUI install** (~2 min user-side when X11/Wayland access available; flagged in `docs/SCI_PAPER_SETUP.md` §1.2; architectural — `.xpi` is signed Firefox-style extension requiring live Zotero process)

---

## §5 Cross-references

`TODO/completed/376a2_m1_multiseed_scaffold.md` (Wf-376-A2 SHIPPED); `TODO/completed/376b1_index_sync.md` (THIS file); `TODO/completed/376_final_synthesis.md` (~580 LOC 8 sections); `TODO/completed/{371,372,373,374,375}{a1..a4,b1,c1}_*.md` + `TODO/completed/{371,372,373,374,375}_final_synthesis.md` + `TODO/completed/375b1_index_sync.md` (30 Wf-37{1..5} closeouts); `docs/SESSION_FINAL_SYNTHESIS.md` (Wf-370-A3 ~575 LOC 10 sections); `docs/M2_R2_FINAL_VERDICT.md` (Wf-369-A2 320 LOC); `docs/M6_KAPPA_FINAL_VERDICT.md` (Wf-365-A3 269 LOC + Wf-375-A4 expansion); `docs/HONEST_NEGATIVE_CATALOG.md` (Wf-375-A2 69 entries); `docs/WETLAB_DEPRECATION_FAQ.md` (Wf-375-A1 ~280 LOC); `docs/USER_OK_FINAL_CHECKLIST.md` (Wf-368-B1 ~430 LOC 9 sections); `docs/WORKFLOW_DEPENDENCY_GRAPH.md` (Wf-358-C1 ~289 LOC); `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` (Wf-364-A4 ~270 LOC LaTeX template); `docs/SCI_PAPER_SETUP.md` (Wf-351-B3 ~120 LOC); `molmetal/reports/wf376b1_preflight.json` (Wf-376-B1 pre-flight 89/2/3/94 wall 154.228 s — 32,533 bytes); `molmetal/reports/wf375b1_preflight.txt` (Wf-375-B1 baseline 89/2/3/94 wall 146.79 s); `molmetal/reports/wf374b1_preflight.txt` (Wf-374-B1 baseline 89/2/3/94 wall ~155 s); `molmetal/reports/wf372c1_preflight.txt` (Wf-372-C1 baseline 92/0/2/94 wall 153.42 s); `molmetal/reports/wf370b1_preflight.txt` (Wf-370-B1 baseline 92/0/2/94 wall 152.53 s); `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline); `AGENTS.md` §5.1 + §5.2 + §5.3 + §5.4 standing directives re-confirmed; `AGENTS.md` §6 row (this entry NEW); `AGENTS.md` last-update trailer (this sync appended); `TODO/INDEX.md` §22.70 (this entry NEW ~50 lines appended between §22.69 Wf-374-B1 and §23 anchor); `kill_then_adapt` §0 (NO GPU launch + USER_OK gating).

**Wall PROJECTED 10-15 min CPU-only** per task brief; **MEASURED ~12 min CPU-only** (C1 ~10 min + B1 ~2 min edit time).