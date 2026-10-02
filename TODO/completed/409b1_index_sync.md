# Wf-409-B1 — docs-only INDEX + AGENTS + final synthesis v40 SHIPPED

**Date:** 2026-09-30T17:55 CST (Wf-409-B1 closeout timestamp)
**Wf:** Wf-409-B1 (ultracode docs-only sync sub-flow per user 2026-09-30 verbatim *"我感觉得基于现状做更具体的规划，和修复，这些内指标还有提升的空间，你可以并行启动多个ultracode来做改进"*)
**Status:** ✅ SHIPPED (docs-only INDEX + AGENTS sync + comprehensive final synthesis v40)
**Owner:** Hugo Lee + Wf-409-B1 ultracode agent
**Skill alignment:** `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline); `mol-metal-multiph-pll` (Wf-409-A1 PLL wire-in); `mol-metal-cfm-training` (Wf-409-A1+A2 wire-ins); `mol-metal-gpu-execution` §0 (`kill_then_adapt` directive — all 4 sub-flows are CPU-only SCAFFOLD/wire-in surface; actual GPU retrain BLOCKED on USER_OK per AGENTS.md §5.1); no `paper/*` touches (TODO-84 active); no `wetlab/*` planning (AGENTS.md §5.4 + §10); atomic write protocol per AGENTS.md §5.6 (this B1 closeout written via `molmetal/scripts/wf386a3_atomic_write.py::write_closeout`).

---

## §0 TL;DR

**Wf-409 wave ACTUAL count: 4 SHIPPED deliverables (A1 + A2 + A3 + A4) + 1 SHIPPED docs-only sync (B1 this entry)** — matches harness narrative verbatim "Wf-409-A1..A4 SHIPPED 4 deliverables: TODO-94 wire-in + TODO-155 CFM wire-in + Wf-246-C SCAFFOLD + Wf-29 extended smoke".

This B1 docs-only sync ships:
1. **`§22.95 Wf-409 ultracode CPU-only TODO-94 + TODO-155 + Wf-246-C + Wf-29 extended wire-ins`** appended to `TODO/INDEX.md` (~120 lines between §22.94 Wf-408 and §22.96 Wf-410)
2. **NEW §6 row in `AGENTS.md`** for Wf-409 (last-update trailer extended)
3. **`TODO/completed/409_final_synthesis.md`** comprehensive final synthesis v40 (~600 LOC 9 sections)

**10 NEW tests PASS** total across the wave:
- Wf-409-A1: 2 NEW tests (CFM PLL wire-in) PASS in 0.66 s wall CPU-only
- Wf-409-A2: 3 NEW tests (CFM chemprop v3 cache wire-in) PASS in <1 s wall CPU-only
- Wf-409-A3: 2 NEW tests (Wf-246-C metal-coordination SCAFFOLD smoke) PASS in ~3 s wall CPU-only
- Wf-409-A4: 3 NEW tests (Wf-29 extended smoke) PASS in 0.40 s wall CPU-only

**0 GPU launches** — all 4 sub-flows are CPU-only docs+code+test+verify per AGENTS.md §5.1 + kill_then_adapt §0.
**0 paper/* touches** — TODO-84 active.
**0 wet-lab planning** — AGENTS.md §5.4 + §10 re-confirmed.

---

## §1 On-disk state verification (per AGENTS.md §5.5 filesystem-truth rule)

**Verified via `ls -la TODO/completed/ | grep -E "409"` at Wf-409-B1 timestamp 2026-09-30T17:55 CST:**

| Sub-flow | On-disk file | Bytes | Verdict |
|---|---:|---:|---|
| Wf-409-A1 (TODO-94 R20-P3 Multi-PH + PLL CFM wire-in) | `409a1_todo_094_wire.md` | 18,295 | ✅ SHIPPED — 3 NEW CLI flags wired (`--cfm-multiph-pll` + `--cfm-multiph-pll-top-k` + `--cfm-multiph-pll-weight`); 50-step §5.7 smoke PASS (CASE A + CASE B); 2 NEW tests PASS in 0.66 s wall |
| Wf-409-A2 (TODO-155 CFM chemprop v3 cache wire-in) | `409a2_todo_155_cfm_wire.md` | 20,917 | ✅ SHIPPED — `--cfm-chemprop-v3-cache` CLI flag wired; 3 NEW tests PASS in <1 s wall |
| Wf-409-A3 (Wf-246-C metal-coordination SCAFFOLD + launch wrapper) | `409a3_wf246c_scaffold.md` | 31,257 | ✅ SHIPPED — CPU-only SCAFFOLD script 435 LOC + bash launch wrapper 236 LOC + 2 NEW tests PASS; kcal sign fix verified; 5-metal subset per Wf-390-A1 Path A |
| Wf-409-A4 (Wf-29 attr-fix EXTENDED smoke: m18 + F5 32-d + click-aug) | `409a4_wf29_extended.md` | 16,726 | ✅ SHIPPED — 3 invariants (inv4 m18 + inv5 F5 SKIP RDKit env-gap + inv6 click-aug); 3 NEW tests PASS in 0.40 s wall |
| Wf-409-B1 (this docs-only INDEX + AGENTS + final synthesis) | `409b1_index_sync.md` + `409_final_synthesis.md` | (this sync) | ✅ SHIPPED |

**On-disk truth:** 4 deliverables SHIPPED on disk (A1 + A2 + A3 + A4) + 1 docs-only sync (B1 this entry) = 5 SHIPPED files on disk. Harness narrative matches disk-truth per AGENTS.md §5.5.

---

## §2 What Wf-409-B1 sync added

### §2.1 `TODO/INDEX.md §22.95 Wf-409 ultracode CPU-only TODO-94 + TODO-155 + Wf-246-C + Wf-29 extended wire-ins`

Appended between §22.94 Wf-408 (line 2282-2338) and §22.96 Wf-410 (line 2340 onward) — ~120 lines. 7 subsections:
- **§22.95.0 — Wf-409 wave structure**: parent directive relay from user 2026-09-30 verbatim + ultracode family context
- **§22.95.1 — Wf-409 sub-flows SHIPPED**: 4 deliverables + 1 docs-only sync table (A1 + A2 + A3 + A4 + B1)
- **§22.95.2 — Wf-409 wave ACTUAL on-disk verification** (per AGENTS.md §5.5 filesystem-truth rule)
- **§22.95.3 — What 彻底解决 (per Wf-409 wave)**: 4 SHIPPED sub-flows + 10 NEW tests + 1 docs-only sync
- **§22.95.4 — What NOT 彻底解决 (forward-looking honest-negatives preserved)**: 7 honest-negative caveats
- **§22.95.5 — MEASURED test pass-rates (per Wf-409 wave)**: 10 NEW tests total table
- **§22.95.6 — User voice answer (per Wf-409-A1..A4 + Wf-409-B1 + Wf-409 final synthesis §7)**: addressing user verbatim
- **§22.95.7 — Cross-link**: 4 Wf-409 SHIPPED closeouts + 2 NEW docs + 2 NEW source files + 2 NEW test files + 4 NEW source artifacts

### §2.2 `AGENTS.md` NEW §6 row (Wf-409 sync — this entry)

Appended to `AGENTS.md` last-update trailer (after Wf-411 row, before §7 critical scripts reference) — ~600-line row documenting:
- Pre-flight cascade PRESERVED at Wf-410/Wf-411 baseline (95/18/4/117 — drift reflects pipeline expansion from Wf-410 + Wf-411 waves; NOT Wf-409-introduced)
- Wf-409 wave structure (4 SHIPPED + 1 docs-only sync)
- 10 NEW tests across 4 sub-flows
- 4 NEW closeouts on disk + 2 NEW source files + 2 NEW test files + 4 NEW source artifacts
- ZERO GPU launches + ZERO paper/* touches + ZERO wet-lab planning
- MEASURED combined wall ~25 min CPU-only across 4 deliverables + B1
- 7 honest-negative caveats preserved
- User voice answer (per Wf-409-A1..A4 + Wf-409 final synthesis §7)

### §2.3 `TODO/completed/409_final_synthesis.md` (~600 LOC 9 sections)

Comprehensive final synthesis v40:
- §0 TL;DR (one-paragraph verdict + key tables)
- §1 Wf-409 wave structure (4 deliverables + 1 docs-only sync + 1 sister wave context)
- §2 What 彻底解决 (TODO-94 wire-in + TODO-155 CFM wire-in + Wf-246-C SCAFFOLD + Wf-29 extended smoke)
- §3 What NOT 彻底解决 (forward-looking honest-negatives preserved)
- §4 Forward-looking USER_OK-gated items
- §5 Honest framing (party-twenty-spirit §3.5)
- §6 Cross-references
- §7 User voice answer
- §8 Wall budget envelope
- §9 Wave architecture notes

---

## §3 Chronology + cross-link summary

**Wf-409-B1 sync chain**: Wf-409-A1 → Wf-409-A2 → Wf-409-A3 → Wf-409-A4 → **Wf-409-B1 this entry** → (Wf-410 + Wf-411 already SHIPPED in INDEX prior syncs).

**Files created/modified by Wf-409-B1 sync**:
- `TODO/INDEX.md` — §22.95 NEW entry appended (~120 lines between §22.94 + §22.96)
- `AGENTS.md` — NEW §6 row appended (last-update trailer extended with Wf-409-B1 sync clause)
- `TODO/completed/409b1_index_sync.md` — THIS file (NEW ~150 LOC 6 sections)
- `TODO/completed/409_final_synthesis.md` — NEW ~600 LOC 9 sections

**Files preserved unchanged** (per Wf-358-D2 + Wf-359-C1 + Wf-360-C1 + ... + Wf-411-B1 prior syncs):
- All §22 entries from §22.5 Wf-389 through §22.94 Wf-408 through §22.97 Wf-411 preserved

**Cross-references in Wf-409 final synthesis**:
- `TODO/completed/{409a1_todo_094_wire,409a2_todo_155_cfm_wire,409a3_wf246c_scaffold,409a4_wf29_extended}_*.md` (4 SHIPPED closeouts)
- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` (Wf-409-A1 +120/-15 LOC + Wf-409-A2 wire-in)
- `molmetal/scripts/wf409a3_metal_coord_retrain_scaffold.py` (NEW 435 LOC) + `scripts/launch_wf409a3_metal_coord_retrain.sh` (NEW 236 LOC)
- `molmetal/scripts/wf409a4_wf29_extended_smoke.py` (NEW ~675 LOC)
- `molmetal/tests/test_wf409{1_cfm_pll_wire,2_chemprop_v3_cache_wire,3_metal_coord_scaffold,4_wf29_extended}.py` (NEW 4 test files)
- `molmetal/reports/wf409a4_wf29_extended/smoke_result.json` (NEW 3,122 bytes; aggregate_verdict=PASS)
- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py.bak.wf409a1` (backup per atomic_write_file §5.6 protocol)

---

## §4 §5 standing directives re-confirmation

Per `party-twenty-spirit` §3.5 + AGENTS.md §5.1..§5.7 + §10 standing directives all re-confirmed in this B1 sync:
- **§5.1** (GPU grant + USER_OK gating) — all 4 Wf-409 sub-flows are CPU-only SCAFFOLD/wire-in surface; actual GPU retrain BLOCKED on USER_OK=1 per `kill_then_adapt` §0
- **§5.2** (honest framing) — no fabrication of m6_kappa lift / m2 R² lift / m1 AUC lift / Wf-246-C retrain metric entries
- **§5.3** (no paper work per TODO-84) — 0 paper/* touches performed across entire Wf-409 wave
- **§5.4** (Tier-2 acceptance bar) — wetlab items DEPRECATED per §10 + 2026-09-29 user directive "我们发二区就可以了，不用规划湿实验"
- **§5.5** (post-wave on-disk verification rule) — used to verify 4 SHIPPED Wf-409 files match harness-claimed 4 deliverables; honest-negative record-keeping preserved
- **§5.6** (atomic write protocol adoption rule) — strictly observed via `atomic_write_file` for B1 closeout + final_synthesis; backup `train_cfm_yuelbond_wasserstein.py.bak.wf409a1` per safe_overwrite_check
- **§5.7** (trainer flag wire-in smoke-required rule) — observed by Wf-409-A1 50-step CPU smoke (CASE A flag OFF + CASE B flag ON + weight=0.1; both loss[0] > loss[49] monotone non-increasing + decode_ratio=1.000) + Wf-409-A2 50-step CPU smoke; cross-ref Wf-399-A1 NEW smoke protocol
- **§10** (wetlab deprecation FAQ) — 0 wetlab planning across entire Wf-409 wave

---

## §5 Self-audit (per AGENTS.md §5.5 filesystem-truth rule)

**§5.5 rule applied to this B1 sync**: the 4 SHIPPED Wf-409 closeouts (A1 + A2 + A3 + A4) are verified on disk BEFORE this B1 closeout was authored. The harness task brief claim "Wf-409-A1..A4 SHIPPED 4 deliverables" matches disk-truth exactly per §5.5:
- `TODO/completed/409a1_todo_094_wire.md` (18,295 bytes mtime 2026-09-30 15:24)
- `TODO/completed/409a2_todo_155_cfm_wire.md` (20,917 bytes mtime 2026-09-30 15:50)
- `TODO/completed/409a3_wf246c_scaffold.md` (31,257 bytes mtime 2026-09-30 16:02)
- `TODO/completed/409a4_wf29_extended.md` (16,726 bytes mtime 2026-09-30 16:16)

This B1 docs-only sync (5th deliverable) closes the wave; 0 NEW tests (verification + docs-only sync only per task brief).

