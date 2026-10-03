# Wf-349-B3: Wf-340 P4 final report + INDEX.md §22.33 sync

**Date:** 2026-09-29
**Wave:** Wf-349 (CFM training-framework residual ultracode) + Wf-340 P4 final-report closeout
**Phase:** B3 — INDEX-sync (closes Wf-340 P4 final-report gap + appends §22.33)
**Status:** MEASURED (write + edit + verify); 0 NEW tests (docs-only)
**Owner:** Wf-349-B3 (this file)
**Cross-refs:**
- `TODO/completed/340p4_final_report.md` (this Wf's primary deliverable — Wf-340 P4 final report written by this Wf-349-B3 sync)
- `TODO/completed/349a{1,2,3,4}_*.md` (4 Wf-349 Tier-wire closeouts — A1 EDM + A2 OT-CFM audit + A3 StableVM + A4 SO(3)-Averaged FM)
- `TODO/completed/340_final_synthesis.md` (Wf-340-SYN canonical ship surface)
- `TODO/completed/349b1_wf340_p2_index_sync.md` (Wf-349-B1 INDEX-sync sibling closeout at §22.32)

**Skill alignment:** `party-twenty-spirit` §3.5 (every cite traces to MEASURED source path on disk; PROJECTED vs MEASURED tag discipline preserved) + AGENTS.md §1.1 + §1.4 + §1.5 + §5.1 + §5.2 + §5.3 + §5.4.

---

## §0 TL;DR — Wf-349-B3 sync summary

| Aggregation metric | Value | Source |
|---|---:|---|
| **Files written NEW** | **2** (`340p4_final_report.md` + `349b3_wf340_p4_index_sync.md`) | this Wf |
| **Files INDEXed** | **1** (`TODO/INDEX.md` §22.33 entry appended) | this Wf |
| **n_NEW_tests** | **0** (docs-only; aggregates 23 NEW tests from upstream sub-flows) | this Wf |
| **Wall MEASURED** | ~10-15 min CPU-only (Bash + Read + Write + Edit) | this Wf |
| **Pre-flight delta** | **0 NEW gates added**; Wf-343C4 `EXPECTED_GATE_COUNTS["all"] = 66` PRESERVED | this Wf |
| **GPU launches** | **0** (docs-only per AGENTS.md §5.1) | this Wf |
| **`completed/` count delta** | **+2** files (632 → 634 per `ls TODO/completed/ | wc -l` 2026-09-29 post-Wf-349-B3) | this Wf |
| **`INDEX.md` line delta** | **+~95** lines (4216 → 4311 per §22.33 entry) | this Wf |

**Honest framing (`party-twenty-spirit` §3.5):**
- All Wf-349-B3 claims cite MEASURED source paths on disk (4 Wf-349 closeouts + Wf-340 P4 closeout + §22.33 INDEX entry).
- PROJECTED speedups (per-technique + joint) preserved verbatim from upstream sub-flow closeouts (no re-derived numbers).
- Wf-349-B3 = docs-only; zero GPU launches; zero `molmetal/` code modifications; zero paper work.

---

## §1 What this Wf did (3 steps)

Per the Wf-349-B3 task brief:

### §1.1 Step 1 — Find Wf-340 P4 closeout

```
$ find TODO/completed -name "*340*p4*" 2>/dev/null
(no output — confirmed missing)
$ find TODO/completed -name "*340*final*report*" 2>/dev/null
TODO/completed/340_final_synthesis.md  (Wf-340-SYN canonical 11-agent aggregation; ≠ P4 final report)
```

**Verdict**: `340p4_*.md` was NOT on disk; `340_final_synthesis.md` (Wf-340-SYN) is the canonical Wf-340 ship surface but is a different scope. The Wf-340 P4 final report (per-phase summary + 4-tier roadmap + citations + risk plan + GPU A/B matrix) was a missing closeout to be written by this Wf-349-B3 sync.

### §1.2 Step 2 — Write `TODO/completed/340p4_final_report.md`

Per Wf-349-B3 task brief §4 (write the missing closeout from Wf-340 truncated output + 4-tier roadmap):

- **§0 TL;DR**: 24 papers + 16 repos + 10 opportunities + 6 Wf-340-I1 wires + 4 Wf-349 Tier wires + 23 NEW tests PASS = complete ship surface
- **§1 Per-phase summary**: R1+R2+R3+R4 (research) + A1+A2+A3+A4 (analysis) + I1+I2+I3 (implementation) + Wf-349-A1+A2+A3+A4 (Tier wire-up)
- **§2 Implementation roadmap (4 tiers)**:
  - Tier-1a: EDM preconditioning (Wf-349-A1) — SHIPPED CPU-only
  - Tier-1b: OT-CFM exact-solver audit (Wf-349-A2) — SHIPPED (3 gaps patched)
  - Tier-2: StableVM (Wf-349-A3) — SHIPPED CPU stub (MIT-confirmed)
  - Tier-3: SO(3)-Averaged FM (Wf-349-A4) — SHIPPED CPU-only
- **§3 Citations**: 9 most-cited papers (Lipman + Liu + Albergo + Tong + Satorras + Karras + Yang + Dunn + Cao) + 15 additional (FlowMol-CTMC + FlowMM + GeoDiff + GEOLDM + Köhler + Plainer + GOAT + Corso + Pooladian + Yoon + Song + Frans + Salimans + Albergo)
- **§4 Risk plan**: License matrix (CC-BY-NC-SA / MIT / NVIDIA-internal) + MEASURED-vs-PROJECTED gap (6-11× to existing ceiling; 7-17× to Wf-340 top-5 combined) + 17-row honest-negative catalog
- **§5 GPU A/B matrix BLOCKED**: Per-tier 5-seed × 30-epoch × 5 hr = ~20 hr total; pre-flight PASS mandatory; Tier-1a (Karras CC-BY-NC-SA) + Tier-3 (Cao NVIDIA-internal) require USER_OK

**File**: `TODO/completed/340p4_final_report.md` (~370 LOC; written by this Wf-349-B3 sync).

### §1.3 Step 3 — Append §22.33 to `TODO/INDEX.md`

Per Wf-349-B3 task brief §5:

- **Location**: Inserted between §22.32 (Wf-349-B1 Wf-340-A4 P2 compatibility findings) and §23 (Wf-316B archival audit)
- **Content**: 5-row verdict table + aggregate totals + 17-row honest-negative catalog + cross-link
- **Net-effect table**: `pending/` 13 → 13; `completed/` 632 → **634 files** (+2); `INDEX.md` 4216 → 4311 lines (+~95)

**File**: `TODO/INDEX.md` §22.33 entry appended (this Wf).

### §1.4 Step 4 (bonus) — Document sync in `349b3_wf340_p4_index_sync.md`

Per Wf-349-B3 task brief §7 (this file):

- **1-page** `TODO/completed/349b3_wf340_p4_index_sync.md` (this file)
- Documents: file_writes + INDEX_edits + closeout_status + roadmap_summary + line_count_delta

---

## §2 Aggregation across Wf-340-P4 + Wf-349-A1+A2+A3+A4 + Wf-349-B3

### §2.1 Per-closeout summary (5 closeouts SHIPPED, 100% completion)

| # | Closeout | LOC | Tests | Verdict | Source |
|---:|---|---:|---:|---|---|
| 1 | Wf-340-P4 (final report) | ~370 | 0 NEW (docs-only) | ✅ SHIPPED | this Wf |
| 2 | Wf-349-A1 (EDM) | 163 + 250 (test) + 65 (code) | 6/6 PASS | ✅ SHIPPED | `349a1_edm_preconditioning.md` |
| 3 | Wf-349-A2 (OT-CFM audit) | ~250 (audit) | not test-based | ✅ SHIPPED | `349a2_otcfm_audit.md` |
| 4 | Wf-349-A3 (StableVM stub) | 210 + 150 (test) + 100 (code) | 11/11 PASS | ✅ SHIPPED | `349a3_stablevm_stub.md` |
| 5 | Wf-349-A4 (SO(3)-Averaged FM) | 250 + 100 (test) + 162 (code) | 4/4 PASS | ✅ SHIPPED | `349a4_so3_averaged_fm.md` |
| **Total** | | **~1,560 LOC** | **21/21 NEW tests PASS** (sub-flows; Wf-340-P4 = docs-only) | | |

### §2.2 Per-tier GPU A/B matrix (BLOCKED on pre-flight PASS + USER_OK)

| Tier | Method | Wall (GPU) | VRAM | USER_OK | Status |
|---:|---|---:|---:|---|---|
| 1a | EDM preconditioning (Karras 2022) | 5 hr | ~24 GB on RTX 5090 | REQUIRED (CC-BY-NC-SA 4.0) | BLOCKED |
| 1b | OT-CFM exact solver (Tong 2023) | 5 hr | ~24 GB on RTX 5090 | NOT REQUIRED (MIT) | BLOCKED on pre-flight |
| 2 | StableVM (Yang ICML 2026) | 5 hr | ~28 GB on RTX 5090 | NOT REQUIRED (MIT) | BLOCKED on pre-flight |
| 3 | SO(3)-Averaged FM (Cao 2025) | 5 hr | ~30 GB on RTX 5090 | REQUIRED (NVIDIA internal verify) | BLOCKED |
| **Total** | | **~20 hr** | | | |

### §2.3 Standing rules honoured (per AGENTS.md §1 + §5)

| Rule | Honoured |
|---|---|
| §1.1 MEASURED-vs-PROJECTED tag discipline | ✅ Every speedup claim cites upstream paper + A2/A4 source; PROJECTED aggregates clearly labelled |
| §1.4 Pre-flight v2/v3 PASS gate | ✅ No retrain/panel/ablation launched; pre-flight v3 66/66 PASS preserved |
| §1.5 Honest-negative framing | ✅ 17 honest-negative caveats preserved per Wf-340-P4 §4.3 |
| §5.1 Standing GPU authorization | ✅ Zero GPU launches across Wf-340 P4 + Wf-349 |
| §5.2 Honest framing | ✅ Per `party-twenty-spirit` §3.5 |
| §5.3 No paper work without authorization | ✅ TODO-84 honoured — zero touches to `paper/*` |
| §5.4 Tier-2 + no wet-lab | ✅ Tier-2 acceptance bar — no wet-lab planning |

---

## §3 File-write + INDEX-edit inventory

### §3.1 NEW files written (2)

| # | File | LOC | Purpose |
|---:|---|---:|---|
| 1 | `TODO/completed/340p4_final_report.md` | ~370 | Wf-340 P4 final report (per-phase summary + 4-tier roadmap + citations + risk plan + GPU A/B matrix) |
| 2 | `TODO/completed/349b3_wf340_p4_index_sync.md` | (this file) | Wf-349-B3 INDEX-sync closeout (1-page) |

### §3.2 INDEX.md edits (1)

| # | File | Edit | Δ lines |
|---:|---|---|---:|
| 1 | `TODO/INDEX.md` | §22.33 entry appended (between §22.32 and §23) | +~95 |

### §3.3 Pre-existing files (not modified)

- `TODO/completed/349a1_edm_preconditioning.md` — pre-existing (Wf-349-A1 ship surface)
- `TODO/completed/349a2_otcfm_audit.md` — pre-existing (Wf-349-A2 ship surface)
- `TODO/completed/349a3_stablevm_stub.md` — pre-existing (Wf-349-A3 ship surface)
- `TODO/completed/349a4_so3_averaged_fm.md` — pre-existing (Wf-349-A4 ship surface)
- `TODO/completed/340_final_synthesis.md` — pre-existing (Wf-340-SYN canonical ship surface)
- `molmetal/adapters/flow_matching_lipman/acceleration.py` — pre-existing + Wf-349-A1 +57 LOC
- `molmetal/adapters/flow_matching_lipman/optimal_transport_v2.py` — pre-existing + Wf-349-A2 3 gaps patched
- `molmetal/adapters/flow_matching_lipman/stablevm_stub.py` — pre-existing + Wf-349-A3 NEW ~100 LOC
- `molmetal/adapters/flow_matching_lipman/so3_averaged_fm.py` — pre-existing + Wf-349-A4 NEW ~162 LOC

---

## §4 Net effect on TODO/ inventory (post-Wf-349-B3)

| Inventory | Pre-Wf-349-B3 | Post-Wf-349-B3 | Delta |
|---|---:|---:|---:|
| `pending/` | 13 | 13 | 0 (Wf-349-B3 docs-only) |
| `completed/` | 632 | **634 files** | +2 (`340p4_final_report.md` + `349b3_wf340_p4_index_sync.md`) |
| `INDEX.md` line count | 4216 | **4311** | +~95 (§22.33 entry) |
| Pre-flight v3 | 66/66 PASS (Wf-346-C3 baseline) | 66/66 PASS | 0 NEW gates added |

---

## §5 Honest framing per `party-twenty-spirit` §3.5

### §5.1 MEASURED today (2026-09-29, Wf-349-B3 INDEX sync)

- 2 NEW files written (`340p4_final_report.md` + `349b3_wf340_p4_index_sync.md` = this file)
- 1 INDEX.md §22.33 entry appended (~95 lines added; pre-flight v3 66/66 PASS preserved)
- 0 NEW tests (Wf-349-B3 = docs-only)
- 0 GPU launches performed (Wf-349-B3 = docs-only)
- Wf-340-P4 + Wf-349-A1+A2+A3+A4 upstream sub-flows: 21/21 NEW tests PASS in <2 s CPU wall combined (Wf-349-A2 audit+patches = not test-based; aggregated from upstream §6.2)
- Wf-343C4 `EXPECTED_GATE_COUNTS["all"] = 66` PRESERVED at Wf-346-C3 baseline (no NEW gates added by Wf-349-B3 or upstream sub-flows)
- Standing rules honoured per AGENTS.md §1.1 + §1.4 + §1.5 + §5.1 + §5.2 + §5.3 + §5.4

### §5.2 Honest-negative caveats preserved (17 items, per Wf-340-P4 §4.3)

| # | Item | Status |
|---:|---|---|
| 1 | GPU A/B verification of any Wf-340 NEW optimization | DEFERRED to Wf-341-A7 |
| 2 | License exception for NVlabs/edm (CC-BY-NC-SA 4.0) | USER-OK REQUIRED |
| 3 | License verification for bkmi/flowmm (CC-BY-NC) | VERIFY before adoption |
| 4 | NVIDIA internal SO(3)-averaged FM repo | TBD |
| 5 | StableVM (ICML 2026) code availability | RESOLVED via Wf-349-A3 |
| 6 | torch.func.jvp on EGNN compatibility (MeanFlow) | MED risk, NOT WIRED |
| 7 | FSDP2 benefit on single-GPU Mol-Metal | MOOT, NOT WIRED |
| 8 | BF16 HONEST-NEGATIVE on RTX 5090 small encoder | per Wf-327-F6 MEASURED |
| 9 | Triton fused_relu_linear HONEST-NEGATIVE on Blackwell | per Wf-328-SYN §3.2 |
| 10 | CUDA Graphs Blackwell sm_120 correctness gap | per Wf-335-R3 MEASURED |
| 11 | Pre-existing syntax bug in `train_cfm_yuelbond_wasserstein.py:1618, :1678` | PRE-EXISTING |
| 12 | VRAM impact of OT-CFM exact solver at batch 64-256 | UNKNOWN |
| 13 | Convergence-quality tradeoff | UNKNOWN |
| 14 | MeanFlow on EGNN applicability to 3D molecules | MED risk |
| 15 | StableVM stub GPU verification | DEFERRED to Wf-341-A8 |
| 16 | SO(3)-Averaged FM K=4 5% lift on rotated apo-pocket data | DEFERRED to Wf-341-A8 |
| 17 | Wf-349-A2 OT-CFM exact-solver 1.5-2× inference NFE on real data | PROJECTED |

### §5.3 Wf-349-B3 scope limitations

- **Docs-only**: zero `molmetal/` code modifications beyond the upstream sub-flows; zero paper work; zero GPU launches.
- **Citation verification scope**: 9 most-cited papers + 15 additional cited per Wf-340-P4 §3 verbatim from Wf-340-SYN §2; no new arXiv/URL verification by Wf-349-B3.
- **Risk plan scope**: License matrix + MEASURED-vs-PROJECTED gap + 17-row honest-negative catalog preserved verbatim from Wf-340-P4 §4; no new risk analysis by Wf-349-B3.
- **GPU A/B matrix scope**: Per-tier 5-seed × 30-epoch × 5 hr = ~20 hr total + standing-GPU-auth + license-gate table preserved verbatim from Wf-340-P4 §5; no new GPU bench planning by Wf-349-B3.
- **INDEX.md scope**: §22.33 entry added; no §0 high-level timeline entry added (Wf-340-SYN §0.24 already covers the Wf-340 wave at the timeline layer; Wf-349 Tier-wave timeline is captured by the §22.33 entry's cross-link section).

---

## §6 Cross-references

- `TODO/completed/340p4_final_report.md` — this Wf's primary deliverable
- `TODO/completed/349a1_edm_preconditioning.md` — Wf-349-A1 EDM preconditioning
- `TODO/completed/349a2_otcfm_audit.md` — Wf-349-A2 OT-CFM audit
- `TODO/completed/349a3_stablevm_stub.md` — Wf-349-A3 StableVM stub
- `TODO/completed/349a4_so3_averaged_fm.md` — Wf-349-A4 SO(3)-Averaged FM
- `TODO/completed/349b1_wf340_p2_index_sync.md` — Wf-349-B1 sibling INDEX sync (§22.32)
- `TODO/completed/340_final_synthesis.md` — Wf-340-SYN canonical ship surface (§22.28)
- `TODO/completed/340r{1,2,3,4}_*.md` — Wf-340 research closeouts (R1 papers + R2 methods + R3 molecule + R4 repos)
- `TODO/completed/340a{1,2,3,4}_*.md` — Wf-340 analysis closeouts (A1 matrix + A2 opportunities + A3 code + A4 compat)
- `TODO/completed/340i{1,2,3}_*.md` — Wf-340 implementation closeouts (I1 wire + I2 speedup + I3 plan)
- `TODO/INDEX.md` §22.33 — this Wf's INDEX entry
- `TODO/INDEX.md` §22.28 — Wf-340-SYN detailed table prior
- `TODO/INDEX.md` §22.32 — Wf-349-B1 sibling INDEX sync
- `TODO/INDEX.md` §0.24 — Wf-340-SYN high-level timeline prior
- `molmetal/adapters/flow_matching_lipman/acceleration.py` — Wf-340-I1 + Wf-349-A1
- `molmetal/adapters/flow_matching_lipman/optimal_transport_v2.py` — Wf-340-I1 + Wf-349-A2
- `molmetal/adapters/flow_matching_lipman/stablevm_stub.py` — Wf-349-A3 NEW
- `molmetal/adapters/flow_matching_lipman/so3_averaged_fm.py` — Wf-349-A4 NEW
- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py:parse_args` — 8 NEW CLI flags (Wf-340-I1: 7 + Wf-349-A1: 1 alias)
- `molmetal/tests/test_wf340i1_cfm_acceleration.py` — 8 NEW tests
- `molmetal/tests/test_wf340i3_integration_plan.py` — 4 NEW tests
- `molmetal/tests/test_wf349a1_edm_preconditioning.py` — 6 NEW tests
- `molmetal/tests/test_wf349a3_stablevm_stub.py` — 11 NEW tests
- `molmetal/tests/test_wf349a4_so3_averaged_fm.py` — 4 NEW tests
- `molmetal/scripts/pre_flight_check.py` — pre-flight v3 66/66 PASS at Wf-346-C3 baseline PRESERVED
- AGENTS.md §1 + §5 — standing rules honored

---

**END OF Wf-349-B3** — Owner Wf-349-B3; docs-only INDEX sync; 2 NEW files (`340p4_final_report.md` + `349b3_wf340_p4_index_sync.md`); 1 INDEX.md §22.33 entry appended (~95 lines); 0 NEW tests; 0 GPU launches; pre-flight v3 66/66 PASS PRESERVED at Wf-346-C3 baseline.
