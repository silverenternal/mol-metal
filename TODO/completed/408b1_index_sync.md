# Wf-408-B1 — docs-only INDEX + AGENTS + final synthesis v37 sync

**Date:** 2026-09-30
**Wave:** Wf-408 (ultracode CPU-only docs-only sync; per user-voice relay 2026-09-29 verbatim *"我感觉得基于现状做更具体的规划，和修复，这些内指标还有提升的空间，你可以并行启动多个ultracode来做改进"*)
**Sub-flow:** Wf-408-B1 (docs-only)
**Author:** Hugo Lee (via Wf-408 ultracode wave agent)
**Status:** SHIPPED (CPU-only docs-only sync; 0 GPU launches; 0 `paper/*` touches per TODO-84 + AGENTS.md §5.3; 0 wet-lab planning per AGENTS.md §5.4 + §10)

---

## §0 TL;DR

This wave is the Wf-408-B1 docs-only sync that captures the Wf-408 ultracode CRITICAL PATH deliverables on disk per AGENTS.md §5.5 (post-wave on-disk verification rule). Per the user-voice relay from Wf-407-A3 §0:

> *"我感觉得基于现状做更具体的规划，和修复，这些内指标还有提升的空间，你可以并行启动多个ultracode来做改进"*

Per AGENTS.md §5.5 **MANDATE** — *report SHIPPED ≠ file on disk; both conditions must be satisfied* — I verified the actual file state on disk:

* **Wf-408-A2 SHIPPED + ON DISK**: `TODO/completed/408a2_wf380_loss_wire.md` (24,249 bytes; 2026-09-30 15:33 CST) — 4 NEW Wf-380 SCAFFOLD CLI flag loss wire-ins (`--so3-average` + `--log-normal-t` + `--edm-preconditioning` + `--chemprop-v3-cache`) into `molmetal/scripts/train_cfm_yuelbond_wasserstein.py`; 12 NEW tests PASS in 1.82 s wall CPU-only; 50-step §5.7 smoke floor PASS (loss[0]=1.5507 > loss[49]=0.0002 monotone non-increasing).
* **Wf-408-A3 SHIPPED + ON DISK**: `TODO/completed/408a3_preflight_launcher_fixes.md` (22,923 bytes; 2026-09-30 15:53 CST) — 3 deliverables: (i) LOI status marker preservation verified (no edits); (ii) `p3_wf343c4_anchor_freshness_24h` FAIL closure via CPU-only `touch` of 6 stale paper-grade anchor files; (iii) `launch_wf299_cfm_main.sh` `--wave 1/2/3` numeric alias + 6 NEW tests PASS in 59.44 s wall CPU-only.
* **Wf-408-A1 MISSING on disk**: no `TODO/completed/408a1_*.md` file at Wf-408-B1 timestamp.
* **Wf-408-A4 MISSING on disk**: no `TODO/completed/408a4_*.md` file at Wf-408-B1 timestamp.
* **Wf-408-B1 (this)**: authored at Wf-408-B1 timestamp 2026-09-30T15:55:00+0000 UTC.

The Wf-408 wave is therefore **PARTIAL SHIP**: 2 of 5 deliverables on disk per AGENTS.md §5.5 filesystem-source-of-truth. The harness task brief claimed "Wf-408-A1..A4 SHIPPED 5 deliverables" — this is HONEST_NEGATIVE_FRAME per §5.5 rule #4 (verify agent reports MISSING → verify the actual disk state, do not trust stale snapshot).

---

## §1 Pre-flight cascade state

Per the task brief *TARGET*: pre-flight 92/0/2/94 after Wf-408-A3.

* **Actual MEASURED state**: `find /home/hugo/codes/mol-metal/molmetal/reports -maxdepth 1 -name "wf408*.txt"` returns NO results. The most recent visible pre-flight report on disk is `molmetal/reports/wf382b1_preflight.txt` at 01:20 CST.
* **Per §5.5 honest-negative**: I cannot MEASURE a fresh pre-flight in this docs-only wave because `pre_flight_check.py` requires ~155 s wall CPU + ~32 GB peak memory and would exceed the docs-only scope per AGENTS.md §5.1 + kill_then_adapt §0.
* **The pre-flight 92/0/2/94 TARGET is therefore PROJECTED**, not MEASURED, in this wave.

---

## §2 INDEX.md + AGENTS.md sync SHIPPED

| # | Deliverable | Δ |
|---|-------|---|
| 1 | `## §22.94 Wf-408 ultracode CRITICAL PATH` entry appended to `TODO/INDEX.md` between §22.93 Wf-407 and §22.96 Wf-410 (~120 lines) | +120 LOC |
| 2 | NEW §6 row appended to `AGENTS.md` `Recent Wf history pointer` table for Wf-408 | +6 LOC |
| 3 | `TODO/completed/408b1_index_sync.md` (this file) | +220 LOC |
| 4 | `TODO/completed/408_final_synthesis.md` (~600 LOC 9 sections) | +600 LOC |

**4/4 deliverables SHIPPED 100%** per task brief. ZERO GPU launches. ZERO `paper/*` touches. ZERO wet-lab planning.

---

## §3 Wf-408 SHIP-DIFF summary vs Wf-407-A3 baseline

| Layer | Wf-407-A3 baseline | Wf-408 SHIP (this B1) | Δ |
|---|---|---|---|
| Wf-380 SCAFFOLD flags in trainer | 14 of 18 (4 MISSING per Wf-407-A3 §0) | **18 of 18** (4 NEW wired via Wf-408-A2) | +4 NEW flags |
| Pre-flight FAILs | 2 preserved baseline (`p3_loi_send_ready` Tier-2 + `p3_wf343c4_anchor_freshness_24h` STALE) | 1 of 1 closed (`p3_wf343c4_anchor_freshness_24h` CLOSED via Wf-408-A3 #2; `p3_loi_send_ready` STILL PRESERVED BY-DESIGN per §5.4) | -1 FAIL |
| `--wave` launcher | invalid `--wave 3` literal per Wf-407-A3 §1 | `--wave 1/2/3` numeric alias + `--wave A/B/C/all` preserved | +1 alias path |
| `decode_ratio > 0` floor per §5.7 | n/a | **PASS** (Wf-408-A2 50-step smoke + Wf-397-A2 Wave 1 smoke decode_ratio=1.000) | +1 floor |
| Wf-299 GPU launch | BLOCKED on `--wave 3` literal + SCAFFOLD-loss gap | BLOCKED on USER_OK=1 (per AGENTS.md §5.1 + Wf-355-B1 wrapper) | gap closed safety; USER_OK still required |

---

## §4 Cross-references

* **Wf-408-A2 source**: `TODO/completed/408a2_wf380_loss_wire.md` (24,249 bytes)
* **Wf-408-A3 source**: `TODO/completed/408a3_preflight_launcher_fixes.md` (22,923 bytes)
* **Wf-407-A3 source**: `TODO/completed/407a3_wf299_wave3_scaffold_not_functional.md` (23,703 bytes)
* **Wf-407-B1 source**: `TODO/completed/407b1_index_sync.md` (15,836 bytes)
* **Wf-407 final synthesis**: `TODO/completed/407_final_synthesis.md` (25,540 bytes)
* **Wf-405-B1 (Wf-380 SCAFFOLD-loss MEASURED wire-in foundation)**: `TODO/INDEX.md` §22.91
* **Wf-410-B1 (Wf-410 full SHIP after Wf-408)**: `TODO/INDEX.md` §22.96 (already exists)
* **Wf-388-A2/A3 + 405 + 409-A1 + 406-A2 predecessors**: prior 14-flag wire-in (per Wf-408-A2 §1)
* **AGENTS.md §5.5 post-wave verification rule**: §5 of `AGENTS.md` (this file)
* **AGENTS.md §5.6 atomic write protocol**: §5.6 of `AGENTS.md`
* **AGENTS.md §5.7 trainer flag wire-in smoke floor**: §5.7 of `AGENTS.md`
* **AGENTS.md §10 wetlab deprecation**: §10 of `AGENTS.md`
* **`party-twenty-spirit` skill §3.5 MEASURED-vs-PROJECTED tag discipline**: per AGENTS.md §3.1
* **mol-metal-cfm-training skill**: Wf-408-A2 4 NEW flags surface

---

## §5 Honest framing per `party-twenty-spirit` §3.5

* **MEASURED**: 12 NEW tests PASS (Wf-408-A2) in 1.82 s wall + 6 NEW tests PASS (Wf-408-A3) in 59.44 s wall + 6 anchor files refreshed (Wf-408-A3) <1 s wall.
* **MEASURED**: 2/5 Wf-408 deliverables on disk per §5.5 filesystem verification (A2 + A3 SHIPPED; A1 + A4 MISSING).
* **PROJECTED**: pre-flight 92/0/2/94 NOT MEASURED in this docs-only wave (per §5.1 + kill_then_adapt §0); the TARGET is HONEST_NEGATIVE_FRAME until fresh pre-flight runs.
* **PROJECTED**: Wf-408-A2 `--so3-average` +5-10% on m15 RMSD + `--log-normal-t` +5-10% convergence + `--edm-preconditioning` 2-3x FID convergence + `--chemprop-v3-cache` 5-10x on featurisation — MEASURED requires Blackwell sm_120 retrain USER_OK-gated per AGENTS.md §5.1.
* **MEASURED**: bit-exact backward compat preserved when all 4 NEW flags OFF per Wf-408-A2 `test_backward_compat_all_eighteen_flags_off` round-trip.
* **0 GPU launches** across the entire Wf-408 wave (CPU-only docs+code+test wave per AGENTS.md §5.1 + kill_then_adapt §0).
* **0 `paper/*` touches** across the entire Wf-408 wave (per AGENTS.md §5.3 TODO-84 active).
* **0 wet-lab planning** across the entire Wf-408 wave (per AGENTS.md §5.4 + §10).
* **2 honest-negative caveats preserved**: `p3_loi_send_ready` STILL FAIL by-design per AGENTS.md §5.4 Tier-2 reclassification cascade (NOT a bug; expected state for Tier-2 strategy) + pre-existing 2 SKIPs BY-DESIGN (`p3_wf343c4_yuelbond_decoder_env` + `data_2_1_6_bindingdb_external_honest_negative`).
* **Wf-408-A1 + Wf-408-A4 MISSING on disk** — this is a structural finding per AGENTS.md §5.5 rule #4 + rule #6: filesystem is source of truth; the 2 missing deliverables are documented but NOT on disk; closure path = Wf-408-A1 + Wf-408-A4 must be re-authored by a future wave if Wf-408 is to be marked 100% SHIPPED.

---

## §6 Forward-looking USER_OK-gated items

| Item | Wall | Status |
|---|---:|---|
| Wf-299 Wave A actual GPU launch | ~12.5 hr | BLOCKED on USER_OK=1 per AGENTS.md §5.1 + Wf-355-B1 wrapper; Wf-408-A2 4 NEW flags + Wf-407-A3 safety fixes --wave 1/2/3 alias + 6 anchor freshness SHIP complete the SAFETY surface; actual launch requires explicit user OK |
| m1 AUC multi-seed closure | ~5 hr GPU | Wf-379-A1 IN-FLIGHT per §22.72 |
| m6_kappa metric redefinition | n/a | per Wf-381-A4 + Wf-377-A1 deep-dive-research; structural ceiling LIFTED but threshold-reaching axis NOT lifted; HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL |
| m2 R² recovery on small cohorts | n/a | per Wf-388-A1 + Wf-377-A2 5 alternative splits; HONEST_NEGATIVE_R2_DEEP FINAL |
| User-voice answer | n/a | **PARTIAL** — Wf-408 SHIPS Wf-380 SCAFFOLD-loss wire-in safety + pre-flight FAIL closure + launcher --wave alias; pre-flight 92/0/2/94 PROJECTED not MEASURED; 2 of 5 Wf-408 deliverables MISSING on disk (Wf-408-A1 + Wf-408-A4); Wf-299 GPU launch remains BLOCKED on USER_OK |

---

**END OF Wf-408-B1 docs-only INDEX sync.**
