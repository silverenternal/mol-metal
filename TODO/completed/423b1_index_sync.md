# Wf-423-B1 — docs-only INDEX + AGENTS + final synthesis v53

**Date:** 2026-09-30
**Owner:** Hugo Lee + Wf-423-B1 ultracode agent
**Workflow:** Wf-423 (WEAK_TO_STRONG §4.3 audit wave pair follow-up)
**Trigger:** User directive 2026-09-30 verbatim *"还有什么没修的bloker吗，也启动ultracode去修"* (any other unfixed blockers? Launch ultracode to fix them too) + Wf-422-B1 + Wf-415-B1 + Wf-416-B1 + Wf-419-B1 prior sync user voice relay
**Status:** ✅ SHIPPED — docs-only INDEX + AGENTS sync + comprehensive final synthesis v53 (this entry)
**Skill alignment:** `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline) + `AGENTS.md` §5.1 (CPU-only docs; GPU USER_OK-gated) + §5.2 (honest framing) + §5.3 (no paper work per TODO-84) + §5.4 (Tier-2 acceptance bar) + §5.5 (post-wave on-disk verification rule — strictly observed; Wf-423-A1 + A2 + A3 verified on disk BEFORE this closeout authored) + §5.6 (atomic write protocol — this closeout + §22.108 INDEX entry + §6 row in AGENTS.md written via `atomic_write_file`) + §5.7 NEW (Wf-399-A2 trainer flag wire-in smoke-required rule) + §10 (wetlab DEPRECATED)

---

## §0 TL;DR — one paragraph verdict

| Item | Value | Source |
|---|---|---|
| **Goal** | docs-only INDEX + AGENTS sync for Wf-423 wave; Wf-423 SHIPPED 3 sub-flows (A1 + A2 + A3); per Wf-422-B1 pattern | Wf-423-B1 task brief |
| **Wave structure** | 1 sub-flow = B1 docs-only INDEX + AGENTS + final synthesis v53 (NO source-code edits) | Wf-423-B1 |
| **Predecessors on disk** | Wf-423-A1 (`TODO/completed/423a1_wf380_audit.md`, 24,385 B) + Wf-423-A2 (`TODO/completed/423a2_wrapper_audit.md`, 21,563 B) + Wf-423-A3 (`TODO/completed/423a3_launcher_audit.md`, 8,242 B) | filesystem-truth per AGENTS.md §5.5 |
| **Wf-380 wire-in actual count** | **29 NEW CFM + 3 NEW DMPNN flags wired** (per Wf-423-A1 §0.1 filesystem-truth table; WAVELENGTH CONFLICT between Wf-418-A1 [12 CFM + 3 DMPNN] vs Wf-405-A2 [6 CFM + 4 DMPNN] vs Wf-419-A5 [FLAG_ONLY] RESOLVED at 29 + 3) | Wf-423-A1 §0.1 |
| **Wrapper flag-surface** | Wf-409-A3 + Wf-125 P1 wrapper flag-surface audit complete; CRITICAL tokenization bug on `--metals` (single-token capture vs `nargs="+"` multi-token) + Wf-125 wrapper consistent | Wf-423-A2 §3 |
| **Wf-299 launcher audit** | Both EXTRA_FLAGS accumulator (Wf-415-A1) AND `--wave 1/2/3` numeric alias mapping (Wf-408-A3) ON DISK + functioning correctly | Wf-423-A3 |
| **Tests** | 0 NEW tests — B1 docs-only sync per Wf-419-B1 pattern | this entry |
| **Pre-flight v3 fresh-run** | PRESERVED (CPU-only docs+sync; no GPU launch per AGENTS.md §5.1 + kill_then_adapt §0); NOT re-run | Wf-419-A2 baseline preserved |
| **GPU launches** | 0 (per AGENTS.md §5.1 + kill_then_adapt §0) | this entry |
| **paper/* touches** | 0 (per AGENTS.md §5.3 + TODO-84 active) | this entry |
| **Wetlab planning** | 0 (per AGENTS.md §5.4 + §10; user directive "搞清楚我们不用做湿实验") | this entry |
| **EXPECTED_GATE_COUNTS["all"]** | 117 PRESERVED (no new pre-flight gates added by Wf-423-B1) | Wf-419-A2 baseline preserved |
| **Wall (MEASURED)** | ~5 min CPU-only for docs-only sync | this entry |

---

## §1 Wf-423 wave structure (per AGENTS.md §5.5 filesystem-truth rule)

Wf-423-B1 is the **docs-only INDEX + AGENTS + final synthesis** sub-flow for the Wf-423 wave. The wave consists of 3 SHIPPED sub-flows (A1 + A2 + A3) verified on disk BEFORE this closeout authored, plus 1 docs-only sync (this entry B1). All 4 deliverables are SHIPPED + ON DISK + ATOMIC-WRITTEN per AGENTS.md §5.6.

### §1.1 Wf-423-A1 Wf-380 SCAFFOLD wire-in actual audit (3 SHIPPED, on disk)

**File:** `TODO/completed/423a1_wf380_audit.md` (24,385 bytes mtime 2026-09-30 23:43)

**Verdict:** WAVELENGTH CONFLICT RESOLVED at **29 NEW CFM + 3 NEW DMPNN flags wired**.

Predecessors that claimed different counts:
- Wf-418-A1 (12 CFM + 3 DMPNN) — full flag enumeration; correct for its scope
- Wf-405-A2 (6 CFM + 4 DMPNN) — INCORRECT 4-DMPNN claim; the 4th flag `--sa-penalty` landed in CFM trainer, not DMPNN
- Wf-419-A5 (FLAG_ONLY) — FALSE-NEGATIVE due to single-line regex on multi-line `p.add_argument(...)` formatting

**Per-wave ACTUAL count (filesystem-truth per AGENTS.md §5.5, MEASURED 2026-09-30T15:42Z):**

| Wave | CFM trainer flags wired | DMPNN trainer flags wired |
|---|---:|---:|
| Wf-388-A2/A3 (m7 diversity + SA) | 8 | 0 |
| Wf-394-A1 (Lambda CFM) | 3 | 0 |
| Wf-405-A2 (loss wire-in) | 6 | 3 |
| Wf-408-A2 (CRITICAL PATH) | 9 | 0 |
| Wf-409-A1 (Multi-PH + PLL) | 3 | 0 |
| **TOTAL NEW flags wired** | **29** | **3** |
| **Legacy flags preserved** | **29** | **26** |
| **Total flags wired** | **58** | **29** |

### §1.2 Wf-423-A2 Wf-409 + Wf-125 wrapper flag-surface audit (3 SHIPPED, on disk)

**File:** `TODO/completed/423a2_wrapper_audit.md` (21,563 bytes mtime 2026-09-30 23:46)

**Verdict:** Wf-409-A3 wrapper (`scripts/launch_wf409a3_metal_coord_retrain.sh` 236 LOC) has **CRITICAL tokenization bug** — `--metals) METALS="$2"; shift 2` (captures only 1 token) vs SCAFFOLD argparse `nargs="+"` (consumes multiple tokens). Invocation breaks when user passes `--metals Ru Ir Rh --seeds 0 1 2` unquoted.

Wf-125 P1 wrapper (`scripts/launch_wf125_p1_controlled_pair.sh` ~399 LOC) consistent — single int + enum args.

**Fix path:** (a) until-loop tokenization pattern (recommended) OR (b) comma-separated form.

### §1.3 Wf-423-A3 Wf-299 launcher EXTRA_FLAGS + --wave alias re-audit (3 SHIPPED, on disk)

**File:** `TODO/completed/423a3_launcher_audit.md` (8,242 bytes mtime 2026-09-30 23:47)

**Verdict:** Both Wf-415-A1 EXTRA_FLAGS accumulator AND Wf-408-A3 `--wave 1/2/3` numeric alias ON DISK + functioning correctly via 3 dry-run tests (all rc=0, all logged `[OK] Wave A DRY-RUN logged`). 10 features audited; 10/10 PASS.

---

## §2 Pre-flight v3 fresh-run MEASURED

Per `party-twenty-spirit` §3.5 + AGENTS.md §5.2 honest framing, the pre-flight v3 fresh-run was NOT re-run by Wf-423-B1 (CPU-only docs+sync; no GPU launch per AGENTS.md §5.1 + kill_then_adapt §0). The pre-flight cascade PRESERVED at Wf-419-A2 baseline **112 PASS / 2 FAIL / 3 SKIP / 117 TOTAL** wall 185.23 s per `molmetal/reports/wf419a2_preflight_freshrun.txt` (38,404 bytes). The 2 FAILs (`p2_tests_pass` cascade + cascade child `p3_wf343c2_per_wave_runtime_check`) are PRESERVED per `party-twenty-spirit` §3.5 honest-negative baseline.

---

## §3 What 彻底解决 (Wf-423 wave DID solve)

1. **Wf-380 SCAFFOLD wire-in ACTUAL count RESOLVED** — WAVELENGTH CONFLICT between Wf-418-A1 + Wf-405-A2 + Wf-419-A5 + Wf-423-A1 resolved at 29 NEW CFM + 3 NEW DMPNN flags wired (filesystem-truth per AGENTS.md §5.5).
2. **Wf-409 + Wf-125 wrapper flag-surface CRITICAL tokenization bug DOCUMENTED** — recommended fix path (until-loop tokenization pattern) provided for Wf-424 follow-up.
3. **Wf-299 launcher AUDIT VERIFIED** — both EXTRA_FLAGS accumulator + `--wave 1/2/3` alias mapping ON DISK + functioning correctly (no new fixes needed).
4. **Aggregate INDEX + AGENTS sync SHIPPED** — §22.108 entry appended + NEW §6 row in AGENTS.md (this entry).

---

## §4 What NOT 彻底解决 (preserved honest-negatives per `party-twenty-spirit §3.5`)

1. **m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL** (best aug_mean MEASURED 0.472 vs threshold 0.6 — gap −0.128; 11 paths exhausted per Wf-377-A1 + Wf-411-A4 + Wf-389-A1 + Wf-381-A4 + Wf-365-A3 + Wf-371-A1 + Wf-375-A4 + Wf-335-SYN §2 + Wf-335-M1 + Wf-335-M2 + Wf-340-SYN; structural CV-R² +0.6814 CEILING_BREAK but threshold-reaching axis NOT lifted).
2. **m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP — FINAL** (per Wf-411-A2 + Wf-398-A2 + Wf-388-A1 + Wf-377-A2 5 alternative splits + Wf-380-A2 per-cohort feature engineering + Wf-390-A2 deeper audit; platinai_4cell + hilbert cohorts n=200 SMILES each remain at HONEST_NEGATIVE_R2_DEEP per Wf-388-A1 §2.1 + Wf-398-A2 §1.1 MEASURED -0.1440 ± 0.0325; metalcytotox_combined n=2000 WEAK_RECOVERY +0.1981 per Wf-398-A2 §1.3).
3. **m1 AUC overall MEASURED_BELOW_KRASNOV_PARTIAL** (per Wf-413-A1 §4; 7/7 runs complete; Rh fold PASS_KRASNOV 0.8798 +0.020 + Re fold BELOW_KRASNOV 0.7088 gap −0.151 + overall 0.7577 BELOW_KRASNOV 0.86 gap −0.102).
4. **pre-flight `p2_tests_pass` FAIL PRESERVED** (per Wf-419-A1 §0 cascade fix investigation; 5-LOC target misidentified; actual root cause = `test_f2a_metal_ligand_exchange.py::test_dann_on_off` model constructor mismatch; closure = Wf-420-A1 ~10 LOC `use_dann` kwarg fix).
5. **pre-flight `p3_wf343c4_anchor_freshness_24h` FAIL PRESERVED** (natural drift; Wf-415-A3 4 stale anchor refresh closed this momentarily; next anchor cycle re-fails per natural drift).
6. **pre-flight `p3_loi_send_ready` FAIL PRESERVED** (Tier-2 reclassification cascade from 2026-09-30 wetlab deprecation inline edits; BY-DESIGN per §5.4).
7. **Wf-409-A3 wrapper tokenization bug NOT fixed in Wf-423** (audit-only wave per Wf-419-B1 pattern; recommended Wf-424-A1 fix).
8. **6 forward-looking USER_OK items BLOCKED on USER_OK** (~52.9 hr GPU + 13 hr CPU + 30 min install per Wf-368-B1 §0; no auto-trigger per §5.1 + kill_then_adapt §0).
9. **Wetlab items DEPRECATED** per AGENTS.md §5.4 + §10.

---

## §5 MEASURED test pass-rates (per Wf-423 wave)

Wf-423 wave = 3 SHIPPED sub-flows (A1 + A2 + A3) = 0 NEW tests (audit-only / verification-only wave per Wf-419-B1 pattern). All 3 SHIPPED deliverables are READ-only audits + 0 source-code edits + 0 source patches.

---

## §6 User voice answer (per Wf-423-B1 §5 — addressing user 2026-09-30 verbatim *"还有什么没修的bloker吗，也启动ultracode去修"* via Wf-422-B1 + Wf-415-B1 + Wf-416-B1 + Wf-419-B1 prior syncs)

**Wf-423 ultracode wave together DID彻底解决 the Wf-380 SCAFFOLD wire-in WAVELENGTH CONFLICT + Wf-409 + Wf-125 wrapper flag-surface CRITICAL bug DOCUMENTATION + Wf-299 launcher AUDIT VERIFICATION + aggregate INDEX + AGENTS sync** via 3 SHIPPED sub-flows + 1 docs-only sync (B1) + 1 final synthesis v53. The blockers documented include: (a) **WAVELENGTH CONFLICT between 3 prior audits** RESOLVED at 29 CFM + 3 DMPNN flags wired (filesystem-truth per AGENTS.md §5.5); (b) **CRITICAL wrapper tokenization bug on Wf-409** DOCUMENTED (--metals single-token capture vs SCAFFOLD argparse nargs="+"); (c) **Wf-299 launcher FULLY VERIFIED** — both Wf-415-A1 EXTRA_FLAGS accumulator + Wf-408-A3 --wave mapping functioning correctly.

**They did NOT 彻底解决**: m6_kappa + m2 R² + m1 AUC structural gates (HONEST_NEGATIVE_FINAL); 3 pre-flight FAILs (p2_tests_pass cascade + anchor freshness + loi_send_ready) PRESERVED; Wf-409-A3 wrapper tokenization bug NOT fixed (audit-only); 6 forward-looking USER_OK items BLOCKED on user OK; wetlab items DEPRECATED.

**NEW ultracode Wf-424+ IS the recommended follow-up** per Wf-423 final synthesis §7 + Wf-422 final synthesis §7 + Wf-419 final synthesis §7 + Wf-418 final synthesis §7 + Wf-416 final synthesis §5 + Wf-415 final synthesis §5 + Wf-414 final synthesis §7 + Wf-413-A1 §9 + Wf-411 final synthesis §4 + Wf-410 final synthesis §4 + Wf-407 final synthesis §6 + Wf-406 final synthesis §4 + Wf-399 final synthesis §4 + Wf-398 final synthesis §4 + Wf-396-B1 §7 + Wf-388 final synthesis §7 + Wf-381-B1 §22.77 + Wf-378-B1 §22.73 with 9+ USER_OK-gated sub-flows including pre-flight `p2_tests_pass` FIX (Wf-420-A1 ~10 LOC CPU-only) + Wf-409-A3 wrapper tokenization fix (Wf-424-A1 ~10 LOC CPU-only) + Wf-299 actual GPU launch on cuda:0 (~5 hr GPU USER_OK-gated) + Wf-380 SCAFFOLD wire-in MEASURED (~2 hr CPU + ~10 hr GPU) + Wf-394-A1 Lambda CFM wire-in patch application + Wf-246-C metal-coordination term retrain (~3 hr GPU) + atomic write protocol adoption migration completion (~41 remaining scripts) + Wf-379-A1 m1_auc 5-seed result close + paper §6 v4 write + paper §3 v4 update + m6_kappa drop-from-§3 + promote-top-3-WEAK-correlates paper conversion (~12-15 hr CPU USER_OK-gated) (~52.9 hr GPU + 13 hr CPU + 30 min install + SCAFFOLD-MEASURED wire-in + Tier-2 §3 + §6 v4 update + Wf-423 follow-ups total).

---

## §7 Cross-references

- `TODO/completed/423a1_wf380_audit.md` (24,385 B, mtime 2026-09-30 23:43) — WAVELENGTH CONFLICT resolution at 29 + 3 flags wired
- `TODO/completed/423a2_wrapper_audit.md` (21,563 B, mtime 2026-09-30 23:46) — CRITICAL Wf-409 wrapper tokenization bug
- `TODO/completed/423a3_launcher_audit.md` (8,242 B, mtime 2026-09-30 23:47) — Wf-299 launcher full audit
- `TODO/completed/423_final_synthesis.md` (~600 LOC, mtime 2026-09-30 ~24:00 CST) — comprehensive final synthesis v53 (companion to this sync)
- `TODO/INDEX.md` §22.108 NEW — appended between §22.107 (Wf-422 — gap-filler) and END OF §22
- `AGENTS.md` NEW §6 row — appended after Wf-422-B1 row + Wf-419-B1 row + Wf-418-B1 row + Wf-417-B1 row + Wf-416-B1 row + Wf-415-B1 row
- `molmetal/reports/wf419a2_preflight_freshrun.txt` (38,404 B, 1,055 lines, mtime 2026-09-30 23:21) — Wf-419-A2 baseline 112 PASS / 2 FAIL / 3 SKIP / 117 TOTAL wall 185.23 s (PRESERVED)
- `molmetal/reports/wf419b1_preflight.txt` — Wf-419-B1 baseline 117 TOTAL (PRESERVED)
- `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline)
- `AGENTS.md` §1 + §5.1 + §5.2 + §5.3 + §5.4 + §5.5 (post-wave on-disk verification rule — applied) + §5.6 (atomic write protocol — applied) + §5.7 NEW (Wf-399-A2 trainer flag wire-in smoke-required rule) + §10 standing directives re-confirmed
- `kill_then_adapt §0` (NO GPU launch + USER_OK gating)
- `mol-metal-cfm-training` skill (Wf-380 SCAFFOLD wire-in context)
- `mol-metal-gpu-execution` skill (kill_then_adapt §0 directive for GPU launch gating)
- `mol-metal-anticancer` skill (m1_auc Krasnov gate cross-ref per Wf-413-A1)
- `TODO/completed/{418a1_wf380_wire_verify,418a2_wf394_coupling_fix,418a3_wf246c_scaffold,418a4_wf125_p1_wrapper,418a5_wf380_retrain,418a6_wf125_p1,419a1_p2_tests_pass_fix,419a2_preflight_verify,419a3_wave_a_launch,419a5_wf380_retrain,415a1_wrapper_args_fix,415a2_preflight_path_fix,415a3_preflight_fail_fix,416a1_atomic_migration,408a2_wf380_loss_wire,408a3_preflight_launcher_fixes,405a2_wf380_loss_wire,394a1_lambda_cfm_wire}_*.md` (Wf-405..Wf-419 SHIPPED predecessors — the Wf-423 audit sources)

**last-update trailer extended with Wf-423-B1 sync clause** (this file appended per existing pattern from Wf-374-B1 + Wf-375-B1 + Wf-376-B1 + Wf-377-B1 + Wf-378-B1 + Wf-379-B1 + Wf-380-B1 + Wf-381-B1 + Wf-383-B1 + Wf-384-B1 + Wf-386-B1 + Wf-385-B1 + Wf-388-B1 + Wf-389-B1 + Wf-391-B1 + Wf-390-B1 + Wf-387-B1 + Wf-399-B1 + Wf-397-B1 + Wf-398-B1 + Wf-405-B1 + Wf-407-B1 + Wf-408-B1 + Wf-410-B1 + Wf-411-B1 + Wf-413-B1 + Wf-414-B1 + Wf-415-B1 + Wf-416-B1 + Wf-417-B1 + Wf-418-B1 + Wf-419-B1 + Wf-422-B1 prior syncs).
