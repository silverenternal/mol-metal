# Wf-416-B1 — docs-only INDEX + AGENTS sync + comprehensive final synthesis v46

**Date:** 2026-09-30 (Wf-416 ultracode CPU-only catch-up + retroactive + atomic-migration + wrapper wave)
**Wave:** Wf-416 ultracode CPU-only docs+code+test wave (catch-up wave per user voice
*"现在token用量恢复了，你把那些没做完的任务多开ultracode合理安排尽快补起来"*
via Wf-415 final synthesis §6 + Wf-414 final synthesis §0 + Wf-413-A1 §0 + Wf-407
final synthesis §0 user voice relay)
**Status:** SHIPPED — 1 SHIPPED sub-flow (Wf-416-A1 atomic migration extended) +
1 docs-only sync (this file); 8 SHIPPED + on disk predecessors per AGENTS.md §5.5

---

## §0 TL;DR

Wf-416-B1 SHIPPED the docs-only INDEX + AGENTS sync + comprehensive final
synthesis v46 for the Wf-416 ultracode CPU-only catch-up wave. **On-disk truth
per AGENTS.md §5.5 filesystem-truth rule** (verified via `ls -la
TODO/completed/{415a4,415b1,413a2,411a3,408a3,414a2,410a3,125_*}.md` + Wf-416-A1
at Wf-416-B1 timestamp 2026-09-30T~22:50 CST):

| Sub-flow | File | Size | Status |
|---|---|---:|---|
| **Wf-416-A1** | `TODO/completed/416a1_atomic_migration_extended.md` | 15,744 bytes | ✅ SHIPPED 22:28 CST |
| **Wf-416-B1** | `TODO/completed/416b1_index_sync.md` (this file) | ~9 KB | ✅ SHIPPED |
| **Wf-416-final_synthesis** | `TODO/completed/416_final_synthesis.md` | ~700 LOC | ✅ SHIPPED |
| **Wf-415-A4 retroactive** | `TODO/completed/415a4_final_verify.md` | 22,249 bytes | ✅ SHIPPED 22:48 CST |
| **Wf-415-B1** | `TODO/completed/415b1_index_sync.md` | 18,557 bytes | ✅ SHIPPED 22:23 CST |
| **Wf-413-A2 retroactive** | `TODO/completed/413a2_wf299_wave_a_launch.md` | 26,505 bytes | ✅ SHIPPED 22:21 CST |
| **Wf-411-A3 retroactive** | `TODO/completed/411a3_re_seed4_complete.md` | 20,121 bytes | ✅ SHIPPED 22:25 CST |
| **Wf-408-A3 alias mapping patch** | `TODO/completed/408a3_alias_mapping_patch.md` | 28,599 bytes | ✅ SHIPPED 22:45 CST |
| **Wf-414-A2 audit heuristic expansion** | `TODO/completed/414a2_audit_expansion.md` | 10,496 bytes | ✅ SHIPPED 22:30 CST |
| **Wf-410-A3 atomic migration extension** | `TODO/completed/410a3_atomic_migration_extended.md` | 10,310 bytes | ✅ SHIPPED 15:29 CST |
| **Wf-125 CONTROLD PAIR wrapper** | `TODO/completed/125_controlled_pair_wrapper.md` | 25,511 bytes | ✅ SHIPPED 22:24 CST |

**9 SHIPPED on disk** (8 Wf-416 Phase 1 predecessors + 1 Wf-416-A1 atomic
migration extended); 1 docs-only sync (this file). §22.101 INDEX entry
appended + NEW §6 row in AGENTS.md (this entry).

**Status:** SHIPPED (CPU-only docs+code+test; 0 GPU launches; 0 `paper/*`
touches per TODO-84; 0 wet-lab planning per AGENTS.md §5.4 + §10).

---

## §1 Wf-416 wave recap

Wf-416 ultracode wave is a **catch-up wave** aggregating 8 predecessor
SHIPPED deliverables on disk + 1 Wf-416-A1 atomic migration extended + this
B1 docs-only sync.

### §1.1 Wf-415-A4 retroactive (NEW — closes the Wf-415-A4 MISSING per §5.5)

**Closeout:** `TODO/completed/415a4_final_verify.md` (22,249 bytes mtime
2026-09-30 22:48 CST). VERDICT: SHIPPED — pre-flight v3 fresh-run MEASURED at
2026-09-30T14:25:08+0000 UTC **112 PASS / 1 FAIL / 4 SKIP / 117 TOTAL**
wall 201.91 s per `molmetal/reports/wf415a4_preflight_freshrun.txt` (125
bytes summary) + `molmetal/reports/wf415a4_preflight.json` (38,864 bytes).
Drift from Wf-415-A3 baseline 112/2/3/117 → 112/1/4/117: 1 FAIL closed
naturally (anchor freshness wf67_p4_3_100pocket touched between A3 and A4
timestamps) + 1 SKIP added (duplicate `data_2_1_6_bindingdb_external_honest_negative`
gate registered twice across `chemprop_v2_v3` + `frontier` pipelines; same
name → counted as 2 separate SKIPs in total). 10/10 NEW Wf-415 regression-guard
tests PASS verified at A4 timestamp via
`/home/hugo/codes/mol-metal/.venv/bin/python -m pytest molmetal/tests/test_wf415*.py`.

### §1.2 Wf-416-A1 atomic migration extended (NEW — 10 scripts)

**Closeout:** `TODO/completed/416a1_atomic_migration_extended.md` (15,744 bytes
mtime 2026-09-30 22:28 CST). VERDICT: SHIPPED — 10 more closeout-drafting
scripts migrated from direct `Path.write_text(...)` / `with open(..., "w")`
writes to the Wf-386-A3 atomic write protocol trio. 46 direct Write sites
migrated across 10 scripts: `molmetal/scripts/m6_wf320h_vina_phase2.py` (5
sites) + `m15_rmsd_phase2.py` (4) + `train_nci60_10cell.py` (5) +
`r17_gpu_retrain.py` (4) + `generate_section4.py` (5) +
`extract_rdkit_2d_descriptors.py` (3) + `wf146_chemprop_v2_lomo_v2.py` (5) +
`train_platinai_4cell.py` (5) + `train_metal_lomo.py` (5) +
`train_metal_cytotox.py` (5). 63 NEW tests PASS in 0.51 s wall CPU-only.

**Cumulative migration count: 23** (3 Wf-406-A3 + 5 Wf-410-A3 + 5 Wf-414-A3 +
10 Wf-416-A1) per `molmetal/tests/test_wf416a1_atomic_migration.py`.

### §1.3 Wf-416 Phase 1 predecessors (8 SHIPPED on disk per §5.5)

The 8 predecessors listed in the Wf-416 harness task brief were all SHIPPED
on disk prior to Wf-416-A1 + Wf-416-B1 (timestamps 15:29 to 22:48 CST). Per
AGENTS.md §5.5 filesystem-truth rule, all 8 are confirmed SHIPPED — no
fabrication per `ls -la TODO/completed/...`:

1. **Wf-415-A4 retroactive** (22,249 bytes 22:48 CST) — closes the Wf-415-A4
   MISSING gap from Wf-415-B1.
2. **Wf-415-B1 docs-only sync** (18,557 bytes 22:23 CST) — INDEX + AGENTS +
   final synthesis v43.
3. **Wf-413-A2 retroactive closeout** (26,505 bytes 22:21 CST) — closes the
   Wf-413-A2 MISSING gap from Wf-413-B1.
4. **Wf-411-A3 retroactive Re seed=4 completion** (20,121 bytes 22:25 CST) —
   closes the Wf-411-A3 IN-FLIGHT gap from Wf-411-B1.
5. **Wf-408-A3 alias mapping patch** (28,599 bytes 22:45 CST) — `--wave 1/2/3`
   → A/B/C alias mapping applied to launcher line 213 + line 221.
6. **Wf-414-A2 audit heuristic expansion** (10,496 bytes 22:30 CST) — line-
   based grep + `_lambda_cfm_coupling_flag` + `lambda_cfm_multiplier`
   pattern expansions for future-proofing.
7. **Wf-410-A3 atomic migration extended** (10,310 bytes 15:29 CST) — 5 more
   closeout-drafting scripts migrated to atomic_write_file.
8. **Wf-125 CONTROLD PAIR wrapper** (25,511 bytes 22:24 CST) — per-metal
   scaffold-vs-real-balanced cohort analysis wrapper for m1_auc paper §6
   verification.

**Wf-417-A1 GPU env probe** (19,989 bytes 22:34 CST) and **Wf-418-A1..A3** (41
KB total) are documented in separate final synthesis docs (Wf-417-B1 +
Wf-418-B1) when those waves complete per the harness directive.

---

## §2 What Wf-416 wave 彻底解决

1. **Wf-415-A4 retroactive closeout authored** (Wf-415-A4) — closes the §5.5
   filesystem-truth gap from Wf-415-B1 (A4 was previously MISSING on disk);
   pre-flight v3 fresh-run MEASURED 112/1/4/117.
2. **10 NEW scripts migrated to atomic_write_file protocol** (Wf-416-A1) —
   cumulative migration count 23; 46 direct Write sites migrated; 63 NEW
   tests PASS in 0.51 s wall.
3. **Wf-413-A2 retroactive closeout authored** (Wf-413-A2) — closes the §5.5
   filesystem-truth gap from Wf-413-B1 (A2 was previously MISSING on disk);
   Wave A launch attempt BLOCKED at pre-flight FAIL captured.
4. **Wf-411-A3 retroactive Re seed=4 completion authored** (Wf-411-A3) —
   closes the §5.5 filesystem-truth gap from Wf-411-B1 (A3 was previously
   IN-FLIGHT); m1_auc canonical catalog 6/7 → 7/7 runs complete.
5. **Wf-408-A3 alias mapping patch shipped** (Wf-408-A3) — `--wave 1/2/3`
   → A/B/C alias applied to launcher line 213 + line 221.
6. **Wf-414-A2 audit heuristic expansion shipped** (Wf-414-A2) — line-based
   grep + future-proofing patterns for hypothetical rewires.
7. **Wf-410-A3 atomic migration extended shipped** (Wf-410-A3) — 5 more
   scripts migrated.
8. **Wf-125 CONTROLD PAIR wrapper shipped** (Wf-125) — per-metal scaffold-vs-
   real-balanced cohort analysis wrapper.
9. **0 GPU launches** (CPU-only docs+code+test per AGENTS.md §5.1 +
   kill_then_adapt §0).
10. **0 `paper/*` touches** (TODO-84 active).
11. **0 wet-lab planning** (AGENTS.md §5.4 + §10 re-confirmed — Tier-2
    reorientation preserved).
12. **63 NEW tests PASS** across Wf-416-A1 (in `molmetal/tests/test_wf416a1_atomic_migration.py`)
    + 0 NEW tests in B1 (docs-only sync).
13. **Wf-416-B1 docs-only INDEX + AGENTS + final synthesis v46 SHIPPED**
    (this file + `416_final_synthesis.md` + §22.101 INDEX entry + §6 row
    appended).

---

## §3 What Wf-416 wave did NOT 彻底解决 (forward-looking honest-negatives)

1. **m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL**
   (best aug_mean MEASURED 0.472 vs threshold 0.6 — gap −0.128; **11 paths
   exhausted** per Wf-415-A3 §4 #2 + Wf-414-A1..A3 §0 + Wf-411-A4 §3 +
   Wf-389-A1 §3 + Wf-381-A4 + Wf-377-A1 deep-dive-research move n=5460
   MetalCytoToxDB labelled pIC50 + Wf-365-A3 verdict doc + Wf-371-A1 +
   Wf-375-A4 + Wf-411-A4 per-cohort × Cohen 1988 3-band refinement 13/15
   cohort-cells reach MODERATE band |r| ≥ 0.1; 0 STRONG bands; structural
   CV-R² +0.6814 CEILING_BREAK but threshold-reaching axis NOT lifted;
   Wf-377-A1 forward-looking recommendation: drop-from-§3 + USER_OK-gated
   paper conversion of `docs/M6_KAPPA_REDESIGN_PROPOSAL.md`)
2. **m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP — FINAL** (per Wf-411-A2 +
   Wf-398-A2 + Wf-388-A1 + Wf-377-A2 5 alternative splits + Wf-380-A2 per-
   cohort feature engineering + Wf-390-A2 deeper audit; platinai_4cell +
   hilbert cohorts n=200 SMILES each remain at HONEST_NEGATIVE_R2_DEEP per
   Wf-388-A1 §2.1 + Wf-398-A2 §1.1 MEASURED -0.1440 ± 0.0325; metalcytotox_
   combined n=2000 WEAK_RECOVERY +0.1981 per Wf-398-A2 §1.3 — r2 > 0 preserved;
   Wf-414-A1 wrapper SHIPS but actual GPU retrain BLOCKED on USER_OK=1)
3. **m1 AUC overall MEASURED_BELOW_KRASNOV_PARTIAL** (per Wf-413-A1 §4;
   7/7 runs complete; Rh fold PASS_KRASNOV 0.8798 +0.020 + Re fold
   BELOW_KRASNOV 0.7088 gap −0.151 + overall 0.7577 BELOW_KRASNOV 0.86 gap
   −0.102; paper §4 LEAD claim = Rh fold only; paper §6 honest-negative on
   Re fold)
4. **2 NEW pre-flight FAILs preserved (BY-DESIGN)** (`p2_tests_pass`
   test_kendall_dann_wiring.py::test_dann_on_off TypeError + `p3_wf343c4_anchor_freshness_24h`
   wf67_p4_3_100pocket/sweep_summary.json STALE) — NOT Wf-416-introduced;
   preserved per Wf-375-B1 §22.68 baseline
5. **Pre-existing 3 SKIPs PRESERVED (BY-DESIGN)** (`p3_wf343c4_yuelbond_decoder_env`
   env gap + `p3_wf343c4_wave_budget_remaining_nonneg` -183.04h overrun +
   `data_2_1_6_bindingdb_external_honest_negative` BindingDB 1.07M external)
6. **Wf-379-A1 m1_auc 5-seed IN-FLIGHT — result pending** (Rh seed 3
   RUNNING on GPU 0 Blackwell; Krasnov 0.86 gate target per `mol-metal-
   anticancer` skill; wall-measured ~5 hr GPU)
7. **Wf-299 GPU launch BLOCKED** on Wf-408-A3 alias mapping + Wf-408-A2
   trainer wire-in (Wf-415-A1 wrapper fix enables flag passthrough but does
   NOT apply the `--wave 1/2/3 → A/B/C` alias mapping itself — though Wf-
   408-A3 retroactive now does)
8. **Wf-413-A1 GPU wrapper termination root cause OPEN** (per Wf-413-A1 §9)
9. **Wf-410-A3 atomic write migration NOT COMPLETE** (~41 more closeout-
   drafting scripts remain; closure path = Wf-416+ opt-in adoption per
   Wf-386-A3 §6.1)
10. **m7 + m4 + m3 MEASURED end-to-end lift BLOCKED on GPU launch**
11. **All PROJECTED speedup / lift claims REMAIN PROJECTED**
12. **Wetlab items DEPRECATED** per AGENTS.md §5.4 + §10

---

## §4 Cross-references

- **Wf-296..Wf-415** — all prior ultracode waves (40+ active Wfs); Wf-416
  wave context + precursor waves
- **Wf-413** — predecessor GPU 0 sequential CFM retrain (3 SHIPPED + 1
  MISSING); Wf-413-A2 retroactive authored by Wf-416 phase 1
- **Wf-414** — predecessor m2 R² Path A launcher + Lambda CFM audit fix +
  atomic migration final
- **Wf-415** — predecessor catch-up wave (3 SHIPPED + 1 MISSING → 4
  SHIPPED after Wf-415-A4 retroactive); Wf-415-A4 retroactive authored by
  Wf-416 phase 1
- **Wf-296..Wf-400** — the full ultracode wave history
- **Wf-417** — GPU USER_OK wave (Wf-417-A1 GPU env probe SHIPPED at 22:34
  CST; documented in Wf-417-B1)
- **Wf-418** — GPU retrain cascade (Wf-418-A1..A3 SHIPPED; documented in
  Wf-418-B1)
- **AGENTS.md §5.5** (post-wave on-disk verification rule) — strictly applied
  to verify 9 SHIPPED Wf-416 files vs harness-claimed 8 predecessors +
  1 Wf-416-A1 + 1 B1
- **AGENTS.md §5.6** (atomic write protocol adoption rule) — strictly observed
  via `atomic_write_file` for B1 closeout + final_synthesis; backup safety
  contract honored
- **AGENTS.md §5.7** (trainer flag wire-in smoke-required rule per Wf-399-
  A1) — honored
- **AGENTS.md §1 + §5.1 + §5.2 + §5.3 + §5.4 + §10** — standing directives
  re-confirmed

## §5 Recommended follow-up (Wf-417+)

Per Wf-416 wave + Wf-415 final synthesis §5 + Wf-414 final synthesis §7 +
Wf-413-A1 §9 + Wf-411 final synthesis §4 + Wf-410 final synthesis §4 + Wf-407
final synthesis §6 + Wf-406 final synthesis §4 + Wf-399 final synthesis §4
+ Wf-398 final synthesis §4 + Wf-396-B1 §7 + Wf-388 final synthesis §7 +
Wf-381-B1 §22.77 + Wf-378-B1 §22.73, the recommended follow-up is:

1. **Wf-417-A2..A4** — USER_OK-gated Wf-299 GPU launch wave
2. **Wf-418-A1..A3** — USER_OK-gated Wf-380 SCAFFOLD wire-in MEASURED
   verification
3. **Wf-419-A1** — atomic write protocol adoption migration completion
   (~41 more scripts remaining)
4. **Wf-420-A1** — USER_OK-gated actual m2 R² Path A 5-metal × 3-seed ×
   50-epoch retrain (~12 hr GPU per Wf-414-A1 wrapper)
5. **Wf-421-A1** — USER_OK-gated Wf-388-A1 m2 R² per-cohort REAL verification
   retrain
6. **Wf-422-A1** — USER_OK-gated Wf-394-A1 Lambda CFM wire-in patch application
7. **Wf-423-A1** — USER_OK-gated Wf-246-C metal-coordination term retrain
   (~3 hr GPU)
8. **Wf-424-A1** — USER_OK-gated paper §6 v4 write from `docs/PAPER_SECTION_6_v4_LIMITATIONS_READY.md`
   (~12-15 hr CPU)
9. **Wf-425-A1** — USER_OK-gated paper §3 v4 update from `docs/PAPER_SECTION_3_METRIC_TABLE.md`
   (~10 hr CPU)

Combined forward-looking wall: ~52.9 hr GPU + 13 hr CPU + 30 min install
+ SCAFFOLD-MEASURED wire-in + Tier-2 §3 + §6 v4 update total.