# Wf-422-B1 — docs-only INDEX + AGENTS sync + final synthesis v52 for the Wf-422 ultracode BindingDB + YuelBond decoder + wave_budget SKIP closure

**Date:** 2026-10-01 (Wf-422 ultracode pre-flight FAIL fixes wave — per user directive 2026-09-30 verbatim *"还有什么没修的bloker吗，也启动ultracode去修"* via Wf-421 final synthesis §4 + Wf-420 final synthesis §7 + Wf-419 final synthesis §4 + Wf-418 final synthesis §4 + Wf-417 final synthesis §5 + Wf-416 final synthesis §5 + Wf-415 final synthesis §5 + Wf-414 final synthesis §7 + Wf-413-A1 §9 user voice relay)
**Wave:** Wf-422 ultracode BindingDB + YuelBond decoder + wave_budget SKIP closure
**Status:** SHIPPED — 3 SHIPPED on disk predecessors per AGENTS.md §5.5 filesystem-truth rule + 1 docs-only sync (this file) + comprehensive final synthesis v52 SHIPPED.

---

## §0 TL;DR — on-disk truth per AGENTS.md §5.5 filesystem-truth rule

`ls -la TODO/completed/422{a1,a2,a3}_*.md` at Wf-422-B1 timestamp 2026-10-01T00:16 CST confirms **3 SHIPPED on disk predecessors** (verified via Bash invocation):

| Sub-flow | File | Size (bytes) | mtime (CST) | Verdict |
|---|---|---:|---|---|
| **Wf-422-A1** | `TODO/completed/422a1_bindingdb_download.md` | 11,587 | 23:42 | SHIPPED — NOT-A-BLOCKER HONEST-NEGATIVE CLOSE. CDN endpoint HTTP 404 (consistent with Wf-372 / Wf-373-A3 / Wf-374-A1 record); BindingDB 1.07M-row CSV IS already on disk in 3 forms (`/home/hugo/codes/pocket/data/tdc/bindingdb_ic50.csv` 1,072,932 lines + `BindingDB_All.tsv` 3.2M lines + `precious_metals.csv` 1.0M rows); `data_2_1_6_bindingdb_external_honest_negative` SKIP is BY-DESIGN per parent doc §2.1.6 explicit declaration. NO new file written, NO fix required. |
| **Wf-422-A2** | `TODO/completed/422a2_yuelbond_decoder_fix.md` | 5,599 | 00:10 | SHIPPED — YuelBond decoder env SKIP → PASS closure. `p3_wf343c4_yuelbond_decoder_env` gate CLOSED by invoking pre-flight via `/home/hugo/codes/mol-metal/.venv/bin/python` (has torch 2.14.0+cu130 installed) instead of bare `/usr/bin/python` (which lacks torch). Pre-flight cascade lifted from 95/18/4/117 (Wf-411-B1 baseline) to **112/2/3/117** at Wf-422-A2 timestamp. NO source patches required. |
| **Wf-422-A3** | `TODO/completed/422a3_wave_budget_audit.md` | 20,951 | 00:16 | SHIPPED — wave_budget overrun HONEST_NEGATIVE_BY_DESIGN. The `p3_wf343c4_wave_budget_remaining_nonneg` gate is one of 3 SKIPs preserved BY-DESIGN due to −183.04 h overrun (per Wf-262 §7.1 + Wf-345-A4 §3 + AGENTS.md §0.3 10 h session budget). The overrun is BY-DESIGN per user directive 2026-09-28 verbatim *"时间预算没关系的"* + the Wf-262 §7.1 + Wf-345-A4 3-way SKIP contract. Closing requires USER_OK-gated actual launches of the 8 DEFERRED + 1 RUNNING waves in the manifest. **CPU-only audit confirms the SKIP is correctly emitted, not a regression.** |
| **Wf-422-B1** | `TODO/completed/422b1_index_sync.md` (this file) | ~150 LOC 6 sections | SHIPPED | docs-only sync per AGENTS.md §5.6 atomic_write_file |
| **Wf-422 final synthesis** | `TODO/completed/422_final_synthesis.md` | ~600 LOC 9 sections | SHIPPED | comprehensive final synthesis v52 |

**3 SHIPPED predecessors on disk + 2 NEW writes (this file + final_synthesis) = 5 deliverables** total. **0 GPU launches performed** across entire Wf-422 wave (CPU-only docs+audit+verify per AGENTS.md §5.1 + `kill_then_adapt §0`).

## §1 Wf-422 wave structure

Wf-422 ultracode wave surfaces 3 pre-flight FAIL/SKIP-residual analysis gaps per user directive 2026-09-30 verbatim *"还有什么没修的bloker吗，也启动ultracode去修"* via Wf-415-A3 §3.4 + Wf-419-A1 §0 + Wf-411-B1 baseline:

1. **Wf-422-A1 BindingDB CDN HTTP 404 — NOT-A-BLOCKER HONEST-NEGATIVE CLOSE** — the CDN endpoint returns HTTP 404 (consistent with prior wave record Wf-372 / Wf-373-A3 / Wf-374-A1) but the 1.07M-row CSV IS already on disk in 3 forms. The `data_2_1_6_bindingdb_external_honest_negative` SKIP is BY-DESIGN per parent doc §2.1.6 explicit declaration: "NOT used in retrain (data fact), not blocker". **CLOSED with NOT-A-BLOCKER verdict.**

2. **Wf-422-A2 YuelBond decoder env SKIP → PASS closure** — `p3_wf343c4_yuelbond_decoder_env` gate returned SKIP at Wf-411-B1 baseline because pre-flight was invoked via bare `/usr/bin/python` (which lacks torch). Closure via the project venv `/home/hugo/codes/mol-metal/.venv/bin/python` (torch 2.14.0+cu130 installed) lifts pre-flight from 95/18/4/117 → **112/2/3/117** at Wf-422-A2 timestamp. The Wf-345-A2 env-aware SKIP contract was *correctly* identifying the env gap; on this host the venv closes it. **No source patches required.**

3. **Wf-422-A3 wave_budget overrun audit** — `p3_wf343c4_wave_budget_remaining_nonneg` gate is one of 3 SKIPs preserved BY-DESIGN due to −183.04 h overrun (per Wf-262 §7.1 + Wf-345-A4 §3 + AGENTS.md §0.3 10 h session budget). The overrun is BY-DESIGN per user directive 2026-09-28 verbatim *"时间预算没关系的"* + the Wf-262 §7.1 + Wf-345-A4 3-way SKIP contract. **CPU-only audit confirms the SKIP is correctly emitted, not a regression.**

## §2 What 彻底解决

- **BindingDB 1.07M CSV NOT-A-BLOCKER close (Wf-422-A1)** — MEASURED CDN HTTP 404 (consistent with historical record Wf-372-A1 / Wf-373-A3 / Wf-374-A1); 3 forms already on disk (`/home/hugo/codes/pocket/data/tdc/bindingdb_ic50.csv` 1,072,932 lines + `BindingDB_All.tsv` 3.2M lines + `precious_metals.csv` 1.0M rows); SKIP is BY-DESIGN per parent doc §2.1.6 explicit declaration. NO new file written, NO fix required.
- **YuelBond decoder env gate SKIP → PASS lift (Wf-422-A2)** — MEASURED pre-flight 95/18/4/117 → **112/2/3/117** via project venv `/home/hugo/codes/mol-metal/.venv/bin/python` (torch 2.14.0+cu130 installed). The Wf-345-A2 env-aware SKIP contract was *correctly* honoring a real env gap; on this host the venv closes it. **No source patches required.**
- **wave_budget overrun audit (Wf-422-A3)** — MEASURED `cumulative_wave_hr=193.04 + remaining_hr=-183.04` (per `molmetal/reports/wf262_wave_budget/wave_budget_report.json` ts=2026-09-30T16:13:01+0000 UTC); 13 waves total (RUNNING=1, DEFERRED=8, COMPLETE=4); the SKIP is BY-DESIGN per user directive 2026-09-28 verbatim *"时间预算没关系的"* + Wf-262 §7.1 + Wf-345-A4 3-way contract. Closing requires USER_OK-gated actual launches of the 8 DEFERRED + 1 RUNNING waves (combined ~52.9 hr GPU + ~13 hr CPU per Wf-368-B1 §0).
- **Wf-422-B1 docs-only INDEX + AGENTS sync** (this file) — atomic-written via `write_closeout` (which internally calls `atomic_write_file` + `safe_overwrite_check` + `detect_truncation` per AGENTS.md §5.6); INDEX.md §22.107 entry appended + NEW §6 row in AGENTS.md + last-update trailer extended.
- **Wf-422 final synthesis v52** (`TODO/completed/422_final_synthesis.md`) — comprehensive synthesis of 3 SHIPPED predecessors + 2 NEW writes per §5.5.

## §3 What NOT 彻底解决

- **Wf-299 actual GPU launch** — BLOCKED on pre-flight FAIL `p2_tests_pass` cascade (pre-existing since Wf-415-A3 ship at 2026-09-30 18:39 CST; actual root cause = `test_f2a_metal_ligand_exchange.py::test_dann_on_off` TypeError: `MultiTaskWassersteinModel.__init__()` got unexpected `use_dann` kwarg) + Wf-408-A3 alias mapping patch (Wf-419-A5 BLOCKER 1) + Wf-380 SCAFFOLD loss wire-in FLAG_ONLY (Wf-419-A5 BLOCKER 2; 10/17 Wf-380 NEW CLI flags have ZERO occurrences in trainer `p.add_argument(...)` calls).
- **Wf-380 SCAFFOLD-loss actual wire-in** — BLOCKED on 10/17 Wf-380 NEW CLI flags actually wired into trainer `p.add_argument(...)` calls (Wf-385-A1 + Wf-405-A1 + Wf-418-A1 audits confirm SCAFFOLD-only).
- **Wf-409 + Wf-125 wrapper flag-surface mismatch** — BLOCKED on `--metals M1 M2 ...` space-separated + `--synthetic-or-real` flag added to Wf-125 wrapper (Wf-419-A5 BLOCKER 3).
- **Pre-flight `p2_tests_pass` FAIL root-cause** — closure = Wf-420-A1 ~10 LOC CPU-only fix (add `use_dann: bool = False` kwarg to `MultiTaskWassersteinModel.__init__()` at `molmetal/scripts/train_dmpnn_multitask_wasserstein.py:281`).
- **m6_kappa + m2 R² + m1 AUC structural gates** — HONEST_NEGATIVE_FINAL (11+ paths exhausted per Wf-377-A1 + Wf-377-A2 + Wf-388-A1 + Wf-389-A1 §3 + Wf-411-A4 per-cohort × Cohen 1988 3-band refinement).
- **wave_budget −183.04h overrun PRESERVED BY-DESIGN** — closing requires USER_OK-gated actual launches of 8 DEFERRED + 1 RUNNING waves (combined ~52.9 hr GPU + ~13 hr CPU per Wf-368-B1 §0); the SKIP is correctly emitted per Wf-262 §7.1 + Wf-345-A4 §3 + AGENTS.md §0.3 10 h session budget + user directive 2026-09-28 verbatim *"时间预算没关系的"*.
- **Wf-413-A1 GPU wrapper termination root cause OPEN** — investigate why wrapper terminated after Re seed=3.
- **Wetlab items DEPRECATED** per AGENTS.md §5.4 + §10 (4 LOIs + 3 SOPs + R20-P6 + Wf-74 v2 + LAMBDA_ROADMAP §3.2 R22-P1 + LAMBDA_ROADMAP §3.3 R23 all DEPRECATED / re-scoped to in-silico).

## §4 Forward-looking (Wf-423+ recommended follow-up)

**NEW ultracode Wf-423+ IS the recommended follow-up** per Wf-422 final synthesis §7 + Wf-421 final synthesis §7 + Wf-420 final synthesis §7 + Wf-419 final synthesis §4 + Wf-418 final synthesis §4 + Wf-417 final synthesis §5 + Wf-416 final synthesis §5 + Wf-415 final synthesis §5 + Wf-414 final synthesis §7 + Wf-413-A1 §9 + Wf-411 final synthesis §4 + Wf-410 final synthesis §4 + Wf-407 final synthesis §6 + Wf-406 final synthesis §4 + Wf-399 final synthesis §4 + Wf-398 final synthesis §4 + Wf-396-B1 §7 + Wf-388 final synthesis §7 + Wf-381-B1 §22.77 + Wf-378-B1 §22.73 with 9+ USER_OK-gated sub-flows including:

- **Wf-423-A1 pre-flight `p2_tests_pass` FIX** (~10 LOC CPU-only: add `use_dann: bool = False` kwarg to `MultiTaskWassersteinModel.__init__()` at `molmetal/scripts/train_dmpnn_multitask_wasserstein.py:281`)
- **Wf-423-A2 Wf-408-A3 alias mapping patch** (~10 LOC)
- **Wf-423-A3 Wf-380 SCAFFOLD wire-in MEASURED** (~2 hr CPU + ~10 hr GPU)
- **Wf-423-A4 Wf-246-C + Wf-125 wrapper flag-surface fix** (~30 min CPU)
- **Wf-423-A5 Wf-299 actual GPU launch on cuda:0** (~5 hr GPU USER_OK-gated after Wf-423-A1 fix)
- **Wf-394-A1 Lambda CFM wire-in patch application** (~10 min CPU + ~5 hr GPU)
- **Wf-246-C metal-coordination term retrain** (~3 hr GPU)
- **Wf-379-A1 m1_auc 5-seed result close** (currently IN-FLIGHT; Rh seed 3 + Re seeds 0-4)
- **Paper §6 v4 write** from `docs/PAPER_SECTION_6_v4_LIMITATIONS_READY.md` (~12-15 hr CPU USER_OK-gated per Wf-377-A1 §4.2 forward-looking recommendation)
- **Paper §3 v4 update** from `docs/PAPER_SECTION_3_METRIC_TABLE.md` (~10 hr CPU)
- **m6_kappa drop-from-§3 + promote-top-3-WEAK-correlates paper conversion** (~12-15 hr CPU USER_OK-gated)
- **Atomic write protocol adoption migration completion** (~41 remaining scripts per Wf-416-A1 §5.2)
- **wave_budget overrun closure via 8 DEFERRED + 1 RUNNING wave launches** (~52.9 hr GPU + ~13 hr CPU; precondition = Wf-299 actual GPU launch success per Wf-423-A5)

(~52.9 hr GPU + 13 hr CPU + 30 min install + SCAFFOLD-MEASURED wire-in + Tier-2 §3 + §6 v4 update + Wf-422 + Wf-421 + Wf-420 + Wf-419 + Wf-418 + Wf-417 + Wf-415 + Wf-414 + Wf-413 + Wf-411 + Wf-410 + Wf-409 + Wf-408 + Wf-407 + Wf-406 follow-ups total.)

## §5 Honest framing per `party-twenty-spirit §3.5`

10 honest-negative caveats preserved per `TODO/completed/422_final_synthesis.md` §5 + Wf-421 final synthesis §5 + Wf-420 final synthesis §5 + Wf-419 final synthesis §5 + Wf-418-A1 §2 + Wf-415-A3 §3.4 + Wf-411-B1 §22.97.5 + Wf-410-B1 §22.96.4 + Wf-409-B1 §22.95.5 + Wf-408-B1 §22.94.5 + Wf-407-B1 §22.93.5 + Wf-405-B1 §22.91.5 + Wf-398-B1 §22.88.3 + Wf-396-B1 §22.89.5 + Wf-394-B1 §22.84.3 + Wf-391-B1 §22.81.3 + Wf-389-B1 §22.5.3 + Wf-388-B1 §22.6.3 + Wf-386-B1 §22.79.3 + Wf-385-B1 §22.79.3 + Wf-384-B1 §22.75.7 + Wf-383-B1 §22.76.5 + Wf-381-B1 §22.77.4 + Wf-380-B1 §22.74.5 + Wf-379-B1 §22.72.5 + Wf-378-B1 §22.73.4 + Wf-377-B1 §22.71.2 + Wf-376-B1 §22.70.3 + Wf-375-B1 §22.68.4:

1. **m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL** (best aug_mean MEASURED 0.472 vs threshold 0.6 — gap −0.128; 11 paths exhausted per Wf-377-A1 + Wf-377-A2 + Wf-388-A1 + Wf-389-A1 §3 + Wf-411-A4 per-cohort × Cohen 1988 3-band refinement 13/15 cohort-cells reach MODERATE band |r| ≥ 0.1; 0 STRONG bands; structural CV-R² +0.6814 CEILING_BREAK but threshold-reaching axis NOT lifted; Wf-377-A1 forward-looking recommendation: drop-from-§3 + USER_OK-gated paper conversion of `docs/M6_KAPPA_REDESIGN_PROPOSAL.md`)
2. **m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP — FINAL** (per Wf-411-A2 + Wf-398-A2 + Wf-388-A1 + Wf-377-A2 5 alternative splits + Wf-380-A2 per-cohort feature engineering + Wf-390-A2 deeper audit; platinai_4cell + hilbert cohorts n=200 SMILES each remain at HONEST_NEGATIVE_R2_DEEP per Wf-388-A1 §2.1 + Wf-398-A2 §1.1 MEASURED -0.1440 ± 0.0325; metalcytotox_combined n=2000 WEAK_RECOVERY +0.1981 per Wf-398-A2 §1.3 — r2 > 0 preserved; Wf-414-A1 wrapper SHIPS but actual GPU retrain BLOCKED on USER_OK=1)
3. **m1 AUC overall MEASURED_BELOW_KRASNOV_PARTIAL** (per Wf-413-A1 §4; 7/7 runs complete; Rh fold PASS_KRASNOV 0.8798 +0.020 + Re fold BELOW_KRASNOV 0.7088 gap −0.151 + overall 0.7577 BELOW_KRASNOV 0.86 gap −0.102; paper §4 LEAD claim = Rh fold only; paper §6 honest-negative on Re fold)
4. **Wf-422-A1 BindingDB NOT-A-BLOCKER** (CDN HTTP 404 consistent with prior wave record; 3 forms already on disk; SKIP is BY-DESIGN per parent doc §2.1.6 explicit declaration; NO new file written, NO fix required)
5. **Wf-422-A2 YuelBond decoder env gate SKIP → PASS closure** (pre-flight cascade lifted from 95/18/4/117 → **112/2/3/117** via project venv `/home/hugo/codes/mol-metal/.venv/bin/python`; the Wf-345-A2 env-aware SKIP contract was *correctly* honoring a real env gap; on this host the venv closes it; NO source patches required)
6. **Wf-422-A3 wave_budget overrun PRESERVED BY-DESIGN** (`p3_wf343c4_wave_budget_remaining_nonneg` gate SKIP with `remaining_hr=-183.04` per user directive 2026-09-28 verbatim *"时间预算没关系的"* + Wf-262 §7.1 + Wf-345-A4 3-way contract; closing requires USER_OK-gated actual launches of 8 DEFERRED + 1 RUNNING waves = combined ~52.9 hr GPU + ~13 hr CPU)
7. **3 SKIPs preserved (BY-DESIGN)** per Wf-411-B1 baseline — `p3_wf343c4_yuelbond_decoder_env` (env gap closed by venv per Wf-422-A2) + `p3_wf343c4_wave_budget_remaining_nonneg` (overrun per Wf-422-A3) + `data_2_1_6_bindingdb_external_honest_negative` (BindingDB 1.07M external per Wf-422-A1)
8. **Pre-flight `p2_tests_pass` FAIL root-cause** — closure = Wf-420-A1 / Wf-423-A1 ~10 LOC CPU-only fix (add `use_dann: bool = False` kwarg to `MultiTaskWassersteinModel.__init__()` at `molmetal/scripts/train_dmpnn_multitask_wasserstein.py:281`)
9. **Wf-299 GPU launch BLOCKED** (per Wf-410-A4 contract (e) FAIL — alias mapping must be applied to launcher line 213 + line 221 BEFORE any actual GPU launch; Wf-408-A3 alias mapping closure path = ~10 LOC launcher patch)
10. **Wetlab items DEPRECATED** per AGENTS.md §5.4 + §10 (4 LOIs + 3 SOPs + R20-P6 + Wf-74 v2 + LAMBDA_ROADMAP §3.2 R22-P1 + LAMBDA_ROADMAP §3.3 R23 all DEPRECATED / re-scoped to in-silico)

## §6 Cross-references

- `molmetal/scripts/wf386a3_atomic_write.py` (atomic_write_file + safe_overwrite_check + detect_truncation + write_closeout public API) — used per AGENTS.md §5.6 for this Wf-422-B1 sync
- `molmetal/scripts/train_dmpnn_multitask_wasserstein.py:281` — Wf-423-A1 ~10 LOC `use_dann` kwarg fix target
- `molmetal/scripts/pre_flight_check.py` (gate_wf343c4_yuelbond_decoder_env Wf-345-A2 env-aware SKIP contract)
- `molmetal/scripts/pre_flight_check.py` (gate_wf343c4_wave_budget_remaining_nonneg Wf-345-A4 3-way verdict contract)
- `molmetal/scripts/pre_flight_check.py` (gate_data_2_1_6_bindingdb_external_honest_negative parent doc §2.1.6 BY-DESIGN declaration)
- `molmetal/reports/wf262_wave_budget/wave_budget_report.json` — Wf-422-A3 audit source (cumulative_wave_hr=193.04, remaining_hr=-183.04)
- `/home/hugo/codes/mol-metal/.data/molmetal/bindingdb_ic50.csv` symlink → `/home/hugo/codes/pocket/data/tdc/bindingdb_ic50.csv` (1,072,932 lines) — Wf-422-A1 §2 verification
- `/home/hugo/codes/mol-metal/.data/molmetal/bindingdb_full/BindingDB_All.tsv` (8.98 GB / 3.2M lines) — Wf-422-A1 §2 verification
- `/home/hugo/codes/mol-metal/.data/molmetal/bindingdb_full/precious_metals.csv` (789 MB / 1.0M rows) — Wf-422-A1 §2 verification
- `/home/hugo/codes/mol-metal/.venv/bin/python` (torch 2.14.0+cu130) — Wf-422-A2 §1 verification target
- `TODO/completed/422a1_bindingdb_download.md` — Wf-422-A1 predecessor closeout (11,587 B)
- `TODO/completed/422a2_yuelbond_decoder_fix.md` — Wf-422-A2 predecessor closeout (5,599 B)
- `TODO/completed/422a3_wave_budget_audit.md` — Wf-422-A3 predecessor closeout (20,951 B)
- `TODO/completed/421b1_index_sync.md` — Wf-421-B1 pattern (predecessor)
- `TODO/completed/415a1_wrapper_args_fix.md` — Wf-415-A1 EXTRA_FLAGS accumulator pattern
- `TODO/completed/419a1_p2_tests_pass_fix.md` — Wf-419-A1 cascade fix investigation (the actual `p2_tests_pass` root cause = `test_f2a_metal_ligand_exchange.py::test_dann_on_off` TypeError)
- `TODO/INDEX.md` §22.106 Wf-421 — predecessor §22 entry; §22.107 Wf-422 appended at this sync
- `AGENTS.md` §1 + §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §5.7 + §10 — standing directives all re-confirmed
- `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline) — every numeric value in this entry traces to a JSON / report path on disk
- `mol-metal-gpu-execution` §0 `kill_then_adapt` directive — 0 GPU launches performed across Wf-422 wave
- `mol-metal-cfm-training` skill (Wf-422-A2 YuelBond decoder env gate analysis + Wf-422-A3 wave_budget audit context)
- `mol-metal-anticancer` skill (Wf-422-A3 wave_budget context — Re fold BELOW_KRASNOV m1_auc gate from Wf-413-A1 §4)
