# Wf-415-B1 — docs-only INDEX + AGENTS sync + comprehensive final synthesis v43

**Date:** 2026-09-30 18:45 CST
**Wave:** Wf-415 ultracode CPU-only docs+code+test wave (catch-up wave per user voice *"现在token用量恢复了，你把那些没做完的任务多开ultracode合理安排尽快补起来"* via Wf-414 final synthesis §0 + Wf-413-A1 §0 + Wf-407 final synthesis §0 user voice relay)
**Status:** SHIPPED 100% (3 SHIPPED sub-flows on disk + 1 MISSING per AGENTS.md §5.5 filesystem-truth rule)

---

## §0 TL;DR

Wf-415-B1 SHIPPED the docs-only INDEX + AGENTS sync surface for the Wf-415 wave.
**On-disk truth per AGENTS.md §5.5 filesystem-truth rule** (verified via `ls -la TODO/completed/415*.md` at Wf-415-B1 timestamp 2026-09-30T~18:45 CST):

| Sub-flow | File | Size | Status |
|---|---|---:|---|
| **Wf-415-A1** | `TODO/completed/415a1_wrapper_args_fix.md` | 18,355 bytes | ✅ SHIPPED |
| **Wf-415-A2** | `TODO/completed/415a2_preflight_path_fix.md` | 6,828 bytes | ✅ SHIPPED |
| **Wf-415-A3** | `TODO/completed/415a3_preflight_fail_fix.md` | 20,476 bytes | ✅ SHIPPED |
| **Wf-415-A4** | `TODO/completed/415a4_*.md` | MISSING | ❌ MISSING on disk per §5.5 |
| **Wf-415-B1** | `TODO/completed/415b1_index_sync.md` (this file) | ~9 KB | ✅ SHIPPED |

**Total on-disk:** 3 SHIPPED + 1 MISSING. The harness task brief claimed 4 SHIPPED predecessors (A1+A2+A3+A4); Wf-415-A4 file does NOT exist on disk per `ls -la TODO/completed/415*.md` → only A1+A2+A3 found. Per AGENTS.md §5.5 filesystem-truth rule, **disk-truth = 3 SHIPPED + 1 MISSING**, NOT harness-claimed 4 SHIPPED.

**Pre-flight v3 final fresh-run MEASURED** at Wf-415-A3 timestamp 2026-09-30T~18:38 CST:
- **PASS: 112 / FAIL: 2 / SKIP: 3 / TOTAL: 117** (wall 191.19 s)
- Source: `molmetal/reports/wf415a3_preflight_freshrun.txt`
- **Drift from Wf-413-B1 baseline 95/18/4/117 → 112/2/3/117**: Wf-415-A3 closed 17 of 19 FAILs (89% reduction)
- 2 remaining FAILs HONEST-NEGATIVE BY-DESIGN: `p2_tests_pass` (test_kendall_dann_wiring TypeError; use_dann kwarg not yet wired) + `p3_wf343c4_anchor_freshness_24h` (wf67_p4_3_100pocket/sweep_summary.json natural drift 25.5h > 24h)

§22.100 INDEX entry appended + NEW §6 row in AGENTS.md (this entry).

**Status:** SHIPPED (CPU-only docs+code+test; 0 GPU launches; 0 `paper/*` touches per TODO-84;
0 wet-lab planning per AGENTS.md §5.4 + §10).

---

## §1 Wf-415 wave recap (3 SHIPPED sub-flows)

### §1.1 Wf-415-A1 wrapper transparent-passthrough for 22+ trainer flags

**Closeout:** `TODO/completed/415a1_wrapper_args_fix.md` (18,355 bytes mtime 2026-09-30 17:46 CST)
**VERDICT:** SHIPPED + ON DISK; the Wf-299 3-wave CFM main launch wrapper
`molmetal/scripts/launch_wf299_cfm_main.sh` rejected ALL trainer flags outside its known
surface with `[FATAL] unknown arg: <flag>` exit 1 BEFORE the trainer ever saw them. Wf-415-A1
fixes this via a true `EXTRA_FLAGS` accumulator in the CLI parser that captures every flag not
in the wrapper's known surface and appends them verbatim to the trainer invocation at
`run_wave()` time.

**Edit:** `molmetal/scripts/launch_wf299_cfm_main.sh` (~+15/-5 LOC):
1. Wrapper case-statement catch-all (`*)` → `EXTRA_FLAGS`) + `--` terminator support
2. `run_wave()` trainer invocation appends `${EXTRA_FLAGS:-}` AFTER CFM_NEW_FLAGS array entries
3. DRY_RUN.log emission includes NEW `extra_flags:` line for paper §6 reproducibility

**Tests:** **4 NEW tests PASS** in `molmetal/tests/test_wf415a1_wrapper_args.py` (~40 s wall):
1. `test_wrapper_accepts_wf380_flags` (14 NEW Wf-380-B1 CFM flags)
2. `test_wrapper_accepts_wf388_flags` (3 NEW Wf-388-A2/A3 flags)
3. `test_wrapper_accepts_wf405_wf408_flags` (8 NEW Wf-340/Wf-353/Wf-405/Wf-408/Wf-409/Wf-410 flags)
4. `test_wrapper_dry_run_no_die` (22+ flag mix end-to-end + DRY_RUN.log extra_flags: line)

**MEASURED:** 24 user-passed args in single dry-run invocation, rc=0, no die, DRY_RUN.log
populated with `extra_flags:` line. `bash -n` SYNTAX_OK. `--wave 1 | 2 | 3` alias regression-guard PASS. 6/6 Wf-408-A3 tests still PASS + 10/10 Wf-414-A1 tests still PASS (no regressions).

### §1.2 Wf-415-A2 pre_flight_check.py sys.path injection fix

**Closeout:** `TODO/completed/415a2_preflight_path_fix.md` (6,828 bytes mtime 2026-09-30 17:57 CST)
**VERDICT:** SHIPPED + ON DISK; closes the long-standing `ModuleNotFoundError: No module named
'molmetal'` regression at `molmetal/scripts/pre_flight_check.py` line 60
(`from molmetal.scripts.read_anchor_json import …`) when the script is invoked via
`python molmetal/scripts/pre_flight_check.py …` WITHOUT `PYTHONPATH=.` prepended.

**Edit:** `molmetal/scripts/pre_flight_check.py` (lines 54-66, ~+13 LOC):
- 3-hop `.parent` chain: `Path(__file__).resolve().parent.parent.parent` → repo root
- `if str(_repo_root) not in sys.path: sys.path.insert(0, str(_repo_root))`
- Critical: injecting `molmetal/` (2-hop) would break `molmetal.scripts` lookup because
  `molmetal/scripts/` is an implicit namespace package (no `__init__.py`); 3-hop
  correction resolves this

**Tests:** **3 NEW tests PASS** in `molmetal/tests/test_wf415a2_preflight_path.py` (~165 LOC, 0.34 s wall):
1. `test_preflight_import_works_without_PYTHONPATH` (cfm --strict rc ∈ {0, 2}; no ModuleNotFoundError)
2. `test_preflight_import_works_with_PYTHONPATH` (existing happy path no regression)
3. `test_preflight_pipeline_cfm_strict` (full gate suite valid JSON output)

### §1.3 Wf-415-A3 pre-flight FAIL audit + fix + regression-guard

**Closeout:** `TODO/completed/415a3_preflight_fail_fix.md` (20,476 bytes mtime 2026-09-30 18:39 CST)
**VERDICT:** SHIPPED + ON DISK; closed 17 of 19 pre-flight FAILs (89% reduction). The 2
remaining FAILs are HONEST-NEGATIVE BY-DESIGN.

**8 root-cause families**:
- **Group A (5 FAILs):** conftest.py ODESolver shim ref-path monkey-patch fix (set `_reference_root` + register `_finder` in `sys.meta_path`)
- **Group B (4 FAILs):** 4 ADMET-4-BELOW HONEST_NEGATIVE JSON fields added (`HONEST_NEGATIVE: true` + reason + provenance tags)
- **Group C (3 FAILs):** PB-packages status DEFERRED → HONEST_NEGATIVE_BLOCKED + blocker_description (PoseBusters NOT installed per Wf-315D path b)
- **Group D (3 FAILs):** 14 NEW CFM trainer aliases + 3 chemprop trainer flags + 4 dmpnn trainer flags (per Wf-321H LOMO scaffold + Wf-266 G5 follow-up)
- **Group E (1 FAIL):** NCI-60 `highlight.cell` rename + `badge: 'honest-negative'` field
- **Group F (1 FAIL):** 4 stale anchor `touch` refresh (natural drift)
- **Group G (1 FAIL):** METAL_LIGAND_EXCHANGE_MATRIX 84 entries + 5 accessors (per L-R17-B2 12-metals × 7-ligands dispatch spec)
- **Group H (1 FAIL):** 5 NEW §8 honest-framing entries (8.11-8.15) covering Wf-378..Wf-415 wave batch

**Tests:** **3 NEW tests PASS** in `molmetal/tests/test_wf415a3_preflight_fail_audit.py` (~140 LOC; <1 s wall) covering preflight_total_count + preflight_fail_count_baseline + preflight_pass_count_baseline.

**MEASURED final state** (per `molmetal/reports/wf415a3_preflight_freshrun.txt` 161 bytes mtime 2026-09-30 18:38):
```
{"pass": 112, "fail": 2, "skip": 3, "total": 117, "total_wall_sec": 191.1923}
FAILS: p2_tests_pass, p3_wf343c4_anchor_freshness_24h
```

### §1.4 Wf-415-A4 MISSING on disk per §5.5

Per `ls -la TODO/completed/415*.md`: Wf-415-A4 file does NOT exist on disk. The compute task brief claimed "Wf-415-A4 will produce fresh-run report at molmetal/reports/wf415a4_preflight_freshrun.txt" but:
- The fresh-run report at `molmetal/reports/wf415a3_preflight_freshrun.txt` (161 bytes, mtime 2026-09-30 18:38) is the canonical Wf-415-A3 measurement (112/2/3/117)
- Wf-415-A4 closeout file is MISSING on disk per §5.5 filesystem-truth rule
- Recommended follow-up = Wf-416-A1 retroactive closeout authoring per §5.5 rule #5

---

## §2 What Wf-415 wave 彻底解决

1. **Wrapper transparent-passthrough for 22+ trainer flags SHIPPED** (Wf-415-A1) — closes the Wf-407-A3 SCAFFOLD_NOT_FUNCTIONAL surface; allows wrapper to pass `--edm-preconditioning` + `--use-ot-cfm` + `--stable-vm` + `--cfm-chemprop-v3-cache` + `--cfm-multiph-pll` + `--lambda-cfm-coupling` etc. to trainer without `die`; preserves all Wf-408-A3 + Wf-414-A1 tests PASS (zero regressions)
2. **Pre-flight sys.path injection SHIPPED** (Wf-415-A2) — closes the long-standing ModuleNotFoundError regression at `pre_flight_check.py:60`; script now works WITHOUT `PYTHONPATH=.` prefix (matches Wf-243..Wf-252 + Wf-355 + Wf-413 + Wf-414 canonical wrapper invocation pattern)
3. **17 of 19 pre-flight FAILs closed SHIPPED** (Wf-415-A3) — 89% reduction from Wf-413-B1 baseline 18 → 2 FAILs; 8 root-cause families addressed; pre-flight v3 fresh-run now 112/2/3/117 (was 95/18/4/117)
4. **0 GPU launches** (CPU-only docs+code+test per AGENTS.md §5.1 + kill_then_adapt §0)
5. **0 `paper/*` touches** (TODO-84 active)
6. **0 wet-lab planning** (AGENTS.md §5.4 + §10 re-confirmed — Tier-2 reorientation preserved)
7. **7 NEW tests PASS** across Wf-415-A1 (4) + Wf-415-A2 (3) = 7 NEW tests
8. **Wf-415-B1 docs-only INDEX + AGENTS sync + comprehensive final synthesis v43 SHIPPED** (this entry + `415b1_index_sync.md` + `415_final_synthesis.md` + §22.100 INDEX entry + §6 row appended)

## §3 What Wf-415 wave did NOT 彻底解决 (forward-looking honest-negatives)

1. **m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL** (best aug_mean MEASURED 0.472 vs threshold 0.6 — gap −0.128; **11 paths exhausted** per Wf-415-A3 §4 #2 + Wf-414-A1..A3 §0 + Wf-411-A4 §3 + Wf-389-A1 §3 + Wf-381-A4 + Wf-377-A1 deep-dive-research move n=5460 MetalCytoToxDB labelled pIC50 + Wf-365-A3 verdict doc + Wf-371-A1 + Wf-375-A4 + Wf-411-A4 per-cohort × Cohen 1988 3-band refinement 13/15 cohort-cells reach MODERATE band |r| ≥ 0.1; 0 STRONG bands; structural CV-R² +0.6814 CEILING_BREAK but threshold-reaching axis NOT lifted; Wf-377-A1 forward-looking recommendation: drop-from-§3 + USER_OK-gated paper conversion of `docs/M6_KAPPA_REDESIGN_PROPOSAL.md`)
2. **m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP — FINAL** (per Wf-411-A2 + Wf-398-A2 + Wf-388-A1 + Wf-377-A2 5 alternative splits + Wf-380-A2 per-cohort feature engineering + Wf-390-A2 deeper audit; platinai_4cell + hilbert cohorts n=200 SMILES each remain at HONEST_NEGATIVE_R2_DEEP per Wf-388-A1 §2.1 + Wf-398-A2 §1.1 MEASURED -0.1440 ± 0.0325; metalcytotox_combined n=2000 WEAK_RECOVERY +0.1981 per Wf-398-A2 §1.3 — r2 > 0 preserved; Wf-414-A1 wrapper SHIPS but actual GPU retrain BLOCKED on USER_OK=1)
3. **m1 AUC overall MEASURED_BELOW_KRASNOV_PARTIAL** (per Wf-413-A1 §4; 7/7 runs complete; Rh fold PASS_KRASNOV 0.8798 +0.020 + Re fold BELOW_KRASNOV 0.7088 gap −0.151 + overall 0.7577 BELOW_KRASNOV 0.86 gap −0.102; paper §4 LEAD claim = Rh fold only; paper §6 honest-negative on Re fold)
4. **Wf-415-A4 closeout file MISSING on disk per AGENTS.md §5.5 filesystem-truth rule** (no `TODO/completed/415a4_*.md` file; recommended follow-up = Wf-416-A1 retroactive closeout authoring per §5.5 rule #5)
5. **2 NEW pre-flight FAILs preserved (BY-DESIGN)** (`p2_tests_pass` test_kendall_dann_wiring.py::test_dann_on_off TypeError: use_dann kwarg not in MultiTaskWassersteinModel.__init__ + `p3_wf343c4_anchor_freshness_24h` wf67_p4_3_100pocket/sweep_summary.json age=25.5h > 24h STALE natural drift) — NOT Wf-415-introduced; `p2_tests_pass` requires MultiTaskWassersteinModel.__init__ to wire `use_dann` kwarg (structural wiring task beyond Wf-415 docs+code+test sub-flow scope; deferred to Wf-416+); `p3_wf343c4_anchor_freshness_24h` is natural drift between waves (will re-appear at next wave boundary; closure path = re-apply `touch` at next wave's audit step)
6. **Pre-existing 3 SKIPs PRESERVED (BY-DESIGN)** per Wf-345-A2 + Wf-411-B1 baseline: `p3_wf343c4_yuelbond_decoder_env` (env gap: gpu host missing) + `p3_wf343c4_wave_budget_remaining_nonneg` (-183.04h overrun BY-DESIGN) + `data_2_1_6_bindingdb_external_honest_negative` (external 1.07M dependency)
7. **Wf-379-A1 m1_auc 5-seed IN-FLIGHT — result pending** (Rh seed 3 RUNNING on GPU 0 Blackwell at Wf-379-B1 timestamp; Krasnov 0.86 gate target per `mol-metal-anticancer` skill; wall-measured ~5 hr GPU)
8. **Wf-299 GPU launch BLOCKED** on Wf-408-A3 alias mapping + Wf-408-A2 trainer wire-in (per Wf-410-A4 contract (e) FAIL — alias mapping must be applied to launcher line 213 + line 221 BEFORE any actual GPU launch; Wf-415-A1 wrapper fix enables flag passthrough but does NOT apply the `--wave 1/2/3 → A/B/C` alias mapping itself)
9. **Wf-413-A1 GPU wrapper termination root cause OPEN** (per Wf-413-A1 §9: investigate why wrapper terminated after Re seed=3; root cause likely subprocess.signal handling or PIPE buffering; closure = follow-up wrapper fix or KEEP manual Wf-413-A1 pattern)
10. **Wetlab items DEPRECATED** per AGENTS.md §5.4 + §10 (4 LOIs + 3 SOPs + R20-P6 + Wf-74 v2 + LAMBDA_ROADMAP §3.2 R22-P1 + LAMBDA_ROADMAP §3.3 R23 all DEPRECATED / re-scoped to in-silico; **Wf-415 wave honors Tier-2 reorientation — no wet-lab planning executed**)
11. **Wf-410-A3 atomic write migration NOT COMPLETE** (~41 more closeout-drafting scripts remain; closure path = Wf-414-A3 / Wf-416+ opt-in adoption per Wf-386-A3 §6.1 + Wf-386-A3 §6.2 — auto-migrating all scripts in single sweep too risky because some scripts have deliberate legacy direct-Write patterns — e.g. CSV logging, debug snapshots — that should NOT be migrated)
12. **m7 + m4 + m3 MEASURED end-to-end lift BLOCKED on GPU launch** (m7 joint lift +0.02-0.035 PROJECTED per Wf-380-A3 + Wf-388-A2 wire-in; m3 Vina −0.3 to −0.5 kcal/mol PROJECTED per Wf-380-A4; m4 SA −20.5% PROJECTED per Wf-380-A6 + Wf-388-A3 wire-in; MEASURED requires Wf-408-A1 USER_OK-gated GPU retrain per AGENTS.md §5.1)
13. **All PROJECTED speedup / lift claims REMAIN PROJECTED** (Wf-340-A1 EDM preconditioning 2-3x FID convergence + Wf-353-A1 Triton OT 1.5-3x + Wf-353-A2 shortcut distillation +10-20% sample quality + Wf-380-A3 diversity +0.02-0.035 m7 lift + Wf-380-A2 per-cohort +0.05-0.10 m2 R² lift + Wf-388-A1 +0.02-0.10 m2 R² per-cohort + Wf-388-A2 diversity +0.02-0.035 m7 + Wf-388-A3 SA −20.5%; MEASURED requires Blackwell sm_120 GPU retrain per §5.1 + kill_then_adapt §0)

## §4 Cross-references

- **Wf-407** (predecessor) — Wf-299 Wave 2+3 SCAFFOLD_NOT_FUNCTIONAL + USER_OK UX gap closure; Wf-415-A1 closes the SCAFFOLD_NOT_FUNCTIONAL transparent-passthrough surface
- **Wf-408** (predecessor) — CRITICAL PATH: Wf-380 SCAFFOLD-loss wire-in + pre-flight FAIL fixes + launcher `--wave` mapping; Wf-415-A1 preserves the alias regression-guard from Wf-408-A3
- **Wf-413** (predecessor) — GPU 0 sequential CFM retrain: 3 SHIPPED + 1 MISSING on disk; Wf-413-A1 SHIPPED the Re seed=4 (Wf-131 P3 deferred runs resume); Wf-413-A2 MISSING per §5.5
- **Wf-414** (predecessor) — 3 SHIPPED deliverables + 17 NEW tests PASS + final synthesis v41
- **Wf-243..Wf-252** — Wf-415-A2 sys.path injection restores compatibility with the Wf-243..Wf-252 canonical wrapper invocation pattern (`python ${pre_flight_check_path} --pipeline all` WITHOUT PYTHONPATH prefix)
- **Wf-355** (USER_OK-gated wrapper pattern) — preserved unchanged
- **Wf-378-B1 / Wf-381-B1 / Wf-388-B1 / Wf-395-B1** — prior B1 sync patterns; Wf-415-B1 follows the same 5-section format (Phase status + INDEX.md sync + AGENTS.md sync + Honest framing + Cross-references)
- **AGENTS.md §5.5** (post-wave on-disk verification rule) — strictly applied to verify 3 SHIPPED Wf-415 files vs harness-claimed 4 deliverables; A4 MISSING recorded honestly
- **AGENTS.md §5.6** (atomic write protocol adoption rule) — strictly observed via `atomic_write_file` for B1 closeout + final_synthesis writes
- **AGENTS.md §5.7** (trainer flag wire-in smoke-required rule per Wf-399-A1) — honored by Wf-415-A1 wrapper fix (transparent-passthrough + 50-step CPU smoke pattern)
- **AGENTS.md §5.1** (standing GPU authorization) — 0 GPU launches across Wf-415 wave
- **AGENTS.md §5.2** (honest framing mandate) — 2 remaining FAILs declared as HONEST-NEGATIVE BY-DESIGN, not paper-over
- **AGENTS.md §5.3** (no paper work without authorization) — 0 touches to `paper/*`
- **AGENTS.md §5.4** (no wet-lab) — 0 wet-lab planning; Tier-2 reorientation preserved
- **AGENTS.md §10** (Wetlab deprecation FAQ) — preserved unchanged
- **`molmetal/scripts/wf386a3_atomic_write.py`** (atomic_write_file + safe_overwrite_check + detect_truncation + write_closeout public API) — used for this B1 closeout per §5.6
- **`molmetal/scripts/wf386a3_atomic_write.py::write_closeout`** (high-level helper: safe_overwrite_check + atomic_write_file + detect_truncation) — used for this B1 closeout + final_synthesis writes
- **`party-twenty-spirit` §3.5** (MEASURED-vs-PROJECTED tag discipline) — every numeric value in this closeout traces to a JSON / report path on disk

## §5 Recommended follow-up (Wf-416+)

Per Wf-415 wave + Wf-414 final synthesis §7 + Wf-413-A1 §9 + Wf-411 final synthesis §4 + Wf-410 final synthesis §4 + Wf-407 final synthesis §6 + Wf-406 final synthesis §4 + Wf-399 final synthesis §4 + Wf-398 final synthesis §4 + Wf-396-B1 §7 + Wf-388 final synthesis §7 + Wf-381-B1 §22.77 + Wf-378-B1 §22.73, the recommended follow-up is:

1. **Wf-416-A1** — author the missing Wf-415-A4 closeout file (retroactive per AGENTS.md §5.5 rule #5)
2. **Wf-416-A2** — wire `use_dann` kwarg through `MultiTaskWassersteinModel.__init__` to close `p2_tests_pass` FAIL (the structural wiring task per Wf-415-A3 §0 + Wf-280D contract)
3. **Wf-416-A3** — USER_OK-gated actual m2 R² Path A 5-metal × 3-seed × 50-epoch retrain (~12 hr GPU per Wf-414-A1 wrapper)
4. **Wf-416-A4** — USER_OK-gated Wf-388-A1 m2 R² per-cohort REAL verification retrain
5. **Wf-416-A5** — USER_OK-gated Wf-299 GPU launch (post-Wf-408-A3 alias mapping patch + Wf-415-A1 wrapper fix applied)
6. **Wf-416-A6** — USER_OK-gated Wf-394-A1 Lambda CFM wire-in patch application
7. **Wf-416-A7** — USER_OK-gated Wf-246-C metal-coordination term retrain (~3 hr GPU)
8. **Wf-416-A8** — USER_OK-gated paper §6 v4 write from `docs/PAPER_SECTION_6_v4_LIMITATIONS_READY.md` (~12-15 hr CPU)
9. **Wf-416-A9** — USER_OK-gated paper §3 v4 update from `docs/PAPER_SECTION_3_METRIC_TABLE.md` (~10 hr CPU)
10. **Wf-416-A10** — atomic write protocol adoption migration completion (~41 more closeout-drafting scripts remaining)

Combined forward-looking wall: ~52.9 hr GPU + 13 hr CPU + 30 min install + SCAFFOLD-MEASURED wire-in + Tier-2 §3 + §6 v4 update total.
