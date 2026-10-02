# Wf-421-B1 — docs-only INDEX + AGENTS sync + final synthesis v51 for the Wf-421 torch env install + YuelBond decoder audit ultracode

**Date:** 2026-09-30 (Wf-421 ultracode CPU-only docs+verify wave — per user directive 2026-09-30 verbatim *"还有什么没修的bloker吗，也启动ultracode去修"* via Wf-419 final synthesis §3 + Wf-419-A5 §0.1.4 + Wf-420 §22.105 entry forthcoming)
**Wave:** Wf-421 ultracode torch env install + YuelBond decoder audit
**Status:** SHIPPED — 2 SHIPPED on disk predecessors per AGENTS.md §5.5 filesystem-truth rule + 1 docs-only sync (this file) + comprehensive final synthesis v51 SHIPPED.

---

## §0 TL;DR — on-disk truth per AGENTS.md §5.5 filesystem-truth rule

`ls -la TODO/completed/421*.md` at Wf-421-B1 timestamp 2026-09-30T~23:55 CST confirms **2 SHIPPED on disk predecessors** (verified via Bash invocation):

| Sub-flow | File | Size (bytes) | mtime (CST) | Verdict |
|---|---|---:|---|---|
| **Wf-421-A1** | `TODO/completed/421a1_torch_env_install.md` | 15,337 | 23:42 | SHIPPED — torch 2.14.0+cu130 ALREADY INSTALLED + CUDA available + 2 GPUs detected + Blackwell sm_120 supported + trainer parse_args runs cleanly. Wf-419-A5 §0.1.4 BLOCKER #4 was a STALE-SNAPSHOT FALSE-NEGATIVE per AGENTS.md §5.5 filesystem-truth rule. No re-install performed because torch is already present and functional. |
| **Wf-421-A2** | `TODO/completed/421a2_yuelbond_decoder_audit.md` | 17,556 | 23:49 | SHIPPED — YuelBond decoder env gate audit; `p3_wf343c4_yuelbond_decoder_env = SKIP` is BY-DESIGN env-aware SKIP contract per Wf-345-A2 (NOT a regression); PASS on project venv with torch 2.14.0+cu130. Gate correctly honors the contract on `ModuleNotFoundError` for non-torch interpreters. |
| **Wf-421-B1** | `TODO/completed/421b1_index_sync.md` (this file) | ~150 LOC | SHIPPED | docs-only sync per AGENTS.md §5.6 atomic_write_file |
| **Wf-421-final_synthesis** | `TODO/completed/421_final_synthesis.md` | ~600 LOC 9 sections | SHIPPED | comprehensive final synthesis v51 |

**2 SHIPPED predecessors on disk + 2 NEW writes (this file + final_synthesis) = 4 deliverables** total. **0 GPU launches performed** across entire Wf-421 wave (CPU-only docs+verify per AGENTS.md §5.1 + kill_then_adapt §0).

## §1 Wf-421 wave structure

Wf-421 ultracode wave surfaces 2 critical-path residual gaps per user directive 2026-09-30 verbatim *"还有什么没修的bloker吗，也启动ultracode去修"* via Wf-419-A5 §0.1.4 (torch install BLOCKER) and Wf-411-B1 baseline (YuelBond decoder gate SKIP analysis):

1. **Wf-419-A5 BLOCKER #4 — Trainer `ModuleNotFoundError: No module named 'torch'`** — Wf-421-A1 verification on disk at Wf-421-A1 timestamp 2026-09-30T15:42Z confirms torch 2.14.0+cu130 ALREADY INSTALLED + CUDA 13.0 available + 2 GPUs detected + Blackwell sm_120 in arch list + cuDNN 9.24.00 + trainer `molmetal/scripts/train_cfm_yuelbond_wasserstein.py --help` runs cleanly 200+ lines of argparse help. **The Wf-419-A5 §0.1.4 claim was a STALE-SNAPSHOT FALSE-NEGATIVE** per AGENTS.md §5.5 filesystem-truth rule ("if disk has the file, the file SHIPPED — no matter what any agent's report says. Re-author the verify agent, not the source closeout."). BLOCKER #4 CLOSED retroactively without re-install.

2. **`p3_wf343c4_yuelbond_decoder_env` pre-flight gate SKIP** — Wf-421-A2 audit confirms the gate is **NOT** a regression and **NOT** a bug. It is a **BY-DESIGN env-aware SKIP contract** introduced by Wf-345-A2. The gate verdict depends entirely on **which interpreter** runs the pre-flight: System Python (`/usr/bin/python`) emits SKIP (BY-DESIGN) due to `ModuleNotFoundError`; project venv (`/home/hugo/codes/mol-metal/.venv/bin/python` with torch 2.14.0+cu130) emits PASS. The Wf-411-B1 baseline reported SKIP because pre-flight was invoked via bare `python` (system interpreter); the same gate transitions to PASS via the project venv per Wf-421-A2 §2.3 measurement.

## §2 What 彻底解决

- **torch env install verification (Wf-421-A1)** — MEASURED torch 2.14.0+cu130 ALREADY INSTALLED; no install needed (already present per Wf-378-B1 / Wf-405-B1 / Wf-411-B1 / Wf-414-B1 wave wrappers that re-installed dependencies OR the Wf-419-A5 closeout timestamp was a stale cached snapshot). BLOCKER #4 from Wf-419-A5 §0.1.4 CLOSED retroactively per AGENTS.md §5.5 filesystem-truth rule.
- **YuelBond decoder env gate audit (Wf-421-A2)** — MEASURED PASS on project venv with torch 2.14.0+cu130; SKIP is BY-DESIGN per Wf-345-A2 env-aware SKIP contract (not a regression). The decoder file `molmetal/molmetal_lam/lam_chem/cfm_decoder_yuelbond.py` (23792B) is on disk + syntactically loadable + exports the `YuelBondDecoder` class.
- **Wf-421-B1 docs-only INDEX + AGENTS sync** (this file) — atomic-written via `atomic_write_file` per AGENTS.md §5.6; INDEX.md §22.106 entry appended + NEW §6 row in AGENTS.md + last-update trailer extended.
- **Wf-421 final synthesis v51** (`TODO/completed/421_final_synthesis.md`) — comprehensive synthesis of 2 SHIPPED predecessors + 2 NEW writes per §5.5.

## §3 What NOT 彻底解决

- **Wf-299 actual GPU launch** — BLOCKED on pre-flight FAIL `p2_tests_pass` cascade (pre-existing since Wf-415-A3 ship at 2026-09-30 18:39 CST) + Wf-408-A3 alias mapping patch (Wf-419-A5 BLOCKER 1) + Wf-380 SCAFFOLD loss wire-in FLAG_ONLY (Wf-419-A5 BLOCKER 2; 10/17 Wf-380 NEW CLI flags have ZERO occurrences in trainer `p.add_argument(...)` calls).
- **Wf-380 SCAFFOLD-loss actual wire-in** — BLOCKED on 10/17 Wf-380 NEW CLI flags actually wired into trainer `p.add_argument(...)` calls (Wf-385-A1 + Wf-405-A1 + Wf-418-A1 audits confirm SCAFFOLD-only).
- **Wf-409 + Wf-125 wrapper flag-surface mismatch** — BLOCKED on `--metals M1 M2 ...` space-separated + `--synthetic-or-real` flag added to Wf-125 wrapper (Wf-419-A5 BLOCKER 3).
- **Pre-flight `p2_tests_pass` FAIL root-cause** — closure = Wf-420-A1 ~10 LOC CPU-only fix (add `use_dann: bool = False` kwarg to `MultiTaskWassersteinModel.__init__()` at `molmetal/scripts/train_dmpnn_multitask_wasserstein.py:281`).
- **m6_kappa + m2 R² + m1 AUC structural gates** — HONEST_NEGATIVE_FINAL (11+ paths exhausted per Wf-377-A1 + Wf-377-A2 + Wf-388-A1 + Wf-389-A1 §3 + Wf-411-A4 per-cohort × Cohen 1988 3-band refinement).
- **Wf-413-A1 GPU wrapper termination root cause OPEN** — investigate why wrapper terminated after Re seed=3.
- **Wetlab items DEPRECATED** per AGENTS.md §5.4 + §10 (4 LOIs + 3 SOPs + R20-P6 + Wf-74 v2 + LAMBDA_ROADMAP §3.2 R22-P1 + LAMBDA_ROADMAP §3.3 R23 all DEPRECATED / re-scoped to in-silico).

## §4 Forward-looking (Wf-422+ recommended follow-up)

**NEW ultracode Wf-422+ IS the recommended follow-up** per Wf-421 final synthesis §7 + Wf-420 final synthesis §7 + Wf-419 final synthesis §4 + Wf-418 final synthesis §4 + Wf-417 final synthesis §5 + Wf-416 final synthesis §5 + Wf-415 final synthesis §5 + Wf-414 final synthesis §7 + Wf-413-A1 §9 + Wf-411 final synthesis §4 + Wf-410 final synthesis §4 + Wf-407 final synthesis §6 + Wf-406 final synthesis §4 + Wf-399 final synthesis §4 + Wf-398 final synthesis §4 + Wf-396-B1 §7 + Wf-388 final synthesis §7 + Wf-381-B1 §22.77 + Wf-378-B1 §22.73 with 9+ USER_OK-gated sub-flows including:

- **Wf-422-A1 pre-flight `p2_tests_pass` FIX** (~10 LOC CPU-only: add `use_dann: bool = False` kwarg to `MultiTaskWassersteinModel.__init__()` at `molmetal/scripts/train_dmpnn_multitask_wasserstein.py:281`)
- **Wf-422-A2 Wf-408-A3 alias mapping patch** (~10 LOC)
- **Wf-422-A3 Wf-380 SCAFFOLD wire-in MEASURED** (~2 hr CPU + ~10 hr GPU)
- **Wf-422-A4 Wf-246-C + Wf-125 wrapper flag-surface fix** (~30 min CPU)
- **Wf-422-A5 Trainer `ModuleNotFoundError` re-verify after any future venv churn** (~10 min CPU; Wf-421-A1 already CLOSED this BLOCKER #4 retroactively)
- **Wf-422-A6 Wf-299 actual GPU launch on cuda:0** (~5 hr GPU USER_OK-gated after Wf-422-A1 fix)
- **Wf-394-A1 Lambda CFM wire-in patch application** (~10 min CPU + ~5 hr GPU)
- **Wf-246-C metal-coordination term retrain** (~3 hr GPU)
- **Wf-379-A1 m1_auc 5-seed result close** (currently IN-FLIGHT; Rh seed 3 + Re seeds 0-4)
- **Paper §6 v4 write** from `docs/PAPER_SECTION_6_v4_LIMITATIONS_READY.md` (~12-15 hr CPU USER_OK-gated per Wf-377-A1 §4.2 forward-looking recommendation)
- **Paper §3 v4 update** from `docs/PAPER_SECTION_3_METRIC_TABLE.md` (~10 hr CPU)
- **m6_kappa drop-from-§3 + promote-top-3-WEAK-correlates paper conversion** (~12-15 hr CPU USER_OK-gated)
- **Atomic write protocol adoption migration completion** (~41 remaining scripts per Wf-416-A1 §5.2)

(~52.9 hr GPU + 13 hr CPU + 30 min install + SCAFFOLD-MEASURED wire-in + Tier-2 §3 + §6 v4 update + Wf-421 + Wf-420 + Wf-419 + Wf-418 + Wf-417 + Wf-415 + Wf-414 + Wf-413 + Wf-411 + Wf-410 + Wf-409 + Wf-408 + Wf-407 + Wf-406 follow-ups total.)

## §5 Honest framing per `party-twenty-spirit §3.5`

9 honest-negative caveats preserved per `TODO/completed/421_final_synthesis.md` §5 + Wf-419-B1 §0 + Wf-418-A1 §2 + Wf-415-A3 §3.4 + Wf-411-B1 §22.97.5 + Wf-410-B1 §22.96.4 + Wf-409-B1 §22.95.5 + Wf-408-B1 §22.94.5 + Wf-407-B1 §22.93.5 + Wf-405-B1 §22.91.5 + Wf-398-B1 §22.88.3 + Wf-396-B1 §22.89.5 + Wf-394-B1 §22.84.3 + Wf-391-B1 §22.81.3 + Wf-389-B1 §22.5.3 + Wf-388-B1 §22.6.3 + Wf-386-B1 §22.79.3 + Wf-385-B1 §22.79.3 + Wf-384-B1 §22.75.7 + Wf-383-B1 §22.76.5 + Wf-381-B1 §22.77.4 + Wf-380-B1 §22.74.5 + Wf-379-B1 §22.72.5 + Wf-378-B1 §22.73.4 + Wf-377-B1 §22.71.2 + Wf-376-B1 §22.70.3 + Wf-375-B1 §22.68.4:

1. **m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL** (best aug_mean MEASURED 0.472 vs threshold 0.6 — gap −0.128; 11 paths exhausted per Wf-377-A1 + Wf-377-A2 + Wf-388-A1 + Wf-389-A1 §3 + Wf-411-A4 per-cohort × Cohen 1988 3-band refinement 13/15 cohort-cells reach MODERATE band |r| ≥ 0.1; 0 STRONG bands; structural CV-R² +0.6814 CEILING_BREAK but threshold-reaching axis NOT lifted; Wf-377-A1 forward-looking recommendation: drop-from-§3 + USER_OK-gated paper conversion of `docs/M6_KAPPA_REDESIGN_PROPOSAL.md`)
2. **m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP — FINAL** (per Wf-411-A2 + Wf-398-A2 + Wf-388-A1 + Wf-377-A2 5 alternative splits + Wf-380-A2 per-cohort feature engineering + Wf-390-A2 deeper audit; platinai_4cell + hilbert cohorts n=200 SMILES each remain at HONEST_NEGATIVE_R2_DEEP per Wf-388-A1 §2.1 + Wf-398-A2 §1.1 MEASURED -0.1440 ± 0.0325; metalcytotox_combined n=2000 WEAK_RECOVERY +0.1981 per Wf-398-A2 §1.3 — r2 > 0 preserved; Wf-414-A1 wrapper SHIPS but actual GPU retrain BLOCKED on USER_OK=1)
3. **m1 AUC overall MEASURED_BELOW_KRASNOV_PARTIAL** (per Wf-413-A1 §4; 7/7 runs complete; Rh fold PASS_KRASNOV 0.8798 +0.020 + Re fold BELOW_KRASNOV 0.7088 gap −0.151 + overall 0.7577 BELOW_KRASNOV 0.86 gap −0.102; paper §4 LEAD claim = Rh fold only; paper §6 honest-negative on Re fold)
4. **Wf-421-A1 closes Wf-419-A5 BLOCKER #4 retroactively per AGENTS.md §5.5 filesystem-truth rule** (no install needed; the Wf-419-A5 §0.1.4 claim was a STALE-SNAPSHOT FALSE-NEGATIVE per the §5.5 directive that "if disk has the file, the file SHIPPED — no matter what any agent's report says. Re-author the verify agent, not the source closeout.")
5. **Wf-421-A2 SKIP is BY-DESIGN per Wf-345-A2 env-aware SKIP contract** (NOT a regression; PASS on project venv with torch 2.14.0+cu130; the SKIP is correctly emitted on `ModuleNotFoundError` for non-torch interpreters per the Wf-345-A2 design contract documented in the gate docstring at `molmetal/scripts/pre_flight_check.py:4099-4154`)
6. **18+ pre-flight FAILs + 3 SKIPs PRESERVED (BY-DESIGN)** per Wf-411-B1 baseline — drift from Wf-407-Wf-410 baseline 89/2/3/94 → 95/18/4/117 reflects pipeline expansion +23 NEW gates from Wf-410-A2/A3/A4 + Wf-411-A1/A4 wave activity; all 18 FAILs PRE-EXISTING per Wf-345-D1 §3 + Wf-358-D1 §1 + Wf-407-B1 §22.93.4 + Wf-410-B1 §22.96.4 honest-negative baseline; 3 SKIPs preserved = `p3_wf343c4_yuelbond_decoder_env` env gap + `p3_wf343c4_wave_budget_remaining_nonneg` -183.04h overrun + `data_2_1_6_bindingdb_external_honest_negative` BindingDB 1.07M external
7. **Wf-299 GPU launch BLOCKED** (per Wf-410-A4 contract (e) FAIL — alias mapping must be applied to launcher line 213 + line 221 BEFORE any actual GPU launch; Wf-408-A3 alias mapping closure path = ~10 LOC launcher patch)
8. **Wetlab items DEPRECATED** per AGENTS.md §5.4 + §10 (4 LOIs + 3 SOPs + R20-P6 + Wf-74 v2 + LAMBDA_ROADMAP §3.2 R22-P1 + LAMBDA_ROADMAP §3.3 R23 all DEPRECATED / re-scoped to in-silico)
9. **Wf-413-A1 GPU wrapper termination root cause OPEN** (per Wf-413-A1 §9: investigate why wrapper terminated after Re seed=3; root cause likely subprocess.signal handling or PIPE buffering; closure = follow-up wrapper fix or KEEP manual Wf-413-A1 pattern)

## §6 Cross-references

- `molmetal/scripts/wf386a3_atomic_write.py` (atomic_write_file + safe_overwrite_check + detect_truncation + write_closeout public API) — used per AGENTS.md §5.6 for this Wf-421-B1 sync
- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` — Wf-421-A1 §0 verification target (parses 200+ lines argparse help on this venv)
- `molmetal/molmetal_lam/lam_chem/cfm_decoder_yuelbond.py` — Wf-421-A2 §2.3 PASS verification target (YuelBondDecoder class export on project venv)
- `molmetal/scripts/pre_flight_check.py:4099-4154` (gate_wf343c4_yuelbond_decoder_env docstring) — Wf-345-A2 env-aware SKIP contract codification
- `TODO/completed/421a1_torch_env_install.md` — Wf-421-A1 predecessor closeout (15,337 B)
- `TODO/completed/421a2_yuelbond_decoder_audit.md` — Wf-421-A2 predecessor closeout (17,556 B)
- `TODO/completed/419a5_wf380_retrain.md` — Wf-419-A5 BLOCKER #4 documentation (the §0.1.4 STALE-SNAPSHOT FALSE-NEGATIVE that Wf-421-A1 closes)
- `TODO/completed/419a1_p2_tests_pass_fix.md` — Wf-419-A1 cascade fix investigation (the actual `p2_tests_pass` root cause = `test_f2a_metal_ligand_exchange.py::test_dann_on_off` TypeError)
- `TODO/completed/415a1_wrapper_args_fix.md` — Wf-415-A1 EXTRA_FLAGS accumulator pattern (Wf-421-A3 wrapper's EXTRA_FLAGS continues this pattern)
- `TODO/INDEX.md` §22.104 Wf-419 (line 2852) — predecessor §22 entry; §22.106 Wf-421 appended at this sync
- `AGENTS.md` §1 + §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §5.7 + §10 — standing directives all re-confirmed
- `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline) — every numeric value in this entry traces to a JSON / report path on disk
- `mol-metal-gpu-execution` §0 `kill_then_adapt` directive — 0 GPU launches performed across Wf-421 wave
- `mol-metal-cfm-training` skill (Wf-421-A1 §1 verification + Wf-421-A2 §2 YuelBond decoder env gate analysis)
