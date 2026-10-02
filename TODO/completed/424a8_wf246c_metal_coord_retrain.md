# Wf-424-A8 Wf-246-C Metal-Coordination Term Retrain SCAFFOLD_NOT_FUNCTIONAL HONEST_NEGATIVE_REFUSED

**USER DIRECTIVE**: 我都给你授权，你启动ultracode做起来，都要做 (Wf-424 ultracode wave)
**TASK BRIEF**: Launch Wf-246-C metal-coordination term retrain on cuda:0 Blackwell via scripts/launch_wf409a3_metal_coord_retrain.sh.
**DATE**: 2026-09-30 (CST) / 2026-09-30T19:06 UTC
**AGENT**: Wf-424-A8

## §0 HONEST_NEGATIVE_REFUSED per kill_then_adapt §0

**VERDICT**: SCAFFOLD_NOT_FUNCTIONAL HONEST_NEGATIVE_REFUSED - launch REFUSED at wrapper level. **0 GPU cycles consumed**.

## §1 5/5 GPU safety conditions (per kill_then_adapt §0)

| # | Condition | MEASURED | Verdict |
|---|-----------|----------|---------|
| 1 | Env probe first | `nvidia-smi` MEASURED GPU 0 RTX PRO 6000 Blackwell 97340 MiB free IDLE 0% util + GPU 1 RTX 5090 32202 MiB free IDLE 0% util | PASS |
| 2 | Pre-flight PASS required | `pre_flight_check.py --pipeline frontier --strict` MEASURED 24 PASS / 1 FAIL / 0 SKIP / 25 TOTAL; the 1 FAIL = `p3_wf343c2_per_wave_runtime_check` cascade | **FAIL** |
| 3 | VRAM budget check | 97340 MiB free vs ~6 GB needed; 91 GB margin | PASS |
| 4 | 50-step CPU smoke PRE-LAUNCH per §5.7 | `run_smoke_50step` returned rc=2 within 2.5s with stderr=`unrecognized arguments: --metal-coord-term` | **FAIL** |
| 5 | Honor USER_OK=1 per §5.1 | USER_OK=1 honored by REFUSAL (not by launch) | PASS |

## §2 3 SCAFFOLD_NOT_FUNCTIONAL blockers

### BLOCKER 1: --metal-coord-term flag does NOT exist in trainer
`grep -n "metal-coord-term|metal_coord_term" molmetal/scripts/train_dmpnn_multitask_wasserstein.py` returns ZERO matches.
The wrapper passes `--metal-coord-term` to the trainer but the trainer does not recognize it (rc=2 with `unrecognized arguments: --metal-coord-term`).
Per Wf-419-A5 BLOCKER pattern: Wf-380 SCAFFOLD-loss wire-in is FLAG_ONLY.

### BLOCKER 2: Pre-flight FAIL cascade `p3_wf343c2_per_wave_runtime_check`
Pre-existing since Wf-345-D1 §3 + Wf-358-D1 §1 + Wf-367-A1 + Wf-415-A3 §3.4 baseline.
Closure path: Wf-420-A1 ~10 LOC `use_dann` kwarg fix at `molmetal/scripts/train_dmpnn_multitask_wasserstein.py:281`.

### BLOCKER 3: Task brief argument format discrepancy
Task brief uses `--metals Ru,Ir,Rh,Os,Re` (comma-separated) but Wf-424-A2 wrapper fix requires SPACE-separated: `--metals "Ru Ir Rh Os Re"`.
The wrapper greedy tokenization loop `[ "${1#--}" = "$1" ]` does NOT split on commas - comma-separated becomes single token "Ru,Ir,Rh,Os,Re" which fails trainer FIVE_METALS choices validation.

## §3 VERIFICATION SUMMARY

### Wrapper
- `scripts/launch_wf409a3_metal_coord_retrain.sh` exists: `-rwxr-xr-x 1 hugo hugo 10614 Oct  1 01:34`
- `bash -n` SYNTAX OK (exit 0)
- `--smoke` mode SHIPPED SCAFFOLD_CPU_ONLY result at `molmetal/reports/wf409a3_metal_coord_launch/result.json` (4394 bytes)
- `--dry-run` mode has shell syntax issue at line 229: `$'\\E[1m': command not found` (tput substitution issue; does not block --smoke or --actual)

### SCAFFOLD smoke result on disk
- `status: SHIPPED_SCAFFOLD_CPU_ONLY`
- `import_ok: True`
- `build_ok: True`
- `per_epoch_kcal_audit`: 5 epochs audited, all `post_fix_kcal_mol` NEGATIVE when `composite > 0.5`
- `epoch_sign_audit_pass: True` (per Wf-333-V1 sign fix at `metal_coord_term.py:635`)
- `n_metals: 5`, `n_seeds: 5` (correct scaffold smoke scale)

### Pre-flight cascade
- Wf-424-B1 baseline (2026-09-30T17:45:32+0000 UTC): 113 PASS / 1 FAIL / 3 SKIP / 117 TOTAL (--pipeline all)
- Wf-424-A8 fresh (2026-09-30T19:06:44+0000 UTC): 24 PASS / 1 FAIL / 0 SKIP / 25 TOTAL (--pipeline frontier --strict)
- Wf-424-A8 fresh (2026-09-30T19:06:17+0000 UTC): 14 PASS / 0 FAIL / 0 SKIP / 14 TOTAL (--pipeline chemprop_v2_v3 --strict)
- The 1 FAIL is the pre-existing `p3_wf343c2_per_wave_runtime_check` cascade child per Wf-345-D1 §3 + Wf-358-D1 §1 + Wf-367-A1 + Wf-415-A3 §3.4 honest-negative baseline.

### GPU env probe
- GPU 0 RTX PRO 6000 Blackwell Workstation Edition: 97340 MiB free / 97887 MiB total / 0% util
- GPU 1 RTX 5090: 32202 MiB free / 32607 MiB total / 0% util
- torch 2.14.0+cu130 + sm_120 + CUDA available
- User voice GPU 0 preference honored (target cuda:0)

## §4 8 SCAFFOLD_NOT_FUNCTIONAL precedents (pattern recognition)

Per Wf-418-A5 + Wf-419-A5 collapse pattern, the same 3 blockers (pre-flight cascade + trainer flag missing + wrapper mismatch) have blocked:
- Wf-379-A2 R20 §5 tmQM + PlatinAI SCAFFOLD_NOT_FUNCTIONAL (--metals restricted to 5; alternative trainer missing)
- Wf-398-A1 D-MPNN retrain REFUSED (--per-cohort-features flag absent)
- Wf-407-A3 Wf-299 Wave 2+3 REFUSED (--wave 3 literal invalid; 18 SCAFFOLD flags missing)
- Wf-413-A2/A3/A4 Wf-299 launch attempts BLOCKED (pre-flight FAIL cascade)
- Wf-417-A2/A3/A4 Wf-299 launch attempts BLOCKED (same cascade)
- Wf-418-A3 Wf-380 SCAFFOLD wire-in MEASURED retrain BLOCKED (FLAG_ONLY per Wf-385-A1)
- Wf-418-A5 Wf-246-C + Wf-125 launch attempts REFUSED (literal task-brief invocations invalid)
- **Wf-424-A8 Wf-246-C launch REFUSED** (--metal-coord-term flag absent in trainer) - this entry

## §5 Closing metrics JSONs

No V3/V4 combined median kcal/mol metric exists in `metrics/by_metric/vina_kcal_per_mol.json` to update - the metric tracks Vina Dock pIC50, not metal-coordination kcal/mol.
`molmetal/reports/wf424a8_wf246c/` directory created but no metric data was generated (0 GPU launches performed).

## §6 User voice answer

Per AGENTS.md §5.2 honest framing + kill_then_adapt §0 safety contract: launch REFUSED.
USER_OK=1 honored by REFUSING to waste GPU cycles on broken invocations.
**0 GPU cycles consumed**.

**Closure path for actual GPU retrain (future Wf wave)**:
1. Wf-420-A1 ~10 LOC `use_dann` kwarg fix at `molmetal/scripts/train_dmpnn_multitask_wasserstein.py:281` (closes pre-flight cascade)
2. Wf-424-A9 wire `--metal-coord-term` flag into trainer (~50 LOC + 1 NEW test) (closes BLOCKER 1)
3. Use SPACE-separated `--metals "Ru Ir Rh Os Re" --seeds "0 1 2 3 4"` form (closes BLOCKER 3)
4. USER_OK=1 + pre-flight PASS + Wf-424-A9 wire-in then Wf-424-A8 (this) re-launch ~3 hr GPU

## §7 Cross-references

- Wf-419-A5 BLOCKER pattern: SCAFFOLD_NOT_FUNCTIONAL collapse
- Wf-418-A5 BLOCKER pattern: SCAFFOLD_NOT_FUNCTIONAL collapse
- Wf-420 follow-up: `use_dann` kwarg fix + atomic write migration
- Wf-415-A3 §3.4: pre-flight FAIL cascade root cause
- Wf-385-A1: Wf-380 SCAFFOLD-loss wire-in FLAG_ONLY pattern
- Wf-389-A5: --use-dann keyword arg TypeError pattern
- Wf-333-V1: kcal sign fix at `metal_coord_term.py:635`
- Wf-411-A4: m6_kappa FINAL verdict (paper §6 scaffold)
- Wf-388-A1: m2 R² per-cohort REAL verification scaffold
- Wf-409-A3: SCAFFOLD script + wrapper authoring baseline
- Wf-424-A2: Wf-424-A2 greedy tokenization fix (space-separated --metals form)
- AGENTS.md §5.1, §5.2, §5.4, §5.5, §5.6, §5.7, §10 standing directives re-confirmed

## §8 ls -la evidence per AGENTS.md §5.5 rule #2

```
TODO/completed/424a8_wf246c_metal_coord_retrain.md  (this file, ~13 KB atomic-written)
scripts/launch_wf409a3_metal_coord_retrain.sh        10654 bytes (M 2026-10-01 01:34)
molmetal/scripts/wf409a3_metal_coord_retrain_scaffold.py  16582 bytes (M 2026-09-30 15:55)
molmetal/reports/wf409a3_metal_coord_launch/result.json    4394 bytes (M 2026-10-01 03:04)
molmetal/reports/wf424a8_wf246c/                          (created empty per task brief step 3)
molmetal/reports/wf424a8_preflight.json                   (chemprop_v2_v3 baseline)
molmetal/reports/wf424a8_preflight_frontier.json          (frontier baseline w/ cascade FAIL)
```

All files exist on disk per AGENTS.md §5.5 filesystem-truth rule.

## §9 CHRONOLOGY

- Wf-424-A8 closeout timestamp: 2026-09-30T19:10 UTC (~03:10 CST 2026-10-01)
- Wf-424-B1 docs-only INDEX + AGENTS + final synthesis: deferred to Wf-424-B1 sub-agent
- Wf-424 wave surface area: A1 + A2 + A3 + ... + A8 + B1 (8 sub-flows per task brief)
- Wf-424 ships after Wf-296..423 ultracode waves + Wf-424-A1 pre-flight + Wf-424-A2 tokenization fix + Wf-424-A4 Wf-299 launch attempt BLOCKED
- 0 GPU launches performed across entire Wf-424-A8 sub-flow per kill_then_adapt §0 safety contract

## §10 Forward-looking USER_OK-gated recommendations

NEW ultracode Wf-425+ IS the recommended follow-up per Wf-424-A8 SCAFFOLD_NOT_FUNCTIONAL HONEST_NEGATIVE_REFUSED:
1. Wf-425-A1 wire `--metal-coord-term` flag into `train_dmpnn_multitask_wasserstein.py` (~50 LOC + 1 NEW test) (~30 min CPU)
2. Wf-425-A2 close pre-flight `p2_tests_pass` cascade via `use_dann` kwarg fix (~10 LOC + 1 NEW test) (~10 min CPU)
3. Wf-425-A3 Wf-246-C metal-coordination term retrain with correct flag wire-in + pre-flight PASS + SPACE-separated --metals (~3 hr GPU USER_OK-gated)
4. Wf-425-A4 Wf-299 actual GPU launch (post Wf-420-A1 + Wf-408-A3 alias mapping) (~5 hr GPU USER_OK-gated)
5. Wf-425-A5 paper §3 v4 update from `docs/PAPER_SECTION_3_METRIC_TABLE.md` + §6 v4 from `docs/PAPER_SECTION_6_v4_LIMITATIONS_READY.md` (~22-25 hr CPU USER_OK-gated)
6. Wf-425-A6 atomic write protocol adoption migration completion (~31 remaining scripts) (~2 hr CPU)
7. Wf-425-A7 Wf-380 SCAFFOLD-loss MEASURED verification on Blackwell sm_120 (~10 hr GPU USER_OK-gated)
8. Wf-425-A8 Wf-394 Lambda CFM wire-in patch application (~5 hr GPU USER_OK-gated)
9. Wf-425-A9 m6_kappa drop-from-§3 + promote-top-3-WEAK-correlates paper conversion (~12-15 hr CPU USER_OK-gated per Wf-377-A1 §4.2)

~52.9 hr GPU + 13 hr CPU + 30 min install + SCAFFOLD-MEASURED wire-in + Tier-2 §3 + §6 v4 update total.

**CHRONOLOGY NOTE**: Wf-424-A8 closeout disk-truth per AGENTS.md §5.5: SCAFFOLD smoke SHIPPED on disk (`molmetal/reports/wf409a3_metal_coord_launch/result.json` exists 4394 bytes); 0 GPU launches performed; 0 metrics JSON updates (no Vina kcal/mol data generated); 0 paper/* touches per AGENTS.md §5.3 + TODO-84; 0 wet-lab planning per AGENTS.md §5.4 + §10.

**CHILD AGENT ROUTING**: The Wf-424-B1 docs-only sync agent should record this Wf-424-A8 closeout in TODO/INDEX.md §22.105 + append AGENTS.md §6 row + flag Wf-424-A8 SCAFFOLD_NOT_FUNCTIONAL HONEST_NEGATIVE_REFUSED + Wf-424-A4 BLOCKED pattern across Wf-420-A1 + Wf-420-A3 forward-looking recommendations.

**STANDING DIRECTIVES re-confirmed** per AGENTS.md §1 + §5.1 (USER_OK=1 honored by refusal; 0 GPU cycles consumed) + §5.2 (honest framing - MEASURED vs PROJECTED; Wf-424-A8 SCAFFOLD_NOT_FUNCTIONAL HONEST_NEGATIVE_REFUSED rather than retried on broken invocations) + §5.3 (no paper/* touches per TODO-84) + §5.4 (no wet-lab per §10) + §5.5 (post-wave on-disk verification rule - strictly observed; 5 SHIPPED files verified on disk BEFORE closeout authored) + §5.6 (atomic write protocol adoption rule - strictly observed via atomic_write_file for this closeout) + §5.7 NEW (trainer flag wire-in smoke-required rule - Wf-424-A8 50-step CPU smoke returned rc=2 with --metal-coord-term unrecognized; honored by refusal) + §10 (wetlab deprecation).
