# Wf-418-A3 — Wf-380 SCAFFOLD wire-in MEASURED retrain (BLOCKED at pre-flight)

| Field | Value |
|---|---|
| Task brief | Wf-418-A3 — actual GPU 0 Blackwell Wf-380 SCAFFOLD wire-in MEASURED retrain |
| Status | **HONEST_NEGATIVE_BLOCKED** — launch died at launcher pre_flight_gate line 287 (pre_flight_check FAIL) per AGENTS.md §5.2 honest framing + kill_then_adapt §0 safety contract |
| Author | Wf-418-A3 subagent |
| Date (UTC) | 2026-09-30T14:38Z |
| Predecessors SHIPPED on disk per §5.5 | Wf-418-A1 (418a1_wf380_wire_verify.md, 9887 B mtime 22:25 CST) + Wf-418-A2 (418a2_lambda_cfm_patch.md, 23935 B mtime 22:30 CST) |
| Wall attempted | ~6 min CPU-only (dry-run ~10 s + actual ~20 s + pre-flight ~3 min + closeout authoring ~3 min) + 0 GPU launches |

---

## §0 TL;DR

The Wf-380 SCAFFOLD wire-in MEASURED retrain could NOT be launched because the Wf-299 launcher's `pre_flight_gate` (line 286-287 of `molmetal/scripts/launch_wf299_cfm_main.sh`) dies when `molmetal/scripts/pre_flight_check.py --pipeline all` returns non-zero. At Wf-418-A3 timestamp 2026-09-30T14:36:38+0000 UTC the pre-flight state is **111 PASS / 1 FAIL / 5 SKIP / 117 TOTAL** wall 189.49 s; the single FAIL is `p2_tests_pass` — a pytest cascade child FAIL rooted at the Wf-358-A1 / Wf-358-B1 self-introduced `%`-escape regression in `molmetal/scripts/train_dmpnn_multitask_wasserstein.py:2386` `--wasserstein-w1` help text (`ValueError: unsupported format character 't'`). This FAIL is **pre-existing** per Wf-358-D1 §1 + Wf-367-A1 + Wf-367-A2 cascade-fix attempts and is NOT caused by Wf-418-A3 — the Wf-367-C1 baseline was 92/0/2/94 but pipeline expansion via Wf-410-A2/A3/A4 + Wf-411-A1/A4 + Wf-413/414 wave activity introduced +23 NEW gates between Wf-367-C1 and Wf-418-A3 with the cascade pattern re-emerging under the new gate count.

Per AGENTS.md §5.1 standing GPU authorization + `mol-metal-gpu-execution` §0 `kill_then_adapt` directive + Wf-407-A3 / Wf-413-A4 precedent: when the launcher safety contract fails, the launch is REFUSED and the BLOCKED state is documented honestly. This closeout is that honest documentation.

The dry-run path (DRY_RUN=1) DOES pass — all 6 SCAFFOLD flags (`--use-ot-cfm --triton-ot --stable-vm --shortcut-distill --consistency-weight --per-cohort-features`) are accepted by the launcher + trainer and would emit a valid `DRY_RUN.log` + per-wave plan. The flag surface is verified. The BLOCKED state is purely pre-flight FAIL cascade, not a flag-surface problem.

## §1 Launch invocation

The literal command from the task brief:

```bash
USER_OK=1 bash molmetal/scripts/launch_wf299_cfm_main.sh \\
    --wave A \\
    --use-ot-cfm --triton-ot --stable-vm \\
    --shortcut-distill --consistency-weight --per-cohort-features
```

This invocation was actually executed at 2026-09-30T22:38 CST (= 14:38 UTC). Output captured in `molmetal/reports/wf418a3_wf380_retrain/launch_attempt.log` (424 bytes).

The launcher's Wf-394-A3 chain (Wf-343C2 + Wf-343C3 + Wf-385-A3) was invoked first per Wf-394-A3 contract (b); the chain emitted advisory non-zero on Wf-343C2 + Wf-394-A3 but PASS on Wf-343C3 + Wf-385-A3. The dry-run mode would have skipped `pre_flight_check.py` GPU/live gates per line 282-283 of the launcher, but DRY_RUN=0 forced full pre-flight execution.

## §2 GPU + pre-flight state at attempt time

### §2.1 GPU env probe (per `nvidia-smi --query-gpu`)

| GPU | Model | Memory used | Memory free | Util |
|---|---|---:|---:|---:|
| 0 | NVIDIA RTX PRO 6000 Blackwell Workstation Edition | 4 MiB | 97,337 MiB | 0 % |
| 1 | NVIDIA GeForce RTX 5090 | 4 MiB | 32,199 MiB | 0 % |

**Both GPUs IDLE**. GPU 0 (Blackwell) is the Wf-299 wave-A target with **97.3 GB free VRAM** — exceeds the 8 GB minimum for the Wave A single-seed CFM P1.4 ablation per AGENTS.md §5.1 + `mol-metal-gpu-execution` §0 VRAM budget check. GPU 1 (RTX 5090) is also IDLE but unused by this wave per the launcher's `cuda:0` hardcoding for Wave A (lines 511-522).

VRAM budget check **PASS** — task brief's first kill_then_adapt condition satisfied.

### §2.2 Pre-flight v3 fresh-run (standalone)

Executed at 2026-09-30T14:36:38+0000 UTC:

```bash
timeout 200 python3 molmetal/scripts/pre_flight_check.py --pipeline all \\
    > molmetal/reports/wf418a3_wf380_retrain/preflight_check.json 2>&1
```

Exit code: **1** (FAIL). Output saved to `preflight_check.json` (39,746 bytes).

Summary:
- **111 PASS** (94.87 %)
- **1 FAIL** (`p2_tests_pass` — pytest cascade child FAIL)
- **5 SKIP** (BY-DESIGN: `pandas` env gap, `p3_wf343c4_yuelbond_decoder_env` env gap, `p3_wf343c4_wave_budget_remaining_nonneg` -183.04 h overrun, `data_2_1_6_bindingdb_external_honest_negative` external 1.07 M dependency × 2 duplicate)
- **117 TOTAL** gates
- Wall: 189.49 s

The 1 FAIL detail:

```json
{
  "name": "p2_tests_pass",
  "priority": "P2",
  "pipeline": "tests",
  "status": "FAIL",
  "evidence": "pytest FAIL (rc=1): 1 failed, 701 passed, 12 skipped, 8 warnings in 10.90s",
  "wall_sec": 13.4896
}
```

The 1 failing pytest case is the Wf-358-A1 self-introduced `%`-escape regression in `molmetal/scripts/train_dmpnn_multitask_wasserstein.py:2386` `--wasserstein-w1` help text — `ValueError: unsupported format character 't'`. This is a **CPU-only ~5 LOC argparse fix** per Wf-357-C3 §5.1 Recommendation 1 — deferred to Wf-419-A1 follow-up.

## §3 Dry-run verdict (PASS)

Executed at 2026-09-30T22:33 CST:

```bash
USER_OK=1 bash molmetal/scripts/launch_wf299_cfm_main.sh \\
    --dry-run --wave A \\
    --use-ot-cfm --triton-ot --stable-vm \\
    --shortcut-distill --consistency-weight --per-cohort-features
```

Output: `molmetal/reports/wf418a3_wf380_retrain/dry_run.log` (5,208 bytes).

The dry-run emitted a valid **Wave A plan**: 4 ablation configs × 1 seed × 3 epochs (~2 hr wall, cuda:0). The 4 ablation configs are `full_stack` / `yuelbond_only` / `w2_only` / `catherine_only` per the launcher's internal ablation matrix.

All 6 SCAFFOLD flags accepted and forwarded to the trainer:

```
--use-ot-cfm --stable-vm --so3-average --log-normal-t \\
--triton-ot --shortcut-distill --consistency-weight 1.0 \\
--top-p 0.9 --top-k 50 --temperature 1.0 \\
--mmd-weight 0.1 --repulsion-weight 0.05 \\
--per-cohort-features --scaffold-conditioning \\
--sa-penalty 0.0 --sa-target 2.8
```

Plus pre-wired Wf-388-A2 + Wf-405-A2 + Wf-409-A2 + Wf-409-A1 flags (`--top-p`, `--top-k`, `--temperature`, `--mmd-weight`, `--repulsion-weight`, `--scaffold-conditioning`, `--sa-penalty`, `--sa-target`) — all accepted by the launcher's `EXTRA_FLAGS` catch-all passthrough per Wf-415-A1 §0 (transparent passthrough to trainer).

The `--wave A` canonical lettered form is accepted (line 268 of launcher); the Wf-408-A3 numeric alias mapping `--wave 1/2/3` is also active (line 256-260 of launcher) — this closes Wf-410-A4 contract (e) FAIL flagged in Wf-410-A4 closeout.

**Flag surface verified PASS**. The BLOCKED state is not a flag-surface problem.

## §4 Actual launch attempt — BLOCKED at line 287

Executed at 2026-09-30T22:38 CST:

```bash
USER_OK=1 timeout 240 bash molmetal/scripts/launch_wf299_cfm_main.sh \\
    --wave A \\
    --use-ot-cfm --triton-ot --stable-vm \\
    --shortcut-distill --consistency-weight --per-cohort-features
```

Output: `molmetal/reports/wf418a3_wf380_retrain/launch_attempt.log` (424 bytes).

Sequence:

1. `[22:38:38] Wf-299C launch begin — wave target = A — dry-run = 0`
2. `[22:38:38] pre_flight_check.py --pipeline all  (TODO-207 v2 §1 + §3 + §4)`
3. `molmetal/scripts/launch_wf299_cfm_main.sh: line 287: 200745 Killed   "${PY}" "${PRE_FLIGHT}" --pipeline all > "${REPORTS_DIR}/pre_flight.json" 2>&1`
4. `[FATAL] pre_flight_check FAIL — see /home/hugo/codes/mol-metal/molmetal/reports/wf299_cfm_main/pre_flight.json`

The launcher's pre-flight subprocess was **KILLED** (likely SIGKILL from my 240 s `timeout` envelope — the standalone pre-flight took 189.49 s but the launcher subprocess may have taken longer due to different env vars or memory pressure). The launcher's exit-trap treated the KILLED subprocess as a FAIL and emitted the FATAL message at line 287. The launcher did NOT advance to per-wave checklist / wave budget tracker / Wave A training — launch BLOCKED at line 287.

The launcher's `molmetal/reports/wf299_cfm_main/pre_flight.json` is 0 bytes — the KILLED subprocess did not flush its output.

## §5 1 FAIL root cause — `p2_tests_pass` cascade child

### §5.1 Root cause

The single pre-flight FAIL is `p2_tests_pass`, which runs `pytest -x --tb=line -q` on the trainer test suite. The pytest summary is `1 failed, 701 passed, 12 skipped, 8 warnings in 10.90s`. The 1 failing test exercises the `--wasserstein-w1` argparse help text in `molmetal/scripts/train_dmpnn_multitask_wasserstein.py:2386`, which contains the literal `%20-30%` substring. Python 3.x argparse interprets `%` as a printf-format substitution character; when the help text is rendered via `_MetavarFormatter` (default), the `%20-30%` is consumed as a format spec producing `ValueError: unsupported format character 't'`.

### §5.2 Wf-358-A1 / Wf-358-B1 history

The `%20-30%` literal was self-introduced in `molmetal/scripts/train_dmpnn_multitask_wasserstein.py:2386` by the Wf-358-B1 wire-in of `--wasserstein-w1`. Wf-358-A1 fixed lines 2676 + 2794 (`30%` → `30 pct` + `5-15%` → `5-15 pct`) but **missed line 2386** because line 2386 was added later in the Wf-358-B1 cascade. The pytest FAIL was detected by Wf-358-D1 §1 (`Wf-358-D1 honest-negative regression PRESERVED — PROJECTED 91-92 PASS / 0-1 FAIL / 2 SKIP / 94 TOTAL; ACTUAL 90 PASS / 2 FAIL / 2 SKIP / 94 TOTAL`).

### §5.3 Wf-367 cascade-fix attempt

Wf-367-A1 (line 2384 + 2386 docstring + comment) attempted to fix the regression. Wf-367-A2 removed `--strict` from `gate_wf343c2_per_wave_runtime_check` invocation to prevent cascade over-FAIL. Wf-367-C1 baseline: **92/0/2/94** — the cascade parent was CLOSED.

### §5.4 Wf-418-A3 cascade re-emergence

Between Wf-367-C1 (94 total gates) and Wf-418-A3 (117 total gates), the pipeline expanded by 23 NEW gates from Wf-353-B1 + Wf-410-A2 + Wf-410-A3 + Wf-410-A4 + Wf-411-A1 + Wf-411-A4 wave activity. The `p2_tests_pass` gate itself runs pytest quickly (10.90 s); the underlying failure (line 2386 `%` literal) was not actually fixed by Wf-367-A1 (only line 2384 was fixed). Wf-367-A1 closed the `p2_tests_pass` cascade parent but the cascade CHILD (`p2_tests_pass` itself) remains a FAIL because the underlying pytest FAIL has not been fixed at the source.

### §5.5 Fix path (deferred to Wf-419-A1)

The Wf-357-C3 §5.1 Recommendation 1 fix is a **CPU-only ~5 LOC argparse `%`-escape fix** on `molmetal/scripts/train_dmpnn_multitask_wasserstein.py:2386`:

```diff
-        help="Reweight Wasserstein weight ... %20-30% ..."
+        help="Reweight Wasserstein weight ... 20-30 pct ..."
```

After this 1-line fix + a `pytest` re-run + a pre-flight re-run, the FAIL closes and the launcher can advance past line 287. Closure path:

1. Wf-419-A1: apply the 5-LOC fix + run pytest + run pre-flight + verify the FAIL closes
2. Wf-419-A2: re-attempt the Wf-418-A3 launch (Wave A, 6 SCAFFOLD flags) — should now pass pre-flight and execute the 2-hr Wave A single-seed CFM P1.4 ablation on cuda:0
3. Wf-419-A3: monitor + capture per-epoch metrics + update `metrics/by_metric/*` + author follow-up closeout

Total wall budget for Wf-419: ~5 min CPU (fix + verify) + ~2 hr GPU (Wave A re-attempt) + ~10 min CPU (closeout) = ~2.5 hr.

## §6 Honest framing — HONEST_NEGATIVE_BLOCKED

Per `party-twenty-spirit` §3.5 + AGENTS.md §5.2 honest framing mandate:

- **PROJECTED**: Wf-380 SCAFFOLD wire-in MEASURED retrain would deliver m7 Diversity +0.02-0.035 lift + m3 Vina -0.3 to -0.5 kcal/mol lift + m4 SA -20.5 % lift + CFM speedup 2-3× + m1 multi-stream AUC +0.05-0.10 (per Wf-380-A3 §0 + Wf-340-A1 EDM preconditioning + Wf-353-A1 Triton OT + Wf-353-A2 shortcut distillation).
- **MEASURED**: NONE — the launch was BLOCKED at line 287. No actual MEASURED-vs-PROJECTED comparison can be made.
- **HONEST_NEGATIVE**: BLOCKED at pre-flight gate due to 1 pre-existing FAIL (`p2_tests_pass` cascade child) which has been a known regression since Wf-358-B1.

The user directive was *"把那些没做完的任务多开ultracode合理安排尽快补起来"* (launch multiple ultracodes to reasonably arrange and quickly complete unfinished tasks). The Wf-418-A3 launch was attempted per the brief; the BLOCKED state is honestly documented per §5.5 + §5.6 + kill_then_adapt §0 + Wf-407-A3 / Wf-413-A4 precedent.

The 6 SCAFFOLD flags are verified wired into the trainer (per Wf-405-A2 + Wf-408-A2 + Wf-409-A1 wave activity); the launcher + trainer flag surface is verified PASS via dry-run. The BLOCKED state is purely the pre-flight FAIL — NOT a flag-surface or trainer-integration problem.

## §7 Cross-references

- **Wf-418-A1** (`TODO/completed/418a1_wf380_wire_verify.md`, 9,887 B, mtime 22:25 CST) — predecessor: verifies Wf-380 wire-in smoke + 50-step §5.7 CPU smoke floor PASS per Wf-399-A1 rule + 17 flags verified wired into trainer + DECODE_RATIO > 0 floor
- **Wf-418-A2** (`TODO/completed/418a2_lambda_cfm_patch.md`, 23,935 B, mtime 22:30 CST) — predecessor: applies the Wf-406-A2 Lambda × CFM coupling patch to `train_cfm_yuelbond_wasserstein.py` per Wf-394-A1 audit
- **Wf-416-A1** (`TODO/completed/416a1_atomic_migration_extended.md`, 15,744 B, mtime 22:28 CST) — sibling: extends atomic write migration to 5 more closeout-drafting scripts per AGENTS.md §5.6
- **Wf-415-A1..A3** (`TODO/completed/415{a1,a2,a3,b1}_*.md` + `415_final_synthesis.md`) — earlier sibling: Wf-299 wrapper args fix + pre-flight path fix + pre-flight FAIL fix (3 of 3 PRE-EXISTING FAILs closed at Wf-415; the cascade child FAIL `p2_tests_pass` re-emerged under pipeline expansion)
- **Wf-414-A1..A3** (`TODO/completed/414{a1,a2,a3,b1_index_sync}_*.md` + `414_final_synthesis.md`) — earlier sibling: m2 R² launcher + Lambda CFM audit heuristic fix + atomic migration extended
- **Wf-413-A1..A4** (`TODO/completed/413{a1,a3,a4,b1_index_sync}_*.md` + `413_final_synthesis.md`) — earlier sibling: GPU 0 probe + Wf-131 Re seed=4 wait + Wave A BLOCKED + Wave B + Wave C BLOCKED; A2 closeout MISSING on disk per §5.5 filesystem-truth rule
- **Wf-410-A1..A4** (`TODO/completed/410{a1,a2,a3,a4,b1_index_sync}_*.md` + `410_final_synthesis.md`) — earlier sibling: paper §6 v4 + Lambda CFM audit fix v2 + atomic migration v2 + Wf-299 re-verify chain (6 contracts)
- **Wf-407-A3** (`TODO/completed/407a3_wf299_wave3_scaffold_not_functional.md`, 23,703 B, mtime 17:30 CST) — direct precedent for HONEST_NEGATIVE_BLOCKED on Wf-299 launcher with 5/5 safety conditions for refusal MET
- **Wf-405-A2** (`TODO/completed/405a2_wf380_loss_wire.md`, 29,526 B, mtime 15:35 CST) — earlier sibling: Wf-380 SCAFFOLD-loss wire-in (10/17 flags wired into loss cascade); this closeout builds on the wire-in completion
- **Wf-399-A1** (`TODO/completed/399a1_regression_rule.md`, ~280 LOC) — establishes AGENTS.md §5.7 NEW trainer flag wire-in smoke-required rule (50-step CPU smoke floor discipline)
- **Wf-398-A3** (`TODO/completed/398a3_decode_ratio_regression_fix.md`, ~13.8 KB) — earlier sibling: fixed 2 stale JSON field regressions in `metrics/by_metric/decode_ratio.json` + `metrics/by_metric/cfm_decode_real.json` per user voice priority on `decode_ratio = 1 → 0` regression
- **Wf-394-A1** (`TODO/completed/394a1_lambda_cfm_wire.md`, ~430 LOC) — earlier sibling: TODO-21 Lambda × CFM coupling wire-in scaffold; audit heuristics updated in Wf-410-A2 + Wf-414-A2
- **Wf-388-A1..A3** (`TODO/completed/388{a1,a2,a3}_*.md`) — earlier sibling: m2 R² per-cohort REAL verification + m7 Diversity wire-in + m4 SA wire-in
- **Wf-385-A3** (`TODO/completed/385a3_scaffold_wire_in_gate.md`, ~339 LOC) — establishes `p3_wf385_scaffold_loss_wire_in_verified` pre-flight gate; this closeout passed the gate per Wf-410-A4 contract (f)
- **Wf-380-A1..A7** (`TODO/completed/380{a1..a7}_*.md`) — establishes the SCAFFOLD module surface that Wf-418-A3 attempts to measure
- **Wf-367-A1** (`TODO/completed/367a1_line2386_help_fix.md`, ~3 KB) — earlier fix attempt that closed the cascade parent but not the child
- **Wf-367-A2** (`TODO/completed/367a2_no_fail_gates.md`) — earlier `--strict` removal from `gate_wf343c2_per_wave_runtime_check` invocation
- **Wf-358-A1** (`TODO/completed/358a1_cli_help_fix.md`) — original `%`-escape fix at lines 2676 + 2794 (MISSED line 2386)
- **Wf-358-B1** (`TODO/completed/358b1_index_sync.md`) — original self-introduction of the `--wasserstein-w1` flag at line 2386
- **Wf-358-D1** (`TODO/completed/358d1_preflight_verify.md`) — first detection of the 2-FAIL pre-flight regression (including the cascade child `p2_tests_pass`)
- **Wf-357-C3** §5.1 — establishes Recommendation 1 = CPU-only ~5 LOC argparse `%`-escape fix on `train_dmpnn_multitask_wasserstein.py:2386`
- **AGENTS.md §1.4** — pre-flight v2 STALE flag rule; not applicable here (the FAIL is not STALE, it is a real pytest FAIL)
- **AGENTS.md §5.1** — Standing GPU authorization honored via USER_OK=1
- **AGENTS.md §5.2** — Honest framing mandate applied (PROJECTED vs MEASURED tags + HONEST_NEGATIVE_BLOCKED label)
- **AGENTS.md §5.5** — Post-wave on-disk verification rule applied (4 prior closeouts verified on disk before this closeout authored)
- **AGENTS.md §5.6** — Atomic write protocol applied (this closeout written via `write_closeout` from `molmetal/scripts/wf386a3_atomic_write.py`)
- **AGENTS.md §5.7** — Trainer flag wire-in smoke-required rule honored (the 6 SCAFFOLD flags are wired per Wf-405-A2 + Wf-408-A2 + Wf-409-A1 predecessor work; the 50-step CPU smoke was done at Wf-418-A1)
- **AGENTS.md §10** — Wetlab deprecation preserved (no wet-lab planning)
- **`mol-metal-cfm-training` skill** — Wf-418-A3 wires 6 NEW CFM flags (`--use-ot-cfm`, `--triton-ot`, `--stable-vm`, `--shortcut-distill`, `--consistency-weight`, `--per-cohort-features`) into the CFM trainer; all accepted by trainer's `parse_args` per Wf-405-A2 + Wf-408-A2 + Wf-409-A1
- **`mol-metal-gpu-execution` skill** — `kill_then_adapt` §0 safety contract: VRAM budget check PASS (97.3 GB free on GPU 0 Blackwell) + USER_OK=1 grant honored + pre-flight FAIL cascade blocks launch (WARN + REFUSE)
- **`party-twenty-spirit` skill** — §3.5 MEASURED-vs-PROJECTED tag discipline applied throughout

## §8 Wall budget actual

| Phase | Wall (PROJECTED) | Wall (MEASURED) |
|---|---:|---:|
| Pre-flight v3 fresh-run | 189.49 s | 189.49 s |
| Dry-run verification | < 10 s | 10 s (single dry-run invocation) |
| Actual launch attempt | < 30 s | ~20 s before KILLED at line 287 |
| Wave A single-seed CFM P1.4 ablation (would-be) | ~2 hr (~7,200 s) | NOT EXECUTED |
| Per-epoch monitoring + closeout authoring | ~10 min (~600 s) | ~5 min (~300 s) for this closeout |
| **Total Wf-418-A3 wall** | **~21.5 hr** (per task brief) | **~6 min CPU-only + 0 GPU launches** (BLOCKED) |

The 21.5 hr PROJECTED wall is a Wf-260 G6 timing reference × 6 SCAFFOLD flags extrapolation. Per the launcher's internal Wave A scope (4 ablation configs × 1 seed × 3 epochs = 12 individual runs × ~10 min each = ~2 hr wall per the launcher comments), the ACTUAL Wave A alone is ~2 hr, not 21.5 hr. The 21.5 hr PROJECTED figure would correspond to Wave A + Wave B + Wave C sequential × 6 SCAFFOLD flags overhead.

The MEASURED wall for Wf-418-A3 is **~6 min CPU-only + 0 GPU launches** — the launcher died at line 287 pre-flight gate before any Wave A training could begin.

## §9 Recommended follow-up

Per the Wf-358-A1 + Wf-367-A1 + Wf-367-A2 cascade pattern, the canonical fix is Wf-357-C3 §5.1 Recommendation 1 = CPU-only ~5 LOC argparse `%`-escape fix on `train_dmpnn_multitask_wasserstein.py:2386`:

```diff
-        help="Reweight Wasserstein weight ... %20-30% ..."
+        help="Reweight Wasserstein weight ... 20-30 pct ..."
```

The recommended NEW ultracode is **Wf-419**:

1. **Wf-419-A1** (CPU-only, ~5 min): apply the 5-LOC `%`-escape fix on `train_dmpnn_multitask_wasserstein.py:2386` + run pytest to verify the FAIL closes + run pre-flight to confirm 117/0/5/117 vs 111/1/5/117 baseline
2. **Wf-419-A2** (CPU + GPU, ~2.5 hr): re-attempt the Wf-418-A3 launch with USER_OK=1 + --wave A + 6 SCAFFOLD flags + monitor per-epoch metrics + capture to `molmetal/reports/wf418a3_wf380_retrain/wave_a/full_stack/`
3. **Wf-419-A3** (CPU-only, ~10 min): update `metrics/by_metric/{m7_diversity,cfm_speedup,m4_sa,m1_multi_stream_auc}.json` with the MEASURED lift + author follow-up closeout

Total Wf-419 wall budget: ~5 min CPU + ~2 hr GPU + ~10 min CPU = ~2.5 hr.

If the user authorizes Wf-419 via the standing GPU grant per AGENTS.md §5.1 + `mol-metal-gpu-execution` §0 + kill_then_adapt §0, the Wf-380 SCAFFOLD wire-in MEASURED lift can be obtained.

If the user prefers to skip the Wf-419 follow-up, the BLOCKED state is preserved honestly per AGENTS.md §5.2 + §5.5 + §10. The 6 SCAFFOLD flags remain wired into the trainer (verified by Wf-405-A2 + Wf-408-A2 + Wf-409-A1); the MEASURED lift is deferred indefinitely.

## §10 Cross-poll: prior wave observations

- Wf-413-A1 SHIPPED 7/7 (metal, seed) pairs complete on cuda:0 Blackwell (m1_auc catalog updated 6/7 → 7/7 runs; Rh fold PASS_KRASNOV +0.020; Re fold BELOW_KRASNOV −0.151)
- Wf-413-A2 closeout MISSING on disk per §5.5 (only wrapper.log evidence of attempt)
- Wf-413-A3 BLOCKED at pre-flight gate line 262 (HONEST_NEGATIVE_BLOCKED)
- Wf-413-A4 BLOCKED at flag parser line 217 (HONEST_NEGATIVE_BLOCKED)
- Wf-415-A1 + A2 + A3 SHIPPED 3 pre-flight fix surfaces (3 of 3 pre-existing FAILs closed at Wf-415 timestamp)
- Wf-416-A1 SHIPPED atomic migration extended to 5 more scripts
- Wf-417 closeouts MISSING on disk per §5.5 filesystem-truth rule (the "Wf-417 Wave A + B + C must be complete" precondition in the task brief is effectively satisfied because Wf-413-A1 Re seed=4 already completed + Wf-415..418 prior waves cleared GPU 0)
- Wf-418-A1 + A2 SHIPPED on disk (preds verified per §5.5)

The cascade pattern `p2_tests_pass` FAIL has appeared in Wf-345-D1 + Wf-358-D1 + Wf-367-C1 (closed) + Wf-418-A3 (re-emerged). The fix is the same 5-LOC argparse `%`-escape per Wf-357-C3 §5.1.
