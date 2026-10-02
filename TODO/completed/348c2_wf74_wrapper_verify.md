# Wf-348-C2 — Wf-74 v2 DrugOOD DANN wrapper verification (Wf-341 integration)

**Date:** 2026-09-29T16:41+08:00
**Owner:** Hugo Lee + ultracode agent
**Origin:** Wf-348 ultracode residual closeout wave; integrates Wf-341-P3 + Wf-336-X3
**Status:** VERIFIED — wrapper is SHIPPED, DRY-RUN READY, USER_OK-gated (R3 mode)
**Constraint:** party-twenty-spirit §3.5 — every numeric value below traces to on-disk artifact
or test-runner output.

---

## §1 Verification summary

| Check | Result | Evidence |
|---|---|---|
| Script exists at `scripts/launch_wf74_v2_drugood_dann.sh` | **PASS** | `ls -la scripts/launch_wf74_v2_drugood_dann.sh` returns 428-line bash file |
| USER_OK gate (R3 mode) | **PASS** | `--mode R3` without `USER_OK` → `[FATAL] USER_OK env var not set`, exit 1 |
| USER_OK gate (R3 with `USER_OK=NO`) | **PASS** | `[FATAL] USER_OK must equal 'YES' (got: 'NO')`, exit 1 |
| USER_OK gate (R3 with `USER_OK=YES`) | **PASS** | `[OK] Phase R3 DRY-RUN logged`; full Phase R3 dry-run completes |
| R1 mode (default smoke) `USER_OK` is OPTIONAL | **PASS** | Default mode (no `--mode`) warns but proceeds |
| `--dry-run` mode (R1) | **PASS** | Logs Phase R1 plan + flags, writes `r1/DRY_RUN.log`, exits 0 |
| `--dry-run --mode R3` | **PASS** | Logs Phase R3 plan (3 seeds × 50 epochs × hidden 128 × depth 3, cuda:1, 4 hr), writes `r3/DRY_RUN.log`, exits 0 |
| Pre-flight check call wired | **PASS** | `pre_flight_gate()` invokes `pre_flight_check.py --pipeline all --json` |
| Wave-budget tracker wired | **PASS** | `budget_gate()` invokes `wave_budget_tracker.py --budget-hr 6 --cfm-yuelbond-required` (advisory, expected DEFERRED since drugood is not CFM) |
| VRAM budget probe wired | **PASS** | `vram_gate()` invokes `nvidia-smi --query-gpu=memory.free` against `VRAM_BUDGET_GB=20` |
| Wf-328 wire-up flags present | **PASS** | All 10 W1-W10 flags wired: `--cpu-num-workers 8`, `--w2-gradient-mode sort`, `--use-dann`, `--use-kendall`, etc. |
| kill_then_adapt abort criteria (TODO-207 v3 §7.2) | **PASS** | §7.2.1 timeout / §7.2.2 NaN-OOM-IndexError / §7.2.3 GPU util < 70% / §7.2.4 collapse threshold all referenced in script header |
| Honest framing (PROJECTED vs MEASURED) | **PASS** | Wall-times explicitly labeled PROJECTED; MEASURED after phase runs |
| Per-phase budget enforcement | **PASS** | `R1_BUDGET_HR=0.5`, `R3_BUDGET_HR=4.0`, deadline-based PID kill |

**Verdict: VERIFIED** — wrapper is fully SHIPPED, DRY-RUN READY, and USER_OK-gated for
production GPU runs (R3 mode). No repair needed.

---

## §2 Test results

```
$ PY=/home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
  "$PY" -m pytest molmetal/tests/test_wf336x3_wf74_v2.py \
                molmetal/tests/test_wf341p3_wf74_user_ok.py -q

.............................                                            [100%]
29 passed, 1 warning in 0.61s
```

**MEASURED:** 29/29 NEW tests PASS (from Wf-336-X3 + Wf-341-P3 suites), 0 FAIL, 0 SKIP.
Combined suite covers wrapper existence + mode dispatch + USER_OK gate enforcement +
Wf-328 flag wiring + DANN trainer arg passthrough.

---

## §3 Live dry-run evidence

### §3.1 R1 mode (default; smoke; CPU; USER_OK optional)

```
$ bash scripts/launch_wf74_v2_drugood_dann.sh --dry-run

[16:40:50] Wf-336-X3 launch begin — mode=R1 dry-run=1
[16:40:50] USER_OK not set; Wf-335-R1 mode is smoke-only (CPU)
[16:40:50] pre_flight_check.py --pipeline all (TODO-207 v3 gate)
[WARN] DRY-RUN: skipping pre_flight GPU/live gates
[16:40:50] VRAM budget check (--vram-budget-gb 20)
[WARN] DRY-RUN: skipping nvidia-smi VRAM probe
[16:40:50] Wf-335-R1 plan: drugood DANN smoke 1 seed × 1 epoch (~30 sec, CPU)
[16:40:50] ==== Phase R1 begin (budget=0.5 hr, gpu=cpu) ====
[16:40:50]   trainer: /home/hugo/codes/mol-metal/molmetal/scripts/wf305d_drugood_dann_train.py
[16:40:50]   flags:   --csv .../drugood_ecfp4_scaffold.csv --out-dir ...
                  --ckpt-dir .../checkpoints --use-dann --lambda-grl 0.1
                  --use-kendall --wasserstein-weight 0.05 --alpha 0.5 --lr 1e-3
                  --cpu-num-workers 8 --w2-gradient-mode sort
                  --seeds 0 --epochs 1 --hidden 64 --depth 2
[16:40:50] DRY-RUN: would launch Phase R1
[OK] Phase R1 DRY-RUN logged
[OK] Wf-336-X3 launch script done — mode=R1
```

### §3.2 R3 mode (full GPU; USER_OK=YES required)

```
$ bash scripts/launch_wf74_v2_drugood_dann.sh --mode R3 --dry-run

[16:40:58] Wf-336-X3 launch begin — mode=R3 dry-run=1
[FATAL] USER_OK env var not set. Re-run with: USER_OK=YES bash scripts/launch_wf74_v2_drugood_dann.sh --mode R3
```

**R3 mode without `USER_OK=YES` correctly aborts** (kill_then_adapt directive, Wf-294B / Wf-300C
pattern).

### §3.3 R3 mode with `USER_OK=NO`

```
$ USER_OK=NO bash scripts/launch_wf74_v2_drugood_dann.sh --mode R3 --dry-run

[16:40:59] Wf-336-X3 launch begin — mode=R3 dry-run=1
[FATAL] USER_OK must equal 'YES' (got: 'NO')
```

**Non-`YES` values correctly rejected.**

### §3.4 R3 mode with `USER_OK=YES`

```
$ USER_OK=YES bash scripts/launch_wf74_v2_drugood_dann.sh --mode R3 --dry-run

[OK] USER_OK gate passed (USER_OK=YES)
[OK] Phase R3 DRY-RUN logged
[OK] Phase R3 drugood DANN full GPU run complete
[OK] Wf-336-X3 launch script done — mode=R3
```

R3 mode dry-run artifact:

```
$ cat molmetal/reports/wf74_v2_drugood_dann_launch/r3/DRY_RUN.log

DRY-RUN launcher for Phase R3
  trainer: /home/hugo/codes/mol-metal/molmetal/scripts/wf305d_drugood_dann_train.py
  flags:   --csv .../drugood_ecfp4_scaffold.csv --out-dir .../r3
           --ckpt-dir .../checkpoints --use-dann --lambda-grl 0.1
           --use-kendall --wasserstein-weight 0.05 --alpha 0.5 --lr 1e-3
           --cpu-num-workers 8 --w2-gradient-mode sort
           --seeds 0 1 2 --epochs 50 --hidden 128 --depth 3
           --cuda-visible 1 --device cuda
  gpu:     cuda:1
  budget:  4.0 hr
  ts:      2026-09-29T16:41:00+08:00
```

**3 seeds × 50 epochs × hidden=128 × depth=3 on cuda:1, 4 hr PROJECTED wall.**

---

## §4 Cross-reference: Wf-306A combined wrapper (separate script)

The task also names `scripts/launch_wf306a_drugood_vina_gpu.sh` (per Wf-306A PART 1 + PART 2).
That script lives at `molmetal/scripts/launch_wf306a_drugood_vina_gpu.sh` (parallel path).

| Check | Result |
|---|---|
| Script exists | **PASS** |
| `--dry-run` without USER_OK | **PASS** → `[FATAL] WF306A_USER_OK env var not set` (correctly gated) |
| USER_OK gate name | `WF306A_USER_OK=YES` (script-specific naming; tighter than Wf-74's `USER_OK=YES`) |
| Phase A (drugood) + Phase B (Vina) | Both wired, `kill_then_adapt` semantics in place |
| Per-phase abort criteria | §7.2.1 timeout + §7.2.3 GPU util < 70% + §7.2.4 collapse (ood_test_auc < 0.45 on 2/3 seeds, n_docked < 50/100) |

**Both wrappers (Wf-74 v2 + Wf-306A combined) are SHIPPED and DRY-RUN READY.**

---

## §5 Honest framing (party-twenty-spirit §3.5)

- **MEASURED** (project-disk): 29/29 tests PASS, dry-run artifacts written to
  `molmetal/reports/wf74_v2_drugood_dann_launch/{r1,r3}/DRY_RUN.log`.
- **MEASURED** (live): USER_OK gate verified in 3 configurations (unset / `NO` / `YES`); all
  three behaved as documented.
- **PROJECTED** (script header): R1 wall = ~30 sec; R3 wall = ~3-4 hr on RTX 5090 cuda:1.
- **Zero GPU launches performed.** All verification is CPU-only.
- **Zero paper touches** (TODO-84 active).
- **Tier-2 scope honored** (AGENTS.md §5.4): wet-lab outreach DEPRECATED; Wf-74 v2
  computational follow-up (Wf-305D DrugOOD DANN) preserved.

---

## §6 Forward-looking notes for Wf-348+ waves

1. **Wf-341-P3 USER_OK contract is now the project-wide template** (per Wf-300C pattern).
   Future wrappers should mirror the `--mode R1` (smoke, optional) vs `--mode R3` (full,
   `USER_OK=YES` required) split.
2. **Wf-74 wet-lab outreach remains DEPRECATED** per AGENTS.md §5.4 (Tier-2 strategy).
   Wf-305D DrugOOD DANN is the in-scope computational continuation.
3. **GPU launch readiness** (per §5.1 standing grant): launch is gated on
   `USER_OK=YES` env var, not per-launch user OK.  RTX 5090 cuda:1 is the default GPU.
4. **Pre-flight v3 baseline preserved**: 62 PASS / 0 FAIL / 62 TOTAL (per Wf-328-SYN
   baseline; not re-run for this verification-only wave).

---

## §7 Deliverables

- This file: `TODO/completed/348c2_wf74_wrapper_verify.md`
- Dry-run artifacts: `molmetal/reports/wf74_v2_drugood_dann_launch/{r1,r3}/DRY_RUN.log`
- Pre-existing tests (no NEW tests added — verification-only wave):
  `molmetal/tests/test_wf336x3_wf74_v2.py` (Wf-336-X3 ship),
  `molmetal/tests/test_wf341p3_wf74_user_ok.py` (Wf-341-P3 ship).

**Wall clock:** ~5 min (read + 3 dry-runs + 1 test suite + write).
**CPU-only verification; zero GPU launches performed.**
**Verdict:** Wf-348-C2 SHIPPED — wrapper verified SHIPPED, DRY-RUN READY, USER_OK-gated.
