# Wf-355-B1 — Wf-131 P3 7 DEFERRED runs re-launch wrapper (USER_OK-gated)

**Date**:        2026-09-29
**Workflow**:    Wf-355-B1 (ultracode residual wave)
**Status**:      SHIPPED — wrapper + 3 NEW tests PASS (CPU-only; 0 GPU launches)
**Owner**:       Hugo Lee + ultracode agent (per AGENTS.md §1.2 + §5.1 + §5.2)
**Skill alignment**: `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline); no `paper/*` touches (TODO-84 honored)

---

## §0 TL;DR

Per Wf-354-A1 §5 HONEST-NEGATIVE: "Wf-131 P3 Re-launch wrapper needed
(USER_OK-gated for >1hr GPU per AGENTS.md §5.1)". The 7 DEFERRED runs
documented in `metrics/by_metric/metal_lomo_auc.json` `wf131_p3_lomo_partial_update`
block needed a USER_OK-gated re-launch wrapper.

**Wf-355-B1 closes this gap with**:

| Component | Path | LOC | Status |
|---|---|---:|---|
| **Launch wrapper** | `scripts/launch_wf131_p3_deferred_runs.sh` | 299 lines (incl. 80-line docstring header) | DRY-RUN READY; USER_OK gate enforced (exit 64) |
| **Test file** | `molmetal/tests/test_wf355b1_wf131_p3_deferred.py` | 3 tests | 3/3 PASS (CPU-only; 0.42 s wall) |
| **plan.json** | `molmetal/reports/wf131_p3_deferred_runs/plan.json` | 26 lines | EMITTED on `--dry-run` |

**Tests**: 3/3 PASS
- `test_wf355b1_user_ok_gate_enforced` — USER_OK gate: refuses without `USER_OK=1` (exit 64) + passes with `USER_OK=1 --dry-run` (exit 0)
- `test_wf355b1_bash_syntax_check` — `bash -n` parses cleanly + size 299 ∈ [200, 350] + shebang + trainer cross-ref
- `test_wf355b1_per_run_enumeration_7_runs` — 7 (metal, seed) pairs enumerated: Rh 3+4 + Re 0-4

---

## §1 Per-run status (verbatim from MEASURED JSON)

### §1.1 Source: `metrics/by_metric/metal_lomo_auc.json` `wf131_p3_lomo_partial_update`

Per Wf-354-A1 §1.2 (verbatim):

| Metal | Seeds done | Per-seed test_auc | Mean | Std | DEFERRED status |
|---|:---:|---|---:|---:|---|
| **Ru** | 5/5 | 0.5911, 0.6185, 0.5881, 0.5917, 0.5810 | **0.5941** | 0.0143 | COMPLETE (no DEFERRED) |
| **Ir** | 5/5 | 0.6336, 0.6062, 0.6228, 0.6505, 0.6080 | **0.6242** | 0.0185 | COMPLETE (no DEFERRED) |
| **Os** | 5/5 | 0.6149, 0.6058, 0.6233, 0.6351, 0.5585 | **0.6075** | 0.0294 | COMPLETE (no DEFERRED) |
| **Rh** | **3/5** | 0.2742, 0.9143, 0.8735 | **0.6874** | 0.3584 | **2 DEFERRED (seeds 3, 4)** |
| **Re** | **0/5** | n/a | n/a | n/a | **5 DEFERRED (seeds 0, 1, 2, 3, 4)** |
| **Total** | **18/25 = 72%** | — | **0.6217** | 0.1284 | **7 DEFERRED total** |

### §1.2 The 7 DEFERRED runs (target of this wrapper)

Per `metal_lomo_auc.json:280-285`:

```
"training_stop_evidence": "Last log entry: 2026-09-27 09:05:40,917 fold=Rh seed=3 epoch=23 (NOT epoch 49); no Re dir on disk; nvidia-smi 0% utilization at 09:07 — training process terminated",
```

The 7 DEFERRED (metal, seed) runs are:

| # | Metal | Seed | DEFERRED reason | Resume target |
|:---:|:---:|:---:|---|---|
| 1 | Rh | 3 | training stopped at epoch 23 (last log 2026-09-27 09:05:40) | re-run to epoch 50 |
| 2 | Rh | 4 | never started | fresh run |
| 3 | Re | 0 | never started | fresh run |
| 4 | Re | 1 | never started | fresh run |
| 5 | Re | 2 | never started | fresh run |
| 6 | Re | 3 | never started | fresh run |
| 7 | Re | 4 | never started | fresh run |

### §1.3 Honest framing on carryover risks

**Per Wf-354-A1 §6**:
- The 13 (metal, seed) remaining Wf-131 P3 + Wf-146 runs are DEFERRED — NO honest-negative lift can be claimed without fresh data.
- The Rh fold regression (0.136) is REAL and NOT RECOVERED.
- The Tier-2 catalog number (0.6220 Re-only Wasserstein, n=1) is documented but does NOT replace the R17 anchor.

**Per `metal_lomo_auc.json:rh_honest_framing`**:
> "Rh fold MEASURED on BUGGY Wf-125 P1 cohort (constant-inactive labels, see TODO-135 §0 root cause). Rh mean 0.6874 is misleading — 3-seed n is too small; seed=0 regressed to 0.274 (paper §6.9 honest-negative), seeds 1+2 lifted to 0.914/0.874. Std=0.358 (high variance from seed-0 outlier)."

**Implication for the 7-run re-launch**: Even after re-running Rh seeds 3+4, the result will still be on the BUGGY Wf-125 P1 cohort. The structural fix-target (TODO-135 Phase 1 = balanced Rh cohort via Wf-309A + Wf-139 P7) is already on disk (`synthetic_rh_pos_neg.json`), but the Wf-131 P3 trainer defaults to the BUGGY cohort path. Re-running on the BUGGY cohort may NOT clear the paper §4 acceptance gate.

**Re fold carryover (per Wf-354-A1 §1.2 footnote)**:
> "Re fold DEFERRED (0/5 seeds done; Re not started before training stopped at Rh seed=3 epoch=23)"

**Implication for Re 0-4**: All 5 Re runs are on the BUGGY cohort. Re cohort is the smallest (~800 rows), so per-seed AUC variance may be high. The lift-vs-R17 comparison will be underpowered regardless.

---

## §2 Re-launch wrapper reference

### §2.1 Wrapper identity

| Property | Value |
|---|---|
| **Path** | `/home/hugo/codes/mol-metal/scripts/launch_wf131_p3_deferred_runs.sh` |
| **LOC** | 299 lines (incl. 80-line docstring header) |
| **Date** | 2026-09-29 |
| **Origin** | Wf-355-B1 ultracode residual closure wave (closes Wf-354-A1 §5 HONEST-NEGATIVE) |
| **Trainer** | `molmetal/scripts/train_metal_lomo_wasserstein.py` (48958 bytes — Wf-125 P2 + Wf-131 P3 canonical entry per Wf-354-A1 §3.2) |
| **LOMO CSV** | `.data/molmetal/metalcytotox_5metal_multitask_v4.csv` (Wf-261 strict scaffold split) |
| **Rh cohort** | `molmetal/reports/wf139_p7_rh_active_cohort/synthetic_rh_pos_neg.json` (250 active + 250 inactive balanced) |
| **GPU** | `cuda:1` (RTX 5090 GPU-1 per Wf-347-B1 G5/G6 parity) |
| **Epochs** | 50 |

### §2.2 Wf-335-R1+R3 lever wiring

The wrapper includes the post-Wf-335 Blackwell cudagraphs safety lever
via Wf-328-W2/W5/W6 + Wf-326B OPTIMAL_NUM_WORKERS (8) + Wf-326A
w2-gradient-mode sort:

| Flag | Default | Source | Rationale |
|---|---|---|---|
| `--wasserstein-weight 0.05` | default | Wf-125 P2 baseline | W₂ regularisation |
| `--rh-oversample 500` | default | Wf-125 P2 baseline | Rh synthetic cohort |
| `--rh-cohort-json` | Wf-139 P7 500-row | Wf-321A + Wf-309A | Balanced cohort for Rh fold |
| `--csv` | v4 strict scaffold | Wf-261 | Zero scaffold leak |
| `--use-l2-guard` | default | Wf-127 P2 | Grad-norm clip |
| `--w2-gradient-mode sort` | default | Wf-328-W5 | Wf-326A MEASURED 2-3× speedup |
| `--cpu-num-workers 8` | default | Wf-328-W2 | Wf-326B OPTIMAL_NUM_WORKERS |

### §2.3 USER_OK gate (per AGENTS.md §5.1)

```bash
if [ "${USER_OK:-0}" != "1" ]; then
    printf '%s[FATAL]%s USER_OK gate refused: set USER_OK=1 in env to launch.\n' \
        "$(printf '\033[0;31m')" "$(printf '\033[0m')" >&2
    printf '       USER_OK=1 bash %s --dry-run   # plan only (still requires USER_OK=1)\n' "$0" >&2
    printf '       USER_OK=1 bash %s                # actual ~5 hr GPU run\n' "$0" >&2
    printf '       USER_OK=1 bash %s --runs "Rh 3 Re 0"   # subset\n' "$0" >&2
    exit 64
fi
```

**Pattern**: matches `mol-metal-gpu-execution` §0 kill_then_adapt
directive + Wf-294B / Wf-300C USER_OK pattern. Exit code 64 is the
conventional "USER_OK refused" code (matches `launch_wf310e_wf135_phase2_gpu.sh`
+ `launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh`).

### §2.4 Subset override via `--runs`

The wrapper supports `--runs Rh 3 Rh 4 Re 0` (without quotes — bash
tokenizes by whitespace) for partial sweeps. Default is all 7 DEFERRED
runs.

---

## §3 Estimated wall time per run

### §3.1 Per-run wall-time (PROJECTED per Wf-260 G6 timing)

| Metal | Cohort size | Wall per run | N runs | Subtotal |
|---|---:|---|:---:|---:|
| Rh | ~1.2k rows | ~15 min | 2 (seeds 3, 4) | **~30 min** |
| Re | ~800 rows | ~30 min | 5 (seeds 0, 1, 2, 3, 4) | **~2.5 hr** |
| Sequential overhead + ckpt write | — | — | — | **~1 hr** |
| **Total** | — | — | **7** | **~5 hr GPU** |

### §3.2 Comparison to Wf-310E Wf-135 Phase 2

The Wf-310E Wf-135 Phase 2 wrapper (`launch_wf310e_wf135_phase2_gpu.sh`)
targets 25 (metal, seed) runs at ~10 hr GPU wall. This Wf-355-B1 wrapper
is the **partial sweep** (7 of 25 runs) — about 1/3 the wall.

### §3.3 Wall budget enforcement

```bash
WALL_BUDGET_HR=5
...
if [ "${elapsed_hr}" -gt "${WALL_BUDGET_HR}" ]; then
    warn "elapsed (${elapsed_hr} hr) exceeded PROJECTED wall budget (${WALL_BUDGET_HR} hr); flagging as MEASURED-over-PROJECTED in plan.json"
    sed -i "s/READY_TO_RUN/READY_TO_RUN_MEASURED_OVER_BUDGET/" "${PLAN_JSON}"
fi
```

**Pattern**: matches `launch_wf310e_wf135_phase2_gpu.sh:236-241` — soft
warning (no hard stop) per kill_then_adapt + Wf-295D per_wave_user_ok
pattern.

---

## §4 Honest framing per party-twenty-spirit §3.5

### §4.1 What this wrapper MEASURED today

1. ✅ Wrapper `scripts/launch_wf131_p3_deferred_runs.sh` SHIPPED (299 LOC, `bash -n` clean)
2. ✅ 3 NEW tests PASS (CPU-only, 0.42 s wall)
3. ✅ USER_OK gate enforced (exit 64 without `USER_OK=1`)
4. ✅ `--dry-run` mode emits plan.json with 7 enumerated (metal, seed) tuples
5. ✅ `--runs Rh 3 Re 0` subset override WORKS
6. ✅ Pre-flight v3 baseline PRESERVED (exit 0; 62/62 PASS at Wf-335-SYN baseline)
7. ✅ Zero GPU launches performed (CPU-only verification)
8. ✅ No `paper/*` touches (TODO-84 honored)

### §4.2 What this wrapper does NOT claim

1. ❌ Does NOT claim the 7 DEFERRED runs are COMPLETE — wrapper only; no training performed
2. ❌ Does NOT claim Rh fold 0.6874 ± 0.3584 will improve on BUGGY cohort re-runs
   - The seed=0=0.274 outlier may be a fluke OR structural — only fresh data resolves
3. ❌ Does NOT claim Re fold will clear paper §4 acceptance gate
   - Re cohort smallest (~800 rows); per-seed AUC variance may be high
4. ❌ Does NOT claim 5-hr wall is sufficient for production
   - PROJECTED per Wf-260 G6 timing; actual may be 30-50% longer with retries
5. ❌ Does NOT modify any metric file or ckpt (CPU-only verification)
6. ❌ Does NOT claim `train_dmpnn_multitask_wasserstein.py` is the trainer
   - Per Wf-354-A1 §3.2, the canonical trainer is `train_metal_lomo_wasserstein.py`

### §4.3 Standing directive alignment (per AGENTS.md §1.1-§1.5)

- **§1.1 MEASURED-vs-PROJECTED tag discipline**: respected — every wall-time number is PROJECTED with citation to Wf-260 G6 timing
- **§1.2 Standing GPU authorization**: respected — ZERO GPU launches performed (CPU-only verification)
- **§1.3 No paper work without authorization**: respected — ZERO paper/* touches
- **§1.4 Pre-flight v2 25/25 PASS gate**: not triggered (no retrain / panel / ablation launch); baseline preserved at Wf-335-SYN 62/62 PASS
- **§1.5 Honest-negative framing**: respected — partial + blocked statuses preserved; no aspirational lift claims

---

## §5 Files touched

| Path | Action | LOC |
|---|---|---:|
| `scripts/launch_wf131_p3_deferred_runs.sh` | NEW | 299 |
| `molmetal/tests/test_wf355b1_wf131_p3_deferred.py` | NEW | ~225 |
| `TODO/completed/355b1_wf131_p3_deferred.md` | NEW (this doc) | ~330 |
| `molmetal/reports/wf131_p3_deferred_runs/plan.json` | EMITTED by `--dry-run` | 26 |
| `molmetal/reports/wf131_p3_deferred_runs/launch.log` | EMITTED by `--dry-run` | log lines |

**No source files were modified** beyond the NEW wrapper + tests + this doc.

---

## §6 Cross-references

- **Wf-354-A1** — `TODO/completed/354a1_wf342_p2_lomo_integration.md` §1.2 + §5 (parent)
- **Wf-131 P3 partial update** — `metrics/by_metric/metal_lomo_auc.json` `wf131_p3_lomo_partial_update` block
- **Wf-354-A1 §3.2** — `train_dmpnn_multitask_wasserstein.py` vs `train_metal_lomo_wasserstein.py` clarification
- **Wf-260 G6 timing** — per-metal wall-time estimate source
- **Wf-310E Wf-135 Phase 2** — `scripts/launch_wf310e_wf135_phase2_gpu.sh` (USER_OK gate pattern)
- **Wf-294B + Wf-300C** — USER_OK gate pattern reference
- **Wf-347-B1 + Wf-347-B2** — per-pair budget tracking pattern (G5/G6 hardening)
- **Wf-335-R1+R3** — torch.compile + `--no-cuda-graphs` safety lever (Blackwell sm_120)
- **Wf-328-W2/W5/W6** — `--cpu-num-workers 8` + `--w2-gradient-mode sort` + `--num-workers 8`
- **Wf-309A + Wf-321A + Wf-139 P7** — Rh balanced cohort (`synthetic_rh_pos_neg.json`)
- **TODO-135 §0 + §5** — Rh fold BUGGY cohort root cause + 50% lift ceiling
- **AGENTS.md** §1.1-§1.5 standing directives (all honored) + §5.2 honest framing mandate + §5.4 Tier-2 markers

---

## §7 What happens after user OK

When the user clears the honest-negative resolution gate (per
user directive 2026-09-29 "先把honest-negative先解决了") and approves
the 7-run re-launch:

1. **Launch command**:
   ```bash
   USER_OK=1 bash scripts/launch_wf131_p3_deferred_runs.sh
   ```
2. **Wall**: ~5 hr GPU sequential (PROJECTED)
3. **Outputs**: per-(metal, seed) AUCs at `molmetal/reports/wf131_p3_deferred_runs/{metal}/seed{N}/per_seed_metrics_seed{N}.json`
4. **Aggregation**: append new per-seed AUCs to `metal_lomo_auc.json` `wf131_p3_lomo_partial_update` block
5. **Paper §4.5.6 honest-framing decision**: per TODO-84 rule (no paper work without authorization)

---

**END of Wf-355-B1** — Owner Hugo Lee; cross-ref `party-twenty-spirit` §3.5 + AGENTS.md §1.2 + §5.1-§5.4; SHIP (CPU-only verification; 3 NEW tests; 0 GPU launches; 1 NEW wrapper; 1 NEW doc; 3 honest-negative caveats preserved per §4.2).
