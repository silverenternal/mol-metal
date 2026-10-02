# Wf-296C — Integrate NEW CLI flags into G6 launcher

**Date:** 2026-09-29
**Owner:** Hugo Lee
**Workflow:** Wf-296C
**Target:** `scripts/launch_wf124_p3_g6.sh`

---

## 1. CONTEXT

Wf-283B + Wf-285B added CLI flags to `molmetal/scripts/train_dmpnn_multitask_wasserstein.py`:

| Flag | Source | Purpose |
|------|--------|---------|
| `--no-kendall` | Wf-283B C10 ablation | Disable Kendall UW MTL loss |
| `--no-dann`    | Wf-283B C11 ablation | Disable DANN GRL |
| `--domain-id`  | Wf-285B              | Per-domain id for DANN |
| `--lambda-grl` | Wf-285B              | GRL lambda strength |
| `--grl-alpha`  | Wf-285B              | GRL alpha |
| `--num-domains`| Wf-285B              | Domain count |
| `--metal-probe-init-scale` | Wf-269A Path C | Metal probe init |

All 7 flags are wired in `train_dmpnn_multitask_wasserstein.py` (verified via `--help`
output). Prior to Wf-296C, only `--no-kendall` and `--no-dann` were relevant to the
G6 per-innovation ablation design — `--domain-id` / `--lambda-grl` /
`--grl-alpha` / `--num-domains` are supplementary DANN hyper-params (left at script
defaults for G6), and `--metal-probe-init-scale` is Wf-269A Path C (not in G6 scope).

---

## 2. PRE-CHANGE STATE

| Check | Result |
|-------|--------|
| `--no-kendall` in `--help` | YES (line 1323) |
| `--no-dann` in `--help`    | YES (line 1331) |
| G6 launcher references `--no-kendall` | YES (header comment §15, line 70 UNVERIFIED_CONFIGS) |
| G6 launcher references `--no-dann`    | YES (header comment §15, line 70 UNVERIFIED_CONFIGS) |
| G6 launcher passes `--no-kendall` / `--no-dann` to `train_*` | **NO** (was UNVERIFIED → SKIP-with-WARN) |
| G6 launcher passes `--no-enforce-budget` to `wave_budget_tracker` | **NO** (missing) |

**Honest framing impact:** Pre-Wf-296C, 30 of 60 tuples (no_kendall + no_dann)
were SKIPPED at launch time per the `verify-flag-before-launch` guard. This meant
G6 could only ship 30 runs as MEASURED; the per-innovation attribution matrix
(TODO-182 §3.4) could only ablate `--single-task` and `--wasserstein-weight 0.0`
fully.

---

## 3. CHANGES APPLIED

### 3.1 Wf-283B flag wiring (per-config dispatcher)

The G6 launcher uses a per-config dispatcher pattern (`CONFIRMED_FLAGS` assoc
array, lines 62-65 of the original). Adding `--no-kendall` / `--no-dann` to the
shared `COMMON_ARGS` block would have **disabled Kendall + DANN across ALL 60
runs**, collapsing the ablation design (since `no_kendall` / `no_dann` configs
would become no-ops vs the all-off baseline).

Per the CONTEXT ("won't surface ablation levers"), the correct wiring is the
per-config dispatcher. Change applied to `CONFIRMED_FLAGS`:

```diff
 declare -A CONFIRMED_FLAGS=(
   ["no_multi_task"]="--single-task"
   ["no_wass"]="--wasserstein-weight 0.0"
+  ["no_kendall"]="--no-kendall"
+  ["no_dann"]="--no-dann"
 )

-VERIFIED_CONFIGS="no_multi_task no_wass"
-UNVERIFIED_CONFIGS="no_kendall no_dann"
+VERIFIED_CONFIGS="no_multi_task no_wass no_kendall no_dann"
+UNVERIFIED_CONFIGS=""
```

### 3.2 Wf-293B `--no-enforce-budget` wiring (wave_budget_tracker call)

The task verification step "Verify Wf-293B `--no-enforce-budget` already in
command block" returned **NOT PRESENT** in the G6 launcher (line 115 of original).
Per the kill_then_adapt directive (G5 §6 header convention) and the wave
budget tracker CLI (`molmetal/scripts/wave_budget_tracker.py:886`), this flag is
required to keep all 60 tuples PLANNED (otherwise over-budget waves get DEFERRED
labeling). Change applied:

```diff
-${PY} molmetal/scripts/wave_budget_tracker.py || { echo "FAIL: wave_budget_tracker"; exit 1; }
+${PY} molmetal/scripts/wave_budget_tracker.py --no-enforce-budget || { echo "FAIL: wave_budget_tracker"; exit 1; }
```

### 3.3 Dry-run summary line update

Pre-Wf-296C dry-run claimed "30 CONFIRMED + 30 UNVERIFIED". Post-Wf-296C, all 60
tuples are CONFIRMED, so the summary line was updated to reflect the new state
(dry-run output verified §5 below).

---

## 4. FLAGS WIRED — INVENTORY

| Flag | Where it lands in G6 | Effective behavior |
|------|----------------------|--------------------|
| `--no-kendall` | `CONFIRMED_FLAGS["no_kendall"]` → 15 tuples (5 metals × 3 seeds) | Kendall UW MTL OFF for those 15 tuples only |
| `--no-dann`    | `CONFIRMED_FLAGS["no_dann"]`    → 15 tuples (5 metals × 3 seeds) | DANN GRL OFF for those 15 tuples only |
| `--no-enforce-budget` | `wave_budget_tracker` invocation | All 60 tuples PLANNED, none DEFERRED |

`--domain-id` / `--lambda-grl` / `--grl-alpha` / `--num-domains` /
`--metal-probe-init-scale` are **not wired in G6** because:
1. The G6 ablation is per-innovation (which loss/method to remove), not DANN
   hyper-param tuning.
2. Adding them would change the DANN training dynamics across all 4 ablation
   configs, confounding attribution.
3. Wf-285B supplementary + Wf-269A Path C surface targets a separate sweep
   (TODO-188 follow-up, out of Wf-296C scope).

---

## 5. DRY-RUN VERIFICATION

**`bash -n` syntax check:** PASS
```
$ bash -n scripts/launch_wf124_p3_g6.sh && echo OK
OK: bash -n passed
```

**Dry-run output (post-change, key lines):**
```
=== Wf-124 P3 G6 — DRY-RUN (CPU verification only) ===
OUT_BASE: results/wf124_p3_per_innovation_ablation_g6
PY: /home/hugo/codes/CadButEaas/.venv-vlm/bin/python
SCRIPT: molmetal/scripts/train_dmpnn_multitask_wasserstein.py
DATA_CSV: /home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv
GPU: cuda:0
COMMON_ARGS: --epochs 50 --batch-size 256 --hidden 300 --depth 3 --lr 1e-3
             --alpha 0.5 --w2-gradient-mode sliced --n-mc 256 --use-l2-guard
             --device cuda:0 --cuda-visible-devices 1 --cpu-num-workers 0
             --data-csv /home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv

--- CSV sanity check ---
OK: CSV exists (/home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv)
n_rows (incl header): 26633

--- Ablation config verification status ---
CONFIRMED flags: no_multi_task no_wass no_kendall no_dann
UNVERIFIED (verify-flag-before-launch):

--- 60 (config, metal, seed) tuples enumerated ---
  [CONFIRMED] cfg=no_multi_task Ru seed=0  ->  ...
  [CONFIRMED] cfg=no_multi_task Ru seed=1  ->  ...
  ... (60 CONFIRMED tuples total, 0 UNVERIFIED)

DRY-RUN OK: 60 (config, metal, seed) tuples PLANNED, no GPU work executed.
  CONFIRMED runs (real launch): 60 (all 4 configs wired post-Wf-296C)
  UNVERIFIED runs (real launch): 0 (all flags verified)
```

---

## 6. SUMMARY

| Metric | Value |
|--------|-------|
| `n_flags_added` | **3** (`--no-kendall`, `--no-dann`, `--no-enforce-budget`) |
| `dry_run_verdict` | **PASS** (60/60 tuples CONFIRMED, CSV sanity OK, bash -n clean) |
| `g6_ready` | **YES** (real launch path can now sweep all 4 ablation configs × 5 metals × 3 seeds = 60 tuples PLANNED) |

### Side effects / follow-ups

- G4 (`scripts/launch_wf124_p3_g4.sh`) does NOT need Wf-296C treatment (no
  ablation dispatcher, just 14-pair full-sweep; Wf-283B flags already in scope
  of next Wf-283C follow-up if user OKs).
- G5 (`scripts/launch_wf124_p3_g5.sh`) similarly does NOT need Wf-296C treatment
  (single `--ablation-config mtl_baseline` mode, no per-config dispatcher).
  Note: G5 header §6 says `--no-enforce-budget` is used, but the actual
  invocation (line 85) doesn't pass it — this is a pre-existing inconsistency
  flagged for a future Wf-296D if user wants to align G5 to the kill_then_adapt
  directive.
- TODO-182 §3.4 per-lever attribution matrix can now be computed across all 4
  ablation levers (pre-Wf-296C: only 2 levers possible).

### Honest framing

- `--no-kendall` / `--no-dann` are verified to exist in
  `train_dmpnn_multitask_wasserstein.py` lines 1323 / 1331 (verified via
  `python ... --help | grep ...` and source inspection).
- MEASURED-vs-PROJECTED: this Wf-296C only integrated the flag WIRING; no GPU
  training has been executed. The first MEASURED G6 sweep result will arrive
  after `bash scripts/launch_wf124_p3_g6.sh` is run on RTX 5090 GPU-1 (planned
  for next session, gated on user OK per kill_then_adapt directive).
