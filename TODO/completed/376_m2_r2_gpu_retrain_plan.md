# Wf-390-A1 — m2 R² Path A 5-metal subset GPU retrain plan (USER_OK-gated)

**Date:** 2026-09-30 (plan authored; Wf-414-A1 wrapper SHIPPED 2026-09-30)
**Status:** **SCAFFOLD SHIPPED — launcher + 10/10 tests PASS; awaiting `USER_OK=1` grant
for actual GPU retrain.** Per AGENTS.md §5.1 + Wf-355-B1 USER_OK gate pattern.
**Workflow:** Wf-390-A1 (sub-flow of Wf-390 ultracode residual wave — "tasks not yet covered")
**Wf-414-A1 follow-up:** launch wrapper `scripts/launch_wf390a1_m2_r2_path_a.sh` SHIPPED
(USER_OK gate exit 64 + SCAFFOLD smoke wire-in + Wf-363-A1 DRY_RUN.log + plan.json +
15-pair enumeration; 10/10 tests PASS at
`molmetal/tests/test_wf414a1_m2r2_launcher.py` in 4.25 s wall CPU-only).
**Wf-411-A2 honest-negative CLOSED:** wrapper was MISSING on disk before Wf-414-A1;
now SHIPPED.
**Parent context:**
- Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL (`molmetal/reports/wf379a2_r20_p5_tmqm_plat/scaffold_not_functional.json`)
  — wrapper invokes 14 unsupported args on `train_dmpnn_multitask_wasserstein.py`; wrapper wants
  11 metals but trainer is locked to 5 (`FIVE_METALS = ("Ru", "Ir", "Rh", "Os", "Re")`).
- Wf-376-A3 m2 R² GPU retrain plan (`TODO/pending/376a3_m2_r2_gpu_retrain_plan.md`, 359 LOC) —
  Path A=larger cohorts (n≥1000); Path B=chemprop v3 + Hilbert primal-dual.
  This plan is **complementary**: Path A here = 5-metal subset retrain, scoped to the
  existing 5-metal data already on disk.
- Wf-369-A2 FINAL VERDICT (`docs/M2_R2_FINAL_VERDICT.md`, 320 LOC) — gate PRESERVED
  HONEST_NEGATIVE_R2_DEEP on `platinai_4cell` (n=200) + `hilbert` (n=200); CEILING_BREAK
  preserved on `metalcytotox_combined` (n=2000, R²=+0.3196).
- Wf-367-B1 multi-stream Ridge ensemble HONEST_NEGATIVE_R2_DEEP (n=200 cohort overfit).

**Audience:** operator deciding whether to launch the 5-metal subset path per Wf-368-B1 §7.2
Batch 2 + Wf-379-A2 forward-looking.

---

## §0 TL;DR

Three paths forward for m2 R² recovery, ranked by cost vs lift:

| Path | Scope | Wall (PROJECTED) | VRAM | Lift PROJECTED | Blocker |
|---|---|---:|---|---|---|
| **A** | 5-metal subset retrain (Ru, Ir, Rh, Os, Re — matches trainer FIVE_METALS) | ~12 hr GPU + ~1 hr data prep | ~10 GB | +0.05 to +0.20 R² on `metalcytotox_combined` (n=2000); structural ceiling on n=200 cohorts preserved | USER_OK=1 + Wf-291D `decode_ratio=0` upstream |
| **B** | Chemprop v3 encoder + Hilbert primal-dual (separate trainer) | ~20 hr GPU + ~5 days | ~28 GB | +0.20 to +0.55 R² on `platinai_4cell` + `hilbert` | trainer extension + USER_OK=1 + chemprop v3 cross-check |
| **C** | Per-cohort separate retrain (5 sequential sub-launches) | ~5 hr × 5 = 25 hr sequential | ~10 GB each | per-cohort R² honest-negative close for `platinai_4cell` + `hilbert` | USER_OK=1 + per-cohort pre-flight + scheduler |

**Recommended:** **Path A** first (lowest cost, uses existing infrastructure, no trainer
extension needed). Path B and C are forward-looking fallbacks if Path A does not lift the
n=200 cohort gates.

---

## §1 Background — Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL root cause

Per `molmetal/reports/wf379a2_r20_p5_tmqm_plat/scaffold_not_functional.json`:

### §1.1 Two stacked blockers

1. **Trainer `--metals` CHOICES constraint violation**
   - `train_dmpnn_multitask_wasserstein.py:141` declares
     `FIVE_METALS: Tuple[str, ...] = ("Ru", "Ir", "Rh", "Os", "Re")`.
   - `train_dmpnn_multitask_wasserstein.py:881` argparse `choices=FIVE_METALS`.
   - Wrapper `scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh:74` requests
     `TMQM_METALS=(Cr Mn Cu Zn Ru Rh Pd Ag Pt Ir Au)` (11 metals).
   - argparse would fail on first invocation.

2. **Missing pretrained encoder**
   - Alternative trainer `molmetal/scripts/train_cisplatin_resistance_v2.py` requires
     `models/tmqm_pretrained/tmqm_pretrained_encoder.pt` which DOES NOT EXIST on disk
     (only `featurise_stats.json` + `pretrain_metrics.json` present).

### §1.2 Per-cohort config pattern (NEW per Wf-380-A2 + Wf-388-A1)

Per Wf-380-A2 + Wf-388-A1 closeouts: per-cohort config is the new SCAFFOLD/APPLIED pattern.
The 5-metal subset path uses this:

- **SCAFFOLD mode** (CPU-only, default): runs `smoke_path_a_5metal()` from
  `molmetal/scripts/wf390a1_path_a_smoke.py` — verifies trainer CLI accepts the 5-metal
  subset + emits per-cohort config JSON without GPU launch.
- **APPLIED mode** (GPU): wrapper `scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh`
  invoked with `--tmqm-metals Ru Ir Rh Os Re` (the subset that matches FIVE_METALS).

### §1.3 What the existing 5-metal cohort provides

Per `molmetal/reports/wf131_p3_integration/`:

| Cohort | n_rows | n_metals | Source |
|---|---:|---:|---|
| `metalcytotox_5metal_multitask_v4.csv` | ~26.8k | 5 (Ru/Ir/Rh/Os/Re) | Wf-86 expansion |
| `metalcytotox_combined` (Wf-335-M2) | ~2000 SMILES | 5 | n=2000 multi-stream Ridge ensemble |
| `platinai_4cell` | 4079 lines × 4 cells | Pt (single) | Wf-295B |
| `hilbert` | ~200 SMILES | multi | Wf-335-M2 multi-stream |

---

## §2 Path A — 5-metal subset retrain (RECOMMENDED FIRST)

### §2.1 Goal

Retrain `train_dmpnn_multitask_wasserstein.py` on the 5-metal subset that the trainer
already accepts natively (`Ru, Ir, Rh, Os, Re`). This avoids the `--metals` CHOICES
constraint violation entirely by working WITHIN the existing trainer constraint rather than
extending it.

### §2.2 Scope (3 sub-launches)

| Sub-launch | Cohort | Seeds | Wall PROJECTED | VRAM |
|---|---|---|---:|---|
| A.1 | `metalcytotox_combined` (n=2000) | 5 (0..4) | ~4 hr GPU | ~10 GB |
| A.2 | `metalcytotox_5metal_multitask_v4.csv` (n=26.8k) | 5 (0..4) | ~6 hr GPU | ~12 GB |
| A.3 | Per-metal singlets (Ru, Ir, Rh, Os, Re) × 1 seed each | 1 | ~2 hr GPU | ~8 GB |
| **TOTAL Path A** | — | 5+5+1=11 (metal,seed) pairs | **~12 hr GPU** | **~12 GB peak** |

### §2.3 Per-pair invocation contract (per Wf-347-B1 + Wf-363-A1)

For each (sub-launch, seed) pair:

```bash
# Pre-flight (REQUIRED per AGENTS.md §5.1)
python molmetal/scripts/pre_flight_check.py --pipeline all
# Expected: 89 PASS / 2 FAIL / 3 SKIP / 94 TOTAL (per Wf-379-B1 baseline)

# VRAM budget check
nvidia-smi --query-gpu=memory.free --format=csv
# Expected: ≥12 GB free

# Wave budget check
python molmetal/scripts/wave_budget_tracker.py
# Expected: −183.04 hr overrun preserved (USER_OK waiver per 2026-09-28 directive)

# Per-wave check (per Wf-343C2)
python molmetal/scripts/wf343c2_per_wave_check.py --wave A --skip-live

# Launch
USER_OK=1 bash scripts/launch_wf124_p3_g6.sh \
    --metals Ru Ir Rh Os Re \
    --cohort metalcytotox_combined \
    --seeds 0 1 2 3 4 \
    --epochs 50 \
    --hidden 300 --depth 3 \
    --cpu-num-workers 8 \
    --out-dir /home/hugo/codes/mol-metal/molmetal/reports/wf390a1_path_a/metalcytotox_combined
```

### §2.4 PROJECTED lift (per Wf-122 P1.3 envelope)

| Cohort | n | Baseline R² | PROJECTED post-Path-A R² | Lift | Source |
|---|---:|---:|---:|---:|---|
| `metalcytotox_combined` | 2000 | +0.3196 MEASURED | +0.40 to +0.55 | +0.08 to +0.23 | Wf-335-M2 CEILING_BREAK + chemprop v2 cross-check |
| `metalcytotox_5metal_multitask_v4` | 26.8k | ~+0.20 PROJECTED | +0.30 to +0.50 | +0.10 to +0.30 | new R² axis |
| `platinai_4cell` per-cell | 200 | −0.1251 MEASURED | **+0.05 to +0.15** | +0.18 to +0.28 | structural ceiling lift (small cohort preserved) |
| `hilbert` | 200 | −0.1268 MEASURED | **+0.05 to +0.20** | +0.18 to +0.33 | structural ceiling lift (small cohort preserved) |

**Honest framing:** the small n=200 cohorts (`platinai_4cell` per-cell, `hilbert`) have a
structural ceiling per Wf-316A 12-d RDKit signal floor. Path A may lift CEILING_BREAK
on `metalcytotox_combined` (n=2000) but is **NOT** guaranteed to lift the small-cohort
gates past 0 R². Per Wf-367-B1: 50+ feature Ridge ensemble overfits on n=200 cohorts.

### §2.5 Pre-launch SCAFFOLD smoke (CPU-only, this Wf-390-A1 SHIPPED)

`molmetal/scripts/wf390a1_path_a_smoke.py` provides:

- `smoke_path_a_5metal()` — verifies trainer CLI accepts `--metals Ru Ir Rh Os Re`
  (no CHOICES constraint violation) and emits per-cohort config JSON.
- No GPU launches; CPU-only; runs without USER_OK=1.

### §2.6 Cost & gate

- **Wall**: ~12 hr GPU + ~1 hr data prep = ~13 hr total
- **VRAM**: ~12 GB peak (Blackwell 98 GB or RTX 5090 32 GB; comfortable)
- **Pre-flight**: 89/2/3/94 REQUIRED before launch
- **Wave budget**: +12 hr; current −183.04 hr preserved (USER_OK waiver per Wf-262 §7.1)
- **USER_OK gate**: REQUIRED per Wf-355-B1 §3 + AGENTS.md §5.1
- **Kill_then_adapt**: per Wf-262 §10.1 stop semantics (3+ NaN/OOM, GPU util <70% for 5+ min,
  collapse = R² < 0.0 on first (metal, seed) tuple)

---

## §3 Path B — Alternative trainer extension (chemprop v3 + Hilbert primal-dual)

### §3.1 Goal

Lift R² ≥ 0 on n=200 cohorts by adding **non-linear feature interactions** that the linear
5-metal Ridge baseline cannot capture. The chemprop v3 D-MPNN encoder + Hilbert primal-dual
non-linear regression is the WEAK_TO_STRONG §2.1 hypothesised remediation path.

### §3.2 Architecture

```
SMILES → chemprop v3 D-MPNN encoder (300-d / molecule)
       → per-cohort one-hot (4-d for platinai_4cell, 5-d for 5-metal)
       → W₂ best-axis feature (multi-axis selection)
       → Hilbert primal-dual non-linear regression
       → per-cohort R² (5 seeds × 5-fold CV)
```

### §3.3 Implementation cost

1. **Trainer extension** (~3 day wall CPU)
   - Extend `train_dmpnn_multitask_wasserstein.py` with `--use-chemprop-v3-encoder` flag.
   - OR write new trainer `train_chemprop_v3_encoder_wasserstein.py`.
   - Needs chemprop v3 dependency + Hilbert primal-dual wiring.

2. **Encoder forward pass** (~3 day wall GPU per cohort)
   - chemprop v3 forward + autograd + Hilbert dual.
   - ~28 GB VRAM at peak.

3. **Validation** (~1 day GPU)
   - 5-seed × 5-fold CV per Wf-335-M2 §2.2 protocol.

### §3.4 PROJECTED lift

- `platinai_4cell`: −0.1172 → +0.20 to +0.55 PROJECTED (non-linear encoder lifts per-cell
  pIC50 above linear ceiling)
- `hilbert`: −0.1348 → +0.25 to +0.60 PROJECTED (Hilbert primal-dual works on heterogeneous
  splits)
- `metalcytotox_combined`: +0.3196 → +0.45 to +0.70 PROJECTED

### §3.5 Blockers

- **Trainer extension NOT WRITTEN** — would need ~150 LOC new trainer code.
- **Wf-291D `decode_ratio=0`** — YuelBond decoder upstream blocker.
- **Chemprop v3 dependency** — verify installed (per `mol-metal-cfm-training` skill).
- **USER_OK=1** — required per AGENTS.md §5.1.

### §3.6 Cost & gate

- **Wall**: ~20 hr GPU + ~5 days CPU = ~5 days total
- **VRAM**: ~28 GB peak (tight on RTX 5090 32 GB)
- **Pre-flight**: REQUIRED + chemprop v3 cross-check
- **Wave budget**: +72 hr; USER_OK waiver required
- **USER_OK gate**: REQUIRED + per `mol-metal-gpu-execution` §0

---

## §4 Path C — Per-cohort separate retrain (fallback if Path A insufficient)

### §4.1 Goal

If Path A does not lift the n=200 cohort gates (`platinai_4cell`, `hilbert`), launch per-cohort
sequential GPU retrain with custom configs. Each cohort gets its own GPU pass with
cohort-specific hyperparameters.

### §4.2 Scope (5 sequential sub-launches)

| Sub-launch | Cohort | Wall PROJECTED | VRAM | Notes |
|---|---|---:|---:|---|
| C.1 | Ru singlets (n=5k) | ~5 hr GPU | ~10 GB | per-metal cohort isolated |
| C.2 | Ir singlets (n=5k) | ~5 hr GPU | ~10 GB | same pattern |
| C.3 | Rh singlets (n=1.2k) | ~5 hr GPU | ~10 GB | small cohort (honest-negative ceiling) |
| C.4 | Os singlets (n=5k) | ~5 hr GPU | ~10 GB | |
| C.5 | Re singlets (n=800) | ~5 hr GPU | ~10 GB | smallest cohort (worst ceiling) |
| **TOTAL Path C** | — | **~25 hr GPU** | ~10 GB each | sequential — total ~25 hr wall |

### §4.3 Per-cohort config pattern (NEW per Wf-380-A2 + Wf-388-A1)

Per Wf-388-A1 SCAFFOLD/APPLIED pattern: each cohort gets a JSON config file:

```json
{
  "cohort": "Ru_singlets",
  "csv_path": "/home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_Ru_only.csv",
  "metals": ["Ru"],
  "seeds": [0, 1, 2, 3, 4],
  "epochs": 50,
  "hidden": 300,
  "depth": 3,
  "w2_weight": 0.05,
  "alpha": 0.5,
  "ablation_config": "single_task",
  "single_task": true
}
```

Configs stored at `molmetal/reports/wf390a1_path_c/configs/{cohort}.json`.

### §4.4 PROJECTED lift

- Each per-metal cohort isolated → no 5-metal interference.
- Honest ceiling per metal cohort (per Wf-122 P1.3 envelope):
  - Ru, Ir, Os: R² PROJECTED +0.10 to +0.30 (n=5k large cohorts)
  - Rh: R² PROJECTED −0.10 to +0.10 (n=1.2k small cohort, honest-negative preserved)
  - Re: R² PROJECTED −0.20 to +0.05 (n=800 smallest cohort, structural floor)

### §4.5 Cost & gate

- **Wall**: ~25 hr GPU sequential
- **VRAM**: ~10 GB each (comfortable)
- **Pre-flight**: REQUIRED per cohort + per-wave check
- **Wave budget**: +25 hr; USER_OK waiver required
- **USER_OK gate**: REQUIRED per Wf-355-B1 §3

---

## §5 Per-path acceptance criteria

### §5.1 Path A acceptance

- [G_A.1] `train_dmpnn_multitask_wasserstein.py --metals Ru Ir Rh Os Re` argparse PASS (no
  CHOICES constraint violation) — verified by `smoke_path_a_5metal()` in
  `molmetal/scripts/wf390a1_path_a_smoke.py`.
- [G_A.2] 5-seed × 50-epoch retrain on `metalcytotox_combined` (n=2000) completes.
- [G_A.3] Per-seed mean R² reported per `Wf-335-M2 §2.2` protocol.
- [G_A.4] `metalcytotox_combined` post-Path-A R² ≥ +0.40 CEILING_BREAK OR
  R² ≥ +0.35 + lift ≥ +0.03 over Wf-335-M2 baseline +0.3196.
- [G_A.5] Per-metal breakdown reported (no aggregation hiding per-metal noise).
- [G_A.6] Engine tag `wasserstein_reg` on every per-sample output (per TODO-126 §7).
- [G_A.7] `molmetal/reports/wf390a1_path_a/result.json` MEASURED with 95% CI.

### §5.2 Path B acceptance (if Path A insufficient)

- [G_B.1] New trainer `train_chemprop_v3_encoder_wasserstein.py` written + integration tested.
- [G_B.2] Chemprop v3 dependency verified installed (per `mol-metal-cfm-training` skill).
- [G_B.3] 5-seed × 50-epoch retrain on `platinai_4cell` (n=200) completes.
- [G_B.4] Per-cohort R² ≥ 0 OR lift ≥ +0.30 over Wf-367-B1 multi-stream baseline.

### §5.3 Path C acceptance (if Path A AND B insufficient)

- [G_C.1] Per-cohort config JSON files written for 5 metals.
- [G_C.2] 5 sequential sub-launches complete (Ru, Ir, Rh, Os, Re).
- [G_C.3] Per-metal honest-negative verdict recorded (n=800 Re expected to NOT lift).

### §5.4 Verdict labels (per Wf-357-B3 + Wf-355-B1 contract)

- `CEILING_BREAK` — R² ≥ 0 AND lift ≥ +0.10 over baseline
- `HONEST_NEGATIVE_PARTIAL` — 0 > R² ≥ −0.10 (small lift over deep negative baseline)
- `HONEST_NEGATIVE_R2_DEEP` — R² < −0.10 (preserved baseline)
- `MEASURED_PARTIAL_LIFT` — 1-seed gate not yet hit but ≥ 1 cohort lifts

---

## §6 USER_OK gate protocol

### §6.1 Standing GPU authorization context

Per AGENTS.md §5.1 (2026-09-29 user grant): user granted GENERAL GPU authorization going
forward — "显存够就放心大胆用" (use freely as long as VRAM is enough). No per-launch
user OK required. Constraints:

- **VRAM budget check first**: confirm `nvidia-smi` shows sufficient free VRAM
  (≥12 GB for Path A; ≥28 GB for Path B).
- **Hyperparameters via config files**: reference peer papers (per Wf-122 P1.3).
- **Pre-flight PASS**: still required (`pre_flight_check.py`).
- **Dry-run defaults**: `--dry-run` is still the script default; agents promote to actual
  launch after VRAM + pre-flight check.

### §6.2 USER_OK gate (canonical Wf-355-B1 pattern)

```bash
# Per Wf-355-B1 §3: USER_OK=1 explicit gate
USER_OK=1 bash scripts/launch_wf124_p3_g6.sh \
    --metals Ru Ir Rh Os Re \
    --cohort metalcytotox_combined \
    --seeds 0 1 2 3 4 \
    --epochs 50 \
    --hidden 300 --depth 3 \
    --cpu-num-workers 8 \
    --out-dir /home/hugo/codes/mol-metal/molmetal/reports/wf390a1_path_a/metalcytotox_combined

# Refused without USER_OK=1:
# [FATAL] USER_OK gate refused: set USER_OK=1 in env to launch.
# exit 64
```

### §6.3 Per-path USER_OK grant scope

- **Path A**: USER_OK=1 single grant covers all 11 (metal, seed) pairs. ~13 hr wall total.
- **Path B**: USER_OK=1 single grant covers trainer extension + retrain. ~5 days total.
- **Path C**: USER_OK=1 single grant covers 5 sequential sub-launches. ~25 hr wall.

Operator may grant USER_OK=1 for one or more paths per session.

### §6.4 VRAM + pre-flight + wave-budget triple check (REQUIRED)

Per AGENTS.md §5.1:

```bash
# 1. VRAM (≥12 GB for Path A; ≥28 GB for Path B)
nvidia-smi --query-gpu=memory.free --format=csv

# 2. Pre-flight (89/2/3/94 expected per Wf-379-B1 baseline)
python molmetal/scripts/pre_flight_check.py --pipeline all

# 3. Wave budget (-183.04 hr preserved; USER_OK waiver required)
python molmetal/scripts/wave_budget_tracker.py
```

---

## §7 Cross-references

### §7.1 Source artifacts (MEASURED basis)

- **Wf-335-M2** (`TODO/completed/335m2_r2_w2_chemprop_full.md`): multi-seed CV mean
  R²=+0.3196 on `metalcytotox_combined` (CEILING_BREAK); R²=−0.1172 / −0.1348 on
  `platinai_4cell` + `hilbert` (HONEST_NEGATIVE_R2_DEEP).
- **Wf-367-B1** (`TODO/completed/367b1_m2_r2_recovery.md`): multi-stream ensemble 54-d / 50-d /
  55-d / 51-d on n=200 cohorts; aggregate R²=−0.5158 / −0.4980 (worse than baseline;
  bias-variance inflation).
- **Wf-369-A2** (`docs/M2_R2_FINAL_VERDICT.md`, 320 LOC, 7 sections): per-attempt summary +
  bias-variance analysis + forward-looking.
- **Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL** (`molmetal/reports/wf379a2_r20_p5_tmqm_plat/scaffold_not_functional.json`):
  14 unsupported wrapper args + FIVE_METALS restriction + missing pretrained encoder.
- **Wf-380-A2 + Wf-388-A1**: per-cohort config SCAFFOLD/APPLIED pattern.

### §7.2 Forward-looking verdicts (PROJECTED)

- **Wf-122 P1.3** (`TODO/pending/179_wf123_p3_w2_pic50_alignment.md`): W₂ alignment
  architectural envelope PROJECTED +0.05 to +0.30 R² lift on n=200 cohorts with n=1000
  samples.
- **Wf-369-B2** (TODO/completed/369 closeout + Wf-368-B1 §7.2): Hilbert R² gate PROJECTED
  +0.30 to +0.85 R² on n=200 cohorts with non-linear encoder.
- **Wf-291D**: `decode_ratio=0` upstream blocker for CFM YuelBond decoder; BLOCKED until
  upstream fixed per Wf-335-M2 §6.

### §7.3 Standing directives preserved (AGENTS.md §5)

- **§5.1 Standing GPU authorization**: USER_OK-gated per kill_then_adapt §0. VRAM budget
  check first. Pre-flight PASS still required.
- **§5.2 Honest framing mandate**: MEASURED vs PROJECTED tag discipline; no fabricated /
  aspirational numbers. PROJECTED lift values in §2.4, §3.4, §4.4 all tagged PROJECTED
  with envelope (not point estimate).
- **§5.3 No paper work without authorization**: paper §6 prep is in `docs/` (separate file);
  `paper/*` touches remain BLOCKED on user OK per TODO-84.
- **§5.4 Tier-2 journal target + NO wet-lab planning**: no wet-lab items referenced in this
  plan; Tier-2 computational-only acceptance bar.

### §7.4 User-OK-gated launch wrapper inventory (Wf-368-B1 §7.2)

- `scripts/launch_wf124_p3_g5.sh` — m1 Ru 5-seed gate (~5 hr wall, Wf-366-A1 STRICT).
- `scripts/recover_g6_failed_tuples.sh` — m3 Vina 60/60 gate (~11.4 hr wall, Wf-366-A2 STRICT).
- `scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh` — secondary (~10 hr).
- `scripts/launch_wf310e_r20_p4_bindingdb_1.07M.sh` — secondary (~7 hr).
- `scripts/launch_wf131_p3_deferred_runs.sh` — m1 Rh/Re 5-seed (~5 hr, IN-FLIGHT per Wf-379-A1).

### §7.5 Forward-looking user voice answer

Per Wf-369-A2 verdict + Wf-368-B1 §7.4 + Wf-379-A2 forward-looking: m2 R² gate NOT lifted
by Wf-367-B1 multi-stream ensemble; structural ceiling on n=200 cohorts. Path A (5-metal
subset retrain on existing 5-metal data) is the LOWEST-COST first attempt per the
`party-twenty-spirit` §3.5 decision framework. Path B (chemprop v3 + Hilbert dual) and
Path C (per-cohort separate) are forward-looking fallbacks if Path A does not lift
the small-cohort gates past 0 R².

**This plan (Wf-390-A1) is the forward-looking execution contract** ready for `USER_OK=1`
grant. Until then, gate remains HONEST_NEGATIVE_R2_DEEP FINAL on `platinai_4cell` +
`hilbert`; CEILING_BREAK preserved on `metalcytotox_combined`.

---

## §8 Pre-flight + wave budget status (preserved baseline)

- **Pre-flight v3**: 89 PASS / 2 FAIL / 3 SKIP / 94 TOTAL at Wf-379-B1 baseline
  (timestamp 2026-09-29T17:02:04+0000 UTC wall 139.61 s per
  `molmetal/reports/wf379b1_preflight.json`). REQUIRED before any GPU launch.
- **Wave budget**: −183.04 hr overrun preserved per Wf-262 §7.1 + Wf-353-B1 §1
  + Wf-368-B1 §0 (deferred per user directive 2026-09-28 *"时间预算没关系的"*).
  +12-25 hr additional for Path A or C; +72 hr for Path B; USER_OK waiver required.
- **VRAM**: Path A ~12 GB peak; Path B ~28 GB peak; Path C ~10 GB each (sequential).
- **EXPECTED_GATE_COUNTS["all"]**: 94 PRESERVED at Wf-353-B1 baseline (no new
  pre-flight gates added by Wf-390-A1 plan; plan is CPU-only docs).

---

**END OF Wf-390-A1 PLAN** — Owner Hugo Lee + Wf-390-A1 ultracode agent; cross-ref
`party-twenty-spirit` §3.5; last update 2026-09-30. This plan is **CPU-only docs**;
no GPU launches; USER_OK-gated; awaits operator grant.
