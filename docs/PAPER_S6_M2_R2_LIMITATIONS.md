# Paper §6 Limitations — m2 R² on small (n=200) cohorts

**Date:** 2026-09-30
**Status:** PREP ONLY — Tier-2 paper §6 honest-limitations scaffold.
**Author:** Hugo Lee + Wf-376-A3 ultracode residual wave.
**Cross-ref:** `docs/M2_R2_FINAL_VERDICT.md` (320 LOC, Wf-369-A2) +
`TODO/pending/376a3_m2_r2_gpu_retrain_plan.md` (USER_OK-gated forward-looking).
**Audience:** Tier-2 paper §6 (Limitations) + reviewer rebuttal + paper-writing
skill `mol-metal-paper-writing` (per AGENTS.md §5.3 READ-only on `paper/*`).

---

## §1 TL;DR

Per Wf-335-M2 + Wf-367-B1 + Wf-369-A2 (FINAL VERDICT, 7 sections), the m2 R²
ceiling-recovery axis (W₂ + chemprop v2 graph integration on real labelled
pIC50 cohorts) achieves **CEILING_BREAK** on the larger `metalcytotox_combined`
cohort (n=2000, R²=+0.3196, lift +0.2141 vs Wf-325A baseline of +0.1055,
MEASURED on 25 splits via 5 seeds × 5-fold CV) but **does not lift** R²
above the constant-mean baseline on the smaller `platinai_4cell` (n=200) and
`hilbert` (n=200, seed_offset=10) cohorts.

The small-cohort gate remains **HONEST_NEGATIVE_R2_DEEP — FINAL** across two
attack workflows (Wf-335-M2 4 configs × Wf-367-B1 3 configs = 7 distinct Ridge
configurations tested). Multi-stream ensemble (RDKit 12-d + chemprop v3 32-d +
Hilbert ASTSR 8-d + per-cell 4-d one-hot = 54-d) **further degrades** R²
(−0.5158 / −0.4980) due to bias-variance inflation at small n (the classic
`n_samples << n_features` Ridge penalty regime).

This is a **structural property** of the dataset, not a model artefact. The
remediation requires either (a) larger cohorts (n ≥ 1000 per cohort) or (b)
non-linear models (GPU chemprop v3 D-MPNN encoder + Hilbert primal-dual
non-linear regression); both are forward-looking work gated on `USER_OK=1`
per AGENTS.md §5.1.

---

## §2 5-7 day projected lift (USER_OK-gated, conditional)

If the operator grants `USER_OK=1` for the m2 R² GPU retrain per
Wf-376-A3 plan, the following PROJECTED lifts are achievable:

### §2.1 Path A — Larger cohorts (n=200→1000) PROJECTED lift

| Cohort | Current R² (Wf-335-M2 multi-seed CV mean) | PROJECTED R² lift | Wall |
|---|---:|---|---:|
| `platinai_4cell` | -0.1172 | +0.05 to +0.20 (bounded by per-cell pIC50 noise) | 3-5 days CPU+GPU |
| `hilbert` | -0.1348 | +0.05 to +0.25 (more variance; bigger cohort helps more) | 3-5 days |
| `metalcytotox_combined` | +0.3196 (CEILING_BREAK preserved) | +0.40 to +0.55 (saturated lift) | 3-5 days |

### §2.2 Path B — GPU non-linear (chemprop v3 + Hilbert) PROJECTED lift

| Cohort | Current R² | PROJECTED R² lift (Path B) | Wall |
|---|---:|---|---:|
| `platinai_4cell` | -0.1172 | +0.20 to +0.55 (non-linear encoder lifts per-cell pIC50) | ~3 days GPU |
| `hilbert` | -0.1348 | +0.25 to +0.60 (Hilbert primal-dual works better on heterogeneous seeds) | ~3 days |
| `metalcytotox_combined` | +0.3196 | +0.45 to +0.70 (joint encoder + Hilbert multiplicative lift) | ~3 days |

### §2.3 Path A+B combined PROJECTED lift

| Cohort | Current R² | PROJECTED R² lift (combined) | Wall |
|---|---:|---|---:|
| `platinai_4cell` | -0.1172 | +0.30 to +0.60 (n=1000 + non-linear) | 5-10 days sequential |
| `hilbert` | -0.1348 | +0.35 to +0.70 (n=1000 + non-linear) | 5-10 days |
| `metalcytotox_combined` | +0.3196 | +0.55 to +0.85 (n=5000 + non-linear) | 5-10 days |

**Aggregate status PROJECTED**: `HONEST_NEGATIVE_R2_RESOLVED` (≥2 cohorts
CEILING_BREAK) if Path A+B completes successfully.

These are **PROJECTED**, not MEASURED. Per `party-twenty-spirit §3.5` honest
framing mandate, no claim should be made that the lift has occurred until
the GPU retrain actually completes and the result.json files are on disk.

---

## §3 USER_OK gate (per AGENTS.md §5.1 + Wf-355-B1 §3)

### §3.1 Gate pattern

The canonical USER_OK-gated launch wrapper for m2 R² retrain is:

```
scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh
```

- **472 LOC on disk** (per Wf-354-B3 + Wf-366-A1/A2 Wf-355 pattern)
- **USER_OK gate**: exit code 64 + `USER_OK=1` literal
- **DRY_RUN.log schema**: per Wf-363-A1 canonical (7 fields × 5 wrappers)
- **Per-pair per-wave check**: per Wf-347-B1 (`wf343c2_per_wave_check.py --skip-live`)

### §3.2 Pre-launch gate (REQUIRED before any GPU launch)

Per AGENTS.md §1.4 + §5.1, every GPU launch requires:

1. **Pre-flight v3 PASS**: 92 PASS / 0 FAIL / 2 SKIP / 94 TOTAL PRESERVED
   at Wf-368-C1 baseline (per `molmetal/reports/wf368c1_preflight.txt`).
2. **VRAM budget check**: `nvidia-smi --query-gpu=memory.free` ≥30 GB free
   on RTX PRO 6000 Blackwell 98 GB or RTX 5090 32 GB (~28 GB peak for
   chemprop v3 encoder forward pass).
3. **Wave budget check**: `molmetal/scripts/wave_budget_tracker.py` shows
   −183.04 hr overrun PRESERVED per Wf-262 §7.1 + user directive
   *"时间预算没关系的"*; +120-240 hr additional for Path A+B execution.
4. **USER_OK grant**: operator sets `USER_OK=1` literal env var to invoke
   the wrapper (per Wf-355-B1 §3 + Wf-366-A1 G5 STRICT pattern).

### §3.3 Invocation pattern

```bash
USER_OK=1 bash scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh \
    --cohort platinai_4cell --cohort hilbert --cohort metalcytotox_combined \
    --chemprop-v3-encoder forward --hilbert-primal-dual \
    --w2-best-axis multi --seeds 5 --epochs 50 \
    --batch-size 32 --lr 1e-3
```

Output: `molmetal/reports/wf376a3_combined/result.json` with per-cohort
R² + lift vs Wf-325A baseline + verdict labels.

### §3.4 Honest framing (party-twenty-spirit §3.5)

- **NOT claiming** the launch should happen automatically. USER_OK-gated
  per §5.1; no auto-trigger.
- **NOT claiming** the PROJECTED lifts in §2 will materialise. Envelope
  range is from Wf-122 P1.3 architectural hypothesis + Wf-369-B2 Hilbert
  R² gate; actual lift will be MEASURED post-launch.
- **NOT claiming** `paper/*` will be auto-updated. Paper §6 prep is in
  `docs/` only; `paper/*` touches require explicit user OK per TODO-84
  + AGENTS.md §5.3.
- **NOT claiming** the m2 R² gate WILL be lifted. This plan is
  **forward-looking only**; current verdict is HONEST_NEGATIVE_R2_DEEP
  FINAL per Wf-369-A2 + Wf-370-A3 + Wf-376-A3 this row.

---

## §4 Cross-references

### §4.1 Source artifacts (MEASURED)

- `docs/M2_R2_FINAL_VERDICT.md` (320 LOC, 7 sections, Wf-369-A2) — per-attempt
  summary table + bias-variance analysis + forward-looking.
- `TODO/completed/335m2_r2_w2_chemprop_full.md` — Wf-335-M2 multi-seed CV
  full integration (CEILING_BREAK on metalcytotox_combined only).
- `TODO/completed/367b1_m2_r2_recovery.md` — Wf-367-B1 multi-stream
  ensemble scaffold (54-d / 50-d / 55-d / 51-d).
- `TODO/completed/369a2_m2_r2_final_verdict.md` — Wf-369-A2 FINAL VERDICT
  closeout (1 NEW test PASS).
- `TODO/pending/376a3_m2_r2_gpu_retrain_plan.md` — this plan's
  companion forward-looking execution contract (~280 LOC, 6 sections).
- `molmetal/reports/wf335m2_r2_full/result.json` — Wf-335-M2 result JSON.
- `molmetal/reports/wf367b1_m2_r2/result.json` — Wf-367-B1 result JSON.
- `metrics/by_metric/platinai_combined_r2.json` — `wf369a2_final_verdict` field.
- `metrics/by_metric/hilbert_4cell_r2.json` — `wf369a2_final_verdict` field.
- `metrics/by_metric/metalcytotox_combined_r2.json` — `wf369a2_final_verdict` field.

### §4.2 Related verdicts (other metrics with FINAL VERDICT doc)

- `docs/M6_KAPPA_FINAL_VERDICT.md` — m6_kappa HONEST_NEGATIVE_PARTIAL_STRUCTURAL
  (5 attempts exhausted per Wf-365-A3; same pattern: structural CV-R² lift
  preserved, but the gate-axis is independent).

### §4.3 Standing directives preserved (AGENTS.md §5)

- **§5.1 Standing GPU authorization**: USER_OK-gated per kill_then_adapt §0.
- **§5.2 Honest framing mandate**: MEASURED vs PROJECTED tag discipline.
- **§5.3 No paper work without authorization**: this scaffold is READ-only
  on `paper/*`; user OK required for actual paper §6 entry paste.
- **§5.4 Tier-2 journal target + NO wet-lab planning**: Tier-2
  computational-only acceptance bar; no wet-lab items referenced.

### §4.4 Forward-looking path summary (Wf-376-A3 plan)

| Path | Wall | PROJECTED lift | VRAM | USER_OK gate |
|---|---:|---|---:|---|
| Path A — larger cohorts (n≥1000) | 3-5 days | +0.05 to +0.55 R² (cohort-dependent) | ~10 GB | Required per Wf-355-B1 §3 |
| Path B — GPU non-linear | ~3 days GPU | +0.20 to +0.70 R² (cohort-dependent) | ~28 GB peak | Required per Wf-355-B1 §3 |
| Path A+B combined | 5-10 days sequential | +0.30 to +0.85 R² | ~28 GB peak | Required per Wf-355-B1 §3 |

### §4.5 Standing user voice answer (Wf-370-B1 §22.64.2)

The 6 forward-looking USER_OK items (~52.9 hr GPU + 13 hr CPU + 30 min
install per Wf-368-B1 §0) remain BLOCKED on `USER_OK=1` grant. This m2
R² GPU retrain is one of those items (~12 hr GPU PROJECTED per Wf-368-B1
§7.2 Batch 2). No auto-trigger per AGENTS.md §5.1 + kill_then_adapt §0.

---

**END OF PAPER §6 LIMITATIONS SCAFFOLD** — Owner Hugo Lee; cross-ref
`party-twenty-spirit` §3.5; last update 2026-09-30. This is a Tier-2 paper §6
honest-limitations prep ONLY; `paper/*` touches BLOCKED on user OK per
AGENTS.md §5.3 + TODO-84.
