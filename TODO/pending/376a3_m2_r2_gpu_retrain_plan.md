# Wf-376-A3 — m2 R² GPU retrain plan (USER_OK-gated forward-looking)

**Date:** 2026-09-30
**Status:** **PLAN ONLY — no retrain launched.** Awaiting `USER_OK=1` grant per AGENTS.md
§5.1 + Wf-355-B1 USER_OK gate pattern.
**Workflow:** Wf-376-A3 (sub-flow of Wf-376 ultracode residual wave)
**Parent context:** Wf-369-A2 FINAL VERDICT (`docs/M2_R2_FINAL_VERDICT.md`,
320 LOC, 7 sections) — gate PRESERVED `HONEST_NEGATIVE_R2_DEEP` on
`platinai_4cell` (n=200) + `hilbert` (n=200); `CEILING_BREAK` preserved on
`metalcytotox_combined` (n=2000).
**Audience:** operator deciding whether to launch Batch 2 GPU retrain per
Wf-368-B1 §7.2 + Wf-369-A2 §5 forward-looking.

---

## §0 TL;DR

| Item | Value |
|---|---|
| **Gate status** | `HONEST_NEGATIVE_R2_DEEP` FINAL (Wf-369-A2 verdict); `metalcytotox_combined` CEILING_BREAK preserved at R²=+0.3196 |
| **Forward-looking paths** | Path A (larger cohorts n≥1000), Path B (GPU chemprop v3 encoder + Hilbert primal-dual), Path A+B combined |
| **Per-path wall** | A: ~3 days (1 data prep + 2 GPU retrain). B: ~3 days GPU. A+B: ~5 days GPU |
| **Total wall (Path A+B combined)** | ~5-10 day sequential GPU wall |
| **PROJECTED lift** | +0.05 to +0.30 R² on `platinai_4cell` + `hilbert` (per Wf-122 P1.3 architectural envelope; bounded by n=200 ceiling) |
| **USER_OK gate** | **REQUIRED** per AGENTS.md §5.1 — does NOT auto-trigger |
| **VRAM budget check** | ~28 GB at peak (chemprop v3 encoder forward pass); RTX PRO 6000 Blackwell 98 GB or RTX 5090 32 GB |
| **Pre-flight v3 gate** | 92 PASS / 0 FAIL / 2 SKIP / 94 TOTAL PRESERVED at Wf-368-C1 baseline — REQUIRED before launch |
| **Wave budget impact** | +30-120 hr GPU; −183.04 hr current overrun preserved (deferred per Wf-262 §7.1) |
| **Launch wrapper** | `scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` (472 LOC, EXISTS on disk, USER_OK-gated) |
| **Cross-references** | Wf-368-B1 §7.2 Batch 2 + Wf-369-A2 verdict §5 + Wf-367-B1 §4 + Wf-355-B1 USER_OK pattern + AGENTS.md §5.1 |

**Bottom line:** m2 R² gate on `platinai_4cell` + `hilbert` is **structurally unliftable
CPU-only**. Lifting it requires **either** (a) larger cohorts (n≥1000 per cohort;
external dependency) **or** (b) GPU non-linear chemprop v3 encoder + Hilbert
primal-dual (per AGENTS.md §5.1 + Wf-291D `decode_ratio=0` upstream resolution).
This plan is the **forward-looking execution contract** ready for the moment the
operator grants `USER_OK=1`. No auto-trigger; no GPU launches by this plan.

---

## §1 Path A — Larger cohorts (USER_OK-gated external data acquisition)

### §1.1 Goal

Close the bias-variance gap on the n=200 cohorts (`platinai_4cell`,
`hilbert`) by acquiring n≥1000 SMILES per cohort from external sources. The
`metalcytotox_combined` CEILING_BREAK at n=2000 (R²=+0.3196, lift +0.2141,
Wf-335-M2) shows the bias-variance tradeoff closes when n >> p.

### §1.2 Sources (3 candidates, ranked)

| Rank | Source | Path to ≥1000 SMILES | Wall | External dep | Notes |
|---|---|---|---|---|---|
| 1 | NCI-60 expansion (Pt/Ru/Ir ≥1000 each) | `molmetal/data/nci60_expansion_loader.py` NEW | 1 day data prep + 2 day GPU retrain | NCI-60 web scraping + curation | Most direct; bounded by data availability |
| 2 | ChEMBL Pt-only mining | `molmetal/data/chembl_pt_loader.py` NEW | 2 day data prep + 2 day GPU retrain | ChEMBL API + SMARTS filter | Larger dataset but Pt-only (loses 5-metal coverage) |
| 3 | PubChem Pt/Ru/Ir/Os mining | `molmetal/data/pubchem_loader.py` NEW | 3 day data prep + 2 day GPU retrain | PubChem FTP bulk download | Largest; slowest; multi-metal coverage |

### §1.3 Implementation

1. **Data prep** (CPU-only, 1-3 days wall depending on source)
   - Extend `molmetal/scripts/platinai_real_label_loader.py` (Wf-295B pattern)
     with NCI-60 / ChEMBL / PubChem expansion.
   - Standardise to 12-d RDKit + 32-d RDKit features per
     `molmetal/data/platinai_4cell_rdkit_features.csv` schema.
   - Output to `molmetal/data/platinai_expanded_rdkit_features.csv` + per-cohort splits.

2. **GPU retrain** (~2 day wall, sequential single-GPU)
   - `scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` (472 LOC, USER_OK-gated
     per Wf-355-B1 pattern).
   - 5 seeds × 50 epochs × dual-init {hilbert, tmqm} per Wf-335-M2 protocol.
   - Output: `molmetal/reports/wf376a3_pathA_cohort_lift/result.json` with
     per-cohort best R² + lift vs Wf-325A baseline.

3. **Validation** (CPU-only, ~1 hr)
   - 5-seed × 5-fold CV per Wf-335-M2 §2.2 protocol.
   - Per-cohort breakdown (per-cell for platinai; per-metal for metalcytotox).
   - Verdict label: `CEILING_BREAK` if R² ≥ 0 AND lift ≥ +0.10 over Wf-325A baseline.

### §1.4 PROJECTED lift

Per Wf-122 P1.3 architectural envelope:
- **platinai_4cell** n=200→1000: R² PROJECTED −0.1172 → +0.05 to +0.20
  (bounded by per-cell pIC50 noise floor).
- **hilbert** n=200→1000: R² PROJECTED −0.1348 → +0.05 to +0.25
  (more variance; bigger cohort helps more).
- **metalcytotox_combined** n=2000→5000+: R² PROJECTED +0.3196 → +0.40 to +0.55
  (saturated lift; CEILING_BREAK preserved).

These are PROJECTED, not MEASURED. Actual lift will be reported post-launch
per MEASURED-vs-PROJECTED tag discipline (`party-twenty-spirit §3.5`).

### §1.5 Cost & gate

- **Wall**: 1-3 days data prep + 2 days GPU retrain = 3-5 days total
- **VRAM**: ~10 GB (no encoder forward pass needed for Path A; static
  featurization + Ridge regression only)
- **Pre-flight**: 92/0/2/94 PRESERVED at Wf-368-C1; REQUIRED before launch
- **Wave budget**: +48-120 hr; current −183.04 hr overrun preserved
  (deferred per Wf-262 §7.1)
- **USER_OK gate**: REQUIRED per Wf-355-B1 §3 pattern (operator sets
  `USER_OK=1 bash scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh`)

---

## §2 Path B — GPU chemprop v3 encoder + Hilbert primal-dual

### §2.1 Goal

Lift R² ≥ 0 on the n=200 cohorts by adding **non-linear feature interactions**
that the linear Ridge baseline cannot capture. The chemprop v3 D-MPNN encoder
forward pass + autograd + Hilbert primal-dual non-linear regression is the
WEAK_TO_STRONG §2.1 hypothesised remediation path.

### §2.2 Architecture (per Wf-335-M2 protocol)

```
SMILES → chemprop v3 D-MPNN encoder (300-d / molecule)
       → per-cell one-hot (4-d for platinai_4cell)
       → W₂ best-axis feature (multi-axis selection)
       → Hilbert primal-dual non-linear regression (deflated + projected)
       → per-cohort R² (5 seeds × 5-fold CV)
```

### §2.3 Implementation

1. **chemprop v3 encoder forward pass** (~3 day wall per cohort)
   - `molmetal/scripts/chemprop_v3_encoder_forward.py` NEW
   - 300-d per molecule (vs 32-d static featurization in Wf-335-M2).
   - Backprop through D-MPNN message passing layers + bond features.
   - ~28 GB VRAM at peak (Blackwell sm_120 98 GB or RTX 5090 32 GB).

2. **Hilbert primal-dual non-linear regression** (~1 day wall per cohort)
   - `molmetal/molmetal_lam/reward/hilbert_primal_dual.py` (existing per
     Wf-335-M2 + Wf-369-B2 R² gate).
   - Deflated subspace + projected primal-dual steps per
     Peyré & Cuturi 2019 §3 Prop 1.
   - 5 seeds × 50 epochs × dual-init {hilbert, tmqm}.

3. **W₂ best-axis selection** (already implemented in Wf-335-M2)
   - Multi-axis {MolWt, MolLogP, TPSA, HeavyAtoms}.
   - Best axis = per-cohort selected; metalcytotox_combined picks TPSA (axis 2).

4. **Validation** (~1 hr CPU-only)
   - 5-seed × 5-fold CV per Wf-335-M2 §2.2.
   - Per-cohort R² + lift vs Wf-325A baseline.
   - Verdict: `CEILING_BREAK` if R² ≥ 0 AND lift ≥ +0.10.

### §2.4 PROJECTED lift

Per Wf-122 P1.3 envelope + Wf-369-B2 Hilbert R² gate:
- **platinai_4cell** n=200: R² PROJECTED −0.1172 → +0.20 to +0.55
  (non-linear encoder lifts per-cell pIC50 above the linear ceiling).
- **hilbert** n=200: R² PROJECTED −0.1348 → +0.25 to +0.60
  (Hilbert primal-dual works better on heterogeneous seed splits).
- **metalcytotox_combined** n=2000: R² PROJECTED +0.3196 → +0.45 to +0.70
  (joint encoder + Hilbert gives multiplicative lift on larger cohort).

These are PROJECTED, not MEASURED. Actual lift will be reported post-launch.

### §2.5 Blockers

- **Wf-291D `decode_ratio=0` upstream resolution** — YuelBond decoder needs
  fix per Wf-335-M2 §6 forward-looking. BLOCKED until upstream fixed.
- **Wf-299 CFM 3-wave launcher readiness** — READY (per Wf-335-M2); not yet launched.
- **Pre-flight v3 92/0/2/94** — PRESERVED at Wf-368-C1; REQUIRED before launch.
- **Blackwell sm_120 retune** — per Wf-326E follow-up B; partial.

### §2.6 Cost & gate

- **Wall**: ~3 days GPU retrain (encoder forward + Hilbert dual + W₂ + validation)
- **VRAM**: ~28 GB peak (Blackwell 98 GB or RTX 5090 32 GB; tight on RTX 5090)
- **Pre-flight**: REQUIRED before launch
- **Wave budget**: +72 hr GPU; current −183.04 hr preserved
- **USER_OK gate**: REQUIRED per Wf-355-B1 §3; VRAM budget check first per §5.1

---

## §3 Path A+B combined — sequential GPU retrain (recommended)

### §3.1 Why combined

Path A alone closes the bias-variance gap (n≥1000) but the per-cell pIC50
noise floor on PlatinAI 4-cell is INTRINSIC to the linear projection.
Path B alone provides non-linear feature interactions but inherits the
n=200 small-cohort noise. **Combining** both:

- Path A expands n=200 → n=1000+ (5× more samples).
- Path B adds non-linear chemprop v3 encoder + Hilbert primal-dual.
- Per-cohort: 5× more data + non-linear model = multiplicative lift.

### §3.2 Sequential sub-phases

| Sub-phase | Wall | Output | Verdict gate |
|---|---:|---|---|
| **3.1 Data prep** (Path A sources) | 1-3 days CPU | `*_expanded_rdkit_features.csv` | NCI-60 / ChEMBL / PubChem loader passes integration test |
| **3.2 Encoder forward** (Path B chemprop v3) | 2 days GPU per cohort × 3 cohorts = 6 days GPU | `*_encoder_300d.npy` | chemprop v3 forward loss <0.01 (sanity) |
| **3.3 Hilbert dual** (Path B primal-dual) | 1 day GPU per cohort × 3 cohorts = 3 days GPU | `*_hilbert_r2.json` | Hilbert R² ≥ 0 per cohort |
| **3.4 W₂ best-axis** (existing Wf-335-M2) | 1 hr CPU per cohort | per-axis R² table | best axis = TPSA (axis 2) for metalcytotox, axis 0 for platinai |
| **3.5 Multi-seed CV validation** | 1 day GPU | `wf376a3_combined/result.json` | ≥ 2 cohorts achieve CEILING_BREAK |

**Total wall**: 5-10 days sequential (3 data prep days + 9 GPU days + 1 validation day)

### §3.3 PROJECTED lift (combined)

- **platinai_4cell** n=1000 + non-linear: R² PROJECTED −0.1172 → +0.30 to +0.60
- **hilbert** n=1000 + non-linear: R² PROJECTED −0.1348 → +0.35 to +0.70
- **metalcytotox_combined** n=5000 + non-linear: R² PROJECTED +0.3196 → +0.55 to +0.85

Aggregate status: PROJECTED `HONEST_NEGATIVE_R2_RESOLVED` (≥2 cohorts
CEILING_BREAK). These are PROJECTED, not MEASURED.

### §3.4 Cost & gate

- **Wall**: 5-10 days sequential
- **VRAM**: ~28 GB peak (same as Path B)
- **Pre-flight**: REQUIRED at start of each sub-phase
- **Wave budget**: +120-240 hr; current −183.04 hr preserved
- **USER_OK gate**: REQUIRED per Wf-355-B1 §3; full-path single grant
- **Per-sub-phase pre-flight check**: per Wf-343C2 + Wf-343C3 pattern

---

## §4 Launch wrapper contract (USER_OK gate pattern)

### §4.1 Wrapper reference

The canonical USER_OK-gated launch wrapper for m2 R² retrain is:

```
scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh
```

- **472 LOC on disk** (per Wf-354-B3 + Wf-366-A1/A2 Wf-355 pattern)
- **USER_OK gate**: per Wf-355-B1 §3 (exit code 64 + USER_OK=1 literal)
- **DRY_RUN.log schema**: per Wf-363-A1 (7 fields × 5 wrappers)
- **Plan.json emission**: per Wf-355-B1 contract
- **Per-pair per-wave check**: per Wf-347-B1 (`wf343c2_per_wave_check.py --skip-live`)
- **Per-pair DRY_RUN.log**: per Wf-363-A1 canonical schema

### §4.2 Invocation pattern

```bash
# 1. Pre-flight check (REQUIRED)
python molmetal/scripts/pre_flight_check.py --pipeline all
# Expected: 92 PASS / 0 FAIL / 2 SKIP / 94 TOTAL (per Wf-368-C1 baseline)

# 2. VRAM budget check (REQUIRED per AGENTS.md §5.1)
nvidia-smi --query-gpu=memory.free --format=csv
# Expected: ≥30 GB free on RTX PRO 6000 Blackwell or RTX 5090

# 3. Wave budget check (REQUIRED per Wf-262 §7.1)
python molmetal/scripts/wave_budget_tracker.py
# Expected: −183.04 hr overrun preserved (deferred); USER_OK waiver required

# 4. Launch with USER_OK gate
USER_OK=1 bash scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh \
    --cohort platinai_4cell \
    --cohort hilbert \
    --cohort metalcytotox_combined \
    --chemprop-v3-encoder forward \
    --hilbert-primal-dual \
    --w2-best-axis multi \
    --seeds 5 \
    --epochs 50 \
    --batch-size 32 \
    --lr 1e-3
```

### §4.3 Per-pair per-wave contract (per Wf-347-B1)

For each (cohort, seed) pair launched:

```
PAIR_PER_WAVE_CHECK wave=N cohort=X seed=Y verdict=PASS skip_live=true
PAIR_BUDGET_START wave=N cohort=X seed=Y est_hr=Z
PAIR_PER_WAVE_CHECK wave=N cohort=X seed=Y verdict=PASS evidence=...
PAIR_BUDGET_END wave=N cohort=X seed=Y actual_hr=Z delta_hr=D
```

Emitted to `molmetal/reports/wf376a3_*/per_pair/` and aggregated in
`molmetal/reports/wf376a3_*/budget_summary.json`.

### §4.4 DRY_RUN.log schema (per Wf-363-A1)

```json
{
  "wrapper_name": "launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh",
  "wrapper_mode": "USER_OK_GATED",
  "per_pair": [
    {"cohort": "platinai_4cell", "seed": 0, "wall_projected_hr": 0.42},
    {"cohort": "platinai_4cell", "seed": 1, "wall_projected_hr": 0.42},
    ...
  ],
  "budget_total_projected_hr": 12.0,
  "user_ok_gate": "USER_OK=1",
  "pre_flight_status": "92 PASS / 0 FAIL / 2 SKIP / 94 TOTAL",
  "vram_check_status": "PASS (≥30 GB free)",
  "wave_budget_status": "OVERRUN +120 hr, USER_OK waiver granted"
}
```

---

## §5 Cross-references

### §5.1 Source artifacts (MEASURED basis)

- **Wf-335-M2** (`TODO/completed/335m2_r2_w2_chemprop_full.md`):
  multi-seed CV mean R²=+0.3196 on metalcytotox_combined (CEILING_BREAK);
  R²=-0.1172 / -0.1348 on platinai_4cell + hilbert (HONEST_NEGATIVE_R2_DEEP).
- **Wf-367-B1** (`TODO/completed/367b1_m2_r2_recovery.md`):
  multi-stream ensemble 54-d / 50-d / 55-d / 51-d on n=200 cohorts;
  aggregate R²=-0.5158 / -0.4980 (worse than baseline; bias-variance inflation).
- **Wf-369-A2** (`TODO/completed/369a2_m2_r2_final_verdict.md` + this row's
  source): FINAL VERDICT doc + 1 NEW test PASS.
- **M2_R2_FINAL_VERDICT.md** (`docs/M2_R2_FINAL_VERDICT.md`, 320 LOC,
  7 sections): per-attempt summary table + bias-variance analysis + forward-looking.

### §5.2 Forward-looking verdicts (PROJECTED)

- **Wf-122 P1.3** (`TODO/pending/179_wf123_p3_w2_pic50_alignment.md`):
  W₂ alignment architectural envelope PROJECTED +0.05 to +0.30 R² lift on
  n=200 cohorts with n=1000 samples.
- **Wf-369-B2** (TODO/completed/369 closeout + Wf-368-B1 §7.2): Hilbert R²
  gate PROJECTED +0.30 to +0.85 R² on n=200 cohorts with non-linear encoder.
- **Wf-291D**: `decode_ratio=0` upstream blocker for CFM YuelBond decoder;
  BLOCKED until upstream fixed per Wf-335-M2 §6.

### §5.3 Standing directives preserved (AGENTS.md §5)

- **§5.1 Standing GPU authorization**: USER_OK-gated per kill_then_adapt §0.
  VRAM budget check first. Pre-flight PASS still required.
- **§5.2 Honest framing mandate**: MEASURED vs PROJECTED tag discipline;
  no fabricated / aspirational numbers. PROJECTED lift values in §1.4, §2.4,
  §3.3 all tagged PROJECTED with envelope (not point estimate).
- **§5.3 No paper work without authorization**: paper §6 prep is in
  `docs/PAPER_S6_M2_R2_LIMITATIONS.md` (separate file); `paper/*` touches
  remain BLOCKED on user OK per TODO-84.
- **§5.4 Tier-2 journal target + NO wet-lab planning**: no wet-lab items
  referenced in this plan; Tier-2 computational-only acceptance bar.

### §5.4 User-OK-gated launch wrapper inventory (Wf-368-B1 §7.2)

- `scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` — this plan
  primary wrapper (~12 hr wall PROJECTED).
- `scripts/launch_wf124_p3_g5.sh` — m1 Ru 5-seed gate (~5 hr wall, Wf-366-A1 STRICT).
- `scripts/recover_g6_failed_tuples.sh` — m3 Vina 60/60 gate (~11.4 hr wall, Wf-366-A2 STRICT).
- `scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh` — secondary (~10 hr).
- `scripts/launch_wf310e_r20_p4_bindingdb_1.07M.sh` — secondary (~7 hr).

### §5.5 Forward-looking user voice answer

Per Wf-369-A2 verdict + Wf-368-B1 §7.4: m2 R² gate NOT lifted by Wf-367-B1
multi-stream ensemble; structural ceiling on n=200 cohorts. Path A (larger
cohorts) and Path B (GPU non-linear) are the only paths forward per
Wf-122 P1.3 envelope. **No more CPU-only architectural paths** to try.

This plan (Wf-376-A3) is the **forward-looking execution contract** ready
for `USER_OK=1` grant. Until then, gate remains HONEST_NEGATIVE_R2_DEEP FINAL.

---

## §6 Pre-flight + wave budget status (preserved baseline)

- **Pre-flight v3**: 92 PASS / 0 FAIL / 2 SKIP / 94 TOTAL PRESERVED at
  Wf-368-C1 baseline (timestamp 2026-09-29T15:54:19+0000 UTC wall 145.75 s
  per `molmetal/reports/wf368c1_preflight.txt`). REQUIRED before any GPU launch.
- **Wave budget**: −183.04 hr overrun preserved per Wf-262 §7.1 + Wf-353-B1 §1
  + Wf-368-B1 §0 (deferred per user directive 2026-09-28 *"时间预算没关系的"*).
  +120-240 hr additional for full Path A+B execution; USER_OK waiver required.
- **VRAM**: ~28 GB peak (Blackwell sm_120 98 GB or RTX 5090 32 GB; tight on RTX 5090).
- **EXPECTED_GATE_COUNTS["all"]**: 94 PRESERVED at Wf-353-B1 baseline
  (no new pre-flight gates added by Wf-376-A3 plan; plan is CPU-only docs).

---

**END OF Wf-376-A3 PLAN** — Owner Hugo Lee; cross-ref `party-twenty-spirit`
§3.5; last update 2026-09-30. This plan is **CPU-only docs**; no GPU launches;
USER_OK-gated; awaits operator grant.
