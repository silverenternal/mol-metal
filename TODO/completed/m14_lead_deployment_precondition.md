# m14 LEAD Deployment Precondition — YuelBond Decoder Must Be Active

**Date**: 2026-09-29
**Reason**: Wf-275C — formal spec doc for the m14 LEAD (CFM `m14_full_metalcytotox`) deployment precondition. Per Wf-269D + TODO-207 §9.AH + TODO-188 §4.5.4, the m14 LEAD claim now explicitly requires the YuelBond decoder to be active as a deployment precondition. Without YuelBond the decode_ratio collapses to 0.000 (Wf-269C MEASURED) — the m14 LEAD is silently invalidated.
**Status**: ✅ SHIPPED (formal spec)
**Owner**: Hugo Lee + Wf-275C ultracode
**Skill alignment**: `party-twenty-spirit` §3.5 (0 容忍 不掩盖; MEASURED-vs-PROJECTED) + `mol-metal-cfm-training` (YuelBond decoder head) + `mol-metal-paper-writing` §4 verbatim template + `mol-metal-gpu-execution` §7 MEASURED-vs-PROJECTED tag discipline.

---

## §0 TL;DR

The m14 LEAD claim (CFM real-cohort decode_ratio, paper §4 LEAD) requires the **YuelBond decoder to be active** as a deployment precondition. Without YuelBond, Wf-269C MEASURED decode_ratio drops from **1.000 → 0.000** (Δ=−1.000, the maximum possible drop) on the real MetalCytoToxDB cohort. The other two CFM levers (Wasserstein regularization + closure-bounded generation) are minor refinements (Δ=0 MEASURED, NOT a decode gate). In production, the CFM ablation flag `--no-yuelbond` MUST NOT be set; this is the only configuration that silently invalidates the m14 LEAD claim without raising any runtime error.

---

## §1 m14 LEAD claim scope

### §1.1 What is "m14"?

The **m14 metric** is the CFM (Conditional Flow Matching) real-cohort decode_ratio — the fraction of generated molecules that pass RDKit-strict validity on a real metal-coordination chemistry dataset. It is one of the 21 paper metrics (m1..m21) tracked per `TODO/METRICS_TARGETS.md` + `TODO/MECHANISM_METRIC_MAPPING.md`.

### §1.2 What is "m14_full_metalcytotox" (the LEAD)?

The m14 LEAD claim is **CFM m14 FULL MetalCytoToxDB decode_ratio = 0.9958** (n=26,801, 5-metal cohort Ru/Ir/Os/Rh/Re), MEASURED per Wf-188 (`molmetal/reports/wf188_m14_full_metalcytotox/result.json` + `synthesis.md`). This supersedes the prior Wf-148 cross-check **decode_ratio = 0.9980** (3 seeds × 5,991 GPU samples, NCI-60 9-cell + DrugOOD native, per `molmetal/reports/wf139_p5_cfm_decode_real/synthesis.md` + `wf148_m14_real_cfm.log`). Both numbers are valid MEASURED data points, but the Wf-188 m14 FULL figure is the paper §4 LEAD because it operates on the larger n=26,801 MetalCytoToxDB cohort.

### §1.3 What is "0.9645" (the smaller number)?

`molmetal/metrics/by_metric/cfm_decode_real.json` reports MEASURED real CFM decode_ratio = **0.9645** (n=818, 1 seed, CPU, PlatinAI pt_subset, pocket 1h36, jitter=0.05) — a deprecated smaller-cohort measurement (per Wf-264 §3.2 audit fix + TODO-188 §5.2 honest-framing note). It is **NOT** the m14 LEAD source. Cite Wf-188 (decode_ratio=0.9958, n=26,801) for the LEAD; cite Wf-148 (0.9980, n=5,991) only as the prior cross-check; do NOT cite 0.9645 as a MEASURED source. Always disclose sample size + seed count + compute (GPU vs CPU) + cohort when citing any decode_ratio figure.

### §1.4 What is NOT part of m14 LEAD scope

- m7 Tanimoto diversity (different metric; +0.820 honest gap to TargetDiff 0.860 per TODO-207 §9.6.2).
- SA / QED / Vina panel on generated molecules (per TODO-207 §9.6.1 + §9.AA; panel MEASURED on the MetalCytoToxDB ligand population, NOT on CFM-generated SMILES).
- chemprop v2 v3 backbone metrics (m1 / m17 / m18 / m19 / m21).
- Wf-124 P2 synthetic-cohort decode_ratio = 1.000 (upper-bound reference per TODO-207 §4.2.11; the m14 LEAD is REAL-cohort only).

---

## §2 Deployment precondition: YuelBond decoder must be active

### §2.1 The single load-bearing precondition

> **YuelBond decoder MUST be active at CFM training + decode time. If `--no-yuelbond` is set, m14 LEAD silently invalidates (decode_ratio = 0.000).**

### §2.2 MEASURED evidence (Wf-269C, real cohort, 2026-09-28)

Per `molmetal/reports/wf269c_cfm_real/wf269c_report.json` (cohort: `metalcytotox_5metal_multitask_v4.csv`, n=26,473 after degenerate-filter, 49 train clouds + 51 val clouds, hidden=32 / n_layers=2, CPU, 5 epochs full + 2 epochs followup):

| Config | decoder_type | n_decode_ok / n_decode_total | final_decode_ratio | Δ vs full | MEASURED status |
|---|---|---:|---:|---:|---|
| **full** | `YuelBondBondDecoder` | 51/51 | **1.000** | (baseline) | ✅ MEASURED |
| **`--no-yuelbond`** | `ReworkedDecoder` + `BondOrderHead` (untrained) | 0/51 | **0.000** | **−1.000** | ✅ MEASURED (load-bearing) |
| **`--no-wasserstein`** | `YuelBondBondDecoder` | 11/11 | **1.000** | **0.000** (NOT −0.013) | ✅ MEASURED (NOT a decode gate; W₂ is velocity-probe anchor) |
| **`--no-closure-bounded`** | `YuelBondBondDecoder` | n/a (not run due to wall-time budget) | **1.000** | **0.000** (projected, NOT −0.008) | ⚠️ PROJECTED (closure-bounded is decode-time bias toward 5-click productive bond types; YuelBond produces valid SMILES regardless) |

### §2.3 Why YuelBond is the single load-bearing lever

- **Δ=−1.000 is the maximum possible drop** on the decode_ratio gate (every generated molecule fails RDKit-strict validity without YuelBond's bond head; see Wf-269C `note`: "BondAwareDecoder + ReworkedDecoder path is too slow on real molecules (>30 atoms) to train past epoch 0. Decode_ratio=0.000 from epoch=0 is the meaningful signal — legacy decoder cannot produce valid SMILES on real cohort").
- The other two levers (Wasserstein reg + closure-bounded) are auxiliary innovations that contribute to training stability (W₂ is a velocity-probe anchor) and chemistry specificity (closure-bounded is a decode-time bias toward productive bond types), NOT to decode validity. Their deltas in the Wf-269C MEASURED run are 0.000, not −0.013 / −0.008 as cited in TODO-207 §9.AF (this is a §9.AF source-value discrepancy; see §7 honest-framing note below).
- Wf-269C `honest_framing_note` clarifies: "the YuelBond decoder IS doing the real work; Wasserstein and closure-bounded are auxiliary innovations that contribute to training stability / chemistry specificity, NOT to decode validity. Caveat: full=1.000 and no_wasserstein=1.000 (both saturated) means the RDKit-strict metric alone cannot separate these two; a chemistry-aware metric (Tanimoto to reference, validity under synthetic noise) would be needed to distinguish them. This is the limit of the decode_ratio signal on this cohort; defer chemistry-aware ablation to TODO-119 P1.3 (real CrossDocked loader)."

---

## §3 Why YuelBond is load-bearing (evidence chain)

### §3.1 Wf-269C real-cohort per-lever ablation (PRIMARY evidence)

The Wf-269C report is the primary MEASURED evidence: real MetalCytoToxDB v4 cohort, 4 configs (full / no_yuelbond / no_wasserstein / no_closure_bounded), per-config decoder_type recorded in JSON. The `--no-yuelbond` config collapses to decode_ratio = 0.000 from epoch 0 because the legacy `ReworkedDecoder` + `BondOrderHead` cannot train past epoch 0 on real molecules (>30 atoms) and never produces a valid SMILES.

### §3.2 Wf-268A flag verification (SECONDARY evidence — flag surface)

Per `molmetal/reports/wf268a_audit/flag_audit.md` (read-only audit, 2026-09-28), all three CFM ablation flags are syntactically and semantically wired into `molmetal/scripts/train_cfm_yuelbond_wasserstein.py`:

| Gate | Flag | PASS/FAIL | Line | Argument definition |
|---|---|:---:|---|---|
| G1 | `--no-yuelbond` | **PASS** | 832-833 | `p.add_argument("--no-yuelbond", dest="use_yuelbond", action="store_false", ...)` |
| G2 | `--no-wasserstein` | **PASS** | 836-838 | `p.add_argument("--no-wasserstein", dest="wasserstein_weight", action="store_const", const=0.0, ...)` |
| G3 | `--no-closure-bounded` | **PASS** | 841-843 | `p.add_argument("--no-closure-bounded", dest="closure_bounded", action="store_false", ...)` |

Default for all three is the AFFIRMATIVE flag (`use_yuelbond=True`, `wasserstein_weight=0.1` default, `closure_bounded=True`), so existing launch scripts (Wf-188 m14 FULL + Wf-124 P2 CFM full + 40 CFM tests) work unmodified. The flags are safe by default; they require explicit opt-in to disable.

### §3.3 Source code architecture (TERTIARY evidence)

- **YuelBond decoder** (`molmetal/molmetal_lam/lam_chem/yuelbond_decoder.py`, ~250 LOC): implements the bond-head that produces valid SMILES on real metal-coordination molecules. Tagged DIAGNOSTIC vs PRODUCTION per Wf-253 v3 ship.
- **BondAwareDecoder + ReworkedDecoder** (legacy path): the alternative decoder path; faster on small molecules (<30 atoms) but cannot produce valid SMILES on real MetalCytoToxDB molecules (Wf-269C `note`).
- **CFM Wasserstein loss** (`molmetal/molmetal_lam/lam_chem/cfm_wasserstein_loss.py`, ~200 LOC): implements the velocity-probe anchor W₂ regularization; GRADIENT path (per TODO-119 §2), NOT DIAGNOSTIC (parallel to Wf-252 §F3 chemprop precedent).
- **closure-bounded** (in `train_cfm_yuelbond_wasserstein.py`): a decode-time bias that nudges logits toward 5-click productive bond types; threaded through `train_one_epoch` (line 567) + `evaluate_decode_ratio` (line 711) per Wf-268A G3.

### §3.4 Cross-references (EVIDENCE-LEVEL)

- Wf-269C report (`molmetal/reports/wf269c_cfm_real/wf269c_report.json`) — PRIMARY
- Wf-268A flag audit (`molmetal/reports/wf268a_audit/flag_audit.md`) — SECONDARY
- Wf-269C summary (`molmetal/reports/wf269c_cfm_real/wf269c_summary.md`) — PRIMARY narrative
- Wf-269C run log (`molmetal/reports/wf269c_cfm_real/run.log`) — execution trace
- Wf-188 m14 FULL result (`molmetal/reports/wf188_m14_full_metalcytotox/result.json` + `synthesis.md`) — m14 LEAD measurement (decode_ratio = 0.9958, n=26,801)

---

## §4 Configuration in production

### §4.1 CFM production launch (the precondition surface)

For CFM training + decode, the production launch MUST satisfy:

1. **`--no-yuelbond` MUST NOT be set** (use_yuelbond MUST default to True; default behavior is safe).
2. **`--wasserstein-weight`** is optional but recommended at default 0.1 (not a precondition; not a decode gate).
3. **`--closure-bounded`** is optional but recommended at default True (not a precondition; not a decode gate).
4. The CFM checkpoint MUST be `molmetal/checkpoints/cfm_yuelbond_wasserstein.pt` (~617 KB Wf-124 P2 ship; TODO-207 §4.2.7).

Example safe launch:
```bash
python molmetal/scripts/train_cfm_yuelbond_wasserstein.py \
    --csv molmetal/data/metallo_drugs_500_train.csv \
    --epochs 50 \
    --device cuda \
    --chemprop-mode <see §4.2> \
    # --no-yuelbond MUST NOT appear
    # --no-wasserstein optional, NOT a precondition
    # --no-closure-bounded optional, NOT a precondition
```

### §4.2 chemprop vs CFM flag disambiguation (NOTE: chemprop-mode is NOT a CFM flag)

`--chemprop-mode` is a **chemprop v2 v3 flag** in `molmetal/scripts/train_chemprop_v2_v3.py` (line 1523), NOT a CFM flag. It accepts `{v2, v3, v3-fallback, v3-fix, shared-encoder}` (the default is `v3-fallback` per TODO-207 §9.7; `v3-fix` is the LOMO-targeted mode per §9.Y). The CFM script `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` does NOT recognize `--chemprop-mode`.

**Common production confusion**: the chemprop and CFM launch scripts share `--chemprop-mode` history (chemprop v2 v3 uses it as a mode selector), but the CFM ablation surface is `--no-yuelbond` / `--no-wasserstein` / `--no-closure-bounded`. An operator who copies the chemprop `--chemprop-mode` flag into a CFM launch will get an argparse error (the flag is unknown); conversely, an operator who copies the CFM `--no-yuelbond` flag into a chemprop launch will also get an argparse error.

For paper §4 m14 LEAD scope, the CFM surface is what matters: the m14 metric is CFM-generated molecule validity, NOT a chemprop head. Cite the chemprop and CFM flags separately in any operational runbook.

### §4.3 Default values (safe-by-default)

Per Wf-268A G1-G3 audit + the source code defaults in `train_cfm_yuelbond_wasserstein.py`:

| Flag | Default | Production posture |
|---|---|---|
| `--use-yuelbond` | True (implicit) | **MUST stay True** — m14 LEAD precondition |
| `--no-yuelbond` | (off, default use_yuelbond=True) | **MUST NOT be set** |
| `--wasserstein-weight` | 0.1 (default at line 834) | Recommended; not a precondition |
| `--no-wasserstein` | (off, default 0.1) | Optional; NOT a precondition |
| `--closure-bounded` | True (implicit) | Recommended; not a precondition |
| `--no-closure-bounded` | (off, default closure_bounded=True) | Optional; NOT a precondition |

Backward-compat: existing CFM launch scripts (Wf-188 m14 FULL, Wf-124 P2 full, Wf-181 m14 smoke) work unmodified because they never pass the `--no-*` ablation flags. The defaults are the safe YuelBond-active state.

---

## §5 Verification protocol

### §5.1 3-config CFM ablation on real cohort (the §3.5 honest-positive anchor)

The verification protocol for the m14 LEAD deployment precondition is the **Wf-269C 4-config real-cohort ablation** (full / `--no-yuelbond` / `--no-wasserstein` / `--no-closure-bounded`), reproduced per `molmetal/reports/wf269c_cfm_real/wf269c_report.json`:

1. **Cohort**: `.data/molmetal/metalcytotox_5metal_multitask_v4.csv` (n=26,473 after degenerate-filter; 49 train clouds + 51 val clouds). The v4 cohort supersedes the v2 cohort (26,633 lines per TODO-207 §1.4 post-Wf-261) and the v1 cohort (26,802 lines, leaky).
2. **Configs**: 4 ablation cells (`full`, `no_yuelbond`, `no_wasserstein`, `no_closure_bounded`).
3. **Per-config epochs**: 5 (full) + 1 (no_yuelbond, epoch-0 collapse makes longer training meaningless) + 2 (no_wasserstein) + 0 (no_closure_bounded, wall-time budget).
4. **Compute**: CPU (Wf-269C was on CPU; RTX 5090 GPU is available for R20+ rerun).
5. **Hidden / n_layers**: 32 / 2 (Wf-269C); production is 128 / 3 (TODO-207 §4.2.8).
6. **Acceptance gate**: `full_decode_ratio - no_yuelbond_decode_ratio ≥ 0.5` (the load-bearing claim requires a large drop; Wf-269C observed 1.000 → 0.000 = Δ=−1.000, well above the 0.5 floor).
7. **Output**: `molmetal/reports/wf269c_cfm_real/wf269c_report.json` with `G1..G10` gates (G9_deltas_visible=true required).

### §5.2 full vs no_yuelbond ratio (the load-bearing ratio)

The single most important diagnostic is the **full vs `--no-yuelbond` decode_ratio ratio**:

- If `full_decode_ratio == 1.000` AND `no_yuelbond_decode_ratio == 0.000` → YuelBond is load-bearing; m14 LEAD valid.
- If `full_decode_ratio == 1.000` AND `no_yuelbond_decode_ratio == 1.000` → YuelBond is NOT load-bearing on this cohort; m14 LEAD premise is false.
- If `full_decode_ratio < 1.000` → CFM is broken (synthetic upper bound not reached); m14 LEAD invalid regardless of YuelBond.
- If `no_yuelbond_decode_ratio > 0.000` → BondAwareDecoder / ReworkedDecoder succeeded on this cohort; m14 LEAD precondition should be revisited (may indicate a cohort where YuelBond is not the load-bearing lever).

### §5.3 Cross-cohort generalization (TODO-119 P1.4 follow-up)

The Wf-269C ablation was on MetalCytoToxDB subset only. TODO-119 P1.4 (per TODO-207 §9.Z Z.5) is the cross-cohort generalization — run Wf-269C ablation protocol on PlatinAI 4-cell + DrugOOD + cross-metal cohorts. Target: confirm YuelBond load-bearing claim generalizes; secondary: measure Wasserstein + closure-bounded lift on smaller cohorts (where the Wf-269C saturation may not hold). Budget re-estimate: ~5 hr GPU (down from 7.5 hr per §9.AF, since per-innovation ablation is already done on MetalCytoToxDB).

---

## §6 Failure modes

### §6.1 YuelBond disabled → 0% decode → m14 LEAD silently invalid

**Failure mode**: An operator (or test script) inadvertently passes `--no-yuelbond` to the CFM launch script. The script parses, runs to completion (no runtime error), emits a `wf269c_report.json`-style output with `final_decode_ratio = 0.000`. The m14 LEAD claim (0.9958 / 0.9980) is silently invalidated. The paper §4 narrative citing "m14 LEAD 0.9958" remains in the text, but the actual measurement is 0.000.

**Detection**: Per-config `decoder_type` field in the output JSON must equal `YuelBondBondDecoder`. If it equals `ReworkedDecoder` or `BondAwareDecoder`, the YuelBond decoder was disabled.

**Mitigation**:
1. Run the Wf-269C ablation verification (§5.1) before any production CFM launch; require `G9_deltas_visible=true` AND `G6_decode_ratio_no_yuelbond=0.000`.
2. Pre-flight gate `molmetal/scripts/pre_flight_check.py --gate 4.2.1` (per TODO-207 §4.2.1) verifies the `--no-yuelbond` / `--no-wass` / `--no-closure` flags are exposed in `--help`; does NOT verify they are NOT used in the launch command.
3. Runbook-level check: the production launch command MUST NOT contain `--no-yuelbond`; review the command line before launch.

### §6.2 Checkpoint loaded without YuelBond head

**Failure mode**: The `molmetal/checkpoints/cfm_yuelbond_wasserstein.pt` checkpoint is loaded by a deployment script that uses the legacy `ReworkedDecoder` instead of `YuelBondBondDecoder`. The model weights load (no error), but inference runs with the wrong decoder → decode_ratio = 0.000 on every molecule.

**Detection**: The deployment script's model class instantiation MUST equal `molmetal/molmetal_lam/lam_chem/yuelbond_decoder.YuelBondBondDecoder`. Add a startup assertion: `assert isinstance(model.decoder, YuelBondBondDecoder), "m14 LEAD deployment requires YuelBond decoder; got %s" % type(model.decoder)`.

**Mitigation**: Pin the model class in the deployment script. The `cfm_yuelbond_wasserstein.pt` checkpoint (~617 KB) contains weights trained with the YuelBond decoder; using it with any other decoder is a misuse.

### §6.3 Wasserstein weight set to 0 silently disables W₂ (NOT a precondition failure)

**Failure mode**: An operator passes `--no-wasserstein` to disable W₂ regularization. Decode_ratio stays at 1.000 (Wf-269C MEASURED); m14 LEAD is NOT invalidated. But the W₂ velocity-probe anchor is disabled, which may degrade downstream training stability and chemistry specificity. This is **NOT** a precondition failure for m14 LEAD; it is a soft regression.

**Detection**: Check `final_decode_ratio` — should stay at 1.000 even with `--no-wasserstein`.

**Mitigation**: None required for m14 LEAD. Recommended to keep `--wasserstein-weight 0.1` (default) for training stability.

### §6.4 Closure-bounded disabled (NOT a precondition failure)

**Failure mode**: An operator passes `--no-closure-bounded`. Decode_ratio stays at 1.000 (Wf-269C PROJECTED, not MEASURED due to wall-time budget). m14 LEAD is NOT invalidated. The 5-click productive bond-type bias is disabled, which may degrade downstream chemistry specificity.

**Detection**: Check `args.closure_bounded == True` in launch logs; alternatively check `evaluate_decode_ratio` call site uses the closure-bounded path (line 711 per Wf-268A G3).

**Mitigation**: None required for m14 LEAD. Recommended to keep `--closure-bounded` (default True) for chemistry specificity.

---

## §7 Honest framing — DO / DO NOT claim lists

### §7.1 DO claim (post-Wf-269C, 2026-09-28)

- "CFM YuelBond decoder is load-bearing for decode_ratio gate on real cohort: full=1.000 vs `--no-yuelbond`=0.000 (Δ=−1.000, Wf-269C MEASURED on n=51 val clouds stratified from 26,473 mol MetalCytoToxDB v4 cohort)."
- "CFM Wasserstein regularization (--no-wasserstein) MEASURED decode_ratio = 1.000 (Δ=0.000 vs full) on Wf-269C — W₂ is a velocity-probe anchor, NOT a decode gate. CFM closure-bounded generation (--no-closure-bounded) PROJECTED decode_ratio = 1.000 (closure-bounded is a decode-time bias; YuelBond produces valid SMILES regardless) — not measured on Wf-269C due to wall-time budget."
- "CFM m14 LEAD claim (decode_ratio 0.9958 on n=26,801, Wf-188) requires YuelBond decoder active as deployment precondition; without YuelBond, the m14 LEAD silently invalidates (decode_ratio = 0.000 on real cohort)."
- "In production CFM launches, `--no-yuelbond` MUST NOT be set; the default `use_yuelbond=True` is the safe YuelBond-active state."
- "The m14 LEAD precondition is verified by the Wf-269C 4-config real-cohort ablation; reproduction requires `molmetal/reports/wf269c_cfm_real/wf269c_report.json` with `G9_deltas_visible=true` and `G6_decode_ratio_no_yuelbond=0.000`."

### §7.2 DO NOT claim (until TODO-119 P1.4 cross-cohort MEASURED)

- "Wf-269C finding generalizes to PlatinAI 4-cell + DrugOOD cohorts" (TODO-119 P1.4 PENDING GPU; Wf-269C is MetalCytoToxDB subset only).
- "YuelBond is the ONLY load-bearing lever across all cohorts" (per-cohort: Wasserstein reg + closure-bounded may differ on smaller cohorts where the Wf-269C saturation may not hold).
- "CFM Wasserstein reg provides a +0.013 lift on decode_ratio" (TODO-207 §9.AF cites Δ=−0.013 for `--no-wasserstein`; Wf-269C MEASURED shows Δ=0.000 — the §9.AF figure is a source-value discrepancy with the actual MEASURED Wf-269C data; cite MEASURED 0.000, not the §9.AF claim).
- "CFM closure-bounded provides a +0.008 lift on decode_ratio" (TODO-207 §9.AF cites Δ=−0.008 for `--no-closure-bounded`; Wf-269C reports this config as PROJECTED 1.000 (not MEASURED due to wall-time budget); cite PROJECTED 1.000 with the wall-time caveat, not the §9.AF claim).
- "The CFM Wasserstein + closure-bounded levers are 'load-bearing' alongside YuelBond" (Wf-269C shows ONLY YuelBond is load-bearing; the other two are auxiliary innovations).
- "m14 LEAD is preserved at decode_ratio 0.9958 on the v2 CSV cohort (26,633 lines, post-Wf-261)" (the Wf-188 MEASURED was on the v2 cohort; but YuelBond-active is the precondition; with `--no-yuelbond` the decode_ratio collapses to 0.000 regardless of cohort).

### §7.3 Source-value discrepancy note (TODO-207 §9.AF vs Wf-269C)

Per party-twenty-spirit §3.5 honest framing: TODO-207 §9.AF cites the Wf-269C ablation deltas as:
- `--no-yuelbond`: 0.000 (Δ=−1.000) ✅ matches Wf-269C JSON
- `--no-wasserstein`: 0.987 (Δ=−0.013) ❌ DOES NOT match Wf-269C JSON (actual MEASURED: 1.000, Δ=0.000)
- `--no-closure-bounded`: 0.992 (Δ=−0.008) ❌ DOES NOT match Wf-269C JSON (PROJECTED 1.000, not MEASURED)

The discrepancy is between the §9.AF narrative claim and the Wf-269C MEASURED JSON. This §7.3 entry flags the discrepancy so reviewers can locate the source-value conflict. Resolution: cite Wf-269C JSON as the MEASURED source of truth; TODO-207 §9.AF requires a follow-up edit to reconcile the source-values. This doc (m14 LEAD deployment precondition) cites Wf-269C JSON verbatim per §3.5 honest framing.

---

## §8 Cross-references

**Primary upstream sources** (this doc synthesizes the following):

- `TODO/completed/188_paper_4_krasnov_protocol_aligned.md` §4.5.4 — Wf-269A + Wf-269C honest results catalog in paper §4 context; m14 LEAD 0.9958 source + YuelBond precondition first stated here.
- `TODO/completed/207_unified_pre_flight_checklist_v2.md` §9.AH — Wf-269A + Wf-269C honest results catalog in pre-flight checklist context; load-bearing claim formalized.
- `TODO/completed/207_unified_pre_flight_checklist_v2.md` §9.AF — Wf-269C YuelBond load-bearing on real cohort (n=1,000 stratified subset, 2026-09-28); flagged source-value discrepancy per §7.3.
- `TODO/completed/207_unified_pre_flight_checklist_v2.md` §9.6.3 — Generated molecule validity 0.9958 (Wf-188) vs synthetic upper bound 1.000; explicit conditional on YuelBond.
- `molmetal/reports/wf269c_cfm_real/wf269c_report.json` — PRIMARY evidence: 4-config real-cohort ablation (full=1.000, no_yuelbond=0.000, no_wasserstein=1.000, no_closure_bounded=1.000 projected); G1..G10 gates.
- `molmetal/reports/wf269c_cfm_real/wf269c_summary.md` — Wf-269C narrative summary.
- `molmetal/reports/wf269c_cfm_real/run.log` — Wf-269C execution trace.
- `molmetal/reports/wf268a_audit/flag_audit.md` — SECONDARY evidence: 3/3 PASS flag audit for `--no-yuelbond` (line 832-833), `--no-wasserstein` (line 836-838), `--no-closure-bounded` (line 841-843).
- `molmetal/reports/wf188_m14_full_metalcytotox/result.json` + `synthesis.md` — m14 LEAD measurement (decode_ratio = 0.9958, n=26,801, gate ≥ 0.05 PASS by 20×).
- `molmetal/reports/wf139_p5_cfm_decode_real/synthesis.md` + `wf148_m14_real_cfm.log` — Wf-148 m14 cross-check (decode_ratio = 0.9980, 3 seeds × 5,991 GPU, NCI-60 9-cell + DrugOOD native).

**Source code** (the precondition enforcement surface):

- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` — CFM launch script; lines 832-843 (3 ablation flags); line 1052 + 1064 (call sites thread `args.closure_bounded`); line 711 (evaluate_decode_ratio with closure_bounded); line 768 (fall-through to plain `_decode_with_rdkit_strict` path when closure_bounded=False); line 624 (training loop guard on `wasserstein_weight > 0.0`); line 567 (`train_one_epoch` accepts closure_bounded).
- `molmetal/molmetal_lam/lam_chem/yuelbond_decoder.py` — YuelBond decoder (~250 LOC P1.1 wire; tagged DIAGNOSTIC vs PRODUCTION per Wf-253 v3 ship).
- `molmetal/molmetal_lam/lam_chem/cfm_wasserstein_loss.py` — CFM Wasserstein loss (~200 LOC P1.2 wire; GRADIENT path per TODO-119 §2).
- `molmetal/checkpoints/cfm_yuelbond_wasserstein.pt` — Production CFM checkpoint (~617 KB Wf-124 P2 ship; weights trained with YuelBond decoder).
- `molmetal/scripts/pre_flight_check.py` — Auto-verify script (gate 4.2.1 verifies the 3 CFM ablation flags are exposed in `--help`; does NOT verify they are NOT used in the launch command — runtime enforcement is the operator's responsibility).

**TODO follow-up hooks**:

- TODO-119 P1.4 cross-cohort generalization: run Wf-269C ablation protocol on PlatinAI 4-cell + DrugOOD + cross-metal cohorts (target: confirm YuelBond load-bearing claim generalizes; secondary: measure Wasserstein + closure-bounded lift on smaller cohorts). Budget re-estimate: ~5 hr GPU (down from 7.5 hr per TODO-207 §9.AF). See `TODO/completed/119_decode_ratio_strengthen.md` §3.
- TODO-207 §9.AF source-value fix: TODO-207 §9.AF cites `--no-wasserstein` Δ=−0.013 and `--no-closure-bounded` Δ=−0.008, which DO NOT match Wf-269C MEASURED JSON (Δ=0.000 and PROJECTED Δ=0.000 respectively); requires follow-up edit per §7.3 source-value discrepancy note.
- TODO-207 §3.1.14 P1 VERIFY gate "145 tests PASS" → "184 tests PASS" update per Wf-270D (+39 new CFM panel tests).

**Skill alignment**:

- `party-twenty-spirit` §3.5 (0 容忍 不掩盖; MEASURED-vs-PROJECTED; every number traces to JSON path or is honest-PROJECTED).
- `mol-metal-cfm-training` (YuelBond bond head + Triton-fused ops + decode_ratio + decode smoke + h=128 / n_layers=3 production defaults; Wf-269C used h=32 / n_layers=2 reduced for CPU speed).
- `mol-metal-paper-writing` §4 verbatim template + §6 limitations template + honest framing.
- `mol-metal-gpu-execution` §7 MEASURED-vs-PROJECTED tag discipline (every §X.Y cites a JSON path or is honest-PROJECTED).
- `mol-metal-mechanism-mapping` §4.5 weak-metric ledger (m14 promoted §4.5 → §4 LEAD per TODO-188 §6.2; YuelBond precondition is the deployment-side guardrail for that promotion).

**Supersedes / supersession chain**:

This doc is NEW (Wf-275C, 2026-09-29); no prior version. The m14 LEAD precondition is first formalized here as a standalone deployment contract. Prior TODO-188 §4.5.4 + TODO-207 §9.AH + TODO-207 §9.AF + TODO-207 §9.6.3 entries each carry partial statements of the precondition in their respective contexts; this doc consolidates them into a single deployment spec.
