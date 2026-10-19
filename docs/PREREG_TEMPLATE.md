# PREREG_TEMPLATE.md — Mol-Metal Pre-Registration Record (Template)

**Status:** TEMPLATE — copy this file to a new location per paper-grade launch (e.g. `docs/prereg/<launch_id>_<date>.md`).
**Owner:** Hugo Lee
**Created:** 2026-09-29 (Wf-350-B2)
**Reference:** Wf-343a-R1 finding #2 (DeepMind responsible-research framework §3.1 pre-registration).
**Tier-2 marker:** AGENTS.md §5.4 — pre-registration is **USER_OK-gated**, NOT auto-enforced; this is a template for paper-grade launches only.

---

## §0 Purpose

This document is a **pre-registration record** for any retrain / panel / ablation launch whose results may land in `paper/` (paper §3/§4/§5) or in any honest-negative closeout that cites a specific metric lift. The template locks in the hypothesis, primary metric, secondary metrics, baseline, expected effect size, and falsification criterion **BEFORE** the launch is executed, so that downstream analysis cannot silently drift (no post-hoc cherry-picking).

Per **AGENTS.md §5.2 honest framing** (party-twenty-spirit §3.5), every numeric value that lands in a paper section MUST trace to a JSON / report path on disk; the pre-registration record is the upstream counterpart that documents what was *promised* (PROJECTED) vs what was *measured* (MEASURED) after the launch.

### §0.1 Canonical schema fields (machine-parseable)

The launch scripts (`scripts/launch_wf124_p3_g5.sh` + `launch_wf124_p3_g6.sh` + `recover_g6_failed_tuples.sh`) verify the following 5 canonical schema fields are present in the prereg.md (snake_case token match):

- `hypothesis`
- `primary_metric`
- `baseline`
- `expected_effect_size`
- `falsification_criterion`

**Tip:** include each token verbatim in the relevant §1, §4, or §6 subsection (e.g. as `Primary metric (primary_metric):` or similar). The Wf-350-B2 test `test_prereg_template_schema_validity` greps for these tokens case-sensitively.

---

## §1 Research question + hypothesis

**Research question** (one-sentence, falsifiable):

> <e.g. "Does adding W₂ sort-mode gradient (vs sliced) to D-MPNN multi-task training on MetalCytoToxDB 5-metal v4 cohort increase mean_hilbert_r2 lift by ≥ +0.05 over the sliced-mode baseline?">

**Hypothesis** (null + alternative):

- **H₀ (null):** <the default null; e.g. "W₂ sort-mode gradient has no effect on mean_hilbert_r2 lift vs the sliced-mode baseline; observed Δ ∈ ±0.02 noise band.">
- **H₁ (alternative):** <the expected effect; e.g. "W₂ sort-mode gradient yields mean_hilbert_r2 lift Δ ≥ +0.05 over sliced-mode baseline on MetalCytoToxDB 5-metal v4 cohort.">

**Direction:** <one-sided | two-sided>

**Significance level:** <α; default 0.05>

---

## §2 Dataset(s) + cohort definition

**Primary dataset:** <name + version + SHA-256 hash; e.g. `metalcytotox_5metal_multitask_v4.csv` at `/home/hugo/codes/mol-metal/.data/molmetal/` + sha256=…>

**Cohort definition:** <e.g. "5 metals (Pt, Ru, Ir, Os, Rh) × 3 seeds (0,1,2) × 50 epochs; scaffold split (Wf-261 v4 key, 0 leak *at that key* — see `docs/DATA_LEAKAGE_AUDIT.md` for the desalted ligand-scaffold overlap); n_train / n_val / n_test = …">

**Exclusion criteria:** <e.g. "compounds with MW > 800 Da excluded; metal-coordination bonds only; SMILES sanitize-required">

**Provenance + license:** <source URL + license + retrieval date>

**Hash verification command:** <e.g. `sha256sum ${DATA_CSV}` — to be re-run at launch time + verified against the pre-registered hash>

---

## §3 Hyperparameters + model architecture

**Config file path:** <e.g. `configs/wf124_p3_50epoch_full_sweep_g5.yaml` — must match the launcher's `--config` arg>

**Config SHA-256:** <sha256sum of the config file at pre-registration time; re-verify at launch>

**Model architecture:** <e.g. "D-MPNN (depth=3, hidden=300, message_passing_steps=3) + chemprop v2/v3 graph encoder + W₂ sort-mode gradient (n_mc=256) + Wasserstein weight=0.05 + L2 guard">

**Hyperparameters (table):**

| Hyperparameter | Value | Source / rationale |
|---|---|---|
| epochs | 50 | TODO-181 §3 |
| batch_size | 256 | Wf-260 G6 timing |
| hidden | 300 | TODO-181 §3 |
| depth | 3 | TODO-181 §3 |
| lr | 1e-3 | Wf-260 |
| alpha | 0.5 | TODO-181 §3 |
| wasserstein_weight | 0.05 | Wf-326A |
| n_mc | 256 | Wf-326A |
| w2_gradient_mode | sort | Wf-328-W5 MEASURED 2-3× speedup |
| use_l2_guard | True | Wf-263C task spec |
| torch_compile_mode | max-autotune-no-cudagraphs | Wf-335-R1 (Blackwell sm_120) |
| cpu_num_workers | 8 | Wf-328-W2 |
| num_workers | 8 | Wf-328-W6 |

**Environment:**

| Component | Version | Verification |
|---|---|---|
| GPU | <RTX PRO 6000 Blackwell 98GB or RTX 5090 32GB> | `nvidia-smi` |
| CUDA | version | `nvcc --version` |
| PyTorch | version | `python -c "import torch; print(torch.__version__)"` |
| chemprop | version (v2 or v3) | `python -c "import chemprop; print(chemprop.__version__)"` |
| git commit SHA | <sha> | `git rev-parse HEAD` |
| git branch | master | `git rev-parse --abbrev-ref HEAD` |

---

## §4 Evaluation metric(s) + acceptance gate

**Primary metric:** <name + direction + threshold; e.g. "mean_hilbert_r2 (5-metal × 3-seed mean, higher=better) ≥ baseline + 0.05">

**Secondary metrics:** <list; e.g. "per_metal_hilbert_r2, per_seed_r2, runtime_per_epoch_s, peak_vram_mib">

**Baseline (PROJECTED):** <e.g. "sliced-mode baseline mean_hilbert_r2 = 0.5000 ± 0.05 (Wf-326A MEASURED)">

**Expected effect size (PROJECTED):** <e.g. "+0.05 mean_hilbert_r2 lift with std-dev σ=+0.02; 95% CI [+0.01, +0.09]">

**Acceptance gate:** <e.g. "Primary metric Δ ≥ +0.05 AND p < 0.05 (paired t-test, 3 seeds × 5 metals = 15 pairs); secondary metrics ≥ baseline - 0.01 (no regression)">

**Falsification criterion:** <e.g. "If mean_hilbert_r2 Δ < +0.02 OR any metal shows ≥ 0.10 regression, declare HYPOTHESIS_REJECTED and emit honest-negative closeout (party-twenty-spirit §3.5).">

---

## §5 Pre-registered ablation plan

**Ablation configs (if any):** <e.g. "per TODO-182 G6: 4 configs × 5 metals × 3 seeds = 60 tuples; configs: no_multi_task, no_wass, no_kendall, no_dann">

**Ablation analysis plan:**

1. Compute per-config mean across (metal, seed).
2. Compute per-config lift = (config mean) - (full model baseline mean).
3. Aggregate per-lever attribution: per-metal × per-lever matrix.
4. Test for interaction effects: 2-way ANOVA per-lever × per-metal.
5. Report top-1 contributing lever per metal.

**Pre-registered ablation stop rule:** <e.g. "Stop after 60 tuples regardless of intermediate results; do not extend tuple count based on partial-data trends.">

---

## §6 Pre-registered analysis plan

**Analysis steps (ordered):**

1. Load per-pair `per_seed_metrics_seed*.json` from `${OUT_BASE}/<metal>/seed<seed>/`.
2. Compute per-pair primary + secondary metrics.
3. Compute aggregate metrics: mean ± std-dev across seeds per metal.
4. Compute aggregate metrics: mean ± std-dev across metals per seed.
5. Compute aggregate metrics: grand mean ± std-dev across all (metal, seed) pairs.
6. Test primary hypothesis: paired t-test (sort vs sliced, 15 pairs).
7. Emit `training_summary.json` + per-metal breakdown JSON.
8. If acceptance gate PASS: declare HYPOTHESIS_CONFIRMED.
9. If acceptance gate FAIL: declare HYPOTHESIS_REJECTED + emit honest-negative closeout.

**Output paths:**

| Artifact | Path |
|---|---|
| Per-pair metrics | `${OUT_BASE}/<metal>/seed<seed>/per_seed_metrics_seed<seed>.json` |
| Training summary | `${OUT_BASE}/training_summary.json` |
| Master log | `${OUT_BASE}/master.log` |
| Honest-negative closeout (if rejected) | `TODO/completed/<wf_id>_<short_name>.md` |

**Pre-registered analysis stop rule:** <e.g. "Do not re-run analysis with alternative aggregations after the first computation; document any deviation as ANALYSIS_DEVIATION with justification.">

---

## §7 Anti-patterns: NO post-hoc cherry-picking

The following actions are **prohibited** after the launch begins, except where explicitly noted as "pre-registered":

1. **Excluding metals/seeds** post-hoc to improve the headline metric. Pre-register the count + list BEFORE launch.
2. **Re-running with different hyperparameters** and reporting the better one as "the result". If a hyperparameter change is needed, document as a NEW launch with a NEW pre-registration record.
3. **Re-framing the primary metric** after seeing the results. If the original primary metric shows no lift, declare HYPOTHESIS_REJECTED — do not pivot to a secondary metric as the new headline.
4. **Cherry-picking cohorts** that show the expected effect while hiding cohorts that don't. Pre-register the cohort definition + exclusion criteria BEFORE launch (§2).
5. **Reporting PROJECTED numbers as MEASURED** (party-twenty-spirit §3.5). Every numeric value MUST trace to a JSON / report path on disk.
6. **Silently changing the config file** between pre-registration and launch. If the config changes, the pre-registration record is INVALID — emit a new pre-registration record first.
7. **Selective reporting of ablation results** (only showing the configs that confirm the hypothesis). Pre-register the full ablation plan (§5) and report ALL results, including HYPOTHESIS_REJECTED configs.

**Audit trail:** the launcher's `master.log` + `training_summary.json` + per-pair `per_seed_metrics_seed*.json` are the canonical evidence; any deviation from the pre-registered analysis plan (§6) MUST be documented in `TODO/completed/<wf_id>_deviations.md` with justification + reviewer sign-off.

---

## §8 Sign-off

**Pre-registration author:** <name + date>
**Pre-registration reviewer:** <name + date; optional but recommended for paper-grade launches>
**Launch authorized at:** <date + time>
**Launch ID:** <e.g. "Wf-350-B2 G5 launch (TODO-181)">
**Launcher script:** `bash scripts/<launch_script>.sh --prereg docs/prereg/<this_file>.md`

---

## §9 References

- AGENTS.md §5.1 standing GPU authorization (2026-09-29 user grant)
- AGENTS.md §5.2 honest framing mandate (party-twenty-spirit §3.5)
- AGENTS.md §5.3 no paper work without authorization (TODO-84 active)
- AGENTS.md §5.4 Tier-2 journal target + NO wet-lab planning
- TODO/INDEX.md §0 + §22 (recent Wf history)
- TODO/completed/207_unified_pre_flight_checklist_v2.md (pre-flight gate catalog)
- TODO/completed/343a1_industry.md §4 recommendation #2 (prereg.md origin)
- TODO/completed/343a5_audit_207.md row 8 (pre-research framing DeepMind §3.1)
- TODO/completed/343a6_audit_per_wave.md row "Pre-registration record"
- TODO/completed/350b2_prereg_template.md (this template's ship doc)
- DeepMind responsible research framework §3.1 (pre-registration)
- party-twenty-spirit §3.5 (MEASURED-vs-PROJECTED tag discipline)

---

**END OF PREREG_TEMPLATE.md** — Tier-2 marker: USER_OK-gated; not auto-enforced.
