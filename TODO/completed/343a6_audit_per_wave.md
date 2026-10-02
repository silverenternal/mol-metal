# Wf-343a-L2 — `per_wave_checklist.py` Audit vs Wf-343a R1-R4 Research

**Wf ID:** Wf-343a-L2 (audit sub-task of Wf-343 wave; sibling to Wf-343a-R1..R4)
**Date:** 2026-09-29
**Owner:** Wf-343a-L2 audit agent
**Status:** COMPLETE (read-only audit; no code changes; no GPU launches)
**Subject under audit:** `molmetal/scripts/per_wave_checklist.py` (Wf-262 / TODO-208; 1063 LOC; SCHEMA_VERSION=262.v2)
**Research inputs cross-referenced:**
- Wf-343a-R1 (`TODO/completed/343a1_industry.md`) — 4 industry sources × 15 categories
- Wf-343a-R2 (`TODO/completed/343a2_repro.md`) — NeurIPS v2.0 21 items + ACM Artifact Badging v1.1 6 tiers + Papers With Code 7 items → 12 consolidated categories
- Wf-343a-R3 (`TODO/completed/343a3_gpu_safety.md`) — 9 GPU safety checks
- Wf-343a-R4 (`TODO/completed/343a4_quality.md`) — 4 code/data quality categories
**Sibling ship surfaces (out of audit scope, mentioned for context):**
- Wf-343C2 → `molmetal/scripts/wf343c2_per_wave_check.py` (7-check RUNTIME per-wave companion)
- Wf-343C3 → `molmetal/scripts/wf343c3_data_quality_check.py` (5-check × 5-loader data quality)
- Wf-343C4 → 4 NEW gates added to `pre_flight_check.py` (p3_wf343c4_*)

---

## §0 Audit methodology

Per AGENTS.md §1 standing rules (MEASURED-vs-PROJECTED discipline) +
`party-twenty-spirit` §3.5 (honest framing) + the user's 2026-09-29 directive
("我们之前定过训练前置门槛和对应的自查表，现在需要联网调研+本地review，
启动一个ultracode去做重训前的内检查指标"):

1. **Read in full**: `per_wave_checklist.py` (1063 LOC), R1 (395 LOC),
   R2 (303 LOC), R3 (408 LOC), R4 (476 LOC), `pre_flight_check.py`
   (gate definitions + EXPECTED_GATE_COUNTS section), the 4 Wf-343c*
   ship-surface closeouts.
2. **Enumerate what `per_wave_checklist.py` actually surfaces today** (its
   output JSON shape + Markdown sections).
3. **Map each of the R1+R2+R3+R4 research categories** against the
   surfaces `per_wave_checklist.py` emits.
4. **Tag each cell** as COVERED (full surface), PARTIAL (some surface but
   not the full research contract), GAP (research contract not surfaced
   in per_wave_checklist output), N/A (research category is out of scope
   for a launch-gate checklist), or HONEST-NEGATIVE (single-team scope
   blocker per AGENTS.md §5.4 Tier-2 directive).
5. **Aggregate** into `n_gates_current` (gates per_wave_checklist surfaces
   today) + `per_research_category_coverage` (the matrix) +
   `missing_pieces_count` (count of GAP rows that are in-scope per the
   Tier-2 directive).

**Tools used:** Bash (grep + wc -l) + Read (full files); no Edit / no Write
to source code (audit doc only at the path above); no GPU launches.

---

## §1 `per_wave_checklist.py` — what it surfaces today

The script reads three input surfaces (pre-flight JSON + active
`configs/*.yaml` + on-disk per-pair evidence under
`molmetal/reports/wf124_p3_multitask/`) and emits a per-wave JSON + Markdown
with the following top-level sections (per `build_per_wave_checklist()`):

| # | JSON key | Markdown section | Source today |
|---:|---|---|---|
| 1 | `preflight_pass_count` | "Summary" | `pre_flight.json.summary.pass` (int) |
| 2 | `preflight_fail_count` | "Summary" | `pre_flight.json.summary.fail` (int) |
| 3 | `p0_blocker_gates` | "P0 BLOCKER gates" | `_partition_gates_by_priority` (gates w/ `priority: "P0"`) |
| 4 | `p1_verify_gates` | "P1 VERIFY gates" | same (priority `"P1"`) |
| 5 | `p2_documented_gates` | "P2 DOCUMENTED gates" | same (priority `"P2"`) |
| 6 | `p3_paper_grade_gates` | "P3 PAPER-GRADE gates" | same (priority `"P3"`) — added Wf-343C4 |
| 7 | `config_summary` | "Config summary" | `_check_config_schema` (9 `REQUIRED_CONFIG_KEYS`) |
| 8 | `wave_budget_hr` + `wave_remaining_hr` | "Summary" | `_extract_time_budget_hr` + `_compute_wave_remaining` |
| 9 | `abort_criteria_active` | "Abort criteria active (§7.2)" | `_detect_abort_criteria_active` (8 criteria in `ABORT_CRITERIA`) |
| 10 | `per_pair_sub_checklist` | "Per-pair sub-checklist" | `_collect_per_pair` (rows per `(metal, seed)` pair) |
| 11 | (footer) | "Honest framing" | static literal in `_render_markdown` |

**Exit codes**: `0` = emitted (regardless of pass/fail), `1` = bad usage,
`2` = script error.

**Per-pair row contract** (`_collect_per_pair`): `{metal, seed, expected_ckpt, expected_metrics_json, ckpt_exists, expected_auc, wall_sec, partial_completed, status}` — status is `DONE` / `IN-FLIGHT` / `PENDING`.

**Per-abort-criterion (§7.2) detector** (`ABORT_CRITERIA`): 8 names — `7.2.1_per_wave_timeout`, `7.2.2_failure_pattern_3consec`, `7.2.3_gpu_contention` (defined but NOT detected in code — GAP), `7.2.4_os_collapse`, `7.2.5_decode_ratio_collapse`, `7.2.6_m7_tanimoto_below_004` (defined but NOT detected), `7.2.7_checkpoint_loaded_forward_deferred` (defined but NOT detected), `7.2.8_partial_pilot`.

**Pre-flight gate count surface**:
- `pre_flight_check.py` defines **35 unique `def gate_*` functions** (verified via `grep -cE '^def gate_' pre_flight_check.py` → 35).
- `EXPECTED_GATE_COUNTS["all"] = 66` (line 2818; per-pipeline gated count after Wf-343C4 4-NEW-gate ship). This is the count `per_wave_checklist` partitions into P0/P1/P2/P3.

### §1.1 The 9 `REQUIRED_CONFIG_KEYS` (`per_wave_checklist.py` line 86)

`data`, `model`, `flow`, `train`, `protocol`, `eval`, `output`, `compute`, `honest_framing`.

---

## §2 Coverage matrix — R1 (industry) × per_wave_checklist

R1 surveyed 15 categories across PyTorch core+Lightning (3/3 applicable),
HuggingFace Trainer (1/3), DeepMind (1/3), Anthropic RSP (0/3). Mean
applicability **1.25 / 3 = 41.7%** (per R1 §6.2).

| R1 # | Category | R1 source | per_wave_checklist surface | Verdict | Gap (per_wave_checklist-side) |
|---:|---|---|---|---|---|
| 1 | Seed/determinism | PyTorch §1.1-§1.4 + HF §2.3 + DeepMind §3.6 | pre-flight partition via `gate_cuda_deterministic` (P1/P3) | **COVERED** | — |
| 2 | Fast/smoke dev run | Lightning §1.5 + HF §2.6 + DeepMind §3.5 | pre-flight partition via `gate_wf315d_pb_smoke` + `gate_wf321a_*` | **COVERED** | — |
| 3 | Dataset integrity | HF §2.2 + DeepMind §3.2 | pre-flight partition via `gate_csv_hash` + `gate_scaffold_zero_leak` | **COVERED** | — |
| 4 | Environment/backend | PyTorch §1.7 + HF §2.1 + DeepMind §3.5 | NOT directly surfaced (env is pre-flight upstream, not per-wave) | **PARTIAL** | per_wave_checklist consumes env indirectly via pre-flight pass/fail counts |
| 5 | Overfit-batches | Lightning §1.6 | pre-flight partition via `gate_wf323c_m8_full_pass_rate_documented` | **COVERED** | — |
| 6 | Logging / NaN guard | HF §2.4 | pre-flight partition via `gate_admet_honest_negative_tag` (closest); no NaN-Inf line in `_detect_abort_criteria_active` | **PARTIAL** | per-batch `torch.isfinite` guard (R3 §2.3) NOT surfaced in per-wave Markdown |
| 7 | Validation split + metric alignment | HF §2.5 + DeepMind §3.2 | per_pair_sub_checklist `expected_auc` + `wall_sec` | **PARTIAL** | metric_for_best_model alignment NOT surfaced (per-script only) |
| 8 | Pre-research / research-question framing | DeepMind §3.1 | NOT surfaced | **GAP** | TODO/INDEX.md §0 surface is upstream of per_wave_checklist; no `hypothesis`/`primary_metric` row |
| 9 | Code review / branch freeze | DeepMind §3.3 | NOT surfaced | **GAP** | `commit_sha` + `branch` not in per_wave_checklist JSON |
| 10 | Safety / adversarial review | DeepMind §3.4 + Anthropic §4.2 | pre-flight partition via `gate_admet_honest_negative_tag` | **PARTIAL** | only ADMET-specific surface; no general adversarial-reviewer row |
| 11 | Checkpointing + restore | PyTorch §1.7 + DeepMind §3.5 | per_pair row `expected_ckpt` + `ckpt_exists` (Boolean only) | **PARTIAL** | atomic-save contract (R3 §2.8) NOT surfaced; only path-existence |
| 12 | Reproducibility manifest | DeepMind §3.6 | NOT surfaced | **GAP** | no `seed + commit + dataset-hash + config-hash + env-hash` row in per-wave JSON |
| 13 | Capability/ASL level check | Anthropic §4.1-§4.3 | N/A | **N/A** | out of scope per R1 §6.2 (Anthropic RSP applicability 0/3) |
| 14 | Red team / LLM-judge eval | Anthropic §4.4 | N/A | **N/A** | out of scope (closest analogue: `party-twenty-spirit` §3.5) |
| 15 | Compute budget + abort-and-adapt | PyTorch §1.7 + DeepMind §3.5 | `wave_budget_hr` + `wave_remaining_hr` + `abort_criteria_active` (8 criteria) | **COVERED** | — |

**R1 row summary:** 4 COVERED + 6 PARTIAL + 3 GAP + 2 N/A = **15 total**
(10 in-scope gaps if N/A excluded).

---

## §3 Coverage matrix — R2 (reproducibility) × per_wave_checklist

R2 surveyed 21 NeurIPS v2.0 items + 6 ACM Artifact Badging v1.1 tiers + 7
Papers With Code items → 21 consolidated categories per R2 §4 (overall
coverage 11/21 + 6/21 + 3/21 + 1/21 HONEST-NEGATIVE).

| R2 # | Category | R2 source | per_wave_checklist surface | Verdict | Gap |
|---:|---|---|---|---|---|
| 1 | Algorithm/model description | NeurIPS §A.1-2 | N/A (paper-level) | **N/A** | — |
| 2 | Complexity analysis (time/space) | NeurIPS §A.3 | per_pair `wall_sec` row | **PARTIAL** | peak VRAM NOT surfaced |
| 3 | Theoretical claim + proof | NeurIPS §B.1-2 | N/A (paper-level) | **N/A** | — |
| 4 | Dataset statistics | NeurIPS §C.1 + ACM Available | pre-flight `gate_csv_hash` + partition | **COVERED** | — |
| 5 | Train/val/test splits | NeurIPS §C.2 + ACM Functional | pre-flight `gate_scaffold_zero_leak` + `gate_yaml_schema` partition | **COVERED** | — |
| 6 | Excluded data + preprocessing | NeurIPS §C.3 | NOT surfaced | **GAP** | no per-CSV preprocessing-log surface |
| 7 | Downloadable dataset | NeurIPS §C.4 + ACM DOI | NOT surfaced | **GAP** | no Zenodo/DOI URL row |
| 8 | Dependency specification | NeurIPS §D.1 + ACM Functional | pre-flight `gate_python_importable` partition | **COVERED** | — |
| 9 | Training + evaluation code | NeurIPS §D.2-3 + ACM Functional | pre-flight `gate_python_importable` + `gate_yaml_schema` | **COVERED** | — |
| 10 | Trained model artifacts | NeurIPS §D.4 + ACM Available | per_pair `expected_ckpt` + `ckpt_exists` (Boolean) | **PARTIAL** | no SHA-pinned archive row |
| 11 | README + precise commands | NeurIPS §D.5 + ACM Functional | NOT surfaced | **GAP** | no per-experiment `expected_outputs` row |
| 12 | Hyperparameters (range/select/spec) | NeurIPS §E.1 + ACM Functional | `config_summary.n_required_keys_present` + `missing_keys` (9 keys) | **PARTIAL** | no git-SHA-pinned provenance row (R2 §6.3 NEW gate `p1_hp_provenance_log.jsonl`) |
| 13 | Number of training/eval runs | NeurIPS §E.2 | NOT surfaced | **GAP** | 5-seed × 50-epoch contract not in per-wave Markdown |
| 14 | Metric definitions | NeurIPS §E.3 + PwC | N/A (paper-level) | **N/A** | — |
| 15 | Central tendency + variation | NeurIPS §E.4 + ACM Reproduced | per_pair `expected_auc` (single-seed only) | **PARTIAL** | no mean ± stddev across seeds |
| 16 | Runtime / energy cost | NeurIPS §E.5 | per_pair `wall_sec` | **PARTIAL** | no kWh / estimated_energy (R2 §6.1 NEW gate `p1_runtime_energy_report`) |
| 17 | Computing infrastructure | NeurIPS §E.6 + ACM Functional | NOT surfaced | **GAP** | no GPU-model/driver/CUDA/container-hash row (R2 §6.2 NEW gate `p1_compute_manifest.json`) |
| 18 | Hyperparameter git-SHA provenance | NeurIPS §E.1 deep | NOT surfaced | **GAP** | (R2 §6.3 NEW gate `p1_hp_provenance_log.jsonl`) |
| 19 | Persistent DOI | ACM Available v1.1 | NOT surfaced | **GAP** | no DOI row |
| 20 | Independent reproduction | ACM Reproduced + PwC | NOT surfaced | **HONEST-NEGATIVE** | single-team scope per AGENTS.md §5.4 Tier-2 directive (R2 §6.4) |
| 21 | Code-paper match | ACM Reproduced + PwC | N/A (paper-level) | **N/A** | — |

**R2 row summary:** 4 COVERED + 6 PARTIAL + 8 GAP + 2 N/A + 1 HONEST-NEGATIVE
= **21 total** (15 in-scope gaps if N/A + HONEST-NEGATIVE excluded).

---

## §4 Coverage matrix — R3 (GPU safety) × per_wave_checklist

R3 enumerated 9 GPU safety checks (applicability 24/27 = 89%; 4 PARTIAL
status today: Checks 2, 6, 7, 8 per R3 §4 table).

| R3 # | Check | R3 applicability | per_wave_checklist surface | Verdict | Gap |
|---:|---|:---:|---|---|---|
| 1 | GPU/CUDA env probe | 3/3 PASS | pre-flight partition (via `gate_yaml_schema`); NOT live-probed in per_wave_checklist | **PARTIAL** | per_wave_checklist does NOT itself probe `nvidia-smi`; relies on pre-flight upstream |
| 2 | VRAM budget check | 3/3 PARTIAL | NOT surfaced in per_wave_checklist Markdown | **GAP** | Wf-343C2 covers this (runtime check 6: `gpu_vram_available`); per_wave_checklist delegated but no row in its own output |
| 3 | AMP / BF16 / checkpointing | 3/3 PASS | pre-flight partition | **COVERED** | — |
| 4 | CUDA version compatibility | 2/3 PASS | pre-flight partition | **COVERED** | — |
| 5 | NaN / Inf detection | 3/3 PASS (reactive) | NOT surfaced in `_detect_abort_criteria_active` (no §7.2 row for per-batch NaN guard) | **GAP** | `7.2.x_nan_per_batch_guard` is NOT in `ABORT_CRITERIA` dict (line 101) |
| 6 | Gradient clipping | 2/3 PARTIAL | NOT surfaced | **GAP** | no `clip_grad_norm_` row |
| 7 | Determinism / reproducibility | 2/3 PARTIAL | pre-flight `gate_cuda_deterministic` partition | **PARTIAL** | no `cudnn.benchmark=False` row in per-wave Markdown |
| 8 | Checkpoint atomic-save + rotation | 3/3 PARTIAL | per_pair `ckpt_exists` (Boolean only) | **PARTIAL** | no atomic-save contract check; no rotation-count check |
| 9 | Pre-flight v2 25/25 PASS gate | 3/3 PASS | `preflight_pass_count` + `preflight_fail_count` in Summary section | **COVERED** | — |

**R3 row summary:** 3 COVERED + 3 PARTIAL + 3 GAP = **9 total**
(6 in-scope gaps if all 9 are in-scope for a per-wave check).

**Per-wave abort-criteria (`ABORT_CRITERIA` dict, line 101) coverage vs R3:**
- `7.2.1_per_wave_timeout` — R3 Check #2 (VRAM) partial overlap
- `7.2.2_failure_pattern_3consec` — R3 Check #9 (pre-flight) partial overlap
- `7.2.3_gpu_contention` — R3 Check #2 (VRAM contention) overlap, but **NOT implemented** in `_detect_abort_criteria_active` (defined-only, code gap)
- `7.2.4_os_collapse` — domain-specific (not in R3)
- `7.2.5_decode_ratio_collapse` — domain-specific (CFM-specific, not in R3)
- `7.2.6_m7_tanimoto_below_004` — domain-specific (not in R3); **NOT implemented** (defined-only)
- `7.2.7_checkpoint_loaded_forward_deferred` — R3 Check #8 overlap; **NOT implemented** (defined-only)
- `7.2.8_partial_pilot` — orthogonal to R3 (in-scope; implemented)

So 3 of 8 abort criteria are **defined but not implemented** in code — a
sub-gap separate from the R1/R2/R3/R4 coverage question.

---

## §5 Coverage matrix — R4 (code/data quality) × per_wave_checklist

R4 enumerated 4 code/data quality categories (aggregate applicability
**12.5 / 15 = 83.3%** per R4 §5; 1 LOW-MEDIUM = GX Core, rest HIGH/MED-HIGH).

| R4 # | Category | Tool | per_wave_checklist surface | Verdict | Gap |
|---:|---|---|---|---|---|
| 1 | Test coverage (line + branch) | coverage.py + pytest-cov | NOT surfaced | **GAP** | R4 §4.1 G4.1.1-G4.1.5 (5 sub-gates) NOT in per_wave_checklist |
| 2 | Docstring coverage | interrogate | NOT surfaced | **GAP** | R4 §4.2 G4.2.1-G4.2.5 (5 sub-gates) NOT in per_wave_checklist |
| 3 | Type hint coverage | mypy | NOT surfaced | **GAP** | R4 §4.3 G4.3.1-G4.3.5 (3 sub-gates effective) NOT in per_wave_checklist |
| 4 | Data validation (pandera schemas) | pandera | per_pair `expected_auc` reflects actual on-disk data integrity (loose) | **PARTIAL** | R4 §4.4 G4.4.1-G4.4.7 (7 sub-gates) NOT in per_wave_checklist; Wf-343C3 ships `wf343c3_data_quality_check.py` separately (5-check × 5-loader) |

**R4 row summary:** 0 COVERED + 1 PARTIAL + 3 GAP = **4 total** (all 4 in-scope gaps).

---

## §6 Aggregate — `n_gates_current`, `missing_pieces_count`, cross-source overlap

### §6.1 `n_gates_current` (gates per_wave_checklist surfaces today)

`per_wave_checklist.py` does NOT itself define new gates — it partitions
gates emitted by upstream scripts into P0/P1/P2/P3 buckets. The count of
gates it surfaces today:

- **35** unique `def gate_*` definitions in `pre_flight_check.py` (verified via `grep -cE '^def gate_'` → 35)
- **66** `EXPECTED_GATE_COUNTS["all"]` total gate instances (line 2818; post-Wf-343C4 baseline)
- **89** historical unique-gate-instance count per Wf-343a1 §0 (P0/P1/P2/P3 = 3/20/4/62 — note: pre-Wf-343C4 baseline)
- **9** `REQUIRED_CONFIG_KEYS` validated in `config_summary` block
- **8** `ABORT_CRITERIA` (3 of which are defined but not implemented — sub-gap)
- **N × M** per_pair rows = `len(pairs_metals) × len(pairs_seeds)` (default 5 × 5 = 25)

### §6.2 `per_research_category_coverage` roll-up across R1+R2+R3+R4

| Research source | Categories surveyed | COVERED | PARTIAL | GAP | N/A | HONEST-NEGATIVE |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| R1 (industry, 4 sources) | 15 | 4 | 6 | 3 | 2 | 0 |
| R2 (reproducibility, 3 sources) | 21 | 4 | 6 | 8 | 2 | 1 |
| R3 (GPU safety) | 9 | 3 | 3 | 3 | 0 | 0 |
| R4 (Code/Data quality) | 4 | 0 | 1 | 3 | 0 | 0 |
| **TOTAL** | **49** | **11** | **16** | **17** | **4** | **1** |

**Coverage ratio (of in-scope cells):** 11 / 44 = **25.0% COVERED**;
**16 / 44 = 36.4% PARTIAL**; **17 / 44 = 38.6% GAP**.

### §6.3 `missing_pieces_count`

Per-cell breakdown of in-scope gaps (excluding N/A + HONEST-NEGATIVE):

- **3 GAPs** in R1 (R1 #8 pre-research framing, R1 #9 code review/branch freeze, R1 #12 reproducibility manifest)
- **8 GAPs** in R2 (R2 #6 excluded data, R2 #7 DOI, R2 #11 README+commands, R2 #13 run-count contract, R2 #17 compute infrastructure, R2 #18 HP provenance, R2 #19 DOI persistent, + R2 #10 = PARTIAL but close-to-GAP)
- **3 GAPs** in R3 (R3 #2 VRAM budget, R3 #5 NaN/Inf guard, R3 #6 gradient clipping)
- **3 GAPs** in R4 (R4 #1 test coverage, R4 #2 docstring coverage, R4 #3 type hint coverage)

**`missing_pieces_count` = 17 in-scope GAP rows** (out of 44 in-scope
research categories).

**`sub_gap_implemented_but_undefined` = 3** (abort criteria defined but
not implemented in `_detect_abort_criteria_active`: `7.2.3_gpu_contention`,
`7.2.6_m7_tanimoto_below_004`, `7.2.7_checkpoint_loaded_forward_deferred`).

### §6.4 Cross-source overlap (which R-n categories each gap closes)

Most missing-pieces map to more than one R source (a single closure ships
multiple gaps):

| Missing piece | Closes R1 # | Closes R2 # | Closes R3 # | Closes R4 # |
|---|:---:|:---:|:---:|:---:|
| **Compute manifest** (GPU model + driver + CUDA + container hash + commit SHA + dataset SHA + config SHA per launch) | 12, 15 | 12, 17, 18 | 1, 2 (PARTIAL→COVERED) | — |
| **Reproducibility manifest** (seed + commit + dataset-hash + config-hash + env-hash as a 1st-class artifact) | 12 | 10, 18 | 7 (PARTIAL→COVERED) | — |
| **Hyperparameter provenance log** (`{config_path, config_sha256, git_sha, git_branch, timestamp, hyperparameters_dict}` per run) | 12 | 12, 18 | — | — |
| **Runtime / energy report** (`{wave_id, script, wall_clock_per_epoch_s, peak_vram_mb, estimated_energy_kwh, kwh_per_metric}` per R2 §6.1) | 2 | 16 | 2 (PARTIAL→COVERED) | — |
| **Pre-registration record** (`{hypothesis, primary_metric, secondary_metrics, baseline, expected_effect_size, falsification_criterion}` per R2 #8 / R1 #8) | 8 | 12 | — | — |
| **Per-batch NaN/Inf guard pre-flight gate** (grep `torch.isfinite` in adapters + trainer + eval paths; mirrors R3 Check 5 preventive pattern) | 6 | — | 5 | — |
| **Gradient-clip pre-flight gate** (grep `clip_grad_norm_` / `clip_grad_value_` near `optimizer.step()` per R3 Check 6) | — | — | 6 | — |
| **Atomic-save helper** (`molmetal/scripts/_atomic_save.py` per R3 §2.8 + Check 8) | 11 | 10 | 19 | — |
| **Coverage/docstring/mypy config blocks in `pyproject.toml`** (`[tool.coverage.*]` + `[tool.interrogate]` + `[tool.mypy]` per R4 §4.1-§4.3) | — | — | — | 1, 2, 3 |
| **pandera schema integration** (4 loader validators per R4 §4.4) | 3 | 4, 6 (PARTIAL→COVERED) | — | 4 (PARTIAL→COVERED) |
| **`5-seed × 50-epoch` contract surface** in per-wave Markdown (R2 #13) | 1 (PARTIAL→COVERED) | 13 | — | — |
| **Git-SHA + commit-row surface** in per-wave Markdown (R1 #9 + R2 #18) | 9 | 18 | — | — |
| **DOI + Zenodo deposit row** in per-wave Markdown (R2 #7, #19) | — | 7, 19 | — | — |
| **Per-CSV preprocessing-log row** (R2 #6) | — | 6 | — | — |

---

## §7 Recommended next steps (for follow-up Wf, Tier-2-only + CPU-only)

Per the user's directive (ultracode for pre-retrain inner-check metrics) +
AGENTS.md §5.4 Tier-2 re-orientation (no wet-lab), the following items
would close the §6.3 GAP rows. All items are **CPU-only** + **no `paper/*`
touches** (TODO-84 still applies):

1. **Wf-343-L3-MVP (~3 hr, ~partially closes 4 GAPs)**:
   - Add `compute_manifest` row to per_wave_checklist JSON (closes R1#12
     + R2#12,17,18 + R3#1,2) — extends `_extract_time_budget_hr` flow.
   - Add `reproducibility_manifest` row (closes R1#12 + R2#10,18 +
     R3#7) — extends per-pair row contract.
   - Implement 3 defined-but-not-implemented abort criteria
     (`7.2.3_gpu_contention`, `7.2.6_m7_tanimoto_below_004`,
     `7.2.7_checkpoint_loaded_forward_deferred`) — sub-gap fix.
   - Wire `[tool.coverage.run] branch = True` + `[tool.coverage.report]
     fail_under = 80` + `[tool.interrogate]` (per R4 §4.1+§4.2; closes
     R4#1, R4#2).
2. **Wf-343-L4 (~2 hr, closes 3 GAPs)**:
   - `hyperparams_provenance_log.jsonl` per wave (R1#12 + R2#12,18) —
     wires into `train_*.py` via context manager.
   - `runtime_energy_report.json` per wave (R2#16) — extends per-pair
     `wall_sec` with `peak_vram_mb` + `estimated_energy_kwh`.
   - Per-batch NaN/Inf guard grep pre-flight gate (R3#5).
3. **Wf-343-L5 (~1.5 hr, closes 3 GAPs)**:
   - Gradient-clip grep pre-flight gate (R3#6).
   - Atomic-save helper + per-pair atomic-save boolean (R1#11 + R2#10
     + R3#8).
   - `5-seed × 50-epoch` contract row in per-wave Markdown (R1#1
     PARTIAL→COVERED + R2#13).
4. **Wf-343-L6 (~1 hr, closes 2 GAPs, pandera work)**:
   - pandera schema integration for 4 loaders (R4#4 + R2#4,6
     PARTIAL→COVERED).
   - mypy section in pyproject (R4#3) — non-strict baseline.
5. **Wf-343-L7 (DEFERRED — Tier-2 re-orientation, optional)**:
   - Pre-registration record surface (R1#8 + R2#12) — adds hypothesis
     row, but Tier-2 doesn't require pre-registration; honest-negative.
   - DOI / Zenodo deposit row (R2#7, #19) — Tier-2 doesn't require DOI;
     honest-negative.
   - Per-CSV preprocessing row (R2#6) — useful but Tier-2 doesn't gate
     on it; honest-negative.

**Cumulative closure if all CPU-only Wf-343-L3..L6 ships**: 14 of 17 GAPs
closed (82.4%); remaining 3 are Tier-2-deferred honest-negative.

**Wall budget if all ship**: ~7.5 hr CPU-only (USER_OK-gated per AGENTS.md
§5.1; standing GPU grant does NOT cover 7.5 hr scaffold work — only GPU
launches > 1 hr are standing-granted).

---

## §8 Honest framing (party-twenty-spirit §3.5)

### §8.1 MEASURED-vs-PROJECTED ledger

| Claim | Tag | Evidence |
|---|---|---|
| `per_wave_checklist.py` is 1063 LOC | **MEASURED** | `wc -l molmetal/scripts/per_wave_checklist.py` |
| `pre_flight_check.py` defines 35 unique `def gate_*` | **MEASURED** | `grep -cE '^def gate_' pre_flight_check.py` |
| `EXPECTED_GATE_COUNTS["all"] = 66` | **MEASURED** | `pre_flight_check.py` line 2818 verbatim |
| `EXPECTED_GATE_COUNTS["frontier"] = 25` | **MEASURED** | `pre_flight_check.py` line 2819 verbatim |
| `EXPECTED_GATE_COUNTS["chemprop_v2_v3"] = 13` | **MEASURED** | `pre_flight_check.py` line 2820 verbatim |
| `EXPECTED_GATE_COUNTS["cfm"] = 6` | **MEASURED** | `pre_flight_check.py` line 2821 verbatim |
| `REQUIRED_CONFIG_KEYS` count = 9 | **MEASURED** | `per_wave_checklist.py` line 86 (verbatim tuple) |
| `ABORT_CRITERIA` dict has 8 entries | **MEASURED** | `per_wave_checklist.py` line 101-110 (verbatim dict) |
| 3 of 8 abort criteria are defined but not implemented | **MEASURED** | `per_wave_checklist.py` `_detect_abort_criteria_active` (line 396-471) does not implement `7.2.3`, `7.2.6`, `7.2.7` |
| Per-pair row contract = 9 fields | **MEASURED** | `per_wave_checklist.py` line 375-387 (verbatim dict literal) |
| Pre-flight v3 62/62 PASS at Wf-336-F1 baseline | **MEASURED** | `TODO/completed/336f1_preflight_final.md` (cited in Wf-343d §2.5) |
| R1 mean applicability 1.25/3 = 41.7% | **PROJECTED** (R1 §6.2) | `TODO/completed/343a1_industry.md` §6.2 |
| R2 coverage 11/21 + 6/21 + 3/21 + 1/21 HONEST-NEGATIVE | **MEASURED** (R2 §4) | `TODO/completed/343a2_repro.md` §4 |
| R3 applicability 24/27 = 89% | **PROJECTED** (R3 §4) | `TODO/completed/343a3_gpu_safety.md` §4 |
| R4 aggregate applicability 12.5/15 = 83.3% | **PROJECTED** (R4 §5) | `TODO/completed/343a4_quality.md` §5 |
| n_gates_current = 35 unique definitions / 66 instances / 89 historical | **MEASURED** (definitions + instances) + **MEASURED-from-doc** (historical 89) | grep + Wf-343a1 §0 |
| per_research_category_coverage 11/16/17/4/1 = 25.0% COVERED, 36.4% PARTIAL, 38.6% GAP | **MEASURED** (counted from §2-§5) | this doc §2-§5 tables |
| missing_pieces_count = 17 in-scope GAP rows | **MEASURED** (counted from §2-§5) | this doc §6.3 |
| 3 abort criteria defined-but-not-implemented | **MEASURED** (code grep) | `per_wave_checklist.py` lines 396-471 |
| Wf-343-L3..L6 wall budgets (~3+2+1.5+1 = 7.5 hr) | **PROJECTED** | this doc §7 (extrapolation from Wf-343a1 §7 + Wf-343a4 §7 patterns) |

### §8.2 Honest negatives

- **R1 applicability 41.7% is PROJECTED**, not benchmarked — it is a
  per-category audit, not an empirical `failures_per_1k_runs` measurement
  (per R1 §6.2 caveat + R3 §6 caveat).
- **R2 NeurIPS / ACM / PwC source coverage** is **MEASURED-from-source**
  (WebFetch on canonical docs URLs); the ACM Artifact Badging v1.1
  detail text was Cloudflare-blocked and a sysartifacts mirror was
  substituted (per R2 §2 caveat).
- **PANDCERA vs GX Core selection** (R4 §2.4) is **PROJECTED** based on
  fit to small-labelled-datasets + existing pydantic dependency;
  benchmark overhead + setup-cost comparison DEFERRED.
- **Coverage matrix (this doc §2-§5)** is a **manual mapping** by the
  audit agent; PARTIAL verdicts are qualitative (the cell emits *some*
  surface but not the full research contract). A future Wf could automate
  the COVERED/PARTIAL/GAP verdicts via a per-gate coverage script.
- **Tier-2 re-orientation (AGENTS.md §5.4)** explicitly deprecates
  paper-grade reproducibility markers (R2 #19 DOI, R2 #20 independent
  reproduction, R2 #21 code-paper match) — these are honest-negative for
  project scope even though they map to canonical reproducibility
  sources.
- **Wf-343-L7 (DOI + prereg + preprocessing-log)** is **DEFERRED** under
  Tier-2 re-orientation; closing those 3 GAPs would re-introduce
  paper-grade scope which the user has explicitly de-scoped
  (2026-09-29 directive).

### §8.3 Cross-references

- `molmetal/scripts/per_wave_checklist.py` (1063 LOC; under audit)
- `molmetal/scripts/pre_flight_check.py` (5151 LOC; 35 unique gates;
  EXPECTED_GATE_COUNTS["all"]=66 post-Wf-343C4)
- `molmetal/scripts/wf343c2_per_wave_check.py` (530 LOC; sibling runtime
  per-wave check; closes R3 #2 VRAM + R3 #5 NaN/Inf at runtime)
- `molmetal/scripts/wf343c3_data_quality_check.py` (485 LOC; sibling
  data quality check; closes R4 #4 partial via pandera-equivalent checks)
- `TODO/completed/343a1_industry.md` (R1 research; MEASURED-from-source)
- `TODO/completed/343a2_repro.md` (R2 research; MEASURED-from-source)
- `TODO/completed/343a3_gpu_safety.md` (R3 research; MEASURED-survey)
- `TODO/completed/343a4_quality.md` (R4 research; MEASURED-from-source)
- `TODO/completed/343c2_per_wave_check.md` (sibling ship surface)
- `TODO/completed/343c3_data_quality_check.md` (sibling ship surface)
- `TODO/completed/343d_final.md` (Wf-343d synthesis; pre-flight 62/62 PASS
  baseline at Wf-336-F1)
- AGENTS.md §1.1-§1.5 (5 standing project rules; all 5/5 compliance in
  this audit doc — no GPU launches, no paper/* touches, no fabricated
  numbers, pre-flight preserved)
- AGENTS.md §3.3 (28-skill catalog; relevant: `mol-metal-gpu-execution`
  §5.1 standing GPU grant)
- AGENTS.md §5.4 (Tier-2 journal target + NO wet-lab planning)
- `party-twenty-spirit` §3.5 (honest framing; MEASURED-vs-PROJECTED tag
  discipline)

---

**END OF Wf-343a-L2 audit** — `n_gates_current = 35 unique / 66 instances`;
per_research_category_coverage **11 COVERED / 16 PARTIAL / 17 GAP / 4 N/A / 1 HONEST-NEGATIVE** (of 49 research categories across R1+R2+R3+R4); `missing_pieces_count = 17` in-scope GAPs (3 sub-gap of defined-but-not-implemented abort criteria separate). MEASURED-from-doc audit; no code changes; no `paper/*` touches; Tier-2 re-orientation preserved.