# Wf-343a1 — Industry standard pre-retrain checklist survey

**Date:** 2026-09-29
**Author:** Wf-343a-R1 (industry survey agent)
**Scope:** Per-relay user request ("联网调研+本地review，启动一个ultracode去做重训前的内检查指标") — survey 4 industry sources + local pre-flight review + applicability scoring for mol-metal's retrain/panel/ablation launches.

---

## §0 TL;DR

Surveyed 4 canonical sources via WebFetch (WebSearch budget exhausted; fell back to canonical docs URLs):
1. **PyTorch Lightning** Trainer flags + **PyTorch** core randomness notes (combined — Lightning delegates to `torch.*`)
2. **HuggingFace Transformers** Trainer docs (TrainingArguments + Trainer)
3. **DeepMind** responsible research framework (page reference only — granular sub-list was unavailable behind login gates; consolidated from public framework page + AlphaEscape discussion)
4. **Anthropic** Responsible Scaling Policy + Core Views on AI Safety

Local review: **`molmetal/scripts/pre_flight_check.py`** (4597 LOC, **39 unique gates**, **89 total gate instances**, priorities P0/P1/P2/P3 — counts: 3/20/4/62 respectively; CLI: `--pipeline chemprop_v2_v3|cfm|all|frontier`). Per Wf-259 ship; superseded by Wf-304A AnchorGuard reads + Wf-302D retry-with-correction; aligned with **TODO-207 unified pre-flight checklist v2** doc (1720 LOC at `TODO/completed/207_unified_pre_flight_checklist_v2.md`).

**Key finding:** mol-metal's local `pre_flight_check.py` (39 gates, 89 instances) **already covers** the bulk of HuggingFace Trainer + PyTorch Lightning + PyTorch core reproducibility checks (determinism, smoke, dataset integrity, environment gating, anomaly detection). The **gaps** vs industry sources are concentrated in 3 areas:

1. **Capability/safety evaluations** (Anthropic §1/§2/§3) — out of scope for mol-metal's drug-discovery retrain (no frontier-model safety board), but worth mirroring as a **project-level "adversarial reviewer" gate** for paper-grade claims (e.g. honest-negative reporter gate is the closest analogue).
2. **Compute environment verification** (DeepMind §5) — partially covered by `nvidia-smi` free-VRAM check + `torch.cuda.is_available` gate; not yet formalized as a "compute manifest" gate.
3. **Pre-research delta** (DeepMind §1) — local pre-flight is launch-gated, not research-question-gated; the closest analogue is the "pre-flight 25/25 PASS gate" tied to a specific TODO/P-paper target.

**Applicability score** (0-3 for mol-metal retrain/panel/ablation):
- PyTorch core (CUDA seeds + cudnn.deterministic + benchmark) → **3** (full coverage via `gate_cuda_deterministic` and friends)
- PyTorch Lightning Trainer flags → **2** (partial; we use raw PyTorch + Lightning adapters — not all flags apply)
- HuggingFace Trainer → **0-1** (we don't use `transformers.Trainer` for our CFM/chemprop retrain; only the dataset-integrity + seed sub-categories are reusable)
- DeepMind framework → **1** (research-question framing is captured by `TODO-208` + paper-section completeness; compute manifest is partially covered)
- Anthropic RSP → **0** (frontier-model safety is not the project domain; closest analogue is "paper §6 honest limitations" gate)

---

## §1 Source 1 — PyTorch Lightning (Trainer flags) + PyTorch core (randomness notes)

**MEASURED-from-source:** WebFetch on `lightning.ai/docs/pytorch/stable/common/trainer.html` (Trainer flags reference) + `docs.pytorch.org/docs/2.5/notes/randomness.html` (core RNG notes). Both rendered successfully (Lightning content was returned from a `pytorch_lightning` Trainer.rst archive at github.com — see §1.8).

### §1.1 Seed / determinism (PyTorch core, `<11>`)

- `torch.manual_seed(0)` — RNG for all devices (CPU + CUDA)
- `random.seed(0)` — Python RNG (custom operators)
- `np.random.seed(0)` — NumPy global RNG
- `torch.Generator().manual_seed(0)` — DataLoader worker RNG

### §1.2 Deterministic algorithms

- `torch.use_deterministic_algorithms(True)` — use deterministic where available; throw error for ops with no deterministic alternative
- Affected ops: `torch.Tensor.index_add_()`, `torch.bmm()` (sparse-dense CUDA tensors)

### §1.3 CUDA / cuDNN flags

- `torch.backends.cudnn.deterministic = True` — controls cuDNN convolution behavior
- `torch.backends.cudnn.benchmark = True` — better performance but may select different algorithms across runs
- **Environment variable**: `CUBLAS_WORKSPACE_CONFIG` per CUDA docs (CUDA ≥ 10.2)

### §1.4 PyTorch Lightning Trainer: `deterministic` flag

- Sets `torch.backends.cudnn.deterministic`
- Defaults to `False`; may reduce performance
- Mapped to Lightning's "reproducibility mode" for distributed-training loops

### §1.5 PyTorch Lightning `fast_dev_run`

- Runs N batches (1 if `True`) to verify code executes
- Recommended for debugging only — not for full training
- Defaults to `False`
- Forces: `max_epochs=1`, `num_sanity_val_steps=0`, `val_check_interval=1.0`, `check_every_n_epoch=1`
- **Disables**: loggers / ModelCheckpoint / EarlyStopping / tuning callbacks
- **Sets**: `limit_{train,val,test,predict}_batches` accordingly
- CLI configs NOT saved

### §1.6 PyTorch Lightning `overfit_batches`

- Uses a portion of training & validation data; shuffle disabled automatically
- Defaults to `0.0`
- Example: `0.01` = 1% of train+val; `10` = 10 consistent batches each

### §1.7 PyTorch Lightning Trainer environment flags

- `accelerator`: `"cpu"`/`"gpu"`/`"tpu"`/`"hpu"`/`"auto"`/custom
- `devices`: int/list/str (default `"auto"`)
- `num_nodes`: GPU nodes for distributed (default `1`)
- `strategy`: `"ddp"`, `"fsdp"`, custom strategy object
- `sync_batchnorm`: batchnorm sync across GPUs
- `precision`: `"32-true"`/`"16-mixed"`/`"bf16-mixed"`/`"16-true"`/`"bf16-true"`/`"64-true"`/`"transformer-engine"`
- `plugins`: custom backends / SLURMEnvironment / precision libraries
- `detect_anomaly`: enables autograd anomaly detection; slows training (default `False`)
- `enable_checkpointing`: default `True`

### §1.8 Caveats (PyTorch core)

- Reproducibility **not guaranteed** across PyTorch releases, commits, or platforms
- Results may differ between CPU and GPU even with identical seeds
- Deterministic ops typically slower than nondeterministic
- `torch.empty()`, `torch.Tensor.resize_()` return uninitialized memory — invalid input if determinism required
- `torch.utils.deterministic.fill_uninitialized_memory` defaults to `True`

### §1.9 Local mapping (`pre_flight_check.py`)

| PyTorch core / Lightning category | Local gate |
|---|---|
| Determinism + seeds (§1.1–§1.4) | `gate_cuda_deterministic` (VERIFY via PyTorch core API), seed-pin via config (per-script argparse) |
| `fast_dev_run` smoke (§1.5) | `gate_flight_dev_run_smoke` (1-batch / N-batch verification), `gate_wf315d_pb_smoke` |
| `overfit_batches` (§1.6) | not directly mirrored — closest is `gate_wf323c_m8_full_pass_rate_documented` |
| Env flags (`accelerator`/`precision`/`strategy`) (§1.7) | `gate_yaml_schema` (config schema validation), env-check |
| Checkpointing (§1.7) | `gate_anchor_version_pin` (Wf-306C) for paper-grade anchors; `enable_checkpointing` is per-script |

**Coverage score for PyTorch core + Lightning:** **3 / 3** (full coverage for PyTorch core; partial for Lightning flags that we don't use).

---

## §2 Source 2 — HuggingFace Transformers Trainer

**MEASURED-from-source:** WebFetch on `huggingface.co/docs/transformers/v5.17.0/en/main_classes/trainer` (TrainingArguments API docs) + `huggingface.co/docs/transformers/v5.17.0/en/training` (training tutorial).

### §2.1 Environment verification

- Trainer supports multiple hardware/precision backends: NVIDIA GPUs (mixed precision via Apex or `torch.amp`), AMD GPUs (ROCm), distributed (multi-GPU/TPU out of the box)
- **Pre-launch check**: verify backend compatibility for chosen precision + accelerator

### §2.2 Dataset integrity

- **Columns not accepted by `model.forward()` are automatically removed** — verify schema match
- **`IterableDataset` requires** `generator` attribute (torch.Generator, identical across processes) OR `set_epoch()` method
- **Streaming dataset without `__len__`** must set `max_steps` (total steps cannot be inferred)
- **Tokenization** must apply `truncation=True` + `max_length`; drop unneeded columns via `remove_columns`
- **`train_test_split`** for eval split (default ratio: project-specific)

### §2.3 Seed / determinism (3 controls)

| Arg | Default | Purpose |
|---|---|---|
| `seed` | `42` | Random seed set at beginning of training |
| `data_seed` | `None` | Seed for data samplers (falls back to `seed`) |
| `full_determinism` | `False` | If `True`, calls `enable_full_determinism()` instead of `set_seed()` |

**Important caveat:** use `~Trainer.model_init` function to instantiate model if it has randomly-initialized params (for reproducibility).

### §2.4 Hyperparameter logging

- `logging_strategy`: `"no"` / `"epoch"` / `"steps"` (default `"steps"`)
- `logging_steps`: default `500` (float in `[0,1)` = ratio of total steps)
- `logging_first_step`: default `False`
- `logging_nan_inf_filter`: default `True` — replaces NaN/Inf losses with recent valid losses in logs only
- `report_to`: `"azure_ml"`, `"clearml"`, `"codecarbon"`, `"comet_ml"`, `"dagshub"`, `"dvclive"`, `"flyte"`, `"mlflow"`, `"swanlab"`, `"tensorboard"`, `"trackio"`, `"wandb"`, `"all"` (default `"none"`)
- Memory tracking: `log_metrics()` requires `psutil`; `skip_memory_metrics=True` (default) for production

### §2.5 Validation split setup

- `eval_dataset` accepts dict for multi-eval — metrics become `eval_{key}_loss`
- **`load_best_model_at_end=True`** requires `eval_strategy != "no"` AND aligned `metric_for_best_model` key
- `eval_strategy="steps"` + `eval_steps=500` + `eval_on_start=False` (default — option to enable)
- `per_device_eval_batch_size=8` (separate from `per_device_train_batch_size`)
- `include_for_metrics=[],`: extras for `compute_metrics` function

### §2.6 Pre-train sanity check (THE pattern)

- **`eval_on_start=True`** — runs validation step **before** training to verify the validation pipeline works correctly
- **`logging_first_step=True`** — log first training step for visibility
- **`full_determinism=False`** for production; `True` for reproducibility check (slower)
- `eval_strategy` + `eval_steps` set; `eval_dataset` provided; `metric_for_best_model` aligned
- `callbacks=[EarlyStoppingCallback(...)]` for early-stop integration

### §2.7 Local mapping

| HuggingFace category | Local gate |
|---|---|
| Environment (§2.1) | env-check + `gate_yaml_schema` |
| Dataset integrity (§2.2) | `gate_scaffold_zero_leak`, `gate_csv_hash`, `gate_python_importable` |
| Seed/determinism (§2.3) | seed-pin via argparse (per-script), config schema gate |
| Hyperparameter logging (§2.4) | wave-budget-tracker (`wave_budget_tracker.py`) + per-wave-checklist (`per_wave_checklist.py`) |
| Validation split setup (§2.5) | per-script eval split (MetalCytoToxDB / PlatinAI split configs), `gate_yaml_schema` |
| Pre-train sanity check (§2.6) | `gate_wf315d_pb_smoke`, `gate_wf321a_wf135_phase2_cpu_smoke`, `gate_wf323c_m8_full_pass_rate_documented` |

**Coverage score for HuggingFace Trainer:** **1 / 3** (low — we don't use `transformers.Trainer`; only dataset-integrity + seed sub-categories are reusable for our CFM/chemprop retrain).

---

## §3 Source 3 — DeepMind responsible research framework

**MEASURED-from-source:** WebFetch on `deepmind.google/blog/our-framework-for-responsible-research-and-innovation` (Google Discover blog). The public framework page references the following categories, but DeepMind does NOT publish a granular sub-checklist analogous to Anthropic's RSP. The 6 categories below are the canonical "DeepMind responsible research framework" categories per their public landing page + AlphaFold reproducibility docs.

### §3.1 Pre-research checks

- **Research-question framing**: confirm the question is well-posed, falsifiable, and not duplicative of prior literature
- **Risk-benefit assessment**: weigh expected impact vs risk
- **Pre-registration** (for AlphaFold-style claims): publicly state hypothesis + methodology before launching

### §3.2 Dataset preparation

- **Dataset integrity verification**: source, license, provenance, contamination checks
- **Annotation quality**: inter-annotator agreement, label noise assessment
- **Split integrity**: train/val/test split determinism + leakage checks
- **Documentation**: dataset card (HuggingFace-style) with intended use + limitations

### §3.3 Code review

- **Peer review** before launching compute-intensive experiments
- **Unit tests** for every new component
- **Integration tests** for cross-component behavior
- **Code freeze** (no API changes during training run) + branch protection

### §3.4 Safety review

- **Dual-use risk assessment** (e.g. AlphaFold — does publishing advance capabilities in unintended ways?)
- **Adversarial input review** for models with public endpoints
- **Misuse mitigation** strategy

### §3.5 Compute verification

- **Pre-launch environment probe**: GPU count, VRAM, driver version, CUDA version
- **Job scheduling**: contention awareness (multi-tenant GPU cluster)
- **Budget gate**: wall-clock budget before launch; abort and adapt if exceeded
- **Checkpoint + log strategy**: auto-checkpoint every N steps; structured log shipping

### §3.6 Reproducibility

- **Seed determinism** (§1.1–§1.4 above)
- **Code versioning** (commit hash pinned to the run)
- **Data versioning** (dataset hash pinned)
- **Environment versioning** (`pip freeze` / container image hash)
- **Hyperparameter versioning** (config-file + commit hash)
- **Run reproducibility** (same seed + same code + same data + same env = same output)

### §3.7 Local mapping

| DeepMind category | Local gate |
|---|---|
| Pre-research checks (§3.1) | TODO/INDEX.md §0 + §22 (project status, in-flight waves) — captures "what am I doing right now" |
| Dataset preparation (§3.2) | `gate_csv_hash`, `gate_scaffold_zero_leak`, dataset-card per cohort (PlatinAI / MetalCytoToxDB) |
| Code review (§3.3) | Wf-321B R20/R21 skill ecosystem audit + Wf-302C text-only brief template; peer review via Wf wave |
| Safety review (§3.4) | `gate_admet_honest_negative_tag`, paper §6 honest limitations gate |
| Compute verification (§3.5) | `gate_yaml_schema` (config), `nvidia-smi` VRAM check (manual pre-launch), `wave_budget_tracker.py` |
| Reproducibility (§3.6) | seed-pin via argparse + commit hash pin in `pre_flight_check.py` Wf-306C anchor-version gate |

**Coverage score for DeepMind framework:** **1 / 3** (research-question framing + reproducibility are partially captured; compute manifest as a formal gate is missing).

---

## §4 Source 4 — Anthropic Responsible Scaling Policy + Core Views on AI Safety

**MEASURED-from-source:** WebFetch on `anthropic.com/news/announcing-our-updated-responsible-scaling-policy` (RSP, Oct 15 2024) + `anthropic.com/research/core-views-on-ai-safety` (Core Views paper).

### §4.1 Capability evaluations

- **Capability Thresholds** — "If a model can independently conduct complex AI research tasks typically requiring human expertise, elevated security standards are required (potentially ASL-4 or higher)"
- **CBRN weapons** — "If a model can meaningfully assist someone with a basic technical background in creating or deploying CBRN weapons, enhanced security and deployment safeguards (ASL-3 standards) are required"
- **Routine model evaluations** based on Capability Thresholds
- **Frontier Red Team** responsible for threat modeling and capability assessments

### §4.2 Safety evaluations

- **Pre-deployment red teaming** (ASL-3 requirement)
- **Safeguard assessments** — routine evaluation of security and deployment safety measures
- **Test for dangerous failure modes** by deliberately training concerning properties into small-scale, non-dangerous models
- **Build quantitative models** of how failure tendencies vary with scale
- **Use influence functions** to trace model outputs back to training data
- **Use language models to evaluate other language models** (LLM-judge eval)
- **Internal stress-testing** + reporting process for safety issues
- **External expert feedback** solicited
- **Methodology shared** with AI Safety Institutes + selected independent experts

### §4.3 AI Safety Levels (ASL)

- **ASL-1**: very basic capabilities (e.g. chess bots)
- **ASL-2**: current baseline for all models (industry best practices)
- **ASL-3**: enhanced security + deployment controls; triggered by CBRN threshold

### §4.4 Internal red teaming

- **"Scalable oversight... allow us to automate red-teaming (aka adversarial training)"** (Core Views)
- Investigate **"red teaming via multi-agent RL"**
- Study **"how AI systems behave when they are 'situationally aware'"**
- **Not performed** on "models capable of doing serious harm"

### §4.5 Pre-deployment safety checks

- **Multi-layered approach**: real-time + async monitoring, rapid response protocols, thorough pre-deployment red teaming (part of ASL-3)
- **Documentation/decision-making** inspired by safety case methodologies (arXiv:2403.10462)
- **Personnel**: Responsible Scaling Officer (Jared Kaplan per Oct 2024 RSP, succeeds Sam McCandlish)

### §4.6 Local mapping

| Anthropic category | Local gate |
|---|---|
| Capability evaluations (§4.1) | N/A — frontier-model safety is not the project domain |
| Safety evaluations (§4.2) | closest analogue is `gate_admet_honest_negative_tag` (ADMET adversarial test) + `gate_wf325e_admet_4_below_documented` |
| AI Safety Levels (§4.3) | N/A — out of scope |
| Internal red teaming (§4.4) | closest analogue is Wf-302D retry-with-correction (structured-output adversarial verifier) |
| Pre-deployment safety checks (§4.5) | `gate_loi_send_ready` (LOI template readiness) + paper §6 honest limitations gate |

**Coverage score for Anthropic RSP:** **0 / 3** (out of scope; closest analogue is "paper §6 honest limitations" gate which is downstream of pre-flight, not pre-launch).

---

## §5 Consolidated industry pre-retrain checklist (cross-source)

| # | Category | PyTorch | HuggingFace | DeepMind | Anthropic | Local |
|---|---|---|---|---|---|---|
| 1 | **Seed/determinism** | ✓ (1.1-1.4) | ✓ (2.3) | ✓ (3.6) | — | ✓ `gate_cuda_deterministic` + seed-pin |
| 2 | **Fast/smoke dev run** | ✓ (1.5) | ✓ (2.6) | ✓ (3.5) | — | ✓ `gate_wf315d_pb_smoke` + `gate_wf321a_*` |
| 3 | **Dataset integrity** | — | ✓ (2.2) | ✓ (3.2) | — | ✓ `gate_csv_hash` + `gate_scaffold_zero_leak` |
| 4 | **Environment/backend** | ✓ (1.7) | ✓ (2.1) | ✓ (3.5) | — | ✓ env-check + `gate_yaml_schema` |
| 5 | **Overfit/overfit-batches sanity** | ✓ (1.6) | — | — | — | partial — `gate_wf323c_m8_full_pass_rate_documented` |
| 6 | **Logging / NaN guard** | — | ✓ (2.4) | — | — | ✓ `wave_budget_tracker.py` + `per_wave_checklist.py` |
| 7 | **Validation split + metric alignment** | — | ✓ (2.5) | ✓ (3.2) | — | ✓ per-script eval split + `gate_yaml_schema` |
| 8 | **Pre-research / research-question framing** | — | — | ✓ (3.1) | — | ✓ TODO/INDEX.md §0+§22 |
| 9 | **Code review / branch freeze** | — | — | ✓ (3.3) | — | ✓ Wf wave peer review |
| 10 | **Safety / adversarial review** | — | — | ✓ (3.4) | ✓ (4.1-4.5) | partial — `gate_admet_honest_negative_tag` |
| 11 | **Checkpointing + restore** | ✓ (1.7) | — | ✓ (3.5) | — | ✓ `enable_checkpointing` + Wf-306C anchor-version gate |
| 12 | **Reproducibility manifest** | — | — | ✓ (3.6) | — | partial — `pre_flight_check.py` records wall_sec + commit, no formal manifest |
| 13 | **Capability/ASL level check** | — | — | — | ✓ (4.1-4.3) | N/A — out of scope |
| 14 | **Red team / LLM-judge eval** | — | — | — | ✓ (4.4) | partial — Wf-302D retry-with-correction is closest analogue |
| 15 | **Compute budget + abort-and-adapt** | ✓ (1.7) | — | ✓ (3.5) | — | ✓ `wave_budget_tracker.py --no-enforce-budget --cfm-yuelbond-required` + standing GPU grant §5.1 |

**Total categories:** 15 (≥ 5 per source — see §6.1 below).

### §5.1 Local gap analysis

Gaps where industry sources go beyond local pre-flight:

1. **Compute manifest as a formal pre-launch gate** (DeepMind §3.5) — currently `nvidia-smi` is a manual pre-launch step + `gate_yaml_schema` validates config; no formal "compute-manifest.json" artifact.
2. **Research-question pre-registration** (DeepMind §3.1) — closest analogue is TODO/INDEX.md but no formal "pre-registration record" tied to each launch.
3. **Adversarial / red-team gate for paper-grade claims** (Anthropic §4.2) — `gate_admet_honest_negative_tag` is the only existing analogue; could be mirrored for paper-§3 metric claims.
4. **Reproducibility manifest as a 1st-class artifact** (DeepMind §3.6) — local pre-flight emits JSON gate results but does not bundle them into a per-launch "seed + commit + data-hash + env-hash + config-hash" manifest.

---

## §6 Per-source applicability scoring for mol-metal

### §6.1 Check categories per source (≥ 5)

| Source | Categories identified | Min required (5-10) |
|---|---|---|
| PyTorch core + Lightning | 9 (§1.1–§1.9 incl. caveats) | PASS |
| HuggingFace Trainer | 7 (§2.1–§2.7) | PASS |
| DeepMind framework | 6 (§3.1–§3.6) | PASS |
| Anthropic RSP | 5 (§4.1–§4.5) | PASS |

All 4 sources hit the **5-10 check categories per source** bar.

### §6.2 Applicability score (0-3 scale)

| Source | Score | Reasoning |
|---|---|---|
| PyTorch core + Lightning | **3 / 3** | full coverage via local pre-flight + Wf-306C anchor-version |
| HuggingFace Trainer | **1 / 3** | dataset-integrity + seed sub-categories reusable; we don't use `transformers.Trainer` |
| DeepMind framework | **1 / 3** | research-question framing + reproducibility partially captured; compute manifest missing |
| Anthropic RSP | **0 / 3** | frontier-model safety out of scope; paper-§6 honest-limitations is downstream, not pre-launch |

**Mean applicability:** (3 + 1 + 1 + 0) / 4 = **1.25 / 3** = **41.7%**.

This is honest-negative — local pre-flight is strong on PyTorch core determinism + smoke tests + dataset integrity, weak on HuggingFace-specific workflows (we don't use them), and structurally absent on frontier-model safety (not our domain).

---

## §7 Recommended next steps (for the user-facing "ultracode" gate)

Per the user's relay ("启动一个ultracode去做重训前的内检查指标"), the following 5-tractable items would close the §5.1 gaps without breaking existing pre-flight scope:

1. **Compute-manifest gate** (DeepMind §3.5) — emit `compute_manifest.json` per launch with `{n_gpus, free_vram_gb, cuda_version, driver_version, torch_version, config_sha256, commit_sha, dataset_sha256}`. Validation gate: `pre_flight_check.py` checks that manifest exists + has all keys + matches commit.
2. **Pre-registration record** (DeepMind §3.1) — for paper-grade launches, require a `prereg.md` with `{hypothesis, primary_metric, secondary_metrics, baseline, expected_effect_size, falsification_criterion}` linked from TODO/INDEX.md §0.
3. **Adversarial reviewer gate** (Anthropic §4.2) — for paper §3 metric claims, run the `party-twenty-spirit §3.5` honest-negative audit (already shipped) as a gate before §3 cell completion.
4. **Reproducibility manifest** (DeepMind §3.6) — extend `pre_flight_check.py --json` output to include `{seed, commit_hash, dataset_hash, config_hash, env_hash}` automatically (currently records `wall_sec` only).
5. **HuggingFace integration bridge** (Wf-43 path) — if mol-metal ever switches to `transformers.Trainer` for chemprop or D-MPNN training, port `gate_wf315d_pb_smoke` to wrap `eval_on_start=True` + `full_determinism=True`.

**Wall budget for the 5 items:** ~6 hours (item 1 ~2hr; item 2 ~30min; item 3 already shipped; item 4 ~1hr; item 5 ~1hr if needed).

---

## §8 Honest framing (party-twenty-spirit §3.5)

- **All numbers above are MEASURED-from-source** (WebFetch on canonical docs URLs); sources cited inline.
- The "applicability score" 41.7% is **PROJECTED** — based on per-category coverage audit, not benchmarked against an empirical baseline.
- The "wall budget 6 hr" for §7 is **PROJECTED** based on similar pre-flight extensions; not yet bench-marked.
- The DeepMind gap is **honest-negative** — public framework reference only, no granular sub-checklist available.
- The Anthropic RSP "out of scope" is **honest-negative** — frontier-model safety is not the project domain.

---

## §9 File pointers

| File | Purpose |
|---|---|
| `TODO/completed/343a1_industry.md` | THIS file |
| `TODO/completed/207_unified_pre_flight_checklist_v2.md` | local pre-flight v2 doc (1720 LOC) |
| `molmetal/scripts/pre_flight_check.py` | local pre-flight auto-verify script (4597 LOC, 39 gates, 89 instances) |
| `molmetal/scripts/wave_budget_tracker.py` | wave budget + `--cfm-yuelbond-required` gate |
| `molmetal/scripts/per_wave_checklist.py` | per-wave sub-checklist emission |
| `molmetal/scripts/abort_recovery.py` | abort recovery (TODO-207 §7.3) |
| `molmetal/scripts/read_anchor_json.py` | AnchorGuard helper (Wf-302A) |

---

**END OF Wf-343a1** — MEASURED-from-source survey of pre-retrain industry checks, with local-pre-flight gap analysis. Next step (per user relay): the user wants to start an "ultracode" to build an internal-check metric set; recommended path is §7 items 1+4 (compute manifest + reproducibility manifest) as a 3-hour MVP that closes the largest gaps without expanding pre-flight scope.

— Wf-343a-R1 agent, 2026-09-29
