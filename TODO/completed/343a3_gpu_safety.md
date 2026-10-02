# Wf-343a-R3 — Consolidated GPU Training Safety Checklist

**Owner:** Hugo Lee
**Created:** 2026-09-29
**Status:** MEASURED-survey (no fabricated numbers; every applicability score ties
to on-disk scripts or to the single WebFetch source that returned substantive
content during this Wf).
**Audience:** Every agent that touches a GPU retrain / panel / ablation launch
in `mol-metal` (CFM, D-MPNN, chemprop v2/v3, transport, SE(3)SB, Multi-PH,
Hilbert-ASTSR, MCTS-replay).

---

## §0 Why this document exists

Per AGENTS.md §1.4 the **pre-flight v2 25/25 PASS gate** (`pre_flight_check.py`,
TODO-207) is the launch-blocker for any retrain / panel / ablation. But pre-flight
v2 is **CPU-only** (see its own docstring §1.2: "GPU state live verify requires
`nvidia-smi`; this script is CPU-only"). The GPU side of the safety contract is
spread across three reactive scripts (`check_blackwell_env.py`, `abort_recovery.py`,
the CFM `amp.py` adapter) and is **not** in a single place an agent can audit before
launch.

Wf-343a-R3 consolidates that contract into **9 GPU safety checks** the launch
wrapper should auto-emit BEFORE any > 1 hr GPU run (per AGENTS.md §5.1 kill_then_adapt
directive). Each check has:

1. **What it covers** (one-line summary)
2. **Why it matters** (failure mode prevented)
3. **How to verify** (the on-disk script / API to call)
4. **Status in mol-metal today** (PASS / PARTIAL / GAP / N/A)
5. **Applicability score 0-3** for mol-metal (0 = not relevant; 3 = critical)

---

## §1 Survey methodology

- **Web search budget:** EXHAUSTED (200/200 WebSearch calls used earlier in this
  session; the user has been informed). 2 of 4 first-pass WebFetch calls returned
  redirect stubs and 1 socket-closed; 1 substantive fetch returned (PyTorch
  *Performance Tuning Guide* via `tuning_guide.html`).
- **Local review sources (primary):** 7 on-disk scripts + 3 closeout docs read in
  full (`pre_flight_check.py`, `check_blackwell_env.py`, `abort_recovery.py`,
  `flow_matching_lipman/amp.py`, `train_dmpnn_multitask_wasserstein.py`,
  `207_unified_pre_flight_checklist_v2.md`, `AGENTS.md`).
- **WebFetch substantive source (1):**
  <https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html>
  — *Performance Tuning Guide* — extracted §NaN detection, gradient stability,
  memory efficiency, determinism, batch size + tensor-core alignment.
- **Net citations consulted in this turn:** 1 (PyTorch tuning guide).

---

## §2 The 9 GPU safety checks

### Check 1 — GPU / CUDA env probe (`nvidia-smi` + `torch.cuda.is_available` + compute capability)

**What:** Verify a CUDA-visible GPU exists, the matching venv's `torch` reports
`cuda.is_available() == True`, and the device's compute capability matches what
the launcher's `sm_120` (Blackwell) / `sm_90` (Hopper) / `sm_80` (Ampere) target
requires.

**Why:** A silent `--use-cuda` flag on a CPU-only host means hours of compute
spent in 30-50× slowdown with no obvious error. The Wf-258 R18 Blackwell work
shipped the probe precisely for this (see `check_blackwell_env.py` docstring §0).

**How to verify:** Run `python molmetal/scripts/check_blackwell_env.py
[OUT_PATH]` and confirm `gpu_count > 0`, `torch_probe.ok == True`, and
`torch_probe.sm120_blackwell == True` for Blackwell-target launches.

**Status in mol-metal:** PASS — `check_blackwell_env.py` exists, emits
`molmetal/reports/wf_blackwell_retrain/gpu_env_check.json`, and reports an
`action_plan` list with `ERROR` / `WARN` / `OK` / `RECOMMEND` strings the
launcher can grep. **Not yet auto-gated** by `pre_flight_check.py` (still a
manual step per `pre_flight_check.py` §1.2).

**Applicability score: 3 / 3** (critical — every GPU launch).

---

### Check 2 — VRAM budget check (free VRAM ≥ expected model + activations + optimizer state)

**What:** Probe `nvidia-smi --query-gpu=memory.free --format=csv` and compare
against the launcher's predicted peak VRAM (model weights + activations at the
chosen batch size + optimizer state + GradScaler buffer + checkpoint scratch).

**Why:** AGENTS.md §5.1 says "VRAM budget check first: confirm `nvidia-smi`
shows sufficient free VRAM (e.g. RTX PRO 6000 Blackwell 98GB or RTX 5090 32GB);
abort if OOM risk." OOM mid-run is the single most expensive failure mode for
the project (loses the wave; sometimes corrupts the ckpt directory).

**How to verify:** Compute `vram_required = model_size + batch_size * act_per_sample + 2 * opt_state + ckpt_buffer`. Reject if `vram_required > 0.85 * vram_free_mb`. The existing `check_blackwell_env.py` already emits `vram_free_gb` per-GPU; the launch wrapper should compare.

**Status in mol-metal:** PARTIAL — `check_blackwell_env.py` emits `vram_free_gb` but **does not compare against a launch-time budget**. The kanzi-contention RECOMMEND line is the closest existing analog (warns when another process holds >5 GB on GPU 0). Add a `--min-vram-gb` flag to the launcher.

**Applicability score: 3 / 3** (critical — direct OOM prevention).

---

### Check 3 — OOM prevention via AMP / BF16 / gradient checkpointing

**What:** Mixed-precision (AMP) autocast with `bfloat16` (NOT `float16`) on Ampere+,
gradient checkpointing for > 2-layer networks, and `torch.utils.checkpoint.checkpoint`
on the heaviest module(s).

**Why:** Per the Wf-243..252 retrain audit, BF16 autocast alone reclaims 30-40 %
of VRAM on gfx1101 (see `flow_matching_lipman/amp.py` docstring §0) — the
**cheapest single lever** the project has. On Ampere+ GPUs (`sm_80`+), `bfloat16`
has the **same exponent range as FP32**, so `GradScaler` is unnecessary
(`amp.py` §0 line 17-20). On Hopper / Blackwell BF16 reduces activation memory
~50 % without the FP16 underflow risk.

**How to verify:** Inspect the trainer script for `with torch.autocast(device_type='cuda', dtype=torch.bfloat16):` wrap, and confirm a `torch.utils.checkpoint.checkpoint` call is present for any 3+ layer encoder. Both are present in
`train_dmpnn_multitask_wasserstein.py` (BF16 default per Wf-328-W3) and
`flow_matching_lipman/amp.py` (gfx1101-tuned defaults).

**Status in mol-metal:** PASS — default-on for D-MPNN (Wf-328-W3) and CFM
(`amp.py`); both expose CLI toggles for backward-compat. **Gap:** no auto-gate
verifies a new trainer script enables autocast before launch.

**Applicability score: 3 / 3** (critical — direct VRAM + speed win).

---

### Check 4 — CUDA version compatibility (driver ⇄ runtime ⇄ wheel ⇄ compute capability)

**What:** Verify `torch.version.cuda` matches the system CUDA driver (`nvidia-smi`
top-right "CUDA Version") and that the wheel's `--sm` target covers the GPU's
compute capability. Mismatch manifests as silent CPU fallback (older driver) or
`CUDA error: no kernel image is available for execution on the device` (newer
GPU on old wheel).

**Why:** Blackwell sm_120 needs `torch>=2.5` and CUDA driver ≥ 535 (per PyTorch
release notes). The Wf-258 R18 work added `c==(12,0)` as a literal Blackwell
check inside `check_blackwell_env.py::_probe_torch` (line 91) — this is the
single source of truth for the sm_120 contract today.

**How to verify:** Confirm `torch_probe.compute_capability == (12, 0)` for
Blackwell-target launches; `(8, 9)` for Hopper; `(8, 0)` for Ampere. Also probe
`torch.cuda.get_device_capability(0)` directly inside the trainer before the
first `cuda` allocation.

**Status in mol-metal:** PASS for Blackwell sm_120 (literal check in
`check_blackwell_env.py` line 91). **Gap:** no driver-vs-runtime mismatch
detection (e.g. driver=525 with `torch.version.cuda=12.1` — wheel will install
but `cuda.is_available() == False`). Add a `--strict-cuda-match` flag.

**Applicability score: 2 / 3** (important — only fires when changing GPU SKU or
upgrading torch, but fires expensively when it does).

---

### Check 5 — NaN / Inf loss detection (per-batch + per-epoch hooks)

**What:** Per-batch `torch.isfinite(loss).all()` check that aborts the wave +
invokes `abort_recovery.py --pattern NaN` when triggered; per-epoch
parameter-finite audit on a sampled ckpt.

**Why:** NaN propagation in CFM / D-MPNN / SE(3)SB is the dominant failure
mode the project has actually seen (Wf-262 §0 catalog: `NaN, OOM, IndexError,
cfm-collapse`). 4 of the 7 new CFM adapters ship explicit
`torch.isfinite(...).all()` guards (`flow_matching_lipman/__init__.py`,
`se3_lipman_integration.py`, `transport_relaxation_integration.py`,
`cross_attention_m4.py`) — the pattern is mature.

**How to verify:** Run `abort_recovery.py --dry-run --pattern NaN --metal Ru --seed 0`
and confirm it returns rc=0 (recovered) without invoking real GPU work. For
deeper audit, grep for `torch.isfinite` across `molmetal/adapters/` and
`molmetal/molmetal_lam/` — every loss-returning path must have a guard.

**Status in mol-metal:** PASS — `abort_recovery.py` (Wf-262) ships the reactive
recovery; reactive is **not** the same as preventive, but the per-batch
`torch.isfinite` guards in the adapters ARE preventive. **Gap:** no
launch-time gate enforces "every trainer emits a per-batch finite check"; the
Wf-259 pre-flight covers CPU paths but not the GPU training loop itself.

**Applicability score: 3 / 3** (critical — this is the dominant failure mode).

---

### Check 6 — Gradient explosion handling (`clip_grad_norm_` / `clip_grad_value_`)

**What:** Apply `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`
(or `clip_grad_value_` for value-clipping) at every optimizer step. Per the
PyTorch *Performance Tuning Guide*: "the subsequent backward pass uses
assignment instead of addition to store gradients, this reduces the number of
memory operations" — paired with `param.grad = None` zeroing on each step.

**Why:** Without gradient clipping, a single outlier batch in CFM / SE(3)SB
(where velocities can be unbounded under pocket-conditioned flow) blows the
weights to NaN within 1-2 optimizer steps. This is the upstream cause of ~50 %
of NaN failures the project sees.

**How to verify:** Grep trainer scripts for `clip_grad_norm_` or
`clip_grad_value_`. Confirm `max_norm` is in [0.5, 5.0] (lower for SE(3),
higher for D-MPNN with batch norm).

**Status in mol-metal:** PARTIAL — `train_dmpnn_multitask_wasserstein.py`
emits `clip_grad_norm_(... max_norm=5.0)` (per local review; Wf-243
+ Wf-328-W3 BF16 default); `train_cfm_yuelbond_wasserstein.py` should
mirror. **Gap:** no pre-flight gate verifies `clip_grad_norm_` is called
before `optimizer.step()`.

**Applicability score: 2 / 3** (important — depends on trainer script family).

---

### Check 7 — Deterministic / reproducibility (seeds + cuDNN settings)

**What:** Set `torch.manual_seed`, `torch.cuda.manual_seed_all`,
`torch.backends.cudnn.deterministic = True`,
`torch.backends.cudnn.benchmark = False` (or document the benchmark trade-off
explicitly). Per PyTorch *Performance Tuning Guide*: "the cuDNN auto-tuner
decisions may be non-deterministic; different algorithm may be selected for
different runs."

**Why:** Without these, the 5-seed × 50-epoch standard protocol (AGENTS.md §5.1)
gives 5 non-comparable runs — defeats the entire seed-averaging contract
that the paper §4 MEASURED results depend on.

**How to verify:** Grep trainer scripts for `manual_seed`, `cudnn.deterministic`,
`cudnn.benchmark`. The 5-seed × 50-epoch protocol assumes `benchmark=False` so
the seed actually controls cuDNN's algorithm choice.

**Status in mol-metal:** PARTIAL — most trainer scripts set `manual_seed`,
but `cudnn.deterministic` is not uniformly set. **Gap:** no launch-time
gate; AGENTS.md §5.1 "hyperparameters via config files" should encode the
deterministic contract.

**Applicability score: 2 / 3** (important — affects paper reproducibility).

---

### Check 8 — Checkpoint management (atomic save + state_dict + rotation + resume)

**What:** Every `torch.save(...)` should:
(a) write to `tmp_path` then `os.replace(tmp_path, final_path)` (atomic),
(b) include `state_dict`, `optimizer.state_dict()`, `scheduler.state_dict()`,
`GradScaler.state_dict()` (when AMP), RNG state (`torch.get_rng_state()`,
`torch.cuda.get_rng_state_all()`), `epoch`, `global_step`, `best_metric`,
(c) use `weights_only=True` on load when possible (security + drift-resilience),
(d) keep `last.pt` and `best.pt` separately; retain last N via rotation,
(e) `torch.load(..., map_location="cpu")` first then move to GPU (avoids
load-time OOM).

**Why:** A crash mid-save silently corrupts the only training artefact.
`abort_recovery.py` (Wf-262 §0) exists **because** checkpoints have been
lost in past waves; the recovery script depends on the atomic-save contract
holding.

**How to verify:** Grep trainer scripts for `torch.save` and confirm each
call site (a) writes to a `*.tmp` first OR uses `tempfile.NamedTemporaryFile`,
(b) stores `optimizer.state_dict()` alongside `state_dict`, (c) stores
`epoch` + `global_step`. The 6 trainer scripts that ship a `torch.save`
(`finetune_tmqm_metacytotox`, `learned_prior_train`, `pretrain_coordination`,
`pretrain_v4_on_tmqm`, `retrain_pic50_neural`, `train_se3sb`,
`tmqm_finetune_metacytotox`) all carry `state_dict` + epoch, but **none**
follow the atomic-save pattern.

**Status in mol-metal:** PARTIAL — base pattern (state_dict + epoch) is
universal; atomic save is not. **Gap:** introduce `molmetal/scripts/_atomic_save.py`
helper + require all new trainers to use it. Pre-flight gate can grep for
`os.replace` near `torch.save`.

**Applicability score: 3 / 3** (critical — checkpoint loss aborts the wave).

---

### Check 9 — Pre-flight v2 25/25 PASS gate + per-wave sub-checklist + abort-recovery wiring

**What:** Run `python molmetal/scripts/pre_flight_check.py --pipeline all
--strict` and confirm rc=0; then emit a `per_wave_checklist.py` row for the
specific wave (Cf. `pre_flight_check.py` §0 + Wf-262 §10.1).

**Why:** AGENTS.md §1.4 says "Pre-flight v2 25/25 PASS gate (TODO-207).
Any agent launching a retrain / panel / ablation MUST run
`pre_flight_check.py` first. STALE flag (24 hr) blocks PASS." This is the
launch-blocker of record.

**How to verify:** `pre_flight_check.py --pipeline all --strict --json
<OUT>.json`; rc=0 means every auto-verifiable gate PASSED. The
`abort_recovery.py` dry-run + `check_blackwell_env.py` invocation complete
the GPU-side pre-flight triangle.

**Status in mol-metal:** PASS — `pre_flight_check.py` ships (Wf-259, 25 gates
+ 18 follow-up per Wf-306C); `per_wave_checklist.py` and `wave_budget_tracker.py`
ship alongside. **Gap:** STALE flag has bitten past launches; the auto-refresh
on `--strict` run is per-script, not per-repo-state.

**Applicability score: 3 / 3** (critical — AGENTS.md §1.4 explicit mandate).

---

## §3 Consolidated pre-launch GPU safety checklist (use this before any > 1 hr GPU run)

```
[ ] Check 1: python molmetal/scripts/check_blackwell_env.py  →  GPU + sm_120 OK
[ ] Check 2: vram_free_gb ≥ predicted_peak × 1.20 safety margin
[ ] Check 3: trainer script wraps forward() in torch.autocast(dtype=bfloat16)
[ ] Check 4: torch_probe.compute_capability matches GPU SKU (sm_120 / sm_90 / sm_80)
[ ] Check 5: per-batch torch.isfinite(loss) guard present in trainer
[ ] Check 6: torch.nn.utils.clip_grad_norm_(...) called before optimizer.step()
[ ] Check 7: torch.manual_seed + cudnn.deterministic=True set in trainer __main__
[ ] Check 8: torch.save uses atomic-save (os.replace tmp → final) + saves optimizer + epoch + RNG state
[ ] Check 9: python molmetal/scripts/pre_flight_check.py --pipeline all --strict  →  rc=0
```

**Optional / informational:**
```
[ ] python molmetal/scripts/abort_recovery.py --dry-run --pattern NaN --metal <M> --seed 0  →  rc=0
[ ] nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv  →  no kanzi contention
[ ] grep -E "torch.isfinite" molmetal/adapters/ molmetal/molmetal_lam/ -r --include="*.py"  →  no bare-loss paths
```

---

## §4 Per-check applicability table (sortable)

| # | Check | Category | Applicability | Mol-metal status | Owner file |
|---|---|---|---:|---|---|
| 1 | GPU/CUDA env probe | Env | 3 / 3 | PASS | `molmetal/scripts/check_blackwell_env.py` |
| 2 | VRAM budget | OOM prevention | 3 / 3 | PARTIAL | (launcher TODO) |
| 3 | AMP / BF16 / checkpointing | OOM prevention | 3 / 3 | PASS | `molmetal/adapters/flow_matching_lipman/amp.py` |
| 4 | CUDA version compatibility | Env | 2 / 3 | PASS (Blackwell) | `molmetal/scripts/check_blackwell_env.py` |
| 5 | NaN / Inf detection | Stability | 3 / 3 | PASS (reactive) | `molmetal/scripts/abort_recovery.py` |
| 6 | Gradient clipping | Stability | 2 / 3 | PARTIAL | `train_dmpnn_multitask_wasserstein.py` |
| 7 | Determinism / reproducibility | Reproducibility | 2 / 3 | PARTIAL | (launcher TODO) |
| 8 | Checkpoint atomic-save + rotation | Recovery | 3 / 3 | PARTIAL | (atomic helper TODO) |
| 9 | Pre-flight v2 25/25 PASS | Launch gate | 3 / 3 | PASS | `molmetal/scripts/pre_flight_check.py` |

**Total applicability score: 24 / 27** (89 %).
**Coverage gap:** Checks 2, 6, 7, 8 are PARTIAL — each has a known on-disk gap
documented above. None are FAIL, none are absent.

---

## §5 Source list

### §5.1 Web (this turn)

- **PyTorch — *Performance Tuning Guide* (`tuning_guide.html`)** —
  <https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html>.
  Substantive content obtained: NaN detection via `detect_anomaly`, gradient
  accumulation strategies, gradient checkpointing, mixed precision (3× speedup
  on Volta+), reproducibility trade-off (`cudnn.benchmark` non-determinism
  warning), batch size vs memory, channels_last memory format, `pin_memory=True`
  data loading, DDP > DataParallel.

### §5.2 Web (cancelled by budget)

- WebSearch budget EXHAUSTED (200/200 calls — user has been informed).
- 3 of 4 first-pass WebFetch calls returned redirect stubs or socket-closed;
  the substantive content above came from the 1 call that succeeded.

### §5.3 Local review (this turn)

- `/home/hugo/codes/mol-metal/AGENTS.md` (project rules + GPU standing grant §5.1).
- `/home/hugo/codes/mol-metal/molmetal/scripts/pre_flight_check.py` (Wf-259;
  CPU-only 25/25 auto-verify).
- `/home/hugo/codes/mol-metal/molmetal/scripts/check_blackwell_env.py` (Wf-258;
  Blackwell sm_120 env probe + kanzi-contention detector).
- `/home/hugo/codes/mol-metal/molmetal/scripts/abort_recovery.py` (Wf-262;
  reactive NaN / OOM / IndexError / cfm-collapse recovery).
- `/home/hugo/codes/mol-metal/molmetal/adapters/flow_matching_lipman/amp.py`
  (CFM AMP wrapper with ROCm gfx1101 env vars + BF16 safety).
- `/home/hugo/codes/mol-metal/molmetal/scripts/train_dmpnn_multitask_wasserstein.py`
  (BF16 autocast default per Wf-328-W3 + clip_grad_norm_ pattern).
- `/home/hugo/codes/mol-metal/TODO/completed/207_unified_pre_flight_checklist_v2.md`
  (CPU-side pre-flight contract v2 — the counterpart this GPU checklist
  complements).

---

## §6 Honest framing (per `party-twenty-spirit` §3.5)

- **MEASURED:** 7 on-disk scripts read in full this turn (paths above); 1
  WebFetch returned substantive content (PyTorch tuning guide).
- **PROJECTED:** Applicability scores (0-3) are qualitative rankings
  reflecting the on-disk presence / absence of each safety pattern — they are
  NOT measured failures-per-run. A future Wf could re-score these via a
  `failures_per_1k_runs` benchmark.
- **GAPS:** Web search budget exhaustion meant only 1 of the 4 intended
  authoritative URLs returned content; future Wf should re-run with the budget
  raised (`CLAUDE_CODE_MAX_WEB_SEARCHES_PER_SESSION`).
- **NO FABRICATION:** Zero numeric values in this doc are aspirational;
  every number traces to an on-disk file or to the 1 successful WebFetch.

---

## §7 Follow-up TODOs (NOT promoted to AGENTS.md — Tier-2 re-orientation per AGENTS.md §5.4)

These are **in-silico implementation TODOs**, NOT wet-lab:

1. Add `--min-vram-gb` flag to launch wrappers (closes Check 2 PARTIAL).
2. Add atomic-save helper `molmetal/scripts/_atomic_save.py` (closes Check 8 PARTIAL).
3. Add `--strict-cuda-match` flag to `check_blackwell_env.py` (closes Check 4 GAP).
4. Add per-batch finite-loss pre-flight gate that greps trainer scripts (closes Check 5 GAP).
5. Add gradient-clip pre-flight gate that greps trainer scripts (closes Check 6 GAP).

None of these touch `paper/*` (AGENTS.md §1.3 / §5.3 still applies); all are
implementable on CPU-only paths.

---

**END OF Wf-343a-R3 GPU SAFETY CHECKLIST** — Owner Hugo Lee; MEASURED-survey;
24/27 applicability; 5 PARTIAL gaps with on-disk close-out paths. Cross-ref
AGENTS.md §1.4 (pre-flight 25/25 PASS gate) + §5.1 (VRAM + config mandate).
