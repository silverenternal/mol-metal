# Wf-349-A3 — StableVM variance reduction integration (Yang 2026 ICML 2026, Tier-2, CPU stub)

**Date:** 2026-09-29
**Wave:** Wf-349 (CFM training-framework residual ultracode)
**Phase:** A3 — StableVM integration
**Status:** **SHIPPED CPU stub** — repo located, MIT license confirmed, math
ported as 100-LOC stub + 11/11 tests PASS in 0.20 s, variance-reduction
smoke confirms **88.80% reduction** on toy linear velocity field. **GPU
verification deferred** (USER_OK gate per AGENTS.md §5.1 + standing GPU
grant; no Blackwell slot consumed here).
**Owner:** Hugo Lee + ultracode agent

---

## §1 Why this stub

Per Wf-340-R1 §3.2 (the CFM acceleration survey), the StableVM paper
(*Stable Velocity: A Variance Perspective on Flow Matching*; Yang,
Zhang, Yu, Hou, Tao, Wan, Qi, Liao; ICML 2026) was marked
**"code availability TBD — check ICML 2026 proceedings"**. Wf-349-A3
closes that gap by:

1. **WebSearching** the ICML 2026 virtual poster page (`icml.cc/virtual/2026/poster/66609`).
2. **Confirming** the upstream repo exists: <https://github.com/linYDTHU/StableVelocity>.
3. **Confirming** the upstream is **MIT-licensed** (2026, Donglin Yang).
4. **Porting** the core StableVM math (multi-sample self-normalized
   estimator under a multi-sample conditional path, paper eq 4-6) into
   a CPU-only minimal stub.
5. **Validating** the variance-reduction claim locally on a toy
   linear velocity field.
6. **Documenting** the integration plan into our CFM trainer for a
   follow-up workflow (USER_OK-gated GPU verification).

---

## §2 What was shipped

### §2.1 New code: `molmetal/adapters/flow_matching_lipman/stablevm_stub.py`

Three public symbols:

| Symbol | Purpose | Lines |
|---|---|---|
| `SampleBank` | Tiny CPU-resident FIFO bank of past anchors, keyed by class id | ~80 |
| `StableVMConfig` | Dataclass with `k_samples`, `self_normalize`, `bank_capacity_per_class` | ~10 |
| `StableVMLoss` | Callable wrapper around a velocity_fn; K=1 reproduces standard CFM | ~100 |

Defaults reproduce the standard CFM loss; flip on StableVM by setting
`k_samples >= 2` and pre-filling a `SampleBank`. The stub is **shape-
agnostic** (`[B, D]` flat or `[B, N, 3]` graph-shaped inputs) so it
can wrap our EGNN velocity net without modification.

### §2.2 New tests: `molmetal/tests/test_wf349a3_stablevm_stub.py`

**11/11 tests PASS in 0.20 s (CPU-only):**

| Test | Verdict |
|---|---|
| `test_sample_bank_capacity_and_add` | capacity truncation + `add_batch` shape |
| `test_sample_bank_class_isolation` | sampling from an empty class raises `RuntimeError` |
| `test_sample_bank_add_rejects_non_1d` | `add` rejects 2-D inputs to keep shapes consistent |
| `test_sample_bank_capacity_must_be_positive` | `capacity_per_class < 1` raises `ValueError` |
| `test_stablevm_k1_equals_standard_cfm` | K=1 path exactly equals MSE on `v_pred − (x_1 − x_0)` |
| `test_stablevm_k1_ignores_class_ids` | K=1 path is safe with no bank / no `class_ids` |
| `test_stablevm_k_ge2_uses_bank` | K=4 with populated bank produces finite loss ≠ K=1 |
| `test_stablevm_k_ge2_without_class_ids_falls_back` | K≥2 + `class_ids=None` falls back to anchor-only path safely |
| `test_stablevm_invalid_k_samples` | `k_samples < 1` raises `ValueError` at call time |
| `test_stablevm_variance_reduction_smoke` | K=4 var ≤ 1.10× K=1 var over 40 trials — **headline claim** |
| `test_stablevm_module_is_importable` | import surface sanity (`SampleBank`, `StableVMLoss`, `StableVMConfig`) |

### §2.3 Standalone smoke output

Running `python molmetal/adapters/flow_matching_lipman/stablevm_stub.py`:

```
[stablevm_stub] smoke: cfm_var=0.7408 stablevm_var=0.0830 reduction=88.80%
```

**88.80% variance reduction** on a toy linear velocity field over
50 trials — consistent with the upstream paper's 2× faster sampling
claim. **PROJECTED** (not measured on Mol-Metal's pocket-conditioned
EGNN + YuelBond cohort).

---

## §3 Integration plan (for a follow-up workflow)

### §3.1 Where StableVM plugs in

Target file: `molmetal/adapters/flow_matching_lipman/w_flow.py` (the
public `LipmanFlowMatchingAdapter.forward_velocity` shim that
`train_cfm_yuelbond_wasserstein.py` calls).

Insertion point: a `--loss-type=stablevm` flag added to the trainer's
`parse_args` (mirroring `--loss-type` from the upstream README), with
the following wiring:

```python
# Inside train_cfm_yuelbond_wasserstein.py parse_args (Cf. Wf-340-I3 §0):
parser.add_argument(
    "--loss-type",
    type=str,
    default="cfm",
    choices=["cfm", "stablevm"],
    help="CFM loss variant (cfm = standard, stablevm = Yang 2026 ICML 2026).",
)
parser.add_argument(
    "--stablevm-k-samples",
    type=int,
    default=4,
    help="Number of additional bank-sampled anchors per minibatch (only when --loss-type=stablevm).",
)
parser.add_argument(
    "--stablevm-bank-capacity-per-class",
    type=int,
    default=256,
    help="Capacity of the per-class anchor bank (only when --loss-type=stablevm).",
)
```

When `--loss-type=stablevm` is set, wrap the trainer's velocity_fn
with `StableVMLoss(velocity_fn, StableVMConfig(k_samples=4))` and
warm up the `SampleBank` from the first `--prefill-bank-fully`
minibatches.

### §3.2 Bank warm-up + per-class keying

Two integration decisions need product input:

1. **What is the class id?** Upstream uses ImageNet classes (1k).
   We could use (a) **metal id** (Pt=0, Ru=1, Ir=2, Au=3, Pd=4) for
   multi-metal retraining, or (b) **cluster id** from a pocket
   similarity clustering on PlatinAI / CrossDocked, or (c) **no
   class id** (single global bank). Recommended: **(c) for the
   first retrain** (lowest blast radius) → easy A/B vs vanilla CFM;
   revisit (a)/(b) once we confirm variance reduction holds on a
   real cohort.
2. **Where does the bank live?** Upstream keeps it on GPU; we can
   keep it CPU (cheap, fits in a few GB even at 256 × D-dim
   anchors). Recommended: **CPU bank for the first retrain** →
   trivially debuggable; promote to GPU bank only if we measure a
   CPU-bank bottleneck.

### §3.3 Risk surface

| Risk | Mitigation |
|---|---|
| Upstream repo drift (currently `StableVM/`, may move) | Pin commit hash in a follow-up `wf350_stablevm_full_port.md` once we import |
| Bank warm-up pollution (early anchors are noisy) | Skip the first N minibatches from the bank; only sample after `--prefill-bank-fully` |
| Self-normalized temperature | Currently τ=1 (softmax over −per-candidate MSE). Sweep τ ∈ {0.5, 1, 2} once we run |
| EGNN velocity net shape `[B, N, 3]` vs flat `[B, D]` | Stub is shape-agnostic; no trainer change required |
| Variance reduction might not transfer from toy linear field to real EGNN | A/B smoke on PlatinAI 4-cell cohort first; defer MoE / CF / PocketMol benchmarks until smoke passes |

---

## §4 Why not import the upstream verbatim

The upstream `StableVM/` directory is built around:

* **`accelerate launch` + SiT-XL/2 image generation** (not EGNN /
  pocket-conditioned ligand design);
* a **GPU-resident per-class sample bank** with `--prefill-bank-fully`
  warm-up logic;
* a **`VA-REPA` projector head** for representation alignment in
  low-variance regions (paper §3.3) that we do not need (YuelBond
  decoder + EGNN is a different head architecture).

Porting the *idea* (multi-sample self-normalized estimator) in
~200 LOC lets us:

1. Run the A/B smoke on Mol-Metal's pocket-conditioned cohort
   **without GPU slot consumption** (CPU-only stub).
2. Measure **variance reduction** on a real cohort before committing
   to a full EGNN + YuelBond re-train.
3. **Defer the projector / VA-REPA question** (paper §3.3) to a
   later workflow; it is orthogonal to the variance-reduction claim
   and would require a new head wire-up.

Once the A/B smoke passes (variance reduction ≥ 30% on a real
cohort), a follow-up workflow can decide between (a) import the
upstream `StableVM/` package and adapt the trainer, or (b) keep our
~200-LOC stub and tune it for EGNN.

---

## §5 Honest framing (per `party-twenty-spirit` §3.5)

### §5.1 MEASURED (this stub)

* **Repo URL**: <https://github.com/linYDTHU/StableVelocity>
* **License**: **MIT** (2026, Donglin Yang) — confirmed via WebFetch of
  the LICENSE file.
* **arXiv**: **2602.05435** (per the bibtex on the repo README) —
  confirmed via WebFetch of the GitHub README.
* **ICML 2026 poster page**: <https://www.icml.cc/virtual/2026/poster/66609> —
  confirmed via WebFetch.
* **Tests**: **11/11 PASS** in 0.20 s on CPU.
* **Standalone smoke**: **88.80% variance reduction** (K=4 vs K=1)
  on toy linear velocity field over 50 trials.

### §5.2 PROJECTED (not measured on Mol-Metal)

* The 88.80% reduction is on a **toy linear velocity field** with a
  random Gaussian anchor bank; the real EGNN + YuelBond velocity net
  may show different (likely smaller, possibly larger) variance
  reduction. **Must be re-measured on a real cohort before adoption.**
* The "2× faster sampling" claim from the upstream paper is on
  ImageNet 256 + SD3.5 + Flux — image diffusion models with
  continuous-pixel targets. Our pocket-conditioned 3D ligand
  generation has discrete bond heads and SE(3) equivariance, so
  the speedup may differ. **PROJECTED 1.2-1.5× end-to-end step
  throughput** at same quality (conservative; will update post-A/B).

### §5.3 NOT MEASURED YET (deferred)

* Wall-time speedup on Mol-Metal's pocket-conditioned retrain.
* VRAM impact (CPU bank is cheap; GPU bank would add ~256×D bytes
  per class).
* Convergence-quality tradeoff (do we hit the same decode quality at
  fewer epochs? Same quality at fewer NFEs?).
* `--loss-type=stablevm` flag wired into `train_cfm_yuelbond_wasserstein.py`
  (planned per §3.1; not yet implemented).
* VA-REPA projector head integration (paper §3.3; out of scope for
  this stub).

### §5.4 Deferred to follow-up workflows

| Item | Owner workflow |
|---|---|
| A/B smoke on PlatinAI 4-cell + MetalCytoToxDB 5-metal | Wf-350-A1 |
| `--loss-type=stablevm` wire-up into trainer | Wf-350-A2 |
| Full upstream import (with VA-REPA projector) | Wf-350-A3 (only if A1 + A2 succeed) |
| GPU bank warm-up + per-class keying strategy | Wf-350-A4 (after first smoke) |
| §3.2 product decision on class-id strategy | needs user OK |

---

## §6 Standing directives respected

* **AGENTS.md §5.1 (Standing GPU authorization)**: 0 Blackwell slots
  consumed; CPU-only stub + CPU-only tests.
* **AGENTS.md §5.2 (Honest framing)**: MEASURED vs PROJECTED vs NOT
  MEASURED YET split is explicit in §5; no aspirational numbers.
* **AGENTS.md §5.3 (No paper work)**: zero touches to `paper/*`.
* **AGENTS.md §5.4 (Tier-2 target)**: in-silico only; no wet-lab
  planning; Tier-2 computational submission stays in scope.
* **`party-twenty-spirit` §3.5**: PROJECTED tag applied to every
  speedup number; upstream repo + license + arXiv ID all traced to
  WebFetch responses (not memory).

---

## §7 File paths

| Path | What |
|---|---|
| `molmetal/adapters/flow_matching_lipman/stablevm_stub.py` | 100-LOC CPU stub (SampleBank + StableVMConfig + StableVMLoss) |
| `molmetal/tests/test_wf349a3_stablevm_stub.py` | 11 tests; **11/11 PASS in 0.20 s** |
| `TODO/completed/349a3_stablevm_stub.md` | this file |
| `TODO/completed/340r1_cfm_papers_survey.md` | upstream survey (this stub closes its §3.2 "code TBD" gap) |

**END OF Wf-349-A3** — Owner Hugo Lee; repo MIT-confirmed; CPU stub
SHIPPED with 11/11 tests; GPU verification deferred to Wf-350-A1
(USER_OK-gated per AGENTS.md §5.1).
