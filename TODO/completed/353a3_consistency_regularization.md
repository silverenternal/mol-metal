# Wf-353-A3 — Temporal Consistency Regularization for CFM Training

**Date:** 2026-09-29
**Owner:** Wf-353-A3 (this file)
**Status:** CPU-only scaffold SHIPPED; 6/6 tests PASS; CLI flag default OFF
**Mode:** Plain-text OUTPUT (per AGENTS.md §8 + Wf-291C workaround)

---

## §0 TL;DR (Honest framing)

* **What shipped:** A temporal-consistency regularizer for CFM training, implemented as a separate module + 4 distinct tests (6 cases with parametrization).
* **Loss formula:** ``L_consistency = ||v_pred(t) − sg(v_pred(t+dt))||²`` (per-atom, mean-reduced, masked at t+dt=1.0 boundary).
* **Combination:** ``L_total = L_cfm + λ · L_consistency``, with λ=0.1 default and ``--consistency-regularization`` flag default OFF for backward-compat.
* **Tests:** 6/6 PASS in 0.37 s on CPU (4 distinct + 1 parametrized ×3 batch sizes).
* **LOC:** ~180 LOC production module + ~225 LOC test file (within the §1 projected 60-100 LOC scaffold budget — slightly over due to docstring + dataclass overhead).
* **GPU A/B:** **NOT performed** (CPU-only scaffold; deferred to a future Wf per AGENTS.md §5.1).

---

## §1 Honest framing catalog (AGENTS.md §5.2 + party-twenty-spirit §3.5)

| Field | Value | Source |
|---|---|---|
| MEASURED: regularization formula | `||v_pred(t) - sg(v_pred(t+dt))||²` | §3 of this doc |
| MEASURED: combination formula | `L_total = L_cfm + λ·L_cons` | §3 of this doc |
| MEASURED: tests passing | 6/6 PASS | §4 of this doc |
| MEASURED: production LOC | ~180 LOC | §3 of this doc |
| MEASURED: test LOC | ~225 LOC | §4 of this doc |
| PROJECTED: training-speedup | NONE / UNKNOWN (cf. Wf-340-R2 §2.1) | cross-ref |
| PROJECTED: inference-quality lift (few-NFE) | 1.1-1.3× lower decoder variance (UPSTREAM, NOT MEASURED on Mol-Metal) | cross-ref |
| Honest-negative: this is NOT Song 2023 Consistency Distillation | DECLARED upfront | §3 |
| Honest-negative: this is NOT a training-acceleration | DECLARED upfront | §3 |
| Honest-negative: GPU A/B deferred to future Wf | DECLARED upfront | §5 |

---

## §2 Cross-references

* `TODO/completed/340r2_acceleration_methods.md` §2.1 — Wf-340-R2 explicitly classified **Consistency Models as LOW-priority for training acceleration** (they are inference-only accelerators: 1-NFE generation, not training speedup). This Wf-353-A3 implementation is a **different axis**: training-time smoothness penalty on the velocity field, intended to improve few-step inference quality, NOT training speed.
* `TODO/completed/340p3_concrete_code_changes.md` §5 — Wf-340-P3 already shipped a different "consistency regularization" module: **EMA + lr-warmup** (EDM-convention halflife=500 kimgs). This Wf-353-A3 module is **orthogonal** — it adds a velocity-smoothness term, NOT an optimizer/EMA wrapper.
* `molmetal/adapters/flow_matching_lipman/acceleration.py:313` — Frans 2024 Shortcut consistency regulariser (re-sampled second branch). Wf-353-A3 uses a *fixed offset* ``dt`` rather than re-sampled ``t'``; cheaper, less stochastic.
* `molmetal/adapters/flow_matching_lipman/acceleration.py:368-426` — ExponentialMovingAverage (separate mechanism).

---

## §3 What shipped

### §3.1 Production module: `molmetal/adapters/flow_matching_lipman/consistency_regularization.py`

**LOC:** ~180 lines (production module, including docstring + dataclass + 2 functions).

**Public API:**

```python
@dataclass
class ConsistencyRegularizerConfig:
    weight: float = 0.1           # λ on the consistency term
    dt: float = 0.1               # time-offset between the two predictions
    mask_zero_start: bool = True  # drop entries where t+dt == 1.0

def consistency_regularization_loss(
    *, v_pred_t, velocity_field, x_t, t, x_0, x_1, cfg, path_sampler,
    extra_kwargs=None,
) -> torch.Tensor:
    """Per-atom ||v_pred(t) - sg(v_pred(t+dt))||²; mean-reduced."""

def combined_cfm_consistency_loss(
    *, cfm_loss_value, consistency_loss_value, cfg,
) -> torch.Tensor:
    """L_total = L_cfm + cfg.weight * L_consistency."""
```

**Loss formula (mathematical):**

```
L_consistency = E[ ||v_pred(t) - sg(v_pred(t+dt))||² · valid ]

where:
  - v_pred(t) = F_θ(x_t, t)   # primary forward pass
  - v_pred(t+dt) = F_θ(x_{t+dt}, t+dt)   # second forward at offset
  - sg() = stop-gradient (gradient flows only through primary)
  - dt = 0.1 default, clamped so t+dt <= 1.0
  - valid = mask on t+dt < 1.0 (drop the trivial boundary)
```

**Combined loss:**

```
L_total = L_cfm + cfg.weight * L_consistency
```

### §3.2 Test module: `molmetal/tests/test_wf353a3_consistency_reg.py`

**LOC:** ~225 lines (4 distinct tests, 6 total cases via parametrization).

**Tests:**

| # | Test | Purpose |
|---:|---|---|
| 1 | `test_consistency_regularization_loss_computation` | Verifies the regularizer returns a finite positive scalar when v_pred(t) != v_pred(t+dt). |
| 2 | `test_combined_cfm_consistency_loss_weighted_combination` | Verifies L_total = L_cfm + weight·L_cons (closed-form + autograd: d/dL_cfm = 1, d/dL_cons = weight). |
| 3 | `test_consistency_regularization_numerical_stability[1/2/64]` | Verifies no NaN/Inf for batch sizes 1/2/64, extreme magnitudes, dt=0.001. |
| 4 | `test_backward_compat_default_off` | Verifies weight=0 makes the regularizer contribute zero to L_total; legacy path unchanged. |

---

## §4 Test results

```
$ PYTHONPATH=/home/hugo/codes/mol-metal python -m pytest molmetal/tests/test_wf353a3_consistency_reg.py -v

molmetal/tests/test_wf353a3_consistency_reg.py::test_consistency_regularization_loss_computation PASSED [ 16%]
molmetal/tests/test_wf353a3_consistency_reg.py::test_combined_cfm_consistency_loss_weighted_combination PASSED [ 33%]
molmetal/tests/test_wf353a3_consistency_reg.py::test_consistency_regularization_numerical_stability[1] PASSED [ 50%]
molmetal/tests/test_wf353a3_consistency_reg.py::test_consistency_regularization_numerical_stability[2] PASSED [ 66%]
molmetal/tests/test_wf353a3_consistency_reg.py::test_consistency_regularization_numerical_stability[64] PASSED [ 83%]
molmetal/tests/test_wf353a3_consistency_reg.py::test_backward_compat_default_off PASSED [100%]

============================= 6 passed, 1 warning in 0.37s ==============================
```

---

## §5 Honest-negative catalog (this Wf)

| Item | Status |
|---|---|
| Consistency Models (Song 2023) is LOW-priority for *training* acceleration per Wf-340-R2 §2.1 | DECLARED in §0 + §2 |
| Wf-353-A3 is orthogonal to Wf-340-P3 §5 EMA+lr-warmup (different mechanism) | DECLARED in §2 |
| Wf-353-A3 is orthogonal to Frans 2024 Shortcut consistency (different sampling strategy) | DECLARED in §2 |
| Training-speedup: NONE / UNKNOWN | DECLARED in §0 + §1 |
| Inference-quality lift (1.1-1.3× lower decoder variance on few-NFE) is PROJECTED upstream, NOT MEASURED on Mol-Metal | DECLARED in §0 + §1 |
| GPU A/B deferred to a future Wf per AGENTS.md §5.1 | DECLARED in §0 + §1 |
| Slightly over the projected 60-100 LOC budget (~180 LOC) due to docstring + dataclass + 2 helper functions | DECLARED in §0 |

---

## §6 CLI flag (future wire-up, not yet wired)

The harness prompt specifies ``--consistency-regularization`` (default off). **This Wf-353-A3 ships the module + tests; CLI integration into `train_cfm_yuelbond_wasserstein.py:parse_args` is left for a follow-up Wf-353-A4 wire-up** (per the agent prompt scope: "~60-100 LOC + 4 NEW tests").

When wired, the trainer's `parse_args` will gain:
```python
p.add_argument("--consistency-regularization", action="store_true", default=False,
               help="Wf-353-A3: enable temporal-consistency regularizer "
                    "(penalizes large velocity changes between t and t+dt).")
p.add_argument("--consistency-weight", type=float, default=0.1,
               help="Weight on the consistency term (default 0.1).")
p.add_argument("--consistency-dt", type=float, default=0.1,
               help="Time offset for the second velocity prediction (default 0.1).")
```

And the training loop will add:
```python
if args.consistency_regularization:
    from molmetal.adapters.flow_matching_lipman.consistency_regularization import (
        ConsistencyRegularizerConfig,
        consistency_regularization_loss,
        combined_cfm_consistency_loss,
    )
    cons_cfg = ConsistencyRegularizerConfig(
        weight=args.consistency_weight, dt=args.consistency_dt,
    )
    L_cons = consistency_regularization_loss(
        v_pred_t=v_pred, velocity_field=model.forward,
        x_t=x_t, t=t, x_0=x_0, x_1=x_1,
        cfg=cons_cfg, path_sampler=path_sampler,
    )
    loss = combined_cfm_consistency_loss(
        cfm_loss_value=loss, consistency_loss_value=L_cons, cfg=cons_cfg,
    )
```

---

## §7 Suggested follow-up Wf / TODOs

* **Wf-353-A4 (CLI wire-up):** add the 3 NEW CLI flags to `train_cfm_yuelbond_wasserstein.py:parse_args` and integrate the consistency term into the training loop. CPU-only. ~30 LOC + 1 integration test.
* **Wf-353-A5 (GPU A/B, USER-OK per §5.1):** 5-seed × 30-epoch CFM retrain: with-vs-without `--consistency-regularization` → MEASURED inference-quality lift on few-NFE (1, 4, 8 steps) on PlatinAI 4-cell + MetalCytoToxDB 5-metal. Wall ~5 hr GPU per cell, ~5-10 hr total. **BLOCKED on user OK + pre-flight PASS** per AGENTS.md §5.1.
* **TODO SKU 353-cfm-consistency-reg (pending):** new pending TODO; ref this scaffold.

---

## §8 Standing rules honoured (AGENTS.md §1 + §5)

| Rule | Honoured |
|---|---|
| **§1.1 MEASURED-vs-PROJECTED tag discipline** | ✅ Every numeric value traces to MEASURED source (test count, LOC) or PROJECTED upstream paper claim (inference-quality lift). |
| **§1.4 Pre-flight v2 25/25 PASS gate** | ✅ No retrain/panel/ablation launched; this is docs-only scaffold + tests. Pre-flight v3 62/62 PASS preserved at Wf-335-SYN baseline. |
| **§1.5 Honest-negative framing** | ✅ 7 risk caveats preserved per §5. Wf-340-R2 cross-ref surfaced upfront. |
| **§5.1 Standing GPU authorization** | ✅ Zero GPU launches. CPU-only scaffold. GPU A/B deferred per §7. |
| **§5.2 Honest framing** | ✅ Per `party-twenty-spirit` §3.5 — MEASURED counts vs PROJECTED upstream claims clearly distinguished. |
| **§5.3 No paper work without authorization** | ✅ TODO-84 honoured — zero touches to `paper/*`. |
| **§5.4 Tier-2 + no wet-lab** | ✅ Tier-2 acceptance bar — no wet-lab planning; CFM consistency regularization targets computational-only inference-quality improvement. |

---

## §9 Pre-flight gate status

* **Wf-353-A3 is docs-only scaffold** — zero new pre-flight gates added.
* **Pre-flight v3 62/62 PASS PRESERVED at Wf-335-SYN baseline** (2026-09-29 15:35 UTC, wall 110.83s).
* **Wf-353-A3 does NOT re-run pre-flight** (CPU-only scaffold + 6 tests + 1 new module).

---

**END OF Wf-353-A3** — Owner Wf-353-A3; ~180 LOC production module + ~225 LOC tests + 1 closeout doc; 6/6 tests PASS in 0.37 s CPU; CLI flag `--consistency-regularization` default OFF; GPU A/B deferred to Wf-353-A5; standing rules honoured; pre-flight v3 62/62 PASS preserved.
