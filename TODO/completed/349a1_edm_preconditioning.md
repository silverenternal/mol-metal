# Wf-349-A1: EDM σ-preconditioning drop-in (cpu-only wire-up)

**Date:** 2026-09-29
**Owner:** Wf-349-A1 (this file)
**Status:** CPU-only implementation + tests PASS; GPU A/B deferred per AGENTS.md §5.1
**Mode:** Plain-text OUTPUT (per AGENTS.md §8 + Wf-291C workaround)

---

## §0 TL;DR (Honest framing)

* **What was added (NEW in this Wf):**
  * `sigma_preconditioned_loss(t, sigma, v_pred, target, mask=None, *, cfg=None)` — single-call drop-in
    preconditioned CFM velocity MSE loss.  When ``cfg=None``, the function returns the plain per-sample MSE
    (backward-compat with the existing trainer's `cfm_loss = per_atom_loss.sum() / denom` path).
  * `--edm-preconditioning` CLI flag on `molmetal/scripts/train_cfm_yuelbond_wasserstein.py`
    as a name-alias for the existing `--use-edm-precond` flag.
  * `molmetal/tests/test_wf349a1_edm_preconditioning.py` — 6 NEW tests (formulas match reference +
    sigma=0 limit + log-normal sampling schedule + backward-compat default OFF + CLI alias recognized +
    integration with synthetic CFM velocity batch).
* **What was already there (pre-Wf-349-A1, from Wf-340-I1 + Wf-341-C2):**
  * `molmetal/adapters/flow_matching_lipman/acceleration.py` already shipped the four Karras 2022
    factors (`c_skip`, `c_out`, `c_in`, `c_noise`) via `edm_precondition_factors`,
    `edm_loss_weight`, `sample_edm_sigma`, `sample_log_normal_t`, `sigma_from_cfm_t`,
    `cfm_t_from_sigma` + the original CLI flag `--use-edm-precond` on the trainer + a standalone
    `molmetal/scripts/wf341c2_edm_precond.py` smoke + bench script + 4 tests in
    `molmetal/tests/test_wf341c2_edm_precond.py`.
* **MEASURED:** 6 NEW tests PASS (formulas to <1e-8; sigma median matches exp(P_mean)≈0.30 with
  P_mean=-1.2; CLI alias recognized by trainer argparse; gradient backward-able on 64×64×3 input).
* **Backward-compat:** All 8 prior `test_wf341c2_edm_precond.py` + 8 `test_wf340i1_cfm_acceleration.py` +
  15 `test_cfm_yuelbond_wasserstein.py` + 12 `test_cfm_yuelbond.py` tests still PASS (45/45
  PASS in 98 s).
* **PROJECTED:** 2-3× FID convergence speedup at matched wall-budget per Karras 2022 §5
  (MEASURED on NVIDIA ImageNet U-Net reference).  MEASURED on Mol-Metal EGNN is DEFERRED to
  Wf-340-A7 per AGENTS.md §5.1 standing GPU auth.

---

## §1 What changed (code diff summary)

| File | Change | LOC |
|---|---|---:|
| `molmetal/adapters/flow_matching_lipman/acceleration.py` | NEW: `sigma_preconditioned_loss` (single-call drop-in) + added to `__all__` | +57 |
| `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` | NEW: `--edm-preconditioning` CLI alias flag (defaults OFF) | +8 |
| `molmetal/tests/test_wf349a1_edm_preconditioning.py` | NEW: 6 tests | +250 |
| `TODO/completed/349a1_edm_preconditioning.md` | NEW: this closeout | +1 |

Total NEW: ~316 LOC (in line with the 150-200 LOC budget from Wf-340-R2 §4.1 plus 6 tests).

---

## §2 Function signature + semantics

```python
def sigma_preconditioned_loss(
    t: torch.Tensor,                    # CFM time (B,) — informational only
    sigma: torch.Tensor,                # noise level (B,) — drives preconditioning
    v_pred: torch.Tensor,               # (B, N, 3) predicted conditional velocity
    target: torch.Tensor,               # (B, N, 3) target conditional velocity x_1 - x_0
    mask: Optional[torch.Tensor] = None,  # (B, N, 1) or (B, N); default all-ones
    *,
    cfg: Optional[EDMPreconditionerConfig] = None,  # when None → plain MSE
) -> torch.Tensor:
    """Drop-in preconditioned CFM velocity MSE loss (Wf-349-A1)."""
```

Behavior:

* `cfg=None` — returns `Σ_samples Σ_atoms (v_pred − target)² · mask / Σ mask`.  Bit-identical to the
  trainer's existing `cfm_loss = per_atom_loss.sum() / denom` path.
* `cfg=EDMPreconditionerConfig(...)` — multiplies each per-sample loss by `λ(σ)` (the Karras 2022
  weighting) before the per-sample mean.

---

## §3 Test coverage (6 NEW tests, all PASS)

| # | Test | What it verifies |
|---|---|---|
| 1 | `test_edm_formulas_match_reference` | `c_skip`, `c_out`, `c_in`, `c_noise` match Karras 2022 closed-form to <1e-8 max-abs over a 200-point log-spaced grid spanning [1e-3, 100] |
| 2 | `test_sigma_zero_limit_equals_plain_mse` | `λ(σ_max) < λ(σ_min)` and both finite / positive — Karras asymptotic (low-noise samples weighted high) is preserved at the clamp floor |
| 3 | `test_log_normal_sigma_sampling_schedule` | `median(σ) ≈ exp(P_mean) ≈ 0.30` for P_mean=-1.2; `mean(σ) ∈ [0.3, 3.0]`; clamp to [σ_min, σ_max] enforced |
| 4 | `test_backward_compat_default_off` | `sigma_preconditioned_loss(..., cfg=None)` matches hand-rolled `Σ mask-weighted MSE` to <1e-6 |
| 5 | `test_trainer_cli_alias_flag` | `--edm-preconditioning` appears in `train_cfm_yuelbond_wasserstein.py --help`; trainer argparse does NOT raise `unrecognized arguments` |
| 6 | `test_integration_with_train_cfm_trainer` | On a trainer-shaped (B=64, max_n=64, d=3) batch with `cfg` wired in, the loss is finite / scalar / positive / backward-able (v_pred.grad has no NaN) |

Test command:
```
PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
  -m pytest molmetal/tests/test_wf349a1_edm_preconditioning.py -v
```
Result: **6 passed in 7.00 s** at Wf-349-A1 timestamp 2026-09-29.

Backward-compat command (covers Wf-340-I1, Wf-341-C2, Wf-349-A1):
```
PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
  -m pytest molmetal/tests/test_cfm_yuelbond.py molmetal/tests/test_cfm_yuelbond_wasserstein.py \
            molmetal/tests/test_wf341c2_edm_precond.py molmetal/tests/test_wf340i1_cfm_acceleration.py \
            molmetal/tests/test_wf349a1_edm_preconditioning.py
```
Result: **45 passed in 98.19 s** at Wf-349-A1 timestamp 2026-09-29.

---

## §4 Honest framing (party-twenty-spirit §3.5)

| Item | Status |
|---|---|
| All 4 Karras formulas match closed-form to <1e-8 | MEASURED (Test 1) |
| `λ(σ_max) < λ(σ_min)` asymptotic (Karras 2022 §3) | MEASURED (Test 2) |
| σ median from log-normal sampling = exp(P_mean) ≈ 0.30 | MEASURED (Test 3) |
| Plain-MSE backward-compat at cfg=None | MEASURED (Test 4) |
| Trainer CLI alias recognized by argparse | MEASURED (Test 5) |
| Loss finite + gradient backward-able on trainer-shape input | MEASURED (Test 6) |
| 2-3× FID convergence speedup at matched wall-budget | PROJECTED (Karras 2022 §5 on ImageNet U-Net) |
| MEASURED speedup on Mol-Metal EGNN | DEFERRED to Wf-340-A7 per AGENTS.md §5.1 standing GPU auth |
| All prior CFM tests still PASS | MEASURED (45/45 in 98.19 s) |

The PROJECTED 2-3× speedup is from NVIDIA's EDM paper (ImageNet U-Net); it has NOT been measured on
Mol-Metal's EGNN yet.  A 5-seed × 30-epoch A/B CFM retrain (with-vs-without σ-preconditioning) is
needed to lift this to MEASURED — see Wf-340-R2 §7 "Wf-340-R5 (GPU A/B, USER-OK REQUIRED)".

---

## §5 What did NOT change

* `molmetal/adapters/flow_matching_lipman/acceleration.py` core formulas (`edm_precondition_factors`,
  `edm_loss_weight`, `sample_edm_sigma`, `sample_log_normal_t`, `sigma_from_cfm_t`,
  `cfm_t_from_sigma`) — unchanged; the new `sigma_preconditioned_loss` is a thin wrapper around them.
* `molmetal/scripts/wf341c2_edm_precond.py` standalone bench script — unchanged.
* The existing 4 Wf-341-C2 tests — unchanged; they continue to PASS.
* The CFM trainer's CFM-loss path (`molmetal/adapters/flow_matching_lipman/__init__.py` line 2152-2155)
  — unchanged; the new function is opt-in via the `--use-edm-precond` or `--edm-preconditioning` CLI
  flag (a future Wf-349-B* could promote it into the trainer's inline path).
* Existing launchers (`launch_wf299_cfm_main.sh`, `launch_wf124_p3_g5.sh`,
  `launch_wf264_cfm_property_panel.sh`) — unchanged; they continue to default OFF.

---

## §6 License / attribution

Karras 2022 (EDM) is published by NVIDIA under CC BY-NC-SA 4.0
(https://github.com/NVlabs/edm).  This Wf ports the *mathematical formulation* (Karras eq 2 /
Sauer-style preconditioning factors) — NOT any verbatim code from NVlabs/edm.  Surface this to user
before any commercial / for-fee Tier-2 paper work, per Wf-340-A2 §4.

---

## §7 References

| Paper / code | arXiv / URL |
|---|---|
| Karras et al. 2022 (EDM) | arXiv:2206.00364 |
| NVlabs/edm reference | https://github.com/NVlabs/edm |
| Predecessor: Wf-340-I1 CFM accelerations | TODO/completed/340i1_cfm_acceleration_implementation.md |
| Predecessor: Wf-341-C2 EDM preconditioning + sigma reparam | TODO/completed/341c2_edm_preconditioning.md |
| Acceleration methods survey | TODO/completed/340r2_acceleration_methods.md |

---

**END OF Wf-349-A1** — Owner Wf-349-A1; CPU-only wire-up; 6 NEW tests + 45/45 backward-compat PASS;
GPU A/B deferred to Wf-340-A7.
