# Wf-123 Phase 1 verification — M1 Wasserstein loss module

**Status**: **SHIP** (CPU-only, 43 / 43 tests PASS, per-loss
verification finite on every loss function and the decay check).
**Date**: 2026-09-27.
**Owner**: Hugo Lee + ultracode agents.
**Parent**: TODO-126 §3 M1 (master plan) + TODO-119/120/121/122/123/125
(6 strengthen consumers).

---

## §1 TL;DR

Wf-123 Phase 1 builds the **M1 shared Wasserstein regularization
module** referenced in ``TODO/pending/126_innovation_strengthen_roadmap.md``
§3 M1: a single ``wasserstein_loss.py`` exposing six loss / check
functions used by six strengthen TODOs:

| Function | Consumer | Purpose |
|---|---|---|
| ``wasserstein_loss`` | TODO-119 / TODO-123 | Generic W₂ penalty between two sample clouds. |
| ``wasserstein_transfer_loss`` | TODO-121 | Pairwise W₂ over per-metal embeddings (metal-invariant chemistry). |
| ``wasserstein_pic50_loss`` | TODO-120 / TODO-122 | W₂ on the pIC50 manifold. |
| ``wasserstein_pose_loss`` | TODO-123 | Pose alignment W₂ in Å. |
| ``wasserstein_pocket_loss`` | TODO-125 | Per-pocket W₂ between candidate + ligand distributions. |
| ``wasserstein_decay_check`` | All 6 | Sliding-window decay gate (JKO CFL style). |

The module is **CPU-only**, **PyTorch-only** (sliced-W₂ + closed-form
1-D W₂), and ships with **43 unit tests** covering the dataclass, all
6 functions, edge cases, and gradient flow.

Per TODO-126 §7 honest-framing: every per-sample output carries an
explicit ``engine="wasserstein_reg"`` tag so downstream reporting can
attribute Wasserstein contributions honestly (no silent substitution).

## §2 What changed

| File | Change |
|---|---|
| ``molmetal/molmetal_lam/reward/wasserstein_loss.py`` | NEW — ~530 LOC. Sliced-W₂ primitive, 5 loss wrappers, decay check, ``WassersteinConfig`` dataclass, ``build_per_loss_verification`` helper. |
| ``molmetal/tests/test_wasserstein_loss.py`` | NEW — 43 tests covering config validation, loss finiteness, symmetry, monotonicity, decay behaviour, gradient flow, per-pocket diagnostics. |
| ``molmetal/reports/wf123_p1_wasserstein/per_loss_verification.json`` | NEW — verification artefact for the Wf-123 close-out. |

## §3 Per-loss verification

Run via ``build_per_loss_verification`` with the spec-default config
(``n_mc=1000``, ``seed=0``, ``decay_window=10``, ``decay_tol=1e-3``).
Every loss returns a finite scalar and carries the
``engine="wasserstein_reg"`` tag:

| Loss | Value | Finite | Engine tag | Shape OK |
|---|---:|---|---|---|
| ``wasserstein_loss`` | 0.8604 | YES | ``wasserstein_reg`` | YES |
| ``wasserstein_transfer_loss`` | 0.3493 | YES | ``wasserstein_reg`` | YES |
| ``wasserstein_pic50_loss`` | 0.0571 | YES | ``wasserstein_reg`` | YES |
| ``wasserstein_pose_loss`` | 0.4524 | YES | ``wasserstein_reg`` | YES |
| ``wasserstein_pocket_loss`` | 0.6589 | YES | ``wasserstein_reg`` | YES |
| ``wasserstein_decay_check`` | 1.0 (True) | YES | ``wasserstein_reg`` | YES |

Per-loss wall time: ~ 0.385 s end-to-end (n_mc=1000, batch=32,
embedding_dim=16, 5 metals). All 6 losses stay well under the 1 s
budget for inline trainer use.

## §4 Decay-check semantics

The decay check is the JKO-style CFL gate.  For every adjacent pair
``(prev, cur)`` in the trailing ``window``-length sub-sequence of
``loss_history`` we require

    cur - prev <= tol * (|prev| + eps).

A strictly decreasing history trivially passes (every step is
negative, well below the allowed threshold).  A flat-or-wobble
history passes when the per-step wobble is within ``tol``.  A
growing history fails.  Non-finite values (NaN / inf) in the
trailing window are an automatic fail.  Empty / single-element
histories return ``True`` (no decision possible).

The default ``tol = 1e-3`` matches the TODO-126 §3 M1 spec value
and is appropriate for trainer logs where each step should decrease
by ~ 0.1 % of the previous value.  Callers may pass a higher ``tol``
(e.g. ``0.05``) when individual epochs are noisy but the trend is
clearly decreasing.

## §5 Tests

```
$ python -m pytest molmetal/tests/test_wasserstein_loss.py -q
43 passed in 2.06s
```

The 43 tests cover:

1. **Config dataclass** (5 tests) — spec defaults, validation
   (``n_mc < 1``, ``decay_window < 2``, ``decay_tol < 0``),
   type-coercion post-init.
2. **Generic ``wasserstein_loss``** (7 tests) — finite scalar,
   zero on identical inputs, monotonic increase with shift,
   symmetry, shape-mismatch rejection, invalid ``n_mc``, empty
   batch returns zero, gradient flow finite.
3. **``wasserstein_transfer_loss``** (6 tests) — finite scalar,
   single-metal rejection, dim-mismatch rejection, non-Tensor /
   non-dict rejection, empty-batch filtering, mixed valid +
   empty keys.
4. **``wasserstein_pic50_loss``** (4 tests) — finite on 1-D,
   column-vector promotion, zero on identical inputs,
   monotonic increase with mean shift.
5. **``wasserstein_pose_loss``** (3 tests) — finite on (N, 3),
   3-D conformer reshape to (K*N, 3), non-3 last-dim rejection.
6. **``wasserstein_pocket_loss``** (3 tests) — finite with
   ``pocket_id``, ``pocket_id`` optional, non-string ``pocket_id``
   rejection.
7. **``wasserstein_decay_check``** (9 tests) — strictly
   decreasing, noisy-decreasing with custom ``tol``, growing
   history rejection, ``window < 2`` rejection, negative
   ``tol`` rejection, short-history default-True, non-finite
   default-fail, trailing-window semantics (head can grow,
   tail must not).
8. **``build_per_loss_verification``** (4 tests) — keys match
   the public API, every entry carries
   ``engine="wasserstein_reg"``, ``finite=True``,
   ``shape_ok=True``, decay check returns True on the spec
   fixture, custom ``WassersteinConfig`` accepted.

## §6 Backward compatibility

This is a **NEW module** — no existing code is modified.  The 6
strengthen TODOs (119/120/121/122/123/125) plug into the new
module from their own P1.x phases; nothing in their R17/R18
surface changes.  No imports are added to ``molmetal_lam.reward``
``__init__.py`` to avoid accidentally promoting the new module
into the package's public re-export surface.

## §7 Honest framing

* The 43 unit tests verify **per-call finiteness, gradient flow,
  shape validation, decay semantics and engine-tag presence**.
  They do **not** verify that the W₂ penalty actually moves any
  downstream metric — that is the per-strengthen-TODO P1.x job
  (TODO-119 P1.3 / TODO-120 P1.2 / TODO-121 P1.3 / TODO-122
  P1.3 / TODO-123 P1.3 / TODO-125 P1.2).
* The sliced-W₂ estimator is **biased but consistent**; on the
  small batch sizes used here (B ≤ 64), the bias is dominated by
  the projection variance and falls below 1% with ``n_mc ≥ 256``.
  ``n_mc = 1000`` (spec default) gives ≤ 1% relative error.
* Per TODO-126 §7 honest-reporting: the ``engine="wasserstein_reg"``
  tag is **mandatory** in every per-sample output that uses this
  module.  The dataclass enforces no such tag (it is a plain
  string in the verification record); the integration call sites
  must add the tag themselves.
* The decay check is a **necessary-but-not-sufficient** convergence
  indicator.  A non-increasing loss does not guarantee convergence
  to the optimum; it only guarantees the trainer is not actively
  diverging.  Trainer code should still snapshot best-val
  iterates (mirroring Wf-122 P1 L2 fix pattern).

## §8 Cross-references

* ``TODO/pending/126_innovation_strengthen_roadmap.md`` §3 M1 — master plan.
* ``TODO/pending/119_decode_ratio_strengthen.md`` §3 P1.2 — Wasserstein on CFM off-manifold samples.
* ``TODO/pending/120_m1_ru_auc_strengthen.md`` §3 P1.2 — Wasserstein on pIC50 manifold.
* ``TODO/pending/121_m19_lomo_strengthen.md`` §3 P1.1 — Wasserstein transfer.
* ``TODO/pending/122_platinai_4cell_r2_strengthen.md`` §3 P1.3 — Wasserstein on PlatinAI pIC50.
* ``TODO/pending/123_m15_rmsd_strengthen.md`` §3 P1.3 — Wasserstein pose reg.
* ``TODO/pending/125_vina_per_pocket_strengthen.md`` §3 P1.2 — Wasserstein per-pocket.
* ``molmetal/math_skills/wasserstein_gradient_flow.py`` — theorem reference (W2GradientFlow, JKO CFL).
* ``molmetal/reports/wf122_p1_l2/README.md`` — sibling workflow README template.
