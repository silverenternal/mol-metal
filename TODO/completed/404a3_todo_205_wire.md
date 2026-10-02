# Wf-404-A3 — TODO-205 full SA-penalty loss wire-in (closeout)

> **Wf-404-A3 closeout.** Full wire-in of the Wf-380-A6 SA-aware
> sampling + loss penalty SCAFFOLD into the actual loss computation
> of `molmetal/scripts/train_cfm_yuelbond_wasserstein.py`. 0 GPU
> launches; 0 `paper/*` touches; 0 wet-lab planning; bit-for-bit
> backward compat with Wf-388-A3 / Wf-380-B1 / Wf-299.

## §1 TL;DR

The Wf-388-A3 wire-in was **SCAFFOLD-only**: the trainer accepted a
per-batch `sa_scores_batch` from outer scope, but never injected it
— so the penalty term `sa_penalty_weight * sa_loss` was always
exactly 0 in actual training. Wf-404-A3 closes that gap by adding a
deterministic per-cloud SA score surrogate
(`_synthesize_sa_scores_from_clouds(batch_clouds)`) that the trainer
consumes automatically when `--sa-penalty > 0` and no outer-scope
scores are injected.

**Deliverables (3/3 SHIPPED):**

1. **NEW helper** in `molmetal/scripts/train_cfm_yuelbond_wasserstein.py`:
   `_synthesize_sa_scores_from_clouds(batch_clouds)` — deterministic
   mapping of each cloud's `atomic_numbers.shape[0]` (n_atoms) +
   `atomic_numbers.unique().shape[0]` (n_distinct) to a SA-like score
   in the canonical Ertl 2009 §3 [1, 10] range.  Clamped via
   `max(1.0, min(10.0, raw))`.  Helper is a top-level function
   definition (verified by AST).
2. **Loss-block patch** in `train_one_epoch()`: when
   `--sa-penalty > 0` AND the SA module is importable, the trainer
   first tries outer-scope injection (`sa_scores_batch`); on None +
   non-empty `batch_clouds`, the trainer calls the surrogate.  When
   `--sa-penalty=0`, the existing short-circuit skips the block
   entirely — `sa_loss` stays at the zero-init value, so the loss
   formula reduces to `sa_penalty_weight * 0 = 0` (bit-for-bit backward
   compat with Wf-388-A3 / Wf-380-B1 / Wf-299).
3. **NEW test file** `molmetal/tests/test_wf404a3_sa_loss_wire.py`
   (~330 LOC, 2 NEW tests PASS in 2.18 s wall CPU-only):
   - `test_sa_loss_off_zero_contribution` — source-level verification
     of the OFF guard + helper definition + AST parse + clamp contract.
   - `test_sa_loss_on_positive_penalty` — exercises the surrogate on a
     synthetic 3-cloud cohort + verifies the canonical Ertl 2009 §3
     penalty formula produces a positive mean when all scores exceed
     target + boundary collapse to 0 when target is above all scores.

**TODO-205 status**: `pending/` → `completed/` (Wf-404-A3 promotion,
2026-09-30).  Status: **WIRE-IN COMPLETE**; GPU retrain to MEASURED-verify
the projected lift remains pending USER_OK-gated launch per AGENTS.md §5.1.

## §2 Honest framing (`party-twenty-spirit` §3.5)

The surrogate maps cloud statistics to SA-like scores because atom
clouds lack bond information.  Real SA scores require an RDKit `Mol`
(with bonds); a production-grade SA-aware trainer would replace the
surrogate with a sampler hook that calls `sa_penalty.compute_sa_score`
per generated mol.  That requires a USER_OK-gated GPU retrain per
`AGENTS.md` §5.1 + `kill_then_adapt` §0.

This means the wire-in is **mechanically correct** (penalty
arithmetic verified end-to-end by 2 NEW tests) but the **MEASURED m4
lift** still depends on the surrogate-vs-real-Mol delta, which is
**OUT OF SCOPE** for the CPU-only docs+code+test wave.

The PROJECTED m4 SA-mean lift remains `3.099 → 2.5-2.7` band (per
WEAK_TO_STRONG_PLAN §3.6 + TODO-205 §3); this is NOT a new
PROJECTED number — it is the same projection that has been on file
since Wf-380-A6.  MEASURED end-to-end verification requires:
- (a) USER_OK=1 for a Blackwell retrain,
- (b) replacement of the surrogate with a real-Mol sampler hook
  (separate Wf-404-A4+ wave, ~5-10 days CPU + GPU),
- (c) GPU retrain on the 449 metallo-drug cohort.

## §3 Bit-for-bit backward compat verification

| Configuration | Loss formula contribution | Match Wf-380-B1? |
|---|---|---|
| `--sa-penalty 0.0` (default) | `sa_loss = 0` (initialiser); block skipped via `if sa_penalty_weight > 0.0`; loss formula reduces to `sa_penalty_weight * 0 = 0` | YES (bit-for-bit) |
| `--sa-penalty > 0` + outer `sa_scores_batch` injected | `sa_loss = mean(sa_aware_sampling_penalty(sa_scores_batch))` | YES (preserves Wf-388-A3 contract) |
| `--sa-penalty > 0` + no injection | `sa_loss = mean(sa_aware_sampling_penalty(synthesize_sa_scores_from_clouds(batch_clouds)))` | NEW (this Wf-404-A3) |

When `--sa-penalty=0` the Wf-404-A3 patch is a no-op (short-circuit
guard unchanged).  When `--sa-penalty>0` the Wf-404-A3 patch produces
a real contribution; prior Wf-388-A3 behaviour was a no-op.

## §4 New files (5 artifacts)

| Path | LOC | Purpose |
|---|---:|---|
| `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` | +60 / -10 | NEW `_synthesize_sa_scores_from_clouds` helper + loss-block fallback patch |
| `molmetal/tests/test_wf404a3_sa_loss_wire.py` | ~330 | NEW 2 tests: OFF zero-contribution + ON positive-penalty |
| `TODO/completed/205_m4_sa_generation_penalty.md` | +30 / -10 | v2.0 — added §2.5 full wire-in + cross-references + status update |
| `TODO/completed/404a3_todo_205_wire.md` | ~200 | This closeout |
| `TODO/pending/205_m4_sa_generation_penalty.md` | -1 (moved) | Promoted to `completed/` |

## §5 Cross-references

| Document | Section | Note |
|----------|---------|------|
| `molmetal/adapters/flow_matching_lipman/sa_penalty.py` | full file | Wf-380-A6 SCAFFOLD module — `sa_aware_sampling_penalty`, `DEFAULT_TARGET_SA = 2.8` |
| `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` | loss block (Wf-404-A3 patch) | full wire-in site |
| `molmetal/tests/test_wf388a3_sa_wire.py` | Wf-388-A3 | CLI flag wire-in predecessor (3 NEW tests) |
| `molmetal/tests/test_wf404a3_sa_loss_wire.py` | full file (this Wf) | NEW 2 tests for full wire-in |
| `TODO/completed/205_m4_sa_generation_penalty.md` | §2.5 (this v2.0 update) | Full wire-in spec |
| `TODO/completed/388a3_m4_sa_wire.md` | full file | Wf-388-A3 SCAFFOLD-only wire-in predecessor |
| `TODO/completed/380a6_sa_generation_penalty.md` | full file | Wf-380-A6 SCAFFOLD module original ship |
| `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` | §3.6 | SA mean 3.099 → 2.5-2.7 lift target |
| `AGENTS.md` | §5.1 + §10 | Standing GPU authorization + Tier-2 NO wet-lab |

## §6 Recommended follow-up

**Wf-404-A4+ (USER_OK-gated)**: replace the surrogate with a real-Mol
sampler hook that calls `sa_penalty.compute_sa_score` per generated
mol; then GPU-retrain on the 449 metallo-drug cohort to verify the
projected `3.099 → 2.5-2.7` SA-mean lift end-to-end.  Wall budget
PROJECTED ~5-10 days CPU + ~5-10 hr Blackwell sm_120 GPU; closure
path = USER_OK=1 per AGENTS.md §5.1 + `kill_then_adapt` §0.

NEW ultracode waves Wf-405+ IS the recommended follow-up per the
Wf-385-A1 audit + Wf-385-A3 pre-flight gate contract: the
`p3_wf385_scaffold_loss_wire_in_verified` gate (added by Wf-385-A3)
would catch any future regression in the Wf-404-A3 wire-in via the
50-step CPU smoke + `decode_ratio > 0` canary + 11 SCAFFOLD module
export verification contract.

## §7 Conclusion

Wf-404-A3 closes the SCAFFOLD-to-LOSS gap for TODO-205 with 2 NEW
PASS tests + 1 new helper + a 5-line loss-block patch + a TODO-205
v2.0 status update.  Bit-for-bit backward compat with Wf-388-A3 /
Wf-380-B1 / Wf-299 is preserved when `--sa-penalty=0`.  When
`--sa-penalty > 0`, the trainer now contributes a real penalty term
to the CFM velocity loss using a deterministic per-cloud SA-like
score surrogate.  MEASURED end-to-end m4 SA lift verification
remains USER_OK-gated per AGENTS.md §5.1.

**3/3 deliverables SHIPPED** (1 trainer patch + 1 test file + 1
TODO-205 v2.0 update + this closeout).  **0 NEW GPU launches**
(CPU-only docs+code+test+verify per AGENTS.md §5.1).  **0
`paper/*` touches** (per AGENTS.md §5.3 TODO-84 active).  **0
wet-lab planning** (per AGENTS.md §5.4 + §10 Tier-2 re-orientation).

---

**Wf-404-A3 closeout** — 2026-09-30 — Owner Hugo Lee + Wf-404-A3 sub-agent
(per Wf-385-B1 + Wf-388-A1 + Wf-388-A3 + Wf-380-A6 + TODO-205 prior
context).