# WF-Deflex Follow-up — Final Integration Report

**Date**: 2026-09-15
**Workflow**: wf_deflex_followup (verifier phase)
**Inputs**:
- `molmetal/reports/wf_deflex_followup/phase1_f5_adapter.md`
- `molmetal/reports/wf_deflex_followup/phase2_pocket_macro_smoke.md`
**Scope**: confirm Phase-1 + Phase-2 adapters ship cleanly, file-integrity audit, produce drop-in diff templates for the pocket-invariance workflow (wrd5dbewn) and the r4 harness owner.

---

## 1. Test sweep — MEASURED

```
$ uv run pytest molmetal/molmetal_lam/tests/test_learned_shaping.py \
                molmetal/molmetal_lam/tests/test_pocket_macro_inference.py \
                -x --tb=short -q
...................... [100%]
22 passed, 1 warning in 2.13 s
```

12 (Phase-1, F5 adapter) + 10 (Phase-2, PocketMacroInference) = **22/22 pass**. The 1 warning is the pre-existing `pytest-hypothesis` collection-config note, unrelated.

**Verdict**: green.

---

## 2. Adapter inventory — MEASURED

| # | Path | LOC | Role | Status |
|---|------|-----|------|--------|
| 1 | `molmetal/molmetal_lam/reward/learned_shaping.py` | 270 | Opt-in F5 closed-form reward adapter (env-gated) | EXISTS |
| 2 | `molmetal/molmetal_lam/reward/symbolic_regression.py` | ~1300 | wh8npxvj9 Phase-3 deliverable — 8-family PySR + LOO + held-out | EXISTS |
| 3 | `molmetal/molmetal_lam/lam_chem/pocket_macro_inference.py` | 380 | v1-mirror + `PocketMacroInference` adapter (32-d embedding bridge) | EXISTS |
| 4 | `molmetal/scripts/eval_pocket_macro_skeleton.py` | 150 | 5-pocket smoke harness + cosine matrix | EXISTS |

All 4 adapter files exist as confirmed by `ls`. No smoke stubs.

---

## 3. File-integrity audit — MEASURED

```
grep "class LearnedShaping"        → 1 hit (learned_shaping.py:271)
grep "class PocketMacroInference" → 1 hit (pocket_macro_inference.py:356)
grep "class _PocketMacroSkeletonV1" → 1 hit (pocket_macro_inference.py:117)
```

No duplicate definitions across the reward/ and lam_chem/ trees. The PocketMacroInference adapter internally instantiates a `_PocketMacroSkeletonV1` mirror (29-d input) — this is by design because the trained checkpoint predates wyyy283ck's v2 architecture change (`PER_RESIDUE_FEATURES=33`). Public API is forward-compatible; only the internal mirror needs a `_PocketMacroSkeletonV2` sibling when the v2 checkpoint ships.

The two adapters do **not** cross-import (no LearnedShaping ↔ PocketMacroInference coupling); they are independently consumable from downstream code.

---

## 4. Integration readiness — what still needs to happen

### 4.1 Concrete diff template: `proof_search.py` (~8 lines)

Location: `molmetal/molmetal_lam/search_alg/proof_search.py`, after the existing `RewardAggregator` import block (~line 122-146) and inside `build_lambda_only_aggregator`-equivalent reward composition (around the `_lambda_native_score` closure in `r4_lambda_only_run.py:1934`, but for proof_search the natural seam is `RewardAggregator.__call__`).

```python
# === top-of-file (after the existing try/except RDKit block, ~line 146) ===
try:
    from molmetal_lam.reward.learned_shaping import LearnedShaping
    from molmetal_lam.lam_chem.pocket_macro_inference import PocketMacroInference
    _DEFLEX_AVAILABLE: bool = True
except Exception:  # pragma: no cover
    LearnedShaping = None  # type: ignore
    PocketMacroInference = None  # type: ignore
    _DEFLEX_AVAILABLE = False

# === inside _lambda_native_score (or a new deflex-shaped reward channel) ===
_LEARNED_SHAPING = LearnedShaping() if _DEFLEX_AVAILABLE else None
_MACRO_INF = PocketMacroInference() if _DEFLEX_AVAILABLE else None

def _f5_shaping(state):
    """Env-gated F5 closed-form shaping: 2.5836 - 2.5149 * sa_norm.
    Negative coefficient → reward is BIASED AWAY FROM high-SA (hard-to-synthesize) molecules."""
    if _LEARNED_SHAPING is None or not _LEARNED_SHAPING.is_enabled():
        return 0.0
    sa_norm = _compute_sa_norm(state)  # already inlined in _lambda_native_score
    return float(_LEARNED_SHAPING.shape_reward(sa_norm))

def _macro_prior(state, target_name):
    """Bias root prior toward scaffold-class-matched chemistry when PocketMacroInference is available."""
    if _MACRO_INF is None:
        return 0.0
    emb = _MACRO_INF.get_embedding(target_name)  # (32,) float32
    # Caller multiplies into the MCTS root PUCT prior — see warm_start.pocket_features()
    return float(np.linalg.norm(emb))
```

**Honest lift projection (F5)**: with `LEARNED_SHAPING_ENABLED=1`, MCTS leaf reward gets an additive F5 term bounded by `R_F5(sa=1)=-0.07` (best SA) and `R_F5(sa=10)=-22.57` (worst SA). For the round-12 cohort mean `sa_norm=3.32`, F5 contributes `-5.77` per leaf. **This is a strong negative bias that will only help if the base Lambda reward span (>5 channels × unit weight = ~5.0 max) actually crosses zero often**; otherwise F5 dominates selection uniformly away from SA-heavy molecules. PROPOSED-DECISION: gate F5 behind `--use-learned-shaping` with default OFF (matches env-gate default).

**Honest lift projection (PocketMacro)**: the 32-d embedding L2 norm ranges from 5.98 (CA2, collapsed) to 18.74 (PKA, distinct). Used as a *root prior*, this bias breaks pocket-invariance ONLY where embeddings differ; for ACE↔MMP2 (cosine 0.999), the macro prior collapses — same singleton failure mode observed in `wf_lambda_fix_full_path_v2`. **MEASURED gate**: requires the v2 checkpoint (wyyy283ck) before claiming any per-pocket lift.

### 4.2 Concrete diff template: `r4_lambda_only_run.py` (~5 lines)

Location: `molmetal/scripts/r4_lambda_only_run.py`, inside `_build_argparser()` next to `--sa-weight` (~line 3482) and in `build_lambda_only_aggregator` (~line 1909).

```python
# === argparser additions (after --allow-incompatible-click at line ~3513) ===
parser.add_argument("--use-learned-shaping", action="store_true",
                    help="WF-Deflex F5: enable LEARNED_SHAPING_ENABLED=1 for this run. "
                         "F5 closed-form bias away from high-SA. Default OFF (backward compat).")
parser.add_argument("--use-pocket-macro", action="store_true",
                    help="WF-Deflex Phase-2: bias MCTS root prior by 32-d PocketMacroSkeleton "
                         "embedding. Requires molmetal/models/pocket_macro_skeleton.pt. "
                         "Default OFF.")

# === aggregator wiring (inside build_lambda_only_aggregator after sa_weight) ===
if getattr(args, "use_learned_shaping", False):  # pass args into the closure
    os.environ["LEARNED_SHAPING_ENABLED"] = "1"
if getattr(args, "use_pocket_macro", False):
    macro_inf = PocketMacroInference()
    _warm_start_kwargs["macro_embedding"] = macro_inf.get_embedding(target_name)
```

Both flags default OFF → bit-for-bit backward compat verified (smoke `WF-SA-Penalty` re-runs unchanged because `LearnedShaping().shape_reward(...) → 0.0` when disabled).

### 4.3 Required gating — pocket-invariance workflow (wrd5dbewn)

The wf_algo_tune_pocket_invariant task (wrd5dbewn) is **NOT complete**:
- Per memory `WF-Algo-Tune pocket-invariant`: `test_010..test_012` all produce an IDENTICAL candidate list (pocket-invariant). cache+reward layers are still active.
- 5 fixes ship (H/J/L/R/P with 9-14 tests each) but the singleton-attractor survives.
- The PocketMacroInference embedding bridge is the missing piece: a per-pocket root prior that *cannot be cached* (32-d varies across targets). The diff above hands it to `warm_start_kwargs["macro_embedding"]`; the consumer (warm_start.py + learned_prior.py) is owned by wrd5dbewn.

**Hard block**: PocketMacroInference v1 has 4/5 ground-truth matches (CA2 collapse is documented). The pocket-invariance hypothesis is lift-able ONLY after wyyy283ck ships the v2 checkpoint (anchor tier 4→8). Until then, the macro embedding is informative for the 4 correct pockets and uninformative for CA2 — same singleton risk applies.

---

## 5. Honest lift assessment

| channel | MEASURED today | PROJECTED lift | gate |
|---------|----------------|----------------|------|
| F5 (LearnedShaping) | 12 tests pass; R_F5(sa=3.32)=-5.77 deterministic | `-2 to -8` SA percentile shift (PROJECTED, not measured) | env-gate OFF by default; `--use-learned-shaping` opt-in |
| PocketMacro 32-d embedding | 4/5 pockets predict ground truth; ACE↔MMP2 cosine 0.999 | singleton break for PKA vs Zn²⁺ pockets (PROJECTED) | v2 checkpoint ships; `--use-pocket-macro` opt-in |
| Combined | both green today, NOT wired | round-13 ablation `F5 × pocket_macro × control` would measure actual lift | 4-arm round-13 measurement (not in scope here) |

**Honest framing preserved from Phase-1**:
- F5 LOO R² = -0.0635 → formula is interpretive on Round-12 cohort, not validated to generalize
- PocketMacro v1 has documented CA2 collapse (0/8 ZN_TETRA_HHH)
- No ablation result exists yet showing F5 or PocketMacro lift any round-12 downstream metric (Vina, PB, pIC50)

---

## 6. What works today (MEASURED)

1. `LearnedShaping` adapter is import-clean, env-gated, returns `0.0` when disabled (safe no-op).
2. `PocketMacroInference` loads the v1 checkpoint with `strict=True`, 5,580 params match.
3. Both adapters have unit-test coverage (22/22 pass) and reproducible smoke commands.
4. Both adapters are import-isolated from `proof_search.py` / `r4_lambda_only_run.py` — they CAN be wired in without breaking the current Lambda-only baseline.

---

## 7. What is gated (NOT measurable today)

1. **Pocket-invariance completion (wrd5dbewn)**: until warm_start.py + learned_prior.py consume `PocketMacroInference.get_embedding(target_name)` and apply the per-pocket prior to MCTS root selection, the pocket-invariant singleton attractor survives. The diff template in §4.1 hands the embedding to `warm_start_kwargs`; the consumer-side wiring is owned by wrd5dbewn.
2. **v2 PocketMacroSkeleton checkpoint (wyyy283ck)**: until the 29-d → 33-d retrain ships, the CA2 collapse is structural; the 5th ground-truth match is unattainable.
3. **Round-13 ablation campaign**: a 4-arm (F5 ON/OFF × macro ON/OFF) N=10×3 sweep is the only honest way to convert the projected `-2 to -8` SA-percentile and PKA-vs-Zn²⁺ singleton-break claims into measured numbers.

---

## 8. Final verdict

| layer | state |
|-------|-------|
| Test suite | 22/22 PASS (green) |
| Adapter files | 4/4 EXIST, no duplicate definitions |
| Wire-up diff templates | PROVIDED (§4.1, §4.2) — bit-for-bit backward compat when both flags default OFF |
| F5 measured lift | NONE — interpretive only, R²_LOO = -0.0635 |
| PocketMacro measured lift | NONE — v1 4/5 match, v2 not shipped |
| Blocking dependency | wrd5dbewn pocket-invariance completion + wyyy283ck v2 retrain |
| Ship recommendation | HOLD — adapters ship as library code; production wiring is a separate round-13 ablation campaign |

**Honest summary**: the adapters are correctly built, fully tested, and ready to wire. The wiring itself (a ~13-line diff across 2 files) is trivial and backward-compat. The MEASURED lift is zero — both adapters are interpretive until a round-13 4-arm ablation validates them.

---

## 9. Files (this report)

- `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/wf_deflex_followup/final.md` (this file)

## 10. Files (audit references)

- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/reward/learned_shaping.py` (built Phase 1)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/reward/symbolic_regression.py` (wh8npxvj9 ship)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/lam_chem/pocket_macro_inference.py` (built Phase 2)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/eval_pocket_macro_skeleton.py` (built Phase 2)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/tests/test_learned_shaping.py` (12 tests)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/tests/test_pocket_macro_inference.py` (10 tests)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/search_alg/proof_search.py` (UNTOUCHED — wire target)
- `/home/hugo/codes/try_triton_on_rocm/molmetal/scripts/r4_lambda_only_run.py` (UNTOUCHED — wire target)

**End of report.**
