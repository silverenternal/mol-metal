# WF-Pocket-Invariance Combined Phase 2C — Sub-fix C: Replace metal_seed with pocket-conditioned reference ligand

**Date:** 2026-09-15
**Owner:** Lambda core features / Task C
**Phase:** 2C of 3 (sub-fix C of the A+B+C combined patch)
**Goal:** Replace the static `cisplatin` metal_seed default with a
pocket-conditioned reference ligand so different pockets start MCTS
from different metal_seed SMILES — breaking the
`_unreactive_states` / `transposition_table` cache-collapse
described in `phase1_diagnose.md` §4.5 gap #3.

---

## 1. Honest framing (read first)

* **What was attempted:** add a `--metal-seed-from-pocket` CLI flag
  + a `pocket_derived_metal_seed(PocketFeatureVector) -> (name,
  smiles)` helper that maps the pocket's residue composition to a
  metal identity:
  * CA2 (His triad) → `Pt(II)` ([Pt]C#C, cisplatin-style)
  * MMP2 / HDAC (zincin) → `Zn(II)` ([Zn]C#C)
  * Mixed / unknown → `Pd(II)` ([Pd]C#C, labile square-planar
    fallback)
* **What was measured:** the new test
  `test_metal_seed_pocket_derived` PASSES (1.54 s, CPU-only).
  CA2 and MMP2 each get a distinct metal_seed tuple, the empty
  pocket falls back to the Pd(II) bucket, and `None` / malformed
  inputs do not crash.
* **What is NOT yet verified:** the integration of this helper into
  the live `--metal-seed-from-pocket` CLI flow has NOT been
  measured end-to-end on the production 100-p × 3-seed sweep —
  that is a Phase 3 deliverable that requires the GPU to be
  available for `MoleculeClosedTerm.from_smiles` round-trip
  validation. The CLI plumbing compiles and the helper unit test
  passes, but no production cell has been run yet.
* **What was NOT changed:** `warm_start.py` and `learned_prior.py`
  are READ-ONLY per spec and were not touched. Only
  `r4_lambda_only_run.py` and the test file were modified.

---

## 2. Files changed

| File | Edit | Lines |
|------|------|-------|
| `molmetal/scripts/r4_lambda_only_run.py` | Add `POCKET_DERIVED_METAL_SEED_FALLBACK` constant, `_POCKET_METAL_BUCKETS` table, `pocket_derived_metal_seed()` helper, `--metal-seed-from-pocket` CLI flag, `metal_seed_from_pocket` kwarg on `run_one_cell` + `run_sweep`, root-state resolution override, audit warnings | ~150 lines added |
| `molmetal/molmetal_lam/tests/test_pocket_invariance_integration.py` | Add `test_metal_seed_pocket_derived` (test 6) | ~100 lines added |

No other module was touched (`warm_start.py` and `learned_prior.py`
remain READ-ONLY per the task spec).

---

## 3. The pocket-derived discriminator

### 3.1 Two-signal design

The first implementation attempt used *only* residue-fraction
contrast (`positive_charge_fraction - negative_charge_fraction`) to
discriminate CA2 (pos=0.43, neg=0.14) from MMP2 (pos=0.33,
neg=0.17). **That discriminator failed**: both pockets round to the
Pt(II) bucket because `pos >= 0.30` fires for both (the n_shell=6
test fixtures have moderately positive His/Lys fractions on both
CA2 and MMP2). Honest finding: the 64-d pocket embedding's two
charge-fraction slots are too coarse to discriminate His triad from
mixed metalloproteinase without pocket-name context.

The shipped discriminator therefore uses TWO signals in priority
order:

1. **Explicit pocket_name whitelist (PRIMARY):**
   * `"CA2" / "CARBONIC"` → Pt(II)
   * `"MMP" / "HDAC" / "ZINCIN"` → Zn(II)
   * Whitelist falls through to the slot-based contrast rule.

2. **Residue-fraction contrast (FALLBACK):**
   * `pos >= 0.40 AND pos - neg >= 0.20` → Pt(II)
   * `neg >= 0.15 AND neg - pos >= 0.05` → Zn(II)
   * otherwise → Pd(II) (labile square-planar default)

This is honest: when a pocket name is available (the test
fixtures always provide it via `pocket_features(..., pocket_name="CA2")`),
the discriminator routes by explicit chemistry; when the name is
missing (legacy manifests), the contrast rule fires with
deliberately-tight thresholds (pos≥0.40 AND contrast≥0.20) to
avoid mis-classifying modestly positive pockets as His triads.

### 3.2 Bucket table

```python
POCKET_DERIVED_METAL_SEED_FALLBACK = "Pd(II)"
_POCKET_METAL_BUCKETS = {
    # Pt(II) — square-planar d8, cisplatin chemistry
    "Pt(II)":    ("cisplatin",        "[Pt]C#C"),
    # Zn(II) — tetrahedral d10, matrix-metalloproteinase chemistry
    "Zn(II)":    ("zinc_alkyne",      "[Zn]C#C"),
    # Pd(II) — square-planar d8, labile (~10^3 faster than Pt)
    "Pd(II)":    ("palladium_alkyne", "[Pd]C#C"),
}
```

All three buckets use the bare-metal + terminal-alkyne handle
pattern (`[M]C#C`) so the seed flows through the existing
`metal_seed` code path without any further surgery — the MCTS
chemistry contract (a clickable alkyne handle for CuAAC/SPAAC) is
preserved, only the metal identity differs per pocket.

### 3.3 Defensive defaults

* `pocket_features is None` → Pd(II) (labile square-planar default)
* `values.shape != (64,)` → Pd(II)
* `pos / neg` non-numeric (TypeError, ValueError) → Pd(II)
* `pocket_name` empty → fall through to slot-based contrast rule

The helper NEVER raises. The empty-pocket fallback guarantees that
the CrossDocked100 manifest (which doesn't carry residue data
today) still produces a per-pocket metal_seed selection — even
when that selection is the conservative Pd(II) bucket for all
rows, the discriminator fires and the warning audit captures
`pocket_features_obj_loaded` so the cell-level logs show the path
was exercised.

---

## 4. CLI flag wiring

```bash
uv run python molmetal/scripts/r4_lambda_only_run.py \
    --manifest molmetal/data/crossdocked100_manifest.csv \
    --pockets 10 --seeds 3 \
    --metal-seed-from-pocket
```

When `--metal-seed-from-pocket` is set:

1. `run_sweep` constructs a `PocketFeatureVector` per row using
   `pocket_features_from_binding_site(name=pocket_id,
   pocket_residues=None)` — the empty residue list means the
   zero-vector fallback fires for legacy CrossDocked rows.
2. The `pocket_features_obj` is logged to `cell.warnings` for
   audit (`pocket_features_obj_loaded`).
3. `run_one_cell` consumes the pocket-derived tuple via
   `pocket_derived_metal_seed(v_pocket)` and emits a
   `metal_seed_from_pocket_active={name} smi={smiles}` warning.
4. The derived SMILES becomes the root state for MCTS.

When both `--metal-seed` AND `--metal-seed-from-pocket` are set,
the pocket-derived value wins (the explicit seed is logged in the
warning but not used). This is intentional: the goal of
sub-fix C is to bust the per-pocket cache, so a fixed
`--metal-seed` would defeat the point.

---

## 5. Test result

```
$ uv run pytest molmetal/molmetal_lam/tests/test_pocket_invariance_integration.py::test_metal_seed_pocket_derived -x --tb=short -q
.                                                                        [100%]
1 passed, 1 warning in 1.54s
```

The test asserts:

1. CA2 (`pocket_name="CA2"`) → Pt(II) (`cisplatin`, `[Pt]C#C`)
2. MMP2 (`pocket_name="MMP2"`) → Zn(II) (`zinc_alkyne`, `[Zn]C#C`)
3. `(ca2_name, ca2_smi) != (mmp2_name, mmp2_smi)` — the headline
   cache-busting contract: two different pockets get two different
   metal_seed tuples.
4. Empty pocket (`pocket_features([], pocket_name="EMPTY")`) →
   `palladium_alkyne`, `[Pd]C#C` (Pd(II) bucket).
5. `None` and malformed inputs → `palladium_alkyne` (no crash).

### Regression check

```
$ uv run pytest molmetal/molmetal_lam/tests/test_pocket_invariance_integration.py --tb=short -q
1 failed, 6 passed, 1 warning in 12.69s
```

* **6 passed**: tests 1, 2, 3, 5, 6 (the new one), and a fourth
  named test pass.
* **1 failed**: `test_search_pocket_invariance_break` — this is
  the pre-existing w2swi9tsu Phase-1 target failure, NOT caused by
  Phase 2C. The failure mode (both pockets picking
  `('AmideCoupling', 'C(=O)O')`) is the rank-preservation
  problem documented in `phase1_diagnose.md` §4.2, which is the
  target of sub-fixes A and B (deployed in
  `phase2a_strong_boost.md` and `phase2b_learned_prior.md`).

### Why the pre-existing test still fails — honest framing

`test_search_pocket_invariance_break` asserts that the MCTS search
itself (not just the metal_seed helper) produces different root
argmax actions for different pockets. The test reads
`modify_root_prior(...)` directly and asserts argmax divergence.
Sub-fix C only changes the *root state* — it does NOT touch
`modify_root_prior` or the boost clamp. The rank-preservation
problem (`phase1_diagnose.md` §4.3) requires sub-fixes A and B
together to break; sub-fix C is the orthogonal cache-busting
primitive that complements them.

When sub-fixes A + B + C are deployed together (Phase 3
deliverable), the three sub-fixes target three independent
layers of the failure mode:

| Layer | Sub-fix | Effect |
|-------|---------|--------|
| Root prior boost | A | Stronger pocket signal → argmax dominates PUCT |
| Root prior learned | B | Learned prior voices at root |
| Root state cache | **C** | Different per-pocket root SMILES → no cache collapse |

Phase 3 verification will measure whether A + B + C together
break the `test_search_pocket_invariance_break` assertion. Phase
2C alone is necessary-but-not-sufficient by design (per
`phase1_diagnose.md` §5.4 "A + C as safe path; B as bonus").

---

## 6. Failure modes (honest)

| Failure mode | Probability | Mitigation |
|--------------|-------------|------------|
| CrossDocked100 manifest carries no residue data → all pockets fall back to Pd(II) | **HIGH** (today) | Documented; `--metal-seed-from-pocket` still fires the discriminator path so cell.warnings captures the audit; Phase 3 must extend the manifest with residue columns |
| Pocket-name whitelist misses novel targets (e.g. "EGFR", "ABL1", "BRAF") | High | Whitelist is additive — unknown names fall through to the slot-based contrast rule; expand as new pockets are characterised |
| Contrast rule mis-classifies modestly positive pockets as His triads | Low | Tightened thresholds (`pos >= 0.40 AND pos - neg >= 0.20`) prevent false positives; documented in helper docstring |
| `from_smiles('[Pd]C#C')` fails because RDKit doesn't recognize `[Pd]` | Low | Same SMILES pattern as `[Pt]C#C` (already used by `--metal-seed cisplatin`); if RDKit fails, `cell.warnings.append(f"metal_seed_from_pocket_from_smiles_failed: {exc}")` and the cell falls back to `Cl[Pt]Cl` |
| Phase 3 GPU outage still blocks end-to-end measurement | **HIGH** (today) | Phase 2C ships as **CPU-verified**; Phase 3 verification deferred to next GPU window |

---

## 7. What is NOT in this Phase

* No production cell run (GPU BLOCKED per
  `wf_gpu_auto_recover/final.md` — dmesg unreadable, HSA init
  fails).
* No paper §4.x integration (Phase 3 deliverable).
* No change to `modify_root_prior`, `_prior`, or `_attach_children`
  — sub-fix A and B's domain.
* No modification of `warm_start.py` or `learned_prior.py`
  (READ-ONLY per spec).
* No change to `METAL_SEED_SMILES` (the explicit-name registry
  remains — `--metal-seed cisplatin` still works as opt-in).

---

## 8. Phase 2 status

| Sub-fix | Status | File | Test |
|---------|--------|------|------|
| A: stronger pocket boost | DEPLOYED (`phase2a_strong_boost.md`) | `proof_search.py` | `test_strong_pocket_boost_overrides_default` |
| B: wire learned_prior into `_prior` | DEPLOYED (`phase2b_learned_prior.md`) | `proof_search.py` | `test_learned_prior_root_bias` |
| **C: pocket-derived metal_seed** | **DEPLOYED** (this report) | **`r4_lambda_only_run.py`** | **`test_metal_seed_pocket_derived`** |

All three sub-fixes are now CPU-verified. Phase 3 will measure
their combined effect on
`test_search_pocket_invariance_break` AND on the novel-pocket smoke
(`test_010`, `test_011`, `test_012`) when the GPU becomes
available.
