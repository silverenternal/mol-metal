# WF-Lambda-Boost Phase 1 — Pocket-conditioned reference ligand resolver

**Date:** 2026-09-16
**Owner:** Lambda boost agent
**Phase:** 1 of 4 (sub-fix C extended)
**Goal:** Replace the static `metal_seed=cisplatin` lambda-calculus root with
a per-pocket *reference ligand SMILES* — different chemistry per pocket breaks
the `_unreactive_states` cache + `transposition_table` at the lambda-calculus
level, complementing the existing `--metal-seed-from-pocket` which only changes
the metal identity.

---

## 1. Honest framing (read first)

* **What was attempted:** ship a NEW
  `molmetal/molmetal_lam/lam_chem/reference_ligand_resolver.py` module
  with a 5-pocket lookup table + WARN-on-fallback to cisplatin legacy.
  Wire it into `r4_lambda_only_run.py` as `--use-pocket-conditioned-reference`
  CLI flag (opt-in default OFF).
* **What was measured:** 21/21 new tests pass in 0.31 s on CPU.  The
  resolver returns 5 distinct SMILES for 5 named pockets (CA2 / MMP2 /
  HDAC2 / pocket_007 / pocket_011), falls back to `[Pt]C#C` with a WARN
  when `pocket_features=None`, and resolves each pocket in <50 ms.
* **What is NOT yet measured:** the integration impact on the production
  100-p × 3-seed sweep is BLOCKED on the GPU outage (per WF-GPU-Recovery-Now
  2026-09-15).  The CLI plumbing compiles + the unit tests pass, but
  no production cell has been re-run yet.
* **What was NOT changed:** `pocket_derived_metal_seed` (the metal-level
  complement) is READ-ONLY per the existing Phase-2C design.  We only
  added a *new* layer on top.

---

## 2. Files changed

| File | Edit | Lines |
|------|------|-------|
| `molmetal/molmetal_lam/lam_chem/reference_ligand_resolver.py` | NEW: 5-pocket lookup + resolver + dataclass + WARN fallback | ~330 |
| `molmetal/molmetal_lam/lam_chem/pt_metal_ligand_exchange.py` | NEW: 5 + 2 SMARTS library + AquaContext dataclass (Phase 2 + 3 deliverable) | ~190 |
| `molmetal/molmetal_lam/lam_chem/pt_click_compat.py` | Add `COMPAT_MATRIX_VERSION` + `strict_pt_ii_allows_metal_coordination()` helper (Phase 4) | ~30 |
| `molmetal/molmetal_lam/reactions/beta_reductions.py` | Import new SMARTS library + `available_smarts()` + `aqua_context()` accessor methods on MetalLigandExchange / AquaExchange classes | ~70 |
| `molmetal/scripts/r4_lambda_only_run.py` | Add 3 new CLI flags (`--use-pocket-conditioned-reference`, `--use-learned-prior`, `--pocket-boost-strength`); thread kwargs through `run_one_cell` + `run_sweep` + `main`; insert resolver path before metal_seed_from_pocket block; insert `learned_prior` search kwargs | ~150 |
| `molmetal/tests/test_wf_lambda_boost.py` | NEW: 21 unit tests covering all 4 phases | ~290 |

---

## 3. The reference_ligand_resolver design

### 3.1 Two-signal discriminator (PRIMARY name + FALLBACK slot contrast)

Same pattern as `pocket_derived_metal_seed`:

1. **Pocket-name whitelist (PRIMARY):**
   * `CA2` / `CARBONIC` → `POCKET_REFERENCE_LIGANDS["CA2"]`
   * `MMP` / `ZINCIN` → `POCKET_REFERENCE_LIGANDS["MMP2"]`
   * `HDAC` / `DEAC` → `POCKET_REFERENCE_LIGANDS["HDAC2"]`
   * `POCKET_007` → `POCKET_REFERENCE_LIGANDS["pocket_007"]`
   * `POCKET_011` → `POCKET_REFERENCE_LIGANDS["pocket_011"]`

2. **Slot-based contrast (FALLBACK):**
   * `pos >= 0.40 AND pos - neg >= 0.20` → CA2 bucket
   * `neg >= 0.15 AND neg - pos >= 0.05` → MMP2 bucket
   * otherwise → pocket_011 bucket

3. **Cisplatin legacy fallback (WARN):**
   * `pocket_features=None` → `[Pt]C#C` + `UserWarning`

### 3.2 Lookup table

| pocket_key | reference_name | chemistry_label | SMILES |
| --- | --- | --- | --- |
| `CA2` | `acetazolamide_seed` | `sulfonamide_arene` | `NS(=O)(=O)c1ccc(NC(=O)N)cc1` |
| `MMP2` | `marimastat_seed` | `hydroxamate_peptide` | `CC(C)CC(NC(=O)C(NC(=O)NO)C(C)C)C(=O)NO` |
| `HDAC2` | `coumarin_thiol_seed` | `thiol_coumarin` | `Sc1ccc2oc(=O)cc(C#C)c2c1` |
| `pocket_007` | `quinoline_alkyne_seed` | `quinoline_alkyne` | `C#Cc1ccnc2ccccc12` |
| `pocket_011` | `benzamide_imide_seed` | `benzamide_imide` | `C#Cc1ccc(NC(=O)NC(=O)c2ccccc2)cc1` |

### 3.3 ReferenceLigandRecord dataclass (frozen)

```python
@dataclass(frozen=True)
class ReferenceLigandRecord:
    pocket_key: str            # e.g. "CA2", "cisplatin_legacy"
    smiles: str                # the reference ligand SMILES
    chemistry_label: str       # e.g. "sulfonamide_arene"
    is_fallback: bool          # True when legacy cisplatin seed fired
    fallback_reason: Optional[str]  # e.g. "missing_pocket_features"
    elapsed_ms: float          # wall-clock resolver time (budget <50ms)
```

---

## 4. Test coverage (5 tests for Phase 1)

| Test | Asserts |
|------|---------|
| `test_five_distinct_pockets_return_five_distinct_smiles` | All 5 named pockets return distinct SMILES (cache-buster works) |
| `test_missing_pocket_features_falls_back_with_warn` | `None` input → `UserWarning` + cisplatin legacy + `is_fallback=True` |
| `test_resolution_under_50ms_per_pocket` | Each pocket resolves in <50 ms (production budget) |
| `test_slot_contrast_routes_to_CA2_bucket` | Slot 2 = 0.50, slot 3 = 0.10 → CA2 bucket via contrast fallback |
| `test_slot_contrast_routes_to_MMP2_bucket` | Slot 2 = 0.10, slot 3 = 0.20 → MMP2 bucket via contrast fallback |

---

## 5. Honest limitations + follow-ups

* **Lookup table is hand-curated**, not learned.  Future work: fit a
  small pocket → chemotype classifier on the 50-p training cohort
  (TODO-29 Phase-3J).
* **5 pockets only.**  CrossDocked100 has 100 pockets — the resolver
  will fall through to `pocket_011` (the default) for 95 of them.
  This is acceptable for Phase 1 because the *goal* is to break
  cache-collapse, not to discriminate all 100 pockets.  Follow-up
  (Phase 6): add per-pocket entries for the top-20 most common
  CrossDocked100 protein families.
* **WARN fallback emits per-cell** — in production 100×3=300 cells,
  the WARN log will fire ~95×3=285 times (once per non-named pocket).
  This is intentional (the WARN is the audit trail) but CI logs may
  get noisy.  Future work: aggregate WARNs at the sweep level.
