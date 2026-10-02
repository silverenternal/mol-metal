# TODO-155 — SMILES featurization cache CPU extension

**Date**: 2026-09-30
**Status**: ✅ **WIRE-IN COMPLETE (Wf-409-A2 2026-09-30)** — chemprop v3 cache SCAFFOLD SHIPPED by Wf-382-A2; full CFM-trainer-primary wire-in SHIPPED by Wf-409-A2 (`--cfm-chemprop-v3-cache` flag + module-level holder + set/get API + tier-1 + tier-2 lookup inside `train_one_epoch` + main() init). chemprop v2/v3 trainer wire-in SHIPPED by Wf-392-A2 + Wf-404-A4. D-MPNN trainer wire-in pending (chemprop_v2 cache SCAFFOLD + chemprop_v3 cache SCAFFOLD + Wf-328-W10 D-MPNN cache; trainer wire-in deferred to Wf-381+ MEASURED wire-in).
**Owner**: Hugo Lee + ultracode agents (Wf-382-A2 + Wf-392-A2 + Wf-404-A4 + Wf-409-A2)
**Parent**: TODO-155 §3 (chemprop v2 featurization cache pivot) + Wf-320D (`molmetal/scripts/wf320d_featurization_fix.py`) + Wf-328-W10 (`molmetal/models/dmpnn_cache.py`).
**Skill alignment**: `party-twenty-spirit` §3.5 (every ETA + cache size number traces to Wf-320D §4.3 / Wf-328-W10 §0 MEASURED or in-repo file path).
**Constraint**: SCAFFOLD-only — full integration into `train_dmpnn_multitask_wasserstein.py` deferred to Wf-381+ MEASURED wire-in. NO GPU launches. NO paper/* touches.

---

## §0 TL;DR — three lines

1. **Wf-382-A2 = SCAFFOLD CPU extension of the existing Wf-320D chemprop v2 cache + Wf-328-W10 D-MPNN cache.** Adds (a) `molmetal/scripts/wf382a2_dmpnn_featurization_cache.py` (~200 LOC) — thin forwarder that wraps `molmetal.models.dmpnn_cache.DMPNNFeaturizationCache` + adds Wf-382 CLI; (b) `molmetal/scripts/wf382a2_chemprop_v3_featurization_cache.py` (~200 LOC) — NEW `ChempropV3FeaturizationCache` class that mirrors the chemprop v2 cache API (Tier-1 per-SMILES + Tier-2 per-batch BatchMolGraph) for chemprop v3 backbone.
2. **Cache extension SCAFFOLD — full integration deferred to Wf-380-B1 + Wf-380 trainer integration.** Projected impact (CPU-only scaffold, not yet MEASURED end-to-end): +5-10× featurization speed on cross-epoch reuse (per Wf-320D §0 + Wf-328-W10 §0 MEASURED baseline).
3. **Pre-flight v3 92 PASS / 0 FAIL / 2 SKIP / 94 TOTAL PRESERVED at Wf-370-B1 baseline.** Per Wf-370-A3 §7 + Wf-370-B1 §22.64.2 honest-negative baseline; Wf-382-A2 is docs+code+light-test only, no pre-flight gate additions, no GPU launches.

---

## §1 Per-trainer status

| Trainer | Featurizer | Existing cache | Wf-382-A2 status | Integration target |
|---------|-----------|----------------|------------------|---------------------|
| `train_chemprop_v2_v2_fastfeaturize.py` | `chemprop.v2.SimpleMoleculeMolGraphFeaturizer` (atom 72 / bond 14) | `molmetal/scripts/wf320d_featurization_fix.py::PerEpochFeatureCache` (587 LOC) | **EXISTING** | already wired in Wf-320D |
| `train_chemprop_v2_v3.py` | `chemprop.v3.SimpleMoleculeMolGraphFeaturizer` (atom 72 / bond 14 by default; configurable) | none on disk | **NEW Wf-382-A2 SCAFFOLD** (`ChempropV3FeaturizationCache`) | deferred to Wf-380-B1 + Wf-380 trainer integration |
| `train_dmpnn_multitask_wasserstein.py` | `molmetal.data.featurize.GraphFeaturizer` (atom 39 / edge 6) | `molmetal/models/dmpnn_cache.py::DMPNNFeaturizationCache` (437 LOC, Wf-328-W10) | **EXISTING** — Wf-382-A2 adds `molmetal/scripts/wf382a2_dmpnn_featurization_cache.py` thin forwarder + Wf-382 CLI | deferred to Wf-380 trainer integration |

Honest framing:
- chemprop v2 cache: SHIPPED per Wf-320D.
- chemprop v3 cache: NEW SCAFFOLD via Wf-382-A2.
- D-MPNN cache: SHIPPED per Wf-328-W10; Wf-382-A2 = thin forwarder (re-export + CLI) so the scripts/ namespace is consistent.

---

## §2 Cache extension design

### §2.1 D-MPNN thin forwarder (`molmetal/scripts/wf382a2_dmpnn_featurization_cache.py`)

NEW file (~200 LOC). Re-exports `molmetal.models.dmpnn_cache.DMPNNFeaturizationCache` + adds:
- `Wf382A2DMPNNCacheAdapter` class that wraps the underlying cache and adds a project-specific helper `featurize_smiles_list(smiles, batch_size=64, n_workers=4)` that fans out featurization across CPU cores (per Wf-326A profile — featurization is MINOR bottleneck but parallelisation still saves wall time on cohort smoke).
- CLI entry point: `python molmetal/scripts/wf382a2_dmpnn_featurization_cache.py --smiles ... --out ...` — emits JSON with `cache_info` + per-tier stats.
- `__all__ = ["DMPNNFeaturizationCache", "Wf382A2DMPNNCacheAdapter", "cached_featurize_batch"]`.

### §2.2 chemprop v3 NEW cache (`molmetal/scripts/wf382a2_chemprop_v3_featurization_cache.py`)

NEW file (~200 LOC). Implements:
- `ChempropV3CacheStats` dataclass (mirrors chemprop v2 `CacheStats`): `n_hit`, `n_miss`, `n_epoch_resets`, `build_wall_s`, `lookup_wall_s` + `hit_rate` property + `to_dict()`.
- `ChempropV3FeaturizationCache` class with **two-tier** cache:
  - **Tier 1 — per-SMILES**: `precompute_smiles(smiles_list, atom_featurizer, bond_featurizer)` — canonicalises SMILES via `Chem.MolFromSmiles` + runs chemprop v3 `SimpleMoleculeMolGraphFeaturizer` (default) OR `FastAtomFeaturizer`/`FastBondFeaturizer` (opt-in `use_fast_featurizer=True`, inherits from `molmetal.scripts.wf320d_featurization_fix`). Stores `{"V", "E", "edge_index", "rev_edge_index"}` per canonical SMILES.
  - **Tier 2 — per-batch**: `begin_epoch(epoch_idx)` (no-op batch-tier rebuild — BatchMolGraph doesn't depend on model weights) + `get_or_build_batch(epoch_idx, batch_idx, smiles_batch, device, batch_molgraph_ctor)` — same contract as `PerEpochFeatureCache.get_or_build_batch`.
- CLI entry point: `python molmetal/scripts/wf382a2_chemprop_v3_featurization_cache.py --smiles ... --out ...` — emits JSON with `cache_info` + per-tier stats.

---

## §3 Test status

NEW `molmetal/tests/test_wf382a2_featurization_cache.py` (~150 LOC) — 2 test functions:
- `test_dmpnn_cache_hit_rate` (~30 lines): builds cache with 50 SMILES + 3 epochs × 16 batches each; asserts `n_batch_hit == 32` (epochs 2 + 3 reuse epoch 1 builds); asserts `batch_hit_rate >= 0.66`.
- `test_chemprop_v3_cache_hit_rate` (~30 lines): same protocol against `ChempropV3FeaturizationCache` with stub `batch_molgraph_ctor`; asserts `n_hit == 32` + `hit_rate >= 0.66`.

Test wall PROJECTED ~1-2 s CPU-only. RUN target: `pytest molmetal/tests/test_wf382a2_featurization_cache.py -v`.

---

## §4 Cross-references

| Reference | Path / Section |
|-----------|----------------|
| Wf-320D chemprop v2 cache (existing) | `molmetal/scripts/wf320d_featurization_fix.py` (587 LOC, full chemprop v2 cache) |
| Wf-328-W10 D-MPNN cache (existing) | `molmetal/models/dmpnn_cache.py` (437 LOC) |
| Wf-326A profile.json finding (cache is MINOR bottleneck) | `molmetal/reports/wf326a_profile/profile.json` §3 |
| TODO-155 §3 (chemprop v2 cache pivot) | `TODO/completed/155_pivot_to_chemprop_v2.md` §3 |
| Wf-380 trainer integration follow-up (DEFERRED) | `TODO/completed/380a1_consistency.md` §3 (Wf-380-A1 = trainer consistency unification) |
| Pre-flight v3 baseline | `molmetal/reports/wf370b1_preflight.txt` (92/0/2/94 wall 152.53 s) |

---

## §5 Forward-looking gaps (NEW ultracode follow-up)

1. **Full chemprop v3 cache integration into `train_chemprop_v2_v3.py`** — SCAFFOLD only here; replacing chemprop v3's per-batch `BatchMolGraph` construction path requires a CPU-only edit on the trainer + a regression test. DEFERRED to Wf-380-B1.
2. **D-MPNN cache integration into `train_dmpnn_multitask_wasserstein.py`** — the existing `DMPNNFeaturizationCache` is wired nowhere; needs a `--dmpnn-cache` CLI flag + a regression test. DEFERRED to Wf-380 trainer integration.
3. **chemprop v3 fast featurizer production check** — `FastAtomFeaturizer` is OPT-IN only (per Wf-320D §0 warning: for production-size datasets ~30K rows, chemprop v3's default C/RDKit featurizer is faster due to per-atom Python attribute-lookup overhead). Default remains `use_fast_featurizer=False`.