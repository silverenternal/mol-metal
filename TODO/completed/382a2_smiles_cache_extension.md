# Wf-382-A2 — SMILES featurization cache CPU extension

**Date**: 2026-09-30
**Status**: ✅ **SHIPPED — SCAFFOLD CPU EXTENSION** of Wf-320D chemprop v2 cache + Wf-328-W10 D-MPNN cache. 3 NEW tests PASS.
**Owner**: Hugo Lee + Wf-382-A2 ultracode agent
**Parent**: TODO-155 §3 (chemprop v2 cache pivot) + Wf-320D (`molmetal/scripts/wf320d_featurization_fix.py`) + Wf-328-W10 (`molmetal/models/dmpnn_cache.py`) + `TODO/pending/155_smiles_featurization_cache_extension.md` (this wave's pending doc).
**Skill alignment**: `party-twenty-spirit` §3.5 (every ETA + cache size number traces to Wf-320D §4.3 / Wf-328-W10 §0 MEASURED or in-repo file path).
**Constraint**: SCAFFOLD-only — full integration into `train_chemprop_v2_v3.py` + `train_dmpnn_multitask_wasserstein.py` deferred to Wf-380-B1 + Wf-380 trainer integration. NO GPU launches. NO paper/* touches.

---

## §0 TL;DR — three lines

1. **Wf-382-A2 SHIPPED the SMILES featurization cache CPU extension scaffold.** Three NEW artifacts on disk: (a) `molmetal/scripts/wf382a2_dmpnn_featurization_cache.py` ~265 LOC (thin forwarder wrapping the existing Wf-328-W10 D-MPNN cache + new `Wf382A2DMPNNCacheAdapter` with multiprocessing fan-out + CLI); (b) `molmetal/scripts/wf382a2_chemprop_v3_featurization_cache.py` ~410 LOC (NEW `ChempropV3FeaturizationCache` mirroring the chemprop v2 `PerEpochFeatureCache` API + `ChempropV3CacheStats` + CLI); (c) `molmetal/tests/test_wf382a2_featurization_cache.py` ~270 LOC (3 NEW tests).
2. **3/3 NEW tests PASS** in 2.09 s wall (CPU-only). `test_dmpnn_cache_hit_rate` + `test_chemprop_v3_cache_hit_rate` + `test_dmpnn_adapter_featurize_smiles_list`. All hit-rate assertions verified (`batch_hit_rate >= 0.66`, `hit_rate >= 0.66`, `smiles_hit_rate >= 0.5`).
3. **Pre-flight v3 92 PASS / 0 FAIL / 2 SKIP / 94 TOTAL PRESERVED at Wf-370-B1 baseline** (NOT re-run by Wf-382-A2 — docs+code+light-test only, 0 GPU launches per AGENTS.md §5.1). EXPECTED_GATE_COUNTS["all"] **94** PRESERVED at Wf-353-B1 baseline (no new pre-flight gates added by Wf-382-A2 — only 3 NEW tests + 2 NEW scripts + 1 INDEX sync + 1 pending + 1 completed).

---

## §1 Per-trainer status (post-Wf-382-A2)

| Trainer | Featurizer | Cache | Wf-382-A2 status | SCAFFOLD vs SHIPPED |
|---------|-----------|-------|------------------|----------------------|
| `train_chemprop_v2_v2_fastfeaturize.py` | `chemprop.v2.SimpleMoleculeMolGraphFeaturizer` (72/14) | `wf320d_featurization_fix.py::PerEpochFeatureCache` | **EXISTING** | SHIPPED per Wf-320D |
| `train_chemprop_v2_v3.py` | `chemprop.v3.SimpleMoleculeMolGraphFeaturizer` (72/14 default) | `wf382a2_chemprop_v3_featurization_cache.py::ChempropV3FeaturizationCache` | **NEW SCAFFOLD** | SCAFFOLD — integration to trainer DEFERRED to Wf-380-B1 |
| `train_dmpnn_multitask_wasserstein.py` | `molmetal.data.featurize.GraphFeaturizer` (39/6) | `dmpnn_cache.py::DMPNNFeaturizationCache` + `wf382a2_dmpnn_featurization_cache.py::Wf382A2DMPNNCacheAdapter` | **EXISTING + NEW forwarder** | forwarder SHIPPED; trainer integration DEFERRED to Wf-380 |

Honest framing: chemprop v2 cache was already SHIPPED pre-Wf-382-A2; chemprop v3 cache was missing on disk; D-MPNN cache was already SHIPPED but lacked a scripts/ forwarder + CLI. Wf-382-A2 = (a) chemprop v3 NEW SCAFFOLD, (b) D-MPNN thin forwarder + adapter + CLI. Both NEW artifacts are CPU-only and integrate cleanly into existing trainer paths via 5-LOC edits deferred to Wf-380.

---

## §2 Cache size per trainer (MEASURED / PROJECTED)

| Trainer | Tier-1 cache entries | Tier-2 cache entries | Hit-rate expected | Source |
|---------|---------------------:|---------------------:|-------------------|--------|
| chemprop v2 (Wf-320D) | up to n_canonical_smiles (~26801 MetalCytoToxDB) | up to n_batches/epoch (~75 × 50 epochs = 3750 unique keys) | `n_batch_hit / (n_batch_hit + n_batch_miss)` ≥ 0.96 in typical 50-epoch run (49 hits / 50 lookups per batch key) | Wf-320D §4.3 |
| chemprop v3 (Wf-382-A2 SCAFFOLD) | up to n_canonical_smiles (same as v2) | up to n_batches/epoch (same as v2) | **PROJECTED** ≥ 0.96 (NOT yet measured end-to-end) | Wf-382-A2 SCAFFOLD |
| D-MPNN (Wf-328-W10 + Wf-382-A2 forwarder) | up to n_canonical_smiles | up to n_batches/epoch (~75 × 50 = 3750 unique keys) | `batch_hit_rate` ≥ 0.96 | Wf-328-W10 §0 |

Per-trainer memory footprint (PROJECTED for chemprop v3, MEASURED for v2 + D-MPNN):
- chemprop v2 (Wf-320D): `PerEpochFeatureCache` reports ~3.5 MB peak for MetalCytoToxDB n=26801 (atom_fdim=72, bond_fdim=14) per Wf-320D §4.3.
- chemprop v3 (Wf-382-A2): **PROJECTED** same as v2 (same shape contract, same per-SMILES storage).
- D-MPNN (Wf-328-W10): `DMPNNFeaturizationCache` reports ~1.8 MB peak for n=26801 (atom_fdim=39, edge_fdim=6) per Wf-328-W10 §0.

---

## §3 Test status (3 NEW tests, all PASS)

`molmetal/tests/test_wf382a2_featurization_cache.py` (3 tests, ~270 LOC):

| Test | Wall | PASS / FAIL | Asserts |
|------|-----:|-------------|---------|
| `test_dmpnn_cache_hit_rate` | <1 s | **PASS** | `n_batch_hit == 32` (epochs 1+2 reuse epoch 0 builds); `batch_hit_rate >= 0.66`; `n_canonical_smiles == 10` |
| `test_chemprop_v3_cache_hit_rate` | <1 s | **PASS** | `n_hit == 32`; `hit_rate >= 0.66`; per-molecule featurisation contract (`V.ndim == 2`, `V.shape[1] == 72`, `E.shape[1] == 14`) |
| `test_dmpnn_adapter_featurize_smiles_list` | <1 s | **PASS** | First feed: `n_added == 10`; second feed: `n_added == 0`; cumulative `smiles_hit_rate >= 0.5` |

Total wall **2.09 s** per `/home/hugo/codes/mol-metal/.venv/bin/python -m pytest molmetal/tests/test_wf382a2_featurization_cache.py -v`.

The chemprop v3 cache test gracefully skips if chemprop v3 isn't installed (`pytest.skip("chemprop v3 not installed in this environment")`). In the project `.venv`, chemprop v3 IS installed (verified: `chemprop.featurizers.SimpleMoleculeMolGraphFeaturizer` importable), so the test runs to completion and PASSes.

---

## §4 Cross-references

| Reference | Path |
|-----------|------|
| **NEW pending doc** | `TODO/pending/155_smiles_featurization_cache_extension.md` (~150 LOC) |
| **NEW D-MPNN forwarder** | `molmetal/scripts/wf382a2_dmpnn_featurization_cache.py` (~265 LOC) |
| **NEW chemprop v3 cache** | `molmetal/scripts/wf382a2_chemprop_v3_featurization_cache.py` (~410 LOC) |
| **NEW tests** | `molmetal/tests/test_wf382a2_featurization_cache.py` (~270 LOC) |
| Wf-320D chemprop v2 cache (existing) | `molmetal/scripts/wf320d_featurization_fix.py` (587 LOC) |
| Wf-328-W10 D-MPNN cache (existing) | `molmetal/models/dmpnn_cache.py` (437 LOC) |
| TODO-155 §3 (chemprop v2 cache pivot) | `TODO/completed/155_pivot_to_chemprop_v2.md` §3 |
| Wf-326A profile.json finding | `molmetal/reports/wf326a_profile/profile.json` §3 |
| Pre-flight v3 baseline | `molmetal/reports/wf370b1_preflight.txt` (92/0/2/94 wall 152.53 s) |
| AGENTS.md §5.1 GPU grant + §5.4 Tier-2 markers | UNCHANGED |

---

## §5 Forward-looking gaps (DEFERRED to follow-up workflows)

1. **Full chemprop v3 cache integration into `train_chemprop_v2_v3.py`** — SCAFFOLD only here; replacing the per-batch `BatchMolGraph` construction path requires a CPU-only edit on the trainer + a regression test. DEFERRED to Wf-380-B1.
2. **D-MPNN cache integration into `train_dmpnn_multitask_wasserstein.py`** — the existing `DMPNNFeaturizationCache` is wired nowhere; needs a `--dmpnn-cache` CLI flag + regression test. DEFERRED to Wf-380 trainer integration.
3. **chemprop v3 fast featurizer production check** — `FastAtomFeaturizer` is OPT-IN only (per Wf-320D §0 warning: for production-size datasets ~30K rows, chemprop v3's default C/RDKit featurizer is faster due to per-atom Python attribute-lookup overhead). Default remains `use_fast_featurizer=False`.

---

## §6 Honest-negative caveats (per `party-twenty-spirit` §3.5)

1. **SCAFFOLD status — NOT end-to-end integrated.** The cache classes are constructed + exercised against a stub `batch_molgraph_ctor` in tests, but the trainers (`train_chemprop_v2_v3.py`, `train_dmpnn_multitask_wasserstein.py`) do NOT yet use the new caches. Full integration DEFERRED to Wf-380-B1 + Wf-380 trainer integration.
2. **PROJECTED vs MEASURED split.** chemprop v2 + D-MPNN cache savings are MEASURED per Wf-320D §4.3 + Wf-328-W10 §0; chemprop v3 cache savings are **PROJECTED** (mirror of v2 API; not yet MEASURED end-to-end on Blackwell sm_120). The +5-10× featurization speed claim is PROJECTED, not MEASURED.
3. **`FastAtomFeaturizer` OPT-IN** — inheriting Wf-320D's warning verbatim: for production-size datasets (~30K rows), chemprop v3's default C/RDKit featurizer is faster than the pure-Python fast featurizer due to per-atom Python attribute-lookup overhead. Default is `use_fast_featurizer=False`.
4. **Multiprocessing fan-out (`n_workers > 0`) NOT tested in unit tests** — `test_dmpnn_adapter_featurize_smiles_list` uses `n_workers=0` (sequential) for test reproducibility. Parallel path is exercised only by the CLI smoke (which is also CPU-only and not gated).
5. **Pre-flight v3 NOT re-run by Wf-382-A2** — docs+code+light-test only; 0 GPU launches; pre-flight state preserved at Wf-370-B1 baseline 92/0/2/94.
6. **NO `paper/*` touches** across the entire Wf-382-A2 wave per AGENTS.md §5.3 TODO-84 active.
7. **0 NEW pre-flight gates** — EXPECTED_GATE_COUNTS["all"] **94** PRESERVED at Wf-353-B1 baseline (no pre-flight wiring changes by Wf-382-A2).
