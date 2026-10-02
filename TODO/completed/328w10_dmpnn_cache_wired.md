# Wf-328-W10: D-MPNN featurization cache wired

**Wf:** Wf-328-W10 (Wf-326F follow-up)
**Date:** 2026-09-29
**Status:** SHIPPED — all 4 tests pass

---

## §1 Goal

Per Wf-326F follow-up, implement a two-tier featurization cache for the
D-MPNN backbone:

- **Tier 1 (per-SMILES)**: pre-featurise each SMILES exactly once via
  `GraphFeaturizer`, cache `(atom_features, edge_index, edge_attr)` per
  canonical SMILES.
- **Tier 2 (per-batch)**: cache padded batched tensors per `batch_idx`
  for cross-epoch reuse (padded shape is invariant across epochs for
  stable batch composition).

---

## §2 What was added

| File | Purpose |
|---|---|
| `molmetal/models/dmpnn_cache.py` (NEW) | `DMPNNFeaturizationCache` + `DMPNNCacheStats` + `cached_featurize_batch` helper |
| `molmetal/models/__init__.py` (MODIFIED) | exports `DMPNNFeaturizationCache`, `DMPNNCacheStats`, `cached_featurize_batch` |
| `molmetal/models/v4_multitask_dmpnn.py` (MODIFIED) | `DMPNNEncoder._featurize_batch` accepts optional `cache` + `batch_idx_key` params |
| `molmetal/tests/test_wf328w10_dmpnn_cache.py` (NEW) | 4 tests (hit/miss, per-epoch reset, integration, speedup benchmark) |
| `molmetal/reports/wf328w10_dmpnn_cache/result.json` (NEW) | MEASURED metrics + honest framing |

---

## §3 MEASURED results

| Metric | Value |
|---|---:|
| Tests pass | 4 / 4 |
| Tier-1 SMILES hit rate | 95.0% |
| Tier-2 batch hit rate | 66.7% (2/3 epochs hit on stable batch composition) |
| Throughput speedup (cached vs not) | **16.83x** on the 50-mol × 5-batch × 3-epoch workload |
| Pre-cache wall time | 27.8 ms |
| Post-cache wall time | 1.6 ms |

---

## §4 Honest framing

Per Wf-326A `profile.json`:

- Featurization is **MINOR** (CPU collate per batch ~7ms vs forward/backward ~2500ms).
- The main bottleneck is the **per-mol Python loop** in `DirectedMPNN.forward` — that fix is **Wf-328-W7 PyG port**, NOT this module.
- This cache is a **PROJECTED 2-5% end-to-end speedup** on the G6 full 5×5×50 sweep (projected, not measured at sweep scale).
- The 16.83x speedup measured here is the **CPU featurization-only** ratio; the end-to-end training ratio will be much smaller because featurization is not the dominant cost.

The cache is **safe** to enable because:

1. Padded batch shape is invariant across epochs for stable batch composition.
2. Pre-computed tensors do not depend on model weights (no gradient staleness risk).
3. Tier-1 cache uses canonical SMILES as key (deduplicates repeats across epochs).
4. Fallback path (no `cache` param) preserves bit-exact original behaviour.

---

## §5 Integration

`DMPNNEncoder._featurize_batch` has two new optional parameters:

- `cache: Optional[DMPNNFeaturizationCache] = None` — when provided,
  uses tier-1 cache for per-SMILES featurization.
- `batch_idx_key: Optional[int] = None` — when provided AND `cache` is
  set, uses tier-2 cache for the full padded batch.

```python
from molmetal.models.dmpnn_cache import DMPNNFeaturizationCache

cache = DMPNNFeaturizationCache()
cache.precompute_smiles(unique_smiles_list)  # tier 1 warm-up

for epoch in range(n_epochs):
    cache.begin_epoch(epoch)
    for batch_idx, batch_smiles in enumerate(dataloader):
        h_atom, edge_index, edge_attr, batch_idx_t, atom_mask = (
            encoder._featurize_batch(
                batch_smiles, device, cache=cache, batch_idx_key=batch_idx
            )
        )
```

For callers that don't pass `batch_idx_key`, the tier-1 cache still
applies (per-SMILES featurization is reused).

---

## §6 Cross-refs

- Wf-326F: parent finding (chemprop v2 cache is useful; D-MPNN cache
  would mirror the same idea).
- Wf-326A: profile.json confirms featurization is MINOR.
- Wf-156: pre-featurize-once (chemprop v2 side) — D-MPNN tier-1 is the
  D-MPNN analogue.
- Wf-320D: chemprop v2 `PerEpochFeatureCache` (model template).
- Wf-328-W7: PyG port — main bottleneck fix (separate workflow).

---

**END OF Wf-328-W10 ship record** — Owner Hugo Lee; cross-ref
`party-twenty-spirit` §3.5.
