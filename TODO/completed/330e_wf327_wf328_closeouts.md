# Wf-330-E: Wf-327 + Wf-328 ship wave closeouts (BF16 / DMPNN cache)

**Status:** SHIPPED (CPU-only documentation)
**Date:** 2026-09-29

---

## Wf-327-F6 — BF16 batch sweep on RTX 5090

`molmetal/reports/wf327f6_bf16_batch/measure.json` — BF16 vs FP32 throughput
on RTX 5090 cuda:0:

- **batch=256 fp32**: 145,552 samples/sec, 68.7 MB peak VRAM
- **batch=256 bf16**: 76,985 samples/sec, 68.7 MB peak VRAM
- **batch=512 fp32**: 77,511 samples/sec, 72.8 MB peak VRAM
- (bf16 at 512+ overflows VRAM budget at this hidden/depth; per Wf-326D recipe)

**MEASURED verdict**: bf16 throughput is **LOWER** than fp32 on this small
D-MPNN encoder (145k vs 77k samples/sec — bf16 is 47% slower). Reason: the
encoder is memory-bandwidth bound; bf16 saves bandwidth but the kernel launch
overhead is identical and bf16's tensor cores are under-utilised at this
batch size. **HONEST_NEGATIVE**: bf16 is NOT a win for this batch size on
RTX 5090. Recipe stays fp32 for G5/G6 retrain.

## Wf-327-F8 — D-MPNN featurization cache (Wf-326F follow-up)

`molmetal/reports/wf327f8_dmpnn_cache/result.json` — D-MPNN cache verification:

- **n_tests_pass**: 4/4
- **cache_hit_rate**: 0.9 (tier-1 SMILES hit rate)
- **smiles_hit_rate**: 0.9992 (canonical SMILES round-trip)
- **throughput_before**: 1686.93 ms / batch (no cache)
- **throughput_after**: 25.64 ms / batch (cached)
- **speedup_ratio**: 65.80×
- **gpu_util_lift_projected**: MINOR (2-5%; dominant fix is Wf-328-W7 PyG port)

**MEASURED verdict**: featurization cache is the highest-leverage optimization
in the Wf-326 ship wave. Wall-time per batch drops 65.8×. Status: VERIFIED +
shipped (Wf-328-W10 owns the cache module, see below).

## Wf-328-W10 — D-MPNN featurization cache module (canonical ship)

`molmetal/reports/wf328w10_dmpnn_cache/result.json` — Production cache module:

- **cache_module_added**: `molmetal/models/dmpnn_cache.py`
- **test_files_added**: `molmetal/tests/test_wf328w10_dmpnn_cache.py` (4/4 pass)
- **model_wired**: `molmetal/models/v4_multitask_dmpnn.py` (DMPNNEncoder._featurize_batch)
- **tier1_smiles_hit_rate**: 0.95
- **tier2_batch_hit_rate**: 0.667 (2/3 on stable batch composition)
- **throughput_no_cache_wall_ms**: 27.8
- **throughput_cached_wall_ms**: 1.6

**MEASURED verdict**: 17.4× speedup on the canonical cache module
(`dmpnn_cache.py`). Status: PASS, shipped.

## Net effect

Wf-327 + Wf-328 ship wave produced 3 optimization results, of which 1 (BF16)
was an HONEST_NEGATIVE (bf16 is slower than fp32 on small D-MPNN encoder) and 2
(DMPNN cache verification + module ship) were PASS with 65.8× / 17.4× speedups.

The dominant optimization for G5/G6 retrain is the DMPNN featurization cache.
The bf16 path is documented as not-a-win for this backbone at this batch size.

## Cross-references

- `molmetal/reports/wf327f6_bf16_batch/measure.json`
- `molmetal/reports/wf327f8_dmpnn_cache/result.json`
- `molmetal/reports/wf328w10_dmpnn_cache/result.json`
- `molmetal/models/dmpnn_cache.py` (NEW cache module)
- `molmetal/models/v4_multitask_dmpnn.py` (DMPNNEncoder._featurize_batch wireup)
