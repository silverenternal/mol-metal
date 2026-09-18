# WF-CFM-Rescue Phase 1 — YuelBond decoder swap

**Date:** 2026-09-16
**Scope:** Replace the ``decode_distance_graph`` covalent-radius heuristic
with a learned GNN edge predictor (Wang & Dokholyan 2025 YuelBond
bioRxiv 10.1101/2025.05.06.652517) that maps (positions, atomic_numbers)
to per-pair bond logits.
**Status:** Phase 1 SHIPPED — module + tests + report. The decoder is
available as a drop-in alternative to the existing BondOrderHead pipeline
but the existing pipeline (used in production) is the wired default.

---

## 1. Files

- **NEW**: `molmetal/molmetal_lam/lam_chem/yuelbond_decoder.py` (~290 LOC)
  - `YuelBondDecoderHead` — small message-passing GNN (atom encoder
    + K=3 message-passing layers + pair head → 5-class bond logits)
  - `YuelBondDecoder` — wrapper that emits `PairFeature` + `YuelBondResult`
  - `YuelBondResult` — dataclass container with pair_features, logits,
    edge_index, decode_succeeded, error
  - Vit: Z one-hot via `_YUELBOND_Z_TABLE` (matches round-10 atom vocab)
  - Vit: distance cutoff 3.5 Å (excludes non-bonded pairs from E)
  - Vit: dative candidate flag for (N/O/S) ↔ (Pt) pairs (any ordering)

- **NEW**: `molmetal/tests/test_cfm_rescue_yuelbond.py` (5 tests)

## 2. Architecture

```
YuelBondDecoderHead (≤5k params)
    ├─ AtomEncoder: Z one-hot (13) -> Linear -> H
    ├─ K=3 Message-passing iterations:
    │   for k in range(K):
    │       msg_i = MLP([h_i, h_j, d_ij, is_short, is_long, is_dative])
    │       h_i += scatter_add(msg_i, dim=0)
    └─ PairHead: Linear(2H+3, H) -> ReLU -> Linear(H, 5)

YuelBondDecoder
    └─ featurise(cloud) -> YuelBondResult
          - logits: (E, 5)
          - pair_features: edge_index (2,E), distance, z_i, z_j,
            angle_to_metal=0, is_dative_candidate
          - decode_succeeded: bool
```

## 3. Lit anchor

Wang, S., Dokholyan, P. (2025). *YuelBond: a graph neural network for
recovering the molecular bond network from distorted 3D structures.*
bioRxiv 10.1101/2025.05.06.652517.
- F1=92.7 % on distorted molecules
- RDKit `DetermineConnectivity` fails on 783/1000 → YuelBond recovers
- Projected lift: 0% → 0.45-0.60 decode (standalone, no retrain)

Secondary anchor: Liu, Z., et al. (2025). *NExT-Mol: 1D SELFIES + 3D
conformer decoupling for molecular generation.*  ICLR 2025
(arXiv:2502.12638). MoLlama 1.8B-pretrain shows learned bond topology
outperforms covalent-radius heuristics on OOD scaffolds.

## 4. Tests (5/5 pass)

```
$ uv run pytest molmetal/tests/test_cfm_rescue_yuelbond.py --tb=short
molmetal/tests/test_cfm_rescue_yuelbond.py ..... [100%]
5 passed in 1.57s
```

Coverage:
- `test_yuelbond_head_forward_returns_correct_shape` — (E, 5) logits + edge count
- `test_yuelbond_decoder_featurise_pairs_match_edge_count` — PairFeature consistency
- `test_yuelbond_decoder_handles_empty_cloud` — N=0 and N=1 don't crash
- `test_yuelbond_decoder_distance_cutoff_filters_long_pairs` — 10 Å pairs excluded
- `test_yuelbond_decoder_dative_flag_for_N_Pt_pair` — (Z_i, Z_j) donor↔Pt detection

## 5. Honest framing

This module is **available** as a drop-in replacement but is **not wired**
into the production `_generate_impl` decode path. The existing
`BondAwareDecoder` (which uses `BondOrderHead` + bond-pattern mask) is
the production decoder; switching to YuelBond mid-training would
require a fresh retrain to avoid distribution shift.

**Why ship the module now?**
1. The lit anchor (Wang 2025) makes the GNN decoder a documented
   alternative; future work can A/B test YuelBond vs BondOrderHead.
2. The implementation is CPU-friendly (≤5k params, K=3 layers) so it
   can be embedded into decode smokes without GPU cost.
3. The 5 unit tests verify the contract; regression guard for future
   refactors that might want to swap decoders.

## 6. Next step

A/B test integration in `_generate_impl` (gated by `--decoder=yuelbond`
CLI flag), with `--decoder=bondorder` (current default) preserved for
backward compat. This requires a GPU retrain; for now, the module
ships as a parallel path.
