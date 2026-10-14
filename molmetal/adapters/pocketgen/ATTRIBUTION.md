# PocketGen Adapter — Attribution & Modifications

This directory contains a vendored subset of the **PocketGen** project
(Zhang et al., 2024, Nature Machine Intelligence), used in mol-metal as a
lightweight protein-pocket representation encoder for the CFM (Conditional
Flow Matching) pocket-conditioning slot (`c`).

## Upstream

- **Project**: PocketGen — *Accurate Protein Pocket Design with a Bilevel
  Graph Transformer and an n-gram Decoding Scheme*
- **Repository**: https://github.com/zaixizhang/PocketGen
- **Paper**: https://www.nature.com/articles/s42256-024-00901-8
- **License**: MIT (see `LICENSE` for full text)
- **Copyright**: (c) 2024 zaixizhang

## What we keep (vendored subset)

Encoder-only — the generation head and chemistry pipeline are intentionally
excluded. Files kept:

| File | Purpose | LOC |
|---|---|---:|
| `models/encoders/cftfm.py` | BilevelEncoder + HierEncoder + GET (3 encoder classes) | 1304 |
| `models/encoders/gnn.py` | GNN_graphpred + MLP + WeightGNN | 540 |
| `models/encoders/get.py` | GET baseline encoder (kept for permutation) | 542 |
| `models/encoders/radial_basis.py` | RadialBasis | 223 |
| `models/encoders/tools.py` | block-edge utilities | 449 |
| `models/encoders/__init__.py` | get_encoder() factory | 38 |
| `models/common.py` | GaussianSmearing + ShiftedSoftplus | 323 |
| `models/protein_features.py` | per-residue feature extraction | ~400 |
| `models/esm2adapter.py` | ESM-2 amino-acid prior adapter | 358 |

## What we remove (not vendored)

| Upstream path | Reason for exclusion |
|---|---|
| `models/PD.py` | full decoder + sampling head; we use encoder only |
| `models/data.py` | their dataloader; we have `crossdocked_pool.py` |
| `models/esmadapter.py` | older ESM; superseded by esm2adapter |
| `utils/` (23 files) | chemistry + docking + relaxation (PyRosetta) |
| `data_preparation/` | pocket extraction; we have `crossdocked_pool` |
| `evaluation/`, `examples/`, `assets/` | not relevant to encoder use |
| `generate_new.py`, `train_recycle.py` | upstream training pipeline |
| `pocketgen.yaml`, `configs/` | upstream configs |
| `.git/`, `.idea/`, `__pycache__/` | upstream build artefacts |

## Modifications (mol-metal applied)

The vendored files are **kept verbatim** at the time of import. Any mol-metal
adaptation happens in wrapper code:

- `molmetal/adapters/pocketgen/__init__.py` (NEW, written by us) — exposes
  `BilevelEncoder` as the canonical pocket-representation encoder and
  documents the wrap-into-CFM contract (see docstring of
  `BilevelEncoder.extract_embedding`).
- The `pocket_embed_dim` adapter (when PocketGen's `hidden_channels=128`
  does not match the CFM velocity-field `hidden_dim=128`) is added in the
  wrapper layer, not by editing PocketGen source.
- Per-layer injection into the EGNN velocity field is handled by
  `molmetal/adapters/egnn_extensions.py::PerLayerPocketInjector`, which is
  pure mol-metal code (not a PocketGen edit).

## License compliance

PocketGen is MIT-licensed. We:

1. Retain the upstream `LICENSE` file unchanged in this directory.
2. Cite the paper in any publication that uses this adapter
   (Zhang et al., 2024, Nat. Mach. Intell.).
3. Make clear in code/docstrings that this code originates upstream
   (e.g. `# Upstream: PocketGen/zaixizhang, MIT, 2024` header on first
   import shim).

The MIT license permits commercial use with attribution; no other
restrictions apply. Compare with `DecompDiff` (CC BY-NC 4.0) which
**would block** commercial metallodrug IP — we did not select that
project for this reason (see Wf-A30 plan §Top-3 ranking).

## Out of scope (deferred)

- Loading the pretrained PocketGen weights (~250 MB). The wrapper exposes
  `BilevelEncoder.from_pretrained(checkpoint_path)` but the download /
  vendor of weights is deferred to Wf-A32 wire-in. The encoder can be
  trained from scratch on CrossDocked 2020 if we want to avoid the
  upstream checkpoint.
- P2Rank upstream pocket detection (separate adapter
  `molmetal/adapters/p2rank_wrapper.py` — Wf-A33).
- ESM-2 weight download (~650 MB for esm2_t33_650M_UR50D). The adapter
  accepts `esm_model_name` and supports both `'esm2_t6_8M_UR50D'`
  (lightweight, 8 M params) and the full `'esm2_t33_650M_UR50D'`.
