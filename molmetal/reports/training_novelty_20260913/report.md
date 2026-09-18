# CrossDocked training chemistry novelty — 2026-09-13

All 100,000 training SDFs and 100 test SDFs were present and valid (missing 0; invalid 0). The train-only index contains **8,765 unique canonical structures** from 100,000 instances, with 91,235 duplicate instances and 4,821 Murcko scaffolds (598 unique structures are acyclic). Each SDF has an individual SHA256 in the external inventory; the split and all index artifacts are hash-locked in `novelty.json`.

Against this complete indexed CrossDocked train chemistry, the **4 unique generated structures / all 70 saved generated instances** have no exact canonical graph overlap and no Murcko scaffold overlap. All four have nonempty scaffolds. This establishes novelty only relative to this declared dataset and canonicalization protocol; it does not establish novelty to AiZynth/REINVENT training data, the literature, or the chemical universe. It also does not resolve the very narrow generated diversity.

| Generated canonical SMILES | Instances | Nearest training Tanimoto | Nearest training canonical SMILES |
|---|---:|---:|---|
| `Cc1cnnn1CCOCCO` | 20 | 0.344828 | `OCCOCCO` |
| `NCc1cnnn1CCOCCO` | 20 | 0.322581 | `OCCOCCO` |
| `NCc1cnnn1Cc1ccccc1` | 10 | 0.400000 | `NCc1ccccc1` |
| `OCCOCCn1nncc1Cc1ccccc1` | 20 | 0.396226 | `O=c1[nH]c(=O)n(COCCO)cc1Cc1ccccc1` |

Unique-structure mean nearest similarity: 0.365909; instance-weighted mean: 0.361038. Candidate selection uses all_candidates where available; the original report has 70 saved candidates and declares 70 generated candidates, so all generated instances are covered here, including the seven without completed docking.

The official split has zero train/test ligand-path overlap, but **34/100 test instances and 32/96 unique test structures** exactly match training structures. Pocket/file splitting does not imply ligand-structure independence and this fact alone does not establish improper leakage for the original CrossDocked task. A ligand-novel generalization protocol must separately declare a structurally disjoint subgroup. The official 100-item test split is retained unchanged. In this first-10-pocket run, 14 generated instances belong to reference-graph-shared pockets and 56 to nonshared pockets; neither subgroup has generated graph/scaffold overlap, and both have mean nearest similarity 0.361038. This full-library overlap audit is distinct from any model experiment that explicitly excludes train/test graph overlap.

Protocol: RDKit 2026.03.6 sanitized SDF, exactly one record required; remove atom maps and ordinary explicit hydrogens while preserving stereochemistry/isotopes/charge/fragments. No salt stripping, neutralization or tautomer normalization. Exact graph equality uses isomeric canonical SMILES. Murcko scaffolds retain atom/bond types and ignore chirality; empty acyclic scaffolds are explicitly labelled. Morgan radius 2, 2048 bits, includeChirality=False. RDKit parsing/fingerprints run on CPU; chunked nearest-Tanimoto reductions ran on ROCm cuda:0, **gfx1101**, torch 2.14.0+rocm7.2, HIP 7.2.53211. Reference chunks have 4096 fingerprints, and exact ties choose the first stable structure ID.

Validation: 5 focused tests pass, including real GPU vs RDKit, canonical chemistry preservation, empty/invalid inputs, deterministic ties, train-only indexing and candidate denominators. An additional independent RDKit BulkTanimoto comparison against **all 8,765 training fingerprints** confirms all four nearest identities with maximum score error 9.62e-9 (`full_index_rdkit_validation.json`).

Large index: `/mnt/storage/data/molmetal/crossdocked/train_novelty_index_20260913/` (`manifest.json`, `index.sqlite`, `morgan_r2_2048.npy`, `source_inventory.jsonl`). Index build took 300.85 seconds with four CPU workers. GPU evaluation, including index hash checks, took 0.561 seconds; no CPU/GPU speedup claim is made.

```bash
uv run --no-sync python -m molmetal.scripts.training_set_novelty build \
  --split /mnt/storage/data/molmetal/crossdocked/split_by_name.pt \
  --root /mnt/storage/data/molmetal/crossdocked/extracted/crossdocked_pocket10 \
  --index /mnt/storage/data/molmetal/crossdocked/train_novelty_index_rebuild --workers 4
HIP_VISIBLE_DEVICES=0 uv run --no-sync python -m molmetal.scripts.training_set_novelty evaluate \
  --index /mnt/storage/data/molmetal/crossdocked/train_novelty_index_20260913 \
  --source molmetal/reports/r4_click_gpu_test10_seed3.json \
  --output molmetal/reports/training_novelty_20260913/novelty.json --device cuda:0
```

`runtime_source_hashes.json` records the initial build implementation; `final_runtime_source_hashes.json` records the evaluated implementation after explicit zero-count defaults, multiprocessing spawn, immutable array copying and reference-overlap subgroup reporting were added. These changes do not change the index chemistry.
