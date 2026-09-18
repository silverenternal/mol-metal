# Larger real CrossDocked CFG control

Runtime: LipmanFlowMatching_v1. All 96 requested samples had finite coordinates; no sample passed full graph decoding. This is not completed molecular-quality or Vina validation.

32 distinct actual 19-heavy-atom C/N/O/F training ligands, with official test-set canonical overlap excluded and invalid coordinate pairings recorded. Each seed (42, 0, 1234) trains a shared CFG1/CFG2 checkpoint for 2,000 updates: hidden32, two layers, batch2, learning rate 0.0001. Tests use exact held-out pairs test_001/test_002, eight samples per CFG/pocket/seed and 64 Euler steps. Encoder pockets are the nearest 64 atoms, ligand-centred and scaled by a train-only statistic. These inputs assume a known binding site.

| Outcome | Count |
|---|---:|
| Requested | 96 |
| Finite coordinates | 96 |
| Connected sanitized decoded graph | 0 |
| Docked / PB pass | 0 / 0 |
| disconnected_distance_graph | 90 |
| atom_outside_training_vocabulary | 6 |

Atom identities are masked during training and unconstrained sampling, with cross-entropy supervising real targets; no output vocabulary mask or atom deletion is applied. The fixed distance graph decoder and its sanitization rules are unchanged from the previous experiments. There is no learned bond-order head. Every rejected output retains all raw coordinates and identities. No Vina or pose quality effect can be estimated when nothing reaches docking.

This expanded training is a diagnostic, not a representative full CrossDocked training regime. It changes training data/capacity/learning rate jointly compared with the eight-pair diagnostic, so that comparison cannot identify a single causal training improvement. The old and v2 32-pair runs use the same protocol except the explicitly versioned velocity architecture and have independently fitted checkpoints. Within each run, CFG1/CFG2 uses an identical checkpoint and initial random seed.

The v2 architecture replaces the non-equivariant H-to-3 projection with a scalar gate multiplying relative position, plus bounded degree-normalized equivariant edge vectors. No sampled coordinate is clamped. Old v1 velocity checkpoints are structurally incompatible and require retraining.

All data identities, runtime/checkpoint/source hashes, common seed losses, raw outputs and failure denominators: [report.json](report.json).
