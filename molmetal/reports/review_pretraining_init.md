# Review: tmQM pretraining init for metal-cytotoxicity transfer

## Pretraining signals actually available
`dmpnn_tmqm_pretrained.pt` (2.5 MB) ships only `encoder_state_dict`
(`atom_embed`, `edge_embed`, 3x `edge_mlp`, 3x `gru_updates`, `atom_to_edge`,
`readout_mlp`) — there is **no saved regression head**. So today's "head
swap" silently maps a model that was trained jointly with a z-scored
coord-number + Wiberg-BO MLP (108 k complexes, z-score MSE) onto a random
pIC50 / active-class head. Coordination number is informative but
mass-action thermodynamics, not cytotoxicity. Net transfer value:
**moderate-low**; mostly conveys ligand-field topology, not bioactivity.

## SOTA precedent for head choice
- **Chemprop** (Yang et al., *J. Chem. Inf. Model.* 2019): pretrain on
  ~1.5 M PubChem logP/solubility, **discard the readout**, fine-tune end-
  to-end with a small LR (1e-4) on the new head. Frozen-encoder fine-tune
  is *not* the recommended default.
- **Uni-Mol** (Zhou et al., *arXiv:2301.01069*, 2023 ICLR): pretrains
  with masked-atom + 3D-position + contrastive pair objectives on 209 M
  conformers, then fine-tunes the **whole encoder** + head; their
  ablation shows >3% AUC drop when only the head is trained.
- **GROVER** (Rong et al., *NeurIPS 2020*): contextual-attribute
  prediction (atom/bond vocab, FG) is treated as a *regularizer* during
  fine-tune, not discarded. Reported +6% mean over 11 benchmarks.

## Diagnosis of current setup
(a) Full frozen encoder + random `pic50_head`/`active_head` discards ~99 %
of the pretraining signal — the only benefit is a better initial
embedding for atoms. (b) No contrastive term means the encoder is free
to drift on canonical-SMILES augmentation. (c) Single ablation run, no
val-AUC curve vs alternatives.

## Improvement plan (concrete, minimal)
1. **LLRD** (Howard & Ruder, *ACL 2018* ULMFiT): `lr_top = 1e-5`,
   `lr_head = 1e-3`, decay 0.95/layer.
2. **Contrastive auxiliary**: InfoNCE on RDKit canonical-SMILES
   augmentations (μ=0.05) — GROVER-style regulariser.
3. **Three-way ablation** logged to JSON: `frozen`, `llrd`, `full`,
   plotted as val-AUC vs epoch.

## Expected gain
Following Uni-Mol / GROVER tables, LLRD + contrastive should lift Ru
val-AUC by ~3–5 pts over frozen-init and shorten convergence ~2× vs
full-unfreeze (less over-fit on the 7 k ligand-dedup split).

## References
1. Yang et al., *J. Chem. Inf. Model.* 2019 — Chemprop (D-MPNN).
2. Zhou et al., *ICLR 2023*, arXiv:2301.01069 — Uni-Mol.
3. Rong et al., *NeurIPS 2020*, arXiv:2007.02835 — GROVER.
4. Howard & Ruder, *ACL 2018* — ULMFiT (LLRD origin).
