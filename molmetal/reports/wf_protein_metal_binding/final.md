# Metric 18 — Protein-Metal Binding prediction head — Final Report

**Date:** 2026-09-25
**Status:** 🟡 Shipped (baseline verified, AUC ≥ 0.85 target **not yet** met)
**Verdict:** `baseline` (mean test AUC = 0.7436 ± 0.0092 over 3 seeds × 50 epochs)
**Author:** Ultracode pass #1 (parallel-line agent for R17 metric closure)

---

## 1. TL;DR

We implemented the **multimodal protein-metal binding prediction head**
described in TODO §6.4 / METRICS_TARGETS §2 (Metric 18). The architecture
fuses four streams — 1D cell-line embedding (protein surrogate), metal-type
embedding, D-MPNN ligand encoder, optional 3D pocket encoder — through a
2-layer MLP head with focal loss.

A first 3-seed × 50-epoch × 5,257-row scaffold-split run achieves
**test AUC = 0.7436 ± 0.0092** (per-seed: 0.7562, 0.7344, 0.7402). The
target of **0.85 was not met** in this pass — this places us in the
`baseline` verdict band of METRICS_TARGETS §2's verdict gate:

- ❌ < 0.70: below baseline (we are not here)
- ⚠️ 0.70–0.85: **baseline band** ← **here**
- ✅ 0.85–0.92: good (target)
- 🌟 ≥ 0.92: matches Tier 1 SOTA (Multimodal Protein-Metal Binding 2025)

The val-AUC trajectory reaches 0.85 in 1 of 3 seeds (best epoch), suggesting
the model has the capacity but the test-set scaffold split is the limiting
factor. We discuss this in §6.

## 2. 2025 paper comparison

We cite the Tier 1 reference **`multimodal_protein_metal_2025`** as the
state-of-the-art target. The published paper achieves **AUC > 0.92** on
*true* protein-metal binding labels (individual proteins with their own
sequences). Our head is operating on MetalCytoToxDB-derived rows where the
"protein" is the cell-line identity (the experimental design records
cell-line response, not individual protein–ligand binding affinity), which
is a less informative supervision signal. This is the largest source of
the gap to 0.92 — see §6 for a candid discussion.

Architecturally we reuse the same multimodal-fusion structure:

| stream | 2025 paper | Mol-Metal (ours) |
|---|---|---|
| 1D protein | ESM-2 embeddings | nn.Embedding on cell-line id |
| 3D pocket | EGNN / equivariant | PointNet (placeholder, 3D coords not in our CSV) |
| Ligand | D-MPNN | D-MPNN (compact, 128-dim × 3-depth) |
| Metal | one-hot + chemistry feats | nn.Embedding |
| Fusion | concat → 2-layer MLP | concat → 2-layer MLP |
| Loss | weighted BCE | focal loss (γ=2, α=auto) |

The pocket stream is **disabled by default** (`pocket_n_pseudo=0`) because
the public MetalCytoToxDB release does not include per-pocket atomic
coordinates — to activate the pocket stream a CSV with `(x, y, z)` per atom
is needed (easy to add once a 3D-pocket backfill lands).

## 3. Multimodal architecture diagram

```
                   ┌────────────────────────────┐
                   │  cell_line_id (LongTensor) │ ─┐
                   └────────────────────────────┘  │
                                                  │
                                                  ▼
                                          [CellLineEncoder]
                                                ↓
                                       32-d embedding
                                                  ↓
                                                   \
                                                    \
   ┌───────────────────────────────┐                \
   │  metal_id (LongTensor)        │ ─┐              \
   └───────────────────────────────┘  │               \
                                      ▼                \
                              [MetalEncoder]            \
                                    ↓                    \
                              16-d embedding              \
                                    ↓                      \
                                     \                      \
   ┌──────────────────────────────────────────────┐         \
   │  smiles (list[str])                          │ ─┐      \
   └──────────────────────────────────────────────┘  │      \
                                                     ▼      \
                                              [DMPNNLigandEncoder]
                                                       ↓
                                            128-d embedding
                                                       ↓
                                                        \      ↓
                                                         \  concat (180-d)
                                                          \  ↓
                                                         ┌────────────────┐
                                                         │  MultimodalFuser │
                                                         │  Linear 256     │
                                                         │  ReLU + Dropout │
                                                         └────────────────┘
                                                                  ↓
                                                            256-d fused
                                                                  ↓
                                                       ┌──────────────┐
                                                       │   Head MLP   │
                                                       │ 256 → 128 → 2│
                                                       │  ReLU + Drop │
                                                       └──────────────┘
                                                                  ↓
                                                          logits (B, 2)
                                                                  ↓
                                                          focal loss
                                                          (γ=2, α=auto)
```

The **MultimodalFuser** projects each modality to a common 256-dim, then
concatenates and feeds a single Linear→ReLU→Dropout layer. The **Head MLP**
collapses the fused representation to the 2-class softmax output.

## 4. Training data

Built from the canonical MetalCytoToxDB CSV at
`.data/molmetal/MetalCytoToxDB.csv` (26,801 rows, 24 columns).

Filters applied (matching Krasnov 2026 §3.1 protocol):

- `metal_whitelist = ("Ru", "Ir", "Rh", "Os", "Re")` (5 precious metals)
- `Time(h) ≤ 24` (short-term measurements only)
- `0.01 µM ≤ IC50_Dark_value ≤ 1000 µM` (drop noisy extremes)
- `IC50_Dark_value not null` (require dark-toxicity assay)
- `Cell_line` non-empty (rows with experimental identity)

After filters:

| metric | value |
|---|---|
| rows kept | **5,257** |
| positives (`active = IC50 < 10 µM`) | 1,621 (30.8 %) |
| negatives | 3,636 (69.2 %) |
| unique cell lines | 209 |
| unique metals | 5 |

Each row carries the ligand SMILES (counter-ions and bare metal atoms
stripped via the `_ligand_only_smiles` helper so the D-MPNN encoder sees
just the ligand chemistry), the metal type, the cell-line identity, and
the binary `active` label.

CSV path: `molmetal/data/protein_metal_binding.csv`

Build command:

```bash
uv run --no-sync python -m molmetal.scripts.train_protein_metal --build-data \
    --epochs 50 --seeds 0 1 2 --out-dir logs/metric18_protein_metal
```

If the CSV is missing, `--build-data` (or just running the script which
auto-builds) re-creates it from the canonical MetalCytoToxDB CSV.

## 5. Training protocol

- 70 / 10 / 20 train / val / test **scaffold (Bemis-Murcko) split**, seed-per-seed
- AdamW, lr=1e-3, weight_decay=1e-4, batch=64
- focal loss γ=2.0, α auto-balanced from batch prior
- 50 epochs with patience-10 early stopping on val AUC
- 3 seeds (0, 1, 2); aggregate mean ± std of best-val-epoch **test AUC**
- all training on shared `uv` env (`.venv-vlm` — torch 2.14 + cu130)

```
test_auc_mean = 0.7436 ± 0.0092
test_auc_per_seed = [0.7562, 0.7344, 0.7402]
test_pr_auc_mean = 0.5695
verdict = "baseline"
elapsed = 376.6 s
```

A second run (100 epochs, patience 25, batch 128) yielded
`test_auc_mean = 0.7393 ± 0.0092` (looser early-stop / 100-epoch run), showing the architecture
plateaus near AUC ≈ 0.74 on this data slice (see logs/metric18_protein_metal_v2/).

## 6. Where we fell short of 0.85 — diagnosis

1. **"Protein" is not a true protein sequence.** MetalCytoToxDB records
   cell-line-level phenotype (cytotoxicity). The "binding" proxy
   (active = IC50 < 10 µM) is therefore a *downstream* phenotype, not a
   direct binding signal. The 2025 multimodal paper (AUC > 0.92) operates
   on real sequence-level protein-metal interaction labels — a much
   richer supervision signal than ours.

2. **The D-MPNN we built inline is compact.** We wrote a small 128-dim
   3-depth message-passing encoder so the head is *self-contained* and
   does not pull in Chemprop dependencies. The production D-MPNN
   (`molmetal.baselines.dmpnn.DMPNNModel`) is 256-dim / 6-depth and
   uses 50+ atomic features — about 6× our parameter budget for the
   encoder. Plugging that in is a single one-line swap
   (`self.ligand_enc = DMPNNModel(...)`), which we leave as a follow-up.

3. **Cell-line identity is the dominant signal.** A 32-dim
   `nn.Embedding` with 209 entries already memorises much of the
   per-cell-line baseline (different cell lines have systematically
   different cytotoxicity baselines); this leaves less learning room
   for the ligand-side D-MPNN. We see this in the val-vs-test gap
   (`val ≈ 0.85`, `test ≈ 0.74`) for seed 1 — the model is overfitting
   to cell-line co-occurrences. L2-regularising the cell-line
   embedding (or pre-scaling it to a small subspace) is a follow-up.

4. **No 3D pocket stream active.** The pocket encoder is wired in code
   but disabled because the public MetalCytoToxDB CSV has no pocket
   coordinates. Activating it requires either (a) docking the ligands
   back into known protein structures for the 209 cell-line proteins
   (a separate work-stream), or (b) waiting for the future
   protein-metal binding benchmark (METRICS_TARGETS §6.4) to land.

## 7. Verdict gate (METRICS_TARGETS §2)

| gate | AUC range | band | ours? |
|---|---|---|---|
| ❌ | < 0.70 | below baseline | no |
| ⚠️ | 0.70 – 0.85 | baseline | **yes — 0.7436 ± 0.009** |
| ✅ | 0.85 – 0.92 | good (target) | not yet |
| 🌟 | ≥ 0.92 | matches Tier 1 SOTA | not yet |

Final verdict: **`baseline`**.

## 8. Code surface

- **`molmetal/molmetal_lam/sbdd_env/protein_metal_binding.py`** (~480 LOC):
  - `PMBConfig` – dataclass for hyperparams
  - `CellLineEncoder`, `MetalEncoder`, `PocketEncoder` – per-modality encoders
  - `DMPNNLigandEncoder` – compact self-contained D-MPNN (no Chemprop)
  - `MultimodalFuser` – concat-then-projection fuser
  - `ProteinMetalBindingHead` – top-level model
  - `focal_loss` – focal cross-entropy (Lin et al. 2017)
  - `PMBVocab`, `PMBDataset`, `collate_pmb` – CSV → torch wiring
  - `build_protein_metal_csv` – row filter + ligand-strip step
  - `roc_auc_safe` – degenerate-folder-safe AUC wrapper

- **`molmetal/scripts/train_protein_metal.py`** (~360 LOC): 3-seed × N-epoch
  loop, scaffold split, focal-loss training, metric JSON output.

- **`molmetal/data/protein_metal_binding.csv`** (5,257 rows): built-once CSV.

- **`metrics/by_metric/metric18_protein_metal.json`** (and `metric18_protein_metal_v2.json`):
  aggregate metrics.

## 9. Future work to reach the 0.85 target

1. **Swap the inline D-MPNN for the production backbone** in
   `molmetal.baselines.dmpnn.DMPNNModel`. Expected to lift test AUC
   by +0.05–0.08 based on Metric 1 (Ru subset) gain.
2. **Add the pocket stream.** Either build a small docking-based 3D
   protein pipeline for the 209 cell-line proteins, or wait for the
   upstream "metalloprotein + metal" binding benchmark mentioned in
   METRICS_TARGETS §6.4.
3. **L2-regularise the cell-line embedding** to reduce overfit on
   identity (force the model to lean on the ligand + metal channels).
4. **Hard-negative mining.** Borderline IC50 ≈ 10 µM samples dominate
   noise; upweighting them via the focal γ knob could help.
5. **Ensemble multiple scaffolds.** Average predictions across 3
   scaffold-split folds (instead of one fixed split) typically gains
   +0.02–0.03 on small-ish tabular datasets like ours.

## 10. Run commands used

```bash
source env.sh

# build CSV (auto-built if missing)
uv run --no-sync python -m molmetal.scripts.train_protein_metal \
    --build-data --epochs 50 --seeds 0 1 2 \
    --out-dir logs/metric18_protein_metal \
    --out-metric metrics/by_metric/metric18_protein_metal.json \
    --out-report molmetal/reports/wf_protein_metal_binding/final.md

# longer-run config (100 epochs, larger batch, looser early-stop)
uv run --no-sync python -m molmetal.scripts.train_protein_metal \
    --epochs 100 --seeds 0 1 2 --batch-size 128 --patience 25 \
    --out-dir logs/metric18_protein_metal_v2 \
    --out-metric metrics/by_metric/metric18_protein_metal_v2.json
```
