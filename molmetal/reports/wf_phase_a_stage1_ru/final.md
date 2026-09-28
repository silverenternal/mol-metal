# Phase A Stage 1.5 — Eval + Ship Decision (Ru AUC, 5-metal multitask)

**Date:** 2026-09-25 (R17)
**Stage:** 1.5 — eval & ship gate (after Stage 1.4 training)
**Owner:** Hugo Lee
**Metric:** m1 — Ru subset binary classification AUC, MetalCytoToxDB
**Recipe:** 3-seed × 100-epoch × 5-metal multi-task V4 D-MPNN (`train_ru_multitask.py`)
**Status:** ❌ **BLOCK** — recipe run did not complete; required metrics file missing.

---

## 1. TL;DR

| Item | Value |
|---|---|
| Decision | **BLOCK** — Stage 1.6 ablation required |
| Reasoning | `metrics/by_metric/ru_5metal_multitask.json` does not exist. The Stage 1.4 training run (`train_ru_multitask.py`, started 22:53) was interrupted before completing seed 0's first eval cycle (only the initial setup line `[train_ru_multitask] seed=0 hidden=300 depth=3 frozen=1,805,400 trainable=1,368,000` was logged). No ckpt, no per-seed metrics JSON, no test AUC. |
| What we have | (a) **R17 1-seed baseline** (`metrics/by_metric/metalcytotox_ru_auc.json`): `0.7123 ± 0.0000` (n=1), scaffold split, plain D-MPNN multi-task — **below gate (0.86) AND below Krasnov (0.81)**. (b) **3-seed LOMO Ru subset** (`metrics/by_metric/metal_lomo_auc.json`, n_runs=15, Ru fold only): `0.5706 ± 0.0032` (n=3 seeds × 1 Ru fold) — **different split protocol (leave-one-metal-out), not directly comparable to Krasnov 2026**. |
| Action | (i) Re-run `train_ru_multitask.py` end-to-end on a host without parallel ultracode contention (current host is also running `train_cisplatin_resistance.py` at 104% CPU since 22:53). (ii) Decide whether to first resolve the tmQM→DirectedMPNN arch mismatch (KNOWN LIMITATION) — without that, the +0.046 ablation lift from tmQM pretrain is unrealisable. |

---

## 2. Decision tree walk-through

| Gate condition | Verdict |
|---|---|
| `mean Ru AUC ≥ 0.86` → SHIP, claim leadership | ⛔ Cannot evaluate — no per-seed data |
| `mean Ru AUC ∈ [0.81, 0.86)` → SHIP, "match Krasnov 2026 within std" | ⛔ Cannot evaluate — no per-seed data |
| `mean Ru AUC < 0.81` → BLOCK, Stage 1.6 ablation | ⛔ Cannot evaluate — no per-seed data |
| **FALLTHROUGH** — `metrics/by_metric/ru_5metal_multitask.json` missing | ❌ **BLOCK** — run incomplete; do not ship |

The decision tree presumes a non-null metrics file. We treat "missing file" as a stronger BLOCK signal than "AUC below gate" because the underlying experiment never produced a measurement — re-evaluating the same incomplete setup cannot lift the gate, only re-running can.

---

## 3. What was attempted (Stage 1.4 training log)

`logs/train_ru_multitask/seed_all.log` (full content, 7 lines, log mtime 22:54, 2026-09-25):

```
warning: `--no-sync` has no effect when used outside of a project
[train_ru_multitask] data=/home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask.csv init=tmqm epochs=100 batch=128 lr=0.001 seeds=[0, 1, 2] hidden=300 depth=3
[train_ru_multitask] split counts: {'Ru': {'train': 13395, 'val': 1913, 'test': 3827}, 'Ir': {'train': 3182, 'val': 455, 'test': 909}, 'Rh': {'train': 794, 'val': 113, 'test': 227}, 'Re': {'train': 607, 'val': 87, 'test': 174}, 'Os': {'train': 782, 'val': 112, 'test': 224}}
[train_ru_multitask] device=cuda
[train_ru_multitask] tmQM encoder arch is incompatible with V4MultiTaskDMPNN.encoder.dmpnn (DirectedMPNN); falling back to random init.  See script docstring KNOWN LIMITATION for the fix path (pretrain_v4multitask_dmpnn.py).
  warnings.warn(
[train_ru_multitask] seed=0 hidden=300 depth=3 frozen=1,805,400 trainable=1,368,000
```

Observations:
* **No epoch lines** — the run never started training; it printed seed 0 architecture info and was killed before epoch 0 forward pass completed (eval cadence is every 10 epochs, first eval at epoch 10).
* **No ckpts** — `.ckpt/` directory is empty (mtime 22:53).
* **No metrics JSON** — `metrics/by_metric/ru_5metal_multitask.json` does not exist.
* **tmQM pretrain FELL BACK to random init** (warning at line 5) — the recipe's expected +0.046 lift from tmQM pretrain is structurally unrealised. Even if the run had completed, the model would be a multi-task V4 with **random-init encoder**, not a tmQM-pretrained encoder.
* **Data prep succeeded** — split counts confirm the 5-metal CSV at `.data/molmetal/metalcytotox_5metal_multitask.csv` parsed correctly (Ru train/val/test = 13 395 / 1 913 / 3 827; total 26 801 rows per the script docstring).

---

## 4. Per-seed AUC table (CANNOT BE FILLED — no data)

The table below records the schema; cells are left blank to make the absence explicit.

| Seed | Train AUC | Val AUC | Test AUC (Ru) | Test PR-AUC (Ru) | Hit@5% (Ru) | Best epoch |
|---|---|---|---|---|---|---|
| 0 | _missing_ | _missing_ | _missing_ | _missing_ | _missing_ | _missing_ |
| 1 | _missing_ | _missing_ | _missing_ | _missing_ | _missing_ | _missing_ |
| 2 | _missing_ | _missing_ | _missing_ | _missing_ | _missing_ | _missing_ |
| **mean** | **—** | **—** | **—** | **—** | **—** | **—** |
| **std** | **—** | **—** | **—** | **—** | **—** | **—** |

---

## 5. Comparison to Krasnov 2026 + R17 baseline

| Source | Model | Split | AUC | Notes |
|---|---|---|---|---|
| Krasnov 2026 (J. Med. Chem.) | Morgan FP + GBDT | Ru subset 5-fold CV | **0.81** | Tier-1 SOTA reference |
| **Mol-Metal (R17, n=1 seed)** | D-MPNN multi-task (alpha=0.5) | Ru scaffold 70/10/20 | **0.7123** | `metrics/by_metric/metalcytotox_ru_auc.json` — gate = 0.86 |
| Mol-Metal (LOMO Ru fold, n=3 seeds) | D-MPNN multi-task | leave-Ru-out across {Ir, Os, Rh, Re, Ru} | **0.5706 ± 0.0032** | `metrics/by_metric/metal_lomo_auc.json` per_fold_auc_mean["Ru"] — different protocol, NOT comparable to Krasnov CV |
| **Mol-Metal (Stage 1.4, 5-metal multitask)** | V4MultiTaskDMPNN (hidden=300, depth=3) + tmQM→random-init fallback | Ru scaffold split (5-metal multitask) | **— NOT MEASURED** | `train_ru_multitask.py` interrupted |

Δ-to-gate for the measured baseline:
- R17 1-seed: 0.712 − 0.86 = **−0.148**
- LOMO 3-seed Ru: 0.571 − 0.86 = **−0.289** (note: LOMO is harder OOD, not directly comparable)

Δ-to-SOTA:
- R17 1-seed: 0.712 − 0.81 = **−0.098**
- LOMO 3-seed Ru: 0.571 − 0.81 = **−0.239**

---

## 6. Confusion matrix on Ru test split (CANNOT BE FILLED)

Stage 1.4 test set = `Ru/test = 3 827 rows` per the split-counts log line (split performed Bemis-Murcko scaffold-wise per the recipe). The Stage 1.4 confusion matrix would have been computed at the end of training for seed-0 best-epoch checkpoint; without the checkpoint, no prediction logits are available. Schema placeholder:

```
Ru test (n=3,827) — scaffold split, 5-metal multitask:
                Predicted
                Active   Inactive
Actual Active     TP       FN      recall = TP/(TP+FN)
       Inactive   FP       TN      TNR    = TN/(FP+TN)
                                     precision = TP/(TP+FP)
                                     F1        = 2*P*R/(P+R)
```

(The seed matrix prediction of the matrix needs the per-seed test logits — Stage 1.6 ablation re-run will fill this in.)

---

## 7. Per-metal breakdown (CANNOT BE FILLED — no data)

The 5-metal multi-task model has heads for Ru / Ir / Rh / Re / Os. The Stage 1.4 recipe reports `test_auc_per_metal` (one entry per metal per seed). Schema placeholder for the eventual table:

| Seed | Ru AUC | Ir AUC | Rh AUC | Re AUC | Os AUC | Mean (5-metal) |
|---|---|---|---|---|---|---|
| 0 | _missing_ | _missing_ | _missing_ | _missing_ | _missing_ | _missing_ |
| 1 | _missing_ | _missing_ | _missing_ | _missing_ | _missing_ | _missing_ |
| 2 | _missing_ | _missing_ | _missing_ | _missing_ | _missing_ | _missing_ |
| **mean** | **—** | **—** | **—** | **—** | **—** | **—** |
| **std** | **—** | **—** | **—** | **—** | **—** | **—** |

For comparison, the LOMO (single-metal-out) per-metal AUC means are recorded in
`metrics/by_metric/metal_lomo_auc.json`:

| Metal | LOMO mean AUC | LOMO std | n_seeds |
|---|---|---|---|
| Ru | 0.5706 | 0.0032 | 3 |
| Ir | 0.6176 | 0.0068 | 3 |
| Os | 0.6113 | 0.0235 | 3 |
| Rh | 0.3677 | 0.0184 | 3 |
| Re | 0.5059 | 0.1281 | 3 |

(These are LOMO transfer AUCs, NOT in-distribution test AUCs; useful only as a lower bound on the multi-task model when each metal is held out.)

---

## 8. Ablation: tmQM pretrain +0.046 lift (EXPECTED, NOT REALISED)

The Stage 1.4 recipe spec (`TODO/03_plans/R17_AUC_RECOVERY_PLAN.md` §2 Stage 1, line 84) expected a **+0.046** lift from tmQM pretraining. This was supposed to bridge the gap from R17 baseline 0.712 to ≈ 0.86 gate:

```
0.712 (R17, 1-seed × 5-ep D-MPNN)
+ 0.046 (tmQM pretrain expected)
+ ~0.10 (3-seed × 100-ep + multi-task, expected)
= ~0.86 (gate)
```

**Critical status (KNOWN LIMITATION, script docstring line 25):**

> ``models/tmqm_pretrained/tmqm_pretrained_encoder.pt`` was trained on the *baseline* ``MetalCytotoxModel`` (baselines/dmpnn.py: 12 tensors, hidden=128, ``_GRUUpdate`` 3-gate GRU), not on the DirectedMPNN encoder that backs V4MultiTaskDMPNN (models/dmpnn.py: 36 tensors, hidden=300, ``GRUMessageUpdate`` = ``msg_proj + GRUCell``).  The two architectures are incompatible (different state-dict keys, different hidden dim, different bond-feature dim 10 vs 6).  A future ``pretrain_v4multitask_dmpnn.py`` is needed to pretrain V4's DirectedMPNN encoder on tmQM before this recipe can load pretrained weights end-to-end.  This script falls back to **random init** for V4 encoder in the meantime.

**Implication:** the Stage 1.4 run that was attempted would have trained a **random-init V4 encoder**, not a tmQM-pretrained encoder. The expected +0.046 lift was structurally not available. The ablation gap is therefore closer to:
```
0.712 (R17)
+ 0.00 (tmQM pretrain — disabled, random init fallback)
+ ~? (3-seed × 100-ep + multi-task — UNMEASURED)
```

If the multi-task + 3-seed × 100-ep lift alone is ~+0.05–0.10, the random-init fallback could still reach gate (0.76–0.81), but likely NOT 0.86. The expected +0.046 from tmQM pretrain was the safety margin for the gate; without it, the recipe is structurally at-risk of missing gate.

---

## 9. Stage 1.6 ablation plan (BLOCKED → unblock recipe)

To unblock Stage 1.5 ship-gate, the recipe must be repaired along one of two paths:

**Path A — Re-run with random-init encoder (cheap, likely insufficient)**

1. Re-launch `train_ru_multitask.py` with `--init random` to make the fallback explicit; expected wall-clock ≈ 12 h on a CUDA GPU at 100 epochs × 3 seeds × eval every 10 epochs.
2. On a host without parallel ultracode contention — current host is running `train_cisplatin_resistance.py` at 104% CPU (PID 3450108, started 22:53, 1:42 CPU time) and likely starved the ru_multitask CUDA job.
3. If mean Ru AUC ≥ 0.81 → SHIP with honest "match Krasnov within std" framing. If still < 0.81 → continue to Path B.

**Path B — Resolve tmQM→DirectedMPNN arch mismatch (2-3 days, expected to give the +0.046 lift)**

1. Write `molmetal/scripts/pretrain_v4multitask_dmpnn.py` — masked-atom + masked-bond BERT-style pretraining of the V4 DirectedMPNN encoder on the 108k tmQM complex set (see TODO/03_plans/R17_AUC_RECOVERY_PLAN.md §2 Stage 1 task 3).
2. Re-train ru_multitask with `--init v4tmqm` after pretraining completes.
3. Expected total wall-clock ≈ 18-24 h (pretrain 6-8 h on tmQM, fine-tune 12 h on MetalCytoToxDB).

**Path C — Skip multi-task, fix the 1-seed scaffold split alone (cheap fallback)**

If neither A nor B lifts the AUC, fall back to a **5-fold CV** (Krasnov 2026 protocol) on the plain multi-task D-MPNN to match Krasnov's evaluation. Honest framing: "different protocol (CV vs scaffold), not paper-promotable as a Krasnov-beating claim".

---

## 10. Files referenced / updated by this report

| File | Action | Status |
|---|---|---|
| `molmetal/reports/wf_phase_a_stage1_ru/final.md` | write | created (this file) |
| `metrics/by_metric/ru_5metal_multitask.json` | read | **missing** — confirms BLOCK |
| `metrics/by_metric/metalcytotox_ru_auc.json` | read | unchanged (1-seed baseline 0.712) |
| `metrics/by_metric/metal_lomo_auc.json` | read | unchanged (3-seed LOMO per-metal) |
| `logs/train_ru_multitask/seed_all.log` | read | unchanged (interrupted at seed 0 init) |
| `.ckpt/` | read | empty — no checkpoints saved |
| `TODO/INDEX.md` | update | m1 row status stays at FAIL — see §11 |
| `TODO/02_metrics/METRICS_TARGETS.md` Table 2 | update | no change needed (gate ≥ 0.86 unchanged; per-seed Ru AUC remains TBD until Stage 1.6) |
| `paper/sections/04_results_table2.tex` | regenerate | no change (regenerate_section4.py pulls `metrics/by_metric/*`; missing file means no entry, current row stays as $0.712$ / REGRESS) |

---

## 11. Honest summary

The Stage 1.4 recipe run did not produce a measurement. The expected tmQM pretrain lift is unrealisable due to an encoder architecture mismatch (baseline MetalCytotoxModel vs V4 DirectedMPNN). Two paths forward:

1. **Re-run with random-init encoder** (cheap, 12 h, expected AUC 0.76–0.81 — likely fails gate).
2. **Pretrain V4's DirectedMPNN on tmQM first** (2-3 days, expected to give the +0.046 lift; then re-run).

The BLOCK verdict is **not** a reflection of a low measurement; it is a reflection of an **absent measurement**. The decision tree in §2 cannot fire because the input file does not exist.

We do NOT claim "Ru precious-metal AUC leadership" because no Stage 1.4 data exists. We do NOT soften the existing R17 1-seed number (0.712) into a "match Krasnov" claim — the 0.712 is BELOW the Krasnov 0.81 baseline, full stop. Re-running the same recipe on the same interrupted host will repeat the interruption; the next attempt needs a cleaner host reservation or a smaller-scope experiment (Path C: 5-fold CV on plain D-MPNN).

End of report.