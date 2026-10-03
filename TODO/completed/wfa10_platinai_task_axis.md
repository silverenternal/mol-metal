# Wf-A10 — PlatinAI 4-cell-line task axis

**Date:** 2026-10-03
**Trigger:** user voice — *"现在前沿的做法是4 个独立任务还是按细胞分头？你调研一下看看在现有基础上能不能跨过这个gap"*, then *"行，这些都得做，你启动ultracode按照你的建议做一下落地"*

---

## §0 TL;DR

**The gap is crossed.** The platinai 4-cell cohort now trains end-to-end on the
existing D-MPNN, producing a per-cell-line result table — the metric shape
Rusanov 2025 reports. **No model code changed:** `MultiTaskWassersteinModel` was
already *"shared encoder + per-task heads"* indexed by an integer task id, so
the cell line simply becomes the task axis.

Verdict: **SHIPPED (plumbing verified, model quality not yet measured).**
The smoke is 1 epoch / 1 seed and returns AUC ≈ 0.50–0.57, i.e. **near chance**.
That is expected for a 1-epoch run and says nothing about the model's ceiling —
it only proves the pipeline is correct. A 50-epoch × 5-seed run is required
before any claim about beating/comparing to Rusanov.

---

## §1 What shipped

File: `molmetal/scripts/train_dmpnn_multitask_wasserstein.py`
Backup: `molmetal/scripts/train_dmpnn_multitask_wasserstein.py.bak.wfa10_taskaxis`

```
-rw-r--r-- 1 hugo hugo 128683 Oct  3 08:26 molmetal/scripts/train_dmpnn_multitask_wasserstein.py
-rw-r--r-- 1 hugo hugo 118641 Oct  3 08:18 molmetal/scripts/train_dmpnn_multitask_wasserstein.py.bak.wfa10_taskaxis
```

| Line | Symbol | What it does |
|---:|---|---|
| 383 | `_stable_hash_int` | SHA-256 based, process-independent hash |
| 478 | `_derive_active_labels` | per-cell median / global binarisation |
| 512 | `build_platinai_cell_line_axis` | builds `(split_df, ds)` for the cell-line axis |
| 2221 | `--task-axis {metal,cell_line}` | selects what the per-task heads are indexed by |
| 2238 | `--active-labeling {per_cell_median,global}` | default `per_cell_median` |
| 2255 | `--active-pic50-threshold` | default 6.0, used only by `global` |

Three defects were fixed along the way, all found by measurement rather than reading:

### 1. The platinai split was non-deterministic (root blocker)

`long_df["scaffold_id"] = ...map(lambda s: abs(hash(s)) % (10**8))` — Python salts
the hash of `str` per process. Measured, three separate processes, `hash("CCO")`:

```
60767441 / 88287468 / 44141501
```

So every run produced a different train/val/test partition while the docstring
claimed *"deterministic hash-based split"*. Every platinai number would have been
irreproducible. Replaced with a SHA-256 digest.

**Verified fixed** — three independent processes now agree byte-for-byte:

```
37c509bb7da095ef7d838036127c1964   (run 1)
37c509bb7da095ef7d838036127c1964   (run 2)
37c509bb7da095ef7d838036127c1964   (run 3)
```

### 2. Cross-cell leakage — checked, and structurally impossible

Because the scaffold id depends only on the SMILES, one molecule lands in one
split across all 4 cell lines. Measured on the real CSV, sampling the 5
lexicographically-first SMILES:

```
{"B1[BH-]2[BH]3...": ["train"], "B=C[P]12...": ["train"],
 "B[BH-]1[BH2-]...": ["test"], ...}
```

Each SMILES maps to exactly one split. Data shape: 16,312 rows = 4,078 SMILES × 4
cells; split sizes train 11,420 / val 1,656 / test 3,236 (70.0 / 10.2 / 19.8 %).

### 3. The active-label threshold did not transfer (found by measuring)

The repo's standard is `active := IC50 < 10 µM` (i.e. `pIC50 > 5.0`,
`molmetal/data/cytotox.py::_add_active`). Measured on the real platinai cohort,
that threshold labels **98.1 %** of rows positive — a degenerate classification
problem whose AUC is meaningless.

| threshold | active fraction |
|---|---|
| pIC50 > 5.0 (repo standard) | 0.981 |
| pIC50 > 5.5 | 0.863 |
| pIC50 > 6.0 | 0.567 |
| pIC50 > 6.5 | 0.244 |
| pIC50 > 7.0 | 0.059 |

A **global** cut cannot be fixed either: the cell lines have different potency
distributions (median pIC50 — a2780 5.69, mcf7 5.99, a549 6.15, a2780cis 6.59),
so at a global 1 µM cut the per-head active fraction is **a2780 0.242 / mcf7 0.491
/ a549 0.632 / a2780cis 0.902** — and a2780cis, the cell line this cohort exists
to characterise, is 90 % positive.

Default is therefore **`--active-labeling per_cell_median`**, which splits each
cell line at its own median. Measured per-task active fraction: **0.50 / 0.50 /
0.50 / 0.50**. `--active-labeling global --active-pic50-threshold 6.0` is retained
for anyone who needs an absolute cut.

A threshold calibrated on one assay's IC50 distribution does not transfer to a
differently pre-filtered cohort; that is why this is a flag and not a constant.

---

## §2 MEASURED evidence

### 2.1 End-to-end smoke — platinai cell-line axis

```
EXIT=0, wall 241.9 s, GPU 0 = RTX PRO 6000 Blackwell (CUDA_VISIBLE_DEVICES=0)
```

Resolved tasks (the `--metals` slot now carries cell lines):

```
[Wf-A6 BLOCKER A] --metals resolved: ['a2780', 'a2780cis', 'a549', 'mcf7']
[Wf-A10 cell-line axis] dataset=platinai_4cell task_axis=cell_line:
  16312 long-format rows; tasks=['a2780','a2780cis','a549','mcf7'];
  per-cell breakdown: {'a2780': 4078, 'a2780cis': 4078, 'a549': 4078, 'mcf7': 4078};
  active_labeling=per_cell_median (per-cell median split)
[wf131-mtw] loaded MetalCytoToxDataset: 16312 rows across
  ['a2780','a2780cis','a549','mcf7']
```

Per-cell result table (the deliverable — one row per cell line, Rusanov's shape):

```
  a2780:    n_train=2855 n_val=414 n_test=809 test_auc=0.5018 test_pearson=0.034
  a2780cis: n_train=2855 n_val=414 n_test=809 test_auc=0.5734 test_pearson=0.081
  a549:     n_train=2855 n_val=414 n_test=809 test_auc=0.5045 test_pearson=-0.005
  mcf7:     n_train=2855 n_val=414 n_test=809 test_auc=0.5087 test_pearson=-0.051
```

**HONEST:** these are 1 epoch, 1 seed, and near chance. They demonstrate the
pipeline, NOT model quality. Do not cite them as a result.

### 2.2 5-metal regression — unchanged

```
EXIT=0.  --metals resolved: ['Ir','Os','Re','Rh','Ru']; 5-metal split table 26801 rows
  Ir: n_train=784  n_val=157 n_test=301
  Os: n_train=66   n_val=7   n_test=14
  Re: n_train=121  n_val=9   n_test=36
  Rh: n_train=60   n_val=10  n_test=38
  Ru: n_train=2509 n_val=363 n_test=796
```

Split sizes are byte-identical to the pre-change run — the metal path is unaffected.

### 2.3 Why this is the right architecture (research)

| approach | representative | status |
|---|---|---|
| separate model per cell line | Davis et al. (early QSAR) | superseded |
| **one shared network over all cell lines** | PECAN (59 NCI-60 lines at once) | current practice — argues the shared objective forces generalisable features |
| shared encoder + cell identity as a condition | NanoToxRadar (110 cell lines) | variant |

Rusanov 2025 (ChemRxiv 2025-06-20 v2, DOI 10.26434/chemrxiv-2025-pp32k) reports
PlatinAI per cell line (MCF-7 548 / A549 517 / A2780 423 / A2780cis 323 72-h IC50
values). Repo's own plan already said *"V4 4-cell-line multi-task head"*
(`TODO/01_research/sota_papers_2025_2026.md:241`).

---

## §3 What did NOT ship / honest-negative

1. **No model-quality result.** 1 epoch / 1 seed, AUC ≈ chance. A 50-epoch ×
   5-seed run is needed before any comparison to Rusanov 2025 is meaningful.
2. **No per-cell R² / Pearson beyond the smoke** — `test_pearson` is ≈ 0 at 1 epoch.
3. **Epoch cost is high**: 42.5 s/epoch for one cell line (12,643 SMILES featurised
   per task). Four tasks × 50 epochs × 5 seeds is a substantial GPU budget that
   has not been scoped here.
4. **A `--task-axis cell_line` run writes checkpoints named after cell lines**
   (`dmpnn_mtw_mcf7_seed0.pt`) into the shared `molmetal/checkpoints/` directory,
   and the default `--resume-from auto` will pick them up on a later run
   (observed: `train_seconds=0.0035` in §2.2, i.e. it resumed instead of
   training). Checkpoint isolation by dataset+axis is still open.
5. **The workflow I first attempted could not run at all** — see §4.
6. `platinai_4cell.csv` is documented in-repo as PlatinAI *predicted* values
   (`platinai_dataset.py` references `PlatinAI_predicted_*.xlsx`). Whether these
   are measured or predicted labels has NOT been verified and would change how
   the per-cell AUC should be framed.

---

## §4 Process finding — AGENTS.md has outgrown subagent injection

The first attempt was an ultracode workflow (5 agents: implement → verify →
synthesize). **All 5 failed in 4.97 s with `Prompt is too long`, 0 tokens,
0 tool calls** — they died during prompt assembly, before doing anything.

Cause: `AGENTS.md` is **1,245,399 bytes across 896 lines**, and every subagent
gets it injected at start. The §6 wave-history table has been appended to by
hundreds of consecutive waves. **Any multi-agent workflow in this repo is
currently impossible.**

This is the same discipline the project already adopted from OmegaFold for
README (§5.10 rule 3: *"ONE README.md ≤ 4 KB … STATUS / QUICKSTART /
ARCHITECTURE / DESIGN-BOUNDARY must live in separate docs/*.md"*) — AGENTS.md
needs the same split. Until then, §5.8's own table applies: *"1-3 file edits,
<50 lines → Direct Edit/Write (no wf)"*, which is what this task actually was.
The work above was done inline.

---

## §5 Forward-looking

1. **Scope the real run**: 4 tasks × 5 seeds × 50 epochs at ~42 s/epoch ≈ 12 GPU-h
   per seed-set; decide batch size / workers before committing.
2. **Checkpoint isolation** by `(dataset, task_axis, task)` so `--resume-from auto`
   cannot mix a cell-line run with a metal run.
3. **Verify the label provenance** of `platinai_4cell.csv` (measured vs predicted).
4. **Split AGENTS.md** (§6 → `docs/WAVE_HISTORY.md`) to unblock multi-agent workflows.
5. Then, and only then, compare per-cell AUC against Rusanov 2025 Table S4–S5.

---

## §6 Cross-references

- `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` (this change)
- `TODO/01_research/sota_papers_2025_2026.md` §1.2 (PlatinAI / Rusanov 2025)
- `molmetal/data/cytotox.py::_add_active` (the 10 µM standard this overrides)
- `AGENTS.md` §5.5 (filesystem truth) / §5.6 (atomic write) / §5.7 (flag smoke) /
  §5.8 (wf token budget) / §5.10 (OmegaFold doc discipline)
- PECAN — https://par.nsf.gov/servlets/purl/10534246
- Rusanov 2025 — DOI 10.26434/chemrxiv-2025-pp32k
