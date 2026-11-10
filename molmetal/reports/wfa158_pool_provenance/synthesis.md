# Wf-A158 BRANCH POOL — pool provenance + round-trip audit

**Pool under audit:** `molmetal/reports/cfm_c8_fullgen_2026_10_24/generated_smiles.csv`
**Artifact:** `pool_provenance.json` (this directory)
**Script:** `molmetal/scripts/wfa158_pool_provenance.py` (CPU-only, RDKit + stdlib, no torch)
**Tests:** `molmetal/tests/test_wfa158_pool_provenance.py` (15 passed)

All numbers below are **MEASURED** on this machine by
`CUDA_VISIBLE_DEVICES="" .venv/bin/python molmetal/scripts/wfa158_pool_provenance.py`
(wall time ≈ 2.1 s) and are re-derivable from the artifact.

---

## 0. Headline

| # | Statement | Status |
|---|---|---|
| 1 | The pool's provenance **is** resolvable from on-disk artifacts — run, date, config, checkpoint files and sampling settings are all pinned by `file:line` citations. | **RESOLVED** |
| 2 | The pool's connectivity columns are **bit-exact reproducible** from the pool's own `smiles` column with RDKit's default parse: `n_frags_raw` matches on **2112 / 2112** rows, `largest_frag_frac` matches on **2112 / 2112** rows. **The RAW-vs-POST-PROCESSED break is therefore NOT a serialisation artefact.** | **MEASURED** |
| 3 | **HONEST-NEGATIVE:** the run that produced the pool **died at epoch 3 of 18** with a CUDA OOM. The pool is a *truncated* run, not a converged model. Any frontier-parity number computed on this pool inherits that. | **MEASURED** |
| 4 | The published columns are **RAW** — the field-standard "keep the largest connected component" post-process was **OFF** for this run. | **MEASURED** |

---

## 1. Q1 — PROVENANCE

`git log` **cannot** be the source: the pool CSV is gitignored
(`.gitignore:142` → `molmetal/reports/**/*.csv`), and the whole run directory is
untracked. Provenance comes from `master.log`, `events.jsonl`, the run's YAML
config, and the checkpoint mtimes.

| Claim | Value | Evidence |
|---|---|---|
| Which run | `train_cfm_yuelbond_wasserstein.py`, the **Wf-124 P2** CFM trainer, tmQM corpus (train=26 715, val=192) | `master.log:9-12`, `master.log:24` |
| Which config | `configs/c8_fullgen_b64_2026_10_24.yaml` (sha256 `39c7acae…3e2`), committed `8fc837b3` *"c8: full-corpus batch-64 re-run of the c7 protocol"* (2026-10-24 23:15 +0800) | config `output.out_dir` (line 636) **is** the pool's own directory |
| Which out_dir / ckpt_dir / log | `output.out_dir` = `molmetal/reports/cfm_c8_fullgen_2026_10_24`; `output.ckpt_dir` = `…/checkpoints`; `output.log_master` = `…/master.log`; `output.save_generated_smiles: true` | config lines 636 / 640 / 665 / 676 |
| Launch command | `/home/hugo/codes/mol-metal/.venv/bin/python molmetal/scripts/train_cfm_yuelbond_wasserstein.py --config {yaml_path} --seed {seed}` (single-venv §5.9, YAML-only §5.11) | config line 780 |
| Date | first event **2026-10-24T20:21:00Z** (= 2026-10-25 04:21 +0800), last event **2026-10-24T22:01:49Z** (= 2026-10-25 06:01 +0800) | `events.jsonl` `train_epoch.ts`; CSV mtime 2026-10-25T06:01:49+08:00 |
| Sampling | ODE `method: midpoint`, `n_steps: 50`; `gen_eval_n: 128` × seeds `0 1 2 3 4 5`; `temperature=1 top_p=1 top_k=0`; `seed: 0` | `master.log:204` (`solver=midpoint/50 seeds=6x128`), `master.log:33` (decode knobs), config lines 632-633 / 614 / 618 / 728 |
| Hyperparameters | `batch_size: 64`, `lr: 3.0e-4`, `epochs: 18` (planned), `checkpoint_interval: 1`, `resume_from: none` | config lines 276 / 280 / 191 / 660 / 664 |
| **From WHICH checkpoint** | **NO checkpoint file.** The CSV rows are decoded from the **live in-memory model** at the end of each epoch: `_gen_smiles` is created inside the epoch loop (`trainer:18926`), filled by the generation eval (`trainer:18969`), and appended to the CSV (`trainer:19215-19251`). | `trainer:18926`, `trainer:18969`, `trainer:19215-19251` |
| Checkpoints on disk (hashed as BYTES, never deserialised) | `cfm_yuelbond_wasserstein_seed0.pt` — 8 395 006 B, mtime 2026-10-25T05:17:40+08:00, sha256 `fbca34d6…06a5f`, **epoch 1** (inferred); `cfm_yuelbond_wasserstein_seed0_last.pt` — 8 395 006 B, mtime 2026-10-25T06:01:49+08:00, sha256 `0ad6b6a6…fabcb`, **epoch 2** (inferred) | epoch attribution **INFERRED** from mtime vs `events.jsonl` `train_epoch.ts`; the gate metric corroborates it (`bond_type_f1` pooled: epoch 0 = 0.475, **epoch 1 = 0.632**, epoch 2 = 0.618 → the best is epoch 1, and `seed{N}.pt` is written only on a gate improvement) |
| **Run completeness** | **epochs completed = [0, 1, 2] of 18 planned.** The process died with `torch.OutOfMemoryError: CUDA out of memory` during the next epoch's backward pass. | `master.log:235` (Traceback), `master.log:257` (OOM) |

Cross-artifact consistency (all MEASURED, all PASS):

* CSV rows per epoch = **{0: 704, 1: 704, 2: 704}** = `events.jsonl` `gen_validity_n` = **[704, 704, 704]** = the **sum of the per-seed `n_total`** (128 + 64 + 128 + 128 + 128 + 128 = 704), with `n_decode_error = 0` and `gen_validity = 1.000`.
  **The pool is COMPLETE for what the run decoded** — no decoded sample is missing from it, and every one of them parsed (`smiles_out` appends only on success, `trainer:10237-10238`).
* `metal_ratio = 1.000` on all three epochs — 2 112 / 2 112 rows contain a metal.

---

## 2. Q2 — ROUND-TRIP (RDKit only, no model)

Recomputation uses RDKit's **defaults**, i.e. exactly what the producing code used:
`Chem.MolFromSmiles(smiles)` (sanitisation ON, explicit H removed) then
`Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)`.
`largest_frag_frac` = `max(fragment atom count) / mol.GetNumHeavyAtoms()`.
Equality tolerance on `largest_frag_frac` is **5e-5** — the writer rounds to 4 dp
(`trainer:19310`), so half a unit in the last published place is the exact-equality bound.

| Quantity | MEASURED |
|---|---|
| Rows | **2112** |
| Rows RDKit can parse | **2112 / 2112** (0 unparseable) |
| Rows that parse only without sanitisation | **0** (so the Wf-A114 `-1` sentinel never arises for this pool) |
| `n_frags_raw` **EQUAL** to the recomputed fragment count | **2112 / 2112** |
| `n_frags_raw` **DIFFERENT** | **0** — difference histogram (stored − recomputed) is **empty** |
| `largest_frag_frac` EQUAL (within 5e-5) | **2112 / 2112** |
| `largest_frag_frac` DIFFERENT | **0** — delta stats: min −5.000e-05, max +5.000e-05, mean −8.76e-07, median 0.0, max abs 5.000e-05; 702 rows have delta exactly 0, the rest are pure 4-dp rounding residue |
| Row whose `smiles` is **not** byte-identical to `Chem.MolToSmiles(mol)` | **1 / 2112 (0.047 %)** — row 437 (epoch 0). It differs only in **ring-closure digit allocation** (`C9C(C936)` → `C9C(C1%10…C1%10…)`); `same_molecule_as_canonical = true`, and the fragment count (3) and the largest-fragment fraction (0.9692) are **unchanged**. |
| Metal present | **2112 / 2112** |
| Metal **shares** its fragment with ≥ 1 non-metal atom | **2082 / 2112 = 98.58 %** |
| Metal **ALONE** in its own fragment | **30 / 2112 = 1.42 %** |
| … of those 30, **surviving the round-trip as a single metal atom** | **30 / 30 = 100 %** |

### What this means for the "RAW vs POST-PROCESSED" break

The published columns are **RAW**. `--decode-largest-fragment` is
`action="store_true"` (`trainer:14210-14211`), the c8 config does not set it, and
the pool's own multi-fragment rows still carry their dots — **1583** rows contain a
`.` and **1583** rows have `n_frags_raw > 1`, so the two sets coincide and truncation
was OFF.

Had the field-standard post-process ("keep the largest connected component") been
applied to **our** pool, it would have made **2112 / 2112 = 100.0 %** of rows
single-fragment, and would have dropped **4 556 heavy atoms** in total across the
**1583** rows that are currently multi-fragment.

**This is a measurement of what the transformation WOULD do to our pool. It asserts
nothing about what any frontier method does — that comparison is a different branch
(BRANCH FRONT) and is deliberately not claimed here.**

---

## 3. Q3 — COLUMN SEMANTICS

| Column | What it actually means | Source |
|---|---|---|
| `n_frags_raw` | The number of **RDKit connected components** of the raw decode: `len(Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False))` on the **sanitised** mol. Not a bond count, not an atom count. It is a **RAW** count — it describes the decode *before* any largest-fragment truncation. | `trainer:9671` (`Chem.MolFromSmiles(smiles)`), `trainer:9674-9675` (frags/sizes), `trainer:9697` (`"n_fragments": len(sizes)`), `trainer:19282-19287` (the writer copies it) |
| `largest_frag_frac` | `max(fragment atom counts) / mol.GetNumHeavyAtoms()` — the atom share of the largest component **of the whole molecule**. Both numerator and denominator **include the metal atom** (it is a heavy atom and sits inside whichever fragment it belongs to). A metal alone in its own fragment adds 1 to the denominator and 0 to the numerator. | `trainer:9674-9676`, `trainer:9703`, `trainer:19310` (`round(_lfrac, 4)`) |
| before / after sanitisation | **AFTER** — `Chem.MolFromSmiles` defaults to `sanitize=True`, so a string that fails sanitisation yields `None` and the writer's `decode_chemistry()` returns `None` | `trainer:9671`, `trainer:19282-19287` |
| metal included? | **YES**, in both columns | `trainer:9676`, `trainer:9703` |
| sentinel history | As of Wf-A114 (2026-10-30) an unparseable row writes `n_frags_raw = -1`. This pool predates that fix, so an unparseable row here would carry the OLD fabricated `1` / `1.0`. **MEASURED: 0 of 2 112 rows fail to parse, so the question does not arise for this pool.** | `trainer:19253-19266` |
| writer path | `save_generated_smiles` gate (`trainer:19215`) → append-only CSV (`trainer:19248`) → header `epoch,smiles,n_frags_raw,largest_frag_frac` (`trainer:19249-19251`) | — |

---

## 4. What was NOT established

1. **Not established:** whether the frontier methods post-process at all, and by how much. Out of scope for this branch — BRANCH FRONT's job. Nothing here asserts it.
2. **Not established:** the **per-row** seed / `template_offset` / cloud-id mapping. The CSV has no `seed`, `template_offset` or cloud-id column, and `per_cloud_out` was not written to the file. Only the epoch-level aggregate is recoverable.
3. **Not established:** why the per-seed decode counts are uneven (per epoch: 128, 64, 128, 128, 128, 128). The run directory records the counts but not the reason.
4. **Not established:** that the pool is a *converged* or *representative* model. It is the output of **3 of 18 planned epochs** of a run that died with a CUDA OOM.
5. **Not established:** the checkpoint **epoch attribution is INFERRED** from mtime alignment plus the gate-metric history, not read out of the `.pt` files — they were never deserialised (no `torch.load`, no GPU, no weights in RAM).
6. **Not established:** that the 1 non-canonical row (row 437) is a *harmless* serialisation artefact in every downstream consumer. It is the same molecule and the connectivity columns are unaffected, but a downstream consumer that string-matches SMILES would see a difference.
7. **Not established:** anything about the 4 556 atoms that truncation would drop beyond the count — no chemistry was scored on the truncated forms.

---

## 5. Reproduce

```bash
cd /home/hugo/codes/mol-metal
CUDA_VISIBLE_DEVICES="" .venv/bin/python molmetal/scripts/wfa158_pool_provenance.py
CUDA_VISIBLE_DEVICES="" .venv/bin/python -m pytest \
    molmetal/tests/test_wfa158_pool_provenance.py -p no:cacheprovider -q
```

Exit codes: `0 = PASS`, `1 = FAIL`, `2 = SKIP` (AGENTS.md §4 stdout invariant).
The script writes `pool_provenance.json` (schema `wfa158_pool_provenance.v1`) via
`molmetal/scripts/wf386a3_atomic_write.py::atomic_write_file` (AGENTS.md §5.6).
