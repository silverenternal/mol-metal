# Wf-A10 — NCI-60 9-cell-line task axis on REAL labels, + 5 scripts repaired

**Date:** 2026-10-03
**Status:** SHIPPED (SCAFFOLD — CPU smoke verified; the GPU launch is the user's)
**Scope:** `molmetal/scripts/train_dmpnn_multitask_wasserstein.py`,
`configs/training/nci60_9cell_5seed_50ep_v1.yaml`,
`molmetal/tests/test_script_import_health.py`, 5 repaired scripts

---

## §0 TL;DR

Two things happened, in this order.

1. **Found and fixed 5 unimportable scripts.** The Wf-416-A1 atomic-write
   migration had reported success while silently corrupting 5 of the 12 files
   it rewrote. All 5 were unimportable. Two were load-bearing:
   `train_nci60_10cell.py` (the script behind the m21 Tier-1 LEAD metric,
   0.8007 AUC) and `train_metal_lomo.py` (the LOMO trainer). Their baselines
   could not be reproduced. Fixed; 12/12 now import. Committed `57d9ae94b5`.

2. **Wired the NCI-60 9-cell cohort into the main trainer on a cell-line task
   axis**, so the real-labelled cohort can run through the newer loss
   machinery rather than only through m21's bespoke runner. The split
   reproduces m21's **byte-for-byte**, so results are comparable to 0.8007.
   CPU smoke: **9/9 cell lines, EXIT=0**; §5.7 **4/4**.

**Not done, and deliberately so:** no GPU launch. Per the standing user
directive the launch is the user's to make.

> **CORRECTION (2026-10-03, same day).** This paragraph originally added
> "the venv currently carries a ROCm torch build (`cuda_available=False`) so a
> real run would silently fall back to CPU anyway." **That is now false and has
> been removed.** Re-measured: `torch==2.14.1+cu130`, `torch.version.hip is
> None`, `torch.cuda.is_available() is True`, 2 devices, real matmul on
> `cuda:0` succeeds. GPU launch works. The claim was true when written and
> went stale within the session — a live example of the environment-fact
> drift the docs-consolidation pass exists to stop.

---

## §1 The 5 broken scripts (found, fixed, guard added)

### §1.1 What was wrong

Wf-416-A1's migration injected a helper and an import block into
`molmetal/scripts/*.py`. It produced two distinct corruptions, neither of
which its own report caught.

**Mode 1 — decorator clobber (4 files).** The helper was spliced *between* a
`@dataclass` decorator and the class it decorated:

```python
@dataclass                              # was decorating the class below
def _wf416_atomic_write_text(...): ...  # now the decorated target
class NCIPreciousDataset: ...           # silently lost its generated __init__
```

Import dies with `AttributeError: 'function' object has no attribute
'__mro__'`. Note the second-order damage: even if you only delete the stray
decorator, the class is still not a dataclass and has no `__init__`.

| File | Class that lost its decorator |
|---|---|
| `train_nci60_10cell.py` | `NCIPreciousDataset` (**m21's 0.8007 script**) |
| `train_metal_lomo.py` | `LOMOSplitter` (**LOMO trainer**) |
| `wf146_chemprop_v2_lomo_v2.py` | `PerMetalRun` |
| `generate_section4.py` | `MetricSpec` (`frozen=True`) |

**Mode 2 — orphaned function tail (1 file).** The import block landed
*inside* `dock_in_process`'s body, between `import vina` and the rest of the
function, pushing the tail into the `except ImportError:` clause:

```
molmetal/scripts/m6_wf320h_vina_phase2.py:416
SyntaxError: 'return' outside function
```

### §1.2 Fixed

Decorator/block moves only — no logic changes. Diff is +34/−17 across the 5
files. Restored dataclasses carry their original fields, including
`MetricSpec`'s `frozen=True`.

### §1.3 Guard so it cannot recur silently

`molmetal/tests/test_script_import_health.py` — **14 tests, PASS**. Any script
carrying the Wf-416-A1 marker is auto-discovered (so future migrations opt
themselves in with no edit here) and import-tested in a fresh interpreter,
plus a fast static check for the exact clobber signature.

**This is the AGENTS.md §5.7 failure mode applied to scripts.** §5.7 exists
because a SCAFFOLD passed its unit tests while never being imported by the
trainer. Here a migration passed its own checks while the migrated scripts
were never imported at all. A `compile()` does not catch Mode 1 — only an
actual import does, which is why the guard spawns a subprocess.

**Negative control (a guard that has never failed is unproven).** Planting the
clobber signature produced exactly 2 failures — the static check *and* the
dynamic import — then the plant was removed and re-verified gone:

```
2 failed, 13 passed   (with planted clobber)
14 passed             (plant removed)
```

### §1.4 Verification

```
12/12 marked scripts import OK
4 restored dataclasses carry original fields (MetricSpec frozen=True)
194 pre-existing tests pass across the 7 suites covering these modules
2 failures in test_lomo_v2_cli.py are PRE-EXISTING and unrelated:
  it targets train_metal_lomo_v2.py (a different file), asserting CLI flags
  are present in --help; that file is not in this diff.
```

---

## §2 NCI-60 9-cell-line task axis

### §2.1 Why this cohort

`platinai_4cell` cannot carry a result — its labels are synthetic.
MEASURED, `molmetal/data/platinai_4cell.meta.json`:

> `"framing": "synthetic IC50 labels for pipeline test; NOT directly
> comparable to Rusanov 2025 Table S4-S5"`

So the per-cell AUC ≈ 0.50 that cohort produces is a **null test** — it shows
there is no label leakage — not a measurement of anything. (That is still
worth something: it is evidence the cell-line axis wiring is sound.)

`molmetal/data/nci60_precious_9cell.csv` carries **real measured** pIC50 from
MetalCytoToxDB: 7,823 rows / 5,207 unique SMILES / 9 cell lines / active rate
0.3818. Its nine lines include all four Rusanov 2025 reports (MCF7, A549,
A2780, A2780cis). **No cohort in the repo has both real labels and Pt** — this
is the one with real labels, so this is the one that can answer the question.

### §2.2 Architecture

`MultiTaskWassersteinModel` is a shared encoder plus one (regression +
classification) head pair per integer task id. Nothing about it is
metal-specific, so the task axis becomes the cell line: **9 heads over one
shared encoder**. That is the PECAN-style shared network the field uses for
multi-cell-line cytotoxicity (PECAN predicts 59 NCI-60 lines from a single
shared network), as opposed to nine independent per-cell models.

### §2.3 Split — deliberately m21's, and MEASURED identical

Reusing m21's protocol is the whole point: a different split would make any
number produced here incomparable to the shipped 0.8007.

`ScaffoldSplitter` (Bemis-Murcko, `largest_first`, 70/10/20, seed=0) is run on
the **unique canonical SMILES**, matching m21 (whose dataset is pivoted to
wide form, so its `.smiles` holds one entry per molecule). Splitting unique
molecules also makes cross-cell leakage structurally impossible.

MEASURED — our loader's split vs m21's protocol run side by side:

```
split   m21_ref     ours  identical  md5
train      3645     3645       True   4442bbfd7e44 == 4442bbfd7e44
val         521      521       True   19503c2260ba == 19503c2260ba
test       1041     1041       True   f941b3bca78b == f941b3bca78b

SPLIT IDENTICAL TO m21 PROTOCOL: True
SMILES spanning >1 split (cross-cell leakage): 0
```

Population match, both directions: `m21-only=0  longform-only=0  overlap=5207`.

### §2.4 Labels

The CSV's own `active` column is used verbatim. `prepare_nci60_data.py:221,289`
computes it as `ic50_um < ACTIVE_THRESHOLD_UM` with the **same 10 µM constant
m21 uses**, so the column *is* m21's label. Re-deriving it from pIC50
reproduces it only to **99.32%** (53/7823 boundary rows differ — an artifact of
pIC50 rounding). Comparability is this cohort's entire purpose, so the column
is used as shipped rather than recomputed.

### §2.5 What was added

| Change | Where |
|---|---|
| `DEFAULT_NCI60_9CELL_CSV` + `NCI60_9CELL_CELL_LINES` | trainer constants |
| `_resolve_default_csv_path` → `nci60_9cell` branch | trainer |
| `_SmilesOnlyAdapter`, `load_nci60_9cell_split_table`, `build_nci60_cell_line_axis` | trainer |
| `--dataset` choices += `nci60_9cell` | trainer |
| `--metals auto` → reads `cell_line` column (long form) | trainer |
| `--task-axis metal` on nci60 → loud `ValueError` | trainer (guard) |
| `configs/training/nci60_9cell_5seed_50ep_v1.yaml` | new config |

`build_nci60_cell_line_axis` deliberately takes **no** threshold/labeling
arguments — accepting a threshold and ignoring it would be a silent footgun.

### §2.6 §5.7 smoke — CPU, isolated

```
dataset nci60_9cell | task_axis cell_line | 7823 rows | 9 tasks | EXIT=0
per-cell rows: A549 2647, MCF7 1964, A2780 1095, HCT116 701, A2780cis 564,
               PC3 275, HL60 225, SKOV3 191, DU145 161   (sums to 7823)
per-split rows: train 5415 / val 804 / test 1604
active_rate 0.3818                      (matches CSV)
grad_norm pinned at 5.0                 (--max-norm wire-in confirmed live)
checkpoint isolation: cold-start at <out_dir>/checkpoints  (NOT the shared dir)
train_seconds 15.29                     (real training; the stale-resume bug
                                         had reported 0.0035)
```

Per-cell test AUC, 1 epoch, seed 0 — **context only, NOT a result** (1 epoch
against a 50-epoch baseline):

```
DU145 0.8333  A2780 0.6589  HL60 0.6552  A2780cis 0.6324  HCT116 0.6242
MCF7  0.5933  PC3   0.5517  A549 0.5079  SKOV3    0.4920
```

**§5.7 four invariants:**

| # | Invariant | Result |
|---|---|---|
| 1 | `parse_args` accepts the flags | **PASS** |
| 2 | Trainer imports + initialises cleanly | **PASS** |
| 3 | `decode_ratio > 0` | **N/A** — D-MPNN has no decoder (per §5.7 / Wf-399-A1) |
| 4 | `loss[0] > loss[-1]` | **PASS** — 2.6620 → 1.1062 (−58%) over 5 epochs |

Invariant 4 trace: `2.6620 → 1.5228 → 1.1530 → 1.0416 → 1.1062`. Not strictly
monotone (a late uptick), but §5.7 requires `loss[0] > loss[-1]`, which holds.

---

## §3 Findings worth carrying forward

### §3.1 The m21 headline is an 8-cell mean, not 9 — HONEST-NEGATIVE

`metrics/by_metric/nci60_precious_9cell_auc.json` reports
`value_overall_mean_auc = 0.8007` and `n_cells = 9`. But **DU145 is `null`
across all three seeds** — m21 could not measure it (161 rows, the smallest
cell). Verifying: the mean of the 8 non-null per-cell values is `0.8007`, i.e.
exactly the headline.

So **0.8007 is a mean over 8 cells.** Any comparison must be made on the same
8, or it is not a comparison. Worth noting for paper §4/§6 framing — the
metric's own description says "9-cell-line".

Interestingly, our run *does* produce a DU145 number (0.8333, 1 epoch), because
the main trainer does not apply m21's per-cell `n_test_min` gate. So DU145 is
measurable here, which is a small capability gain — but it also means DU145 is
not comparable to "null".

### §3.2 `model.hidden_dim` is a dead key — the platinai config never applied it

MEASURED: the argparse dest is `hidden` (`--hidden`, default `DMPNN_HIDDEN=128`).
A `model.hidden_dim` key has **no matching dest**, so the YAML loader skips it
silently and the value never reaches the model.

Consequence: `configs/training/platinai_pt_4cell_5seed_50ep_v1.yaml` sets
`hidden_dim: 300` and **has been running at 128 anyway**. Not fixed here (it
would change that config's behaviour); flagged for its owner. Our config uses
`hidden:`, which works.

### §3.3 `output.base_dir` does not set `out_dir`

MEASURED: `--config`-only invocation leaves `out_dir` at the trainer default
`molmetal/reports/wf131_p1_ru_auc_multitask` — `base_dir` only drives
start.sh's `{out_dir}` substitution. The platinai config has this gap too.
Harmless when launched via start.sh (which passes `--out-dir` explicitly), but
a `--config`-only run would write one cohort's results into another's tree.
Our config sets **both** keys.

### §3.4 `_nci60_scaffold_helpers.py` carries the same salted-hash defect — LATENT

`molmetal/scripts/_nci60_scaffold_helpers.py:63,89` uses Python's salted
`hash()` on strings:

```python
sc = f"__circular_{hash(smi) % 1024}__"          # line 63
label = f"scaffold_ood_group_{abs(hash(sc)) % 10**8}_n={len(idx_list)}"  # line 89
```

Same class as the platinai loader defect fixed earlier (Wf-A10): the value
differs per process, so the split (line 63) and the run's identity label
(line 89) are not reproducible.

**This is latent, not live** — verified no shipped metric contains a
`scaffold_ood_group_*` or `__circular_*` string:

```
grep -rn "scaffold_ood_group\|__circular_" metrics/ molmetal/reports/  → 0 hits
```

So no published number is contaminated. But m21's 0.8007 does **not** depend on
this file (it uses `ScaffoldSplitter`, measured deterministic across three
`PYTHONHASHSEED` values). Left unfixed here as out of scope; recorded so it is
not rediscovered from scratch.

### §3.5 Determinism of the split m21 rides on — MEASURED

Since the whole comparison rests on it: `ScaffoldSplitter` is deterministic
across processes. Three runs, `PYTHONHASHSEED` ∈ {0, 12345, random}:

```
train:6261:7edaad5f87ea|val:521:f927be155e80|test:1041:9961708057eb   (×3, identical)
```

(The `6261` there is *row*-level over all 7,823 long rows; the `3645` in §2.3
is *unique-molecule*-level. Different groupings, both stable.)

---

## §4 Standing directives

| Rule | Status |
|---|---|
| §5.1 GPU authorization | **0 GPU launches.** CPU smokes only. Launch is the user's. |
| §5.2 honest framing | Every number above is MEASURED with its source; the 1-epoch AUCs are labelled context, not results |
| §5.3 no paper work | 0 `paper/*` touches |
| §5.4 / §10 no wet-lab | 0 wet-lab planning |
| §5.5 filesystem-truth | `ls -la` evidence in §5 |
| §5.6 atomic write | closeout written via `atomic_write_file` |
| §5.7 flag smoke | 4/4 (inv 3 N/A by rule) |
| §5.8 token budget | work done inline (this wave predates the AGENTS.md trim). AGENTS.md is now 47,350 B, not 1.2 MB — subagent spawns work again (commit `6007efda6d`) |
| §5.9 single venv | all invocations via `.venv/bin/python` |
| YAML-only control | run config is YAML; no CLI hyperparameters |

---

## §5 ls -la evidence (§5.5 rule #2)

Raw `ls -la` at closeout time (2026-10-03), 8 files:

```
-rw-r--r-- 1 hugo hugo  13139 Oct  3 08:56 configs/training/nci60_9cell_5seed_50ep_v1.yaml
-rw-r--r-- 1 hugo hugo  26892 Oct  3 08:42 molmetal/scripts/generate_section4.py
-rw-r--r-- 1 hugo hugo  36465 Oct  3 08:43 molmetal/scripts/m6_wf320h_vina_phase2.py
-rw-r--r-- 1 hugo hugo 139550 Oct  3 08:51 molmetal/scripts/train_dmpnn_multitask_wasserstein.py
-rw-r--r-- 1 hugo hugo  21956 Oct  3 08:43 molmetal/scripts/train_metal_lomo.py
-rw-r--r-- 1 hugo hugo  47394 Oct  3 08:42 molmetal/scripts/train_nci60_10cell.py
-rw-r--r-- 1 hugo hugo  30199 Oct  3 08:43 molmetal/scripts/wf146_chemprop_v2_lomo_v2.py
-rw-r--r-- 1 hugo hugo   5750 Oct  3 08:44 molmetal/tests/test_script_import_health.py
```

Import health re-confirmed at closeout: **12/12 marked scripts import OK**
(`molmetal/tests/test_script_import_health.py` — 14 passed in 25.38s).

---

## §6 Cross-references

- `57d9ae94b5` — the repair commit (5 scripts + guard)
- `AGENTS.md` §5.7 — the same "unit tests pass while the integration point is
  never exercised" failure mode, applied here to script imports
- `TODO/completed/wfa10_platinai_task_axis.md` — the platinai cell-line axis
  and the synthetic-label finding that redirected this work
- `molmetal/data/platinai_4cell.meta.json` — the synthetic-label declaration
- `metrics/by_metric/nci60_precious_9cell_auc.json` — the 0.8007 baseline
- `molmetal/scripts/train_nci60_10cell.py` — the script whose split this reproduces

---

## §7 Not done / next

- **No GPU launch** — user's call. (An earlier revision of this line claimed
  the venv's torch was ROCm / `cuda_available=False`. **Re-measured and false:**
  `2.14.1+cu130`, CUDA available, matmul on `cuda:0` verified. See the
  correction in §0.)
- **The 50-epoch comparison is unrun.** Everything here is 1-epoch plumbing
  verification. Whether the main trainer's loss machinery beats 0.8007 is
  *unknown* and is the actual experiment.
- **Not fixed:** `_nci60_scaffold_helpers.py` salted hash (§3.4, latent);
  platinai's `hidden_dim` dead key (§3.2); platinai's `out_dir` gap (§3.3).
