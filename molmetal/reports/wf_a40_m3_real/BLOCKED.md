# Wf-A40-3 m3 PlatinAI 3-cell re-measurement — BLOCKED on data quality

**Status:** ❌ BLOCKED (no training launched per AGENTS.md §5.2 honest framing + §5.5 filesystem-truth)
**Date:** 2026-10-15
**Author:** Hugo Lee (via ultracode subagent — Wf-A40-3 closeout)
**Cross-ref:** `docs/HONEST_NEGATIVE_CATALOG.md` §6.65 (2026-10-15)

---

## 1. Brief vs. on-disk premise

The F4 brief requested:

```bash
# First filter the real pool to 3 cells
python -c "import pandas as pd; df = pd.read_csv('.../platinai_4cell_real_label_pool_with_smiles.csv'); df3 = df[df.cell_line.isin(['MCF-7','A549','A2780cis'])].to_csv('/tmp/platinai_3cell.csv')"

# Then run m3 trainer
python molmetal/scripts/train_cisplatin_resistance.py \
    --csv /tmp/platinai_3cell.csv \
    --out-dir molmetal/reports/wf_a40_m3_real \
    --seeds 0 1 2 3 4 --epochs 100 --device cpu
```

The brief assumed: (a) the source CSV is at `/home/codes/mol-metal/platinai_4cell_real_label_pool_with_smiles.csv` (chr-decoded); (b) the pool has MCF-7 rows; (c) the pool has SMILES for enough rows to run D-MPNN; (d) the trainer accepts `--csv`; (e) the trainer accepts long-format input.

**All 5 assumptions are wrong on disk** per §5.5 filesystem-truth verification.

---

## 2. On-disk verification (AGENTS.md §5.5)

### 2.1 Source CSV path

The chr-decoded source path is `/home/codes/mol-metal/platinai_4cell_real_label_pool_with_smiles.csv` — the `/hugo/` segment of the actual working directory `/home/hugo/codes/mol-metal/` is missing from the chr() sequence. The actual SMILES-enriched pool is at:

```
.data/molmetal/platinai_4cell_real_label_pool_with_smiles.csv  (240,662 bytes, 2054 lines)
```

created by Wf-A40-2 (commit `2efd1ccf`). Verified via `ls -la .data/molmetal/platinai_4cell_real_label_pool_with_smiles.csv`.

### 2.2 SMILES coverage diagnostic

```bash
$ /home/hugo/codes/mol-metal/.venv/bin/python -c "
import pandas as pd
df = pd.read_csv('.data/molmetal/platinai_4cell_real_label_pool_with_smiles.csv')
print('rows:', len(df))
print('cell_line_long unique:', df['cell_line_long'].unique())
print('MCF-7 rows:', (df['cell_line_long']=='MCF-7').sum())
print('smiles != __no_smiles__:', (df['smiles'] != '__no_smiles__').sum())
"
# rows: 2053
# cell_line_long unique: ['A549' 'A2780' 'A2780cis']
# MCF-7 rows: 0
# smiles != __no_smiles__: 1
```

**Three findings on disk:**
1. **SMILES coverage 1/2053** (0.05%) — `MB-Pt-002238-1` is the only row with a real SMILES; the other 2052 have `smiles = '__no_smiles__'`. Wf-A40-2 commit message verbatim: *"F2 partial - n_with_smiles=1/2053"*.
2. **MCF-7 absent** — `cell_line_long` ∈ {`A549`, `A2780`, `A2780cis`}. Zero MCF-7 rows. The m3 protocol requires MCF-7/A549/A2780cis (drop A2780 per `molmetal/scripts/train_cisplatin_resistance.py:108` `CELL_LINES = ('MCF-7', 'A549', 'A2780cis')`).
3. **Long-format mismatch** — the pool has columns `compound_code, canonical_smiles, cell_line_long, cell_line_short, value_mean, value_sd, doi, method, incubationTime, label_source, smiles` (one row per compound×cell_line). The trainer at `molmetal/scripts/train_cisplatin_resistance.py:351-392` `load_dataset()` hardcodes the wide-format column list `("smiles", "ic50_mcf7_nm", "ic50_a549_nm", "ic50_a2780cis_nm")`.

### 2.3 Trainer argparse audit

```bash
$ grep -nE "add_argument|--data|--csv" molmetal/scripts/train_cisplatin_resistance.py
# 487:    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
# 488:    p.add_argument("--encoder", type=Path, default=DEFAULT_ENCODER_CKPT)
# 489:    p.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
# 490:    p.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
# 491:    p.add_argument("--epochs", type=int, default=EPOCHS)
# 492:    p.add_argument("--batch", type=int, default=BATCH)
# 493:    p.add_argument("--lr", type=float, default=LR)
# 494:    p.add_argument("--eval-every", type=int, default=EVAL_EVERY)
# 496:    p.add_argument("--limit", type=int, default=None, ...)
# 498:    p.add_argument("--config", type=str, default=None, ...)
# (no --csv, no --out-dir, no --device)
```

The trainer accepts `--data`, `--encoder`, `--out`, `--seeds`, `--epochs`, `--batch`, `--lr`, `--eval-every`, `--limit`, `--config`. **No `--csv`, no `--out-dir`, no `--device`** flags exist. The trainer uses GPU via `CUDA_VISIBLE_DEVICES` env var (per `_cuda_default.ensure_cuda_visible_devices_default()` at `molmetal/scripts/train_cisplatin_resistance.py:70`), not from a `--device` CLI flag.

---

## 3. Brief CLI failure mode (3 sequential failures)

The brief's CLI would fail at 3 sequential points:

1. **FS lookup**: the chr-decoded source path is missing `/hugo/` so `pd.read_csv(...)` raises `FileNotFoundError` on a non-existent path. Even corrected to `.data/molmetal/platinai_4cell_real_label_pool_with_smiles.csv`, the pool has no MCF-7 column per §2.2.
2. **Filter**: `df.cell_line.isin([...])` raises `KeyError: 'cell_line'` — the actual column is `cell_line_long` (and `cell_line_short`), per §2.2.
3. **Trainer argv**: `--csv /tmp/platinai_3cell.csv` → `argparse.ArgumentError: unrecognized argument: --csv` (the trainer's flag is `--data`). Even if corrected to `--data`, the long-format `/tmp/platinai_3cell.csv` lacks `ic50_mcf7_nm`/`ic50_a549_nm`/`ic50_a2780cis_nm` columns, and `load_dataset` would silently skip every row (per `molmetal/scripts/train_cisplatin_resistance.py:364-371`: `if v == "" or v is None: continue`). The 3-cell scaffold split would yield empty fold splits and the test AUC would be NaN — the trainer would NOT crash; it would produce a JSON with `test_auc = NaN` for every seed (this is the silent-degradation pattern that §6.51 documents).

**Per AGENTS.md §5.2 honest framing + memory `no-self-made-tables.md` + `help-probe-misses-broken-compute-path.md`, an --help probe is NOT sufficient evidence that the run succeeded — running the smoke + asserting the numbers is required. This BLOCKED training was never launched because the upstream data is incomplete at the source; running it would either crash (argparse) or produce NaN folds (silent). Both outcomes yield no usable m3 number, and a fabricated success would be an honest-negative per §5.2.**

---

## 4. m3 baseline reference (synthetic, stays on disk)

Per `metrics/by_metric/platinai_3cell_resistance.json` (timestamp 2026-09-26):

```json
{
  "metric": "platinai_3cell_resistance",
  "value_overall_mean_auc": 0.545851474519129,
  "value_a2780cis_auc": 0.6298878799735552,
  "per_cell_line": {
    "MCF-7": { "auc_mean": 0.5030384976752029 },
    "A549": { "auc_mean": 0.5046280459086289 },
    "A2780cis": { "auc_mean": 0.6298878799735552 }
  },
  "n_rows": 4078,
  "timestamp": "2026-09-26",
  "model": "dmpnn_3cell_resistance_with_sidefeat",
  "focal_loss_gamma": 2.0,
  "split": "scaffold_70_10_20",
  "encoder_init": "tmqm_pretrained_encoder.pt (12/12 backbone tensors)"
}
```

This is the **synthetic** baseline from `molmetal/data/platinai_4cell.csv` (4078 synthetic rows). It is the m3 reference number that the F4 brief expected to be replaced with a REAL measurement. **Per §6.65c, the replacement is BLOCKED on data quality; the synthetic 0.5459 stays as the m3 baseline.**

No re-emission of `metrics/by_metric/platinai_3cell_resistance.json` was performed (would be fabrication per AGENTS.md §5.2).

---

## 5. Remediation hooks (DEFERRED)

To unblock m3 PlatinAI re-measurement, the following MUST be resolved in a future Wf-A40-X (user-OK-gated per AGENTS.md §5.1):

1. **Re-attempt SMILES fetch**: the Wf-A40-1 page-850+ `get-substances` scan returned 0 hits for the target `compound_code` set (likely imageId format mismatch: MB-Finder pads to `MB-Pt-NNNNN` 6-digit while PlatinAI has non-padded or `MB-Pt-NNNNN-N` suffix forms). A future scan with imageId-format-aware matching + PSILI/PubChem fallback is DEFERRED.
2. **MCF-3 source**: the real PlatinAI pool as fetched from MB-Finder has 0 MCF-7 rows. Either (a) re-fetch MCF-3 data with different query parameters, or (b) drop MCF-3 from the m3 protocol (m3 becomes 2-cell = AUC-A549 + AUC-A572-A2780cis — note this is a protocol change, not a measurement change).
3. **Long→wide pivot + µM→nM conversion wrapper**: write a `wf_a40_pivot_wrapper.py` that pivots the long-format pool to the trainer's wide-format columns, with `value_mean * 1000` for µM → nM conversion. This is a low-risk code change, but is MEANINGLESS until blockers 1 and 2 are resolved.
4. **Re-run m3 trainer on corrected pool**: after 1+2+3, run `molmetal/scripts/train_cisplatin_resistance.py --data <pivoted_pool.csv> --seeds 0 1 2 3 4 --epochs 100 --batch 64 --lr 5e-4 --out <NEW JSON>` (CPU-only per the standing "gpu先别动" directive).

---

## 6. Files NOT created (per §5.2 + §6.51 silent-degradation pattern)

Per AGENTS.md §5.2 honest framing + §6.51 + memory `no-self-made-tables.md`, the following files are **NOT created** because no training was launched:

- `molmetal/reports/wf_a40_m3_real/run.log` — would be empty or contain the 3-failure-mode trace; not informative
- `molmetal/reports/wf_a40_m3_real/per_seed_metrics_seed*.json` — would contain NaN fold AUCs (silent-degradation per §6.51)
- `molmetal/reports/wf_a40_m3_real/training_summary.json` — would aggregate NaN to NaN mean

Only `BLOCKED.md` (this file) is created. The m3 metric JSON `metrics/by_metric/platinai_3cell_resistance.json` is **NOT re-emitted**; the synthetic 0.5459 stays on disk per §6.65c.

---

## 7. Pre-flight + git status

Pre-flight v3 92/0/2/94 baseline PRESERVED at Wf-A40-2 (`2efd1ccf`) timestamp. This edit is docs-only and follows the same `SKIP_PREFLIGHT=1` documented escape-hatch pattern as Wf-A40-1 (`6f5fb04b`) per AGENTS.md §1.4 + the pre-flight gate's documented stale-flag rule (the stale entries are m7/m19/m21/etc. trainer-metric staleness, NOT data-pipeline staleness — this docs-only edit does not touch training scripts / metrics / configs).

`git status` expected after this commit:
```
M  docs/HONEST_NEGATIVE_CATALOG.md     (status line + §6.65 body)
?? molmetal/reports/wf_a40_m3_real/    (NEW dir with BLOCKED.md)
```

No `paper/*` touches per AGENTS.md §5.3. No wet-lab per §5.4 + §10. CPU-only docs + manifest surface per AGENTS.md §5.1.

---

## 8. Source-of-truth refs

- `docs/HONEST_NEGATIVE_CATALOG.md` §6.65 (this Wf-A40 closeout — MEASURED-vs-PROJECTED ledger + active count delta)
- `docs/HONEST_NEGATIVE_CATALOG.md` §6.64c (previous brief's "synthetic 0.7105" verified NOT in narrative — same premise error pattern)
- `docs/00_MODEL_NARRATIVE.md` §2.1 line 64 (existing m3 row is MoleculeNet bace pool=mean, NOT PlatinAI 3-cell resistance)
- `molmetal/scripts/train_cisplatin_resistance.py:108` (CELL_LINES = ('MCF-7', 'A549', 'A2780cis'))
- `molmetal/scripts/train_cisplatin_resistance.py:111-115` (IC50_COLS = ('ic50_mcf7_nm', 'ic50_a549_nm', 'ic50_a2780cis_nm'))
- `molmetal/scripts/train_cisplatin_resistance.py:351-392` (load_dataset long-format rejection)
- `molmetal/scripts/train_cisplatin_resistance.py:485-505` (argparse — `--data` not `--csv`)
- `.data/molmetal/platinai_4cell_real_label_pool_with_smiles.csv` (2053 rows, 1 SMILES, 0 MCF-7)
- `.data/molmetal/platinai_4cell_real_label_pool_with_smiles.meta.json` (coverage stats: n_with_smiles=1/2053)
- `molmetal/scripts/wf_a40_smiles_fetcher.py` (Wf-A40-1 fetcher — 0 effective hits from page-850+ scan)
- `molmetal/scripts/wf_a40_smiles_integrator.py` (Wf-A40-2 integrator — joined cache into pool)
- `git log master` (Wf-A40-2 `2efd1ccf` is the SMILES-integrator commit; no F3 m2 re-measurement commit exists)
- `metrics/by_metric/platinai_3cell_resistance.json` (synthetic 0.5459 baseline, 2026-09-26, NOT re-emitted)