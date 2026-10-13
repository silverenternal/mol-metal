# Wf-A2 m19 Rh fold diagnosis — 2026-10-11

**Question**: Is the m19 Rh fold AUC (~0.38, below random) a label-side/cohort-side defect or a model-side limitation?

**Verdict**: **COHORT_TRUNCATION_DOMINANT.** The Rh fold's sub-random AUC is driven by the `<=24h` exposure-time filter applied to the trainer's *default* CSV (`metal_lomo.csv`), which drops 90.5% of Rh rows and shifts the Rh test prior `0.468 -> 0.324`. The M1 lever (`--time-threshold 96`) was a **dead no-op** because its operand CSV is already truncated at 24h.

All numbers below are MEASURED and cite an on-disk path (§5.5). Machine-readable twin: `diagnosis.json`.

---

## 1. The defect: the default cohort is a 24h snapshot

`molmetal/scripts/train_metal_lomo_wasserstein.py` reads `DEFAULT_LOMO_CSV = molmetal/data/metal_lomo.csv` (line 180), then applies `CytotoxFilter(time_threshold=args.time_threshold)`.

`metal_lomo.csv` **already has a `Time(h)` column and is already filtered to <=24h**:

| metal | rows in `metal_lomo.csv` | rows in canonical source | loss |
|---|---:|---:|---:|
| Ru | 3,849 | 19,135 | 80% |
| Ir | 1,250 | 4,546 | 73% |
| Re | 166 | 868 | 81% |
| **Rh** | **108** | **1,134** | **90.5%** |
| **Os** | **87** | **1,118** | **92.2%** |

Rh `Time(h)` distribution in the default CSV: `{6h: 2, 24h: 106}` — **every row is <=24h**. A threshold sweep `{24, 48, 72, 96, 120, 999}` returns **Rh=108 for all of them**. The filter can only *remove* rows; it can never recover the 1,026 dropped Rh rows.

## 2. The filter also shifts the label prior (label-side)

Rh label balance by exposure window in the canonical source:

| window | n | active | pos_rate | unique ligands |
|---|---:|---:|---:|---:|
| <=24h (kept) | 108 | 35 | **0.324** | 51 |
| 24-48h | 489 | 207 | 0.423 | 125 |
| 48-72h | 289 | 158 | 0.547 | 131 |
| 72-96h | 245 | 129 | 0.527 | 119 |
| **<=96h (all real)** | **1,131** | **529** | **0.468** | **337** |
| all | 1,134 | 531 | 0.468 | 339 |

The discarded long-exposure rows have a **higher** positive rate (0.42–0.55) than the kept <=24h rows (0.324). So the filter does not merely shrink Rh — it selects a **non-representative** Rh sub-distribution and removes 288 of 339 ligands.

## 3. Proof that M1 was a dead lever

`configs/wf_a2_m19_time96.yaml` set `data.time_threshold: 96`. Verification:

- `_parse_args(['--config', ...])` → `time_threshold = 96.0` (the key **did bind**) and `csv = molmetal/data/metal_lomo.csv`.
- `molmetal/reports/wf_a2_m19_time96/master.log:6` → `dataset loaded: n=5271 metals={'Rh': 108, ...}` — **identical to the baseline**.
- Rh fold result: **0.3845 ± 0.0077**, `n_test=108`.

The threshold bound correctly but operated on a pre-truncated CSV. M1's headline `0.5186` vs baseline `0.5575` is other-fold/seed noise, **not** a cohort change.

## 4. The shipped paper number uses the truncated fold

`metrics/by_metric/metal_lomo_auc_wasserstein.json`: `value = 0.5575 ± 0.0949`, **Rh fold = 0.3812**, `n_test = 108`. The Rh bottleneck in the shipped m19 metric is the truncated fold.

## 5. Counter-evidence: a full-Rh cohort scores 0.80–0.88

`wf131_p3_pt_4cells` used `--csv .data/molmetal/metalcytotox_5metal_multitask_v4.csv`. That CSV **has no `Time(h)` column**, so `_apply_filters` **skips the time filter** (guard at `molmetal/data/cytotox.py:314-315`). Result:

| run | Rh n_test | Rh AUC | seeds | protocol |
|---|---:|---:|---:|---|
| `wf131_p3_pt_4cells` | 1,118 | **0.7971 ± 0.0238** | 5 | 50ep / hidden300 / batch256 / l2-guard |
| `wf131_p3_deferred_runs` | 1,618 | **0.8798 ± 0.0081** | 2 | — |
| `wf131_p3_lomo_full` (trunc) | 108 | 0.3812 ± 0.0270 | 5 | current family |
| `wf_a2_m19_time96` (M1) | 108 | 0.3845 ± 0.0077 | 5 | current family |
| `m19_rh_fix_2026_10_08` | 51 | 0.4056 ± 0.0597 | 5 | rh_loso halved |
| `wf125_p1 scaffold_smoke` | 1,618 | 0.3211 | 1 | **outlier** |

**Caveats (honest, §5.2):**
1. The big-cohort runs differ in **more than cohort** (50ep/hidden300/batch256/l2-guard vs the current 30ep/hidden128/batch64). `0.7971` is **not** a clean single-variable lift of the current protocol — it establishes feasibility, not a measured lift.
2. The `wf125_p1 scaffold_smoke` outlier (`0.3211` at `n_test=1618`) shows cohort size alone does not guarantee a good AUC — the split/protocol matters.
3. The v4 CSV lacks `Time(h)`, so it is an **unfiltered** cohort (all exposures mixed). A principled repair uses the **canonical source at <=96h**, not the unfiltered v4.

## 6. Proposed next experiment (cohort repair)

Single change vs `wf_a2_m19_time96.yaml`:

```yaml
data:
  csv: .data/molmetal/MetalCytoToxDB.csv   # canonical, HAS Time(h)
  time_threshold: 96                        # now actually operative
```

Verified via the real builder: `_build_dataset(csv=canonical, time_threshold=96)` → **n=25,802** (Rh 1,131, Os 1,100, Ru 18,189, Ir 4,514, Re 868), 0.1s, no GPU.

**Framing**: this is a **COHORT REPAIR** producing a **NEW protocol number** — it is **not comparable** to the 0.5575 baseline (which used the truncated 5,271-row cohort). Per §5.2 it must not be reported as a lift. Expected wall ~35 min (5× M1's cohort; M1 took ~7 min).

---

**MEASURED** (every number cites an on-disk path). NO paper work (§5.3); NO wet-lab (§5.4).
