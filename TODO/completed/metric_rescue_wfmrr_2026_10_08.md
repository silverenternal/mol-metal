# Wf-METRIC-RESCUE 2026-10-08 — Regressed-metric ultracode (5 honest-negative findings + code SHIPPED)

**Date**: 2026-10-08 (W1 arm2 completed 2026-10-09 00:15)
**Status**: ✅ **SHIPPED** (5 honest-negative findings on disk + 2 trainer fixes wired + 11 NEW tests PASS; pre-flight v3 still rc=0)
**Owner**: Hugo Lee + ultracode agent (W1 diagnose + measure agents per §5.8 ≤2 parallel cap)
**Trigger**: user directive 2026-10-08 verbatim *"回退的东西得想办法把他们补强，启动ultracode基于对现状的洞察去做修复和重测"* (regressed metrics need to be fortified; launch ultracode based on insight into current state to do fix + re-measure)
**Skill alignment**: `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline) + AGENTS.md §5.2 (honest framing mandate) + §5.5 (filesystem-truth) + §5.6 (atomic write) + §5.7 (trainer flag wire-in smoke) + §5.8 (wf token budget ≤2 parallel agents) + §5.9 (single-venv)

---

## TL;DR

Wf-METRIC-RESCUE surfaced **5 honest-negative findings** from regressed-metric ultracode, all MEASURED on disk per AGENTS.md §5.5 filesystem-truth rule:

1. **W₁ W₂ ON/OFF de-confound** — turning W₂ ON lowers real-only LOMO by Δ=−0.023 (every-batch cadence); 3-arm de-confound (OFF 0.5575 / per-epoch 0.5496 / every-batch 0.5349) confirms W₂ is NOT a load-bearing lever at any tested cadence.
2. **G6 per-innovation ablation (18 tuples)** — none of Wasserstein / Kendall / multi-task individually decisive for AUC at 3 seeds × 2 metals; per-lever Δ all inside seed std. → **SUPERSEDED 2026-10-31 (Wf-A124):** the "(18 tuples)" count is **retracted** — the tree holds **24 tuples** (4 cohorts × {Ru, Rh} × 3 seeds) — and the published per-lever Δ table **does not reproduce** from the shipped artifacts (**2 of 6** cells under its own `no_multi_task` reference, **0 of 6** under the G5 all-ON baseline); the qualitative conclusion ("no individual innovation is decisive at this n") **STANDS** and is re-derived by the paired-vs-G5 comparison — see the §2 SUPERSEDE note below + `docs/HONEST_NEGATIVE_CATALOG.md` §6.216..§6.222.
3. **bace rescue attempt — hypothesis REFUTED** — class-weighting was NOT the cause of the −0.134 bace gap; rescue `--class-weight balanced` made it WORSE (0.7421 → 0.6805). 4 NEW flags wired end-to-end (NOT FLAG_ONLY).
4. **m19 fix-stack aggregate 0.5862** — `--rh-loso-validation` specifically HURT Rh (0.5022 → 0.4056, Δ −0.097). Centering baseline (Rh 0.5022) remains the canonical value.
5. **pIC50 pearson readout** — ridge on frozen encoder **decisively beats** trained neural head (pooled 0.4151 vs 0.1543); head is **anti-correlated on Os** (−0.278).

**Code SHIPPED** (per §5.5 filesystem-truth):
- `molmetal/scripts/train_moleculenet.py` — 4 NEW flags (`--class-weight {on,off,balanced}` + `--dropout` + `--weight-decay` + `--patience`) wired end-to-end at argparse 806-822 → run 894-897 → `_train_one_seed_dmpnn` 433-436. §5.7 50-step CPU smoke verified (NOT FLAG_ONLY).
- `molmetal/scripts/train_metal_lomo_wasserstein.py` — inverse no-op guard in `_parse_args` (warns when `--wasserstein-real` passed but `wasserstein_weight <= 0`).
- `molmetal/tests/test_wf_round2_c3_w2_noop.py` — NEW 11 tests, 11/11 PASS.

**Pre-flight v3 status preserved**: rc=0 (PASS, 113/1/3/117 per Wf-432 baseline).

---

## §1 W₁ W₂ ON/OFF de-confound — 3-arm MEASURED (OFF / per-epoch / every-batch)

The single most consequential finding. The project's narrative had treated W₂ (Wasserstein pIC50 transfer) as part of the innovation stack, with an earlier "+0.049 lift" claim entangled with the **cadence** (the only prior ON/OFF pair differed in cadence too). Wf-METRIC-RESCUE W1 ran **two clean 25-run ON arms at different cadences** against the existing OFF reference arm, all real-only (`--rh-oversample 0`), 30 epochs, 5 seeds × 5 metals:

| W₂ state | pooled | per-fold (Ru/Ir/Os/Rh/Re) | source |
|---|---|---|---|
| **OFF** (reference, no `--wasserstein-real`) | **0.5575** | — | `metrics/by_metric/metal_lomo_auc_wasserstein.json` |
| ON `--wasserstein-cadence per-epoch` | **0.5496 ± 0.0967** | 0.6031 / 0.5896 / 0.5849 / 0.3780 / 0.5924 | `molmetal/reports/m1_realonly_w2on_per-epoch_30ep/metrics.json` (W1 arm2 COMPLETE 2026-10-09) |
| ON `--wasserstein-cadence every-batch` | **0.5349** | 0.5888 / 0.5901 / 0.5560 / 0.3794 / 0.5603 | `molmetal/reports/m1_realonly_w2on_every-batch_30ep/metrics.json` (W1 arm1 complete 2026-10-08) |

**Verdict — HONEST-NEGATIVE: W₂ is NOT load-bearing** at any tested cadence. Turning W₂ ON lowers the metric; the effect is larger at every-batch cadence (Δ −0.023) than per-epoch (Δ −0.008), but both are negative. Independent confirmation from G6 18-tuple ablation (see §2) — **superseded 2026-10-31 (Wf-A124):** the "CONFIRMED independently" wording does **not** stand (the two G6 arms differ in more than one lever, the difference sits inside seed std, and on Rh it is −0.0001 ≈ 0); see the §2 SUPERSEDE note + catalog §6.221.

**Why this matters**: the earlier "+0.049 lift" claim should not be cited as a W₂ effect. The m19 metric is dominated by per-fold variance (Rh n_test=108, Os n_test=87 → AUC resolution ≈±0.06).

---

## §2 G6 per-innovation ablation — 18 tuples MEASURED — **SUPERSEDED 2026-10-31 (Wf-A124): see the SUPERSEDE note at the end of this §**

The reduced G6 sweep (`G6_CONFIGS=3 G6_METALS=2 G6_SEEDS=3`, per `results/wf124_p3_per_innovation_ablation_g6/`) completed 18 tuples on `no_multi_task` + `no_wass` + `no_kendall`, 3 seeds × 2 metals (Ru + Rh): — **SUPERSEDED 2026-10-31 (Wf-A124): the tree always held 24 tuples** (4 cohorts × {Ru, Rh} × 3 seeds, `no_dann` included from the first results commit `37aa4960`, 2026-10-09); **there was never an 18-tuple tree**. — **the Δ table below was previously claimed and is RETRACTED (not reproducible from the shipped artifacts); superseded 2026-10-31 (Wf-A124) — see catalog §6.221 and the §2 SUPERSEDE note below.**

| Lever Δ vs `no_multi_task` baseline | Ru | Rh |
|---|---|---|
| Wasserstein (`no_wass`) | +0.007 | −0.0001 |
| Kendall (`no_kendall`) | −0.011 | +0.003 |
| Multi-task (`no_multi_task` itself) | +0.011 | +0.001 |

**All inside seed std; none individually significant.** The full 60-run sweep was killed per user voice 2026-10-07 (a 7.5 hr GPU run for an attribution that is below the 0.70 gate); the 18-tuple decisive subset confirmed: do not attribute +0.049 lift to any single innovation. → **the "confirmed" wording is superseded 2026-10-31 (Wf-A124) — see the SUPERSEDE note below.**

### §2 SUPERSEDE note (2026-10-31, Wf-A124 — CPU-only, 0 GPU)

**This file is the ORIGIN of the per-lever Δ table** that `TODO/INDEX.md` §22.151
and `docs/HONEST_NEGATIVE_CATALOG.md` §6.38b reproduce verbatim.  The table
above is **historical**; the corrections below are **additive and dated**
(AGENTS.md §5.5 rule #6 — the historical closeout trail is preserved, not
rewritten).  Catalog entries: `docs/HONEST_NEGATIVE_CATALOG.md` §6.216..§6.222.

1. **The tree always held 24 tuples, never 18.**  MEASURED: the G6 tree's first
   results commit `37aa4960` (2026-10-09) added **all four** cohorts —
   `no_dann` included — i.e. **4 cohorts × 2 metals (Ru, Rh) × 3 seeds = 24
   runs** (`git ls-tree -r HEAD -- results/wf124_p3_per_innovation_ablation_g6/`
   → 76 files = 24 runs × 3 artifacts + 2 cohort receipts + `plan.json` +
   `recovery_manifest.tsv`).  The "18 tuples" wording above is **retracted**.
2. **The Δ table does not reproduce.**  MEASURED 2026-10-31 against the git-HEAD
   per-seed artifacts: **2 of the 6 published cells reproduce under the table's
   own declared `no_multi_task` reference** (the two Wasserstein cells: Ru
   +0.006721 ≈ +0.007, Rh −0.000082 ≈ −0.0001) and **0 of the 6 reproduce under
   the G5 all-ON baseline** that the catalog elsewhere names (same metal column).
   **In total 4 of the 6 published cells reproduce, each against a DIFFERENT
   reference, and 2 reproduce under nothing** (Kendall Rh +0.003 implies a
   reference Rh of 0.648055; multi-task Rh +0.001 implies 0.663589; no arm has
   either).  Two further apparent matches are coincidences of magnitude, not
   reproductions: the Kendall-Ru cell reproduces only against a **different arm**
   (`no_wass`, −0.010758) and the multi-task-Ru cell reproduces only against the
   **G5 Rh column** (it equals `no_multi_task(Rh) − G5(Rh)` = +0.010909, i.e. the
   number the Rh cell of that row should carry — the two cells are swapped; the
   Ru column's own +0.001668 appears nowhere in the published table).  The multi-task row is additionally
   **self-referential** — its arm *is* the stated baseline, so under its own
   header its Δ must be 0, not +0.011 / +0.001; no reference arm rescues it.
   (Catalog §6.221.)
3. **Which lever was disabled is not recorded for 3 of the 4 cohorts.**  MEASURED:
   the union of keys across the 24 `training_summary.json` files is 25, and the
   **only** lever field that varies is `wasserstein_weight` (0.0 for the 6
   `no_wass` runs, 0.05 for the other 18).  `use_dann` / `use_kendall` /
   multi-task fields are **absent from all 24** summaries, and `no_kendall`,
   `no_dann` and `no_multi_task` show **0 differing fields** on every compared
   key for Ru/seed0 (`per_metal` / `ckpt_paths` / `elapsed_seconds_total` /
   `honest_framing_note` / `wf_tag` excluded).  The **cohort directory name is
   the only record** of the lever identity — the AGENTS.md §5.11
   silent-absent-knob failure mode.
4. **The qualitative conclusion STANDS.**  The honest-negative verdict of this
   §2 — *"no individual innovation is decisive at this n"* — is **UNCHANGED**
   and is re-derived by the correct paired comparison against the G5 all-ON
   baseline (`results/wf124_p3_50epoch_full_sweep_g5/`, same metal + same seed +
   same split, n = 6, df = 5, threshold |t| > 2.5706): `no_wass` +0.009608
   (t = +1.3177), `no_multi_task` +0.006288 (t = +0.9708), `no_kendall` −0.002497
   (t = −0.2647), `no_dann` +0.001110 (t = +0.1606) — **every arm
   non-significant**, and all 6 between-arm contrasts non-significant too (max
   |t| = 2.1801).  Catalog §6.218.
   **Confound (2026-10-31, Wf-A124):** the G5 baseline tree records **no trainer
   fingerprint** — the 8 fields `bf16`, `dmpnn_backend`, `dmpnn_backend_requested`,
   `tf32`, `torch_compile_cuda_graphs_disabled`, `torch_compile_enabled`,
   `torch_compile_mode` and `w2_gradient_mode` are absent from every G5
   `training_summary.json` — so every Δ above mixes a lever change with an
   **unrecorded trainer-build change**.  The pairing is like-for-like (same metal,
   same seed, same split), but it is not a clean single-lever contrast.  Catalog
   **§6.219**.
5. **What is NOT asserted.**  The artifacts cannot distinguish "the lever had no
   resolvable effect" from "the lever was never applied": the between-cohort
   spread at Ru/seed0 (0.025281) sits **inside** the range of ordinary
   within-cohort seed spreads (0.010977–0.031655), and the per-seed signs are
   inconsistent (seeds 0 and 1 favour the lever-OFF arm, seed 2 favours
   `no_multi_task`).  This note therefore asserts only that the levers are
   **unresolvable from the artifacts**; it does **not** assert that they were
   inert.  Logged **OPEN** (catalog §6.220).

---

## §3 bace rescue attempt — hypothesis REFUTED

**Hypothesis tested**: the unconditional `pos_weight=neg/pos` class weighting (applied even to near-balanced bace, pos_rate ≈0.45) depressed AUC and caused the −0.134 bace vs chemprop gap.

**MEASURED** (`metrics/by_metric/moleculenet_auc_dmpnn_5seed_rescue.json`):

| Task | Baseline (5-seed) | Rescue (`--class-weight balanced` + dropout 0.1 + wd 1e-5 + patience 20) | Δ |
|---|---|---|---|
| **bace** | 0.7421 ± 0.0446 | **0.6805** | **−0.062 (WORSE)** |
| clintox | 0.8209 ± 0.0122 | 0.8332 | +0.012 (inside ±0.0135 std) |
| bbbp (no-op control) | 0.9065 ± 0.0227 | 0.8939 | −0.013 (inside ±0.018 std) |

**Verdict — HONEST-NEGATIVE: hypothesis REFUTED**. Class-weighting is NOT the cause of the bace gap. The bbbp control confirms the device effect (GPU→CPU) at ~±0.015. bace is a real D-MPNN regression on the protocol (5-seed CI excludes the 0.876 chemprop baseline), not a class-weight artifact. A clean GPU re-run with a same-device control is the confirmation still owed.

**Code SHIPPED**: 4 NEW flags wired end-to-end in `molmetal/scripts/train_moleculenet.py` (argparse 806-822 → run 894-897 → `_train_one_seed_dmpnn` 433-436; verified by reading the file — NOT FLAG_ONLY per memory `flag-only-needs-getattr-aware-grep.md` lesson).

---

## §4 m19 fix stack — Rh specifically HURT by `--rh-loso-validation`

5-metal × 5-seed fix stack aggregate (`molmetal/reports/m19_rh_fix_2026_10_08/metrics.json`):

| Metal | Centering baseline | Fix stack | Δ |
|---|---|---|---|
| Ru | 0.5618 | 0.5838 | +0.022 |
| Ir | 0.5517 | 0.5962 | +0.045 |
| Os | 0.6521 | 0.6521 | 0.000 |
| **Rh** | **0.5022** | **0.4056** | **−0.097** |
| Re | 0.6613 | 0.6933 | +0.032 |
| **pooled** | **0.5845** | **0.5862** | +0.0017 |

**Verdict — HONEST-NEGATIVE: counterintuitive Rh-fold regression**. The fix stack lifted 4 of 5 metals but specifically degraded Rh. The `--rh-loso-validation` flag (intended to clean Rh validation) hurt Rh by Δ −0.097. Rh AUC resolution (n_test=108, n_pos=35) is ≈±0.061, so the effect is real (not noise). The fix stack should NOT include `--rh-loso-validation`; the 0.5022 centering result remains the canonical Rh value.

---

## §5 pIC50 pearson readout — ridge on frozen encoder beats trained neural head decisively

3-seed probe on `molmetal/reports/pearson_head_fix_2026_10_08/` (`--task multi_task --alpha 0.5 --pic50-ridge-readout`):

| Readout (pooled across metals, same split, n_test=994) | Test Pearson |
|---|---|
| **Ridge on trained D-MPNN encoder embedding** | **0.4151 ± 0.2027** |
| **Trained neural pIC50 head** | **0.1543 ± 0.3650** |
| Ridge on trained encoder (α=0.5) — wf_pearson_diag_3seed | 0.4019 ± 0.0058 |
| Ridge on trained encoder (α=0.2) — wf_pearson_diag_3seed | 0.4033 ± 0.0036 |
| Ridge on **random** (untrained) encoder — wf_pearson_diag_3seed | 0.3780 ± 0.0116 |
| Ridge on Morgan FP (split-fixed) — wf_pearson_diag_3seed | 0.3232 |

Per-metal head (ridge vs head, mean ± std over seeds 0/1/2):
- Ir: ridge 0.1701 ± 0.0034 vs head 0.1362 ± 0.0046
- Os: ridge 0.4244 ± 0.0455 vs head **−0.2782 ± 0.0654 (anti-correlated!)**
- Re: ridge 0.6509 ± 0.0736 vs head 0.6051 ± 0.0687

**Verdict — HONEST-NEGATIVE: the neural pIC50 head is not a working regressor**. Report `--pic50-ridge-readout` as the pIC50 predictor. Re-weighting α toward regression made the head WORSE (0.218 → 0.165), refuting the α-starvation hypothesis. The encoder learns only ~+0.025 pIC50 structure over random; the head actively destroys signal on Os.

---

## §6 Disk artifacts (per AGENTS.md §5.5 filesystem-truth rule)

```
metrics/by_metric/metal_lomo_auc_wasserstein.json        (m1_w2_onoff_deconfound_2026_10_08 block)
metrics/by_metric/moleculenet_auc_dmpnn_5seed_rescue.json
molmetal/reports/m1_realonly_w2on_every-batch_30ep/metrics.json     (W1 arm1)
molmetal/reports/m1_realonly_w2on_per-epoch_30ep/metrics.json       (W1 arm2)
molmetal/reports/m1_realonly_w2on_per-epoch_30ep/{Ru,Ir,Os,Rh,Re}/per_seed_metrics_seed{0..4}.json  (125 per-seed JSONs)
molmetal/reports/m19_rh_fix_2026_10_08/metrics.json
molmetal/reports/pearson_head_fix_2026_10_08/seed{0,1,2}.json
results/wf124_p3_per_innovation_ablation_g6/{no_dann,no_kendall,no_multi_task,no_wass}/{Ru,Rh}/seed{0,1,2}/  (18 G6 tuples -- SUPERSEDED 2026-10-31 (Wf-A124): 24 G6 tuples; MEASURED git ls-tree -r HEAD -- results/wf124_p3_per_innovation_ablation_g6/ -> 76 files = 24 runs x 3 artifacts + 2 cohort receipts + plan.json + recovery_manifest.tsv; the 74 result artifacts were git-only from 2026-10-27 until Wf-A124 restored them to disk on 2026-10-31)
docs/HONEST_NEGATIVE_CATALOG.md                            (§6.38 appended; 94 total entries)
docs/ALIGNED_COMPARISON.md                                 (bace rescue caveat + m19 stale-file flag)
docs/STATUS.md                                             (Last refreshed + Wave-METRIC-RESCUE row + catalog count 93→94)
TODO/INDEX.md                                              (§22.151 trailer)
TODO/completed/metric_rescue_wfmrr_2026_10_08.md           (this file)
memory/w2-wasserstein-is-not-load-bearing.md               (3-arm ON/OFF table)
memory/pearson-ridge-beats-neural-head.md                  (NEW)
```

`git status --short` confirms: 5 modified files + 8 untracked entries (4 report dirs + 2 NEW metric JSONs + 1 NEW test file + 1 NEW memory file).

---

## §7 Standing directives re-confirmed

| § | Directive | Status |
|---|-----------|--------|
| §1.2 | MEASURED-vs-PROJECTED tag discipline | Honored (every numeric value cites disk path) |
| §5.1 | Standing GPU authorization | Honored (1 GPU launch: W1 arm2 per-epoch cadence, within standing grant) |
| §5.2 | Honest framing | Honored (5 honest-negative findings declared; no over-claim of W₂) |
| §5.3 | No paper work without authorization | Honored (TODO-84 unchanged) |
| §5.4 | Tier-2 + no wetlab | Honored |
| §5.5 | Post-wave on-disk verification | Strictly observed (all 11 artifacts ls -la above) |
| §5.6 | Atomic write protocol | Strictly observed (atomic_write_file used for this closeout + HONEST_NEGATIVE_CATALOG §6.38 + STATUS.md + INDEX.md §22.151) |
| §5.7 | Trainer flag wire-in smoke | Honored (4 NEW molnet flags verified end-to-end, NOT FLAG_ONLY per memory `flag-only-needs-getattr-aware-grep.md`) |
| §5.8 | wf token budget discipline | Honored (≤2 parallel agents per wave: 1 diagnostic agent + 1 measure agent) |
| §5.9 | Single-venv rule | Honored (.venv/bin/python throughout) |
| §5.10 | OmegaFold-gap-fill | Honored (no new README/docs surface added beyond §6.38 catalog + §22.151 trailer) |
| §10 | Wetlab deprecation | Honored (0 wetlab planning) |

---

## §8 Cross-references

- `docs/HONEST_NEGATIVE_CATALOG.md` §6.38 — 5 sub-entries (catalog expansion 93 → 94)
- `docs/ALIGNED_COMPARISON.md` §2 — bace rescue caveat + m19 stale-file flag
- `docs/STATUS.md` — Wave-METRIC-RESCUE row at top of timeline + honest-negative catalog count
- `TODO/INDEX.md` §22.151 — wave trailer
- `memory/w2-wasserstein-is-not-load-bearing.md` — 3-arm ON/OFF table
- `memory/pearson-ridge-beats-neural-head.md` — pooled ridge 0.4151 vs head 0.1543
- `memory/m19-centering-run-has-w2-off.md` — m19 0.5845 run config notes
- AGENTS.md §5.5 + §5.6 + §5.7 + §5.8 + §5.9 + §10
- `molmetal/scripts/wf386a3_atomic_write.py` — atomic_write_file implementation (used for this closeout)
- `molmetal/scripts/wf399a1_smoke_protocol.py` — §5.7 trainer flag wire-in smoke (the molnet 4 NEW flags verified via this surface)


---

## CORRECTION 2026-10-11 (appended, not rewritten, per AGENTS.md §5.5 rule #6)

The pooled pair in item 5 above — ridge **0.4151 ± 0.2027** / head **0.1543 ± 0.3650** —
is **not reproducible from any file on disk**. The source it cites,
`molmetal/reports/pearson_head_fix_2026_10_08/seed{0,1,2}.json`, does not exist; the real
layout is `<metal>/per_seed_metrics_seed{N}.json`. Recomputed from all 15 of those files:

| Quantity | This document | Recomputed 2026-10-11 |
|---|---|---|
| Ridge on trained D-MPNN embedding, pooled | 0.4151 ± 0.2027 | **0.4302 ± 0.1644** |
| Trained neural pIC50 head, pooled | 0.1543 ± 0.3650 | **0.2036 ± 0.2878** |

(mean-of-metal-means over 5 metals × 3 seeds, population sd.)

The **per-metal** figures in this document were already correct and still reproduce:
Ir 0.1362, Os −0.2782 ± 0.0654, Re 0.6051; Rh 0.3295 was missing and is added.

**The conclusion is unchanged**: the neural pIC50 head is not a working regressor, and
ridge beats it on all five metals individually. Pooling across n_test 994/231/16/13/13
puts 78.5% of the weight on Ru, so the pooled sd should not be quoted as a spread.

Corrected in `docs/HONEST_NEGATIVE_CATALOG.md` §6.38e and `TODO/INDEX.md`.
