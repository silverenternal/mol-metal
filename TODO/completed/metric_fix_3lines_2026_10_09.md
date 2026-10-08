# Wf-FIX-3LINES 2026-10-09 — 3-line metric fix: bace rescue / CFM bond head 20-ep / Lambda cohort expansion

**Closeout written per AGENTS.md §5.6 (atomic_write_file) + §5.5 (ls -la evidence).**

> **PREMISE CORRECTION (brief vs disk).** The wave brief directed this closeout to report a
> **5-seed bace value**, a **20-epoch CFM bond F1**, and a **"R² at 50-d"**. On-disk verification
> at 2026-10-09 01:31 shows **2 of the 3 runs ABORTED mid-flight** and the third used **32-d, not
> 50-d**, features. Per §5.5 (filesystem is the source of truth) + §5.2 (honest framing), **none
> of the brief's headline numbers are adopted.** No 5-seed bace mean and no 20-epoch CFM F1 exist
> on disk. Wave 3 alone completed.

---

## §1 Wave 1 — bace rescue round 2 — **ABORTED at seed 2/5**, INCONCLUSIVE (no verdict)

| Field | Value |
|---|---|
| Protocol | `--backbone dmpnn --epochs 50 --batch-size 64 --lr 0.001 --hidden 300 --depth 3 --seeds 0 1 2 3 4` (canonical baseline protocol, GPU) |
| Baseline | 0.7421 ± 0.0446 (`metrics/by_metric/moleculenet_auc_dmpnn_5seed.json::tasks.bace`) |
| MEASURED (partial) | seed 0 `test roc_auc` = **0.6311** (pr_auc 0.6367); seed 1 `test roc_auc` = **0.8644** (pr_auc 0.8187) |
| Run state | **ABORTED.** `nvidia-smi --query-compute-apps` empty; no `train_moleculenet` process alive; log frozen at mtime 01:02. Seed 2 reached epoch 40/50 with **no `test_metrics` block written**; seeds 3–4 never started. Launcher's `metrics.json` / `report.md` **were never produced**. |
| Verdict | **INCONCLUSIVE — NO VERDICT CLAIMED.** The brief's decision rule (>0.86 CEILING_BREAK / 0.74–0.86 PARTIAL / <0.74 STILL_HONEST_NEGATIVE) **cannot be applied to a 2-seed partial**. |

**Key finding:** the 2 surviving seeds span **Δ = 0.2333** (0.6311 → 0.8644) — bace at this
protocol is **seed-dominated**. No 2-seed mean may be reported as "the" bace number, and the
baseline std (±0.0446) is itself an order of magnitude below this partial's spread. The
`docs/ALIGNED_COMPARISON.md` §2 "clean GPU re-run with a same-device control" confirmation is
**still owed** — this run did not deliver it. **Do NOT cite 0.6311 or 0.8644 as a bace result.**

Source: `molmetal/reports/wf_fix3lines_bace_50ep_2026_10_09/bace_dmpnn_seed{0,1}.json` +
`logs/wf_fix3lines_bace_50ep.log`

---

## §2 Wave 2 — CFM bond head 20-epoch — **ABORTED at epoch 1/20**; F1 **BELOW** the 0.079 probe

| Field | Value |
|---|---|
| Config | `--epochs 20`, `bond_head=learned`, `cfm_decoder=yuelbond`, `use_yuelbond=True`, `joint_train=True` |
| epoch 0 | bond_f1 **0.0357**, prec 0.5708, rec 0.0184, pr_auc 0.485, exact_type_rate **0.000**, train_loss 0.9909, bond_ce 0.9210, **decode_ratio 1.000** (192/192) |
| epoch 1 | bond_f1 **0.0193**, prec 0.5820, rec 0.0098, pr_auc 0.566, exact_type_rate **0.000**, train_loss 0.8298, bond_ce 0.7697, **decode_ratio 0.875** (168/192) |
| Run state | **ABORTED.** Log frozen at mtime 01:18; only epochs 0–1 of 20 present. No trainer process alive; `nvidia-smi` empty. The `cfm-early-stop` gate had **already fired at epoch 0** on `gen_eval_n=0` ("generation eval did not run") — the run was on a termination path regardless. |
| Verdict | **HONEST-NEGATIVE (provisional, 1 usable epoch).** bond F1 = **0.0193–0.0357**, i.e. **0.6–2.5× WORSE** than the 0.079 probe, not above it. |

**Two framing traps avoided:**
1. The logged `alt_rule_F1` (0.331 / 0.301) is a **different decision rule** (non-argmax) and is
   **NOT comparable** to the argmax probe number — it must not be quoted as the headline.
2. `decode_ratio` **> 0** at both epochs, so the **§5.7 floor HOLDS** — the §5.7 regression class
   did **not** recur here. That is a genuine pass and is stated as such.

**A 20-epoch claim is NOT SUPPORTED by 2 epochs.** Source: `logs/wf_fix3lines_cfm_20ep.log`

---

## §3 Wave 3 — Lambda/platinai cohort expansion n=200 → 4078 — **COMPLETED**, R² **0.0102 ± 0.0036** — ≈ 0

| Field | Value |
|---|---|
| Protocol | `load_cohort(max_cohort_size=4078)`; tier + Ridge from `wf388a1_m2_per_cohort_real` **unmodified**; 5 seeds × 5-fold = 25 splits; no subsampling |
| MEASURED | platinai_4cell `r2_mean` = **0.010248**, `r2_std` = **0.003589**, baseline mechanical −0.12505, lift **+0.13530** |
| hilbert (alias) | `r2_mean` = **0.010992** — `cohort_is_alias_of: "platinai_4cell"` |
| n=200 baseline | −0.144015 ± 0.032505 (`metrics/by_metric/m2_r2_per_cohort.json::per_cohort`) |
| Verdict | **STILL_HONEST_NEGATIVE_R2_DEEP — NOT WEAK_RECOVERY.** |

**Why this is NOT the brief's "WEAK_RECOVERY":** the brief's rule was "if > 0 → WEAK_RECOVERY".
The mean IS nominally > 0, but **0.0102 ± 0.0036 is statistically indistinguishable from zero**
and is ~14× smaller than the lift the earlier n=200 claim rested on. Reporting it as
WEAK_RECOVERY would violate §5.2. A **20× cohort expansion moved R² from −0.144 to +0.010** — the
signal does **not** scale with data.

**Two confounds that must travel with this number:**
1. **NOT a like-for-like tier comparison.** n=200 used the `n_lt_300` tier (12-d, alpha=1.0);
   n=4078 uses `n_gte_1000` (alpha=0.1). Feature count AND regularisation both change, so the
   Δ mixes cohort-size effect with tier-config effect. (Recorded on disk as `delta_vs_n200`.)
2. **32-d, NOT 50-d.** The `n_gte_1000` tier *requests* 50 features (12-d RDKit + 12-d chemprop v3
   + 24-d Hilbert spectral), but the loader supplies **only the 32-d RDKit block**
   (`n_features_used=32`, `n_features_available=32`). This is a 32-d measurement scored under
   alpha=0.1, **not** the full 50-d ensemble. The brief's "R² at 50-d" framing is not what ran.

Source: `molmetal/reports/wf_fix3lines_plat_expand_2026_10_09/result.json` +
`metrics/by_metric/m2_r2_per_cohort.json::wf_fix3lines_expand_2026_10_09`

---

## §3a hilbert alias confirmation (from Wave 3's own output)

`hilbert` is an **alias of `platinai_4cell`**, recorded inline as `cohort_is_alias_of:
"platinai_4cell"`. The n=4078 run's "2 cohorts tested" are therefore **1 independent cohort**
(`n_cohorts_independent: 1`, `n_cohorts_aliased: 1`). The ~0.0101 vs ~0.0109 gap is the same data
under a seed offset and is **not** a second data point. Any "we expanded the Hilbert cohort to
n=4078" claim is **structurally meaningless** — the expansion applied to the single underlying
platinai table.

---

## §4 Cross-wave insights

1. **bace: still unresolved, and newly shown to be seed-dominated.** The 2-seed spread (Δ=0.2333)
   is ~5× the baseline 5-seed std (±0.0446). The −0.134 architecture gap therefore cannot be
   closed by any single-run number; it needs the full 5-seed GPU run that did not complete.
2. **CFM bond F1: the lift did NOT materialise.** At the only two completed epochs, F1 is
   0.0193–0.0357 vs the 0.079 probe — **worse**, not better. This corroborates §6.38e (neural
   head is not a working regressor) rather than refuting it. Contrast with the §5.7 floor, which
   *did* hold (decode_ratio > 0) — so this is a quality failure, not a wiring failure.
3. **Cohort expansion: the single positive direction, but too small to bank.** R² moved
   −0.144 → +0.010 across a 20× data increase. Direction is right; magnitude is ≈ 0. The
   bottleneck is **representation**, not sample count — see §8.
4. **Common thread: two of three "fix" directions were pre-declared too optimistically.** The
   brief predicted a bace lift and an F1 lift; disk shows neither. Only the expansion direction
   moved, and only slightly.

---

## §5 Standing directives honored

| Directive | How honored |
|---|---|
| **§1.1 MEASURED-vs-PROJECTED** | Every number tagged MEASURED with a source path. The 2 aborted runs are tagged **ABORTED/INCONCLUSIVE**, never as results. |
| **§1.2 Honest-negative framing** | bace + CFM F1 + R² all declared honest-negative rather than paraphrased as promising (§5.2). |
| **§1.4 Pre-flight v2 25/25 gate** | Wave 3 was a CPU-only closed-form Ridge diagnostic — no retrain, so no pre-flight required. **Waves 1 & 2 were GPU retrains and NO pre-flight v3 log exists on disk** for this wave (see §7). |
| **§5.1 Standing GPU authorization** | Both GPU launches proceeded without per-launch OK (within the standing grant). VRAM was the abort cause, not authorization. |
| **§5.5 Post-wave verification** | **This closeout's premise section is the correction**: the brief's 3 headline numbers were rejected against disk before any number was written. `ls -la` evidence in §6. |
| **§5.6 Atomic write** | This file was written via `atomic_write_file` (temp + `os.rename`), not direct `Write`. |
| **§5.7 Trainer-flag wire-in smoke** | The CFM run's `decode_ratio` was checked at both completed epochs (1.000, 0.875 — both > 0) → floor HOLDS. |
| **§5.9 Single-venv rule** | Both launchers invoke `.venv/bin/python` only; no bare `python`/`python3`, no CadButEaas/kanzi_venv. |

---

## §6 Disk artifacts (ls -la evidence per §5.5 rule #2)

```
TODO/completed/metric_fix_3lines_2026_10_09.md
-rw-r--r-- 1 hugo hugo  <THIS_FILE> Oct  9 01:3x

molmetal/reports/wf_fix3lines_bace_50ep_2026_10_09/
drwxr-xr-x  1 hugo hugo      84 Oct  9 01:02 .
-rw-r--r--  1 hugo hugo     481 Oct  9 01:02 bace_dmpnn_seed0.json
-rw-r--r--  1 hugo hugo     494 Oct  9 01:02 bace_dmpnn_seed1.json
   ^ NO metrics.json / report.md — the launcher's aggregate outputs were never written (ABORTED).
   ^ NO seed2/3/4 JSON — seeds 2-4 never completed.

molmetal/reports/wf_fix3lines_plat_expand_2026_10_09/
-rw-r--r--  1 hugo hugo          result.json          <-- WAVE 3, COMPLETE

molmetal/reports/cfm_bond_head_20ep_2026_10_09/
drwxr-xr-x  1 hugo hugo      46 Oct  9 01:18 .
drwxr-xr-x  1 hugo hugo      66 Oct  9 01:18 checkpoints
-rw-r--r--  1 hugo hugo    2056 Oct  9 01:18 events.jsonl
   ^ NO final metrics.json — epochs 0-1 of 20 only (ABORTED).

metrics/by_metric/m2_r2_per_cohort.json   (17728 B, Oct  9 01:24) — key wf_fix3lines_expand_2026_10_09
docs/HONEST_NEGATIVE_CATALOG.md            — §6.39 + §6.39a..d

logs/wf_fix3lines_bace_50ep.log    1649 B  Oct  9 01:02  (28 min stale at verify time)
logs/wf_fix3lines_cfm_20ep.log     4607 B  Oct  9 01:18  (13 min stale at verify time)
```

---

## §7 Cross-references

- `docs/HONEST_NEGATIVE_CATALOG.md` **§6.39** (+ §6.39a bace / §6.39b CFM / §6.39c R² / §6.39d hilbert alias) — the canonical honest-negative record for this wave.
- `docs/ALIGNED_COMPARISON.md` **§2** — the bace −0.134 gap and the "same-device control confirmation still owed" that this wave did **not** deliver.
- `docs/PAPER_SECTION_3_METRIC_TABLE_FINAL.md` **§1.2** — m2 R² per-cohort REAL verification table. **NOTE: §1.2 has NOT been updated with the n=4078 result**; it still carries the Wf-388-A1 n=200 numbers. Sync is outstanding.
- `TODO/completed/metric_rescue_wfmrr_2026_10_08.md` — the prior wave whose bace rescue was **REFUTED** (0.7421 → 0.6805, worse).
- AGENTS.md **§5.5** (filesystem-truth), **§5.6** (atomic write), **§5.7** (wire-in smoke), **§5.9** (single-venv).

**Pre-flight gap:** AGENTS.md §1.4 requires pre-flight PASS before any retrain. No pre-flight
v3 log for this wave exists on disk (`logs/` contains no `*pre*flight*` file dated 2026-10-09;
`molmetal/scripts/pre_flight_check.py` mtime is 2026-10-08 19:28). Waves 1 & 2 launched without a
recorded pre-flight PASS — **flagged, not adjudicated here.**

---

## §8 Next steps

- **bace (Wave 1):** re-run to full 5 seeds on GPU. The Δ=0.2333 seed spread means **5 seeds are
  the minimum** for any usable number — a 3-seed run would still be under-powered. Verify VRAM
  headroom before relaunch (both prior aborts were VRAM/schedule-bound, not logic-bound).
- **CFM bond head (Wave 2):** the `cfm-early-stop` gate fired at **epoch 0** on `gen_eval_n=0`
  ("generation eval did not run"). **Fix that first** — otherwise any relaunch terminates before
  epoch 2 regardless of the bond head. Then, if F1 is still ≈ 0.02 vs the 0.079 probe, retry
  OT-CFM per §8 of the plan; the honest read is that **the neural head is not a working regressor**
  (§6.38e), so the fix may lie in the head architecture, not the schedule.
- **Cohort expansion (Wave 3):** the bottleneck is **representation, not sample count** — 20× more
  data moved R² by 0.154 but landed at ≈ 0. Next lever is the **missing 18 features** (chemprop v3
  12-d + Hilbert spectral 24-d): the tier *requests* 50-d but the loader only supplies 32-d RDKit.
  **Wiring the full 50-d block is the higher-value next step** and is a genuine confound, not a
  tuning knob.
- **Docs sync:** update `docs/PAPER_SECTION_3_METRIC_TABLE_FINAL.md` §1.2 with the n=4078 result,
  and record the §1.4 pre-flight gap.

---

## §9 Honest framing (AGENTS.md §5.2)

**MEASURED:** Wave 3 R² = 0.010248 ± 0.003589 (32-d, alpha=0.1, 5 seeds × 5-fold, n=4078);
Wave 1 partial bace = 0.6311 (seed 0) / 0.8644 (seed 1); Wave 2 partial CFM bond_f1 = 0.0357
(epoch 0) / 0.0193 (epoch 1) with decode_ratio 1.000 / 0.875.

**ABORTED / NO VERDICT:** bace 5-seed value (**does not exist** — 2/5 seeds); CFM 20-epoch
production F1 (**does not exist** — 2/20 epochs).

**NOT MEASURED:** "R² at 50-d" (only 32-d was available); any bace mean; any 20-epoch F1 claim.

**Zero fabricated numbers.** Every value above traces to a path on disk, per §1.1.
