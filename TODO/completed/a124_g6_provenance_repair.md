# Wf-A124 — G6 PROVENANCE REPAIR — close-out (2026-10-31)

**Wave**: Wf-A124 · **Mode**: CPU-only (0 GPU; every python/pytest invocation prefixed
`CUDA_VISIBLE_DEVICES=""`) · **Single venv**: `/home/hugo/codes/mol-metal/.venv/bin/python`
(AGENTS.md §5.9) · **No commit, no push, no `paper/`, no `metrics/`, no `registry/`**
(AGENTS.md §5.3 / TODO-84).

---

## 1. What was wrong, and what was done

The G6 per-innovation ablation tree (`results/wf124_p3_per_innovation_ablation_g6/`)
existed **only in git**: 76 files were tracked in HEAD while 74 of them were deleted
from the working tree (every deletion `" D"`), and the tree's own four logs gave
four **mutually exclusive** accounts of the same 60 tuples (`PLAN_ONLY` / 60 ×
`PENDING` / `OK: 0` / `OK 2 + FAILED 28 + RUNNING 1 + PENDING 29`). The published
per-lever Δ table (Wasserstein Ru +0.007 / Rh −0.0001; Kendall Ru −0.011 / Rh
+0.003; multi-task Ru +0.011 / Rh +0.001) **does not reproduce** from the shipped
per-seed artifacts under any single reference arm, the published `0.7888` has **no
source** anywhere in the tree it is attributed to, and the claim "18 tuples" was
wrong (the tree always held **24**). This wave restored the 74 files to disk,
recomputed every headline number from the per-seed JSON, and converted the
resulting negatives into **7 new honest-negative catalog entries (§6.216–§6.222)**
plus **6 in-place dated supersedes** of the project's own earlier overclaims
(§6.38a, §6.38b, §6.54f, §6.54i, §6.89.5, §6.91c). The historical text is
**PRESERVED VERBATIM** per AGENTS.md §5.5 rule #6 — every retraction is an added,
dated annotation, never a deletion. A regression guard
(`molmetal/tests/test_g6_receipt_integrity.py`) locks the findings: the four guard
holes the critic found are closed and were proven closed by mutation testing
(re-add the Δ table → fail; empty the G5 baseline tree → fail; flip a verdict cell
→ fail).

---

## 2. The SEVEN new catalog entries

All in `docs/HONEST_NEGATIVE_CATALOG.md` (line numbers verified with `grep -n`,
not guessed). The catalog's own counter: **COUNTER 270 → 277**, ACTIVE **226 → 233**.

| # | Section | Title (verbatim) | File + line |
|---|---|---|---|
| 1 | §6.216 | THE G6 ABLATION TREE WAS GIT-ONLY (RESTORED 2026-10-31 BY Wf-A124), AND ITS OWN FOUR LOGS GIVE FOUR MUTUALLY EXCLUSIVE ACCOUNTS OF THE SAME 60 TUPLES (2026-10-31) | `docs/HONEST_NEGATIVE_CATALOG.md:15006` |
| 2 | §6.217 | THE TWO COHORT RECEIPTS RECOMPUTE BIT-EXACTLY, BUT THEIR "std" IS THE POPULATION STDEV AND THEY CARRY NO n_test (2026-10-31) | `docs/HONEST_NEGATIVE_CATALOG.md:15129` |
| 3 | §6.218 | NO G6 LEVER IS RESOLVABLE: EVERY PAIRED EFFECT AGAINST THE G5 ALL-ON BASELINE SITS INSIDE SEED VARIANCE (2026-10-31) | `docs/HONEST_NEGATIVE_CATALOG.md:15201` |
| 4 | §6.219 | THE G5 BASELINE CARRIES NO TRAINER FINGERPRINT, AND BOTH SIBLING TREES' OWN ACCOUNTING IS FALSE (2026-10-31) | `docs/HONEST_NEGATIVE_CATALOG.md:15265` |
| 5 | §6.220 | THE ARTIFACTS DO NOT RECORD WHICH LEVER WAS DISABLED: 3 OF THE 4 G6 COHORTS ARE INDISTINGUISHABLE BY THEIR OWN CONFIG (2026-10-31) | `docs/HONEST_NEGATIVE_CATALOG.md:15348` |
| 6 | §6.221 | THE PUBLISHED PER-LEVER Δ TABLE DOES NOT REPRODUCE: 4 OF 6 CELLS REPRODUCE ONLY AGAINST A DIFFERENT REFERENCE, 2 REPRODUCE UNDER NOTHING (2026-10-31) | `docs/HONEST_NEGATIVE_CATALOG.md:15409` |
| 7 | §6.222 | THE PUBLISHED 0.7888 HAS NO SOURCE, AND 330b4's TRAINER-SPLIT AND FAILED-RUN CLAIMS ARE REFUTED BY THE ARTIFACTS (2026-10-31) | `docs/HONEST_NEGATIVE_CATALOG.md:15497` |

---

## 3. The SIX in-place corrections (dated supersedes, original text PRESERVED)

| # | Catalog anchor | What changed | Where |
|---|---|---|---|
| C5 | **§6.38a** — "CONFIRMED independently by G6 per-innovation ablation" | That clause is an **overclaim** (the two arms differ in the W₂ weight *and* in the multi-task head → CONFOUNDED). Superseded by §6.221. The **Verdict** row ("W₂ is NOT load-bearing") is **UNCHANGED** and is independently supported by §6.218. | `docs/HONEST_NEGATIVE_CATALOG.md:1021` (banner) and `:1037` (MEASURED-row table row) |
| C6 | **§6.38b** — "G6 per-innovation ablation (18 tuples)" | 4 dated corrections: **tuple count** 18 → **24** (there was never an 18-tuple tree); **Δ rows** withdrawn (they do not reproduce — §6.221); **n_test figures** "Rh 108 / Os 87" withdrawn (the G6 Rh test set is **223** rows; **no Os run exists**); the "Os reference" row removed. | `docs/HONEST_NEGATIVE_CATALOG.md:1041` (banner), `:1060` (tuple count), `:1061` (Δ rows), `:1062` (n_test figures) |
| C1 | **§6.54f** — the deferral clause | **PROVENANCE SUPERSEDE**: the G6 tree was git-only, so the clause rested on artifacts a reader could not open; the tree is on disk again since 2026-10-31. | `docs/HONEST_NEGATIVE_CATALOG.md:1313` |
| C2 | **§6.54i** — the anchor's own receipt claims | **PROVENANCE SUPERSEDE**: two of the anchor's claims are contradicted by the restored artifacts (receipt std = population stdev; 24 tuples, not 18). | `docs/HONEST_NEGATIVE_CATALOG.md:1343` |
| C3 | **§6.89.5** | **SUPERSEDED**: the row above recorded an accounting that the artifacts refute (both sibling trees' own accounting is false — §6.219). | `docs/HONEST_NEGATIVE_CATALOG.md:3601` |
| C4 | **§6.91c** | **SUPERSEDED**: the row above and the receipts it leaned on are refuted (0.7888 has no source; the failed-run claim is refuted — §6.222). | `docs/HONEST_NEGATIVE_CATALOG.md:3818` |

Two further in-place edits outside the six: `docs/HONEST_NEGATIVE_CATALOG.md:13`
(stale "108 … 129 total entries" count annotated **HISTORICAL** — it was already
stale in HEAD) and the §6.216 §5.5 blanket claim softened to name the three
rewritten docs (`TODO/INDEX.md`, `TODO/README.md`,
`TODO/completed/metric_rescue_wfmrr_2026_10_08.md`).

---

## 4. MEASURED headline numbers

Every value below is **MEASURED** (recomputed 2026-10-31 by Wf-A124 from the
per-seed artifacts on disk or from `git HEAD`), with its source path. No
PROJECTED value appears in this wave.

### 4.1 The four paired Δ + t against the G5 all-ON baseline [F5]
**Source**: `results/wf124_p3_per_innovation_ablation_g6/<arm>/{Ru,Rh}/seed{0,1,2}/per_seed_metrics.json`
× `results/wf124_p3_50epoch_full_sweep_g5/{Ru,Rh}/seed{0,1,2}/{ru,rh}/per_seed_metrics.json`;
n = 6 paired differences, df = 5, two-sided 5 % critical value **|t| > 2.570581835636**.
Table + verdict live at `docs/HONEST_NEGATIVE_CATALOG.md:15201` (§6.218).

| G6 arm | paired Δ mean (n=6) | sample sd | t (df=5) | significant? |
|---|---:|---:|---:|---|
| `no_wass` | +0.009608140325 | 0.017860450804 | **+1.3177182050052332** | **NO** |
| `no_multi_task` | +0.006288457682 | 0.015866059684 | **+0.9708467570400957** | **NO** |
| `no_kendall` | −0.002496574199 | 0.023100400540 | **−0.2647284354420802** | **NO** |
| `no_dann` | +0.001109568805 | 0.016919484354 | **+0.1606359478438517** | **NO** |

**Verdict (MEASURED): no G6 lever is resolvable.** All 6 between-arm contrasts are
non-significant too (max |t| = **2.1801078623848413**, `no_kendall − no_wass`).
**Qualification (honest):** "no lever is significant" is **not** "no lever matters"
— n = 6 with 3 seeds × 2 metals cannot resolve the effect sizes present.

### 4.2 The 8-field trainer-fingerprint confound [F18-adjacent]
**Source**: `results/wf124_p3_50epoch_full_sweep_g5/{Ru,Rh}/seed{0,1,2}/training_summary.json`
(catalog `docs/HONEST_NEGATIVE_CATALOG.md:15265`, §6.219).
**MEASURED**: the G5 baseline summaries carry **no** trainer fingerprint. The 8
fields `bf16`, `dmpnn_backend`, `dmpnn_backend_requested`, `tf32`,
`torch_compile_cuda_graphs_disabled`, `torch_compile_enabled`,
`torch_compile_mode` and `w2_gradient_mode` are **absent from every G5 summary**
(present in the G6 summaries). Every paired Δ above therefore mixes a lever change
with an **unrecorded trainer-build change**. The pairing is like-for-like (same
metal, same seed, same split) but is **not** a clean single-lever contrast.

### 4.3 The receipts recompute [F2]
**Source**: `results/wf124_p3_per_innovation_ablation_g6/{no_wass,no_multi_task}/{Ru,Rh}/training_summary.json`
(catalog `docs/HONEST_NEGATIVE_CATALOG.md:15129`, §6.217).
**MEASURED**: **4 of 4** cells' `test_auc_mean` are **mean-exact** (Δ = 0.0 against
the 3-seed recompute) and **4 of 4** cells' `test_auc_std` is **exactly the
population stdev (ddof=0)**, bit-identical to `statistics.pstdev`, and matches the
sample stdev in **none**. The sample stdev is **22.47 % larger in all four cells**
(ratio exactly √(3/2) = 1.2247448713915892 for n = 3). **Neither receipt carries an
`n_test` field.** Receipt byte sizes: `no_wass` **4283**, `no_multi_task` **4325**.

### 4.4 The git-only window [F1]
**Source**: `git ls-tree -r HEAD -- results/wf124_p3_per_innovation_ablation_g6/` +
`git status --porcelain -- results/wf124_p3_per_innovation_ablation_g6/` +
`results/wf124_p3_per_innovation_ablation_g6/recovery_manifest.tsv` +
`results/wf124_p3_per_innovation_ablation_g6/master.log` +
`TODO/completed/337g1_g6_recovery.md:31-46` (catalog `docs/HONEST_NEGATIVE_CATALOG.md:15006`, §6.216).
**MEASURED**: **76** files tracked in HEAD; **74** of them deleted in the working
tree, every one `" D"`; **4** survived (`DRY_RUN.log` + `master.log`, both
gitignored by `.gitignore:33 **/*.log`, plus `plan.json` + `recovery_manifest.tsv`).
**Four mutually exclusive accounts of the same 60 tuples**: (A) `plan.json` —
`PLAN_ONLY`; (B) `recovery_manifest.tsv` — 60 rows, every one `PENDING`; (C)
`master.log` — `OK: 0`, `TO_RECOVER: 60` (266 lines, mtime 2026-10-27 02:03:33);
(D) `337g1:31-46` — `OK 2 / FAILED 28 / RUNNING 1 / PENDING 29`, citing a
`run.log` that does not exist in HEAD (`git ls-tree` count of `run.log` in the tree
= **0**) and a "master.log (66 lines, mtime 15:03:52)" snapshot that contradicts
the on-disk file. Account D is the **origin** of the `ALL_SEEDS_FAILED` claim that
§6.222 refutes.
**Restored 2026-10-31**: **79** files on disk (76 tracked + 2 gitignored `.log` + 1
`RECONCILIATION.md`); repo-wide unstaged `" D"` count = **0**; the 74 restored
files are byte-identical to HEAD.

### 4.5 The retracted Δ table [F9] and the retracted 0.7888 [F10]
**Δ table (F9)** — the published table was: Wasserstein Ru **+0.007** / Rh
**−0.0001**; Kendall Ru **−0.011** / Rh **+0.003**; multi-task Ru **+0.011** / Rh
**+0.001**. **MEASURED** (catalog `docs/HONEST_NEGATIVE_CATALOG.md:15409`, §6.221;
table at `TODO/INDEX.md` and `TODO/completed/metric_rescue_wfmrr_2026_10_08.md`):
the table **does not reproduce**.

### 4.6 The Δ table's ORIGIN file and its four stale spots [F15]
**ORIGIN file**: `TODO/completed/metric_rescue_wfmrr_2026_10_08.md` §2 (the table
lives there in its **tabular** form — the form the pre-Wf-A124 guard did not
match). **MEASURED** — four stale spots, all now carrying a dated supersede while
the original text is preserved:
1. `:16` — summary item 2, "G6 per-innovation ablation (18 tuples)" → count
   retracted (24 tuples) + Δ table withdrawn.
2. `:46` — the §2 heading "G6 per-innovation ablation — 18 tuples MEASURED" →
   SUPERSEDE pointer added.
3. `:48` — the §2 body "completed 18 tuples" and the **caption immediately above
   the Δ table** → retraction marker added to the caption.
4. `:56` — "All inside seed std; none individually significant … the 18-tuple
   decisive subset **confirmed**" → the "confirmed" wording superseded.
Plus: `:194` (the Source row's "(18 G6 tuples)" → annotated "24 G6 tuples") and
`:109` (item 4's paired-Δ paragraph → the §4.2 confound caveat appended).

### 4.7 Cell-level reproduction [F16]
**MEASURED** (catalog §6.221): of the 6 published cells, **4 of 6 reproduce, each
against a DIFFERENT reference**, and **2 reproduce under nothing**.

| Lever | Metal | Published Δ | vs `no_multi_task` (the table's own reference) | vs the G5 all-ON baseline (paired, same metal) | vs `no_wass` (a different arm) |
|---|---|---:|---:|---:|---:|
| Wasserstein | Ru | +0.007 | **+0.006721 ✓** | +0.008390 | +0.000000 (self) |
| Wasserstein | Rh | −0.0001 | **−0.000082 ✓** | +0.010827 | +0.000000 (self) |
| Kendall | Ru | −0.011 | −0.004037 | −0.002368 | **−0.010758 ✓** |
| Kendall | Rh | +0.003 | −0.013533 | −0.002625 | −0.013451 |
| multi-task | Ru | +0.011 | +0.000000 | +0.001668 | −0.006721 |
| multi-task | Rh | +0.001 | +0.000000 | +0.010909 | +0.000082 |

The two cells that reproduce under nothing: Kendall Rh +0.003 implies a reference
Rh of **0.648055** and multi-task Rh +0.001 implies **0.663589** — no arm has
either. The multi-task row is self-referential (its arm *is* the stated
baseline), and the multi-task **Ru** cell equals the G5 **Rh** column's Δ
(+0.010909) — the two cells of that row are swapped.

### 4.8 24 tuples, never 18 [F17]
**MEASURED**: the G6 tree always held **24** tuples = 4 cohorts (`no_dann`,
`no_kendall`, `no_multi_task`, `no_wass`) × {Ru, Rh} × 3 seeds
(`git ls-tree -r HEAD -- results/wf124_p3_per_innovation_ablation_g6/` → 76 files
= 24 runs × 3 artifacts + 2 cohort receipts + 4 tree-level files). `no_dann` was
present from the **first** results commit `37aa4960` (2026-10-09) — the "18"
came from `G6_CONFIGS=3`, never from the tree.

### 4.9 No lever flag for 3 of 4 cohorts [F18]
**Source**: `results/wf124_p3_per_innovation_ablation_g6/<arm>/<metal>/seed*/training_summary.json`
+ `molmetal/scripts/train_dmpnn_multitask_wasserstein.py:3061` (catalog
`docs/HONEST_NEGATIVE_CATALOG.md:15348`, §6.220).
**MEASURED**: `use_dann`, `use_kendall` and the multi-task flags are recorded
**nowhere** in the artifacts; `no_dann`, `no_kendall` and `no_multi_task` show
**0 differing fields** on every compared key for Ru/seed0 (`per_metal`,
`ckpt_paths`, `elapsed_seconds_total`, `honest_framing_note`, `wf_tag` excluded).
**3 of the 4 cohorts are indistinguishable by their own config**; the **cohort
directory name is the only record** of the lever identity — the AGENTS.md §5.11
silent-absent-knob failure mode. The trainer *has* the flags; it does not write
them.

### 4.10 Between-cohort spread vs within-cohort seed spread [F19]
**Source**: `results/wf124_p3_per_innovation_ablation_g6/RECONCILIATION.md:253-262`
(= `no_dann` 0.7941348, `no_kendall` 0.7716835, `no_multi_task` 0.7688538,
`no_wass` 0.7867674 for Ru/seed 0).
**MEASURED**: between-cohort spread **0.025281** (Ru, seed 0); within-cohort seed
spreads `no_dann` 0.017189, `no_kendall` 0.031655, `no_multi_task` **0.016296**,
`no_wass` 0.010977 → range **0.010977–0.031655**. The between-cohort spread sits
**INSIDE** the range of ordinary within-cohort seed spreads.
**Labelled OPEN — and explicitly NOT a claim that the levers were inert.** This
is equally consistent with "no lever had a resolvable effect" and with "the lever
was never applied"; the artifacts cannot distinguish the two, so **neither is
asserted**.

### 4.11 §6.38b's n_test figures [F20]
**MEASURED**: "Rh n_test=**108** has AUC resolution ≈±0.061, Os n_test=**87**
≈±0.077" is **WRONG for the G6 protocol**. Every G6 **Rh** run records
`n_train=696 / n_val=199 / n_test=**223**` (identical across all runs), and there
is **no Os run in G6 at all** (G6 = Ru + Rh only; `plan.json` lists 5 metals but
only 2 were run). Superseded at `docs/HONEST_NEGATIVE_CATALOG.md:1062`.

### 4.12 Every run is single-metal; G5 and G6 share identical splits [F21]
**MEASURED**: every G6 run is **single-metal** (`metals = ["Ru"]` or `["Rh"]`;
`per_metal` has one entry) — `RECONCILIATION.md:246`. G5 and G6 share the
**identical split per metal** (Ru 12471/2166/4350, Rh 696/199/223) —
`RECONCILIATION.md:145`.

### 4.13 §6.38a's "CONFIRMED independently" overclaim [F22]
**MEASURED**: the clause "CONFIRMED independently by G6 per-innovation ablation
`no_wass` vs `no_multi_task` (Ru +0.007, Rh −0.0001, both inside seed std)" is an
**overclaim** — the two arms differ in the W₂ weight **and** in the multi-task
head, so the contrast is **CONFOUNDED**. **The verdict that still stands:**
"**W₂ is NOT load-bearing**" — unchanged, and independently supported by the
correct paired G5 comparison (§4.1, all four arms non-significant).
Superseded at `docs/HONEST_NEGATIVE_CATALOG.md:1021` and `:1037`.

---

## 5. What was NOT fixed

Copied from the critic's **C9** list. These remain OPEN; none is in this wave's
owned set, and none was silently dropped:

1. `TODO/completed/337g1_g6_recovery.md` — the 2026-09-29 account of the 60 tuples
   + origin of the refuted ALL_SEEDS_FAILED claim; not in the owned set, but must be
   logged.
2. `docs/STATUS.md:20` — live retracted Δ/18-tuples; not owned.
3. `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` — does **not** write
   `use_dann`/`use_kendall`/multi-task into `training_summary.json`, the root cause
   of [F18]; needs a code change + a 50-step smoke (§5.7).
4. `results/wf124_p3_50epoch_full_sweep_g5/training_summary.json` still says
   `todo_181_g5_total_pairs: 0` / `runs: []`, and G5's `plan.json` still
   `PLAN_ONLY` — documented in §6.219 but not repaired (not owned).
5. `.gitignore:33 **/*.log` is why the operational logs vanish; not owned.

Items 1 and 2 are **logged as OPEN** inside the owned file
(`docs/HONEST_NEGATIVE_CATALOG.md:15006`, §6.216's OPEN-anomaly table, with
`docs/STATUS.md:20`, `TODO/completed/337g1_g6_recovery.md:31-46` and
`TODO/completed/339d2_lambda_roadmap_audit.md:132` named verbatim), and 337g1 is
also recorded there as the **fourth** mutually exclusive account. The underlying
files were **not** edited — they are outside the owned set.

---

## 6. party-twenty-spirit §4 mapping (二十大精神 × Wf-A124)

**日期**: 2026-10-31 · **阶段**: Wf-A124 G6 provenance repair (CPU-only, 0 GPU) — ship

### 三个务必
- 不忘初心: ⚠️ 远离 headline — 本波不动任何 AUC，只把 G6 结论从"不可复现"改成"可复现的否定"；paper §4 Tier-1 AUC 未推进。
- 谦虚谨慎: ✅ 承认 7 条新 honest-negative，其中 5 处是本项目自己此前发布的过度声明（Δ 表、0.7888、18 tuples、CONFIRMED independently、Δ rows）。
- 敢于斗争: 下阶段硬骨头 = `molmetal/scripts/train_dmpnn_multitask_wasserstein.py:3061` 不把 lever flag 写进 artifact（§6.220 根因），需 code change + §5.7 50-step smoke。

### 守正创新
- 守: MLC + CFM + click chemistry + PLL + 84-rule MLE 未触碰；§5.5 rule #6 "PRESERVED verbatim" 全程执行，撤回一律是加 dated annotation。
- 创新: `molmetal/tests/test_g6_receipt_integrity.py`（17 tests + 26 subtests）+ `results/wf124_p3_per_innovation_ablation_g6/RECONCILIATION.md`（18.7 KB）——两个都能独立失败的真货。
- 不抄: 无 SOTA 拼盘；每条结论从 per-seed JSON 重算，不引用二手表格。

### 系统观念
- 与 Phase A 关系: 独立 — 不碰 Ru/Pt AUC 训练。
- 与 R18 接口: §6.216..§6.222 给 R18 ablation 一条硬约束：arm 必须把 lever flag 写进 artifact，否则结论不可复现。
- 5-gate: docs ✅（catalog 7 新 + 6 修正）· integration ✅（pytest 17 passed）· smoke ✅（mutation 4/4 触发）· bench N/A · paper ⛔（`paper/` 0 改动，TODO-84）。

### 高质量发展
- 创新点: 上面两个文件（test + RECONCILIATION）；7 条 catalog entry 每条都带可点开的 source path。
- 凑数项: 无 —— 0 新 workflow、0 GPU、0 训练、0 采样。
- 绿色: 全 CPU-only；`nvidia-smi` 无 mol-metal 进程（仅 `/home/hugo/codes/planB` 两个进程）。

### 自我革命
- 数据真实: ✅ 全部 MEASURED，源为 per-seed JSON / git HEAD；本波 0 个 PROJECTED。
- 失败暴露: ✅ 四个日志互相矛盾、Δ 表不可复现、0.7888 无来源、18→24 —— 全部写进 §6.216..§6.222。
- 0 容忍: ✅ 撤回时保留原文；两份原文（INDEX.md 一句话、metric_rescue §2）逐字留存。

### 决策
- 继续当前路线? **是** — 先修 provenance，再谈 ablation 结论。
- 调整方向: 下一步 = trainer 写 lever flag → 重跑 G6 → 才允许再引用 per-lever Δ。
- TODO 更新操作: `TODO/INDEX.md` G6 行 · `TODO/README.md:3` · `TODO/completed/metric_rescue_wfmrr_2026_10_08.md` §2 · `docs/HONEST_NEGATIVE_CATALOG.md`（7 新 + 6 修正 + 2 就地）· 新建本文件。

---

## 7. Filesystem-truth evidence (§5.5) — `ls -la` of every file this wave wrote

```
$ ls -la AGENTS.md docs/HONEST_NEGATIVE_CATALOG.md TODO/INDEX.md TODO/README.md \
    TODO/completed/330b4_regression_smoke.md TODO/completed/metric_rescue_wfmrr_2026_10_08.md \
    molmetal/tests/test_g6_receipt_integrity.py \
    results/wf124_p3_per_innovation_ablation_g6/RECONCILIATION.md \
    results/wf124_p3_per_innovation_ablation_g6/plan.json \
    results/wf124_p3_remaining_metals_g4/C/Ru/seed0/ru/per_seed_metrics_seed0.json
-rw-r--r-- 1 hugo hugo  26482 Oct 31 08:11 AGENTS.md
-rw-r--r-- 1 hugo hugo 1585727 Oct 31 08:47 docs/HONEST_NEGATIVE_CATALOG.md
-rw-r--r-- 1 hugo hugo  31339 Oct 31 08:46 molmetal/tests/test_g6_receipt_integrity.py
-rw-r--r-- 1 hugo hugo   9048 Oct 31 08:20 results/wf124_p3_per_innovation_ablation_g6/plan.json
-rw-r--r-- 1 hugo hugo  18732 Oct 31 08:45 results/wf124_p3_per_innovation_ablation_g6/RECONCILIATION.md
-rw-r--r-- 1 hugo hugo   1321 Oct 31 08:22 results/wf124_p3_remaining_metals_g4/C/Ru/seed0/ru/per_seed_metrics_seed0.json
-rw-r--r-- 1 hugo hugo  28406 Oct 31 08:16 TODO/completed/330b4_regression_smoke.md
-rw-r--r-- 1 hugo hugo  20322 Oct 31 08:45 TODO/completed/metric_rescue_wfmrr_2026_10_08.md
-rw-r--r-- 1 hugo hugo 181582 Oct 31 08:45 TODO/INDEX.md
-rw-r--r-- 1 hugo hugo  45162 Oct 31 08:45 TODO/README.md
```

The **74 restored G6 artifacts** (deleted in the working tree before this wave,
restored byte-identical to HEAD) — `results/wf124_p3_per_innovation_ablation_g6/`:

```
$ ls -la results/wf124_p3_per_innovation_ablation_g6/
total 56
drwxr-xr-x 1 hugo hugo   210 Oct 31 08:45 .
drwxr-xr-x 1 hugo hugo  4172 Oct 27 01:10 ..
-rw-r--r-- 1 hugo hugo  3993 Oct 27 02:03 DRY_RUN.log
-rw-r--r-- 1 hugo hugo 12464 Oct 27 02:03 master.log
drwxr-xr-x 1 hugo hugo     8 Oct 31 08:14 no_dann
drwxr-xr-x 1 hugo hugo     8 Oct 31 08:14 no_kendall
drwxr-xr-x 1 hugo hugo    32 Oct 31 08:14 no_multi_task
drwxr-xr-x 1 hugo hugo    32 Oct 31 08:14 no_wass
-rw-r--r-- 1 hugo hugo  9048 Oct 31 08:20 plan.json
-rw-r--r-- 1 hugo hugo 18732 Oct 31 08:45 RECONCILIATION.md
-rw-r--r-- 1 hugo hugo  3644 Oct 27 02:03 recovery_manifest.tsv
$ find results/wf124_p3_per_innovation_ablation_g6 -type f | wc -l
79
```

`git status --porcelain` over the wave's owned set (MEASURED):

```
 M AGENTS.md
 M TODO/INDEX.md
 M TODO/README.md
 M TODO/completed/330b4_regression_smoke.md
 M TODO/completed/metric_rescue_wfmrr_2026_10_08.md
 M docs/HONEST_NEGATIVE_CATALOG.md
 M results/wf124_p3_per_innovation_ablation_g6/plan.json
 T results/wf124_p3_remaining_metals_g4/C/Ru/seed0/ru/per_seed_metrics_seed0.json
?? molmetal/tests/test_g6_receipt_integrity.py
?? results/wf124_p3_per_innovation_ablation_g6/RECONCILIATION.md
```

The ` T` row is the G4 slot that was a **live git symlink** (mode 120000) whose
105-byte blob is a path string; as a symlink `json.load()` FOLLOWED it and silently
served the G5 Ru/seed0 metrics instead of raising. It is now a regular-file
provenance stub with `"status": "MIS_PATHED_DUPLICATE"`,
`"is_a_measurement": false`, `"MEASURED_vs_PROJECTED": "NOT_A_MEASUREMENT"`.

This close-out file itself (written last, so its own size is not in the block above):

```
$ ls -la TODO/completed/a124_g6_provenance_repair.md
-rw-r--r-- 1 hugo hugo 26147 Oct 31 08:50 TODO/completed/a124_g6_provenance_repair.md
```

---

## 8. Close-out checks (MEASURED, 2026-10-31, CPU-only)

| Check | Command | Result |
|---|---|---|
| `paper/` untouched | `git status --porcelain -- paper/` | **empty** (0 lines) |
| G6 tree | `git status --porcelain -- results/wf124_p3_per_innovation_ablation_g6/` | **2 lines**: ` M plan.json` + `?? RECONCILIATION.md` — the wave's own two deliverables. Repo-wide unstaged `" D"` count = **0** (the 74 restored files are byte-identical to HEAD and do not appear as diffs). |
| Registry | `molmetal/scripts/registry.py check` | **REGISTRY: PASS** (datasets 35/35, protocols 8/8, metrics **578 current, 0 stale, 0 missing**) |
| Regression guard | `pytest molmetal/tests/test_g6_receipt_integrity.py -q` (`CUDA_VISIBLE_DEVICES=""`) | **17 passed, 26 subtests passed** in 0.51 s |
| GPU state | `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader` | 2 processes, both `/home/hugo/codes/planB/.venv/bin/python3` (2858 MiB, 1656 MiB) — **no mol-metal GPU process** |
| Catalog counter | `grep -c "277 entries\*\* (233 active)" AGENTS.md` | **3** (all three AGENTS.md count lines agree with the catalog's `COUNTER 270 → 277` / ACTIVE 226 → 233) |

**Blocking defects**: the critic reported **12**; **all 12 are resolved** (B1 = this
close-out file; B2 `RECONCILIATION.md` "2 lines" → "74 lines, every one ` D`";
B3 "139 files" → "79 files"; B4 §6.217 "15 %…23 %" → "22.47 % in all four cells";
B5 §6.221's wrong `vs no_wass` cell + two spurious ✓ + 4-of-6 framing;
B6 RESTORED banners on §6.216/§6.217 + the §6.219 G4 row rewritten to the stub's
present state; B7 the confound caveat added to INDEX.md + metric_rescue and §6.219
added to INDEX.md's Sources; B8 both "(lever removed)" headers → "lever status
**NOT recorded in the artifact** — see §6.220"; B9/B10 `docs/STATUS.md:20`,
337g1 and 339d2:132 logged as OPEN in §6.216 with 337g1 added as the fourth
account; B11 the four guard holes closed and proven by mutation (Δ table re-added
in either form → 1 failed; G5 tree emptied → 2 failed; verdict cell flipped → 1
failed); B12 INDEX.md's original one-liner restored verbatim with a dated
supersede). Non-blocking C1/C2/C3/C10 were fixed as well.

**Not done, by instruction**: no commit, no push, no `paper/` touch, no `metrics/`
touch, no `registry/` touch, no GPU job, no wet-lab work.
