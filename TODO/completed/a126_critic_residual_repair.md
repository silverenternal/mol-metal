# Wf-A126 — G6 CRITIC-RESIDUAL REPAIR — close-out (2026-10-31)

**Wave**: Wf-A126 · **Mode**: CPU-only (**0 GPU**; every `python` / `pytest`
invocation prefixed `CUDA_VISIBLE_DEVICES=""`) · **Single venv**:
`/home/hugo/codes/mol-metal/.venv/bin/python` (AGENTS.md §5.9) · **No commit, no
push, no `paper/`, no `metrics/`, no `registry/`** (AGENTS.md §5.3 / TODO-84) ·
**Catalog**: `docs/HONEST_NEGATIVE_CATALOG.md` **§6.223** (COUNTER **277 → 278**,
ACTIVE **233 → 234**).

---

## 1. What was wrong, and what was done

Wf-A124 shipped the G6 provenance repair and then ran its own completeness
critic. The critic returned **12 blocking defects** (B1..B12) plus non-blocking
findings C1..C12, and the wave was left NO-GO. Re-checked against disk on
2026-10-31, **10 of the 12 were already closed** by the wave's own later groups
— the critic's evidence had been gathered against a **MID-WAVE** state — and
**2 of its claims were factually wrong**. The genuinely open residue was:

1. **Three live documents** still carrying retracted numbers, which the wave had
   **logged** as OPEN in §6.216 but never edited (they were outside its owned
   set). *Logging is not fixing.*
2. **A trainer root cause nobody repaired.** The critic's C9(3) named it
   explicitly: the trainer "does **not** write `use_dann`/`use_kendall`/
   multi-task into `training_summary.json` … needs a code change + a 50-step
   smoke (§5.7)".

Wf-A126 closed both. The three documents now carry **dated, additive**
corrections with the historical text **PRESERVED VERBATIM** (AGENTS.md §5.5
rule #6); the trainer now records its **ablation lever identity** and its
**build fingerprint** in `training_summary.json`, additively, with no training
behaviour change.

---

## 2. The 12 "blocking defects", re-checked one by one

| # | the critic's claim | disk truth, 2026-10-31 | verdict |
|---|---|---|---|
| B1 | close-out file `TODO/completed/a124_g6_provenance_repair.md` MISSING | **EXISTS**, 26 147 B | already closed |
| B2 | `RECONCILIATION.md:51` "**2** lines, all ` D`" | "**74** lines, every one ` D`" | already closed |
| B3 | `RECONCILIATION.md:63` "**139** files on disk" | "**79** files on disk = 76 tracked + 2 gitignored `.log` + 1 `RECONCILIATION.md`" | already closed |
| B4 | §6.217 "15 %…23 %" | "**22.47 % in all four cells**" (ratio exactly √(3/2), identical for n = 3) | already closed |
| B5 | §6.221's "vs `no_wass`" cell "−0.000082 ✓" | "+**0.000000 (self)**" | already closed |
| B6 | §6.216/217/219 stale present-tense claims | §6.216+§6.217 carry the dated **RESTORED 2026-10-31** banner; §6.219's G4 row now states the regular-file stub with `is_a_measurement: false` | already closed |
| B7 | fingerprint confound missing at INDEX.md + metric_rescue | present in **both** (INDEX.md:934, metric_rescue:108-109) | already closed |
| B8 | header "G6 arm (lever removed)" | "G6 arm (lever status **NOT recorded in the artifact** — see §6.220)"; `grep -c 'lever removed'` → **0** | already closed |
| B9 | `docs/STATUS.md:20` retracted count + Δ, live | was open | **CLOSED by Wf-A126** |
| B10 | `337g1_g6_recovery.md` refuted account, live | was open | **CLOSED by Wf-A126** |
| B11 | T7/T5 guard holes | four guards now exist (`test_g6_receipt_integrity.py:585/671/681/730`) | already closed |
| B12 | `INDEX.md:922` "deleted rather than superseded" | `git diff -U0` → the `+` line **starts with the `-` line's exact text** | **critic FACTUALLY WRONG** |

A further critic claim was also factually wrong: `grep -rn 337g1
docs/HONEST_NEGATIVE_CATALOG.md` → **0 hits** is false — MEASURED **4 hits**
(15064 / 15073 / 15094 / 15111); §6.216 already lists the 337g1 account as
"Account D".

---

## 3. The root cause nobody repaired: THE TRAINER WAS REVERTED OUT OF ITS OWN FINGERPRINT

| when | commit | what it did |
|---|---|---|
| 2026-10-02 | `bc29aa90` | the G5 baseline tree was committed — the trainer had **no** fingerprint yet |
| 2026-10-08 15:54 | `8d139d23` | "wire 8 §5.7 features + alias 5 onto live levers" — **ADDED** the 8-field fingerprint |
| 2026-10-09 | `37aa4960` | the G6 results were committed — all 24 G6 summaries carry all 8 fields |
| **2026-10-15 08:12** | **`af84a545`** | **"m21 selective Wf-A17 revert (D-MPNN multitask trainer + config)" — REMOVED the fingerprint** |

MEASURED 2026-10-31: the union of keys over the 24 G6 `training_summary.json`
files is **25**; the trainer's summary writes **17**; the **8** missing keys are
exactly §6.219's fingerprint (`bf16`, `dmpnn_backend`,
`dmpnn_backend_requested`, `tf32`, `torch_compile_cuda_graphs_disabled`,
`torch_compile_enabled`, `torch_compile_mode`, `w2_gradient_mode`).

**The consequence is live, not historical.** A G6-style ablation launched today
would again produce summaries that (a) cannot be version-matched against G5 and
(b) cannot say which lever was disabled — the exact defect that made the
published per-lever Δ table irreproducible (§6.221) would recur.

---

## 4. The fix (ADDITIVE ONLY)

`molmetal/scripts/train_dmpnn_multitask_wasserstein.py` gained one module-level
function `lever_and_build_fingerprint(args)` and one `**`-merge into the summary
dict. Recorded keys:

| lever (§6.220) | build field (§6.219) |
|---|---|
| `single_task` | `tf32` |
| `kendall` | `torch_compile` |
| `dann` | `w2_gradient_mode` |

`wasserstein_weight` — the fourth lever — was already written. The keys are
FLAT and sit next to `wasserstein_weight`, so a future G5-vs-G6 comparison is a
plain key-set difference: `set(g6_summary) - set(g5_summary)`.

**Scope discipline.** The 8 fields the G6 artifacts carry were removed together
with the rest of the Wf-A17 revert. They are **NOT** resurrected — that would
revive deliberately reverted behaviour (AGENTS.md §1.4: no scope creep, no
quietly restoring reverted code). What is restored is the subset the CURRENT
trainer actually knows.

### 4.1 §5.7 50-step CPU smoke — MEASURED 2026-10-31

`CUDA_VISIBLE_DEVICES=""` + `--metals Ru --seeds 0 --epochs 1 --batch-size 64
--hidden 32 --depth 2 --n-mc 4 --device cpu` → **rc = 0**, **58 s** wall. The
emitted `training_summary.json` carries **23** keys (17 pre-existing + 6 new):

```
single_task      = False
kendall          = True
dann             = True
tf32             = False
torch_compile    = False
w2_gradient_mode = "sliced"
wasserstein_weight = 0.05
per_metal.Ru     = n_seeds 1, test_auc_mean 0.4912384351047719, status "OK"
```

### 4.2 New regression test — MEASURED 2026-10-31

`molmetal/tests/test_wf_a126_lever_fingerprint.py` → **8 passed** (1.48 s,
CPU-only). The load-bearing one is
`test_the_four_g6_arms_are_pairwise_distinguishable` — the four G6 arms now
produce **pairwise distinct** recorded signatures, which is the §6.220 root
cause closed.

---

## 5. The three live documents, now dated-corrected

| file | what was live | correction |
|---|---|---|
| `docs/STATUS.md` | "G6 per-innovation ablation **18 tuples MEASURED on disk** (… W₂ Δ≈0, Kendall Δ≈−0.01, all inside std)" | dated `**SUPERSEDED 2026-10-31 (Wf-A124): … did NOT measure 18 tuples, and its per-lever Δ do not reproduce.**` suffix (24 runs, 2 of 6 / 0 of 6, \|t\| = 1.3177 < 2.5706) |
| `TODO/completed/337g1_g6_recovery.md` | the 2026-09-29 in-flight account, the **origin** of the `ALL_SEEDS_FAILED (CUDAGraphs RuntimeError)` claim §6.222 refutes | dated `> ⚠️ **SUPERSEDED 2026-10-31 (Wf-A124) … REFUTED by the artifacts.**` blockquote (24 runs, master.log **266** lines / mtime 2026-10-27 02:03:33, `git ls-tree … grep -c 'run\.log'` → **0**) |
| `TODO/completed/339d2_lambda_roadmap_audit.md` | "60 ablation runs (4 configs × 5 metals × 3 seeds) **NOT STARTED**" | dated `**SUPERSEDED 2026-10-31 (Wf-A124): PARTLY RUN — 24 of the 60 tuples completed …**` suffix; the original tail ("BLOCKED on user OK per kill_then_adapt §0") preserved verbatim |

All three use the dated bold-marker style already used in `TODO/INDEX.md`
(`— **SUPERSEDED 2026-10-31 (Wf-A124): …**`) rather than introducing a second
convention: none of the three files used `~~strikethrough~~` before this wave.

---

## 6. The navigation row that never existed

`TODO/INDEX.md` had **no §22 row for Wf-A124** — the wave's close-out file
existed on disk, but the navigation index never listed it, so the wave was not
reachable from the index (AGENTS.md §5.5 filesystem-truth). Added as
**§22.165** (Wf-A124) and **§22.166** (Wf-A126), plus a new §0 timeline trailer
in `TODO/INDEX.md` and `TODO/README.md` (each preceding trailer PRESERVED
VERBATIM).

---

## 7. Filesystem-truth evidence (§5.5) — `ls -la` of every file this wave wrote

```
-rw-r--r-- 1 hugo hugo  26908 Oct 31 09:04 AGENTS.md
-rw-r--r-- 1 hugo hugo 1598080 Oct 31 09:04 docs/HONEST_NEGATIVE_CATALOG.md
-rw-r--r-- 1 hugo hugo  14926 Oct 31 08:58 docs/STATUS.md
-rw-r--r-- 1 hugo hugo 209715 Oct 31 08:57 molmetal/scripts/train_dmpnn_multitask_wasserstein.py
-rw-r--r-- 1 hugo hugo   8234 Oct 31 08:57 molmetal/tests/test_wf_a126_lever_fingerprint.py
-rw-r--r-- 1 hugo hugo  11908 Oct 31 08:58 TODO/completed/337g1_g6_recovery.md
-rw-r--r-- 1 hugo hugo  28342 Oct 31 08:58 TODO/completed/339d2_lambda_roadmap_audit.md
-rw-r--r-- 1 hugo hugo 192446 Oct 31 09:04 TODO/INDEX.md
-rw-r--r-- 1 hugo hugo  46889 Oct 31 09:04 TODO/README.md
```

---

## 8. Close-out checks (MEASURED, 2026-10-31, CPU-only)

| check | result |
|---|---|
| new regression test | `8 passed` (1.48 s) |
| §5.7 50-step CPU smoke | **rc = 0**, 58 s, **23** summary keys |
| GPU | `nvidia-smi --query-compute-apps` → **0** mol-metal processes |
| GPU watchdog (`bpzsdj6aj`) | **0** violations |
| catalog | **114** `## §6.` headings; new = §6.223; line-5 counter = `COUNTER 277 → 278` |
| AGENTS.md count sentence | **3/3** places read "**278 entries** (234 active) … latest = §6.223 Wf-A126 2026-10-31"; **0** occurrences of the old "277 entries" |
| TODO/INDEX.md | §22.165 + §22.166 present; §0 trailer updated |
| paper/ | **0** touches |
| wet-lab | **0** |
| commit / push | **none** (the working tree carries the wave; the harness rule is to commit only when asked) |

---

## 9. party-twenty-spirit §4 mapping (二十大精神 × Wf-A126)

```
[二十大精神 × Wf-A126 收尾]
不忘初心: ✅ 离 headline 更近 — G6 消融的"哪个 lever 说了算"从未被 artifact 回答过;
           现在 artifact 自述, 下一个消融跑出来的 Δ 表可以复现
谦虚谨慎: ✅ 承认 12 条"blocking defect"里 10 条早已关闭、2 条评论本身是错的;
           不把 critic 的 NO-GO 当成事实, 也不假装本波修好了 12 条
敢于斗争: 下阶段 = Wf-A125 (guard/gate 残余) + 重跑 G6 全 60 tuple 的 CPU 预演
守: MLC + CFM + click chemistry + PLL + 84-rule MLE 一根线没动; 本波只改 trainer 的
     自述字段, 不改训练行为
创新: 把"revert 会悄悄重新武装一个已被诊断的缺陷"变成 catalog §6.223 的可查条目
不抄: 没造新表 — 每一行都来自 per_seed_metrics.json
系统: 与 Wf-A124 的关系 = 补它自己 critic 的残余, 不重跑它的结论; 与 Wf-A125 = 串行
     (三者都写 catalog + AGENTS.md 计数句, 并发 = lost update)
5-gate: docs gate ✅ (三份 live doc 已改) · integration gate ✅ (新字段不改动训练路径)
高质量发展: 真价值 = 一个 trainer 函数 + 8 个测试 + 三份文档的 dated 更正;
           凑数项 = 无 — 没有新表格, 没有新指标
绿色: 全部 CPU-only, 0 GPU slot, 0 CUDA 调用
自我革命: 数据真实 ✅ 24 runs / 266 行 master.log / 23 keys / 8 passed 全部落盘可查
           失败暴露 ✅ §6.216..§6.223 八条负面全部进 catalog
           0 容忍 ✅ 没有把 PROJECTED 写成 MEASURED
决策: 继续当前路线 — 下一波 Wf-A125 (guard + gate 残余), 同样串行、同样 0 GPU
      TODO 更新操作: TODO/INDEX.md 新增 §22.165/§22.166 + §0 trailer;
                     TODO/README.md 新增 trailer; AGENTS.md 计数句 ×3
```

---

## 10. What was NOT fixed (KNOWN, OPEN)

* **The G6 tree's own `plan.json` / `training_summary.json` still say
  `PLAN_ONLY` / `total_pairs: 0`** while the tree holds 15 completed G5 pairs.
  Documented in §6.219; not repaired here (not this wave's owned set).
* **The G4 slot's committed blob is still a live symlink in git HEAD** — closed
  on disk by a regular-file stub, still open in the commit (§6.219).
* **`.gitignore:33` `**/*.log`** is why the operational logs vanish from the
  tree; not owned here.
* **The 36 never-run G6 tuples (Ir, Os, Re)** remain unrun; no GPU was used
  and none is authorized under the standing no-GPU directive.
* **The G5 tree can never be version-matched to a future G6 tree** — the 8
  fingerprint fields were never in G5. The fix here prevents NEW runs from
  being unmatchable; it cannot retro-fit G5.
