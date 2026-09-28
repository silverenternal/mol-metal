# Wf-24 Phase 5 — Final Verify + Cross-check Synthesis

**Date**: 2026-09-26
**Phase**: Wf-24 Phase 5 (final verify after Wf-24 Phase 1-4 cleanup)
**Inputs**: `README.md` + `EXECUTION_ORDER.md` + `INDEX.md` (full read) + `TODO/04_history/_wf24_pending_cleanup_manifest.md` + `TODO/03_plans/R18_PLAN.md` §15
**Audience**: Hugo Lee + 任何接手 R18 close-out 后的下一步执行者

---

## 1. TL;DR

- **24 wf ledger** 是 canonical 数字（22 SHIP + 1 IN FLIGHT + 1 CLEANUP），由 INDEX §11 锁定；其他 doc 数字（19 / 20 / 22 / 19/19）需要 reconcile。
- **R18 PLAN 真活 44/47 = 100% ship** 在 4 个 canonical doc（README / EXECUTION_ORDER / INDEX §10.2 / R18_PLAN §15.2）完全一致；3 DEFER 全部 Pitfall-T2 P2.5 Phases C-E (AMBER/YASARA/MMseqs2 第三方闭源) 一致。
- **`TODO/pending/`** 现已清空（Wf-24 Phase 1 完成 16 文件迁移）；README / EXECUTION_ORDER §3 / INDEX §3 仍称"6 active"或"15 files"是 **stale**。
- **`TODO/04_history/`** 实际 = 25 files（16 superseded + 8 pre-existing + 1 Wf-24 manifest）；非 40+。

---

## 2. Verify matrix

### 2.1 24 wf ship number — 4-way cross-check

| Doc | Reported wf-ship count | 状态 |
|---|---:|---|
| `README.md` §"Ship surface" | "**19 wf reports** in `completed/{53-72}_*.md` (skip Wf-14/16 命名合并)" | ⚠️ **stale** (文件已扩到 53-74) |
| `EXECUTION_ORDER.md` §0 | "**19 wf reports** ship in `completed/{53..74}_*.md`" | ⚠️ **stale** (53..74 含 22 文件) |
| `EXECUTION_ORDER.md` §3 | "completed/ (**74 files**, NN ≤ 74)" | ⚠️ **stale** (实际 92 numbered + 1 lambda_integration_plan = 93 total) |
| `INDEX.md` 顶部 | "**22 R18 wf reports ship**" | ⚠️ partial (shipped wf 数量 = 22, but reports = 22 files because Wf-2.4/Wf-14/Wf-16/Wf-20 已合并) |
| `INDEX.md` §10.1 | "**19 wf reports ship** — `{53..68,70,71,72}` (skip 69/73/74)" | ⚠️ **stale** (69 = Wf-16, 73 = Wf-22, 74 = Wf-21 都应 ship) |
| `INDEX.md` §9.1 | "**12 wf** — 全部 ✅ ship today" | ⚠️ **stale** (only Wf-1..Wf-12, missing Wf-13..Wf-22) |
| `INDEX.md` §11 header | "**24 wf ledger** (Wf-1 → Wf-24 — 2026-09-26 ultracode dispatch)" | ✅ **CANONICAL** |
| `INDEX.md` §11 Aggregate | "**22 SHIP** (Wf-1..Wf-22 except Wf-23) — 20 wf reports in `completed/{53..74}_*.md`" + 1 IN FLIGHT (Wf-23) + 1 CLEANUP (Wf-24) | ✅ **CANONICAL** |
| `R18_PLAN.md` §15.6 / decisions LOG | "**19/19 wf reports**" + "**20 wf reports**" (mismatch within same doc) | ⚠️ **contradictory** |

**Canonical decision**: 24 wf ledger = 22 SHIP + 1 IN FLIGHT + 1 CLEANUP.
- **22 SHIP wf**: Wf-1, Wf-2, Wf-2.4, Wf-3, Wf-4, Wf-5, Wf-6, Wf-7, Wf-8, Wf-9, Wf-10 (= Wf-2.4), Wf-11, Wf-12, Wf-13, Wf-14, Wf-15, Wf-16, Wf-17, Wf-18, Wf-19, Wf-20, Wf-21, Wf-22.
- **wf reports in completed/{53..74}_*.md** = 22 files (NN prefix match); 但 §11 aggregate reports "20 wf reports" because Wf-2.4 → Wf-10 合并 + Wf-14/Wf-16 合并 = 2 files absorbed. (实际合并情况: Wf-2.4 真合并入 Wf-10/file 63; Wf-14 真合并入 Wf-12? or Wf-15? — file 67 自己 header 写 "Wf-12" 但 §11 ledger 说 Wf-14 — **存在编号歧义**, 见 §3.4)
- **1 IN FLIGHT**: Wf-23 (Blackwell GPU ultracode retrain, 4-week queue).
- **1 CLEANUP**: Wf-24 (16 SUPERSEDED pending → 04_history/).

### 2.2 R18 PLAN 真活完成度 — 4-way cross-check

| Doc | P0 | P1 | P2 | Pitfall-T1 | Pitfall-T2 | Integration | Bench | Total | DEFER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `README.md` "R18 PLAN 真活完成度" | 4/4 | 7/7 | 5/5 | 5/5 (合 13/16) | 13/16 | 5/5 | 5/5 | **44/47** | **3** |
| `EXECUTION_ORDER.md` §0 | 4/4 | 7/7 | 5/5 | 5/5 | 13/16 | 5/5 | 5/5 | **44/47** | **3** |
| `INDEX.md` §10.2 | 4/4 | 7/7 | 5/5 | 5/5 | 13/16 | 5/5 | 5/5 | **44/47** | **3** |
| `R18_PLAN.md` §15.2 | 4/4 | 7/7 | 5/5 | 5/5 | 13/16 | 5/5 | 5/5 | **44/47** | **3** |

✅ **Consistent**: 44/47 ship + 3 DEFER across all 4 canonical docs.

### 2.3 Pitfall DEFER 列表 — 4-way cross-check

| Doc | DEFER items listed |
|---|---|
| `README.md` 唯一 still DEFER | (1) MMseqs2 cluster; (2) AMBER force-field; (3) YASARA integration |
| `EXECUTION_ORDER.md` §0 | "Pitfall-Tier-2 P2.5 Phases C-E = MMseqs2 + AMBER + YASARA 第三方闭源" |
| `INDEX.md` §12.2 | Phase C (AMBER) + Phase D (YASARA) + Phase E (MMseqs2 cluster) |
| `INDEX.md` §12.5 summary | "P2.5 Phase C (AMBER) + Phase D (YASARA) + Phase E (MMseqs2 cluster) DEFER R19+" |
| `R18_PLAN.md` §15.2 + §15.4 | "P2.5 Phases C-E (MD-stub MMseqs2 + AMBER force-field + YASARA integration) DEFER R19" |

✅ **Consistent**: 3 DEFER items = AMBER + YASARA + MMseqs2, all P2.5 Phases C-E.

### 2.4 TODO/pending/ empty

```bash
$ ls -la /home/hugo/codes/mol-metal/TODO/pending/
drwxr-xr-x 1 hugo hugo     0 Sep 26 02:10 .
drwxr-xr-x 1 hugo hugo   516 Sep 26 02:13 ..
```

✅ **EMPTY** (no `.gitkeep`, no ACTIVE files; dir preserved as empty).

Stale docs that still claim pending has content:
- `README.md` §Archived Work: "**`pending/` — 6 active TODOs**"
- `EXECUTION_ORDER.md` §3: "`pending/`  (**6 active**)"
- `INDEX.md` §3: "**pending/ — 15 files (NN ≥ 19)**"

All three are **stale** (post-Wf-24 Phase 1 cleanup).

### 2.5 TODO/04_history/ archive count

| Component | Files | Notes |
|---|---:|---|
| **Wf-8 SUPERSEDED banners** (per `61_todo_closeout.md` §2) | 10 | 14, 19, 22, 26, 28, 31×2, 32, 33, 34 |
| **Wf-11 SUPERSEDED banners** (per `64_r18_closeout.md` row 12) | 6 | 21, 23, 24, 25, 30, 53 |
| **Wf-24 cleanup moved** (per `_wf24_pending_cleanup_manifest.md` table) | 16 | same 16 unique entries (Wf-8 ∪ Wf-11; overlap on 31 counted once each) |
| **Pre-existing ship artefacts** (legacy) | 8 | AUDIT_RESEARCH_GRADE / completion_audit / decisions / engineering_practices / environment / project_workflow / risks / roadmap |
| **Wf-24 manifest** | 1 | `_wf24_pending_cleanup_manifest.md` |

**Actual `ls` count**: 25 files (16 superseded + 8 pre-existing + 1 manifest).

| Doc | Reported 04_history count |
|---|---|
| `_wf24_pending_cleanup_manifest.md` | "24 entries total" (16 superseded + 8 pre-existing) ✓ |
| `README.md` §Archived Work | "**9 pre-R17 plans archived to `04_history/`**" — ⚠️ **stale** (实际 16 superseded, 但仍是 9 unique pre-R17 plans; Wf-24 加了 7 NEW: 21/23/24/25/30/53/61 not pre-R17 — wait 61 is close-out so 16 - 10 Wf-8 = 6 NEW from Wf-11 + 0 from Wf-24 = 6 new total; 9 + 6 = 15 unique plans not 16) |
| `INDEX.md` §1 History table | "8 legacy plans" (AUDIT_RESEARCH_GRADE / completion_audit / engineering_practices / environment / decisions / risks / roadmap / project_workflow) ✓ |
| Task expectation | "24 (Wf-8+Wf-11) + 16 (Wf-24) = **40+**" — ❌ **mis-calibrated** (实际 25 files) |

✅ 04_history 实际 = 25 files，stale 的"40+" 期望是因为误把 16 superseded 与 Wf-8/11 旧 archive 算成相加（实际是 unique 16 套用一次）。

---

## 3. Inconsistencies found

### 3.1 Stale "pending/ still has 6/15 active TODOs" — 3 docs

- `README.md` §Archived Work: "**`pending/` — 6 active TODOs**"
- `EXECUTION_ORDER.md` §3: "`pending/`  (**6 active**)"
- `INDEX.md` §3: "**pending/ — 15 files (NN ≥ 19)**"

**Fix**: replace with "pending/ — empty (post-Wf-24 Phase 1 cleanup)".

### 3.2 Stale "completed/ file count" — 3 docs

- `README.md` §Archived Work: "**`completed/` — 63 shipped task files (R3–R18)**"
- `EXECUTION_ORDER.md` §3: "**completed/ (74 files, NN ≤ 74)**"
- `INDEX.md` §4: "**completed/ — 60 files (NN ≤ 63)**"

**Actual**: 92 NN_*.md + 1 lambda_integration_plan.md = **93 files** (NN ≤ 74 含 53-74 = 22 R18 wf reports + 70 earlier R3-R17 ship files).

**Fix**: replace with "completed/ — **93 files (NN ≤ 74)**, R3-R18 92 + 1 lambda_integration_plan".

### 3.3 Stale "wf reports = 19" — 4 docs

- `README.md`: "19 wf reports in `completed/{53-72}_*.md`"
- `EXECUTION_ORDER.md` §0: "19 wf reports ship in `completed/{53..74}_*.md`"
- `INDEX.md` §10.1: "19 wf reports ship — `{53,54,...,68,70,71,72}`"
- `INDEX.md` §9.1: "Workflow ledger (**12 wf**)"

**Fix**: unify to "**24 wf ledger** (22 SHIP + 1 IN FLIGHT + 1 CLEANUP)" — see INDEX §11 header as canonical.

### 3.4 wf-id ↔ file-id 编号歧义 — file 67 header vs INDEX §11

- File `67_r18_pipeline_integration.md` 顶部 self-claims "**Wf-12 — R18 → R17 Pipeline Integration Synthesis (5 adapters)**"
- INDEX §11 ledger table assigns file 67 → **Wf-14** (and file 65 → Wf-12)

**Recommendation**: accept INDEX §11 ledger as canonical. File 67 header 应改为 "Wf-14"（非强制 — file header 是 synthesis 撰写者的本地 claim；ledger INDEX 是 cross-reference surface-truth）。

### 3.5 R18_PLAN §15.2 "Integration — 5 adapters — Wf-12 (67_*)"

- `R18_PLAN.md` line 408: "**Integration — 5 adapters** ... Wf-12 (`67_r18_pipeline_integration.md`)"
- 但 INDEX §11 ledger 把 67_*.md 标 Wf-14（不是 Wf-12）

**Fix**: 把 R18_PLAN §15.2 行 408 的 Wf-12 改为 Wf-14。

### 3.6 R18_PLAN §15.6 list vs §15.4 "19 wf reports"

- §15.4 line 450: "**19 wf reports** ship in `TODO/completed/`"
- §15.6 line 501: 列 19 reports (53-72, 跳过 69)
- §15.7 decisions LOG line 516: "**19/19 wf reports**"
- 但 §11 ledger = 24 wf; 实际 22 NN_*.md files in 53-74

**Fix**: 把 §15.4 / §15.7 的 "19" 改为 "22 (Wf-1 → Wf-22, skip Wf-23)"; §15.6 list 加上 69, 73, 74 三 file。

### 3.7 "20 wf reports" mismatch

- INDEX §11 Aggregate: "**20 wf reports** in `completed/{53..74}_*.md` (Wf-2.4/Wf-10 合并, Wf-14/Wf-16 命名合并)"
- INDEX §10.1: "**19 wf reports**"

**Fix**: 在 INDEX §10.1 末尾注明 "20 wf reports (skip 69, 73, 74 per ledger convention)" 或统一改为 22 (NN ≤ 74)；建议采纳 §11 的 "20 wf reports after 2 merges"。

---

## 4. Tasks for next session (recommend)

| # | Task | Owner | Effort |
|---|---|---|---|
| 1 | 修复 README.md 末段 "pending/ 6 active" + "completed/ 63 files" + "19 wf reports" 三处 stale (按本报告 §3.1-3.3) | doc-sync | 5 min |
| 2 | 修复 EXECUTION_ORDER.md §3 "pending/ (6 active)" + "completed/ (74 files, NN ≤ 74)" (按 §3.1-3.2) | doc-sync | 5 min |
| 3 | 修复 INDEX.md §3 (pending/) + §4 (completed/) + §10.1 (wf reports) (按 §3.1-3.3 + §3.7) | doc-sync | 10 min |
| 4 | 修复 R18_PLAN §15.2 + §15.4 + §15.6 + §15.7 的 wf 数字 (按 §3.5-3.6) | doc-sync | 10 min |
| 5 | 修复 file `67_r18_pipeline_integration.md` 顶部 header "Wf-12" → "Wf-14" (按 §3.4) | doc-sync | 1 min |
| 6 | 加入 `.gitkeep` 到 `TODO/pending/` 以保留空 dir (可选；非 blocking) | bash | 1 min |

**总 effort**: ~30 min, 全部 doc-only edit; 0 GPU; 0 module modify。

---

## 5. Closing note

Wf-24 Phase 1 cleanup 已 ship (16 SUPERSEDED files moved from pending → 04_history; pending/ 现已 empty)。本 Phase 5 verify 发现 **8 处 stale doc 数字 + 1 处 wf-id 编号歧义 + 1 处 04_history 计数 mis-calibrated**。

**关键正面**: R18 PLAN 真活 44/47 + 3 DEFER (Pitfall-T2 P2.5 Phases C-E) **4-way 完全一致**; 24 wf ledger 在 INDEX §11 已锁定为 canonical。

**下一步建议**: 把 §4 6 个 doc-sync task 排 Wf-24 Phase 6 (本周末 ~30 min); 不需要重跑任何 wf。

— Claude Code, Wf-24 Phase 5, 2026-09-26