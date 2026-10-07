# Wf-R39 — Uncovered TODO / planned-task audit (triage wave closeout)

**Date**: 2026-10-08
**Workflow**: `wf_r39_uncovered_todo_audit` (`/tmp/wf_r39_uncovered_todo_audit.js`, 4 phases)
**Scope**: Inventory + categorize + resolve the `TODO/pending/` + `TODO/03_plans/` tasks NOT covered by wf-R31..R38 + the L1-L5 wave chain.
**Constraints honored**: AGENTS.md §5.4 (no wet-lab) · §5.5 (filesystem-truth) · §5.6 (atomic write) · §5.7 (smoke) · §5.8 (token budget) · §5.9 (single venv) · §5.10 (OmegaFold gap-fill) · §5.3 / TODO-84 (no `paper/*`).

---

## §0 TL;DR

Wf-R39 was a **triage wave, not a compute wave**. Of the 11 audited open task files (7 wf-numbered `TODO/pending/*.md` + 4 `TODO/pending/plan_a_*.md`), **3 were moved `pending/ → completed/`** (each backed by a self-declared terminal `Status:` line *plus* a `TODO/completed/` counterpart verified by `ls -la`), **8 were correctly RETAINED as still-open** (GPU-USER_OK / GPU-BLOCKED / live gating chains). In the CPU bucket, 1 latent import-time bug was fixed (`extract_rdkit_2d_descriptors.py` lazy-import guard was dead code → `NameError` on an RDKit-less host) with a NEW 6-case regression test **6/6 PASS in 1.86 s**. Documentation surfaces updated: `docs/HONEST_NEGATIVE_CATALOG.md` **NEW §6.37** and `TODO/INDEX.md` **NEW §22.150**. Final pre-flight: **106 PASS / 0 FAIL / 2 SKIP / 108** (MEASURED, 295.9 s wall). **Wf-R39 did NOT unblock L4 or L5** — both remain BLOCKED on the P-A protocol patch + 5-fold real-only re-measure, with 4 yaml gate FAILs still open. Net honest count: **3 task files closed, 8 task files remain open, 1 CPU code fix shipped, 0 GPU launches, 0 trainer patches, 0 `paper/*` touches.**

---

## §1 Inventory results (Phase 1 — 2 parallel read-only agents)

Two read-only inventory agents covered (a) the 7 wf-numbered `TODO/pending/*.md` files and (b) `TODO/03_plans/` + the 4 `plan_a_*.md` + `PROTOCOL_RECONCILIATION_AND_RE_PLAN_2026_10_07.md`.

### §1.1 The 7 wf-numbered pending files (post-R39 disk state, `ls -la` MEASURED)

| # | File | Location | Size (B) | Self-declared `Status:` | Coverage | Aligns to |
|---:|---|---|---:|---|---|---|
| 177 | `177_wf123_p3_m1m2_downstream_metric_lift.md` | `pending/` (RETAINED) | 9 732 | `🚧 P1 — IN-FLIGHT … downstream lift NOT yet measured` | partial | L2 (m2 R²) |
| 179 | `179_wf123_p3_w2_pic50_alignment.md` | `pending/` (RETAINED) | 17 079 | `🚧 P1 — IN-FLIGHT` | partial | L4 / L5 (W2 pIC50) |
| 183 | `183_wf123_wf124_p3_gap_routing_plan.md` | `pending/` (RETAINED) | 16 733 | `📐 PLANNING` | uncovered | L4 / L5 (routes TODOs 177-182) |
| 206 | `206_m6_kappa_metric_redefinition.md` | `completed/` (**MOVED**) | 12 898 | `SHIPPED-PARTIAL → FINALIZED via Wf-381-A4` | **covered** | m6 κ |
| 234 | `234_wf234_frontier_math_audit_integration.md` | `completed/` (**MOVED**) | 30 335 | `AUDIT COMPLETE — read-only` | **covered** | frontier math |
| 376a3 | `376a3_m2_r2_gpu_retrain_plan.md` | `pending/` (RETAINED) | 17 852 | `PLAN ONLY — no retrain launched. Awaiting USER_OK=1` | uncovered | L2 (m2 R² GPU) |
| 383 | `383_se3_sb_fast_converging_integration.md` | `completed/` (**MOVED**) | 33 161 | `PLAN_READY (CPU-only plan)` | **covered** | SE(3)-SB wire |

**Coverage summary (Phase 1)**: **3 covered** (206 / 234 / 383) / **2 partial** (177 / 179) / **2 uncovered** (183 / 376a3).

### §1.2 The 4 `plan_a_*.md` gating chain (all RETAINED)

| File | Scope | GPU-hr | USER_OK? | BLOCKED-on |
|---|---|---:|---|---|
| `plan_a_overview.md` | 方案A master overview | 0 | n/a | — (`APPROVED by user` 2026-10-07) |
| `plan_a_data_pipeline.md` | Step 1 data pipeline | 0 | no | gates Step 2-3 |
| `plan_a_step2_cfm_prior.md` | Step 2 CFM prior | ~10 | yes | Step 1 + `cfm_r10` fix |
| `plan_a_step3_closed_loop.md` | Step 3 closed loop | ~30 | yes | Step 1 + 2 |

None of the 4 is a finished doc — they are a **live serial gating chain** (Step 1 → 2 → 3).

### §1.3 `TODO/03_plans/` — P-A..P-F master plan status

`PROTOCOL_RECONCILIATION_AND_RE_PLAN_2026_10_07.md` (the wf-R32 P-A..P-F master plan) drives the L4/L5 chain. **P-A (protocol patch) remains open** — this is the single root blocker for L4 and everything downstream of it. The 5 active yamls still FAIL 4 yaml gates (per the Wf-A8c closeout "Residual gaps").

---

## §2 Categorization matrix (Phase 2 — `categorize` agent)

Buckets: (a) docs-only · (b) CPU-only · (c) GPU-USER_OK · (d) GPU-BLOCKED · (e) DEPRECATED · (f) already-covered.

| Task | Category | Est. wall | USER_OK? | BLOCKED-on | R39 action |
|---|---|---|---|---|---|
| 206 m6 κ redefinition | (a) docs-only | 0 | — | — | **MOVED** → `completed/` |
| 234 frontier-math audit | (a) docs-only | 0 | — | — | **MOVED** → `completed/` |
| 383 SE(3)-SB integration | (a) docs-only | 0 | — | — | **MOVED** → `completed/` |
| rdkit-2d lazy-import guard | (b) CPU-only | ~10 s | no | — | **FIXED** + 6-case test |
| Catalog §6 + INDEX §22 entries | (a) docs-only | 0 | — | — | **ADDED** (§6.37 / §22.150) |
| 177 m1/m2 downstream lift | (d) GPU-BLOCKED | ~4 hr | yes | P-A / m2 R² | RETAINED |
| 179 w2 pIC50 alignment | (d) GPU-BLOCKED | ~4 hr | yes | P-A + 5-fold real-only | RETAINED |
| 183 gap-routing plan | (d) GPU-BLOCKED | routing-only | yes | target TODOs 177-182 open | RETAINED |
| 376a3 m2 R² GPU retrain | (c) GPU-USER_OK | ~6 hr | yes | `USER_OK=1` | RETAINED |
| plan_a Step 1 / 2 / 3 | (c)+(d) | 0 / ~10 / ~30 | yes (2,3) | serial chain | RETAINED |
| 7 `wetlab_*` files | (e) DEPRECATED | — | — | — | **RETAINED NOT DELETED** (§10.5) |

**Bucket counts**: 5 (a) docs · 1 (b) CPU · 3 (c) GPU-USER_OK · 4 (d) GPU-BLOCKED · 7 (e) DEPRECATED · 3 (f) covered.

**Top-3 priority (value-per-token)**: 1) the CPU rdkit lazy-import fix (real latent bug, ~10 s, 1 test); 2) the 3-file docs triage (removes 3 phantom "open" items from the queue); 3) the §6.37 + §22.150 doc sync (keeps the single-source-of-truth honest).

**Re-categorization note**: the rdkit lazy-import item was initially filed as "docs / covered" during Phase 1 inventory but Phase 3's disk grep against the module source proved it a genuine **(b) CPU code defect** — the guard set `_HAS_RDKIT=False` yet two module-scope descriptor registries dereferenced `Descriptors.MolWt` unconditionally, so the module raised `NameError` on import on any RDKit-less host. Per memory `verify-premises-before-proposing.md`, the premise was re-checked against source before acting.

---

## §3 Resolution outcomes (Phase 3 — 2 parallel agents)

### §3.1 `resolve-docs` (commit `d5a29f7787`)

**Files MOVED `pending/` → `completed/` via `git mv` per §5.6 (3)**:

| From | To | Size (B) | Terminal `Status:` basis | `completed/` counterpart (MEASURED) |
|---|---|---:|---|---|
| `206_m6_kappa_metric_redefinition.md` | `completed/206_…md` | 12 898 | `FINALIZED via Wf-381-A4` | `381a4_m6kappa_finalize.md` 17 120 B |
| `234_wf234_frontier_math_audit_integration.md` | `completed/234_…md` | 30 335 | `AUDIT COMPLETE` | `189_frontier_math_integration.md` 17 778 B |
| `383_se3_sb_fast_converging_integration.md` | `completed/383_…md` | 33 161 | `PLAN_READY` | `390a3_se3_sb_wire.md` 17 731 B + `383a2_…md` 20 987 B |

**Docs surfaces updated**:
- `docs/HONEST_NEGATIVE_CATALOG.md` — **NEW §6.37** `wfr39_docs_only_pending_to_completed_closeout` (`HONEST_NEGATIVE_DOCS_ONLY_TRIAGE`), appended immediately before `## §7 WETLAB_DEPRECATED`. Grep-confirmed 0 pre-existing `§6.37` before the write. File 140 450 B → 146 344 B (+3 613 B / +16 lines).
- `TODO/INDEX.md` — **NEW §22.150** Wf-R39 docs-only pending→completed triage entry.

### §3.2 `resolve-cpu` (commit `b85acf6e0f`)

| File | Δ |
|---|---|
| `molmetal/scripts/extract_rdkit_2d_descriptors.py` | +143 / −45 LOC |
| `molmetal/tests/test_wf_r39_cpu_lazy_rdkit.py` | **NEW** 195 lines / 7 890 B |

**Fix**: both descriptor registries were stored as frozen `(column_name, rdkit_attr_name)` pairs; the RDKit callables are now resolved only when RDKit is importable, so the module imports cleanly on an RDKit-less host instead of dying with `NameError: name 'Descriptors' is not defined`.

**Test verdict (MEASURED, re-run at wave-close)**: `.venv/bin/python -m pytest molmetal/tests/test_wf_r39_cpu_lazy_rdkit.py -q` → **6 passed in 1.86 s** (1 hypothesis-plugin warning, unrelated).

**Pre-flight verdict**: **106 PASS / 0 FAIL / 2 SKIP / 108**, 295.9194 s wall (MEASURED at wave-close, `WF35_PROTOCOL_LOCK=1 WFA2_CONFIG_STRICT=1`).

**§5.7 smoke**: not applicable — no trainer flag was wired in this wave; the CPU change is descriptor extraction, not a training-loop site.

**§5.9 single-venv**: every invocation above used `/home/hugo/codes/mol-metal/.venv/bin/python`.

### §3.3 §5.5 filesystem-truth evidence (`ls -la`, MEASURED 2026-10-08)

```
-rw-r--r-- 1 hugo hugo 12898 Oct  3 17:58 TODO/completed/206_m6_kappa_metric_redefinition.md
-rw-r--r-- 1 hugo hugo 30335 Oct  3 17:58 TODO/completed/234_wf234_frontier_math_audit_integration.md
-rw-r--r-- 1 hugo hugo 33161 Oct  3 17:58 TODO/completed/383_se3_sb_fast_converging_integration.md
-rw-r--r-- 1 hugo hugo  9732 Sep 30 10:04 TODO/pending/177_wf123_p3_m1m2_downstream_metric_lift.md
-rw-r--r-- 1 hugo hugo 17079 Oct  1 17:00 TODO/pending/179_wf123_p3_w2_pic50_alignment.md
-rw-r--r-- 1 hugo hugo 16733 Oct  3 17:02 TODO/pending/183_wf123_wf124_p3_gap_routing_plan.md
-rw-r--r-- 1 hugo hugo 17852 Oct  3 17:58 TODO/pending/376a3_m2_r2_gpu_retrain_plan.md
-rw-r--r-- 1 hugo hugo  7890 Oct  8 00:42 molmetal/tests/test_wf_r39_cpu_lazy_rdkit.py
```

---

## §4 Cross-references — L1 / L3 / L4 / L5 + wf-R38 ship evidence

| Wave | Status after Wf-R39 | Evidence |
|---|---|---|
| **L1** — CFM m7 diversity (~4 hr GPU) | **LAUNCHED 2026-10-07** (in flight) | `scripts/launch_wf124_p3_g5.sh`; `--stable-vm` bond-loss guard in `train_cfm_yuelbond_wasserstein.py` |
| **L2** — m2 R² per-cell lift | covered by m2 lock entry; **LOMO real-only still GPU-USER_OK** | `TODO/pending/376a3_m2_r2_gpu_retrain_plan.md` (awaiting `USER_OK=1`) |
| **L3** — VariancePUCT bench | **SHIPPED 2026-10-07** | commits `f82ed1e04f` + `0aa0221b40` (wf-R37/R38 chain); `ab_comparison.json` reads bench JSON `agg[]` |
| **L4** — LOMO real-only (~6 hr GPU) | **BLOCKED** | P-A protocol patch + 5-fold real-only re-measure; 4 yaml gate FAILs open |
| **L5** — Wasserstein re-bench (~4 hr GPU) | **BLOCKED on L4** | upstream of L4 |
| **wf-R38 ship evidence** | `--stable-vm` is **FLAG_ONLY** on bond loss; removed from `scripts/launch_wf299_cfm_main.sh` `CFM_NEW_FLAGS`; trainer's startup guard correctly refused to train | catalog §6.35 (2026-10-07 23:33 MEASURED) + commit `5dc930601a` |
| **wf-R39 CPU fix** | rdkit-2d lazy-import guard made live + 6-case regression test | commit `b85acf6e0f` |
| **wf-R39 docs triage** | 3 MOVED / 8 RETAINED + catalog §6.37 + INDEX §22.150 | commit `d5a29f7787` |

**Final `git diff --stat HEAD`**: `52 files changed, 3039 insertions(+), 826 deletions(-)` — this is the whole working-tree delta, not Wf-R39's own contribution (Wf-R39's own commits are `d5a29f7787` and `b85acf6e0f`, both already committed; the remainder is pre-existing uncommitted work from earlier waves).

**`git status --short -- paper/`**: **empty** — TODO-84 honored, 0 `paper/*` touches.

---

## §5 Honest framing

**Closed by Wf-R39 (3 of 11 audited task files)**:
- `206_m6_kappa_metric_redefinition.md` — genuinely FINALIZED (counterpart `381a4_m6kappa_finalize.md` on disk, MEASURED).
- `234_wf234_frontier_math_audit_integration.md` — genuinely AUDIT-COMPLETE read-only work (counterpart `189_…md` on disk).
- `383_se3_sb_fast_converging_integration.md` — genuinely PLAN_READY and CPU-only (two counterparts on disk).

These 3 were **not** closed by new work in Wf-R39 — the work was already done in wf-R31..R38 and the files were simply never moved out of `pending/`. **What Wf-R39 added was queue hygiene, not capability.** That distinction is the honest reading of this wave.

**Shipped as genuine new work (1)**: the `extract_rdkit_2d_descriptors.py` lazy-import fix + 195-line regression test. This is the only item in Wf-R39 that changed runtime behaviour on any host.

**Still open (8 of 11)**:
- GPU-BLOCKED (4): 177, 179, 183, and the plan_a Step 2/3 chain.
- GPU-USER_OK (1 + 2): 376a3 m2 R² retrain (~6 hr, needs `USER_OK=1`); plan_a Step 2 (~10 hr) and Step 3 (~30 hr).
- Serial gating: plan_a Step 1 → 2 → 3 is strictly serial; Step 2 cannot start until Step 1 lands, Step 3 cannot start until Step 2 lands.

**BLOCKED items untouched by Wf-R39**:
1. **L4** (LOMO real-only, ~6 hr GPU) — BLOCKED on the P-A protocol patch + 5-fold real-only re-measure; 5 yamls still FAIL 4 yaml gates.
2. **L5** (Wasserstein re-bench, ~4 hr GPU) — BLOCKED on L4.
3. **`--stable-vm` velocity-loss wire-in** — deliberately deferred by wf-R38 to "Wf-R39+"; **Wf-R39 did NOT wire it**. It remains FLAG_ONLY (catalog §6.35).
4. **7 `wetlab_*` files** — DEPRECATED per AGENTS.md §5.4 / §10; RETAINED NOT DELETED per §10.5 retention policy. Tier-2 target unchanged.

**Did Wf-R39 move L4/L5 toward unblock?** **No.** Wf-R39 made **0 progress on the P-A protocol patch**, which is the sole root blocker for L4 and L5. It also closed no GPU-BLOCKED task, because closing any of them requires a measurement that only a GPU run can produce. Honest verdict: **Wf-R39 is a triage wave; the P-A patch remains the critical path for the entire L4/L5 branch.**

**MEASURED vs PROJECTED**: every number in this doc is MEASURED from disk (`ls -la`, `pytest`, pre-flight JSON summary, `git diff --stat`) except where explicitly labelled as a plan-file GPU-hour *estimate* (§1.2, drawn from the plan files' own text, which are PROJECTED wall-clock estimates, not measured runtimes).

**Cross-ref**: AGENTS.md §5.2 · §5.4 · §5.5 · §5.6 · §5.7 · §5.8 · §5.9 · §5.10 · §10 · TODO-84 · `docs/HONEST_NEGATIVE_CATALOG.md` §6.35 + §6.37 · `TODO/INDEX.md` §22.150 · `TODO/03_plans/PROTOCOL_RECONCILIATION_AND_RE_PLAN_2026_10_07.md` · commits `d5a29f7787` + `b85acf6e0f`.
