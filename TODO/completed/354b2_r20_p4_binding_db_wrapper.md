# Wf-354-B2 — R20 §4 BindingDB 1.07M launch wrapper — verify + USER_OK gate

**Date:** 2026-09-29
**Wf:** Wf-354-B2 (USER_OK-gate verification per AGENTS.md §5.1 kill_then_adapt)
**Parent:** `TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` §3 (Phase 3 — BindingDB full 1.07M retrain, 1 week GPU)
**Status:** ⚠️ **HONEST-NEGATIVE** — wrapper does NOT exist on disk; per §3 plan, the wrapper was never written; BindingDB 1.07M data NOT on disk (only 5,257 subset per `TODO/completed/148_wf146_lomo_retrain_prep_status.md` §1.1)

> **Scope ambiguity note.** Task title reads "R20 §4" but the R20 plan has Phase 3 = BindingDB (§3) and Phase 4 = tmQM+PlatinAI (§4). Both are absent on disk. The task is mapped to Phase 3 BindingDB since the title explicitly names BindingDB 1.07M. The §4 tmQM+PlatinAI wrapper is covered by `TODO/completed/354b3_r20_p5_tmqm_plat_joint_retrain_wrapper.md` (sister closeout, also HONEST-NEGATIVE).

---

## §1 Search performed

```bash
$ find . -name "*r20*p4*" -o -name "*binding_db*launch*" 2>/dev/null | head -20
./TODO/completed/95_r20_p4_3way_100pocket.md

$ find . -path ./node_modules -prune -o -type f -name "*.sh" -print \
    | xargs grep -l "bindingdb\|BindingDB\|wf175\|bd_full\|bd-2" 2>/dev/null
./scripts/wf175_bindingdb_metal_filter.py  (a Python pre-filter, not a launch wrapper)
(no launch_*.sh wrappers match BindingDB)

$ ls scripts/launch_*.sh
launch_wf119_p1_4_chemprop_v3_lomo.sh
launch_wf124_p3_g4.sh
launch_wf124_p3_g5.sh
launch_wf124_p3_g6.sh
launch_wf264_cfm_property_panel.sh
launch_wf300_cfm_main.sh
launch_wf310b_g6_final.sh
launch_wf310e_wf135_phase2_gpu.sh
launch_wf310e_wf219_production_gpu.sh
launch_wf338v2_wf176_m19_lomo_cpu.sh
launch_wf74_v2_drugood_dann.sh
```

**Verdict**: No `scripts/launch_*r20_p4*bindingdb*.sh` wrapper exists on disk. The only `*r20*p4*` glob match is `TODO/completed/95_r20_p4_3way_100pocket.md` (TODO-95 R20-P4 3-way integration doc — DIFFERENT ship target; not a launch wrapper). The only BindingDB-related code is `scripts/wf175_bindingdb_metal_filter.py` (a CPU SMARTS-based metal pre-filter, no GPU launch).

## §2 Plan spec (per `TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` §3)

| Item | Spec |
|---|---|
| **Data** | BindingDB full 1.07M → metal-filtered subset (~50K-200K) |
| **Targets** | Ki / Kd / IC50 affinity (Boltz-2 baseline Pearson r = 0.86) |
| **Model** | chemprop v2 (hidden=300, depth=3, 5 seeds × 50 epochs) |
| **Validation target** | Affinity Pearson r ≥ 0.65 |
| **ETA** | ~1 week GPU-0/1 (per §3 table) |
| **Trainer candidates** | `molmetal/scripts/train_chemprop_v2_v3.py` (Wf-266 levers WIRED) — would be the candidate trainer; no BindingDB-specific launcher exists |

## §3 Data availability gap HONEST-NEGATIVE (per `TODO/completed/148_wf146_lomo_retrain_prep_status.md` §1.1)

Even if the wrapper were written, the data is NOT on disk:

- **MEASURED**: BindingDB subset = **5,257 rows** (`molmetal/data/protein_metal_binding.csv`)
- **MEASURED**: Full BindingDB = **NOT on disk**; only the 5K subset (`TODO/completed/148_wf146_lomo_retrain_prep_status.md` §1.1 + `TODO/completed/144_q2_paper_ship_done.md` §2.4.2)
- **Note**: A 1.07M BindingDB row source *is* symlinked via `pocket/data/tdc/` per `TODO/completed/144_q2_paper_ship_done.md` §2.4.2 ("BindingDB domain-3 lifts from ~5K fallback → 1.07M real via the symlink to pocket/data/tdc/"), giving 1,072,931 rows in the four-domain combined panel. **However, this is NOT the 8-metal metal-filtered subset** required for the per-metal regression on chemprop v2; it's a generic BindingDB panel for cross-domain uncertainty.

Per `party-twenty-spirit §3.5`: the wrapper would not have data to consume. Even with a wrapper + chemprop v2 + hidden=300/depth/3 + 5-seed × 50-epoch, the run is blocked on the metal-filter step (BindingDB full download + 8-metal SMARTS filter) which lives in `scripts/wf175_bindingdb_metal_filter.py` but has not been executed end-to-end against the full 1.07M source.

## §4 Why no wrapper exists (HONEST-NEGATIVE audit)

Per the Wf-354-B3 sister closeout pattern (R20 §5 tmQM+PlatinAI joint retrain wrapper — also NOT-WRITTEN):

| Phase | Goal | Status | Why ultracode not launched |
|---|---|---|---|
| **§3 Phase 3** | BindingDB 1.07M chemprop v2 (1 week GPU) | ⏸️ NOT LAUNCHED (USER_OK-gated + DATA-HOLD) | per `TODO/completed/148_wf146_lomo_retrain_prep_status.md` §1.1 — full BindingDB NOT on disk; only 5K subset |

**Root cause**: Per Wf-148 §1.1, only the 5,257-row BindingDB subset is on disk; the full 1.07M source requires a TSV.gz download from bindingdb.org + RDKit SMARTS filter via `scripts/wf175_bindingdb_metal_filter.py` (which is in tree but never run end-to-end on the full source). With DATA-HOLD, the wrapper was never authored — there was nothing to wrap.

**Sister workflows on the same axis** (already WRITTEN, for context):
- `scripts/launch_wf310e_wf219_production_gpu.sh` — 244 LOC, USER_OK + PAPER_OK gates (Wf-219 §4.2 chemprop v2 production backbone, MetalCytoToxDB 5-metal)
- `scripts/launch_wf310e_wf135_phase2_gpu.sh` — 244 LOC, USER_OK gate (Wf-135 Phase 2 LOMO 5-metal × 5-seed × 50-epoch)
- `scripts/launch_wf124_p3_g4.sh` + `g5.sh` + `g6.sh` — USER_OK-gated TODO-180/181/182 GPU wrappers
- `scripts/launch_wf338v2_wf176_m19_lomo_cpu.sh` — CPU scaffold wrapper (Wf-176 m19 LOMO Phase 2)

All 6 existing wrappers follow a uniform pattern: USER_OK=1 gate at line ~82-88, dry-run mode that emits `plan.json` + skips GPU launch, exit-code-64 refusal when gate fails. The R20 §3 BindingDB wrapper would mirror this pattern if/when authored.

## §5 USER_OK-gate verification — N/A (no wrapper)

Per the Wf-354-A2 precedent (USER_OK-gate verification), the verification path requires a wrapper file to exist. Since no wrapper exists, **there is nothing to verify**.

The verification would proceed as follows IF the wrapper existed:

```bash
# Step 1: Confirm wrapper exists
$ test -f scripts/launch_r20_p3_bindingdb_full.sh && echo "EXISTS" || echo "MISSING"
MISSING

# Step 2: Confirm USER_OK gate
$ grep -n "USER_OK" scripts/launch_r20_p3_bindingdb_full.sh
(grep: scripts/launch_r20_p3_bindingdb_full.sh: No such file or directory)

# Step 3: Confirm dry-run mode
$ grep -n "dry-run\|DRY-RUN\|--dry-run" scripts/launch_r20_p3_bindingdb_full.sh
(grep: scripts/launch_r20_p3_bindingdb_full.sh: No such file or directory)
```

All three steps fail with ENOENT. **Verification verdict: NOT_APPLICABLE**.

## §6 Honest framing (per AGENTS.md §5.2 + `party-twenty-spirit §3.5`)

| Item | Status |
|---|---|
| Wrapper existence | ❌ **HONEST-NEGATIVE NOT-WRITTEN** (per R20 plan §3 + Wf-148 §1.1 DATA-HOLD on BindingDB 1.07M) |
| USER_OK gate | N/A — no wrapper |
| Dry-run mode | N/A — no wrapper |
| Data availability | ❌ **HONEST-NEGATIVE DATA-HOLD** — only 5,257 subset on disk; full 1.07M NOT downloaded |
| Tests | 0 NEW |
| Wall | PROJECTED 10-15 min CPU-only (verification only); 0 actual GPU launch |
| Sister gap | `TODO/completed/354b3_r20_p5_tmqm_plat_joint_retrain_wrapper.md` covers R20 §4 tmQM+PlatinAI — also HONEST-NEGATIVE NOT-WRITTEN |

## §7 Cross-references

- `TODO/03_plans/R20_BIG_DATASET_FULL_RETRAIN_PLAN.md` §3 Phase 3 — BindingDB full 1.07M retrain spec
- `TODO/completed/148_wf146_lomo_retrain_prep_status.md` §1.1 — BindingDB 5K subset on disk; 1.07M NOT on disk
- `TODO/completed/144_q2_paper_ship_done.md` §2.4.2 — 1.07M row generic BindingDB via `pocket/data/tdc/` symlink (NOT the 8-metal metal-filtered subset)
- `scripts/wf175_bindingdb_metal_filter.py` — CPU SMARTS-based metal pre-filter; in tree but never run end-to-end on 1.07M source
- `TODO/completed/354b3_r20_p5_tmqm_plat_joint_retrain_wrapper.md` — sister closeout (R20 §4 tmQM+PlatinAI)
- `TODO/completed/354a2_wf310e_user_ok_verify.md` — Wf-354-A2 USER_OK-gate verification template
- `AGENTS.md` §5.1 (standing GPU grant) + §5.2 (honest framing mandate)

---

**END OF Wf-354-B2** — wrapper NOT-WRITTEN; DATA-HOLD on BindingDB 1.07M; per `party-twenty-spirit §3.5` zero aspirational claims; sister gap is `354b3` (R20 §4 tmQM+PlatinAI). 0 NEW tests; 0 GPU launches. Owner Hugo Lee + ultracode agent (Wf-354-B2 sub-flow).
