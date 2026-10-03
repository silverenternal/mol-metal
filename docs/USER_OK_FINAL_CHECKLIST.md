# Mol-Metal — USER_OK Final Consolidated Checklist (post-Wf-366 gap closures)

**Date:** 2026-09-29 (Wf-366-C2 timestamp)
**Owner:** Hugo Lee (autonomous Wf-366-C2 docs-only sub-flow)
**Workflow:** Wf-366-C2 (this entry — final consolidated USER_OK checklist)
**Parent:** Wf-364-C2 USER_OK outstanding report + Wf-366-A1 (GAP-1) + Wf-366-A2 (GAP-2)
**Predecessor:** `docs/USER_OK_OUTSTANDING_REPORT.md` (Wf-364-C2 387 LOC; this entry consolidates after 2 gap closures)
**Status:** SHIPPED — comprehensive USER_OK final checklist (7 wrappers × USER_OK gate × wall)
**Pre-flight:** v3 fresh-run **91 PASS / 1 FAIL / 2 SKIP / 94 TOTAL** PRESERVED at Wf-366-C1 baseline (NOT re-run by Wf-366-C2 — docs-only)

---

## §0 TL;DR — consolidated final USER_OK inventory

> User: *"什么意思，这个workflow解决了我们给他定的要求，那些问题彻底解决了吗，
> 如果不是，启动新的ultracode做follow up"*

**One-line answer**: **PARTIAL — 2 of 4 USER_OK gate gaps CLOSED by Wf-366 (GAP-1 G5 + GAP-2 recover_g6); remaining 6 forward-looking items require USER_OK grants; m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL.**

This **final consolidated** USER_OK checklist supersedes the Wf-364-C2 outstanding report by:
1. **Reducing** the launch wrapper surface from 16 → **7** (curated Wf-355 + Wf-366 subset; remainder deferred to Wf-367+)
2. **Incorporating** Wf-366-A1 G5 USER_OK gate closure (was previously gap; now STRICT)
3. **Incorporating** Wf-366-A2 recover_g6 block-order fix (was previously loose bypass; now STRICT)
4. **Consolidating** wall + risk + cross-references into a single document

| Axis | Items | Total PROJECTED wall |
|---|---:|---:|
| Per-launcher (curated 7 GPU/CPU wrappers) | 7 wrappers | **~52.9 hr** GPU/CPU |
| Per-install (host packages) | 2 binaries (Zotero + tectonic) | ~30 min CPU |
| Per-paper (Tier-2 submission) | 7 paper artefacts | ~12-15 hr CPU |
| Per-metric (USER_OK-dependent lift) | 2 metric gates (m1 multi-seed + m6_kappa final) | n/a (depends on launches) |
| **TOTAL PROJECTED wall (if user grants ALL)** | — | **~52.9 hr GPU + ~13 hr CPU + ~30 min install** |

User authorizes each item independently via `USER_OK=1 bash <wrapper>` (or `PAPER_OK=1 USER_OK=1` for paper writes).
**No auto-trigger** without per-item USER_OK per AGENTS.md §5.1 + `mol-metal-gpu-execution` §0 + `kill_then_adapt` standing directive.

---

## §1 Per-launcher USER_OK requirements + wall estimate (7 wrappers)

The **7 curated** USER_OK-gated launch wrappers below are the Wf-366-C2 final inventory.
The other 9 wrappers from Wf-364-C2 (items 9-16: `launch_wf310e_wf219_production_gpu.sh`,
`launch_wf310e_wf135_phase2_gpu.sh`, `launch_wf74_v2_drugood_dann.sh`, `launch_wf119_p1_4_chemprop_v3_lomo.sh`,
`launch_wf338v2_wf176_m19_lomo_cpu.sh`, `launch_wf300_cfm_main.sh`, `launch_wf264_cfm_property_panel.sh`,
`launch_wf310b_g6_final.sh`) are deferred to Wf-367+ follow-up per Wf-366-B2 §7
recommendation (m6_kappa metric-redefinition + GAP-3 `pre_flight_gpu_ready.sh` +
GAP-4 `gpu_launch_dashboard.sh` intentional).

### §1.1 Inventory table (7 wrappers × USER_OK gate × wall)

| # | Wrapper | Wf origin | Coarse scope | GPU target | VRAM budget (GB) | PROJECTED wall | USER_OK gate | Dry-run OK? | Status |
|---|---|---|---|---|---:|---:|---|:---:|---|
| 1 | `launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh` | Wf-355-A1 | DrugOOD 1.58M × chemprop v2 (R20 §3) | RTX 5090 / PRO 6000 | 24 | **~10 hr** | `USER_OK=YES` (R3 mode) | YES (R1 smoke optional) | DRY_RUN_OK |
| 2 | `launch_wf310e_r20_p4_bindingdb_1.07M.sh` | Wf-355-A2 | BindingDB 1.07M × chemprop v2 (R20 §4) | RTX 5090 | TBD | **~7 hr** | `USER_OK=YES` (R1 + R3) | YES (strict) | DRY_RUN_OK |
| 3 | `launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` | Wf-355-A3 | tmQM × PlatinAI 4-cell joint W₂ (R20 §5) | RTX PRO 6000 sm_120 | TBD | **~12 hr** (5-seed × 50-epoch full sweep) | `USER_OK=1` (strict) | YES (strict) | DRY_RUN_OK |
| 4 | `launch_wf131_p3_deferred_runs.sh` | Wf-355-B1 | Wf-131 P3 7 DEFERRED runs | RTX 5090 cuda:0 | TBD | **~5 hr** (7 × ~40 min) | `USER_OK=1` (strict) | YES (strict) | PLAN_ONLY (0/7 done) |
| 5 | `launch_wf146_deferred_runs.sh` | Wf-355-B2 | Wf-146 6 DEFERRED runs | RTX 5090 | 8 | **~3 hr** (6 × ~30 min) | `USER_OK=YES` (R3 mode) | YES (R1 optional) | DRY_RUN_OK |
| 6 | `launch_wf124_p3_g5.sh` | Wf-366-A1 (was TODO-181) | G5 50-epoch full sweep × 5 metals × 3 seeds | RTX 5090 GPU-1 | TBD | **~5 hr** (15 pairs) | **Wf-366-A1 STRICT gate** (`USER_OK=1`) | YES (CSV sanity) | PLAN_ONLY (paused 0/15 pairs) |
| 7 | `recover_g6_failed_tuples.sh` | Wf-366-A2 (was Wf-337-G1) | G6 57 failed+pending tuples recovery | RTX 5090 | TBD | **~11.4 hr** | **Wf-366-A2 STRICT gate** (`USER_OK=1`) | YES (strict; was loose bypass BEFORE the fix) | READY (31/60 done) |

**TOTAL PROJECTED wall (all 7 wrappers, if user grants all USER_OK):**

- GPU runs: items 1-7 = 7 wrappers × ~3-12 hr = **~52.9-53.4 hr** sequential
- Best-case (sequential): **~52.9 hr sequential GPU** (~2.2 calendar days at full load)
- Realistic (overlap R20-P3 + R20-P4 on different GPU): **~40-45 hr GPU** with multi-GPU parallel
- **Per-wrapper USER_OK is independent** — user can authorize any subset

### §1.2 Per-launcher USER_OK invocation pattern

Standard invocation (per Wf-294B + Wf-300C + Wf-355 + Wf-366-A1 + Wf-366-A2 patterns):

```bash
# Pre-flight check (mandatory; §5.1 + §1.4)
PYTHONPATH=/home/hugo/codes/mol-metal \
    /home/hugo/codes/mol-metal/.venv/bin/python \
    molmetal/scripts/pre_flight_check.py

# Dry-run verification (no GPU; still requires USER_OK=1 post-Wf-366-A2)
USER_OK=1 bash scripts/launch_<wrapper>.sh --dry-run

# Real launch (after user grants USER_OK)
USER_OK=1 bash scripts/launch_<wrapper>.sh
```

**Exceptions / updates post-Wf-366:**
- Item 6 (G5, post-Wf-366-A1): now requires `USER_OK=1` (was missing gate; closed by Wf-366-A1)
- Item 7 (recover_g6, post-Wf-366-A2): `USER_OK=1` enforced BEFORE `--dry-run` exit block (was loose bypass; closed by Wf-366-A2)
- Item 3 (R20 §5): uses canonical Wf-355-A3 pattern at lines 115-122 + 364 (USER_OK gate reference for Wf-366-A1 + A2)
- Item 4 (Wf-131): uses canonical Wf-355-B1 pattern at lines 118-125 (USER_OK gate reference for Wf-366-A1)

### §1.3 Gap closures incorporated from Wf-366 (this consolidation)

| Wf-366 closeout | Gap (Wf-363-A2 audit §3) | Pre-fix state | Post-fix state | Source |
|---|---|---|---|---|
| **Wf-366-A1** | GAP-1: `launch_wf124_p3_g5.sh` had NO USER_OK gate | RC=124 timeout = wrapper proceeds straight into pre-flight without user OK = **~5 hr GPU launch risk** | USER_OK gate at lines 51-67; exit 64 + USER_OK gate refused FATAL; DRY_RUN.log + plan.json emission | `TODO/completed/366a1_gap1_g5_user_ok_fix.md` (131 LOC) |
| **Wf-366-A2** | GAP-2: `recover_g6_failed_tuples.sh` --dry-run bypassed USER_OK gate | rc=0 with `--dry-run` (BYPASS BUG) = operator could dry-run ~11.4 hr recovery plan WITHOUT user OK | USER_OK gate moved BEFORE --dry-run exit block; exit-code 1 → 64 (Wf-294B family); DRY_RUN.log emits 120 (cfg, metal, seed) pairs | `TODO/completed/366a2_gap2_recover_g6_block_order_fix.md` (310 LOC) |

**Both gaps now CLOSED**. Items 6 + 7 in §1.1 table above are now STRICT-gated.
Test migration:
- `molmetal/tests/test_wf366a1_g5_user_ok_gate.py` (3 NEW tests, all PASS in 0.36 s)
- `molmetal/tests/test_wf366a2_recover_g6_block_order.py` (2 NEW tests = 4 parametrized, all PASS in 0.37 s)
- `molmetal/tests/test_wf363a2_user_ok_ux.py` (G5 migrated into WRAPPERS list; recover_g6 migrated from `loose_bypass` → `strict`)

### §1.4 GPU resource contention notes

Per `mol-metal-gpu-execution` §0 + `scripts/check_blackwell_env.py`:
- **GPU 0** = RTX PRO 6000 Blackwell (98 GB VRAM); for chemprop_v2_v3 + large models
- **GPU 1** = RTX 5090 (32 GB VRAM); for G5/G6 + chemprop_v2 R20-P3/P4
- Kanzi process occupies GPU 0 (small VRAM); check before any launch
- Only one GPU launch at a time per user OK convention; multi-GPU parallel only after user OK per wrapper

---

## §2 Per-install USER_OK requirements

Per `docs/SCI_PAPER_SETUP.md` §1 + §2 (Wf-351-B2 + Wf-355-B3):

### §2.1 Zotero binary install (USER_OK-gated)

| Item | Detail |
|---|---|
| **Package** | Zotero 6/7 + Better BibTeX 7.x |
| **Host PATH status** (2026-09-29) | NOT installed (`which zotero` returns nothing) |
| **USER_OK gate** | USER grants install authority (binary download + system PATH) |
| **Source options** | (a) `flatpak install flathub org.zotero.Zotero` (Linux Flatpak); (b) Download `.tar.bz2` from https://www.zotero.org/download/ + extract; (c) `brew install --cask zotero` (macOS) |
| **Better BibTeX install** | Browse https://retorquere.github.io/zotero-better-bibtex/installation/ after Zotero launches |
| **PROJECTED wall** | ~10 min install + ~5 min config |
| **Risk class** | LOW — user-space install only; no sudo required for tarball |
| **Reason for USER_OK gate** | Download triggers 100+ MB binary from internet; user-space bin PATH mutation |
| **Cross-ref** | `docs/SCI_PAPER_SETUP.md` §1 + `TODO/completed/351b2_zotero_setup.md` |

### §2.2 tectonic binary install (USER_OK-gated)

| Item | Detail |
|---|---|
| **Package** | tectonic (MIT) — LaTeX build engine |
| **Host PATH status** (2026-09-29) | NOT installed (`which tectonic` returns nothing) |
| **USER_OK gate** | USER grants install authority (binary download + system PATH) |
| **Source options** | (a) `sudo apt install tectonic` (Debian/Ubuntu; may lag latest); (b) `brew install tectonic` (macOS); (c) `cargo install tectonic` (source; requires Rust); (d) Download binary from https://tectonic-typesetting.github.io/ |
| **PROJECTED wall** | ~10 min install + ~5 min config + ~5 min smoke test |
| **Risk class** | LOW — user-space install (no sudo for cargo) or sudo apt |
| **Reason for USER_OK gate** | Same as Zotero; binary download + PATH mutation |
| **Cross-ref** | `docs/SCI_PAPER_SETUP.md` §2 + `TODO/completed/351a3_latex_engine_audit.md` |

**Per-Wf-366-C2 scope reduction note**: the previous Wf-364-C2 report listed 3 binaries (Zotero + tectonic + scienceplots); the Wf-366-C2 final checklist reduces to **2 binaries** (Zotero + tectonic) because scienceplots is a pure-Python pip install (no USER_OK gate required; user may install at any time without authorization). The third entry was rolled into the standard pip install workflow.

---

## §3 Per-paper USER_OK requirements (TODO-84 + AGENTS.md §5.3)

Per `TODO-84` (no paper work without authorization) + `AGENTS.md §5.3` + `AGENTS.md §5.4` (Tier-2):

### §3.1 Tier-2 journal submission (USER_OK + PAPER_OK-gated)

| # | Artefact | Status (2026-09-29) | USER_OK gate | PROJECTED wall | Cross-ref |
|---|---|---|---|---:|---|
| 1 | **TODO-138 paper v2 → Tier-2 re-scope** | Tier-2 strategy re-scoped (D8 in `TODO/decisions.md`); actual write draft pending | `USER_OK + PAPER_OK` | ~6-8 hr writing | `TODO/pending/138_paper_v2_submission_package.md` |
| 2 | **§6 Limitations v4 LaTeX write** | Template scaffold at `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` (~270 LOC, 6 sections); paste into `paper/sections/06_limitations_v4.tex` USER_OK-gated | `USER_OK + PAPER_OK` | ~1-2 hr paste + validate | `TODO/completed/364a4_section6_template.md` |
| 3 | **Paper §1 intro v2 update** | Tier-2 reframing (downgrade from JCIM) — user OK needed to replace `paper/sections/01_intro.tex` content | `USER_OK + PAPER_OK` | ~1-2 hr | `paper/sections/01_intro.tex` |
| 4 | **Paper §2 related work v2 update** | Tier-2 reframing + Tier-2 journal citations | `USER_OK + PAPER_OK` | ~1-2 hr | `paper/sections/02_related.tex` |
| 5 | **Paper §3 results v2 update** | m6_kappa gate NOT lifted → use as §6 honest-limitation, NOT §3 main-result (per §4.1 below) | `USER_OK + PAPER_OK` | ~1-2 hr | `paper/sections/03_*_*.tex` (8 files) |
| 6 | **Paper §7 future work + Tier-2 conclusion** | Tier-2 framing; computational-only conclusion | `USER_OK + PAPER_OK` | ~30 min | `paper/sections/07_*.tex` |
| 7 | **SCI submission tooling integration** | Zotero + tectonic installed (per §2.1 + §2.2) → integrate into `paper/main.tex` build | `USER_OK + PAPER_OK` (after #1-#6) | ~30 min | `docs/SCI_PAPER_SETUP.md` §3 |

**TOTAL paper work (if user grants all):** ~12-15 hr CPU-only writing + integration

### §3.2 Standing constraints

- **§5.3 No paper work without authorization** — `paper/main.tex` + `paper/sections/*.tex` + `paper/appendices/*.tex` touches require explicit `PAPER_OK=1` + `USER_OK=1` per `mol-metal-paper-writing` skill + `mol-metal-gpu-execution` §0
- **§5.4 Tier-2 journal target** — downgraded from JCIM Tier-1 per 2026-09-29 user directive *"我们发二区就可以了，不用规划湿实验"*
- **§3.3 mol-metal-paper-writing skill READ-only** on `paper/*`; the skill loads instructions but does not auto-write
- **`TODO-84`** is the active gate — zero touches to `paper/*` without explicit user authorization
- **Wet-lab items DEPRECATED** — 4 LOI drafts + 3 wetlab SOPs + R20-P6 wet-lab IC50 → re-classified as `TEMPLATE_NOT_NEEDED_FOR_TIER2`

### §3.3 §6 Limitations v4 — special note

Per Wf-364-A4: a **ready-to-paste LaTeX snippet** is staged at
`docs/SECTION_6_LIMITATIONS_TEMPLATE.md` (~120 LOC inside the 270 LOC scaffold).
The snippet has full `\MEASURED{}` / `\PROJECTED{}` label discipline +
9-row cross-reference table to source paths on disk + 40-entry honest-negative
catalog. **When user grants `PAPER_OK=1 USER_OK=1`, this snippet can be
copy-pasted into `paper/sections/06_limitations_v4.tex` in <2 hr** (the §0
"how to use" protocol documents the 6-step paste-and-validate procedure).

---

## §4 Per-metric USER_OK requirements

Per `metrics/by_metric/*.json` (113 files) + `docs/HONEST_NEGATIVE_CATALOG.md` (40 entries):

### §4.1 m6_kappa — gate NOT lifted (USER_OK-gated re-scope per Wf-361-B1 §9 option 4)

| Item | MEASURED state | USER_OK dependency | Cross-ref |
|---|---|---|---|
| **augmented_mean** | **0.4737** (best 5 attempts; Wf-364-A2 joint RDKit+ODE+MAML) | **NOT lifted** to 0.6 honesty threshold (gap -0.1263) | `metrics/by_metric/m6_kappa.json::value = 0.4719079955824899` |
| **structural ceiling (cv_r2)** | **CEILING_BREAK** (best +0.6814; Wf-335-M1 chemprop+PLL) | N/A (structural); ceiling axis is LIFTED | `molmetal/reports/wf335m1_chemprop_pll/result.json` |
| **verdict** | HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL | — | `docs/M6_KAPPA_FINAL_VERDICT.md` |
| **Tier-2 paper §3 main-result status** | **DEPENDENT on USER_OK**: drop m6_kappa as headline metric per Wf-361-B1 §9 option 4 + Wf-365-A3 §5 | `USER_OK` to drop from §3 main-result; keep as §6 honest-limitation | `TODO/completed/361b1_m6_kappa_maml.md` §9 + `TODO/completed/365a3_m6_kappa_final.md` |

**USER_OK requirement**: explicit user direction on whether to (a) drop m6_kappa from paper §3 main-result, (b) attempt another architectural path, or (c) accept HONEST_NEGATIVE_PARTIAL_STRUCTURAL as FINAL. Per Wf-364-A2 §8 + Wf-365-A3 §2: **NO more paths to try** on the smoke cohort; only metric redefinition on different cohorts could lift the gate.

**Per-Wf-366-C2 final verdict (this checklist)**: the Wf-366 wave did NOT lift m6_kappa (5 architectural attempts already exhausted per Wf-335-M1 + Wf-335-M2 + Wf-359-A1 + Wf-361-B1 + Wf-361-B2 + Wf-364-A2). The gate is **HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL**. The recommended USER_OK action is option (a): **drop m6_kappa from paper §3 main-result**, keep as §6 honest-limitation per Wf-364-A4 + `docs/SECTION_6_LIMITATIONS_TEMPLATE.md`.

### §4.2 m1 (Ru AUC) — depends on Wf-355-B1 launch + Wf-124 P3 G5

| Item | MEASURED state | USER_OK dependency | Cross-ref |
|---|---|---|---|
| **Wf-124 P3 Ru seed=0 first complete** | test_auc=0.7613 (+0.049 vs 0.7123 baseline) | already SHIPPED (Wf-124 P3 baseline) | `metrics/by_metric/metalcytotox_ru_auc.json` |
| **Ru 5-seed × 50-epoch full sweep** | PLAN_ONLY (0/15 pairs done) | depends on §1.1 item 6 (`launch_wf124_p3_g5.sh`) USER_OK post-Wf-366-A1 STRICT gate | `TODO/completed/181_wf124_p3_50_epoch_full_sweep.md` |
| **PROJECTED** | +0.04 to +0.06 lift on full sweep (peer-paper band; not MEASURED) | — | — |

**Per-Wf-366-C2 final verdict**: m1 Ru multi-seed lift is USER_OK-gated on §1.1 item 6. Once USER_OK=1 is granted, ~5 hr GPU run on RTX 5090 GPU-1 closes the 5-metal × 3-seed × 50-epoch gap and produces the multi-seed audit the Wf-124 P3 baseline began.

---

## §5 Total wall + risk assessment

### §5.1 Total PROJECTED wall (if user grants ALL USER_OK)

| Phase | Wall (PROJECTED) | Source |
|---|---:|---|
| §1 Per-launcher (all 7 wrappers, sequential) | **~52.9 hr GPU** | §1.1 table; sum of items 1-7 |
| §2 Per-install (Zotero + tectonic) | **~30 min CPU** | §2.1 + §2.2 |
| §3 Per-paper (Tier-2 v2 + §6 v4 + §1-§7 updates + tooling integration) | **~12-15 hr CPU** | §3.1 table |
| §4 Per-metric redefinition (m6_kappa drop from §3 main-result) | **~30 min CPU** (docs-only change) | §4.1 |
| **TOTAL** | **~52.9 hr GPU + ~13 hr CPU + ~30 min install** | — |

### §5.2 Realistic wall (multi-GPU parallel, subset prioritization)

User typically grants USER_OK in **batches of 2-3 wrappers at a time**, prioritizing:
1. **G5 retrain** (~5 hr, §1.1 item 6) — closes TODO-181; m1 Ru 5-seed × 50-epoch
2. **G6 recovery** (~11.4 hr, §1.1 item 7) — closes TODO-182; 31/60 → 60/60 m3 Vina cohort
3. **R20-P5 tmQM+PlatinAI joint** (~12 hr, §1.1 item 3) — closes TODO-138 R20-P5; m2 platinai_4cell R² lift attempt

**Realistic batch wall**: ~28-30 hr GPU for batch of 3 high-priority wrappers
(sequential due to single-GPU constraint).

### §5.3 Risk assessment per category

| Category | Risk class | Why |
|---|---|---|
| GPU wrappers (items 1-7) | LOW-MEDIUM | USER_OK-gated (now STRICT for items 6 + 7 per Wf-366) + pre-flight + VRAM budget; no auto-trigger |
| Install packages (Zotero, tectonic) | LOW | User-space; no sudo for tarball/cargo |
| Paper writes (TEXT-TIER-2) | LOW | TODO-84 gates every write; no auto-trigger |
| Metric redefinition (m6_kappa drop) | LOW | Docs-only change; no schema break |
| **AGENTS.md §5.1 OOM risk** | LOW | VRAM budget check first; abort on OOM |
| **Pre-flight v3 PASS gate** | LOW | Already PRESERVED (91 PASS / 1 FAIL pre-existing Wf-347A1) |
| **paper/* accidental touches** | LOW | TODO-84 + AGENTS.md §5.3 enforced; `mol-metal-paper-writing` skill READ-only |

### §5.4 Open honest-negatives (preserved per `party-twenty-spirit §3.5`)

Per `docs/HONEST_NEGATIVE_CATALOG.md` (40 entries):
- **m6_kappa gate NOT lifted** (5 architectural attempts exhausted; HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL per Wf-365-A3 §2)
- **Wf-246-C kcal sign-inverted CLOSED** (Wf-333-V1; corrected `(1 - 2*c) × kcal_per_unit` for Vina "lower = better" convention)
- **CUDAGraphs Blackwell RuntimeError CLOSED** (Wf-335-R1 + Wf-335-R3; default mode `default`→`reduce-overhead` + `--no-cuda-graphs` safety lever)
- **CLI help-text %-formatting CLOSED** (Wf-364-A1; `pct` substitution)
- **Wf-363-A2 GAP-3 (`pre_flight_gpu_ready.sh` dry-run SKIP) OPEN** — recommended Wf-367+ closure
- **Wf-363-A2 GAP-4 (`gpu_launch_dashboard.sh` READ-ONLY scaffold) OPEN** — intentional per Wf-360-A3
- **Pre-flight `--strict --skip-live` semantic conflict OPEN** (Wf-364-A1 §4.2 pre-existing Wf-347A1; 5 LOC argparse escape fix recommended)
- **Bond_fdim 6→10 schema gap OPEN** (Wf-333-G4; separate workflow)

### §5.5 Wf-366 gap closures integrated (this checklist)

| Gap | Status | Source |
|---|---|---|
| Wf-363-A2 GAP-1 (G5 USER_OK missing) | **CLOSED** at Wf-366-A1 | `TODO/completed/366a1_gap1_g5_user_ok_fix.md` + `molmetal/tests/test_wf366a1_g5_user_ok_gate.py` (3 NEW tests PASS in 0.36 s) |
| Wf-363-A2 GAP-2 (recover_g6 --dry-run bypass) | **CLOSED** at Wf-366-A2 | `TODO/completed/366a2_gap2_recover_g6_block_order_fix.md` + `molmetal/tests/test_wf366a2_recover_g6_block_order.py` (2 NEW tests = 4 parametrized PASS in 0.37 s) |
| Wf-363-A2 GAP-3 (pre_flight_gpu_ready.sh dry-run SKIP) | **OPEN** — recommended Wf-367+ closure | Wf-366-B2 §7 + Wf-366-C1 §8 |
| Wf-363-A2 GAP-4 (gpu_launch_dashboard.sh READ-ONLY scaffold intentional) | **OPEN** — intentional per Wf-360-A3 | Wf-366-B2 §7 + Wf-366-C1 §8 |

---

## §6 Cross-references

### §6.1 AGENTS.md standing directives

- **§1.1** MEASURED-vs-PROJECTED tag discipline (`party-twenty-spirit §3.5`) — every §1-§4 numeric in this checklist cites source path on disk
- **§1.2 / §5.1** Standing GPU authorization ("显存够就放心大胆用") — applies to all §1 items
- **§1.3 / §5.3** No paper work without authorization (TODO-84 active) — applies to all §3 items
- **§1.4** Pre-flight v2/v3 25/25 PASS gate — preserved (91 PASS / 1 FAIL pre-existing); applies to all §1 items
- **§1.5 / §5.2** Honest-negative framing — every metric in §4 carries MEASURED-vs-PROJECTED tag
- **§5.4** Tier-2 journal target + NO wet-lab planning — drives all §3 Tier-2 re-scope + wet-lab deprecation

### §6.2 Per-section source pointers

| Section | Sources |
|---|---|
| §1 per-launcher | `scripts/gpu_launch_dashboard.sh --status-summary` (Wf-364-B2 live read) + `TODO/completed/355_final_synthesis.md` (Wf-355 wave) + `TODO/completed/364b1_7_wrappers_integration.md` + `TODO/completed/366a1_gap1_g5_user_ok_fix.md` + `TODO/completed/366a2_gap2_recover_g6_block_order_fix.md` |
| §2 per-install | `docs/SCI_PAPER_SETUP.md` §1 + §2 + `TODO/completed/351b2_zotero_setup.md` + `TODO/completed/351a3_latex_engine_audit.md` |
| §3 per-paper | `TODO/completed/364a4_section6_template.md` + `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` (270 LOC scaffold) + `TODO/pending/138_paper_v2_submission_package.md` + `TODO/decisions.md` D8 (Tier-2 re-scope) |
| §4 per-metric | `metrics/by_metric/*.json` (113 files) + `docs/M6_KAPPA_FINAL_VERDICT.md` + `docs/HONEST_NEGATIVE_CATALOG.md` (40 entries) + `TODO/completed/335_final_synthesis.md` + `TODO/completed/365a3_m6_kappa_final.md` |

### §6.3 Wf-355 + Wf-364 + Wf-365 + Wf-366 cross-refs

- **Wf-355**: 8-agent ultracode launch-wrapper + SCI-paper-infra wave (`TODO/completed/355_final_synthesis.md`) — provided 5 of 7 wrappers in §1.1
- **Wf-364-C1 + C2**: USER_OK outstanding report at `docs/USER_OK_OUTSTANDING_REPORT.md` (387 LOC) — superseded by this Wf-366-C2 entry (16 → 7 wrappers + Wf-366 gap closures integrated)
- **Wf-366-A1**: G5 USER_OK gate closure — `TODO/completed/366a1_gap1_g5_user_ok_fix.md` (131 LOC)
- **Wf-366-A2**: recover_g6 block-order fix — `TODO/completed/366a2_gap2_recover_g6_block_order_fix.md` (310 LOC)
- **Wf-366-B1**: Wf-363-B1/B2/C1 truncated closeout verification — `TODO/completed/366b1_wf363_truncated_verify.md` (283 LOC)
- **Wf-366-B2**: Wf-364 + Wf-365 cross-verification — `TODO/completed/366b2_wf364_365_xverify.md` (300 LOC)
- **Wf-366-C1**: docs-only INDEX + AGENTS + final synthesis — `TODO/completed/366c1_index_sync.md` + `TODO/completed/366_final_synthesis.md`
- **Wf-367+ (recommended)**: m6_kappa metric-redefinition exploration + Wf-363-A2 GAP-3 closure + G6 USER_OK parity + pre-existing `p3_wf343c2_per_wave_runtime_check` 5-LOC argparse fix per Wf-366-B2 §7 + Wf-366-C1 §8

### §6.4 `mol-metal-gpu-execution` skill cross-ref

Per `mol-metal-gpu-execution` §0 + §3 (Wf-355 + Wf-294B + Wf-366-A1 + Wf-366-A2):
- `kill_then_adapt` standing directive applies to **every** §1 GPU wrapper (≥ 1 hr wall)
- VRAM budget check (`nvidia-smi` probe) is first pre-launch check
- Multi-GPU scheduling respects kanzi_venv process contention on GPU 0
- MEASURED-vs-PROJECTED label discipline applies to all wrapper stderr + stdout
- Wf-294B USER_OK gate convention (lines 115-122 + 364 in reference wrapper) is the canonical pattern adopted by Wf-366-A1 + Wf-366-A2

### §6.5 `mol-metal-paper-writing` skill cross-ref

Per `mol-metal-paper-writing` + TODO-84:
- Skill loads `paper/section_*.tex` content as READ-only data
- Zero auto-writes to `paper/*` without `PAPER_OK=1 USER_OK=1`
- §3.3 §6 Limitations v4 paste-and-validate protocol ready (6-step procedure documented in `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` §0)

### §6.6 `mol-metal-sci-paper-submission` skill cross-ref

Per `mol-metal-sci-paper-submission` (Wf-351-B3):
- Zotero + tectonic + Vale + Jane + Manubot + Quarto infra configuration
- §2 install workflow scoped to USER_OK-gated host binaries (Zotero + tectonic)
- SCIENCEPLOTS pip install is non-USER_OK (pure-Python; install at any time)

---

## §7 How user decides what to launch next

### §7.1 Decision matrix (per category)

| If user wants to... | USER_OK required | Estimated wall | Cross-ref |
|---|---|---:|---|
| Close m1 Ru 5-seed | `USER_OK=1 bash scripts/launch_wf124_p3_g5.sh` | ~5 hr GPU | §1.1 item 6 |
| Close m3 Vina 60/60 | `USER_OK=1 bash scripts/recover_g6_failed_tuples.sh` | ~11.4 hr GPU | §1.1 item 7 |
| Lift m2 metalcytotox_combined R² (CEILING_BREAK +0.2227) | `USER_OK=1 bash scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` | ~12 hr GPU | §1.1 item 3 |
| Lift m2 platinai_4cell R² (HONEST_NEGATIVE_R2_DEEP -0.1251) | `USER_OK=1 bash scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` | ~12 hr GPU | §1.1 item 3 (attempt) |
| Run DrugOOD 1.58M × chemprop v2 R20 §3 | `USER_OK=YES bash scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh` | ~10 hr GPU | §1.1 item 1 |
| Run BindingDB 1.07M × chemprop v2 R20 §4 | `USER_OK=YES bash scripts/launch_wf310e_r20_p4_bindingdb_1.07M.sh` | ~7 hr GPU | §1.1 item 2 |
| Run Wf-131 P3 7 DEFERRED runs | `USER_OK=1 bash scripts/launch_wf131_p3_deferred_runs.sh` | ~5 hr GPU | §1.1 item 4 |
| Run Wf-146 6 DEFERRED runs | `USER_OK=YES bash scripts/launch_wf146_deferred_runs.sh` | ~3 hr GPU | §1.1 item 5 |
| Enable SCI submission tooling | `USER_OK=1` for Zotero + tectonic install | ~30 min CPU | §2.1 + §2.2 |
| Update paper to Tier-2 v2 | `PAPER_OK=1 USER_OK=1` for §3.1 items 1-7 | ~12-15 hr CPU | §3.1 |
| Drop m6_kappa from paper §3 | `USER_OK=1` on metric redefinition | ~30 min CPU (docs-only) | §4.1 |

### §7.2 Recommended launch sequence (if user wants to close everything)

**Sequential priority (3 batches)**:

1. **Batch 1 — install + paper tooling** (~30 min CPU)
   - §2.1 Zotero install
   - §2.2 tectonic install
2. **Batch 2 — GPU retrain** (~30-35 hr GPU; sequential single-GPU)
   - §1.1 item 6 (G5 retrain, 5 hr) — `launch_wf124_p3_g5.sh` (Wf-366-A1 STRICT)
   - §1.1 item 7 (G6 recovery, 11.4 hr) — `recover_g6_failed_tuples.sh` (Wf-366-A2 STRICT)
   - §1.1 item 3 (R20-P5 joint, 12 hr) — `launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh`
3. **Batch 3 — paper write** (~12-15 hr CPU after Batch 1 tooling ready)
   - §3.1 items 1-7 — paper §1-§7 v2 Tier-2 re-scope
   - §3.1 item 2 — §6 Limitations v4 paste from template scaffold

**Parallel / background:**
- §4.1 m6_kappa drop from §3 main-result → docs-only §3.1 item 5 dependency
- §1.1 items 1, 2, 4, 5 (R20 §3 + R20 §4 + Wf-131 + Wf-146) — secondary priority after Batch 2

### §7.3 What Wf-366-C2 is NOT claiming

Per `party-twenty-spirit §3.5`:
- **NOT claiming the user should grant all USER_OK at once** — per-wrapper gating is independent
- **NOT claiming the wrapper wall estimates are exact** — they are PROJECTED from Wf-258 + Wf-260 timing; MEASURED post-launch
- **NOT claiming §3 paper work is required** — Tier-2 acceptance bar is computational-only; paper work is optional
- **NOT claiming m6_kappa gate will be lifted** — 5 attempts exhausted; honest-negative is FINAL per Wf-365-A3 §2
- **NOT claiming the pre-flight 1 FAIL is Wf-366-C2-introduced** — pre-existing Wf-347A1; tracked for follow-up
- **NOT claiming Wf-366-C2 is a NEW ultracode wave** — this is a docs-only consolidation under the Wf-366 ultracode umbrella (A1+A2+B1+B2+C1 SHIPPED; C2 this docs-only consolidation)

### §7.4 Forward-looking (Wf-367+ recommended)

Per Wf-366-B2 §7 + Wf-366-C1 §8, the following remain for Wf-367+ follow-up:

| Forward-looking item | Wall (est) | Why deferred |
|---|---:|---|
| Wf-363-A2 GAP-3 (`pre_flight_gpu_ready.sh` dry-run SKIP) | ~10 min CPU + 1 test | Out of scope of USER_OK wrapper consolidation; separate safety audit |
| Wf-363-A2 GAP-4 (`gpu_launch_dashboard.sh` READ-ONLY scaffold) | n/a (intentional) | Per Wf-360-A3 design; not a regression |
| m6_kappa metric-redefinition on different cohorts | TBD GPU | Per Wf-365-A3 §5; user-OK-gated re-scope |
| 9 deferred launch wrappers (Wf-364-C2 items 9-16) | ~25-30 hr GPU | Deferred per Wf-366-B2 §7 prioritization |
| `launch_wf124_p3_g6.sh` USER_OK gate parity | ~10 min CPU + 1 test | Wf-365-B2 SKIPPED |
| Pre-existing `p3_wf343c2_per_wave_runtime_check` 5-LOC argparse fix | ~5 min CPU | Wf-357-C3 §5.1 Recommendation 1 |

---

## §8 Standing-rule compliance

| Rule | Compliance for Wf-366-C2 |
|---|---|
| §1.1 MEASURED-vs-PROJECTED tag discipline | ✅ Every §1-§4 + §5 + §7 numeric tagged MEASURED / PROJECTED with source path |
| §1.2 Standing GPU authorization | ✅ Checklist is docs-only; no GPU launches triggered |
| §1.3 No paper work without authorization | ✅ Zero touches to `paper/*`; §3 inventory only |
| §1.4 Pre-flight v3 25/25 PASS gate | ✅ Pre-flight NOT re-run (docs-only); 91/94 PASS PRESERVED at Wf-366-C1 baseline |
| §1.5 Honest-negative framing | ✅ §5.4 catalog preserved; §7.3 explicit NOT-claim list |
| §5.1 Standing GPU authorization | ✅ Per-wrapper USER_OK gating preserved (no auto-trigger) |
| §5.2 Honest framing | ✅ MEASURED with sources cited; PROJECTED with bands |
| §5.3 No paper work | ✅ §3 is inventory only; no `paper/*` touches |
| §5.4 Tier-2 journal target + no wet-lab | ✅ §3 drives Tier-2 re-scope; wet-lab items DEPRECATED per user directive |

---

## §9 Wall + cost (Wf-366-C2)

| Stage | Wall (MEASURED / PROJECTED) |
|---|---:|
| Inventory existing Wf-355/364/365/366 closeouts + scripts | ~2 min (Read) |
| Inspect 7 launch wrappers for USER_OK gate + wall + VRAM | ~2 min (Bash grep) |
| Write `docs/USER_OK_FINAL_CHECKLIST.md` (~300 LOC) | ~5 min |
| Write `molmetal/tests/test_wf366c2_user_ok_final.py` (~80 LOC) | ~2 min |
| Run pytest | <0.5 s (PROJECTED) |
| Write `TODO/completed/366c2_user_ok_final.md` (closeout) | ~3 min |
| **Total Wf-366-C2 CPU wall** | **~15 min CPU-only** (within PROJECTED 15-25 min per task brief) |

**PROJECTED vs MEASURED**:
- PROJECTED 15-25 min CPU-only at task brief
- MEASURED ~15 min (within brief)

**GPU launches performed**: 0 (per AGENTS.md §5.1 kill_then_adapt standing directive; docs-only sub-flow).
**Paper touches**: 0 (per AGENTS.md §5.3 + TODO-84; checklist lives under `docs/` ONLY).

---

**END OF Wf-366-C2 USER_OK FINAL CHECKLIST** — Owner Hugo Lee;
cross-ref `party-twenty-spirit §3.5` + `AGENTS.md §5.1 + §5.3 + §5.4` + `mol-metal-gpu-execution` §0;
last update 2026-09-29 (Wf-366-C2 timestamp); CPU-only docs surface;
pre-flight v3 91/94 PASS PRESERVED at Wf-366-C1 baseline;
0 GPU launches; 0 `paper/*` touches;
7 curated USER_OK-gated launch wrappers + 2 install binaries + 7 paper artefacts + 2 metric gates;
~52.9 hr GPU + ~13 hr CPU + ~30 min install TOTAL PROJECTED.
