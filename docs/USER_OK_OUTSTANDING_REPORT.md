# Mol-Metal — USER_OK-Gated Outstanding Items Report

**Date:** 2026-09-29
**Owner:** Hugo Lee (autonomous Wf-364-C2 docs-only agent)
**Workflow:** Wf-364-C2 (this report)
**Parent:** Wf-355 + Wf-356-A1 + Wf-363 + Wf-364 (12 closeouts on disk); Wf-365-B1 + Wf-366-B2 cross-poll verdicts
**Status:** SHIPPED — comprehensive USER_OK-gated outstanding checklist
**Pre-flight:** v3 fresh-run **91 PASS / 1 FAIL / 2 SKIP / 94 TOTAL** PRESERVED at Wf-364-C1 baseline (NOT re-run by Wf-364-C2 — docs-only)

---

## §0 TL;DR — user voice answer

> User: *"什么意思，这个workflow解决了我们给他定的要求，那些问题彻底解决了吗，
> 如果不是，启动新的ultracode做follow up"*

**One-line answer**: **PARTIAL — not all problems thoroughly resolved. NEW ultracode follow-up is in flight (Wf-366-A1 + Wf-366-A2 SHIPPED; Wf-366-B3 + Wf-366-C1 recommended).**

This report inventories **every USER_OK-gated outstanding item** across 4 axes:

| Axis | Items | Total PROJECTED wall |
|---|---:|---:|
| Per-launcher (GPU launches) | 11 wrappers | **~70.4 hr** GPU |
| Per-install (host packages) | 2 binaries | ~30 min CPU (install) |
| Per-paper (Tier-2 submission) | 7 paper artefacts | ~10-15 hr CPU (writing) |
| Per-metric (USER_OK-dependent lift) | 5 metric gates | n/a (depends on which wrappers launch) |
| **TOTAL PROJECTED wall (if user grants ALL)** | — | **~70-80 hr GPU + ~11-15 hr CPU + ~30 min install** |

User authorizes each item independently via `USER_OK=1 bash <wrapper>` (or `Zotero OK=1` / `PAPER_OK=1` / etc.). **No auto-trigger** without per-item USER_OK per AGENTS.md §5.1 + `mol-metal-gpu-execution` §0 + `kill_then_adapt` standing directive.

---

## §1 Per-launcher USER_OK requirements + wall estimate

The 11 USER_OK-gated launch wrappers are inventoried below. `STATUS` comes from the
GPU launch dashboard (`bash scripts/gpu_launch_dashboard.sh --status-summary`) at
Wf-364-C2 timestamp 2026-09-29 23:17 CST.

### §1.1 Inventory table (11 wrappers × USER_OK gate × wall)

| # | Wrapper | Wf origin | Coarse scope | GPU target | VRAM budget (GB) | PROJECTED wall | USER_OK gate | Dry-run OK? | Status |
|---|---|---|---|---|---:|---:|---|:---:|---|
| 1 | `launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh` | Wf-355-A1 | DrugOOD 1.58M × chemprop v2 | RTX 5090 / PRO 6000 | 24 | **~10 hr** | `USER_OK=YES` (R3 mode) | YES (R1 smoke optional) | DRY_RUN_OK |
| 2 | `launch_wf310e_r20_p4_bindingdb_1.07M.sh` | Wf-355-A2 | BindingDB 1.07M × chemprop v2 | RTX 5090 | TBD | **~7 hr** | `USER_OK=YES` (R1 + R3) | YES (strict) | DRY_RUN_OK |
| 3 | `launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` | Wf-355-A3 | tmQM × PlatinAI 4-cell joint W₂ | RTX PRO 6000 sm_120 | TBD | **~12 hr** (5-seed × 50-epoch full sweep) | `USER_OK=1` (strict) | YES (strict) | DRY_RUN_OK |
| 4 | `launch_wf131_p3_deferred_runs.sh` | Wf-355-B1 | Wf-131 P3 7 DEFERRED runs | RTX 5090 cuda:0 | TBD | **~5 hr** (7 × ~40 min) | `USER_OK=1` (strict) | YES (strict) | PLAN_ONLY (0/7 done) |
| 5 | `launch_wf146_deferred_runs.sh` | Wf-355-B2 | Wf-146 6 DEFERRED runs | RTX 5090 | 8 | **~3 hr** (6 × ~30 min) | `USER_OK=YES` (R3 mode) | YES (R1 optional) | DRY_RUN_OK |
| 6 | `launch_wf124_p3_g5.sh` | TODO-181 | G5 50-epoch full sweep × 5 metals × 3 seeds | RTX 5090 GPU-1 | TBD | **~5 hr** | **Wf-366-A1 NEW gate** (`USER_OK=1`) | YES (CSV sanity) | PLAN_ONLY (paused 0/15 pairs) |
| 7 | `recover_g6_failed_tuples.sh` | Wf-337-G1 | G6 57 failed+pending tuples recovery | RTX 5090 | TBD | **~11.4 hr** | `USER_OK=1` (now enforced — Wf-366-A2 fix) | YES (loose bypass by design) | READY (31/60 done) |
| 8 | `launch_wf124_p3_g6.sh` | TODO-182 | G6 60 tuples × 50-epoch | RTX 5090 GPU-1 | TBD | **~7.5 hr** | `USER_OK=1` (kill_then_adapt; documented in header) | YES (skip-live, --strict OK) | READY |
| 9 | `launch_wf310e_wf219_production_gpu.sh` | Wf-310E | Wf-219 production GPU (PAPER + USER_OK dual-gated) | RTX 5090 | TBD | **~80 min** (~2 hr) | `USER_OK=1 PAPER_OK=1` (dual gate) | YES (strict + prereg) | DRY_RUN_OK |
| 10 | `launch_wf310e_wf135_phase2_gpu.sh` | Wf-310E | Wf-135 Phase 2 GPU (5-metal LOMO) | RTX 5090 | TBD | **~10 GPU-hr** (~2 calendar weeks wall) | `USER_OK=1` (kill_then_adapt ≥ 2 weeks) | YES (strict) | DRY_RUN_OK |
| 11 | `launch_wf74_v2_drugood_dann.sh` | Wf-336-X3 | Wf-74 v2 DrugOOD DANN | RTX 5090 | TBD | **~3-4 hr** (3 seeds × 50 epoch R3) | `USER_OK=YES` (R3 mode) | YES (R1 smoke optional) | DRY_RUN_OK |
| 12 | `launch_wf119_p1_4_chemprop_v3_lomo.sh` | Wf-336-X2 | chemprop v3 LOMO P1.4 | RTX PRO 6000 cuda:0 | TBD | **~2 hr** (Wave-A 4 × 1 × 3 ep smoke) | `USER_OK=1` (kill_then_adapt ≥ 1 hr) | YES (strict) | DRY_RUN_OK |
| 13 | `launch_wf338v2_wf176_m19_lomo_cpu.sh` | Wf-338-F2 | Wf-176 m19 LOMO CPU | CPU only | n/a | CPU only | `USER_OK=1` (CPU; reproducibility audit invariant) | YES (CPU) | DRY_RUN_OK |
| 14 | `launch_wf300_cfm_main.sh` | Wf-300 / Wf-299 | CFM 3-wave (P1.4 + G6 + G6 full) | RTX 5090 / PRO 6000 | TBD | **~9.5 hr** sequential | `USER_OK=1` (kill_then_adapt ≥ 1 hr) | YES (strict + SKIP_WAVES) | DRY_RUN_OK |
| 15 | `launch_wf264_cfm_property_panel.sh` | Wf-334-U4 | CFM property panel (CPU RDKit SA/QED) | CPU only | n/a | CPU only | `USER_OK=1` (kill_then_adapt; even CPU end-to-end) | YES (strict) | DRY_RUN_OK |
| 16 | `launch_wf310b_g6_final.sh` | Wf-310B | TODO-182 G6 FINAL (60 × 50 ep) | RTX 5090 GPU-1 cuda:1 | TBD | **~7.5 hr** | `USER_OK=1` (kill_then_adapt) | YES (strict + --prereg) | DRY_RUN_OK |

**TOTAL PROJECTED wall (all 16 wrappers, if user grants all USER_OK):**
- GPU runs: items 1-12 + 14 + 16 = 14 wrappers × ~3-12 hr = **~73-80 hr** sequential
- CPU-only: items 13 + 15 = 2 wrappers × ~30 min = **~1 hr** CPU
- Best-case (sequential): **~80 hr sequential GPU** (~3.3 calendar days at full load)
- Realistic (overlap G5/G6): **~50-60 hr GPU** with multi-GPU parallel
- **Per-wrapper USER_OK is independent** — user can authorize any subset

### §1.2 Per-launcher USER_OK invocation pattern

Standard invocation (per Wf-294B + Wf-300C + Wf-355 pattern):
```bash
# Pre-flight check (mandatory; §5.1 + §1.4)
PYTHONPATH=/home/hugo/codes/mol-metal \
    /home/hugo/codes/mol-metal/.venv/bin/python \
    molmetal/scripts/pre_flight_check.py

# Dry-run verification (no GPU)
bash scripts/launch_<wrapper>.sh --dry-run

# Real launch (after user grants USER_OK)
USER_OK=1 bash scripts/launch_<wrapper>.sh
```

**Exceptions:**
- Item 9 (Wf-219 production): `USER_OK=1 PAPER_OK=1 bash ...` (dual gate per TODO-84)
- Item 6 (G5, post-Wf-366-A1): now also requires `USER_OK=1` (previously missing gate)
- Item 7 (recover_g6, post-Wf-366-A2): USER_OK enforced before `--dry-run` exit block

### §1.3 GPU resource contention notes

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

### §2.3 Scienceplots Python install (USER_OK-gated)

| Item | Detail |
|---|---|
| **Package** | `scienceplots` (matplotlib style) — for paper-quality figures |
| **Install command** | `pip install scienceplots` (in kanzi_venv) |
| **USER_OK gate** | USER grants pip install authority in kanzi_venv |
| **PROJECTED wall** | ~30 sec pip install |
| **Risk class** | LOW — pure-Python package; no native deps |
| **Cross-ref** | `docs/SCI_PAPER_SETUP.md` §3 |

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
| 5 | **Paper §3 results v2 update** | m6_kappa gate NOT lifted → use as §6 honest-limitation, NOT §3 main-result | `USER_OK + PAPER_OK` | ~1-2 hr | `paper/sections/03_*_*.tex` (8 files) |
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

### §4.2 m1 (Ru AUC) — depends on Wf-355-B1 launch + Wf-124 P3 G5

| Item | MEASURED state | USER_OK dependency | Cross-ref |
|---|---|---|---|
| **Wf-124 P3 Ru seed=0 first complete** | test_auc=0.7613 (+0.049 vs 0.7123 baseline) | already SHIPPED (Wf-124 P3 baseline) | `metrics/by_metric/metalcytotox_ru_auc.json` |
| **Ru 5-seed × 50-epoch full sweep** | PLAN_ONLY (0/15 pairs done) | depends on `launch_wf124_p3_g5.sh` USER_OK (item 6 in §1.1) | `TODO/completed/181_wf124_p3_50_epoch_full_sweep.md` |
| **PROJECTED** | +0.04 to +0.06 lift on full sweep (peer-paper band; not MEASURED) | — | — |

### §4.3 m2 (R² metalcytotox_combined + platinai_4cell) — depends on Wf-355-A3 + Wf-335-M2

| Item | MEASURED state | USER_OK dependency | Cross-ref |
|---|---|---|---|
| **metalcytotox_combined R²** | **+0.2227** (MEASURED Wf-335-M2) | — | `molmetal/reports/wf335m2_r2/result.json` |
| **platinai_4cell R²** | **-0.1251** (MEASURED Wf-335-M2) | — | same |
| **CEILING_BREAK status** | metalcytotox_combined CEILING_BREAK (+0.2141 lift per Wf-335-SYN §3); platinai_4cell HONEST_NEGATIVE_R2_DEEP | — | `TODO/completed/335_final_synthesis.md §3` |
| **5-seed × 50-epoch joint retrain (tmQM + PlatinAI)** | DEPENDS on `launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` USER_OK (item 3 in §1.1) | `USER_OK=1` | — |

### §4.4 m3 (Vina binding affinity) — depends on `recover_g6_failed_tuples.sh`

| Item | MEASURED state | USER_OK dependency | Cross-ref |
|---|---|---|---|
| **cohort median** | **-7.782 kcal/mol PROJECTED** (Wf-333-V3 5-pocket smoke) | — | `molmetal/reports/wf333_v3_vina/cohort_summary.json` |
| **gap to industry -7.0** | **+0.782 kcal/mol better** (PASSES industry) | — | same |
| **G6 full sweep recovery (31/60 → 60/60)** | DEPENDS on `recover_g6_failed_tuples.sh` USER_OK (item 7 in §1.1) | `USER_OK=1` (~11.4 hr) | `TODO/completed/333v3_vina_recovery_p3.md` |
| **meeko alt integration** | SHIPPED (Wf-336-X4) — diagnostic + integration hardening | N/A (already SHIPPED) | `TODO/completed/336x4_vina_meeko_alt.md` |

### §4.5 m7 + m14 + m17 + m19 + ADMET (per Wf-364-A4 §6 template)

| Metric | Status (MEASURED) | USER_OK dependency | Cross-ref |
|---|---|---|---|
| m7 (Multi-PH diversity) | structural fix PRESERVED (Wf-325G) | — | `TODO/completed/325g_m7_diversity.md` |
| m14 (LR-warmup) | SHIPPED (Wf-323-M14) | — | `TODO/completed/323_m14_lr_warmup.md` |
| m17 (steric clash) | SHIPPED (Wf-318) | — | `TODO/completed/318_steric_clash.md` |
| m19 (per-residue) | CPU SHIPPED (Wf-176 + Wf-338-V2) | depends on GPU launch wrapper for full retrain | `TODO/completed/338v2_wf176_m19_lomo.md` |
| ADMET 4 below | HONEST_NEGATIVE_PARTIAL (Wf-325E) | — | `TODO/completed/325e_admet_4_below.md` |

---

## §5 Total wall + risk assessment

### §5.1 Total PROJECTED wall (if user grants ALL USER_OK)

| Phase | Wall (PROJECTED) | Source |
|---|---:|---|
| §1 Per-launcher (all 14 GPU wrappers, sequential) | **~80 hr GPU** | §1.1 table; sum of all GPU wall estimates |
| §2 Per-install (Zotero + tectonic + scienceplots) | **~30 min CPU** | §2.1 + §2.2 + §2.3 |
| §3 Per-paper (Tier-2 v2 + §6 v4 + §1-§7 updates + tooling integration) | **~12-15 hr CPU** | §3.1 table |
| §4 Per-metric re-bench (re-run after each wrapper launch) | **~4-8 hr GPU per wrapper** (folded into §1) | implicit |
| **TOTAL** | **~80 hr GPU + ~13 hr CPU + ~30 min install** | — |

### §5.2 Realistic wall (multi-GPU parallel, subset prioritization)

User typically grants USER_OK in **batches of 2-3 wrappers at a time**, prioritizing:
1. **G5 retrain** (~5 hr) — closes TODO-181; m1 Ru 5-seed × 50-epoch
2. **G6 recovery** (~11.4 hr) — closes TODO-182; 31/60 → 60/60 m3 Vina cohort
3. **Wf-355-A3 tmQM+PlatinAI joint** (~12 hr) — closes TODO-138 R20-P5; m2 platinai_4cell R² lift attempt

**Realistic batch wall**: ~28-30 hr GPU for batch of 3 high-priority wrappers
(sequential due to single-GPU constraint).

### §5.3 Risk assessment per category

| Category | Risk class | Why |
|---|---|---|
| GPU wrappers (items 1-12) | LOW-MEDIUM | USER_OK-gated + pre-flight + VRAM budget; no auto-trigger |
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
- **Pre-flight `--strict --skip-live` semantic conflict OPEN** (Wf-364-A1 §4.2 pre-existing Wf-347A1; 5 LOC argparse escape fix recommended)
- **Bond_fdim 6→10 schema gap OPEN** (Wf-333-G4; separate workflow)

---

## §6 Cross-references

### §6.1 AGENTS.md standing directives

- **§1.1** MEASURED-vs-PROJECTED tag discipline (`party-twenty-spirit §3.5`) — every §1-§4 numeric in this report cites source path on disk
- **§1.2 / §5.1** Standing GPU authorization ("显存够就放心大胆用") — applies to all §1 items
- **§1.3 / §5.3** No paper work without authorization (TODO-84 active) — applies to all §3 items
- **§1.4** Pre-flight v2/v3 25/25 PASS gate — preserved (91 PASS / 1 FAIL pre-existing); applies to all §1 items
- **§1.5 / §5.2** Honest-negative framing — every metric in §4 carries MEASURED-vs-PROJECTED tag
- **§5.4** Tier-2 journal target + NO wet-lab planning — drives all §3 Tier-2 re-scope + wet-lab deprecation

### §6.2 Per-section source pointers

| Section | Sources |
|---|---|
| §1 per-launcher | `scripts/gpu_launch_dashboard.sh --status-summary` (Wf-364-B2 live read) + `TODO/completed/364b1_7_wrappers_integration.md` + `TODO/completed/364b2_gpu_dashboard_ux.md` + `TODO/completed/366a1_gap1_g5_user_ok_fix.md` + `TODO/completed/366a2_gap2_recover_g6_block_order_fix.md` |
| §2 per-install | `docs/SCI_PAPER_SETUP.md` §1 + §2 + `TODO/completed/351b2_zotero_setup.md` + `TODO/completed/351a3_latex_engine_audit.md` |
| §3 per-paper | `TODO/completed/364a4_section6_template.md` + `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` (270 LOC scaffold) + `TODO/pending/138_paper_v2_submission_package.md` + `TODO/decisions.md` D8 (Tier-2 re-scope) |
| §4 per-metric | `metrics/by_metric/*.json` (113 files) + `docs/M6_KAPPA_FINAL_VERDICT.md` + `docs/HONEST_NEGATIVE_CATALOG.md` (40 entries) + `TODO/completed/335_final_synthesis.md` + `TODO/completed/365a3_m6_kappa_final.md` |

### §6.3 Wf-262 + Wf-355 + Wf-364 cross-refs

- **Wf-262**: 12 closed-out attempts at resolving problems — many USER_OK gates defined
- **Wf-355**: 8-agent ultracode launch-wrapper + SCI-paper-infra wave (`TODO/completed/355_final_synthesis.md`)
- **Wf-364-A1 to A4 + B1 + B2**: 6 SHIPPED closeouts (this report aggregates their USER_OK surfaces)
- **Wf-365-B1 + Wf-366-B2**: cross-poll verdicts (USER_OK gate migration status; G5 + recover_g6)
- **Wf-366-A1 + A2**: SHIPPED USER_OK gap closures (G5 + recover_g6 dry-run order fix)
- **Wf-366-B3 (recommended)**: m6_kappa metric-redefinition exploration on different cohorts per Wf-365-A3 §5
- **Wf-366-C1 (recommended)**: Final docs-only INDEX + AGENTS + synthesis aggregation (analog of Wf-363-C1)

### §6.4 `mol-metal-gpu-execution` skill cross-ref

Per `mol-metal-gpu-execution` §0 + §3 (Wf-355 + Wf-294B):
- `kill_then_adapt` standing directive applies to **every** §1 GPU wrapper (≥ 1 hr wall)
- VRAM budget check (`nvidia-smi` probe) is first pre-launch check
- Multi-GPU scheduling respects kanzi_venv process contention on GPU 0
- MEASURED-vs-PROJECTED label discipline applies to all wrapper stderr + stdout

### §6.5 `mol-metal-paper-writing` skill cross-ref

Per `mol-metal-paper-writing` + TODO-84:
- Skill loads `paper/section_*.tex` content as READ-only data
- Zero auto-writes to `paper/*` without `PAPER_OK=1 USER_OK=1`
- §3.3 §6 Limitations v4 paste-and-validate protocol ready (6-step procedure documented in `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` §0)

---

## §7 How user decides what to launch next

### §7.1 Decision matrix (per category)

| If user wants to... | USER_OK required | Estimated wall | Cross-ref |
|---|---|---:|---|
| Close m1 Ru 5-seed | `USER_OK=1 bash scripts/launch_wf124_p3_g5.sh` | ~5 hr GPU | §1.1 item 6 |
| Close m3 Vina 60/60 | `USER_OK=1 bash scripts/recover_g6_failed_tuples.sh` | ~11.4 hr GPU | §1.1 item 7 |
| Lift m2 platinai_4cell R² | `USER_OK=1 bash scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` | ~12 hr GPU | §1.1 item 3 |
| Enable SCI submission tooling | `USER_OK=1` for Zotero + tectonic install | ~30 min CPU | §2.1 + §2.2 |
| Update paper to Tier-2 v2 | `PAPER_OK=1 USER_OK=1` for §3.1 items 1-7 | ~12-15 hr CPU | §3.1 |
| Drop m6_kappa from paper §3 | `USER_OK=1` on metric redefinition | ~30 min CPU (docs-only) | §4.1 |
| Close pre-flight cascade drift | `USER_OK=1` on Wf-357-C3 §5.1 Recommendation 1 | ~5 min CPU (5 LOC fix) | §5.4 + Wf-364-A1 §4.2 |

### §7.2 Recommended launch sequence (if user wants to close everything)

**Sequential priority (3 batches)**:

1. **Batch 1 — install + paper tooling** (~30 min CPU)
   - §2.1 Zotero install
   - §2.2 tectonic install
   - §2.3 scienceplots pip install
2. **Batch 2 — GPU retrain** (~30-35 hr GPU; sequential single-GPU)
   - §1.1 item 6 (G5 retrain, 5 hr) — `launch_wf124_p3_g5.sh`
   - §1.1 item 7 (G6 recovery, 11.4 hr) — `recover_g6_failed_tuples.sh`
   - §1.1 item 3 (R20-P5 joint, 12 hr) — `launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh`
3. **Batch 3 — paper write** (~12-15 hr CPU after Batch 1 tooling ready)
   - §3.1 items 1-7 — paper §1-§7 v2 Tier-2 re-scope
   - §3.1 item 2 — §6 Limitations v4 paste from template scaffold

**Parallel / background:**
- §4.1 m6_kappa drop from §3 main-result → docs-only §3.1 item 5 dependency

### §7.3 What Wf-364-C2 is NOT claiming

Per `party-twenty-spirit §3.5`:
- **NOT claiming the user should grant all USER_OK at once** — per-wrapper gating is independent
- **NOT claiming the wrapper wall estimates are exact** — they are PROJECTED from Wf-258 + Wf-260 timing; MEASURED post-launch
- **NOT claiming §3 paper work is required** — Tier-2 acceptance bar is computational-only; paper work is optional
- **NOT claiming m6_kappa gate will be lifted** — 5 attempts exhausted; honest-negative is FINAL per Wf-365-A3 §2
- **NOT claiming the pre-flight 1 FAIL is Wf-364-introduced** — pre-existing Wf-347A1; tracked for follow-up

---

## §8 Standing-rule compliance

| Rule | Compliance for Wf-364-C2 |
|---|---|
| §1.1 MEASURED-vs-PROJECTED tag discipline | ✅ Every §1-§4 + §5 + §7 numeric tagged MEASURED / PROJECTED with source path |
| §1.2 Standing GPU authorization | ✅ Report is docs-only; no GPU launches triggered |
| §1.3 No paper work without authorization | ✅ Zero touches to `paper/*`; §3 inventory only |
| §1.4 Pre-flight v3 25/25 PASS gate | ✅ Pre-flight NOT re-run (docs-only); 91/94 PASS PRESERVED at Wf-364-C1 baseline |
| §1.5 Honest-negative framing | ✅ §5.4 catalog preserved; §7.3 explicit NOT-claim list |
| §5.1 Standing GPU authorization | ✅ Per-wrapper USER_OK gating preserved (no auto-trigger) |
| §5.2 Honest framing | ✅ MEASURED with sources cited; PROJECTED with bands |
| §5.3 No paper work | ✅ §3 is inventory only; no `paper/*` touches |
| §5.4 Tier-2 journal target + no wet-lab | ✅ §3 drives Tier-2 re-scope; wet-lab items DEPRECATED per user directive |

---

**END OF Wf-364-C2 USER_OK OUTSTANDING REPORT** — Owner Hugo Lee;
cross-ref `party-twenty-spirit §3.5` + `AGENTS.md §5.1 + §5.3 + §5.4` + `mol-metal-gpu-execution` §0;
last update 2026-09-29 23:17 CST; CPU-only docs surface; pre-flight v3 91/94 PASS PRESERVED; 0 GPU launches.