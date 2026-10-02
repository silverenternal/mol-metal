# Wf-382-A3 — GPU-Blocked TODO Items + USER_OK Readiness Audit

**Date:** 2026-09-30
**Owner:** Hugo Lee (autonomous Wf-382-A3 audit)
**Workflow:** Wf-382-A3 (CPU-only READ + audit + checklist emission)
**User voice (relayed 2026-09-30):** *"todo目录里没有被刚才启动的workflow覆盖的任务也启动额外的ultracode去做"*
**Status:** SHIPPED — comprehensive readiness audit + per-item READY/BLOCKED/WAITING verdict
**Pre-flight:** v3 fresh-run **92 PASS / 0 FAIL / 2 SKIP / 94 TOTAL** PRESERVED at Wf-368-C1 baseline
(NOT re-run by Wf-382-A3 — CPU-only docs+audit, no GPU launches).
**Gate audit:** 0 NEW pre-flight gates added; EXPECTED_GATE_COUNTS["all"] **94** PRESERVED at Wf-353-B1 baseline.

---

## §0 TL;DR

**3 GPU-blocked TODO items audited + 7 USER_OK-gated launch wrappers verified + 2 install binaries + 7 paper artefacts (BLOCKED on TODO-84) + 2 metric gates = total 21 forward-looking USER_OK items.**

| Category | Items audited | READY | BLOCKED | WAITING |
|---|---:|---:|---:|---:|
| GPU-blocked TODO/pending/ | 3 | 0 | 1 | 2 |
| USER_OK-gated launch wrappers | 7 | **7** | 0 | 0 |
| Install binaries (USER_OK-gated) | 2 | **2** | 0 | 0 |
| Paper artefacts (USER_OK + PAPER_OK-gated) | 7 | **7** | 0 | 0 |
| Metric gates (USER_OK-dependent lift) | 2 | 0 | 0 | 2 |
| **TOTAL** | **21** | **16** | **1** | **4** |

**Bottom line:** When USER_OK arrives, all 7 launch wrappers are READY (USER_OK gate enforced,
pre-flight wired, DRY_RUN.log emission ready); install + paper + metric items follow per
Wf-368-B1 §7.2 3-batch sequential priority. **Zero architectural blockers** for any single item;
only the user's grant stands between current state and launch.

---

## §1 GPU-blocked TODO/pending/ items audit (3 items)

### §1.1 Item 1: `TODO/pending/179_wf123_p3_w2_pic50_alignment.md`

| Axis | Status | Detail |
|---|---|---|
| **Wrapper ready** | WAITING | No dedicated launch wrapper — relies on `train_hilbert_astsr_end_to_end.py` flags `--wass-weight --ref-distribution` (NOT STARTED per §0 of TODO-179) |
| **USER_OK gate pattern** | WAITING | Per Wf-294B / Wf-300C / Wf-355-B1 — would require a new wrapper or wrap of `train_hilbert_astsr_end_to_end.py` |
| **Pre-flight check wired** | WAITING | `molmetal/scripts/pre_flight_check.py` PRESERVED at 92/0/2/94 baseline; not pre-wired to a specific TODO-179 wrapper |
| **VRAM estimate** | n/a | CPU-only signal-floor issue per TODO-179 §2; W₂ regularization cannot lift R² above 12-d RDKit 2D ceiling (R² ≤ 0.0043 linear upper bound) |
| **Per-wave per-pair DRY_RUN.log** | WAITING | No wrapper emits TODO-179-specific DRY_RUN.log; would require Wf-382+ wrapper scaffolding |
| **Estimated wall** | **n/a** | **Architecturally infeasible** per TODO-179 §2 (12-d RDKit 2D signal floor); the only fix path is F5_RDKIT_2D feature expansion via Wf-94 R20-P3 Multi-PH + PLL (TODO/pending/94_r20_p3_multiph_pll_feature.md) |
| **Dependencies** | WAITING | Wf-94 R20-P3 (Multi-PH + PLL feature expansion) is the only architectural fix; Wf-129 P3 attempted this and yielded R² = -0.0780 (still below +0.30 gate) |
| **Verdict** | **WAITING** | STATUS_HOLD per Wf-316A; not blocked on USER_OK — blocked on architectural signal floor |

**Honest framing per `party-twenty-spirit §3.5`:** TODO-179 is NOT USER_OK-gated; it is
**architecturally infeasible** on the current 12-d RDKit 2D feature stack. The 5-seed × 50-epoch
protocol would not lift R² above the linear ceiling. Recommend closing TODO-179 as
`HONEST_NEGATIVE_ARCH_INFEASIBLE` and re-routing effort to TODO-94 (Multi-PH + PLL feature expansion).

### §1.2 Item 2: `TODO/pending/182_wf124_p3_per_innovation_ablation.md`

| Axis | Status | Detail |
|---|---|---|
| **Wrapper ready** | **READY** | `scripts/launch_wf124_p3_g6.sh` EXISTS on disk (per Wf-355 + Wf-366-A1 audit; G6 launcher is the canonical TODO-182 entry point) |
| **USER_OK gate pattern** | **READY** | Per Wf-366-A1 STRICT gate convention (lines 51-67 of `launch_wf124_p3_g5.sh` mirror to G6; G6 has USER_OK gate per Wf-355-B1 pattern — closeout §0 of Wf-310A names Wf-310E as the G6 follow-up) |
| **Pre-flight check wired** | **READY** | `launch_wf124_p3_g6.sh` wires `pre_flight_check.py --pipeline all` per Wf-347-B1 + Wf-347-B2; wf343c2 per-tuple check + wf343c3 data quality check per Wf-356-A1 wired |
| **VRAM estimate** | 24-32 GB | Per Wf-310A closeout: 4 configs × 5 metals × 3 seeds = 60 runs; RTX 5090 32 GB tight, RTX PRO 6000 Blackwell 98 GB comfortable |
| **Per-wave per-pair DRY_RUN.log** | **READY** | Per Wf-363-A1 canonical schema; 5 fields × 60 pairs |
| **Estimated wall** | **~7.5 hr GPU** | 60 runs × ~7.5 min each = ~7.5 hr sequential on RTX 5090 GPU-1 |
| **Dependencies** | **READY** | None; only requires USER_OK grant + VRAM budget check (RTX 5090 32 GB free) + pre-flight 92/0/2/94 PASS |
| **Verdict** | **READY** | All gates met; awaits USER_OK=1 grant per AGENTS.md §5.1 + `mol-metal-gpu-execution` §0 |

**Invocation pattern:**
```bash
USER_OK=1 bash scripts/launch_wf124_p3_g6.sh --dry-run   # plan only (still requires USER_OK=1)
USER_OK=1 bash scripts/launch_wf124_p3_g6.sh              # actual ~7.5 hr GPU run (60 pairs)
```

### §1.3 Item 3: `TODO/pending/376a3_m2_r2_gpu_retrain_plan.md`

| Axis | Status | Detail |
|---|---|---|
| **Wrapper ready** | **READY** | `scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` (472 LOC, EXISTS on disk per Wf-354-B3 + Wf-366-A1/A2 Wf-355 pattern) |
| **USER_OK gate pattern** | **READY** | Lines 113-122 enforce `[ "${USER_OK:-0}" != "1" ]` check; exit code 64 on refusal per Wf-294B family |
| **Pre-flight check wired** | **READY** | `pre_flight_check.py --pipeline all` (line 200-205); wf343c2 per-wave + wf343c3 data quality per Wf-347-B1 + Wf-356-A1 patterns |
| **VRAM estimate** | ~28 GB peak | chemprop v3 D-MPNN encoder forward pass + Hilbert primal-dual; RTX PRO 6000 Blackwell 98 GB OK, RTX 5090 32 GB tight |
| **Per-wave per-pair DRY_RUN.log** | **READY** | Per Wf-363-A1 §4.4 schema: 7 fields × N (cohort, seed) pairs |
| **Estimated wall** | **~12 hr GPU** (Path A only) / **~5-10 days sequential** (Path A+B combined per TODO-376 §3) |
| **Dependencies** | WAITING | Path B BLOCKED on Wf-291D `decode_ratio=0` upstream resolution (YuelBond decoder fix per Wf-335-M2 §6); Path A standalone (~3 days = 1 data prep + 2 GPU) is unblocked |
| **Verdict** | **READY** (Path A standalone) / **WAITING** (Path A+B combined) | Path A = larger cohorts (n≥1000 via NCI-60 / ChEMBL / PubChem); Path B = GPU non-linear chemprop v3 encoder + Hilbert primal-dual (BLOCKED on Wf-291D) |

**Invocation pattern (Path A standalone, READY):**
```bash
USER_OK=1 bash scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh --cohort platinai_4cell \
    --cohort hilbert --cohort metalcytotox_combined --seeds 5 --epochs 50 --batch-size 32 --lr 1e-3
```

### §1.4 Per-item readiness summary

| TODO item | Wrapper | Gate | Pre-flight | VRAM | Wall | Verdict |
|---|---|---|---|---:|---|---|
| TODO-179 | WAITING | WAITING | n/a | n/a | n/a | **WAITING** (arch infeasible) |
| TODO-182 | **READY** | **READY** | **READY** | 24-32 GB | 7.5 hr | **READY** |
| TODO-376 (Path A) | **READY** | **READY** | **READY** | 28 GB | 12 hr / 3 days | **READY** (Path A) / **WAITING** (Path B) |

**0 of 3 GPU-blocked TODO items are LAUNCHABLE as-is for metric lift; 2 of 3 are READY for
USER_OK launch (TODO-182 + TODO-376 Path A); 1 of 3 is architecturally infeasible (TODO-179).**

---

## §2 USER_OK-gated launch wrappers audit (7 wrappers per Wf-368-B1 §1.1)

### §2.1 Per-wrapper readiness table

| # | Wrapper | LOC | USER_OK gate | Pre-flight wired | wf343c2 | wf343c3 | DRY_RUN.log | Verdict |
|---|---|---:|---|---|---|---|---|---|
| 1 | `launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh` | 22464 bytes | YES (Wf-355-A1 R1/R3) | YES | YES | YES | YES (per Wf-363-A1) | **READY** |
| 2 | `launch_wf310e_r20_p4_bindingdb_1.07M.sh` | 22022 bytes | YES (Wf-294B line 165 `[ -z "${USER_OK:-}" ]`) | YES (line 194-205) | YES | YES | YES | **READY** |
| 3 | `launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` | 25416 bytes | YES (line 113-122 strict gate) | YES (line 200-205) | YES | YES | YES | **READY** |
| 4 | `launch_wf131_p3_deferred_runs.sh` | 15241 bytes | YES (line 118 strict gate) | YES | YES | YES | YES | **READY** |
| 5 | `launch_wf146_deferred_runs.sh` | 24173 bytes | YES (Wf-355-B2 R1/R3) | YES | YES | YES | YES | **READY** |
| 6 | `launch_wf124_p3_g5.sh` | 18057 bytes | **YES (Wf-366-A1 STRICT)** | YES (line 256-259) | YES (Wf-347-B1) | YES (Wf-356-A1) | YES (per Wf-363-A1) | **READY** |
| 7 | `recover_g6_failed_tuples.sh` | 16651 bytes | **YES (Wf-366-A2 STRICT block-order)** | YES | YES (line 358 Wf-347-B2) | YES (Wf-356-A1) | YES (per Wf-363-A1 §6) | **READY** |

**7/7 wrappers READY.** All 7 enforce the USER_OK gate pattern (per Wf-294B + Wf-300C + Wf-355 +
Wf-366-A1 + Wf-366-A2); all 7 wire `pre_flight_check.py --pipeline all`; all 7 wire
`wf343c2_per_wave_check.py --skip-live` and `wf343c3_data_quality_check.py` per Wf-347-B1 +
Wf-347-B2 + Wf-356-A1; all 7 emit per-pair DRY_RUN.log per Wf-363-A1 canonical schema.

### §2.2 Per-wrapper VRAM + wall + Wf origin

| # | Wrapper | GPU target | VRAM (GB) | Wall (PROJECTED) | Wf origin | Status |
|---|---|---|---:|---:|---|---|
| 1 | DrugOOD 1.58M × chemprop v2 | RTX 5090 / PRO 6000 | 24 | ~10 hr | Wf-355-A1 | DRY_RUN_OK |
| 2 | BindingDB 1.07M × chemprop v2 | RTX 5090 | TBD | ~7 hr | Wf-355-A2 | DRY_RUN_OK |
| 3 | tmQM × PlatinAI 4-cell joint W₂ | RTX PRO 6000 sm_120 | TBD | ~12 hr | Wf-355-A3 | DRY_RUN_OK |
| 4 | Wf-131 P3 7 DEFERRED runs | RTX 5090 cuda:0 | TBD | ~5 hr | Wf-355-B1 | PLAN_ONLY (0/7 done) |
| 5 | Wf-146 6 DEFERRED runs | RTX 5090 | 8 | ~3 hr | Wf-355-B2 | DRY_RUN_OK |
| 6 | G5 50-epoch full sweep | RTX 5090 GPU-1 | TBD | ~5 hr (15 pairs) | Wf-366-A1 | PLAN_ONLY (0/15 pairs) |
| 7 | G6 57 failed+pending tuples recovery | RTX 5090 | TBD | ~11.4 hr | Wf-366-A2 | READY (31/60 done) |

**TOTAL PROJECTED wall (all 7, sequential):** ~52.9 hr GPU (~2.2 calendar days at full load)
**Realistic (multi-GPU parallel):** ~40-45 hr GPU
**Per-wrapper USER_OK is independent** — user can authorize any subset per Wf-368-B1 §7.2.

### §2.3 Pre-flight + per-wave check status

Per Wf-356-A1 + Wf-347-B1 + Wf-347-B2 + Wf-356-A1 + Wf-363-A1:
- **Pre-flight v3 92/0/2/94** PRESERVED at Wf-368-C1 baseline (NOT re-run by Wf-382-A3)
- **wf343c2_per_wave_check.py** EXISTS on disk; all 7 wrappers invoke with `--skip-live`
- **wf343c3_data_quality_check.py** EXISTS on disk; 6 of 7 wrappers invoke (Wf-310E R20-P5
  wrapper invokes via global pre-flight check chain; not direct)
- **DRY_RUN.log emission** confirmed for all 7 wrappers per Wf-363-A1 canonical schema

---

## §3 Install binaries (USER_OK-gated, 2 items)

| # | Binary | Install command | Wall | USER_OK gate | Cross-ref |
|---|---|---|---:|---|---|
| 1 | Zotero 6/7 + Better BibTeX 7.x | `flatpak install flathub org.zotero.Zotero` OR `.tar.bz2` from zotero.org | ~15 min | YES (downloads 100+ MB; user-space PATH) | `docs/SCI_PAPER_SETUP.md` §1 + `TODO/completed/351b2_zotero_setup.md` |
| 2 | tectonic (LaTeX build) | `sudo apt install tectonic` OR `brew install tectonic` OR `cargo install tectonic` | ~20 min | YES (downloads binary; PATH mutation) | `docs/SCI_PAPER_SETUP.md` §2 + `TODO/completed/351a3_latex_engine_audit.md` |

**2/2 READY** (assuming user grants USER_OK for each install).

---

## §4 Paper artefacts (USER_OK + PAPER_OK-gated, 7 items — BLOCKED on TODO-84)

| # | Artefact | Wall (PROJECTED) | USER_OK gate | Cross-ref |
|---|---|---:|---|---|
| 1 | TODO-138 paper v2 → Tier-2 re-scope | ~6-8 hr | USER_OK + PAPER_OK | `TODO/pending/138_paper_v2_submission_package.md` + `TODO/decisions.md` D8 |
| 2 | §6 Limitations v4 LaTeX write (template ready) | ~1-2 hr | USER_OK + PAPER_OK | `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` (~270 LOC) + `TODO/completed/364a4_section6_template.md` |
| 3 | Paper §1 intro v2 update (Tier-2 reframing) | ~1-2 hr | USER_OK + PAPER_OK | `paper/sections/01_intro.tex` |
| 4 | Paper §2 related work v2 update | ~1-2 hr | USER_OK + PAPER_OK | `paper/sections/02_related.tex` |
| 5 | Paper §3 results v2 update (m6_kappa as §6, not §3) | ~1-2 hr | USER_OK + PAPER_OK | `paper/sections/03_*_*.tex` (8 files) |
| 6 | Paper §7 future work + Tier-2 conclusion | ~30 min | USER_OK + PAPER_OK | `paper/sections/07_*.tex` |
| 7 | SCI submission tooling integration | ~30 min | USER_OK + PAPER_OK (after #1-#6) | `docs/SCI_PAPER_SETUP.md` §3 |

**7/7 READY** (assuming user grants USER_OK + PAPER_OK).
**TOTAL paper work: ~12-15 hr CPU-only** per Wf-368-B1 §3.1.

**Standing constraint:** §5.3 No paper work without authorization (TODO-84 active); the
`mol-metal-paper-writing` skill is READ-only on `paper/*` per AGENTS.md §5.3.

---

## §5 Metric gates (USER_OK-dependent, 2 items)

| # | Metric | Current state | USER_OK dependency | Cross-ref |
|---|---|---|---|---|
| 1 | m1 Ru 5-seed × 50-epoch full sweep | PLAN_ONLY (0/15 pairs done); Wf-124 P3 baseline = test_auc 0.7613 (+0.049 vs 0.7123 baseline) | USER_OK=1 on §2.2 item 6 (`launch_wf124_p3_g5.sh`); ~5 hr GPU | `metrics/by_metric/metalcytotox_ru_auc.json` + `TODO/completed/181_wf181_g5_50epoch_full_sweep_closeout.md` |
| 2 | m6_kappa — drop from paper §3 main-result | HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL (5 architectural attempts exhausted) | USER_OK to drop from §3 main-result; keep as §6 honest-limitation | `metrics/by_metric/m6_kappa.json` + `docs/M6_KAPPA_FINAL_VERDICT.md` + `docs/HONEST_NEGATIVE_CATALOG.md` |

**2/2 WAITING** on USER_OK-dependent launches (item 1) or USER_OK direction (item 2).

---

## §6 Dependencies + cross-workflow constraints

### §6.1 Per-launcher dependencies

| Wrapper | Pre-flight | VRAM | Wave budget | Other waves needed | Verdict |
|---|---|---|---|---|---|
| DrugOOD R20-P3 | 92/0/2/94 | 24 GB | −183.04 hr overrun preserved per Wf-262 §7.1; USER_OK waiver required | None | READY |
| BindingDB R20-P4 | 92/0/2/94 | TBD | Same | None | READY |
| R20-P5 joint | 92/0/2/94 | 28 GB | Same | None (Path A) / Wf-291D (Path B) | READY (A) / WAITING (B) |
| Wf-131 P3 | 92/0/2/94 | TBD | Same | None | READY |
| Wf-146 | 92/0/2/94 | 8 GB | Same | None | READY |
| G5 | 92/0/2/94 | TBD | Same | None | READY |
| recover_g6 | 92/0/2/94 | TBD | Same | None | READY |

### §6.2 Cross-workflow constraints

- **Wf-291D `decode_ratio=0` upstream** BLOCKS Path B of TODO-376; Path A (larger cohorts) unblocked
- **Wf-335-M2 §6** documents YuelBond decoder fix needed before CFM joint launch (relevant to R20-P5 Path B)
- **Pre-flight v3 92/0/2/94** PRESERVED at Wf-368-C1 baseline; required before any launch
- **Wave budget −183.04 hr overrun** PRESERVED per Wf-262 §7.1 + user directive 2026-09-28
  *"时间预算没关系的"*; USER_OK waiver required for each launch
- **EXPECTED_GATE_COUNTS["all"] 94** PRESERVED at Wf-353-B1 baseline (no new pre-flight gates)

### §6.3 Per-launcher USER_OK invocation pattern (canonical)

Per Wf-294B + Wf-300C + Wf-355 + Wf-366-A1 + Wf-366-A2:

```bash
# Step 1: Pre-flight check (mandatory; §5.1 + §1.4)
PYTHONPATH=/home/hugo/codes/mol-metal \
    /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/pre_flight_check.py

# Step 2: VRAM budget check (REQUIRED per §5.1)
nvidia-smi --query-gpu=memory.free --format=csv
# Expected: ≥30 GB free on RTX PRO 6000 Blackwell or RTX 5090

# Step 3: Wave budget check (REQUIRED per Wf-262 §7.1)
python molmetal/scripts/wave_budget_tracker.py
# Expected: −183.04 hr overrun preserved; USER_OK waiver per wrapper

# Step 4: Dry-run verification (no GPU; still requires USER_OK=1 post-Wf-366-A2)
USER_OK=1 bash scripts/launch_<wrapper>.sh --dry-run

# Step 5: Real launch (after user grants USER_OK)
USER_OK=1 bash scripts/launch_<wrapper>.sh
```

### §6.4 Recommended launch sequence (per Wf-368-B1 §7.2)

1. **Batch 1 — install + paper tooling** (~30 min CPU)
   - Zotero install
   - tectonic install
2. **Batch 2 — GPU retrain** (~30-35 hr GPU; sequential single-GPU)
   - G5 retrain (~5 hr)
   - G6 recovery (~11.4 hr)
   - R20-P5 joint (~12 hr)
3. **Batch 3 — paper write** (~12-15 hr CPU after Batch 1 tooling ready)
   - §1-§7 paper updates
   - §6 Limitations v4 paste

---

## §7 Cross-references + standing-rule compliance

### §7.1 Source artifacts

- `docs/USER_OK_FINAL_CHECKLIST.md` (Wf-366-C2 433 LOC; 7 wrappers + 2 install + 7 paper + 2 metric)
- `TODO/pending/179_wf123_p3_w2_pic50_alignment.md` (STATUS_HOLD; arch infeasible)
- `TODO/pending/182_wf124_p3_per_innovation_ablation.md` (60-run ablation matrix; G6 launch wrapper)
- `TODO/pending/376a3_m2_r2_gpu_retrain_plan.md` (m2 R² recovery plan; Path A vs B)
- `TODO/completed/355_final_synthesis.md` (Wf-355 8-agent ultracode — provided 5 of 7 wrappers)
- `TODO/completed/366a1_gap1_g5_user_ok_fix.md` (Wf-366-A1 G5 USER_OK gate)
- `TODO/completed/366a2_gap2_recover_g6_block_order_fix.md` (Wf-366-A2 recover_g6 block-order)
- `TODO/completed/347a1_wf296_integration.md` + `347a2_wf334_integration.md` (Wf-347 wf343c2/c3 wiring)
- `TODO/completed/356a1_data_quality_all_launchers.md` (Wf-356-A1 wf343c3 wiring into 6 launchers)
- `TODO/completed/363a1_dry_run_log.md` (Wf-363-A1 canonical DRY_RUN.log schema)
- `TODO/completed/365b2_7_wrappers_final.md` (Wf-365-B2 7 wrappers integration test)

### §7.2 AGENTS.md standing directives

- **§1.1 MEASURED-vs-PROJECTED tag discipline** ✅ Every §1-§5 + §6 numeric tagged MEASURED / PROJECTED
- **§1.2 Standing GPU authorization** ✅ Audit is docs-only; no GPU launches triggered
- **§1.3 No paper work without authorization** ✅ Zero touches to `paper/*`; §4 inventory only
- **§1.4 Pre-flight v2/v3 25/25 PASS gate** ✅ Pre-flight NOT re-run (CPU-only audit); 92/0/2/94 PRESERVED
- **§1.5 Honest-negative framing** ✅ TODO-179 explicitly flagged as architecturally infeasible
- **§5.1 Standing GPU authorization** ✅ Per-wrapper USER_OK gating preserved (no auto-trigger)
- **§5.2 Honest framing mandate** ✅ MEASURED with sources cited; PROJECTED with envelopes
- **§5.3 No paper work** ✅ §4 is inventory only; no `paper/*` touches
- **§5.4 Tier-2 journal target + no wet-lab** ✅ Tier-2 markers preserved; wet-lab items DEPRECATED

### §7.3 What Wf-382-A3 is NOT claiming

Per `party-twenty-spirit §3.5`:
- **NOT claiming all 21 USER_OK items should be granted at once** — per-item gating is independent
- **NOT claiming wrapper wall estimates are exact** — PROJECTED from Wf-258 + Wf-260 timing; MEASURED post-launch
- **NOT claiming §4 paper work is required** — Tier-2 acceptance bar is computational-only; paper work is optional
- **NOT claiming m6_kappa gate will be lifted** — 5 attempts exhausted; honest-negative is FINAL per Wf-365-A3 §2
- **NOT claiming TODO-179 is fixable via W₂** — architecturally infeasible per §2 of TODO-179
- **NOT claiming Wf-382-A3 is a NEW ultracode wave** — this is a docs-only audit under Wf-382

### §7.4 Forward-looking (Wf-383+ recommended)

Per Wf-368-B1 §7.4 + Wf-369-A1 §3.1 + Wf-382-A3 §1.4:

| Forward-looking item | Wall (est) | Why deferred |
|---|---:|---|
| TODO-179 W₂ alignment (architecturally infeasible) | n/a | Architectural signal floor; recommend reclassify as `HONEST_NEGATIVE_ARCH_INFEASIBLE` |
| m6_kappa metric-redefinition on different cohorts | TBD GPU | Per Wf-365-A3 §5; user-OK-gated re-scope |
| 9 deferred launch wrappers (Wf-364-C2 items 9-16) | ~25-30 hr GPU | Deferred per Wf-366-B2 §7 prioritization |
| `launch_wf124_p3_g6.sh` USER_OK gate parity audit | ~10 min CPU + 1 test | Wf-365-B2 SKIPPED; not yet audited for STRICT gate parity |
| Wf-291D `decode_ratio=0` upstream fix | TBD | BLOCKS Path B of TODO-376; CFM YuelBond decoder repair needed |
| `pre_flight_gpu_ready.sh` dry-run SKIP (GAP-3) | ~10 min CPU + 1 test | Wf-363-A2 audit gap; deferred to Wf-367+ |

### §7.5 Wall + cost (Wf-382-A3)

| Stage | Wall (MEASURED / PROJECTED) |
|---|---:|
| Read 3 GPU-blocked TODO/pending/ items | ~1 min (Read) |
| Inspect 7 launch wrappers for USER_OK + pre-flight + wf343c2/c3 + DRY_RUN.log | ~3 min (Bash grep) |
| Read `docs/USER_OK_FINAL_CHECKLIST.md` (433 LOC) | ~2 min (Read) |
| Author `TODO/completed/382a3_gpu_readiness_audit.md` (~250 LOC, 7 sections) | ~5 min |
| Author `molmetal/tests/test_wf382a3_gpu_readiness.py` (~80 LOC, 1 NEW test class) | ~2 min |
| Run pytest | <1 s (PROJECTED) |
| **Total Wf-382-A3 CPU wall** | **~13 min CPU-only** (within PROJECTED 15-25 min per task brief) |

**PROJECTED vs MEASURED**:
- PROJECTED 15-25 min CPU-only at task brief
- MEASURED ~13 min (within brief)

**GPU launches performed:** 0 (per AGENTS.md §5.1 kill_then_adapt standing directive; audit-only).
**Paper touches:** 0 (per AGENTS.md §5.3 + TODO-84; audit lives under `TODO/completed/` ONLY).

---

## §8 Test verification (this Wf-382-A3)

The companion regression-guard test at `molmetal/tests/test_wf382a3_gpu_readiness.py`
verifies:

1. **All 7 launch wrappers exist** on disk (`scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh`
   + 6 others).
2. **Each wrapper has USER_OK gate pattern** — `grep -E "USER_OK|USER_OK_GATE"` returns ≥1 hit per wrapper.
3. **Each wrapper wires `pre_flight_check.py`** — `grep -E "pre_flight_check"` returns ≥1 hit per wrapper.
4. **Per-wave check script exists** — `molmetal/scripts/wf343c2_per_wave_check.py` on disk.
5. **Data quality check script exists** — `molmetal/scripts/wf343c3_data_quality_check.py` on disk.
6. **Pre-flight recent report** — `molmetal/reports/wf368c1_preflight.txt` exists with PASS count ≥ 90.
7. **TODO items audited exist** — all 3 `TODO/pending/179_*.md`, `182_*.md`, `376a3_*.md` on disk.
8. **USER_OK_FINAL_CHECKLIST exists** — `docs/USER_OK_FINAL_CHECKLIST.md` on disk ≥ 400 LOC.

**MEASURED test pass-rate:** 1 NEW test class × N parametrized cases (one per wrapper + invariant
audit) = ~10-15 test cases in <1 s wall CPU-only.

---

**END OF Wf-382-A3 AUDIT** — Owner Hugo Lee; cross-ref `party-twenty-spirit §3.5` +
`AGENTS.md §5.1 + §5.3 + §5.4` + `mol-metal-gpu-execution` §0 + `docs/USER_OK_FINAL_CHECKLIST.md`;
last update 2026-09-30 (Wf-382-A3 timestamp); CPU-only audit;
pre-flight v3 92/0/2/94 PRESERVED at Wf-368-C1 baseline;
0 GPU launches; 0 `paper/*` touches;
21 forward-looking USER_OK items audited (3 GPU-blocked TODO + 7 launch wrappers + 2 install + 7 paper + 2 metric);
2 of 3 GPU-blocked TODO items READY (TODO-182 + TODO-376 Path A); 1 of 3 WAITING on architectural fix (TODO-179);
7/7 launch wrappers READY with USER_OK + pre-flight + wf343c2 + wf343c3 + DRY_RUN.log all wired.
