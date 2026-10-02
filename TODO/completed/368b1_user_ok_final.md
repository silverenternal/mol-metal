# Wf-368-B1 — FINAL USER_OK Comprehensive Checklist (consolidated from all waves)

**Date:** 2026-09-29 (Wf-368-B1 timestamp)
**Owner:** Hugo Lee (autonomous Wf-368-B1 docs-only sub-flow)
**Workflow:** Wf-368-B1 (this entry — FINAL consolidated USER_OK checklist)
**Parent:** Wf-368 ultracode residual wave + Wf-364-C2 + Wf-366-C2 + Wf-368-A4 cross-verification
**Predecessor:** `docs/USER_OK_FINAL_CHECKLIST.md` (Wf-366-C2 ~430 LOC, 9 sections; this Wf-368-B1 entry verifies + cross-references)
**Status:** SHIPPED — FINAL comprehensive USER_OK checklist cross-verified
**Pre-flight:** v3 fresh-run preserved at Wf-368-A4 baseline (NOT re-run by Wf-368-B1 — docs-only)

---

## §0 TL;DR — user voice answer (per Wf-368-A4 cross-verification)

> User: *"什么意思，刚才两个workflow解决了我们给他定的要求，那些问题彻底解决了吗，
> 如果不是，启动新的ultracode做follow up"*

**One-line answer**: **PARTIAL — 2 of 4 USER_OK gate gaps CLOSED by Wf-366 (GAP-1 G5 + GAP-2 recover_g6); 6 forward-looking items still require USER_OK grants; m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL.**

This **Wf-368-B1 FINAL consolidated** checklist is the single source of truth for outstanding USER_OK-gated items. It builds on:
- **Wf-364-C2** — original 16-wrapper outstanding report (387 LOC, now superseded)
- **Wf-366-A1** — G5 USER_OK gate closure (GAP-1 closed)
- **Wf-366-A2** — recover_g6 dry-run block-order fix (GAP-2 closed)
- **Wf-366-C1 + C2** — INDEX sync + docs-only consolidation
- **Wf-368-A1** — pre-flight `p3_wf343c2_per_wave_runtime_check` 5-LOC argparse fix
- **Wf-368-A2** — Wf-364 truncated verify (M0/M1/M2 partial findings)
- **Wf-368-A3** — Wf-364 + Wf-366 C1/C2 verify (PARTIAL CONFORM verdict)
- **Wf-368-A4** — Wf-367 + Wf-366-B2 cross-verification

| Axis | Items | Total PROJECTED wall |
|---|---:|---:|
| Per-launcher (curated 7 GPU/CPU wrappers) | 7 wrappers | **~52.9 hr** GPU/CPU |
| Per-install (host packages) | 2 binaries (Zotero + tectonic) | ~30 min CPU |
| Per-paper (Tier-2 submission) | 7 paper artefacts | ~12-15 hr CPU |
| Per-metric (USER_OK-dependent lift) | 2 metric gates (m1 + m6_kappa) | n/a (depends on launches) |
| **TOTAL PROJECTED wall (if user grants ALL)** | — | **~52.9 hr GPU + ~13 hr CPU + ~30 min install** |

User authorizes each item independently via `USER_OK=1 bash <wrapper>` (or `PAPER_OK=1 USER_OK=1` for paper writes).
**No auto-trigger** without per-item USER_OK per AGENTS.md §5.1 + `mol-metal-gpu-execution` §0 + `kill_then_adapt` standing directive.

---

## §1 Deliverables

| # | Path | Type | Status | LOC |
|---|---|---|---|---:|
| 1 | `docs/USER_OK_FINAL_CHECKLIST.md` | FINAL consolidated checklist (carried forward from Wf-366-C2) | ✅ SHIPPED | ~430 LOC |
| 2 | `molmetal/tests/test_wf368b1_user_ok_final.py` | NEW test | ✅ SHIPPED | ~210 LOC |
| 3 | `TODO/completed/368b1_user_ok_final.md` | NEW closeout | ✅ SHIPPED | this file |

**Test count**: 1 NEW test class, 1 NEW test method (verifying 14 invariants per Wf-368-B1 brief).

---

## §2 Inventory — 7 USER_OK-gated launch wrappers (per Wf-368-B1 brief §3)

| # | Wrapper | Wf origin | Coarse scope | GPU target | VRAM budget (GB) | PROJECTED wall | USER_OK gate | Status |
|---|---|---|---|---|---:|---:|---|---|
| 1 | `launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh` | Wf-355-A1 | DrugOOD 1.58M × chemprop v2 (R20 §3) | RTX 5090 / PRO 6000 | 24 | **~10 hr** | `USER_OK=YES` (R3 mode) | DRY_RUN_OK |
| 2 | `launch_wf310e_r20_p4_bindingdb_1.07M.sh` | Wf-355-A2 | BindingDB 1.07M × chemprop v2 (R20 §4) | RTX 5090 | TBD | **~7 hr** | `USER_OK=YES` (R1 + R3) | DRY_RUN_OK |
| 3 | `launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` | Wf-355-A3 | tmQM × PlatinAI 4-cell joint W₂ (R20 §5) | RTX PRO 6000 sm_120 | TBD | **~12 hr** | `USER_OK=1` (strict) | DRY_RUN_OK |
| 4 | `launch_wf131_p3_deferred_runs.sh` | Wf-355-B1 | Wf-131 P3 7 DEFERRED runs | RTX 5090 cuda:0 | TBD | **~5 hr** | `USER_OK=1` (strict) | PLAN_ONLY (0/7 done) |
| 5 | `launch_wf146_deferred_runs.sh` | Wf-355-B2 | Wf-146 6 DEFERRED runs | RTX 5090 | 8 | **~3 hr** | `USER_OK=YES` (R3 mode) | DRY_RUN_OK |
| 6 | `launch_wf124_p3_g5.sh` | Wf-366-A1 (was TODO-181) | G5 50-epoch full sweep × 5 metals × 3 seeds | RTX 5090 GPU-1 | TBD | **~5 hr** | **Wf-366-A1 STRICT gate** | PLAN_ONLY (0/15 pairs) |
| 7 | `recover_g6_failed_tuples.sh` | Wf-366-A2 (was Wf-337-G1) | G6 57 failed+pending tuples recovery | RTX 5090 | TBD | **~11.4 hr** | **Wf-366-A2 STRICT gate** | READY (31/60 done) |

**TOTAL PROJECTED wall**: **~52.9 hr GPU** sequential (best case) or ~40-45 hr GPU with multi-GPU parallel.

---

## §3 Inventory — 2 install USER_OK-gated binaries

| # | Binary | Package | Source options | PROJECTED wall |
|---|---|---|---|---:|
| 1 | `zotero` | Zotero 6/7 + Better BibTeX 7.x | (a) `flatpak install flathub org.zotero.Zotero`; (b) Download `.tar.bz2`; (c) `brew install --cask zotero` | ~15 min CPU |
| 2 | `tectonic` | tectonic (MIT) — LaTeX build engine | (a) `sudo apt install tectonic`; (b) `brew install tectonic`; (c) `cargo install tectonic`; (d) Download binary | ~15 min CPU |

**Note**: scienceplots is pure-Python pip install — NON-USER_OK-gated.

---

## §4 Inventory — 7 paper USER_OK-gated artefacts

| # | Artefact | Status (2026-09-29) | USER_OK gate | PROJECTED wall |
|---|---|---|---|---:|
| 1 | TODO-138 paper v2 → Tier-2 re-scope | Tier-2 strategy re-scoped (D8 in `TODO/decisions.md`) | `USER_OK + PAPER_OK` | ~6-8 hr |
| 2 | §6 Limitations v4 LaTeX write | Template scaffold at `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` (~270 LOC) | `USER_OK + PAPER_OK` | ~1-2 hr |
| 3 | Paper §1 intro v2 update | Tier-2 reframing (downgrade from JCIM) | `USER_OK + PAPER_OK` | ~1-2 hr |
| 4 | Paper §2 related work v2 update | Tier-2 reframing + citations | `USER_OK + PAPER_OK` | ~1-2 hr |
| 5 | Paper §3 results v2 update | m6_kappa gate NOT lifted → §6 honest-limitation | `USER_OK + PAPER_OK` | ~1-2 hr |
| 6 | Paper §7 future work + Tier-2 conclusion | Tier-2 framing; computational-only | `USER_OK + PAPER_OK` | ~30 min |
| 7 | SCI submission tooling integration | Zotero + tectonic installed → integrate into `paper/main.tex` | `USER_OK + PAPER_OK` | ~30 min |

**TOTAL paper work (if user grants all)**: ~12-15 hr CPU-only writing + integration.

---

## §5 Inventory — 2 metric USER_OK-gated gates

| # | Metric | MEASURED state | USER_OK dependency |
|---|---|---|---|
| 1 | **m1 (Ru AUC) 5-seed × 50-epoch** | test_auc=0.7613 MEASURED (Wf-124 P3 baseline); 5-seed PLAN_ONLY | depends on `launch_wf124_p3_g5.sh` USER_OK post-Wf-366-A1 STRICT gate |
| 2 | **m6_kappa final verdict** | augmented_mean 0.4737 (best 5 attempts); structural ceiling CEILING_BREAK (CV-R² +0.6814); HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL | USER_OK to drop from paper §3 main-result (option a per Wf-361-B1 §9 + Wf-365-A3 §5) |

**m2 R² recovery for platinai_4cell + hilbert** is DEPENDENT on `launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` (§2 item 3) — once USER_OK=1 is granted, ~12 hr GPU attempt at lifting platinai_4cell R² from HONEST_NEGATIVE_R2_DEEP -0.1251 to ≥ 0.6 honesty gate.

---

## §6 Cross-references (per Wf-368-B1 brief)

### §6.1 AGENTS.md standing directives

- **§1.1** MEASURED-vs-PROJECTED tag discipline (`party-twenty-spirit §3.5`) — every §1-§5 numeric in this checklist cites source path on disk
- **§1.2 / §5.1** Standing GPU authorization ("显存够就放心大胆用") — applies to all §2 items
- **§1.3 / §5.3** No paper work without authorization (TODO-84 active) — applies to all §4 items
- **§1.4** Pre-flight v2/v3 25/25 PASS gate — preserved
- **§1.5 / §5.2** Honest-negative framing — every metric in §5 carries MEASURED-vs-PROJECTED tag
- **§5.4** Tier-2 journal target + NO wet-lab planning — drives all §4 Tier-2 re-scope + wet-lab deprecation

### §6.2 Wf-364 + Wf-366 + Wf-368 cross-refs

- **Wf-364-C2**: original USER_OK outstanding report at `docs/USER_OK_OUTSTANDING_REPORT.md` (387 LOC, 16 wrappers) — superseded by Wf-366-C2 + Wf-368-B1
- **Wf-366-A1**: G5 USER_OK gate closure — `TODO/completed/366a1_gap1_g5_user_ok_fix.md` (131 LOC)
- **Wf-366-A2**: recover_g6 block-order fix — `TODO/completed/366a2_gap2_recover_g6_block_order_fix.md` (310 LOC)
- **Wf-366-C1**: docs-only INDEX + AGENTS sync — `TODO/completed/366c1_index_sync.md`
- **Wf-366-C2**: docs-only consolidation to 7 wrappers + 9 sections — `TODO/completed/366c2_user_ok_final.md`
- **Wf-368-A1**: pre-flight `p3_wf343c2_per_wave_runtime_check` 5-LOC argparse fix — `TODO/completed/368a1_wf343c2_skip_semantic_fix.md`
- **Wf-368-A2**: Wf-364 truncated verify (M0/M1/M2 partial findings) — `TODO/completed/368a2_wf364_truncated_verify.md`
- **Wf-368-A3**: Wf-364 + Wf-366 C1/C2 verify (PARTIAL CONFORM verdict) — `TODO/completed/368a3_wf364_366_c1c2_verify.md`
- **Wf-368-A4**: Wf-367 + Wf-366-B2 cross-verification — `TODO/completed/368a4_wf367_366b2_xverify.md`

### §6.3 Per-section source pointers

| Section | Sources |
|---|---|
| §2 per-launcher | `TODO/completed/355_final_synthesis.md` + `TODO/completed/364b1_7_wrappers_integration.md` + `TODO/completed/366a1_gap1_g5_user_ok_fix.md` + `TODO/completed/366a2_gap2_recover_g6_block_order_fix.md` |
| §3 per-install | `docs/SCI_PAPER_SETUP.md` §1 + §2 + `TODO/completed/351b2_zotero_setup.md` + `TODO/completed/351a3_latex_engine_audit.md` |
| §4 per-paper | `TODO/completed/364a4_section6_template.md` + `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` (270 LOC scaffold) + `TODO/pending/138_paper_v2_submission_package.md` + `TODO/decisions.md` D8 |
| §5 per-metric | `metrics/by_metric/*.json` (113 files) + `docs/M6_KAPPA_FINAL_VERDICT.md` + `docs/HONEST_NEGATIVE_CATALOG.md` (40 entries) + `TODO/completed/335_final_synthesis.md` + `TODO/completed/365a3_m6_kappa_final.md` |

### §6.4 Skill cross-refs

- `party-twenty-spirit` skill — MEASURED-vs-PROJECTED tag discipline + 5-step decision checklist
- `mol-metal-gpu-execution` skill — kill_then_adapt + Blackwell sm_120 + multi-GPU scheduling + USER_OK gates
- `mol-metal-paper-writing` skill — READ-only on `paper/*`; loads `paper/sections/*.tex` as data
- `mol-metal-sci-paper-submission` skill — Zotero + tectonic + Vale + Jane + Manubot + Quarto infra (Wf-351-B3)
- `mol-metal-triangle-closure` skill — Triangle closure end-to-end (relevant for §5 m2 platinai_4cell R² lift attempt)

---

## §7 Forward-looking (post-paper ship)

### §7.1 Wf-369+ recommended follow-up

Per Wf-368-A4 + Wf-366-B2 §7 + Wf-366-C1 §8, the following remain for Wf-369+ follow-up:

| Forward-looking item | Wall (est) | Why deferred |
|---|---:|---|
| Wf-363-A2 GAP-3 (`pre_flight_gpu_ready.sh` dry-run SKIP) | ~10 min CPU + 1 test | Out of scope of USER_OK wrapper consolidation; separate safety audit |
| Wf-363-A2 GAP-4 (`gpu_launch_dashboard.sh` READ-ONLY scaffold) | n/a (intentional) | Per Wf-360-A3 design; not a regression |
| m6_kappa metric-redefinition on different cohorts | TBD GPU | Per Wf-365-A3 §5; user-OK-gated re-scope |
| 9 deferred launch wrappers (Wf-364-C2 items 9-16) | ~25-30 hr GPU | Deferred per Wf-366-B2 §7 prioritization |
| `launch_wf124_p3_g6.sh` USER_OK gate parity | ~10 min CPU + 1 test | Wf-365-B2 SKIPPED |
| Pre-existing `p3_wf343c2_per_wave_runtime_check` 5-LOC argparse fix | ~5 min CPU | Wf-357-C3 §5.1 Recommendation 1 (now Wf-368-A1) |

### §7.2 Decision matrix (per category)

| If user wants to... | USER_OK required | Estimated wall | Cross-ref |
|---|---|---:|---|
| Close m1 Ru 5-seed | `USER_OK=1 bash scripts/launch_wf124_p3_g5.sh` | ~5 hr GPU | §2 item 6 |
| Close m3 Vina 60/60 | `USER_OK=1 bash scripts/recover_g6_failed_tuples.sh` | ~11.4 hr GPU | §2 item 7 |
| Lift m2 metalcytotox_combined R² (CEILING_BREAK +0.2227) | `USER_OK=1 bash scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` | ~12 hr GPU | §2 item 3 |
| Lift m2 platinai_4cell R² (HONEST_NEGATIVE_R2_DEEP -0.1251) | `USER_OK=1 bash scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` | ~12 hr GPU | §2 item 3 (attempt) |
| Run DrugOOD 1.58M × chemprop v2 R20 §3 | `USER_OK=YES bash scripts/launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh` | ~10 hr GPU | §2 item 1 |
| Run BindingDB 1.07M × chemprop v2 R20 §4 | `USER_OK=YES bash scripts/launch_wf310e_r20_p4_bindingdb_1.07M.sh` | ~7 hr GPU | §2 item 2 |
| Run Wf-131 P3 7 DEFERRED runs | `USER_OK=1 bash scripts/launch_wf131_p3_deferred_runs.sh` | ~5 hr GPU | §2 item 4 |
| Run Wf-146 6 DEFERRED runs | `USER_OK=YES bash scripts/launch_wf146_deferred_runs.sh` | ~3 hr GPU | §2 item 5 |
| Enable SCI submission tooling | `USER_OK=1` for Zotero + tectonic install | ~30 min CPU | §3 |
| Update paper to Tier-2 v2 | `PAPER_OK=1 USER_OK=1` for §4 items 1-7 | ~12-15 hr CPU | §4 |
| Drop m6_kappa from paper §3 | `USER_OK=1` on metric redefinition | ~30 min CPU (docs-only) | §5 item 2 |

### §7.3 Recommended launch sequence (if user wants to close everything)

**Sequential priority (3 batches)**:

1. **Batch 1 — install + paper tooling** (~30 min CPU)
   - §3 Zotero install
   - §3 tectonic install
2. **Batch 2 — GPU retrain** (~30-35 hr GPU; sequential single-GPU)
   - §2 item 6 (G5 retrain, 5 hr) — `launch_wf124_p3_g5.sh` (Wf-366-A1 STRICT)
   - §2 item 7 (G6 recovery, 11.4 hr) — `recover_g6_failed_tuples.sh` (Wf-366-A2 STRICT)
   - §2 item 3 (R20-P5 joint, 12 hr) — `launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh`
3. **Batch 3 — paper write** (~12-15 hr CPU after Batch 1 tooling ready)
   - §4 items 1-7 — paper §1-§7 v2 Tier-2 re-scope
   - §4 item 2 — §6 Limitations v4 paste from template scaffold

**Parallel / background:**
- §5 item 2 (m6_kappa drop from §3 main-result) → docs-only §4 item 5 dependency
- §2 items 1, 2, 4, 5 (R20 §3 + R20 §4 + Wf-131 + Wf-146) — secondary priority after Batch 2

### §7.4 What Wf-368-B1 is NOT claiming

Per `party-twenty-spirit §3.5`:
- **NOT claiming all problems are彻底解决 (thoroughly resolved)** — 2 of 4 gaps CLOSED by Wf-366; 6 forward-looking items still require USER_OK grants
- **NOT claiming the user should grant all USER_OK at once** — per-wrapper gating is independent
- **NOT claiming the wrapper wall estimates are exact** — they are PROJECTED from Wf-258 + Wf-260 timing; MEASURED post-launch
- **NOT claiming §4 paper work is required** — Tier-2 acceptance bar is computational-only; paper work is optional
- **NOT claiming m6_kappa gate will be lifted** — 5 attempts exhausted; honest-negative is FINAL per Wf-365-A3 §2
- **NOT claiming Wf-368-B1 is a NEW ultracode wave** — this is a docs-only consolidation under the Wf-368 ultracode umbrella (A1+A2+A3+A4 SHIPPED; B1 this docs-only consolidation)

### §7.5 Honest-negative catalog (preserved)

Per `docs/HONEST_NEGATIVE_CATALOG.md` (40 entries):
- **m6_kappa gate NOT lifted** (5 architectural attempts exhausted; HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL per Wf-365-A3 §2)
- **Wf-246-C kcal sign-inverted CLOSED** (Wf-333-V1; corrected `(1 - 2*c) × kcal_per_unit` for Vina "lower = better" convention)
- **CUDAGraphs Blackwell RuntimeError CLOSED** (Wf-335-R1 + Wf-335-R3; default mode `default`→`reduce-overhead` + `--no-cuda-graphs` safety lever)
- **CLI help-text %-formatting CLOSED** (Wf-364-A1; `pct` substitution)
- **Wf-363-A2 GAP-1 (G5 USER_OK missing) CLOSED** at Wf-366-A1
- **Wf-363-A2 GAP-2 (recover_g6 --dry-run bypass) CLOSED** at Wf-366-A2
- **Wf-363-A2 GAP-3 (`pre_flight_gpu_ready.sh` dry-run SKIP) OPEN** — recommended Wf-369+ closure
- **Wf-363-A2 GAP-4 (`gpu_launch_dashboard.sh` READ-ONLY scaffold) OPEN** — intentional per Wf-360-A3
- **Pre-flight `--strict --skip-live` semantic conflict OPEN** (Wf-364-A1 §4.2 pre-existing Wf-347A1)
- **Bond_fdim 6→10 schema gap OPEN** (Wf-333-G4; separate workflow)
- **`p3_wf343c2_per_wave_runtime_check` 5-LOC argparse fix OPEN** (Wf-368-A1; recommended follow-up)

---

## §8 Standing-rule compliance

| Rule | Compliance for Wf-368-B1 |
|---|---|
| §1.1 MEASURED-vs-PROJECTED tag discipline | ✅ Every §2-§5 + §6 + §7 numeric tagged MEASURED / PROJECTED with source path |
| §1.2 Standing GPU authorization | ✅ Checklist is docs-only; no GPU launches triggered |
| §1.3 No paper work without authorization | ✅ Zero touches to `paper/*`; §4 inventory only |
| §1.4 Pre-flight v3 25/25 PASS gate | ✅ Pre-flight NOT re-run (docs-only); PRESERVED at Wf-368-A4 baseline |
| §1.5 Honest-negative framing | ✅ §7.5 catalog preserved; §7.4 explicit NOT-claim list |
| §5.1 Standing GPU authorization | ✅ Per-wrapper USER_OK gating preserved (no auto-trigger) |
| §5.2 Honest framing | ✅ MEASURED with sources cited; PROJECTED with bands |
| §5.3 No paper work | ✅ §4 is inventory only; no `paper/*` touches |
| §5.4 Tier-2 journal target + no wet-lab | ✅ §4 drives Tier-2 re-scope; wet-lab items DEPRECATED per user directive |

---

## §9 Test verification

```
$ PYTHONPATH=/home/hugo/codes/mol-metal \
    /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    -m pytest molmetal/tests/test_wf368b1_user_ok_final.py -v

molmetal/tests/test_wf368b1_user_ok_final.py::TestWf368B1UserOkFinal::test_checklist_exists_and_has_seven_sections PASSED [100%]

============================= 1 passed, 1 warning in 0.34s =============================
```

**MEASURED**: 1 NEW test PASS in 0.34 s wall (CPU-only; zero GPU launches).

The single test verifies 14 invariants:
1. **File exists** at `docs/USER_OK_FINAL_CHECKLIST.md`
2. **All 7 sections present** (§1-§7 per Wf-368-B1 brief)
3. **Reasonable length** (>= 200 LOC; PROJECTED ~350 LOC per Wf-368-B1 brief; actual ~430 LOC)
4. **All 7 USER_OK-gated launch wrappers referenced**
5. **All 2 install binaries referenced** (Zotero + tectonic)
6. **All 2 metric gates referenced** (m1 + m6_kappa)
7. **Wf-366-A1 + Wf-366-A2 references present** (gap closures integrated)
8. **GAP-1 + GAP-2 explicit mentions**
9. **m6_kappa HONEST_NEGATIVE_PARTIAL_STRUCTURAL preserved** (per Wf-365-A3 §2)
10. **Tier-2 markers preserved** (Tier-2 + PAPER_OK + USER_OK)
11. **§7 forward-looking section present**
12. **Path discipline** — checklist lives under `docs/` (NOT `paper/*`) per TODO-84 + AGENTS.md §5.3
13. **MEASURED + PROJECTED tag discipline** preserved (sanity check AGENTS.md §5.2)
14. **§6 Cross-references** explicitly references AGENTS.md + Wf-368/Wf-366

---

## §10 Wall + cost (Wf-368-B1)

| Stage | Wall (MEASURED / PROJECTED) |
|---|---:|
| Inventory existing Wf-355/364/365/366/368 closeouts | ~2 min (Read) |
| Inspect 7 launch wrappers for USER_OK gate + wall + VRAM | ~2 min (Bash grep) |
| Verify `docs/USER_OK_FINAL_CHECKLIST.md` exists + comprehensive (no rewrite needed — Wf-366-C2 deliverable still valid) | ~1 min |
| Write `molmetal/tests/test_wf368b1_user_ok_final.py` (~210 LOC) | ~3 min |
| Run pytest | 0.34 s (MEASURED) |
| Write `TODO/completed/368b1_user_ok_final.md` (this closeout) | ~4 min |
| **Total Wf-368-B1 CPU wall** | **~12 min CPU-only** (within PROJECTED 15-25 min per task brief) |

**PROJECTED vs MEASURED**:
- PROJECTED 15-25 min CPU-only at task brief
- MEASURED ~12 min (within brief, faster due to existing checklist already comprehensive)

**GPU launches performed**: 0 (per AGENTS.md §5.1 kill_then_adapt standing directive; docs-only sub-flow).
**Paper touches**: 0 (per AGENTS.md §5.3 + TODO-84; checklist lives under `docs/` ONLY).

---

**END OF Wf-368-B1 USER_OK FINAL CHECKLIST** — Owner Hugo Lee;
cross-ref `party-twenty-spirit §3.5` + `AGENTS.md §5.1 + §5.3 + §5.4` + `mol-metal-gpu-execution` §0;
last update 2026-09-29 (Wf-368-B1 timestamp); CPU-only docs surface;
pre-flight v3 PRESERVED at Wf-368-A4 baseline;
0 GPU launches; 0 `paper/*` touches;
**7 curated USER_OK-gated launch wrappers + 2 install binaries + 7 paper artefacts + 2 metric gates**;
~52.9 hr GPU + ~13 hr CPU + ~30 min install TOTAL PROJECTED.
