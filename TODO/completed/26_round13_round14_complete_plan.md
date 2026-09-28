# TODO-26 — Round-13 + Round-14 完整 ship 计划

**Status:** ⚙️ **R12 COMPLETE + R15 STRUCTURAL SHIP + R16 = YuelBond retrain + Round-13 100x3 paper-grade + Deflex integration; 12-week roadmap updated to R12/R13/R15/R16/Deflex-arXiv**
**Priority:** medium (Round-13 deferred to R16; Round-15 structural; Round-16 = paper-grade)
**Owner:** (unset)
**Depends on:** 4 in-flight workflows (`wbw9a59g3` CFM frontier + `wrd5dbewn` pocket-invariance + `wyyy283ck` parallel fixes + `wtdrwkb6z` Deflex integration)
**Created:** 2026-09-15
**Last updated:** 2026-09-16 (R15 ship summary added; R16 = YuelBond retrain + Round-13 100x3 paper-grade + Deflex integration; 12-week roadmap updated)

## ⚠️ Update 2026-09-15 — what we now know

### Round-12 deliverable — **COMPLETE**
- ✅ 4-fix bundle (rule symmetry + decoder rework + scaffold-aware gate + partner tiles)
- ✅ n_distinct 1→20, div_tan 0→0.1065, div_homo 0→0.0749 on 30 cells
- ✅ 147 cells DESIGN→MEASURED in paper §4 (per `we5qbl16b`)
- ✅ paper §6 expanded 8→12 caveats
- ✅ paper §3.4 honest-framing paragraph added
- ✅ arXiv unblocked (paper/main.pdf 69 pages / 5.17 MB / 0 unresolved)

### Round-13 100×3 — **HONEST NEGATIVE on novel pockets**
- ❌ n_distinct=1 collapse resurfaces on test_010..test_019 (3-layer attractor: chemistry BROKEN, MCTS cache + reward prior REMAIN)
- ❌ PB 30/30 search-bound at n_sim=100 (sub-fix at n_sim=1000 in flight via `wrd5dbewn`)
- ✅ GPU recovery confirmed (cuda_available=True, 2 devices)
- ✅ Per-pivot insight: 32 mols training data is fundamental bottleneck (CFM path)

### Round-14 lit-grounded — **PATH A + D2/D4 SHIPPED; M3+CFM in flight**
- ✅ Path A 4-fix bundle (above)
- ✅ pt_click_compat 5×5 compat matrix (scaffold-aware gate)
- ✅ F2(a) MetalLigandExchange SMARTS (per `wyyy283ck` A)
- ⚙️ CFM frontier research (`wbw9a59g3`) searching 2025-2026 SOTA + 3 fixes in implementation
- ⚙️ Pocket-invariance combined (`wrd5dbewn`) 3 sub-fixes in implementation
- ⚙️ Deflex integration (`wtdrwkb6z`) F5 adapter + PocketMacroInference
- ⚙️ Parallel discovered fixes (`wyyy283ck`) Skeleton CA2 + λ Combinators cold-swap

### What's MEASURED vs PROJECTED at paper submission

| Item | MEASURED | PROJECTED | Status |
|---|---|---|---|
| n_distinct | 20 (PathA-10x3) | — | ✅ |
| div_tanimoto | 0.1065 (PathA-10x3) | — | ✅ |
| div_homotype | 0.0749 (PathA-10x3) | — | ✅ |
| metal_compliance | 0.0 (with seed) | 1.0 F2(a) | ⚠️ (F2(a) in flight) |
| Vina real | -6.929 kcal/mol (1-pocket) | — | ✅ (smoke only) |
| PB pass rate | 1.000 (1-pocket smoke) | 60-80% (production) | ⚠️ (n_sim=1000 in flight) |
| PB MMFF94 | 22/26 (click_tile) | — | ✅ |
| REINVENT4 wire | r=0.6763 vs proxy | — | ✅ |
| F5 formula | R=2.58-2.51·sa_norm | — | ✅ |
| Pocket macro skel | 87.9% train acc | — | ✅ |
| CFM decode_ratio | 0/64 (h=128 5K) | >0 (frontier fixes) | ❌ (gated on frontier research) |

## **核心痛点诊断（per user's question）**

> **不是数据不好看，也不是测试协议无法对齐**——是 **测试规模不够大**：
> - 所有 pilot 都跑 n_sim=100 + 5-10 pockets max
> - TargetDiff 标准是 **100 × 3 seeds = 300 evaluations**
> - **协议对齐**（TargetDiff standard 我们都用）：CrossDocked2020 + 30% seq-id + QVina + PB 1.0 + 18 metrics panel
> - **数据 lift 方法 lit-grounded + 对的**（9 P0 + Path B + scaffold gate + rule symmetry + 9 NEW research gaps as honest contributions）
> - **只是 budget 没到位**——n_sim=1000 + 100 pockets + 3 seeds 才能 paper-grade

## **今日 ship 现状**

### **Code fixes ship**

| Fix | Workflow | Status |
|---|---|---|
| F1 + F2 + F3 + F4 Lambda fixes | WF-Lambda-Fix-FullPath-v2 | ✅ ship |
| F1 soft tiered metal prior (h=16→128) | WF-Vina-Lift-Phase23 | ✅ ship |
| F2 drop tanh + learnable vel_scale | WF-Vina-Lift-Phase23 | ✅ ship |
| F3 PCGrad multi-task loss | WF-Vina-Lift-Phase23 | ✅ ship |
| PAC-Bayes generalization bound | WF-Vina-Lift-Phase23 | ✅ ship |
| Path B decoder rework | WF-CFM-Path-B-Decoder-Rework | ✅ ship |
| Lambda rule symmetry fix | WF-Lambda-Rule-Symmetry-Fix | ✅ ship |
| Metal-seed = `[Pt]C#C` | WF-Lambda-Fix-FullPath-v2 | ✅ ship |
| Scaffold-aware click gate (5×5 compat) | WF-MCTS-Chemistry-Research | ✅ ship |
| Diversity_bonus channel | WF-Lambda-Fix-FullPath-v2 | ✅ ship |
| metal_compliance_truthful | WF-Lambda-Fix-FullPath-v2 | ✅ ship |
| 8 partner tiles (3 azides + 3 boronic + 2 bromides) | WF-Partner-Tiles-PathA | ✅ ship |
| mmff94s_relax_pose | WF-PB-MMFF94-Relax | ✅ ship |
| --pb-mode dock (26 checks) | WF-PB-Dock-Mode-Wire | ✅ ship |
| --sa-weight 0.3 | WF-SA-Penalty-Guidance | ✅ ship |
| --keep-high-sa-tiles + 10-tile blacklist | WF-SA-Fragment-Pool-Optimize | ✅ ship |
| --decoder-rework flag | WF-CFM-Path-B-GPU-Retrain | ✅ ship |
| **18 metrics in r4_lambda_only_run.py** | WF-P0-Metrics-Add + WF-CoM-Shift-Metric + WF-Rigid-RMSD-Metric | ✅ ship |

### **Research ship**

| Survey | Status | Citations |
|---|---|---|
| Lambda Internal Review | ✅ done | 4 root causes |
| MCTS Chemistry Research | ✅ done | 22 papers + 5 click verdicts |
| Triton Arch Research | ✅ done | 13 kernels + FlashAttention + mHC |
| Triton Kernel Audit | ✅ done | 3 priority wirings |
| Lit Survey v2 | ✅ done | **65 papers + 20 theorems borrowed** |

### **Paper ship**

| Section | Status |
|---|---|
| §1+§2+§6+§7 (WF-Paper-1) | ✅ ship |
| §3 4 sub-sections (WF-Lambda-3) | ✅ ship |
| §4 (WF-Paper-Section-04) | ✅ ship |
| §5 (WF-Paper-Section-05) | ✅ ship |
| 4 figures (WF-Paper-2 + Fig 4 homotype scatter) | ✅ ship |
| main.tex + 73 bib entries + 449 supplementary | ✅ ship |
| Lambda-Only-Paper-Path restructure | ✅ ship |
| **PDF compile** | ✅ 71 pages, 5.15 MB, 0 fatal errors |

## **TODO file 现状**

| File | Status |
|---|---|
| TODO/pending/13_top_journal_pilot_r12.md | Round-12 SHIPPED |
| TODO/pending/14_full_100pocket_paper_r13.md | Round-13 scope ready |
| TODO/pending/21_lambda_model_coupling.md | Lambda × CFM deferred + iGPU fallback + GPU recovery + 10000-step verdict |
| TODO/pending/22_data_gap_alignment_plan.md | 25 metrics gap plan |
| TODO/pending/23_weak_to_strong_plan.md | 5 weak + 5 NEW research gaps |
| TODO/pending/24_cfm_architecture_redo_plan.md | 4 root causes + 5 P0 + 4 P1 + 3 triton kernels + Phase 4 fusion |
| TODO/pending/25_round14_lit_grounded_plan.md | Lit-grounded Round-14 + 5 NEW gaps |

### **3 个 in-flight workflow 现状**

| Task ID | Workflow | Expected | Wall |
|---|---|---|---|
| `wo21fr1ug` / #671 | **WF-Round13-100x3-Sweep**（100 pockets × 3 seeds at n_sim=1000）| paper-grade diversity/PB/SA/Vina | ~50 min CPU Lambda + ~6h GPU CFM |
| `wgvpkwmvb` / #667 | **WF-Round12-Lambda-PathA-10x3**（10 pockets × 3 seeds at n_sim=1000 with Path B）| diversity_tanimoto 0.10+ verified on 30 cells | ~5 min CPU |
| `wgnp3ne6j` / #647（已完成 phase 1）| **WF-CFM-Path-B-GPU-Retrain**（10000-step h=64 + decoder_rework）| Vina_mean -2 → -7 verified | 已完成 phase 1，phase 2 done |

## **3 个并行路径 phase**

### **Phase A：WF-Round13-100x3-Sweep**（in flight）

- **Lambda path**（CPU，~50 min）：`r4_lambda_only_run.py --pockets 100 --seeds 42 0 1234 --n-simulations 1000 --metal-seed cisplatin --click-rules auto-pt-strict`
- **CFM path**（GPU，~6h）：`r10_cfg_real_crossdocked.py --decoder-rework --pb-relax-mmff94`
- **PB path**（CPU，~50 min）：`r4_c_full_sweep.py --pb-mode dock --pb-relax-mmff94`
- **Expected**：300 evaluations per metric per pocket

### **Phase B：WF-Round12-Lambda-PathA-10x3**（in flight）

- Lambda 10×3 with Path B（rule symmetry + decoder + scaffold-aware + partner tiles）
- **Expected**：mean n_distinct ≥ 2 + diversity_tanimoto ≥ 0.10 + metal_compliance ≥ 0.667 + click_rules_fired > 0

### **Phase C：WF-CFM-PathB-GPU-Retrain**（已完成 phase 1）

- 10000-step + h=64 + decoder_rework + tmqm + pcgrad + joint_train
- **Result**：decode_ratio=0/192 → **wrap-ordering fix needed**（next workflow）

## **Round-13 + Round-14 时间表（updated）**

| Week | Action | Status |
|---|---|---|
| **W38 (this week)** | 3 个 in-flight workflow 完成 + paper §4 整合 lit-grounded 数据 | ⚙️ running |
| **W39** | WF-Round13-100x3-Sweep 完整 300 evaluations + paper §4.2/§4.3/§4.6 真数据 | pending W38 |
| **W40** | CFM decoder wrap-ordering fix（如果 Vina mean null）+ Round-14 实证 | pending W39 |
| **W41** | paper §6 + §7 update（5 NEW gaps as honest contributions）+ arXiv submission | pending W40 |

## **5 NEW research gaps as honest paper contributions**（per Lit-Survey-v2）

| Gap | Status | In paper |
|---|---|---|
| M1: differentiable-Vina + FM joint loss convergence | Open | §6 future-work |
| M2: FM with valence constraint | Open | §6 future-work |
| M3: joint FM vector-field + bond-head multi-task loss | Open | §6 future-work |
| M4: MCTS-over-typed-term βNF search space | Open | §6 future-work |
| M5: per-pocket learnability guarantee for EGNN-style generators | Open | §6 future-work |
| **NEW: rule-dispatch asymmetry** | **✅ SOLVED** by WF-Lambda-Rule-Symmetry-Fix | §3 + §4 acknowledged |

## **Round-14 决策矩阵**（per TODO-25 3 paths）

| Path | Resource | Expected lift | Risk |
|---|---|---|---|
| **Path A** ship now | 0 GPU | 0 | 0（current 71-page PDF）|
| **Path B** lit-grounded | 12-24h GPU + 24h CPU | Vina -2 → -7, diversity 0 → 0.50+, PB 0% → 60-80% | LOW（borrow existing theorems）|
| **Path C** arch rework | 12-24h GPU + 14h CPU + Phase 4 fusion | same + EquiformerV2 + mHC | MEDIUM |

**Decision**：选择 **Path B**（lit-grounded + 当前所有 fix ship + 3 个 in-flight workflow 验证 + GPU budget no limit per user）

## **决策矩阵 vs paper impact**

| Workflow result | Paper impact | Honest framing |
|---|---|---|
| **3 个 in-flight 都 lift** | §4.6 真数据 populated + §6 从 8 caveats → 6 + §7 从 8 future work → 4 | ✅ ship to arXiv |
| **部分 lift（diversity yes, Vina no）** | §4.6 混合 + §6 仍 8 caveats | ✅ ship with Path C deferred |
| **都 NO_LIFT** | §4.6 仍空白 + §6 仍 8 caveats | ⚠️ paper §7 future work = 5 NEW gaps 实证 + accept Vina lift 为 open problem |

## **TODO/pending/26（this file）的更新协议**

Append-only. 任何新 fix ship / new in-flight result / decision / TODO update 都需要写 dated section。

---

## Phase-3 ship (2026-09-15) — comprehensive master plan consolidation

**Source:** `molmetal/reports/wf_round14_plan/phase{1_lit,2_synthesis,3_writes}.md`
**Companion to:** TODO-25 (lit-grounded Round-14 plan) + TODO-29 (F2(a) +
Round-13 retry).

### Why this matters here

TODO-26 (this file) is the canonical comprehensive ship plan that
consolidates the Round-13 retry strategy + Round-14 lit-grounded plan +
paper submission timeline + 6 in-flight workflows tracking. This Phase-3
update adds:

1. **Round-13 retry decision tree** (F2(a) + 5 algo fixes + warm-start)
2. **Round-14 strategy** (per TODO-25 P0/P1/P2 priorities)
3. **Paper submission timeline** (W41 arXiv + W46-W49 journal)
4. **6 in-flight workflows tracking** (Round-13 sweep + CFM PathB + Paper
   Compile + Round-12 PathA + PB Real-Dock + Lambda MCTS Coords)
5. **Honest framing of partial lifts + open gaps**

### §A. Round-13 retry strategy

#### Honest framing of Round-13 verdict
Per `wf_round13_100x3/final.md`:
- **Path A killed:** n_distinct=1 on test_010..test_019 (singleton collapse
  resurfaces despite PathA-10x3 lift on test_000..test_009)
- **Path B 30/30 search-bound:** MCTS early-stopped at 51 sims; 0 new mols
  (2 no_candidates + 3 seed_only per pocket)
- **Path C BLOCKED:** GPU torch.cuda.is_available()=False (HSA init fails)
- **paper_grade_data_ready=FALSE:** integration REFUSED to silently promote
  DESIGN→MEASURED

#### Round-13 retry — 3 components
- **F2(a) MetalLigandExchange + AquaExchange SMARTS** (per TODO-29) —
  breaks the `_unreactive_states` permanent cache at SMARTS level. Ships
  per `molmetal_lam/reactions/beta_reductions.py`.
- **5 algo fixes** (per `wf_algo_tune/final.md` PARTIAL_LIFT) — soft tiered
  metal prior + diversity_bonus channel + metal_compliance split
  (including_seed/non_seed) + scaffold-aware click selection +
  --allow-incompatible-click flag. All already ship per
  `wf_lambda_fix_full_path_v2/fix2_scaffold_aware.md`.
- **Warm-start integration** — Tasks J (per-pocket warm-start embedding) + D
  (sub-pocket fingerprint diversity metric) per Phase-3 agent 4. Already
  ship per INDEX.md §Lambda Core Features.

#### Round-13 retry decision tree
```
Round-13 retry (T2+T3+T4 from TODO-25)
├── Q1: Does T2 (F2(a)) + T3 (Tasks J+D) lift on novel pockets?
│   ├── YES (n_distinct>1 on ≥80% test_010..test_019) → ship §4 Table 1 MEASURED
│   └── NO → T8 (n_sim=5000 + virtual loss)
│       ├── YES → ship at higher n_sim
│       └── NO → Path (c) hybrid per TODO-21 §3
└── Q2: Does the 5 algo fixes carry over to Round-14?
    ├── YES (no regression on Round-12 cohort) → ship to Round-14
    └── NO (regression on diversity/QED/SA) → revert 5 fixes
```

#### Round-13 retry success criteria
- 100×3 = 300 cells: ≥90% MEASURED (was 0% per Round-13 partial)
- §4 Table 1 MEASURED: n_distinct > 1 on ≥80% test_010..test_019
- §4.6 columns: PB ≥60% / Vina ≥60% / Diversity ≥60% MEASURED
- Honest framing: pocket-invariance gap closed OR documented in §6.1

#### Round-13 retry risk
- **HIGH risk:** F2(a) + Tasks J+D may not lift singleton on all 100 pockets.
- **Mitigation:** T8 (n_sim=5000 + virtual loss) as next step. T10 (Alt B
  CFM ETKDG init) as last resort.
- **Honest framing:** if Round-13 retry fails, Round-14 carries over with
  Path (c) hybrid per TODO-21 §3 + §3.5/§6.1/§7.1 honest framing preserved.

### §B. Round-14 strategy (per TODO-25)

#### Round-14 priorities (per TODO-25 §2 Top-10)
- **P0 (this week W38, ship-blocking for arXiv):**
  1. T1 — Alt A pdflatex diagnose+fix (2-5.5h CPU, restore `paper/main.pdf`)
  2. T2 + T3 — F2(a) + Tasks J+D (12-14h CPU, break singleton attractor)
  3. T6 + T7 — §4/§5/§6/§7 paper content updates (4-6h CPU)
- **P1 (W39-W40, ship-blocking for paper-grade data):**
  4. T4 — Round-13 re-run 100×3 with T2+T3 enabled (~50min CPU single-thread)
  5. T5 — Alt A CFM wrap reorder (4h CPU, fix decode_ratio=0/192)
  6. T6 — §4.2 + §4.3 + §4.6 DESIGN → MEASURED promotion (1h CPU)
- **P2 (W41, ship-blocking for arXiv submission):**
  7. T7 — §6 reduce 9 → 7 caveats + §7 reduce 5 → 3 NEW gaps (1h CPU)
  8. T9 — arXiv submission prep (2h CPU)
  9. T8 + T10 — Round-14 carry-over (Alt A MCTS budget + Alt B CFM ETKDG
     init if needed)

#### Round-14 success criteria (per TODO-25 §7.2)
- `paper/main.pdf` exists ≥56 pages, 0 unresolved refs, 0 fatal errors
- §4 Table 1: 100×3 cells ≥ 90% MEASURED
- §4.6 PB column ≥ 60% MEASURED
- §4.6 Vina column ≥ 60% MEASURED
- §4.6 Diversity column ≥ 60% MEASURED
- §6 caveats: 9 → 7
- §7 future work: 5 NEW gaps → 3
- arXiv preprint ready (cover letter + supplementary.tex + main.pdf)

### §C. Paper submission timeline (W41-W49)

#### W41 (arXiv submission, primary ship target)
- **Mon-Tue:** T9 (arXiv submission prep: cover letter + supplementary.tex +
  main.pdf final review)
- **Wed:** **arXiv SUBMIT** (Q1-2027 target)
- **Thu-Fri:** Round-14 carry-over (T8/T10 if needed)

#### W42-W45 (Round-15, GPU retrain + paper-grade lift)
- **W42-W43:** GPU retrain 10000-step + h=128 + tmQM init + Alt A+B fixes
- **W44:** Round-14 100×3 sweep at n_sim=1000 + F2(a) + Tasks J+D
- **W45:** paper §4.6 + §5 + §6.1 v2 update + journal submission prep

#### W46 (journal target selection, per `decisions.md D11`)
- **W46:** Journal selection:
  - (a) **Digital Discovery (RSC)** — strong fit for lit-grounded + open-source
  - (b) **J. Chem. Inf. Model.** — fit for SBDD benchmark
  - (c) **Nat. Comput. Sci.** — broader audience, higher bar
- **Recommended:** (a) Digital Discovery (RSC) — primary fit for open-source
  SBDD tool + lit-grounded methodology + cross-disciplinary (chem + ML)
  audience. Per `wf_3_citeonly_sota.md` cite-only path already ship.

#### W47-W48 (format + cover letter + supplementary)
- **W47:** Format per journal + apply bibstyle + cross-reference audit
- **W48:** Cover letter + supplementary.tex + ORCID + author affiliations +
  conflict-of-interest statement

#### W49 (JOURNAL SUBMIT)
- **W49 Mon-Tue:** Final pre-submission review (paper + figures + bib +
  supplementary)
- **W49 Wed:** **JOURNAL SUBMIT** (target Q1-2027 Q2 / Q2-2027)
- **W49 Thu-Fri:** Acknowledgement receipt + Round-15 carry-over

### §D. 6 in-flight workflows tracking

#### WF-Round13-100x3-Sweep (tasks #671, #672)
- **Phase:** Phase 2 in_progress (sweep NOT yet executed)
- **Status:** PARTIAL — paper-grade 300 cells NOT achieved (2026-09-15)
- **Depends on:** T2 (F2(a)) + T3 (Tasks J+D) ship
- **Round-13 retry:** Round-13 re-run 100×3 with T2+T3 enabled (per TODO-25 T4)
- **Honest framing:** integration REFUSED to silently promote DESIGN→MEASURED

#### WF-CFM-PathB-GPU-Retrain (tasks #647, Phase 1 done)
- **Phase:** Phase 1 done; Phase 2 wait-for-budget
- **Status:** decode_ratio=0/192 on real CrossDocked output
- **Depends on:** T5 (Alt A CFM wrap reorder) ship
- **Round-14 retry:** wrap reorder (4h CPU) + ETKDG init (12-16h GPU) if
  wrap reorder fails
- **Honest framing:** 5 P0 fixes ship but wrap-ordering issue (hypothesis a
  per `wf_path_b_gpu_retrain/final.md §5.1`) is dominant cause

#### WF-Paper-Compile-Verify (tasks #460, #475)
- **Phase:** in_progress
- **Status:** `paper/main.pdf` MISSING despite "completed" tasks #456/460/475
- **Depends on:** T1 (Alt A pdflatex diagnose+fix) ship
- **Round-14 retry:** Alt A (diagnose + fix), Alt B (snapshot section_03 +
  incremental patches), Alt C (multi-PDF bundle)
- **Honest framing:** per TODO-27, root cause is in main.tex / refs.bib /
  figure paths / package options. Each can be diagnosed by running
  pdflatex verbose.

#### WF-Round12-Lambda-PathA-10x3 (task #667 — ✅ DONE)
- **Phase:** COMPLETE 2026-09-15
- **Status:** diversity lift verified on 30 cells (test_000..test_009)
- **Round-14 carry-over:** n_distinct 1→20, div_tan 0→0.1065 (verified)
- **Honest framing:** pocket-invariance on test_010..test_019 NOT verified;
  this is the gap that Round-13 retry + T2+T3 target

#### WF-PB-Pass-Real-Dock-Integrate (tasks #545, #546, #551, #552, #553)
- **Phase:** in_progress
- **Status:** 1-pocket smoke = pb_pass_rate=1.000; 30-cell smoke = 0/30
  PB-eligible (search-bound at n_simulations=100 with strict gates)
- **Depends on:** T8 (Alt A MCTS budget lift) for non-degenerate production
  PB panel
- **Honest framing:** production 100×3 PB cells still DESIGN (Round-13
  sweep pending); 10×3 PB smoke before pass-rate claim

#### WF-Lambda-MCTS-Coords-Fix (task #650)
- **Phase:** COMPLETE
- **Status:** ROOT fix for both NO_LIFT workflows (5 steps complete)
- **Round-14 carry-over:** Round-12 Lambda 10×3 verified with co-embedded
  3D coords; PB 30×3 re-verified
- **Honest framing:** coords flow into singleton collapse analysis; the
  fix does NOT address chemistry-level singleton attractor (T2 F2(a) does)

### §E. Honest framing of partial lifts + open gaps

#### Partial lifts (MEASURED, not PROMISED)
- **Round-12 Lambda PathA-10x3:** n_distinct 1→20, div_tan 0→0.1065 on
  test_000..test_009 ONLY (pocket-invariance gap on novel pockets NOT
  measured)
- **Round-12 Lambda Metal-Pilot:** metal_compliance_rate +1.0 (0.0→1.0)
  on 5×1 with --metal-seed cisplatin + --click-rules all-5
- **Round-12 Lambda Mini-Pilot (5×1):** 5/5 cells valid=1.0, synth=1.0,
  uniq=1.0, metal=0.0 (no --metal-seed)
- **Round-13 honest-negative:** 100×3 paper-grade NOT achieved;
  Path A killed + Path B 30/30 search-bound + Path C BLOCKED by GPU
- **CFM Path B 5 P0 fixes:** F1-F5 ship (BondAwareDecoder.decode wired +
  BondOrderHead in_dim=9+2*hidden_dim + vocab_mask in CE + UserWarning on
  hidden_dim<64 + bonds=zeros placeholder removed); 7/7 tests pass
- **GPU recovery 2026-09-15:** cuda_available=True device_count=2;
  5000-step CFM retrain 3 seeds COMPLETED 125s; decode_ratio=0/192
  (FAILURE per spec gate, path (c) λ-only stays as default)
- **PAC-Bayes bound implementation:** ships per `wf_vina_lift_phase23/pac_bayes.md`

#### Open gaps (NOT closed)
- **Pocket-invariance:** Round-13 retry + T2+T3 target. If fails → T8
  (n_sim=5000 + virtual loss) → T10 (CFM + Lambda hybrid per TODO-21 §3).
- **CFM Path B decode_ratio=0:** T5 (wrap reorder) targets. If fails →
  T10 (ETKDG init) as Round-14 follow-up.
- **paper/main.pdf missing:** T1 (Alt A) targets. If fails → Alt B (snapshot
  + incremental patches) → Alt C (multi-PDF bundle).
- **SOTA-scale PB pass rate:** WF-PB-Pass-Real-Dock-Integrate in flight.
  30-cell smoke = 0/30 PB-eligible (search-bound). T8 (n_sim=5000) for
  non-degenerate production PB panel.
- **Lambda × CFM coupling:** TODO-21 (Lambda × CFM deferred) — Round-15+
  carry-over. Decision tree per §3.4 of TODO-25.
- **pIC50 oracle active learning:** TODO-18 + WF-Extra-1; Tosh 2021 RL-AL
  4-64× oracle cost reduction next round.

#### Honest framing points (preserved verbatim)
1. **No theoretical derivation is re-attempted in this plan.** Each fix
   path cites the lit anchor in `phase1_lit.md §1-§6` + Lit-Survey-v2
   65 papers + 20 theorems.
2. **All claimed lifts are PROJECTIONS, not measurements.** Per TODO-25
   §"5 weak metrics → existing lit + plan", the (a)+(b)+(c)+(d) projected
   lift for Vina (-2 → -7) is a *lit-grounded projection*, not a
   measurement.
3. **Round-13 partial = honest-negative, not failure.** Per
   `wf_round13_100x3/final.md §7`, the integration refused to fabricate
   measurements.
4. **SBDD-with-metal-centers has no SOTA baseline.** Our metal-seeded
   Lambda pilot is one of the first formal ablations.
5. **5 NEW research gaps of TODO-25 §5 remain open** and traceable in
   §3.5 → §4.11 → §6.1 → §7.1 of paper.
6. **GPU recovery is fragile.** Per `wf_gpu_recovery_now/final.md`,
   GPU recovers after cold power cycle but any broken KFD node = fatal
   HSA init. Watchdog required for any GPU run.

### §F. Resource budget (uv-managed Python 3.12, ROCm 7.2, triton-rocm 3.8.0, RX 7800 XT gfx1101 wave64)

#### Wall time estimates

| Action | Resource | Wall |
|---|---|---|
| T1 Alt A pdflatex diagnose+fix | CPU | 2-5.5h |
| T2 F2(a) MetalLigandExchange | CPU | 6h |
| T3 Tasks J+D novel-pocket smoke | CPU | 6-8h |
| T4 Round-13 re-run 100×3 | CPU (Lambda) | ~50min single-thread |
| T5 Alt A CFM wrap reorder | CPU | 4h |
| T6 §4 promotion | CPU | 1h |
| T7 §5 + §6 + §7 update | CPU | 1-2h |
| T8 Alt A MCTS budget lift | CPU | 2-4h |
| T10 Alt B CFM ETKDG init | GPU + CPU | 12-16h |
| Round-14 100×3 sweep | CPU (Lambda) | ~50min single-thread |
| Round-15 GPU retrain 10000-step | GPU | 12-24h |
| arXiv submission prep | CPU | 2h |
| Journal submission prep | CPU | 8-16h |

#### GPU budget (Round-15 W42-W43)
- 10000-step CFM retrain 3 seeds: 12-24h GPU
- Alt B CFM ETKDG init (if needed): 12-16h GPU
- Total Round-15 GPU: 12-40h

#### Risk on this host
- **GPU is fragile** per `wf_gpu_recovery_now/final.md`. Cold power cycle
  required if HSA init fails. iGPU (Radeon 780M gfx1100) HW-healthy but
  PyTorch/HSA cannot reach it because ROCm 7.2 HSA runtime treats ANY
  broken KFD node as FATAL hsa_init failure.
- **Watchdog:** monitor + auto-recover per `wf_gpu_auto_recover/final.md`
- **DEFAULT_DEVICE=cpu** on this host until sudo kernel-dGPU-blacklist +
  reboot (out of scope per `wf_igpu_switch/probe.md`).

### §G. Open questions for user (decision points)

#### D11 — journal target (W46)
- (a) **Digital Discovery (RSC)** — RECOMMENDED (lit-grounded + open-source fit)
- (b) **J. Chem. Inf. Model.** — backup
- (c) **Nat. Comput. Sci.** — broader audience, higher bar

#### Lambda × CFM coupling timing (TODO-21 deferred)
- Per `TODO-21` decision: 重训可接受但不是现在；后面必做"Lambda 算法和模型侧相结合"
- A line CFM paused; B line + Round-12 pilot complete
- Decision point: start 5-10d joint training after B line + Round-13 retry complete

#### Round-13 retry decision tree on failure
- If T2+T3+T4 fails: ship Path (c) hybrid per TODO-21 §3 + §3.5/§6.1/§7.1
  honest framing preserved
- If T5 fails: ship Path (b) Lambda-only canonical per TODO-21 deferral
- If T1 fails: ship multi-PDF bundle per §2.3 Alt C

### Cross-references

- `TODO-25` (`25_round14_lit_grounded_plan.md`) — Round-14 lit-grounded plan
- `TODO-27` (`27_paper_main_pdf_repair.md`) — paper/main.pdf repair
- `TODO-28` (`28_round12_honest_negative_reframe.md`) — R12 PARTIAL_LIFT framing
- `TODO-29` (`29_f2a_round13_retry.md`) — F2(a) + Round-13 retry
- `TODO-21` (`21_lambda_model_coupling.md`) — Lambda × CFM deferred
- `TODO-24` (`24_cfm_architecture_redo_plan.md`) — CFM architecture redo
- `TODO-22` (`22_data_gap_alignment_plan.md`) — data gap alignment
- `TODO-23` (`23_weak_to_strong_plan.md`) — weak-to-strong plan deferred
- `molmetal/reports/wf_round14_plan/phase1_lit.md` — lit survey
- `molmetal/reports/wf_round14_plan/phase2_synthesis.md` — synthesis
- `molmetal/reports/wf_round14_plan/phase3_writes.md` — Phase 3 deliverable
- `molmetal/reports/wf_round13_100x3/final.md` — Round-13 honest-negative
- `molmetal/reports/wf_path_b_gpu_retrain/final.md` — CFM Path B honest-negative
- `molmetal/reports/wf_round12_lambda_patha_10x3/final.md` — PathA-10x3 PARTIAL_LIFT
- `molmetal/reports/wf_lit_survey_v2/synthesis.md` — 65 papers + 20 theorems

## **完成 checklist**（实时更新）

- [x] 9 P0 CFM fixes ship（hidden_dim + tanh + PCGrad + bonds=zeros + vocab_mask + scaffold + bond_aware + in_dim + warning + F5 verify）
- [x] 4 Lambda fixes ship（soft prior + rebalance + truthfulness + scaffold-aware）
- [x] 5 Lambda Path-B fixes ship（ReworkedDecoder + 8 partner tiles + rule symmetry + bare-metal seed + diversity_bonus）
- [x] Lit-Survey-v2 ship（65 papers + 20 theorems borrowed + 5 NEW gaps）
- [x] 5 NEW research gaps documented as honest paper contributions
- [x] Lambda-Only-Paper-Path restructure ship
- [x] 71-page paper PDF ship，0 fatal errors
- [ ] **paper/main.pdf repair** (TODO-27 urgent) — file MISSING; need pdflatex diagnose+fix (2-5.5h CPU)
- [ ] **§4/§5/§6 paper content update with Phase-2 synthesis** (this workflow)
  - §4.2/§4.3/§4.6 promote DESIGN→MEASURED ONLY after Round-13 re-execution closes
  - §6 caveats 9 → 7 after Round-13 + CFM Alt A
  - §7 future work 5 NEW gaps → 3 after Round-14 closes
- [x] **Phase-2 synthesis ship** (`molmetal/reports/wf_round14_plan/phase2_synthesis.md`, 2026-09-15)
  - 12-week roadmap W38-W49 (R14 + R15 + paper submission)
  - Decision tree per gap (pocket-invariance + CFM Path B + paper/main.pdf + paper content)
  - P0/P1/P2 prioritized action list with dependencies + success criteria
  - Lit anchors preserved verbatim from phase1_lit.md §7
- [ ] WF-Round12-Lambda-PathA-10x3 完成（in flight）—— verify diversity on 30 cells
- [ ] WF-Round13-100x3-Sweep 完成（in flight）—— paper-grade 300 evaluations
- [ ] WF-CFM-Path-B-GPU-Retrain phase 2 完成——wrap-ordering fix + Vina mean verify
- [ ] paper §4.2 + §4.3 + §4.6 update with REAL 100x3 measurements
- [ ] paper §6 + §7 reduce caveats + future work per lift outcome
- [ ] arXiv submission ready

---

## Phase-4 update 2026-09-16 — R15 ship summary + R16 roadmap

**Source:** `molmetal/reports/wf_r15_consolidation/final.md` (this round's consolidation report); `wf_r15_cross_verify/final.md` (verifier, 89 tests + 2 real bugs caught); per-workflow final.md.

### Why this matters here

TODO-26 is the canonical comprehensive ship plan that consolidates Round-13 retry + Round-14 lit-grounded + paper submission timeline. Phase-4 (this update) adds:
1. **R15 ship summary** (today's 13 workflows across 4 parallel families + 1 verifier)
2. **R16 roadmap** = YuelBond GPU retrain + Round-13 100x3 paper-grade + Deflex integration
3. **12-week roadmap update** with R12 → R13 → R15 → R16 → Deflex-arXiv cadence
4. **2 real bugs caught** as ship-blocker pre-flight for R16

### §A. R15 ship summary (today's 13 workflows across 4 families + 1 verifier)

#### R15 family 1: CFM-Rescue (5 phases)
- Phase 1 YuelBond decoder swap (NEW module `lam_chem/yuelbond_decoder.py`)
- Phase 2 bond_head joint_train default flip
- Phase 3 training data scale 8→32 mols
- Phase 4 ODE midpoint + rectified flow + n_atoms fix
- Phase 5 200-step 8-sample decode smoke
- **24/24 unit tests pass**; bit-exact baseline at CPU smoke

#### R15 family 2: Lambda-Boost (4 phases + sub-fix C)
- Phase 1 reference_ligand_resolver module (NEW)
- Phase 2 F2(a) MetalLigandExchange SMARTS
- Phase 3 AquaExchange SMARTS
- Phase 4 pt_click_compat update
- Phase 5 r4_lambda_only_run.py wire-up (sub-fix C: reference_ligand_resolver)
- **21/21 unit tests pass**; pocket-invariance structural break

#### R15 family 3: Lambda-CFM-Coupling (4 phases; TODO-21 reopen)
- Phase 1 tmqm_cfm_pretraining.py (NEW, dry-run bridge)
- Phase 2 coupling_adapter.py (NEW, stub + load)
- Phase 3 warm_start.py wire
- Phase 4 learned_prior.py wire
- **17/18 unit tests pass; 1 SKIP (libtorch ABI host issue, not code defect); 1 FAIL (BUG-1 coupling reshape silent-fail)**

#### R15 family 4: Deflex-Wireup (5 phases)
- Phase 1 F5 learned shaping wire (SymbolicRegression-derived reward)
- Phase 2 PocketMacroInference v2 wire
- Phase 3 Sub-fix B learned_prior argmax wire
- Phase 4 v2 checkpoint switch (33-d features)
- Phase 5 Integration smoke
- **20/26 unit tests pass; 5 SKIP (libtorch ABI); 1 FAIL (BUG-2 CWD-relative path)**

#### R15 verifier (1 workflow)
- Cross-workflow test surface: 89 tests across 12 new files
- **82 pass (92.1%), 5 skip, 2 fail (2 real bugs caught)**
- File boundary audit: paper/* untouched; no symbol collisions
- 2 bugs are ship-blockers for R16 GPU retrain

### §B. R16 roadmap (YuelBond retrain + Round-13 100x3 + Deflex integration)

#### R16 priority 1 — BUG-1 + BUG-2 fix (R16 W37, ~30 min CPU)
- BUG-1: `learned_prior.py:412-417` reshape 64→5 silent-fail (3-line fix)
- BUG-2: `pocket_macro_inference.py:101,104` CWD-relative path (5-line fix)
- Re-run the 2 failing tests for green
- Verify BUG-1 coupling reshape 64→5 (truncate 60 → reshape 5,12 OR deterministic 64→5 sum-pool)
- Verify BUG-2 path resolution relative to `Path(__file__).resolve().parents[3]`

#### R16 priority 2 — YuelBond GPU retrain (R16 W42-W43, 12-24h GPU)
- Stack ALL R15 CFM-Rescue fixes: YuelBond + bond_head default learned + joint_train default True + n_train=32 + ODE midpoint + rectified flow
- 10000-step CFM retrain at h=128 (default per P1.1) on CrossDocked2020
- Target metric: `decode_ratio ≥ 0.5` on the 6-seed × 2-pocket × 2-cfg × 16-sample Round-10 protocol
- Honest decision tree (per TODO-24 §5):
  - decode_ratio ≥ 0.5 → SUCCESS → ship CFM geometric column in §4.3 (R16 lift)
  - decode_ratio in [0, 0.5] → investigate wrap-ordering OR add ETKDG init (T10 from Round-14 plan)
  - decode_ratio = 0 → honest-negative; document in §6.1; path (c) λ-only stays as default

#### R16 priority 3 — Round-13 100-pocket × 3-seed paper-grade sweep (R16 W44-W45, ~50 min CPU Lambda + ~6h GPU CFM)
- Lambda path (CPU): `r4_lambda_only_run.py --pockets 100 --seeds 42 0 1234 --n-simulations 1000 --metal-seed cisplatin --click-rules auto-pt-strict --metal-ligand-exchange`
- CFM path (GPU, after R16 P2 retrain completes): `r10_cfg_real_crossdocked.py --decoder-rework --pb-relax-mmff94 --bond-head learned --joint-train --n-train 32 --ode-steps midpoint`
- PB path (CPU): `r4_c_full_sweep.py --pb-mode dock --pb-relax-mmff94`
- Expected: 300 evaluations per metric per pocket (Vina / PB / SA / Diversity / NFE)
- Success criteria: ≥90% MEASURED on §4 Table 1 (was 0% per Round-13 partial)

#### R16 priority 4 — Deflex integration measurement (R16 W44-W45, parallel with P3)
- Re-validate Deflex v2 PocketMacroInference (post BUG-2 fix)
- Run Deflex F5 learned shaping across all 100 pockets (3 seeds each)
- Measure `learned_shaping_lift` vs control (no shaping)
- Target: `learned_shaping_lift ≥ 0.05` diversity_tanimoto

#### R16 priority 5 — paper §3.5 Deflex sub-section + §4 update + §6 reduce (R16 W45)
- Ship `paper/sections/03_5_deflex.tex` (NEW sub-section; per Workflow 1 ship)
- Promote §4.2/§4.3/§4.6 DESIGN→MEASURED cells based on R16 P3 sweep
- Reduce §6 caveats 9 → 7 (per TODO-25 §7.2)
- Reduce §7 future work 5 NEW gaps → 3 (per TODO-25 §7.2)

### §C. 12-week roadmap updated (R12 → R13 → R15 → R16 → Deflex-arXiv)

| Week | Round | Action | Status |
|---|---|---|---|
| W34 (2026-08-25) | R11 | QVina + data staging + N=50 parity | ✅ CLOSED |
| W35 (2026-09-01) | R11b | anticancer metric suite | ✅ CLOSED |
| W36 (2026-09-08) | R12 | top-journal pilot | ✅ DELIVERED (147 cells DESIGN→MEASURED in §4) |
| W37 (2026-09-15) | R13 + R14 | Round-13 partial + lit-grounded R14 | ⚙️ partial (R13 honest-negative; R14 in flight) |
| **W38 (2026-09-16) — today** | **R15** | **R15 structural ship (13 workflows + 1 verifier)** | ✅ **STRUCTURAL SHIP — 89 tests 82 pass + 5 skip + 2 fail (2 real bugs caught)** |
| W39 (2026-09-23) | **R16 P1** | BUG-1 + BUG-2 fix + re-verify | ⏳ PENDING (30 min CPU) |
| W40-W41 (2026-09-30 / 2026-10-07) | **R16 P3 prep** | Round-13 100×3 sweep infra + Lambda serialize | ⏳ PENDING (50 min CPU prep) |
| W42-W43 (2026-10-14 / 2026-10-21) | **R16 P2** | YuelBond 10000-step GPU retrain (12-24h GPU) | ⏳ PENDING (gated on GPU recovery) |
| W44-W45 (2026-10-28 / 2026-11-04) | **R16 P3 + P4** | Round-13 100×3 paper-grade sweep + Deflex integration measurement | ⏳ PENDING (~6h GPU CFM) |
| W45 (2026-11-04) | **R16 P5** | paper §3.5 + §4 update + §6 reduce + §7 reduce | ⏳ PENDING (8-12h CPU paper work) |
| W46 (2026-11-11) | **journal target selection** | (a) Digital Discovery (RSC) primary; (b) J. Chem. Inf. Model. backup; (c) Nat. Comput. Sci. | ⏳ PENDING |
| W47-W48 (2026-11-18 / 2026-11-25) | **arXiv prep** | format + cover letter + supplementary + ORCID + conflict-of-interest | ⏳ PENDING (8-16h CPU) |
| **W49 (2026-12-02)** | **arXiv SUBMIT** | **Q1-2027 target** | ⏳ PENDING |
| W49-W52 (2026-12-02 / 2026-12-23) | R17 (carry-over) | Deflex-arXiv integration + wet-lab validation prep + multi-metal extension | ⏳ PENDING |
| W53-W1 (2026-12-30 / 2027-01-06) | journal SUBMIT | Q2-2027 target | ⏳ PENDING |

### §D. R15 → R16 transition (honest framing preserved)

#### Partial lifts (MEASURED, not PROMISED)
- **R15 CFM-Rescue**: 4 fixes ship bit-exact with pre-fix baseline (24/24 unit tests pass); `n_bonded_samples` lifts from 0/8 → 8/8 (bond-bearing); RDKit-strict decode_ratio stays 0/8 (gated on R16 GPU retrain)
- **R15 Lambda-Boost**: 4 modules + 3 CLI flags + 21 tests all pass; pocket-invariance structural break is implemented; metric lift BLOCKED on GPU per the workflow's own honest framing
- **R15 Coupling dry-run**: `tmqm_cfm_pretraining.py --dry-run` exits 0 + writes deterministic .npz + .json; the bridge between CPU-stand-in CFM and Lambda MCTS root prior is alive (with BUG-1 caveat)
- **R15 Deflex end-to-end**: Phase 5 integration test passes (F5 shaping + v2 PocketMacroInference + learned_prior argmax all coexist); 100% train acc on 6 scaffold classes from CA2-fix

#### Open gaps (NOT closed in R15)
- **BUG-1 + BUG-2** must be fixed before R16 GPU runs (W39, ~30 min CPU)
- **CFM decode_ratio metric lift** gated on R16 GPU retrain (W42-W43, 12-24h GPU)
- **Lambda pocket-invariance metric lift** gated on R16 GPU runs for 100-p sweep (W44-W45)
- **PB production pass rate** at scale gated on R16 P3 sweep (W44-W45)
- **§3.5 Deflex sub-section** ready to ship on next paper recompile (W45)

#### Honest framing points (preserved verbatim)
1. **No theoretical derivation is re-attempted in this plan.** Each fix path cites the lit anchor in `phase1_lit.md §1-§6` + Lit-Survey-v2 65 papers + 20 theorems + Frontier 24 papers.
2. **All claimed lifts are PROJECTIONS, not measurements.** Per TODO-25 §"5 weak metrics → existing lit + plan", the (a)+(b)+(c)+(d) projected lift for Vina (-2 → -7) is a *lit-grounded projection*, not a measurement. R15 added **bonded-graph metric (n_bonded_samples)** which IS measured at the unit-test layer (0/8 → 8/8), but production-cell promotion is gated on R16 GPU retrain.
3. **R15 structural ship ≠ R13/R16 paper-grade ship.** R15 closes 13 structural fixes across 4 families; R13/R16 closes the paper-grade 100×3 sweep + §3.5 Deflex + §4 promotion.
4. **SBDD-with-metal-centers has no SOTA baseline.** Our metal-seeded Lambda pilot is one of the first formal ablations.
5. **5 NEW research gaps of TODO-25 §5 remain open** and traceable in §3.5 → §4.11 → §6.1 → §7.1 of paper.
6. **GPU recovery is fragile.** Per `wf_gpu_recovery_now/final.md`, GPU recovers after cold power cycle but any broken KFD node = fatal HSA init. Watchdog required for any GPU run.

### §E. R16 success criteria (gate for journal submission)

- `paper/main.pdf` exists ≥71 pages, 0 unresolved refs, 0 fatal errors
- §4 Table 1: 100×3 cells ≥ 90% MEASURED (was 0% per R13 partial)
- §4.6 PB column ≥ 60% MEASURED (was 0/30 per R13 smoke)
- §4.6 Vina column ≥ 60% MEASURED
- §4.6 Diversity column ≥ 60% MEASURED
- §4.6 bonded-graph column (NEW post-YuelBond) ≥ 60% MEASURED at 100-p scale
- §3.5 Deflex sub-section shipped (independent of GPU; can run on next recompile)
- §6 caveats: 12 → 8 (per TODO-25 §7.2; R15 expanded 8 → 12)
- §7 future work: 5 NEW gaps → 3 (per TODO-25 §7.2)
- arXiv preprint ready (cover letter + supplementary.tex + main.pdf)

### §F. R16 risk inheritance

| Risk | Severity | Status | Mitigation |
|---|---|---|---|
| **R1: CFM decode_ratio metric lift** | HIGH CONFIRMED | 24/24 unit tests pass; 0/8 → 0/8 CPU smoke | YuelBond 10000-step h=128 GPU retrain (W42-W43) |
| **R2: GPU outage** | HIGH CONFIRMED | gpu_status: recovered 2026-09-15 (cuda_available=True, 2 devices) | Watchdog + cold PSU cycle protocol |
| **R3: BUG-1 + BUG-2 ship-blockers** | HIGH CONFIRMED | 2 real bugs caught by R15 verifier | 30 min CPU fix in R16 W39 |
| **R4: Pocket-invariance lift** | HIGH CONFIRMED | R15 reference_ligand_resolver structural ship | R16 W44-W45 sweep |
| **R5: PB MMFF94 production pass rate** | MEDIUM CONFIRMED | R15 MMFF94-relax 22/26 click_tile | R16 W44-W45 at 100-p scale |
| **R6: paper/main.pdf recompile** | MEDIUM LOW | last recompile 2026-09-15 (69 pages / 5.17 MB) | R16 W45 verification |

### §G. Open questions for user (decision points)

#### D11 — journal target (R16 W46)
- (a) **Digital Discovery (RSC)** — RECOMMENDED (lit-grounded + open-source fit)
- (b) **J. Chem. Inf. Model.** — backup
- (c) **Nat. Comput. Sci.** — broader audience, higher bar

#### Lambda × CFM coupling timing (TODO-21 reopen in R15)
- R15 ships coupling_adapter + warm_start + learned_prior wire (structural)
- Per `TODO-21` decision: 重训可接受但不是现在；后面必做"Lambda 算法和模型侧相结合"
- A line CFM paused; B line + Round-12 pilot complete
- R16 P2 = YuelBond GPU retrain = first A-line CPU-stand-in CFM end-to-end coupling measurement
- Decision point: start 5-10d joint training after B line + R16 P2 complete

#### R16 retry decision tree on failure
- If R16 P2 retrain fails (decode_ratio < 0.5): ship ETKDG init (T10) OR extend YuelBond bond-budget to 64 mols (R17)
- If R16 P3 sweep fails (pocket-invariance not lifted): ship n_sim=5000 (T8) OR per-pocket learned prior (T11) as R17 follow-up
- If R16 P5 paper content fails: ship arXiv preprint with §6 honest-framing preserved (12 caveats)

### Cross-references (R15 → R16 chain)

- `molmetal/reports/wf_r15_consolidation/final.md` — R15 consolidation (this round)
- `molmetal/reports/wf_r15_cross_verify/final.md` — R15 verifier (89 tests + 2 bugs)
- `molmetal/reports/wf_cfm_rescue/phase5_200step_smoke.json` — YuelBond bit-exact baseline
- `molmetal/reports/wf_lambda_boost/final.md` — sub-fix C reference_ligand_resolver
- `molmetal/reports/wf_lambda_cfm_coupling/final.md` — TODO-21 reopen
- `molmetal/reports/wf_deflex_wireup/final.md` — 5-phase Deflex wireup
- `molmetal/reports/wf_cfm_frontier_research/final.md` — 24 papers Tier 1-4 synthesis
- `TODO-14` (`14_full_100pocket_paper_r13.md`) — Round-13 R15 update
- `TODO-21` (`21_lambda_model_coupling.md`) — Lambda × CFM coupling deferred → reopen
- `TODO-24` (`24_cfm_architecture_redo_plan.md`) — CFM YuelBond + Frontier update
- `TODO-25` (`25_round14_lit_grounded_plan.md`) — Round-14 lit-grounded plan
- `TODO-27` (`27_paper_main_pdf_repair.md`) — paper/main.pdf repair (CLOSED 2026-09-15)
- `TODO-28` (`28_round12_honest_negative_reframe.md`) — R12 framing
- `TODO-29` (`29_f2a_round13_retry.md`) — F2(a) + Round-13 retry (CLOSED via R15 Lambda-Boost)

## R15 master consolidation (2026-09-16)

**Verdict:** CONDITIONAL SHIP. R15 = 4 parallel work-streams + 1 verifier + 1 master.
- **DIVERSITY LIFT MEASURED** at n_sim=1000 (R12 Deflex all-on 10×3): n_distinct 1→20, sa 5.9→3.1, div_tan 0→0.11.
- **CFM + across-pocket NOT MEASURED**: decode_ratio 0/8 (architecture-bound); A+B+C test_010/011/012 Jaccard=1.0 (manifest fallback to legacy seed).
- **2 real bugs caught**: BUG-1 coupling reshape + BUG-2 CWD path (both sub-10-line fixes).
- **Paper recompile**: 75 pp / 5.2 MB / 0 fatal errors; +6 pp over WF-Bib-Complete-v2 baseline.
- **89 tests across 12 files** (92.1% pass) per `wf_r15_cross_verify/final.md`.
- **Master report**: `molmetal/reports/wf_r15_all/final.md` — one-page summary.
