# Cross-Skill Summary — Mol-Metal 8 Skills × 5 Dimensions

**Date**: 2026-09-26
**Source**: `.claude/skills/{mol-metal-lambda-calculus,mol-metal-cfm-training,mol-metal-click-chemistry,mol-metal-anticancer,mol-metal-mechanism-mapping,mol-metal-paper-writing,mol-metal-r18-math,party-twenty-spirit}/SKILL.md`
**Workflow**: wf_cross_skill_summary (in-flight)
**Convention**: 1-line summary per skill; full 5-dim rows below; cross-link table at end.

---

## 1. Skill summary rows (1 line × 8 skills)

| ID | Skill | One-line summary |
|---|---|---|
| **S1** | `mol-metal-lambda-calculus` | Typed β-NF substrate (atoms-as-combinators + bonds-as-application); KataGo-MCTS + PLL over MLC terms; LambdaASTSR reward shaper; Deflexpressor coupling to CFM. |
| **S2** | `mol-metal-cfm-training` | Lipman CFM backbone (EGNN + YuelBond + Triton-fused ODE solver); R17 30/30 decode probes; 23 R18 adapters wrap it without modifying R17 close-out. |
| **S3** | `mol-metal-click-chemistry` | 18 click rules (5 base + 13 SyntheMol) + 101 MLE matrix (84 + 17 NEW) + 21 non-click med-chem = 43 total; SMARTS-only, no pretrained weights. |
| **S4** | `mol-metal-anticancer` | Cytotoxicity data spine (MetalCytoToxDB / PlatinAI / tmQM / NCI-60); m1/m2/m17/m18/m21 Tier-1 metrics; kNN oracle caveat + Krasnov 2026 / Rusanov 2025 baselines. |
| **S5** | `mol-metal-mechanism-mapping` | Horizontal diagnostic layer; 21 R17 + 23 R18 = 44 mechanisms × 20 metrics; failure-mode → ablation → §6 limitations bridge. |
| **S6** | `mol-metal-paper-writing` | §1-§7 + appendix drafting helper; MEASURED / PROJECTED / ARCHITECTURAL tag discipline; §4 LEAD-claim 8 (m11/m12/m13/m14/m17/m2/m21/Deflexpressor). |
| **S7** | `mol-metal-r18-math` | 9 mathematical theorems (T1-T9 + T10) → 23 R18 mechanisms (M22-M44); CPU-only ship, GPU retrain DEFER per Blackwell single-card. |
| **S8** | `party-twenty-spirit` | Mandatory pre-action skill; 5-dim decision checklist + ≤30-line mapping conclusion + R18 self-check; forces "不忘初心" before any TODO update. |

---

## 2. 5-Dim × 8-Skill matrix

### S1 — `mol-metal-lambda-calculus`

| Dim | Content |
|---|---|
| **When to use** | User asks about lambda calculus, Molecular Lambda Calculus, β-reduction / β-NF, Church encoding, atoms-as-combinators, bonds-as-application, curried partial application, synthesis certificate, closure theorem, Deflex. Triggers: `molmetal/molmetal_lam/lam_chem/{ast,lambda_combinators,closure}.py`, `paper/sections/03_1_mlc_formalism.tex`, `TODO/13_lambda_clickchem/`. |
| **How to use** | Read AST nodes (`ast.py`); PRIMITIVE_ATOMS = 42-entry (8 organic + 4 halogen + 30 d-block); arity = valence + lone_pairs; `closure.py` gates β-reduction; MCTS via `proof_search.py` (KataGo + PLL); reward shaper = LambdaASTSR (Deflexpressor-style λ-AST SR). Verify: `pytest molmetal/molmetal_lam/tests/ -q --tb=short` (89 tests). |
| **Examples** | (1) Verify cisplatin `Pt_II(NH3)(NH3)(Cl)(Cl)` is closed + β-NF via `closure.is_closed_term` + `closure.is_in_beta_nf`. (2) Generate analogue via `AquaExchange` step (β-reduction); confirm `USE_LAMBDA_ASTSR is True` (R²=1.0000 vs F5=0.9324). |
| **Cross-ref** | → S2 (CFM bond head decodes MLC products) · → S3 (click rules = β-reduction steps) · → S4 (A2780/MCF7 score MLC β-NF witnesses) · → S5 (M1-M4, M8-M11 owned here) · ← S8 (must call before any MLC refactor). |
| **R18 mechanism ref** | M42 (21-rule non-click med-chem extends MLC PRIMITIVE_RULES) · M31 quantum semantics (paper §3.8 symplectic point per atom) · M32 Hochschild cohomology (β-reduction cochain). Cite keys: `abdel_magid_1996_joc` + `maxiaonan2026quantization` + `lisi2026hochschild`. |

### S2 — `mol-metal-cfm-training`

| Dim | Content |
|---|---|
| **When to use** | User mentions CFM, flow matching, Lipman, EGNN, YuelBond, bond head, decode_ratio, decode smoke, CFM retrain, fm_pocket, AMP/autocast, BF16, velocity_net, ODE solver, joint_train, h=128/n_layers=3. Triggers: `molmetal/adapters/flow_matching_lipman/`, `molmetal/scripts/metallo_drug_smoke_retrain.py`, `TODO/pending/24_cfm_architecture_redo_plan.md`. |
| **How to use** | Lipman 2023 CFM + SE(3)-equivariant EGNN + YuelBond bond head; AMP/BF16/AMP-DTYPE env vars; checkpoints in `molmetal/checkpoints/`; default flags `--hidden-dim 128 --n-layers 3 --bond-head learned --joint-train --decode-smoke-every 1000`; pitfall guard checklist P1-P7. GPU Blackwell + Triton 3.8 + torch 2.14.0+cu130. |
| **Examples** | (1) 200-step decode-smoke probe (`--decode-smoke-every 50`) — cheapest regression catch. (2) 1000-step × 3-seed stability (`--seed 0/1/2`); expect mean=1.000 stdev=0.000 on clean Blackwell; stdev>0.10 flags P6/P7. |
| **Cross-ref** | ← S1 (MLC β-NF = substrate) · → S3 (5×5/18×18 SMARTS = bond head targets) · → S4 (m1/m2/m21 from CFM-decoded candidates) · → S5 (M5-M7, M12-M17 owned here) · ← S8. |
| **R18 mechanism ref** | **W2.1** SE(3) SB (M22, `se3_sb_integration.py`) · **W2.2** Transport Relaxation (M23, additive-only) · **W2.3** Multi-PH features (M24) · **M27** W-Flow ODE (5×3 windows) · **M29** Coordination-Aware GNN (independent) · **M30** Quotient-Space Diffusion · **M34** Fractional ODE (Caputo D^α). Cite: `mahmood2026sb_lie` + `wflow2026` + `arvind2026qsdm`. |

### S3 — `mol-metal-click-chemistry`

| Dim | Content |
|---|---|
| **When to use** | User mentions CuAAC/SPAAC/ThiolEne/Suzuki/AmideCoupling/MetalLigandExchange/AquaExchange, click rule, SMARTS, RDKit AllChem.ReactionFromSmarts, building block, azide/alkyne/boronic acid/strained alkyne/terminal alkyne. Triggers: `molmetal/molmetal_lam/lam_chem/pt_click_compat.py`, `molmetal/molmetal_lam/reactions/`, `molmetal/molmetal_lam/sbdd_env/syntemol_reactions.py`. |
| **How to use** | 5 click + 2 metal exchanges = 7 base rules; `pt_click_compat.py` is **single source of truth** (5×5 / 18×18 matrix); gate `can_apply(rule, metal_state)` BEFORE AllChem call; never hard-code SMARTS outside `pt_click_compat.py`; never load SyntheMol Chemprop weights (IP). 101-rule MLE matrix = 84 + 17 NEW (transmetalation + Cl-OH swap + ligand-shell-MMA + Pt/Pd transmetalation). |
| **Examples** | (1) Gate CuAAC on Pt(II): `can_apply("CuAAC", "Pt_II") == True`; Suzuki on Pt_II == False (Pd-catalyzed). (2) Add oxime ligation: register SMARTS + `register_compat` for every Pt oxidation state. |
| **Cross-ref** | ← S1 (atoms-as-combinators = MLC substrate) · → S2 (YuelBond bond head learns click-reacted bonds) · → S4 (A2780/MCF7 correlates with click rule) · → S5 (M10-M11 owned here) · ← S8. |
| **R18 mechanism ref** | **M38** FG-compat / leave-group veto (84-cell FGCompatMatrix; `--fg-veto-strict` default OFF) · **M36** skeleton jump (Bemis–Murcko fragment swap; single-step vs M39 3-stage) · **M42** non-click med-chem (peptide + Boc + reductive amination; 18→21 rule expansion via sidecar). Cite: `march_1985_advanced_organic` + `bemis_murcko_1996_jmc`. |

### S4 — `mol-metal-anticancer`

| Dim | Content |
|---|---|
| **When to use** | User mentions cisplatin/carboplatin/oxaliplatin/nedaplatin/satraplatin/picoplatin, Pt(II)/Ru(II)/Ir(III)/Au(I/III)/Pd(II), square-planar/octahedral/linear geometry, IC50/pIC50/GI50, A2780/MCF7/NCI-60, MetalCytoToxDB, PlatinAI, MBFinder, Krasnov 2026, Rusanov 2025. Triggers: `molmetal/data/cytotox.py`, `molmetal/molmetal_lam/lam_chem/platinai_dataset.py`, `paper/03_7_metallodrug_vertical.tex`. |
| **How to use** | Data spine: MetalCytoToxDB (4.8 MB / 26801 rows) + tmQM (300 MB / 100849 rows) + NCI-60 (420 MB / 4.85M rows) + PlatinAI/MBFinder (500-mol MaxMin subset pre-shipped); D-MPNN 5-metal multi-task; kNN oracle with `PlatinAIDataset` (NOT wet-lab ground truth); always cite Rusanov 2025 when reporting. |
| **Examples** | (1) Score cisplatin via `KNNEvaluator(ds, k=5, fp_radius=2, fp_bits=2048).predict([smiles])` → cite Rusanov 2025. (2) m17 + m21 metric pair: `train_drugood --splits assay scaffold size --seeds 0 1 2` + `train_nci60_10cell --cells MCF7 A549 A2780 ...`. |
| **Cross-ref** | ← S1 (MLC β-NF = scored candidates) · ← S2 (CFM-decoded conformers) · ← S3 (click rules = which candidates exist) · → S5 (M9, M18-M21 owned here) · ← S8. |
| **R18 mechanism ref** | **M25** Hilbert primal-dual (replaces F5/LambdaASTSR evolutionary SR with Chambolle-Pock 2011 — *J. Math. Imaging Vision* 40(1): 120-145, issued 2010-12-21, DOI 10.1007/s10851-010-0251-1 — the primal-dual splitting actually implemented in `molmetal/molmetal_lam/reward/hilbert_primal_dual.py`; Hilbert-space primal-dual dynamics: Sun & Zheng 2026, 孙祥凯 (Sun Xiangkai) & 郑丽娟 (Zheng Lijuan), 《数学学报(中文版)》 2026年04期, DOI 10.12386/A20250022, English preprint arXiv:2408.06884 (*J. Optim. Theory Appl.* 207(1): 12, 2025, DOI 10.1007/s10957-025-02766-6) — volume / issue / pages UNVERIFIED; affects m1 + wall-time) · **M26** Score-Repellent MC (variance-controlled PUCT; ≥30% MCTS var reduction) · **M43** Bergman kernel (d-orbital occupancy; RBFE ΔG 1.8→1.2 kcal/mol). Cite: Chambolle & Pock 2011 (DOI 10.1007/s10851-010-0251-1) + Sun & Zheng 2026 (DOI 10.12386/A20250022 / arXiv:2408.06884) + `score_repellent_mc_icml2026` + `ma_xiaonan2025bergman_kernel`. (No `sun2026hilbert` entry exists in `paper/refs.bib` — 0 hits across its 100 entries — so that key is replaced by the verified inline anchors; see the Wf-A149 note below.) |

> **NOTE-Wf-A149 (2026-10-31, citation-truth wave)** — the superseded
> wording of the M25 mechanism row is retained verbatim below as a
> historical record (never delete a superseded record — AGENTS.md §5.5
> rule 6):
>
>     **M25** Hilbert primal-dual (replaces F5/LambdaASTSR evolutionary SR with Chambolle-Pock 2011; affects m1 + wall-time) · **M26** Score-Repellent MC (variance-controlled PUCT; ≥30% MCTS var reduction) · **M43** Bergman kernel (d-orbital occupancy; RBFE ΔG 1.8→1.2 kcal/mol). Cite: `sun2026hilbert` + `score_repellent_mc_icml2026` + `ma_xiaonan2025bergman_kernel`.
>
> Superseded because `sun2026hilbert` named a fabricated record: its
> arXiv id `arXiv:2510.08423` resolves to "Computational Bell
> Inequalities" (Merkulov & Arnon, quant-ph, v2 2025-10-10) — NOT a
> Hilbert-space paper — and no `sun2026hilbert` entry exists in
> `paper/refs.bib` (0 hits across its 100 entries).  The Sun & Zheng
> paper itself IS real (孙祥凯 / 郑丽娟, 《数学学报(中文版)》 2026年04期,
> DOI 10.12386/A20250022; English preprint arXiv:2408.06884), so the
> citation was re-pointed at the verified record rather than deleted.
> The algorithm actually implemented is Chambolle–Pock (verified on
> disk: `hilbert_primal_dual.py` — `x_{k+1} = prox(x_k − lr∇F(x_k))`,
> `y_{k+1} = y_k + lr_dual·(x_{k+1} − x_k)`), so that is the honest
> mechanism anchor.  The Douglas-Rachford substitute proposed for the
> Hilbert anchor is NOT-FOUND (Crossref has no such record; DOI
> 10.1137/110822976 → HTTP 404) and is NOT used.

### S5 — `mol-metal-mechanism-mapping`

| Dim | Content |
|---|---|
| **When to use** | (a) Any metric regressed → which mechanism is responsible? (b) Designing ablation experiment. (c) Reviewing PR that changes a mechanism → predict affected metrics. (d) On-boarding new team. (e) Writing §6 Limitations. (f) Planning R18+. Triggers: `TODO/MECHANISM_METRIC_MAPPING.md`, `TODO/METRICS_TARGETS.md`, `TODO/EVALUATORS_MIGRATED.md`, "metric regression", "ablation design", "mechanism attribution", "why did m_X drop". |
| **How to use** | 21 R17 mechanisms (M1-M21) + 23 R18 (M22-M44) = **44 mechanisms total** × 20 metrics (m1-m20) + paper §3 math; §1 mechanisms / §2 metrics / §3 contribution matrix (✓ 主要 / ○ 次要 / blank 无影响) / §4 failure modes / §5 ablation design / §6 diagnostic checklist. **Owns M1-M21 + M22-M44 horizontal view**; per-skill skills own vertical module implementation. |
| **Examples** | (1) m14 decode_ratio regressed 1.000→0.300: run `python molmetal/scripts/diagnose_metric.py --metric m14` (M15→M16→M17→M13→M14 chain). (2) Ablate M7 YuelBond: backup `bond_head.py` → swap `bond_head.py.no_yuel` → 200-step smoke → restore. (3) Add m21 NCI-60 9-cell: update §2/§3/§4 + cross-ref S4. |
| **Cross-ref** | ← S1 (M1-M4, M8-M11) · ← S2 (M5-M7, M12-M17) · ← S3 (M10-M11) · ← S4 (M9, M18-M21) · → S6 (paper §6 limitations) · → S7 (R18 §5/§6/§7 cross-mapping) · ← S8. |
| **R18 mechanism ref** | **This skill is the integration point** — full M22-M44 × key metric table (§5 W1 / §6 W2 / §7 W3). Diagnostic commands: `eval_all_evaluators.py` (14/14) + `wf_pitfall_guard/check_all.py` (7/7) + `pytest test_pitfall_guards.py`. Cite keys: all of S1-S7 + S8 (party-twenty-spirit cited via TODO/completed/). |

### S6 — `mol-metal-paper-writing`

| Dim | Content |
|---|---|
| **When to use** | User asks to write/edit/draft/review/extend any of: `paper/main.tex`, `paper/sections/*.tex`, `paper/appendices/*.tex`, "write §3", "draft §4", "results table", "limitations", "future work", "abstract", "MEASURED vs PROJECTED", "honest framing", "figure caption", paper-ready TODO claim. Triggers: section templates + traceability.csv + bibtex unresolved=0. |
| **How to use** | 3 source docs: `TODO/02_metrics/METRICS_TARGETS.md` + `TODO/02_metrics/MECHANISM_METRIC_MAPPING.md` + `paper/main.tex`; tag discipline MEASURED / PROJECTED @ R-round / ARCHITECTURAL (never drop bold); §4 LEAD-claim 8 = m11/m12/m13/m14/m17/m2/m21/Deflexpressor; §4.5 honest-negative + §6 limitations for 8-9 weak metrics; traceability.csv generated by `build_traceability_csv.py`. |
| **Examples** | (1) §3.4 SE(3) SB reference: add `\citep{mahmood2026sb_lie}` + tag math `\textbf{ARCHITECTURAL}` + §6 "Blackwell single-card, CPU-only shipping" + `\textbf{PROJECTED @ R18}` for m3 uplift. (2) §4 Results table: pull every Metric N row, cite source row inline as `\todo{metric=N,file=METRICS_TARGETS.md}`. |
| **Cross-ref** | ← S1 (§3.1 MLC, Appendix A closure) · ← S2 (§3.5 Deflex) · ← S3 (§4.4 wet-lab yield, Appendix C protocol) · ← S4 (§3.2 datasets, §4 metric) · ← S5 (§3.3 reward rationale, §4 interpretation, §6 limitations) · → S7 (§3 method extensions + refs.bib 23 cite keys) · ← S8 (must call before any §-section bulk ship). |
| **R18 mechanism ref** | 23 cite keys for M22-M44 (one per row); §3.x paragraphs per mechanism; §4 metric claims all `\textbf{PROJECTED @ R18}` (GPU DEFER); only M25 (CPU smoke 452× speedup) ships `\textbf{MEASURED}`; §6 must enumerate every R18 mechanism that ships CPU-only; self-check: `bibtex paper/main && grep -c '^\\bibcite{' main.aux` = 0 unresolved. |

### S7 — `mol-metal-r18-math`

| Dim | Content |
|---|---|
| **When to use** | (a) R18 upgrade planning: which theorem to graft onto which component. (b) Paper §3 method extension drafting. (c) Reward/flow redesign (SB vs Wasserstein GF vs Bass martingale). (d) Pocket-conditioned generation on SE(3). (e) Multi-PH features. (f) Risk analysis (PH approximation hardness). Do NOT use for: pure plumbing, dataset glue, D-MPNN head wiring (defer to S2). |
| **How to use** | 9 theorems (T1-T9) + T10 quantum semantics = 10 math objects; priority map P0 = T1 SB on Lie Groups / T2 Transport Relaxation / T5 multi-PH; P1 = T3 Monge-Ampère / T4 WGF / T7 SB-Bass; P2 = T8 Bass existence (cite-only) / T9 Bergman / T6 PH hardness (guardrail). Each: theorem statement → Mol-Metal anchor → integration sketch → expected delta → fallback → cite key. |
| **Examples** | (1) m3 Vina Dock uplift: match metric → Lie-group geodesic (T1) → check P0 priority → verify ship (`se3_schrodinger_bridge.py` + `se3_sb_integration.py`, 25/25 tests) → M22 mechanism → `\citep{mahmood2026sb_lie}` → §3.5 ARCHITECTURAL + §6 Blackwell CPU-only. (2) Reject T3 Monge-Ampère for m3 (convexity fails on multi-modal pockets). |
| **Cross-ref** | ← S2 (training-loop mechanics, CUDA kernels) · → S5 (§5/§6/§7 R18 cross-mapping) · → S6 (paper §3.x drafting + refs.bib) · → S1 (S1 owns M31/M32 paper-§3 math) · → S4 (M25/M26/M43 anchors) · ← S8. |
| **R18 mechanism ref** | **Full M22-M44 mechanism↔theorem table** in this skill — 23 cite keys; 3/9 theorems upgraded P1→shipped (T1, T2, T5); W1/W2/W3 wave distribution (M22-M28 / M29-M35 / M36-M44); 308/308 tests pass CPU-only; 0 GPU retrain shipped. Cite: 23 keys in `paper/refs.bib` per skill §6 self-check. |

### S8 — `party-twenty-spirit`

| Dim | Content |
|---|---|
| **When to use** | **MANDATORY pre-action** — (1) "阶段性结束 + 更新 TODO 之前"; (2) R17→R18 / R18→R19 round switch; (3) paper section / workflow batch close-out; (4) 投稿 / drop metric / 启动新 R decision; (5) BLOCK / failure / honest-negative close-out. Triggers: EXECUTION_ORDER.md §5/§7 决策点 + workflow 收尾. |
| **How to use** | 3 steps: (1) Read EXECUTION_ORDER.md §5/§7 current stage; (2) Apply §3 5-dim decision checklist (三个务必 / 守正创新 / 系统观念 / 高质量发展 / 自我革命) → output ≤30-line mapping conclusion; (3) THEN update TODO 目录. R18 §11 self-check 5 questions mandatory. Anti-pattern: "re-using this skill to optimize itself" = 凑数, refuse + write §10 log. |
| **Examples** | (1) Phase A Stage 1 BLOCK close-out: 不忘初心 ❌ (Ru 0.712 没修); 谦虚谨慎 ✅; 决策=Phase A.6 ablation. (2) R18 PLAN 12-file PR audit: approve 600-LOC adapter (真价值), reject 11 thin shims (凑数) — `no_overclaim=true`. |
| **Cross-ref** | **First skill called** before S1-S7; provides 决策输入 for all downstream. §7 chain: S8 → S5 (horizontal diagnostic) → {S1/S2/S3/S4} (technical execution) → {S6/S7} (paper/R18). §8 hooks: keyword trigger + decision-point hook + BLOCK close-out hook. |
| **R18 mechanism ref** | §11 R18 mechanism self-check 5 questions + W1/W2/W3 ship quick-reference table (23/23 CPU-only shipped 2026-09-26; 308/308 tests pass; 0 GPU retrain); §10 verification log enforces "skill 价值 = 调用次数 × 决策真被影响", not文档字数. |

---

## 3. Skill cross-link map (edges between 8 skills)

```
S8 (party-twenty-spirit, mandatory pre)
  ↓ ≤30-line mapping conclusion
S5 (mechanism-mapping, horizontal diagnostic — owns M22-M44 cross-ref)
  ↓ failure-mode → ablation → §6 limitations
{S1 (lambda) ← substrate}
{S2 (cfm-training) ← backbone}
{S3 (click-chemistry) ← rules}
{S4 (anticancer) ← data + Tier-1 metrics}
  ↓ code/test/data
S6 (paper-writing, §1-§7 + MEASURED/PROJECTED)
S7 (r18-math, 9 theorems + 23 mechanisms)
  ↓ paper §3.x / §4.x / §6 + refs.bib
[Final artifact]
```

| Edge | From → To | Nature | Anchor |
|---|---|---|---|
| S8 → all | pre-action | 决策输入 | §3 checklist |
| S1 → S2 | MLC substrate → CFM refiner | typed β-NF → 3D atom cloud | `lambda_combinators.py` → `flow_matching_lipman/` |
| S2 → S3 | CFM bond head → click rules | bond-type learn → SMARTS target | `bond_head.py` → `pt_click_compat.py` |
| S3 → S4 | click rules → scored candidates | β-reduction → A2780/MCF7 | `syntemol_reactions.py` → `platinai_dataset.py` |
| S4 → S5 | Tier-1 metrics → mechanism attribution | m1/m2/m17/m18/m21 → M9/M18-M21 | `mol-metal-anticancer` → `MECHANISM_METRIC_MAPPING.md` |
| S1 → S5 | MLC AST → mechanism | M1-M4, M8-M11 | `lambda_combinators.py` → §3 matrix |
| S2 → S5 | CFM pipeline → mechanism | M5-M7, M12-M17 | `flow_matching_lipman/` → §3 matrix |
| S3 → S5 | click rules → mechanism | M10-M11 | `pt_click_compat.py` → §3 matrix |
| S5 → S6 | §6 limitations → paper text | failure-mode → Limitations row | `MECHANISM_METRIC_MAPPING.md` §4 → `paper/sections/06_*` |
| S7 → S6 | theorems → §3 method | T1-T10 → §3.x paragraph | `math_skills/*` → `paper/sections/03_*` |
| S7 → S5 | mechanisms → diagnostic | M22-M44 → §5/§6/§7 R18 cross-mapping | `MECHANISM_METRIC_MAPPING.md` §5/§6/§7 |
| S6 → S7 | cite key demand → bib | paper §3 needs → refs.bib 23 keys | `paper/refs.bib` |
| S1 → S7 | MLC owns paper-§3 math | M31 quantum semantics + M32 Hochschild | `quantum_semantics.py` + `hochschild_cohomology.py` |
| S4 → S7 | anticancer owns R18 scoring math | M25/M26/M43 | `hilbert_primal_dual.py` + `score_repellent_mc.py` + `bergman_kernel.py` |

---

## 4. Counts + status snapshot (2026-09-26)

| Skill | Code modules | Tests | R17 ship | R18 ship (CPU-only) |
|---|---:|---:|---|---|
| S1 lambda | 5 (atoms/lam_chem/search_alg/reward/tests) | 89 + 20 + 11 + 69 + 17 = 206+ | ✅ close-out | M42/M31/M32 |
| S2 cfm-training | 7 (adapters + scripts + tests + triton) | 18 + 30 + 14 | ✅ close-out | M22/M23/M27/M29/M30/M34 (6) |
| S3 click-chemistry | 6 (reactions + pt_click_compat + tests) | 101-rule contract + 25/25/33 = 83+ | ✅ 18 click + 84+17 MLE | M38/M36/M42 |
| S4 anticancer | 6 (data + scripts + tests) | 17/17 tmQM + 14/14 evaluators + 3-seed | ✅ m17/m18/m21 | M25/M26/M43 |
| S5 mechanism-mapping | 3 docs (METRICS/MECHANISM/EVALUATORS) | 14 evaluators + 7 pitfall + 15 credibility = 36 | ✅ 21 mechanisms | M22-M44 = 23 |
| S6 paper-writing | 1 doc + bib + traceability.csv | n/a (writing helper) | ✅ §1-§7 + 4 appendices | M22-M44 cite keys |
| S7 r18-math | 10 math_skills + 6 adapters | 25/25+25/25+25/25+32/32+36/36+31/31+49/49 = 250+ | n/a | T1-T10 = 10 ships |
| S8 party-twenty-spirit | 1 doc + verification log | n/a (pre-action skill) | ✅ §10 4 entries | R18 §11 self-check |

**Grand total**: 8 skills × 5 dimensions = **40 cells filled** (above); 23 R18 mechanisms × 5 dims (when/where/file/cite/self-check) covered across skills; 308+ R18 tests pass CPU-only; 0 GPU retrain shipped.

---

## 5. Use this summary when

- On-boarding new team member → read S1-S5 + cross-link map + counts table.
- Designing R18 PR → read S5 §5/§6/§7 + S7 mechanism↔theorem table + S6 tag discipline + S8 pre-action.
- Writing paper §4 LEAD claim → check S6 §4 LEAD-claim 8 + S4 Tier-1 metric results + S5 §3 contribution matrix.
- Diagnosing m_X regression → read S5 §4 failure modes → cross-link to S1/S2/S3/S4 owning mechanism.
- Reviewing 凑数 vs 真价值 PR → apply S8 §3.4 高质量发展 + §10 verification log + R18 §11 self-check.

**End of cross-skill table.**