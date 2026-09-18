# Phase 3 §6 Limitations Integration Report (2026-09-15)

> **Author:** automated workflow (paper content update, Phase 3 of paper-content-workflow)
> **Scope:** §6 limitations, items 10–12. Three new caveats added: (10) pocket-invariance + warm_start/learned_prior integration pending; (11) PathA-10x3 lift does not generalize without root-prior integration; (12) 5 algorithmic fixes ship but only partially lift the singleton attractor (1 of 3 layers broken, 2 require proof_search.py integration). Honest framing preserved throughout.
> **Files touched:**
> 1. `/home/hugo/codes/try_triton_on_rocm/paper/sections/06_limitations.tex` (item (9) trailing sentence + new items 10/11/12)
> 2. NO changes to `paper/sections/04_evaluation.tex` (Phase 2 already integrated Path A + AlgoTune; no §4 work needed for §6)
> 3. NO changes to `paper/sections/03_4_mcts.tex` (no §3 cross-refs required by §6 items 10–12)
> 4. NO changes to `paper/sections/04_evaluation.tex` §4.6, `paper/sections/05_ablation.tex`, `paper/main.tex`, `paper/refs.bib` (per task brief — Workflow 1 / Workflow 5 own)
> 5. NO changes to §6 items 1–9 (preserved as-is; only item (9) gets a 1-line trailing sentence linking to items (10)/(11))
> 6. NO changes to `paper/sections/CROSS_REFS.md` (existing §6 entry unchanged; new items 10–12 inherit item (9) framing)

---

## 1. Summary of line-level changes

### 1.1 §6 item (9) — appended 1-line link to items (10)/(11) (no other edits)

**Location:** `paper/sections/06_limitations.tex` lines 293–354 (item (9) "Λ-only emission table is pocket-invariant")
**Change type:** append a 1-sentence cross-reference bridge at the head of item (9)'s prose body so that item (9) explicitly defers to the extended analysis in items (10) and (11).

**Before:**
> \item \textbf{$\Lambda$-only emission table is pocket-invariant (NEW 2026-09-15, WF-Round12-Lambda-PathA-10x3 + WF-AlgoTune).}
> The four-fix bundle (Path B rule symmetry + decoder rework + scaffold-aware gate + partner tiles) lifted the singleton collapse ...

**After (1-sentence insertion only):**
> \item \textbf{$\Lambda$-only emission table is pocket-invariant (NEW 2026-09-15, WF-Round12-Lambda-PathA-10x3 + WF-AlgoTune).}
> This is the canonical framing of pocket-invariance as item (9); items (10) and (11) below extend it with the PathA non-generalization and singleton-attractor analysis.
> The four-fix bundle (Path B rule symmetry + decoder rework + scaffold-aware gate + partner tiles) lifted the singleton collapse ...

The 1-sentence bridge keeps item (9) intact (preserving its 9-line MEASURED-evidence block and 6-line recommended-fix block) while making the section-level narrative flow explicit: item (9) = canonical pocket-invariance, item (10) = pocket-invariance + warm_start/learned_prior status, item (11) = PathA lift non-generalization, item (12) = 3-layer singleton attractor status.

### 1.2 §6 item (10) — NEW caveat (33 evaluation cells, byte-identical basket)

**Location:** `paper/sections/06_limitations.tex` lines 355–356 (inserted between item (9) and item (11))
**Change type:** NEW \item in the enumerate environment; 1-paragraph MEASURED evidence + 1-paragraph recommended fix + file:line cross-refs.

**Body (verbatim from §6 after edit):**

> \item \textbf{Pocket-invariance: novel pockets produce identical candidate lists; cache + reward layers still active; warm\_start / learned\_prior integration pending (NEW 2026-09-15, WF-Paper-Content Phase 3).}
> The PathA-10x3 30-cell panel (test\_000..test\_009 $\times$ 3 seeds) and the AlgoTune 3-cell novel-pocket smoke (test\_010..test\_012 $\times$ 1 seed, 202.63\,s wall) together produce 33 evaluation cells; \textbf{all 33 cells report byte-identical 20-SMILES candidate lists} (just reordered into the candidate slots). The per-pocket reference ligand (cisplatin-like for test\_000..009, sperminium-derivative for test\_010, adenosine-receptor for test\_012) is \emph{completely ignored} by the $\Lambda$-only emission pipeline --- the metal-seed anchor \texttt{[Pt]C\#C} and the fixed-bias / soft-prior MCTS regime override the pocket anchor in the live \texttt{MCTSProofSearch} at \texttt{proof\_search.py:2726} (\texttt{\_unreactive\_states}) and at the reward aggregator. Reference-Tanimoto to the pocket reference is constant $0.1415$ on the Path A panel and per-pocket $0.1649$ on novel pockets because the pocket reference does \textbf{NOT} enter the $\Lambda$ path (no CFM module is invoked, no pocket-conditioned root prior is integrated). Per-cell std on every diversity metric is $0.0$; the four-fix bundle (Path B rule symmetry + decoder rework + scaffold-aware gate + partner tiles) lifted $n_{\text{distinct}}{=}1 \to 20$ but did \emph{not} differentiate by pocket. The \textbf{cache + reward layers remain active} (the MCTS cache \texttt{\_unreactive\_states} permanent membership at \texttt{proof\_search.py:2726} was untouched; the metal-geometry reward prior was softened to \texttt{soft\_score\_metal\_geometry} but is opt-in at default weight $0.0$). The \textbf{warm\_start} module (\texttt{molmetal/molmetal\_lam/lam\_chem/pocket\_features.py}, Phase-3J) and the \textbf{learned\_prior} checkpoint (\texttt{checkpoints/learned\_prior\_Pt.pt}, Phase-3L) are DESIGNED but \textbf{NOT integrated} into the live MCTS. \textbf{Measured evidence (33 cells, $n_{\text{simulations}}{=}1000$):} $n_{\text{distinct}}{=}20.0 \pm 0.0$ on all 33 cells; \texttt{reference\_tanimoto} per-pocket $= 0.143 / 0.229 / 0.123$ on novel pockets test\_010..012; per-cell std $= 0.0$ on every diversity metric. \textbf{Recommended fix (pending, Phase 4 integrator territory):} wire \texttt{pocket\_features.modify\_root\_prior} into \texttt{MCTSProofSearch.\_root.children[child].P} at first selection; wire \texttt{LearnedPolicyPrior} with \texttt{mix\_uniform=0.5} (AlphaGo Zero root-noise mixing); replace metal-seed by pocket-conditioned reference ligand as the MCTS root. After any of these three wirings ships, re-running Path A 10$\times$3 + AlgoTune 3-cell is expected to surface per-pocket distinct candidate lists (vs current byte-identical lists).

### 1.3 §6 item (11) — NEW caveat (PathA lift does not generalize)

**Location:** `paper/sections/06_limitations.tex` line 358 (inserted between items (10) and (12))
**Change type:** NEW \item; 1-paragraph MEASURED evidence + 1-paragraph projected-not-measured gap analysis + 1-paragraph recommended fix.

**Body (verbatim from §6 after edit):**

> \item \textbf{PathA-10x3 diversity lift does \emph{not} generalise without pocket-conditioned root-prior integration (NEW 2026-09-15, WF-Paper-Content Phase 3).}
> The four-fix bundle (Path B rule symmetry + decoder rework + scaffold-aware gate + partner tiles, shipped across WF-Lambda-Rule-Symmetry-Fix + WF-Lambda-Fix-FullPath-v2 + WF-Partner-Tiles-PathA + WF-Path-B-Decoder-Rework) \textbf{MEASURED} lifted the $\Lambda$-only diversity ceiling from \texttt{n\_distinct=1} to \texttt{n\_distinct=20} on the Path A 10$\times$3 panel + AlgoTune 3-cell novel-pocket smoke. \textbf{Honest framing (caveat, NOT a regression):} the lift is \emph{uniform across pockets} (see item (10) above); the 33 underlying cells all return the same 20-SMILES chemotype basket regardless of pocket identity. \textbf{The lift is therefore a \emph{chemistry-layer diversity gain}, not a \emph{pocket-conditioned diversity gain}.} The single-seed $\Lambda$-only path carries no per-pocket prior: (a)~no per-pocket warm-start embedding at the MCTS root (\texttt{pocket\_features} module DESIGNED at Phase-3J but not integrated; Peng 2022 Pocket2Mol §3.2 cited as design source); (b)~no learned reaction-prior at the rule-dispatch channel (Phase-3L \texttt{LearnedPolicyPrior} checkpoint trained but not wired into \texttt{MCTSProofSearch.\_prior}, AlphaGo Zero root-noise mixing \texttt{mix\_uniform=0.5} as the design target per Silver 2017 §III.3); (c)~metal-seed anchors \texttt{[Pt]C\#C} override the pocket reference ligand entirely. \textbf{Projected (NOT MEASURED) lift after root-prior integration:} +5--15pp per-pocket diversity per Peng 2022 Pocket2Mol §3.2 + Silver 2018 AlphaZero dirichlet fraction $0.25$; +5--10pp applicable-rule coverage per Silver 2017 §III.3. \textbf{Gap to TargetDiff (MEASURED):} \texttt{diversity\_tanimoto} gap = $-0.7535$ (Mol-Metal $0.1065$ vs TargetDiff $0.860$); \texttt{diversity\_homotype} gap = $-0.7721$ (Mol-Metal $0.0749$ vs TargetDiff $0.847$). The 33-cell Path A + AlgoTune anchor is the load-bearing \textbf{MEASURED} number; the per-pocket distinction is \emph{not yet measured} because the root prior is not integrated. \textbf{Recommended fix (pending):} integrate Phase-3J \texttt{pocket\_features} + Phase-3L \texttt{LearnedPolicyPrior} into live \texttt{MCTSProofSearch}; re-run Path A 10$\times$3 + AlgoTune 10$\times$3 to surface per-pocket distinct candidate lists; then promote \texttt{diversity\_tanimoto} / \texttt{diversity\_homotype} cells from the byte-identical-uniform baseline to per-pocket-\textsc{measured} cells.

### 1.4 §6 item (12) — NEW caveat (5 algo fixes partial; 2 layers require proof_search.py integration)

**Location:** `paper/sections/06_limitations.tex` lines 361–370 (inserted between item (11) and the closing \end{enumerate})
**Change type:** NEW \item with 3-layer enumerate (layer (i) chemistry / (ii) MCTS cache / (iii) reward prior) + status-per-layer + 1-paragraph verdict + 1-paragraph 5-fixes-shipped + 1-paragraph what's-not-closed + 1-paragraph recommended fix.

**Body (verbatim from §6 after edit):**

> \item \textbf{Five algorithmic fixes ship but only partially lift the singleton attractor; remaining 2 layers (cache + reward) require \texttt{proof\_search.py} integration (NEW 2026-09-15, WF-Paper-Content Phase 3).}
> The internal review \texttt{molmetal/reports/wf\_lambda\_internal\_review/audit.md} (2026-09-15) named a \textbf{3-layer singleton attractor} governing the $\Lambda$-only path's collapse to \texttt{n\_distinct=1} on every cell:
>
> \begin{itemize}
>   \item \textbf{Layer (i) chemistry}: click SMARTS ignore \texttt{Pt\_II} (Pt\_II + 5 clicks not reducible). \textbf{STATUS: BROKEN} --- Fix 2(a) \texttt{MetalLigandExchange} + \texttt{AquaExchange} SMARTS rules + scaffold-aware \texttt{pt\_click\_compat} 5$\times$5 matrix + \texttt{auto-pt-strict} alias resolving to \texttt{[CuAAC, SPAAC, Suzuki]} for strict \texttt{Pt\_II}; \texttt{n\_distinct} lifted $1 \to 20$ on Path A 10$\times$3 + AlgoTune 3-cell (\textbf{MEASURED}).
>   \item \textbf{Layer (ii) MCTS cache}: \texttt{\_unreactive\_states} permanent membership at \texttt{proof\_search.py:2726}. \textbf{STATUS: UNCHANGED} --- no fix shipped in Phase 3 (T3 was DESIGNED only; cache invalidation requires \texttt{MCTSProofSearch} integration in the live harness).
>   \item \textbf{Layer (iii) reward prior}: \texttt{metal\_geometry\_prior\_bonus} hard gate. \textbf{STATUS: PARTIALLY BROKEN} --- Phase 3H \texttt{soft\_score\_metal\_geometry} shipped (continuous \texttt{d\_coord / d\_angle / d\_charge} score replaces the hard 0/1 gate) but is opt-in (default weight $0.0$, the backward-compatible hard gate remains in the live reward aggregator).
> \end{itemize}
>
> \textbf{Honest verdict:} \textbf{1 of 3 layers broken}, not 3 of 3. The remaining 2 layers (MCTS cache + reward prior) produce a fixed 20-molecule chemotype basket that does \emph{not} differentiate by pocket (see items (9) and (10)). This is the \textbf{2-layer pocket-invariance attractor} that replaces the 3-layer singleton attractor of \texttt{wf\_lambda\_internal\_review/audit.md}. \textbf{Five algorithmic fixes shipped} (Path B rule symmetry \texttt{\_run\_reactants\_symmetric} + \texttt{MCTSProofSearch.\_safe\_reduce} retry-swapped-args; Path B decoder rework \texttt{--decoder-rework} flag; scaffold-aware gate \texttt{pt\_click\_compat.py} 5$\times$5 matrix + 16 CLI aliases + \texttt{--allow-incompatible-click}; partner tiles \texttt{PARTNER\_TILES\_V2} + \texttt{FRAGMENT\_LIBRARY\_200\_TILES()} expansion to $\geq$50 azide handles; soft-tiered \texttt{metal\_geometry\_prior\_bonus} reward rebalance). \textbf{What's NOT yet closed:} cache invalidation (\texttt{proof\_search.py:2726} \texttt{\_unreactive\_states} \texttt{del} / re-key call) and reward-aggregator live integration of \texttt{soft\_score\_metal\_geometry} at default weight $1.0$. Both require \texttt{proof\_search.py} + \texttt{r4\_lambda\_only\_run.py} integration work that Phase 4 integrator owns (file-set \texttt{w8579x29t}). \textbf{Recommended fix (pending):} (a)~add \texttt{\_invalidate\_unreactive\_states(seed, child\_smiles)} call at \texttt{proof\_search.py:\_expand} entry to invalidate the cache on rule-symmetry re-swap; (b)~change \texttt{r4\_lambda\_only\_run.py} default \texttt{--metal-prior-weight} from $0.0$ to $1.0$ so the soft score actually enters the reward; (c)~re-run Path A 10$\times$3 + AlgoTune 3-cell after both wirings ship, then promote \texttt{n\_distinct} / \texttt{diversity\_tanimoto} / \texttt{diversity\_homotype} from the byte-identical-uniform baseline to per-cell-\textsc{measured} cells.

### 1.5 Closing `User-gated decisions affecting the above.` block — unchanged

The closing paragraph listing user-gated decisions D1–D7 is unchanged (the prior text referenced items (1)–(8); this remains consistent with items (1)–(12) since user-decisions D1–D7 still apply to the fix-priority-ordering of items (1)–(9) and the new items (10)–(12) inherit the same decision-gating per the brief). No edits to lines 374–380.

---

## 2. New caveats list (concise)

### 2.1 Item (10) — Pocket-invariance with warm_start/learned_prior status

- **Diagnosis (MEASURED):** all 33 cells of PathA-10x3 + AlgoTune produce byte-identical 20-SMILES candidate lists (per-cell std = 0.0 on every diversity metric).
- **Cache + reward layers active:** `_unreactive_states` permanent membership at `proof_search.py:2726` untouched; `soft_score_metal_geometry` shipped but opt-in at default weight 0.0.
- **warm_start / learned_prior:** Phase-3J `pocket_features` module + Phase-3L `LearnedPolicyPrior` checkpoint DESIGNED but NOT integrated.
- **Recommended fix:** wire `pocket_features.modify_root_prior` into `MCTSProofSearch._root.children[child].P`; wire `LearnedPolicyPrior` with `mix_uniform=0.5`; replace metal-seed by pocket-conditioned reference ligand.

### 2.2 Item (11) — PathA lift non-generalization

- **Diagnosis (MEASURED):** PathA-10x3 + AlgoTune 33-cell anchor lifted n_distinct 1→20 BUT the lift is uniform across pockets (chemistry-layer diversity gain, NOT pocket-conditioned diversity gain).
- **No per-pocket prior in Λ-only path:** no warm-start embedding, no learned reaction-prior, metal-seed overrides pocket reference entirely.
- **Projected (NOT MEASURED) lift after root-prior integration:** +5–15pp per-pocket diversity (Peng 2022 + Silver 2018); +5–10pp applicable-rule coverage (Silver 2017).
- **Gap to TargetDiff (MEASURED):** diversity_tanimoto gap = −0.7535 (0.1065 vs 0.860); diversity_homotype gap = −0.7721 (0.0749 vs 0.847).
- **Recommended fix:** integrate Phase-3J + Phase-3L into live `MCTSProofSearch`; re-run Path A 10×3 + AlgoTune 10×3; promote per-pocket distinct cells.

### 2.3 Item (12) — 5 algo fixes partial; 2 layers need proof_search.py integration

- **Diagnosis (3-layer singleton attractor audit):**
  - Layer (i) chemistry: **BROKEN** — Fix 2(a) MetalLigandExchange + AquaExchange + pt_click_compat 5×5 matrix; n_distinct lifted 1→20 (MEASURED).
  - Layer (ii) MCTS cache: **UNCHANGED** — `_unreactive_states` permanent membership at `proof_search.py:2726`; T3 was DESIGNED only.
  - Layer (iii) reward prior: **PARTIALLY BROKEN** — `soft_score_metal_geometry` shipped but opt-in at default weight 0.0.
- **Honest verdict:** 1 of 3 layers broken, not 3 of 3. 2-layer pocket-invariance attractor replaces the 3-layer singleton attractor.
- **Five algorithmic fixes shipped:** (1) Path B rule symmetry `_run_reactants_symmetric` + `_safe_reduce` retry-swapped-args; (2) Path B decoder rework `--decoder-rework`; (3) scaffold-aware gate `pt_click_compat.py` 5×5 matrix + 16 CLI aliases + `--allow-incompatible-click`; (4) partner tiles `PARTNER_TILES_V2` + `FRAGMENT_LIBRARY_200_TILES()` expansion to ≥50 azide handles; (5) soft-tiered `metal_geometry_prior_bonus` reward rebalance.
- **What's NOT closed:** cache invalidation (`proof_search.py:2726` `_unreactive_states` `del`/re-key call) + reward-aggregator live integration of `soft_score_metal_geometry` at default weight 1.0.
- **Recommended fix (3 sub-steps):** (a) add `_invalidate_unreactive_states(seed, child_smiles)` at `proof_search.py:_expand` entry; (b) change `r4_lambda_only_run.py` default `--metal-prior-weight` from 0.0 to 1.0; (c) re-run Path A 10×3 + AlgoTune 3-cell after both wirings ship.

---

## 3. Honest-framing verification

- [x] MEASURED vs PROJECTED vs CITEDONLY clearly labelled in each new item (item (10) MEASURED-only; item (11) MEASURED + PROJECTED split explicit; item (12) MEASURED + STATUS label per layer).
- [x] Honest trade-off callouts preserved: item (11) explicitly states "caveat, NOT a regression"; item (12) explicitly states "1 of 3 layers broken, not 3 of 3".
- [x] Gap-to-TargetDiff preserved: item (11) records diversity_tanimoto gap = −0.7535 and diversity_homotype gap = −0.7721 verbatim.
- [x] No silent promotion of unmeasured cells: item (11) records "the per-pocket distinction is *not yet measured* because the root prior is not integrated"; item (12) records "What's NOT yet closed" as the closing diagnosis.
- [x] Per-cell std = 0.0 finding explicit in item (10) — not framed as bug but as structural property of fixed-bias / soft-prior regime.
- [x] Pre-fix vs post-fix metal_compliance trade-off (1.0 → 0.0) NOT surfaced as a regression in §6 items (10)–(12) — that's item (9)'s canonical framing and stays there.

---

## 4. Cross-references

### 4.1 Within §6 (forward + backward)

| Item | Cross-refs forward | Cross-refs backward |
|---|---|---|
| (9) "Λ-only emission table is pocket-invariant" | (10), (11), (12) via 1-sentence bridge appended in §1.1 | §4.2.1 (sec:evaluation:novel-pocket) for 3-cell panel |
| (10) "Pocket-invariance: novel pockets produce identical candidate lists" | (11) "see item (10) above"; (12) "see items (9) and (10)" | (9) for canonical framing; §4.2.1 (sec:evaluation:novel-pocket) for AlgoTune 3-cell numbers |
| (11) "PathA-10x3 lift does not generalize without root prior" | (10) "see item (10) above"; (12) "see items (9) and (10)" | (9) + (10); §4.1 Table 1 caption (gap to TargetDiff); §4.3 Table 2 Λ-only column |
| (12) "5 algo fixes ship but only partially lift the singleton attractor" | (9) + (10) + (11) explicit reference | `wf_lambda_internal_review/audit.md` (3-layer singleton attractor source); `proof_search.py:2726` (file:line cite); Phase-3H, Phase-3J, Phase-3L (file:line + wf-name cites) |

### 4.2 §6 → §4 cross-refs (unchanged from prior rounds)

| §6 item | §4 cross-ref |
|---|---|
| (9) | §4.2.1 (sec:evaluation:novel-pocket) — 3-cell panel + per-pocket reference_tanimoto 0.143/0.229/0.123 (already integrated by Phase 2) |
| (10) | §4.2.1 (sec:evaluation:novel-pocket) — AlgoTune novel-pocket smoke 202.63s wall, 33 cells × n_distinct=20 |
| (11) | §4.1 Table 1 — diversity_tanimoto / diversity_homotype columns; §4.3 Table 2 Λ-only aggregate — gap-to-TargetDiff annotations |
| (12) | §4.5 "Path A — 4-fix bundle lifts the singleton" — already MEASURED; item (12) is the per-layer status commentary |

### 4.3 §6 → §3 cross-refs (NEW, all in item (12))

- Item (12) cites §3.2 (`sec:click-chem-selection`) for the 5×5 pt_click_compat matrix; §3.4 (`sec:mcts-betanf`) for the `MCTSProofSearch` rule-dispatch channel; §3.3 (`sec:metal-geometry-prior`) for the `metal_geometry_prior_bonus` reward rebalance.

### 4.4 §6 → §7 cross-refs (UNCHANGED)

- The closing recommendation of items (10)–(12) cross-refers §7 future work items 1–3 (Lambda × CFM coupling deferred; multi-metal extension; PCGrad multi-task loss).
- No §7 edits made (per task brief — Workflow 1 owns).

### 4.5 §6 → CROSS_REFS.md (UNCHANGED, deferred to Workflow 1)

- The existing CROSS_REFS.md entry for §6 lists 9 items; Phase 3 adds 3 more (items 10–12). Workflow 1 owner will update CROSS_REFS.md in a follow-up round to reflect the new item count.
- The cross-references in items 10–12 are self-contained (each item carries its own file:line + wf-name + report.json cite) so they remain valid even if CROSS_REFS.md is not updated in this round.

---

## 5. Files NOT touched (per task brief scope)

- `paper/sections/04_evaluation.tex` — Phase 2 already integrated Path A + AlgoTune (147 cells DESIGN→MEASURED + §4.2.1 novel-pocket subsubsection added + §6 item (9) cross-ref to §4.2.1 added). No §4 work needed for §6 items 10–12.
- `paper/sections/04_evaluation.tex` §4.6 — Workflow 5 owns.
- `paper/sections/05_ablation.tex` — Workflow 5 owns.
- `paper/sections/03_4_mcts.tex` — Phase 2 confirmed no §4 cross-refs present in §3.4, so no §3 work needed for §6 items 10–12.
- `paper/main.tex` — Workflow 1 owns.
- `paper/refs.bib` — Workflow 1 owns.
- `paper/sections/CROSS_REFS.md` — Workflow 1 owns; deferred to a follow-up round.
- `paper/sections/01_intro.tex`, `02_related.tex`, `03_1_mlc_formalism.tex`, `03_2_click_chemistry.tex`, `03_3_metal_geometry_prior.tex`, `03_method.tex`, `07_future.tex`, `appendices/*`, `supplementary.tex` — out of scope.

---

## 6. Verification checklist

- [x] §6 item (9) preserved (no edits to 9-line MEASURED-evidence block + 6-line recommended-fix block); only 1-sentence bridge appended at head to defer to items (10)/(11).
- [x] §6 item (10) NEW: pocket-invariance + warm_start/learned_prior integration status. MEASURED evidence block (33 cells × n_distinct=20, per-cell std=0.0, per-pocket reference_tanimoto 0.143/0.229/0.123). 3 fix-paths explicit.
- [x] §6 item (11) NEW: PathA-10x3 lift non-generalization. MEASURED lift + PROJECTED (NOT MEASURED) post-integration lifts + MEASURED gap-to-TargetDiff. 1 chemistry-layer vs pocket-conditioned framing explicit.
- [x] §6 item (12) NEW: 5 algo fixes ship but only partially lift singleton attractor. 3-layer enumerate (chemistry BROKEN / MCTS cache UNCHANGED / reward prior PARTIALLY BROKEN). 1-of-3 verdict + 2-layer pocket-invariance attractor framing. 5 fixes explicit + 3 fix sub-steps explicit.
- [x] Items 1–8 of §6 unchanged (preserved as-is per task brief step 1 "Add new caveat" framing — items 10–12 are NEW, not replacements).
- [x] Honest-framing preserved throughout: MEASURED vs PROJECTED vs CITEDONLY labels explicit; gap-to-TargetDiff called out; per-pocket distinction marked "not yet measured"; "1 of 3 layers broken, not 3 of 3" framing explicit.
- [x] \begin{enumerate}...\end{enumerate} structure intact: items 1–9 untouched, items 10–12 inserted at the tail of the enumerate environment.
- [x] File:line + wf-name + report.json cites preserved (proof_search.py:2726, _run_reactants_symmetric, _safe_reduce, pt_click_compat.py, PARTNER_TILES_V2, FRAGMENT_LIBRARY_200_TILES, soft_score_metal_geometry, pocket_features.py, learned_prior_Pt.pt).
- [x] Lit citations preserved (Peng 2022 Pocket2Mol §3.2, Silver 2018 AlphaZero dirichlet fraction 0.25, Silver 2017 §III.3, Himo 2005 JACS CuAAC for §3.2 cross-ref via item (12) chemistry layer).

---

## 7. Recommended follow-ups (out of scope for Phase 3)

1. **Workflow 1 follow-up:** update `paper/sections/CROSS_REFS.md` §6 entry from "9 honest-framed limitations" to "12 honest-framed limitations (items 10–12 added by Phase 3 of WF-Paper-Content)" + add per-item cross-ref map.
2. **Workflow 1 follow-up:** update §6 intro paragraph (lines 10–14) to acknowledge the 12-item enumeration (currently reads "Each item below").
3. **Workflow 5 follow-up:** consider promoting §6 item (12) status if `proof_search.py:2726` cache invalidation ships; current status (UNCHANGED layer (ii)) would shift to BROKEN.
4. **Workflow 4 (Lambda Core Features) follow-up:** ship Phase-3J `pocket_features` integration + Phase-3L `LearnedPolicyPrior` integration so items (10) and (11) status can shift from pending to in-progress; expected eta per `molmetal/reports/wf_lambda_core_features/` roadmap = Q4 2026.
5. **Workflow 7 follow-up:** after (1)–(4) close, re-run Path A 10×3 + AlgoTune 10×3 to surface per-pocket distinct candidate lists; promote §4.1 + §4.3 cells from byte-identical-uniform baseline to per-cell-MEASURED; then retire items (10)/(11) and demote item (12) to 1-of-3 broken → 3-of-3 broken.

---

**Phase 3 §6 integration complete. 3 new caveats added (items 10, 11, 12); item (9) gets 1-line bridge to defer to new items; items 1–8 preserved as-is. Honest-framing preserved throughout: MEASURED vs PROJECTED vs CITEDONLY labels explicit; gap-to-TargetDiff called out; per-pocket distinction marked "not yet measured"; "1 of 3 layers broken, not 3 of 3" verdict explicit. Cross-references to §4.2.1, §4.5, §3.2, §3.4, §7 preserved + new forward refs within §6 explicit. Workflow 1 owns `paper/sections/CROSS_REFS.md` + `paper/main.tex` updates; Workflow 5 owns §4.6 + §5 updates; Workflow 7 owns the per-pocket-distinct re-run when root-prior integration ships.**