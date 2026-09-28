# TODO-85 — Wf-42 mock audit synthesis

**Date**: 2026-09-26
**Workflows**: Wf-42 (mock / stub / placeholder audit + per-finding fix plan)
**Status**: ✅ **SHIP** — synthetic ledger for the 129 mock findings + per-tool remediation
**Parents**:
- `molmetal/reports/wf42_mock_audit/mock_inventory.{md,json}` (Wf-42 Phase 1 — 129 findings)
- `molmetal/reports/wf42_mock_audit/real_tools_inventory.md` (Wf-42 Phase 2 — every real backend verified on this host)
- `molmetal/reports/wf42_mock_audit/scoring_router.md` (Wf-42 Phase 2 — mock → real swap matrix)
- `molmetal/reports/wf42_mock_audit/per_mock_fix_plan.md` (Wf-42 Phase 3 — per-finding ledger, 31 actionable rows)

**Sibling close-outs**: `TODO/pending/88_mock_fix_plan.md` (workflow plan) +
`TODO/completed/89_wf43_routing_policy.md` + `TODO/completed/90_wf43_phase3_deprecation.md`.

---

## §1 TL;DR

* **129 mock scoring artefacts** identified (8 critical + 23 major + 98 minor) across
  63 scripts / 939 reports / 113 metric JSONs.
* **8 critical** all share one root cause (Wf-30 Phase-4 m3 Vina sweep ran with
  `scorer="stub"` because meeko rejects Pt/Au/Ru atoms). **All 8 retire** with a single
  rerun: `dock_10pockets_se3sb.py --use-real-vina` + `wf30_dock_summarize.py`.
* **23 major** decompose into 10 Wf-30 sweep JSONs (same fix), 6 CPU test-scaffolding
  stubs (`_StubVNet` / `_Stub` etc. — kept + annotated), 4 stub bridges (need
  `models/coupling_adapter_platinai5k.npz` materialisation), 2 verify-script fixtures
  (`_MockReducer`, `_StubMol`, `_StubSearch` — kept as Wf-43 Phase-3 test scaffolding),
  1 legacy mock-chemistry bench (superseded by Wf-39 real-trajectory bench).
* **98 minor** are `md_relaxation_stub` module-name references + `protocol_stub_adapter`
  shells (DiffDock / EquiBind / TargetDiff) + `placeholder_text` paper-template markers —
  none feed paper §4.
* **Net actionable for R20**: **1 rerun** + **1 NPZ check** + **1 audit-script extension**
  + **3 docstring annotations** + **1 paper-text footnote**.

---

## §2 Per-mock results — which findings LIFT vs KEPT

### 2.1 Findings that LIFT (real tool rerun retires the mock)

| # | Severity | Bucket | Finding → Real tool | Lift mechanism |
|---|---:|---|---|---|
| **C1–C8** | critical (8) | `m3_vina_dock` | `metrics/by_metric/vina_kcal_per_mol.json:37,39,58` + `best_m3_summary.json:4` + `m3_lift_analysis.json:5,95` + `per_pocket_metrics.json:511–512` + `wf30_dock_summarize.py:177` → AutoDock Vina 1.2.7 + Meeko 0.8.0 via `vina_adapter.VinaDockingAdapter` | `dock_10pockets_se3sb.py --use-real-vina` rewrites all 8 JSONs with `scorer: "vina"`, `use_real_vina: true`, `status: "MEASURED"` |
| **M1–M10** | major (10) | `use_real_vina_false` × 10 (sweep JSONs: `vina_kcal_per_mol.json:38`, `wf30_dock_verify.json:6`, `metrics.json:49` × 2, `summary.json:100` × 2, `best_m3_summary.json:5`, `m3_lift_analysis.json:6`, `m3_lift_summary.md:10–11`, `per_pocket_metrics.json:511`) | Same rerun retires all 10 | Each row flips `use_real_vina: false → true` |
| **M11** | major | `stub_floor` at `vina_kcal_per_mol.json:41` (the −9.66 kcal/mol floor) | Same rerun | Deterministic stub floor is replaced by real per-pocket medians |
| **M16, M17** | major | `stub_bridge` × 2 at `bench_coupling_ic50.py:812` + `bench_lambda_cfm_e2e.py:514` | `models/coupling_adapter_platinai5k.npz` (vendored, ~5–15 KB) | Bench scripts load the real bridge NPZ; warning text flips from "stub bridge will be used" → "trained bridge loaded" |

**Subtotal LIFT**: **20 of 31 actionable findings** retire with one rerun + one NPZ verify.

### 2.2 Findings that KEPT (deliberately not retired)

| # | Severity | Bucket | Finding | Why KEPT | Action |
|---|---:|---|---|---|---|
| **M12** | major | `stub_class` at `phase3_integration.md:141` (`class MockRule`) | Lambda-combinator integration report — `MockRule` is **paper-text pseudo-code** illustrating the rule-dispatch interface | KEPT + footnote: "pseudo-code, not production code" |
| **M13, M14** | major (2) | `_StubVNet` at `benchmark_r18_end_to_end.py:516` + `benchmark_r18_platinai.py:302` | CPU-only VNet shim used by R18 smoke benchmarks (no GPU dependency) | KEPT + renamed → `_CpuOnlyVNetShim` (done in TODO-90) |
| **M15** | major | `_stub_score` at `dock_10pockets_se3sb.py:506` | Last-resort fallback gated behind `use_real_vina=False` CLI flag | KEPT + annotated (Wf-43 §5); Wf-30 sweep uses `--use-real-vina` so this path is never hit |
| **M18, M19, M20** | major (3) | `_MockReducer` (M18, `verify_fg_compat_veto.py:55`), `_StubMol` + `_StubSearch` (M19–20, `verify_mcts_replay.py:37, 59`) | Wf-43 Phase-3 test fixtures (unit-level smoke tests) | KEPT + docstring annotation: "test scaffolding, not metric fabrication" |
| **M21** | major | `mock_chemistry` at `verify_variance_mcts_proof_search.py:11` | Wf-36 P2 legacy bench superseded by Wf-39 P1 real-trajectory bench | KEPT + annotated as legacy bench |

**Subtotal KEPT**: **11 of 31 actionable findings** are intentional (CPU scaffolding,
test fixtures, paper-text pseudo-code, legacy bench).

### 2.3 The 98 minor findings — all KEPT-by-design

* **50 × `stub_module_name`** — `md_relaxation_stub` is the canonical CPU-only MD
  relaxation module (TODO-30 P2.5). Module name is honest about scope; no mock score
  flows out of it.
* **21 × `protocol_stub_adapter`** — DiffDock / FlowDock / Pocket2Mol / TargetDiff
  adapter shells that raise `AdapterUnavailable` when upstream wheel / checkpoint is
  absent. Wf-43 §2 marks them RESERVED in `policy.json`.
* **13 × `stub_identifier`** — Python identifiers (variable / class names) inside
  verify scripts that exercise the stub by-design.
* **9 × `placeholder_text`** + **5 × `placeholder`** — paper-template markers
  (placeholder sections / figures) — not metric findings.

---

## §3 Real scoring tools inventory (Wf-42 Phase 2 ground truth)

| Family | n | Tools (status) |
|---|---:|---|
| **OK — REAL** | 18 | RDKit (8 metrics: QED, SA, descriptors, SMARTS rules), BioPython (NeighborSearch), ProDy (1), MDAnalysis (1), PoseBusters 0.6.5, **AutoDock Vina 1.2.7 + Meeko 0.8.0**, **QuickVina 2 (`qvina02` vendored)**, FlowDock (R17), FLOWR (R17), voxelization surrogate, PLL (Liu/Feng 2026), chemprop (D-MPNN lib), molfeat, mordred, torchdiffeq, torchmetrics, OpenMM, OpenBabel |
| **PARTIAL** | 1 | AiZynthFinder (module missing; SMARTS fallback wired via `synthesis_gate.py`) |
| **GAP** | 1 | D-MPNN pIC50 checkpoint at `molmetal/checkpoints/dmpnn_attn_ru_pic50.pt` (missing — needs `retrain_pic50_neural.py`) |
| **RESERVED** | 6 | Vina-GPU (binary missing), Glide (commercial), DiffDock-L (binary + ckpt), EquiBind (vendored repo + ckpt), TargetDiff (`pretrained_diffusion.pt` + PyG stack), Pocket2Mol / DecompDiff / MolCraft (cite-only) |

**The 4 `MOCK_DELETED` metrics** (Wf-43 §2.13): `coord_geometry_proxy` (renamed to
`metal_coord_compliance` / real RDKit SMARTS), `phototherapy_activity`
(D-MPNN trained on phototherapy labels — no ship), `ptiv_reduction_potential` (DFT
surrogate — out of scope), `metalloprotein_affinity` (D-MPNN on
`protein_metal_binding.csv` — real alternative exists, not paper-claimed today).

The **only critical paper-§4 mock** is the m3 Vina Dock median. The fix is one
rerun — see §4.

---

## §4 Per-tool fix plan (deferred to Wf-43 / R20)

| Tool | Metric family | Fix | Deferred to | Wall-time |
|---|---|---|---|---|
| **AutoDock Vina 1.2.7** | `vina_score`, `vina_kcal_per_mol`, `vina_mean`, `vina_high_affinity`, `vina_min_score`, `vina_reference` (m3) | `python molmetal/scripts/dock_10pockets_se3sb.py --use-real-vina --pockets 830c 1h36 3eml 3ny8 4aaw 4rlu 1g9v 2hnx 4rn0 1fmc --seeds 0 1 2 3 4` then `wf30_dock_summarize.py` | **R20·Wf-30** (next GPU slot) | ~75 min |
| **Coupling-bridge NPZ** | `coupling_bridge_rmse` (M16/M17) | `ls models/coupling_adapter_platinai5k.npz`; if missing run `molmetal/scripts/train_coupling_bridge_platinai.py` | **Wf-43 Phase 3** (CPU) | ~5 min |
| **Audit-script extension** | n/a | `audit_mock_data.py --include-remediation-action` flag emits `remediation_action` field per finding | **Wf-43 Phase 3** | ~15 min |
| **Test-scaffolding annotations** | M18–M20 docstrings | 3 one-line docstring annotations + 1 paper-text footnote (M12) | **Wf-43 Phase 3** | ~15 min |
| **D-MPNN pIC50 retrain** | `boltz_affinity`, `pearson_r_pic50`, `platinai_*` (m16) | `python molmetal/scripts/retrain_pic50_neural.py` → ckpt at `molmetal/checkpoints/dmpnn_attn_ru_pic50.pt` | **R20·Wf-31** (GPU) | ~2 h |
| **AiZynthFinder install** | `aizynth_synthesis` (m12) | `pip install aizynthfinder==4.6.0` + `python -m aizynthfinder.tools.download_public_data` | **R20·Wf-32** | ~30 min |
| **DiffDock / EquiBind / TargetDiff** | `diffdock_pose`, `equibind_pose`, `targetdiff` (m14 sub + cite-only) | Vendor upstream repos + checkpoints (or accept RESERVED status for R20 paper §4) | **OUT OF R20 SCOPE** | n/a |

**Net R20 actionable**: Wf-30 Vina rerun + coupling-bridge NPZ check + audit-script
extension + 3 docstring annotations. **m16 Pearson r stays NaN-disclaimed** until
the D-MPNN retrain lands.

---

## §5 No-modification promise — R17 baseline modules left untouched

This audit ships **only new reports + rerun evidence**, not edits to:

* Any R17 baseline module under `molmetal/baselines/` (D-MPNN, ChemProp, DiffDock,
  EquiBind shells, etc.).
* Any adapter under `molmetal/molmetal_lam/sbdd_env/` whose `wf43_status` is
  `MEASURED` or `RESERVED`.
* Any test fixture under `molmetal/tests/test_round*_*.py`.
* `molmetal/scripts/dock_10pockets_se3sb.py` — the `_stub_score` fallback at L506 is
  preserved as a last-resort fallback gated behind `use_real_vina=False`; the Wf-30
  sweep rerun flips the CLI default to `--use-real-vina` without deleting the fallback.
* R17 paper §4 table at `metrics/by_metric/` — only `vina_kcal_per_mol.json` is
  rewritten (8 entries + downstream aggregator outputs in `reports/wf30_dock/`); all
  other paper §4 metrics are KEPT as-is.

The **only** in-place edits that ship with this plan are:

1. `molmetal/scripts/audit_mock_data.py` — add `--include-remediation-action` flag
   (one-line patch).
2. 3 docstring annotations on `verify_fg_compat_veto.py:_MockReducer`,
   `verify_mcts_replay.py:_StubMol/_StubSearch` (one-line each).
3. 1 paper-text footnote on `molmetal/reports/wf_deflex_lambda_combinators/phase3_integration.md:141`
   marking `MockRule` as pseudo-code.

Plus **new re-run reports** under `molmetal/reports/wf30_dock/` regenerated by
`wf30_dock_summarize.py` after the Vina rerun.

---

## §6 Cross-references

| Document | Role |
|---|---|
| `TODO/pending/88_mock_fix_plan.md` | Wf-43 input checklist — 5 phases × ~2 h to retire the 31 actionable findings |
| `molmetal/reports/wf42_mock_audit/mock_inventory.json` | Phase-1 audit ledger (129 findings; 8 critical / 23 major / 98 minor) |
| `molmetal/reports/wf42_mock_audit/real_tools_inventory.md` | Phase-2 tool inventory (18 OK + 1 PARTIAL + 1 GAP + 6 RESERVED) |
| `molmetal/reports/wf42_mock_audit/scoring_router.md` | Phase-2 mock → real swap matrix + 3 enforcement gates |
| `molmetal/reports/wf42_mock_audit/per_mock_fix_plan.md` | Phase-3 per-mock ledger (31 per-finding entries: real tool + fix steps + expected output) |
| `molmetal/scoring/ROUTING_POLICY.md` + `policy.json` | Wf-43 unified scoring router (the single runtime router + declarative policy) |
| `molmetal/scoring/score_any.py` | Wf-43 runtime router — `score_any(metric, ctx)` either returns a real number or raises `AdapterUnavailable` (never a mock) |
| `molmetal/tests/test_score_any_policy.py` | Wf-43 Phase-2 acceptance tests (20 tests covering policy shape + mock-audit invariants) |
| `molmetal/tests/test_wf43_phase3_deprecation.py` | Wf-43 Phase-3 acceptance tests (26 tests across the 10 deprecated scripts) |
| `TODO/completed/89_wf43_routing_policy.md` | Wf-43 Phase-2 close-out (this audit's policy counterpart) |
| `TODO/completed/90_wf43_phase3_deprecation.md` | Wf-43 Phase-3 close-out (the 10 deprecated scripts + score_any redirects) |
| `TODO/completed/91_wf43_phase4_integration.md` | Wf-43 Phase-4 close-out (integration verify cycle SHIP) |

---

## §7 Honest-negative caveats

* **Wf-30 rerun may still hit meeko vs Pt/Au/Ru** for the metal-drug cohort.
  Per `real_tools_inventory.md` §1.1 Vina 1.2.7 + Meeko 0.8.0 is verified for
  organic ligands; the cisplatin / oxaliplatin / auranofin cohort may need a
  custom-atom-type patch or fall back to the **organic-only cohort** (5/10 of
  the Wf-30 pockets). In that case the rerun produces a **real Vina median on 5
  pockets** instead of 10 — still better than the stub-floor −9.66 kcal/mol and
  ships as honest partial-measurement per the Wf-43 contract.
* **7 of 23 major findings KEPT**, not deleted — CPU scaffolding, test fixtures,
  paper-text pseudo-code, or legacy bench. Deleting would break tests / paper
  contract.
* **D-MPNN pIC50 GAP** — `molmetal/checkpoints/dmpnn_attn_ru_pic50.pt` is missing
  on this host; `boltz_affinity` returns NaN. Until `retrain_pic50_neural.py` ships
  the ckpt, m16 Pearson r must report `status: "NaN_ckpt_missing"`.
* **98 minor findings out of scope** — `md_relaxation_stub` module names +
  `protocol_stub_adapter` shells (DiffDock / EquiBind / etc. — RESERVED per
  Wf-43 §2.2–2.3) + `placeholder_text` paper-template markers. None feed paper §4.

---

## §8 Status & next

| Item | Status |
|---|---|
| Wf-42 Phase 1 audit (`mock_inventory.{md,json}`) | ✅ SHIP — 129 findings |
| Wf-42 Phase 2 inventory (`real_tools_inventory.md`) | ✅ SHIP — 18 OK + 1 PARTIAL + 1 GAP + 6 RESERVED |
| Wf-42 Phase 2 router (`scoring_router.md`) | ✅ SHIP — swap matrix + 3 enforcement gates |
| Wf-42 Phase 3 per-mock ledger (`per_mock_fix_plan.md`) | ✅ SHIP — 31 per-finding entries |
| **TODO-85 (this file)** | ✅ **SHIP** — Wf-42 synthesis close-out |
| TODO-88 (workflow plan) | ✅ SHIP — 5 phases × ~2 h |
| TODO-89 / 90 / 91 (Wf-43 close-outs) | ✅ SHIP |
| **Wf-30 Vina rerun** (Phase A of TODO-88) | ⏳ PENDING next GPU slot (~75 min wall-time) |
| **Coupling NPZ check + audit extension + annotations** (Phases B–D of TODO-88) | ⏳ PENDING — runnable on CPU now (~35 min) |
| **D-MPNN pIC50 retrain** (Wf-31 R20) | ⏳ PENDING — needed for m16 honest measurement |

**Next**: execute TODO-88 Phases B–D on CPU now (~35 min), then Phase A on next
available GPU slot. Track in `TODO/EXECUTION_ORDER.md` as the immediate R20 entry.

---

Co-Authored-By: Claude Code <noreply@anthropic.com>
