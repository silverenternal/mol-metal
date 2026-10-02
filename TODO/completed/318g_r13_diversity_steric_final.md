# Wf-318G: r13_100x3_partial + diversity_subpocket + steric_clash_pb_dock — FINAL honest-negative close-out

**Date**: 2026-09-29
**Workflow**: Wf-318G (Wf-311F §7 ultracode follow-up candidates #3 + #4 FINAL close-out)
**Skill alignment**: `party-twenty-spirit` §3.5 (MEASURED-vs-DEFERRED tag discipline)
**Status**: ✅ **SHIPPED — 3 workstreams CLOSED as honest-negative for Tier-2 scope; pre-flight 49/49 PASS**

---

## §0 Why this doc exists

Per `AGENTS.md` §5.4 (2026-09-29 user directive "我们发二区就可以了，不用规划湿实验"),
the project re-orients from Tier-1 (JCIM) to Tier-2 (JCAMD / J. Mol. Graph. Model.
/ Comput. Chem. WIREs / Mol. Divers. / Comput. Biol. Chem.) journal target.
**Tier-2 acceptance bar is computational-only**; wet-lab validation is a
6–12 month external-dependency that does NOT gate Tier-2 acceptance.

Wf-311F §7 surfaced 7 ultracode follow-up candidates. Per `TODO/completed/314a_wf311e_part4_deterministic_edits.md`
§7, the candidates are:

1. m3 LOMO 5/5 × 5-seed × 50-ep full sweep — DEFERRED_GPU (Wf-145 SHIPPED)
2. Wf-219 Ru per_metal_summary aggregation fix — CPU-only
3. **r13_100x3_partial paper-grade 300 cells — CPU-only** ← **Wf-318G closes**
4. **Wf-311E diversity_subpocket + steric_clash + com_shift staging — DESIGN_ONLY / DEFERRED / NOT-staged** ← **Wf-318G closes 2 of 3**
5. Phase-3G metal_coord_compliance 10×3 rotation — BLOCKED on TODO-29 F2(a) + GPU
6. OOD cross-domain supplement RDKit env fix — CPU-only

Wf-318G closes candidates #3 + #4 (the 3 workstreams r13_100x3_partial +
diversity_subpocket + steric_clash_pb_dock) as **honest-negative for Tier-2
scope**. Candidate #4's com_shift sub-item (1h36 / test_001 NOT staged per
Wf-121 deferral) is a separate DEFERRED workstream not covered by Wf-318G.

---

## §1 Files shipped (Wf-318G)

| # | Path | Type | Edit |
|---|---|---|---|
| 1 | `TODO/completed/r13_100x3_final_honest_negative.md` | Doc | NEW — workstream #1 closeout |
| 2 | `TODO/completed/diversity_subpocket_final_honest_negative.md` | Doc | NEW — workstream #2 closeout |
| 3 | `TODO/completed/steric_clash_pb_dock_final_honest_negative.md` | Doc | NEW — workstream #3 closeout |
| 4 | `TODO/completed/318g_r13_diversity_steric_final.md` | Doc | NEW — this file |
| 5 | `metrics/by_metric/r13_100x3_partial.json` | JSON | NEW — mirrors `metrics/by_round/r13_100x3_partial.json` + adds `HONEST_NEGATIVE=true` + `honest_negative_reason` + `honest_framing_note` + `verdict: FAIL` schema |
| 6 | `metrics/by_metric/diversity_subpocket.json` | JSON | +4 fields (`HONEST_NEGATIVE=true`, `honest_negative_reason`, `honest_framing_note`, `wf_design_ref`) preserving all existing fields including `verdict: DESIGN_ONLY` |
| 7 | `metrics/by_metric/steric_clash_pb_dock.json` | JSON | +5 fields (`HONEST_NEGATIVE=true`, `honest_negative_reason`, `honest_framing_note`, `wf_design_ref`, `verdict: FAIL`) preserving `status: DEFERRED` |
| 8 | `molmetal/scripts/paper_dashboard.py` | Python | +3 HONEST_NEGATIVE_CATALOG entries (`hn_wf318g_r13_100x3_not_paper_grade`, `hn_wf318g_diversity_subpocket_design_only`, `hn_wf318g_steric_clash_pb_dock_830c_deferred`); counter bumps (`n_new_honest_negatives_added` 13 → 16, `new_honest_negative_ids` list +3 IDs) |

**n_closeout_docs = 4** (3 workstreams + 1 master summary)
**n_metrics_updated = 3**
**n_honest_negatives_added = 3** (HONEST_NEGATIVE_CATALOG)
**n_counter_bumps = 2** (`n_new_honest_negatives_added`, `new_honest_negative_ids`)

---

## §2 Per-workstream honest framing (party-twenty-spirit §3.5)

### §2.1 r13_100x3_partial — NEGATIVE_RESULT_HONEST

- **MEASURED on disk**: `metrics/by_round/r13_100x3_partial.json` carries
  `verdict.status = NEGATIVE_RESULT_HONEST` since 2026-09-15; this Wf-318G
  pass promotes the round-level verdict into the paper_dashboard.py §9
  honest-negative catalog + adds the canonical schema to a new
  `metrics/by_metric/r13_100x3_partial.json` mirror file.
- **Verbatim values cited**: `n_evaluations_total = 30` (10% of spec 300);
  `pb_path_evaluated = 30`; `pb_path_n_no_candidates = 15`;
  `pb_path_n_seed_only = 15`; `pb_path_pb_pass_rate_aggregate = null`;
  `pb_path_wall_seconds_mean = 7.75`; `cfm_path_status = BLOCKED`;
  `paper_grade_data_ready = false`.
- **Recovery path**: TODO-29 F2(a) MetalLigandExchange SMARTS rule +
  GPU HSA userland recovery + GPU pipeline Phase 3 (R16-W42-W44).

### §2.2 diversity_subpocket — NEGATIVE_RESULT_HONEST

- **MEASURED on disk**: `metrics/by_metric/diversity_subpocket.json` carries
  `verdict: DESIGN_ONLY` since 2026-09-15; this Wf-318G pass adds the
  canonical `HONEST_NEGATIVE=true` schema + preserves the existing
  `verdict: DESIGN_ONLY` field + promotes the metric into the §9 catalog.
- **Verbatim values cited**: smoke-only test fixtures
  (`empty=NaN, N=1=0.0, 2-disjoint>0`); `aggregate.all_measured_lambda_diversity_subpocket
  = DESIGN only (no full sweep yet)`; Phase-3D module + 5 unit tests + Phase
  4 wire-in SHIPPED.
- **Recovery path**: TODO-29 F2(a) MetalLigandExchange SMARTS rule +
  Phase-3G 10×3 rotation + GPU pipeline Phase 3 (R16-W44).

### §2.3 steric_clash_pb_dock (830c pocket) — NEGATIVE_RESULT_HONEST

- **MEASURED on disk**: `metrics/by_metric/steric_clash_pb_dock.json` carries
  `status: DEFERRED` since 2026-09-18; this Wf-318G pass adds the canonical
  `HONEST_NEGATIVE=true` + `verdict: FAIL` schema + preserves `status:
  DEFERRED` + `deferred_reason: deferred_posebusters_unavailable_AND_830c_docked_poses_missing`.
- **Verbatim values cited**: 830c pocket `value.rate = null`,
  `value.n_total = 100`, `receptor_pdb = /mnt/disk1/Hugo_Lee/mol-metal/molmetal/data/mmp13_real/830c.pdb`;
  1h36 pocket `m8_protein_aware_pass_rate = 1.0000 (15/15 pass)` per
  `metrics/by_metric/steric_clash_pb_dock_1h36.json::MEASURED_SMOKE`
  (Wf-315D 2026-09-29 cross-reference).
- **Recovery path**: 830c receptor PDB staging + 830c docked poses staging +
  PoseBustersAdapter.dock() recompute (RDKit 2026.03.6 + PoseBusters 0.6.5
  NOW available per Wf-315D verification).
- **Important scope note**: 1h36 MEASURED_SMOKE = 1.0000 (15/15) is
  **independent of this close-out** and remains Tier-2-shippable as the
  per-pose metric citation.

---

## §3 Verification

### §3.1 JSON parse (3/3 PASS)
```
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -c "import json; [json.load(open(f'metrics/by_metric/{n}.json')) for n in ['r13_100x3_partial','diversity_subpocket','steric_clash_pb_dock']]; print('JSON parse: 3/3 PASS')"
JSON parse: 3/3 PASS
```

### §3.2 AST parse (dashboard)
```
$ /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -c "import ast; ast.parse(open('molmetal/scripts/paper_dashboard.py').read()); print('AST parse: PASS')"
AST parse: PASS
```

### §3.3 Pre-flight check (TODO-207 v2 gate)
```
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python molmetal/scripts/pre_flight_check.py
{
  "summary": {
    "pass": 49,
    "fail": 0,
    "skip": 0,
    "total": 49,
    "total_wall_sec": 250.4369
  },
  ...
}
```

**Result**: 49/49 PASS (no regression; was 45/45 PASS pre-Wf-318G; expansion
reflects 4 new honest-negative schema gates). Pre-flight PASS floor
maintained per AGENTS.md §1.4.

---

## §4 Honest framing checklist (party-twenty-spirit §3.5)

### §4.1 MEASURED vs PROJECTED vs DEFERRED
- **MEASURED**: All edits are pure JSON tag additions + paper_dashboard.py
  catalog entries + closeout docs. No metric values fabricated.
- **PROJECTED**: N/A — no compute, no projection.
- **DEFERRED**: r13_100x3_partial paper-grade 300 cells (DEFERRED to R16-W42-W44),
  diversity_subpocket production 100-pocket sweep (DEFERRED to R16-W44),
  steric_clash_pb_dock 830c (DEFERRED to R16-W45 + 830c staging).

### §4.2 0 容忍 fabricate check
- ✅ No duplicate metric values invented
- ✅ No silent skip
- ✅ Honest-negative framing preserved (mirrors Wf-308B aqueous_solubility_logS
  + Wf-311E Part 4 + Wf-314A pattern verbatim)
- ✅ Existing verdict fields preserved (`verdict: DESIGN_ONLY` for
  diversity_subpocket; `status: DEFERRED` for steric_clash_pb_dock 830c)
- ✅ 1h36 MEASURED_SMOKE preserved (NOT flagged honest-negative)

### §4.3 §3.4 高质量发展 (凑数 check)
- ✅ Each closeout doc is necessary for §9 honest-negative catalog wiring
- ✅ No decoration without function (all 3 metrics cited are Wf-311F §7
  candidates #3 + #4)
- ✅ Recovery paths documented but NOT executed (CPU-only doc-edit + JSON
  tag + paper_dashboard.py wiring)

### §4.4 R20 §11 self-check (Wf-318G specific)
1. **哪个 mechanism 实际动了代码?** — N/A (no mechanism M22-M44 change;
   pure data + dashboard catalog wiring).
2. **哪些 metric 受影响?** — `r13_100x3_partial` + `diversity_subpocket`
   + `steric_clash_pb_dock` (3 honest-negative Tier-2 deferred metrics).
3. **0 GPU retrain shipped 这条限制 paper §6 limitations 写了吗?** — N/A
   (this Wf is CPU-only deterministic edits; no GPU).
4. **Wave 1/2/3 ship status 准确?** — N/A (this Wf is not R18 W1/W2/W3;
   not a mechanism wave).
5. **23 个 mechanism cite key 都进 paper/refs.bib 了吗?** — N/A (this Wf
   adds 0 new mechanisms).

---

## §5 Cross-references (verified)

| Wf | Path | Relation |
|---|---|---|
| Wf-311F | `molmetal/reports/wf311_synthesis/paper_4_final_state.md` | §7 ultracode follow-up candidates #3 + #4 source |
| Wf-314A | `TODO/completed/314a_wf311e_part4_deterministic_edits.md` | §7 (Wf-311F candidates list verbatim) + Wf-311E Part 4 deterministic edit pattern |
| Wf-314F | (synthesis doc, per task brief) | Cited in closeout docs §5 |
| Wf-315D | `metrics/by_metric/steric_clash_pb_dock.json::notes` + `metrics/by_metric/steric_clash_pb_dock_1h36.json` | RDKit + PoseBusters availability cross-ref (2026-09-29) |
| Wf-317 | (Tier-2 strategy doc) | Per Wf-317, 4 LOIs + 3 SOPs are TEMPLATE_NOT_NEEDED_FOR_TIER2; this Wf extends the principle to 3 DEFERRED metric workstreams |
| Wf-317G | `TODO/completed/317g_wetlab_sop_tier2_marker.md` | Tier-2 marker pattern (Wf-318G applies analogous marker to metrics) |
| Wf-308B | `metrics/by_metric/aqueous_solubility_logS.json` | HONEST_NEGATIVE schema precedent |
| Wf-311E Part 4 / Wf-314A | `metrics/by_metric/{m6_kappa,hepatotox_index,logp7_4}.json` | HONEST_NEGATIVE schema precedent (m6_kappa + hepatotox + logp7_4) |
| TODO-14 | `TODO/completed/14_full_100pocket_paper_r13.md` | Round-13 master (where the r13_100x3_partial partial deliverable is preserved) |
| TODO-26 | `TODO/pending/26_round13_round14_complete_plan.md` | R16 master plan (forward-pointer for recovery paths) |
| TODO-29 | `TODO/completed/29_f2a_closed.md` | F2(a) MetalLigandExchange SMARTS rule (structural singleton-attractor fix) |
| AGENTS.md | §1.4, §5.1, §5.4 | Pre-flight gate + kill_then_adapt + Tier-2 journal target |
| party-twenty-spirit | §3.5 | MEASURED-vs-DEFERRED tag discipline; data真实 |

---

## §6 Files touched summary (Wf-318G)

| Path | Type | Edit |
|---|---|---|
| `TODO/completed/r13_100x3_final_honest_negative.md` | Doc | NEW |
| `TODO/completed/diversity_subpocket_final_honest_negative.md` | Doc | NEW |
| `TODO/completed/steric_clash_pb_dock_final_honest_negative.md` | Doc | NEW |
| `TODO/completed/318g_r13_diversity_steric_final.md` | Doc | NEW (this file) |
| `metrics/by_metric/r13_100x3_partial.json` | JSON | NEW (mirror of by_round + HONEST_NEGATIVE schema) |
| `metrics/by_metric/diversity_subpocket.json` | JSON | +4 fields (preserve existing verdict) |
| `metrics/by_metric/steric_clash_pb_dock.json` | JSON | +5 fields (preserve existing status) |
| `molmetal/scripts/paper_dashboard.py` | Python | +3 HONEST_NEGATIVE_CATALOG entries + counter bump (13→16) + 3 IDs append |

**Total edits**: 4 NEW docs + 1 NEW JSON + 2 JSON tag additions + 1 dashboard
catalog update (3 entries + 2 counter bumps). **Zero GPU launches, zero
compute, zero fabricated numbers.**

---

## §7 Remaining Wf-311F §7 candidates (NOT covered by Wf-318G)

Per Wf-314A §7 (5 remaining candidates):

1. m3 LOMO 5/5 × 5-seed × 50-ep full sweep — DEFERRED_GPU (Wf-145 SHIPPED;
   user-OK needed per AGENTS.md §5.1)
2. Wf-219 Ru per_metal_summary aggregation fix — CPU-only (separate
   workstream)
3. com_shift 1h36 / test_001 NOT staged per Wf-121 deferral — separate
   DEFERRED workstream
4. Phase-3G metal_coord_compliance 10×3 rotation — BLOCKED on TODO-29 F2(a)
   + GPU outage (separate workstream)
5. OOD cross-domain supplement RDKit env fix — CPU-only (separate workstream)

Wf-318G closes 2/7 Wf-311F §7 candidates (#3 + #4 partial). Remaining
5 candidates are scoped for separate workstreams.

---

*Generated 2026-09-29 by Wf-318G close-out. 3 workstreams CLOSED as
honest-negative for Tier-2 scope; pre-flight 49/49 PASS; dashboard reflects
+3 NEW honest-negative catalog entries + +2 counter bumps.*
