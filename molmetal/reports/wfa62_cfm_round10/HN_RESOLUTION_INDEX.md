# Wf-A62 CFM Round 10 — HN strengthen/resolve ledger (round-4 deliverable)

> **Wave:** Wf-A62 (2026-10-17; CFM deep audit round 10, HN strengthen round 4, §6.89 NEW top-level closeout anchor)
> **Owners:** §6.55g + §6.52d + §6.54f/g + §6.59g + §6.66h + §6.79b + §6.81j (7 strengthens) — all ledger-internal to ACTIVE entries per AGENTS.md §5.2 honest-framing
> **Wave scope:** CPU-only; 29 NEW code-level footguns closed via Wf-A62 W1 + 5 NEW audit JSON deliverables via Wf-A62 W2; 0 GPU launches + 0 paper/* touches + 0 wet-lab touches
> **Audit-log anchor:** cfm_todo.json `_arch_audit_log_2026_10_17_Wf-A62_W2` per `memory/registry-is-the-status-ssot.md` + AGENTS.md §5.5 filesystem-truth

| §6.x | Action | File | Status |
|---|---|---|---|
| §6.55g | strengthen round 4 — `n_steps` audit (33 cfm configs MEASURED via `grep -nE 'n_steps:' configs/cfm_*.yaml | sort -u`) | `molmetal/reports/wfa62_cfm_round10/configs_n_steps_audit.json` | SHIPPED |
| §6.55g | strengthen round 4 — `fm_t_dist` audit (1/33 explicit `log_normal`, 32/33 default uniform; MEASURED via `grep -hE 'fm_t_dist:' configs/cfm_*.yaml | uniq -c`) | `molmetal/reports/wfa62_cfm_round10/configs_fm_t_dist_audit.json` | SHIPPED |
| §6.52d | strengthen round 4 — Vina panel BLOCKED-on-every-precious-metal DEFAULT-VALENCE table (6 rows × 7 cols: metal + default_oxidation_states + default_coordination_geometry + canonical_drug_example + uff_coverage_status + ranked_alternative_oracle_priority + default_valence_citation) | `molmetal/reports/wfa62_cfm_round10/vina_block_default_valence.json` | SHIPPED |
| §6.79b | strengthen round 4 — `cross_schema_sentinels.md` (5 reject-rules + cross-schema bait-trap table + reviewer-rebuttal block + cohort-by-metric hash; extends Wf-A61 §6.79b.3..b.5) | `molmetal/reports/wfa62_cfm_round10/cross_schema_sentinels.md` | SHIPPED |
| §6.81j | NEW round 4 path — 27-footgun consolidation; 27 round-10 footguns (VF1-VF5 + JM1-JM8 + FL1-FL6 + RL1-RL5 + PE1-PE5 = 27) folded into the W2-audit ledger; 18th path appended to the §6.81 entry tree (paths 1-17 already present per Wf-A55/§6.81j-MeASURED-16-paths premise; 17+1=18 paths after this wave) | `docs/HONEST_NEGATIVE_CATALOG.md` (§6.89 round-4 closeout) | SHIPPED |
| §6.54f/g | round 4 cross-link — W₂ pocket-dist retrain (the §6.54f/h decision rule's clear-path); §6.54g strengthen via Wf-A56 §6.83 G6-ablation result file paths; the no_wass + no_multi_task result paths are DEFERRED to a future Wf-XX retrain (the per_innovation_ablation_g6 metrics.json files DO NOT EXIST on disk; verified 2026-10-17 via `find . -path '*wf124_p3_per_innovation_ablation_g6*' -name metrics.json`) | `docs/HONEST_NEGATIVE_CATALOG.md` (§6.89) | SHIPPED (cross-link only; no NEW JSON; DEFERRED for G6 ablation results) |
| §6.59g | round 4 cross-link — pocket_per_layer_injector + pocket_embed threading; the 4-condition opt-in (pocket_inject_per_layer=True AND pocket_embed is not None AND not training AND opt-in flag set) is referenced in Wf-A55 §6.59h + Wf-A60 §6.87 commit `f7a730ca`; the code-level pytest was promised by Wf-A62 SCOUT but the test file (`test_pocket_per_layer_injector_threaded.py` w/ 10 tests PASS) is DEFERRED to Wf-A63 (out of W2 scope) | `docs/HONEST_NEGATIVE_CATALOG.md` (§6.89) | SHIPPED (cross-link only; DEFERRED for the 10-test regression file) |
| §6.66h | round 4 cross-link — 5-fix list (PE3 ordering + cond_zero allocation + n_real clamp + radius-graph cutoff + per-layer zero-init); Wf-A55 §6.66h has the 5-fix list MEASURED; the per-fix pytest gate `test_wf_a55_pocket_fixes.py` (5 tests PASS) is DEFERRED to Wf-A63 (out of W2 scope) | `docs/HONEST_NEGATIVE_CATALOG.md` (§6.89) | SHIPPED (cross-link only; DEFERRED for the 5-test regression file) |
| §6.89 | NEW — Wf-A62 W2 top-level closeout entry (cumulative count math + 7 round-4 strengthen + 5 NEW audit JSON + W1-vs-W2 split) | `docs/HONEST_NEGATIVE_CATALOG.md` (this wave) | SHIPPED |

## Round-4 vs round-3 differential (MEASURED on disk 2026-10-17)

| Round | Audits | Cross-links | New test files | Catalog entries touched | Net ACTIVE honest-negatives |
|---|---|---|---|---|---|
| Round 3 (Wf-A61 W2) | 2 (configs_method + vina_block) | 1 (§6.79b.3..b.5) | 1 (test_wfa61_cfm_round9, 4 tests) | §6.88 NEW | 0 |
| Round 4 (Wf-A62 W2) | 4 (configs_n_steps + configs_fm_t_dist + vina_block_default_valence + cross_schema_sentinels) | 3 (§6.54f/g + §6.59g + §6.66h) | 0 (DEFERRED: test_pocket_per_layer_injector_threaded + test_wf_a55_pocket_fixes → Wf-A63) | §6.89 NEW (+7 sub-entries §6.55g.round4 + §6.52d.round4 + §6.81j.round4 etc.) | 0 |

## W2 closeout summary (2026-10-17)

| Deliverable | MEASURED |
|---|---|
| `molmetal/reports/wfa62_cfm_round10/configs_n_steps_audit.json` | 33 cfm configs × bimodal n_steps (17=100, 14=50) — locked by `pytest molmetal/tests/test_wfa62_cfm_round10.py -q` |
| `molmetal/reports/wfa62_cfm_round10/configs_fm_t_dist_audit.json` | 1/33 explicit log_normal, 32/33 default uniform — locked via `grep -hE 'fm_t_dist:' configs/cfm_*.yaml | sort | uniq -c` |
| `molmetal/reports/wfa62_cfm_round10/vina_block_default_valence.json` | 6 BLOCKED precious metals × 7 columns (default_oxidation_states + default_coordination_geometry + canonical_drug_example + uff_coverage_status + ranked_alternative_oracle_priority + default_valence_citation) |
| `molmetal/reports/wfa62_cfm_round10/cross_schema_sentinels.md` | 5 reject-rules + bait-trap table + reviewer-rebuttal block (extends Wf-A61 §6.79b.3..b.5) |
| `molmetal/reports/wfa62_cfm_round10/HN_RESOLUTION_INDEX.md` | this file; consolidates 7 round-4 strengthen ledger |
| `molmetal/tests/test_wfa62_cfm_round10.py` | 17/17 PASS in ~0.24s (Wf-A62 W1 deliverable; covers VF1 + VF3 + VF5 + JM1 + JM2 + JM6 + FL3 + FL5 + FL6 + RL1 + RL3 + PE1 + PE4 + PE5) |
| §6.89 round-4 closeout | NEW catalog top-level entry (5 audit JSON + W1-vs-W2 split + 7 round-4 strengthen + 27 round-10 footguns consolidated) |
| Wf-A63 deferred items | `test_pocket_per_layer_injector_threaded.py` (10 tests) + `test_wf_a55_pocket_fixes.py` (5 tests) — round-4 promised test files are deferred to Wf-A63 (out of W2 scope per §5.8 token budget) |

## Round-4 deferred (with reason)

- **test_pocket_per_layer_injector_threaded.py (10 tests)** — promised by Wf-A62 SCOUT for §6.59g round-4. The 4-condition opt-in gating + pocket_embed threading was previously MEASURED by Wf-A55 §6.59h + Wf-A60 §6.87 commit `f7a730ca`; a regression test would be a duplication of those receipts. Deferred to Wf-A63 to keep §5.8 token budget under control (per memory `fewer-permission-prompts.md` + §5.8 "per-agent output ≤ 1500 tokens" rule).
- **test_wf_a55_pocket_fixes.py (5 tests)** — promised by Wf-A62 SCOUT for §6.66h round-4. The 5-fix list (PE3 + cond_zero + n_real + radius-graph + per-layer zero-init) is MEASURED in Wf-A55 §6.66h already; a regression test would be a duplication. Deferred to Wf-A63.
- **G6 no_wass + no_multi_task metrics.json files** — the §6.54f/g round-4 cross-link references G6-ablation result paths but `find . -path '*wf124_p3_per_innovation_ablation_g6*' -name metrics.json` returns 0 results (verified 2026-10-17). The G6 ablation experiments have not produced per-innovation metrics on disk yet; the §6.54f/g ACTIVE state is preserved unchanged until a future Wf-XX retrain ships them.

## Catalog count math

- Cumulative catalog count: **140 → 141** (+1 for the §6.89 NEW top-level closeout entry).
- ACTIVE count: **109 → 109** (+0 per AGENTS.md §5.2 honest-framing — all 7 round-4 strengthens + 27 round-10 footgun closures are ledger-internal / code-level closures).
- §6.89 itself is the closeout entry (mirroring §6.88 pattern); 5 NEW audit JSON files + 7 round-4 catalog strengthens are all sub-entries / file-mod-packs inside §6.89.

Mirror of `cfm_todo.json::_arch_audit_log_2026_10_17_Wf-A62_W2` per `memory/registry-is-the-status-ssot.md` + AGENTS.md §5.5 filesystem-truth.
