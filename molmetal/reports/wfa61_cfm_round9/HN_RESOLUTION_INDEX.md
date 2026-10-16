# Wf-A61 CFM Round 9 — HN strengthen/resolve ledger

| §6.x | Action | File | Status |
|---|---|---|---|
| §6.55g | strengthen — cross-YAML rk4 config audit | `molmetal/reports/wfa61_cfm_round9/configs_method_audit.json` | SHIPPED |
| §6.52d | strengthen — Vina panel BLOCKED-on-every-precious-metal | `molmetal/reports/wfa61_cfm_round9/vina_block_audit.json` | SHIPPED |
| §6.79b | strengthen — chemprop_v2 vs D-MPNN schema-version paper §4 rule (b.2 + b.3 + b.4 + b.5 sub-entries) | `docs/HONEST_NEGATIVE_CATALOG.md` (this wave) | SHIPPED |
| §6.54f/g | strengthen — W₂ lift paper-claim rule + G6-ablation empirical anchor (§6.54i) | `docs/HONEST_NEGATIVE_CATALOG.md` (this wave) | SHIPPED |
| §6.81j | NEW — Wf-A58 W10-A fork_rng determinism gate | `docs/HONEST_NEGATIVE_CATALOG.md` (this wave) | SHIPPED |
| §6.81k | NEW — Wf-A60 W12-C1 F401 cleanup | `docs/HONEST_NEGATIVE_CATALOG.md` (this wave) | SHIPPED |
| §6.81l | NEW — Wf-A60 W12-C3 importlib refactor + path 17 | `docs/HONEST_NEGATIVE_CATALOG.md` (this wave) | SHIPPED |
| §6.81m | NEW — Wf-A61 L1/L2/L13/L14/L18/L21 fixes (5 NEW footguns closed) | `docs/HONEST_NEGATIVE_CATALOG.md` (this wave) | SHIPPED |
| §6.81n | NEW — Wf-A61 W1 CFM deep audit round 9 (5 NEW footguns + 4 verify-clean + 7 sub-entries) | `docs/HONEST_NEGATIVE_CATALOG.md` (this wave) | SHIPPED |
| §6.81o | NEW — Wf-A61 W2 HN strengthen/resolve round 3 closeout anchor | `docs/HONEST_NEGATIVE_CATALOG.md` (this wave) | SHIPPED |
| §6.88 | NEW — Wf-A61 W2 top-level closeout entry (cumulative count math + file-on-disk receipts) | `docs/HONEST_NEGATIVE_CATALOG.md` (this wave) | SHIPPED |
| HN-cfm-property-panel | RESOLVED — memory retire (Wf-A55 W7-P4 receipt + Wf-A61 W2 confirmation NO-OP) | `~/.claude/projects/.../memory/cfm-property-panel-measures-no-model.md:44-69` | SHIPPED |
| §6.59h | MEASURED — fm_velocity pocket conditioning structural fix | `train_cfm_yuelbond_wasserstein.py:1864-1971` | CITED |
| §6.59i | MEASURED commit-ref anchor — `f7a730ca` (Wf-A60 W3 §6.87) + Wf-A32 | `docs/HONEST_NEGATIVE_CATALOG.md` | CITED |
| §6.66h | MEASURED — pocket_per_layer_injector wire-in | `train_cfm_yuelbond_wasserstein.py` (Wf-A32 commit) | CITED |
| §6.66i | MEASURED commit-ref anchor — Wf-A32 + Wf-A41 W2 (C14/C11/C12/F4/I.1) | `docs/HONEST_NEGATIVE_CATALOG.md` | CITED |

## W2 closeout summary (2026-10-17)

| Deliverable | MEASURED |
|---|---|
| `molmetal/reports/wfa61_cfm_round9/configs_method_audit.json` | 15 rk4 cohorts (pytest `test_configs_method_audit_rk4_count` PASS) |
| `molmetal/reports/wfa61_cfm_round9/vina_block_audit.json` | `n_rows == 6` + `all_blocked == true` (pytests PASS) |
| `molmetal/tests/test_wfa61_cfm_round9.py` | 4/4 PASS in ~0.2s |
| §6.79b.3..b.5 | 3 NEW sub-entries (prose-rule + metric-name + MEASURED code-level anchor) |
| §6.81o + §6.88 | 2 NEW catalog closeout anchors |
| Memory retire | NO-OP per Wf-A55 W7-P4 receipt (line 44 + 69 already on disk) |
| Catalog count delta | **+1** (§6.88 entry; ACTIVE count unchanged at 109) |

Mirror of `cfm_todo.json::_arch_audit_log_2026_10_17_Wf-A61_W2` per `memory/registry-is-the-status-ssot.md` + AGENTS.md §5.5 filesystem-truth.