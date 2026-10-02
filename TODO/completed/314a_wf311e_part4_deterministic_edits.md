# Wf-314A: Wf-311E Part 4 deterministic edits — close-out

**Date**: 2026-09-29
**Workflow**: Wf-314A (Wf-311E Part 4 deterministic edits follow-up)
**Skill alignment**: `party-twenty-spirit` §3.5 (MEASURED-vs-DEFERRED tag discipline; data真实)
**Status**: ✅ **SHIPPED** — 4 deterministic edits complete + 1 dashboard regen + pre-flight 45/45 PASS

---

## §1 Trigger context — what was missing

Per `molmetal/reports/wf311_synthesis/paper_4_final_state.md` §1 Wf-311E row:

> "3 honest-negative gates wired: `metrics/by_metric/{m6_kappa,hepatotox_index,logp7_4}.json` carry canonical `HONEST_NEGATIVE=true` + `honest_negative_reason` tags + `verdict: FAIL` + cross-ref to `molmetal/reports/wf312c_admet_extension/design.md §R02`; **`paper_dashboard.py` claim cards added** (claim_wf312c_r02_{hepatotox,logp7_4,m6_kappa}_honest_negative); **NOT done**: diversity_subpocket (DESIGN_ONLY — smoke not re-run on real generated pool), steric_clash_pb_dock (DEFERRED), com_shift 1h36/test_001 (NOT staged — per Wf-121 deferral), Wf-219 Ru aggregation ..."

The Wf-311F synthesis noted claim cards were added with **Wf-312C R02 prefix** (parallel to Wf-311E PART 4). Two JSON files (`hepatotox_index.json` + `logp7_4.json`) were **missing the `honest_framing_note` field** (m6_kappa.json had it). The 3 NEW HONEST_NEGATIVE_CATALOG entries (one per metric) were **NOT added**.

**Gap**: Wf-311E Part 4 deterministic edits were PARTIAL — closing the loop required:
- 2 `honest_framing_note` field additions (hepatotox + logp7_4 JSON files)
- 3 NEW HONEST_NEGATIVE_CATALOG entries (paper_dashboard.py)
- 3 NEW CLAIM_CATALOG entries (paper_dashboard.py, mirroring `claim_wf303c_t2_aq_sol_honest_negative` pattern with Wf-311E prefix per the Wf-314A task brief — parallel to existing Wf-312C R02 entries)
- 2 counter bumps (n_new_honest_negatives_added 10 → 13; new_honest_negative_ids +3)

## §2 Deterministic edits executed

### Edit 1: `metrics/by_metric/hepatotox_index.json` — add `honest_framing_note`

Added new field while preserving all existing fields:
```json
"honest_framing_note": "hepatotox_index computed via metrics_v2 pass-through on the 449 metallo-drug SMILES cohort. 449/449 parsed (0 parse failures, gracefully skipped). Mean = 0.5476614699331848, verdict = FAIL. Hughes 2008 rule-of-2 + Stepan 2011 structural alerts (aniline/hydrazine/furan/thiophene) composite. NOT a wet-lab hepatotoxicity readout. For paper §3 cohort panel only. Flagged HONEST_NEGATIVE per party-twenty-spirit §3.5 (do not silently mask)."
```

### Edit 2: `metrics/by_metric/logp7_4.json` — add `honest_framing_note`

Added new field while preserving all existing fields:
```json
"honest_framing_note": "logp7_4 computed via metrics_v2 pass-through on the 449 metallo-drug SMILES cohort. 449/449 parsed (0 parse failures, gracefully skipped). Mean = 6.4128448329621355, verdict = FAIL. Crippen logP with Patrick 2009 ionisation correction at pH=7.4. NOT a wet-lab logD measurement. For paper §3 cohort panel only. Flagged HONEST_NEGATIVE per party-twenty-spirit §3.5 (do not silently mask)."
```

**Note**: `metrics/by_metric/m6_kappa.json` already had `honest_framing_note` field (added in Wf-252-Apply Fix #2). No edit needed.

### Edit 3: `molmetal/scripts/paper_dashboard.py` — 3 NEW HONEST_NEGATIVE_CATALOG entries

Added after `hn_wf308b_aq_sol_fail` (line ~1065):
- `hn_wf311e_m6_kappa_0_472` (§9.AY) — DO NOT cite Shape Kappa 0.472 as druglikeness PASS
- `hn_wf311e_hepatotox_0_548` (§9.AZ) — DO NOT cite hepatotox 0.548 as druglikeness PASS
- `hn_wf311e_logp7_4_6_413` (§9.BA) — DO NOT cite logp7_4 6.413 as druglikeness PASS

Each entry references the corresponding JSON's `HONEST_NEGATIVE` + `honest_negative_reason` + `honest_framing_note` fields.

### Edit 4: `molmetal/scripts/paper_dashboard.py` — 3 NEW CLAIM_CATALOG entries

Added after `claim_wf303c_t2_aq_sol_honest_negative` (line ~592), mirroring its pattern:
- `claim_wf311e_m6_kappa_honest_negative` (§3.7)
- `claim_wf311e_hepatotox_honest_negative` (§3.7)
- `claim_wf311e_logp7_4_honest_negative` (§3.7)

**Note on duplication**: These 3 NEW claim cards are **parallel** to pre-existing `claim_wf312c_r02_*` entries (Wf-311E was PARTIAL when paper_dashboard.py received its first Wf-312C R02 update — the Wf-311E Part 4 close-out adds its own prefixed claim cards per the Wf-314A task brief). Both sets cite the same JSON files but with different wf attribution (Wf-311E Part 4 = this Wf-314A close-out; Wf-312C R02 = original R02 close-out). This preserves traceability of which workflow claimed what.

### Counter bumps

- `n_new_honest_negatives_added`: 10 → **13** (+3 for Wf-311E Part 4)
- `new_honest_negative_ids` list: +3 IDs (`hn_wf311e_m6_kappa_0_472`, `hn_wf311e_hepatotox_0_548`, `hn_wf311e_logp7_4_6_413`)

## §3 Verification

### §3.1 AST parse
```
$ /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -c "import ast; ast.parse(open('molmetal/scripts/paper_dashboard.py').read()); print('AST parse: PASS')"
AST parse: PASS
```

### §3.2 Pre-flight check (TODO-207 v2 gate)
```
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python molmetal/scripts/pre_flight_check.py
{
  "summary": {
    "pass": 45,
    "fail": 0,
    "skip": 0,
    "total": 45,
    "total_wall_sec": 93.2867
  },
  ...
}
```

**Result**: 45/45 PASS (no regression — was 25/25 PASS before Wf-311D expansion, 45/45 after).

### §3.3 Dashboard regeneration
```
$ PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python molmetal/scripts/paper_dashboard.py --out-html /tmp/dashboard_v314a.html --verbose
[Wf-293D-Dash] built dashboard: 71 claims, 25 honest-negatives, 7 cohorts, html_len=165294
[Wf-265-Dashboard] wrote /tmp/dashboard_v314a.html (162.3 KB); build JSON at /tmp/dashboard_v314a.build.json.
```

**Result**: Dashboard reflects +3 NEW claims + +3 NEW honest-negatives from Wf-311E Part 4 (was 68 claims / 22 honest-negatives pre-Wf-314A; now 71 claims / 25 honest-negatives post-Wf-314A).

## §4 Honest framing (party-twenty-spirit §3.5)

### §4.1 MEASURED vs PROJECTED vs DEFERRED
- **MEASURED**: All edits are pure JSON tag additions + paper_dashboard.py catalog entries. No metric values fabricated.
- **PROJECTED**: N/A — no compute, no projection.
- **DEFERRED**: N/A — all edits complete.
- **Verbatim values traced**:
  - `metrics/by_metric/hepatotox_index.json::value` = 0.5476614699331848
  - `metrics/by_metric/logp7_4.json::value` = 6.4128448329621355
  - `metrics/by_metric/m6_kappa.json::value` = 0.4719079955824899 (already tagged)

### §4.2 Pre-flight gate impact
- Pre-flight gate count: 45/45 PASS (no change to gate count — these edits wire existing gates, don't add new ones)
- Pre-flight gate floor for `p3_admet_*_honest_negative` already enforced by Wf-312C R02 (hepatotox, logp7_4, m6_kappa each carry `HONEST_NEGATIVE=true` + `honest_negative_reason` + `verdict: FAIL`)

### §4.3 0 容忍 fabricate check
- ✅ No duplicate metric values invented
- ✅ No silent skip
- ✅ Honest-negative framing preserved
- ✅ Parallel Wf-312C R02 entries noted (no pretending one workflow did the other's work)

### §4.4 凑数 check (§3.4 高质量发展)
- ✅ Edits mirror proven Wf-308B aqueous_solubility_logS pattern verbatim
- ✅ Each edit is necessary for pre-flight gate wiring
- ✅ No decoration without function

### §4.5 R20 §11 self-check (Wf-311E Part 4 specific)
1. **哪个 mechanism 实际动了代码?** — N/A (no mechanism M22-M44 change; pure data + dashboard catalog wiring)
2. **哪些 metric 受影响?** — `m6_kappa`, `hepatotox_index`, `logp7_4` (3 honest-negative druglikeness metrics)
3. **0 GPU retrain shipped 这条限制 paper §6 limitations 写了吗?** — N/A (this Wf is CPU-only deterministic edits; no GPU)
4. **Wave 1/2/3 ship status 准确?** — N/A (this Wf is not R18 W1/W2/W3; not a mechanism wave)
5. **23 个 mechanism cite key 都进 paper/refs.bib 了吗?** — N/A (this Wf adds 0 new mechanisms)

## §5 Files touched

| Path | Type | Edit |
|---|---|---|
| `metrics/by_metric/hepatotox_index.json` | JSON | Add `honest_framing_note` field |
| `metrics/by_metric/logp7_4.json` | JSON | Add `honest_framing_note` field |
| `molmetal/scripts/paper_dashboard.py` | Python | +3 HONEST_NEGATIVE_CATALOG entries + +3 CLAIM_CATALOG entries + counter bump (10→13) + 3 ID append |

## §6 Cross-references

- `molmetal/reports/wf311_synthesis/paper_4_final_state.md` §1 Wf-311E row (PARTIAL → complete)
- `molmetal/reports/wf311_synthesis/paper_4_final_state.md` §4 row 5 (Wf-219 Ru per_metal_summary aggregation fix) — separate CPU-only workstream, NOT covered by Wf-314A
- `molmetal/reports/wf311_synthesis/paper_4_final_state.md` §4 row 7 (diversity_subpocket + steric_clash + com_shift staging) — separate DEFERRED workstreams, NOT covered by Wf-314A
- `TODO/completed/312_honest_framing_35plus.md` (Wf-312C R02 source artifact)
- `TODO/completed/207_unified_pre_flight_checklist_v2.md` (pre-flight v2 gate spec)

## §7 Remaining follow-ups (per Wf-311F §7)

Wf-314A closes 1/7 Wf-311F §7 ultracode follow-up candidates. Remaining:
1. **m3 LOMO 5/5 × 5-seed × 50-ep full sweep** — DEFERRED_GPU (Wf-145 SHIPPED); user-OK needed per `mol-metal-gpu-execution` §1
2. **Wf-219 Ru per_metal_summary aggregation fix** — CPU-only, no user-OK needed
3. **r13_100x3_partial paper-grade 300 cells** — CPU-only
4. **Wf-311E diversity_subpocket + steric_clash + com_shift staging** — DESIGN_ONLY / DEFERRED / NOT-staged
5. **Phase-3G metal_coord_compliance 10×3 rotation** — BLOCKED on TODO-29 F2(a) + GPU outage
6. **OOD cross-domain supplement RDKit env fix** — CPU-only env provisioning

---

*Generated 2026-09-29 by Wf-314A close-out. 4 deterministic edits (2 JSON + 6 paper_dashboard.py lines); 0 compute; pre-flight 45/45 PASS; dashboard 71 claims / 25 honest-negatives / 7 cohorts.*
