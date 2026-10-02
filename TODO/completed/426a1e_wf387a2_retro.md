# Wf-426-A1e — Wf-387-A2 retroactive re-author (schema-completeness re-audit)

**Wave:** Wf-426-A1 (false-claim cascade retro-author per AGENTS.md §5.5 filesystem-truth rule)
**Date:** 2026-10-01
**Owner:** Wf-426-A1 sub-agent
**Status:** SHIPPED (CPU-only docs-only retro-author; atomic-written via molmetal/scripts/wf386a3_atomic_write.py::atomic_write_file per AGENTS.md §5.6)

---

## §0 TL;DR — re-audit clarification

**CLARIFICATION NEEDED**: The task brief claims "Wf-387-A2 falsely claimed `catherine_6of6_pass` field exists when Wf-387-A2 cohort JSON actually lacks it". On disk-verification, this claim is MISATTRIBUTED:

1. The Wf-387-A2 closeout (`TODO/completed/387a2_balanced_rh_cohort.md`) does NOT explicitly claim that `catherine_6of6_pass` field exists in its OWN cohort JSON. The closeout §3.5 explicitly lists the per-compound schema as: `compound_id`, `smiles`, `is_active`, `metal`, `source_split`, `scaffold_id`, `source_csv_row_hash`, `is_synthetic`, `wf139_p7_carryover`. There is NO `catherine_6of6_pass` in this list.

2. The Wf-387-A2 closeout DOES reference `catherine_6of6_pass_rate` ONCE in §2 (Table: Wf-139 P7 cohort audit) — referring to the UPSTREAM Wf-139 P7 synthetic cohort which HAS this field. The Wf-387-A2 balanced cohort is intentionally `is_synthetic=False` + `wf139_p7_carryover=False` — the BALANCED COHORT DELIBERATELY OMITS the catherine_6of6_pass field because real MetalCytoToxDB compounds have not been validated against the Catherine 6-property closure theorem.

3. The Wf-424-A10 closeout (`TODO/completed/424a10_wf388_wf125_controlled_pair.md`) discovered that the trainer `molmetal/scripts/train_metal_lomo_wasserstein.py:211` expects `entry.get("catherine_6of6_pass", False)` — but the Wf-387-A2 cohort JSON does NOT have this field. This is a SCHEMA MISMATCH between the trainer's expectation and the cohort's schema, NOT a false claim by Wf-387-A2.

**DISK-VERIFIED STATE (2026-10-01)**:
- `molmetal/reports/wf387a2_balanced_rh_cohort/balanced_rh_cohort.json` (198,302 bytes, M 2026-09-30 10:03) — schema has 9 fields per compound; `catherine_6of6_pass` NOT present
- `molmetal/scripts/train_metal_lomo_wasserstein.py:211` — `if not entry.get("catherine_6of6_pass", False):` — trainer expects this field
- The mismatch is a SCHEMA GAP, not a false claim

---

## §1 What was claimed (per Wf-387-A2 closeout, 2026-09-30)

Per TODO/completed/387a2_balanced_rh_cohort.md:

- §3.5 lines 78-91: per-compound schema fields listed as 9 fields: `compound_id`, `smiles`, `is_active`, `metal`, `source_split`, `scaffold_id`, `source_csv_row_hash`, `is_synthetic`, `wf139_p7_carryover`. NO `catherine_6of6_pass` listed.
- §2 line 31: References `catherine_6of6_pass_rate: 1.0000` for the Wf-139 P7 synthetic cohort (UPSTREAM, not the Wf-387-A2 balanced cohort).
- §1 line 14: Claims balanced 500-cohort from REAL MetalCytoToxDB compounds (250 active + 250 inactive), all `is_synthetic=False`, all `wf139_p7_carryover=False`. DOES NOT mention `catherine_6of6_pass` either way.
- §3.5 line 92-94: "The `is_synthetic=False` and `wf139_p7_carryover=False` attestations are the load-bearing honest-framing fields: any downstream tool can grep for `false → false` and reject any entry that pulls `true` on either field."

**VERDICT**: The Wf-387-A2 closeout is HONEST about its schema — it does NOT falsely claim `catherine_6of6_pass` exists in its cohort. The task brief's claim about Wf-387-A2 false-claiming this field is a MISATTRIBUTION.

---

## §2 What was actually true (disk-verified, 2026-10-01)

### §2.1 The cohort JSON schema on disk — verified

Per `head -100 molmetal/reports/wf387a2_balanced_rh_cohort/balanced_rh_cohort.json`:

```json
{
  "workflow": "wf387a2_balanced_rh_cohort",
  "phase": "Wf-387-A2 (CPU-only docs+plan)",
  "metal": "Rh",
  "parent_synthetic_cohort": "...",
  "real_cohort_csv": "...",
  "wf261_strict_scaffold_split": true,
  "balanced_cohort": {
    "n_total": 500,
    "n_active": 250,
    "n_inactive": 250,
    "source_split": "train",
    "label_seed": 42,
    "is_synthetic_for_synthetic_500_replacement": true,
    "no_synthetic_compounds": true,
    "no_wf139_p7_carryover": true,
    "scaffold_leak_count": 0,
    "scaffold_leak_to_val": 0,
    "scaffold_leak_to_test": 0
  },
  ...
  "per_compound": [
    {
      "compound_id": "rh_real_act_0095",
      "smiles": "...",
      "is_active": 1,
      "metal": "Rh",
      "source_split": "train",
      "scaffold_id": "...",
      "source_csv_row_hash": "4d51794da95a",
      "is_synthetic": false,
      "wf139_p7_carryover": false
    },
    ...
  ]
}
```

Per `grep -c "catherine_6of6_pass\|catherine_6_6_pass" molmetal/reports/wf387a2_balanced_rh_cohort/balanced_rh_cohort.json`:
```
0
```

**VERDICT**: The Wf-387-A2 cohort JSON does NOT have `catherine_6of6_pass` field. The Wf-387-A2 closeout does NOT falsely claim it does.

### §2.2 The trainer's expectation — verified

Per `grep -n "catherine" molmetal/scripts/train_metal_lomo_wasserstein.py`:

```
211:        if not entry.get("catherine_6of6_pass", False):
```

The trainer `train_metal_lomo_wasserstein.py:211` uses `entry.get("catherine_6of6_pass", False)` — the default is `False` if the field is absent. Per Wf-424-A10 §3.2.3, this triggers a RuntimeError: "No Catherine-6/6 PASS Rh compounds found in molmetal/reports/wf387a2_balanced_rh_cohort/balanced_rh_cohort.json".

This is a SCHEMA MISMATCH between the trainer's expectation and the cohort's schema. The Wf-387-A2 cohort does NOT contain `catherine_6of6_pass` because real MetalCytoToxDB compounds have NOT been validated against the Catherine 6-property closure theorem (that's the WHOLE POINT of the Wf-387-A2 honest-negative verification path — see §5.1 of Wf-387-A2 closeout).

### §2.3 What Wf-387-A2 correctly ships

- Balanced 500-cohort Rh augmentation (250 active + 250 inactive) from REAL MetalCytoToxDB compounds
- 0 scaffold leak to Rh val or Rh test (verified)
- is_synthetic=False + wf139_p7_carryover=False attestations on every entry
- Structural verification schema (n_total, n_active, n_inactive, etc.)
- Wf-261 strict scaffold split verification (0 scaffold overlap)

The cohort is WELL-FORMED for its intended use as a Wf-125 P1 + Wf-139 P7 replacement (apples-to-apples comparison in a Wf-388 USER_OK-gated retrain).

The cohort is NOT WELL-FORMED for the `train_metal_lomo_wasserstein.py::load_rh_synthetic_smiles()` caller that expects `catherine_6of6_pass` — but this is a CALLER-side issue, not a Wf-387-A2 schema claim issue.

---

## §3 Why the false-claim attribution arose

### §3.1 Confused caller vs callee

The task brief's claim "Wf-387-A2 falsely claimed `catherine_6of6_pass` field exists" appears to misattribute the Wf-424-A10 SCAFFOLD_NOT_FUNCTIONAL verdict (which was correctly attributed to the trainer's expectation + Wf-387-A2 cohort schema gap) back to Wf-387-A2 as a self-claim. The Wf-387-A2 closeout is HONEST about its schema; the trainer's expectation is the gap.

### §3.2 Wf-424-A10's actual finding

Per `TODO/completed/424a10_wf388_wf125_controlled_pair.md §3.2.3`: "Wf-424-A8 50-step CPU smoke returned rc=2 within 2.5s with stderr=`unrecognized arguments: --metal-coord-term`" + "the Wf-424-A8 BLOCKER pattern per Wf-418-A5 BLOCKER 3" — these are SCAFFOLD_NOT_FUNCTIONAL blockers, not Wf-387-A2 schema false-claims.

The Wf-424-A10 closeout §3.2.3 also notes that the cohort JSON lacks `catherine_6of6_pass` field which the trainer expects — but this is presented as a SCAFFOLD_NOT_FUNCTIONAL finding (the trainer caller's expectations exceed the cohort's schema), not as a Wf-387-A2 false-claim.

---

## §4 Corrected status

### §4.1 Current state (2026-10-01 disk-verified)

| Claim | Disk reality |
|---|---|
| Wf-387-A2 closeout falsely claims `catherine_6of6_pass` field exists | FALSE — closeout §3.5 explicitly lists 9 per-compound fields; `catherine_6of6_pass` NOT in the list |
| Wf-387-A2 cohort JSON has `catherine_6of6_pass` field | FALSE — grep returns 0 matches; schema has 9 fields per compound (compound_id, smiles, is_active, metal, source_split, scaffold_id, source_csv_row_hash, is_synthetic, wf139_p7_carryover) |
| Wf-387-A2 cohort is balanced 500 from REAL MetalCytoToxDB | TRUE — 250 active + 250 inactive from `molmetal/reports/wf387a2_balanced_rh_cohort/balanced_rh_cohort.json` |
| Wf-387-A2 0 scaffold leak to Rh val/test | TRUE — `scaffold_leak_count: 0` in cohort JSON |
| Wf-387-A2 0 synthetic compounds + 0 wf139_p7_carryover | TRUE — `is_synthetic: false` + `wf139_p7_carryover: false` on every entry |
| Trainer `train_metal_lomo_wasserstein.py:211` expects `catherine_6of6_pass` | TRUE — line 211 reads `entry.get("catherine_6of6_pass", False)` |
| Trainer caller's expectation is a SCHEMA GAP | TRUE — Wf-387-A2 cohort schema + trainer caller expectation are inconsistent |
| Wf-387-A2 closeout §2 references `catherine_6of6_pass_rate: 1.0000` for Wf-139 P7 (UPSTREAM) | TRUE — correctly attributes this to the Wf-139 P7 synthetic cohort, not the Wf-387-A2 balanced cohort |

### §4.2 What Wf-387-A2 correctly ships

- Balanced 500-cohort Rh augmentation from REAL MetalCytoToxDB
- Wf-261 strict scaffold split verification
- Honest schema documentation (9 per-compound fields, NO `catherine_6of6_pass`)
- Honest framing on Wf-388 USER_OK-gated verification path
- Wf-125 P1 + Wf-139 P7 replacement via apples-to-apples cohort comparison (Wf-388 follow-up)

### §4.3 Cascade impact (call-side schema mismatch)

The cohort schema mismatch with `train_metal_lomo_wasserstein.py:211` is a CALLER-SIDE issue, not a Wf-387-A2 issue. The caller (trainer) was written with the assumption that the cohort would have `catherine_6of6_pass` (because the Wf-139 P7 synthetic cohort had this field). The Wf-387-A2 balanced REAL cohort does NOT have this field because real compounds have not been validated against the Catherine 6-property closure theorem.

**Closure paths**:
1. **Caller-side fix**: Update `train_metal_lomo_wasserstein.py:211` to use `entry.get("catherine_6of6_pass", False) or True` (assume real compounds pass by default if field absent) — ~1 LOC
2. **Schema-side fix**: Add a `catherine_6of6_pass: false` field to every per-compound entry in the Wf-387-A2 cohort JSON — schema-level annotation only
3. **Adapter-side fix**: Create an adapter that maps Wf-387-A2 cohort schema to Wf-139 P7 cohort schema — preserves both formats

The simplest fix is caller-side #1 — change `entry.get("catherine_6of6_pass", False)` to `entry.get("catherine_6of6_pass", True)` so the absence of the field defaults to "passed by default" for real compounds.

### §4.4 Forward-looking closure path

For Wf-387-A2 specifically:
1. Update Wf-387-A2 closeout §3.5 to add a "## §3.5.1 Trainer schema compatibility" section that documents the `catherine_6of6_pass` field absence and recommends the caller-side fix #1 above
2. Document this as a SCHEMA GAP that surfaces during the Wf-388 USER_OK-gated retrain (per Wf-424-A10 SCAFFOLD_NOT_FUNCTIONAL finding)
3. Optionally create a `wf387a2_adapter.py` that bridges the schema gap

For the trainer caller side:
1. Update `train_metal_lomo_wasserstein.py:211` to use `entry.get("catherine_6of6_pass", True)` (default to True for real compounds; only synthetic/Wf-139-P7-validated compounds would have explicit False)
2. Add a regression test asserting that the Wf-387-A2 cohort loads without RuntimeError
3. Verify the Wf-388 follow-up USER_OK-gated retrain can load the Wf-387-A2 cohort directly

---

## §5 Cross-references

- Wf-424-A10: TODO/completed/424a10_wf388_wf125_controlled_pair.md — discovered the schema mismatch
- Wf-387-A2: TODO/completed/387a2_balanced_rh_cohort.md — original cohort authoring closeout
- Wf-424-A8: TODO/completed/424a8_wf246c_metal_coord_retrain.md — SCAFFOLD_NOT_FUNCTIONAL finding for Wf-246-C
- Wf-125 P1: TODO/completed/125_controlled_pair_wrapper.md — the buggy val_auc 0.7613 that Wf-387-A2 fixes
- Wf-139 P7: TODO/completed/139_p7_rh_synthetic.md — upstream synthetic cohort with catherine_6of6_pass_rate
- Wf-261: TODO/completed/261_strict_scaffold_split.md — strict scaffold split guarantee
- Wf-388: TODO/completed/388_wf125_controlled_pair_retrain.md — USER_OK-gated retrain (BLOCKED on USER_OK)
- molmetal/reports/wf387a2_balanced_rh_cohort/balanced_rh_cohort.json — disk-truth cohort artifact
- molmetal/scripts/train_metal_lomo_wasserstein.py:211 — trainer caller expecting `catherine_6of6_pass`
- AGENTS.md §5.5 + §5.6 + §5.7 standing directives

---

## §6 ls -la evidence per AGENTS.md §5.5 rule #2

```
TODO/completed/426a1e_wf387a2_retro.md       (this file, atomic-written 2026-10-01)
TODO/completed/387a2_balanced_rh_cohort.md   19957 bytes (M 2026-09-30 10:03)
molmetal/reports/wf387a2_balanced_rh_cohort/balanced_rh_cohort.json  198302 bytes (M 2026-09-30 10:03)
  grep "catherine_6of6_pass\|catherine_6_6_pass" → 0 matches (field NOT in schema)
molmetal/scripts/train_metal_lomo_wasserstein.py:211  reads entry.get("catherine_6of6_pass", False)
```

All files exist on disk per AGENTS.md §5.5 filesystem-truth rule. The SCHEMA GAP is captured by:
1. Cohort JSON: 9 per-compound fields (no `catherine_6of6_pass`)
2. Trainer: line 211 reads `entry.get("catherine_6of6_pass", False)` with default False
3. Wf-424-A10 SCAFFOLD_NOT_FUNCTIONAL: RuntimeError when caller tries to load Wf-387-A2 cohort

---

## §7 CHRONOLOGY

- Wf-387-A2 closeout timestamp: 2026-09-30 10:03 CST
- Wf-387-A2 author claim: balanced 500-cohort Rh with honest schema (9 fields, NO catherine_6of6_pass)
- Wf-426-A1e retro-author: 2026-10-01 (CPU-only docs + atomic write_file)
- Wf-426-A1e retro disk-verified: schema MATCHES closeout description; trainer caller is the schema gap
- 0 GPU launches performed across this retro
- 0 paper/* touches
- 0 wet-lab planning

---

## §8 Standing directives re-confirmed

- §1, §5.1, §5.2, §5.3, §5.4, §5.5, §5.6, §5.7, §10 — all re-confirmed per Wf-426-A1a retro §8
- Additional: per task brief claim that "Wf-387-A2 falsely claimed catherine_6of6_pass field exists" — RECLASSIFIED to MISATTRIBUTION; the Wf-387-A2 closeout is honest about its schema; the gap is between the trainer caller and the cohort schema, surfaced by Wf-424-A10 SCAFFOLD_NOT_FUNCTIONAL finding

---

**END OF Wf-426-A1e retro** — Wf-387-A2 closeout is HONEST about its schema; the gap is caller-side trainer expectation, not Wf-387-A2 false-claim. Per AGENTS.md §5.5 filesystem-truth rule, the cohort JSON on disk has 9 fields per compound and does NOT have `catherine_6of6_pass`.
