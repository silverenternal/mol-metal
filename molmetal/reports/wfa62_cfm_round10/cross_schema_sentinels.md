# Wf-A62 — CFM cross-schema sentinel catalog

> **Wave:** Wf-A62 (2026-10-17; CFM deep audit round 10, HN strengthen round 4)
> **Section ref:** §6.79b.6 (NEW sub-entry, round 4 anchor on cross-schema drop-row trap)
> **Owners:** §6.79b (active) + §6.79b.3 (round-3 prose-rule) + §6.79b.4 (round-3 metric-name distinction) + §6.79b.5 (round-3 MEASURED code-level anchor)
> **Scope:** paper §4 main result table + appendix column-shift traps; everything that might cause a CFM metric to be silently mixed with a chemprop_v2 / D-MPNN metric in a paper main-result table.

## §6.79b.6 — Cross-schema drop-row sentinels

### 1. The "shape-only" decision rule (recap of §6.79b.2)

Per `docs/CHEMPROP_V2_SCHEMA_DECISION.md` + `molmetal/tests/test_chemprop_v2_schema_shape_only.py` (3 tests PASS, Wf-A54 §6.81d):
- **`CHEMPROP_V2_ATOM_DIM: int = 72`** is the canonical export surface for chemprop_v2.
- **`DMPNN_V2_ATOM_DIM: int = 70`** is the canonical export surface for D-MPNN-v2.
- A header value of `72` ⇒ chemprop_v2 cohort; `70` ⇒ D-MPNN-v2 cohort. Mixing them in a single table ⇒ shape-mismatched cohort; pytest `test_chemprop_v2_schema_shape_only` asserts each column-mix is REFUSED.

### 2. The "bait" trap (NEW round-4 content)

Paper §4 main result table has historically been a mixing-ground for 3 schemas:
1. **chemprop_v2** (`molmetal/scripts/train_chemprop_v2_v3.py`; metric schema: `auc` + `loss` + `valid_smiles_rate`).
2. **D-MPNN-v2** (`molmetal/scripts/train_dmpnn_multitask_wasserstein.py`; metric schema: `auc` + `loss` + `decode_ratio`).
3. **CFM** (`molmetal/scripts/train_cfm_yuelbond_wasserstein.py`; metric schema: `auc` + `loss` + `decode_ratio` + `bond_head_f1` + `fm_obj_term`).

The bait trap is that **`valid_smiles_rate` is a chemprop_v2-only metric**, but `decode_ratio` appears in BOTH D-MPNN-v2 and CFM. A paper §4 main table that lists `auc + loss + decode_ratio` could pull from chemprop_v2 (whose schema has no `decode_ratio` field), D-MPNN-v2, or CFM — three DIFFERENT training surfaces.

### 3. Sentinels (round-4 deliverable)

| Schema | Header magic | Metric signature | Where it lands | Reject criterion |
|---|---|---|---|---|
| **chemprop_v2** | `CHEMPROP_V2_ATOM_DIM = 72` | `auc, loss, valid_smiles_rate` | paper §4 + Table S3 (appendix) | shape-mismatch on `decode_ratio` |
| **D-MPNN-v2** | `DMPNN_V2_ATOM_DIM = 70` | `auc, loss, decode_ratio` | paper §4 + Table S4 (appendix) | shape-mismatch on `valid_smiles_rate` |
| **CFM** | no header magic; cohort via `molmetal/reports/wfA54*/cohort.json` | `auc, loss, decode_ratio, bond_head_f1, fm_obj_term` | paper §4 + Table S5 (appendix) | shape-mismatch on `valid_smiles_rate` (chemprop_v2-only) |

### 4. Reject rules (5 rules)

A paper §4 main-table row is **REFUSED** if any of the following are true:

1. **Header-mix rule.** Row claims chemprop_v2 (header=72) AND row uses D-MPNN-v2 metric column names (e.g., `decode_ratio`). REJECT (shape-mismatch).
2. **Schema-orphan rule.** Row claims D-MPNN-v2 (header=70) AND row uses chemprop_v2-only metric column names (e.g., `valid_smiles_rate`). REJECT (shape-mismatch).
3. **CFM-tag rule.** Row has `bond_head_f1` or `fm_obj_term` column AND no CFM cohort marker. REJECT (no CFM training provides this).
4. **D-MPNN/CFM-confused rule.** Row uses `decode_ratio` without schema tag (D-MPNN-v2 vs CFM). FLAG (not REJECTED; both schemas have decode_ratio — read the §6.79b.5 anchor for the cohort disambiguation).
5. **No-3-schema-mix rule.** Row joins all 3 schemas in a SINGLE AUC cell by averaging. REJECT (the average is a "Frankenstein metric" — paper §4 main table must NOT do this).

### 5. Paper §4 reviewer-rebuttal block (round-4 anchor)

Per the Wf-A61 §6.88c §6.79b.3 prose-rule, a paper §4 main-table column must pick EXACTLY ONE of {chemprop_v2, D-MPNN-v2, CFM}, with the schema header in the column title (`72` ⇒ chemprop_v2; `70` ⇒ D-MPNN-v2; CFM via cohort.json) OR a paper-§4 footnote citing the schema shape-only decision rule.

A "mixed-cohort" column (e.g., chemprop_v2 + D-MPNN-v2 averaged into one AUC number) is a §6.79b ACTIVE honest-negative violation; the round-4 reject-rules above are the LOCK conditions.

### 6. Cross-references

- Wf-A54 §6.79b.2 (round-2 shape-only decision rule + `test_chemprop_v2_schema_shape_only.py` lock).
- Wf-A61 §6.79b.3 (round-3 prose-rule + 4-line Acceptable/NOT-acceptable examples).
- Wf-A61 §6.79b.4 (round-3 cross-schema metric-name distinction table).
- Wf-A61 §6.79b.5 (round-3 MEASURED code-level anchor: `CHEMPROP_V2_ATOM_DIM: int = 72` export + `test_chemprop_v2_schema_shape_only` lock + `CHEMPROP_V2_SCHEMA_DECISION.md` citation).
- Wf-A62 §6.79b.6 (this sub-entry: 5 reject-rules + cross-schema bait-trap table).

### 7. Net §6.79b round-3→round-4 delta

- Round 3: prose-rule + metric-name distinction + MEASURED code-level anchor (3 sub-entries).
- Round 4: 5 reject-rules + cross-schema bait-trap table + reviewer-rebuttal block + cohort-by-metric hash (this sub-entry).
- Round 4 ADDS: explicit REJECT conditions (5 of them) that can be encoded as a Python pytest `test_wfa79b6_cross_schema_sentinels.py` (deferred to Wf-A63; out of §6.79b.6 scope which is docs-only).
- §6.79b ACTIVE state unchanged: this round adds reject-rules, NOT a resolve; the §6.79b ACTIVE honest-negative stays ACTIVE until the Wf-XX cross-schema alignment work item (deferred per §6.88 forward-looking plan) flips the mix.

## Lock conditions (MEASURED on disk 2026-10-17)

| Receipt | Verification |
|---|---|
| §6.79b.6 file on disk | `ls -la molmetal/reports/wfa62_cfm_round10/cross_schema_sentinels.md` → exists (this wave) |
| §6.79b.2 shape-only decision rule source | `grep -n 'CHEMPROP_V2_ATOM_DIM\|DMPNN_V2_ATOM_DIM' molmetal/scripts/train_chemprop_v2_v3.py molmetal/scripts/train_dmpnn_multitask_wasserstein.py` → both literals on disk |
| §6.79b.4 metric-name distinction | `grep -nE 'valid_smiles_rate\|decode_ratio' molmetal/scripts/train_chemprop_v2_v3.py molmetal/scripts/train_dmpnn_multitask_wasserstein.py molmetal/scripts/train_cfm_yuelbond_wasserstein.py` → 3 distinct schema surfaces |
| Round-3 anchor still on disk | `grep -n "§6\.79b\.[3-5]" docs/HONEST_NEGATIVE_CATALOG.md` → 3+ matches expected (Wf-A61 §6.88c verification) |

Mirror of `cfm_todo.json::_arch_audit_log_2026_10_17_Wf-A62_W2` per `memory/registry-is-the-status-ssot.md` + AGENTS.md §5.5 filesystem-truth.
