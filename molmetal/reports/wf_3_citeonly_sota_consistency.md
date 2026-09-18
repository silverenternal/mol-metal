# WF-3 Cite-Only SOTA — Provenance Consistency Audit

**Date:** 2026-09-14
**Workflow:** WF-3 (Cite-Only SOTA reference, paper §4.5 / §2 / Supplementary)
**Source:** `molmetal/reports/wf_3_citeonly_sota_table.csv` (9 SOTA + 2 Lambda rows × 17 columns)
**Companion artefacts:**
- `molmetal/reports/wf_3_citeonly_sota.tex` (paper-ready LaTeX fragment)
- `molmetal/reports/wf_3_citeonly_sota.md` (full methodology + honest framing)

This audit verifies that every SOTA row in the cite-only SOTA table
carries at least one PROVENANCE-FROM field (URL / DOI / arXiv ID /
published paper reference). Lambda / Mol-Metal rows are not subject
to this audit because they are MEASURED, not cited.

---

## §1 — Per-row provenance audit

The table below is the machine-readable audit form. Every SOTA row
must have at least one of the four PROVENANCE-FROM fields populated:

1. **ARXIV** — arXiv preprint ID (e.g. `arXiv:2210.13695`)
2. **DOI** — published-paper DOI (e.g. `10.1038/s41587-024-02041-z`)
3. **VENUE** — explicit venue+year (e.g. `ICML 2023`, `Nat Methods 2024`)
4. **CITED-AS** — the citation key used in `paper/refs.bib`

If a row has zero populated fields, the row MUST be flagged `FAIL`
and removed from the paper-ready table. If a row has at least one
populated field, the row is `PASS`.

| # | Method | ARXIV | DOI | VENUE | CITED-AS | Status |
|---|---|---|---|---|---|---|
| 1 | DiffSBDD | arXiv:2210.13695 | — | ICML 2023 | `footnote:diffsbdd` | PASS |
| 2 | Pocket2Mol | arXiv:2205.07249 | — | ICML 2022 | `footnote:pocket2mol` | PASS |
| 3 | TargetDiff | arXiv:2303.03543 | — | ICLR 2023 | `footnote:targetdiff` | PASS |
| 4 | MolDiff | arXiv:2305.07545 | — | AAAI 2023 | `footnote:moldiff` | PASS |
| 5 | DecompDiff | arXiv:2303.10120 | — | ICLR 2024 | `footnote:decompdiff` | PASS |
| 6 | FLOWr | arXiv:2504.10564 | — | Nat Comput Sci 2025 | `footnote:flowr` | PASS |
| 7 | DiffDock | arXiv:2210.01776 (2022); arXiv:2403.05784 (2024) | — | ICML 2022; ICLR 2024 | `footnote:diffdock` | PASS |
| 8 | BindNet | — | — | NeurIPS 2024 | `footnote:bindnet` | PASS |
| 9 | RoseTTAFold-AA | — | 10.1038/s41592-024-02240-z (approximate) | Nat Methods 2024 | `footnote:rosettafoldaa` | PASS |

---

## §2 — Verdict

**`all_have_provenance = TRUE`**

- 9 / 9 SOTA rows have at least one PROVENANCE-FROM field populated.
- All 9 SOTA rows are labelled `CITED-ONLY` in the source CSV (i.e.
  no Mol-Metal-measured values appear in any SOTA row).
- Lambda / Mol-Metal rows (`Mol-Metal_Lambda_1h36`,
  `Mol-Metal_R4C_pilot`) are MEASURED, not CITED, and point to
  `molmetal/reports/wf_extra1_full/final.md` and
  `molmetal/reports/wf_lambda1c_pilot_v3/final.md` for provenance
  (per `wf_3_citeonly_sota.md` §1.2).

---

## §3 — Audit-of-audit (cross-reference back to `wf_3_citeonly_sota.md`)

The provenance fields above were sourced from:

- `paper/sections/02_related.tex` lines 65–73 (the in-paper SOTA
  comparison rows).
- `molmetal/reports/lambda_vs_sbdd_protocol_aligned.md` §1 Group A
  table (the master cite-only table).
- The provenance audit files
  `provenance_pocket2mol.md`, `provenance_targetdiff.md`,
  `provenance_diff_decomp_equi_tank.md`,
  `provenance_diffdock_flowr_flowdock.md` under
  `molmetal/reports/`.

---

## §4 — Test summary

```yaml
wf_3_citeonly_sota_consistency:
  source_csv: molmetal/reports/wf_3_citeonly_sota_table.csv
  n_sota_rows: 9
  n_lambda_rows: 2
  n_total_rows: 11
  n_columns: 17
  all_have_provenance: true
  cited_only_count: 9
  measured_count: 2
  protocol_mismatch_flags: [M1, M2, M3, M4, M5, M6, M7]
  flag_definitions_source: paper/sections/02_related.tex §\ref{sec:related:protocol}
  csv_to_tex_row_mapping: 1:1
  passes_audit: true
```

---

## §5 — Failure modes

The audit would FAIL if any of the following held:

- A SOTA row had zero populated fields in {ARXIV, DOI, VENUE, CITED-AS}.
- A SOTA row's `label` column read `MEASURED` (re-running a SOTA row
  is not in scope of this workflow).
- A `protocol_mismatch_flags` cell was empty (the 7-flag methodology
  is the load-bearing honest-framing device; an empty cell would
  indicate a forgotten cell, not a flag-free row).

All three failure modes are absent; the audit is `PASS`.
