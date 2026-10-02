# Wf-346-B1 — Pipeline drift audit (frontier / chemprop_v2_v3 / cfm / all)

**Status:** SHIPPED (audit-only — no code changes)
**Date:** 2026-09-29
**Wall:** ~2 min CPU-only (audit-only)
**Tests:** 0 NEW (audit-only)
**Drift found:** NO — all 4 pipelines match EXPECTED_GATE_COUNTS

---

## §1 Context

Per Wf-343D1 + Wf-343D3 + the 2026-09-29 ultracode directive, the project
needed to verify whether registration drift on the pre-flight `build_gates()`
dispatcher was limited to the `all` pipeline (where Wf-343D1 originally
found the issue) or had silently propagated to the three subset pipelines
(`frontier`, `chemprop_v2_v3`, `cfm`). This is the audit-only companion
investigation; Wf-347 follow-up will only be needed if any drift is found.

Per Wf-343D3 reference per-pipeline sub-counts:

| Pipeline | Expected |
|---|---:|
| `all` | 66 |
| `frontier` | 25 |
| `chemprop_v2_v3` | 13 |
| `cfm` | 6 |

The `all` pipeline is the union of three branches in `build_gates()`:

1. `if pipeline in ("chemprop_v2_v3", "all"):` → 13 gates
2. `if pipeline in ("cfm", "all"):` → 6 gates
3. `if pipeline == "all":` → 47 gates (all-only)

13 + 6 + 47 = 66 (matches expected `all` total).

The `frontier` pipeline has its own branch (`if pipeline == "frontier":`)
that surfaces a curated 25-gate subset of the P3 frontier-math workstream
for paper-rule parity.

---

## §2 Method

1. Located `def build_gates()` at `molmetal/scripts/pre_flight_check.py:3652`.
2. Located `EXPECTED_GATE_COUNTS` dict at lines 2817-2822 (the canonical
   source of truth for per-pipeline counts).
3. Sectioned the dispatcher body by `if pipeline in (...)` / `if pipeline == ...`
   markers; each branch contains a `runners += [...]` literal whose elements
   are `(name, lambda)` tuples.
4. Statically counted the leading string of each tuple in each branch;
   deduplicated (the gate name appears twice — once as tuple arg, once in
   the `gate_xxx()` helper call).
5. Cross-checked: 13 + 6 + 47 = 66 (matches `EXPECTED_GATE_COUNTS["all"]`).
6. Wrote machine-readable JSON audit at:
   `molmetal/reports/wf346b1_pipeline_drift_audit/audit.json`.

No subprocess was launched. No `pre_flight_check.py` execution required —
this is a static-source audit only. CPU-only; zero GPU launches per
AGENTS.md §5.1 standing GPU authorization (no GPU work was needed).

---

## §3 Per-pipeline results

### §3.1 `frontier`

| Field | Value |
|---|---|
| Expected | 25 |
| Actual | 25 |
| Drift | 0 |
| Verdict | **MATCH** |
| Missing gates | (none) |

All 25 expected frontier gates are present. The branch lives at
`pre_flight_check.py:4916-5218` (`if pipeline == "frontier":`).

### §3.2 `chemprop_v2_v3`

| Field | Value |
|---|---:|
| Expected | 13 |
| Actual | 13 |
| Drift | 0 |
| Verdict | **MATCH** |
| Missing gates | (none) |

All 13 expected chemprop_v2_v3 gates are present. The branch lives at
`pre_flight_check.py:4062-4168` (`if pipeline in ("chemprop_v2_v3", "all"):`).

### §3.3 `cfm`

| Field | Value |
|---|---:|
| Expected | 6 |
| Actual | 6 |
| Drift | 0 |
| Verdict | **MATCH** |
| Missing gates | (none) |

All 6 expected cfm gates are present. The branch lives at
`pre_flight_check.py:4171-4225` (`if pipeline in ("cfm", "all"):`).

### §3.4 `all` (composed of 3 branches)

| Field | Value |
|---|---:|
| Expected | 66 |
| Actual | 66 |
| Drift | 0 |
| Verdict | **MATCH** |
| Missing gates | (none) |
| Composition | 13 (chemprop) + 6 (cfm) + 47 (all-only) = 66 |

The `all` pipeline total matches the expected count via the documented
union construction.

---

## §4 Honest framing (per AGENTS.md §5.2 / party-twenty-spirit §3.5)

- **MEASURED:** All 4 per-pipeline counts are MEASURED via static source
  extraction of `build_gates()` in `molmetal/scripts/pre_flight_check.py`.
  No values were estimated or projected.
- **Audit-only:** This Wf-346-B1 agent performed NO code changes — no
  Edit, no Write to `pre_flight_check.py`. The audit verifies the
  post-Wf-343 + Wf-343D1 + Wf-343C4 dispatcher state.
- **No drift:** The original Wf-343D1 drift was on the `all` pipeline
  (the 4 NEW Wf-343C4 gates brought it from 62 → 66 with the
  `gate_count_consistency_audit` gate). That drift has since been
  documented and reconciled; no further drift has appeared.
- **Cross-check:** The Wf-321D `p3_gate_count_consistency_audit` gate
  (which subprocess-runs the `all` pipeline and asserts
  `summary.total == EXPECTED_GATE_COUNTS["all"]`) is itself one of the
  66 `all`-pipeline gates. If the count ever drifts, that gate FAILs
  on the next pre-flight run — this audit is a one-time static
  confirmation that the documented counts match the source.
- **WF-347 follow-up:** NOT required (no drift detected).

---

## §5 Files written

| Path | Purpose |
|---|---|
| `molmetal/reports/wf346b1_pipeline_drift_audit/audit.json` | Machine-readable audit (schema-versioned) |
| `TODO/completed/346b1_pipeline_drift_audit.md` | This closeout doc |

---

## §6 Conclusion

**STATUS: CLEAN.** All 4 pipelines (`frontier`, `chemprop_v2_v3`, `cfm`,
`all`) match `EXPECTED_GATE_COUNTS` exactly. No registration drift
detected on the `build_gates()` dispatcher. The Wf-343D1 finding
was a one-time event that has been fully reconciled. The audit
JSON at `molmetal/reports/wf346b1_pipeline_drift_audit/audit.json`
is the canonical record.
