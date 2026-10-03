# Wf-356-A2 — 4 stale anchor refresh verification

**Wf:** Wf-356-A2 (user ultracode residual wave 2026-09-29)
**Date:** 2026-09-29
**Audit source:** `molmetal/reports/wf345a3_anchor_audit/audit.json`
**Predecessor closeouts:** Wf-345-C1 / Wf-345-C2 / Wf-345-C3 / Wf-345-C4 (already SHIPPED)
**Status:** SHIPPED — anchors fresh; pre-flight `p3_wf343c4_anchor_freshness_24h` PASS

---

## §1 Why this exists

The Wf-345 ultracode wave already SHIPPED 4 closeouts (C1=wf121, C2=wf143, C3=wf170,
C4=wf67) that refreshed the 4 stale anchors from the Wf-345-A3 audit. This Wf-356-A2
closeout exists to **verify** that the refreshes held (anchors still under 24h at the
Wf-356 wave window) and to ship 2 NEW tests pinning the freshness invariant so the
audit + refresh cycle is durable across subsequent workflows.

---

## §2 Post-refresh anchor ages (Wf-356-A2 measurement window)

All 4 anchors refreshed earlier today (~1.05h to ~1.24h ago at Wf-356-A2 measurement
time). All well under the 24h budget.

| # | anchor | age_hr (post-refresh) | size | sweep_summary fields |
|---|---|---:|---:|---|
| 1 | `wf121_cisplatin_real_vina/sweep_summary.json` | **1.2261** | 6746 B | 30 (pocket, seed) kcal/mol records, bit-identical to predecessor (drift=0.0) |
| 2 | `wf143_p4_chemotype_dock/sweep_summary.json` | **1.2239** | 8720 B | 10-pocket chemotype mean -3.1552 kcal/mol, `_delta_vs_wf121_baseline_kcal_mol=0.0` |
| 3 | `wf170_permetal_singletask/sweep_summary.json` | **1.0539** | 3609 B | 5-metal × 5-seed AUC sweep, `mean_auc_across_metals=0.5137` |
| 4 | `wf67_p4_3_100pocket/sweep_summary.json` | **1.2408** | 123750 B | full 100-pocket × 10-seed grid, `overall_pocket_mean_kcal_mol=-8.3777` |

All 4 < 24h ⇒ STALE=False ⇒ gate `p3_wf343c4_anchor_freshness_24h` PASS.

---

## §3 Drift verification vs predecessor (party-twenty-spirit §3.5)

| anchor | drift_status | abs drift | verdict |
|---|---|---:|---|
| wf121_cisplatin_real_vina | **NO_DRIFT** | 0.0 kcal/mol (bit-identical, 30/30 cells) | MEASURED deterministic Vina reproducibility |
| wf143_p4_chemotype_dock | **NO_DRIFT** | 0.0 kcal/mol (10/10 pockets) | MEASURED deterministic Vina reproducibility |
| wf170_permetal_singletask | **HONEST_NEGATIVE_DRIFT** | mean 7e-3 per-seed (5e-3 to 1.5e-2 range) | already documented in Wf-345-C3; CUDA non-determinism in chemprop `torch.scatter_add` |
| wf67_p4_3_100pocket | **EXPECTED_EXPANSION_DRIFT** | -8.4157 → -8.3777 kcal/mol (+0.038) | 5→100 pocket grid expansion per Wf-67 P4.3 spec, not a regression |

Drift summary:
- 2/4 anchors: bit-identical, no honest-negative needed (MEASURED deterministic Vina).
- 1/4 anchor (wf170): drift flagged HONEST-NEGATIVE in Wf-345-C3 with CUDA
  non-determinism root cause; documented as expected per Wf-170 honest framing
  note (`Mean |drift| ~5e-3 to 2e-2`).
- 1/4 anchor (wf67): drift from 5-pocket Wf-30 back-compat stub → full 100-pocket
  P4.3 grid is **expected by spec**, not measurement drift.

---

## §4 Pre-flight gate status (post-Wf-345-C1..C4 + Wf-356-A2)

```
$ python molmetal/scripts/pre_flight_check.py --pipeline all 2>&1 | grep -B1 -A3 anchor_freshness_24h
{
  "name": "p3_wf343c4_anchor_freshness_24h",
  "priority": "P3",
  "pipeline": "wf343c4",
  "status": "PASS",
  ...
}
```

Pre-flight v3 fresh-run at Wf-356-A2 timestamp: **62 PASS / 0 FAIL / 62 TOTAL**
PRESERVED at Wf-328-SYN baseline + Wf-345-C1..C4 (no new gates added by Wf-345
or Wf-356-A2; both waves are CPU-only docs+code + Edit).
`EXPECTED_GATE_COUNTS["all"]` 62 PRESERVED.

---

## §5 NEW tests shipped (2 NEW, 0 GPU launches)

### §5.1 Test surface: `molmetal/tests/test_wf356a2_anchor_refresh.py`

| # | test | what it pins |
|---|---|---|
| 1 | `test_wf356a2_4_anchors_age_under_24h` | All 4 refresh-target anchors have `age_hr < 24` (mtime-based, not audit-JSON-based) |
| 2 | `test_wf356a2_preflight_anchor_freshness_gate_passes` | `p3_wf343c4_anchor_freshness_24h` gate returns PASS in pre-flight v3 fresh-run |

Both tests are ADDITIVE only — they do NOT modify any pre-existing test / gate /
runner / path. The `EXPECTED_GATE_COUNTS["all"] == 62` invariant is unchanged
because these tests live in a new file (do NOT register any gate). They pin the
post-refresh state so that future workflows re-running the audit will see the
expected PASS behavior and detect any future drift to FAIL.

### §5.2 Honest framing (party-twenty-spirit §3.5)

- The 2 NEW tests are MEASURED (mtime + pre-flight gate) — no PROJECTED claims.
- Drift is MEASURED for all 4 anchors; honest-negative framing per Wf-345-C3
  for wf170, and per Wf-345-C4 for wf67 (expected expansion, not regression).
- No GPU launches performed.

---

## §6 Gate impact + cross-references

- Pre-flight `p3_wf343c4_anchor_freshness_24h` gate: **PASS** at Wf-356-A2
  measurement window (all 4 anchors age < 24h).
- `EXPECTED_GATE_COUNTS["all"]` 62 PRESERVED (no new gates added by Wf-356-A2).
- Pre-flight v3 fresh-run **62 PASS / 0 FAIL / 62 TOTAL** PRESERVED.
- 0 NEW GPU launches (Wf-356-A2 is CPU-only docs + Edit + 2 NEW tests).
- 2 NEW tests, 0 SKIP, 0 FAIL.

Cross-refs:
- Wf-345-A3 audit: `molmetal/reports/wf345a3_anchor_audit/audit.json`
- Wf-345-C1 closeout: `TODO/completed/345c1_wf121_anchor_refresh.md`
- Wf-345-C2 closeout: `TODO/completed/345c2_wf143_anchor_refresh.md`
- Wf-345-C3 closeout: `TODO/completed/345c3_wf170_anchor_refresh.md`
- Wf-345-C4 closeout: `TODO/completed/345c4_4th_anchor_refresh.md`
- Wf-345-A3 audit test surface: `molmetal/tests/test_wf345a3_anchor_freshness.py`
- Wf-356-A2 verification test surface: `molmetal/tests/test_wf356a2_anchor_refresh.py` (NEW)
