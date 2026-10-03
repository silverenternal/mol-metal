# Wf-324F — TODO/02_metrics/METRICS_TARGETS.md refresh

**Date**: 2026-09-29
**Wf**: wf324f_metrics_targets_refresh
**Owner**: Hugo Lee + Wf-324F agent (party-twenty-spirit §3.5 honest framing)
**Status**: ✅ SHIPPED — METRICS_TARGETS.md updated with Wf-310A G5 + Wf-318B
cascade delta + Wf-323B schema fix + Tier-2 journal strategy per AGENTS.md §5.4.

---

## §1 Trigger

Per Wf-324F task brief — the file `TODO/02_metrics/METRICS_TARGETS.md` was
stale at 2026-09-25 (R17) and needed to reflect the Wf-310A + Wf-318B +
Wf-323B ships that landed 2026-09-29:

1. **Wf-310A** — G5 50-epoch 5-metal re-aggregation (path+schema fix +
   2-pair retry).
2. **Wf-318B** — TODO-181 cascade delta 30ep→50ep extended to all 5 metals
   (3 MEASURED lifts + 2 HONEST_NEGATIVE regressions).
3. **Wf-323B** — MetalCytoToxDB loader v3/v4 schema drift fix.
4. **Tier-2 journal target** — 2026-09-29 user directive "我们发二区就可以了，
不用规划湿实验" (Tier-2 journal is sufficient; do NOT plan wet-lab experiments)
per AGENTS.md §5.4.

---

## §2 Edits applied to METRICS_TARGETS.md

| Section | Change | Lines (post-edit) |
|---|---|---|
| Header (line 3) | Date `2026-09-25` → `2026-09-29`; added R17 + Wf-310A/318B/323B refresh + Tier-2 journal target annotation | 3-9 |
| §3 integration table | Ru subset AUC row updated to `0.7860 ± 0.0145 ✅ MEASURED` (was 0.712 single-task chemprop_v2 1-seed); added **5-metal grand aggregate** row at `0.7227 ± 0.0417 ✅ MEASURED` with Wf-310A annotation | 401-425 |
| §3.1 NEW | **Wf-318B cascade delta 5-metal table** added (Ru/Ir/Os LIFT, Rh/Re HONEST_NEGATIVE) | 428-448 |
| §5 honest framing | 3 NEW entries: Wf-181 G5 2-metal HONEST_NEGATIVE regression; Tier-2 journal target framing; Wf-310A G5 5-metal grand aggregate framing | 460-471 |
| §6 current progress | 3 NEW status rows: 5-metal grand aggregate (Wf-310A); cascade delta 5-metal (Wf-318B); MetalCytoToxDB v3/v4 schema fix (Wf-323B) | 473-489 |
| §11 NEW | Full Wf-310A + Wf-318B + Wf-323B refresh section (6 subsections: G5 per-metal table, cascade delta, schema fix, pre-flight status, Tier-2 strategy, cross-references) | 633-720 |

**Total**: 1 file updated, 6 sections touched, ~150 LOC added. **No fabrication**:
every value traces to JSON on disk or to a closeout doc in
`TODO/completed/`.

---

## §3 Pre-flight verification

```bash
cd /home/hugo/codes/mol-metal && \
PYTHONPATH=/home/hugo/codes/mol-metal \
/home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
molmetal/scripts/pre_flight_check.py --json /tmp/wf324f_preflight.json
```

- **TOTAL: 59 / PASS: 58 / FAIL: 1 / SKIP: 0**
- The 1 FAIL (`p3_gate_count_consistency_audit`) is the pre-existing gate
  count drift per `TODO/completed/321d_preflight_gate_audit.md` — NOT
  loader-related, NOT a Wf-324F blocker. Already flagged in Wf-321D audit.
- Wf-324F is text-only / metric-target-doc-only; **no new measurements**,
  no GPU launches, no test additions. Pre-flight status unchanged from
  Wf-323B baseline (58 PASS + 1 pre-existing FAIL).

**Note on venv path**: The task brief specified `/home/hugo/cades/CadButEaas/.venv-vlm/bin/python`
but the actual venv is at `/home/hugo/codes/CadButEaas/.venv-vlm/bin/python`.
Used the correct path. No effect on pre-flight status.

---

## §4 Honest framing per party-twenty-spirit §3.5

- **Wf-310A 5-metal grand aggregate (0.7227 ± 0.0417)** is reported MEASURED
  on 13/15 pairs (Ru/seed=2 + Ir/seed=0 pending). 2 retries were in-flight
  as of Wf-310A ship; Ir/seed=0 has since landed (Wf-318B incorporated the
  retry at 0.6888), Ru/seed=2 still in-flight (epoch 27/50). The
  grand aggregate will refresh to 15/15 once Ru/seed=2 lands.
- **Wf-318B cascade delta 5-metal**: 3 MEASURED lifts (Ru +0.030, Ir +0.027,
  Os +0.025) + 2 HONEST_NEGATIVE regressions (Rh −0.043, Re −0.037). All
  reported in `molmetal/scripts/paper_dashboard.py` CLAIM_CATALOG with
  explicit HONEST_NEGATIVE badge. 5-metal aggregate delta (+0.0003) is
  reported unfiltered.
- **Tier-2 journal target** (per AGENTS.md §5.4): paper §6 no longer claims
  wet-lab validation as in-scope. This is a strategic re-scope, not a
  capability reduction.
- **Wf-323B schema fix**: v3 and v4 are both legitimate versions of the same
  data; loader now handles both for backward compatibility. Pure CPU fix,
  no GPU.
- **0 fabrication**: every value traces to JSON on disk or to a closeout doc
  in `TODO/completed/`. No aspirational timing for "when wet-lab will happen".

---

## §5 Files written / updated

| Path | Type | Size |
|---|---|---|
| `TODO/02_metrics/METRICS_TARGETS.md` | MODIFIED (6 sections touched, ~150 LOC added) | +150 LOC |
| `TODO/completed/324f_metrics_targets_refresh.md` | NEW (this file) | — |
| `/tmp/wf324f_preflight.json` | NEW (pre-flight gate audit JSON) | — |

---

## §6 Cross-references

| Doc | Path |
|---|---|
| Wf-310A closeout (G5 5-metal) | `TODO/completed/181_wf181_g5_50epoch_full_sweep_closeout.md` |
| Wf-318B closeout (cascade delta) | `TODO/completed/318b_g5_cascade_delta_5metal.md` |
| Wf-323B closeout (schema fix) | `TODO/completed/323b_v3_v4_schema_drift_fix.md` |
| Wf-317D Tier-2 strategy | `TODO/completed/317d_improvement_enum_tier2_update.md` |
| Wf-318J Tier-2 integration | `TODO/completed/318j_tier2_integration_verified.md` |
| Wf-321D pre-flight gate audit | `TODO/completed/321d_preflight_gate_audit.md` |
| Metric JSON (G5 50-ep) | `metrics/by_metric/wf181_g5_50epoch_full_sweep.json` |
| Metric JSON (cascade delta 5-metal) | `metrics/by_metric/wf181_g5_50epoch_cascade_delta_5metal.json` |
| Loader (post-Wf-323B) | `molmetal/data/metal_cytotox.py` |
| Loader tests (post-Wf-323B) | `molmetal/tests/test_wf323b_v3_v4_schema_drift.py` |

---

## §7 Follow-ups

1. **Wf-318C** (next): re-run `compute_cascade_delta_5metal.py` after
   Ru/seed=2 retry lands. Ru row should update from n=2 to n=3 overlap;
   5-metal grand aggregate refresh from 13/15 to 14/15 (Ir already
   incorporated in Wf-318B).
2. **TODO-182 G6 per-innovation ablation**: 5×3×50=75 pairs to attribute the
   Rh/Re regression to W₂ vs L2-guard vs baseline (Wf-310F follow-up).
3. **Wf-309A early-stopping** may be preferable for small-cohort metals
   (Rh/Os/Re < 1.2k train rows) — review once cascade delta reproduces
   with shorter horizon.
4. **TODO-138 paper v2 + Tier-2 submission** — Tier-2 journal target
   package (JCAMD / WIREs Comput. Mol. Sci / Mol. Divers. / J. Mol. Graph.
   Model.) per AGENTS.md §5.4.
5. **Pre-flight gate count drift (p3_gate_count_consistency_audit)** — out
   of scope for Wf-324F; already flagged in Wf-321D audit doc. Will require
   `EXPECTED_GATE_COUNTS` update in `TODO/completed/321d_preflight_gate_audit.md`.

---

**END of Wf-324F doc**
