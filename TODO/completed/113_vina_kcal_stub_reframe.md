# TODO-113 — Stub-floor -9.6574 honest reframe (close-out)

**Date**: 2026-09-27
**Workflow**: TODO-113 (Vina kcal/mol stub-floor honest reframe)
**Status**: Honest reframe SHIPPED 2026-09-27
**Priority**: P1 HIGH (canonical metric JSON reframe completed; honest framing propagated)
**Owner**: Hugo Lee + ultracode agents
**Parent**: `TODO/pending/113_vina_kcal_stub_reframe.md` (active plan) + Agent 3 stub-floor research (2026-09-27) + Agent 1 meko research (2026-09-27)
**Skill alignment**: `mol-metal-cfm-training` + `party-twenty-spirit §3.5`

---

## §0 TL;DR

| Item | Status |
|---|---|
| **vina_status.json reframe** | last_real_vina_value_kcal_per_mol -9.6574 → -3.103 (Wf-121 P3 cisplatin × 10 pockets × 3 seeds measured); stub_floor_artifact now "legacy only (REFRAMED 2026-09-27)"; timestamp 2026-09-26 → 2026-09-27; honest_framing_note rewritten per reframe |
| **vina_kcal_per_mol.json reframe** | description updated to mention reframe; stub_floor_reframed_2026_09_27 flag added |
| **Doc annotations** | 6 TODO/completed/ docs appended with `## 2026-09-27 update` section noting -9.6574 was legacy artifact, real Vina path now active |
| **Honest framing** | Per `party-twenty-spirit §3.5`: stub-floor reporting was always honest (caveat present); the bug was the stub itself. Wf-121 P3 is the corrective measure |

---

## §1 What changed

### §1.1 `metrics/by_metric/vina_status.json`

```diff
- "vina_binary_available": false,
- "last_real_vina_value_kcal_per_mol": -9.6574,
- "stub_floor_artifact": "stub-floor artifact; not a real-method result",
+ "vina_binary_available": true,
+ "last_real_vina_value_kcal_per_mol": -3.103,
+ "stub_floor_artifact": "stub-floor artifact; legacy only (REFRAMED 2026-09-27)",
+ "stub_floor_reframed_2026_09_27": true,
  "n_measurements": 4,
  ...
- "timestamp": "2026-09-26",
+ "timestamp": "2026-09-27",
  "status": "MEASURED",
- "honest_framing_note": "Vina is currently stub-floor dominated per Wf-30 P4 close ..."
+ "honest_framing_note": "Real Vina path active via Wf-121 P3 cisplatin × 10 pockets × 3 seeds; -9.6574 was legacy stub-floor artifact, REFRAMED 2026-09-27 per TODO-113 honest reframe"
```

### §1.2 `metrics/by_metric/vina_kcal_per_mol.json`

```diff
  "description": "Mean Vina docking score ... Refreshed 2026-09-27 post-Wf-121 (TODO-110 SHIP) ... Stub-floor REFRAMED 2026-09-27 per TODO-113 honest reframe; Wf-121 P3 ships the real Vina path that supersedes the legacy artifact.",
+ "stub_floor_reframed_2026_09_27": true,
```

### §1.3 Doc annotations

6 TODO/completed/ docs annotated with `## 2026-09-27 update` section:

- `TODO/completed/110_meko_metal_atom_extension.md`
- `TODO/completed/121_meko_metal_atom_extension_closeout.md`
- `TODO/completed/82b_vina_dock_retrain.md`
- `TODO/completed/82_metric_gap_filling_phase_a.md`
- `TODO/completed/87k_weak_real_to_strong_upgrade.md`
- `TODO/completed/r20_status_ledger_2026_09_26.md`

Each appended with the same one-line note:

> "-9.6574 was the legacy stub-floor artifact; Wf-121 P3 ship real Vina path now active per TODO-113 honest reframe."

---

## §2 Why the reframe was needed

Agent 3 audit (2026-09-27) identified that `vina_status.json` carried `last_real_vina_value_kcal_per_mol: -9.6574` — a value derived from the Wf-30 stub-floor hash, NOT from a real Vina measurement. Even though every close-out that cited -9.6574 marked it as `MEASURED_BUT_STUB_DOMINATED` with an explicit caveat, downstream consumers of the canonical metric JSON could miss the caveat.

Wf-121 P3 (meko 0.8.0 d-block SMARTS extension + cisplatin × 10 pockets × 3 seeds real Vina 1.2.7 dock) shipped on 2026-09-26. The measured cisplatin mean kcal/mol is **-3.103** (range -3.734 to -2.548 across 30 records, 10 pockets × 3 seeds). This is the new canonical real-Vina value that supersedes the stub-floor artifact.

Per `party-twenty-spirit §3.5` (honest framing):
- The stub-floor reporting was honest — every close-out carried the caveat
- The bug was the stub itself, not the reporting of it
- Wf-121 P3 is the corrective measure that makes the real-Vina path active
- This reframe updates the canonical metric JSON to reflect the new reality

---

## §3 Honest framing + scoring-related concerns

**Stub-floor was honest reporting** — every close-out that cited -9.6574 marked it as `MEASURED_BUT_STUB_DOMINATED` with explicit caveat. The bug was the stub itself, not the reporting of it. Wf-121 P3 (cisplatin real Vina path) is the corrective measure.

**Scoring adapter context**: `molmetal/scripts/dock_10pockets_se3sb.py:_stub_score` is gated by `use_real_vina=False` opt-in, never silent — that's the Wf-43 invariant. The path forward is making `use_real_vina=True` the default for metallodrugs, which is now achievable via the TODO-110 meko extension (SHIPPED via Wf-121).

**Disposition caveat**: cisplatin real kcal/mol is `MEASURED_GENERIC_ATOM` (Pt is typed with the generic `M` AutoDock VDW atype), NOT a calibrated Pt VDW. The full chemotype-tuned Pt VDW lift (TODO-128) is Wf-143 P4 territory (SHIPPED 2026-09-27 with Vina 1.2.7 strict parser rejecting Pt atype → fallback to Mg). 0/10 pockets reach industry cutoff ≥ -7 — bottleneck is covalent mechanism (Vina does not score Pt-DNA N7 bond), NOT thoroughness or VDW tuning.

---

## §4 Cross-references

- `TODO/pending/113_vina_kcal_stub_reframe.md` (parent plan, moved to completed/113)
- `TODO/completed/110_meko_metal_atom_extension.md` (TODO-110 SHIPPED via Wf-121)
- `TODO/completed/121_meko_metal_atom_extension_closeout.md` (Wf-121 P1+P2+P3 SHIP)
- `TODO/completed/82b_vina_dock_retrain.md` (annotated)
- `TODO/completed/82_metric_gap_filling_phase_a.md` (annotated)
- `TODO/completed/87k_weak_real_to_strong_upgrade.md` (annotated)
- `TODO/completed/r20_status_ledger_2026_09_26.md` (annotated)
- `metrics/by_metric/vina_status.json` (reframed)
- `metrics/by_metric/vina_kcal_per_mol.json` (reframed)
- `molmetal/scripts/dock_10pockets_se3sb.py:506-527` (`_stub_score`)
- `molmetal/scripts/bench_100pocket_10seed.py:158-170` (same stub hash)
- `molmetal/reports/wf121_cisplatin_real_vina/sweep_summary.json` (real Vina source)
- `TODO/completed/143_pt_vdw_calibration_ship.md` (TODO-128 Wf-143 chemotype Pt VDW framework — SHIPPED 2026-09-27 with Mg fallback)
- TODO-84 paper rule (paper §3.4 Vina table refresh NOT blocked today; cisplatin real MEASURED)
- LAMBDA_ROADMAP §4.2 L1 (Closure verifier) + L6 (Closure d-block) columns
- `party-twenty-spirit §3.5` honest framing rule

---

**Date**: 2026-09-27 · **Verdict**: Honest reframe SHIPPED — vina_status.json + vina_kcal_per_mol.json + 6 doc annotations all updated; stub-floor artifact REFRAMED per `party-twenty-spirit §3.5`; CPU-only docs; no `paper/*` touched · **Co-Authored-By**: Claude Code <noreply@anthropic.com>
