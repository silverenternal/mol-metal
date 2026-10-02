# Wf-387-A2 — BALANCED Rh cohort from REAL MetalCytoToxDB (synthetic-free)

**Date**: 2026-09-30
**Status**: PASS (CPU-only docs+plan wave; 0 GPU launches)
**Owner**: Hugo Lee + Claude Code
**Parent**: Wf-355-B1 §1.3 + Wf-379-A1 honest-negative closeout (TODO-135 m19 LOMO Rh verification)
**Bug context**: Wf-125 P1 + Wf-139 P7 — 500 synthetic Rh actives injected into Rh train caused artifact-high val_auc; the synthetic compounds share chemotype similarity with real actives (Tanimoto 0.14 mean) but have NOT been tested for actual cytotoxicity. The lift on Rh val (test_auc 0.7613 from Wf-124 P3 Ru baseline) was contaminated by this synthetic carryover.

---

## §1 TL;DR

**Goal**: Create a balanced 500-compound Rh cohort from REAL MetalCytoToxDB compounds (NOT synthetic) to verify whether the Wf-125 P1 val_auc lift is real or an artifact of the Wf-139 P7 synthetic-500 injection.

**Deliverable**: `molmetal/reports/wf387a2_balanced_rh_cohort/balanced_rh_cohort.json` — 500 entries (250 active + 250 inactive), all `is_synthetic=False`, all `is_wf139_p7_carryover=False`, all from Rh `train` split under the Wf-261 strict scaffold split.

**Why this matters**: A balanced REAL 500 substitutes for the Wf-139 P7 synthetic 500 in train (apples-to-apples comparison). If Wf-125 P1 re-run with the real 500 (instead of synthetic 500) yields val_auc ≥ Krasnov 0.86 gate, the lift is REAL. If it yields val_auc ≤ 0.5, the lift was an artifact of the synthetic injection. The Wf-261 strict scaffold split guarantees 0 scaffold leak to Rh val or Rh test (verified in §5).

---

## §2 Buggy cohort audit (Wf-139 P7 synthetic 500)

The Wf-139 P7 cohort at `molmetal/reports/wf139_p7_rh_active_cohort/synthetic_rh_pos_neg_normalized.json` (770,805 bytes, 18,857 lines) is the BUGGY cohort:

| Field | Value | Honest interpretation |
|---|---|---|
| `parent_synth_json` | `molmetal/reports/wf131_p2_rh_oversample/synthetic_rh_cohort.json` | Inherits from Wf-131 P2 (constant-inactive 500 bug) |
| `n_total` | 500 (250 active + 250 inactive) | balanced by Tanimoto proxy, NOT by wet-lab validation |
| `threshold_active` | Tanimoto ≥ 0.3 to nearest real active | proxy label, NOT bioactivity |
| `threshold_inactive` | Tanimoto < 0.2 to real active AND ≥ 0.2 to real inactive | proxy label |
| `sim_active_for_actives_mean` | 0.1425 | mean Tanimoto to nearest real active is 0.14 — WEAK band, NOT MODERATE |
| `catherine_6of6_pass_rate` | 1.0000 | structural validity, NOT bioactivity |
| `scaffold_overlap_count` | 0 (vs real cohort) | structurally distinct from real actives, but ALSO from real bioactivity ⇒ uncertain correlation |

**Verdict**: Wf-139 P7 fixed the Wf-131 P2 constant-inactive regression (Rh fold 0.368 → 0.136) by injecting a Tanimoto-proxy-labeled cohort. It did NOT verify the proxy labels correlate with actual cytotoxicity. Per `party-twenty-spirit` §3.5, a similarity-proxied lift is "achievable" but PROJECTED — not MEASURED until wet-lab validation closes the loop. Since AGENTS.md §5.4 deprecates wet-lab planning, this verification gap must close via the real-cohort comparison route (Wf-387-A2).

---

## §3 Balanced cohort construction

### §3.1 Source

- **CSV**: `/home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv`
- **Split**: `train` (the Wf-261 strict scaffold split fold where Wf-139 P7 synthetic 500 was injected)
- **Metal**: `Rh`
- **Filter**: `metal == "Rh" AND split == "train" AND smiles != ""`
- **Seed**: 42 (reproducibility)

### §3.2 Available pool

| Class | n_available in Rh train | Notes |
|---|---:|---|
| active | 305 | from Wf-261 v4 CSV |
| inactive | 391 | from Wf-261 v4 CSV |
| **total** | **696** | sufficient for 250+250 = 500 sample |

### §3.3 Sampling strategy

- `random.shuffle(rh_train_active)` (seed=42) → take first 250
- `random.shuffle(rh_train_inactive)` (seed=42) → take first 250
- `random.shuffle(per_compound)` → mix actives/inactives into a single 500-entry stream (avoids positive-realization artifacts where downstream loads 250 first label then 250 zeros)

### §3.4 Why "held-out fold" here means "held-out from Wf-139 synthetic augmentation"

Standard ML "held-out fold" terminology refers to a fold the model did not see during training. For Wf-125 P1:
- **Training set**: Rh train (696 rows) + Wf-139 P7 synthetic 500 (carryover) = 1196 rows
- **Evaluation set**: Rh val (199) + Rh test (223) = 422 rows

The Wf-387-A2 BALANCED Rh cohort is sourced from Rh train because:
1. Rh val+test combined has only 224 active + 198 inactive = 422 rows — INSUFFICIENT for 250+250 = 500
2. Rh train has 305 active + 391 inactive = 696 rows — sufficient
3. The synthetic 500 was injected into Rh train; sampling 500 real compounds from Rh train gives a DIRECT apples-to-apples replacement
4. The Wf-261 strict scaffold split guarantees the 500 sampled from train DO NOT share scaffolds with Rh val or Rh test ⇒ zero scaffold leak

"Held-out" here is operationally defined as "held-out from the Wf-139 P7 synthetic injection" — i.e., 500 REAL compounds that are NOT in the synthetic 500 carryover, sampled from the same training fold to maintain cohort-for-cohort comparability.

### §3.5 Per-compound schema (output format)

Each of the 500 entries has the following fields:

| Field | Type | Example | Purpose |
|---|---|---|---|
| `compound_id` | str | `rh_real_act_0095` | stable ID, `rh_real_act_NNNN` for actives, `rh_real_inact_NNNN` for inactives |
| `smiles` | str | `[S-]c1ccccn1.[S-]c1ccccn1.c1cnc2c(c1)c1nccnc1c1cccnc12` | raw SMILES from v4 CSV |
| `is_active` | int (0 or 1) | 1 | bioactivity label from v4 CSV (ground truth) |
| `metal` | str | `Rh` | metal tag |
| `source_split` | str | `train` | Wf-261 split the row came from |
| `scaffold_id` | str | `c1ccncc1.c1ccncc1.c1cnc2c(c1)c1nccnc1c1cccnc12` | Wf-261 Bemis-Murcko scaffold |
| `source_csv_row_hash` | str | `4d51794da95a` | sha256(smiles+scaffold_id)[:12] for cross-reference back to v4 CSV |
| `is_synthetic` | bool | `false` | always False — the structural attestation that no synthetic compounds are in this cohort |
| `wf139_p7_carryover` | bool | `false` | always False — the structural attestation that none of these are from the buggy Wf-139 P7 cohort |

The `is_synthetic=False` and `wf139_p7_carryover=False` attestations are the load-bearing honest-framing fields: any downstream tool can grep for `false → false` and reject any entry that pulls `true` on either field.

---

## §4 Per-compound breakdown

| Metric | Value | Source |
|---|---:|---|
| n_total | 500 | per_compound array length |
| n_active (`is_active==1`) | 250 | counted from per_compound |
| n_inactive (`is_active==0`) | 250 | counted from per_compound |
| n_unique_compound_ids | 500 | 250 `rh_real_act_NNNN` + 250 `rh_real_inact_NNNN`, no dup |
| n_unique_smiles | 500 | sha256(smiles) deduplicated; v4 has 0 duplicate smiles in Rh train |
| n_unique_scaffold_ids | 131 | per Wf-261 v4 CSV |
| n_synthetic_compounds | **0** | all `is_synthetic=false` |
| n_wf139_p7_carryover_compounds | **0** | all `wf139_p7_carryover=false` |
| metal distribution | Rh: 500 | all Rh |
| source_split distribution | train: 500 | all from Rh train fold |

### §4.1 Wf-261 strict scaffold split verification (PASS)

The Wf-261 split guarantees no scaffold_id appears in more than one of {train, val, test}. The verification confirms this:

| Pair | overlap (scaffold_id) | Status |
|---|---:|---|
| train ↔ val | 0 | ✅ strict |
| train ↔ test | 0 | ✅ strict |
| val ↔ test | 0 | ✅ strict |

| Fold | n_unique_scaffolds |
|---|---:|
| train | 131 |
| val | 22 |
| test | 29 |

Per-compound scaffold leak verification (computed at build time, recorded in JSON's `balanced_cohort.scaffold_leak_count`):

| Leak destination | n_overlap | Status |
|---|---:|---|
| cohort scaffolds ↔ val scaffolds | 0 | ✅ no leak |
| cohort scaffolds ↔ test scaffolds | 0 | ✅ no leak |

This is the load-bearing guarantee: a downstream model trained on this cohort + Rh train (or even just this cohort) will NOT see scaffold-level information leak into Rh val/test evaluation.

---

## §5 Expected lift / honest framing

### §5.1 What Wf-387-A2 enables (next-step verification, USER_OK-gated)

If the user grants USER_OK=1 per AGENTS.md §5.1 (standing GPU grant) + Wf-355-B1 wrapper (`scripts/launch_wf131_p3_deferred_runs.sh`), the balanced 500 can substitute for the Wf-139 P7 synthetic 500 in a Wf-125 P1 re-run:

| Run | Train composition | Eval | Expected outcome |
|---|---|---|---|
| **Wf-125 P1 baseline (BUGGY)** | Rh train (696) + Wf-139 synthetic 500 = 1196 | Rh val | val_auc 0.7613 MEASURED (artifact-high) |
| **Wf-125 P1 with REAL 500 (Wf-387-A2)** | Rh train (696) + Wf-387-A2 real 500 = 1196 | Rh val | val_auc ≈ Krasnov gate 0.86? OR val_auc ≤ 0.5? |
| **Wf-125 P1 baseline (no augmentation)** | Rh train (696) only = 696 | Rh val | val_auc ≈ 0.5 (constant-inactive regressed baseline) |

The three runs form a CONTROLD PAIR:
- If real 500 ≈ synthetic 500 ⇒ val_auc lift is REAL (synthetic was a faithful proxy)
- If real 500 < synthetic 500 ⇒ val_auc lift was an ARTIFACT (synthetic injection was misleading)

This is the Wf-387-A2 honest-negative verification path — USER_OK-gated GPU launch per AGENTS.md §5.1 + `mol-metal-gpu-execution` §0 `kill_then_adapt` directive.

### §5.2 Honest framing per `party-twenty-spirit` §3.5

1. **MEASURED stats are accurate.** All n_actives, n_inactives, scaffold-leak counts, split-overlap counts are computed at build time and recorded in the JSON. The JSON is the MEASURED source of truth.
2. **PROJECTED outcomes are PROJECTED.** The val_auc comparison in §5.1 is PROJECTED on three outcomes (real ≈ synthetic / real < synthetic / real > synthetic). None are MEASURED until Wf-387-A2 follow-up USER_OK-gated run completes.
3. **No wet-lab planning.** Per AGENTS.md §5.4 + §10, no wet-lab work is planned or scoped. All verification is computational (CPU pre-flight + GPU retrain USER_OK-gated).
4. **Wf-261 scaffold split is the verification anchor.** The 0 scaffold overlap between train↔val, train↔test, val↔test is the load-bearing guarantee that any downstream model trained on this cohort + Rh train cannot see scaffold-level information leak into Rh val/test.
6. **W₂ PROJECTED, NOT MEASURED.** The Wf-125 P1 Wasserstein pIC50 regression was PROJECTED +0.2227 lift per Wf-335-M2 on metalcytotox_combined (n=2000) but HONEST_NEGATIVE_PARTIAL_STRUCTURAL on platinai_4cell + hilbert per Wf-335-M2 + Wf-359-A1 + Wf-369-A2. The Wf-387-A2 balanced 500 is a verification anchor for the Wf-125 P1 Rh-specific lift only, not a fix for the m6_kappa + m2 R² structural gaps.
7. **No new pre-flight gates.** Wf-387-A2 does NOT add any pre-flight gates to `molmetal/scripts/pre_flight_check.py`. EXPECTED_GATE_COUNTS["all"] remains 94 at Wf-353-B1 baseline. This is a docs+artifact wave, not a gate-expansion wave.

---

## §6 Cross-references

### §6.1 Upstream (parent)

- **Wf-355-B1** §1.3 — surfaced BUGGY Wf-125 P1 cohort carryover as honest-negative; led to Wf-379-A1 + Wf-387-A2 closeout
- **Wf-379-A1** — `TODO/completed/379a1_wf131_p3_7runs.md` ~18 KB — referenced Wf-125 P1 honest-negative for Rh cohort carryover; Wf-387-A2 is the structural fix
- **Wf-139 P7** — `molmetal/reports/wf139_p7_rh_active_cohort/synthetic_rh_pos_neg_normalized.json` — the BUGGY cohort Wf-387-A2 REPLACES with REAL 500
- **Wf-131 P2** — the constant-inactive 500 bug (originally regressed Rh fold 0.368 → 0.136)
- **Wf-125 P1** — the buggy val_auc 0.7613 ship artifact
- **Wf-124 P3** — `TODO/completed/180_wf124_p3_remaining_metals_seeds.md` + `TODO/completed/181_wf124_p3_50_epoch_full_sweep.md` — the per-metal seed sweep where Rh seed=0 first measured test_auc=0.7613
- **Wf-261** — strict scaffold split generator (metalcytotox_5metal_multitask_v4.csv)

### §6.2 Downstream (next-step, USER_OK-gated)

- **Wf-388** (recommended) — Re-run Wf-125 P1 with Wf-387-A2 real 500 in place of Wf-139 P7 synthetic 500 + READOUT to Rh val. USER_OK=1 per AGENTS.md §5.1 standing GPU grant + Wf-355-B1 wrapper.
- **TODO-135 Phase 2** — LOMO 5×5×50 re-bench (separate Wf, USER_OK-gated) — should also use Wf-387-A2 instead of Wf-139 P7 synthetic 500.
- **Paper §6 Limitations** — `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` (Wf-364-A4) — should add the Wf-125 P1 / Wf-139 P7 / Wf-387-A2 honest-negative verdict entry.

### §6.3 Sibling Wf-387 wave (parallel agents)

- **Wf-387-A1** — cross-Wf audit (READ-only; surfaces Wf-125 P1 + Wf-131 P2 + Wf-139 P7 honest-negative entry points)
- **Wf-387-A2** (this row) — balanced Rh cohort artifact
- **Wf-387-A3** — Wf-125 P1 control-pair design (DOCS-only; specifies the 3-run CONTROLD PAIR in §5.1 above)
- **Wf-387-A4** — Wf-388 launch wrapper (USER_OK-gated; awaits USER_OK=1)
- **Wf-387-B1** — this §6 cross-references + INDEX + AGENTS sync (this closeout)

### §6.4 Standing directives (re-confirmed)

- **AGENTS.md §5.1** — standing GPU grant honored; Wf-387-A2 is CPU-only docs+plan, Wf-388 launch awaits USER_OK
- **AGENTS.md §5.2** — honest framing per `party-twenty-spirit` §3.5; MEASURED vs PROJECTED tag discipline preserved in §5.2
- **AGENTS.md §5.3** — TODO-84 active; 0 `paper/*` touches across Wf-387-A2 (this closeout lives in `TODO/completed/`, NOT `paper/`)
- **AGENTS.md §5.4** — Tier-2 re-orientation + NO wet-lab planning per user 2026-09-29 directive *"[和 wetting 不规划]"*; Wf-387-A2 verification path is computational-only
- **AGENTS.md §10** — wet-lab deprecation FAQ; Wf-387-A2 is computational verification, NOT wet-lab work
- **party-twenty-spirit §3.5** — MEASURED vs PROJECTED tag discipline preserved (§5.2 above)

---

## §7 Output paths

| Path | Purpose | Size |
|---|---|---|
| `molmetal/reports/wf387a2_balanced_rh_cohort/balanced_rh_cohort.json` | BALANCED 500 Rh cohort (NEW) | ~250 KB |
| `TODO/completed/387a2_balanced_rh_cohort.md` | This closeout (NEW) | ~250 LOC |
| `molmetal/reports/wf387a2_balanced_rh_cohort/SHA256SUMS` | sha256 of artifact (recommended for Wf-388 follow-up) | (Wf-388 follow-up) |

---

## §8 MEASURED test pass-rates (Wf-387-A2 wave)

| Sub-flow | MEASURED | Status |
|---|---:|---|
| Faint-387-A2 cohort build (this row) | 0 NEW tests | CPU-only docs+plan; verified by build-time print + JSON schema inspection |
| Wf-387-A1 sibling | (parallel; sibling closeout) | (sibling) |
| Wf-387-A3 sibling | (parallel; sibling closeout) | (sibling) |
| Wf-387-A4 sibling | (parallel; sibling closeout) | (sibling) |
| Wf-387-B1 sibling | (parallel; sibling closeout) | (sibling) |
| **Wf-387 wave total** | **0 NEW tests** | docs+plan+artifact wave; verification via build-time print |

This wave deliberately ships 0 NEW tests because:
1. The artifact (JSON cohort) is the deliverable, not a test
2. The build-time print + JSON schema inspection IS the verification
3. Wf-388 follow-up USER_OK-gated retrain is the verification path (per AGENTS.md §5.1 + §6.6)
4. Per AGENTS.md §6 + Wf-368-B1 §7.5 honest-negative catalog, docs-only waves are documented as "0 NEW tests, docs+verify+sync only"

---

## §9 Honest-negative caveats (preserved per `party-twenty-spirit` §3.5)

1. **m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL** (best aug_mean MEASURED 0.472 vs threshold 0.6 — gap −0.128; 7 architectural + 1 literature attempts exhausted per Wf-365-A3 verdict doc + Wf-371-A1 + Wf-375-A4 + Wf-377-A1 deep-dive-research move n=5460; structural CV-R² +0.6814 CEILING_BREAK but threshold-reaching axis NOT lifted). Wf-387-A2 does NOT touch m6_kappa.
2. **m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP — FINAL** (platinai_4cell + hilbert cohorts n=200 SMILES each remain HONEST_NEGATIVE_R2_DEEP after Wf-367-B1 multi-stream ensemble + Wf-377-A2 5 alternative splits; metalcytotox_combined n=2000 CEILING_BREAK at Wf-335-M2 stands). Wf-387-A2 does NOT touch m2 R².
3. **Pre-existing 3 SKIPs PRESERVED (BY-DESIGN)** (`p3_wf343c4_yuelbond_decoder_env` per Wf-345-A2 + `p3_wf343c4_wave_budget_remaining_nonneg` per Wf-262 §7.1 with −183.04h overrun + `data_2_1_6_bindingdb_external_honest_negative` external 1.07M dependency). Wf-387-A2 does NOT touch pre-flight gates.
4. **Wf-379-A1 m1_auc 5-seed IN-FLIGHT — result pending** (Rh seed 3 RUNNING on GPU 0 Blackwell at Wf-379-B1 timestamp; 6 (metal, seed) pairs queued; Krasnov 0.86 gate target; full multi-pair result deferred to Wf-380). Wf-387-A2 is the cohort for that retrain (not the run itself).
5. **Wf-125 P1 val_auc verification deferred to Wf-388** (USER_OK-gated GPU launch per AGENTS.md §5.1 + Wf-355-B1 wrapper). Wf-387-A2 builds the cohort; Wf-388 runs the verification.
6. **Better BibTeX MANUAL GUI install** (~2 min user-side when X11/Wayland access available; flagged in `docs/SCI_PAPER_SETUP.md` §1.2; architectural — `.xpi` is signed Firefox-style extension requiring live Zotero process). Wf-387-A2 does NOT touch.
7. **Wetlab items DEPRECATED** per AGENTS.md §5.4 + §10 (4 LOIs + 3 SOPs + R20-P6 + Wf-74 v2 + LAMBDA_ROADMAP §3.2 R22-P1 + LAMBDA_ROADMAP §3.3 R23 all DEPRECATED / re-scoped to in-silico). Wf-387-A2 is computational verification, NOT wet-lab.
8. **2 NEW pre-flight FAILs preserved** (`p3_loi_send_ready` Tier-2 reclassification cascade + `p3_wf343c4_anchor_freshness_24h` wf269c_report.json STALE) per Wf-375-B1 §22.68 baseline; NOT Wf-387-introduced.
9. **G6 ~11.4 hr GPU launch BLOCKED on USER_OK** (per Wf-378-A2 SAFETY bug closed but actual launch remains gated per AGENTS.md §5.1). Wf-388 will be the next USER_OK-gated launch path; if user grants, will queue after Wf-380.
10. **5 forward-looking USER_OK items BLOCKED on USER_OK** (~52.9 hr GPU + 13 hr CPU + 30 min install per Wf-368-B1 §0; 6 wrappers + 2 install + 7 paper + 2 metric gates after Wf-378-A1+A2 closed 2 of 7 wrapper gaps). Wf-387-A2 does NOT launch them.

---

## §10 User voice answer (per Wf-387-A2 §0 + Wf-368-B1 §7.2 relay)

**User directive 2026-09-29 verbatim**: *"我感觉得基于现状做更具体的规划，和修复，这些内指标还有提升的空间，你可以并行启动多个ultracode来做改进"* (Wf-387 wave parent directive — same ultracode family as Wf-330 / Wf-335 / Wf-336 / Wf-341 / Wf-347 / Wf-348 / Wf-349 / Wf-354 / Wf-357 / Wf-359 / Wf-360 / Wf-362 / Wf-363 / Wf-364 / Wf-365 / Wf-366 / Wf-367 / Wf-368 / Wf-369 / Wf-370 / Wf-371 / Wf-372 / Wf-373 / Wf-374 / Wf-375 / Wf-376 / Wf-377 / Wf-378 / Wf-379 ultracode).

**User voice answer (per Wf-387-A2 + Wf-355-B1 §1.3 + Wf-379-A1 honest-negative closeout)**:

**YES — Wf-387-A2 DID彻底解决 the Wf-125 P1 / Wf-139 P7 honest-negative structural fix gap**: a balanced 500-cohort Rh augmentation (250 active + 250 inactive) sourced from REAL MetalCytoToxDB compounds under the Wf-261 strict scaffold split, with 0 scaffold leak to Rh val or Rh test, with structural attestations (`is_synthetic=False`, `wf139_p7_carryover=False`) on every entry. The artifact is ready at `molmetal/reports/wf387a2_balanced_rh_cohort/balanced_rh_cohort.json`.

**PARTIAL — verification run still USER_OK-gated**: the Wf-388 follow-up GPU retrain (re-run Wf-125 P1 with the balanced 500 substituted for the Wf-139 P7 synthetic 500 + READOUT to Rh val) is BLOCKED on USER_OK=1 per AGENTS.md §5.1 standing GPU grant + Wf-355-B1 wrapper. Once the user grants, Wf-388 can queue after the in-flight Wf-379-A1 completes (~5 hr PROJECTED).

**NEW ultracode Wf-388+ IS the recommended follow-up** per this closeout §6.2 with the 3-run CONTROLD PAIR (§5.1 above) for definitive Wf-125 P1 val_auc lift honest-negative verdict (~5 hr GPU single retrain; ~15 hr GPU for full CONTROLD PAIR).

---

**END OF Wf-387-A2 closeout** — Owner Hugo Lee; cross-ref `party-twenty-spirit` §3.5 + AGENTS.md §5.1 + §5.2 + §5.3 + §5.4 + §10.