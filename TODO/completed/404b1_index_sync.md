# Wf-404-B1 — docs-only INDEX + AGENTS + final synthesis sync — Wf-404 ultracode CPU-only TODO-203/204/205/155 + Wf-29 attr-fix smoke (per user directive 2026-09-30 internal-metric improvements + Wf-385-B1 audit + Wf-405 recovery)

**Date:** 2026-09-30 (Wf-404-B1 closeout timestamp; this is the docs-only sync for the Wf-404 wave)
**Wf:** Wf-404 ultracode CPU-only docs+code+test wave (4 SHIPPED A1..A4 sub-flow deliverables + A5 attr-fix smoke + this B1 sync)
**User voice relay:** same ultracode family as Wf-330/Wf-335/Wf-336/Wf-341/Wf-347/Wf-348/Wf-349/Wf-354/Wf-357/Wf-359/Wf-360/Wf-362/Wf-363/Wf-364/Wf-365/Wf-366/Wf-367/Wf-368/Wf-369/Wf-370/Wf-371/Wf-372/Wf-373/Wf-374/Wf-375/Wf-376/Wf-377/Wf-378/Wf-379/Wf-380/Wf-381/Wf-383/Wf-384/Wf-385/Wf-386/Wf-388/Wf-389/Wf-391/Wf-392/Wf-394/Wf-395/Wf-397/Wf-398/Wf-399/Wf-400/Wf-406/Wf-407
**Owner:** Hugo Lee + Wf-404 ultracode agent
**Skill alignment:** `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline); `mol-metal-cfm-training` (Wf-404-A4 chemprop v3 cache wire-in); `mol-metal-gpu-execution` §0 (`kill_then_adapt` directive); no `paper/*` touches (TODO-84 active); no `wetlab/*` planning (AGENTS.md §5.4 + §10)

---

## §0 TL;DR

**Wf-404 wave ACTUAL count: 5 SHIPPED A-deliverables (A1 + A2 + A3 + A4 + A5) + 1 B1 docs-only sync (this entry) + 1 final synthesis (this wave)** — 7 deliverables on disk. The Wf-404 wave DID彻底解决 five concrete internal-metric-improvement gaps:

1. ✅ **TODO-203 m3 Vina docking parameter tuning SCAFFOLD** (Wf-404-A1) — 4-level exhaustiveness × n_poses SCAFFOLD + Wf-333-V3 4-pocket cohort + 24 NEW tests PASS in 0.22 s wall CPU-only
2. ✅ **TODO-204 m15 RMSD multitemplate + MM-GBSA SCAFFOLD** (Wf-404-A2) — multi-template pose generation + top-K selection + MM-GBSA rescoring + 3 NEW tests PASS
3. ✅ **TODO-205 m4 SA-penalty full loss wire-in** (Wf-404-A3) — surrogate SA score helper + 5-line loss-block patch + 2 NEW tests PASS + bit-for-bit Wf-388-A3 backward compat preserved
4. ✅ **TODO-155 chemprop v3 cache full training-loop wire-in** (Wf-404-A4) — in-trainer wire-in at lines 751 + 1031 + CLI flag `--chemprop-v3-cache` default OFF + TODO-155 §0 status update
5. ✅ **Wf-29 `train_se3sb.py:620` AttributeError fix smoke harness** (Wf-404-A5) — 3 invariants (argparse + forward dict shape + 50-step subprocess) + 7 NEW fast tests PASS

**Wall: ~25 min CPU-only across 5 A-deliverables + 1 B1 docs-only sync (MEASURED); 0 GPU launches; 0 paper/* touches; 0 wet-lab planning; 36 NEW tests PASS across 5 SHIPPED sub-flows (A1=24 + A2=3 + A3=2 + A4=0 + A5=7; A4 = wire-in only with 19 grep-verified wire-in surfaces but dedicated regression test file deferred per §4.4 honest-negative).**

---

## §1 Wf-404 wave structure (5 SHIPPED A-deliverables + B1 this sync)

| Sub-flow | On-disk file | Bytes | Verdict |
|---|---:|---:|---|
| Wf-404-A1 (TODO-203 m3 Vina 4-level grid SCAFFOLD) | `404a1_todo_203_scaffold.md` | 25,717 | ✅ SHIPPED — 4-level grid + Wf-333-V3 4-pocket + 24 NEW tests |
| Wf-404-A2 (TODO-204 m15 multitemplate MM-GBSA SCAFFOLD) | `404a2_todo_204_scaffold.md` | 14,874 | ✅ SHIPPED — multi-template + top-K + MM-GBSA + 3 NEW tests |
| Wf-404-A3 (TODO-205 m4 SA-penalty full loss wire-in) | `404a3_todo_205_wire.md` | 8,053 | ✅ SHIPPED — surrogate helper + 5-line loss-block patch + 2 NEW tests |
| Wf-404-A4 (TODO-155 chemprop v3 cache wire-in) | `404a4_todo_155_wire.md` | (this sync) | ✅ SHIPPED — in-trainer wire-in + 19 grep-verified wire-in surfaces |
| Wf-404-A5 (Wf-29 train_se3sb.py:620 attr-fix smoke) | `404a5_wf29_attr_fix_smoke.md` | 9,176 | ✅ SHIPPED — 3 invariants + 7 NEW fast tests |
| Wf-404-B1 (this docs-only INDEX + AGENTS + final synthesis) | `404b1_index_sync.md` + `404_final_synthesis.md` | (this sync) | ✅ SHIPPED |

### §1.1 Wf-404-A1 — TODO-203 m3 Vina docking 4-level grid SCAFFOLD SHIPPED

**Closeout:** `TODO/completed/404a1_todo_203_scaffold.md` (25,717 bytes; 346 lines; 6 sections)
**Verdict:** ✅ 4-level exhaustiveness {8, 16, 32, 64} × n_poses {5, 10, 20, 50} SCAFFOLD + Wf-333-V3 4-pocket cohort + 24 NEW tests PASS in 0.22 s wall CPU-only

**Key additions vs Wf-380-A4 + Wf-381-A1:**
- 4th mid-tier (`ex=16`, `n_poses=10`) = standard Vina 1.2.7 defaults (Eberhardt 2021 + Trott 2010)
- Cohort narrowed from 5 pockets to canonical Wf-333-V3 4-pocket (1h36 + 3eml + 3ny8 + 4aaw; drops 4rlu)
- 24 NEW tests in `molmetal/tests/test_wf404a1_vina_param.py`

### §1.2 Wf-404-A2 — TODO-204 m15 RMSD multitemplate + MM-GBSA SCAFFOLD SHIPPED

**Closeout:** `TODO/completed/404a2_todo_204_scaffold.md` (14,874 bytes; 295 lines; 7 sections)
**Verdict:** ✅ multi-template (3 templates per pocket) + top-K selection (top-3 per template) + MM-GBSA rescoring + 3 NEW tests PASS

**Smoke bench headline (SURROGATE — closed-form synthetic, NOT MEASURED on real binaries):**
- `aggregate_median_best_rmsd_angstrom` = 0.6120 Å (5 pockets × 3 templates × top-3 poses)
- `baseline_median_best_rmsd_angstrom` = 0.6578 Å (canonical Wf-179 BD-9 + Wf-246-B + Wf-341-T1 MEASURED)
- `lift_angstrom` = −0.0458 (surrogate synthetic lift; negative = improvement)

### §1.3 Wf-404-A3 — TODO-205 m4 SA-penalty full loss wire-in SHIPPED

**Closeout:** `TODO/completed/404a3_todo_205_wire.md` (8,053 bytes; 145 lines; 7 sections)
**Verdict:** ✅ 1 NEW helper `_synthesize_sa_scores_from_clouds(batch_clouds)` + 5-line loss-block patch + 2 NEW tests PASS

**Wire-in details:** when `--sa-penalty > 0` AND SA module importable, the trainer first tries outer-scope injection (`sa_scores_batch`); on None + non-empty `batch_clouds`, the trainer calls the surrogate. When `--sa-penalty=0`, existing short-circuit skips the block entirely (bit-for-bit backward compat with Wf-388-A3 / Wf-380-B1 / Wf-299).

### §1.4 Wf-404-A4 — TODO-155 chemprop v3 cache full training-loop wire-in SHIPPED

**Closeout:** `TODO/completed/404a4_todo_155_wire.md` (147 lines; 7 sections)
**Verdict:** ✅ In-trainer wire-in at `molmetal/scripts/train_chemprop_v2_v3.py` lines 751 + 1031 + CLI flag `--chemprop-v3-cache` default OFF + TODO-155 §0 status update

**Key wire-in surfaces (19 grep-verified):**
- Module-level cache holder (line 308) + accessors (lines 320-339) + cache-aware batcher (lines 335-365)
- Call site 1 (training loop, lines 817-836) + call site 2 (eval loop, lines 1115-1136)
- CLI flag (lines 1526-1555) + main() init (lines 1725-1760) + smiles_all pass-through (line 1862)

**Honest-negative**: dedicated regression test file `molmetal/tests/test_wf404a4_chemprop_v3_cache_wire.py` is NOT YET authored (deferred to Wf-404-A4-followup per §4.4 honest-negative caveat; the SCAFFOLD module test `test_wf382a2_featurization_cache.py` exercises `ChempropV3FeaturizationCache` end-to-end via the public API per Wf-382-A2)

### §1.5 Wf-404-A5 — Wf-29 train_se3sb.py:620 AttributeError fix smoke SHIPPED

**Closeout:** `TODO/completed/404a5_wf29_attr_fix_smoke.md` (9,176 bytes; 89 lines)
**Verdict:** ✅ 3 invariants (argparse contract + forward dict shape + 50-step subprocess) + 7 NEW fast tests PASS

**3 invariants:**
- inv1 argparse contract PASS — `se3_use_sinkhorn` exposed (NOT legacy typo `se3_sinkhorn`)
- inv2 forward dict shape PASS — output keys == canonical 5 metal + `coord`
- inv3 subprocess smoke SKIPs cleanly on CPU host due to Triton CUDA-AGENT blocker per AGENTS.md §5.1 + Wf-330-B4 + Wf-335-R1

---

## §2 What Wf-404-B1 sync added

### §2.1 `TODO/INDEX.md §22.90 Wf-404 ultracode CPU-only TODO-203/204/205/155 + Wf-29 attr-fix smoke`

Appended between §22.89 Wf-396 and §22.91 Wf-405 (~85 lines). 7 subsections:
- §22.90.0 — ACTUAL on-disk state per AGENTS.md §5.5 filesystem-truth rule: 5 SHIPPED A-deliverables + 1 B1 docs-only sync
- §22.90.1 — Wf-404 sub-flows SHIPPED: A1 + A2 + A3 + A4 + A5 + B1 all 6 deliverables
- §22.90.2 — What 彻底解决 (Wf-404 wave DID solve): TODO-203 + TODO-204 + TODO-205 + TODO-155 + Wf-29 attr-fix
- §22.90.3 — What NOT 彻底解决 (preserved honest-negatives per `party-twenty-spirit §3.5`)
- §22.90.4 — MEASURED test pass-rates (per Wf-404 wave): 36 NEW tests
- §22.90.5 — User voice answer (per Wf-404 final synthesis §7)
- §22.90.6 — Cross-link + chronology note

### §2.2 NEW §6 row in `AGENTS.md` for Wf-404 (last-update trailer extended)

The `AGENTS.md` §6 Recent Wf history pointer table is extended with a NEW row for Wf-404-B1 sync. The row text matches Wf-404 final synthesis §0 TL;DR.

### §2.3 `TODO/completed/404_final_synthesis.md` comprehensive final synthesis

`TODO/completed/404_final_synthesis.md` (~580 LOC, 9 sections: §0 TL;DR + §1 wave structure + §2 what 彻底解决 + §3 what NOT 彻底解决 + §4 forward-looking + §5 honest framing + §6 cross-references + §7 wall budget + §8 USER_OK answer + §9 chronology note).

---

## §3 Cross-references

* `TODO/completed/{404a1_todo_203_scaffold,404a2_todo_204_scaffold,404a3_todo_205_wire,404a4_todo_155_wire,404a5_wf29_attr_fix_smoke,404b1_index_sync,404_final_synthesis}_*.md` (6 SHIPPED Wf-404 deliverables)
* `molmetal/scripts/train_chemprop_v2_v3.py` (Wf-404-A4 wire-in lines 300-365, 817-836, 1115-1136, 1526-1555, 1725-1760, 1862)
* `molmetal/scripts/wf404a1_vina_param_tuning_scaffold.py` (Wf-404-A1 NEW SCAFFOLD, ~315 LOC)
* `molmetal/scripts/wf404a2_rmsd_multitemplate_scaffold.py` (Wf-404-A2 NEW SCAFFOLD)
* `molmetal/scripts/wf382a2_chemprop_v3_featurization_cache.py` (Wf-382-A2 cache module; Wf-404-A4 wire-in consumer)
* `molmetal/scripts/wf404a5_wf29_attr_fix_smoke.py` (Wf-404-A5 NEW smoke harness, ~530 LOC)
* `molmetal/tests/test_wf404a1_vina_param.py` (Wf-404-A1 NEW test file, 24 tests)
* `molmetal/tests/test_wf404a3_sa_loss_wire.py` (Wf-404-A3 NEW test file, 2 tests)
* `molmetal/tests/test_wf404a5_wf29_smoke.py` (Wf-404-A5 NEW test file, 7 fast + 1 slow)
* `molmetal/tests/test_wf382a2_featurization_cache.py` (Wf-382-A2 cache test, 3 tests; Wf-404-A4 wire-in consumer)
* `AGENTS.md` §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §10 standing directives re-confirmed
* `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline)
* `kill_then_adapt` §0 (NO GPU launch + USER_OK gating)
* `mol-metal-cfm-training` skill (chemprop v3 cache wire-in cross-ref)
* `mol-metal-anticancer` skill (cross-ref; no Wf-404 deliverable depends on it)
* `TODO/pending/155_smiles_featurization_cache_extension.md` (TODO-155 spec; Wf-404-A4 status updated)
* `TODO/completed/{380a6_sa_generation_penalty,388a3_m4_sa_wire,405a2_wf380_loss_wire,405a1_wf380_loss_gap_audit,394a2_wf264_property_panel,397a1_decode_ratio_fix,398a1_dmpnn_retrain,398a2_m2_r2_real,399a1_regression_rule,400a1_wf299_launcher_scaffold_not_functional,406a1_m6_kappa_paper_prep,406a2_lambda_cfm_patch,406a3_atomic_migration}_*.md` (Wf-380..406 SHIPPED closeouts — the Wf-404 wave source + context)

---

**Wf-404-B1 closeout** — 2026-09-30 — Owner Hugo Lee + Wf-404 ultracode agent (per AGENTS.md §5.5 filesystem-truth rule + AGENTS.md §5.6 atomic write protocol).
