# Wf-133 Phase 4 — Orphan close-out synthesis + cross-link

**Date**: 2026-09-27
**Workflow**: Wf-133 Phase 4 (synthesis + cross-link of 7 orphan close-outs)
**Owner**: Hugo Lee + ultracode agents
**Skill alignment**: `party-twenty-spirit` §3.5 (honest framing mandatory)
**Inputs**:
- 7 close-out docs at `TODO/completed/{121,122,123,125,126,127,129}_*.md` (Wf-133 P1-P3 ships)
- `molmetal/reports/wf132_index_audit/data_assembly.md` (Wf-132 cross-reference, §7 ship-without-closeout list)
- `TODO/INDEX.md` §17.10 (the list this Wf-133 closes)
**Output**:
- This synthesis doc (`synthesis.md`, ≤ 200 lines)
- `TODO/INDEX.md` §17.10 patched (ship-without-closeout → 0)
- `closeout_inventory.json` (machine-readable inventory of 7 close-outs)

---

## §1 What this synthesis establishes

1. **Audit-3 ship-without-closeout count was 7** (Wf-121 / Wf-122 / Wf-123 / Wf-125 / Wf-126 / Wf-127 / Wf-129), per `wf132_index_audit/data_assembly.md` §7 (verbatim `Wf-121 .. Wf-129` rows).
2. **As of 2026-09-27**, all 7 close-out docs ship on disk. The Wf-132 INDEX §17.10 net count of **5** was a transitional number reflecting only Wf-121 + Wf-123 having closed at that point.
3. **Wf-133 Phases 1-3 + this Phase 4** close the loop: §17.10 ship-without-closeout drops from **5 → 0** (or read from scratch: **7 → 0**).
4. Every close-out doc follows the same **5-section shape**: `§0 TL;DR` + `§1 Deliverables (on disk)` + `§2 MEASURED` + `§3 NOT SHIP / deferred` + `§4 Cross-references` + `§5 Honest framing (party-twenty-spirit §3.5)`.

---

## §2 The 7 close-outs at a glance

| Wf | Date | Path | Status | Headline |
|---|---|---|---|---|
| Wf-121 | 2026-09-27 | `TODO/completed/121_meko_metal_atom_extension_closeout.md` | 🚀 **P0 PARTIAL LIFT SHIP** | cisplatin × 10 pockets MEASURED kcal/mol -2.548 to -3.734; 8 d-block Pt-group SMARTS ship |
| Wf-122 | 2026-09-27 | `TODO/completed/122_hilbert_l2_astsr_guard_closeout.md` | ⚠️ **HONEST PARTIAL SHIP** | L2 patch 16/16 PASS + ASTR guard 4/5 seeds; P3 post-fix bench still negative on PlatinAI 4-cell (12-d RDKit signal floor per Wf-49 §Cause 2) |
| Wf-123 | 2026-09-27 | `TODO/completed/123_master_plan_wasserstein_closure_closeout.md` | 🚀 **P0 MASTER SHIP** | M1 wasserstein (43/43 tests) + M2 pocket_closure_bias (55/55 tests); foundation for 6 strengthen TODOs |
| Wf-125 | 2026-09-27 | `TODO/completed/125_rh_oversample_wasserstein_transfer_closeout.md` | ✅ **SHIP** | 500 Rh synthetic cohort, Catherine 6/6 PASS rate 14.92%; W₂ decay +38.30% (diversity, NOT density) |
| Wf-126 | 2026-09-27 | `TODO/completed/126_rmsd_sa_vina_stretch_closeout.md` | ✅ **P1+P2 SHIP** | 5/5 pockets READY + SE(3) Schrödinger-Bridge adapter PASS; P3 SA+per-pocket Vina deferred |
| Wf-127 | 2026-09-27 | `TODO/completed/127_l5_hilbert_diagnostic_closeout.md` | ✅ **P1+P2 SHIP** | Parameter-sweep diagnostic + L2 weight-decay guard; root cause = 12-d signal floor (NOT L2/tune) |
| Wf-129 | 2026-09-27 | `TODO/completed/129_l2_l7_variance_wasserstein_closeout.md` | ✅ **P1 SHIP (script+tests)** | bench_variance_mcts_wasserstein + bench_multiph_pll_end_to_end_guarded on disk; bench did NOT execute (no-torch host) |

**Net**: 7/7 close-out docs on disk; **3 SHIP**, **3 PARTIAL SHIP** (P3 deferred), **1 SCRIPT-SHIP (bench non-execute on CPU)**.

---

## §3 Honest framing — what each close-out DOES NOT yet claim (per §3 of each doc)

| Wf | What ships | What is **NOT** claimed |
|---|---|---|
| Wf-121 | cisplatin × 10 pockets MEASURED | Full 8 d-block metallodrug sweep (Tc/Pm/lanthanide + 4-6 wk R20 GPU deferred); `M` atype = `MEASURED_GENERIC_ATOM` (NOT calibrated Pt VDW) |
| Wf-122 | L2/ASTR guards on synthetic + controlled probes | PlatinAI 4-cell R² lift (combined R² still negative — 12-d RDKit signal floor is the architectural ceiling; Wf-49 §Cause 2 territory) |
| Wf-123 | M1+M2 modules + integration tests | Any downstream metric lift (M1+M2 are primitives — 6 strengthen TODOs consume them) |
| Wf-125 | Rh synthetic cohort + W₂ decay | LOMO Rh-bottleneck fix (requires joint W₂ training with multi-task D-MPNN, Wf-131 P3 deliverable) |
| Wf-126 | 5-pocket staging + SE(3) bridge | m5 SA / m15 RMSD / per-pocket Vina real-data lift (P3 IN-FLIGHT, depends on Wf-123 M2 + 5-pocket completion) |
| Wf-127 | sweep diagnostic + L2 guard on synthetic | PlatinAI 4-cell joint sweep (depends on Wf-56 R20-P2 GPU retrain); the **root cause** is 12-d RDKit signal floor, NOT L2/tune |
| Wf-129 | bench scripts + 12 unit tests PASS | Real GPU bench (deferred to R20-P1 / R20-P3 RTX 5090); current `variance_reduction = 1.0` is a `1 - 0/1e-12` artifact, NOT a real measurement |

---

## §4 Cross-link to Wf-132 data_assembly.md §7

Per `molmetal/reports/wf132_index_audit/data_assembly.md` §7 verbatim list (audit-snapshot 2026-09-26 19:20 UTC), all 7 rows now have ✅ close-out:

| Wf (wf132 §7 row) | wf132 row verdict | Wf-133 P4 verdict |
|---|---|---|
| Wf-121 (row 1) | ✅ SHIP 2026-09-27 (post-audit delta noted) | ✅ confirmed (close-out verbatim unchanged) |
| Wf-122 (row 2) | still ❌ at audit time | ✅ closed 2026-09-27 (Wf-133 P1) |
| Wf-123 (row 3) | ✅ SHIP 2026-09-27 (post-audit delta noted) | ✅ confirmed (close-out verbatim unchanged) |
| Wf-125 (row 4) | still ❌ at audit time | ✅ closed 2026-09-27 (Wf-133 P1) |
| Wf-126 (row 5) | still ❌ at audit time | ✅ closed 2026-09-27 (Wf-133 P2 P1+P2 ship) |
| Wf-127 (row 6) | still ❌ at audit time | ✅ closed 2026-09-27 (Wf-133 P2 P1+P2 ship) |
| Wf-129 (row 7) | still ❌ at audit time | ✅ closed 2026-09-27 (Wf-133 P2 P1 ship) |

**Net delta vs Wf-132 §7**: 5 rows (Wf-122 / Wf-125 / Wf-126 / Wf-127 / Wf-129) flipped from ❌ → ✅. The 2 already-shipped rows (Wf-121 / Wf-123) confirmed verbatim.

---

## §5 INDEX.md §17.10 patch (this is what this Phase 4 ships)

**Old line 479 heading**: `### §17.10 Ship-without-closeout 7 (Wf-121/122/123/125/126/127/129)`

**New line 479 heading**: `### §17.10 Ship-without-closeout 0 (Wf-121/122/123/125/126/127/129 — all closed in Wf-133)`

**Old line 493 verdict**: "Net ship-without-closeout as of 2026-09-27: **5** (Wf-122, Wf-125, Wf-126, Wf-127, Wf-129) — reduced from audit3's 7 after Wf-121 + Wf-123 closed out 2026-09-27."

**New line 493 verdict**: "Net ship-without-closeout as of 2026-09-27: **0** — all 7 closed in Wf-133 Phases 1-3 (Wf-121/123 already-shipped confirmed; Wf-122/125/126/127/129 closed 2026-09-27). Down from audit3's 7 and Wf-132's interim 5."

The 7-row table is preserved (each row's post-audit delta column updated to ✅ SHIP 2026-09-27 with the Wf-133 close-out path); §17.12 cross-references add rows 13-19 (one per close-out doc) pointing to `wf133_orphan_closeout/synthesis.md` as the umbrella anchor.

---

## §6 Cross-link to §17.2 / §17.6 / §17.11 (party-twenty-spirit §3.5 trace-back)

| §17 row | What §17 cited before Wf-133 | What §17 cites after Wf-133 |
|---|---|---|
| §17.2 row 1 (Wf-121 P0 partial lift) | close-out only in §17.10 | unchanged — also now linked from §17.10 row 1 → row 1 of §17.2 via `closeout_inventory.json` |
| §17.6 row 3 (vina_status stub-floor) | Wf-121 P3 MEASURED (post-audit delta) | unchanged (no ship delta from Wf-133) |
| §17.11 health-ratio verdict | 47%/53% → ~49%/51% post-audit uplift (Wf-121 + Wf-123 + Wf-119 + Wf-130) | unchanged (no ship delta from Wf-133); the 5 newly-closed close-outs are **defensive** (L2/ASTR guard, diagnostic, bench infrastructure), NOT metric-lifting — honest-negative uplift only |
| §17.12 cross-refs (rows 9, 10, 12) | Wf-121, Wf-123, Wf-130 close-out cross-refs | + rows 13-19 for Wf-122/125/126/127/129 close-outs |

**Honest framing**: closing the ship-without-closeout loop is a **documentation discipline** win, not a **metric lift**. The health ratio verdict (47%/53% real/mock split) is unchanged by Wf-133. Paper §6 honest-negative count is unchanged.

---

## §7 Verification (5-section grep + line count)

Per task spec, verification of all 7 close-outs:

```
$ for f in 121_meko_metal_atom_extension_closeout 122_hilbert_l2_astsr_guard_closeout \
           123_master_plan_wasserstein_closure_closeout 125_rh_oversample_wasserstein_transfer_closeout \
           126_rmsd_sa_vina_stretch_closeout 127_l5_hilbert_diagnostic_closeout \
           129_l2_l7_variance_wasserstein_closeout; do
    echo "=== $f ==="
    grep -E "^## §[0-9]" TODO/completed/${f}.md
  done

121: §0 TL;DR · §1 Deliverables (3 phases) · §2 MEASURED · §3 NOT SHIP/deferred · §4 Cross-references · §5 Honest framing
122: §0 TL;DR · §1 Deliverables (on disk) · §2 MEASURED (CPU-only) · §3 NOT SHIP/deferred · §4 Cross-references · §5 Honest framing
123: §0 TL;DR · §1 Deliverables (M1+M2 modules) · §2 MEASURED (per-loss/function verification) · §3 NOT SHIP/deferred · §4 Cross-references · §5 Honest framing
125: §0 TL;DR · §1 Deliverables (on disk) · §2 MEASURED (CPU-only) · §3 NOT SHIP/deferred · §4 Cross-references · §5 Honest framing
126: §0 TL;DR · §1 Deliverables (on disk) · §2 MEASURED · §3 NOT SHIP/deferred · §4 Cross-references · §5 Honest framing
127: §0 TL;DR · §1 Deliverables (on disk) · §2 MEASURED · §3 NOT SHIP/deferred · §4 Cross-references · §5 Honest framing
129: §0 TL;DR · §1 Deliverables (on disk) · §2 MEASURED · §3 NOT SHIP/deferred · §4 Cross-references · §5 Honest framing
```

All 7 close-outs: 5-section shape (with §0 TL;DR) ✅; line count 121-197 (well above the minimum) ✅; verbatim close-out path on disk ✅.

---

## §8 Down-stream follow-ups (out of Wf-133 scope, captured for traceability)

These remain open after Wf-133 P4 — captured here so they don't get lost:

1. **Wf-122 P3 post-fix bench** — PlatinAI 4-cell combined R² still negative; the **real** fix is Multi-PH+PLL feature augmentation (Wf-79 P2 / Wf-94 R20-P3), NOT Wf-122's L2/ASTR guards.
2. **Wf-126 P3** — SA + per-pocket Vina re-bench IN-FLIGHT (depends on Wf-123 M2 + 5-pocket completion).
3. **Wf-127 P3** — joint sweep on PlatinAI 4-cell IN-FLIGHT (depends on Wf-56 R20-P2 GPU retrain).
4. **Wf-129 P1 bench** — `variance_reduction = 1.0` is a CPU-host artifact; real GPU bench deferred to R20-P1 / R20-P3 RTX 5090.
5. **Wf-121 full 8 d-block sweep** — cisplatin + 7/8 Pt-group metals dock; full sweep (Tc/Pm/lanthanide) deferred 4-6 wk R20 GPU.
6. **Wf-124 `yuelbond_decoder.py`** — still NOT shipped (in-flight, per wf132 §6); NOT a Wf-133 close-out target.

None of these are blockers for paper-acceptable honest-negative framing under `party-twenty-spirit §3.5`.

---

## §9 §17 INDEX.md cross-link anchor (so future Wf finds this synthesis)

For Wf-134+ readers: `molmetal/reports/wf133_orphan_closeout/synthesis.md` is the umbrella synthesis for the Wf-133 orphan close-out cleanup. The 7 individual close-out docs at `TODO/completed/{121,122,123,125,126,127,129}_*.md` each have their own §4 Cross-references — none of those point to this synthesis doc explicitly; the INDEX.md §17.12 cross-link is the canonical pointer.

**This synthesis is the trace-back anchor** for: "did Wf-133 really close all 7 ship-without-closeout rows from audit3?" — yes (see §4 cross-link table above + `closeout_inventory.json` for machine-readable evidence).

---

**End of Wf-133 P4 synthesis — 0 ship-without-closeout remaining as of 2026-09-27.**
