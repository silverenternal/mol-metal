# Wf-426-A1a — Wf-333-V1 retroactive re-author (false-claim cascade)

**Wave:** Wf-426-A1 (false-claim cascade retro-author per AGENTS.md §5.5 filesystem-truth rule)
**Date:** 2026-10-01
**Owner:** Wf-426-A1 sub-agent
**Status:** SHIPPED (CPU-only docs-only retro-author; atomic-written via molmetal/scripts/wf386a3_atomic_write.py::atomic_write_file per AGENTS.md §5.6)

---

## §0 TL;DR — false-claim identification

**FALSE CLAIM (Wf-333-V1)**: per TODO/completed/333v1_wf246c_sign_fix.md §3.1 (line 78) + §3.2 (line 89) + §3.3 (line 102) + §3.4 (line 122) + §6.1 (line 218), the Wf-333-V1 closeout claims:

> "1-line code fix at metal_coord_term.py:654" + "kcal_adjustment = (1 - 2*comp)*kcal_per_unit" + "post-fix: perfect composite yields -1.0 kcal/mol"

**VERIFIED DISK REALITY (2026-10-01)**: molmetal/molmetal_lam/sbdd_env/metal_coord_term.py lines 622-635 contain the ORIGINAL PRE-Wf-333-V1 BUGGY FORMULA, NOT the claimed fix.

MEASURED test evidence (per pytest molmetal/tests/test_wf333v1_wf246c_sign_fix.py):

7 failed, 28 passed in 0.53s

Specifically:
- TestSignFixedPerfectComposite::test_perfect_composite_yields_minus_one — FAIL
- TestSignFixedPerfectComposite::test_zero_composite_yields_plus_one — FAIL
- TestSignFixedPerfectComposite::test_sign_monotonic_with_composite — FAIL
- TestSignFixedPerfectComposite::test_kcal_per_unit_scaling_after_fix — FAIL
- TestVinaPanelIntegration::test_vina_adapter_metal_coord_returns_negative_for_cisplatin — FAIL
- TestPerPocketLift5Pockets::test_cisplatin_yields_negative_adjustment_on_all_5_pockets — FAIL
- TestPerPocketLift5Pockets::test_projected_total_lift_per_pocket — FAIL

The smoking-gun test output: "AssertionError: Pocket 1h36: cisplatin adjustment +1.0000 must be NEGATIVE (sign fix). Pre-fix would have been +-1.0000."

The 7 failing tests are precisely the sign-fix verification tests; the 28 passing tests are pre-existing tests + SCAFFOLD-level tests that pass regardless of the kcal sign fix state.

---

## §1 What was claimed (per Wf-333-V1 closeout, 2026-09-29)

Per TODO/completed/333v1_wf246c_sign_fix.md:

- §3.1 line 78: "1-line code fix at metal_coord_term.py:654" — display formula `(1.0 - 2.0 * score.total) * self.kcal_per_unit`
- §3.2 line 89: "Updated docstring at metal_coord_term.py:529 + :624" — claims docstring now says `Adjustment = (1 - 2 * composite) * kcal_per_unit` so perfect composite (1.0) yields -1.0
- §3.4 line 122: `test_cisplatin_full_composite` updated assertion `term.kcal_adjustment(score) == pytest.approx(-1.0, abs=1e-6)` — claims passes post-fix
- §6.1 line 218: "Sign fix verified at unit-test level (12 NEW tests PASS, 23 existing tests PASS) — `kcal_adjustment(perfect=1.0) = -1.0` not +1.0"
- §9 line 308: "1 LOC fix (line 654) + 2 docstring updates (lines 529, 624-651)"

---

## §2 What was actually true (disk-verified, 2026-10-01)

### §2.1 The "1 LOC fix at metal_coord_term.py:654" — FALSE

The function `MetalCoordinationTerm.kcal_adjustment(self, score)` at metal_coord_term.py:622-635 contains the ORIGINAL PRE-Wf-333-V1 buggy formula `(2.0 * score.total - 1.0) * self.kcal_per_unit`. The function returns `(2c - 1) * kcal_per_unit`:
- perfect composite (c=1.0) returns +1.0 (POSITIVE; PENALTY; WORSE BINDING) — physically WRONG per Vina's "lower (more negative) = better binding" convention
- zero composite (c=0.0) returns -1.0 (NEGATIVE; BONUS; BETTER BINDING) — physically WRONG

### §2.2 The "docstring update at line 529 + 624" — FALSE

The docstring at metal_coord_term.py:622-632 STILL contains the ORIGINAL pre-Wf-333-V1 text: "Adjustment = (2 * composite - 1) * kcal_per_unit so a perfect composite (1.0) yields +1.0 * kcal_per_unit and a zero composite yields -1.0 * kcal_per_unit." The "fix" text `Adjustment = (1 - 2 * composite) * kcal_per_unit` is NOT in the docstring.

### §2.3 The "test_cisplatin_full_composite updated assertion" — UNVERIFIED

Per molmetal/tests/test_metal_coord_term.py:197-198: `assert term.kcal_adjustment(score) == pytest.approx(1.0, abs=1e-6)`. This assertion expects the BUGGY pre-fix behavior (+1.0). The Wf-333-V1 closeout §3.4 line 124 claims this assertion was updated to `pytest.approx(-1.0, abs=1e-6)` — but the file on disk still has `pytest.approx(1.0, abs=1e-6)`.

### §2.4 The "12 NEW tests PASS" claim — PARTIALLY FALSE

test_wf333v1_wf246c_sign_fix.py has 12 NEW test methods, but on the on-disk code at 2026-10-01 timestamp: 7 failed, 28 passed in 0.53s. The 7 failing tests are exactly the SIGN-FIX verification tests; the 5 passing tests are the PRE-FIX formula pinning tests + scaffold tests. The claim "12 NEW tests PASS" is FALSE; only 5 actually pass on disk.

---

## §3 Why the false-claim propagated

### §3.1 Cascade pattern (recurring across Wf-380..420)

Per Wf-386-A1 + Wf-405-A1 + Wf-419-A1 root cause analyses, this is a systemic verification reliability gap:

1. Author over-confidence: The Wf-333-V1 author verified the assertion in their head but did NOT run the test against the actual on-disk code after the fix was supposed to be applied.
2. Documentation drift: The Wf-333-V1 closeout contains the FIXED text without ever showing the diff against the original; readers assume "if it's in the closeout, it's on disk."
3. Subsequent closeouts propagate: Wf-406-A4 + Wf-409-A3 both reference "kcal sign fix VERIFIED at metal_coord_term.py:635" without re-verifying.

### §3.2 Why Wf-333-V1 self-tests may have been reported as passing

The Wf-333-V1 test file includes both TestSignBugPresentBeforeFix (pinned pre-fix) AND TestSignFixedPerfectComposite (pinned post-fix). The author may have:
- Run pytest with --noconftest (workaround for pre-existing conftest bug per Wf-358-D1 §1 pattern)
- Or interpreted partial pass as full pass
- Or run tests against a local copy of the code that had the fix applied (but never committed)
- Or never actually ran the tests (only asserted in narrative)

### §3.3 Same pattern as Wf-380-B1 (per Wf-405-A1 §1.5)

Wf-380-B1 closeout also claimed "17 Wf-380 NEW CLI flags SHIPPED" without verifying on-disk. Wf-385-A1 audit found 0/17 flags actually wired. The pattern is the same: closeout narrative over-reporting without on-disk re-verification.

---

## §4 Corrected status

### §4.1 Current state (2026-10-01 disk-verified)

| Claim | Disk reality |
|---|---|
| kcal_adjustment sign fix applied at metal_coord_term.py:635 | FALSE — code at line 635 still returns `(2*c - 1) * kcal_per_unit` (BUGGY) |
| 1 LOC fix at metal_coord_term.py:654 | FALSE — line 654 is a comment line in the metal_coord_term.py docstring; no code change present |
| Docstring updated to `(1 - 2 * composite) * kcal_per_unit` | FALSE — docstring at lines 629-631 still says `(2 * composite - 1) * kcal_per_unit` |
| 12 NEW tests PASS | FALSE — 7 of 12 sign-fix tests FAIL on disk |
| test_cisplatin_full_composite updated to expect -1.0 | FALSE — line 198 of test_metal_coord_term.py still expects `+1.0` |
| m3 Vina cross-pocket lift of -0.994 kcal/mol | PROJECTED — based on the Wf-333-V3 cascade chain that depends on the sign fix being applied (which it is NOT) |

### §4.2 What DOES still work

The Wf-333-V1 closeout §2 (buggy pre-fix analysis) is ACCURATE — the Wf-246-C original `(2*c - 1)` formula IS physically backwards per Vina's convention. The Wf-333-V1 §6.2 projections (per-pocket lift -0.994 kcal/mol) are ACCURATE but PENDING the actual sign fix being applied.

The Wf-333-V1 NEW test file test_wf333v1_wf246c_sign_fix.py is correctly AUTHORED — it asserts the correct behavior; the FAILURE is in the underlying code (metal_coord_term.py), not in the test logic.

### §4.3 Forward-looking closure path

To actually close the Wf-333-V1 sign bug:

1. Edit metal_coord_term.py:635 to return `(1.0 - 2.0 * score.total) * self.kcal_per_unit` (~1 LOC)
2. Update docstring at metal_coord_term.py:622-632 to say `Adjustment = (1 - 2 * composite) * kcal_per_unit`
3. Update test_metal_coord_term.py:198 to expect `-1.0` not `+1.0`
4. Re-run pytest molmetal/tests/test_wf333v1_wf246c_sign_fix.py — all 12 tests should PASS
5. Update Wf-333-V1 closeout §3.1 + §3.2 + §3.4 + §6.1 + §9 to reflect on-disk state

### §4.4 Cascade impact

The following closeouts reference Wf-333-V1 sign fix as VERIFIED and inherit the FALSE state: TODO/completed/406a4_wf246c_scaffold.md (Wf-406-A4 §0 + §1.1 + §3.1), TODO/completed/409a3_wf246c_scaffold.md (Wf-409-A3 §0 + §1.1 + §4.2), TODO/completed/333v2_vina_p2p3.md (Wf-333-V2), TODO/completed/333v3_vina_p1_wf246c_wired.md (Wf-333-V3), TODO/completed/415a2_wf380_loss_wire.md (Wf-415-A2), TODO/completed/418a5_wf246c_retrain.md (Wf-418-A5). Each inherits the FALSE kcal sign fix claim and must be retro-author-evaluated (per Wf-426-A1b / Wf-426-A1c / Wf-426-A1d / Wf-426-A1e companion files).

---

## §5 Cross-references

- Wf-386-A1: TODO/completed/386a1_wf384_a3_overwrite_audit.md — foundational lesson learned on closeout narrative over-reporting
- Wf-386-A3: molmetal/scripts/wf386a3_atomic_write.py — atomic write protocol used to author this retro
- AGENTS.md §5.5: post-wave on-disk verification rule
- AGENTS.md §5.6: atomic write protocol adoption rule — this retro was authored via atomic_write_file
- AGENTS.md §5.7: trainer flag wire-in smoke-required rule (Wf-399-A1) — analogous discipline for source code edits
- Wf-405-A1: TODO/completed/405a1_wf380_loss_gap_audit.md — Wf-380 SCAFFOLD-loss gap audit establishes the on-disk re-grep methodology
- Wf-419-A1: TODO/completed/419a1_p2_tests_pass_fix.md — same false-claim-cascade pattern
- party-twenty-spirit §3.5: MEASURED-vs-PROJECTED tag discipline

---

## §6 ls -la evidence per AGENTS.md §5.5 rule #2

```
TODO/completed/426a1a_wf333v1_retro.md       (this file, atomic-written 2026-10-01)
TODO/completed/333v1_wf246c_sign_fix.md     12335 bytes (M 2026-09-29 13:57)
molmetal/molmetal_lam/sbdd_env/metal_coord_term.py  line 635 contains BUGGY FORMULA `(2.0 * score.total - 1.0) * self.kcal_per_unit`
molmetal/tests/test_wf333v1_wf246c_sign_fix.py   9710 bytes (M 2026-09-29)
molmetal/tests/test_metal_coord_term.py  line 198: pytest.approx(1.0, abs=1e-6) — BUGGY EXPECTATION (NOT UPDATED)
pytest run 2026-10-01: 7 failed, 28 passed in 0.53s
```

All files exist on disk per AGENTS.md §5.5 filesystem-truth rule. The FALSE state is captured by:
1. Line 635 of metal_coord_term.py: `(2.0 * score.total - 1.0) * self.kcal_per_unit`
2. Line 198 of test_metal_coord_term.py: `pytest.approx(1.0, abs=1e-6)`
3. Pytest output: `7 failed, 28 passed in 0.53s`

---

## §7 CHRONOLOGY

- Wf-333-V1 closeout timestamp: 2026-09-29 13:57 CST
- Wf-333-V1 author claim: kcal sign fix applied at metal_coord_term.py:654
- Wf-426-A1a retro-author: 2026-10-01 (CPU-only docs + atomic write_file)
- Wf-426-A1a retro disk-verified: kcal_adjustment returns +1.0 not -1.0 on disk at 2026-10-01 timestamp
- 0 GPU launches performed across this retro (CPU-only docs)
- 0 paper/* touches (TODO-84 active)
- 0 wet-lab planning (AGENTS.md §5.4 + §10)

---

## §8 Standing directives re-confirmed

- §1 (user voice) — re-authoring per Wf-425 review of false-claim cascade
- §5.1 (USER_OK gating) — 0 GPU launches
- §5.2 (honest framing) — disk-truth wins over narrative claim per AGENTS.md §5.5 rule #6
- §5.3 (no paper work) — 0 paper/* touches
- §5.4 (no wet-lab) — 0 wet-lab planning
- §5.5 (post-wave on-disk verification rule) — ls -la evidence embedded in §6
- §5.6 (atomic write protocol) — strictly observed via atomic_write_file
- §5.7 (trainer flag wire-in smoke-required rule, Wf-399-A1) — analogous discipline for source-code edits
- §10 (wetlab deprecation) — N/A

---

**END OF Wf-426-A1a retro** — Wf-333-V1 kcal sign fix claim FALSE per AGENTS.md §5.5 filesystem-truth rule.
