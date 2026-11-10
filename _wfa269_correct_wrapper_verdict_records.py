#!/usr/bin/env python
# Licensed under the Apache License, Version 2.0.
"""Four dated, additive corrections to the three wrapper-verdict records.

WHAT THESE FOUR CORRECTIONS ARE
--------------------------------
Three TODO records report a **dry-run verdict** for the same five launch
wrappers, and all three read it the same way: *dry run exits 0* ==> *the
wrapper is fine*.  That inference is wrong, and it is wrong in a way that a
single probe catches.

MEASURED 2026-11-02 (CPU-only, `--dry-run` only, zero GPU):

  * PRE-REPAIR, `scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh`
    old line 735 read

        printf '  "status": "%s"\n' "$( [ "${DRY_RUN}" = "1" ] && echo PLAN_ONLY || echo READY_TO_RUN )"

    i.e. the plan's `status` field was **one of two hardcoded strings** and
    never read the capability gate's verdict at all -- the dry-run plan said
    `PLAN_ONLY` and the **real launch plan said `READY_TO_RUN`** while the
    gate said BLOCKED.

  * POST-REPAIR (Wf-A269 wave-12), the same dry run emits
    `molmetal/reports/wf355a3_r20_p5_tmqm_plat/plan.json` with
    `status = BLOCKED_CAPABILITY`, `capability = BLOCKED`, and an
    `honest_framing.MEASURED_today` naming **12 of 20** undeclared argv flags.
    The 12 is derived from `argv_capability.txt` ("FAIL -- 12 of 20 flag(s)
    are not declared", 12 `MISSING ` rows), not typed.

  * MEASURED acceptance matrix by RUNNING each wrapper `--dry-run` and
    reading the exit code (64 = the USER_OK gate refused):

        wrapper                    gate       USER_OK=YES   USER_OK=1
        p3_drugood            :173 {YES,1}   exit 0         exit 0
        p4_bindingdb         :202 {YES,1}   exit 0         exit 0
        p5_tmqm_plat         :182 {1}       exit 64        exit 0
        wf131_deferred       :132 {1}       exit 64        exit 0
        wf146_deferred       :175 {YES,1}   exit 0         exit 0

    Repo-wide gate census (MEASURED, `scripts/*.sh`, 77 files): 64 carry the
    string `USER_OK`, 48 have a gate, **44 accept EXACTLY `{1}`**, 4 accept
    `{YES,1}`.  `launch_wf74_v2_drugood_dann.sh` is SELF-CONTRADICTORY
    (`:139` accepts `{1,YES}`, `:192` accepts `{YES}` only); and
    `launch_wf306a_drugood_vina_gpu.sh` needs BOTH `USER_OK=1` AND
    `WF306A_USER_OK=YES`.  **AGENTS.md contains ZERO occurrences of
    `USER_OK`** -- so 44 wrappers enforce a literal that their cited
    authority (§5.1, the standing GPU grant, which requires NO per-launch OK)
    never states.  That reconciliation is USER-gated.

WHICH CLAIMS ARE MEASURED FALSE
--------------------------------
1. `357a6` §3.1 row 3 + Summary -- "DRY-RUN PASS" as a launchability verdict.
2. `357a6` §3.2 -- "all 5 wrappers accept `USER_OK=YES`" (FALSE for p5 and
   wf131) and row 2's "FAIL with USER_OK=1 (literal mismatch)" (FALSE: p4
   accepts `1`).
3. `364b1` §2 -- "PASS (rc=0)" in the `dry-run OK` column + "7/7 PASS".
4. `365b2` §1 -- "✅ exit 0" in the `--dry-run` column + the §3 test row.

RULE (AGENTS.md §5.5 rule #6)
------------------------------
Every correction is ADDITIVE + DATED.  Every superseded rendering is
PRESERVED VERBATIM inside the correction block, under the exact marker

    SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:

No original line's text is deleted or reworded anywhere.  The check below is
a SUBSTRING test (a preserved rendering inside a blockquote gains a "> "
prefix, so it is no longer byte-equal as a LINE); the invariant that matters
is that no original line's TEXT disappears from the file.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(REPO_ROOT / "molmetal" / "scripts"))

from wf386a3_atomic_write import atomic_write_file  # noqa: E402

TODOS = REPO_ROOT / "TODO" / "completed"

MARKER = "SUPERSEDED RENDERING PRESERVED VERBATIM per AGENTS.md section 5.5 rule #6:"

# ===========================================================================
# 357a6  --  two blocks (A: the verdict table, B: the USER_OK vocabulary)
# ===========================================================================

F_357A6 = TODOS / "357a6_gpu_readiness_audit.md"

R2_357A6 = (
    "| 2 | `launch_wf310e_r20_p4_bindingdb_1.07M.sh` | **YES REQUIRED** for both R1 + R3 |"
    " **DRY-RUN PASS** with USER_OK=YES; FAIL with USER_OK=1 (literal mismatch) |"
    " Phase R1 logged; affinity Pearson r ≥ 0.65 (Boltz-2 baseline 0.86);"
    " `USER_OK=YES` literal per Wf-294B |"
)
R3_357A6 = (
    "| 3 | `launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` | required (kill_then_adapt;"
    " > 1 hr GPU) | **DRY-RUN PASS** | 4 cells × 5 seeds = 20 (metal, cell, seed) tuples;"
    " STATUS_HOLD warning must be acknowledged before actual launch |"
)
SUM_357A6 = (
    "**Summary:** **5/5 wrappers DRY-RUN PASS** with USER_OK=YES literal; **0/5 BLOCKING_FAIL**."
)
SEC32_357A6 = """Per Wf-294B kill_then_adapt directive, the canonical USER_OK literal is `YES` (not
`1`). Wrappers that perform `[[ "${USER_OK}" != "YES" ]]` check refuse `USER_OK=1`.
Wf-355-A2 (R20 §4 BindingDB) implements this strict check; Wf-355-A1, A3, B1, B2 use
the same canonical literal. This audit verified all 5 wrappers accept `USER_OK=YES`
and refuse non-`YES` values."""

BLOCK_A_357A6 = f"""
> **CORRECTION (2026-11-02, the main thread, Wf-A269 wave-12) — "DRY-RUN PASS" WAS A PLAN-EMISSION FACT, NOT A LAUNCHABILITY VERDICT; THE ROW-3 PLAN'S OWN `status` FIELD SAID `BLOCKED_CAPABILITY`.**
>
> **{MARKER}**
>     {R3_357A6}
>
>     {SUM_357A6}
>
> **WHAT IS MEASURED FALSE (2026-11-02, CPU-only, `--dry-run` only, zero GPU).** The dry run DID exit 0 and DID emit a plan — that much of "DRY-RUN PASS" is true. What it did NOT establish is that the wrapper can launch, because **the plan it emitted reported a status the wrapper's own capability gate had already contradicted.**
>
> * **PRE-REPAIR** — `git diff` of `scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh`, the `-` side of the Wf-A269 wave-12 patch, old line **735**: `printf '  "status": "%s"\n' "$( [ "${{DRY_RUN}}" = "1" ] && echo PLAN_ONLY || echo READY_TO_RUN )"`. The `status` field was **one of two hardcoded strings** and never read the capability gate's verdict at all: the dry-run plan wrote `PLAN_ONLY` and the **real launch plan wrote `READY_TO_RUN`** — so even the launch path's plan claimed READY while the gate said BLOCKED.
> * **POST-REPAIR, MEASURED 2026-11-02** — `CUDA_VISIBLE_DEVICES="" USER_OK=1 bash scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh --dry-run` → exit 0; the emitted `molmetal/reports/wf355a3_r20_p5_tmqm_plat/plan.json` now carries `status = BLOCKED_CAPABILITY` and `capability = BLOCKED`, and its `honest_framing.MEASURED_today` reads *"wrapper NOT launchable as written: the trainer does not declare 12 of 20 of this launcher's argv flags (MEASURED via molmetal/scripts/wf_argv_capability_check.py; see argv_capability.txt)"*.
> * The **12 of 20** is **DERIVED, not typed**: `molmetal/reports/wf355a3_r20_p5_tmqm_plat/argv_capability.txt` ends `FAIL — 12 of 20 flag(s) are not declared by train_dmpnn_multitask_wasserstein.py: …` and carries exactly **12** `MISSING ` rows — the plan's count and the checker's own report agree (12 vs 12).
>
> **⇒ THE RULE THIS ESTABLISHES, AND IT APPLIES TO EVERY "rc=0" / "✅ exit 0" DRY-RUN COLUMN IN THIS FILE AND IN `TODO/completed/364b1_7_wrappers_integration.md` + `TODO/completed/365b2_7_wrappers_final.md`: A DRY RUN'S EXIT CODE IS NOT A VERDICT.** `rc=0` means *the plan was produced*; the verdict lives in the plan's `status` field (and in the gate's own report), which the dry run writes. A wrapper whose plan says `BLOCKED_CAPABILITY` is not launchable, whatever its exit code.
>
> **⇒ WHAT IS NOT WRONG.** Row 3's *Notes* column ("4 cells × 5 seeds = 20 tuples") and its *USER_OK gate* column are unaffected by this correction; and §5's "**No GPU launches performed**" stands — this audit never launched a trainer.
""".rstrip("\n")

BLOCK_B_357A6 = f"""
> **CORRECTION (2026-11-02, the main thread, Wf-A269 wave-11 + wave-12) — THE `USER_OK` VOCABULARY IN THIS SECTION IS WRONG FOR TWO OF THE FIVE WRAPPERS, AND ONE ROW NOTE IS WRONG FOR A THIRD.**
>
> **{MARKER}**
>     {SEC32_357A6}
>
>     {R2_357A6}
>
> **MEASURED 2026-11-02 — THE ACCEPTANCE MATRIX, BY RUNNING each wrapper in `--dry-run` mode on CPU and reading its exit code (exit 64 = the USER_OK gate refused; exit 0 = accepted). No GPU; the dry run emits a plan and trains nothing:**
>
> | wrapper | gate (file:line) | `USER_OK=YES` | `USER_OK=1` |
> |---|---|---|---|
> | `launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh` | `:173` `{{YES,1}}` | exit 0 — ACCEPTED | exit 0 — ACCEPTED |
> | `launch_wf310e_r20_p4_bindingdb_1.07M.sh` | `:202` `{{YES,1}}` | exit 0 — ACCEPTED | exit 0 — ACCEPTED |
> | `launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` | `:182` `{{1}}` | **exit 64 — REFUSED** | exit 0 — ACCEPTED |
> | `launch_wf131_p3_deferred_runs.sh` | `:132` `{{1}}` | **exit 64 — REFUSED** | exit 0 — ACCEPTED |
> | `launch_wf146_deferred_runs.sh` | `:175` `{{YES,1}}` | exit 0 — ACCEPTED | exit 0 — ACCEPTED |
>
> **⇒ THREE CLAIMS IN THIS SECTION ARE MEASURED FALSE.**
> 1. *"the canonical USER_OK literal is `YES` (not `1`)"* — **FALSE for this wrapper set**: two of the five accept `{{1}}` ONLY. The repo-wide canonical value, measured over `scripts/*.sh` (77 files: 64 carry the string `USER_OK`, 48 have a gate), is **`USER_OK=1`** — **44** wrappers accept EXACTLY `{{1}}` and only **4** accept `{{YES,1}}`. `YES` is the **minority** spelling, not the canonical one.
> 2. *"This audit verified all 5 wrappers accept `USER_OK=YES` and refuse non-`YES` values"* — **FALSE**: `launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` and `launch_wf131_p3_deferred_runs.sh` **REFUSE `YES` with exit 64**. The §3 command block above (`USER_OK=YES bash $w --dry-run`) therefore exits 64 for rows 3 and 4, so the verdicts this section records for those two rows cannot have come from that command succeeding.
> 3. Row 2's note *"FAIL with USER_OK=1 (literal mismatch)"* — **FALSE**: `launch_wf310e_r20_p4_bindingdb_1.07M.sh` **ACCEPTS** `USER_OK=1` (exit 0); its gate is the `&&` pair `{{YES,1}}`.
>
> **⇒ TWO SIBLING RECORDS ALREADY HAD IT RIGHT, AND THEY CONTRADICT THIS ONE.** `TODO/completed/364b1_7_wrappers_integration.md` §0 lists rows 3 and 4 as `USER_OK=1`, and `TODO/completed/365b2_7_wrappers_final.md` §1 lists them as `USER_OK=1, exit 64` while listing rows 1, 2 and 5 as `USER_OK=YES`. **This file is the outlier** — it reports `YES` for all five.
>
> **⇒ HONEST-NEGATIVE, AND IT IS USER-GATED — NOT AN AGENT DECISION.** MEASURED 2026-11-02: `AGENTS.md` contains **ZERO** occurrences of `USER_OK` (and zero of `user_ok`); `docs/STANDING_DIRECTIVES.md` mentions the gate only in passing (`:199`, `:612`) without fixing a literal. Yet **44** wrappers enforce one — and they cite **AGENTS.md §5.1**, which is the standing GPU grant that requires **NO** per-launch OK at all. So the vocabulary that 44 wrappers enforce is anchored to a rule that never mentions it, and the literal that the majority enforce is the one the minority record calls non-canonical. **Reconciling the literal — and whether a per-launch OK is needed at all under the standing grant — is a policy decision for the user**; this record states the measurement and does not decide it. Separately USER-gated: `scripts/launch_wf74_v2_drugood_dann.sh` is **self-contradictory** (its gate at `:139` accepts `{{1,YES}}`; a second gate at `:192` accepts `{{YES}}` only, so `USER_OK=1` alone is refused), and `scripts/launch_wf306a_drugood_vina_gpu.sh` requires **BOTH** `USER_OK=1` (`:103`) **AND** `WF306A_USER_OK=YES` (`:203`).
""".rstrip("\n")

# ===========================================================================
# 364b1  --  one block
# ===========================================================================

F_364B1 = TODOS / "364b1_7_wrappers_integration.md"

R3_364B1 = (
    "| `launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` | PASS | PASS (strict) |"
    " PASS (rc=0) | PASS (sources helper) |"
)
VERD_364B1 = """**Per-wrapper integration verdict: 7/7 PASS** (with 2 honest-negative gaps
explicitly documented in §3)."""

BLOCK_364B1 = f"""
> **CORRECTION (2026-11-02, the main thread, Wf-A269 wave-12) — "PASS (rc=0)" IN THE `dry-run OK` COLUMN IS A PLAN-EMISSION FACT, NOT A LAUNCHABILITY VERDICT; ONE OF THE SEVEN WRAPPERS' PLAN SAYS `BLOCKED_CAPABILITY`.**
>
> **{MARKER}**
>     {R3_364B1}
>
>     {VERD_364B1}
>
> **MEASURED 2026-11-02 (CPU-only, `--dry-run` only, zero GPU).** The `rc=0` is real — the dry run exits 0 and emits a plan. But `rc=0` answers *"did the wrapper produce a plan?"*, not *"can this wrapper launch?"*.
>
> * `scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` writes `molmetal/reports/wf355a3_r20_p5_tmqm_plat/plan.json`. MEASURED after the Wf-A269 wave-12 repair: `status = BLOCKED_CAPABILITY`, `capability = BLOCKED`; the plan's `honest_framing.MEASURED_today` states that the trainer `train_dmpnn_multitask_wasserstein.py` does not declare **12 of 20** of the launcher's argv flags — cross-checked against `argv_capability.txt`'s own line `FAIL — 12 of 20 flag(s) are not declared` and its **12** `MISSING ` rows.
> * PRE-REPAIR, the same file's `status` field was **hardcoded**: old line 735 read `printf '  "status": "%s"\n' "$( [ "${{DRY_RUN}}" = "1" ] && echo PLAN_ONLY || echo READY_TO_RUN )"` — the dry-run plan said `PLAN_ONLY` and the **real** launch plan said `READY_TO_RUN`, both **independent of the capability gate**.
>
> **⇒ THE TEST THIS FILE RAN DID NOT AND COULD NOT TEST LAUNCHABILITY.** §1.1 records `test_per_wrapper_dry_run_with_user_ok_works` as *"With USER_OK + `--dry-run` → exit 0 (CPU-only)"* — that assertion is correct **for what it claims**, and the record's evidence is not contradicted by this correction. What it does NOT establish is launchability, because the verdict is in the plan's `status` field, which the test never read. **⇒ THE "7/7 PASS" VERDICT IS AN INTEGRATION VERDICT (bash syntax, gate enforcement, plan emission), NOT A LAUNCHABILITY VERDICT** — the two are different questions and the verdict line above does not distinguish them.
>
> **⇒ THE USER_OK COLUMN OF THIS FILE IS CORRECT AND SURVIVES.** §0 already lists rows 3 and 4 as `USER_OK=1` — MEASURED: `launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh:182` and `launch_wf131_p3_deferred_runs.sh:132` both gate on `${{USER_OK:-0}} != "1"`, so `USER_OK=YES` is REFUSED with exit 64 and `USER_OK=1` is accepted. This record and `TODO/completed/365b2_7_wrappers_final.md` §1 both contradict `TODO/completed/357a6_gpu_readiness_audit.md` §3.2, which claims all five accept `USER_OK=YES`; see that file's dated correction for the full matrix.
""".rstrip("\n")

# ===========================================================================
# 365b2  --  two blocks (D: the §1 table, E: the §3 test row)
# ===========================================================================

F_365B2 = TODOS / "365b2_7_wrappers_final.md"

R3_365B2 = (
    "| 3 | Wf-355-A3 | `scripts/launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh` | ✅ |"
    " ✅ gate enforced (USER_OK=1, exit 64) | ✅ exit 0 | ✅ emitted at"
    " `wf355a3_r20_p5_tmqm_plat/` |"
)
SUM_365B2 = (
    "**Summary**: 6/7 wrappers fully conformant (USER_OK-gated + dry-runnable + DRY_RUN.log)."
    " 1/7 (G5) HONEST-NEGATIVE on USER_OK gate (legacy predates Wf-300C pattern)."
)
T3_365B2 = (
    "| Dry-run with USER_OK | `test_dry_run_all_7[7 wrappers]` | ✅ 7/7 dry-run exit 0 |"
)

BLOCK_D_365B2 = f"""
> **CORRECTION (2026-11-02, the main thread, Wf-A269 wave-12) — "✅ exit 0" IN THE `--dry-run` COLUMN IS A PLAN-EMISSION FACT; THE ROW-3 PLAN'S OWN VERDICT FIELD SAYS `BLOCKED_CAPABILITY`.**
>
> **{MARKER}**
>     {R3_365B2}
>
>     {SUM_365B2}
>
> **MEASURED 2026-11-02 (CPU-only, `--dry-run` only, zero GPU).** The exit code is real — the dry run exits 0 and emits a plan into `wf355a3_r20_p5_tmqm_plat/`. But the plan it emits now carries `status = BLOCKED_CAPABILITY` and `capability = BLOCKED`, and its `honest_framing.MEASURED_today` names the reason: the trainer `train_dmpnn_multitask_wasserstein.py` does not declare **12 of 20** of the launcher's argv flags (cross-checked against `argv_capability.txt`'s `FAIL — 12 of 20 flag(s) are not declared` line and its **12** `MISSING ` rows). PRE-REPAIR the `status` field was hardcoded (`PLAN_ONLY` on the dry-run path, `READY_TO_RUN` on the launch path), so no dry-run plan could ever have carried the gate's real verdict.
>
> **⇒ THIS FILE'S USER_OK COLUMN IS THE CORRECT ONE, AND IT IS CONFIRMED BY BOTH THE CODE AND RUNTIME PROBES.** Rows 3 and 4 say `USER_OK=1, exit 64` — MEASURED: `launch_wf310e_r20_p5_tmqm_plat_joint_retrain.sh:182` and `launch_wf131_p3_deferred_runs.sh:132` gate on `${{USER_OK:-0}} != "1"`; probed, `USER_OK=YES` → **exit 64 (REFUSED)**, `USER_OK=1` → exit 0. Rows 1, 2 and 5 say `USER_OK=YES` — MEASURED: `launch_wf310e_r20_p3_drugood_1.58M_chemprop_v2.sh:173`, `launch_wf310e_r20_p4_bindingdb_1.07M.sh:202` and `launch_wf146_deferred_runs.sh:175` use the `&&` pair form, so they accept **both** `YES` and `1`.
>
> **⇒ ⇒ THE TWO "✅" COLUMNS MEASURE DIFFERENT THINGS, AND NEITHER IS A LAUNCHABILITY MEASUREMENT.** `USER_OK gate` = the wrapper refuses to proceed without an OK token. `--dry-run` = the wrapper produced a plan. **Neither column reads the plan's `status`, which is the launchability verdict.** ⇒ The "6/7 wrappers fully conformant" summary is **TRUE as stated** (conformant = gated + dry-runnable + log emitted) and must not be read as "6/7 launchable"; the difference is now on the record rather than implied.
""".rstrip("\n")

BLOCK_E_365B2 = f"""
> **CORRECTION (2026-11-02, the main thread, Wf-A269 wave-12) — THE `7/7 dry-run exit 0` TEST ROW IS EXACT AS FAR AS IT GOES, AND IT IS NOT A LAUNCHABILITY MEASUREMENT.**
>
> **{MARKER}**
>     {T3_365B2}
>
> **MEASURED 2026-11-02 by probe, CPU-only.** The five Wf-355 / Wf-131 / Wf-146 wrappers audited in this file (p3 drugood, p4 bindingdb, p5 tmqm_plat, wf131, wf146) all exit **0** under `USER_OK=… --dry-run`; the two G5/G6 wrappers were not re-probed in this correction. **⇒ The row is EXACT.** What it does not measure is the launchability verdict, which lives in the emitted plan's `status` field — for row 3 that field reads `BLOCKED_CAPABILITY` (see the §1 correction above). The test's own assertion — *"With USER_OK + `--dry-run` → exit 0"* — is correct for what it claims; this entry records only the boundary of what "exit 0" licenses a reader to conclude.
""".rstrip("\n")


def _insert_after(anchor: str, payload: str, text: str) -> str:
    idx = text.index(anchor)
    end = idx + len(anchor)
    return text[:end] + payload + text[end:]


def _check(original: str, out: str, label: str) -> None:
    lost = [ln for ln in original.splitlines() if ln.strip() and ln.strip() not in out]
    assert not lost, f"[{label}] {len(lost)} original line(s) lost: {lost[:3]}"
    assert out.count(MARKER) >= 1, f"[{label}] marker missing"
    assert out.endswith("\n"), f"[{label}] must end with a newline"


def main() -> int:
    # ---------------- 357a6 ----------------
    orig = F_357A6.read_text(encoding="utf-8")
    for a in (R3_357A6, SUM_357A6, SEC32_357A6, R2_357A6):
        assert orig.count(a) == 1, f"357a6 anchor missing/ambiguous: {a[:70]!r}"
    out = _insert_after(R3_357A6, "\n\n" + BLOCK_A_357A6, orig)
    out = _insert_after(SEC32_357A6, "\n\n" + BLOCK_B_357A6, out)
    assert out.count(R3_357A6) == 2 and out.count(SUM_357A6) == 2
    assert out.count(SEC32_357A6) == 2 and out.count(R2_357A6) == 2
    _check(orig, out, "357a6")
    atomic_write_file(F_357A6, out, encoding="utf-8")
    print(f"357a6  {len(orig)} -> {len(out)} chars (+{len(out) - len(orig)}); 2 correction blocks")

    # ---------------- 364b1 ----------------
    orig = F_364B1.read_text(encoding="utf-8")
    for a in (R3_364B1, VERD_364B1):
        assert orig.count(a) == 1, f"364b1 anchor missing/ambiguous: {a[:70]!r}"
    out = _insert_after(VERD_364B1, "\n\n" + BLOCK_364B1, orig)
    assert out.count(R3_364B1) == 2 and out.count(VERD_364B1) == 2
    _check(orig, out, "364b1")
    atomic_write_file(F_364B1, out, encoding="utf-8")
    print(f"364b1  {len(orig)} -> {len(out)} chars (+{len(out) - len(orig)}); 1 correction block")

    # ---------------- 365b2 ----------------
    orig = F_365B2.read_text(encoding="utf-8")
    for a in (R3_365B2, SUM_365B2, T3_365B2):
        assert orig.count(a) == 1, f"365b2 anchor missing/ambiguous: {a[:70]!r}"
    out = _insert_after(SUM_365B2, "\n\n" + BLOCK_D_365B2, orig)
    out = _insert_after(T3_365B2, "\n\n" + BLOCK_E_365B2, out)
    assert out.count(R3_365B2) == 2 and out.count(SUM_365B2) == 2 and out.count(T3_365B2) == 2
    _check(orig, out, "365b2")
    atomic_write_file(F_365B2, out, encoding="utf-8")
    print(f"365b2  {len(orig)} -> {len(out)} chars (+{len(out) - len(orig)}); 2 correction blocks")

    print("  no original line's text lost in any of the three files: True")
    print(f"  marker present in all three: True")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
