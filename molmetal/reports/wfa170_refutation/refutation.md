# Wf-A170 — Independent Adversarial Re-Derivation of the Wf-A169 Finding Set

**Verdict: PARTLY REFUTED.**
Tag: MEASURED. 0 GPU used. Single venv (`/home/hugo/codes/mol-metal/.venv/bin/python`).
Full per-check evidence: `molmetal/reports/wfa170_refutation/refutation.json`.

Every mechanical measurement in Wf-A169 reproduces; the **cause** the main thread gives is
wrong, three of its stated numbers do not reproduce, and its "they have NEVER run" framing
is false.

## Claim-by-claim

| Claim | Verdict | Measured |
|---|---|---|
| **1** — 24 gates DEFINED but registered in NO pipeline, "so they have NEVER run" | **PARTLY REFUTED** | "defined" 24/24 CONFIRMED; "registered in no pipeline" 0/24 CONFIRMED (AST census validated EXACT vs live `build_gates`, HEAD 126 / worktree 129 live `all`); **"NEVER run" REFUTED** — they ran 2026-10-02 (`TODO/completed/432a1_pre_training_gate_v2.md` §3: total 149, pass 139, fail 3, skip 3; the 32 new gates at 26 PASS / 4 WARN / 2 FAIL), then were **deliberately removed** by the Wf-437 A2 audit. |
| **2** — three-leg proof that registration was LOST, not never written | **PARTLY REFUTED** | Leg (a) CONFIRMED as a measurement (all 24 names → exactly 1 commit, `bc29aa90`; no removal) but the inference drawn in `registration_table.md` ("no registration call was ever written") is **REFUTED**. Leg (b) CONFIRMED (348,994 / 359,637 / 10,643 / `.bak` 307,077 / witness 18,070; 359,637 appears in **none** of the 41 committed sizes) but the gap covers **29** removed tuples, not 24. Leg (c) CONFIRMED in substance (**zero** registrations removed) but the COUNTS are wrong: **133 → 137** (union over 5 pipelines), not 115 → 119. |
| **3** — L5 triple defect D1 / D2 / D3 | **NOT REFUTED** | D1 CONFIRMED by independent probe: HEAD emitter no-ops on a missing `hook_dir`, 9/9 gates return PASS with the evidence string "hook emitted", **0 files written**, `molmetal/reports/protocol_validation/` never populated in git. D2 CONFIRMED (HEAD replaces the file; worktree merges → 3 files × 3 hook_id keys). D3 CONFIRMED **still open**: PASS = file existence + emission, `verdict_field` hardcoded; measured 25 / 9 / 3 records vs spec 125+ / 45+ / 12. Repair is real, **zero** live blast radius (`build_gates("all")` = 129 = EXPECTED), and **uncommitted** (sha256 `ae878bcd…`, 734,797 B, mtime 11:54:50). |
| **4** — live counts 129 / 131 / 25 / 18 / 8 == `EXPECTED_GATE_COUNTS` | **NOT REFUTED** | CONFIRMED exactly by live import, `CUDA_VISIBLE_DEVICES=""`. |
| **5** — RED surface 36/72 before patch, 29/79 after | **PARTLY REFUTED** | "29 failed / 79 passed" describes no tree I could reproduce: **35 failed / 53 passed** at 12:23 (pre-resync) and **32 failed / 56 passed** at 12:31, across the 5 resync-patch files (the main thread resynced only `test_wf350c1_gpu_env_probe.py`, at 12:24:06, mid-measurement). Measured split of the 32: **24** parametrized + **1** "all 32 present" + **7** stale count pins (4 × `test_wf433a1`, `test_wfa127`, `test_wfa128`, `test_wf323f_m4`). The claimed "3 GPU-required-session-artefact" bucket is **empty** — those 3 failures were stale pins (`94`), not a missing GPU session. The 6th file was **not identified**. |

## Strongest surviving counter-argument

The 24 are not "lost registrations" a wave forgot. `molmetal/scripts/pre_flight_check.py:14441-14450`
(HEAD `:12480`) is the repo's **own committed comment**: the Wf-437 A2 audit removed **29**
"SCAFFOLD-only doc-trackers" (15 `p3_protocol_*` + 5 `p3_mock_to_real_*` + 9
`krasnov_*`/`rusanov_*`/`drugood_*`) and kept only the 3 `p3_honest_neg_closure_*`.
`EXPECTED_GATE_COUNTS["all"]` at `bc29aa90` is **98** with the in-file comment
"Pre-Wf-437 baseline: 127 unique named gate tuples" (127 − 29 = 98, and live import at that
commit measures exactly 98 unique names with exactly the 3 kept gates present).
`TODO/completed/321d_preflight_gate_audit.md:148-152` records that 5 of those 29 were
later **restored** by Wf-R42 once proven REAL. The remaining 24 were never re-examined.

So the correct framing is **"deferred by an audit whose deferral was never revisited"** — and
D3 is direct evidence the deferral's premise no longer holds.

## Material corrections the main thread must make

1. **Reverse** `registration_table.md`'s "no registration call was ever written (not 'written, then removed')". It was written (2026-10-02, 359,637 B, 127 unique) and **deliberately removed** by Wf-437 A2. The `git log -S` single-commit result shows only that the registered state was **never committed**.
2. **Replace** Claim 1's "so they have NEVER run" — they ran once, on 2026-10-02.
3. **Fix** leg 2(c): "HEAD 115 / worktree 119" → **133 / 137** (union) or live `all` 126 → 129. The 4 added names are correct; `p3_cfm_fragmentation_preflight` is on the `cfm` pipeline only.
4. **Fix** leg 2(b): the 10,643 B gap covers **29** tuples (367 B/tuple), not 24.
5. **Fix** Claim 5's numbers and split (see table). State that the resync is only **1 of 5** files applied.
6. **Label** the D1/D2 repair **uncommitted**.
7. **Name or drop** the 6th test file in the 6-file RED-surface claim.

## Honest negative

This refutation does **not** clear the 24 gates. Registering them makes the `all` pipeline
FAIL on `p3_protocol_alignment_baseline_paper_refs_bib` and WARN on 4 optional artefacts
(19 PASS / 4 WARN / 1 FAIL over the 24, per the Wf-A169 branch's `apply_probe.json`).
The open question is narrower than "were they lost": **"was the Wf-437 A2 classification of
these 29 as SCAFFOLD-only doc-trackers ever re-examined?"** For the 9 L5 hooks the answer is
no — they PASS while their sources hold 25 / 9 / 3 records against 125+ / 45+ / 12 thresholds.

`metrics/` was snapshotted before and after both pytest runs: **UNCHANGED** (38 lines, sha256
`025a37ef…`). No published metric was overwritten.
