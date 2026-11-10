# Wf-A169 — the 24 dead Wf-432-A1 gates: wiring patch

**Branch 1 of the Wf-A169 parallel ultracode.** Tag: **MEASURED** (2026-11-01).
CPU-only; no GPU was used or requested (`CUDA_VISIBLE_DEVICES=""` on every
invocation; the module imports no torch, no trainer was run, no molecule was
decoded).

## Deliverables

| File | What it is |
|---|---|
| `patch_wire_24_gates.diff` | Unified diff that registers the 24 dead gates in `molmetal/scripts/pre_flight_check.py`. **+295 / −1 lines, 2 hunks.** The one removed line is a stale path (§6 of the registration table). |
| `registration_table.md` | One row per gate: layer, target pipeline, priority string, every argument with its exact value + the spec line / sibling line that justifies it. Plus the re-derivation of the finding and the honest negatives the registration surfaces. |
| `apply_probe.json` | Machine-readable result of applying the patch to a scratch copy and importing the patched module: `git apply --check` rc, byte-identity, per-pipeline counts before/after, the DELTA, and each of the 24 gates' on-disk verdict. |
| `regen_patch.py` + `regen_block.py` | Re-base the patch onto whatever `pre_flight_check.py` looks like now (the main thread was editing it concurrently), then re-run the probe. |

## Headline numbers (all MEASURED, all from `apply_probe.json`)

* Gates registered by the patch: **24 / 24** — no gate left out.
* `git apply --check` against the scratch copy: **rc=0**. `git apply`: **rc=0**.
  Applied file **byte-identical** to the expected patched file.
* `git apply --check` against the **live working tree** also returns **rc=0**
  (check-only; the working tree was not modified).
* `len(build_gates("all"))`: **129 → 153 (Δ +24)**.
* `len(build_gates("launch"))`: **131 → 155 (Δ +24)**.
* `len(build_gates("frontier" / "chemprop_v2_v3" / "cfm"))`: **25 / 18 / 8 —
  unchanged (Δ 0)**.
* **0 UNDETERMINED gates.** Every argument was recoverable from the spec +
  the gate signatures + the already-defined (and previously unused) `_p2_*`
  locals + the witness test's `arg_sets`. Nothing was invented.
* The 24 gates' own verdicts once registered: **19 PASS / 4 WARN / 1 FAIL** —
  the FAIL and WARNs are pre-existing honest negatives on disk, not wiring
  defects (details in `registration_table.md` §7).

## The finding, re-derived (not taken on faith)

`git log --all -S'<name>' -- molmetal/scripts/pre_flight_check.py` returns
**exactly one commit for each of the 24 names** — `bc29aa90` (2026-10-02),
the commit that added the function **definitions**. The occurrence count never
rose above the definition count, so the registration call was never written
(it was not "written, then removed"). The Wf-437 comment in `build_gates` that
claims "29 SCAFFOLD-only gate tuples REMOVED ... p3_protocol_* (15 gates)" does
**not** refer to these 24: `git show 140e1f82 -- molmetal/scripts/pre_flight_check.py`
removes no `p3_protocol_alignment_*` / `*_recommended_*` / `krasnov_*` /
`rusanov_*` / `drugood_*` line, and the only `p3_protocol_*` gates registered
today are the 4 `p3_protocol_alignment_yaml_*` (Wf-350-C1 family).

Corroborating signal: 26 `_p2_*` path locals sit in the `all` block with
**usage count 1** (definition only) — they were written by Wf-432-A1 for
exactly these 24 gates and were never wired. The 8 `_p2_*` locals that *are*
used (count ≥ 2) belong to the already-registered L3/L4 gates.

**Correction to the task brief:** the brief's byte-gap proof (348,994 committed
vs 359,637 claimed in the Wf-432-A1 close-out) no longer holds on this tree —
the file is now 734,797 bytes. The git-history proof above is what survives, and
it is stronger.

## Two things the main thread must do alongside this patch

1. **Bump the count pins.** `EXPECTED_GATE_COUNTS["all"]` 129 → 153 and
   `["launch"]` 131 → 155. This patch deliberately does **not** touch
   `EXPECTED_GATE_COUNTS` (another branch owns it). Until it is bumped, the
   explicit `expected_total` drift-check gate inside `build_gates("all")` will
   FAIL with `actual=153 vs expected=129 (+DRIFT (silent gate bloat))`.
2. **Re-verify before applying.** The patch was generated against a snapshot
   (sha256 in the diff header and in `apply_probe.json`) of a file that was
   being edited concurrently. If `git apply --check` fails, re-run
   `regen_patch.py`.

## Side effects to be aware of

* The 24 gates were designed to surface honest negatives. Registered, they
  make the `all` pipeline **FAIL** on `p3_protocol_alignment_baseline_paper_refs_bib`
  (`paper/refs.bib` is missing `krasnov2026metalcytotoxdb`,
  `rusanov2025platinai`, `ji2022drugood`) and **WARN** on 4 missing optional
  artifacts. That is the gates working, not the wiring failing.
* The witness test `molmetal/tests/test_wf433a1_gate_v2_wire.py` pins
  `EXPECTED_GATE_COUNTS["all"] == 127` and a 127-unique-name count; both pins
  are already stale (today 129) and belong to the count-pin branch.
* `paper/*` was **read only** (AGENTS.md §5.3). Nothing under `paper/`,
  `docs/`, `TODO/`, `configs/`, `metrics/`, `registry/`, `AGENTS.md` or
  `README.md` was written by this branch.
