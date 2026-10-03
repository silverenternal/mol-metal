# Wf-446 L4 — wire `check_data_leakage.py` into pre-flight (`p2_data_leakage_diag`) — SHIPPED

**Date:** 2026-10-03 · **Lane:** `molmetal/scripts/pre_flight_check.py`, `molmetal/tests/`, `TODO/completed/`
**Status:** SHIPPED (disk-verified below per AGENTS.md §5.5; written via `atomic_write_file` per §5.6)

## §1 What changed

1. **NEW gate function** `gate_p2_data_leakage_diag` —
   `molmetal/scripts/pre_flight_check.py:8015` (def) .. 8194. Runs
   `molmetal/scripts/check_data_leakage.py` as a subprocess with EXPLICIT
   inputs (`molmetal/data/platinai_5k.csv` +
   `.data/molmetal/metalcytotox_5metal_multitask_v4.csv`) and
   `--out-path results/gate_scratch/wf44_leak/leakage_check.json`
   (gitignored, verified `git check-ignore` OK; the directory did not
   exist before — created lazily by the script). Requires
   `strict_pass` on: platinai scaffold split; metalcytotox scaffold split
   + shipped split. Random-split overlap is recorded as the expected
   honest-negative (it leaks by design — gating it would be red-by-design).
   Host contract: SKIP (not FAIL) when the gitignored `.data/**` v4 CSV is
   absent; FAIL on missing script / missing tracked platinai CSV /
   subprocess crash / unreadable report / missing dataset entry /
   missing `shipped` column.
2. **Registration** — `p2_data_leakage_diag`, `P2` / `data`, all-only, in
   the §5 P2 block at `pre_flight_check.py:8997`.
3. **Count sync** — `EXPECTED_GATE_COUNTS["all"]` 103 → 104
   (`pre_flight_check.py:3951`), MEASURED via
   `len(build_gates("all")) == 104`; the docs-drift guard
   `TODO/completed/321d_preflight_gate_audit.md` §2.1 / §5.2 / §6 / §6.4
   updated in the same edit.
4. **NEW test** `molmetal/tests/test_wf446_l4_data_leakage_gate.py`
   (8/8 PASS, 22.3 s including the 21.4 s live integration case).

## §2 FAIL proof (constructed bad input — real output)

Direct script run with the v1 CSV (the known-leaky artifact):

```
$ .venv/bin/python molmetal/scripts/check_data_leakage.py --platinai-csv molmetal/data/platinai_5k.csv \
    --metalcytotox-csv .data/molmetal/metalcytotox_5metal_multitask.csv --out-path /tmp/wf44_leak_v1.json
[check_data_leakage]   metalcytotox_5metal: random_split: strict=FAIL (1935 shared canon SMILES; 1213 shared Bemis-Murcko scaffolds)
[check_data_leakage]   metalcytotox_5metal: scaffold_split: strict=PASS (0 shared scaffolds; 0.0% of test scaffolds in train)
[check_data_leakage]   metalcytotox_5metal: shipped_split: strict=FAIL (48 shared scaffolds)
[check_data_leakage] n_datasets=2 strict_random_pass=1 strict_scaffold_pass=2
```

Gate verdict (real output):

```
STATUS: FAIL
EVIDENCE: data leakage diagnostic FAIL: metalcytotox_5metal: shipped split NOT strict (strict_pass['shipped']=False)
| report=/tmp/wf446_l4_leak_fail.json | tail: [check_data_leakage]   shipped_split: strict=FAIL (48 shared scaffolds)
| [check_data_leakage] wrote /tmp/wf446_l4_leak_fail.json | ... n_datasets=2 strict_random_pass=1 strict_scaffold_pass=2
```

Synthetic FAIL proofs (unit tests, no `.data/**` dependency):
injected shipped-split leak → FAIL; `split` column removed → FAIL;
missing script → FAIL; missing platinai CSV → FAIL; missing v4 CSV → SKIP.

## §3 PASS proof (healthy tree — real output)

```
$ .venv/bin/python -c "... r = dict(build_gates('all'))['p2_data_leakage_diag'](); print(r.status, r.evidence)"
STATUS: PASS
NAME/PRIORITY/PIPELINE: p2_data_leakage_diag P2 data
WALL: 21.4
EVIDENCE: check_data_leakage.py exit=0 in 21.4s; platinai_5k scaffold strict=PASS; metalcytotox_5metal
scaffold strict=PASS + shipped strict=PASS (0 shared scaffolds — the split trainers consume);
HONEST-NEGATIVE expected random-split overlap: platinai_5k random: 0 shared SMILES / 93 shared scaffolds;
metalcytotox_5metal random: 1931 shared SMILES / 1198 shared scaffolds (random splits leak by design —
the strict shipped/scaffold splits are the contract); report=.../results/gate_scratch/wf44_leak/leakage_check.json
```

Full pre-flight `--pipeline all` was NOT run (per task hard rule: it spawns CPU
smoke training). The registration was proven instead by calling the runner
extracted from `build_gates("all")` (above), plus:

```
$ len(build_gates(...)) → all expected=104 actual=104 OK; frontier 25/25; chemprop_v2_v3 18/18; cfm 7/7
$ pytest molmetal/tests/test_wf327f11_gate_count_consistency.py -q → 3 passed (0.28s, docs-drift guard green)
$ pytest molmetal/tests/test_wf446_l4_data_leakage_gate.py -q → 8 passed (22.29s)
$ git status --porcelain results/gate_scratch → (empty: scratch output is gitignored, no git dirt)
```

## §4 `ls -la` evidence (AGENTS.md §5.5 rule #2)

```
-rw------- 1 hugo hugo 473359 Oct  3 15:19 molmetal/scripts/pre_flight_check.py
-rw-r--r-- 1 hugo hugo   6318 Oct  3 15:20 molmetal/tests/test_wf446_l4_data_leakage_gate.py
-rw-r--r-- 1 hugo hugo   9051 Oct  3 15:19 results/gate_scratch/wf44_leak/leakage_check.json
-rw-r--r-- 1 hugo hugo  20266 Oct  3 15:21 TODO/completed/321d_preflight_gate_audit.md
```

## §5 Deliberately not fixed (reasons)

* **Repointing the 6+ trainer defaults from v1 to v4** (`train_ru_multitask.py`
  etc. still default `DEFAULT_DATA` at v1) — out of this lane; v1 bytes are
  hash-frozen by `p0_csv_hash_verify` (blocking ON), so the leak cannot drift
  silently; visibility item only. Reported, not fixed.
* **Gating the random-split axis** — red-by-design (a random split on a
  duplicate-molecule cohort leaks by construction; MEASURED 1931/1198 on v4).
  Gating it would violate the block-only-when-green policy.
* **`check_data_leakage.py` output path default** (`molmetal/reports/…`, a
  git-tracked dir) — the gate always passes `--out-path` explicitly, so the
  default is never exercised by pre-flight; changing the script's default is
  out of lane.

## §6 Honest framing

* The gate verifies the SPLIT-STRICTNESS property of the v4 artifact + the
  platinai scaffold splitter — it does not verify trainer behaviour, and it
  does not cover the excluded v1 artifact's leak (see §5).
* 21.4 s wall per run (CPU-only, repo venv), inside the 300 s timeout headroom.
* Case for the blocking question: GREEN under it (tree stays green, git clean)
  and the remedy is one command in the failure message — the gate may block on
  its axis; it adds no blocking behaviour to any hook (pre-flight is not a
  commit hook).
