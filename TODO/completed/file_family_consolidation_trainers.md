# File-family consolidation — trainer/chemprop dedup (2026-10-08)

**Method**: usage-evidence inventory (grep over scripts/*.sh launchers,
molmetal/**.py imports, tests/) + independent re-verification by the main
session (grep for `subprocess`/`os.system`/`python -m` exec refs, `import
molmetal` + pytest collection). Per AGENTS.md §5.5 filesystem-truth.

**Note on commit bundling**: these 6 deletions were staged concurrently by the
prune agent and rode into commit `16c8a58f7f` (the encoder-deliverables commit)
without appearing in its message. This doc is the explicit record; the deletion
list below is the authoritative one.

## §1 Canonical member per family (kept)

| Family | Canonical (KEPT) | Evidence |
|---|---|---|
| train_hybrid* | `train_hybrid.py` | base module; 5 live imports from siblings + test_metal_hybrid_v3 |
| train_metal_lomo* | `train_metal_lomo_wasserstein.py` | 7 launcher refs (incl. launch_wf131), 14 test imports, newest + largest |
| train_chemprop_v2* | `train_chemprop_v2_v3.py` | 6 launchers (incl. start.sh), 14 imports, 31 tests |
| wf1xx/wf3xx chemprop | `wf185_chemprop_v2_drugood_full.py` | launcher ref (launch_wf310e_r20_p3_drugood_1.58M) + tests |

Also KEPT (live launcher AND/OR test refs — not provably dead):
`train_metal_lomo.py`, `train_metal_lomo_v2.py`, `baseline_lomo_cv.py`,
`train_multimetal_lomo.py`, `train_chemprop_v2_lomo.py`,
`train_chemprop_v2_per_metal.py` (canonical of the per-metal pair — has a
dedicated `G1: Script exists + importable` test), `train_metal_hybrid_v4.py` +
`train_metal_hybrid_v4_mb1.py` (distinct v4 architecture), `wf146_chemprop_lomo.py`,
`wf174`/`wf182` (import chain), `wf331r5`/`wf335m2`/`wf382a2`/`wf392a2`
(import+test), `train_cisplatin_resistance_v2.py` + its verifier.

## §2 Deleted (6 files, 2,601 lines) — all 0 live exec/import refs

| File | Rationale | Independent re-check |
|---|---|---|
| `train_metal_hybrid.py` | superseded by `train_hybrid.py` base + `train_metal_hybrid_v4.py` | only ref = scripts/README.md doc line; v4* hits are a different file |
| `train_hybrid_ablation.py` | 0 refs; ablation superseded | grep: 0 hits anywhere |
| `train_hybrid_ablation_v3.py` | 0 refs | grep: 0 hits anywhere |
| `train_chemprop_v2_v2_fastfeaturize.py` | 0 live refs | refs in `wf320d_featurization_fix.py` are docstring (L44) + comment (L560) only — no subprocess/exec |
| `wf170_chemprop_drugood.py` | self-ref only | ref in wf174 is a comment mention (L548), not an import |
| `wf335m1_chemprop_pll_smoke.py` | 0 refs everywhere | grep: 0 hits anywhere |

## §3 Verification (post-deletion, by main session)

- `.venv/bin/python -c "import molmetal"` → rc 0, IMPORT OK.
- `.venv/bin/python -m pytest molmetal/tests/ --co -q` → **12808 tests collected,
  0 errors**.
- `git stash list` → empty (no dangling stash; the concurrent-agent stash the
  prune agent recovered is reconciled).

## §4 Reversibility

Each deletion is single-path reversible:
`git checkout 16c8a58f7f~1 -- <path>`.

## §5 Honest framing

Deletion is destructive by nature; the safety bar applied was "0 live
import/exec references outside reports/*.md" **and** independent re-verification
for real exec refs (not just exact-word grep). The test-count delta reported by
the prune agent came from concurrent new tests, not from these removals. No
family member with a live launcher or test import was deleted.
