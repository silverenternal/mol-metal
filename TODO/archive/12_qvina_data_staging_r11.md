# Round-11 — QVina install + CrossDocked2020 data staging

**Status:** QuickVina 2 identified and runnable; test manifest paths corrected; N=50 parity pending
**Priority:** high
**Effort:** 5 days (1 ultracode round)
**Owner:** (unset)
**Depends on:** Round-10 (5 click reactions wired)
**Blockers:** none for local QuickVina 2 discovery or test-pair staging.
The N=50 empirical parity run is remaining experiment work, not an environment blocker.
`qvina` and `quickvina2` resolve the same verified binary; see
`molmetal/reports/quickvina2_binary_identity.md`.
**Created:** 2026-09-13

## Goal

Get the two **non-algorithmic** blockers for top-journal strict-protocol
comparison resolved:
1. **QVina engine** parity (exhaustiveness=8 to match TargetDiff paper)
2. **CrossDocked2020 100-pocket** (Luo 2021 split) test set staged locally

Both are env / data work — **no experiments, no sweep, no production code
rewrites**. If env blocks, fall back to Vina 1.2.7 + exhaustiveness=16 with
empirical parity documentation (cite Alhossary 2015 + Hassan 2017).

## Scope (4 axes)

### A — QVina binary install
Try in order:
1. `uv tool install qvina` (or `pip install qvina` if uv fails)
2. `conda install -c conda-forge qvina` (if conda available)
3. Download QVina 2.1 binary release from official site
4. Build QVina from source (https://github.com/QVina/qvina)
5. **Fallback**: document Vina 1.2.7 + exhaustiveness=16 parity rigorously

Once installed: extend `molmetal/molmetal_lam/sbdd_env/vina_adapter.py` engine
field to support `qvina` as the actual scoring path (not just the resolution
fallback); add `--qvina-binary PATH` flag.

### B — QuickVina 2.1 (Pocket2Mol engine) install
Same fallback ladder as A. Target engine parity for Pocket2Mol comparison.

### C — CrossDocked2020 data staging
Try in order:
1. `curl`/`wget` from https://bits.csb.pitt.edu/files/ crossdocked100_test.tar.gz
   (or any cached mirror: deepchem, huggingface, zenodo)
2. Look for pre-staged 100-pocket subset in `molmetal/data/`,
   `molmetal/references/`, `~/.cache/`
3. **Fallback**: use the 5-10 pockets already available (1h36, 830c, MMP2,
   MMP13, CA2) + document the protocol-divergence caveat in the paper

If staging succeeds: write `molmetal/scripts/stage_crossdocked100.py` that
downloads + parses + indexes 100 test pockets to `molmetal/data/crossdocked100/`.

### D — Empirical Vina engine parity study
On N=5 pockets × N=10 molecules × both engines (Vina 1.2.7 exh=16 + QVina
exh=8 + QuickVina2 exh=16 if available), report:
- Per-molecule kcal/mol delta
- Pearson correlation
- Mean absolute difference (MAD)

Cite Trott 2010 (Vina), Alhossary 2015 (QVina), Hassan 2017 (QuickVina).
This data feeds into the paper's "protocol alignment" appendix.

## Success criterion

- QVina binary callable from `molmetal_lam/sbdd_env/vina_adapter.py` via
  `--engine qvina` (engine parity ≥ 0.6 kcal/mol MAD vs published QVina runs)
- CrossDocked2020 100-pocket subset staged (or fallback proxy documented)
- Engine parity report `molmetal/reports/round11_engine_parity.md` with
  empirical N=50 kcal/mol table
- 158 + ~10 new tests pass

## Out of scope (deferred to round-12+)

- SOTA checkpoint re-runs (Pocket2Mol / TargetDiff / DiffSBDD / DecompDiff /
  FLOWR); round-11 only attempts fetch + smoke-test, no real eval
- N=5+ pilot at top-journal protocol (round-12)
- 100-pocket full sweep (round-13)

## Recommended orchestration

Single ultracode workflow `w_round11_qvina_data`, 3 phases:

| Phase | Agents | Goal |
|---|---|---|
| 1 — Install + stage (parallel attempts) | 3 parallel | A (QVina) + B (QuickVina2) + C (CrossDocked) each tries install ladder, reports status |
| 2 — Engine parity empirical study | 2 parallel | D (parity harness) + fetch-and-smoke SOTA checkpoints (1 agent) |
| 3 — Verify + report | 1 | Tests pass; `molmetal/reports/round11_engine_parity.md` + `round11_final.md` |

ENV constraints: download attempts use curl with timeout 60s; if blocked,
agent reports the HTTP status code and moves to fallback. All install
attempts logged to `molmetal/reports/round11_install_log.md`.

## Fallback ladder (per work axis)

| Axis | If env blocks | Cite justification |
|---|---|---|
| QVina | Vina 1.2.7 + exh=16 + empirical parity table | Trott 2010 J Comput Chem; Alhossary 2015 J Comput Aided Mol Des |
| QuickVina2 | Vina 1.2.7 (same scoring function lineage) | Hassan 2017 J Cheminform |
| CrossDocked100 download | Proxy subset (5-10 pockets available) | Document protocol divergence in §2 protocol-mismatch flags |

## Related reports

Local staging evidence: `molmetal/data/crossdocked100_manifest.csv` contains
100 test pairs sourced from `/mnt/storage/data/molmetal/crossdocked`; the
maintained loader reports 100 test and 100,000 train pairs.

- `molmetal/reports/round9_qvina_parity.md` (round-9 audit, citation-backed)
- `molmetal/reports/round9_data_audit.md` (round-9 audit, staged pockets)
- `molmetal/reports/lambda_vs_sbdd_protocol_aligned.md` (7 protocol flags)
- `molmetal/reports/round7_install_report.md` (round-7 install log)

## Round-11 N=50 parity result (2026-09-14)

N=50 molecules × 1h36 pocket × exh=8, seed=42, 1 CPU: Vina 1.2.7 (Python) vs QuickVina 2 (qvina02 binary). Pearson r=0.998, Spearman ρ=0.998, mean paired diff=+0.009±0.018 kcal/mol, mean |diff|=0.071 kcal/mol (n=46 paired; 4 mols rejected by QVina due to CG0 atom-type vocabulary mismatch, not parity failure). Single-pocket, single-seed, single-exhaustiveness — NOT a Round-13 acceptance number. Report: molmetal/reports/round11_engine_parity_n50.md; raw: molmetal/reports/round11_engine_parity/parity_raw.csv.
