# WF-Round12-SOTA-Integrate — cite-only SOTA column promotion (Round-12 integration)

**Date:** 2026-09-14
**Status:** PARTIAL (consistent with upstream `wf_round12_sota_subset/final.md`; honest framing preserved)
**Goal:** Integrate the WF-Round12-SOTA-Subset outcome into `paper/sections/04_evaluation.tex` §4.4 cite-only SOTA column and add a new live-DiffDock subsection, without fabricating any number.

## 1. Summary of upstream run

`molmetal/reports/wf_round12_sota_subset/final.md` reports:

| Field | Value |
|---|---|
| pockets requested | 5 |
| pockets completed (no timeout) | **0** |
| seeds | 1 (intended: 42) |
| n_ligands_total | **0** |
| confidence_mean | n/a |
| top1_confidence | n/a |
| gpu_blocked | True (`torch.cuda.is_available() == False`, ROCm 7.2 / RX 7800 XT gfx1101 wave64 not enumerated as `cuda`) |
| n_smoke_pockets (1-pocket smoke fallback ran) | 0 (blocked at `inference.py` parser-import; `torch_geometric` missing) |
| network_blocked (cannot fetch `diffdock_models.zip`) | True |
| esm_blocked (cannot embed receptor residues) | True |
| torch_geometric_blocked (parser-import fails) | True |
| wall-clock consumed | $<2$ min (pre-flight only) |
| wall-clock budget | 30 min |

All three blockers (GPU, ESM, network) hit before any generation
step ran. The brief's 1-pocket smoke fallback is also blocked at
parser-import (`from torch_geometric.data import Batch, Data`), so
even a single-pocket DiffDock confidence value is unavailable.

## 2. Cells promoted (DESIGN -> MEASURED) in `paper/sections/04_evaluation.tex` Table cite-only SOTA column

**n_cells_promoted = 0.**

No cite-only SOTA column cell can be promoted from `\DESIGN{}` to
`\MEASURED{}` because the upstream run produced **zero measured
DiffDock confidence values** (`n_ligands_total = 0`). Per the
§4.10 promotion rule of `04_evaluation.tex` (and the user-brief's
honest-framing clause), a `\DESIGN{}` cell is promoted only when a
real measurement replaces it; the Round-12 SOTA-subset outcome
provides no such measurement for the 5 mini-pilot pockets
(`test_000..test_004`).

Specifically:

- **0 of 5** `test_000..test_004` rows were touched in Table 1
  (`tab:per-pocket`). The Round-12 mini pilot (search-only, no
  physical docking, see `wf_round12_integrate.md`) tagged these 5
  rows as `\SEARCHONLY{}` for SA / QED / Lip-pass / LogP / TPSA /
  RotB and as `n/a (proxy)` for Vina. The DiffDock-subset attempt
  did not produce a new live number to overwrite any of these
  cells.
- **0 of 9** SOTA rows in the cite-only SOTA column of Table 2
  (`tab:aggregate`) were touched. DiffDock's row in the CSV
  (`wf_3_citeonly_sota_table.csv`) was already populated from the
  original paper and is `\CITEDONLY{}`; the Round-12 attempt was
  the *per-pocket* anchor, not an aggregate re-run, so no
  aggregate cite-only cell moves either.
- **0 of 5** rows in the new §4.4.1 subsubsection live-DiffDock
  per-pocket table were filled. The new subsubsection explicitly
  records the zero outcome in its *Result* paragraph.

The brief asked "for each of the 5 measured pockets, replace the
corresponding `\DESIGN{}` cell in Table 3 cite-only SOTA column
with the MEASURED DiffDock confidence". Since zero pockets were
measured, zero cells were replaced. **This is the honest outcome.**

## 3. Files modified

| File | Change |
|---|---|
| `paper/sections/04_evaluation.tex` | Added new subsubsection §4.4.1 `Live SOTA anchor attempt (DiffDock, 5 pockets × 1 seed)` (`sec:evaluation:sota:live-diffdock`) inside the existing §4.4 cite-only SOTA subsection. Six paragraphs: Goal, Protocol (planned), Result, Smoke-test fallback, Limitations, "What this changes in §4.4". No `\DESIGN{}` cell was promoted. |
| `paper/sections/CROSS_REFS.md` | §4.4 row extended with a one-line entry pointing to the new §4.4.1 subsubsection; cites `wf_round12_sota_subset/final.md` and the new integration report (`wf_round12_sota_integrate.md`). The original 7-flag block and `wf_3_citeonly_sota.md` §5 citation are preserved verbatim above the new entry. |
| `molmetal/reports/wf_round12_sota_integrate.md` | This report (new). |

## 4. §4.4.1 subsubsection anatomy (added to §4 evaluation)

The new §4.4.1 (`sec:evaluation:sota:live-diffdock`) explicitly
covers the four items called out in the integration brief:

1. **Protocol** --- DiffDock v1.1 `inference.py` CLI flags audited
   at pre-flight (`--protein_path`, `--ligand_description`,
   `--samples_per_complex=40` budget, `--inference_steps=20`,
   `--model_dir`/`--confidence_model_dir`, `--out_dir`); 5
   CrossDocked2020 pockets `test_000..test_004` × seed 42 × 40
   samples × 20 steps = 800 reverse-diffusion steps per pocket.
2. **Sample budget** --- 40 ligands/pocket, 5 pockets, 1 seed;
   30 min wall-clock ceiling; per-pocket wall-clock estimate
   4--8 h on CPU documented as the reason the CPU fallback is
   blocked even if blockers B and C were lifted.
3. **Result** --- `n_ligands_total = 0`; empty
   `wf_round12_sota_subset/diffdock/` directory; pre-flight
   stopped at $<2$ min; three independent blockers hit
   (GPU unavailable, ESM missing, network unreachable).
4. **Limitations (single-pocket if GPU blocked)** --- the 1-pocket
   smoke fallback was also blocked at parser-import
   (`torch_geometric` not in uv env); partial outcome recorded
   in the new subsubsection's *Smoke-test fallback* paragraph
   rather than fabricating a number.

## 5. Cross-reference map updates

`paper/sections/CROSS_REFS.md` §4 row was extended with the
§4.4.1 subsubsection pointer. The §4.4 cite-only SOTA narrative
(7 protocol-mismatch flags + `wf_3_citeonly_sota.md` §5) is
preserved verbatim above the new entry; nothing about the
`\CITEDONLY{}` cell-tag invariant is relaxed.

## 6. Follow-ups

The follow-up list (matching `wf_round12_sota_subset/final.md`
§6.5) for a future round that does ship a live SOTA anchor:

1. **Get ROCm enumerated as `cuda`** on this RX 7800 XT
   (\texttt{gfx1101}, wave64) box. Two candidate workarounds:
   (a) set `\texttt{HSA\_OVERRIDE\_GFX\_VERSION=10.3.0}` and
   verify `\texttt{torch.cuda.is\_available()}` returns True;
   (b) rebuild PyTorch against ROCm~7.2 with the gfx1101
   backend enabled (`PYTORCH_ROCM_ARCH=gfx1101`). Without this,
   DiffDock's CPU fallback is 4--8 h/pocket and exceeds the
   30 min budget by an order of magnitude.
2. **Install `fair-esm` and `torch_geometric`** into the uv
   env (network permitting; both are required by DiffDock's
   receptor-preprocessing pipeline and `inference.py`
   top-level imports respectively).
3. **Pre-fetch `diffdock_models.zip`** (best-EMA inference ckpt
   + 75-epoch confidence ckpt, 120 MB total) and unpack into
   `\texttt{molmetal/references/DiffDock/workdir/v1.1/\{score\_
   model,confidence\_model\}}` so the auto-download branch
   (`inference.py:120--141`) is skipped.
4. **Re-run the 5-pocket sweep** with
   `--samples_per_complex 40` (CPU path) or `--samples_per_complex 100`
   (GPU path), 1 seed 42, and write
   `\texttt{molmetal/reports/wf\_round12\_sota\_subset/diffdock/diffdock\_<pocket>/}`
   per-pocket CSVs of `(rank, smiles, confidence)`.
5. **Promote the cite-only SOTA column cells** for the 5
   CrossDocked2020 mini-pilot pockets from `\CITEDONLY{}` to
   `\MEASURED{}` *only at the point of use* (per the §4.10
   promotion rule); add a per-pocket DiffDock
   `\texttt{confidence\_mean}` cell to `tab:per-pocket` and
   cross-link from the §4.4.1 subsubsection.
6. **Pocket2Mol fallback** (per the upstream brief §1, deferred
   to a future round): if DiffDock remains blocked after the
   above, evaluate Pocket2Mol's
   `\texttt{molmetal/references/Pocket2Mol/sample\_for\_pdb.py}`
   + RDKit preprocessing + downloaded `ckpt/` dir. Pocket2Mol
   was not exercised in Round-12 because the upstream brief
   preferred the simpler one-shot DiffDock CLI; this preference
   is preserved here and not overridden.
7. **Citation hygiene**: if the live DiffDock run completes and
   the per-pocket confidence values differ from the cite-only
   SOTA CSV, document the delta in §6 Limitations (cite-only
   SOTA protocol-mismatch Flag~3, which currently covers PoseBusters
   validity for FLOWr, will need a parallel DiffDock PB-valid
   arm to remain honest).

## 7. Conclusion

**status = partial**, **n_cells_promoted = 0**, **n_pockets_measured_live = 0**.
The §4.4 cite-only SOTA column retains its `\CITEDONLY{}` tag for
all 9 SOTA rows; no live DiffDock number is papered over the
gap. The §4.4.1 subsubsection is the artefact --- it documents
the partial live-DiffDock attempt honestly and points readers to
the 3 unblock paths above.