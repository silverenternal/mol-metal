# Wf-A168 REFUTATION — EF-TALFM primary-source axis

VERDICT: **NOT REFUTED**

Adversarial re-derivation of the branch `wfa168_eftalfm_axis` (PAPER branch: arXiv 2609.08333v4).
0 GPU, CPU-only, `CUDA_VISIBLE_DEVICES=""` on every call, single venv, no forbidden tree written.

| # | check | result |
|---|---|---|
| 1 | re-derive the numbers (own live fetch + own extraction) | **PASS** |
| 1b | re-derive the repo-side numbers (recomputed, not copied) | **PASS** |
| 1c | published artifacts are reproducible, not hand-edited | **PASS** |
| 2 | the comparison is not overstated | **PASS** |
| 3 | the regression guard really bites (negative controls) | **PASS** (caveat: no test of its own) |
| 4 | no frontier number came from a search summary | **PASS** |
| 5 | `metrics/` + `registry/` unmodified by the branch | **PASS** |
| 6 | no GPU used | **PASS** |

## 1 — Re-derivation (own fetch)

Own live fetch of `https://arxiv.org/e-print/2609.08333v4` -> HTTP 200, 8508562 B,
sha256 `e13ac0f4fd13f22ecc0b70bc15954ffe0c30661970cf6c89cae73789ef742224` — **byte-identical to the
branch's recorded digest**. Own fetch of the abs page -> HTTP 200, 41623 B, sha256
`54e9eb0a9a7cda5f2e46d48d6e44a5ce769c6661b906b5b29d3dba17c6c18a90` — identical. Independently
extracted the tarball and read every cited site verbatim; all 26 cited lines match character-for-character.

- **P1 87.9 %** — `src/abstract.tex:1`; body `src/results_unconditional.tex:35`. FOUND.
- **P2 92.98 %** — `src/appendix.tex:115`, verified **structurally** to be the `C.` row, EF-TALFM
  column of panel (a) (header `UAE-3D | FlowMol | EF-TALFM` at `:110-113`); definition at `:60`.
- **P3 94.33 %** — `src/appendix.tex:127`, the bold `N.&U.&S.` row, EF-TALFM column; caption `:89`.
- **P4 93.13 %** — `src/appendix.tex:171`, panel (b) titled *"(b) PoseBusters results among N.&U.&S.
  molecules (%)"*, columns `UAE-3D | FlowMol | EF-TALFM` -> EF-TALFM = 93.13. Individual EF-TALFM
  rates recomputed = 99.79 / 99.32 / 97.22 / 99.93 / 98.09 / 97.69 -> **97.22–99.93**, as the paper says.
- Denominators re-read at `:53`, `:54`, `:55`, `:93` (incl. *"9,433 for EF-TALFM"*). Arithmetic
  cross-check reproduced: `9433 x 93.13 % / 10000 = 87.849529 %` ~ the abstract's 87.90 %.
- **"89.4" census re-run**: **0** occurrences across all 9 extracted `.tex` files, **0** on the abs
  page. The `0.8942` coincidence the branch mentions really is at `src/appendix.tex:832`.
- **Bond rule** `:1043` verbatim; `\lambda_\pi` occurs exactly **once** in the whole source; **no**
  `lambda*pi` product -> **pi IS A SUBSCRIPT**, confirmed. All-atom branch -> Open Babel (`:1016`, `:1018`).
- **Appendix lettering re-derived from the source's own hierarchy** (A unconditional / B conditional /
  **C Methods** at `:884` / D training-configs): `C.2.1` = `\subsubsection{Molecular graph reconstruction}`
  (`:987`), `C.2.2` = `:1028`; the reconstruction algorithm is the **3rd** `\begin{algorithm}` float
  (`:886, :942, :1034, :1118`) -> **Algorithm 3**. The branch's adjudication is **correct**.
- Wf-A160's independent record carries the **same** digest (`verification.json -> provenance/sha256`).

## 1b — Repo side (recomputed from disk, not copied)

`1078/12668 = 0.08509630565203663` — **identical** to the published `single_fragment_rate_over_parsed`.
Denominators 12669 / 12668 / 12668; post-processed axis 1.0 / 0 multi-component / 0 pp-fail /
`uff_ok_rate = 0.0`; `fraction_of_raw_mass_discarded = 0.22474803194442472`; `2663/12668 = 21.02 %`;
`10003/12668 = 78.96 %`. c8: `529/2112 = 0.25047348484848486`, `60/2112 = 0.028409090909090908` over
2112 rows — all match, and the branch correctly keeps c8 flagged as a different pool.
Cordero: **19** entries, the `molmetal/data/batch.py` table compared **equal**, max pair-sum
**2.78 A** at Z=(53,53)=I-I, `3.5/2.78 = 1.2589928057553958`. Deviation block at
`yuelbond_decoder.py:263`; `cutoff: float = 3.5` at 430/830/969/1742/2184; `radii_scaled_candidacy`
`= False` at **all three** constructor sites (1042/1778/2208); the Cordero table feeds only the
separate `0.7`-ratio steric veto.

## 1c — Reproducibility

Re-ran the branch's script into a temp out-dir from **my own** cache: `sources.json` diff = **0 lines**
(after normalising timestamp / cache_path / served_from / http_status), `report.md` diff = 2 lines
(the HTTP column reads `cache` vs `200`). **The published artifacts are exactly what the script
produces — they were not hand-edited.** In the branch window the only eftalfm writes are its own
script + its own report dir; `paper/ docs/ TODO/ configs/ metrics/ registry/ AGENTS.md README.md`
= **0** modifications.

## 2 — Not overstated

The branch states our 8.51 % is *"not on the same scale as ANY of the paper's four numbers"* and names
**four independent breaks**; it claims the **axis structure** is shared (both sides RAW), never a
magnitude. The house-format strings are labelled **PROPOSAL ONLY** and were applied to nothing.

## 3 — The guard bites

`test_wfa155_decoder_bond_candidacy.py` -> **34 passed**; `test_wfa156_bond_threshold_frontier.py`
-> **11 passed** (both with the CPU-lock banner, 0 GPU tests ran).

- **NEGATIVE CONTROL 1**: monkeypatched `radii_scaled_factor` back to the pre-repair `lambda * math.pi`
  *in memory* (no repo file touched) -> **15 FAILED / 19 passed**, incl.
  `test_the_factor_is_lambda_pi` and `test_the_radii_scaled_rule_is_stricter_than_the_flat_cutoff_at_lambda_one`.
- **NEGATIVE CONTROL 2**: re-ran the branch script against a tarball with `89.4` injected ->
  `scan.n_89_4_in_source = 1`, site `src/abstract.tex:1`, P1 reported NOT-FOUND.
- **Caveat**: the branch ships **no test of its own**.

## 4 / 5 / 6

Every frontier percentage has a `value_site` in the paper **and** a URL + status + sha256 + date
retrieval record; the one search-summary number is labelled NOT-FOUND. `find metrics registry
-newermt "2026-11-01 11:35"` -> **empty** (the working tree's metrics/registry drift is from earlier
waves, not this branch). The watchdog log still holds **exactly 24** lines, all the historical
`pid=700241, 550 MiB` record, mtime Oct 31 17:52 — **no new line**.

## Surviving concerns

- **C1 (low, latent)** — the script **crashes** (`TypeError` on `ck['abstract_pct']`, `render_report`
  line 710) on the NOT-FOUND path it explicitly provides for; `sources.json` is written, `report.md` is
  not. No published number affected.
- **C2 (low, doc drift)** — the branch's §2a break 2 ("parsed rows only, 1 unparseable excluded of
  12,669") is **more accurate** than `docs/HONEST_NEGATIVE_CATALOG.md` §6.264 §6 break 2 ("over ALL
  samples drawn, unparseable against"); the branch did not record this supersession. 8.51 % is unaffected.
- **C3 (low, framing)** — the branch presents the main caption `:93` as less ambiguous than `:183`,
  but connectivity *is* a SMILES-based metric, so `:93` is at least as ambiguous. The verdict (RAW)
  rests on the value (92.98 % cannot be post-LCC) and on `:60`, so it stands.
- **C4 (low)** — no test of its own; `definition_site` null for P2/P3 (both definitions are present
  at `:60` / `:89` and cited in the body).

## Verdict

**NOT REFUTED.** Every quoted percentage, denominator, post-processing qualifier, the 89.4 %
NOT-FOUND finding, the pi-is-a-subscript verdict, the Appendix C.2.1 / Algorithm 3 adjudication and
every repo-side number were independently re-derived and all match. The comparison is explicitly
framed as NOT like-for-like. The artifacts are byte-reproducible, the forbidden trees are untouched,
and no GPU was used. The residual defects are latent-robustness and documentation-drift items, none
of which alters a published number.
