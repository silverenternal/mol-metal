# D-MPNN line — compressed overview (2026-10-05)

Consolidates 9 historical wave docs (~1900 LOC total) that overlap heavily on the same trainer (`molmetal/scripts/train_dmpnn_multitask_wasserstein.py`).

**Canonical references (DO NOT DELETE — they are on-disk evidence per AGENTS.md §5.5):**

| Topic | Wave | Doc | Status |
|---|---|---|---|
| Seed integrity (Rh seed0=seed1 bug) | Wf-120 + TODO-112 | `120_dmpnn_seed_integrity.md` | 25 ckpts × 5 seeds, SHA256 diversity verified |
| Featurisation cache (pickle) | Wf-327-F8 / Wf-328-W10 | `327f8_dmpnn_cache_verified.md` + `328w10_dmpnn_cache_wired.md` | Wf-ultracode FIX now adds stderr warning on corrupt cache |
| PyG-style batched MP | Wf-328-W7 | `328w7_dmpnn_pyg_port.md` | CPU-correctness gate MEASURED |
| Pre-featurise optimisation | Wf-333-G4 | `333g4_pre_featurize_dmpnn.md` | batched featurisation |
| Dynamic `--metals` resolution | Wf-A6 BLOCKER A | `A1_dmpnn_dynamic_metals.md` | USER VOICE 2026-10-03 — auto-detect from CSV |
| M2 R² retrain (REFUSED) | Wf-398-A1 | `398a1_dmpnn_retrain.md` | SCAFFOLD_NOT_FUNCTIONAL refusal |

**Redundant / superseded (deleted 2026-10-05 per Wf-dmpnn-prune-docs):**

- ~~`112_dmpnn_seed_integrity.md` (73 LOC)~~ — DELETED 2026-10-05; superseded by `120_dmpnn_seed_integrity.md` (380 LOC, §6 final verdict covers same ground)
- ~~`112_dmpnn_seed_integrity_closeout.md` (116 LOC)~~ — DELETED 2026-10-05; same content as `120_*.md` §6 final verdict
- 01-10 `_done.md` series (8 files: `01_closed_loop_rewire_done.md` / `02_proof_search_prior_done.md` / `03_l3_tile_wireup_done.md` / `04_qvina_swap_done.md` / `06_posebusters_fix_done.md` / `08_tmqm_egnn_wireup_done.md` / `09_metal_prior_done.md` / `10_minibatch_ot_done.md`) — DELETED 2026-10-05; superseded by 14-23 `_done.md` series (byte-identical except for `Status:` line, round-8 vs pending)
- `18_reinvent4_install_done.md` — DELETED 2026-10-05; superseded by `05_reinvent4_install.md` (6211 b, full REINVENT4 install + AMD learned + multiproperty bridge)

**Redundancy map:** `112_*.md` and `120_*.md` cover the SAME Rh seed0=seed1 identity bug + SHA256 audit. `120_*.md` is the canonical closeout (Wf-120 wave). `112_*.md` (plan + closeout) predates Wf-120 and is subsumed.

**Wf-ultracode 2026-10-05 fixes (NEW, this wave — see `molmetal/tests/test_dmpnn_ultracode_fixes.py`):**

1. Pool-size silent-drop — recompute `n_metals_pool` after bare-except at 1633-1641 (was 5 even when one metal dropped)
2. best_state NaN fallback — set `best_epoch = -1` sentinel when no epoch won (was `best_epoch=epochs` silent)
3. predict_joint invalid-row bias — skip invalid SMILES entirely (was `score=0.0, label=int(a)` biased ROC down)
4. Cache load silent fail — emit stderr warning on corrupt pickle (was bare `except: pass`)

**Wf-ultracode 2026-10-05 model-architecture fixes (NEW, this wave — see `molmetal/tests/test_dmpnn_metal_lines_cleanups.py`):**

5. **Silent ligand-only graphs (Defect 1, CONFIRMED)** — `DMPNNEncoder` now strips precious-metal atoms (Pt, Pd, Ru, Ir, Rh, Os, Re, Au) from SMILES via the new `_strip_metal_atoms` helper in `molmetal/data/featurize.py`, default-on. This removes the cross-cohort asymmetry where platinai SMILES carry `[Pt+2]` (4078/4078 rows) while v4 (0/26632) and NCI60 (0/7823) carry no metal token. The metal id is still passed via `metal_ids` / `metal_embed`, so no information is lost — only the double-counting between the graph path and the metal-id tensor. `strip_metal_atoms=False` restores the legacy encoding for callers that need the metal atom in the graph.
6. **Multi-fragment disconnection (Defect 2, PARTIALLY CONFIRMED)** — `GraphFeaturizer.__call__` output dict now exposes `n_fragments` (dot-separated fragment count) so callers can detect and reason about disconnected graphs without re-parsing RDKit. The underlying disconnection remains (99.94% of v4 rows, 99.87% of NCI60 rows are multi-fragment); this fix is observability-only. The new `_count_fragments` helper provides the same count for direct use.

**Verified by 5/5 regression tests in `molmetal/tests/test_dmpnn_metal_lines_cleanups.py`** (all failed pre-fix, all pass post-fix):

- `test_strip_metal_atoms_helper_strips_pt` — `_strip_metal_atoms` removes Pt from platinai SMILES.
- `test_strip_metal_atoms_idempotent_on_no_metal` — `_strip_metal_atoms` no-op on ligand-only SMILES.
- `test_count_fragments_helper_basic` — `_count_fragments` returns correct count (4 for the multi-fragment fixture).
- `test_graph_featurizer_exposes_n_fragments` — `GraphFeaturizer.__call__` output includes `n_fragments`.
- `test_dmpnn_encoder_strips_metal_atoms` — encoder output for `[Pt+2]`-bearing SMILES equals output for the ligand-only strip (pre-fix, embeddings diverge because Pt is an atom in the graph).