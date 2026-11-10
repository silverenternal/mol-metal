#!/usr/bin/env python
"""(§6.242 extension) — every repo-relative path an SSOT document cites must resolve.

**The gap this closes.**  Catalog §6.242 records that the four SSOT documents
still cite **48** repo-relative paths that do not resolve on disk, and that
the Wf-A127 guard does not cover them (it owns one family:
``TODO/(completed|pending|archive)/*.md``).  §6.242 also says the number is a
*path-binding* measurement, not a verdict on each path.  This file is that
verdict, and the guard that keeps it from growing again.

**Value-directed, not shape-directed.**  A path-shaped citation is a claim of
the form *"there is an artifact at this location"*.  The resolver answers it
against **values**:

* the **filesystem** answers "does this exact path exist?";
* the **registry** (``registry/metrics.yaml`` — the metric SSOT) answers "is
  this name a metric, and if so, what is its canonical artifact?".

That asymmetry is what lets a short-form metric id (``metrics/by_metric/m1.json``)
be exempt while a typo'd metric path (``metrics/by_metric/m22.json``) is
reported: the two strings have the same shape, but only one of them names a
thing the registry declares.

**Why a baseline and not zero tolerance.**  The **54** pre-existing broken
citations are historical sentences in four SSOT documents.  *(MEASURED
2026-10-31.  The count is now **56** — see :data:`BASELINE_SIZE`; the two
extra entries were added 2026-11-01 by catalog §6.264.  The **54**
rendering above is PRESERVED VERBATIM per AGENTS.md §5.5 rule #6.
RE-MEASURED 2026-11-02 (Wf-A269 wave-5): the count is now **79** — the
**56** rendering above is PRESERVED VERBATIM per AGENTS.md §5.5 rule #6, and
the growth is recorded entry-by-entry in :data:`BASELINE`.)*  Rewriting them is
forbidden (AGENTS.md §5.5 rule #6 — the historical trail is preserved
verbatim; corrections are additive and dated) and restoring 54 files is a
different wave's work.  So this guard is a **growth detector**: any failing
citation outside the audited baseline fails the test.  Repairing a baseline
entry shrinks the failing set and must be re-adjudicated here.

**Nothing here is a silent hole.**  Every exemption is NAMED and carries its
own assertion:

* :data:`SHORT_FORM_METRIC_IDS` — the four short-form metric ids the corpus
  cites.  Each is asserted to be declared by the registry AND to have its
  canonical artifact on disk AND to still be cited (so the exemption is
  load-bearing, not dead code).
* :data:`BASELINE` — the **59** broken citations (**54** pre-existing + **2**
  added 2026-11-01 by catalog §6.264 + **2** added 2026-11-01 by catalog
  §6.267 + **1** added 2026-11-02 by catalog §6.269), each annotated
  with its adjudication (``ROOT_PREFIX`` / ``METRIC_NAMESPACE`` / ``MISSING``)
  and each asserted against the filesystem: a ``ROOT_PREFIX`` entry must name
  a real file a reader could be redirected to, a ``MISSING`` entry must have no
  same-basename file anywhere, a ``METRIC_NAMESPACE`` entry must genuinely be
  undeclared.  *(The **59** rendering above is PRESERVED VERBATIM per
  AGENTS.md §5.5 rule #6; the census is now **81** — 60 -> 79 at wave-5
  (2026-11-02) and 79 -> 81 at wave-17 (2026-11-04), catalog §6.269.24.  The
  per-step growth narrative is in the :data:`BASELINE_SIZE` comment block.)*
  *(RE-MEASURED 2026-11-04, Wf-A269 wave-25/26: the census is now **84** — the
  **81** rendering above is PRESERVED VERBATIM per AGENTS.md §5.5 rule #6;
  81 -> 82 at wave-22 (catalog §6.269.32), 82 -> 83 at wave-22 (catalog
  §6.269.35), 83 -> 84 at wave-25/26 (catalog §6.269.36).  The per-step growth
  narrative is in the :data:`BASELINE_SIZE` comment block.)*
* :data:`PATTERN_BASELINE` — the five glob/template citations that expand to
  nothing.
* :data:`SUPERSEDED_POINTERS` — the **1** citation that points at a
  **transient, run-scoped** artefact (MEASURED 2026-11-04).  It is kept
  OUT of :data:`BASELINE` on purpose: the baseline is compared for EQUALITY
  against the failing set, and a file a tool rewrites on every run is broken
  SOMETIMES and resolved OTHERS — a baseline entry for it would make the
  guard flap with an unrelated run.  It carries its own four assertions
  (still cited / not tracked / parent dir survives / dated
  ``SUPERSEDED-POINTER`` marker in the SSOT corpus), so the exemption is
  visible and dies with the correction it depends on.

**No filesystem walk.**  The repo is 201 GB; ``os.walk`` is forbidden by
AGENTS.md.  The path index comes from ``git ls-files`` (tracked +
untracked-but-not-ignored).  Existence is always decided by the filesystem; the
index is used only to find *where else* a basename lives.

**Scope.**  The same four SSOT documents the Wf-A127 guard reads, and the
same ``TODO/(completed|pending|archive)/*.md`` family is **delegated** to that
guard rather than duplicated (see
:func:`test_the_wfa127_family_is_delegated_not_duplicated`).
"""

from __future__ import annotations

import re
from pathlib import Path

from molmetal.tests import wfa131_path_reference_guard as G

REPO = G.REPO

#: The SSOT documents whose citations are checked.
DOCUMENTS: tuple[str, ...] = G.DOCUMENTS

#: The PRE-EXISTING broken citations, adjudicated path by path.
#:
#: MEASURED 2026-10-31 against the working tree of ``/home/hugo/codes/mol-metal``:
#: **54** distinct failing paths across the four SSOT documents.
#:
#: RE-MEASURED 2026-11-01: **56**.  The two extra paths are the ones catalog
#: §6.264 introduced with a wrong directory chain — they are still cited (the
#: superseded rendering is preserved VERBATIM inside a strike per AGENTS.md
#: §5.5 rule #6), so they still need an adjudication.  Both are
#: ``ROOT_PREFIX``: a file with that basename exists, under a different
#: directory chain, and the named target is the REAL file.
#:
#: The annotation is the adjudication, and each kind carries its own
#: assertion (see :func:`test_every_baseline_entry_is_adjudicated`):
#:
#: * ``ROOT_PREFIX -> <path>`` — a file with that basename exists, but under
#:   a different directory chain.  The named path is the REAL file, so the
#:   repair is a redirection, not a restoration.
#: * ``METRIC_NAMESPACE`` — the citation is inside the ``metrics/``
#:   namespace and the registry declares no such artifact there.
#: * ``MISSING`` — the cited path does not exist, and no file of that
#:   basename is in the git index (tracked + untracked-but-not-ignored).
#:   The index scope is a MEASURED limitation, not a convenience: the repo
#:   is 201 GB and a filesystem walk is forbidden (AGENTS.md), so a
#:   **gitignored** file of the same basename is INVISIBLE to this
#:   adjudication.  The §6.269 entry below is a case where one exists.
#:
#: This is the same "48 distinct (89 occurrences)" population that catalog
#: §6.242 recorded, re-adjudicated: the catalog's number was a path-binding
#: measurement taken before Wf-A128 redirected the ``TODO/`` family, and it
#: did not disambiguate short-form metric ids from genuine breakages.
#: Adjudicating them yields **54**, not 48 — see the report for the
#: reconciliation.
BASELINE: dict[str, str] = {
    "TODO/02_metrics/ARCHITECTURE.md": "ROOT_PREFIX -> docs/ARCHITECTURE.md",
    "TODO/03_baselines/ARCHITECTURE.md": "ROOT_PREFIX -> docs/ARCHITECTURE.md",
    "TODO/03_baselines/HONEST_NEGATIVE_CLOSURE_PLAN.md": "MISSING",
    "TODO/03_baselines/MOCK_TO_REAL_PLAN.md": "MISSING",
    "TODO/03_baselines/PROTOCOL_VALIDATION.md": "MISSING",
    "configs/c7_calib_noterms_2026_10_20.yaml": "MISSING",
    "configs/m7_phase1_gate_bace_clintox_2026_10_14.yaml": "MISSING",
    "configs/training/m17_pt_platinai_4cell_5seed_50ep_v1.yaml": "MISSING",
    "configs/training/m17_pt_platinai_4cell_5seed_50ep_v1.yaml.bak.wf449n3": "MISSING",
    "configs/training/m17_pt_platinai_4cell_5seed_50ep_v1.yaml.bak.wf449n4": "MISSING",
    "metrics/by_metric/m1_m2_m3_ablation.json": "METRIC_NAMESPACE",
    "metrics/by_metric/m22.json": "METRIC_NAMESPACE",
    "metrics/by_metric/m25.json": "METRIC_NAMESPACE",
    "metrics/by_metric/m2_hilbert.json": "METRIC_NAMESPACE",
    "metrics/by_metric/moleculenet_phase1_gate.json": "METRIC_NAMESPACE",
    "metrics/by_metric/vina_covalent.json": "METRIC_NAMESPACE",
    "metrics/by_metric/vina_min.json": "METRIC_NAMESPACE",
    "molmetal/adapters/admet8.py": "ROOT_PREFIX -> molmetal/eval/scoring/admet8.py",
    "molmetal/checkpoints/cfm_fullcorpus_2026_10_10_last.pt": "MISSING",
    "molmetal/checkpoints/cfm_yuelbond_wasserstein.pt": "MISSING",
    "molmetal/data/platinai_4cell_mbfinder.csv": "MISSING",
    "molmetal/eval/scoring/metric_hooks.py": "ROOT_PREFIX -> molmetal/molmetal_lam/sbdd_env/metric_hooks.py",
    "molmetal/models/cfm_decoder_yuelbond.py": "ROOT_PREFIX -> molmetal/molmetal_lam/lam_chem/cfm_decoder_yuelbond.py",
    "molmetal/molmetal/tests/test_cfm_yuelbond.py": "ROOT_PREFIX -> molmetal/tests/test_cfm_yuelbond.py",
    "molmetal/molmetal_lam/chemprop_v3.py": "MISSING",
    "molmetal/molmetal_lam/lam_chem/metal_smiles.py": "ROOT_PREFIX -> molmetal/data/metal_smiles.py",
    "molmetal/molmetal_lam/reward/equivariant_message_passing.py": "MISSING",
    "molmetal/molmetal_lam/reward/exact_minibatch_ot_coupling.py": "MISSING",
    "molmetal/molmetal_lam/reward/metal_coord_term.py": "ROOT_PREFIX -> molmetal/molmetal_lam/sbdd_env/metal_coord_term.py",
    "molmetal/molmetal_lam/reward/sa_penalty.py": "ROOT_PREFIX -> molmetal/adapters/flow_matching_lipman/sa_penalty.py",
    "molmetal/molmetal_lam/reward/tanimoto_diversity.py": "MISSING",
    "molmetal/molmetal_lam/reward/wasserstein_w1.py": "ROOT_PREFIX -> molmetal/adapters/flow_matching_lipman/wasserstein_w1.py",
    "molmetal/reports/cfm_fullcorpus_2026_10_10/checkpoints/cfm_yuelbond_wasserstein_seed0.pt": "ROOT_PREFIX -> results/wfa108h_b32_chunk2_vector_channels/ckpt/cfm_yuelbond_wasserstein_seed0.pt",
    "molmetal/reports/cfm_fullcorpus_2026_10_10/checkpoints/cfm_yuelbond_wasserstein_seed0_last.pt": "ROOT_PREFIX -> results/wfa108h_b32_chunk2_vector_channels/ckpt/cfm_yuelbond_wasserstein_seed0_last.pt",
    "molmetal/reports/r17_smoke_50step/checkpoints/cfm_yuelbond_wasserstein_seed0_epoch0.pt": "MISSING",
    "molmetal/reports/wf314e_ood_reachability/per_metal_diagnosis.json": "MISSING",
    "molmetal/reports/wf333_v3_vina/cohort_summary.json": "MISSING",
    "molmetal/reports/wf335m2_w2_chemprop/result.json": "ROOT_PREFIX -> tmp/wf409a3_launch_smoke/result.json",
    "molmetal/reports/wf_a39_m21_rerun/launch_and_aggregate.sh": "MISSING",
    "molmetal/scripts/_config_binding_gate.py": "ROOT_PREFIX -> scripts/_config_binding_gate.py",
    "molmetal/scripts/aggregate_phase1_gate.py": "MISSING",
    "molmetal/scripts/cfm_pad_groups_mcnemar.py": "ROOT_PREFIX -> scripts/cfm_pad_groups_mcnemar.py",
    "molmetal/scripts/launch_wf299_cfm_main.sh": "ROOT_PREFIX -> scripts/launch_wf299_cfm_main.sh",
    "molmetal/scripts/wf125_p1_controlled_pair.py": "MISSING",
    "molmetal/scripts/wf347_b3_launcher.sh": "MISSING",
    "molmetal/scripts/wf385a3_decode_ratio_regression_check.py": "MISSING",
    "molmetal/scripts/wfa103_shortcut_sigma_sweep.py": "MISSING",
    "molmetal/tests/test_cfm_decoder_anti_aromatic.py": "MISSING",
    "molmetal/tests/test_rocm_lipman.py": "MISSING",
    "molmetal/tests/test_wf399a2_catalog_regression_entry.py": "MISSING",
    "molmetal/train_cfm_yuelbond_wasserstein.py": "ROOT_PREFIX -> molmetal/scripts/train_cfm_yuelbond_wasserstein.py",
    "results/cfm_tmqm_all6/checkpoints/cfm_yuelbond_wasserstein_seed0.pt": "ROOT_PREFIX -> results/wfa108h_b32_chunk2_vector_channels/ckpt/cfm_yuelbond_wasserstein_seed0.pt",
    # --- Wf-A269 wave-3 (2026-11-02) re-adjudication ---------------------
    # The wave-3 launcher move made this citation RESOLVE again (the
    # launcher now lives at ``scripts/``), so the recorded adjudication no
    # longer describes the corpus.  The entry is RETIRED, not deleted: the
    # superseded rendering is preserved VERBATIM here, per AGENTS.md §5.5
    # rule #6 (a correction is ADDITIVE + DATED, never a silent deletion):
    #   "scripts/launch_wf74_v2_drugood_dann.sh": "ROOT_PREFIX -> TODO/archive/wf437_simplify/launch_wf74_v2_drugood_dann.sh",
    #
    # Two NEW citations are adjudicated ``MISSING`` in the same wave: the
    # USER_OK-gate tests used to ``shutil.rmtree`` the repo's tracked
    # results tree, which destroyed the two artefacts below.  MEASURED
    # (2026-11-02): neither name occurs in the git index at all
    # (``git ls-files | grep -E 'master\.log|RECONCILIATION\.md'`` -> 0
    # hits), so the ``MISSING`` adjudication is the honest one — there is no
    # same-basename file to redirect a reader to.
    "results/wf124_p3_per_innovation_ablation_g6/RECONCILIATION.md": "MISSING",
    # Wf-A269 wave-12 (2026-11-02): the `master.log` row that used to sit
    # beside the `RECONCILIATION.md` row above was REMOVED, because the
    # corpus was REPAIRED -- the file came BACK (4 592 B, mtime
    # 2026-11-02 22:32:54) and the citation resolves again, so the
    # exemption had become a hole.  The guard's own rule says exactly this
    # ("the corpus was repaired and the baseline must be tightened").  It is
    # still UNTRACKED (`git ls-files --error-unmatch` errors for it), which
    # is a fact about the working tree, not about the citation.  The
    # historical fact is kept here: the results tree had destroyed the
    # artefact, and the `RECONCILIATION.md` sibling is STILL missing, so
    # ITS row stays.
    # "results/wf124_p3_per_innovation_ablation_g6/master.log": "MISSING",
    "scripts/launch_wf_fix3lines_bace.sh": "ROOT_PREFIX -> scripts/_retired/launch_wf_fix3lines_bace.sh",
    "scripts/launch_wf_fix3lines_cfm.sh": "ROOT_PREFIX -> scripts/_retired/launch_wf_fix3lines_cfm.sh",
    "tests/test_rocm_env.py": "MISSING",
    # --- Wf-A175 (2026-11-01) re-adjudication, catalog §6.267 ------------
    # A GIT BLOB, not a disk file: the pre-removal snapshot of the
    # pre-flight lives at `aee5f8ab^:TODO/archive/wf437_simplify/
    # pre_flight_check.py.bak.wf437` (359,637 B) and was junk-purged from
    # disk.  The catalog cites it as evidence; the citation cannot be
    # deleted (AGENTS.md §5.5 rule #6), so it is adjudicated here.
    "TODO/archive/wf437_simplify/pre_flight_check.py.bak.wf437": "MISSING",
    # NEVER TRACKED IN GIT: the emitter is the spec source of the dead
    # gate `p3_protocol_recommended_section_6_honest_minor_section_auto_emit`.
    # `git log --all --oneline -- molmetal/scripts/auto_emit_section_6_minor.py`
    # is EMPTY (MEASURED 2026-11-01).
    "molmetal/scripts/auto_emit_section_6_minor.py": "MISSING",
    # --- Wf-A269 (2026-11-02) re-adjudication, catalog §6.269 -------------
    # The catalogue's OWN new §6.269.4c sentence cites the checkpoint the
    # arm needed:
    #   `molmetal/reports/wf_pearson_ridge_gpu/checkpoints/dmpnn_mtw_ru_seed0.pt`
    # That path does NOT exist — its parent directory survives on disk and is
    # EMPTY (A201's D4).  The sentence cannot be rewritten (AGENTS.md §5.5
    # rule #6 — the historical trail is preserved VERBATIM), so the broken
    # citation is adjudicated here: the same mechanism as the §6.264 and
    # §6.267 pairs above.
    #
    # WHY ``MISSING`` AND NOT ``ROOT_PREFIX``.  The ONLY file of that
    # basename anywhere on disk is
    #   `molmetal/checkpoints/wf196_50ep/dmpnn_mtw_ru_seed0.pt`
    # and it is a DIFFERENT RUN — MEASURED 2026-11-02 by loading it
    # (CPU-only, `torch.load(map_location="cpu")`): `n_train/n_val/n_test =
    # 2509/363/796`, `epochs = 50`, `batch_size = 256`, `hidden = 64`,
    # `depth = 2`; whereas the arm's OWN record
    # (`molmetal/reports/wf_pearson_ridge_gpu/ru/per_seed_metrics_seed0.json`)
    # is `2262/394/994`, `epochs = 30`, `batch_size = 64`, `hidden = 128`,
    # `depth = 3`.  It is therefore NOT a redirect target — pointing the
    # citation at it would make the catalogue claim the arm IS reproducible,
    # which it is not (that is the whole point of the sentence it appears in).
    #
    # WHY THE CLASSIFIER STILL SAYS ``DANGLING`` (not ``BROKEN_ROOT_PREFIX``).
    # That checkpoint is **gitignored**, so it is absent from the git index
    # the resolver uses for "where else does this basename live"
    # (`git ls-files`, tracked + untracked-but-not-ignored).  The guard cannot
    # see gitignored artifacts, so the token has no indexed same-basename
    # sibling — and ``MISSING`` is exactly the adjudication that matches that
    # bucket (MEASURED 2026-11-02: `classify(...) -> ("DANGLING", "")`).
    "molmetal/reports/wf_pearson_ridge_gpu/checkpoints/dmpnn_mtw_ru_seed0.pt": "MISSING",
    # --- Wf-A269 wave-5 (2026-11-02) re-adjudication, catalog §6.269.16 -----
    # MEASURED: the corpus grew from 60 to 79 broken citations.  ALL 19 new
    # ones come from `docs/HONEST_NEGATIVE_CATALOG.md` ALONE -- the guard's
    # own DOCUMENT set is (catalog, TODO/INDEX.md, TODO/README.md,
    # docs/WF_HISTORY.md), and the other three documents contribute none.
    #
    # THEY ARE NOT WRONG DOCUMENTS.  They are the catalogue's OWN §5.5
    # rule-#6 historical renderings: the launcher paths the wave-2/3/4
    # narrative QUOTES, which the wave-3 launcher move (and the `_retired`
    # move) turned into dead paths, plus three deliberate fixture/example
    # spellings.  A scanned document claims nothing wrong about a path that
    # must exist, and nothing wrong about a path that must NOT exist -- so
    # the correct repair is to RECORD the adjudication, not to rewrite the
    # catalogue (which AGENTS.md §5.5 rule #6 forbids).
    #
    # (a) THE WAVE-3 LAUNCHER MOVE -- 10 ROOT_PREFIX entries.  The launcher
    #     moved from TODO/archive/wf437_simplify/ (or from molmetal/scripts/)
    #     to scripts/; each citation still resolves under the NEW spelling.
    "TODO/archive/wf437_simplify/launch_wf264_cfm_property_panel.sh": "ROOT_PREFIX -> scripts/launch_wf264_cfm_property_panel.sh",
    "TODO/archive/wf437_simplify/launch_wf306b_wf219_ru_gpu.sh": "ROOT_PREFIX -> scripts/launch_wf306b_wf219_ru_gpu.sh",
    # Wf-A269 wave-12 (2026-11-02): the same shape, one more instance.  The
    # launcher was MOVED from TODO/archive/wf437_simplify/ to scripts/ (the
    # working-tree move is staged; the committed location is still the
    # archive -- `git log --all -- 'scripts/launch_wf310c_t179_w2_gpu.sh'`
    # returns NOTHING).  Catalog §6.269.16.7 cites the ARCHIVE path as a
    # historical git fact ("the launcher's only committed location is ..."),
    # so the citation is a historical pointer, not a live one -- exactly the
    # shape the two rows above already adjudicate.
    "TODO/archive/wf437_simplify/launch_wf310c_t179_w2_gpu.sh": "ROOT_PREFIX -> scripts/launch_wf310c_t179_w2_gpu.sh",
    "molmetal/scripts/launch_wf124_p3_g6.sh": "ROOT_PREFIX -> scripts/launch_wf124_p3_g6.sh",
    "molmetal/scripts/launch_wf306a_drugood_vina_gpu.sh": "ROOT_PREFIX -> scripts/launch_wf306a_drugood_vina_gpu.sh",
    "molmetal/scripts/launch_wf306b_wf219_ru_gpu.sh": "ROOT_PREFIX -> scripts/launch_wf306b_wf219_ru_gpu.sh",
    "molmetal/scripts/launch_wf308c_todo220_gpu.sh": "ROOT_PREFIX -> scripts/launch_wf308c_todo220_gpu.sh",
    "molmetal/scripts/launch_wf310c_t179_w2_gpu.sh": "ROOT_PREFIX -> scripts/launch_wf310c_t179_w2_gpu.sh",
    "molmetal/scripts/launch_wf310c_todo220_gpu.sh": "ROOT_PREFIX -> scripts/launch_wf310c_todo220_gpu.sh",
    "molmetal/scripts/launch_wf310f_frontier_math_gpu.sh": "ROOT_PREFIX -> scripts/launch_wf310f_frontier_math_gpu.sh",
    "molmetal/scripts/launch_wf333_vina_4step_gpu_bench.sh": "ROOT_PREFIX -> scripts/launch_wf333_vina_4step_gpu_bench.sh",
    # (b) THE G4 ARCHIVE + THE `_retired/` MOVE -- 2 ROOT_PREFIX entries.
    #     G4 must NOT exist under scripts/ (asserted by
    #     molmetal/tests/test_wf337g2_launcher_v2.py); the retired launcher
    #     lives under scripts/_retired/ with a `_5seed` suffix.
    "scripts/launch_wf124_p3_g4.sh": "ROOT_PREFIX -> TODO/archive/wf437_simplify/launch_wf124_p3_g4.sh",
    "scripts/launch_wf_fix3lines_cfm_30ep_onecycle.sh": "ROOT_PREFIX -> scripts/_retired/launch_wf_fix3lines_cfm_30ep_onecycle.sh",
    # (c) THE TYPO CLASS -- 2 MISSING entries.  The catalogue itself calls
    #     these TYPO-class literals: the `_retrain` spelling exists nowhere
    #     (`git ls-files` -> 0 hits); the real names are
    #     scripts/launch_wf380_scaffold_loss.sh and
    #     scripts/launch_wf394_lambda_cfm.sh.
    "molmetal/scripts/launch_wf380_scaffold_loss_retrain.sh": "MISSING",
    "molmetal/scripts/launch_wf394_lambda_cfm_retrain.sh": "MISSING",
    # (d) NEVER TRACKED IN GIT -- 1 MISSING entry.  Same class as
    #     molmetal/scripts/auto_emit_section_6_minor.py above: the shipped
    #     copy of the patch gate lives outside the repo (/tmp).
    "molmetal/scripts/wf_a269_patch_gate.py": "MISSING",
    # (e) DELIBERATE FIXTURES / EXAMPLES -- 3 MISSING entries.  NOT defects:
    #     `scripts/launch_child.sh` is a RUNTIME fixture created by
    #     molmetal/tests/test_wf_launcher_hardening_gate.py at runtime;
    #     `scripts/launch_x.sh` is a DOCSTRING EXAMPLE in
    #     scripts/_launcher_hardening_gate.py; `scripts/launch_X.sh` is the
    #     same generic placeholder spelled with a capital X.
    "scripts/launch_child.sh": "MISSING",
    "scripts/launch_x.sh": "MISSING",
    "scripts/launch_X.sh": "MISSING",
    # (f) THE RETIRED-BASENAME CLASS -- 1 MISSING entry.  The catalogue
    #     quotes the pre-repair comment naming a launcher that NEVER existed
    #     under that spelling (the retired one carries the `_5seed` suffix).
    "scripts/launch_wf_blk_d1_attentivefp.sh": "MISSING",
    # (g) THE §6.269.24 EVIDENCE CLASS -- 2 MISSING entries (2026-11-04).
    #     Both are cited by catalog §6.269.24 as the EVIDENCE of the 0-GPU
    #     contract breach, and both must STAY cited: AGENTS.md §5.5 rule #6
    #     forbids rewriting the entry to hide a path that no longer resolves
    #     (the same mechanism as the §6.264 / §6.267 / §6.269 pairs -- a
    #     historical rendering whose artefact is gone for good).
    #       * molmetal/scripts/smoke/_tmp_orig_smoke_cfm.sh -- the untracked
    #         COPY of HEAD's smoke script that the main thread left INSIDE
    #         the repo's smoke directory.  §6.269.24.6 records its DELETION,
    #         so its ABSENCE is the point of the entry: the file's very
    #         deletion is the evidence that the /tmp symlink farm was not an
    #         isolation.  (It also had to be deleted: left in place, the new
    #         smoke-lock guard would have globbed a file that is not a smoke
    #         script -- a false positive created by the guard's own
    #         population rule.)
    #       * scripts/run_nlp_structure_ablation.py -- the planB project's
    #         entry script.  It resolves in /home/hugo/codes/planB, NOT in
    #         this repository: §6.269.24.1 cites it to show that the ONE
    #         remaining GPU compute app was ANOTHER project's, i.e. that the
    #         measured 0-GPU state was ours and not ours.  It is a
    #         path-shaped citation of an EXTERNAL artefact; this census is a
    #         REPO-relative measurement, and the token is repo-rooted only
    #         by coincidence.
    #     MEASURED 2026-11-04: the resolver's `by_base` holds NO entry for
    #     either basename (`git ls-files` finds 0 files named either), so
    #     both are adjudicated MISSING.
    "molmetal/scripts/smoke/_tmp_orig_smoke_cfm.sh": "MISSING",
    "scripts/run_nlp_structure_ablation.py": "MISSING",
    # (h) THE §6.269.32 SELF-DISCLOSED PHANTOM — 1 MISSING entry (2026-11-04).
    #     catalog §6.269.32 records the repair of the BORN-FALSE pin
    #     `molmetal/tests/test_wf375a2_total_entries_count_is_71.py` and
    #     states, IN ITS OWN TEXT, that "the path in the superseded
    #     docs/STATUS.md line NEVER existed".  Naming a path to say that it
    #     never existed is a SELF-DISCLOSED broken citation: the catalogue is
    #     not wrong about it (it says so), but the token does not resolve, so
    #     the census counts it.  Same mechanism as (a)-(g): the rendering is
    #     preserved VERBATIM (AGENTS.md §5.5 rule #6 — the catalogue MUST NOT
    #     be reworded to hide the path), so the broken citation stays cited
    #     and still needs its adjudication.  The correct repair is to RECORD
    #     the adjudication (this table), NOT to rewrite the catalogue.
    #     MEASURED 2026-11-04: `git ls-files` finds 0 files with that
    #     basename and `ls molmetal/tests/` shows 0 — the live file is
    #     `molmetal/tests/test_wf375a2_catalog_wetlab.py` (the pin was
    #     consolidated into it), and the resolver's `by_base` holds NO entry
    #     for the phantom basename, so it is adjudicated MISSING.
    "molmetal/tests/test_wf375a2_total_entries_count_is_71.py": "MISSING",
    # (i) THE WAVE-22 SELF-DISCLOSED PHANTOM — 1 MISSING entry (2026-11-04).
    #     catalog §6.269.35.6 records THIS WAVE's own self-correction: the
    #     enforcement-stack population was first written as "22 files", and the
    #     list behind that number carried a path that NEVER existed.  Naming it
    #     is a SELF-DISCLOSED broken citation — the catalogue is not wrong about
    #     it (it says so), but the token does not resolve, so the census counts
    #     it.  Same mechanism as (a)-(h): the rendering is preserved VERBATIM
    #     (AGENTS.md §5.5 rule #6 — the catalogue MUST NOT be reworded to hide
    #     the path), so the broken citation stays cited and still needs its
    #     adjudication.  The correct repair is to RECORD the adjudication (this
    #     table), NOT to rewrite the catalogue.  §6.269.35.7 states the
    #     census mechanism (the census counts TOKENS, not truth).
    #     MEASURED 2026-11-04: `git ls-files` finds 0 files with that basename
    #     and `ls molmetal/scripts/` shows 0 — the receipt's real producer is
    #     `molmetal/scripts/wf269a_receipt_locator_refresh.py` (already in the
    #     catalogue's list of 21) and the receipt itself is the TRACKED
    #     `docs/wf_a44_verify_receipts.md`.
    "molmetal/scripts/wf_a44_verify_receipts.py": "MISSING",
    # (j) THE Wf-A269 WAVE-25/26 SELF-DISCLOSED PHANTOM — 1 MISSING entry
    #     (2026-11-04).  catalog §6.269.36 (the F6 family — "MEASURE IT, PRINT
    #     IT, GATE ON IT — THEN DISCARD IT" + the DEAD-GATE class in two
    #     costumes) names `molmetal/reports/wf426a3_retrain_readiness.json`
    #     THREE times, and every one of the three is a citation IN ORDER TO SAY
    #     THAT THE FILE DOES NOT EXIST:
    #       §6.269.36 (subject) — the launcher "publishes
    #         molmetal/reports/wf426a3_retrain_readiness.json";
    #       §6.269.36.3 — it "was **never produced** — the file does not exist
    #         on disk";
    #       §6.269.36.8 — it "remains **ABSENT**".
    #     The launcher `molmetal/scripts/wf426a3_post_fix_retrain_launcher.sh`
    #     DECLARES the path (READINESS_JSON, line 29) and writes it only AFTER
    #     the §1.4 gate passes; the live pre-flight measures 3 FAIL gates, so
    #     the deliverable is unreachable BY DESIGN and the catalogue says so.
    #     Same mechanism as (a)-(i): the rendering is preserved VERBATIM
    #     (AGENTS.md §5.5 rule #6 — the catalogue MUST NOT be reworded to hide
    #     the path), so the broken citation stays cited and still needs its
    #     adjudication.  The correct repair is to RECORD the adjudication (this
    #     table), NOT to rewrite the catalogue.
    #     MEASURED 2026-11-04: `git ls-files
    #     molmetal/reports/wf426a3_retrain_readiness.json` returns 0 paths and
    #     the path is absent from disk (ENOENT) — the file has NEVER existed in
    #     this repository.
    "molmetal/reports/wf426a3_retrain_readiness.json": "MISSING",
}

#: The recorded size of :data:`BASELINE`.
#:
#: MEASURED 2026-10-31: **54**.  RE-MEASURED 2026-11-01: **56** — the two
#: extra entries are the paths catalog §6.264 introduced with a wrong
#: directory chain (the superseded rendering is preserved VERBATIM inside a
#: strike per AGENTS.md §5.5 rule #6, so the broken citation is still
#: cited and still needs its adjudication).
#:
#: RE-MEASURED 2026-11-01 (Wf-A175): **58** — two more entries, both
#: introduced by catalog §6.267 (2026-11-01):
#:
#:   * ``TODO/archive/wf437_simplify/pre_flight_check.py.bak.wf437`` — a
#:     GIT BLOB (``aee5f8ab^:…``, 359,637 B), junk-purged from disk;
#:   * ``molmetal/scripts/auto_emit_section_6_minor.py`` — NEVER TRACKED
#:     IN GIT (the spec source of the dead gate
#:     ``p3_protocol_recommended_section_6_honest_minor_section_auto_emit``).
#:
#: Both are adjudicated ``MISSING``: MEASURED 2026-11-01, `git ls-files`
#: finds **0** files with either basename, and the
#: ``TODO/archive/wf437_simplify/`` directory survives on disk holding only
#: launcher ``.sh`` files.  Same mechanism as the §6.264 pair — the
#: superseded rendering is preserved VERBATIM (AGENTS.md §5.5 rule #6), so
#: the broken citation is still cited and still needs its adjudication.
#:
#: RE-MEASURED 2026-11-02 (Wf-A269): **59** — one more entry, introduced by
#: catalog §6.269 (2026-11-02): the catalogue's OWN new §6.269.4c sentence
#: cites ``molmetal/reports/wf_pearson_ridge_gpu/checkpoints/
#: dmpnn_mtw_ru_seed0.pt``, a checkpoint that was never persisted.  Third
#: instance of the same mechanism (after §6.264 and §6.267).  It is
#: adjudicated ``MISSING``; see the entry's own comment for why it is NOT
#: ``ROOT_PREFIX`` (the same-basename file that DOES exist on disk is a
#: different run AND is gitignored).
# Wf-A269 wave-3 (2026-11-02) — 59 -> 60: the wave-3 launcher move
# RETIRED one baseline entry (the citation resolves again) and the
# destructive-test loss ADDED two ``MISSING`` entries.  The superseded
# rendering (59) is preserved VERBATIM above per AGENTS.md §5.5 rule #6.
#
# Wf-A269 wave-5 (2026-11-02) — 60 -> 79: NINETEEN more entries, ALL of
# them the catalogue's OWN §5.5 rule-#6 historical renderings (the launcher
# paths the wave-2/3/4 narrative quotes) plus three deliberate
# fixture/example spellings.  MEASURED: every one of the 19 is cited by
# `docs/HONEST_NEGATIVE_CATALOG.md`; the guard's other three documents
# (TODO/INDEX.md, TODO/README.md, docs/WF_HISTORY.md) contribute NONE; and
# NO scanned document is wrong about a path that must exist or must NOT
# exist.  The correct repair is to RECORD the adjudication (this table),
# NOT to rewrite the catalogue -- which AGENTS.md §5.5 rule #6 forbids.
# The superseded rendering (60) is preserved VERBATIM above per
# AGENTS.md §5.5 rule #6.
#
# Wf-A269 wave-17 (2026-11-04) — 79 -> 81: TWO more entries, introduced by
# catalog §6.269.24 (2026-11-04) — the entry that recorded the 0-GPU
# contract breach.  Both are its EVIDENCE:
# ``molmetal/scripts/smoke/_tmp_orig_smoke_cfm.sh`` (the scaffolding copy
# that §6.269.24.6 records as DELETED) and
# ``scripts/run_nlp_structure_ablation.py`` (the planB project's entry
# script, cited to show that the one remaining GPU compute app belonged to
# another project).  Same mechanism as every pair above: the rendering is
# preserved VERBATIM (AGENTS.md §5.5 rule #6), so the broken citation stays
# cited and still needs its adjudication.  The correct repair is to RECORD
# the adjudication (this table), NOT to rewrite the catalogue.
# MEASURED 2026-11-04: the corpus holds 81 broken citations (79 before the
# §6.269.24 append, 81 after — the two paths above are the ONLY delta; no
# baseline entry went stale, and no citation was repaired).
# The superseded rendering (79) is preserved VERBATIM above per AGENTS.md
# §5.5 rule #6.
#
# Wf-A269 wave-22 (2026-11-04) — 81 -> 82: ONE more entry, introduced by
# catalog §6.269.32 (2026-11-04) — the entry that repaired the BORN-FALSE
# §6.269.32 pin.  The path is
# `molmetal/tests/test_wf375a2_total_entries_count_is_71.py`: §6.269.32
# states IN ITS OWN TEXT that this path "never existed", so naming it is a
# SELF-DISCLOSED broken citation.  Same mechanism as every pair above — the
# rendering is preserved VERBATIM (AGENTS.md §5.5 rule #6 forbids rewording
# the catalogue to hide the path), so the broken citation stays cited and
# still needs its adjudication.  The correct repair is to RECORD the
# adjudication (this table), NOT to rewrite the catalogue.
# MEASURED 2026-11-04: the corpus holds 82 broken citations (81 before the
# §6.269.32 append, 82 after — this path is the ONLY delta; no baseline
# entry went stale, and no citation was repaired).
# The superseded rendering (81) is preserved VERBATIM above per AGENTS.md
# §5.5 rule #6.
# Wf-A269 wave-22 (2026-11-04) — 82 -> 83: ONE more entry, introduced by
# catalog §6.269.35.6 (2026-11-04) — this wave's OWN self-correction.  The
# enforcement-stack population was first written as "22 files"; the list behind
# that number carried `molmetal/scripts/wf_a44_verify_receipts.py`, a path that
# NEVER existed (the receipt's real producer is
# `molmetal/scripts/wf269a_receipt_locator_refresh.py`; the receipt itself is
# the tracked `docs/wf_a44_verify_receipts.md`).  The corrected number is 21.
# The path is named in the catalogue IN ORDER TO SAY THAT IT NEVER EXISTED, so
# it is a SELF-DISCLOSED broken citation: same mechanism as every pair above —
# the rendering is preserved VERBATIM (AGENTS.md §5.5 rule #6 forbids rewording
# the catalogue to hide the path), so the broken citation stays cited and still
# needs its adjudication.  The correct repair is to RECORD the adjudication
# (this table), NOT to rewrite the catalogue.
# MEASURED 2026-11-04: the corpus holds 83 broken citations (82 before the
# §6.269.35 append, 83 after — this path is the ONLY delta; no baseline
# entry went stale, and no citation was repaired).
# The superseded rendering (82) is preserved VERBATIM above per AGENTS.md
# §5.5 rule #6.
#
# Wf-A269 wave-25/26 (2026-11-04) — 83 -> 84: ONE more entry, introduced by
# catalog §6.269.36 (2026-11-04) — this wave's OWN new entry, the F6 family
# ("MEASURE IT, PRINT IT, GATE ON IT — THEN DISCARD IT" + the DEAD-GATE class
# in two costumes).  The path is
# `molmetal/reports/wf426a3_retrain_readiness.json`: the launcher's OWN
# deliverable, cited THREE times, and every one of the three citations is a
# SELF-DISCLOSED phantom — §6.269.36 says the launcher "publishes" it,
# §6.269.36.3 says it "was **never produced** — the file does not exist on
# disk", and §6.269.36.8 says it "remains **ABSENT**".  The launcher DECLARES
# it (READINESS_JSON, line 29) and writes it only AFTER the §1.4 gate passes;
# the live pre-flight measures 3 FAIL gates, so the deliverable is unreachable
# BY DESIGN — and the catalogue says so.
# MEASURED 2026-11-04: `git ls-files` returns 0 paths for that name and the
# path is absent from disk (ENOENT) => it has NEVER existed in this
# repository.  Same mechanism as every pair above — the rendering is preserved
# VERBATIM (AGENTS.md §5.5 rule #6 forbids rewording the catalogue to hide the
# path), so the broken citation stays cited and still needs its adjudication.
# The correct repair is to RECORD the adjudication (this table), NOT to rewrite
# the catalogue.
# MEASURED 2026-11-04: the corpus holds 84 broken citations (83 before the
# §6.269.36 append, 84 after — this path is the ONLY delta; 0 baseline
# entries went stale, and no citation was repaired).
# The superseded rendering (83) is preserved VERBATIM above per AGENTS.md
# §5.5 rule #6.
BASELINE_SIZE = 84

#: The short-form metric ids the corpus cites.  These are metric IDS, not
#: file paths: the registry resolves each to a canonical artifact, so they
#: are exempt — but only because the registry says so, and only for these
#: four ids.
#:
#: The exemption is NAMED (not a string shape) and every entry carries its
#: own assertion that the thing it exempts is real: the registry must
#: declare the id, the canonical artifact must be on disk, the corpus must
#: still cite it, and the token must not be failing.
SHORT_FORM_METRIC_IDS: tuple[str, ...] = ("m1", "m2", "m7", "m19")

#: The glob / template citations that expand to nothing.  MEASURED
#: 2026-10-31: 47 glob citations in the corpus, 4 of which expand to nothing.
PATTERN_BASELINE: dict[str, str] = {
    "TODO/completed/313b_wetlab_": "MISSING",
    "molmetal/reports/pearson_head_fix_2026_10_08/seed0|"
    "molmetal/reports/pearson_head_fix_2026_10_08/seed1|"
    "molmetal/reports/pearson_head_fix_2026_10_08/seed2": "MISSING",
    "molmetal/scripts/wf125": "MISSING",
    "molmetal/tests/test_wf399a2": "MISSING",
    # --- Wf-A175 (2026-11-01) re-adjudication, catalog §6.267 ------------
    # The DIRECTORY exists but holds `per_seed_results.json`, so this glob
    # expands to nothing.  MEASURED 2026-11-01.
    "molmetal/reports/wf131_p3_deferred_runs/per_seed_metrics_seed": "MISSING",
    # --- Wf-A269 wave-5 (2026-11-02) re-adjudication, catalog §6.269.16 -----
    # MEASURED: the pattern corpus grew from 5 to 6 broken globs.  The new
    # one is `molmetal/scripts/launch_` -- the catalogue's shorthand for the
    # launcher family.  `molmetal/scripts/` holds ZERO `launch_*` files on
    # disk (the launchers live under the repo-root `scripts/`), so the glob
    # expands to nothing.  It is cited by the catalogue at 4 sites.
    "molmetal/scripts/launch_": "MISSING",
    # --- Wf-A269 wave-6 (2026-11-04) re-adjudication, catalog §6.269.20 ----
    # The literal token `configs/wfa269_c1_dmpnn_pair_sweep_g5_Metal` is the
    # PREFIX of a TEMPLATE citation: the catalogue writes
    # `configs/wfa269_c1_dmpnn_pair_sweep_g5_{Metal}_seed{N}.yaml`, so the
    # guard's pattern scan stops at the first `{` and the prefix expands to
    # nothing.
    #
    # The TEMPLATE is not a dead reference.  MEASURED 2026-11-04 on this
    # working tree: the `_g5_` family holds **15** files
    # (`configs/wfa269_c1_dmpnn_pair_sweep_g5_<Metal>_seed<N>.yaml` — 5 metals
    # x 3 seeds) and the whole `wfa269_c1*` set holds **197** files across 13
    # families of 15 plus 2 `retry_g5` members.  ALL 197 are UNTRACKED
    # (`git ls-files configs/` returns 0 of them) — a separate finding,
    # recorded in the catalogue, not a broken citation.
    "configs/wfa269_c1_dmpnn_pair_sweep_g5_Metal": "MISSING",
}

#: The recorded size of :data:`PATTERN_BASELINE`.  MEASURED 2026-10-31.
#:
#: RE-MEASURED 2026-11-01 (Wf-A175): **5** — the fifth is
#: ``molmetal/reports/wf131_p3_deferred_runs/per_seed_metrics_seed``, a
#: glob the catalog §6.266/§6.267 cites; the directory exists but holds
#: ``per_seed_results.json``, so the glob expands to nothing.
#:
#: RE-MEASURED 2026-11-02 (Wf-A269 wave-5): **6** — the sixth is
#: ``molmetal/scripts/launch_``, the catalogue's shorthand for the launcher
#: family; ``molmetal/scripts/`` holds zero ``launch_*`` files on disk, so
#: the glob expands to nothing.  The superseded rendering (5) is preserved
#: VERBATIM above per AGENTS.md §5.5 rule #6.
#:
#: RE-MEASURED 2026-11-04 (Wf-A269 wave-6): **7** — the seventh is
#: ``configs/wfa269_c1_dmpnn_pair_sweep_g5_Metal``, the literal PREFIX of the
#: catalogue's TEMPLATE citation
#: ``configs/wfa269_c1_dmpnn_pair_sweep_g5_{Metal}_seed{N}.yaml``.  The
#: template names a REAL 15-member family (5 metals x 3 seeds), all present
#: on disk; the prefix expands to nothing only because the guard's pattern
#: scan stops at the first ``{``.  The superseded rendering (6) is preserved
#: VERBATIM above per AGENTS.md §5.5 rule #6.
PATTERN_BASELINE_SIZE = 7

#: Citations that point at a **TRANSIENT, RUN-SCOPED** artefact — a file a
#: tool REWRITES on every run, so the path names a DIFFERENT file each time it
#: exists, and the artefact it originally described is gone for good.
#:
#: **WHY THIS IS NOT A :data:`BASELINE` ENTRY.**  The baseline IS the census of
#: currently-broken citations, and :func:`test_the_baseline_matches_the_recorded_measurement`
#: compares it for **EQUALITY** against the failing set.  A transient path is
#: broken SOMETIMES and resolved OTHERS, so a baseline entry for it would make
#: the guard flap: the census would change with an unrelated concurrent run.
#: MEASURED 2026-11-04: the file below was PRESENT at 01:12 and GONE at 01:16
#: on the same day, while a pre-flight run was writing it — the flapping is
#: real, not hypothetical.
#:
#: **WHY IT IS NOT SILENT.**  Each entry carries four assertions (see
#: :func:`test_every_superseded_pointer_is_adjudicated`): the citation is
#: still CITED by an SSOT document; the file is NOT TRACKED anywhere (it is
#: run-scoped, not a repository artefact); its parent directory EXISTS; and the
#: SSOT corpus carries a dated ``SUPERSEDED-POINTER`` marker for it — so the
#: exemption is tied to a VISIBLE correction, and deleting that correction
#: fails this guard.
SUPERSEDED_POINTERS: dict[str, str] = {
    "molmetal/reports/wf146_chemprop_lomo/pre_flight.json": (
        "transient pre-flight report — rewritten by every pre-flight run of "
        "scripts/launch_wf146_deferred_runs.sh; the 22:27 observation it "
        "described is superseded by the dated catalogue correction"
    ),
}

#: The recorded size of :data:`SUPERSEDED_POINTERS`.  MEASURED 2026-11-04: **1**.
SUPERSEDED_POINTERS_SIZE = 1


# ------------------------------------------------------------------ helpers
def _documents_text() -> dict[str, str]:
    return {rel: (REPO / rel).read_text(encoding="utf-8") for rel in DOCUMENTS}


def _scan() -> dict[str, tuple[str, str]]:
    """``{token: (bucket, evidence)}`` over the live SSOT documents."""
    return G.scan(REPO, DOCUMENTS)


def _failing(res: dict[str, tuple[str, str]]) -> set[str]:
    """The broken citations, minus the adjudicated transient pointers.

    :data:`SUPERSEDED_POINTERS` are subtracted HERE rather than being listed as
    :data:`BASELINE` entries.  The reason is structural, not cosmetic: the
    baseline IS the census of currently-broken citations and it is compared for
    EQUALITY against this set (:func:`test_the_baseline_matches_the_recorded_measurement`).
    A transient path is broken SOMETIMES and resolved OTHERS, so a baseline
    entry for it would make the census — and therefore the guard — flap
    whenever an unrelated run rewrites the file.

    The exemption is not a hole: :func:`test_every_superseded_pointer_is_adjudicated`
    asserts, for each pointer, that it is still cited, that it is NOT a
    tracked repository artefact, that its parent directory survives, and that
    the SSOT corpus carries a dated ``SUPERSEDED-POINTER`` marker for it.
    """
    return {
        tok
        for tok, (bucket, _) in res.items()
        if bucket in G.FAILING_BUCKETS and tok not in SUPERSEDED_POINTERS
    }


def _citation_sites() -> dict[str, list[str]]:
    """``{token: [document, ...]}`` — where each token is cited."""
    out: dict[str, list[str]] = {}
    for rel, text in _documents_text().items():
        for tok in G.extract_tokens(text, REPO):
            out.setdefault(tok, []).append(rel)
    return out


# -------------------------------------------------------------------- tests
def test_no_new_broken_repo_path_reference_was_introduced() -> None:
    """Every repo-relative path the SSOT docs cite must resolve, or be baseline.

    This is the load-bearing assertion: a NEW broken citation fails.
    """
    failing = _failing(_scan())
    new = sorted(failing - set(BASELINE))
    assert not new, (
        "these repo-relative paths do not resolve to anything on disk and "
        "are NOT part of the audited §6.242 baseline — a document now points "
        f"at nothing (AGENTS.md §5.5 filesystem-truth): {new}"
    )


def test_the_baseline_matches_the_recorded_measurement() -> None:
    """The baseline is a live measurement, not a historical record.

    Unlike a constant-to-constant comparison (which is a tautology), this
    asserts BOTH directions against the corpus:

    * a baseline entry that is no longer broken means the corpus was
      repaired — the baseline must be tightened here, or the exemption has
      silently become a hole;
    * the constant must equal the size of the recorded baseline.
    """
    assert len(BASELINE) == BASELINE_SIZE, (
        f"the audited baseline holds {len(BASELINE)} paths but the recorded "
        f"measurement is {BASELINE_SIZE} — the baseline was edited without "
        "updating the entry."
    )
    failing = _failing(_scan())
    assert len(failing) == BASELINE_SIZE, (
        f"the corpus now holds {len(failing)} broken citations, but the "
        f"recorded §6.242 adjudication is {BASELINE_SIZE} — the baseline has "
        f"drifted.  Offending paths: {sorted(failing - set(BASELINE))}"
    )
    stale = sorted(set(BASELINE) - failing)
    assert not stale, (
        f"these baseline entries are no longer broken — the corpus was "
        f"repaired and the baseline must be tightened: {stale}"
    )


def test_every_baseline_entry_is_load_bearing() -> None:
    """A baseline entry that is no longer CITED is dead code.

    The baseline must describe the corpus as it is.  An entry whose citation
    disappeared (the sentence was rewritten) must be removed, otherwise the
    exemption outlives the thing it exempts.
    """
    sites = _citation_sites()
    for path in sorted(BASELINE):
        assert path in sites, (
            f"baseline entry {path} is no longer cited by any SSOT document — "
            "the exemption is dead code and must be removed from the baseline"
        )


def test_every_baseline_entry_is_adjudicated() -> None:
    """Each baseline entry's adjudication must be TRUE against the filesystem.

    This is the per-entry assertion the baseline needs: it is not enough that
    the path is listed — the KIND recorded for it must still hold.

    * ``ROOT_PREFIX -> <real file>`` — the named file must exist, so the
      repair is a redirection to something real;
    * ``MISSING`` — no file with that basename may exist anywhere in the
      repository, so the citation really does point at nothing;
    * ``METRIC_NAMESPACE`` — the registry must declare no such artifact, so
      the namespace really does disagree.
    """
    resolver = G._resolver(REPO)
    for path, note in sorted(BASELINE.items()):
        kind, _, target = note.partition(" ")
        if kind == "ROOT_PREFIX":
            assert target.startswith("-> "), (
                f"baseline entry {path} is annotated ROOT_PREFIX but names no "
                f"target: {note!r}"
            )
            real = target[3:]
            assert (REPO / real).exists(), (
                f"baseline entry {path} is annotated ROOT_PREFIX -> {real}, "
                "but that file does not exist — the repair target is not real"
            )
        elif kind == "MISSING":
            base = path.rsplit("/", 1)[-1]
            assert not resolver.by_base.get(base), (
                f"baseline entry {path} is annotated MISSING, but a file named "
                f"{base} exists at {resolver.by_base.get(base)} — the "
                "adjudication is stale"
            )
        elif kind == "METRIC_NAMESPACE":
            base = path.rsplit("/", 1)[-1]
            assert base not in resolver.src_by_base, (
                f"baseline entry {path} is annotated METRIC_NAMESPACE, but the "
                f"registry now declares {base} — the adjudication is stale"
            )
            stem = base.rsplit(".", 1)[0]
            assert stem not in resolver.legacy, (
                f"baseline entry {path} is annotated METRIC_NAMESPACE, but the "
                f"registry now declares the short-form id {stem} — the "
                "adjudication is stale"
            )
        else:
            raise AssertionError(
                f"baseline entry {path} has an unknown adjudication {note!r}"
            )


def test_every_superseded_pointer_is_adjudicated() -> None:
    """Each transient pointer's exemption must be TRUE against the corpus.

    :data:`SUPERSEDED_POINTERS` is the ONLY exemption this guard allows that
    is not a :data:`BASELINE` entry, so it carries its own four assertions —
    without them it would be an unaudited hole in a guard whose whole subject
    is unaudited exemptions.

    1. **still cited** — the pointer must still be referenced by an SSOT
       document; a pointer nothing cites is dead code.
    2. **NOT a tracked artefact** — the path must not be in the git index.
       ``resolver.by_base`` is built from the index, so this asserts the
       absence of any TRACKED file with that basename.  A transient pointer
       that gets COMMITTED stops being transient, and this assertion fires.
    3. **parent directory survives** — the run-scoped directory the pointer
       lives in must still exist; a deleted directory means the citation
       should be re-pointed, not exempted.
    4. **a dated marker exists** — the SSOT corpus must carry a line that
       names the path, carries the ``SUPERSEDED-POINTER`` marker, and carries
       a measurement date.  This is what ties the exemption to a VISIBLE
       correction: deleting the correction breaks this guard.
    """
    resolver = G._resolver(REPO)
    sites = _citation_sites()
    corpus = _documents_text()
    for path, why in sorted(SUPERSEDED_POINTERS.items()):
        assert why, f"superseded pointer {path} carries no reason"
        assert path in sites, (
            f"superseded pointer {path} is no longer cited by any SSOT "
            "document — the exemption is dead code and must be removed"
        )
        base = path.rsplit("/", 1)[-1]
        assert not resolver.by_base.get(base), (
            f"superseded pointer {path} names {base}, and a TRACKED file with "
            f"that basename now exists at {resolver.by_base.get(base)} — the "
            "artefact stopped being transient and the exemption must be "
            "re-adjudicated"
        )
        assert (REPO / path).parent.is_dir(), (
            f"superseded pointer {path} lives in a directory that no longer "
            "exists — the citation should be re-pointed, not exempted"
        )
        marker_lines = [
            (rel, line)
            for rel, text in corpus.items()
            for line in text.splitlines()
            if "SUPERSEDED-POINTER" in line and path in line
        ]
        assert marker_lines, (
            f"superseded pointer {path} has no dated SUPERSEDED-POINTER "
            "marker in the SSOT corpus — the exemption must be tied to a "
            "visible correction (AGENTS.md §5.5 rule #6)"
        )
        dated = [
            (rel, line)
            for rel, line in marker_lines
            if re.search(r"\b20\d{2}-\d{2}-\d{2}\b", line)
        ]
        assert dated, (
            f"superseded pointer {path} has a SUPERSEDED-POINTER marker but "
            f"no measurement date on it: {marker_lines!r}"
        )


def test_every_short_form_metric_id_resolves_to_a_live_registry_artifact() -> None:
    """The named short-form exemption, with its own assertion.

    ``metrics/by_metric/m1.json`` is a metric ID, not a file path.  The
    exemption is earned by a resolution against ``registry/metrics.yaml`` —
    the metric SSOT — not granted by a string shape.  For each named id:

    * the registry must declare it;
    * its canonical artifact must be on disk;
    * the corpus must still cite it (the exemption is load-bearing);
    * the token must not be failing.
    """
    resolver = G._resolver(REPO)
    failing = _failing(_scan())
    sites = _citation_sites()
    for mid in SHORT_FORM_METRIC_IDS:
        assert mid in resolver.legacy, (
            f"short-form metric id {mid} is no longer declared by the registry "
            f"— the exemption must be re-adjudicated.  Declared ids: "
            f"{sorted(resolver.legacy)}"
        )
        canonical = resolver.legacy[mid]
        assert (REPO / canonical).is_file(), (
            f"short-form metric id {mid} resolves to {canonical}, which is not "
            "on disk — the exemption would be hiding a lost artifact"
        )
        token = f"metrics/by_metric/{mid}.json"
        assert token not in failing, (
            f"{token} is a short-form metric id the registry resolves to "
            f"{canonical}; it must not be reported as a broken pointer"
        )
        assert token in sites, (
            f"short-form metric id {mid} is no longer cited by any SSOT "
            "document — the exemption is dead code"
        )


def test_a_metric_path_that_contradicts_the_registry_is_broken() -> None:
    """The SAME namespace fires when the registry has no such metric.

    This is the falsifiable half of the metric rule: ``m1.json`` and ``m22.json``
    have exactly the same shape, and only the registry can tell them apart.
    """
    failing = _failing(_scan())
    for token in ("metrics/by_metric/m22.json", "metrics/by_metric/m25.json"):
        assert token in failing, (
            f"{token} is not a declared metric anywhere; the citation is a "
            "broken pointer and must be reported"
        )


def test_a_bare_metric_filename_is_not_a_path_token() -> None:
    """A filename with no directory is a NAME, not a path — never a token."""
    toks = G.extract_tokens(
        "the metric is vina_min.json, see metrics/by_metric/vina_min.json",
        REPO,
    )
    assert "vina_min.json" not in toks
    assert "metrics/by_metric/vina_min.json" in toks


def test_a_template_path_is_not_a_path_token() -> None:
    """``<spot>`` is a template placeholder, not a path."""
    toks = G.extract_tokens(
        "traces to `metrics/by_metric/bond_f1_<spot>_<seed>.json`", REPO
    )
    assert not [t for t in toks if t.startswith("metrics/by_metric/bond_f1")]


def test_the_dot_data_prefix_is_not_split() -> None:
    """``.data/molmetal/x.csv`` must be one token, never ``data/molmetal/x.csv``."""
    toks = G.extract_tokens("run `wc -l .data/molmetal/platinai_4cell.csv`", REPO)
    assert ".data/molmetal/platinai_4cell.csv" in toks
    assert "data/molmetal/platinai_4cell.csv" not in toks


def test_a_package_relative_citation_resolves() -> None:
    """``models/bond_head.py`` is a package-relative spelling — it resolves.

    Dropping a package root is a legitimate spelling; ADDING a spurious root
    (``molmetal/scripts/launch_wf299_cfm_main.sh``) is a wrong pointer.  Both
    directions are decided by the filesystem, not by a convention.
    """
    bucket, ev = G.classify("models/bond_head.py", REPO)
    assert bucket == "RESOLVED_PACKAGE_RELATIVE", f"{bucket} / {ev}"
    assert ev == "molmetal/models/bond_head.py"


def test_a_path_at_the_end_of_a_sentence_is_tokenized_and_audited() -> None:
    """A citation immediately followed by a sentence period must STILL be a token.

    REGRESSION (Wf-A269 wave-32, 2026-11-07).  The token regex's lookahead
    excluded ``.`` as well as word/path characters, so a path at the END OF A
    SENTENCE (``see docs/x.md.``) produced **NO TOKEN AT ALL**: the greedy
    character class swallowed the sentence period, the backtrack that would
    have yielded ``docs/x.md`` was rejected by the lookahead, and the whole
    match FAILED.  A broken pointer in that position was therefore **INVISIBLE**
    to this guard — not "extracted then rejected", but never tokenised at all.

    The corpus exposure of that shape was MEASURED (catalogue §6.269.42): one
    path+period shape across the four SSOT documents, and that one was already
    visible because a backtick terminated the token.  ⇒ the defect was
    **LATENT**, not firing.  A latent blind shape is still a blind shape, and
    it must be pinned — otherwise the next refactor re-arms it silently.

    The token must also stay EXACT: the class is greedy, so it consumes every
    ``.`` it can, and the final character class forces the sentence period
    back out — the match ends BEFORE the period, never inside it.
    """
    good = "molmetal/scripts/pre_flight_check.py"
    toks = G.extract_tokens(f"See {good}.", REPO)
    assert good in toks, f"the sentence-final path was not tokenised: {toks}"
    assert f"{good}." not in toks, "the sentence period leaked into the token"
    assert "molmetal/scripts/pre_flight_check" not in toks, (
        "the token was TRUNCATED to the stem — that is a DIFFERENT file"
    )

    # The falsifiable half: a BROKEN pointer in the same position must be
    # reported, not silently dropped.
    bad = "docs/definitely_not_a_real_file_zzz.md"
    res = G.scan_texts(REPO, {"probe": f"See {bad}."})
    assert bad in res, f"a broken pointer at the end of a sentence is INVISIBLE: {res}"
    bucket = res[bad][0]
    assert bucket in G.FAILING_BUCKETS, (
        f"the broken pointer was classified {bucket!r}, which is not a failing "
        f"bucket — the guard would not block on it"
    )


def test_the_wfa127_family_is_delegated_not_duplicated() -> None:
    """The ``TODO/(completed|pending|archive)`` family belongs to Wf-A127.

    Delegated, not duplicated: this guard must never re-implement the other
    guard's family, or the two will disagree about the same citation.
    """
    toks = G.extract_tokens(
        "see TODO/completed/999_zzz.md and TODO/completed/234_...md", REPO
    )
    assert not [t for t in toks if t.startswith("TODO/completed/")]


def test_the_guard_does_not_depend_on_the_ellipsis_convention() -> None:
    """Outside the delegated family, ``...`` is NOT an escape hatch."""
    toks = G.extract_tokens("see metrics/by_metric/m9_...json", REPO)
    assert "metrics/by_metric/m9_...json" in toks


def test_the_guard_is_not_vacuous() -> None:
    """The guard must scan a real corpus and find real breakages."""
    res = _scan()
    assert len(res) >= 400, f"only {len(res)} path tokens found — not scanning"
    assert (
        len(_failing(res)) >= 20
    ), "the corpus of broken pointers collapsed to nothing"


def test_no_broken_pattern_was_introduced() -> None:
    """A glob / template citation must expand to something.

    A pattern is not a pointer to one file, so it cannot be "resolved" the
    way a path can — but a pattern that expands to NOTHING is still a dead
    end, and must be reported.
    """
    res = G.scan_patterns(REPO, _documents_text())
    broken = {k for k, (b, _) in res.items() if b == "BROKEN_PATTERN"}
    new = sorted(broken - set(PATTERN_BASELINE))
    assert not new, (
        "these glob / template citations expand to nothing and are NOT part "
        f"of the audited baseline: {new}"
    )


def test_the_pattern_baseline_matches_the_recorded_measurement() -> None:
    """The pattern baseline is a live measurement in both directions."""
    assert len(PATTERN_BASELINE) == PATTERN_BASELINE_SIZE, (
        f"the pattern baseline holds {len(PATTERN_BASELINE)} entries but the "
        f"recorded measurement is {PATTERN_BASELINE_SIZE}"
    )
    res = G.scan_patterns(REPO, _documents_text())
    broken = {k for k, (b, _) in res.items() if b == "BROKEN_PATTERN"}
    assert len(broken) == PATTERN_BASELINE_SIZE, (
        f"the corpus now holds {len(broken)} broken glob citations, but the "
        f"recorded measurement is {PATTERN_BASELINE_SIZE}: "
        f"{sorted(broken ^ set(PATTERN_BASELINE))}"
    )


# --------------------------------------------------------------------------
# NEGATIVE TESTS — the guard must be able to FAIL.
# --------------------------------------------------------------------------


def test_a_perturbed_document_copy_adds_exactly_one_new_failure() -> None:
    """NEGATIVE TEST: perturbing a COPY of a document makes the guard fire.

    The perturbation is applied to an in-memory copy and scanned against the
    REAL repository, so the repository is never written to.  The assertion is
    exact: the perturbed corpus fails on the baseline PLUS the injected path
    and nothing else — the guard detects the new breakage and does not
    invent or lose any.
    """
    rel = "TODO/README.md"
    texts = _documents_text()
    clean = _failing(G.scan_texts(REPO, texts))
    assert clean == set(BASELINE), (
        f"the unperturbed corpus does not fail on exactly the baseline — the "
        f"control is wrong: {sorted(clean ^ set(BASELINE))}"
    )
    texts[rel] = texts[rel] + (
        "\n\nSee `scripts/no_such_launcher.sh` for the launch recipe.\n"
    )
    perturbed = _failing(G.scan_texts(REPO, texts))
    assert perturbed == set(BASELINE) | {"scripts/no_such_launcher.sh"}, (
        "the perturbed corpus must fail on exactly the baseline plus the "
        f"injected path; got {sorted(perturbed ^ (set(BASELINE) | {'scripts/no_such_launcher.sh'}))}"
    )


def test_a_new_broken_pointer_would_fail(tmp_path: Path) -> None:
    """NEGATIVE TEST: the failing path is reachable on a purpose-built repo.

    A synthetic mini-repository: ``scripts/real.py`` resolves, ``scripts/gone.py``
    does not, and ``metrics/by_metric/m1.json`` resolves through the registry.
    """
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    (repo / "scripts").mkdir()
    (repo / "metrics/by_metric").mkdir(parents=True)
    (repo / "registry").mkdir()
    (repo / "metrics/by_metric/ru.json").write_text("{}", encoding="utf-8")
    (repo / "registry/metrics.yaml").write_text(
        "schema: x\nmetrics:\n  m1_ru_auc:\n"
        "    source_json: metrics/by_metric/ru.json\n",
        encoding="utf-8",
    )
    (repo / "scripts/real.py").write_text("x = 1\n", encoding="utf-8")
    doc = repo / "docs/D.md"
    doc.write_text(
        "ok: `scripts/real.py`\n"
        "broken: `scripts/gone.py`\n"
        "metric id: `metrics/by_metric/m1.json`\n",
        encoding="utf-8",
    )
    res = G.scan(repo, ("docs/D.md",))
    assert res["scripts/real.py"][0] == "RESOLVED_DIRECT"
    assert res["scripts/gone.py"][0] == "DANGLING"
    assert res["metrics/by_metric/m1.json"][0] == "RESOLVED_METRIC_ID"
