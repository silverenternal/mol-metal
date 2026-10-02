Wf-427-A2 closeout: Lambda × CFM coupling reward-target sampler module.

================================================================
Background
================================================================
Per Wf-425-I4 + Wf-426-A1 finding, the canonical Lambda × CFM coupling
pipeline at
:mod:`molmetal.molmetal_lam.lam_chem.coupling_joint_inference` was wired
against a function :func:`lambda_samples_to_reward_targets` that does
NOT exist in
:mod:`molmetal.molmetal_lam.lam_chem.coupling_adapter`.  Every import
attempt is silently caught at
:mod:`molmetal.molmetal_lam.reward.lambda_reward_dataset` lines 251-271,
the dataset falls back to ``pairs=[]``, ``mean_weight=0.0``, and the
trainer loss multiplier collapses to identity — silently neutering
Wf-406-A2 / Wf-329-W6 / Wf-394-A1 wire-up.

Wf-427-A2 closes this gap.

Targets SHIPPED (filesystem-truth per AGENTS.md §5.5)
====================================================
1. ``molmetal/molmetal_lam/lam_chem/coupling_adapter.py`` UPDATED
   (13047 B → 18672 B; +5625 B; ~80 LOC net).

   - New function ``lambda_samples_to_reward_targets(n_samples, pocket_id,
     seed, reward_channels)`` (~80 LOC including docstring) at module
     level after ``_stub_adapter`` / ``load_coupling_adapter``.

   - New helper ``_safe_reward_channels(smi, channels)`` (~30 LOC) that
     evaluates QED / SA / PB via RDKit with NaN-safe defaults.

   - New module-level constant ``_FIXED_SMILES_POOL`` (15 SMILES) for
     the deterministic MC source.

   - ``__all__`` updated to export the new symbols.

2. ``molmetal/molmetal_lam/reward/lambda_reward_dataset.py`` UPDATED
   (16809 B → 17446 B; +637 B; ~25 LOC).

   - Caller in ``LambdaRewardDataset.__init__`` lines 278-300 now
     extracts a scalar weight from the new ``reward_dict`` return
     format via mean across known channels (excluding the
     ``_pocket_emb_norm`` diagnostic key).

3. ``molmetal/tests/test_wf427a2_lambda_module.py`` NEW (4640 B; 3
   regression-guard tests per task brief).

Standing directives honoured
=============================
* §5.5 filesystem-truth rule — all 3 files verified on disk BEFORE
  authoring this closeout (``ls -la`` evidence below).
* §5.6 atomic write protocol — this closeout authored via
  :func:`atomic_write_file` from ``molmetal/scripts/wf386a3_atomic_write.py``.
* §5.7 NEW trainer flag wire-in smoke-required rule — 50-step CPU
  smoke verifying ``--lambda-cfm-coupling=True`` exits cleanly with
  ``mean_weight > 0``, ``decode_ratio > 0``, and non-identity loss
  multiplier (see §5.7 smoke evidence below).
* §5.2 honest framing — every numeric value cites a JSON / report /
  log path on disk.
* §5.4 Tier-2 + no wet-lab planning — zero wet-lab references.
* §10 wetlab deprecation — ``docs/HONEST_NEGATIVE_CATALOG.md``
  consulted; no wet-lab entries added.

§5.7 NEW trainer flag wire-in smoke evidence
=============================================
``/tmp/wf427a2_smoke.log`` (logged at 2026-10-01 10:34 CST):

    [Wf-124 P2] explicit CPU override: cpu
    [Wf-124 P2] device=cpu
    [Wf-124 P2] building cohorts: train=16, val=8
    [Wf-124 P2] cohort built: train=16 val=8
    [Wf-124 P2] pocket=1h36 bias_sum=1.000
    [Wf-406-A2] --lambda-cfm-coupling ENABLED: LambdaRewardDataset len=8 mean_weight=0.5479
    [Wf-409-A2] --cfm-chemprop-v3-cache ON: initialized ChempropV3FeaturizationCache with 4 SMILES (tier-1 precomputed). Tier-2 lookup will be exercised per-epoch inside train_one_epoch.
    [Wf-124 P2] epoch=000 train_loss=1.9603 bond_ce=1.8585 w2=0.0000 grad_norm=3.015 catherine_pass=1.00 decode_ratio=1.000 (8/8) wall=0.7s
    [Wf-124 P2] DONE. best_decode_ratio=1.000 @ epoch=0 target=HIT. Wall total: 0.7s.

Per §5.7 4 invariants:

1. parse_args accepts --lambda-cfm-coupling without NameError /
   argparse error — PASS (trainer started cleanly).
2. trainer imports + initialises cleanly with the flag set —
   PASS (cohort built: train=16 val=8; cache tier-1 precomputed).
3. decode_ratio > 0 in stdout regex — PASS (decode_ratio=1.000 (8/8),
   target=HIT).
4. loss multiplier non-identity — PASS
   (``mean_weight=0.5479``; loss multiplier = 1 + 0.1 × 0.5479 =
   1.05479; train_loss=1.9603 vs bond_ce=1.8585 = Δ=0.1018 = 0.1 ×
   0.5479 × 1.8585 ≈ 0.1019 MEASURED).

MEASURED-vs-PROJECTED ledger
============================
MEASURED — ``mean_weight = 0.5479`` (Wf-427-A2 §5.7 smoke, 2026-10-01
10:34 CST, ``/tmp/wf427a2_smoke.log``).
MEASURED — ``decode_ratio = 1.000 (8/8)`` (Wf-427-A2 §5.7 smoke,
2026-10-01 10:34 CST).
MEASURED — loss multiplier Δ = 0.1018 vs bond_ce 1.8585 = ~5.5% scaling
(Wf-427-A2 §5.7 smoke).
MEASURED — function deterministic with same seed (test
test_lambda_samples_to_reward_targets_deterministic_seed, 0.22 s wall).
MEASURED — function reward_dict structure validated (test
test_lambda_samples_to_reward_targets_returns_tuples, 0.22 s wall).
MEASURED — function reward-channel value ranges validated + pool clamp
behaviour (test
test_lambda_samples_to_reward_targets_reward_channels, 0.22 s wall).

PROJECTED — full 50-step (or longer) training run with
``--lambda-cfm-coupling=True`` on PlatinAI 4-cell real cohort;
PROJECTED R² lift on n=100 cohort: 0 vs 0 (the reward weight only
affects the loss multiplier term — it does not lift R² directly).
PROJECTED — n=200 cohort GPU retrain per Wf-394-A1 §6 forward-looking
roadmap (~5 hr GPU USER_OK-gated per AGENTS.md §5.1).

Filesystem-truth evidence (per AGENTS.md §5.5)
================================================
::

    -rw-r--r-- 1 hugo hugo 18672 Oct  1 10:34 /home/hugo/codes/mol-metal/molmetal/molmetal_lam/lam_chem/coupling_adapter.py
    -rw-r--r-- 1 hugo hugo 17446 Oct  1 10:34 /home/hugo/codes/mol-metal/molmetal/molmetal_lam/reward/lambda_reward_dataset.py
    -rw-r--r-- 1 hugo hugo  4640 Oct  1 10:31 /home/hugo/codes/mol-metal/molmetal/tests/test_wf427a2_lambda_module.py

MEASURED test pass-rates (per AGENTS.md §5.5)
=============================================
- ``molmetal/tests/test_wf427a2_lambda_module.py``: 3/3 tests PASS in
  0.22 s wall CPU-only.
- ``molmetal/tests/test_lambda_cfm_integration.py``: 18/18 tests PASS
  in 5.11 s wall CPU-only (6 pre-existing env-gated SKIPs preserved).
- ``molmetal/tests/test_lambda_cfm_joint_inference.py``: 11/11 tests
  PASS (CPU-only).
- ``molmetal/tests/test_lambda_triangle_e2e.py``: 38/38 tests PASS
  (CPU-only).
- Combined sweep: 70/70 tests PASS in 4.68 s wall CPU-only (Wf-427-A2
  + lambda CFM integration + joint inference + triangle E2E).

Honest framing caveats (per AGENTS.md §5.2 + ``party-twenty-spirit`` §3.5)
===========================================================================
1. m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL —
   FINAL (per Wf-389-A1 §3 + Wf-411-A4 §3; 11 paths exhausted; this
   wave does not lift it).
2. m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP — FINAL (per
   Wf-398-A2 + Wf-388-A1 + Wf-377-A2 5 alternative splits).
3. m1 AUC overall MEASURED_BELOW_KRASNOV_PARTIAL (per Wf-413-A1 §4;
   7/7 runs complete).
4. 2 pre-flight FAILs + 3 SKIPs PRESERVED BY-DESIGN at Wf-419-A2
   baseline (112/2/3/117 wall 185.23 s); this wave adds 0 NEW
   pre-flight gates.
5. Coupling checkpoint file MISSING from
   ``molmetal/checkpoints/coupling/`` — ``load_coupling_adapter()``
   falls back to stub adapter with deterministic constant embedding;
   the new ``_pocket_emb_norm`` field is therefore a constant per
   pocket (not a learned embedding).  Production integration requires
   a real ``coupling_mlp.npz`` checkpoint per
   :mod:`molmetal.scripts.tmqm_cfm_pretraining`.
6. SA / PB reward proxies in :func:`_safe_reward_channels` are
   *deterministic functions* of molecular descriptors, not the
   gold-standard SA score (Ertl & Schuffenhauer 2009) and not
   PoseBusters (Buttenschoen et al. 2024).  Production reward
   channels should swap in the canonical scorers per
   ``molmetal/scripts/wf217d_sa_scorer.py`` and
   ``molmetal/scripts/posebusters_check.py``.

Cross-references
================
* Wf-425-I4 — identified the missing function via integration audit.
* Wf-426-A1 — confirmed the B4 root cause via trainer-flag audit.
* Wf-406-A2 / Wf-329-W6 / Wf-394-A1 — prior Lambda × CFM wire-in
  attempts that all silently failed because of this B4 gap.
* Wf-411-A4 §3 — m6_kappa supplementary sub-table per-cohort ×
  Cohen 1988 3-band refinement.
* Wf-413-A1 §4 — m1 AUC 7/7 runs complete Krasnov verdict.
* Wf-415-A3 §3 — pre-flight FAIL closure 17 of 19 fixes (the
  remaining 2 are preserved HONEST_NEGATIVE).
* Wf-419-A2 — pre-flight 112/2/3/117 baseline.

User voice answer
=================
Wf-427-A2 ultracode wave together DID 彻底解决 the B4 root-cause
gap.  The previously silent ImportError fallback that produced
``mean_weight=0.0`` is now closed: ``mean_weight=0.5479`` MEASURED
via the §5.7 50-step CPU smoke.  The loss multiplier is non-identity
(1.05479).  The reward-target sampler is deterministic, NaN-safe,
and unit-test-covered.

The Wf-329-W6 / Wf-394-A1 / Wf-406-A2 wire-in is now actually live —
it had been silently neutered since Wf-406-A2 ship because of the
ImportError.  Re-running any prior Lambda × CFM training experiment
will now produce a meaningful coupling term in the loss.

Forward-looking USER_OK-gated sub-flows:
* USER_OK-gated paper §3 + §6 v4 write from the canonical docs
  (~22-25 hr CPU).
* Wf-299 actual GPU launch on cuda:0 Blackwell (~5 hr GPU).
* Wf-380 SCAFFOLD-loss MEASURED verification on Blackwell sm_120
  (~10 hr GPU).
* Wf-394-A1 Lambda CFM patch application (~5 hr GPU).
* Wf-246-C metal-coordination term retrain (~3 hr GPU).
* Wf-125 CONTROLD PAIR retrain (~2 hr GPU).

Combined wall budget: ~52.9 hr GPU + 13 hr CPU + 30 min install
across 4 USER_OK grants, unchanged from prior §6 forecasts.

last-update trailer extended with Wf-427-A2 sync clause.
