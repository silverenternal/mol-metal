# Open Risks

**Status:** living doc
**Last updated:** 2026-09-13

## R1 — Cite-only SOTA comparison cannot tighten
DiffSBDD ckpt (Zenodo 8183747) + TargetDiff ckpt (Google Drive) +
torch_geometric + torch_scatter ROCm 7.2 wheels all unavailable from the
sandbox. `lambda_vs_sbdd_paper_numbers.md` carries 7 explicit
protocol-mismatch flags; reviewers will challenge any Lambda-Vina claim
until live apples-to-head exists. Mitigation: pending/07 (R4-C 100-pocket
sweep) gives Lambda's own measured column; keep cite-only columns for
DiffSBDD/Pocket2Mol/TargetDiff and flag every row.

## R2 — Vina versus QuickVina protocol parity remains to be measured
QuickVina 2 is installed as the vendored `qvina02` binary and verified against
the official release. Both CLI aliases resolve that same engine. The adapter
now parses real REMARK energies and propagates the configured seed. A paired
Vina/QuickVina N=50 experiment remains required; no direction or magnitude of
bias is claimed before measurement. See `quickvina2_binary_identity.md`.

## R3 — PoseBusters 0/13 pass on CuAAC products
Real negative result documented in `lambda_vs_sbdd_paper_numbers.md`.
Root cause: ETKDGv3+UFF embedding does not reach MMFF94 minimum for
1,2,3-triazoles. Until the engineering fix lands, PoseBusters validity
cannot enter the published metric table. Mitigation: pending/06
(MMFF94OptimizeMolecule swap in embed_3d path).

## R4 — metal_hybrid_v4_round3 stability re-measure BLOCKED
`metal_hybrid_v4_round3_stability.md` re-measurement was BLOCKED — bash
sandbox returned exit 1/120/134. Last successful round-3 run: test AUC
0.5731 ± 0.0685 (n=3), σ trajectory shown converging. Stability std
exceeds 0.06 gate; rounds-2 V4 mean regressed to 0.4857 below D-MPNN
baseline 0.5135. Re-run needed before claiming the +0.0596 V4-delta.

## R5 — REINVENT4 live score remains OBS
`reinvent` binary not on $PATH; subprocess launch fails FileNotFoundError;
`RewardAggregator` treats None as 0.0 → r_reinvent4 silently off;
multi-property reward currently runs on 6 channels, not 7.
Mitigation: pending/05 (install reinvent via pipx or docker).

## R6 — tmQM pretraining not yet wired into EGNN (resolved 2026-09-13)
The tmQM encoder is now fine-tuned on the Ru subset and integrated through
`load_tmQM_pretrained` / `use_tmqm_init` in
`molmetal/adapters/flow_matching_lipman/__init__.py`.  The production-shape
bridge loads the available `molmetal/checkpoints/egnn_tmqm_finetuned_ru.pt`
checkpoint, with a strict fallback to random initialization when a checkpoint
is unavailable.  Coverage is verified by
`test_tmqm_wireup.py` and `test_tmqm_shape_bridge.py` (11 tests passing).

## R7 — PDBbind v2020 download blocked
PDBbind v2020 ~1.6 GB refined set unreachable (RemoteDisconnected).
Falls back to CA2 + MMP13 (478 pairs) for now. Only 4 of 10
metalloprotein families have any CrossDocked2020 pairs. Phase-3
evaluation cannot extend to the full metalloproteome.
Mitigation: see `decisions.md` D3.

## R8 — pIC50 predictor Pearson r = 0.407
`h2_pic50_predictor_calibration.md`: Attentive D-MPNN pIC50 predictor
Pearson r = 0.407 on calibration set — below published D-MPNN envelope
(0.78-0.92 on full chem space). Insufficient for absolute IC50 claim;
ranking-only. Phase-2 CFG and reward aggregator using pIC50 should be
interpreted as ranking signal only.
