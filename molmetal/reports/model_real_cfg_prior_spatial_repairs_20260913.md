# Real-data model controls and production contract repairs

The model line now has executable real-data CFG controls, an active explicit-topology metal prior, masked atom prediction, and a versioned equivariant velocity architecture. It does **not** yet generate connected valid molecules in these controls and has no measured CFG or Pt-prior Vina advantage. This report preserves that distinction.

## Atom supervision and forward-flow contract

The old training step supervised atomic numbers with CE but simultaneously provided those labels as embedding inputs. The sampling ODE used all-zero inputs and the endpoint head all-one inputs (hydrogen, despite an incorrect carbon comment). That encouraged copying the labels and created train/sample mismatch. Training now uses masked Z=0 inputs, supervises actual atomic numbers only through CE, and uses the same identity input during unconstrained generation. No output-vocabulary mask was added. Old teacher-forced checkpoints need retraining.

Training and generation use identical complete directed graphs without self loops. Actual molecular bonds are unused in both paths: this avoids graph leakage but also exposes the missing bond-learning task. Gaussian noise and the CondOT forward direction match: x_t=(1−t)x_0+t x_1, target v=x_1−x_0, integration from 0 to 1. Real-data training and inference use the same train-only coordinate scale and ligand-centred pocket convention. A malformed training receptor/ligand pair with no receptor atom within 12 Å is rejected and logged rather than recentered silently.

A receptor sulfur atom exposed a vocabulary-bound error in the initial small harness; the encoder now raises a useful ValueError before a GPU embedding index fault. The real harness covers all receptor atomic numbers using vocabulary100. All setup failures remain recorded.

## Velocity architecture v2

An invariant hidden scalar vector cannot be projected to a physical 3D vector by an H-to-3 linear layer. The old trained/nonzero-head counterexample produced a rotation/translation velocity discrepancy of 0.069927 for velocities of magnitude 0.071410. It also allowed unbounded coordinate-dependent edge coefficients, creating a superlinear ODE outside the training support. Merely increasing Euler steps did not fix this: in the eight-pair seed42 checkpoint the maximum normalized coordinate grew 7.33 → 242 → nonfinite at 16/64/256 steps.

The v2 velocity is

    v_i = tanh(a(h_i)) (x_i − centroid(x))
          + [1/(N−1)] sum_j tanh(phi_ij) (x_j − x_i).

Scalar gates are invariant; the spatial bases rotate as vectors and are translation invariant. Each coefficient is bounded inside the learned architecture and messages are degree-normalized, giving at-most-linear spatial growth of the output. Generated coordinates are never clamped. Zero initialization keeps the initial flow zero, while the real relative-position bases give both readout and final message projection a nonzero first gradient. The H-to-1 readout is deliberately incompatible with old H-to-3 checkpoints; metadata identifies `LipmanFlowMatching_v2_equivariant` and the retraining requirement. The bounded-message option is enabled only for the Lipman velocity field; root MolFlow/QM9 behavior is unchanged.

CPU and ROCm tests check nonzero velocity rotation/translation, atom-logit invariance, coordinate-growth bound, first-step gradients, and actual `generate` equivariance under a common rigid transform of initial noise. The prior's fixed atomic identities are used only for the explicit prior and final identity constraint; they are not supplied as unseen features to the masked-trained velocity network.

## Controlled real CFG results

| Run | Train pairs / updates / seed | Requested | Finite outputs | Valid decoded graphs |
|---|---:|---:|---:|---:|
| Original teacher-forced atom head | 8 / 200 | 96 | 64 | 0 |
| Masked atom head | 8 / 200 | 96 | 64 | 0 |
| Larger training, old spatial readout | 32 / 2000 | 96 | 96 | 0 |
| Larger training, v2 spatial readout | 32 / 2000 | 96 | 96 | 0 |

All runs use real CrossDocked training SDF coordinates and official test-SMILES exclusion, exact test_001/test_002 pairs, seeds42/0/1234 and matched CFG1/CFG2 from a common checkpoint with identical sample seeds. The larger runs share hidden32, two layers, batch2, lr0.0001 and 64 Euler steps; comparing them with the smaller runs cannot isolate any one training-budget effect.

The masked eight-pair control removes unsupported atoms from its 64 finite outputs (1216/1216 C/N/O/F), whereas the original had 58/64 molecules containing at least one unsupported atom. Nevertheless, all graph decoding fails. The final v2 control has 92 disconnected graphs and 4 outputs containing a training-vocabulary-external atom. No output reaches docking, so Vina/PoseBusters effects are unmeasured, with all 96 requests retained as failures in success denominators. The decoder preserves every atom and uses unchanged distance connectivity, single bonds/implicit H and sanitization; it never deletes fragments or relaxes thresholds.

Independent v2 checkpoint resolution replays at 16/64/256 steps stay finite for every seed, with normalized maximum coordinate ranges 2.42–2.89, 2.54–3.06 and 2.57–3.10 respectively. This supports numerical stability in the measured cases, not molecular validity.

Artifacts, each with protocol, data/checkpoint/runtime hashes, raw outputs and README:

- `r10_cfg_real_crossdocked/`
- `r10_cfg_real_crossdocked_masked_atoms/`
- `r10_cfg_real_crossdocked_train32_2000/`
- `r10_cfg_real_crossdocked_v2_train32_2000/`
- `r10_cfg_integrator_diagnostic.json` (old module hooks observed endpoint logits only; correctly labelled, integration outcomes still valid)
- `r10_cfg_integrator_v2_diagnostic.json` (actual forward-call trajectories)
- `r10_equivariant_velocity_repair/` (before/after spatial counterexamples and archived pre-v2 source)

## Metal prior: explicit donors and actual sampler

`MetalGeometryPrior` previously ignored supplied edge codes' endpoints and treated every atom within 0.4–5 Å as a donor. It now requires an explicit `(2,E)` edge_index whenever dative edges exist, uses unique donor→metal endpoints and validates index alignment. The legacy square-planar routine now accepts both 90° cis and 180° trans targets. Both opt-in velocity-field loss helpers evaluate every batch graph, not just graph0.

The sampler previously passed placeholder atoms and an empty dative edge vector, so positive prior weight did nothing. It now accepts `GenerationConfig.conditioning.fixed_atom_types` and `.dative_edge_index` as explicit known-identity/topology constraints. Missing data leaves the unconstrained sampler `inactive`; invalid graphs raise. Per-graph energies are summed so gradients do not shrink with batch size, and `torch.enable_grad` works inside the ODE solver's no-grad context. Metadata records eligibility, energy evaluations and actual nonzero per-graph gradient updates. A real ROCm sampler test with two known graphs and 20 Euler steps records 40 nonzero graph updates, preserves every supplied atom identity and lowers each graph's actual prior energy. This is an untrained integration contract test, not a trained-generation result.

On eight real tmQM Pt CN4 full structures × three seeds, the primitive and repaired production energy are separately compared against identical noisy starts using DFT coordinates only to evaluate recovery. In the production control, mean donor-vector RMSE decreases 0.071322 Å (18/24 pairs), donor-angle MAE to DFT decreases 4.374783° (19/24), and full-coordinate RMSE decreases 0.001625 Å (20/24). Negative pairs remain. The angular function and distance sanity filters do not constitute a full coordination validator, and CN4 alone does not prove Pt(II). No receptor-paired Pt scoring setup is validated, so neither control establishes Vina improvement.

- `r10_tmqm_geometry_control/`: original explicit-edge primitive, full structures and all24 paired cases.
- `r10_tmqm_production_prior_control/`: repaired production energy function, same real-data protocol, separately hashed and reported.

## Validation and remaining model work

**83 passed, 1 existing timing-benchmark skipped** across 15 relevant test files, including spatial contracts, atom supervision/noise/graph contracts, real prior sampling, prior legacy tests, import isolation, batching, CFG, OT, ROCm sampling and tmQM shape bridge (19.96s). The earlier 30-update toy optimization check no longer met its unchanged 5% criterion after atom labels were masked (6.4553→6.5204); 100 updates now meet the same criterion. Its common fixed-noise evaluation uses masked inputs, and no threshold was relaxed. Historical diagnostic failure is retained here; this test is optimization plumbing, not molecular evidence.

Remaining scientific model work is substantive: jointly learn chemical topology with atom/coordinate generation on an adequate real training split; provide ligand–receptor equivariant directional interactions (the current receptor encoder globally pools invariant features and cannot specify pose orientation); validate complete generated molecular geometry and binding against matched actual baselines. Zero graph validity must not be marked as model-line completion. The Pt Vina requirement likewise remains open until real paired data and validated scoring parameters are available.
