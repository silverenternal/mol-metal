# Real CrossDocked CFG output-quality control

This is a negative generation-quality result, not completed molecular-quality or Vina validation. The matrix completed, but 4 / 12 cells failed during coordinate integration. All 96 planned/requested samples remain the denominator.

Eight distinct real 19-heavy-atom C/N/O/F training ligands were selected from the official training split, excluding canonical SMILES overlap with all available test ligands. One eligible pair had no receptor atoms within 12 Å of its ligand and was excluded with its identity/reason retained. The encoder uses the nearest 64 receptor atoms; positions are centered on the provided ligand center and scaled by a training-only statistic. Ligand-centred pocket extraction assumes a known binding site, not blind docking.

For each seed (42, 0, 1234), 200 updates fit one hidden16/layer1 checkpoint shared by CFG1 and CFG2. The same sample seed, decoding protocol, and two held-out manifest pairs (test_001, test_002) are used. Each of 12 cells requests 8 samples. This tiny-data control does not represent a fully trained model.

| Outcome | Count |
|---|---:|
| Requested | 96 |
| Finite raw outputs | 64 |
| Decoded connected, sanitized graphs | 0 |
| Docked | 0 |
| PoseBusters passes | 0 |
| disconnected_distance_graph | 54 |
| connectivity_or_valence_failure:AtomValenceException | 10 |
| FloatingPointError: Atom prediction produced non-finite probabilities | 32 |

No atom is deleted or substituted. Distance connectivity with single bonds and implicit hydrogens is explicitly a heuristic, not a learned bond-order model; valid graph decoding is only a necessary gate. Raw model coordinates and all failures are retained. No valid graph reached docking, so Vina and docking PoseBusters remain unmeasured. The graph decoder rejects incompatible atoms, disconnected graphs and invalid valences without repair.

The original run trained on true atomic-number inputs, integrated with masked zeros and predicted endpoint logits with all-one inputs. This allowed atom-label leakage and caused severe train/sample mismatch. The masked_atoms run instead inputs zeros in both training and unconstrained sampling, supervising true atomic numbers only with CE. No generation vocabulary mask is introduced. Old teacher-forced checkpoints require retraining.

In the original finite outputs, 58/64 molecules contain at least one unsupported atom. After masked-input retraining, all 1,216 atoms in 64 finite outputs are from C/N/O/F; however, 54 graphs are disconnected and 10 fail valence. The atom distribution bug is repaired but graph/coordinate generation is still not scientifically adequate. Seed0 remains nonfinite in both protocols.

Artifact hashes, train and test identities, individual losses, common checkpoint hashes, failures and raw coordinate paths are in [report.json](report.json). Planned versus attempted requested counts are separate for budget aborts. Historical setup errors are retained in the original directory.
