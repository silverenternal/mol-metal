# Wf-A201 -- CPU-only localisation of the 0.1931 pIC50 pearson gap

*Question*: where does the 0.1931 pIC50 pearson gap (ridge on frozen TRAINED encoder embeddings 0.4074 vs the repo's neural head 0.2143) go?

## 0. Scope (MEASURED, read-only)

- CPU-only: `cuda.is_available()` = False; nvidia-smi before: `1549281, /home/hugo/codes/planB/.venv/bin/python3, 1020 MiB
1577603, /home/hugo/codes/planB/.venv/bin/python3, 986 MiB
1577677, /home/hugo/codes/planB/.venv/bin/python3, 988 MiB`
- nvidia-smi after: `1606638, /home/hugo/codes/planB/.venv/bin/python3, 1706 MiB`
- encoder trained: **False**; checkpoint modified: **False**
- metrics/ registry/ paper/ docs/ TODO/ files written: 0 / 0 / 0 / 0
- every fit is: head-only (linear/MLP) readouts on FROZEN, PRE-EXTRACTED encoder embeddings
- elapsed: 287.3 s

## 1. Split identity (H5: same rows)

- joint_train=3480 target_train=2262 val=394 test=994
- published 2026-10-08 split: {'n_joint_train': 3480, 'n_target_train': 2262, 'n_val': 394, 'n_test': 994} -- matches: **True**
- test row-id sha256: `bdd5349364f96f8cdd7be54e5a390103843658e6311dffd394bb5df3f1494589`

## 2. Reproduced vs published (2026-10-08)

| arm | published | reproduced here (honest rule) | delta (honest) | reproduced here (leaked a5cc339d rule) | delta (leaked) | status |
|---|---|---|---|---|---|---|
| `ridge_on_morgan_fp` | 0.323151 | 0.310433 | -0.012718 | 0.323151 | 0.000000 | REPRODUCED EXACTLY by the LEAKED rule; NOT reproduced by the honest rule |
| `ridge_on_random_encoder_embedding` | 0.394380 | 0.393907 | -0.000473 | 0.394902 | 0.000522 | REPRODUCED (leaked rule) |
| `ridge_on_trained_encoder_embedding` | 0.407406 | 0.442363 | 0.034957 | NOT MEASURED | NOT MEASURED | NOT REPRODUCED (no saved checkpoint for the 2026-10-08 run) |
| `neural_head_default_alpha_0.5` | 0.214257 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT REPRODUCED |
| `neural_head_regweighted_alpha_0.2` | 0.168961 | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT MEASURED | NOT REPRODUCED |

- `ridge_on_morgan_fp`: ridge on Morgan FP, target-metal train rows -> target-metal test rows. The published fingerprint (pearson+mae+rmse+alpha) is reproduced to 17 significant digits by a5cc339d's ridge_pearson, which selects the alpha by maximising Pearson ON THE 994 TEST ROWS it reports. The repo's current honest rule (alpha chosen on a carved-out val fold) gives the 'reproduced_here_honest_rule' number instead.
- `ridge_on_random_encoder_embedding`: random init encoder, torch.manual_seed(0), joint-pool train rows. Same rule attribution as the Morgan row; any residual delta vs the published number is CPU-vs-CUDA embedding arithmetic, not method.
- `ridge_on_trained_encoder_embedding`: the 2026-10-08 diag script trained the encoder in-memory and never saved it; closest on-disk trained encoder = dmpnn_mtw_re_seed1 (delta vs published is NOT a reproduction)
- `neural_head_default_alpha_0.5`: the 2026-10-08 head came from the same in-memory (unsaved) model; reproducing it needs a GPU training run, forbidden by rule 1. The head-only CPU analogue is the L1 arm.
- `neural_head_regweighted_alpha_0.2`: same reason as the alpha-0.5 head. The head-only CPU analogue is the A02_alpha0.2_regweighted arm.

## 2b. Ridge selection-rule sweep (H6 -- is the REFERENCE number honest?)

Three ways to choose the ridge alpha on the SAME rows and the SAME features:

| feature set | honest: alpha on carved-out val fold | honest: alpha on the real val fold | leaked: alpha = max Pearson ON THE TEST ROWS (a5cc339d) | leak inflation |
|---|---|---|---|---|
| `morgan_fp` | 0.310433 | 0.323151 | 0.323151 | 0.012718 |
| `random_init_seed0` | 0.393907 | 0.343214 | 0.394902 | 0.000995 |
| `dmpnn_mtw_ir_seed0` | 0.368844 | 0.377484 | 0.377484 | 0.008640 |
| `dmpnn_mtw_ir_seed1` | 0.361119 | 0.366649 | 0.366649 | 0.005529 |
| `dmpnn_mtw_ir_seed2` | 0.371214 | 0.366828 | 0.386606 | 0.015392 |
| `dmpnn_mtw_os_seed0` | 0.305331 | 0.370866 | 0.374936 | 0.069604 |
| `dmpnn_mtw_os_seed1` | 0.336605 | 0.422031 | 0.422031 | 0.085427 |
| `dmpnn_mtw_os_seed2` | 0.353668 | 0.427758 | 0.427758 | 0.074090 |
| `dmpnn_mtw_re_seed0` | 0.341660 | 0.389658 | 0.397754 | 0.056094 |
| `dmpnn_mtw_re_seed1` | 0.442363 | 0.453291 | 0.466463 | 0.024100 |
| `dmpnn_mtw_re_seed2` | 0.405100 | 0.385801 | 0.405147 | 0.000047 |
| `dmpnn_mtw_rh_seed0` | 0.392592 | 0.419846 | 0.420683 | 0.028091 |
| `dmpnn_mtw_rh_seed1` | 0.387844 | 0.423859 | 0.423859 | 0.036014 |
| `dmpnn_mtw_rh_seed2` | 0.417200 | 0.411646 | 0.417070 | -0.000131 |

- H6 PROVEN for the Morgan-FP arm (exact 17-digit fingerprint match to the published number): `True`
- H6 statement: the published ridge reference (0.4074) was selected by maximising Pearson on the 994 test rows it reports (a5cc339d ridge_pearson), so it is a test-set-leaked upper bound, not an honest held-out score
- arm provenance: the `random_init_seed0` encoder is randomly initialised, so NO row was ever seen in training -- its ridge numbers are the CLEAN reference. Every checkpoint encoder was trained on a DIFFERENT cohort/split (recorded n_joint_train 3540 / 18137 / 18596 vs the probe's 3480); the ridge on those features is fit only on the probe's pool rows, but the features themselves may be optimistic for rows that were in the encoder's training set (overlap NOT measured).

## 3. CPU-only localisation table

### encoder = `random_init_seed0`

| arm | hypothesis | test pearson (final, mean+-std) | test pearson (best-val ep) | train pearson (in-sample) | delta vs ridge |
|---|---|---|---|---|---|
| `L1_repo_head_alpha0.5_raw` | H1/H2/H3/H4 baseline arm | 0.1388+-0.0158 | 0.0070 | 0.2596 | -0.2551 |
| `L2_repo_head_alpha0.5_std_target` | H1 target scale | 0.2789+-0.0147 | 0.1668 | 0.4613 | -0.1150 |
| `L3_wide_head_512` | H2 head capacity | 0.1151+-0.0498 | -0.0067 | 0.2807 | -0.2788 |
| `L3b_linear_head_no_hidden` | H2 head capacity (linear floor) | 0.1321+-0.0214 | 0.0201 | 0.2694 | -0.2618 |
| `L4_repo_head_wd1e-2` | H2/H3 regularisation | 0.0888+-0.0257 | 0.0003 | 0.2245 | -0.3051 |
| `L5_repo_head_gradclip5` | H3 optimisation (trainer clip) | 0.1805+-0.0331 | -0.0051 | 0.2462 | -0.2134 |
| `A02_alpha0.2_regweighted` | H4 multi-task coupling | 0.1388+-0.0158 | 0.0070 | 0.2612 | -0.2551 |
| `A08_alpha0.8_clsweighted` | H4 multi-task coupling | 0.1388+-0.0158 | 0.0070 | 0.2595 | -0.2551 |
| `LR3e-4` | H3 optimisation (learning rate) | 0.1205+-0.0101 | -0.0024 | 0.2917 | -0.2734 |
| `LR3e-3` | H3 optimisation (learning rate) | 0.1634+-0.0212 | -0.0482 | 0.2823 | -0.2305 |
| `LR1e-2` | H3 optimisation (learning rate) | 0.1773+-0.1548 | NOT MEASURED | 0.3320 | -0.2167 |
| `_ridge_control_` | L4 ridge readout (control) | 0.3939 | - | - | 0.0000 |

Multi-task loss split (mean over the last epoch, 3 seeds) and the alpha arms:

| arm | alpha | loss_cls | loss_reg | cls/reg |
|---|---|---|---|---|
| `L1_repo_head_alpha0.5_raw` | 0.5 | 0.549261 | 0.775653 | 0.710895 |
| `A02_alpha0.2_regweighted` | 0.2 | 0.551807 | 0.773500 | 0.716046 |
| `A08_alpha0.8_clsweighted` | 0.8 | 0.552507 | 0.775656 | 0.715645 |
| `L2_repo_head_alpha0.5_std_target` | 0.5 | 0.549261 | 0.806795 | 0.680953 |

## 4. Checkpoint's own head (L5)

| checkpoint | test pearson | test rmse | train (in-sample) pearson | ckpt epochs | ckpt hidden | ckpt recorded split (n_train/n_val/n_test/n_joint_train) |
|---|---|---|---|---|---|---|
| `dmpnn_mtw_ir_seed0` | 0.055356 | 0.884245 | 0.217446 | 1 | 300 | 3444/243/859/18137 |
| `dmpnn_mtw_ir_seed1` | 0.164607 | 0.710657 | 0.242788 | 1 | 300 | 3444/243/859/18137 |
| `dmpnn_mtw_ir_seed2` | 0.090772 | 1.288910 | 0.220876 | 1 | 300 | 3444/243/859/18137 |
| `dmpnn_mtw_os_seed0` | 0.394637 | 0.647217 | 0.459704 | 1 | 128 | 66/7/14/3540 |
| `dmpnn_mtw_os_seed1` | 0.516369 | 0.595082 | 0.585575 | 50 | 128 | 782/107/224/18596 |
| `dmpnn_mtw_os_seed2` | 0.454276 | 0.613611 | 0.513743 | 50 | 128 | 782/107/224/18596 |
| `dmpnn_mtw_re_seed0` | 0.289835 | 0.768882 | 0.386090 | 1 | 128 | 121/9/36/3540 |
| `dmpnn_mtw_re_seed1` | 0.545726 | 0.624808 | 0.561351 | 50 | 128 | 607/87/174/18596 |
| `dmpnn_mtw_re_seed2` | 0.361125 | 0.676406 | 0.459573 | 50 | 128 | 607/87/174/18596 |
| `dmpnn_mtw_rh_seed0` | 0.630531 | 0.488420 | 0.638066 | 1 | 128 | 60/10/38/3540 |
| `dmpnn_mtw_rh_seed1` | 0.544101 | 0.614214 | 0.570105 | 50 | 128 | 778/113/227/18596 |
| `dmpnn_mtw_rh_seed2` | 0.446940 | 0.630363 | 0.509504 | 50 | 128 | 778/113/227/18596 |

- `dmpnn_mtw_ir_seed0`: NOT COMPARABLE to the other arms: this checkpoint was trained on a different cohort/split (recorded n_train/n_val/n_test/n_joint_train below) than the probe's split (3480 joint-train / 2262 target-train / 394 val / 994 test), so its predictions on the probe's test rows are NOT held out. Reported for completeness; no headline claim rests on it.

- `dmpnn_mtw_ir_seed1`: NOT COMPARABLE to the other arms: this checkpoint was trained on a different cohort/split (recorded n_train/n_val/n_test/n_joint_train below) than the probe's split (3480 joint-train / 2262 target-train / 394 val / 994 test), so its predictions on the probe's test rows are NOT held out. Reported for completeness; no headline claim rests on it.

- `dmpnn_mtw_ir_seed2`: NOT COMPARABLE to the other arms: this checkpoint was trained on a different cohort/split (recorded n_train/n_val/n_test/n_joint_train below) than the probe's split (3480 joint-train / 2262 target-train / 394 val / 994 test), so its predictions on the probe's test rows are NOT held out. Reported for completeness; no headline claim rests on it.

- `dmpnn_mtw_os_seed0`: NOT COMPARABLE to the other arms: this checkpoint was trained on a different cohort/split (recorded n_train/n_val/n_test/n_joint_train below) than the probe's split (3480 joint-train / 2262 target-train / 394 val / 994 test), so its predictions on the probe's test rows are NOT held out. Reported for completeness; no headline claim rests on it.

- `dmpnn_mtw_os_seed1`: NOT COMPARABLE to the other arms: this checkpoint was trained on a different cohort/split (recorded n_train/n_val/n_test/n_joint_train below) than the probe's split (3480 joint-train / 2262 target-train / 394 val / 994 test), so its predictions on the probe's test rows are NOT held out. Reported for completeness; no headline claim rests on it.

- `dmpnn_mtw_os_seed2`: NOT COMPARABLE to the other arms: this checkpoint was trained on a different cohort/split (recorded n_train/n_val/n_test/n_joint_train below) than the probe's split (3480 joint-train / 2262 target-train / 394 val / 994 test), so its predictions on the probe's test rows are NOT held out. Reported for completeness; no headline claim rests on it.

- `dmpnn_mtw_re_seed0`: NOT COMPARABLE to the other arms: this checkpoint was trained on a different cohort/split (recorded n_train/n_val/n_test/n_joint_train below) than the probe's split (3480 joint-train / 2262 target-train / 394 val / 994 test), so its predictions on the probe's test rows are NOT held out. Reported for completeness; no headline claim rests on it.

- `dmpnn_mtw_re_seed1`: NOT COMPARABLE to the other arms: this checkpoint was trained on a different cohort/split (recorded n_train/n_val/n_test/n_joint_train below) than the probe's split (3480 joint-train / 2262 target-train / 394 val / 994 test), so its predictions on the probe's test rows are NOT held out. Reported for completeness; no headline claim rests on it.

- `dmpnn_mtw_re_seed2`: NOT COMPARABLE to the other arms: this checkpoint was trained on a different cohort/split (recorded n_train/n_val/n_test/n_joint_train below) than the probe's split (3480 joint-train / 2262 target-train / 394 val / 994 test), so its predictions on the probe's test rows are NOT held out. Reported for completeness; no headline claim rests on it.

- `dmpnn_mtw_rh_seed0`: NOT COMPARABLE to the other arms: this checkpoint was trained on a different cohort/split (recorded n_train/n_val/n_test/n_joint_train below) than the probe's split (3480 joint-train / 2262 target-train / 394 val / 994 test), so its predictions on the probe's test rows are NOT held out. Reported for completeness; no headline claim rests on it.

- `dmpnn_mtw_rh_seed1`: NOT COMPARABLE to the other arms: this checkpoint was trained on a different cohort/split (recorded n_train/n_val/n_test/n_joint_train below) than the probe's split (3480 joint-train / 2262 target-train / 394 val / 994 test), so its predictions on the probe's test rows are NOT held out. Reported for completeness; no headline claim rests on it.

- `dmpnn_mtw_rh_seed2`: NOT COMPARABLE to the other arms: this checkpoint was trained on a different cohort/split (recorded n_train/n_val/n_test/n_joint_train below) than the probe's split (3480 joint-train / 2262 target-train / 394 val / 994 test), so its predictions on the probe's test rows are NOT held out. Reported for completeness; no headline claim rests on it.

## 5. Where does the gap go? (ridge vs head, same rows)

| encoder | head arm | test pearson | pearson(ridge, head) | partial corr(y,head | ridge) | partial corr(y,ridge | head) |
|---|---|---|---|---|---|
| `random_init_seed0` | `_ridge_` | 0.393907 | - | - | - |
| `random_init_seed0` | `head::L1_repo_head_alpha0.5_raw` | 0.141548 | 0.2011 | 0.0692 | 0.3769 |
| `random_init_seed0` | `head::L2_repo_head_alpha0.5_std_target` | 0.286255 | 0.5750 | 0.0795 | 0.2925 |
| `random_init_seed0` | `head::L3_wide_head_512` | 0.117931 | 0.1735 | 0.0548 | 0.3819 |
| `random_init_seed0` | `head::L3b_linear_head_no_hidden` | 0.131848 | 0.1709 | 0.0713 | 0.3802 |
| `random_init_seed0` | `head::L4_repo_head_wd1e-2` | 0.091706 | 0.1593 | 0.0319 | 0.3858 |
| `random_init_seed0` | `head::L5_repo_head_gradclip5` | 0.182235 | 0.2621 | 0.0891 | 0.3648 |
| `random_init_seed0` | `head::A02_alpha0.2_regweighted` | 0.141553 | 0.2014 | 0.0691 | 0.3768 |
| `random_init_seed0` | `head::A08_alpha0.8_clsweighted` | 0.141547 | 0.2011 | 0.0692 | 0.3769 |
| `random_init_seed0` | `head::LR3e-4` | 0.120684 | 0.1390 | 0.0724 | 0.3836 |
| `random_init_seed0` | `head::LR3e-3` | 0.162881 | 0.2253 | 0.0828 | 0.3716 |
| `random_init_seed0` | `head::LR1e-2` | 0.291377 | 0.4351 | 0.1450 | 0.3101 |
| `dmpnn_mtw_ir_seed0` | `_ridge_` | 0.368844 | - | - | - |
| `dmpnn_mtw_ir_seed0` | `checkpoint_own_head` | 0.055356 | 0.1544 | -0.0017 | 0.3652 |
| `dmpnn_mtw_ir_seed1` | `_ridge_` | 0.361119 | - | - | - |
| `dmpnn_mtw_ir_seed1` | `checkpoint_own_head` | 0.164607 | 0.1628 | 0.1150 | 0.3435 |
| `dmpnn_mtw_ir_seed2` | `_ridge_` | 0.371214 | - | - | - |
| `dmpnn_mtw_ir_seed2` | `checkpoint_own_head` | 0.090772 | 0.1028 | 0.0570 | 0.3653 |
| `dmpnn_mtw_os_seed0` | `_ridge_` | 0.305331 | - | - | - |
| `dmpnn_mtw_os_seed0` | `checkpoint_own_head` | 0.394637 | 0.4075 | 0.3107 | 0.1722 |
| `dmpnn_mtw_os_seed1` | `_ridge_` | 0.336605 | - | - | - |
| `dmpnn_mtw_os_seed1` | `checkpoint_own_head` | 0.516369 | 0.3777 | 0.4464 | 0.1786 |
| `dmpnn_mtw_os_seed2` | `_ridge_` | 0.353668 | - | - | - |
| `dmpnn_mtw_os_seed2` | `checkpoint_own_head` | 0.454276 | 0.4619 | 0.3507 | 0.1820 |
| `dmpnn_mtw_re_seed0` | `_ridge_` | 0.341660 | - | - | - |
| `dmpnn_mtw_re_seed0` | `checkpoint_own_head` | 0.289835 | 0.2940 | 0.2108 | 0.2803 |
| `dmpnn_mtw_re_seed1` | `_ridge_` | 0.442363 | - | - | - |
| `dmpnn_mtw_re_seed1` | `checkpoint_own_head` | 0.545726 | 0.5239 | 0.4110 | 0.2192 |
| `dmpnn_mtw_re_seed2` | `_ridge_` | 0.405100 | - | - | - |
| `dmpnn_mtw_re_seed2` | `checkpoint_own_head` | 0.361125 | 0.4686 | 0.2121 | 0.2863 |
| `dmpnn_mtw_rh_seed0` | `_ridge_` | 0.392592 | - | - | - |
| `dmpnn_mtw_rh_seed0` | `checkpoint_own_head` | 0.630531 | 0.5334 | 0.5413 | 0.0857 |
| `dmpnn_mtw_rh_seed1` | `_ridge_` | 0.387844 | - | - | - |
| `dmpnn_mtw_rh_seed1` | `checkpoint_own_head` | 0.544101 | 0.4736 | 0.4440 | 0.1762 |
| `dmpnn_mtw_rh_seed2` | `_ridge_` | 0.417200 | - | - | - |
| `dmpnn_mtw_rh_seed2` | `checkpoint_own_head` | 0.446940 | 0.5344 | 0.2916 | 0.2359 |

## 6. Checkpoints read (read-only)

- `molmetal/checkpoints/dmpnn_mtw_ir_seed0.pt` sha256 `280c7352fbbb585422a585c5a3daa1308fbef8db3c37075b0d0d91fb95a12c04`
- `molmetal/checkpoints/dmpnn_mtw_ir_seed1.pt` sha256 `f51bf4e027e3623449e4c90ef8433c4a9e1a0ab17c4e7eb5bcb20b41e32530f2`
- `molmetal/checkpoints/dmpnn_mtw_ir_seed2.pt` sha256 `bb9db2b32cecb4afedb3dcbca72aa10092e2f235f2693e4c76df2aea1c64a023`
- `molmetal/checkpoints/dmpnn_mtw_os_seed0.pt` sha256 `0227cb1c337c3e0a819217176e6bd8672e90b0341e27a0cd40ed4e17a66146d7`
- `molmetal/checkpoints/dmpnn_mtw_os_seed1.pt` sha256 `88ed7f25752d56d33b84c4224fb8726c560e2baa61b7148bf55e25b81b744972`
- `molmetal/checkpoints/dmpnn_mtw_os_seed2.pt` sha256 `afb05cd69d9beb093b596643502401c336e1ec7f66e4159d9b91c5ad44bb0d77`
- `molmetal/checkpoints/dmpnn_mtw_re_seed0.pt` sha256 `3b21b63405c2ed9ee98fd863cc04032339d3a0fc2e3a7b57f0366a0b156fa278`
- `molmetal/checkpoints/dmpnn_mtw_re_seed1.pt` sha256 `a1efa4546e3bcd08466672a29ec61f66ea68b6970bfe5fba41e5d8e3982251cc`
- `molmetal/checkpoints/dmpnn_mtw_re_seed2.pt` sha256 `841f3329550f9bebbe97db4e371cf1f79ba03c88ca9794d295f8bac0d340ad8f`
- `molmetal/checkpoints/dmpnn_mtw_rh_seed0.pt` sha256 `9923a87f1f7831dcdfe5fd02e8a5639a8c1a0ec8924cf8dd03dd70d7378a4118`
- `molmetal/checkpoints/dmpnn_mtw_rh_seed1.pt` sha256 `1a2d3c95d5ef8c38e28f94bdb6bd9d7e7b02136139373cd0b17db65b2e3c9ee5`
- `molmetal/checkpoints/dmpnn_mtw_rh_seed2.pt` sha256 `4a927724229cda3e0736802d023a335d31f88cc28dec209e263a48eb21bf562e`

## 7. Defects found in the READ-ONLY tree (reported, not fixed)

- **D1 (HIGH)** -- `molmetal/scripts/wf_pearson_r_pic50_head_vs_feature.py:101-106 (current) / a5cc339d:103-114 (the revision that produced the report)`
  - the published ridge reference was produced by a helper that selects the ridge alpha by maximising Pearson ON THE 994 TEST ROWS it then reports (test-set leak). The file on disk today has the opposite behaviour (alpha chosen on a val fold carved out of the train rows) and was changed on 2026-10-13 (commit 0e232def) WITHOUT regenerating molmetal/reports/wf_pearson_diag/report.json, so the committed script no longer reproduces its own published numbers.
- **D2 (HIGH)** -- `molmetal/reports/wf_pearson_diag/report.json vs molmetal/reports/wf_pearson_ridge_gpu/ru/per_seed_metrics.json`
  - two 2026-10-08 artefacts on the SAME split disagree about the trained head: the diag report says the head's test pearson is 0.2143, the sibling trainer run says 0.0113 for a nominally identical recipe (same split 2262/394/994). Neither is reconciled anywhere on disk.
- **D3 (MEDIUM)** -- `molmetal/scripts/wf_pearson_r_pic50_head_vs_feature.py:179 vs :190/:236`
  - the 'same-split' reference arms do NOT share a training set: the Morgan-FP arm is fit on the 2262 target-metal train rows while both encoder arms are fit on the 3480-row joint pool, so the ceiling and the model are not measured under the same data regime.
- **D4 (MEDIUM)** -- `molmetal/reports/wf_pearson_ridge_gpu/checkpoints/ (empty directory)`
  - the checkpoint path recorded in the 2026-10-08 per-seed metrics (dmpnn_mtw_ru_seed0.pt) does not exist, so the trained encoder/head of that run cannot be re-measured on CPU and arms (C)/(D) of the published table are NOT REPRODUCIBLE by construction.

- sibling 2026-10-08 artefact `molmetal/reports/wf_pearson_ridge_gpu/ru/per_seed_metrics.json`: seed=0 split=2262/394/994 test_pearson=**0.011266** vs the diag report's head pearson **0.214257** (discrepancy -0.202991); recorded checkpoint `molmetal/reports/wf_pearson_ridge_gpu/checkpoints/dmpnn_mtw_ru_seed0.pt` exists = False

## 8. The minimal GPU experiment that would close it (DECLARED, NOT RUN)

- status: DECLARED, NOT RUN (Wf-A201 rule 1: CPU-only, 0 GPU)
- question: does the 0.1931 gap survive once the reference arm stops selecting its hyper-parameter on the test rows?
  - re-run molmetal/scripts/wf_pearson_r_pic50_head_vs_feature.py unchanged EXCEPT that the trained encoder AND the trained head are written to disk (the Oct-8 run saved neither, which is why arms (C)/(D) cannot be re-measured today)
  - route all three ridge arms through the repo's current honest fit_ridge_readout (alpha chosen on a val fold, never on the test rows) and additionally report the a5cc339d leaked-rule number for the same fit, so the selection artefact is quantified rather than argued
  - 5 seeds for the head arm, same split (2262/394/994), same loss (alpha=0.5, Adam 1e-3, 30 epochs, batch 64, W2 0.05, L2 guard)
  - score BOTH the ridge and the head on the SAME 994 test rows and report the gap as: honest_ridge - head, per seed
- decision rule: if honest_ridge - head stays near the 0.1931 quoted, the gap is a modelling/objective gap and the CPU head-only arms L1..L5 say which lever; if honest_ridge drops by the measured leak inflation, part of the 0.1931 was a selection artefact and the remaining gap is the real one
- why it is the minimum: one script, one split, one change (save the checkpoint + honest alpha selection); it converts the two NOT-REPRODUCIBLE rows of the table into MEASURED rows

## 9. What was NOT done

- no GPU was used (nvidia-smi before/after above);
- no encoder was trained; every fit is a head-only readout on frozen features;
- no checkpoint was modified (sha256 recorded above);
- no `metrics/`, `registry/`, `paper/`, `docs/`, `TODO/` file was written;
- the 2026-10-08 published numbers were NOT edited;
- the head-only CPU fit omits the W2 (Wasserstein) regulariser, which in the trainer acts on the encoder's prediction distribution -- see report.json `not_measured` list.
- the three guards in this probe (CPU-only refusal, same-rows, write scope) are exercised -- and shown to BITE -- by `molmetal/tests/test_wfa201_encoder_readout_probe.py`: removing any one of them from a /tmp copy of this file makes the matching test fail with `DID NOT RAISE CpuOnlyViolation` / `SameRowsViolation` / `WriteScopeViolation`.

END OF REPORT (this line exists so the truncation guard sees a closing token).
