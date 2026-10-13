# bace capacity-lift — MEASURED record

**Date:** 2026-10-11 · **Scope:** MoleculeNet `bace` ROC-AUC, D-MPNN backbone, weak-metric re-opening round.
**Status:** all values MEASURED (each traces to a JSON on disk, read 2026-10-11). No PROJECTED values.
**Machine-readable companion:** [`bace_arms.json`](bace_arms.json) (same table + provenance).

## Headline

| | value | source |
|---|---|---|
| Historical baseline (varying per-seed splits, hidden 300 / depth 3) | **0.7505 ± 0.0456** | `molmetal/reports/wf_fix3lines_bace_50ep_2026_10_09/metrics.json` |
| **Best re-measured (varying splits, hidden 512 / depth 4)** | **0.7803 ± 0.0210** | `molmetal/reports/wf_a1_bace_cap512_d4/metrics.json` |
| Δ vs baseline | **+0.0298** (mean) | — |
| chemprop D-MPNN literature reference | 0.876 | (split-match not proven) |

**bace is partially lifted: 0.7505 → 0.7803 (+0.0298), with the seed std roughly halved (0.0456 → 0.0210).**
The mechanism is **model capacity**, and it is not a split artefact (see the fixed-split confirmation below).
The 0.876 chemprop gap is **not** closed — bace remains an honest-negative against the literature reference.

## Protocol note — the "5-seed std" mixes two variances

`molmetal/scripts/train_moleculenet.py::_scaffold_split(smiles, seed=seed)` seeds the Bemis-Murcko
scaffold split by the **training** seed, so each seed sees a **different** test set (MEASURED: bace
5-seed test-set intersection = 0). The historical "5-seed mean ± std" therefore mixes *split* variance
with *seed* variance. To separate them, the fixed-split arms below pin one split with the new
`--split-seed` flag (added this round; default `None` preserves the historical behaviour byte-for-byte).

## Arm table (all 5 seeds, 50 epochs, dmpnn, batch 64, lr 1e-3)

### Varying per-seed splits (the paper protocol — directly comparable to 0.7505)

| arm | lever | mean | std | verdict | source |
|---|---|---|---|---|---|
| baseline | hidden 300 / depth 3 | 0.7505 | 0.0456 | — | `wf_fix3lines_bace_50ep_2026_10_09` |
| B1 | `class_weight=off` | 0.7449 | 0.0433 | **null** | `wf_a1_bace_cw_off` |
| B2 | `dropout=0.2` | 0.7434 | 0.0505 | **null** | `wf_a1_bace_dropout02` |
| **B3** | **hidden 512 / depth 4** | **0.7803** | **0.0210** | **WIN (+0.0298, variance halved)** | `wf_a1_bace_cap512_d4` |
| B4 | lr 5e-4 + 100ep + wd 1e-5 + patience 20 | 0.7247 | 0.0868 | regression | `wf_a1_bace_lr5e4_100ep` |
| B5 | focal loss (γ=2.0) | 0.6935 | 0.0665 | regression | `wf_a1_bace_focal` |

### Fixed split (`--split-seed 0`) — removes split variance

| arm | lever | mean | std | Δ vs control | source |
|---|---|---|---|---|---|
| control | hidden 300 / depth 3 | 0.6369 | 0.0409 | — | `wf_a1_bace_cap300_split0` |
| **512** | **hidden 512 / depth 4** | **0.7467** | 0.0233 | **+0.1098** | `wf_a1_bace_cap512_d4_split0` |
| 768 | hidden 768 / depth 4 | 0.7433 | 0.0287 | +0.1064 | `wf_a1_bace_cap768_d4_split0` |
| 512+d | hidden 512 / depth 4 + dropout 0.1 | 0.7381 | 0.0250 | +0.1012 | `wf_a1_bace_cap512_d4_do01_split0` |
| 512+ens | hidden 512 / depth 4 + seed ensemble | 0.7495 | 0.0113 | +0.1126 | `wf_a1_bace_cap512_d4_ens_split0` |

- **Capacity, not split luck.** On the shared seed-0 split, hidden 512/depth 4 beats hidden 300/depth 3 by **+0.1098** — far beyond either std. The same split makes the *control* drop to 0.6369 (seed 0 is a hard split), so the varying-split 0.7505 and the fixed-split numbers are **not** cross-comparable; only arms sharing a split are.
- **Capacity saturates at 512.** 768/depth 4 (0.7433) does not beat 512/depth 4; dropout 0.1 (0.7381) does not help either. → the best configuration is **hidden 512 / depth 4**.
- **Seed ensemble** adds a further ~**+0.008–0.011** (ensemble 0.7572 vs single-seed mean 0.7495) and pushes the per-seed std down to 0.0113. Secondary to the capacity effect.

## Falsified levers (honest-negative; do not re-try as-is)

| lever | result | source |
|---|---|---|
| `class_weight=off` (isolated) | 0.7449 vs 0.7505 — **null**; class weighting is **not** the cause of the bace gap (confirms the earlier §6.38c hypothesis on a clean single variable) | `wf_a1_bace_cw_off` |
| `dropout=0.2` | 0.7434 — null | `wf_a1_bace_dropout02` |
| lower LR + 100 ep + wd + patience | 0.7247 ± 0.0868 — regression | `wf_a1_bace_lr5e4_100ep` |
| focal loss (γ=2.0) | 0.6935 ± 0.0665 — regression, noisiest arm; the "loss-function" axis is closed for bace | `wf_a1_bace_focal` |
| backbone swap to attentivefp (prior round) | ties D-MPNN (Δ +0.0131, noise) | `narrative_claims.yaml::bace_attentivefp_auc` |

## Reproduce

```bash
source env.sh
python molmetal/scripts/pre_flight_check.py --pipeline launch    # must PASS
USER_OK=1 bash scripts/start.sh configs/wf_a1_bace_cap512_d4.yaml --no-tui
```

## Honest framing (§5.2 / §5.5)

- Every number above is MEASURED and traces to the listed JSON path; none is PROJECTED.
- The bace gap vs chemprop (0.876) is **not** eliminated — capacity closes ~0.030 of the ~0.126 gap. Report the residual as an honest-negative; do **not** present 0.7803 as beating the literature.
- The varying-split (0.7803) and fixed-split (0.7467) numbers are on **different splits** and must not be compared to each other as if same-protocol.
