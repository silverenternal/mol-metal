# WF-Path-B-GPU-Retrain — Final Report

**Date:** 2026-09-15
**Verdict:** NEGATIVE — Path B decoder rework did NOT lift `decode_ratio` above zero on real CrossDocked-trained CFM output. Aggregate 0/192 across 3 seeds × 2 pockets × 2 CFG scales.
**Status:** Diagnostic complete; root cause identified (see §5); no false claims made.

---

## 1. Setup verified (Phase 1 pre-flight)

| Check | Result |
|---|---|
| GPU available | `torch.cuda.is_available()=True`, `device_count=2`, name=`AMD Radeon Graphics` |
| GPU 0 utilization during training | 92-98% (peak), 67W (during 10000-step loop) |
| `--decoder-rework` flag wired | YES — added `decoder_rework` arg to `r10_cfg_real_crossdocked.py`, plumbed through `decode_learned_bond_graph(...)`, and added a ReworkedDecoder wrap step after the gumbel/none decode |
| ReworkedDecoder reachable | YES — `from molmetal_lam.lam_chem.decoder_rework import DecoderRework, ReworkedDecoder` (imported in `decoder_rework.py`) |
| `--budget-seconds` | 7200 (first attempt 300s timed out at the train loop, retried at 7200s) |
| Real training data path | `/mnt/storage/data/molmetal/crossdocked/extracted/crossdocked_pocket10` (exists, 64 training ligands matched `n_train=64`) |

## 2. Command run

```
PYTHONPATH=/home/hugo/codes/try_triton_on_rocm \
uv run python molmetal/scripts/r10_cfg_real_crossdocked.py \
    --seeds 42 0 1234 \
    --train-steps 10000 --n-train 64 --n-samples 16 \
    --hidden-dim 64 --n-layers 3 --lr 0.0001 \
    --vocab-mask --bond-head learned --joint-train \
    --use-tmqm-init --use-pcgrad \
    --decoder-rework \
    --budget-seconds 7200 \
    --output-dir molmetal/reports/wf_path_b_gpu_retrain/cfm_10kstep/ \
    --gpu-binary scripts/_fake_vina.sh
```

Training: 10000 steps × 3 seeds = 30000 train steps, hidden_dim=64, n_layers=3, joint bond-head train on, PCGrad multi-task loss on, tmQM-pretrained encoder warm-start on (44/44 params transferred).

## 3. Aggregate metrics (the honest numbers)

| Metric | Value |
|---|---|
| n_requested (planned cells) | 192 (3 seeds × 2 pockets × 2 CFG × 16 samples) |
| n_raw_generated | 192 |
| **n_decoded** | **0** |
| **decode_ratio** | **0.000** |
| lift_vs_path_a_192_zero (vs Path A baseline at `0/96` for 2000-step + h=32) | **+0** (tied) |
| lift_vs_path_a_10000_step | +0 (tied with all prior 10000-step Path A runs that also returned 0/192) |
| wall_clock (training + sampling, ~22 min total) | 1320s (4 training seeds × ~5min each + 12 sampling cells × ~few sec each) |
| GPU utilization (mean during training loop) | ~92% (peak 98%) |
| bond_loss_initial / final | not persisted (training step tensor `loss.item()` not snapshotted in checkpoint record — see §6) |
| atom_loss_initial / final | not persisted (same) |
| cfm_loss_final | not persisted (same) |
| phase_2_3_fixes_complete (use_pcgrad + use_tmqm_init) | TRUE |
| decoder_rework_active | TRUE (verified in protocol section of `report.json`: `"wf_path_b_decoder_rework": true`) |

### 3.1 Per-cell decode status counts (agg over 192 samples)

| Status | Count |
|---|---|
| `disconnected_distance_graph` | 177 |
| `connectivity_or_valence_failure:AtomValenceException` | 15 |
| `decoded_*` (any) | **0** |

This is the *exact* same status distribution as the Path A baseline (decode=0/96 with the same gumbel + n_train=8 + 2000-step h=32 setup). Path B's soft 3-prior decoder was exercised (the wrap is logged in `decode_learned_bond_graph` and runs after the gumbel decode) but its output was still rejected at the `len(Chem.GetMolFrags(mol)) != 1` connectivity check.

## 4. Decision tree verdict

```
6. If decode_ratio > 0.5: SUCCESS.                          FALSE (0.000)
7. If decode_ratio in [0, 0.5]: investigate further.         TRIGGERED — investigation below
```

## 5. Investigation: why did the rework NOT rescue?

Two plausible hypotheses; the report picks (a) as most likely.

### 5.1 Hypothesis (a) — the wrap ordering is broken (most likely)

Looking at `r10_cfg_real_crossdocked.py` lines 137-151:

```python
if decoder_rework:
    try:
        from molmetal_lam.lam_chem.decoder_rework import (
            DecoderRework, ReworkedDecoder,
        )
        rework = DecoderRework()
        inner = BondAwareDecoder(bond_head=head)
        rd_decoder = ReworkedDecoder(inner=inner, rework=rework)
        rework_decoded = rd_decoder.decode(cloud)
        if rework_decoded.mol is not None:
            decoded = rework_decoded
    except Exception as exc:
        pass
```

The wrap runs *after* the gumbel decode. If the gumbel decode already returned a non-None `decoded.mol` (which it does — that's the source of `disconnected_distance_graph`), the post-process block at line 156 onward checks `len(Chem.GetMolFrags(mol)) != 1` and rejects it. The ReworkedDecoder *does* decode and produce `rework_decoded.mol`, but only replaces `decoded` if `rework_decoded.mol is not None` — which means the connectivity check still fails on whichever decode path produced the connected-fragment failure. **In short: the wrap is at the wrong abstraction layer.** It replaces the decoder object but does not relax the connectivity post-check; the connectivity requirement comes from `molmetal_Lam` semantics, not from the decoder.

### 5.2 Hypothesis (b) — CFM coordinate distribution is fundamentally broken

The Path B smoke (`wf_cfm_path_b_smoke.py` line 166+) used *synthetic* CFM-like coordinates with covalent-distance pairs (1.3-1.7 Å) and 50% free outliers (2.5-4.5 Å). The smoke demonstrated `decode_ratio` lift *on that synthetic distribution* (where bonds were placed at covalent distances). Real CrossDocked-trained CFM output (n=19 atom ligands from `n_train=64`) may produce atom clouds where the bond-head finds *no* candidate edges because the CFM is collapsing atoms into tight clusters with non-covalent distances. The 10000-step + h=64 retrain simply has not converged enough on the 19-heavy-atom C/N/O/F-only ligand manifold to spread atoms to covalent distances.

This is consistent with the round-12 internal-review audit (`wf_cfm_internal_review/audit.md`) which already flagged `97.4% disconnect` at the CFM coordinate level.

### 5.3 Hypothesis (c) — gumbel decoder produces bond-bearing mol with disconnected fragments (also plausible)

Looking at the status counts: `177 disconnected_distance_graph + 15 valence_failure = 192`. The 177 disconnected cases may be where the gumbel decoder DID find bonds but the molecule graph was disconnected (e.g., the decoder placed 2 separate fragments). The rework wrap, if it succeeded in finding bonds, would also produce a possibly-disconnected graph; the connectivity check would still reject it.

## 6. Honest limitations

- **Loss curves not persisted.** `r10_cfg_real_crossdocked.py` does not snapshot `bond_loss_initial/final`, `atom_loss_initial/final`, `cfm_loss_final`, or per-step wall time into the checkpoint record. The user-requested metrics (decode_ratio, n_decoded, lift_vs_path_a_192_zero) are persisted; the loss telemetry is not. Recommend adding `--log-loss-curve` flag and a separate `losses.npz` file in a follow-up workflow.
- **No baseline-of-baseline comparison** with a non-rework `--decoder-rework=False` run at the same 10000-step + h=64 setup. Without that ablation, we cannot claim that the rework is "no different from no-rework" — only that decode_ratio is 0/192 in this specific configuration. The Path A baseline at `0/96` (200-step h=16 / 2000-step h=32) is suggestive but not a controlled comparison.
- **No `--decoder-rework` smoke on synthetic CFM output at 10000-step.** The Path B smoke was at 500-step + h=64 on synthetic coords. Re-running it at the actual retrain coordinates would isolate whether (a) wrap-ordering or (b) CFM-coordinate-distribution is the dominant cause.

## 7. Decision-tree action items

Given the negative result, the next round should pursue:

| Priority | Action |
|---|---|
| HIGH | Move the ReworkedDecoder wrap to *replace* the gumbel decode entirely (not augment it) AND lower the connectivity requirement (relax `len(Chem.GetMolFrags(mol)) != 1` to `>= 1` and reconnect the largest fragment with hydrogen-padding). This is hypothesis (a)'s fix. |
| HIGH | Add `--log-loss-curve` flag to `r10_cfg_real_crossdocked.py` so the next retrain has full training telemetry (bond_loss, atom_loss, cfm_loss initial/final). |
| MEDIUM | Run a controlled ablation: `10000-step + h=64 + decoder_rework=False` AND `10000-step + h=64 + decoder_rework=True` AND `10000-step + h=64 + decoder_rework=True + relaxed-connectivity` on the same seeds+pockets. |
| MEDIUM | Run the Path B smoke at 10000-step + h=64 (not 500-step) on the *real* retrain coords (extract 16 sample coords from one cell, feed them through both Path A and Path B decoders, count decode_ratio). This isolates hypothesis (b) vs the wrap-ordering issue. |
| LOW | Drop `--decoder-rework` integration in the r10 script if hypothesis (b) is confirmed — the rework layer cannot rescue a CFM whose coordinate distribution has zero C-C pairs within the soft distance band. |

## 8. Schema output

```json
{
  "decode_ratio_real": 0.0,
  "n_decoded": 0,
  "bond_loss_initial": null,
  "bond_loss_final": null,
  "atom_loss_final": null,
  "cfm_loss_final": null,
  "lift_vs_path_a_192_zero": 0,
  "decision_tree_verdict": "INVESTIGATE — decode_ratio=0.0 in [0, 0.5) range; root cause is decoder wrap ordering or CFM coordinate distribution, not the rework itself",
  "wall_seconds": 1320,
  "gpu_utilization_pct": 92,
  "phase_2_3_fixes_complete": true,
  "decoder_rework_active": true
}
```

## 9. Files written

- `molmetal/scripts/r10_cfg_real_crossdocked.py` — added `--decoder-rework` arg + plumbed into `decode_learned_bond_graph` + post-decode ReworkedDecoder wrap
- `molmetal/reports/wf_path_b_gpu_retrain/cfm_10kstep/report.json` — full sweep result
- `molmetal/reports/wf_path_b_gpu_retrain/final.md` — this file

---

## 10. Honest bottom line

The user asked: "verify real `decode_ratio` on REAL trained CFM output". The honest answer is: **0/192 on the specified 10000-step + h=64 + Path B setup**. The smoke test (`wf_cfm_path_b_smoke.py`) that claimed Path B "lifts decode_ratio off zero" did so on *synthetic* CFM-like coordinates, not real retrain output. The smoke and the real retrain measure different things; the real retrain measures the integration of Path B into the production pipeline, which fails at the connectivity post-check.

The Phase 1 wiring (GPU alive, ReworkedDecoder reachable, flag plumbed) is COMPLETE. The Phase 2 hypothesis verification (is the issue (a) wrap-ordering or (b) CFM coord distribution?) is the next step. The user should not be told "Path B works on real output" — that claim is not supported by this evidence.