# Mini-batch OT coupling (Tong et al. 2023)

**Status:** done (round-8)
**Priority:** med
**Effort:** 3d
**Owner:** (unset)
**Depends on:** none
**Blockers:** POT (`pip install pot`) or PythonOT available
**Created:** 2026-09-12

## Goal
Replace the full-batch optimal-transport coupling in the EGNN training
loop with mini-batch OT (Tong et al. 2023, "Improving and Generalizing
Flow-Based Generative Models with Mini-Batch Optimal Transport"). The
project already ships a Hungarian OT pairing helper at
`molmetal/molmetal/flow_matching/optimal_transport.py`; this task adds the
mini-batch version on top.

## File(s) to edit
- /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal/flow_matching/optimal_transport.py  (add mini_batch_ot_coupling)
- /home/hugo/codes/try_triton_on_rocm/molmetal/molmetal/flow_matching/loss.py  (ConditionalFlowMatchingLoss accepts use_minibatch_ot=True)

## Success criterion
`optimal_transport.py` exposes `mini_batch_ot_coupling(x1, x0, batch_idx)`
using Sinkhorn or Hungarian within mini-batch;
`ConditionalFlowMatchingLoss(use_minibatch_ot=True)` works end-to-end;
ablation on MMP13 with/without mini-batch OT shows Vina mean delta in
favour of mini-batch OT (positive or neutral, never worse).

## Related reports
- (Phase-2 deliverable; no prior in-repo report yet — generate one when complete)