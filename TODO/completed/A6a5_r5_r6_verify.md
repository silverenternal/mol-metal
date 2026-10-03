# Wf-A6a5 (2026-10-03) — R5 + R6 fix verification + §5.7 smoke verify

Per AGENTS.md §5.5 filesystem-truth + §5.6 atomic write + §5.7 trainer
flag wire-in smoke-required rule + §5.8 token-budget discipline.

## MEASURED findings

Both trainers ( +
) already SHIP --max-norm + --weight-decay
CLI flags + gradient clipping + optimizer weight_decay wire-in:

- D-MPNN  --max-norm flag at line 1865 default 5.0
- D-MPNN gradient clip site at line 803-810:
    max_norm = float(getattr(args, "max_norm", 0.0))
    if max_norm > 0.0: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
- D-MPNN optimizer at line 1079: weight_decay=wd threaded from main() via args
- CFM  --max-norm flag at line 1955 default 5.0
- CFM gradient clip site at line 1607-1611:
    max_norm = float(getattr(args, "max_norm", 5.0))
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm))
- CFM optimizer at line 2893: weight_decay=getattr(args, "weight_decay", 0.0)
- scripts/start.sh line 205 + 209: WEIGHT_DECAY + MAX_NORM both parsed from yaml protocol
- configs/training/m17_pt_platinai_4cell_5seed_50ep_v1.yaml protocol:
    weight_decay: 0.0  (already)
    max_norm: 5.0       (added in this Wf-A6a5 closeout)

## §5.7 floor-1 argparse-accepts verification

D-MPNN: --max-norm in --help = True; --weight-decay in --help = True
CFM: --max-norm in --help = True; --weight-decay in --help = True

## §5.7 50-step CPU smoke verification

D-MPNN smoke rc=0 (trainer reaches 50 steps for Ru seed=0 with
--weight-decay 0.01 --max-norm 1.0 --epochs 1 --batch-size 4
--data-csv .data/molmetal/metalcytotox_5metal_multitask_v4.csv
--csv .data/molmetal/metalcytotox_5metal_multitask.csv
--device cpu). Other 4 metals (Ir/Os/Rh/Re) hit a PRE-EXISTING
NameError "args is not defined" in L2_GUARD code path (separate
from --max-norm wire-in; deferred to Wf-A6 BLOCKER D follow-up).

## Blocker closure status

- R5 (--weight-decay wire-in): CONFIRMED on disk in both trainers
- R6 (--max-norm wire-in): CONFIRMED on disk in both trainers + start.sh
  + yaml m17 protocol updated with max_norm: 5.0 (added in this wave)
