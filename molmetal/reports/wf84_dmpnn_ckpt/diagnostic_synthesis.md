# Wf-84 Phase 3 — D-MPNN pIC50 checkpoint diagnostic

- ckpt_dir: `molmetal/checkpoints`
- pattern: `dmpnn_pic50_*_seed*.pt`
- n_checkpoints: **10**
- n_strict_ok: **10**
- n_load_errors: **0**
- n_inference_ok: **10**
- elapsed: 0.7s
- device: cuda:0

## Per-checkpoint summary

| ckpt | size | keys | strict_ok | pic50 range |
|------|------|------|-----------|-------------|
| dmpnn_pic50_ir_seed0.pt |   758733 | 22 | yes | [-0.140, -0.061] |
| dmpnn_pic50_ir_seed1.pt |   758733 | 22 | yes | [+0.055, +0.394] |
| dmpnn_pic50_os_seed0.pt |   758733 | 22 | yes | [-0.374, -0.086] |
| dmpnn_pic50_os_seed1.pt |   758733 | 22 | yes | [+0.043, +0.517] |
| dmpnn_pic50_re_seed0.pt |   758733 | 22 | yes | [-0.171, -0.064] |
| dmpnn_pic50_re_seed1.pt |   758733 | 22 | yes | [+0.032, +0.217] |
| dmpnn_pic50_rh_seed0.pt |   758733 | 22 | yes | [-0.339, -0.122] |
| dmpnn_pic50_rh_seed1.pt |   758733 | 22 | yes | [+0.075, +0.340] |
| dmpnn_pic50_ru_seed0.pt |   758733 | 22 | yes | [-0.637, +0.044] |
| dmpnn_pic50_ru_seed1.pt |   758733 | 22 | yes | [+0.029, +0.372] |

## Honest framing

This diagnostic confirms ckpt existence + load + finite forward pass.
It does **not** validate pIC50 R² quality; see `training_summary.json`.
