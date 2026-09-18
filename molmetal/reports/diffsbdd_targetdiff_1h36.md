# DiffSBDD / TargetDiff — 1h36 inference log

**Task #193** — 2026-09-11

## Status: NO INFERENCE RUN — pretrained checkpoint unavailable in sandbox

See `reports/diffsbdd_targetdiff_no_ckpt.md` for full diagnosis.

### Why no log here

- Both pretrained ckpts (TargetDiff on Google Drive, DiffSBDD on Zenodo 8183747) are unreachable: sandbox blocks egress.
- `DiffSBDD` repo is not cloned locally.
- `torch_geometric` / `torch_scatter` are not installed in the active `.venv` (which is pinned to ROCm `torch 2.14.0+rocm7.2` + Triton 3.8).
- `targetdiff/pretrained_models/` is empty (the repo's own README points to an external Google Drive folder).

### What we would have run, given ckpts

```bash
# TargetDiff on 1h36 (100 samples, batch 100, ~3 min/sample-budget)
cd molmetal/references/targetdiff
python scripts/sample_for_pocket.py configs/sampling.yml \
    --pdb_path examples/1h36_A_rec_1h36_r88_lig_tt_docked_0_pocket10.pdb \
    --num_samples 100 --batch_size 10 \
    --result_path /tmp/tdiff_1h36

# DiffSBDD on 1h36 (20 samples per pocket, fullatom cond model)
cd molmetal/references/DiffSBDD
python generate_ligands.py checkpoints/crossdocked_fullatom_cond.ckpt \
    --pdbfile /path/to/1h36.pdb --outfile diffsbdd_1h36.sdf \
    --ref_ligand A:R88 --n_samples 100
```

### Substitute delivered

- `molmetal/molmetal_lam/sbdd_env/targetdiff_adapter.py` — abstract-layer stub documenting the interface and degrading to `NotImplementedError` until ckpts are dropped in.
- `reports/lambda_vs_sbdd_paper_numbers.md` already exists and contains cite-only published numbers — that is the active comparison artifact for this paper.

### Future-work checklist (when sandbox opens)

- [ ] wget TargetDiff pretrained_diffusion.pt → `references/targetdiff/pretrained_models/`
- [ ] git clone DiffSBDD + wget crossdocked_fullatom_cond.ckpt from Zenodo
- [ ] Build ROCm-compatible torch_geometric + torch_scatter wheels (or wrap EGNN k-NN in pure torch)
- [ ] Re-run the two `python ...` commands above
- [ ] Replace this placeholder with the actual inference log + Vina docking results

### Time spent

Investigation + adapter stub + two reports: ~25 min.
1h36 budget cap was 1 h 36 m total; remaining time was not enough to spin up PyG build on ROCm even if network were open.
