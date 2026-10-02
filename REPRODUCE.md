# REPRODUCE.md — Single-file Reproducibility Contract

**Last updated:** 2026-10-02 (Wf-438 simplification + Wf-439 wrapper audit fix + 4 new doc audits shipped per §10.2 or user voice)

This is the **1-command reproducibility contract** for the Mol-Metal project.
If you have ROCm 7.2 + `triton-rocm` 3.8.0 + Python 3.12 + uv installed, this single command reproduces the
canonical R7-mate ship surface on an AMD RX 7800 XT (16 GB, gfx1101):

```bash
uv run python main.py --smoke --metal Pt --cuiddcer scaffold_split --seeds 0 1 2
```

**Expected outputs** (after ~6 min wall on RX 7800 XT):
- `molmetal/reports/wf124_p3_pt_4cells/synthesis.md` — 3-seed × 30-epoch aggregation
- `metrics/by_metric/metal_lomo_auc_wasserstein.json` — canonical 7/7 pair aggregate
- `paper/main.pdf` — 69 pages / 5.17 MB / 0 unresolved bibitems (recompile on next paper edit)
- Pre-flight v3 PASS gate: 93 PASS / 3 FAIL / 2 SKIP / 98 TOTAL wall 176.68 s

**Honest framing** (per AGENTS.md §1.2 + party-twenty-spirit §3.5):
- 5 of 27 baseline primers are MEASURED at scale (Krasnov 0.8007 / Rusanov / DrugOOD / Vina / MeVina)
- 2 honest-negative gates have structural closure paths (m2 R² + m6_kappa)
- CFM decode_ratio = 0.998 MEASURED per `docs/PAPER_SECTION_3_METRIC_TABLE_FINAL.md`

**Cross-links** (per Wf-440 / Wf-441 / Wf-442 / Wf-443 + this wave's Wf-438 simplification):
- Top-level [README.md](README.md) — overview + quickstart
- Top-level [CHANGELOG.md](CHANGELOG.md) — versioned changes (wf-free naming)
- [TODO/README.md](TODO/README.md) — TODO directory overview + navigation
- [TODO/02_metrics/](TODO/02_metrics/) — metrics + mechanism mapping + architecture
- [TODO/03_baselines/](TODO/03_baselines/) — 3 Tier-1 baselines + protocol alignment
- [TODO/03_plans/](TODO/03_plans/) — R17-R20 strategic plans + index
- `docs/ARCHITECTURE.md` — molmetal/ hexagonal architecture
- `docs/DESIGN_BOUNDARY.md` — adapter vs port boundary

**Versioning** (per 2026-09-30 user directive *"扁平化这些wave，审稿人考到项目的时候是不会在一这些wave信息的"*):
- All version history lives in `git log` + `CHANGELOG.md`
- No per-wave file naming convention
- `git log --oneline` shows the canonical ship surface
