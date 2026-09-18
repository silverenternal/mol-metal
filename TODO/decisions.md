# Pending Decisions

**Status:** living doc
**Last updated:** 2026-09-12

---

## Approved decisions (2026-09-12)

### D1 ✅ DiffSBDD ckpt access strategy
- **Decision:** option (c) — stay on cite-only path for arXiv preprint; rely on the
  new `molmetal/scripts/r4_c_full_sweep.py` orchestrator for Lambda's measured column.
  The orchestrator subprocess-calls `references/DiffDock/inference.py` whenever the
  DiffDock model_dir + torch_geometric are available; otherwise it records
  `status=torch_geometric_missing` or `status=checkpoint_missing` (honest fallback
  per TODO/risks.md R1).
- **Reopens if:** DiffSBDD authors grant ckpt access OR ROCm torch_geometric wheels
  land. Then revisit (a)/(b) for camera-ready.
- **Approved by:** user, 2026-09-12

### D2 ✅ MMP2 vs MMP13 vs CA2 evaluation scope
- **Decision:** MMP13 primary (PDB 830c, 442 CrossDocked2020 pairs), CA2 secondary
  (stress test), MMP2 skipped (0 pairs).
- **Evidence:** `TODO/completed/13_mmp13_surrogate.md` + `mmp13_vina_real.md`
  (100/100 docked, mean=-1.355).
- **Approved by:** user, 2026-09-12

### D3 ✅ PDBbind mirror fallback
- **Decision:** option (c) — skip PDBbind in headline; lean on CrossDocked +
  MMP13 holdout. BindingDB (option b) is the training-data fallback if labels
  are usable.
- **Approved by:** user, 2026-09-12
- **Status:** remains open operationally (PDBbind download still blocked per R7)

### D4 ✅ EGNN vs DiS diffusion backbone (for the metal-prior)
- **Decision:** EGNN first via `TODO/pending/09_metal_prior.md` (now archived to
  `TODO/completed/22_metal_prior_done.md`) — dative edge type +
  90° Pt(II) angle constraint wired into `molmetal/adapters/egnn_rocm.py` +
  `molmetal/molmetal_lam/priors/metal_geometry.py` (new file).
- **Reopens if:** EGNN r=0.407-style correlation gap doesn't close by Phase-3
  midpoint (per R8). Then swap to DiS-style diffusion.
- **Approved by:** user, 2026-09-12

### D5 ✅ PoseBusters vs internal geometry check (fast-path)
- **Decision:** keep PoseBusters as the headline metric for paper credibility;
  internal RDKit-based geometric check as fast-path filter in MCTS expansion
  (gate candidate branching).
- **Approved by:** user, 2026-09-12

---

## New pending decisions (emerged from round-4 work)

### D6 ⏳ REINVENT4 install path
- **Context:** the r4_c_full_sweep orchestrator can call REINVENT4 via the
  existing `reinvent4_subprocess_adapter.py` once `reinvent` is on PATH. The
  cloned REINVENT4 repo at `molmetal/references/REINVENT4/` ships its own
  `uv.lock` and `pyproject.toml` — it can be installed via
  `uv pip install --python /path/to/.venv/bin/python -e
  molmetal/references/REINVENT4/` (one-liner). But that pulls in REINVENT4's own
  deps into our shared .venv which may conflict with our torch 2.14.0+rocm7.2 pin.
- **Options:**
  - (a) Install REINVENT4 into a separate venv (`uv venv .venv-reinvent4` then
    `uv pip install -e molmetal/references/REINVENT4`) — clean isolation, but
    the subprocess RPC needs to point to the other venv's `reinvent` binary
  - (b) Install REINVENT4 into our shared .venv — simpler subprocess, but risk
    of torch version conflict (REINVENT4 wants torch>=2.5; we have torch 2.14.0+rocm7.2)
  - (c) Use Docker image `mricci/reinvent:latest` (per REINVENT4 README) — no
    local venv pollution, but requires docker daemon (unconfirmed on this host)
- **Default:** (a) — separate venv. Decide by 2026-09-19.

### D7 ⏳ Vina → QVina engine swap activation
- **Context:** task 04 added the `--engine {vina|qvina|quickvina2}` flag to
  `molmetal/molmetal_lam/sbdd_env/vina_adapter.py`. Default stays Vina 1.2.7
  to preserve mmp13_vina_real.md parity (100/100 docked, mean=-1.355).
- **Options:**
  - (a) Keep Vina 1.2.7 default forever — Lambda numbers published as-is, accept
    systemic bias vs SBDD literature (which used QVina)
  - (b) Flip default to QVina at v1.1 release — Lambda numbers stay defensible
    but lose mmp13_vina_real.md bit-equivalence
  - (c) Run both engines in the headline table, label each row — most honest,
    but doubles the work for the 100-pocket sweep
- **Default:** (c) for the arXiv table; (a) until QVina binary is actually
  installed on the run host. Decide when QVina is installed.

---

## Cross-cutting

- See `risks.md` for the 8 open risks
- See `completed/` for the 23 archived items (13 original + 10 round-4)
- See `molmetal/reports/round4_pending_tasks_done.md` for the full evidence
  trail on the round-4 work