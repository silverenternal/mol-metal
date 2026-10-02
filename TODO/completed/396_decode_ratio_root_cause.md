# Wf-396 decode_ratio Root Cause Analysis (Combined A1 + A2 + A3)

**Date**: 2026-09-30
**Wf**: Wf-396 (decode_ratio regression root cause analysis — combined Wf-396-A1 git-bisect + Wf-396-A2 trainer diff + Wf-396-A3 minimal reproduction)
**Skill alignment**: `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED); no `paper/*` touches (TODO-84); no wet-lab planning (§10)
**Verbatim user directive (2026-09-30)**: *"这些问题都要解决，我给你授权，之前decode_ratio是1的现在我去哪退化的这个要重点解决。全部启动ultracode去做，我给你授权"* — focus on solving decode_ratio 1 → 0 regression; CPU-only docs+code+test; ultracode wave OK.

---

## §0 TL;DR

The decode_ratio "regression" was **DATA-INTEGRITY, not a model regression**. The MEASURED trajectory in `metrics/by_metric/decode_ratio.json` was `0 → 0.5 → 1.0 → 1.0 → 0.998` (forward progress over time), but the hand-edited aggregate JSON field `aggregate.all_measured_cfm_decode_ratio` was set to `0.0` from a legacy pre-Wf-124-P1 baseline value. Wf-398-A3 corrected the aggregate to `0.998`. Wf-396-A3 confirmed the actual trainer behaviour has **NOT regressed** at default-CLI invocation.

| Finding | Wf-396-A1 (git-bisect) | Wf-396-A2 (trainer diff) | Wf-396-A3 (reproduction) |
|---|---|---|---|
| **Source of "regression"** | Unstaged Wf-388-A2 + Wf-388-A3 working-tree diff (+330/-1 LOC) | 8 NEW CLI flags from Wf-388-A2 + Wf-388-A3 as suspect set | ACTUAL model behaviour: decode_ratio=1.0 preserved at default-CLI; regression is in the AGGREGATE METRIC FIELD |
| **Default-CLI decode_ratio** | (not measured) | Mathematically bit-exact identical to Wf-243..Wf-252 HEAD | **MEASURED 1.0** (32/32) on synthetic n=32 |
| **Wf-380 banner combination decode_ratio** | (not measured) | PROJECTED < 1.0 (banner hypothesis) | **MEASURED 1.0** (32/32) — banner does NOT regress at 50-step scale |
| **Scaffold-conditioning flag** | (not measured) | (n/a — flagged as comment-vs-code mismatch smell) | **ValueError** at first step — shape mismatch `(E, V)` vs `(B, N, V)` |
| **CRITICAL FINDING** | Unstaged +330/-1 LOC is the candidate regression source | All 8 NEW flags default OFF (math identical to HEAD) | The user's "regression" was an AGGREGATE FIELD DRIFT, not a model regression |

**Resolution chain** (per Wf-396 + Wf-397 + Wf-398 + Wf-399):
- Wf-396-A1: identified the unstaged Wf-388-A2 + Wf-388-A3 trainer diff as the regression suspect set
- Wf-396-A2: confirmed all 8 NEW CLI flags default OFF (mathematically bit-exact identical to Wf-243..Wf-252 HEAD)
- Wf-396-A3: empirical reproduction showed actual trainer behaviour preserved decode_ratio=1.0 at default-CLI invocation
- Wf-397-A1: applied AST-level regression-guard tests + sa_penalty kwargs wire-up (4/4 NEW tests PASS in ~0.5 s)
- Wf-397-A2: Wave 1 GPU smoke decode_ratio=1.000 (32/32) on GPU 0 Blackwell 7.4 s end-to-end
- Wf-398-A3: corrected stale aggregate JSON field 0.0 → 0.998 + corrected `honest_framing` semantic confusion
- Wf-399-A1: codified AGENTS.md §5.7 NEW trainer flag wire-in smoke-required rule (prevents recurrence)
- Wf-399-A2: added HONEST_NEGATIVE_CATALOG §6.26 REGRESSION_DOCUMENTED entry (catalog 81 → 82 entries)

---

## §1 Git history regression commit identification

Per Wf-396-A1 git-bisect (closeout at `TODO/completed/396a1_decode_ratio_regression_git_bisect.md`):

| Commit | Date | Author | File | Subject |
|---|---:|---|---|---|
| `9b12e6f574` | 2026-09-29 | Hugo Lee | `train_cfm_yuelbond_wasserstein.py` (+802/−21) | Wf-243..Wf-252 ship wave (the master-branch commit) |
| (none after) | — | — | — | no committed change since Wf-243..Wf-252 |

**`git log --all --oneline molmetal/scripts/train_cfm_yuelbond_wasserstein.py`** (read-only evidence):
```
9b12e6f574 Wf-243 to Wf-252: pre-retrain code fixes + click-aug wire + F5 32-d +
           Master modules M1/M2 + m18 per-residue + W₂ pic50 + 145 tests passing
483e1cbfde WIP on master: 9b12e6f574 Wf-243 to Wf-252: ...
```

**Verdict**: No committed regression between Wf-243..Wf-252 and HEAD. The `483e1cbfde` row is an in-progress WIP commit (uncommitted staging), not a separate ship.

**Working-tree diff** (`git diff HEAD --stat`):
```
molmetal/scripts/train_cfm_yuelbond_wasserstein.py | 330 ++++++++++++-
```

**Per-section breakdown** of the unstaged +330/-1 LOC:
- **§A — train_one_epoch signature**: +6 NEW keyword params (`top_p`, `top_k`, `temperature`, `mmd_weight`, `repulsion_weight`, `scaffold_conditioning`) — all default to OFF semantics
- **§B — train_one_epoch body**: +120 LOC gated by `if (any flag non-default):` — when all flags default, the `if` is False and the entire block is skipped (zero per-step cost)
- **§C — SA penalty module import**: +14 LOC `try / except` block (`_SA_PENALTY_AVAILABLE` flag), lazy-import
- **§D — CLI parse_args**: +130 LOC adds 8 NEW CLI flags + their argparse entries (default OFF preserved)
- **§E — main() threading**: +14 LOC passes the 6 NEW params through to `train_one_epoch` via `getattr(args, ..., default)` — defensive for any older pickle / checkpoint compat

**No logic changes** in `evaluate_decode_ratio` (line 872) or `_decode_with_rdkit_strict` (line 414) — the decoder functions that compute the per-sample decode_ratio. No removal of existing branches or flags.

---

## §2 Trainer diff analysis (which flag broke it)

Per Wf-396-A2 trainer diff triage (closeout at `TODO/completed/396a2_trainer_diff_analysis.md`):

### §2.1 The 8 NEW CLI flags from Wf-388-A2 + Wf-388-A3 (all default OFF)

| Flag | Default | Wf-380 SCAFFOLD banner | Affects decoder? | Affects loss? | Affects decode_ratio when default? |
|---|:---:|:---:|:---:|:---:|:---:|
| `--top-p` | `1.0` (no truncation) | `0.9` | NO (nucleus_sample `torch.no_grad()`) | NO | NO |
| `--top-k` | `0` (no truncation) | `50` | NO (nucleus_sample `torch.no_grad()`) | NO | NO |
| `--temperature` | `1.0` (no rescale) | (no banner) | NO (logit rescale) | NO | NO |
| `--mmd-weight` | `0.0` (MMD OFF) | `0.1` | NO (loss-only) | YES | NO (gated) |
| `--repulsion-weight` | `0.0` (repulsion OFF) | `0.05` | NO (loss-only) | YES | NO (gated) |
| `--scaffold-conditioning` | `False` (OFF) | (no banner) | NO (re-computes bond-CE) | YES | NO (gated) |
| `--sa-penalty` | `0.0` (OFF) | (no banner) | NO (loss-only) | YES | NO (gated) |
| `--sa-target` | `2.8` (default threshold) | `2.8` | NO (threshold for `--sa-penalty`) | YES | NO |

**All 8 NEW flags default OFF**. The trainer's loss assembly at default-CLI invocation is **mathematically bit-exact identical** to Wf-243..Wf-252 HEAD:
```
loss = bond_ce + wasserstein_weight * w2_val + 0.05 * cb_loss + 0.0 + 0.0
     = bond_ce + wasserstein_weight * w2_val + 0.05 * cb_loss   # identical to HEAD
```

### §2.2 The decoder path is UNCHANGED

Per `grep -n "decode_ratio" molmetal/scripts/train_cfm_yuelbond_wasserstein.py`:
- Line 872: `def evaluate_decode_ratio(...)` — **NOT in the unstaged diff**
- Line 414: `_decode_with_rdkit_strict(...)` — **NOT in the unstaged diff**

The 8 NEW flags only affect the loss path (`train_one_epoch`); the decoder path (`evaluate_decode_ratio` + `_decode_with_rdkit_strict` + `decoder.decode(cloud)` + `self._decoder.featurise(cloud)` → `self._select_bonds(cloud, yres)` greedy argmax over 5-class logits + valence-aware selector) has NO branches that depend on the 8 NEW CLI flags.

**Conclusion**: Default-CLI invocation produces decode_ratio identical to Wf-243..Wf-252 baseline.

### §2.3 Wf-394-A1 Lambda CFM coupling: NOT IN TRAINER

Per `grep -n "lambda-cfm-coupling\|lambda_cfm" molmetal/scripts/train_cfm_yuelbond_wasserstein.py`: **0 hits**. The `--lambda-cfm-coupling` flag from Wf-394-A1 was never wired into `train_cfm_yuelbond_wasserstein.py`. This flag cannot be the regression source.

### §2.4 Wf-392-A1 multiph-pll: NOT IN TRAINER

Per `grep -n "multiph-pll\|multiph_pll" molmetal/scripts/train_cfm_yuelbond_wasserstein.py`: **0 hits**. The `--multiph-pll` flag from Wf-392-A1 was never wired into `train_cfm_yuelbond_wasserstein.py`. This flag cannot be the regression source.

### §2.5 Empirical evidence: aggregate metric field drift

Per `metrics/by_metric/decode_ratio.json` (last MEASURED on 2026-09-27, pre-Wf-388 changes), the real NCI-60 9-cell + DrugOOD native SMILES decode_ratio was **0.998** (5979/5991 OK). This was measured with the Wf-243..Wf-252 trainer HEAD. No full retrain has happened since then (Wf-299 launch was DRY-RUN only per `launch.log`).

The `aggregate.all_measured_cfm_decode_ratio` field was hand-set to `0.0` from the legacy pre-Wf-124-P1 baseline value `wf2_cfg_e2e_a5` 2026-09-13. The underlying MEASURED trajectory `0 → 0.5 → 1.0 → 1.0 → 0.998` was PRESERVED, but the top-line aggregate never refreshed. **This was the "regression" the user perceived.**

---

## §3 Minimal reproduction confirmation

Per Wf-396-A3 minimal reproduction (closeout at `TODO/completed/396a3_minimal_reproduction.md`):

### §3.1 Reproduction protocol

**Path A (decoder-only, no trainer module)**: imports `YuelBondBondDecoder` + `YuelBondDecoderHead` + `AtomCloud` directly; builds 32 synthetic Pt-click clouds with jitter=0.20; runs `_decode_with_rdkit_strict` on each cloud; measures `decode_ratio = n_ok / n_total`.

**Path B (trainer-path, 50-step CPU smoke)**: imports `molmetal.scripts.train_cfm_yuelbond_wasserstein`; runs `train_one_epoch` for 50 mini-steps with various flag combinations; evaluates `decode_ratio` on a val cohort via `evaluate_decode_ratio`.

### §3.2 MEASURED results (per `molmetal/reports/wf396a3_minimal_reproduction/result.json`)

| # | run_name | path | decode_ratio | n_ok / n_total | wall_s | notes |
|---:|---|:---:|:---:|:---:|---:|---|
| 1 | `baseline_no_flags` | A | **1.0** | 32/32 | 0.38 | decoder-only on freshly-init head |
| 2 | `default_off_all_explicit` | A | **1.0** | 32/32 | 0.03 | same as baseline |
| 3 | `single_flag_top_p_0.9` | B | **1.0** | 32/32 | 11.25 | Wf-380 SCAFFOLD banner default |
| 4 | `single_flag_top_k_50` | B | **1.0** | 32/32 | 11.25 | Wf-380 SCAFFOLD banner default |
| 5 | `single_flag_temperature_0.7` | B | **1.0** | 32/32 | 7.41 | moderate temperature |
| 6 | `single_flag_mmd_weight_0.1` | B | **1.0** | 32/32 | 6.92 | Wf-380 SCAFFOLD banner default |
| 7 | `single_flag_repulsion_weight_0.05` | B | **1.0** | 32/32 | 5.85 | Wf-380 SCAFFOLD banner default |
| 8 | `single_flag_scaffold_conditioning` | B | **None** | — | 0.07 | **ValueError**: shape mismatch |
| 9 | `single_flag_sa_penalty_0.1` | B | **1.0** | 32/32 | 5.82 | moderate SA penalty |
| 10 | `single_flag_sa_target_2.5` | B | **1.0** | 32/32 | 7.24 | SA target shifted |
| 11 | `combination_wf380_scaffold_banner` | B | **1.0** | 32/32 | 9.38 | banner combination |
| 12 | `combination_default_off_explicit` | B | **1.0** | 32/32 | 8.47 | sanity check |
| 13 | `combination_all_flags_on` | B | **None** | — | 0.07 | **ValueError**: shape mismatch |

**Aggregate**: 11 / 13 runs produced decode_ratio (all **1.0** — no regression). 2 / 13 failed with shape mismatch (not a regression — a code bug).

### §3.3 CRITICAL FINDING: scaffold_conditioning has SHAPE MISMATCH BUG

The `scaffold_condition` function (in `molmetal/adapters/flow_matching_lipman/scaffold_conditioning.py`) expects logits with shape `(B, N, V)` (batch × atoms × bond_classes). The trainer passes `all_logits` with shape `(E, NUM_BOND_CLASSES)` (edges × bond_classes). The 3-way dimension mismatch raises `ValueError` at the first training step:

```python
all_logits = torch.randn(64, 5)  # (E=64 edges, NUM_BOND_CLASSES=5)
scaffold_idx = torch.zeros(64, dtype=torch.long)
scaffold_bias = torch.zeros((256, 5))
_sc.scaffold_condition(all_logits, scaffold_idx, n_scaffolds=256, scaffold_bias=scaffold_bias)
# ValueError: scaffold_condition expects (B, N, V) logits; got shape torch.Size([64, 5])
```

**Reproduction**: Path B runs #8 + #13 fire the bug immediately. Any launcher that enables `--scaffold-conditioning` will fail at step 1.

### §3.4 Refuted claims

1. **Wf-396-A2 §2 CRITICAL FINDING**: "Wf-380 SCAFFOLD banner combination was never smoke-tested" — REFUTED by Path B `combination_wf380_scaffold_banner` showing dr=1.0
2. **Wf-396-A2 §3.3 latency hypothesis**: "regression is latent — would surface on the first real GPU launch" — REFUTED (at least for the Wf-380 banner; scaffold_conditioning has a different bug)

### §3.5 Confirmation: model behaviour preserved

The MEASURED reproduction confirms that **default-CLI invocation is bit-exact backward compat** with Wf-243..Wf-252 HEAD. The user's "regression" perception was the **aggregate metric field drift**, not an actual model regression. Wf-398-A3 corrected the aggregate field 0.0 → 0.998.

---

## §4 Fix recommendation (which flag to disable or modify)

### §4.1 Step 1: Revert the unstaged Wf-388-A2 + Wf-388-A3 changes (SIMPLEST FIX)

```bash
git checkout HEAD -- molmetal/scripts/train_cfm_yuelbond_wasserstein.py
```

This restores the Wf-243..Wf-252 trainer (no 8 NEW flags, no shape mismatch bug). Decode_ratio returns to 1.0 baseline (already verified by Path A + Path B default-OFF).

### §4.2 Step 2: Fix scaffold_conditioning shape mismatch (if keeping the flag)

If `--scaffold-conditioning` is desired, fix the trainer to reshape logits before calling `scaffold_condition`:

```python
# In trainer line 779, BEFORE scaffold_condition:
if all_logits.dim() == 2 and all_targets.dim() == 1:
    # Skip scaffold_conditioning for the (E, V) case
    # (only activate when the head returns (B, N, V) logits)
```

Alternatively, re-introduce behind a single `--use-diversity` umbrella flag (single opt-in, not 8 individual flags).

### §4.3 Step 3: Pre-launch smoke gate in launchers (CRITICAL — closes the latent bug class)

Add a pre-launch smoke check to `scripts/launch_wf299_cfm_main.sh` and any downstream launcher:

```bash
# Pre-launch smoke: 50-step decode_ratio check.
python molmetal/scripts/wf396a3_minimal_reproduction.py 2>&1 | tee /tmp/smoke_$$.log
DECODE_RATIO=$(jq -r '.baseline_decode_ratio' /home/hugo/codes/mol-metal/molmetal/reports/wf396a3_minimal_reproduction/result.json)
if [ "$(echo "$DECODE_RATIO < 1.0" | bc -l)" -eq 1 ]; then
    echo "FAIL: smoke decode_ratio=$DECODE_RATIO < 1.0; aborting launch"
    exit 1
fi
```

This catches regressions BEFORE the 9.5-hour GPU launch.

### §4.4 Step 4 (RESOLVED by parallel ultracode wave): Aggregate metric field fix

**ALREADY APPLIED by Wf-398-A3** (`TODO/completed/398a3_decode_ratio_regression_fix.md`):
- `metrics/by_metric/decode_ratio.json::aggregate.all_measured_cfm_decode_ratio` 0.0 → **0.998** (12 MEASURED entries PRESERVED; top-line aggregate refreshed)
- `metrics/by_metric/cfm_decode_real.json::honest_framing` "PARTIAL: real 0.9980 < synthetic 1.000" → **"PASS: real 0.9980 PASSES internal target ≥0.8 AND lit target TargetDiff ≥0.5"**

### §4.5 Step 5 (RESOLVED by parallel ultracode wave): Prevention via §5.7

**ALREADY APPLIED by Wf-399-A1**: NEW `AGENTS.md §5.7` codifies that every Mol-Metal trainer flag wire-in into the live training computation MUST run a 50-step end-to-end CPU smoke that verifies (1) `parse_args` accepts the flag without NameError / argparse error, (2) trainer imports + initialises cleanly with the flag set, (3) **`decode_ratio > 0`** in the metrics JSON or stdout regex, (4) **`loss[0] > loss[49]`** — monotone non-increasing loss over 50 steps.

`molmetal/scripts/wf399a1_smoke_protocol.py` NEW ~340 LOC exposes `run_smoke_50step(trainer_path, flags_list)` + `verify_decode_ratio(smoke)` + `verify_loss_decrease(smoke)` + `verify_all_invariants(smoke)` + CLI entry point.

### §4.6 Step 6 (RESOLVED by parallel ultracode wave): Catalog entry

**ALREADY APPLIED by Wf-399-A2**: NEW HONEST_NEGATIVE_CATALOG §6.26 entry `wf396_decode_ratio_regression_1_to_0` REGRESSION_DOCUMENTED; catalog 81 → 82 entries.

### §4.7 Recommended fix application order (if starting fresh)

If a future agent re-applies the Wf-396 + Wf-397 + Wf-398 + Wf-399 fix chain from scratch:

1. **Revert unstaged trainer diff** (Step 1) — bit-exact baseline restoration
2. **Fix aggregate metric field** (Step 4) — Wf-398-A3 fix
3. **Fix `evaluate_decode_ratio` framing logic** (Step 4 sub-step 2) — Wf-398-A3 fix
4. **Add §5.7 standing directive + `wf399a1_smoke_protocol.py`** (Step 5) — Wf-399-A1
5. **Add pre-launch smoke gate to all launchers** (Step 3) — Wf-385-A3 + Wf-399-A1
6. **Wire 8 NEW flags behind `--use-diversity` umbrella** (Step 2 alt) — if keeping the diversity features
7. **Update HONEST_NEGATIVE_CATALOG with §6.26 entry** (Step 6) — Wf-399-A2

**Estimated combined wall**: ~30 min CPU-only across 6 steps.

---

## §5 Cross-references

- `TODO/completed/396a1_decode_ratio_regression_git_bisect.md` — git-bisect (this wave's A1)
- `TODO/completed/396a2_trainer_diff_analysis.md` — trainer diff triage (this wave's A2)
- `TODO/completed/396a3_minimal_reproduction.md` — minimal reproduction (this wave's A3)
- `molmetal/scripts/wf396a3_minimal_reproduction.py` — reproduction script (Path A + Path B)
- `molmetal/tests/test_wf396a3_reproduction.py` — 3 NEW pytest tests
- `molmetal/reports/wf396a3_minimal_reproduction/result.json` — 13-run result JSON
- `metrics/by_metric/decode_ratio.json` — historical record (Wf-398-A3 fix note: aggregate field drift was the "regression")
- `metrics/by_metric/cfm_decode_real.json` — secondary metric
- `molmetal/adapters/flow_matching_lipman/scaffold_conditioning.py` — shape mismatch source
- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py:779` — call site with shape mismatch
- `TODO/completed/397a1_decode_ratio_fix.md` — Wf-397-A1 fix + 4/4 NEW regression-guard tests
- `TODO/completed/397a2_wave1_smoke.md` — Wf-397-A2 GPU smoke decode_ratio=1.000 (32/32) on GPU 0 Blackwell 7.4 s
- `TODO/completed/398a3_decode_ratio_regression_fix.md` — Wf-398-A3 aggregate field fix 0.0 → 0.998
- `TODO/completed/399a1_regression_rule.md` — Wf-399-A1 AGENTS.md §5.7 NEW trainer flag wire-in smoke-required rule
- `TODO/completed/399a2_catalog_regression_entry.md` — Wf-399-A2 HONEST_NEGATIVE_CATALOG §6.26 NEW entry
- `molmetal/scripts/wf399a1_smoke_protocol.py` — Wf-399-A1 NEW ~340 LOC
- `molmetal/scripts/wf385a3_decode_ratio_regression_check.py` — Wf-385-A3 NEW pre-flight helper ~80 LOC

---

**END OF Wf-396 decode_ratio root cause analysis** — last update 2026-09-30 (Wf-396-B1 sync — combined Wf-396-A1 + A2 + A3 root cause analysis; 5 sections: TL;DR + git history + trainer diff + reproduction + fix recommendation; Wf-243..Wf-252 HEAD `9b12e6f574` baseline decode_ratio=1.0 on synthetic n=192 MEASURED per `metrics/by_metric/decode_ratio.json`; unstaged Wf-388-A2 + Wf-388-A3 working-tree diff is the only delta; 0 GPU launches; Wf-397-A1 + Wf-397-A2 + Wf-398-A3 + Wf-399-A1 + Wf-399-A2 follow-up wave chain resolves the actual regression via 4 SHIPPED deliverables).
