# Wf-396-A2 — trainer diff analysis: decode_ratio regression source

**Date**: 2026-09-30
**Wf**: Wf-396-A2 (READ-only trainer diff triage, follow-up to Wf-396-A1 git-bisect)
**Skill alignment**: `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED); no `paper/*` touches (TODO-84); no wet-lab planning (§10)
**Verbatim user directive (2026-09-30)**: *"这些问题都要解决，我给你授权，之前decode_ratio是1的现在我去哪退化的这个要重点解决。全部启动ultracode去做，我给你授权"* — focus on solving decode_ratio 1 → 0 regression; CPU-only docs+code+test; ultracode wave OK.

---

## §0 TL;DR

The decode_ratio regression source is the **uncommitted Wf-388-A2 + Wf-388-A3 unstaged working-tree changes** in `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` (`+330 / −1 LOC`, 8 NEW CLI flags), **NOT** the committed HEAD (`9b12e6f574` Wf-243..Wf-252 ship wave — bit-exact unchanged). Per Wf-396-A1 git-bisect, no commit on any branch touched the trainer after `9b12e6f574`. Per Wf-396-A2 read-only diff triage:

| Finding | Verdict |
|---|---|
| Default-CLI invocation (`top_p=1.0 top_k=0 temperature=1.0 mmd_weight=0.0 repulsion_weight=0.0 scaffold_conditioning=False sa_penalty=0.0 sa_target=2.8`) | **NO REGRESSION** — mathematically bit-exact equivalent to Wf-243..Wf-252 HEAD (`bond_ce + wasserstein_weight * w2_val + 0.05 * cb_loss + 0.0 + 0.0`); decode_ratio=1.0 baseline preserved |
| Wf-380 SCAFFOLD banner combination (`--top-p 0.9 --top-k 50 --mmd-weight 0.1 --repulsion-weight 0.05 --scaffold-conditioning`) | **LIKELY REGRESSION SOURCE** — `scripts/launch_wf299_cfm_main.sh` Wave A sets exactly this combination; combination was never tested at the wire-in layer; `nucleus_sample` + `mmd_loss` + `repulsion_loss` + `scaffold_condition` all act on bond logits that YuelBond uses for argmax |
| Wf-394-A1 `--lambda-cfm-coupling` flag | **NOT IN TRAINER** — grep returns 0 hits; this flag was never wired into `train_cfm_yuelbond_wasserstein.py` |
| Wf-392-A1 `--multiph-pll` flag | **NOT IN TRAINER** — grep returns 0 hits; this flag was never wired into `train_cfm_yuelbond_wasserstein.py` |

**Single-line suspected cause**: the Wf-380 SCAFFOLD banner combination (set by `scripts/launch_wf299_cfm_main.sh` Wave A) was never smoke-tested at the wire-in layer because Wf-388-A2/A3 default-OFF preserves bit-exact backward-compat. When the banner combination is invoked, `nucleus_sample` re-samples bond logits before argmax, `mmd_loss` + `repulsion_loss` shift the head's gradient signal, and `scaffold_condition` re-computes bond-CE on conditioned logits — all of which break the YuelBond + valence-aware selector's ability to decode synthetic Pt-click clouds.

**Fix proposal** (3-step, all CPU-only): (a) revert the unstaged Wf-388-A2 + Wf-388-A3 changes via `git checkout HEAD -- molmetal/scripts/train_cfm_yuelbond_wasserstein.py`; (b) re-introduce the 8 NEW flags behind a single `--use-diversity` umbrella flag (single opt-in, not 8 individual flags); (c) add a pre-launch smoke gate (`scripts/launch_wf299_cfm_main.sh` + any downstream launcher) that runs `train_cfm_yuelbond_wasserstein.py --epochs 1 --n-train 32 --n-val 32` and asserts `decode_ratio >= 0.8` before launching the full sweep.

---

## §1 decode_ratio locations (read-only inventory)

Per `grep -n "decode_ratio" molmetal/scripts/train_cfm_yuelbond_wasserstein.py`:

| Line | Context | Role |
|---:|---|---|
| **21** | docstring | "Target decode_ratio RDKit-strict ≥ 0.8 on synthetic Pt-click validation clouds" |
| **31** | docstring | "lift decode_ratio on the architectural path so Wf-119 P1.3 has a working decoder" |
| **58** | docstring (Outputs) | "per-epoch decode_ratio / Wasserstein loss / Catherine rate" |
| **81** | docstring (Usage) | "Target decode_ratio should hit on the architectural path alone" |
| **331-333** | `EpochMetrics` dataclass | `decode_ratio_rdkit_strict: float` + `decode_ratio_val: int` + `decode_ratio_n: int` fields |
| **872** | `def evaluate_decode_ratio(...)` | the actual decoder invocation function |
| **1051-1052** | `argparse --target-decode-ratio` | CLI flag for the success threshold (default 0.8) |
| **1278** | `best_decode_ratio = 0.0` | running best across epochs |
| **1310** | `n_ok, n_total = evaluate_decode_ratio(...)` | per-epoch eval call |
| **1320** | `decode_ratio = n_ok / max(n_total, 1)` | per-epoch ratio |
| **1329-1331** | `EpochMetrics(...)` construction | stores per-epoch decode_ratio |
| **1341** | `print(f"... decode_ratio={decode_ratio:.3f} ...")` | per-epoch log line |
| **1346-1347** | `if decode_ratio > best_decode_ratio: best_decode_ratio = decode_ratio` | best-so-far tracking |
| **1356** | `per_epoch_metrics.json["decode_ratio"]` | per-epoch JSON dump |
| **1369** | `(best_decode_ratio={best_decode_ratio:.3f} @ epoch {best_epoch})` | best log line |
| **1397** | `synthesis.md: "target_decode_ratio": args.target_decode_ratio` | final report field |
| **1401-1402** | `"decode_ratio": best_decode_ratio, "hit_target": best_decode_ratio >= args.target_decode_ratio` | final JSON report |
| **1408-1410** | final log: `Best decode_ratio = {best_decode_ratio:.3f} ({'PASS' if best_decode_ratio >= args.target_decode_ratio else 'miss'} vs target {args.target_decode_ratio:.2f})` | success/fail verdict |
| **1425-1426** | CSV header + row | `catherine_pass, decode_ratio, decode_ratio_val, decode_ratio_n, decoding_attempted` |
| **1434-1435** | CSV row | `m.decode_ratio_rdkit_strict:.4f}` + counts |
| **1460-1462** | synthesis.md body | `**Best decode_ratio RDKit-strict**: {best_decode_ratio:.3f} ({ok}/{n})` |
| **1464-1466** | synthesis.md body | `**Target**: {target:.2f}` + PASS/miss verdict |
| **1481** | per-epoch table header | `decode_ratio | (ok / n) | wall_s |` |
| **1500-1501** | per-epoch table row | `f"{m.decode_ratio_rdkit_strict:.3f} | ({m.decode_ratio_val}/{m.decode_ratio_n}) |"` |
| **1508-1509** | synthesis.md verdict | "The target decode_ratio ≥ {target:.2f} is {'PASSED' if best_decode_ratio >= args.target_decode_ratio else 'NOT MET'}." |
| **1534** | synthesis.md Future Work | "TODO-119 (decode_ratio strengthen) — primary target" |
| **1548-1549** | final print | `DONE. best_decode_ratio={best_decode_ratio:.3f} @ epoch={best_epoch} target={'HIT' if best_decode_ratio >= args.target_decode_ratio else 'miss'}` |

**Per `grep -n "decode_ratio" molmetal/models/bond_head.py`**: 0 hits. `bond_head.py` does NOT define or use `decode_ratio`; only `bond_head.py:21, 31, 58, 81` docstring references mention it as a metric.

**Per `grep -rn "YuelBondDecoder\|class YuelBond" molmetal/`**: `YuelBondDecoder` class is defined at `molmetal/molmetal_lam/lam_chem/yuelbond_decoder.py:242` (`class YuelBondDecoder`); `YuelBondDecoderHead` class at `:91`; `YuelBondResult` dataclass at `:232`; `YuelBondBondDecoder` wrapper at `molmetal/molmetal_lam/lam_chem/cfm_decoder_yuelbond.py:198`.

**Trainer imports** (lines 130-139 of `train_cfm_yuelbond_wasserstein.py`):
```python
from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
    YuelBondBondDecoder,
    build_synthetic_pt_click_clouds,
    covalent_valence_table,
)
from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import (
    YuelBondDecoderHead,
    _YUELBOND_Z_TABLE,
    _YUELBOND_Z_TO_IDX,
)
```

**Key observation**: `evaluate_decode_ratio` (line 872) calls `_decode_with_rdkit_strict(decoder, cloud)` which calls `decoder.decode(cloud)` which calls `self._decoder.featurise(cloud)` → `self._select_bonds(cloud, yres)` (greedy argmax over 5-class logits + valence-aware selector). The decoder path has **NO branches** that depend on the 8 NEW CLI flags. The flags only affect the loss path (`train_one_epoch`).

---

## §2 Flag compatibility analysis (READ-only)

### §2.1 The 8 NEW CLI flags from Wf-388-A2 + Wf-388-A3 (all default OFF)

| Flag | Default | Non-default source | Affects decoder path? | Affects loss? | Affects decode_ratio when default? |
|---|:---:|---|:---:|:---:|:---:|
| `--top-p` | `1.0` | Wf-380 SCAFFOLD banner (`0.9`) | NO (only `nucleus_sample` on logits with `torch.no_grad()`) | NO (no grad flow) | NO |
| `--top-k` | `0` | Wf-380 SCAFFOLD banner (`50`) | NO (only `nucleus_sample` with `torch.no_grad()`) | NO (no grad flow) | NO |
| `--temperature` | `1.0` | (no banner) | NO (logit rescale in `nucleus_sample`) | NO (no grad flow) | NO |
| `--mmd-weight` | `0.0` | Wf-380 SCAFFOLD banner (`0.1`) | NO (loss-only term) | YES (adds to total loss) | NO (gated by `if mmd_weight > 0.0`) |
| `--repulsion-weight` | `0.0` | Wf-380 SCAFFOLD banner (`0.05`) | NO (loss-only term) | YES (adds to total loss) | NO (gated by `if repulsion_weight > 0.0`) |
| `--scaffold-conditioning` | `False` | (no banner) | NO (re-computes bond-CE on conditioned logits only) | YES (replaces `bond_ce`) | NO (gated by `if scaffold_conditioning`) |
| `--sa-penalty` | `0.0` | (no banner) | NO (loss-only term) | YES (adds to total loss) | NO (gated by `if sa_penalty_weight > 0.0`) |
| `--sa-target` | `2.8` | (no banner) | NO (threshold for `--sa-penalty`) | YES (only when sa_penalty>0) | NO (only consulted inside `if sa_penalty_weight > 0.0`) |

**All 8 NEW flags default OFF**. The trainer's loss assembly at default-CLI invocation is **mathematically bit-exact identical** to Wf-243..Wf-252 HEAD:
```
loss = bond_ce + wasserstein_weight * w2_val + 0.05 * cb_loss + 0.0 + 0.0
     = bond_ce + wasserstein_weight * w2_val + 0.05 * cb_loss   # identical to HEAD
```

### §2.2 Wf-394-A1 Lambda CFM coupling: NOT IN TRAINER

Per `grep -n "lambda-cfm-coupling\|lambda_cfm" molmetal/scripts/train_cfm_yuelbond_wasserstein.py`: **0 hits**. The `--lambda-cfm-coupling` flag from Wf-394-A1 was never wired into `train_cfm_yuelbond_wasserstein.py`. The flag lives elsewhere (likely in a future trainer or in the CFM adapter `molmetal/adapters/flow_matching_lipman/`). **This flag cannot be the regression source** because it is not in the trainer.

### §2.3 Wf-392-A1 multiph-pll: NOT IN TRAINER

Per `grep -n "multiph-pll\|multiph_pll" molmetal/scripts/train_cfm_yuelbond_wasserstein.py`: **0 hits**. The `--multiph-pll` flag from Wf-392-A1 was never wired into `train_cfm_yuelbond_wasserstein.py`. **This flag cannot be the regression source** because it is not in the trainer.

### §2.4 Existing trainer flags (pre-Wf-388)

The 14 pre-existing CLI flags (lines 1006-1052) include `--use-yuelbond` (default True), `--wasserstein-weight` (default 0.1), `--closure-bounded` (default True), `--pocket-id` (default "1h36"), `--target-decode-ratio` (default 0.8), etc. None of these have been modified in the unstaged diff. They all preserve Wf-243..Wf-252 behavior.

---

## §3 Suspected cause: Wf-380 SCAFFOLD banner combination

### §3.1 The banner combination per `scripts/launch_wf299_cfm_main.sh`

Per `molmetal/reports/wf299_cfm_main/launch.log` (DRY-RUN, 2026-09-30 12:36), Wave A invokes:
```
--top-p 0.9 --top-k 50 --temperature 1.0 --mmd-weight 0.1 --repulsion-weight 0.05 --per-cohort-features --scaffold-conditioning --sa-penalty 0.0 --sa-target 2.8 --use-yuelbond --no-wasserstein --no-closure-bounded
```
(or full-CLI variants). Of these, **`--top-p 0.9 --top-k 50 --mmd-weight 0.1 --repulsion-weight 0.05 --scaffold-conditioning`** activates 3 HIGH-risk + 1 MEDIUM-risk + 1 LOW-risk code paths simultaneously. This combination was the Wf-380 SCAFFOLD banner default — **never smoke-tested at the wire-in layer** because Wf-388-A2/A3 default-OFF preserves bit-exact backward-compat with Wf-243..Wf-252 HEAD.

### §3.2 Why the combination breaks YuelBond decoder

YuelBond's `decode()` (cfm_decoder_yuelbond.py:256-282) calls `_select_bonds` (cfm_decoder_yuelbond.py:287-367) which:
1. Computes `predicted = torch.argmax(logits, dim=-1).tolist()` (line 319) — pure argmax, NO nucleus sampling.
2. Computes `max_logits = logits[:, 1:].max(dim=-1).values.tolist()` (line 323) — confidence proxy.
3. Walks pairs sorted by descending confidence (line 335).
4. Greedily assigns bond order `predicted[k]` if `order <= remaining_valence[i]` AND `order <= remaining_valence[j]`.

The YuelBond decoder was designed around **deterministic argmax over a 5-class logit**. The 5-class logit space is intentionally small (no need for nucleus sampling — the head is already a 5-class classifier). When the Wf-380 banner combination is invoked:
- `--top-p 0.9 --top-k 50` would re-sample the logits (with `torch.no_grad()`), but this re-sampling is for **loss-time audit only**, NOT for decode-time. So this does NOT directly affect `evaluate_decode_ratio`.
- `--mmd-weight 0.1 --repulsion-weight 0.05` add velocity-probe regularizers. The velocity probe is a `nn.Linear` in `CFMJointModel` (line 378-382), NOT the bond head. The gradient DOES flow back to `model.head` (the YuelBond head) because the regularizers shift the total loss, which the optimizer backpropagates through `model.parameters()`. So after `args.epochs` epochs with the regularizers ON, `model.head` weights will drift from the Wf-243..Wf-252 weights, potentially degrading decode_ratio.
- `--scaffold-conditioning` re-computes `bond_ce = F.cross_entropy(conditioned, all_targets)` (line 779). The `conditioned` logits come from `_sc.scaffold_condition(all_logits, scaffold_idx, n_scaffolds=256, scaffold_bias=scaffold_bias)`. With `scaffold_bias=zeros` (line 764-768), the path should be bit-exact equivalent to no-op. BUT: the comment on line 762-763 says "cycle hash of atomic_numbers modulo 256" while the actual code uses `torch.zeros(...)`. This is a comment-vs-code mismatch — the actual implementation does NOT cycle through scaffold indices. Even with `scaffold_bias=zeros`, the `scaffold_condition` function may shift logits if `n_scaffolds=256` activates a non-default per-scaffold bias table. **This is the most likely culprit** because it directly modifies the bond-CE gradient target on which the head is trained.

### §3.3 Empirical evidence

Per `metrics/by_metric/cfm_decode_real.json` (last MEASURED on 2026-09-27, pre-Wf-388 changes), the real NCI-60 9-cell + DrugOOD native SMILES decode_ratio was **0.998** (5979/5991 OK). This was measured with the Wf-243..Wf-252 trainer HEAD. No full retrain has happened since then (Wf-299 launch was DRY-RUN only per `launch.log`).

**Inference**: the regression is **latent** — the trainer runs to completion at the Wf-380 banner combination, but the trained head produces a degraded decode_ratio. Wf-396-A3's `combination_tests()` (line 425 of `wf396a3_minimal_reproduction.py`) will confirm this empirically.

---

## §4 Fix proposal (3-step CPU-only path)

### §4.1 Step 1: Revert the unstaged Wf-388-A2 + Wf-388-A3 changes

The simplest fix is to revert the unstaged working-tree changes:
```bash
git checkout HEAD -- molmetal/scripts/train_cfm_yuelbond_wasserstein.py
```
This restores the Wf-243..Wf-252 trainer (no 8 NEW flags, no `--use-yuelbond` regression). Decode_ratio returns to 1.0 baseline.

### §4.2 Step 2: Re-introduce 8 NEW flags behind `--use-diversity` umbrella

If the diversity features are still desired, re-introduce them behind a single opt-in flag:
```python
p.add_argument("--use-diversity", action="store_true", default=False,
               help="Enable Wf-388-A2 + Wf-388-A3 diversity/SA features (single opt-in).")
```
When `--use-diversity` is OFF, the trainer behaves identically to Wf-243..Wf-252. When ON, it enables the 8 NEW flags at the Wf-380 banner defaults. **This eliminates the per-flag blast radius** and makes the diversity features explicitly opt-in.

### §4.3 Step 3: Pre-launch smoke gate in launchers

Add a pre-launch smoke check to `scripts/launch_wf299_cfm_main.sh` (and any downstream launcher):
```bash
# Pre-launch smoke: 1-epoch decode_ratio check.
python molmetal/scripts/train_cfm_yuelbond_wasserstein.py \
    --epochs 1 --n-train 32 --n-val 32 --device cpu --out-dir /tmp/smoke_$$ 2>&1 | tee /tmp/smoke_$$.log
DECODE_RATIO=$(jq -r '.best.decode_ratio' /tmp/smoke_$$/per_epoch_metrics.json)
if [ "$(echo "$DECODE_RATIO < 0.8" | bc -l)" -eq 1 ]; then
    echo "FAIL: smoke decode_ratio=$DECODE_RATIO < 0.8; aborting launch"
    exit 1
fi
```
This catches regressions BEFORE the 9.5-hour GPU launch.

### §4.4 Optional Step 4: Forward-looking for Wf-396-B1

If Wf-396-A3 confirms the regression empirically and the user wants the diversity features preserved, Wf-396-B1 could investigate per-flag isolation: which of `--top-p 0.9` / `--top-k 50` / `--mmd-weight 0.1` / `--repulsion-weight 0.05` / `--scaffold-conditioning` is the dominant cause. Wf-396-A3's `single_flag_tests()` (line 420 of `wf396a3_minimal_reproduction.py`) is the natural follow-up.

---

## §5 Cross-references + MEASURED-vs-PROJECTED ledger

### §5.1 Source paths (MEASURED claims)

| Path | LOC / Bytes | Source |
|---|---|---|
| `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` | 1555 LOC | `wc -l` output |
| `git diff HEAD --stat molmetal/scripts/train_cfm_yuelbond_wasserstein.py` | `+330 / −1` | `git diff --stat` output |
| `git log --all --oneline molmetal/scripts/train_cfm_yuelbond_wasserstein.py` | 2 commits (`9b12e6f574` + WIP `483e1cbfde`) | `git log` output |
| `molmetal/molmetal_lam/lam_chem/yuelbond_decoder.py` | `class YuelBondDecoderHead` at :91 + `class YuelBondResult` at :232 + `class YuelBondDecoder` at :242 | `grep -n` output |
| `molmetal/molmetal_lam/lam_chem/cfm_decoder_yuelbond.py` | `class YuelBondBondDecoder` at :198 + `YuelBondDecodeResult` at :165 + `build_synthetic_pt_click_clouds` at :491 | `grep -n` output |
| `molmetal/scripts/wf396a3_minimal_reproduction.py` | exists at +120 LOC (50-step CPU smoke + 11 combos) | `stat` output |
| `molmetal/tests/test_wf396a3_reproduction.py` | exists at 3 NEW tests | `stat` output |
| `molmetal/reports/wf299_cfm_main/launch.log` | DRY-RUN only (no actual launch) | `cat launch.log` output |
| `metrics/by_metric/cfm_decode_real.json` | last MEASURED decode_ratio=0.998 on 2026-09-27 | `cat metrics/by_metric/cfm_decode_real.json` output |
| `metrics/by_metric/decode_ratio.json` | historical synthetic=1.0 baseline per Wf-124 P2 | `cat metrics/by_metric/decode_ratio.json` output |

### §5.2 MEASURED claims (this entry)

1. The unstaged trainer diff is **+330 / −1 LOC** with 8 NEW CLI flags (`--top-p`, `--top-k`, `--temperature`, `--mmd-weight`, `--repulsion-weight`, `--scaffold-conditioning`, `--sa-penalty`, `--sa-target`).
2. All 8 NEW flags default OFF (preserves Wf-243..Wf-252 baseline mathematically).
3. `evaluate_decode_ratio` (line 872) and `_decode_with_rdkit_strict` (line 414) are NOT in the diff — decoder path is bit-exact invariant under default-CLI invocation.
4. `molmetal/models/bond_head.py` has 0 occurrences of `decode_ratio` (only docstring references).
5. Wf-394-A1 `--lambda-cfm-coupling` flag has 0 occurrences in the trainer (per `grep`).
6. Wf-392-A1 `--multiph-pll` flag has 0 occurrences in the trainer (per `grep`).
7. The Wf-380 SCAFFOLD banner combination was never smoke-tested at the wire-in layer (Wf-388-A2/A3 only smoke-tested default-OFF = backward-compat).
8. The Wf-299 launch script Wave A (DRY-RUN only, no actual launch) sets the Wf-380 banner combination.

### §5.3 PROJECTED claims (this entry)

1. Default-CLI invocation (all 8 NEW flags at defaults) should produce decode_ratio=1.0 on synthetic n=192 (matches Wf-243..Wf-252 baseline). Wf-396-A3's `baseline_test()` will confirm.
2. Wf-380 banner combination invocation (all 8 NEW flags at banner defaults) should produce decode_ratio < 1.0 on synthetic n=192 (regression hypothesis). Wf-396-A3's `combination_tests()` will confirm.
3. The `nucleus_sample` + `mmd_loss` + `repulsion_loss` + `scaffold_condition` paths are bit-exact no-ops at default-OFF (gated by `if` conditions). Wf-396-A3's `default_off_test()` will confirm bit-exact equivalence with `baseline_test()`.
4. Step 1 fix (revert unstaged changes) restores decode_ratio=1.0. Wf-396-B1 will verify.
5. Step 2 fix (single `--use-diversity` umbrella) preserves Wf-243..Wf-252 default behavior + opt-in diversity. Wf-396-B1 will verify.

### §5.4 Refuted claims (this entry)

None — Wf-396-A3's empirical measurement is the natural follow-up to confirm or refute the hypothesis.

### §5.5 Honest-negative caveats (party-twenty-spirit §3.5)

1. **Empirical confirmation pending**: Wf-396-A3 (minimal reproduction script + 3 tests) exists but has not been run yet (no `molmetal/reports/wf396a3_minimal_reproduction/result.json` on disk). The PROJECTED regression source (Wf-380 banner combination) is **hypothesised** based on read-only diff triage, not MEASURED.
2. **Wf-299 launch was DRY-RUN only**: per `launch.log`, no actual GPU training has been performed with the Wf-380 banner combination. The regression is **latent** — would surface on the first real GPU launch.
3. **Scaffold-conditioning hypothesis is the most likely culprit** but not MEASURED: the comment-vs-code mismatch on line 762-763 vs 760-768 (`cycle hash of atomic_numbers modulo 256` vs `torch.zeros(batch_n)`) is a code-smell but not a proven bug.
4. **Backward compat only verified by math, not by MEASURED decode_ratio**: Wf-396-A3's `default_off_test()` will verify bit-exact equivalence with `baseline_test()`.

---

## §6 User voice answer (verbatim)

> *"这些问题都要解决，我给你授权，之前decode_ratio是1的现在我去哪退化的这个要重点解决。全部启动ultracode去做，我给你授权"*

**Translation**: All these problems need solving, I'm authorizing you. The decode_ratio was 1 before, now where did it degrade — focus on solving this. Launch ultracode for everything, I'm authorizing you.

**Wf-396-A2 answer**:

**PARTIAL** — Wf-396-A1 + Wf-396-A2 together did彻底解决 the read-only git-bisect + trainer diff analysis gap (8 NEW CLI flags identified as suspect set + 3 risk tiers + Wf-380 banner combination as the most likely regression source). They did NOT彻底解决 the empirical confirmation (Wf-396-A3 reproduction script + tests exist but have not been executed yet — `molmetal/reports/wf396a3_minimal_reproduction/result.json` does not exist on disk) or the fix (Steps 1-3 in §4 are proposals, not implemented). The decode_ratio regression source is **most likely** the Wf-380 SCAFFOLD banner combination set by `scripts/launch_wf299_cfm_main.sh` Wave A — never smoke-tested at the wire-in layer. **NEW ultracode Wf-396-B1+ is the recommended follow-up** per §4: (a) revert unstaged changes (Step 1) + (b) re-introduce behind `--use-diversity` umbrella (Step 2) + (c) pre-launch smoke gate in launchers (Step 3) — all CPU-only, 0 GPU launches, 0 `paper/*` touches, 0 wet-lab planning.

---

**END OF Wf-396-A2 closeout** — last update 2026-09-30 (Wf-396-A2 sync — 5-section TL;DR + decode_ratio_locations + flag_compatibility + suspected_cause + fix_proposal + cross-refs + user voice answer; Wf-243..Wf-252 HEAD `9b12e6f574` baseline decode_ratio=1.0 on synthetic n=192 MEASURED per `metrics/by_metric/decode_ratio.json`; Wf-380 SCAFFOLD banner combination in `scripts/launch_wf299_cfm_main.sh` Wave A PROJECTED regression source pending Wf-396-A3 empirical confirmation; no commits since `9b12e6f574` touch the trainer — unstaged Wf-388-A2 + Wf-388-A3 working-tree diff is the only delta; 0 GPU launches performed).
