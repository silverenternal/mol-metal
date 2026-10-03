# Wf-396-A1 — decode_ratio regression git-bisect

**Date**: 2026-09-30
**Wf**: Wf-396-A1 (git-bisect on the trainer file)
**Skill alignment**: `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED); no `paper/*` touches (TODO-84); no wet-lab planning (§10)

---

## §0 TL;DR

Git-bisect on `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` identifies the
**Wf-388-A2 + Wf-388-A3 unstaged working-tree changes** as the candidate regression
source.  The committed HEAD is `9b12e6f574` (Wf-243..Wf-252 ship wave — pre-retrain
code fixes + click-aug wire + F5 32-d + Master modules M1/M2 + m18 per-residue +
W₂ pIC50 + 145 tests passing).  Working-tree diff is **+330 / −1 lines** that
adds **8 NEW CLI flags** (`--top-p`, `--top-k`, `--temperature`, `--mmd-weight`,
`--repulsion-weight`, `--scaffold-conditioning` from Wf-388-A2 + `--sa-penalty`,
`--sa-target` from Wf-388-A3) wired into `train_one_epoch`.  **No committed
regression between Wf-243..Wf-252 and HEAD** (the only diff to the trainer on
the master branch is the Wf-243..Wf-252 commit, no Wf-296..Wf-379 commit
touched the trainer).

---

## §1 Git-bisect methodology

Per `party-twenty-spirit` §3.5 + `Wf-302A` AnchorGuard discipline: every claim
must trace to a JSON / report path on disk.  No PROJECTED claims without
MEASURED evidence on this entry.

**Method** (read-only):
1. `git log --all --oneline molmetal/scripts/train_cfm_yuelbond_wasserstein.py` —
   confirms only **1 commit** has touched the file in the past 30 days:
   `9b12e6f574` (Wf-243..Wf-252 ship wave, 2026-09-29 ish).
2. `git diff HEAD molmetal/scripts/train_cfm_yuelbond_wasserstein.py` —
   surfaces the **+330 / −1 LOC unstaged diff** in the working tree.
3. `git log --all --oneline --since="2026-09-01" -- molmetal/adapters/flow_matching_lipman/diversity_sampling.py molmetal/adapters/flow_matching_lipman/scaffold_conditioning.py molmetal/adapters/flow_matching_lipman/sa_penalty.py molmetal/molmetal_lam/lam_chem/cfm_decoder_yuelbond.py`
   — confirms no decoder or import-side regressions on HEAD.
4. `git diff HEAD --stat` — only ONE Python file has unstaged changes
   (`molmetal/scripts/train_cfm_yuelbond_wasserstein.py`).

**Verdict**: the regression candidate is the **uncommitted Wf-388-A2 + Wf-388-A3
wire-in changes** to the trainer, currently sitting in the working tree.

---

## §2 Commit inventory

| Commit | Date | Author | File | Subject |
|---|---:|---|---|---|
| `9b12e6f574` | 2026-09-29 | Hugo Lee | `train_cfm_yuelbond_wasserstein.py` (+802/−21) | Wf-243..Wf-252 ship wave (the master-branch commit) |
| (none after) | — | — | — | no committed change since Wf-243..Wf-252 |

**git log on file** (read-only evidence):

```
9b12e6f574 Wf-243 to Wf-252: pre-retrain code fixes + click-aug wire + F5 32-d +
           Master modules M1/M2 + m18 per-residue + W₂ pic50 + 145 tests passing
483e1cbfde WIP on master: 9b12e6f574 Wf-243 to Wf-252: ...
```

The `483e1cbfde` row is an in-progress WIP commit (uncommitted staging), not a
separate ship.  No other commit on any branch touched this file.

---

## §3 Unstaged diff summary

`git diff HEAD --stat` reports **+330 / −1 LOC** in the trainer file, all in
the unstaged working tree:

```
molmetal/scripts/train_cfm_yuelbond_wasserstein.py | 330 ++++++++++++-
```

The diff is **purely additive** except for **one removed line** (`loss = bond_ce
+ wasserstein_weight * w2_val + 0.05 * cb_loss`) which is replaced by an
expanded assembly that adds **two new zero-cost terms** when flags default OFF
(`diversity_term` + `sa_penalty_weight * sa_loss`, both multiplied by 0.0 in
default state).

**Per-section breakdown**:
- **§A — train_one_epoch signature**: +6 NEW keyword params (`top_p`,
  `top_k`, `temperature`, `mmd_weight`, `repulsion_weight`,
  `scaffold_conditioning`) — all default to OFF semantics
  (`1.0` / `0` / `1.0` / `0.0` / `0.0` / `False`).
- **§B — train_one_epoch body**: +120 LOC gated by `if (any flag non-default):`
  — when all flags default, the `if` is `False` and the entire block is
  skipped (zero per-step cost).
- **§C — SA penalty module import**: +14 LOC `try / except` block
  (`_SA_PENALTY_AVAILABLE` flag), lazy-import.
- **§D — CLI parse_args**: +130 LOC adds 8 NEW CLI flags + their argparse
  entries (default OFF preserved).
- **§E — main() threading**: +14 LOC passes the 6 NEW params through to
  `train_one_epoch` via `getattr(args, ..., default)` — defensive for any
  older pickle / checkpoint compat.

**No logic changes** in `evaluate_decode_ratio` or `_decode_with_rdkit_strict`
(the decoder functions that compute the per-sample decode_ratio).  No removal
of existing branches or flags.

---

## §4 Suspected culprits (forwarded to Wf-396-A2)

Per Wf-396-A1's read-only git-bisect, the **8 NEW CLI flags** are the suspect
set.  Each flag is gateable individually so a flag-isolation smoke
(Wf-396-A3) can confirm or rule out each one:

| Flag | Default (Wf-388-A2/A3 wire-in) | Wf-380 SCAFFOLD banner | Forward-looking question |
|---|:---:|:---:|---|
| `--top-p` | `1.0` (no truncation) | `0.9` | Does `--top-p < 1.0` affect `evaluate_decode_ratio`? |
| `--top-k` | `0` (no truncation) | `50` | Does `--top-k > 0` affect `evaluate_decode_ratio`? |
| `--temperature` | `1.0` (no rescale) | `1.0` | Does `--temperature != 1.0` affect `evaluate_decode_ratio`? |
| `--mmd-weight` | `0.0` (MMD OFF) | `0.1` | Does `--mmd-weight > 0.0` affect decode_ratio via the loss path? |
| `--repulsion-weight` | `0.0` (repulsion OFF) | `0.05` | Does `--repulsion-weight > 0.0` affect decode_ratio via the loss path? |
| `--scaffold-conditioning` | `False` (OFF) | (n/a) | Does enabling `--scaffold-conditioning` re-write `bond_ce`? |
| `--sa-penalty` | `0.0` (OFF) | (n/a) | Does `--sa-penalty > 0.0` modify the decoder? |
| `--sa-target` | `2.8` (default threshold) | `2.8` | Does `--sa-target` shift the SA penalty boundary? |

**Forward-looking hypothesis** (Wf-396-A3 to confirm or refute):
- Default OFF flags SHOULD preserve Wf-243..Wf-252 decode_ratio=1.000 on
  synthetic 192-sample.
- Any flag at its NON-default value MAY degrade decode_ratio (expected for
  `--top-p < 1.0`, `--top-k > 0`, `--temperature != 1.0` — these are
  nucleus-sampling flags that change the discrete sample distribution).
- `--mmd-weight`, `--repulsion-weight`, `--scaffold-conditioning`, `--sa-penalty`
  affect the loss, not the decoder.  Decoder behaviour should be unchanged
  when these flags are at default 0.0/False.

---

## §5 Honest framing per party-twenty-spirit §3.5

**Source paths**:
- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` (working tree, +330/−1 LOC)
- `git log --all --oneline molmetal/scripts/train_cfm_yuelbond_wasserstein.py`
  (only `9b12e6f574` in the past 30 days; the `483e1cbfde` is the WIP
  in-progress commit, not a separate ship)

**MEASURED claims**:
1. The unstaged trainer diff is **+330 / −1 LOC** (one Edit replaces a single
   loss-assembly line).
2. **8 NEW CLI flags** wired into `train_one_epoch` (6 from Wf-388-A2 +
   2 from Wf-388-A3).
3. **All defaults preserve backward compat** mathematically
   (`diversity_term = 0.0` + `sa_penalty_weight * sa_loss = 0.0` when flags at
   defaults).

**PROJECTED claims**:
1. Default-OFF flags should NOT degrade decode_ratio on synthetic 192-sample
   (the loss assembly is mathematically identical to Wf-243..Wf-252 HEAD).
2. NON-default flags may degrade decode_ratio (especially `--top-p < 1.0`,
   `--top-k > 0`, `--temperature != 1.0` — these are sampling-time knobs).
3. The Wf-396-A3 minimal reproduction (50-step CFM smoke) will confirm
   or refute these hypotheses.

**Refuted claims** (none at this stage — waiting on Wf-396-A3).

---

## §6 Cross-references

- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py:557-855` — `train_one_epoch`
  with the 6 NEW keyword params (Wf-388-A2)
- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py:775-820` — SA penalty wire-in
  (Wf-388-A3)
- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py:872-937` —
  `evaluate_decode_ratio` (unchanged; not in the diff)
- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py:414-440` —
  `_decode_with_rdkit_strict` (unchanged; not in the diff)
- `metrics/by_metric/decode_ratio.json` — historical decode_ratio record
  (synthetic=1.0 per Wf-124 P2; real NCI-60=0.998 per Wf-148)

---

## §7 Forward-looking

Wf-396-A3 (minimal reproduction) is the natural follow-up:
- Run 50-step CFM CPU smoke with `(a)` baseline (no flags) `(b)` each flag
  isolated `(c)` the Wf-380 SCAFFOLD banner defaults combination.
- Measure `decode_ratio` per run.
- Confirm whether the regression is real, and which flag is the culprit.

Until Wf-396-A3 completes, the regression source is **not formally confirmed**
— Wf-396-A1's git-bisect only narrows the suspect set to the 8 NEW CLI
flags.
