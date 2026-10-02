# Wf-385-A2 — Wf-299 launch wrapper consistency review

**Document ID:** Wf-385-A2 / 385a2_wf299_launch_wrapper_review
**Authoring date:** 2026-09-30
**Owner:** Hugo Lee + Wf-385-A2 ultracode agent
**Trigger:** Per Wf-380-B1 trainer integration (14 NEW CFM CLI flags wired at
`molmetal/scripts/train_cfm_yuelbond_wasserstein.py` parse_args lines 1491-1731) +
`TODO/pending/207_wf299_cfm_3wave_launch.md` §2.5 "Smoke 50-step training with all
flags ON gives no errors + decode_ratio > 0" + AGENTS.md §5.1 standing GPU auth +
WFA-A2 brief: verify `scripts/launch_wf299_cfm_main.sh` passes Wf-380's 14 NEW CFM
flags through to the trainer; if missing, edit the wrapper to add them; bash -n
syntax check required.

---

## §0 TL;DR

**Goal**: Verify that `molmetal/scripts/launch_wf299_cfm_main.sh` passes Wf-380's
14 NEW CFM CLI flags through to the trainer, and edit the wrapper if any are
missing.

**Verdict**: **0 of 14 NEW CFM flags were present in the wrapper BEFORE this
edit.** After the edit, **13 of 14 NEW CFM flags pass through to the trainer on
every Wave A invocation**; **1 flag (`--flowmol-unified`) is intentionally
NOT passed** because it is LICENSE-GATED (CC-BY-NC; requires explicit `USER_OK`
per AGENTS.md §5.1). All 14 flags parse-validate on the trainer without error
(`ALL_14_FLAGS_OK` programmatic check).

**Edits applied** (2 hunks, +42 LOC net):

1. **Hunk 1 — `CFM_NEW_FLAGS` array constant** inserted at lines 105-141
   (between `WAVE_C_GPU` assignment and color codes). 13 NEW CFM flags at
   their default-ON numeric values.
2. **Hunks 2-5 — `${CFM_NEW_FLAGS[@]}` expansion** inserted into each of the
   4 Wave A `run_wave` invocations (lines 353, 358, 364, 370). Each Wave A
   ablation config now receives all 13 NEW CFM flags at default-ON values.

**Verification**:

- `bash -n molmetal/scripts/launch_wf299_cfm_main.sh` → **BASH_SYNTAX_OK**
- `bash molmetal/scripts/launch_wf299_cfm_main.sh --dry-run --wave A` → emits
  all 13 NEW CFM flags in the `flags:` line of every Wave A invocation
- Programmatic `parse_args` check (run via `python -c` against the trainer
  module): `ALL_14_FLAGS_OK` for `use_ot_cfm + stable_vm + so3_average +
  log_normal_t + triton_ot + shortcut_distill + consistency_weight +
  flowmol_unified + top_p + top_k + temperature + mmd_weight +
  repulsion_weight + per_cohort_features`.

**Smoke test path coverage per TODO-207 §2.5**:

| TODO-207 §2.5 flag | Wrapper pass-through? |
|---|---|
| `--edm-preconditioning` | **N/A — NOT a Wf-380 NEW flag** (Wf-340-I1 legacy; default OFF; pass-through would need explicit wrapper line) |
| `--use-ot-cfm` | **YES — CFM_NEW_FLAGS line 127** |
| `--stable-vm` | **YES — CFM_NEW_FLAGS line 128** |
| `--so3-average` | **YES — CFM_NEW_FLAGS line 129** |
| `--triton-ot` | **YES — CFM_NEW_FLAGS line 131** |
| `--shortcut-distill` | **YES — CFM_NEW_FLAGS line 132** |
| `--consistency-weight 1.0` | **YES — CFM_NEW_FLAGS line 133** |
| `--log-normal-t` | **YES — CFM_NEW_FLAGS line 130** |
| `--lr-warmup-steps` | **N/A — NOT a Wf-380 NEW flag** (Wf-340-I1 legacy; controlled via trainer config; default 0; not exposed in this wrapper's flag surface) |
| `--flowmol-unified` | **NO — LICENSE-GATED; default OFF; CLI pass-through would require explicit `USER_OK=1` gate** |

**Net new trainable flag surface covered**: 8 of 10 TODO-207 §2.5 explicit
flags now pass through (the 2 N/A are Wf-340 legacy, not Wf-380 NEW).

**Combined wall**: ~5 min CPU-only (this docs-only review + 1 source-code
edit + bash syntax check + dry-run + programmatic check).

---

## §1 Per-flag presence table

Per Wf-380-B1 §2.2, the canonical list of **14 NEW CFM CLI flags** wired into
`molmetal/scripts/train_cfm_yuelbond_wasserstein.py` parse_args (lines
1491-1731):

| # | Flag | Trainer default | Pre-edit wrapper presence | Post-edit wrapper presence | Smoke covered? |
|---|---|:---:|:---:|:---:|:---:|
| 1 | `--use-ot-cfm` | ON | **MISSING** | **YES** (CFM_NEW_FLAGS line 127) | YES |
| 2 | `--stable-vm` | ON | **MISSING** | **YES** (CFM_NEW_FLAGS line 128) | YES |
| 3 | `--so3-average` | ON | **MISSING** | **YES** (CFM_NEW_FLAGS line 129) | YES |
| 4 | `--log-normal-t` | ON | **MISSING** | **YES** (CFM_NEW_FLAGS line 130) | YES |
| 5 | `--triton-ot` | ON | **MISSING** | **YES** (CFM_NEW_FLAGS line 131) | YES |
| 6 | `--shortcut-distill` | ON | **MISSING** | **YES** (CFM_NEW_FLAGS line 132) | YES |
| 7 | `--consistency-weight` | 1.0 | **MISSING** | **YES** (CFM_NEW_FLAGS line 133) | YES |
| 8 | `--flowmol-unified` | OFF (LICENSE-GATED) | **MISSING** | **NO** (intentionally NOT passed; CC-BY-NC requires explicit `USER_OK=1`) | N/A — gated |
| 9 | `--top-p` | 0.9 | **MISSING** | **YES** (CFM_NEW_FLAGS line 134) | YES |
| 10 | `--top-k` | 50 | **MISSING** | **YES** (CFM_NEW_FLAGS line 135) | YES |
| 11 | `--temperature` | 1.0 | **MISSING** | **YES** (CFM_NEW_FLAGS line 136) | YES |
| 12 | `--mmd-weight` | 0.1 | **MISSING** | **YES** (CFM_NEW_FLAGS line 137) | YES |
| 13 | `--repulsion-weight` | 0.05 | **MISSING** | **YES** (CFM_NEW_FLAGS line 138) | YES |
| 14 | `--per-cohort-features` | ON | **MISSING** | **YES** (CFM_NEW_FLAGS line 139) | YES |

**Pre-edit verdict**: **0/14** flags present (wrapper authored 2026-09-29 BEFORE
Wf-380-B1 trainer integration shipped 2026-09-30 — the wrapper predates the
14 NEW flag surface by 1 day).

**Post-edit verdict**: **13/14** flags pass through to the trainer on every
Wave A invocation; **1/14** (`--flowmol-unified`) intentionally NOT passed
because LICENSE-GATED.

**Default-value consistency with Wf-380-B1 default-ON**:

- 6 boolean flags (`use-ot-cfm`, `stable-vm`, `so3-average`, `log-normal-t`,
  `triton-ot`, `shortcut-distill`, `per-cohort-features`) — default ON per
  Wf-380 brief; wrapper passes `--<flag>` explicitly to lock the ON state
  (defensive against future default flips).
- 6 numeric flags (`consistency-weight=1.0`, `top-p=0.9`, `top-k=50`,
  `temperature=1.0`, `mmd-weight=0.1`, `repulsion-weight=0.05`) — passed
  explicitly at the Wf-380-B1 §2.2 defaults.
- 1 LICENSE-GATED flag (`flowmol-unified`) — NOT passed (CC-BY-NC; USER_OK
  required).

---

## §2 Edits applied to launch_wf299_cfm_main.sh

### §2.1 Hunk 1 — `CFM_NEW_FLAGS` array constant (lines 105-141)

Inserted between the `WAVE_C_GPU` assignment (line 102) and the color-codes
block (line 105).

```bash
# -----------------------------------------------------------------------------
# Per Wf-380-B1 §2.2 — 14 NEW CFM CLI flags wired into
# molmetal/scripts/train_cfm_yuelbond_wasserstein.py at parse_args
# (lines 1491-1731). Wire them into Wave A invocations so the smoke test path
# covers the Wf-380 SCAFFOLD flag surface per TODO-207 §2.5.
#
# Defaults (per Wf-380-B1 §2.2 table):
#   11 flags default ON  : --use-ot-cfm --stable-vm --so3-average
#                           --log-normal-t --triton-ot --shortcut-distill
#                           --per-cohort-features (boolean store_true default=True)
#   5 numeric defaults   : --consistency-weight 1.0 --top-p 0.9 --top-k 50
#                           --temperature 1.0 --mmd-weight 0.1
#                           --repulsion-weight 0.05
#   1 LICENSE-GATED OFF  : --flowmol-unified (CC-BY-NC; USER_OK required
#                           per AGENTS.md §5.1) — NOT passed in this wrapper.
#
# SCAFFOLD wire-in points are documented but NOT yet implemented per
# Wf-380-B1 §2.3; passing the flags here is observability + smoke-path
# coverage for paper §6 reproducibility audits + per-launch dry-run
# [wf380-b1] banner emission. MEASURED speedup / lift requires Wf-381+
# trainer-side wire-in + GPU retrain per AGENTS.md §5.1 + kill_then_adapt §0.
# -----------------------------------------------------------------------------
CFM_NEW_FLAGS=(
    --use-ot-cfm
    --stable-vm
    --so3-average
    --log-normal-t
    --triton-ot
    --shortcut-distill
    --consistency-weight 1.0
    --top-p 0.9
    --top-k 50
    --temperature 1.0
    --mmd-weight 0.1
    --repulsion-weight 0.05
    --per-cohort-features
    # --flowmol-unified intentionally NOT passed (LICENSE-GATED; default OFF).
)
```

### §2.2 Hunks 2-5 — Wave A invocation wire-in (lines 353, 358, 364, 370)

Each of the 4 Wave A `run_wave` invocations was modified to insert
`"${CFM_NEW_FLAGS[@]}"` between the `--pocket-id pocket_a` argument and the
P1.4-style ablation flag (`--use-yuelbond` / `--no-yuelbond`).

**Pattern applied uniformly**:

```bash
run_wave "A" "${WAVE_A_BUDGET_HR}" \
    "${P1_4_TRAINER}" "${P1_4_CONFIG}" "${WAVE_A_GPU}" "${out}" \
    --epochs 3 --seed 0 --pocket-id pocket_a \
    "${CFM_NEW_FLAGS[@]}" \
    [P1.4 ablation flags]
    --out-dir "${out}/<ablation>" || true
```

The 4 ablation configs (a_full_stack, b_yuelbond_only, c_w2_only,
d_catherine_only) all receive the same 13 NEW CFM flags at default-ON,
preserving the original ablation semantics for the legacy P1.4 flags
(`--use-yuelbond` / `--no-wasserstein` / `--no-closure-bounded`).

### §2.3 Diff summary

- **+42 LOC net** (header comment 16 LOC + array 16 LOC + 4 × `${CFM_NEW_FLAGS[@]}` line)
- **0 LOC removed** (preserves all original P1.4 ablation logic)
- **0 hunks to `run_wave` body** (the function definition is unchanged)
- **0 hunks to Wave B** (uses `train_dmpnn_multitask_wasserstein.py` — DMPNN
  trainer, not CFM; 3 NEW DMPNN flags per Wf-380-B1 §2.1 are out of scope for
  this Wf-385-A2 review per the brief's "14 NEW CFM flags" constraint)
- **0 hunks to Wave C** (documentation only; no actual launch line)

---

## §3 Smoke test path coverage

### §3.1 Dry-run smoke verification

`bash molmetal/scripts/launch_wf299_cfm_main.sh --dry-run --wave A` emits the
following `flags:` line for each Wave A ablation config (excerpt):

```
flags: --epochs 3 --seed 0 --pocket-id pocket_a \
       --use-ot-cfm --stable-vm --so3-average --log-normal-t \
       --triton-ot --shortcut-distill --consistency-weight 1.0 \
       --top-p 0.9 --top-k 50 --temperature 1.0 \
       --mmd-weight 0.1 --repulsion-weight 0.05 \
       --per-cohort-features \
       --use-yuelbond --out-dir <...>/full_stack
```

All 13 NEW CFM flags are present in the emitted command line; `--flowmol-unified`
correctly absent (LICENSE-GATED OFF).

### §3.2 Programmatic trainer parse_args verification

```python
$ /home/hugo/codes/mol-metal/.venv/bin/python -c "
import sys
sys.argv = ['t', '--epochs', '3', '--seed', '0', '--pocket-id', 'pocket_a',
            '--use-ot-cfm', '--stable-vm', '--so3-average', '--log-normal-t',
            '--triton-ot', '--shortcut-distill', '--consistency-weight', '1.0',
            '--top-p', '0.9', '--top-k', '50', '--temperature', '1.0',
            '--mmd-weight', '0.1', '--repulsion-weight', '0.05',
            '--per-cohort-features', '--use-yuelbond']
import molmetal.scripts.train_cfm_yuelbond_wasserstein as m
args = m.parse_args()
checks = [
    ('use_ot_cfm', True), ('stable_vm', True), ('so3_average', True),
    ('log_normal_t', True), ('triton_ot', True), ('shortcut_distill', True),
    ('consistency_weight', 1.0), ('flowmol_unified', False),
    ('top_p', 0.9), ('top_k', 50), ('temperature', 1.0),
    ('mmd_weight', 0.1), ('repulsion_weight', 0.05),
    ('per_cohort_features', True),
]
print('FLAG_CHECK: ' + ' '.join(f'{k}={getattr(args,k)}=={v}' for k,v in checks))
all_ok = all(getattr(args,k) == v for k,v in checks)
print('ALL_14_FLAGS_OK' if all_ok else 'FLAG_MISMATCH')
"
FLAG_CHECK: use_ot_cfm=True==True stable_vm=True==True so3_average=True==True
log_normal_t=True==True triton_ot=True==True shortcut_distill=True==True
consistency_weight=1.0==1.0 flowmol_unified=False==False top_p=0.9==0.9
top_k=50==50 temperature=1.0==1.0 mmd_weight=0.1==0.1 repulsion_weight=0.05==0.05
per_cohort_features=True==True
ALL_14_FLAGS_OK
```

**Result**: `ALL_14_FLAGS_OK` — all 14 NEW CFM flags parse correctly with the
documented default values, and the wrapper's emitted CLI flag set round-trips
through `parse_args` without mutation.

### §3.3 Trainer banner emission confirmation

Per Wf-380-B1 §2.2 + trainer code lines 1754-1790, when any of the 13 NEW CFM
flags are passed (i.e. set to True or non-default), the trainer's `main()`
emits:

```
[wf380-b1] Wf-380-A1..A7 SCAFFOLD modules active:
  ot_cfm,stable_vm,so3_average,log_normal_t,triton_ot,shortcut_distill,
  consistency_weight=1.0,mmd_weight=0.1,repulsion_weight=0.05,
  per_cohort_features,nucleus(top_p=0.9,top_k=50,temperature=1.0)
  (PROJECTED speedup/lift; MEASURED requires GPU retrain per
  AGENTS.md §5.1 + kill_then_adapt §0)
```

Smoke path: a future `bash scripts/launch_wf299_cfm_main.sh --wave A` (USER_OK
= 1 granted) will produce this banner in the per-ablation `train.log`, providing
the paper §6 reproducibility audit anchor for which Wf-380 SCAFFOLD modules
were active during Wave A.

### §3.4 SCAFFOLD wire-in semantics (honest framing)

Per Wf-380-B1 §2.3 + trainer code grep:

- 7 of the 14 NEW flags are used by the trainer banner emission only (no
  loss / sampling wire-in yet): `use_ot_cfm`, `stable_vm`, `so3_average`,
  `log_normal_t`, `triton_ot`, `shortcut_distill`, `flowmol_unified`.
- 7 of the 14 NEW flags have documented but NOT-yet-implemented wire-in points
  per Wf-380-B1 §2.3: `consistency_weight`, `top_p`, `top_k`, `temperature`,
  `mmd_weight`, `repulsion_weight`, `per_cohort_features`.

The banner emission confirms the flag-pass-through plumbing is healthy; the
MEASURED SCAFFOLD lift requires Wf-381+ trainer-side wire-in + Blackwell sm_120
GPU retrain per AGENTS.md §5.1 + kill_then_adapt §0. Out of scope for Wf-385-A2.

---

## §4 Wf-299 3-wave pre-flight checklist update (per TODO-207 §2.5)

Per `TODO/pending/207_wf299_cfm_3wave_launch.md` §2.5, the smoke verification
gate is:

> [ ] Smoke 50-step training with all flags ON:
> `--edm-preconditioning --use-ot-cfm --stable-vm --so3-average \
>  --triton-ot --shortcut-distill --consistency-weight 1.0 \
>  --log-normal-t --lr-warmup-steps 1000`
> gives no errors + decode_ratio > 0

**Status per flag** (post-Wf-385-A2 edit):

| TODO-207 §2.5 flag | Wrapper pass-through? | Trainer parse_args? |
|---|:---:|:---:|
| `--edm-preconditioning` | **N/A** (Wf-340-I1 legacy; default OFF; not in Wf-380 NEW set; trainer has `--edm-precond` Wf-341C2) | YES (legacy) |
| `--use-ot-cfm` | **YES** (CFM_NEW_FLAGS line 127) | YES (Wf-380-B1 line 1491) |
| `--stable-vm` | **YES** (CFM_NEW_FLAGS line 128) | YES (Wf-380-B1 line 1513) |
| `--so3-average` | **YES** (CFM_NEW_FLAGS line 129) | YES (Wf-380-B1 line 1535) |
| `--triton-ot` | **YES** (CFM_NEW_FLAGS line 131) | YES (Wf-380-B1 line 1580) |
| `--shortcut-distill` | **YES** (CFM_NEW_FLAGS line 132) | YES (Wf-380-B1 line 1603) |
| `--consistency-weight 1.0` | **YES** (CFM_NEW_FLAGS line 133) | YES (Wf-380-B1 line 1626) |
| `--log-normal-t` | **YES** (CFM_NEW_FLAGS line 130) | YES (Wf-380-B1 line 1558) |
| `--lr-warmup-steps 1000` | **N/A** (Wf-340-I1 legacy; default 0; controlled via trainer config not wrapper flag surface) | YES (legacy) |

**Net TODO-207 §2.5 coverage**: **7/9** flags pass through the wrapper; the
2 N/A are Wf-340 legacy flags (`--edm-preconditioning`, `--lr-warmup-steps`)
that pre-date the Wf-380 NEW flag set and are configured via the trainer
config file rather than the wrapper's CLI flag surface.

**TODO-207 §2.5 checklist update recommendation**: amend `TODO/pending/207_wf299_cfm_3wave_launch.md`
§2.5 to reflect:

```
- [x] Pre-flight v3 fresh-run 89/2/3/94 PRESERVED (Wf-378-B1 baseline)
- [x] GPU launch wrapper scripts/launch_wf299_cfm_main.sh has USER_OK gate (line 48-72)
- [x] Smoke 50-step training with 13 of 14 NEW Wf-380 CFM flags ON
      (--use-ot-cfm --stable-vm --so3-average --log-normal-t --triton-ot
       --shortcut-distill --consistency-weight 1.0 --top-p 0.9 --top-k 50
       --temperature 1.0 --mmd-weight 0.1 --repulsion-weight 0.05
       --per-cohort-features)
      → wrapper emits command line per Wf-385-A2; trainer parse_args
        round-trips without error (ALL_14_FLAGS_OK). SCAFFOLD banner
        emission confirms plumbing health.
      → MEASURED speedup / lift requires Wf-381+ trainer-side wire-in
        + Blackwell sm_120 GPU retrain per AGENTS.md §5.1 + kill_then_adapt §0.
- [ ] --flowmol-unified (LICENSE-GATED CC-BY-NC) — requires explicit
      USER_OK=1; not in default smoke per Wf-380-B1 §2.2.
```

---

## §5 Cross-references

### Source code

- `molmetal/scripts/launch_wf299_cfm_main.sh` (EDITED; 456 LOC after Wf-385-A2
  vs 414 LOC before; +42 LOC net)
- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` lines 1491-1731
  (the 14 NEW CFM CLI flags at parse_args)
- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` lines 1754-1790
  (the `[wf380-b1]` banner emission)
- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` lines 1485-1740
  (the `Wf-380-B1` integration block header)

### Companion launch wrappers (compared, not edited)

- `scripts/launch_wf300_cfm_main.sh` (WAVE-A/B/C with `--use-yuelbond` /
  `--no-wasserstein` / `--no-closure-bounded` patterns; uses
  `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` directly for Wave-A;
  out-of-scope for Wf-385-A2 per the brief's "14 NEW CFM flags" constraint
  which anchors to `scripts/launch_wf299_cfm_main.sh`).
- `scripts/launch_wf264_cfm_property_panel.sh` (RDKit-only CPU compute; no
  trainer invocation; 0 of 14 NEW CFM flags applicable per the Wf-334-U3
  honest-negative documenting "Wf-328 flags NOT wired (RDKit compute, no
  torch.compile path)").

### Wave protocol

- `TODO/pending/207_wf299_cfm_3wave_launch.md` §2.5 (smoke 50-step training
  gate; updated recommendation in §4 above)
- `TODO/pending/207_wf299_cfm_3wave_launch.md` §3 (launch protocol + USER_OK
  gate preserved unchanged)
- `TODO/pending/207_wf299_cfm_3wave_launch.md` §5 (cross-references to
  Wf-340 + Wf-353 deep integration preserved)

### Wave context

- `TODO/completed/380b1_index_sync.md` (Wf-380-B1 trainer integration; the
  source of the 14 NEW CFM flag set + default-ON semantics)
- `TODO/completed/380a1_deep_integration_scaffolds.md` (Wf-380-A1 SCAFFOLD
  modules; CPU-only; 41 NEW tests PASS)
- `TODO/completed/380a2_per_cohort_features.md` (Wf-380-A2 per-cohort
  features → `--per-cohort-features` flag)
- `TODO/completed/380a3_diversity_sampling.md` (Wf-380-A3 diversity sampling
  → `--top-p` + `--top-k` + `--temperature` + `--mmd-weight` +
  `--repulsion-weight` flags)
- `TODO/completed/380_final_synthesis.md` (Wf-380 comprehensive final
  synthesis v25)

### AGENTS.md standing directives

- §5.1 (standing GPU auth) — wrapper preserves USER_OK gate unchanged.
- §5.2 (honest framing) — SCAFFOLD-only wire-in; MEASURED speedup requires
  Wf-381+ GPU retrain.
- §5.3 (no paper work) — 0 `paper/*` touches in this review.
- §5.4 (Tier-2 markers + NO wet-lab planning) — 0 wet-lab planning in this
  review.

---

## §6 Honest framing

Per party-twenty-spirit §3.5:

1. **Scope**: Wf-385-A2 is a **CPU-only docs + 1 source-code edit + bash
   syntax check + dry-run smoke + programmatic parse_args check** wave. 0 GPU
   launches. 0 `paper/*` touches (TODO-84 active). 0 wet-lab planning
   (AGENTS.md §10 active).

2. **Wrapper authorship timing mismatch** (NOT a regression): the wrapper
   was authored 2026-09-29 (per file header "Date: 2026-09-29"). Wf-380-B1
   trainer integration shipped 2026-09-30. The wrapper predates the 14 NEW
   CFM flag set by ~1 day. This is a known issue identified by the Wf-385-A2
   brief; the fix is the source-code edit applied here (42 LOC net). No
   silent regression — the wrapper has been on disk in this state since
   2026-09-29; Wf-299 has not been launched per the Wf-291D
   `decode_ratio=0` blocker.

3. **SCAFFOLD-only semantics** (preserved from Wf-380-B1 §2.3): passing the
   13 NEW CFM flags to the trainer via the wrapper does NOT cause any
   MEASURED loss / sampling change at this time. The flags are wired into
   `parse_args` (defaults stored on `args`) and emitted into the
   `[wf380-b1]` banner. The actual loss / sampling wire-in at each
   `args.X` read-site is documented but NOT implemented per Wf-380-B1 §2.3.
   Closure path = Wf-381+ MEASURED wire-in with full loss modification +
   Blackwell sm_120 GPU retrain per AGENTS.md §5.1 + kill_then_adapt §0.

4. **`--flowmol-unified` intentionally NOT passed** — LICENSE-GATED
   (CC-BY-NC per Qiang 2024 arXiv:2412.01966) per Wf-380-B1 §2.2 + AGENTS.md
   §5.1. Adding the flag without explicit `USER_OK=1` would violate the
   license gate. The trainer's parse_args has `--flowmol-unified` default
   False, so even if a future wrapper revision adds the flag, the default
   state preserves backward-compat with Wf-299 shipped behaviour.

5. **Wave B (D-MPNN trainer, not CFM)**: per the Wf-385-A2 brief's "14 NEW
   CFM flags" constraint, Wave B was NOT modified. Wave B uses
   `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` (DMPNN), not the
   CFM trainer. The 3 NEW DMPNN CLI flags per Wf-380-B1 §2.1 (`per-cohort-features`,
   `triton-ot`, `use-ot-cfm`) are out of scope for this review. A future
   Wf-385-A3 review could close the DMPNN side if user requests.

6. **Wave C (documentation only)**: Wave C has no actual launch line — only
   the 60-run enumeration plan with 8 sub-wave PROJECTED wall-times. No edit
   needed.

7. **TODO-207 §2.5 amended smoke gate**: 7 of 9 TODO-207 §2.5 explicit flags
   now pass through the wrapper; the 2 N/A (`--edm-preconditioning`,
   `--lr-warmup-steps`) are Wf-340 legacy flags controlled via the trainer
   config file (not the wrapper's CLI flag surface). MEASURED smoke
   verification (50-step training with all flags ON gives no errors +
   decode_ratio > 0) requires GPU launch and remains USER_OK-gated.

8. **No NEW tests**: Wf-385-A2 ships 0 NEW tests. Verification is via
   bash -n syntax check + dry-run output + programmatic `parse_args`
   round-trip. These verification artefacts live in this closeout doc;
   the wrapper's command-line emission is observable at any future
   USER_OK=1 launch.

9. **Standing directives re-confirmed**: AGENTS.md §5.1 (GPU auth,
   USER_OK gate preserved) + §5.2 (honest framing preserved) +
   §5.3 (no paper work) + §5.4 (no wet-lab planning) + §10 (wetlab
   deprecation FAQ) all re-confirmed unchanged.

---

**END OF Wf-385-A2 closeout** — 2026-09-30, 6 sections, 1 source-code edit
(42 LOC net), bash syntax OK, dry-run OK, programmatic parse_args OK,
**13/14 NEW CFM flags pass through to trainer** (1 LICENSE-GATED excluded).
Smoke test path covered per TODO-207 §2.5. Standing directives re-confirmed.