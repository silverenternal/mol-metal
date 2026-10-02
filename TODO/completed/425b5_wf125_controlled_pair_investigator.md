# Wf-425 B5 (Wf-388 Wf-125 P1 CONTROLD PAIR) — Investigator Review Report

**Date:** 2026-10-01
**Wave:** Wf-425 investigator Phase 1 (review-only; Phase 2 fixers apply)
**Targets:** `scripts/launch_wf125_p1_controlled_pair.sh` + `molmetal/reports/wf387a2_balanced_rh_cohort/balanced_rh_cohort.json` + `molmetal/scripts/train_metal_lomo_wasserstein.py`
**Standing directives:** AGENTS.md §5.5 (filesystem-truth) + §5.6 (atomic write) + §5.7 (50-step CPU smoke) + §5.4 (no wet-lab per §10) + §1.2 (honest framing per party-twenty-spirit §3.5)

---

## §0 TL;DR (1-2 sentence verdict)

The Wf-388 Wf-125 P1 CONTROLD PAIR retrain is blocked on **3 distinct root causes**, not just the apparent "missing `.py`" claim: (B5-a) `molmetal/scripts/wf125_p1_controlled_pair.py` is indeed a phantom target — the wrapper `scripts/launch_wf125_p1_controlled_pair.sh` (USER_OK-gated, ~6 hr wall PROJECTED) IS the canonical entry point and was authored as a wrapper per Wf-416 Wf-125 CONTROLD PAIR design (Wf-387-A2 §5.1); (B5-b) the wrapper does NOT accept task brief's literal `--epochs 30 --seeds 0,1,2 --balanced-cohort --user-ok` flag set — wrapper hardcodes EPOCHS=50 + --metals Rh + --seeds 0 + only accepts `--run-mode synthetic|real|none` + `--epochs` + `--out-dir`; (B5-c) the SCHEMA MISMATCH is a REAL BLOCKER: trainer `molmetal/scripts/train_metal_lomo_wasserstein.py:211` calls `entry.get("catherine_6of6_pass", False)` but Wf-387-A2 cohort JSON `per_compound[]` schema uses `is_active` (bioactivity label) NOT `catherine_6of6_pass` (structural validity). Closure requires **2 minimal fixes**: (1) patch `train_metal_lomo_wasserstein.py:211` to filter on `is_active` (the Wf-387-A2 design's actual ground truth label) instead of `catherine_6of6_pass` — ~1 LOC; (2) re-author task-brief invocation per wrapper `--run-mode real` + `--epochs 30` (no `--seeds 0,1,2` override because wrapper only plans 3 CONTROLD PAIR runs).

---

## §1 Code state (file:line refs to every relevant code location)

### 1.1 Wrapper `scripts/launch_wf125_p1_controlled_pair.sh` (19,892 B mtime 2026-09-30 22:22)

**USER_OK gate** (lines 96-102):
```
96: if [ "${USER_OK:-0}" != "1" ]; then
97:     printf '%s[FATAL]%s USER_OK gate refused: set USER_OK=1 in env to launch.\n' \
98:         "$(printf '\033[0;31m')" "$(printf '\033[0m')" >&2
99:     printf '       USER_OK=1 bash %s --dry-run   # CPU plan only\n' "$0" >&2
100:    printf '       USER_OK=1 bash %s                # actual ~6 hr GPU run (3-run CONTROLD PAIR)\n' "$0" >&2
101:    exit 64
102: fi
```
**KEY FINDING:** Wrapper uses `USER_OK` **environment variable** (per Wf-294B family convention) — task brief's literal `--user-ok` CLI flag is NOT a real flag (the wrapper has NO `--user-ok` handler).

**Constants** (lines 62-64):
```
62: EPOCHS=50
63: GPU="cuda:0"               # RTX PRO 6000 Blackwell sm_120 (per Wf-411-A1)
64: DEFAULT_OUT_BASE="${PROJECT_ROOT}/molmetal/reports/wf125_p1_controlled_pair"
```
**KEY FINDING:** `EPOCHS=50` is HARDCODED — overridable via `--epochs=` (line 79) but the smoke test (line 174) uses `--epochs 2`.

**Flag parsing** (lines 73-91):
```
73: for arg in "$@"; do
74:     case "$arg" in
75:         --dry-run) DRY_RUN=1 ;;
76:         --run-mode)  shift_next=1 ;;
77:         --run-mode=*) RUN_MODE_OVERRIDE+=("${arg#--run-mode=}") ;;
78:         --epochs)    shift_next=1 ;;
79:         --epochs=*)  EPOCHS_OVERRIDE="${arg#--epochs=}" ;;
80:         --out-dir)   shift_next=1 ;;
81:         --out-dir=*) OUT_DIR_OVERRIDE="${arg#--out-dir=}" ;;
82:     esac
83: done
```
**KEY FINDING:** Wrapper accepts ONLY `--dry-run`, `--run-mode`, `--epochs`, `--out-dir`. NO `--user-ok`, NO `--seeds`, NO `--balanced-cohort`, NO `--metals`, NO `--device`.

**Per-run COMMON_ARGS** (lines 133-143):
```
133: COMMON_ARGS=(
134:   --epochs "${EPOCHS}"
135:   --metals "Rh"
136:   --seeds 0
137:   --wasserstein-weight 0.05
138:   --csv "${LOMO_CSV}"
139:   --use-l2-guard
140:   --no-cuda-graphs
141:   --device "cuda:0"
142:   --cuda-visible-devices 0
143: )
```
**KEY FINDING:** `--seeds 0` HARDCODED — wrapper does not pass task brief's `--seeds 0,1,2` to trainer. `--metals "Rh"` HARDCODED — task brief's `--metals Ru,Ir,Rh,Os,Re` not honored (but this is INTENTIONAL per Wf-387-A2 §5.1 CONTROLD PAIR design which targets Rh-only for the lift question).

**build_cohort_args dispatch** (lines 157-166):
```
157: build_cohort_args() {
158:     local synthetic_or_real="$1"
159:     case "${synthetic_or_real}" in
160:         synthetic) printf '%s\n' "--rh-oversample" "500" "--rh-cohort-json" "${SYNTH_RH_COHORT}" ;;
161:         real)      printf '%s\n' "--rh-oversample" "500" "--rh-cohort-json" "${REAL_RH_COHORT}" ;;
162:         none)      printf '%s\n' ;;
163:         *) printf '%s[FATAL]%s invalid --run-mode: %s (must be synthetic|real|none)\n' \
164:                 "$(printf '\033[0;31m')" "$(printf '\033[0m')" "${synthetic_or_real}" >&2; return 1 ;;
165:     esac
166: }
```
**KEY FINDING:** Wrapper dispatches `--rh-cohort-json` based on `--run-mode synthetic|real|none` — task brief's `--balanced-cohort` does NOT exist; equivalent semantic = `--run-mode real`.

**SCAFFOLD smoke** (lines 171-188):
```
171: echo "=== Wf-125 P1 CONTROLD PAIR SCAFFOLD smoke (CPU; Wf-399-A1 §5.7 floor) ===" | tee -a "${LOG}"
172: SMOKE_RC=0
173: PYTHONPATH="${PROJECT_ROOT}" ${PY} "${TRAINER}" \
174:     --epochs 2 \
175:     --metals Rh --seeds 0 \
176:     --wasserstein-weight 0.05 \
177:     --csv "${LOMO_CSV}" --use-l2-guard \
178:     --device cpu \
179:     --rh-oversample 500 --rh-cohort-json "${REAL_RH_COHORT}" \
180:     --out-dir "${OUT_BASE}/scaffold_smoke" \
181:     > "${OUT_BASE}/scaffold_smoke.log" 2>&1 || SMOKE_RC=$?
```
**KEY FINDING:** The smoke at line 173-181 invokes the trainer with REAL_RH_COHORT (Wf-387-A2 schema) — this is EXACTLY where `RuntimeError: No Catherine-6/6 PASS Rh compounds found` triggers at trainer line 218.

### 1.2 Trainer `molmetal/scripts/train_metal_lomo_wasserstein.py` (48,958 B mtime 2026-09-27 22:18)

**`load_rh_synthetic_smiles` schema filter** (lines 193-220):
```
193: def load_rh_synthetic_smiles(cohort_json: Path) -> List[str]:
194:     """Return the SMILES of every Catherine 6/6 PASS compound in the cohort.
...
210:     for entry in payload.get("per_compound", []):
211:         if not entry.get("catherine_6of6_pass", False):
212:             continue
213:         s = str(entry.get("smiles", "")).strip()
214:         if s:
215:             smis.append(s)
216:     if not smis:
217:         raise RuntimeError(
218:             f"No Catherine-6/6 PASS Rh compounds found in {cohort_json}"
219:         )
220:     return smis
```
**KEY FINDING (B5-c ROOT CAUSE):** Line 211 filters on `catherine_6of6_pass` — Wf-387-A2 cohort JSON `per_compound[]` does NOT have this field; ALL 500 entries are filtered out → RuntimeError at line 218.

**`append_synthetic_rh_to_dataset` call-chain** (line 251):
```
251:     smis = load_rh_synthetic_smiles(cohort_json)
```
**KEY FINDING:** Line 251 is the immediate caller — raises RuntimeError before any further processing.

**Trainer invocation chain** (lines 766-769):
```
766:     if rh_over > 0:
767:         ds, n_appended = append_synthetic_rh_to_dataset(
768:             ds, Path(args.rh_cohort_json), n_target=rh_over
769:         )
```
**KEY FINDING:** Wrapper passes `--rh-cohort-json ${REAL_RH_COHORT}` → line 768 receives Wf-387-A2 cohort path → line 251 calls `load_rh_synthetic_smiles` → line 218 RuntimeError.

**CLI flags** (lines 680-705):
```
680:     p.add_argument("--metals", nargs="+", default=["Ru", "Ir", "Os", "Rh", "Re"])
681:     p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
682:     p.add_argument("--epochs", type=int, default=30)
...
690:     p.add_argument("--rh-oversample", type=int, default=500,
692:     p.add_argument("--rh-cohort-json", type=str, default=str(DEFAULT_RH_COHORT_JSON),
```
**KEY FINDING:** Trainer `--seeds` accepts `nargs="+"` (multi-value list) — task brief's `--seeds 0,1,2` IS valid trainer syntax. BUT wrapper line 136 hardcodes `--seeds 0` in COMMON_ARGS and doesn't expose `--seeds` to user.

### 1.3 Cohort JSON `molmetal/reports/wf387a2_balanced_rh_cohort/balanced_rh_cohort.json` (198,302 B mtime 2026-09-30 10:03)

**Schema fields present in `per_compound[]`** (9 keys per entry, audited via JSON parse):
```
['compound_id', 'is_active', 'is_synthetic', 'metal', 'scaffold_id', 'smiles',
 'source_csv_row_hash', 'source_split', 'wf139_p7_carryover']
```
**KEY FINDING (B5-c ROOT CAUSE):** `catherine_6of6_pass` field is **NOT PRESENT** in Wf-387-A2 cohort schema. Only `is_active` (bioactivity label) is present, and **all 500 entries have `is_synthetic=False` + `wf139_p7_carryover=False` + `is_active` balanced 250/250**.

**Wf-387-A2 §3.5 schema documentation** (per TODO/completed/387a2_balanced_rh_cohort.md lines 77-103):
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `is_active` | int (0 or 1) | 1 | bioactivity label from v4 CSV (ground truth) |
| `catherine_6of6_pass_rate` | float | 1.0000 | structural validity, NOT bioactivity |

**KEY FINDING:** Wf-387-A2 closeout §3.5 documents `catherine_6of6_pass_rate` as an AGGREGATE stat (not per-compound field), and the bioactivity ground truth is `is_active`. The trainer code at line 211 was authored before Wf-387-A2 cohort existed (per Wf-387-A2 closeout header referencing Wf-139 P7 which used `catherine_6of6_pass` field).

### 1.4 Phantom target `molmetal/scripts/wf125_p1_controlled_pair.py` (FILE_NOT_FOUND)

```bash
$ ls -la molmetal/scripts/wf125_p1_controlled_pair.py
ls: cannot access 'molmetal/scripts/wf125_p1_controlled_pair.py': No such file or directory
```
**KEY FINDING (B5-a):** No `.py` file exists. The canonical entry point is the wrapper. Task brief's literal `python molmetal/scripts/wf125_p1_controlled_pair.py` invocation is invalid — the wrapper IS the canonical surface per Wf-387-A2 §5.1 CONTROLD PAIR design.

### 1.5 Cross-script `catherine_6of6_pass` field usage (canonical field name)

| File | Line | Usage |
|------|------|-------|
| `molmetal/scripts/wasserstein_transfer_rh.py` | 652 | `c.catherine_6of6_pass for c in cohort.per_compound` |
| `molmetal/scripts/wasserstein_transfer_rh.py` | 743 | `"catherine_6of6_pass": bool(c.catherine_6of6_pass)` (WRITES field) |
| `molmetal/scripts/rh_active_label_generation.py` | 89 | `catherine_6of6_pass: bool` (Pydantic model) |
| `molmetal/scripts/rh_active_label_generation.py` | 195, 235 | `catherine_6of6_pass=syn.get("catherine_6of6_pass", False)` |
| `molmetal/scripts/wf379a1_normalize_rh_cohort.py` | 40 | `e.get("catherine_6of6_pass", False)` |
| `molmetal/molmetal_lam/reward/closure_oversample.py` | 204 | `catherine_6of6_pass : bool` |
| `molmetal/scripts/train_metal_lomo_wasserstein.py` | 211 | `entry.get("catherine_6of6_pass", False)` ← FILTER |

**KEY FINDING:** `catherine_6of6_pass` IS the canonical field name used across the codebase. Wf-139 P7 synthetic cohort writes it (per `wasserstein_transfer_rh.py:743`). Wf-387-A2 real balanced cohort OMITTED this field, creating the schema mismatch with the trainer's pre-existing filter.

---

## §2 Root cause analysis (chain of evidence from claim → code → gap)

### Claim 1: `molmetal/scripts/wf125_p1_controlled_pair.py` is missing

**Evidence chain:**
- Claim: BLOCKER B5-a says `wf125_p1_controlled_pair.py` doesn't exist on disk.
- Verification: `ls -la molmetal/scripts/wf125_p1_controlled_pair.py` → `No such file or directory`.
- Context: `scripts/launch_wf125_p1_controlled_pair.sh` (19,892 B mtime 2026-09-30 22:22) IS on disk and IS the canonical Wf-125 P1 CONTROLD PAIR launch wrapper per Wf-387-A2 §5.1 design and Wf-416-B1 §22.101.4 (this is a USER_OK-gated wrapper, NOT a Python script entry point).
- **Verdict: BLOCKER B5-a is NOT-A-BLOCKER** — the task brief's literal `python molmetal/scripts/wf125_p1_controlled_pair.py` invocation is INVALID per the actual project surface. The wrapper `scripts/launch_wf125_p1_controlled_pair.sh` is the canonical entry point per Wf-387-A2 §5.1 + Wf-416-B1 §22.101.4.

### Claim 2: Wrapper does NOT accept `--epochs 30 --seeds 0,1,2 --balanced-cohort --user-ok` flags

**Evidence chain:**
- Claim: BLOCKER B5-b says wrapper flag surface doesn't match task brief.
- Verification: Wrapper flag parsing at lines 73-91 accepts ONLY `--dry-run`, `--run-mode`, `--epochs`, `--out-dir`. Task brief's `--seeds 0,1,2` is NOT in wrapper parser; task brief's `--balanced-cohort` is NOT in wrapper parser; task brief's `--user-ok` is NOT in wrapper parser (wrapper uses `USER_OK` env var which is CORRECT per Wf-294B family convention).
- Task brief intent mapped to wrapper semantics:
  - `--epochs 30` → `--epochs=30` (override EPOCHS=50 to 30) ✅ VALID
  - `--seeds 0,1,2` → NO EQUIVALENT (wrapper hardcodes `--seeds 0` in COMMON_ARGS; task brief wants multiple seeds but wrapper plans 3 CONTROLD PAIR runs not 3 seeds of same run)
  - `--balanced-cohort` → `--run-mode real` (selects Wf-387-A2 REAL balanced cohort) ✅ SEMANTIC EQUIVALENT
  - `--user-ok` → `USER_OK=1` env var (CORRECT per Wf-294B convention; wrapper reads env var at line 96)
- **Verdict: BLOCKER B5-b PARTIAL** — `--user-ok` → env var is correct convention (NOT a wrapper bug); `--epochs 30` works as override; `--balanced-cohort` → `--run-mode real` is semantic equivalent; `--seeds 0,1,2` is the only true mismatch (wrapper plans 3 CONTROLD PAIR runs not 3 seeds of same run).

### Claim 3: Trainer `load_rh_synthetic_smiles` raises RuntimeError due to schema mismatch

**Evidence chain:**
- Claim: BLOCKER B5-c says trainer fails with `RuntimeError: No Catherine-6/6 PASS Rh compounds found`.
- Verification: Trainer `train_metal_lomo_wasserstein.py:211` calls `entry.get("catherine_6of6_pass", False)` — this filters per-compound entries on the `catherine_6of6_pass` field.
- Wf-387-A2 cohort JSON `per_compound[]` 500 entries — NONE have `catherine_6of6_pass` field (audited via JSON parse; field set is exactly 9 keys: `compound_id`, `is_active`, `is_synthetic`, `metal`, `scaffold_id`, `smiles`, `source_csv_row_hash`, `source_split`, `wf139_p7_carryover`).
- Trainer line 211 with default `False` → all 500 entries filtered → line 216 `if not smis: raise RuntimeError(...)`.
- **Verdict: BLOCKER B5-c REAL — schema mismatch is a REAL blocker.** The Wf-387-A2 cohort was designed with `is_active` (bioactivity) as the ground truth label, not `catherine_6of6_pass` (structural validity). The trainer's pre-existing filter at line 211 was authored for Wf-139 P7 synthetic cohort (which has `catherine_6of6_pass`).

### Root cause chain summary:

```
Task brief invocation:
  python molmetal/scripts/wf125_p1_controlled_pair.py --epochs 30 --seeds 0,1,2 --balanced-cohort --user-ok
   ↑ B5-a  NOT-A-BLOCKER (file doesn't exist; wrapper is canonical)
   ↑ B5-b  PARTIAL (3 of 4 flags mapped via wrapper semantics; --seeds not honored)
   ↑ B5-c  REAL BLOCKER (trainer schema filter on missing field)
```

---

## §3 Predecessor closeout cross-check (which Wf-X closeout made the false claim)

| Block | Predecessor closeout | Claim | Verification |
|-------|---------------------|-------|--------------|
| B1 (pre-flight gate) | `TODO/completed/347a1_wf296_integration.md` | `p3_wf343c2_per_wave_runtime_check` gate wired | VERIFIED — gate exists; pre-flight currently PASS at 114/0/3/117 |
| B1 (cascade fix) | `TODO/completed/367a2_preflight_no_fail_gates_fix.md` | `--strict` removal at pre_flight_check.py:4590-4598 | VERIFIED — `--strict` removed from gate's invocation; cascade parent closed |
| B2 (launcher bug) | `TODO/completed/413a3_wf299_wave_b_launch.md` | `--json` argparse bug fixed at `launch_wf299_cfm_main.sh:257-258` | VERIFIED — bug fixed in Wf-299 wrapper; Wf-125 wrapper uses different convention (USER_OK env var) |
| B3 (wrapper convention) | `TODO/completed/409a3_wf246c_scaffold.md` | Wf-294B USER_OK gate family | VERIFIED — Wf-125 wrapper follows Wf-294B convention (USER_OK env var, exit 64 on refusal, gate BEFORE dry-run block) |
| B3 (kcal sign) | Wf-299B kcal sign convention | `(1 - 2*c) × kcal_per_unit` for Vina convention | OUT OF SCOPE for Wf-125 CONTROLD PAIR (no Vina kcal/mol in this retrain) |
| B4 (Lambda CFM) | `TODO/completed/394a1_lambda_cfm_wire.md` | Lambda × CFM coupling wire-in audit | OUT OF SCOPE for Wf-125 (different trainer — `train_metal_lomo_wasserstein.py` not `train_cfm_yuelbond_wasserstein.py`) |
| B4 (canonical shape) | `TODO/completed/329w6_todo21_lambda_cfm_verified.md` | Lambda × CFM canonical wire shape | OUT OF SCOPE for Wf-125 |
| **B5 (cohort build)** | `TODO/completed/387a2_balanced_rh_cohort.md` | Wf-387-A2 balanced 500 cohort built with `is_active` ground truth | **VERIFIED — claim is TRUE; cohort schema uses `is_active` NOT `catherine_6of6_pass`** |
| **B5 (cohort field)** | Wf-139 P7 (canonical synthetic cohort) | `catherine_6of6_pass` is the canonical filter field for synthetic Rh cohort | **VERIFIED — Wf-139 P7 synthetic cohort DOES have `catherine_6of6_pass` field per `wasserstein_transfer_rh.py:743`; Wf-387-A2 real balanced cohort OMITS this field — that's the schema gap** |
| B6 (wire-in false claim) | `TODO/completed/405a2_wf380_loss_wire.md` | Wf-380 + Wf-405-A2 wire-in claims | OUT OF SCOPE for Wf-125 (different trainer) |
| B6 (per-cohort feature) | `TODO/completed/380a2_per_cohort_features.md` | Wf-380-A2 per-cohort feature extractor | OUT OF SCOPE for Wf-125 |
| B5 (current cycle) | `TODO/completed/424a10_wf388_wf125_controlled_pair.md` | Wf-424-A10 prior HONEST_NEGATIVE_BLOCKED verdict | **VERIFIED — same 3 blockers identified at Wf-424-A10 timestamp 2026-10-01 03:42 CST; same findings apply at Wf-425 investigator timestamp** |

---

## §4 Proposed fix plan (specific file paths + line numbers + LOC estimate + new code snippet)

### Fix 1 (PRIMARY — closes B5-c schema mismatch; ~1 LOC)

**File:** `molmetal/scripts/train_metal_lomo_wasserstein.py`
**Line:** 211
**Current code:**
```python
        if not entry.get("catherine_6of6_pass", False):
            continue
```
**Proposed code (Option A — minimal change, accept both fields):**
```python
        # Wf-387-A2: real cohort uses is_active; Wf-139 P7 synthetic cohort uses catherine_6of6_pass.
        # Accept either field to support both synthetic + real cohorts per Wf-387-A2 §5.1.
        if not (entry.get("catherine_6of6_pass", False) or entry.get("is_active", False)):
            continue
```
**Proposed code (Option B — strict Wf-387-A2 design: filter on is_active):**
```python
        # Wf-387-A2: real cohort uses is_active as bioactivity ground truth per §3.5.
        # All 500 entries are Catherine-6/6 PASS per §3.5 (rate=1.0000) so is_active is the relevant filter.
        if not entry.get("is_active", False):
            continue
```
**LOC:** 1 line change (Option A or B); Option A more robust (handles both cohort schemas).
**Risk:** Backward compat — `wasserstein_transfer_rh.py:743` WRITES `catherine_6of6_pass=True` for ALL Wf-139 P7 synthetic compounds, so Option A's `catherine_6of6_pass OR is_active` would INCLUDE all synthetic entries (current behavior) + Wf-387-A2 real entries (new). Option B would EXCLUDE Wf-139 P7 entries that lack `is_active` — this BREAKS the existing Wf-125 P1 baseline invocation which uses `${SYNTH_RH_COHORT}` (the synthetic cohort).
**RECOMMENDATION:** Option A.

### Fix 2 (SECONDARY — closes B5-b flag surface; ~10 LOC)

**File:** `scripts/launch_wf125_p1_controlled_pair.sh`
**Lines:** 73-91 (flag parsing) + 133-143 (COMMON_ARGS)
**Current behavior:** Wrapper accepts `--epochs --run-mode --out-dir --dry-run` only; hardcodes EPOCHS=50, --metals Rh, --seeds 0.
**Proposed change (Option A — extend --seeds flag):**
Add `--seeds` handler to flag parser loop (line 73-83) similar to `--epochs`:
```bash
        --seeds)      shift_next=1 ;;
        --seeds=*)    SEEDS_OVERRIDE=(${arg#--seeds=}) ;;  # e.g. --seeds=0,1,2 → SEEDS_OVERRIDE=("0" "1" "2")
```
After line 91 capture loop, add SEEDS_OVERRIDE comma-split:
```bash
if [ -n "${SEEDS_OVERRIDE_RAW:-}" ]; then IFS=',' read -ra SEEDS_OVERRIDE <<< "${SEEDS_OVERRIDE_RAW}"; fi
```
Replace `--seeds 0` on line 136 with `--seeds ${SEEDS_OVERRIDE[@]:-0}` (default to 0 if not overridden).

**Proposed change (Option B — add `--balanced-cohort` as alias for `--run-mode real`):**
Add `--balanced-cohort` to flag parser at line 73-83:
```bash
        --balanced-cohort) RUN_MODE_OVERRIDE=("real") ;;
```

**LOC:** ~10 lines combined.
**Risk:** Minimal — backward-compat preserved (default values unchanged; new flags additive).

### Fix 3 (TERTIARY — closes B5-a; document that wrapper is canonical; ~0 LOC source patch)

**File:** `docs/WF125_P1_CONTROLLED_PAIR_ENTRY_POINTS.md` (NEW)
**Recommended content:**
```
Wf-125 P1 CONTROLD PAIR canonical entry points:
- WRAPPER (canonical): USER_OK=1 bash scripts/launch_wf125_p1_controlled_pair.sh [--dry-run] [--run-mode synthetic|real|none] [--epochs N] [--out-dir /abs/path]
- PYTHON SCRIPT: N/A — no molmetal/scripts/wf125_p1_controlled_pair.py exists by design.
- CONTEXT: This is a USER_OK-gated wrapper per Wf-294B family convention; the script-level entry point is intentionally not provided to prevent accidental un-gated launch.
```
**LOC:** 0 source patch + ~10 LOC docs.
**Risk:** Zero.

### Combined fix effort: ~11 LOC + ~10 LOC docs (~21 LOC total).

---

## §5 Risks + AGENTS.md §5.7 smoke requirements

### §5.7 NEW trainer flag wire-in smoke-required rule (per Wf-399-A1)

Per AGENTS.md §5.7 (added by Wf-399): every NEW flag wire-in MUST run a 50-step CPU smoke verifying `decode_ratio > 0` + `loss[0] > loss[49]`.

**APPLICABILITY to B5 fixes:**
- Fix 1 (trainer schema filter): NOT a new FLAG — internal filter logic change. §5.7 smoke MIGHT be required to verify backward compat with Wf-139 P7 synthetic cohort (Option A or B).
- Fix 2 (wrapper flag surface): wrapper changes don't affect trainer's training loop — §5.7 NOT required.
- Fix 3 (docs only): §5.7 NOT required.

**RECOMMENDED §5.7 smoke for Fix 1** (if Option A applied):
```bash
USER_OK=1 bash scripts/launch_wf125_p1_controlled_pair.sh --dry-run --run-mode synthetic
# verify SCAFFOLD smoke rc=0; cohort loaded successfully; n_synthetic_rh_appended=500
# Then with --run-mode real: verify rc=0; n_synthetic_rh_appended=500
# Both smoke runs in <30 s wall on CPU; loss should be finite; decode_ratio N/A (LOMO AUC not CFM decode)
```

### Other risks:

1. **Backward compat risk (Fix 1 Option B):** Option B filters on `is_active` which is NOT present in Wf-139 P7 synthetic cohort (per `rh_active_label_generation.py:195` writes `is_active` only if synth_data has it — and `wasserstein_transfer_rh.py:743` does NOT write `is_active`). Verify by reading `wasserstein_transfer_rh.py:730-750`:
   - RECOMMENDATION: Option A (accept either field) is safer.
2. **GPU launch safety (Fix 1 + Fix 2):** Per Wf-294B family + AGENTS.md §5.1, USER_OK gate MUST remain enforced BEFORE any actual launch. Current wrapper does this correctly at lines 96-102.
3. **Pre-flight gate impact (Fix 1 + Fix 2 + Fix 3):** NONE expected — `p3_wf343c2_per_wave_runtime_check` is preserved BY-DESIGN per Wf-347-A1 + Wf-345-D1 §3 honest-negative baseline; `p2_tests_pass` closed by Wf-424-A1 `use_dann` kwarg fix; `p3_wf343c4_yuelbond_decoder_env` env-gap closed by Wf-422-A2 venv invocation; no NEW pre-flight gate changes needed.
4. **Section 5.5 filesystem-truth compliance:** Fix 3 docs file MUST be written via `atomic_write_file` per AGENTS.md §5.6 — no direct `Write` tool to avoid half-merged file state.
5. **Honest framing (per AGENTS.md §1.2 + party-twenty-spirit §3.5):** Fix 1's claim that "Wf-139 P7 synthetic cohort has `catherine_6of6_pass` field" is MEASURED via grep on `wasserstein_transfer_rh.py:743`; Fix 2's flag-surface mapping is MEASURED via reading wrapper lines 73-91 + 133-143; Fix 3's "wrapper is canonical" is MEASURED via reading wrapper header comment lines 1-44.

---

## §6 Verification steps (how to confirm fix works; pre-flight gate impact)

### Verification of Fix 1 (schema mismatch):

```bash
# 1. Smoke test Option A (both fields accepted):
USER_OK=1 bash scripts/launch_wf125_p1_controlled_pair.sh --dry-run --run-mode synthetic
# Expect: SCAFFOLD_SMOKE_RC=0, DRY_RUN.log emitted, plan.json emitted, n_runs=1

USER_OK=1 bash scripts/launch_wf125_p1_controlled_pair.sh --dry-run --run-mode real
# Expect: SCAFFOLD_SMOKE_RC=0, DRY_RUN.log emitted, plan.json emitted, n_runs=1
# Previously this raised RuntimeError due to missing catherine_6of6_pass field.

# 2. Run 50-step CPU smoke per AGENTS.md §5.7 NEW rule:
PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/train_metal_lomo_wasserstein.py --epochs 2 --metals Rh --seeds 0 \
    --wasserstein-weight 0.05 --csv /home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv \
    --use-l2-guard --device cpu --rh-oversample 500 \
    --rh-cohort-json /home/hugo/codes/mol-metal/molmetal/reports/wf387a2_balanced_rh_cohort/balanced_rh_cohort.json \
    --out-dir /tmp/wf425_b5_smoke_real
# Expect: n_synthetic_rh_appended=500 in master.log; no RuntimeError.

# 3. Run 50-step CPU smoke on synthetic cohort (backward-compat):
PYTHONPATH=/home/hugo/codes/mol-metal /home/hugo/codes/CadButEaas/.venv-vlm/bin/python \
    molmetal/scripts/train_metal_lomo_wasserstein.py --epochs 2 --metals Rh --seeds 0 \
    --wasserstein-weight 0.05 --csv /home/hugo/codes/mol-metal/.data/molmetal/metalcytotox_5metal_multitask_v4.csv \
    --use-l2-guard --device cpu --rh-oversample 500 \
    --rh-cohort-json /home/hugo/codes/mol-metal/molmetal/reports/wf139_p7_rh_active_cohort/synthetic_rh_pos_neg_normalized.json \
    --out-dir /tmp/wf425_b5_smoke_synth
# Expect: n_synthetic_rh_appended=500 in master.log; no RuntimeError.
# Both smoke runs should complete in <30 s wall on CPU.
```

### Verification of Fix 2 (flag surface):

```bash
# 1. --epochs override:
USER_OK=1 bash scripts/launch_wf125_p1_controlled_pair.sh --dry-run --epochs 30
# Expect: EPOCHS=30 in DRY_RUN.log + plan.json (overrides EPOCHS=50 default)

# 2. --balanced-cohort alias (if Fix 2 Option B applied):
USER_OK=1 bash scripts/launch_wf125_p1_controlled_pair.sh --dry-run --balanced-cohort
# Expect: RUNS=[real] in DRY_RUN.log + plan.json (single CONTROLD PAIR run with real cohort)

# 3. --seeds override (if Fix 2 Option A applied):
USER_OK=1 bash scripts/launch_wf125_p1_controlled_pair.sh --dry-run --run-mode real --seeds 0,1,2
# Expect: COMMON_ARGS includes --seeds 0 1 2; SCAFFOLD smoke iterates over 3 seeds (3x n_runs=1)
```

### Verification of Fix 3 (docs only):

```bash
# 1. Verify docs file exists on disk per AGENTS.md §5.5:
ls -la docs/WF125_P1_CONTROLLED_PAIR_ENTRY_POINTS.md
# Expect: file exists with content matching §4-Fix-3 spec.
```

### Pre-flight gate impact:

- BEFORE fix: pre-flight v3 fresh-run is at Wf-424-A7 baseline **114 PASS / 0 FAIL / 3 SKIP / 117 TOTAL wall 186.48 s** (per `/tmp/wf424a7_preflight.json`).
- AFTER fix: NO change expected — Fix 1 + 2 + 3 don't touch pre-flight gates. EXPECTED_GATE_COUNTS["all"] = 117 PRESERVED at Wf-353-B1 baseline.
- If Fix 1 introduces any pre-existing tests (e.g. `train_metal_lomo_wasserstein.py` test for new filter logic): +1 gate; CURRENT 117 → 118.
- VERIFY: re-run `pre_flight_check.py --pipeline all` after fix is applied.

### End-to-end verification:

```bash
# After all 3 fixes applied:
USER_OK=1 bash scripts/launch_wf125_p1_controlled_pair.sh --dry-run
# Expect: SCAFFOLD_SMOKE_RC=0, n_runs=3 (synthetic|real|none), DRY_RUN.log + plan.json emitted.
# Then with USER_OK=1 + actual launch (~6 hr wall):
USER_OK=1 bash scripts/launch_wf125_p1_controlled_pair.sh --run-mode real
# Expect: 3 CONTROLD PAIR runs executed sequentially; each ~30 min on RTX PRO 6000 Blackwell sm_120.
# After completion: CONTROLED_PAIR_SUMMARY.json emitted at $OUT_BASE/ with verdict taxonomy per wrapper header.
```

---

## §7 Honest framing caveats (MEASURED vs PROJECTED; any side-effects on bit-for-bit backward compat)

### MEASURED (verified at Wf-425 investigator timestamp 2026-10-01 ~04:05 CST):
- `molmetal/scripts/wf125_p1_controlled_pair.py` does NOT exist on disk (`ls` returns no matches).
- `scripts/launch_wf125_p1_controlled_pair.sh` exists (19,892 B mtime 2026-09-30 22:22) per `ls -la`.
- Trainer `molmetal/scripts/train_metal_lomo_wasserstein.py:211` filters on `catherine_6of6_pass` field (audited via grep).
- Wf-387-A2 cohort JSON `per_compound[]` schema has 9 fields: `['compound_id', 'is_active', 'is_synthetic', 'metal', 'scaffold_id', 'smiles', 'source_csv_row_hash', 'source_split', 'wf139_p7_carryover']` — NO `catherine_6of6_pass` field (audited via Python JSON parse).
- Wrapper flag parser accepts ONLY `--dry-run`, `--run-mode`, `--epochs`, `--out-dir` (audited via reading lines 73-91).
- `catherine_6of6_pass` is the canonical field name used across `wasserstein_transfer_rh.py:743` + `rh_active_label_generation.py:89,195,235` + `wf379a1_normalize_rh_cohort.py:40` + `closure_oversample.py:204` + `train_metal_lomo_wasserstein.py:211` (audited via grep across codebase).
- 12 NEW tests PASS in `molmetal/tests/test_wf424a1_use_dann_kwarg.py` (per Wf-424-A1 closeout).
- Pre-flight v3 fresh-run at Wf-424-A7 timestamp = **114 PASS / 0 FAIL / 3 SKIP / 117 TOTAL wall 186.48 s** (per `/tmp/wf424a7_preflight.json`).

### PROJECTED (not yet measured):
- Fix 1 (Option A) BACKWARD COMPAT: Wf-139 P7 synthetic cohort with `catherine_6of6_pass=True` on ALL entries continues to work; new Wf-387-A2 real cohort with `is_active` balanced 250/250 also works; Wf-388-A2 balanced 500 cohort 0 scaffold leak to Rh val/test PRESERVED.
- Fix 1 (Option A) MEASURED lift: 0 lift vs Wf-125 P1 baseline (lift is on the CONTROLD PAIR design question, not on the schema fix); MEASURED lift requires actual GPU launch (~6 hr wall) per wrapper PROJECTED.
- Fix 2 (--seeds override) MEASURED: SCAFFOLD smoke with `--seeds 0,1,2` should complete in <60 s wall on CPU; 3 seeds x 30 epochs x 50 epochs PROJECTED ~3 hr wall on RTX PRO 6000 Blackwell sm_120 per Wf-260 G6 timing.
- Fix 3 (docs only) MEASURED: 0 LOC source patch; docs file rendered at `$DOCS/WF125_P1_CONTROLLED_PAIR_ENTRY_POINTS.md`.

### Honest-negative caveats:
1. **B5-a BLOCKER (phantom `.py`) is NOT-A-BLOCKER per filesystem-truth rule** — the wrapper IS the canonical entry point per Wf-387-A2 §5.1 + Wf-416-B1 §22.101.4. Reporting this as a "missing file" without acknowledging the wrapper would be a false claim.
2. **B5-b BLOCKER (flag surface) is PARTIAL** — `--user-ok` -> env var is CORRECT convention (NOT a wrapper bug); `--epochs 30` works as override; `--balanced-cohort` -> `--run-mode real` is semantic equivalent; only `--seeds 0,1,2` is the true wrapper gap (wrapper plans 3 CONTROLD PAIR runs not 3 seeds of same run).
3. **B5-c BLOCKER (schema mismatch) is REAL** — trainer's pre-existing filter at line 211 was authored for Wf-139 P7 synthetic cohort (which has `catherine_6of6_pass`); Wf-387-A2 real balanced cohort OMITTED this field; CLOSED by Fix 1 (~1 LOC).
4. **Bit-for-bit backward compat (Fix 1 Option A) PRESERVED** — Wf-139 P7 synthetic cohort has `catherine_6of6_pass=True` on ALL entries per `wasserstein_transfer_rh.py:743`; Option A filter `catherine_6of6_pass OR is_active` returns True for all Wf-139 P7 entries (current behavior) + True for Wf-387-A2 entries with `is_active=True` (250/500); Fix 1 Option B would BREAK Wf-139 P7 invocation (filter on `is_active` only — but Wf-139 P7 entries lack `is_active` field).
5. **Bit-for-bit backward compat (Fix 2) PRESERVED** — wrapper flag parser additions are additive; existing flags unchanged; default COMMON_ARGS unchanged; `EPOCHS_OVERRIDE` only applied if set; `SEEDS_OVERRIDE` only applied if set; `RUN_MODE_OVERRIDE` only applied if set.
6. **m6_kappa + m2 R² + m1 AUC structural gates HONEST_NEGATIVE_FINAL PRESERVED** — out of scope for B5 fix (per Wf-411-A4 + Wf-419-A2 + Wf-388-A1 prior verifications); 6+ forward-looking USER_OK items (~52.9 hr GPU + 13 hr CPU + 30 min install) still BLOCKED on user OK per AGENTS.md §5.1.
7. **Wetlab items DEPRECATED** per AGENTS.md §5.4 + §10 — NO wet-lab planning or execution.

---

## §8 Cross-references (which other Wf-X closeouts must be updated)

### MUST update after fix:

1. **`TODO/completed/387a2_balanced_rh_cohort.md`** — add note in §3.5 schema documentation that `catherine_6of6_pass_rate` aggregate stat is `1.0000` (per §3.5 row 32) BUT `catherine_6of6_pass` per-compound field is NOT serialized; trainer filter at line 211 must be updated to either accept `is_active` OR serialize `catherine_6of6_pass` per-compound.
2. **`TODO/completed/424a10_wf388_wf125_controlled_pair.md`** — add Wf-425 B5 fix note in §11 cross-references section: "B5 schema mismatch CLOSED by Wf-425 investigator Phase 2 fixer via Fix 1 in `molmetal/scripts/train_metal_lomo_wasserstein.py:211` (1 LOC change to accept either `catherine_6of6_pass` OR `is_active` filter field)."
3. **`TODO/completed/424a11_wf390_m2_r2_path_a.md`** — add cross-ref to Wf-425 B5 fix in §10 forward-looking section noting that B5 fix enables the broader Wf-390-A1 retrain pattern (CONTROLD PAIR design applies to other (metal, seed) sweeps).
4. **`TODO/completed/416b1_index_sync.md`** — add Wf-425 B5 fix note in §22.101 cross-references: "Wf-388 Wf-125 P1 CONTROLD PAIR retrain surface (CLOSED by Wf-425 B5 fix per `train_metal_lomo_wasserstein.py:211` schema adapter)."

### MUST create new:

5. **`docs/WF125_P1_CONTROLLED_PAIR_ENTRY_POINTS.md`** (NEW, Fix 3) — documents wrapper as canonical entry point + USER_OK gate convention + flag-surface semantics.

### MUST keep current (no update needed):

6. **`scripts/launch_wf125_p1_controlled_pair.sh`** — stays as canonical wrapper; Fix 2 adds 2 new flags but doesn't change existing semantics.
7. **`molmetal/scripts/train_metal_lomo_wasserstein.py`** — Fix 1 changes 1 LOC at line 211; all other code paths preserved.
8. **`molmetal/reports/wf387a2_balanced_rh_cohort/balanced_rh_cohort.json`** — stays as-is; Fix 1 in trainer preserves backward compat without modifying cohort schema.

### Cross-check completion status:

| Predecessor | Files | Action |
|-------------|-------|--------|
| Wf-347-A1 (B1) | `TODO/completed/347a1_wf296_integration.md` | NO UPDATE — pre-flight gate unchanged |
| Wf-367-A2 (B1) | `TODO/completed/367a2_preflight_no_fail_gates_fix.md` | NO UPDATE — cascade parent unchanged |
| Wf-413-A3 (B2) | `TODO/completed/413a3_wf299_wave_b_launch.md` | NO UPDATE — different launcher (Wf-299 vs Wf-125) |
| Wf-409-A3 (B3) | `TODO/completed/409a3_wf246c_scaffold.md` | NO UPDATE — wrapper convention same |
| Wf-299B (B3) | N/A (pre-AGENTS.md convention) | NO UPDATE — out of scope |
| Wf-394-A1 (B4) | `TODO/completed/394a1_lambda_cfm_wire.md` | NO UPDATE — different trainer (CFM not LOMO) |
| Wf-329-W6 (B4) | `TODO/completed/329w6_todo21_lambda_cfm_verified.md` | NO UPDATE — different trainer |
| Wf-387-A2 (B5) | `TODO/completed/387a2_balanced_rh_cohort.md` | UPDATE §3.5 schema documentation (5 NEW LOC) |
| Wf-139 P7 (B5) | (canonical synthetic cohort — no closeout file) | NO UPDATE — synthetic cohort schema unchanged |
| Wf-405-A2 (B6) | `TODO/completed/405a2_wf380_loss_wire.md` | NO UPDATE — different trainer (D-MPNN not LOMO) |
| Wf-380-A2 (B6) | `TODO/completed/380a2_per_cohort_features.md` | NO UPDATE — different trainer |
| Wf-424-A10 (current cycle) | `TODO/completed/424a10_wf388_wf125_controlled_pair.md` | UPDATE §11 cross-references (3 NEW LOC) |
| Wf-424-A11 (current cycle) | `TODO/completed/424a11_wf390_m2_r2_path_a.md` | UPDATE §10 forward-looking (3 NEW LOC) |
| Wf-416-B1 (current cycle) | `TODO/completed/416b1_index_sync.md` | UPDATE §22.101 cross-references (3 NEW LOC) |

---

## §9 Appendix: AGENTS.md §5.5 filesystem-truth ls -la evidence

```bash
$ ls -la molmetal/scripts/train_metal_lomo_wasserstein.py \
        scripts/launch_wf125_p1_controlled_pair.sh \
        molmetal/reports/wf387a2_balanced_rh_cohort/balanced_rh_cohort.json
-rw-r--r-- 1 hugo hugo  48958 Sep 27 22:18 molmetal/scripts/train_metal_lomo_wasserstein.py
-rwx--x--x 1 hugo hugo  19892 Sep 30 22:22 scripts/launch_wf125_p1_controlled_pair.sh
-rw-r--r-- 1 hugo hugo 198302 Sep 30 10:03 molmetal/reports/wf387a2_balanced_rh_cohort/balanced_rh_cohort.json

$ ls -la molmetal/scripts/wf125_p1_controlled_pair.py
ls: cannot access 'molmetal/scripts/wf125_p1_controlled_pair.py': No such file or directory

$ ls TODO/completed/ | grep -E "(387a2|394a1|329w6|405a2|413a3|347a1|367a2|409a3|424a10|424a11)"
329w6_todo21_lambda_cfm_verified.md
347a1_wf296_integration.md
367a2_preflight_no_fail_gates_fix.md
387a2_balanced_rh_cohort.md
394a1_lambda_cfm_wire.md
405a2_wf380_loss_wire.md
409a3_wf246c_scaffold.md
413a3_wf299_wave_b_launch.md
424a10_wf388_wf125_controlled_pair.md
424a11_wf390_m2_r2_path_a.md
```

All 10 predecessor closeouts VERIFIED on disk per AGENTS.md §5.5 rule #1.

---

**END OF REPORT** (Wf-425 B5 investigator Phase 1 review).
