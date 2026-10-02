# Wf-421-A2 — YuelBond Decoder Env Gate Audit

**Wave:** Wf-421 ultracode residual CPU-only docs+verify wave (response to user voice
2026-09-30 verbatim *"还有什么没修的bloker吗，也启动ultracode去修"* via Wf-419-A2 closeout
relay)
**Author:** Wf-421-A2 subagent (sub-flow A2 of the Wf-421 residual ultracode wave)
**Date:** 2026-09-30 (Wf-421 wave timestamp)
**Pipeline:** `chemprop_v2_v3` (lookup context) + `wf343c4` (gate pipeline)
**Cross-refs:** Wf-419-A2 (sister sub-flow), Wf-345-A2 (env-aware SKIP contract),
AGENTS.md §5.4 (Tier-2 + NO wet-lab), §5.5 (post-wave on-disk verification rule),
§5.6 (atomic write protocol adoption rule), §5.7 NEW (trainer flag wire-in
smoke-required rule), TODO-84 (no `paper/*` work), Wf-272I (CFM YuelBond-
conditionality enforcement), Wf-269C (MEASURED `--no-yuelbond` → `decode_ratio = 0.000`
on real cohort honest-framing)

---

## §0 TL;DR

The `p3_wf343c4_yuelbond_decoder_env` pre-flight gate is **NOT** a regression and
**NOT** a bug. It is a **BY-DESIGN env-aware SKIP contract** introduced by
Wf-345-A2. The gate verdict depends entirely on **which interpreter** runs the
pre-flight:

| Interpreter | YuelBond gate verdict | Reason |
|---|---|---|
| System Python (`/usr/bin/python` / bare `python`) | **SKIP** (BY-DESIGN) | `ModuleNotFoundError: No module named 'torch'` because the canonical `cfm_decoder_yuelbond.py` line 59 does `import torch` at module top-level |
| `/home/hugo/codes/mol-metal/.venv/bin/python` (with torch 2.14.0+cu130) | **PASS** | decoder file (23792B) is on disk + syntactically loadable + exports the `YuelBondDecoder` class |

The Wf-411-B1 baseline reported `p3_wf343c4_yuelbond_decoder_env = SKIP` because
**Wf-411-A1 ran `pre_flight_check.py` with bare `python`** (system interpreter,
no torch) — see `TODO/completed/411a1_gpu_env_probe.md` shell snippet:

```
$ PYTHONPATH=/home/hugo/codes/mol-metal python molmetal/scripts/pre_flight_check.py \
    --pipeline chemprop_v2_v3 --strict
```

When the same `pre_flight_check.py` script is invoked via the project venv
(`/home/hugo/codes/mol-metal/.venv/bin/python`, which has torch 2.14.0+cu130
installed), the same gate transitions from SKIP → **PASS**. **No fix is needed**
because the gate is honoring its Wf-345-A2 env-aware SKIP contract correctly:
it surfaces the env dependency to the operator without blocking (the gate only
raises exit-code 2 under `--strict`).

---

## §1 Decoder file existence + import check

### §1.1 File existence + size (MEASURED)

```
$ ls -la /home/hugo/codes/mol-metal/molmetal/molmetal_lam/lam_chem/cfm_decoder_yuelbond.py
-rw-r--r-- 1 hugo hugo 23792 Sep 27 03:14 \
    /home/hugo/codes/mol-metal/molmetal/molmetal_lam/lam_chem/cfm_decoder_yuelbond.py

$ stat -c "%s" /home/hugo/codes/mol-metal/molmetal/molmetal_lam/lam_chem/cfm_decoder_yuelbond.py
23792
```

**MEASURED file size: 23792B (~23.8 KB)**, NOT the 56 KB that Wf-419-A2's closeout
referenced. The Wf-419-A2 reference appears to have either pre-dated a refactor
or used a different file path. The canonical file at
`molmetal/molmetal_lam/lam_chem/cfm_decoder_yuelbond.py` is 23792B at
2026-09-27 03:14 mtime.

### §1.2 Import check (MEASURED — PASS on this venv)

```
$ /home/hugo/codes/mol-metal/.venv/bin/python -c \
    "from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import \
        YuelBondDecoder; print('OK')"
[molmetal] applied gfx1101 env vars: TOKENIZERS_PARALLELISM=false, TRANSFORMERS_VERBOSITY=error
OK
```

**MEASURED verdict**: `YuelBondDecoder` import SUCCEEDS on the project venv.

### §1.3 Class export verification (MEASURED — PASS)

```
$ grep -c "class YuelBondDecoder" \
    /home/hugo/codes/mol-metal/molmetal/molmetal_lam/lam_chem/cfm_decoder_yuelbond.py
1
```

`YuelBondDecoder` class IS defined and exported in `cfm_decoder_yuelbond.py`.

---

## §2 Gate SKIP reason — env gap, NOT regression

### §2.1 Wf-411-A1 SKIP context (the prior wave that triggered this audit)

Wf-411-A1 ran the pre-flight with **bare system Python** (no torch). The gate
detected `ModuleNotFoundError: No module named 'torch'` from the decoder module's
top-level `import torch` (line 59 of `cfm_decoder_yuelbond.py`) and emitted SKIP
with this evidence:

> decoder module requires 'torch' (env gap on host Python);
> cfm_decoder_yuelbond.py CANNOT be loaded without the expected torch-enabled
> runtime (e.g. kanzi_venv or /home/hugo/codes/CadButEaas/.venv-vlm/bin/python per
> AGENTS.md §5.1). Module is on disk
> (/home/hugo/codes/mol-metal/molmetal/molmetal_lam/lam_chem/cfm_decoder_yuelbond.py,
> 23792B) and structurally loadable — this is an env gap, not a regression.
> Original probe error: ModuleNotFoundError: No module named 'torch'

Verbatim evidence pulled from `molmetal/reports/wf411a1_preflight.json` (gate
verdict line for `p3_wf343c4_yuelbond_decoder_env`, status: SKIP).

### §2.2 Wf-345-A2 env-aware SKIP contract (the design contract that was honored)

The gate's docstring at `molmetal/scripts/pre_flight_check.py:4099-4154` codifies
the contract:

> Env-aware SKIP contract (Wf-345A2): The canonical `cfm_decoder_yuelbond.py`
> imports torch / import torch.nn as nn at module top-level. On a host Python
> without torch (e.g. the bare /usr/bin/python interpreter, the default pytest
> invocation), the structural check is **not verifiable** — not because the
> on-disk module is broken, but because the gate's import probe requires the
> runtime that the canonical CFM retrain runs in (per AGENTS.md §5.1 standing
> GPU grant + §5.2 honest-framing; expected runtime = the torch-enabled venv,
> e.g. kanzi_venv or /home/hugo/codes/CadButEaas/.venv-vlm/bin/python). We treat
> this as an **honest-negative SKIP** rather than a FAIL because:
> 1. The decoder module is on disk + syntactically loadable.
> 2. The only reason the probe cannot PASS is the host env gap, not a regression
>    in the module itself.
> 3. A SKIP surfaces the env dependency to the operator without blocking it
>    (the gate only raises exit-code 2 under `--strict`).
> On `ModuleNotFoundError` / `ImportError` for `torch` the gate emits SKIP with
> reason naming the missing dependency + expected runtime. CPU-only (no
> torch.cuda, no subprocess) — runs in < 5 ms on the host.

### §2.3 PASS on project venv (MEASURED — this audit run)

When invoked via the project venv, the gate transitions SKIP → PASS:

```
$ .venv/bin/python -c "
import importlib.util as _ilu
from pathlib import Path
from molmetal.scripts.pre_flight_check import gate_wf343c4_yuelbond_decoder_env, LAM_CHEM_DIR
decoder_module = LAM_CHEM_DIR / 'cfm_decoder_yuelbond.py'
result = gate_wf343c4_yuelbond_decoder_env(
    'p3_wf343c4_yuelbond_decoder_env', 'P3', 'wf343c4', decoder_module,
)
print('RESULT:', result.status)
print('EVIDENCE:', result.evidence)
"
[molmetal] applied gfx1101 env vars: TOKENIZERS_PARALLELISM=false, TRANSFORMERS_VERBOSITY=error
RESULT: PASS
EVIDENCE: cfm_decoder_yuelbond.py (23792B) exports YuelBondDecoder \
    (per Wf-272I / Wf-269C YuelBond-conditionality structural check; \
    --no-yuelbond → decode_ratio 0.000 MEASURED on real cohort)
```

Wall_sec ≈ 0.36 (≈ same speed as the SKIP path — torch import + class lookup is
microseconds).

---

## §3 Fix attempt — none required (env gap is BY-DESIGN)

### §3.1 No code fix attempted

Per the Wf-345-A2 env-aware SKIP contract, **no code fix is required**. The gate is
working exactly as designed:

  * PASS when torch is importable + decoder class exported.
  * SKIP when torch is missing + decoder module is on disk (operator-friendly
    honest-negative; not a regression).
  * FAIL when decoder module is missing, empty, or doesn't export `YuelBondDecoder`.

### §3.2 Operator guidance (not a fix)

When running pre-flight on a host without torch (e.g. bare `python` invocation in
a CI runner), the `p3_wf343c4_yuelbond_decoder_env` gate will SKIP. To convert
this SKIP → PASS, simply invoke `pre_flight_check.py` via the project's torch-
enabled venv:

```
# SKIP (system Python, no torch):
$ python molmetal/scripts/pre_flight_check.py --pipeline all --strict
...
p3_wf343c4_yuelbond_decoder_env: SKIP (env gap, not regression)

# PASS (project venv, torch available):
$ /home/hugo/codes/mol-metal/.venv/bin/python \
    molmetal/scripts/pre_flight_check.py --pipeline all --strict
...
p3_wf343c4_yuelbond_decoder_env: PASS \
    (cfm_decoder_yuelbond.py (23792B) exports YuelBondDecoder)
```

This is the **canonical fix** for the env gap, NOT a code change.

### §3.3 Gate docstring cross-reference

The gate's failure modes are explicit at `pre_flight_check.py:4143-4144`:

> FAIL on (a) the module is missing / empty, (c) the requested class is not
> exported. Other unexpected exceptions still FAIL (preserving the original
> Wf-343C4 contract).

The Wf-411-A1 SKIP context did NOT match any of the FAIL criteria — only the
SKIP criteria. Therefore the gate is behaving correctly.

---

## §4 Honest framing per `party-twenty-spirit` §3.5

### §4.1 MEASURED outcomes

  * **File size: 23792B** (MEASURED via `stat -c "%s"` at Wf-421-A2 timestamp 2026-09-30).
  * **`YuelBondDecoder` class exported: YES** (MEASURED via `grep -c "class YuelBondDecoder"` = 1).
  * **Direct import under `.venv/bin/python`: PASS** (MEASURED at Wf-421-A2 timestamp).
  * **Direct invocation under system `python`: ModuleNotFoundError on `torch`** (MEASURED per Wf-411-A1 evidence verbatim).
  * **Pre-flight verdict under `.venv/bin/python` for `pipeline=all`: PASS** (MEASURED at Wf-421-A2 timestamp via `/tmp/wf421a2_preflight_all.json` 38,391 bytes; gate verdict line:
    `{"name":"p3_wf343c4_yuelbond_decoder_env","priority":"P3","pipeline":"wf343c4","status":"PASS","evidence":"cfm_decoder_yuelbond.py (23792B) exports YuelBondDecoder (per Wf-272I / Wf-269C YuelBond-conditionality structural check; --no-yuelbond → decode_ratio 0.000 MEASURED on real cohort)","wall_sec":0.358}`).

### §4.2 PROJECTED claims

  * None — this audit reports only MEASURED outcomes.

### §4.3 Honest-negative framing

The `p3_wf343c4_yuelbond_decoder_env = SKIP` preserved in Wf-411-B1 baseline is
**NOT** a regression, **NOT** a blocker, and **NOT** a fix-needed item. It is
the **correct** verdict under the Wf-345-A2 env-aware SKIP contract for a
torch-less Python interpreter. When the pre-flight is invoked via the project
venv, the gate transitions SKIP → PASS, confirming that:

  1. The decoder module IS on disk + structurally loadable.
  2. The `YuelBondDecoder` class IS exported.
  3. The torch import is the **only** blocker, and it's an env gap (not a code
     regression).

Therefore: **NO action needed**; **NO code fix needed**; **NO env install
needed** (the project venv already has torch 2.14.0+cu130).

### §4.4 Other blockers PRESERVED

This audit does NOT resolve any of the other 2 FAILs + 2 SKIPs in the
Wf-411-B1 baseline. The full preserved honest-negative set per AGENTS.md §5.5
filesystem-truth rule:

  * `p2_tests_pass` — FAIL preserved per Wf-358-D1 §1 + Wf-367-A1 cascade fix
    (parent closed; child FAIL pending on `train_dmpnn_multitask_wasserstein.py:2386`
    argparse `%`-escape fix per Wf-357-C3 §5.1 Recommendation 1, ~5 LOC CPU fix).
  * `p3_loi_send_ready` — FAIL preserved per Wf-375-B1 §22.68 baseline (Tier-2
    reclassification cascade from 2026-09-30 wet-lab deprecation inline edits).
  * `p3_wf343c4_anchor_freshness_24h` — FAIL preserved per Wf-375-B1 §22.68
    baseline (wf269c_report.json age > 24h STALE natural drift; closure path =
    next-touch refresh).
  * `p3_wf343c4_wave_budget_remaining_nonneg` — SKIP preserved (BY-DESIGN; Wf-262
    §7.1 -183.04h overrun).
  * `data_2_1_6_bindingdb_external_honest_negative` — SKIP preserved
    (BY-DESIGN; BindingDB 1.07M external dependency).
  * `p3_wf343c4_yuelbond_decoder_env` — SKIP preserved (BY-DESIGN env-aware;
    converts to PASS under `.venv/bin/python`).

---

## §5 Cross-references

### §5.1 Upstream

  * `TODO/completed/419a2_*.md` (Wf-419-A2 — sister sub-flow; referenced 56 KB
    file size — likely pre-dated a refactor OR referred to a different path;
    current canonical file is 23792B per §1.1 MEASURED).
  * `TODO/completed/345a2_yuelbond_decoder_env.md` (Wf-345-A2 — env-aware SKIP
    contract author; this audit verifies the contract is honored correctly).
  * `TODO/completed/411a1_gpu_env_probe.md` (Wf-411-A1 — the prior wave that
    triggered the SKIP that became the focus of this audit; Wf-411-A1 ran
    `pre_flight_check.py` with bare `python`).
  * `TODO/completed/421_final_synthesis.md` (Wf-421-B1 — comprehensive final
    synthesis for the Wf-421 ultracode wave; this closeout is referenced).
  * `TODO/completed/421a1_*.md` (Wf-421-A1 — sister sub-flow).

### §5.2 Downstream

  * `TODO/completed/421b1_index_sync.md` (Wf-421-B1 — INDEX + AGENTS sync;
    this row added per filesystem-truth rule).
  * `TODO/INDEX.md` §22 (NEW §22.x Wf-421 entry appended per §5.5).
  * `AGENTS.md` §6 (NEW §6 row appended per §5.5).

### §5.3 AGENTS.md standing directives referenced

  * §1.5 (MEASURED-vs-PROJECTED tag discipline — applied; §4 only MEASURED).
  * §5.1 (Standing GPU authorization — preserved; no GPU launches performed
    during this audit; torch is already in the project venv).
  * §5.2 (Honest framing — applied; no fabrication of metrics; SKIP verdict is
    honestly preserved as BY-DESIGN env gap).
  * §5.3 (No paper work without authorization — preserved; this is a docs+verify
    audit; no `paper/*` touches).
  * §5.4 + §10 (Tier-2 + NO wet-lab — preserved; no wet-lab planning).
  * §5.5 (Post-wave on-disk verification rule — strictly observed; this closeout
    + the Wf-421 final synthesis + INDEX entry are ALL on disk before B1 sync).
  * §5.6 (Atomic write protocol adoption rule — strictly observed; this closeout
    was written via `atomic_write_file` per Wf-386-A3; backup at
    `421a2_yuelbond_decoder_audit.md.bak.wf421a2` per `safe_overwrite_check`).
  * §5.7 NEW (Trainer flag wire-in smoke-required rule — observed; this audit
    ran a direct import smoke + a `pre_flight_check.py --pipeline all` smoke).

### §5.4 Code artifacts

  * `molmetal/molmetal_lam/lam_chem/cfm_decoder_yuelbond.py` (23792B; canonical
    decoder module; unchanged; PASS verification target).
  * `molmetal/scripts/pre_flight_check.py:4092-4232` (`gate_wf343c4_yuelbond_decoder_env`;
    gate implementation; verified honored correctly under both system Python
    [SKIP] and project venv [PASS]).
  * `molmetal/scripts/wf386a3_atomic_write.py` (atomic_write_file primitive
    used to author this closeout).
  * `molmetal/reports/wf411a1_preflight.json` (Wf-411-A1 prior preflight; shows
    SKIP for yuelbond gate under system Python).
  * `/tmp/wf421a2_preflight_all.json` (this audit's fresh preflight; shows
    PASS for yuelbond gate under project venv).

### §5.5 Cross-wave priority

This audit (Wf-421-A2) is **not** in the blocker-resolution priority queue.
It is a documentation + verification closeout that confirms the Wf-411-B1
SKIP-on-yuelbond-gate observation is BY-DESIGN per Wf-345-A2 contract and
NOT a fix-needed item. Future wave priority (per Wf-368-B1 + Wf-370-B1 +
Wf-375-B1 + Wf-377-B1 + Wf-407-B1 + Wf-413-B1 + Wf-414-B1 forward-looking
recommendations) remains:

  1. Wf-415 / Wf-416+ USER_OK-gated m2 R² Path A GPU retrain (Wf-414-A1 wrapper
     ready; BLOCKED on USER_OK=1 + 12 hr GPU).
  2. Wf-415+ USER_OK-gated paper §6 v4 write from
     `docs/PAPER_SECTION_6_v4_LIMITATIONS_READY.md` (~12-15 hr CPU).
  3. Wf-415+ USER_OK-gated paper §3 v4 update from
     `docs/PAPER_SECTION_3_METRIC_TABLE.md` (~10 hr CPU).
  4. Wf-415+ USER_OK-gated Wf-394-A1 Lambda CFM wire-in patch application
     (~10 min CPU + ~5 hr GPU).
  5. Wf-415+ USER_OK-gated Wf-246-C metal-coordination term retrain
     (~3 hr GPU).
  6. Wf-415+ atomic write protocol adoption migration completion
     (~41 more scripts; ~30 min CPU).
  7. Wf-415+ USER_OK-gated Wf-379-A1 m1_auc 5-seed result close + Wf-379-A2
     SCAFFOLD_NOT_FUNCTIONAL trainer extension (already 7/7 complete per
     Wf-413-A1; remains a Wf-378-A2 SCAFFOLD_NOT_FUNCTIONAL gap).
  8. pre-flight FAIL closure: `p3_loi_send_ready` regex multi-line fix
     (~20 LOC per Wf-405-A3); `p3_wf343c4_anchor_freshness_24h` next-touch
     refresh (single `touch` per Wf-405-A3).
  9. pre-flight `p2_tests_pass` cascade child closure: argparse `%`-escape
     fix at `train_dmpnn_multitask_wasserstein.py:2386` (~5 LOC per
     Wf-357-C3 §5.1 Recommendation 1; CPU-only fix).
 10. pre-flight `p3_wf343c4_yuelbond_decoder_env` is NOT a fix-needed item —
     preserve BY-DESIGN SKIP contract; document env-gap fix path (use
     `.venv/bin/python` instead of bare `python` for the gate run).

---

**wf421a2_pcn_summary:** YuelBond decoder env gate is **NOT a fix-needed item**.
The Wf-411-B1 SKIP-on-yuelbond-gate observation is the **correct** verdict
under the Wf-345-A2 env-aware SKIP contract for a torch-less Python interpreter.
When the pre-flight is invoked via the project venv, the gate transitions SKIP
→ PASS. **No code fix needed; no env install needed (project venv already has
torch 2.14.0+cu130).** Document the env-gap fix path (use `.venv/bin/python`
instead of bare `python`) for operator guidance. MEASURED-only claims;
PROJECTED = none. Wf-421-A2 SHIPPED; this is the closeout record per
AGENTS.md §5.5 filesystem-truth rule + §5.6 atomic write protocol adoption
rule.
