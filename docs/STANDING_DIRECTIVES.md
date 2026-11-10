# Mol-Metal Standing Directives

**Last updated:** 2026-10-09 (Wf-AGENTS-TRIM)
**Audience:** AI agents (canonical reference for AGENTS.md §5.4 / §5.5 / §5.6 / §5.7 / §5.8 / §5.9 / §5.10)
**Purpose:** Self-contained expansion of AGENTS.md §5 standing-directive bullets. AGENTS.md only carries
a 3-line pointer per rule + a one-line "why" + cross-ref here. THIS file carries the full mandate text.

The split is deliberate: every subagent inherits AGENTS.md at spawn time, so AGENTS.md must stay
slim. A subagent only needs the cross-ref pointer to know "this rule exists and where to read it".
The full text is for the agent that is actively applying the rule.

---

## §A Tier-2 journal target + NO wet-lab planning (AGENTS.md §5.4)

**⚠️ HARD MANDATE — DO NOT PLAN, EXECUTE, OR SEND ANY WET-LAB WORK. ⚠️**

**Verbatim user directives across multiple sessions:**
- 2026-09-29: "我们发二区就可以了，不用规划湿实验" — Tier-2 sufficient; do NOT plan wet-lab
- 2026-09-30: "还有就是搞清楚我们不用做湿实验，这一点我已经明确了好多遍了" — explicit frustration; reinforced

User said: "我们发二区就可以了，不用规划湿实验" — Tier-2 journal is sufficient; do NOT
plan wet-lab experiments.

**Paper target:** Tier-2 (downgrade from JCIM Tier-1). Examples: J. Comput. Aided Mol. Des.
(JCAMD), J. Mol. Graph. Model., Comput. Chem. (WIREs), Mol. Divers., Comput. Biol. Chem. No
wet-lab data required for Tier-2 acceptance — pure computational submission is in scope.

**Wet-lab items DEPRECATED for project scope** (keep templates in `TODO/pending/` as reference only):
- 4 LOI drafts (Rusanov / PKU / 协和 / ORNL) → `TODO/pending/wetlab_loi_*.md` — DO NOT SEND
- 3 wetlab SOPs (cytotoxicity / cellular uptake / binding affinity) → `TODO/pending/wetlab_sop_*.md` — DO NOT EXECUTE
- 1 IC50 analysis pipeline → `TODO/pending/wetlab_ic50_analysis_pipeline.md` — DO NOT RUN
- TODO-138 paper v2 → Tier-2 journal target (re-scope from JCIM)
- R20-P6 wet-lab IC50 validation (TODO-97) → DEPRECATED; re-scope to in-silico metric lift
- Wf-74 v2 wet-lab outreach → DEPRECATED
- LAMBDA_ROADMAP §3.2 R22-P1 (m18 wet-lab) → re-scope as computational-only
- LAMBDA_ROADMAP §3.3 R23 (Phase I clinical) → DEPRECATED

**Why:** Tier-2 acceptance bar is computational-only; wet-lab validation is 6-12 month
external-dependency that does not gate Tier-2 acceptance.

**How to apply:**
- Every `metrics/by_metric/*.json` and `TODO/completed/*.md` is Tier-2-shippable as is.
- Wf-316 TODO directory audit should reclassify wet-lab items as `TEMPLATE_NOT_NEEDED_FOR_TIER2`.
- Plan files (WEAK_TO_STRONG / IMPROVEMENT_ENUM / LAMBDA_ROADMAP) should re-scope wet-lab items.
- TODO-84 still applies for `paper/*` touches — paper work is user-authorization gated.

**Cross-ref:** `docs/WETLAB_DEPRECATION_FAQ.md` (the canonical wetlab-FAQ doc with
§10.1-§10.6 detail) + `docs/HONEST_NEGATIVE_CATALOG.md` (per-wetlab-file honest-negative
entries, Wf-375 deferred).

---

## §B Post-wave on-disk verification rule (AGENTS.md §5.5)

**MANDATE** (per Wf-384-B1 process improvement — Wf-378-A3 FALSE-NEGATIVE lesson learned):

1. **报告 SHIPPED ≠ 文件存在**. A closeout report claiming "SHIPPED" does NOT mean the file is on disk.
   **Both conditions must be satisfied** before claiming SHIPPED.
2. **Every closeout report MUST include `ls -la <file>` evidence** in the closeout doc itself (or a
   `Bash` invocation with `wc -l` + `stat` output) — not just a narrative claim.
3. **The next-wave A3 verify agent MUST `ls -la` the actual files** (not query stale snapshots /
   cached closeout lists / prior-wave aggregations). Verification must be against the **current
   disk state**, not the prior wave's narrative.
4. **If a verify agent reports "MISSING" but the files ARE on disk**, the verify agent's output
   is a FALSE-NEGATIVE — root-cause via Wf-384-A2 pattern (timing race + stale-snapshot +
   cached-narrative propagation).
5. **Trigger retroactive re-author workflow** if: (a) closeout says SHIPPED + disk MISSING, OR
   (b) cross-wave verify says MISSING + closeout says SHIPPED — either direction triggers
   Wf-384-A1 (audit) + Wf-384-A2 (root cause) + Wf-384-A3 (re-author) + Wf-384-B1 (final sync).
6. **Filesystem is the source of truth.** If disk has the file, the file SHIPPED — no matter what
   any agent's report says. Re-author the verify agent, not the source closeout.
   **AND a correction is APPEND-ONLY** — when a published artifact or document is corrected,
   the superseded rendering is **PRESERVED VERBATIM** and the correction is **ADDITIVE +
   DATED**, never a silent edit and never a deletion. *(Clause restored 2026-10-31 by
   Wf-A152 — see the note below.)*

> **⚠️ WHY THIS CLAUSE IS BEING RESTORED (2026-10-31, Wf-A152).** §5.5's item 6 was
> **REPLACED WHOLESALE** on 2026-09-30 (Wf-384-B1) when the filesystem-truth mandate
> took over this slot; the APPEND-ONLY clause it used to carry was dropped in the
> rewrite and never restored. MEASURED 2026-10-31: the repo carries **145**
> citations of `§5.5 rule #6` for the append-only practice (across **~100**
> documents — the `Append-only correction log.` footer in 8+ close-out reports, the
> `HONEST_NEGATIVE_CATALOG.md` Wf-A131..Wf-A151 entries, `AGENTS.md` §2,
> `TODO/README.md`, `TODO/INDEX.md`) and **~18** more for the filesystem-truth
> meaning; **every one of the append-only citations pointed at a rule that did not
> state it**, because the rule the working-tree AGENTS.md carried before the
> 2026-09-30 rewrite was never committed and does not exist anywhere in `git
> history`. The *practice* was correct and is preserved verbatim in every document
> it touched; only its *stated authority* was broken. Both clauses now live in
> item 6, so every citation in the repo resolves. Catalog **§6.264**.

> **⚠️ THE COUNTS ABOVE ARE SUPERSEDED (2026-11-01, Wf-A154).** The **145** /
> **~100 documents** / **~18** figures were true when measured on 2026-10-31 and are
> **not** reproducible under any of four definitions re-tested on 2026-11-01. MEASURED
> 2026-11-01, `rg --binary` over `git ls-files` **plus** untracked-not-ignored
> (**9868** files): the **rule-6-form** citations (`§5.5` within 45 chars of
> `rule #6` / `rule 6` / `rule-6` / `item 6`) number **285** across **70** files
> (tracked-only: **248** / **59** files); the **filesystem-truth-form** citations
> (`§5.5` followed by `filesystem-truth`) number **1017** across **329** files
> (tracked-only: **1008** / **322** files); **2545** citations of `§5.5` in any form
> across **592** files. Excluding `paper/**` and `.data/**` changes nothing — those
> paths carry none of these citations. The count grew because the re-quote wave
> itself added citations: the catalog alone now carries **96** rule-6-form citations
> (`TODO/INDEX.md` **38**, `TODO/README.md` **16**). **A citation count is not a stable
> property of a repo that is actively citing the rule it documents** — which is the
> same lesson §6.263 records about `grep`, applied to a number. The superseded
> rendering above is **PRESERVED VERBATIM** per AGENTS.md §5.5 rule 6. Catalog
> **§6.264** §3.

**Why this rule exists**: Wf-378-A3 cross-poll verify agent reported "Wf-377 0/5 SHIPPED" based
on a stale snapshot query. The FALSE-NEGATIVE propagated downstream to Wf-379..Wf-383 wave
briefs. Wf-384-A1 immediately disproved the FALSE-NEGATIVE: all 5 files ARE on disk. This is
a process gap (cross-wave verify-agent handoff), NOT a delivery gap.

**How to apply**: Every future ultracode wave + paper work + Tier-2 ship surface. Wave authors
MUST include `ls -la` evidence in closeout docs (rule #2); verify agents MUST run `ls -la`
from current disk state (rule #3); the 4-step retroactive re-author workflow (rule #5) is the
canonical recovery pattern.

**Cross-ref:** `TODO/completed/384a1_wf377_on_disk_audit.md` + `TODO/completed/384a2_wf377_root_cause.md`
+ `TODO/completed/384b1_index_sync.md` + `TODO/completed/384_final_synthesis.md` + `TODO/INDEX.md`
§22.75.

---

## §C Atomic write protocol (AGENTS.md §5.6)

**MANDATE** (per Wf-386-B1 — Wf-371/373/377/378/384 recurring "verification reliability + write
overwriting" pattern, deeper-level fix per user voice *"如果是同一个问题一直没修好的话得做调研
+review尝试更深层的修复"*):

This rule sits at the **I/O substrate layer** complementing §B's **agent-prompt layer**.
Belt-and-braces at two levels:

| Layer | Mechanism | Rule |
|---|---|---|
| Agent-prompt | §B | Every closeout MUST include `ls -la` evidence; next-wave A3 verify MUST `ls -la` from current state |
| I/O substrate | §C | Every closeout write MUST use `atomic_write_file`; overwrite-write MUST run `safe_overwrite_check`; agent verification MUST run `detect_truncation` |

**Three rules:**

1. **All future closeout writes MUST use `atomic_write_file`** (NOT direct `Write`). The
   atomic temp-file + rename with per-path process-local lock closes the half-merged file
   state failure mode. Implementation in `molmetal/scripts/wf386a3_atomic_write.py::atomic_write_file`
   (POSIX `os.replace` atomicity; concurrent writers serialise via per-path lock).
2. **All overwrite-write tasks MUST run `safe_overwrite_check` first.** Line-set Jaccard
   distance heuristic (default threshold 0.5) flags substantial content loss. Implementation
   in `molmetal/scripts/wf386a3_atomic_write.py::safe_overwrite_check`. When the threshold
   is exceeded, raise `OverwriteRisk(existing_lines, diff_ratio)` so the caller must
   explicitly opt-in. Wf-384-A3 closeout pattern (claim "RE-AUTHORED" but agent did NOT
   actually re-author) is structurally impossible under this rule.
3. **All agent verification tasks MUST run `detect_truncation` on agent output.** Closing-
   token heuristic + length check catches mid-sentence agent output before the calling
   workflow accepts it. Default `expected_min_length=200`; callers may raise on detection.

**How to apply**: Every future closeout-drafting agent. Per Wf-386-A3 §6.1 adoption sequence:
Wf-387+ should migrate existing closeout-drafting scripts in batches with regression-guard
tests. The protocol is **opt-in** for existing scripts (auto-migrating all in a single sweep
is too risky per Wf-386-A3 §6.2). §C mandate governs FUTURE closeout writes; existing
scripts continue to work as before.

**Cross-ref:** `molmetal/scripts/wf386a3_atomic_write.py` (NEW ~366 LOC) +
`molmetal/tests/test_wf386a3_atomic_write.py` (NEW 27-case test suite) +
`docs/ATOMIC_WRITE_PROTOCOL.md` (NEW ~290 LOC canonical reference with §0 audit + §1
problem statement + §2 atomic write protocol + §3 overwrite detection protocol + §4
truncation detection protocol + §5 agent prompt template + §6 examples + §7 cross-refs +
§8 honest framing) + `TODO/completed/386a1_wf384_a3_overwrite_audit.md` +
`TODO/completed/386a2_wf378_a3_wrong_verdict_root_cause.md` +
`TODO/completed/386a3_atomic_write_protocol.md` + `TODO/completed/386_final_synthesis.md`.

---

## §D Trainer flag wire-in smoke-required rule (AGENTS.md §5.7)

**MANDATE** (per Wf-399-A1 — third recurring "CLI flag declared but not wired" regression
pattern, lesson learned via deeper-level fix per user voice *"之前decode_ratio是1的现在我
去哪退化的这个要重点解决"* — focus on solving the decode_ratio=1 → 0 regression):

Every time an agent wires a NEW flag into a Mol-Metal trainer's training loop — i.e. into
`loss = ...` cascade, `train_one_epoch` body, sampling/decode step, or any other live-
computation site — the wire-in is **INCOMPLETE** until a **50-step end-to-end smoke** has
verified:

1. `parse_args` accepts the flag without `NameError` / argparse error.
2. Trainer imports + initialises cleanly with the flag set (catches SCAFFOLD-module-not-
   imported regression — see Wf-385-A1).
3. `decode_ratio > 0` in the metrics JSON or stdout regex (catches the canonical
   `decode_ratio = 0/192` regression — Wf-291D + Wf-299).
4. `loss[0] > loss[49]` — monotone non-increasing loss over 50 steps (catches silent
   loss-divergence from broken loss-modifier wire-in).

**Three rules:**

1. **Every trainer flag wire-in MUST be smoke-tested end-to-end before SHIPPED.** A flag
   whose SCAFFOLD module has 4/4 unit tests PASS but is NOT imported by the trainer's main
   loop is **FLAG_ONLY**, not SHIPPED (Wf-385-A1: 17/17 Wf-380 NEW CLI flags were FLAG_ONLY).
2. **Smoke MUST use CPU + tiny config** (`--device cpu --epochs 1 --batch-size 8` or
   equivalent) so it completes in <60 s. GPU launches are still USER_OK-gated per §5.1.
3. **`verify_decode_ratio > 0` is the canonical honest-negative floor** for any CFM /
   YuelBond / decoder-class trainer. If `decode_ratio` regresses to 0 on a 50-step smoke,
   the wire-in MUST be reverted or fixed before SHIPPED. The historical decode_ratio=1 → 0
   regression pattern appeared at Wf-291D, Wf-299, Wf-380, Wf-385-A1.

**Smoke template** (CPU-only, <60 s):

```python
from molmetal.scripts.wf399a1_smoke_protocol import run_smoke_50step, verify_all_invariants
smoke = run_smoke_50step(
    trainer_path="molmetal/scripts/train_cfm_yuelbond_wasserstein.py",
    flags_list=["--my-flag", "--epochs", "1", "--batch-size", "8", "--device", "cpu"],
)
verdict = verify_all_invariants(smoke)
assert verdict["pass"], f"WIRE-IN REGRESSION: {verdict}"
```

**Cross-ref:** `molmetal/scripts/wf399a1_smoke_protocol.py` (~340 LOC) +
`molmetal/tests/test_wf399a1_smoke_protocol.py` (~210 LOC, 2 tests PASS) +
`TODO/completed/399a1_regression_rule.md` (Wf-399-A1 closeout) +
`TODO/completed/385a1_wf380_cli_flag_audit.md` (Wf-385-A1 audit) +
`TODO/completed/394a1_lambda_cfm_wire.md` (canonical wire-in reference) +
`TODO/completed/380b1_index_sync.md` §5 (SCAFFOLD-only integration documentation).

---

## §E Workflow (wf) token-budget discipline (AGENTS.md §5.8)

**User voice (verbatim, 2026-10-02)**: *"随便一个workflow动不动就几M的token"*.

**Three rules** (apply to ALL workflows launched from this repo going forward):

| # | Rule | Rationale |
|---:|---|---|
| 1 | **Per-workflow parallel agents ≤ 2** (was 4-6) | Each agent loads AGENTS.md + skills + TODO/README.md → 0.5-1M tokens per agent context |
| 2 | **Per-agent file-read scope ≤ 3 files** (was 10+) | Each Read = 50-200K tokens; reading 20 files = ~2M wasted |
| 3 | **Per-agent output ≤ 1500 tokens** (was 5000+) | Verbose JSON returns burn downstream phase context |

**When to use wf vs direct Edit/Write:**

| Scope | Method | Rationale |
|---|---|---|
| 1-3 file edits, <50 lines, single domain | **Direct Edit/Write** (no wf) | Agent context already loaded |
| Single Python script bug fix | **Direct Edit/Write + Bash verify** | No need for parallel exploration |
| Audit / refactor of >5 files OR cross-domain | **wf with 2 parallel agents** | Parallelism adds value; cap at 2 |
| Multi-step orchestration (>3 phases) | **wf with sequential agents** | Pipeline + barrier if cross-item; cap phase agents at 2 |

**When NOT to use wf:** single Edit/Write/Bash invocation; reading a file once and quoting it;
"verify X exists" → `ls -la` is faster than spawning an agent.

**Cookie budget memory aid:**

| wf scale | tokens | use case |
|---|---|---|
| Single direct Edit | ~10K | Bug fix, 1-3 lines |
| Single-agent wf | ~50-200K | One well-scoped task |
| 2-agent wf | ~300K-1M | Audit or refactor across 2 domains |
| 4-agent wf (LEGACY) | ~1.5-3M | Only for genuinely independent parallel investigations |
| 6-agent wf (LEGACY) | ~3-9M | Only for full domain audit; NOT for routine fixes |

**Migration note** (Wf-432..Wf-435 retrospective 2026-10-02): legacy waves ran at 2-3M tokens
because the agent spawned 4-6 parallel agents per wave. Going forward, ALL wf invocations must
apply the 3 rules above OR document the exception in the wf name (e.g.,
`wf-NNN-large-audit-only`).

**Cross-ref:** `docs/WORKFLOW_DEPENDENCY_GRAPH.md` (Wf-358-C1 + Wf-415-A4 cross-workflow
tracking) + `workflow-authoring` skill §"Scale to what the user asked for" + `party-twenty-spirit`
skill §3.5 honest-framing mandate.

---

## §F Single-venv rule (AGENTS.md §5.9)

**MANDATE** (per user directive *"我们不需要任何wave类型的代码版本管理,git就够了"*): all
launch wrappers in `scripts/` MUST invoke the project venv `/home/hugo/codes/mol-metal/.venv/bin/python`
— NOT bare `/usr/bin/python` or `python3`, NOT the legacy `CadButEaas` venvs (R17-R21 era), and
NOT `kanzi_venv` (R18 era).

**Why:**
1. **Reproducibility**: `.venv/bin/python` is the canonical Mol-Metal venv; carries torch 2.14.0+cu130
   + Blackwell sm_120 + YuelBond decoder + all SCAFFOLD modules. Bare `python` lacks torch and
   triggers the `p3_wf343c4_yuelbond_decoder_env` SKIP.
2. **No environment drift**: cross-wave wrappers historically invoked `CadButEaas` venvs from
   R17-R21; each wave rewrote their venv layout, causing transient divergence in installed
   torch / chemprop / scipy versions. Single-venv rule closes that drift permanently.
3. **Git-only version control**: per user voice *"我们不需要任何wave类型的代码版本管理"*, the
   project is moving from wave-typed code → pure git-tracked code. The single-venv rule is the
   env-layer counterpart: ONE `.venv/` per repo, no per-wave sub-venvs.

**Pattern (canonical):**

```bash
#!/bin/bash
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec "$REPO_ROOT/.venv/bin/python" -m molmetal.scripts.train_metal_lomo_wasserstein \
  --epochs 50 --seeds 0 --no-cuda-graphs --cuda-0 --user-ok
```

**Forbidden patterns:** bare `python`; `CadButEaas` venv; `kanzi_venv`; any per-wave sub-venv.

**Audit surface:** `scripts/_wrapper_audit.py` is the shipped §F gate — its REAL-wrapper marker
set requires the canonical `PY=/home/hugo/codes/mol-metal/.venv/bin/python`.

**Cross-ref:** Wf-421-A2 §0 (YuelBond decoder env SKIP closed via venv) + Wf-427-B1 §22.123
(CLEAN pre-flight 113/0/4/117 via venv).

---

## §G OmegaFold-gap-fill rule (AGENTS.md §5.10)

**MANDATE** (per Wf-442 docs flatten + project hygiene): every Mol-Metal code pattern MUST mirror
the OmegaFold reference repo (https://www.biorxiv.org/content/10.1101/2022.07.21.500999v1; local
clone at `/home/hugo/OmegaFold/`). Six concrete rules:

1. **REFERENCE**: All layout / entry-point / docs / license / CI decisions cite the OmegaFold repo
   as the canonical pattern. When in doubt, `ls /home/hugo/OmegaFold/` first.
2. **FLAT PACKAGE LAYOUT**: collapse nested structures to a single `molmetal/` package per the
   OmegaFold `omegafold/` pattern (51 files, 1 console-script entry point). No nested top-level
   sub-packages, no per-script folders at repo root.
3. **ONE README.md**: ≤ 4 KB per the OmegaFold README.md size discipline. STATUS / QUICKSTART /
   ARCHITECTURE / DESIGN-BOUNDARY must live in separate `docs/*.md` files.
4. **console_scripts ENTRY_POINT**: `molmetal = "molmetal.__main__:cli_main"` in `pyproject.toml`.
   `molmetal/__main__.py` is the single dispatch surface — no per-script entry points.
5. **MINIMAL CI**: `.github/workflows/test.yml` must exist with **1 job + 1 pytest** invocation
   as a smoke gate. Heavy multi-job CI lives in `.github/workflows/ci.yml` separately.
6. **FLAT LICENSE**: Apache-2.0 header per `.py` file (boilerplate at top). No `LICENSE.md` /
   `LICENSE.txt` duplication — the canonical license is at repo root only.

**Audit surface:** `scripts/wf442_omegafold_audit.py` (~50 LOC) scans for: (a) README.md size
≤ 4 KB; (b) `molmetal` console-script entry-point present; (c) `molmetal/__main__.py` exists;
(d) `.github/workflows/test.yml` exists with ≤ 2 jobs; (e) all `.py` files have Apache-2.0
header. Returns non-zero if any rule violated.

**Cross-ref:** Wf-442 Phase 1 A1 docs flatten (QUICKSTART.md + DESIGN_BOUNDARY.md + ARCHITECTURE.md
created) + Wf-442 Phase 1 A2 project hygiene (`scripts/_pre_flight_check.sh` created) + §F
single-venv + §C atomic-write + §B post-wave verification + §D trainer-flag smoke.

---

## Cross-reference summary

| AGENTS.md § | Full text in this file | Status |
|---|---|---|
| §5.1 | (kept inline in AGENTS.md — high-frequency) | n/a |
| §5.2 | (kept inline in AGENTS.md — high-frequency) | n/a |
| §5.3 | (kept inline in AGENTS.md — high-frequency) | n/a |
| §5.4 | §A | Tier-2 + no-wetlab |
| §5.5 | §B | Filesystem-truth |
| §5.6 | §C | Atomic-write |
| §5.7 | §D | Trainer-flag smoke |
| §5.8 | §E | WF token budget |
| §5.9 | §F | Single-venv |
| §5.10 | §G | OmegaFold gap-fill |
| §5.11 | §H (below) | YAML-only launch — also summarized inline in AGENTS.md |

---

---

---

## §H YAML-only launch — no CLI hyperparameters (AGENTS.md §5.11)

**User directive (2026-10-09, verbatim):** *"以后要求能且只能用yaml配置启动训练和推理，
no cli参数"* — training and inference launch from the YAML config. Every
hyperparameter is declared in `configs/*.yaml`; a launch wrapper sets up the
environment, detaches the process, and redirects output, nothing more.

**The rule.** A wrapper may pass only `--config`, `--out-dir`, `--ckpt-dir`,
`--help` — flags that select *which run*, never *what to train*. `--extra-flags`
is **REFUSED** by `scripts/start.sh`, not deprecated.

**Enforcement surfaces:**

| Surface | Role |
|---|---|
| `scripts/_yaml_only_gate.py` | Fails the launch if any `launch_*.sh` smuggles a trainer flag, or if `start.sh` still accepts `--extra-flags` |
| `scripts/_config_binding_gate.py` | Fails if any config key resolves to no dest of the parser named by that config's own `entry.script` |
| `pre_flight_check.py` | `p3_yaml_only_launch_check` + `p3_config_trainer_binding_check`, both launch-pipeline |
| `molmetal/tests/test_wf_yaml_only_gate.py` + `test_wf_config_binding_gate.py` | 35 regression tests over the gates themselves |
| `molmetal/scripts/wfA2_config_loader.py` | Resolves YAML leaves → trainer dests; holds `_NON_TRAINER_PATHS` and `DECLARED_INERT_PATHS` |

**Three dispositions for a key that resolves to no dest.** The middle one is the
whole point of the audit:

1. **resolves to a dest** — live;
2. **`_NON_TRAINER_PATHS` / `_NON_TRAINER_KEYS`** — live, read by something that is
   not the trainer, with the consumer named in the dict;
3. **`DECLARED_INERT_PATHS`** — documented intent with **zero consumers repo-wide**
   (verified by grepping the literal key across every `*.py`/`*.sh`): split-protocol
   prose, G4/G5/G6 sweep structure, and MEASURED baselines recorded as the
   comparison target. Reported as `DECLARED-INERT` on **every** gate run and never
   fatal — a key being inert is a fact a reader needs, and hiding it would recreate
   the blind spot this audit was run to close;
4. **anything else** — a violation; a new unresolvable key fails the launch.

**`entry.command_template` is a third escape route, and it was open.** `--extra-flags`
is refused in `start.sh` and the wrappers are scanned, but a command_template is a
raw shell string inside the YAML. The check distinguishes *placeholder* from
*literal*: `--epochs {epochs}` is yaml-driven (start.sh fills it from the same YAML
via `entry.format(...)`, and `{metal}`/`{seed}` are what make the G5/G6 sweep loops
work at all), while `--device cuda:0` and bare `--use-multi-ph` are literals nothing
reads from the YAML. Verified 2026-10-09: every placeholder used by every config
resolves to a field `start.sh` actually passes, so no template raises `KeyError`.

**Why this rule exists (measured, not stylistic).** Before it, **54
hyperparameters across 6 launch wrappers** were set on command lines rather than
in configs — including all five BLK-C5 levers and every MoleculeNet backbone
hyperparameter. The cost was not untidiness. It was a silently mis-wired model:

`use_fm_objective` was absent from 5 of 7 CFM configs, so the argparse default
(False) applied while `use_egnn_velocity: true` was present. With
`fm_objective=False`, `fm_term` is an exact `torch.zeros(())`
(`train_cfm_yuelbond_wasserstein.py:2710`, gated at `:2715` by
`if fm_objective and batch_clouds:`). The 518,276-parameter `EGNNVelocityField`
was therefore trained by the Wasserstein geometry anchor alone
(`:2355`, `wasserstein_weight=0.1` pulling predicted velocity toward a reference
square-planar Pt geometry) — not by the flow-matching regression onto
`dx = x1 - x0`. Measured gradient mass on a CPU 1-epoch run of
`configs/cfm_blk_c5_repair_2026_10_09.yaml`:

```
head.msg_mlps 2.24e+03 | other 1.14e+03 | vel_field 1.27e+01
```

while `method: rk4` drove an ODESolver (`:3740`) that integrated `vel_field` as
the velocity function. A knob that lives only in a shell script is a knob nobody
audits; §5.11 is the fix for that, not for line length.

**How to apply:**

1. To add a hyperparameter: add it to the YAML, then verify the round-trip —
   `trainer._parse_args(["--config", cfg])` must return the non-default value.
   `wfA2_config_loader`'s leaf-fallback resolves any leaf under `compute`, `data`,
   `eval`, `flow`, `model`, `output`, `protocol`, `train` by matching the leaf name
   against the parser's dests, so most keys need no alias entry.
2. To add a new trainer that must be config-driven: wire the loader into its
   `_parse_args` in the same shape the CFM trainer uses —
   `args = apply_yaml_overrides(p, args, args.config)` after `p.parse_args(argv)`.
   Watch for `required=True` arguments: argparse enforces them *before* the YAML
   is applied, so such a flag is unsatisfiable from a config. `train_chemprop_v3_
   frozen_dmpnn_head.py` had exactly this bug with `--out-dir`.
3. Per AGENTS.md §5.7, a new `--config` wire-in is not shipped until an end-to-end
   CPU smoke proves it — `parse_args` round-trip alone does not catch a trainer
   that dies later on the resolved values.

**Enforcement note.** The gate's own regression suite caught two real holes in the
gate on its first run, both worth recording because either one would have made it
report PASS on a violating wrapper: (a) a per-physical-line scan misses flags on
backslash-continuation lines — which is exactly how every launcher in this repo was
written; (b) `Path.relative_to(REPO_ROOT)` raised `ValueError` on any path outside
the repo. Both are fixed and covered.

**Cross-ref:** AGENTS.md §5.11 (inline summary) + `docs/WORKFLOW_DEPENDENCY_GRAPH.md`.

**⚠️ LIST-valued flags must be YAML lists — both gates miss this.** A flag declared
`nargs="+"` (e.g. `--metals`) arrives from the CLI as a list of tokens. Writing
`metals: Ru Ir Os Re` in a config delivers **one string** instead. The trainer then
searches for a metal literally named `"Ru Ir Os Re"`, matches no rows, and every
fold produces an empty test set → `test_auc = nan` on every seed.

This slipped through **both** §5.11 gates, and cost a full run:

- `scripts/_config_binding_gate.py` checks the key resolves to a dest — it does;
- `p3_wf343c4_hyperparams_config_complete` checks four scalars (`batch_size`, `lr`,
  `hidden_dim`, `n_layers`) — a `metals` list is not one of them;
- the one-variable comparison table printed `metals = 'Ru Ir Os Re'` beside
  `metals = ['Ru Ir Os Rh Re']` and was read as "expected difference", not as a
  **type** difference.

The fix is `metals: [Ru, Ir, Os, Re]`. The rule that generalises: **verify the
round-trip by type, not by value** — `isinstance(args.metals, list)` rather than
`args.metals != default`. §5.7's smoke must assert the shape, not just that the
run exits 0.

**A second-order bug this exposed:** `train_metal_lomo_wasserstein.py` aggregated
per-fold AUCs with `statistics.pstdev(_aucs)` and no finiteness filter, so one
degenerate fold raised `AttributeError('float' object has no attribute
'numerator')` — *at the very end*, after every fold had trained and written its
per-seed JSON. One bad fold destroyed the aggregate for all the others. Fixed
2026-10-09 to drop non-finite AUCs from the mean/std and report
`n_seeds_nonfinite_auc` in the per-metal payload rather than aborting.

**Cross-ref**: `configs/m19_lomo_4metal_sensitivity.yaml` header (records the
first launch's all-NaN output), `molmetal/scripts/train_metal_lomo_wasserstein.py`
aggregate block.

---

## §I Enforcement matrix — which rules are gated, which are convention (2026-10-10)

> The first table above answers *where the full text lives*. This one answers the
> question that actually matters for discipline: **what stops a rule from being
> violated?** A rule with a live gate cannot rot silently; a rule that is only
> prose is one bad week away from being ignored.
>
> Method: every claim below was checked against disk on 2026-10-10, and the gate
> was **run**, not merely grepped (a gate that is present but never invoked is
> the failure mode this table exists to catch).

| § | Rule | Enforcement surface | Live? | Verdict |
|---|---|---|---|---|
| §5.1 | GPU authorization | `pre_flight_check.py` GPU probes + `check_blackwell_env.py` | yes | strong |
| §5.2 | Honest MEASURED/PROJECTED framing | **none** — `HONEST_NEGATIVE_CATALOG.md` + the `party-twenty-spirit` skill only | no | convention |
| §5.3 | No `paper/*` without OK | **none** | no | convention |
| §5.4 | Tier-2, no wet-lab | `pre_flight_check.py :: p3_loi_send_ready` | yes, **semantics lag** — see F4 | needs rework |
| §5.5 | Filesystem-truth (§5.5) | **none** | no | convention |
| §5.6 | Atomic write | `wf386a3_atomic_write.py`, **259 call sites**, `test_wf386a3_atomic_write.py` | yes | strongest |
| §5.7 | 50-step CPU smoke on new flag | `wf399a1_smoke_protocol.py::run_smoke_50step` — **0 callers repo-wide** | **no** | dead — see F2 |
| §5.8 | WF token budget | **none** | no | convention |
| §5.9 | Single venv | `scripts/_wrapper_audit.py` + `test_wrapper_python_path.py` | yes — **currently RED** | see F5 |
| §5.10 | OmegaFold gap-fill | `_add_license_headers.py` (present); **`wf442_omegafold_audit.py` does not exist** | **phantom** | see F6 |
| §5.11 | YAML-only launch | `_yaml_only_gate.py` + `_config_binding_gate.py`, wired at `pre_flight_check.py:13311,13326` | yes — **verified biting today** | strong |

**Summary: 6 gated-and-passing, 1 gated-and-RED, 1 phantom, 4 convention-only.**
The convention-only set (§5.2 §5.3 §5.5
§5.7 §5.8) is where erosion happens first, because nothing fails when they are
skipped. §5.2 and §5.3 are nonetheless the two rules this project has been most
scrupulous about — that adherence is cultural, not mechanical, and should be
described as such rather than as "enforced".

**Verification command.** Every "live" verdict above was obtained by *running*
the gate on 2026-10-10, not by grepping for its presence. Two of them turned out
differently than a grep would have suggested — see F5 and F6. That is the whole
argument for this table: presence of a gate file is not evidence of enforcement.

### F1 — pre-flight is temporally fragile (found 2026-10-10, real)

`pre_flight_check.py` runs `_config_binding_gate.py` (§I table). The gate is
sound: given a config with a key under a non-whitelisted section it emits one
`VIOLATION <config>: '<key>' resolves to no dest` per key and returns FAIL.

The failure mode is **ordering**, not the gate. Launching the
`bench_mcts_search_quality` ON-arm today, pre-flight was run at 01:37 and the
config was authored at 01:39 — so the gate validated a world in which the config
did not exist, returned PASS, and the run then silently used argparse defaults
for four keys. It surfaced only because the run's own telemetry reported
`dmpnn_enabled=false, n_dmpnn_eval=0`.

AGENTS.md §4's STALE flag is a **time** predicate (`now - last_refresh > 24 hr`).
Nothing binds a pre-flight PASS to the **content** it validated.

> **Rule: pre-flight must run AFTER every config it is meant to validate is on
> disk, and must be re-run after any config edit.** A pre-flight run performed
> before authoring the artifact it checks is void, not PASS.


### F2 — §5.7's smoke helper has zero callers (found 2026-10-10)

`grep -rn run_smoke_50step` outside its own module returns **0 hits**. The rule
exists precisely because the project already learned that a scaffold can import
and print `--help` yet crash at compute time; the codified answer is a 50-step
CPU smoke, and nothing calls it. §5.7 is currently prose.

Cheapest real fix: have `pre_flight_check.py` assert, for each trainer whose
flag surface changed, that a smoke artifact newer than the trainer's mtime
exists. Until then, treat §5.7 as advisory and say so.


### F3 — `_LEAF_FALLBACK_SECTIONS` is a section-name allowlist (found 2026-10-10)

`wfA2_config_loader._LEAF_FALLBACK_SECTIONS` is
`{protocol, model, flow, data, compute, eval, output, train}`. A YAML leaf under
**any other section name** is flattened to `dmpnn_dmpnn_value` and resolves to
no dest. Writing a new, perfectly reasonable `dmpnn:` block therefore drops the
entire block, silently.

The loader *does* warn per key (AGENTS.md §H: "reported, not swallowed"), which
is the correct behaviour — but the warning is one line among thousands of RDKit
messages and does not stop the run. **New config section names must be added to
`_LEAF_FALLBACK_SECTIONS` or spelled flat at top level.** Both are recorded in
`configs/mcts_search_dmpnn_value_on.yaml`.


### F4 — §5.4's LOI gate still grades send-readiness (found 2026-10-10)

`p3_loi_send_ready` "verifies the 4 wetlab LOI drafts are ready-to-send". Wf-405-A3
extended its status regex to also accept `TEMPLATE_NOT_NEEDED_FOR_TIER2` and
`DEPRECATED`, so it is **not** an active contradiction with §10, and it sends
nothing. But its PASS condition is still *"the LOIs are structurally ready to
send"* — a criterion AGENTS.md §10 declares deprecated ("DO NOT SEND"). The
prohibition lives in comments, not in the pass condition.

The honest description today: the gate is **permissive**, not mandating. It
should be inverted to assert deprecation, not readiness.

---


### F5 — §5.9's gate was RED; resolved 2026-10-10 by retiring, not rewriting

`scripts/_wrapper_audit.py` exited **rc=1** with **7 violations** — the entire
violation set was the historical single-line-fix wrappers from 2026-10-09
(`launch_wf_fix3lines_*` × 5, `launch_wf_blk_d1/d2_*` × 2).

**Resolution: retire, don't rewrite** (user decision). All seven moved to
`scripts/_retired/` — see that directory's README for the full table and
reasoning. `_wrapper_audit.py::discover_wrappers` is
`sorted(scripts_dir.glob("launch_wf*.sh"))`, **non-recursive**, so they left the
audit's scope by construction rather than via an exemption list.

Before retiring, the one genuinely risky marker was fixed rather than waved away:
five of the seven invoked the interpreter through the **relative** path
`.venv/bin/python`, relying on an earlier `cd` to resolve it. AGENTS.md §5.9
requires the absolute form. All seven now use
`/home/hugo/codes/mol-metal/.venv/bin/python` (two via a canonical `PY=`
assignment). Verified with `bash -n` on each — a real syntax check, not a
`--help`-style probe.

Why retire rather than rewrite: compliant wrappers run 9,187–34,915 B
(median 17,413 B) while these were 2,464–5,176 B, so closing the remaining four
markers (USER_OK gate / DRY_RUN emission / plan.json emission / USER_OK env
propagation) meant rewriting each into the full template — real work for
launchers whose results are already on disk in `metrics/` and `todo.json`. They
were **moved, not deleted**, so the launch provenance for those numbers survives.

Gates after the change:

```
_wrapper_audit.py       rc=0   25 wrappers scanned, all REAL
_yaml_only_gate.py      rc=0   PASS (40 wrappers)
_config_binding_gate.py rc=0   PASS
```

`launch_wf_fix3lines_cfm_c5_repair.sh` was **not** moved — it is compliant and is
cited by live code at `molmetal/scripts/train_cfm_yuelbond_wasserstein.py:5849`.

### F6 — §5.10 names a gate that does not exist (measured 2026-10-10)

AGENTS.md §5.10 states the enforcement surface is
`scripts/wf442_omegafold_audit.py`. That file is **absent from disk and from git
history** (`git log --diff-filter=D` returns nothing). The only OmegaFold-related
module on disk is `molmetal/scripts/wf443a3_omegafold_integration_helper.py` —
a different wave number and a different role.

`_add_license_headers.py` (§5.10's other named surface) does exist. So §5.10 is
partially enforced: the license-header half is real, the layout-audit half is
cited but absent. Either restore the gate or correct AGENTS.md §5.10 — leaving a
phantom path in the agent entry point is worse than having no gate, because an
agent will trust it.

---

### F7 — this file is not in git (measured 2026-10-10)

`docs/STANDING_DIRECTIVES.md` is **untracked** (`git status --porcelain` → `??`).
AGENTS.md §5 opens with "Full mandate text + cross-refs for §5.4-§5.10 live in
`docs/STANDING_DIRECTIVES.md`", and §5.4 / §5.5 / §5.7 / §5.9 / §5.10 each carry
a pointer to their §A-§G section here. In a fresh clone every one of those
pointers is a dead link, and the six directives revert to their two-line inline
headlines in AGENTS.md.

Same status for `docs/GENERATION_OBJECTIVE_METRIC_DESIGN.md` (referenced by
todo.json BLK-C15). `docs/00_MODEL_NARRATIVE.md` is new work, untracked as
expected.

**The same gap is wider than this file.** `scripts/_wrapper_audit.py` audits 32
launchers; **6 are untracked**, and 5 of those 6 are in §F5's violation set:

```
launch_wf_blk_d1_attentivefp_5seed.sh
launch_wf_blk_d2_chemprop_v3_frozen_5seed.sh
launch_wf_fix3lines_cfm_100ep.sh
launch_wf_fix3lines_cfm_100ep_qed.sh
launch_wf_fix3lines_cfm_30ep_onecycle.sh
launch_wf_fix3lines_cfm_c5_repair.sh
```

So the pattern is consistent: work gets done, results land in `metrics/` and
`todo.json`, and the **launcher and directive documents that produced it stay
outside git**. Nothing in the review flow notices, because every SSOT that an
agent actually reads (`todo.json`, `metrics/`) *is* tracked.

Fix: `git add docs/STANDING_DIRECTIVES.md docs/GENERATION_OBJECTIVE_METRIC_DESIGN.md
scripts/launch_wf_blk_d1_attentivefp_5seed.sh scripts/launch_wf_blk_d2_chemprop_v3_frozen_5seed.sh
scripts/launch_wf_fix3lines_cfm_100ep.sh scripts/launch_wf_fix3lines_cfm_100ep_qed.sh
scripts/launch_wf_fix3lines_cfm_30ep_onecycle.sh scripts/launch_wf_fix3lines_cfm_c5_repair.sh`.
Not done here because committing is the user's call, not the audit's.
