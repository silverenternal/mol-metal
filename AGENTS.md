# AGENTS.md — Mol-Metal Agent Entry Point

> **Wave-versioning note (2026-10-01 user directive):** All version history lives in `git log`. No per-wave file naming convention is in use going forward.

**Last updated:** 2026-10-02 (**Wf-432 6-问题 ultracode audit + Wf-433 32-NEW-gates wire-in + Wf-434 BLOCKER 3 fix + Wf-435 pre-flight FAIL closure SHIPPED**; user directive 2026-10-02 verbatim *"我觉得可能是现在的设计有问题，你可以参考一下我们之前clone的那些项目看看别人是怎么修的"* → **§5.8 wf token budget discipline** added per user voice; previous wave history preserved below).

This file is the **agent-facing entry point** for the Mol-Metal project. Claude Code / Claude Agent
SDK sessions working on this repo should read this first. It is the agent counterpart to
[`README.md`](README.md) (project TOC) and [`TODO/README.md`](TODO/README.md) (status TOC).

| Doc | Audience | Purpose |
|---|---|---|
| **AGENTS.md** (this) | **AI agent** | Project rules + skill catalog + critical workflows + standing directives |
| [`README.md`](README.md) | Human + agent | Top-level project status (Sep 28 Wf-265 snapshot) |
| [`TODO/README.md`](TODO/README.md) | Human + agent | TODO directory navigation + recent R/Wf timeline |
| [`TODO/INDEX.md`](TODO/INDEX.md) | Human + agent | **Single source of truth** for every file's status (1449 lines, Sep 28) |

---

## §1 Five standing project rules

These rules apply to **every** agent action in this repo. Violations are honest-negatives.

1. **MEASURED-vs-PROJECTED tag discipline** (`party-twenty-spirit` §3.5). Every numeric value in
   any agent output MUST trace to a JSON / report path on disk. PROJECTED vs MEASURED tag is
   required. Zero tolerance for fabricated / aspirational numbers in main results.
2. **Standing GPU authorization (2026-09-29 user grant)**. User has granted GENERAL GPU
   authorization going forward — **"显存够就放心大胆用"** (use freely as long as VRAM is enough).
   No per-launch user OK required. Constraints:
   - **VRAM budget check first**: confirm `nvidia-smi` shows sufficient free VRAM (e.g. RTX
     PRO 6000 Blackwell 98GB or RTX 5090 32GB); abort if OOM risk.
   - **Hyperparameters via config files**: reference peer papers in the same domain; encode
     in `--batch-size`, `--lr`, `--hidden`, `--depth` config rather than guessing.
   - **Pre-flight PASS**: still required (`pre_flight_check.py`) before any retrain / panel /
     ablation.
   - **Dry-run defaults**: `--dry-run` is still the script default; agents should run the
     actual launch path but always confirm GPU OK + pre-flight PASS first.
3. **No paper work without authorization** (TODO-84 active). Zero touches to `paper/*` unless
   user explicitly OKs. Any reference to paper §1-§7 is READ-only via `mol-metal-paper-writing`.
4. **Pre-flight v2 25/25 PASS gate** (TODO-207). Any agent launching a retrain / panel /
   ablation MUST run `pre_flight_check.py` first. STALE flag (24 hr) blocks PASS.
5. **Honest-negative framing** (`party-twenty-spirit` §3.5). When a metric is BELOW field-avg
   or BLOCKED on external dependency, declare it as honest-negative — never paraphrase as
   "achievable" / "promising" without PROJECTED tag + wall budget.

---

## §2 Single source of truth pointers

| Question | Read |
|---|---|
| "What's every file's status?" | [`TODO/INDEX.md`](TODO/INDEX.md) — 1449 lines, updated by hand on every Wf ship |
| "What's the recent Wf timeline?" | [`TODO/INDEX.md`](TODO/INDEX.md) §0 + §22 |
| "What's the project status today?" | [`README.md`](README.md) §0 (snapshot table) |
| "What was the 2026-09-15 R15 ship?" | [`TODO/README.md`](TODO/README.md) §R15 ship summary |
| "What was the 2026-09-28 Wf-256..268 ship wave?" | [`TODO/INDEX.md`](TODO/INDEX.md) §0.1 + §22 |
| "Where are measured numbers?" | [`metrics/by_metric/*.json`](metrics/) (113 files) |
| "What are the architectural decisions?" | [`TODO/decisions.md`](TODO/decisions.md) |
| "What are the open blockers?" | [`TODO/risks.md`](TODO/risks.md) |
| "What's the env / GPU setup?" | [`TODO/environment.md`](TODO/environment.md) |
| "What's the engineering conventions?" | [`TODO/engineering_practices.md`](TODO/engineering_practices.md) |

---

## §3 Skill catalog (29 skills in `.claude/skills/` — 20 baseline + 8 NEW per Wf-321B + 1 NEW per Wf-351-B3; all re-validated per Wf-329-W8 + Wf-351-B3)

Skills are loaded via the `Skill` tool. The first 5 are mandatory / hot-path; the rest activate
on keyword match. All 28 are project-owned (not from a plugin registry). Wf-321B ship surface
(R20+R21 skill ecosystem audit, TODO-99): 4 NEW R20 stubs (variance-puct-default + hilbert-astsr-e2e
+ closure-theorem + multiph-pll) + 4 NEW R21 skills (score-rep-variance-mcts + mcts-replay-learner
+ closure-dblock-bench [aliased to catherine-dblock-bench] + triangle-closure) + 2 NEW meta-packs
(r20-skill-pack + r21-skill-pack). All 8 NEW Wf-321B stubs were re-verified per Wf-329-W8
(`TODO/completed/329w8_skill_wiring_verified.md`); catherine-dblock-bench was additionally wired
per Wf-327-F10 (108 LOC, satisfies `closure-dblock-bench` R21-P5 alias per Wf-321B §1.3);
closure-catherine was wired per Wf-329-W8 (227 LOC, 8 sections, YAML frontmatter added).

### §3.1 Mandatory / hot-path (always invoke first)

| Skill | When to invoke | File |
|---|---|---|
| **`party-twenty-spirit`** | Before ANY TODO update, R-switch, paper section completion, major decision, BLOCK / honest-negative close-out. **Mandatory pre-flight for TODO work.** | `.claude/skills/party-twenty-spirit/SKILL.md` |
| **`mol-metal-gpu-execution`** | Before ANY GPU launch > 1 hr. Covers kill_then_adapt directive + Blackwell sm_120 + RTX 5090 env probe + multi-GPU scheduling. | `.claude/skills/mol-metal-gpu-execution/SKILL.md` |
| **`mol-metal-cfm-training`** | When the user mentions CFM / Flow Matching / Lipman / EGNN / YuelBond / bond head / decode_ratio / decode smoke / AMP / BF16 / ODE solver / `molmetal/adapters/flow_matching_lipman/` / `train_cfm_yuelbond_wasserstein.py`. | `.claude/skills/mol-metal-cfm-training/SKILL.md` |

### §3.2 Domain / surface skills (activate by keyword)

| Skill | Domain | Activate on |
|---|---|---|
| `mol-metal-anticancer` | Pt/Ru/Ir/Au/Pd cytotoxicity, A2780/MCF7/NCI-60, MetalCytoToxDB (Krasnov), IC50/pIC50/GI50 | "anticancer", "cytotoxicity", "IC50", "MetalCytoToxDB" |
| `mol-metal-paper-writing` | paper §1-§7 + appendix writing (READ-only on `paper/*`) | "paper §X", "draft paper", "appendix" |
| `mol-metal-lambda-calculus` | Molecular Lambda Calculus (MLC), β-NF formalism, Lambda MCTS drug design | "lambda calculus", "MLC", "β-NF", "Lambda MCTS" |
| `mol-metal-click-chemistry` | 5 click reactions (CuAAC / SPAAC / ThiolEne / Suzuki / AmideCoupling), MetalLigandExchange, AquaExchange | "click chemistry", "CuAAC", "SPAAC", "Suzuki coupling" |
| `mol-metal-mechanism-mapping` | Mechanism ↔ metric mapping (debugging regressed metrics, designing ablations) | "regressed metric", "which mechanism", "ablation design" |
| `mol-metal-closure-theorem` | Closure theorem main result + Catherine 6/6 properties + cross-domain OOD on PlatinAI | "closure theorem", "Catherine properties", "P1-P6" |
| `mol-metal-triangle-closure` | ASTSR ↔ Hilbert ↔ MCTS triangle closure end-to-end on PlatinAI 4-cell + MetalCytoToxDB 5-metal | "triangle closure", "ASTSR + Hilbert + MCTS" |
| `mol-metal-wetlab-protocol` | Wet-lab IC50 validation, MTT / SRB / CellTiter-Glo cell-viability assays | "wet lab", "MTT assay", "cell viability protocol" |
| `mol-metal-wetlab-bayesian` | Bayesian prioritization for wet-lab compounds | "Bayesian wetlab", "compound prioritization" |
| `mol-metal-sci-paper-submission` | SCI paper submission workflow — end-to-end build → validate → submit → archive using audited open-source tools (sn-jnl + tectonic + Zotero + matplotlib/scienceplots). READ-only on `paper/*` per TODO-84 + §5.3. | "submit paper", "SCI submission", "投稿", "cover letter", "ScholarOne", "Editorial Manager", "Mol. Divers.", "JCAMD", "J. Mol. Graph. Model.", "build PDF", "tectonic", "Zotero export", "Better BibTeX", "CSL style" |

### §3.3 R20 plan skills (forward-looking, GPU-blocked or §5.1 standing GPU auth)

| Skill | R20 phase | GPU budget |
|---|---|---|
| `mol-metal-r18-math` | R18 foundation | n/a (math integration) |
| `mol-metal-catherine-dblock-bench` | R20-P5 (TODO-96) | TBD (**WIRED per Wf-327-F10** — 108 LOC; also satisfies `closure-dblock-bench` R21-P5 alias per Wf-321B §1.3) |
| `mol-metal-closure-catherine` | R20-P5 (TODO-96) | TBD (**WIRED per Wf-329-W8** — 227 LOC, 8 sections, YAML frontmatter added) |
| `mol-metal-hierarchical-multimetal` | R20-P4 (TODO-95) | TBD |
| `mol-metal-3way-multimetal` | R20-P4 (TODO-95) | TBD (NEW per Wf-321B) |
| `mol-metal-hilbert-astsr-e2e` | R20-P2 (TODO-93) | TBD (NEW per Wf-321B; **WIRED per Wf-327-F9** — 526 LOC) |
| `mol-metal-hilbert-astsr-r2` | R20-P2 (TODO-93) | TBD |
| `mol-metal-mcts-replay-learner` | R20-P2 (TODO-93) | TBD (NEW per Wf-321B; **WIRED per Wf-327-F10** — 483 LOC, also R21-P2 cross-ref) |
| `mol-metal-multiph-pll` | R20-P3 (TODO-94) | TBD (NEW per Wf-321B; **WIRED per Wf-327-F9** — 518 LOC) |
| `mol-metal-multi-ph-pll-feature` | R20-P3 (TODO-94) | TBD |
| `mol-metal-neural-ode-kinetics` | R20-P4 (TODO-95) | TBD |
| `mol-metal-pareto-multiobj` | R20-P4 (TODO-95) | TBD |
| `mol-metal-variance-puct-default` | R20-P1 (TODO-92) | CPU-only first (NEW per Wf-321B; **WIRED per Wf-327-F9** — 114 LOC) |
| `mol-metal-closure-theorem` | R20-P5 (TODO-96) | TBD (NEW per Wf-321B; **WIRED per Wf-327-F9** — 615 LOC) |
| `mol-metal-score-rep-variance-mcts` | R21-P1 (TODO-98) | TBD (NEW per Wf-321B R21 expansion; **WIRED STUB per Wf-327-F10** — 104 LOC) |
| `mol-metal-triangle-closure` | R21 (TODO-93 cross-ref L8) | TBD (NEW per Wf-321B R21 expansion; **WIRED per Wf-327-F10** — 574 LOC) |
| `mol-metal-r20-skill-pack` | R20 summary pack | n/a (meta-pack aggregating 4 R20 + 12 R19 → 20-skill R20 ship surface) (NEW per Wf-321B) |
| `mol-metal-r21-skill-pack` | R21 summary pack | n/a (meta-pack aggregating 4 R21 + 16 R17-R20 → 20-skill R21 ship surface) (NEW per Wf-321B; **WIRED per Wf-327-F10**) |

### §3.4 How skills compose with workflows

A workflow (`/tmp/wfNNN_*.js`) is the orchestration layer for parallel + multi-stage work. Skills
provide domain context. The pattern:

1. Workflow script inlines `agent(prompt, {label, phase})` calls — each agent inherits the
   project CLAUDE.md / AGENTS.md + the relevant skill's instructions.
2. For a specific sub-task (e.g. "implement Wf-263A v3-fallback mode"), the workflow's prompt
   includes "use Bash + Edit + Read" and "no StructuredOutput" (per Wf-302C text-only brief
   template to sidestep the Wf-291C pipeline regression).
3. Synthesis agents `agent(prompt, {label, phase})` aggregate upstream results — READ only.

Reference: `workflow-authoring` skill (system-bundled) for workflow scripting conventions.

---

## §4 Critical workflows (always run before retrain / panel / ablation)

| Script | Purpose | When |
|---|---|---|
| `python molmetal/scripts/pre_flight_check.py` | Pre-flight v2 25/25 PASS gate (TODO-207) | Before any retrain / panel / ablation launch |
| `python molmetal/scripts/per_wave_checklist.py` | Per-wave sub-checklist emission (Wf-262 §10.1) | Between waves in a multi-wave launch |
| `python molmetal/scripts/wave_budget_tracker.py` | Wave budget + `--no-enforce-budget` + `--cfm-yuelbond-required` (Wf-262/291D) | Before any wave (gate: PLANNED only) |
| `python molmetal/scripts/wf343c2_per_wave_check.py --skip-live` | Wf-343C2 per-wave runtime gate (7 checks: preflight + hyperparams + yuelbond + wave_budget + VRAM + disk + freshness) (Wf-347-A1/A2/B1/B2) | Before each (metal, seed) pair launch in G5 + G6 sweep; standalone gate |
| `python molmetal/scripts/wf343c3_data_quality_check.py --strict` | Wf-343C3 data quality gate (5 loaders × 5 checks: csv_loads + per_metal_balance + scaffold_split + class_balance + label_range) (Wf-347-A1/A2/B1/B2) | Before any retrain; standalone gate |
| `python molmetal/scripts/abort_recovery.py` | Abort recovery (TODO-207 §7.3) | After any unexpected abort / timeout |
| `bash scripts/launch_wf124_p3_g5.sh --dry-run` | TODO-181 G5 dry-run (5 hr GPU) | After pre-flight PASS + user OK |
| `bash scripts/launch_wf124_p3_g6.sh --dry-run` | TODO-182 G6 dry-run (7.5 hr GPU) | After pre-flight PASS + user OK |
| `bash scripts/launch_wf299_cfm_main.sh` | Wf-299 3-wave CFM launch (P1.4 + G6 + G6 full) | After pre-flight PASS + user OK |
| `bash scripts/gpu_pipeline_master.sh wf343c4 [--strict]` | Wf-343C4 opt-in gate runner (Wf-347-A2): invokes wf343c2 + wf343c3 + cross-references 4 NEW Wf-343C4 gates; gated behind `WF343C4_GATE=1` env var (default off) | Before any retrain / panel; opt-in surface for kill_then_adapt §0 diagnostics |

**Stdout invariant:** every script emits either `PASS` / `FAIL` / `SKIP` (with `--strict` for
SKIP-as-fail). Exit codes: 0=PASS, 1=FAIL, 2=SKIP.

**Stale flag rule:** pre-flight auto-marks the report STALE when `now - last_refresh > 24 hr`.
STALE ⇒ auto-verify refuses PASS until refreshed.

---

## §5 Standing directives (2026-09-29 snapshot)

### §5.1 Standing GPU authorization (2026-09-29 user grant)

User has granted GENERAL GPU authorization — **"显存够就放心大胆用"** (use freely as long as
VRAM is enough). No per-launch user OK required. Constraints:

- **VRAM budget check first**: `nvidia-smi` to confirm free VRAM. OOM = abort.
- **Hyperparameters via config files**: reference peer papers in the same domain.
- **Pre-flight PASS** still required (`pre_flight_check.py`).
- **Dry-run defaults**: still default for scripts; agents promote to actual launch after
  VRAM + pre-flight check.
- **Wf-343C2 per-wave runtime gate + Wf-343C3 data quality gate + USER_OK gate** (per
  Wf-347-A1/A2/B1/B2 ultracode integration wave 2026-09-29): the canonical pre-launch
  gating surface for any retrain / panel / ablation. Both gate scripts are also
  wired into `pre_flight_check.py::build_gates("all")` as subprocess-style gates
  (`p3_wf343c2_per_wave_runtime_check` + `p3_wf343c3_data_quality_check`) and into
  `scripts/gpu_pipeline_master.sh wf343c4 [--strict]` (opt-in via `WF343C4_GATE=1`).
  Per-tuple invocation is wired into G5 + G6 launchers
  (`scripts/launch_wf124_p3_g5.sh` + `scripts/launch_wf124_p3_g6.sh` +
  `scripts/recover_g6_failed_tuples.sh`) — see §4 for the full script list.
  Standalone invocation examples:
  - `python molmetal/scripts/wf343c2_per_wave_check.py --wave <wave> --pairs-metals Pt --pairs-seeds 0 --skip-live`
  - `python molmetal/scripts/wf343c3_data_quality_check.py --strict`
  USER_OK gating on Wf-335 / Wf-336 / Wf-337-G3 launch wrappers follows
  `kill_then_adapt` §0 (TODO-84 still applies for paper/* per §5.3).

### §5.4 Tier-2 journal target + NO wet-lab planning (2026-09-29 user directive + 2026-09-30 reinforcement)

**⚠️ HARD MANDATE — DO NOT PLAN, EXECUTE, OR SEND ANY WET-LAB WORK. ⚠️**

**Verbatim user directives across multiple sessions:**
- **2026-09-29**: "我们发二区就可以了，不用规划湿实验" — Tier-2 sufficient; do NOT plan wet-lab
- **2026-09-29**: "我们发二区就可以了，不用规划湿实验" (restated multiple times)
- **2026-09-30**: "还有就是搞清楚我们不用做湿实验，这一点我已经明确了好多遍了" — explicit frustration; reinforced; make this visible in TODO + AGENTS

User said: "我们发二区就可以了，不用规划湿实验" — Tier-2 journal is sufficient; do NOT
plan wet-lab experiments. This re-orients the project:

- **Paper target: Tier-2** (downgrade from JCIM Tier-1). Examples: J. Comput. Aided Mol.
  Des. (JCAMD), J. Mol. Graph. Model., Comput. Chem. (WIREs), Mol. Divers., Comput. Biol.
  Chem. **No wet-lab data required for Tier-2 acceptance** — pure computational submission
  is in scope.
- **Wet-lab items DEPRECATED for project scope**:
  - 4 LOI drafts (Rusanov / PKU / 协和 / ORNL) → keep templates in `TODO/pending/` as
    reference only; do NOT send (per TODO-84 anyway; per Tier-2 strategy nullified).
  - 3 wetlab SOPs (cytotoxicity / cellular uptake / binding affinity) → keep templates
    in `TODO/pending/` as reference; do NOT execute.
  - TODO-138 paper v2 → Tier-2 journal target (re-scope from JCIM).
  - R20-P6 wet-lab IC50 validation → DEPRECATED; re-scope to in-silico metric lift.
  - Wf-74 v2 wet-lab outreach → DEPRECATED.
  - LAMBDA_ROADMAP §3.2 R22-P1 (m18 wet-lab) → re-scope as computational-only.
  - LAMBDA_ROADMAP §3.3 R23 (Phase I clinical) → DEPRECATED.

**Why:** Tier-2 acceptance bar is computational-only; wet-lab validation is 6-12 month
external-dependency that does not gate Tier-2 acceptance.

**How to apply:**
- Every `metrics/by_metric/*.json` and `TODO/completed/*.md` is Tier-2-shippable as is.
- Agent Wf-316 TODO directory audit should reclassify wet-lab items as
  `TEMPLATE_NOT_NEEDED_FOR_TIER2` rather than `STILL_PENDING`.
- Plan files (WEAK_TO_STRONG / IMPROVEMENT_ENUM / LAMBDA_ROADMAP) should re-scope
  wet-lab items to in-silico alternatives.
- TODO-84 still applies for `paper/*` touches — paper work is user-authorization gated,
  not auto-shippable.

Queued GPU launches (2026-09-29):

| Launch | Wall | Status |
|---|---:|---|
| TODO-181 G5 (50-epoch full sweep) | ~5 hr | in flight (Ru seed 2 + Ir seed 0 running on GPU 0) |
| TODO-182 G6 (per-innovation ablation) | ~7.5 hr | dry-run READY; launch wrapper ready |
| TODO-119 P1.4 (chemprop v3 LOMO) | TBD | pre-flight 25/25 PASS; launch when ready |
| Wf-299 CFM 3-wave | ~9.5 hr sequential | dry-run READY; launch wrapper ready |
| Wf-264 CFM property panel | TBD | launch script not yet written |
| §9.Z 5 CFM forward-looking TODOs | 37.8 hr total | per Wf-258 §2.1 |

### §5.2 Honest framing mandate

Per `party-twenty-spirit` §3.5. **Every numeric value MUST trace to a JSON / report path on
disk**. Projected vs MEASURED tag is required. Zero tolerance for fabricated / aspirational
numbers in main results. The §9 honest framing catalog in `TODO/completed/207_unified_pre_flight_checklist_v2.md`
has 9 entries (per Wf-263B expansion); agents MUST respect these when generating claims.

### §5.3 No paper work without authorization

TODO-84 is active. Zero touches to `paper/*` unless user explicitly OKs. `mol-metal-paper-writing`
skill is READ-only on paper sections.

### §5.5 Post-wave on-disk verification rule (NEW 2026-09-30 per Wf-384-B1)

**MANDATE** (added per Wf-384-B1 process improvement proposal — Wf-378-A3 FALSE-NEGATIVE lesson learned):

1. **报告 SHIPPED ≠ 文件存在**. A closeout report claiming "SHIPPED" does NOT mean the file is on disk.
   **Both conditions must be satisfied** before claiming SHIPPED.
2. **Every closeout report MUST include `ls -la <file>` evidence** in the closeout doc itself (or a
   `Bash` invocation with `wc -l` + `stat` output) — not just a narrative claim.
3. **The next-wave A3 verify agent MUST `ls -la` the actual files** (not query stale snapshots /
   cached closeout lists / prior-wave aggregations). Verification must be against the **current
   disk state**, not the prior wave's narrative.
4. **If a verify agent reports "MISSING" but the files ARE on disk**, the verify agent's output
   is a FALSE-NEGATIVE — root-cause via Wf-384-A2 pattern (timing race condition + stale-snapshot
   query + cached-narrative propagation).
5. **Trigger retroactive re-author workflow** if: (a) closeout report says SHIPPED + disk shows
   MISSING, OR (b) cross-wave verify agent says MISSING + closeout report says SHIPPED — either
   direction triggers Wf-384-A1 (audit) + Wf-384-A2 (root cause) + Wf-384-A3 (re-author) +
   Wf-384-B1 (final sync) sequence.
6. **Filesystem is the source of truth.** If disk has the file, the file SHIPPED — no matter what
   any agent's report says. Re-author the verify agent, not the source closeout.

**Why this rule exists**: Wf-378-A3 cross-poll verify agent reported "Wf-377 0/5 SHIPPED" based on a
stale snapshot query. The FALSE-NEGATIVE propagated downstream to Wf-379..Wf-383 wave briefs, each
assuming Wf-377 was missing. Wf-384 was spawned as a "ultracode follow-up" to fix the supposed
missing files. Wf-384-A1 immediately disproved the FALSE-NEGATIVE: all 5 files ARE on disk. This
is a process gap (cross-wave verify-agent handoff), NOT a delivery gap (the Wf-377 deliverables
are correct, complete, and on disk).

**How to apply**: Every future ultracode wave + paper work + Tier-2 ship surface. The rule applies
to ALL agents, NOT just verify agents. Wave authors MUST include `ls -la` evidence in closeout docs
(rule #2); verify agents MUST run `ls -la` from current disk state (rule #3); the 4-step retroactive
re-author workflow (rule #5) is the canonical recovery pattern when either direction's gap surfaces.

**Cross-ref**: `TODO/completed/384a1_wf377_on_disk_audit.md` (the Wf-384-A1 audit that surfaced the
gap) + `TODO/completed/384a2_wf377_root_cause.md` (the Wf-384-A2 root cause investigation) +
`TODO/completed/384b1_index_sync.md` (the Wf-384-B1 docs-only sync + process improvement proposal)
+ `TODO/completed/384_final_synthesis.md` (the Wf-384-B1 comprehensive final synthesis v26) +
`TODO/INDEX.md` §22.75 (the INDEX entry for this wave).

### §5.6 Atomic write protocol adoption rule (NEW 2026-09-30 per Wf-386-B1)

**MANDATE** (added per Wf-386-B1 process improvement proposal — Wf-371 / Wf-373 / Wf-377
/ Wf-378 / Wf-384 recurring "verification reliability + write overwriting" pattern lesson
learned via deeper-level fix per user directive 2026-09-30 verbatim *"如果是同一个问题一直
没修好的话得做调研+review尝试更深层的修复"*):

This §5.6 sits at the **I/O substrate layer** complementing §5.5's **agent-prompt layer**.
The protocol provides belt-and-braces defence at TWO levels:

| Layer | Mechanism | Rule |
|---|---|---|
| Agent-prompt | Wf-384 §5.5 | Every closeout MUST include `ls -la` evidence; next-wave A3 verify MUST `ls -la` from current state |
| I/O substrate | Wf-386 §5.6 NEW | Every closeout write MUST use `atomic_write_file`; overwrite-write MUST run `safe_overwrite_check`; agent verification MUST run `detect_truncation` |

**Three rules**:

1. **All future closeout writes MUST use `atomic_write_file`** (NOT direct `Write`). The
   atomic temp-file + rename with per-path process-local lock closes the half-merged file
   state failure mode. Implementation in
   `molmetal/scripts/wf386a3_atomic_write.py::atomic_write_file` (POSIX `os.replace`
   atomicity guaranteed on Linux; concurrent writers serialise via per-path lock).

2. **All overwrite-write tasks MUST run `safe_overwrite_check` first.** The line-set
   Jaccard distance heuristic (default threshold 0.5) flags substantial content loss when
   re-authoring an existing closeout. Implementation in
   `molmetal/scripts/wf386a3_atomic_write.py::safe_overwrite_check`. When the threshold
   is exceeded, raise `OverwriteRisk(existing_lines, diff_ratio)` so the caller must
   explicitly opt-in to the overwrite. The Wf-384-A3 closeout pattern (claim "RE-AUTHORED"
   but agent did NOT actually re-author) is structurally impossible under this rule.

3. **All agent verification tasks MUST run `detect_truncation` on agent output.** The
   closing-token heuristic + length check catches mid-sentence agent output before the
   calling workflow accepts it as a complete closeout. Implementation in
   `molmetal/scripts/wf386a3_atomic_write.py::detect_truncation`. Default
   `expected_min_length=200`; callers may raise on detection.

**Why this rule exists**: The §5.5 post-wave verification rule governs agent prompts but
does NOT prevent agents from writing files directly via the `Write` tool or via direct
filesystem manipulation. The Wf-386-A1 audit confirmed that Wf-384-A3 ran the correct
commands and produced correct output (no re-authoring actually happened) but Wf-384-B1
narrative still over-reported "RE-AUTHORED" — the deeper-level I/O fix ensures that
*when* an agent attempts an overwrite, the diff threshold raises a flag. Combined with
§5.5, the two layers prevent both: (a) report SHIPPED without file existence (§5.5) AND
(b) file overwrite without report (§5.6).

**How to apply**: Every future closeout-drafting agent. Per Wf-386-A3 §6.1 adoption
sequence: Wf-387+ should migrate existing closeout-drafting scripts in batches with
regression-guard tests. The protocol is **opt-in** for existing scripts (auto-migrating
all existing scripts in a single sweep is too risky per Wf-386-A3 §6.2). The §5.6
mandate governs FUTURE closeout writes; existing scripts continue to work as before.

**Cross-ref**: `molmetal/scripts/wf386a3_atomic_write.py` (NEW ~366 LOC protocol
implementation) + `molmetal/tests/test_wf386a3_atomic_write.py` (NEW 27-case test suite
covering atomic_write_file creates/concurrent + safe_overwrite_check warn/silent +
detect_truncation flag/pass + write_closeout integration; 27/27 PASS in <1 s wall
CPU-only) + `docs/ATOMIC_WRITE_PROTOCOL.md` (NEW ~290 LOC canonical reference doc with
§0 audit + §1 problem statement + §2 atomic write protocol + §3 overwrite detection
protocol + §4 truncation detection protocol + §5 agent prompt template + §6 examples +
§7 cross-refs + §8 honest framing) + `TODO/completed/386a1_wf384_a3_overwrite_audit.md`
(Wf-386-A1 closeout that surfaced the narrative over-reporting) +
`TODO/completed/386a2_wf378_a3_wrong_verdict_root_cause.md` (Wf-386-A2 closeout that
identified the timing race condition) + `TODO/completed/386a3_atomic_write_protocol.md`
(Wf-386-A3 closeout) + `TODO/completed/386b1_index_sync.md` (Wf-386-B1 docs-only sync) +
`TODO/completed/386_final_synthesis.md` (Wf-386-B1 comprehensive final synthesis) +
`TODO/INDEX.md` §22.79 (the INDEX entry for this wave) +
`TODO/completed/302d_retry_with_correction.py` (sibling defence for StructuredOutput).

### §5.7 Trainer flag wire-in smoke-required rule (NEW 2026-09-30 per Wf-399-A1)

**MANDATE** (added per Wf-399-A1 — the third recurring "CLI flag declared but
not wired" regression pattern, lesson learned via deeper-level fix per user
directive 2026-09-30 verbatim *"之前decode_ratio是1的现在我去哪退化的这个要重点
解决"* — focus on solving the decode_ratio=1 → 0 regression that has recurred
across 3+ waves):

Every time an agent wires a NEW flag (`--my-flag`, `--use-my-feature`,
`--no-my-feature`, etc.) into a Mol-Metal trainer's training loop — that is,
into the `loss = ...` cascade, the `train_one_epoch` body, the sampling /
decode step, or any other live-computation site — the wire-in is **INCOMPLETE**
until a **50-step end-to-end smoke** has been run that verifies:

1. **`parse_args` accepts the flag** without `NameError` / argparse error
   (catches flag-name typo / dest-mismatch early).
2. **The trainer imports + initialises cleanly** with the flag set (catches
   SCAFFOLD-module-not-imported regression — see Wf-385-A1 below).
3. **`decode_ratio > 0`** in the metrics JSON or stdout regex (catches the
   canonical `decode_ratio = 0/192` regression — see Wf-291D + Wf-299 below).
4. **`loss[0] > loss[49]`** — monotone non-increasing loss over 50 steps
   (catches silent loss-divergence from a broken loss-modifier wire-in).

**Implementation** (CPU-only, available today):

| File | Purpose |
|---|---|
| `molmetal/scripts/wf399a1_smoke_protocol.py` (~340 LOC) | `run_smoke_50step(trainer_path, flags_list)` + `verify_decode_ratio(smoke)` + `verify_loss_decrease(smoke)` + `verify_all_invariants(smoke)` + CLI entry point |
| `molmetal/tests/test_wf399a1_smoke_protocol.py` (~210 LOC, 2 NEW tests PASS) | Verifies the protocol helper itself is correct (mock trainer + multi-source decode_ratio parsing) |
| `TODO/completed/399a1_regression_rule.md` | Wf-399-A1 closeout (this entry) |

**Three rules**:

1. **Every trainer flag wire-in MUST be smoke-tested end-to-end before the
   wire-in is reported as SHIPPED.** A flag whose SCAFFOLD module has 4/4 unit
   tests PASS but is NOT imported by the trainer's main loop is **FLAG_ONLY**,
   not SHIPPED — see Wf-385-A1 §1 (17/17 Wf-380 NEW CLI flags were FLAG_ONLY;
   this is the exact regression class §5.7 prevents).

2. **Smoke invocation MUST use CPU + tiny config** (`--device cpu --epochs 1
   --batch-size 8` or equivalent) so it completes in <60 s on the test host.
   GPU launches are still USER_OK-gated per §5.1 — the §5.7 smoke is a
   CPU-only regression gate, not a GPU retrain.

3. **`verify_decode_ratio > 0` is the canonical honest-negative floor** for
   any CFM / YuelBond / decoder-class trainer. If `decode_ratio` regresses to
   0 on a 50-step smoke, the wire-in MUST be reverted or fixed before the
   wave is reported as SHIPPED. The historical decode_ratio=1 → 0 regression
   pattern has appeared at Wf-291D, Wf-299, Wf-380 (banner code), Wf-385-A1
   (audit) — §5.7 codifies the fix.

**Why this rule exists**: The Wf-385-A1 audit confirmed that the Wf-380 wave
SHIPPED 17 NEW CLI flags + 12 SCAFFOLD modules with 45/45 unit tests PASS,
yet ZERO of the 17 flags were wired into either trainer's main training loop.
Unit tests at the SCAFFOLD-module level do NOT catch trainer-integration
regressions — only an end-to-end smoke that exercises the trainer's `main`
function with the new flag set does. The Wf-291D / Wf-299 canonical
`decode_ratio = 0/192` regression (192 attempts, 0 successful decodes) was
invisible at the unit-test layer until a full pipeline smoke was run. §5.7
makes the smoke a mandatory gate, not an optional check.

**How to apply**: When your wave brief says "wire `--my-flag` into the trainer",
include a 50-step smoke invocation step (3 lines):

```python
from molmetal.scripts.wf399a1_smoke_protocol import run_smoke_50step, verify_all_invariants
smoke = run_smoke_50step(
    trainer_path="molmetal/scripts/train_cfm_yuelbond_wasserstein.py",
    flags_list=["--my-flag", "--epochs", "1", "--batch-size", "8", "--device", "cpu"],
)
verdict = verify_all_invariants(smoke)
assert verdict["pass"], f"WIRE-IN REGRESSION: {verdict}"
```

If the smoke fails, the wave cannot be reported as SHIPPED — re-open the
wire-in until `decode_ratio > 0` + `loss[0] > loss[49]` both hold. The
honest-framing caveat for the wave's closeout is: "Wire-in INCOMPLETE —
50-step smoke did NOT pass; flag is FLAG_ONLY."

**Cross-ref**: `molmetal/scripts/wf399a1_smoke_protocol.py` (NEW) +
`molmetal/tests/test_wf399a1_smoke_protocol.py` (NEW 2/2 PASS) +
`TODO/completed/399a1_regression_rule.md` (Wf-399-A1 closeout, NEW) +
`TODO/completed/385a1_wf380_cli_flag_audit.md` (Wf-385-A1 audit that surfaced
the FLAG_ONLY pattern) +
`TODO/completed/394a1_lambda_cfm_wire.md` (Wf-394-A1 wire-in audit + smoke
scaffold — the canonical reference for §5.7's pattern) +
`TODO/completed/38_lit_honest_synthesis.md` §N1 (Wf-291D + Wf-299 canonical
decode_ratio=0/192 → 1.000 fix) +
`TODO/completed/380b1_index_sync.md` §5 honest framing caveat #1
(SCAFFOLD-only integration documentation).

---

### §5.8 Workflow (wf) token-budget discipline (NEW 2026-10-02 per user voice)

**User voice (verbatim, 2026-10-02)**: *"随便一个workflow动不动就几M的token"* (any workflow casually burns several million tokens). Per user direction, this discipline rule caps per-workflow token consumption so sessions don't blow up.

**Three rules** (apply to ALL workflows launched from this repo going forward):

| # | Rule | Rationale | Enforcement |
|---:|---|---|---|
| 1 | **Per-workflow parallel agents ≤ 2** (was 4-6) | Each agent loads AGENTS.md + multiple .claude/skills + TODO/README.md → 0.5-1M tokens per agent context | Use `parallel([2 agents])` not `parallel([4-6 agents])`; combine tasks into 1 agent instead of splitting |
| 2 | **Per-agent file-read scope ≤ 3 files** (was 10+) | Each Read = 50-200K tokens; reading 20 files = ~2M tokens wasted on context | Use `grep -l` / `find` / Glob to locate first; Read ONLY the canonical design doc + 1-2 sibling docs + 1 representative data file |
| 3 | **Per-agent output ≤ 1500 tokens** (was 5000+) | Verbose JSON returns burn downstream phase context | Use compact markdown bullets; structured output ONLY for required schema fields; "files_modified" + "verification" + 1-line "honest_framing" is sufficient |

**When to use wf vs direct Edit/Write**:

| Scope | Method | Rationale |
|---|---|---|
| 1-3 file edits, <50 lines change, single domain | **Direct Edit/Write** (no wf) | Agent context already loaded; avoid spinning up parallel agents for trivial changes |
| Single Python script bug fix | **Direct Edit/Write + Bash verify** | No need for parallel exploration |
| Audit / refactor of >5 files OR cross-domain verification | **wf with 2 parallel agents** | Parallelism adds value; cap at 2 to limit token |
| Multi-step orchestration (>3 phases) | **wf with sequential agents** | Pipeline + barrier if cross-item; cap phase agents at 2 |

**When NOT to use wf**:

- Single Edit / Write / Bash invocation
- Reading a file once and quoting it
- "Verify X exists" → `ls -la` is faster than spawning an agent

**Cookie budget memory aid** (for ongoing session budgeting):

| wf scale | tokens | use case |
|---|---|---|
| **Single direct Edit** | ~10K | Bug fix, 1-3 lines |
| **Single-agent wf** | ~50-200K | One well-scoped task |
| **2-agent wf** | ~300K-1M | Audit or refactor across 2 domains |
| **4-agent wf** (LEGACY) | ~1.5-3M | Only for genuinely independent parallel investigations |
| **6-agent wf** (LEGACY) | ~3-9M | Only for full domain audit; NOT for routine fixes |

**Standing directives re-confirmed**: §5.1 (USER_OK-gated GPU) + §5.2 (honest framing) + §5.3 (no paper work) + §5.4 (no wet-lab planning) + §5.5 (filesystem-truth) + §5.6 (atomic write) + §5.7 (trainer flag wire-in smoke) + **§5.8 (wf token budget discipline — NEW)** all re-confirmed.

**Cross-references**: `docs/WORKFLOW_DEPENDENCY_GRAPH.md` (Wf-358-C1 + Wf-415-A4 cross-workflow tracking) + `workflow-authoring` skill §"Scale to what the user asked for" + `party-twenty-spirit` skill §3.5 honest-framing mandate.

**Migration note** (per Wf-432..Wf-435 retrospective 2026-10-02): the legacy Wf-432..Wf-435 waves ran at 2-3M tokens each because the agent spawned 4-6 parallel agents per wave. Going forward, ALL wf invocations must apply the 3 rules above OR document the exception in the wf name (e.g., `wf-NNN-large-audit-only`).

### §5.9 Single-venv rule (NEW per Wf-437 — 2026-10-02)

**MANDATE** (added per user directive *"我们不需要任何wave类型的代码版本管理,git就够了"*): all launch wrappers in `scripts/` MUST invoke the project venv `/home/hugo/codes/mol-metal/.venv/bin/python` — NOT bare `/usr/bin/python` or `python3`, NOT the legacy `CadButEaas` venvs (R17-R21 era), and NOT `kanzi_venv` (R18 era). Reasons:

1. **Reproducibility**: `.venv/bin/python` is the canonical Mol-Metal venv; it carries torch 2.14.0+cu130 + Blackwell sm_120 + YuelBond decoder + all SCAFFOLD modules. Bare `python` lacks torch and triggers the `p3_wf343c4_yuelbond_decoder_env` SKIP.
2. **No environment drift**: cross-wave wrappers historically invoked `CadButEaas` venvs from R17-R21; each wave rewrote their venv layout, causing transient divergence in installed torch / chemprop / scipy versions across waves. Single-venv rule closes that drift permanently.
3. **Git-only version control**: per user voice *"我们不需要任何wave类型的代码版本管理"*, the project is moving from wave-typed code → pure git-tracked code. The single-venv rule is the env-layer counterpart: ONE `.venv/` per repo, no per-wave sub-venvs.

**Pattern (canonical)**:

```bash
#!/bin/bash
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec "$REPO_ROOT/.venv/bin/python" -m molmetal.scripts.train_metal_lomo_wasserstein \
  --epochs 50 --seeds 0 --no-cuda-graphs --cuda-0 --user-ok
```

**Forbidden patterns**:

```bash
# WRONG: bare python
exec python molmetal/scripts/train_xxx.py
# WRONG: CadButEaas venv
exec /home/hugo/.venvs/CadButEaas/bin/python ...
# WRONG: kanzi_venv
exec /home/hugo/.venvs/kanzi_venv/bin/python ...
```

**Audit surface**: `wf437a1_single_venv_audit.py` (~30 LOC) scans `scripts/launch_*.sh` for `python` invocations; flags wrappers that use bare/python3/CadButEaas/kanzi_venv; returns non-zero if violations > 0.

**Enforcement on disk (2026-10-03)**: `scripts/_wrapper_audit.py` is the shipped §5.9 gate — its REAL-wrapper marker set requires the canonical `PY=/home/hugo/codes/mol-metal/.venv/bin/python`; the `wf437a1_single_venv_audit.py` cited above is NOT on disk.

**Cross-ref**: Wf-421-A2 §0 (YuelBond decoder env SKIP closed via venv) + Wf-427-B1 §22.123 (CLEAN pre-flight 113/0/4/117 via venv).

### §5.10 OmegaFold-gap-fill rule (NEW 2026-10-02 per Wf-442)

**MANDATE** (added per Wf-442 docs flatten + project hygiene + Wf-443 §5.10 sync): every Mol-Metal code pattern MUST mirror the OmegaFold reference repo (https://www.biorxiv.org/content/10.1101/2022.07.21.500999v1; local clone at `/home/hugo/OmegaFold/`). Six concrete rules:

1. **REFERENCE**: All layout / entry-point / docs / license / CI decisions cite the OmegaFold repo as the canonical pattern. When in doubt, `ls /home/hugo/OmegaFold/` first.
2. **FLAT PACKAGE LAYOUT**: collapse nested structures to a single `molmetal/` package per the OmegaFold `omegafold/` pattern (51 files, 1 console-script entry point). No nested top-level sub-packages, no per-script folders at repo root.
3. **ONE README.md**: ≤ 4 KB per the OmegaFold README.md size discipline. STATUS / QUICKSTART / ARCHITECTURE / DESIGN-BOUNDARY must live in separate `docs/*.md` files, NOT inside README.md.
4. **console_scripts ENTRY_POINT**: `molmetal = "molmetal.__main__:cli_main"` in `pyproject.toml` per the OmegaFold setup.py pattern. `molmetal/__main__.py` is the single dispatch surface — no per-script entry points.
5. **MINIMAL CI**: `.github/workflows/test.yml` must exist with **1 job + 1 pytest** invocation as a smoke gate. Heavy multi-job CI lives in `.github/workflows/ci.yml` separately.
6. **FLAT LICENSE**: Apache-2.0 header per `.py` file (boilerplate at top). No `LICENSE.md` / `LICENSE.txt` duplication — the canonical license is at repo root only.

**Audit surface**: `scripts/wf442_omegafold_audit.py` (~50 LOC, NEW per Wf-442) scans for: (a) README.md size ≤ 4 KB; (b) `molmetal` console-script entry-point present; (c) `molmetal/__main__.py` exists; (d) `.github/workflows/test.yml` exists with ≤ 2 jobs; (e) all `.py` files have Apache-2.0 header. Returns non-zero if any rule violated.

**Cross-ref**: Wf-442 Phase 1 A1 docs flatten (QUICKSTART.md + DESIGN_BOUNDARY.md + ARCHITECTURE.md created) + Wf-442 Phase 1 A2 project hygiene (`scripts/_pre_flight_check.sh` created) + Wf-437 single-venv rule (§5.9) + Wf-385 atomic-write rule (§5.6) + Wf-386 §5.5 post-wave verification rule + Wf-399 §5.7 trainer-flag wire-in smoke-required rule.

---



## §6 Recent Wf history pointer

**Full wave history is in [`docs/WF_HISTORY.md`](docs/WF_HISTORY.md).**

It was relocated there on 2026-10-03 because AGENTS.md had reached 1.2 MB,
and since project instructions are injected into every subagent that made
every agent spawn fail with `Prompt too long` — i.e. no workflow in this repo
could start. See that file's §0 for the measurement. **Nothing was deleted.**

Most recent work:

- **Wf-A10 (2026-10-03)** — NCI-60 9-cell-line task axis on REAL labels; split
  byte-identical to m21's; 5 scripts repaired (Wf-416-A1 migration corruption)
  plus an import-health guard. `TODO/completed/wfa10_nci60_cellline_axis.md`
- **Wf-A3..Wf-A8c (2026-10-02/03)** — YAML unification, startup-chain repair,
  TUI deep integration. `TODO/completed/A*_*.md`

For per-wave detail read `docs/WF_HISTORY.md`.

## §7 Critical scripts reference (cheat sheet)

| Path | Purpose |
|---|---|
| `molmetal/scripts/pre_flight_check.py` | Pre-flight v2 (TODO-207) |
| `molmetal/scripts/abort_recovery.py` | Abort recovery |
| `molmetal/scripts/per_wave_checklist.py` | Per-wave sub-checklist |
| `molmetal/scripts/wave_budget_tracker.py` | Wave budget + `--cfm-yuelbond-required` |
| `molmetal/scripts/read_anchor_json.py` | AnchorGuard helper (Wf-302A; max_age_sec=86400, required_version check) |
| `molmetal/scripts/wf302d_retry_with_correction.py` | StructuredOutput retry-with-correction helper (Wf-302D) |
| `molmetal/scripts/wf301e_harness_detector.py` | Harness shape detector (Wf-301E, Wf-298D) |
| `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` | CFM training entry (YuelBond decoder REQUIRED) |
| `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` | D-MPNN multi-task Wasserstein training |
| `molmetal/scripts/train_chemprop_v2_v3.py` | chemprop v2/v3 training (default `v3-fallback` per Wf-263A) |
| `molmetal/scripts/cfm_property_panel.py` | CFM property panel (SA / QED / diversity) |
| `molmetal/scripts/cfm_vina_dock_panel.py` | CFM Vina docking panel (BLOCKED on Pt+2 parameterisation per Wf-299B) |
| `molmetal/scripts/platinai_real_label_loader.py` | MB-Finder PlatinAI loader (Wf-295B) |
| `molmetal/scripts/platinai_selenium_loader.py` | PlatinAI Selenium S1 loader (Wf-292D / Wf-304B) |
| `scripts/launch_wf124_p3_g4.sh` | G4 launcher (TODO-180 CLOSED) |
| `scripts/launch_wf124_p3_g5.sh` | G5 launcher (TODO-181, 5 hr GPU, user OK required) |
| `scripts/launch_wf124_p3_g6.sh` | G6 launcher (TODO-182, 7.5 hr GPU, user OK required) |
| `scripts/launch_wf299_cfm_main.sh` | Wf-299 3-wave CFM launcher |

---

## §8 Workflow authoring conventions (Mol-Metal specifics)

Per `workflow-authoring` skill (system-bundled) + Wf-291C regression workaround:

1. **Plain-text briefs only.** Per Wf-291C root-cause + Wf-302C text-only template, agents
   confused StructuredOutput complex schema with the schema-as-deliverable. **Do NOT use
   StructuredOutput** in Mol-Metal workflows. Use plain-text OUTPUT sections (5-8 lines max).
2. **REPL canonical worktree location:** `/tmp/wfNNN_*.js` (persists via `scriptPath`).
3. **Plain-text OUTPUT template** (copy-paste this into each agent prompt):
   ```
   OUTPUT (PLAIN TEXT): <5-8 line summary>. Do NOT call StructuredOutput.
   ```
4. **Resume from cached results:** `Workflow({scriptPath, resumeFromRunId})` —
   unchanged (prompt, opts) replay from cache.
5. **Phase titles in `meta.phases`** must match `phase()` calls exactly.

---

## §9 When unsure

- "Where do I start?" → Read [`TODO/INDEX.md`](TODO/INDEX.md) §0 (recent ship wave) + §22 (detailed table).
- "Should I launch a GPU job?" → Check §5.1 kill_then_adapt directive. If > 1 hr, ASK USER.
- "Should I touch paper?" → Check §5.3. Default NO.
- "How do I frame a metric?" → Check §5.2 honest framing. MEASURED + cite source, or label PROJECTED.
- "What skill do I need?" → Check §3 skill catalog by keyword.
- "What workflow is in flight?" → Check §6 recent Wf history + active background IDs.
- **"What's the canonical honest-negative catalog?"** → Read
  [`docs/HONEST_NEGATIVE_CATALOG.md`](docs/HONEST_NEGATIVE_CATALOG.md) (Wf-358-C2 +
  Wf-359-B1 expansion 9 → 40 entries + Wf-369-A3 expansion 40 → 65 entries;
  6 sections: per-metric + per-workflow + per-launcher + resolved + open +
  Wf-340..368 consolidated). Verified by
  `molmetal/tests/test_wf358c2_honest_neg_catalog.py` (1 NEW test PASS) +
  `molmetal/tests/test_wf359b1_honest_neg_catalog.py` (1 NEW test PASS) +
  `molmetal/tests/test_wf369a3_catalog_size.py` (1 NEW test PASS).
  Cross-ref `TODO/completed/207_unified_pre_flight_checklist_v2.md` §9 for the
  legacy 9-entry catalog (per Wf-263B) + `TODO/completed/335_final_synthesis.md` §7 for
  Wf-335-SYN honest framing + `TODO/completed/358c2_honest_neg_catalog.md` for Wf-358-C2
  + `TODO/completed/369a3_catalog_update.md` for Wf-369-A3 expansion
  ship evidence.

---

## §10 Wetlab deprecation FAQ (NO WET-LAB WORK — DO NOT PLAN, EXECUTE, OR SEND)

**⚠️ HARD MANDATE — DO NOT PLAN, EXECUTE, OR SEND ANY WET-LAB WORK. ⚠️**

Per AGENTS.md §5.4 + 2026-09-30 user reinforcement "搞清楚我们不用做湿实验，这一点我已经明确了好多遍了":

### §10.1 What wet-lab items are DEPRECATED?

- **4 LOI drafts** (Rusanov / PKU / 协和 / ORNL) — **DO NOT SEND**
  - `TODO/pending/wetlab_loi_ornl_protein_metal.md`
  - `TODO/pending/wetlab_loi_pku_cisplatin_ic50.md`
  - `TODO/pending/wetlab_loi_xiehe_nci60.md`
  - (Rusanov PKU referenced historically)
- **3 wetlab SOPs** (cytotoxicity / cellular uptake / binding affinity) — **DO NOT EXECUTE**
  - `TODO/pending/wetlab_sop_binding_affinity_assay.md`
  - `TODO/pending/wetlab_sop_cellular_uptake_assay.md`
  - `TODO/pending/wetlab_sop_cytotoxicity_assay.md`
- **1 IC50 analysis pipeline** — **DO NOT RUN**
  - `TODO/pending/wetlab_ic50_analysis_pipeline.md`
- **R20-P6 wet-lab IC50 validation** (TODO-97) — DEPRECATED scope; re-scope to in-silico metric lift
- **Wf-74 v2 wet-lab outreach** — DEPRECATED scope
- **LAMBDA_ROADMAP §3.2 R22-P1 (m18 wet-lab)** — re-scope as computational-only
- **LAMBDA_ROADMAP §3.3 R23 (Phase I clinical)** — DEPRECATED

### §10.2 Why wet-lab removed?

- **Paper target: Tier-2** (downgrade from JCIM Tier-1). Examples: J. Comput. Aided Mol.
  Des. (JCAMD), J. Mol. Graph. Model., Comput. Chem. (WIREs), Mol. Divers., Comput. Biol.
  Chem. **No wet-lab data required for Tier-2 acceptance** — pure computational submission
  is in scope.
- **Wet-lab validation is 6-12 month external-dependency** that does not gate Tier-2 acceptance.
- **User directive (verbatim, 2026-09-29)**: "我们发二区就可以了，不用规划湿实验"
- **User reinforcement (verbatim, 2026-09-30)**: "搞清楚我们不用做湿实验，这一点我已经明确了好多遍了"

### §10.3 What to do if a task, plan, or paper section requests wet-lab work?

1. **REFUSE** — do not plan, write, or execute wet-lab work
2. **RE-SCOPE** — substitute in-silico alternative (computational simulation, literature values, peer-paper citation)
3. **CITE** — add HONEST_NEGATIVE entry to `docs/HONEST_NEGATIVE_CATALOG.md` flagging the wetlab → insilico substitution
4. **FLAG** — surface to user via TodoWrite / explicit message in final synthesis

### §10.4 What if a peer-reviewer rebuttal requires experimental data?

- **Re-frame** as computational substitution (per §10.3)
- **Cite** existing peer-paper experimental references (literature values are Tier-2 acceptable)
- **Mark** as `PROJECTED_FUTURE_WORK` rather than `MEASURED` in paper §6 Limitations

### §10.5 Wetlab templates retention policy

- **Retain** — keep all 7 wetlab pending files in `TODO/pending/` as **historical artifacts** (do NOT delete)
- **DO NOT execute / DO NOT send** — any agent action attempting to run an LOI or execute an SOP must REFUSE per §10.3
- **Cross-ref** — `TODO/README.md` `pending/` section has been updated to flag DEPRECATED status

### §10.6 Related directives

- **AGENTS.md §5.4** — Tier-2 journal target + NO wet-lab planning
- **TODO/README.md** — top-level wetlab deprecation notice in `pending/` section
- **docs/HONEST_NEGATIVE_CATALOG.md** — per-wetlab-file honest-negative entries (Wf-375 deferred)
- **TODO/03_plans/LAMBDA_ROADMAP.md** §3.2 R22-P1 + §3.3 R23 — DEPRECATED / re-scoped (Wf-375 verify)

---
