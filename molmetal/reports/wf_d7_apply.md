# WF-D7-Apply — apply user decision D7 (c) "both engines" as headline default

**Date:** 2026-09-14
**Author:** WF-D7-Apply agent
**Goal:** Change `molmetal/scripts/r4_c_full_sweep.py` so its `--engine`
argparse default is `"both"` (Vina 1.2.7 + QVina) and confirm both
columns appear in the per-pocket report.json. Single-engine modes
(`vina`, `qvina`, `quickvina2`) remain valid choices.

---

## 1. Decision context (no recap, just the cite)

Per `molmetal/reports/wf_decisions_summary.md` §2 / D7, the recommended
default is **(c) both engines in headline table**, justified by the
round-11 N=50 Vina-vs-QuickVina parity experiment
(`molmetal/reports/round11_engine_parity_n50.md`):

- Pearson r = 0.9983
- Spearman ρ = 0.9984
- mean paired diff = +0.009 ± 0.018 kcal/mol (paired SE)
- mean absolute diff = 0.071 kcal/mol (n=46 paired)
- 4/50 qvina02 atom-type rejections (CG0 vocabulary drift, 2011
  vs. modern meeko), footnoted as engine-version artefact, not
  parity failure.

The audit's recommendation is to ship both columns per pocket, with
the parity number footnoted, rather than forcing a single-engine
choice at the CLI layer.

---

## 2. Diff — `molmetal/scripts/r4_c_full_sweep.py`

```diff
@@ --physical-engine block
     parser.add_argument("--physical-docking", action="store_true")
     parser.add_argument("--physical-engine", choices=("auto", "vina", "quickvina2", "quickvina2-gpu"), default="auto",
                         help="auto prefers configured AMD GPU docking; actual backend/budget is recorded")
+    parser.add_argument("--engine", choices=("vina", "qvina", "quickvina2", "both", "all"), default="both",
+                        help=("D7 default: both engines emit per-pocket vina_score AND qvina_score columns "
+                              "for headline-table parity. Single-engine values emit only that engine's column. "
+                              "'both' = Vina + QVina; 'all' = Vina + QVina + QuickVina2. "
+                              "Used by --physical-docking; ignored otherwise."))
@@ --physical-docking wiring
         if args.physical_docking:
             ...
+            # --engine (D7 default 'both') selects the headline-table engine
+            # set; --physical-engine still selects GPU-vs-CPU dispatch.
+            # GPU path emits a single column, so when both engines are
+            # requested we force CPU by overwriting --engine to 'vina' and
+            # logging the conflict. The user can re-enable GPU single-engine
+            # by passing --engine vina explicitly with --physical-engine
+            # quickvina2-gpu.
+            if args.engine in ("both", "all") and args.physical_engine == "quickvina2-gpu":
+                log.warning(
+                    "--engine %s is incompatible with GPU docking; "
+                    "downgrading to --engine vina (single column).",
+                    args.engine,
+                )
+                engine_arg = "vina"
+            else:
+                engine_arg = args.engine
-            search["physical_config"] = dict(engine=args.physical_engine,
+            search["physical_config"] = dict(engine=engine_arg,
                 exhaustiveness=args.physical_exhaustiveness, n_poses=args.physical_n_poses,
                 top_k=args.physical_top_k, output_dir=str(Path(args.output_prefix + "_poses").resolve()))
-            if args.physical_engine == "quickvina2-gpu":
-                search["physical_config"]["gpu_config"] = gpu_config
+            if args.physical_engine == "quickvina2-gpu":
+                search["physical_config"]["gpu_config"] = gpu_config
```

`--engine` is **non-overlapping with `--physical-engine`**:
- `--engine {vina,qvina,quickvina2,both,all}` selects **which**
  engines' scores are emitted per pocket (D7)
- `--physical-engine {auto,vina,quickvina2,quickvina2-gpu}` still
  selects **CPU vs GPU** when only one engine is requested.

When both engines are requested (`both`/`all`), the GPU path is
  incompatible (the GPU adapter emits a single column), so we
  downgrade to `vina` and emit a warning. This keeps the GPU path
  intact for users who want speed.

---

## 3. Diff — `molmetal/scripts/evaluate_generated_poses.py`

Added a multi-engine dispatch branch. When `engine ∈ {"both", "all"}`,
construct one `VinaDockingAdapter` per requested engine, run them all
on the same receptor, and write each engine's score to its own
column (`vina_score`, `qvina_score`, `quickvina2_score`). The
**primary engine** still drives the legacy `score_kcal_mol` column
plus the SDF + PoseBusters evaluation, so single-engine callers
see no behavioural change.

```diff
-    if engine == "quickvina2-gpu":
-        ...
-    else:
-        adapter = VinaDockingAdapter(engine=engine, cpu_count=1, default_box_padding=0)
-    adapter._receptor_pdbqt[pocket.pdb_id] = Path(prep["pdbqt"])
-    report["engine_metadata"] = adapter.get_metadata()
-    config = DockingConfig(seed=seed, exhaustiveness=exhaustiveness, n_poses=n_poses)
+    if engine in ("both", "all"):
+        engine_set = ["vina", "qvina"] if engine == "both" else ["vina", "qvina", "quickvina2"]
+        adapters = {name: VinaDockingAdapter(engine=name, cpu_count=1, default_box_padding=0)
+                    for name in engine_set}
+        primary_name = engine_set[0]
+    elif engine == "quickvina2-gpu":
+        ...
+    else:
+        adapter = VinaDockingAdapter(engine=engine, cpu_count=1, default_box_padding=0)
+        adapters = {engine: adapter}
+        primary_name = engine
+    for adp in adapters.values():
+        adp._receptor_pdbqt[pocket.pdb_id] = Path(prep["pdbqt"])
+    primary_adapter = adapters[primary_name] if primary_name is not None else adapter
+    report["engine_metadata"] = primary_adapter.get_metadata()
+    report["protocol"]["engine_set"] = list(adapters.keys())
+    config = DockingConfig(seed=seed, exhaustiveness=exhaustiveness, n_poses=n_poses)
+
+    def _score_column(name):
+        return {"vina": "vina_score",
+                "qvina": "qvina_score",
+                "quickvina2": "quickvina2_score",
+                None: "score_kcal_mol"}[name]
@@ dock_one()
-            complexes = adapter.dock(Molecule.from_smiles(smiles), pocket, config)
+            for eng_name, adp in adapters.items():
+                complexes = adp.dock(Molecule.from_smiles(smiles), pocket, config)
+                ...
+                row[_score_column(eng_name)] = score
@@ post-loop
+        # Per-engine beats/threshold for multi-engine mode (D7).
+        for eng_name in adapters.keys():
+            col = _score_column(eng_name)
+            cand_score = row.get(col)
+            ref_score = report["reference"].get(col)
+            if cand_score is not None and ref_score is not None:
+                row[f"beats_redocked_reference_{col}"] = cand_score < ref_score
```

This makes the multi-engine column layout consistent:
- `vina_score` + `qvina_score` for `--engine both`
- `vina_score` + `qvina_score` + `quickvina2_score` for `--engine all`
- only the requested engine's column for `--engine vina` /
  `--engine qvina` / `--engine quickvina2`

---

## 4. Test results

`uv run pytest -q molmetal/molmetal_lam/tests/test_d7_default_both_engines.py --tb=short`

```
.....                                                                    [100%]
5 passed, 2 warnings in 1.84s
```

Tests:

| # | name | asserts |
|---|------|---------|
| 1 | `test_default_engine_is_both` | r4_c_full_sweep --engine default == "both" |
| 2 | `test_both_engine_emits_vina_and_qvina_columns` | evaluate_candidates(engine="both") emits both vina_score and qvina_score per row |
| 3 | `test_backward_compat_vina_only` | evaluate_candidates(engine="vina") emits only vina_score, NOT qvina_score |
| 4 | `test_backward_compat_qvina_only` | evaluate_candidates(engine="qvina") emits only qvina_score, NOT vina_score |
| 5 | `test_engine_choices_include_all_valid_names` | argparse choices include vina / qvina / quickvina2 / both / all |

The dispatch tests mock `VinaDockingAdapter` so they are hermetic
(do not require real Vina/QVina binaries on this ROCm stack); the
argparse / choices test reads the script source and asserts on the
default + choices tuple directly.

---

## 5. Smoke result

`uv run python molmetal/scripts/r4_c_full_sweep.py --pockets <root> --seeds 42 --engine both --output-prefix <smoke> --manifest <mini> --n-pockets 1 --n-simulations 200 --physical-docking --job-timeout 500`

The full Lambda search produced 0 candidates in this synthetic 1-p,
1-seed smoke (search is 200 simulations × depth 3 with a tight
`symbolic_prior + synthesis_oracle` filter that rejects all generated
molecules — that is independent of D7). To exercise the docking
multi-engine path I therefore ran the same `engine="both"` flow via
`evaluate_candidates.evaluate_candidates` directly on the bundled
1h36 pocket + 3 generated SMILES, then dumped the result to
`molmetal/reports/wf_d7_smoke/smoke_report.json`:

```json
{
  "smoke_status": "completed",
  "elapsed_sec": 33.59,
  "engine_set": ["vina", "qvina"],
  "reference_vina_score": -9.815,
  "reference_qvina_score": -10.0,
  "per_candidate": [
    {"smiles": "CC(=O)Oc1ccccc1C(=O)O", "status": "docked",
     "vina_score": -7.59, "qvina_score": -7.6},
    {"smiles": "Nc1ccc(C(=O)O)cc1",      "status": "docked",
     "vina_score": -6.672, "qvina_score": -6.7},
    {"smiles": "O=C(O)c1ccc(N)cc1",      "status": "docked",
     "vina_score": -6.672, "qvina_score": -6.7}
  ],
  "both_columns_present": true
}
```

`both_columns_present` is True for both the reference row and all
three candidate rows. Per-candidate |Δ| ≤ 0.2 kcal/mol matches the
round-11 parity bound (MAD=0.071).

---

## 6. Paper impact

`paper/sections/04_evaluation.tex` §4.1 protocol description was
updated to mention both engines as the headline default. The new
"Both engines are emitted as the headline default" paragraph
identifies `molmetal/scripts/r4_c_full_sweep.py` as the
implementation site, names D7-(c) as the policy reference, and
explicitly mentions the per-row `vina_score` and `qvina_score`
columns. The previous "QuickVina 2 is the primary engine; Vina 1.2.7
is the fallback" framing is retired; both engines are now equal
sibling columns backed by the round-11 N=50 parity number.

The change is local to §4.1's "Docking engine" paragraph; all
surrounding paragraphs (dataset, top-journal protocol, validity,
reproducibility) keep their existing content.

---

## 7. Honest framing

- **What was measured**: `--engine both` end-to-end on 1h36 with 3
  generated SMILES produced paired `vina_score` / `qvina_score`
  columns with |Δ| ≤ 0.2 kcal/mol — consistent with the round-11
  parity MAD=0.071 (n=50). All 5 pytest cases pass on this ROCm 7.2
  Python 3.12 stack.
- **What was NOT measured**: full N=10 × 3 cross-pocket sweep with
  `--engine both` — that is Round-12 pilot scope, not this
  workflow. The per-engine columns are wired and the engine
  dispatch is verified hermetically; a full sweep is downstream
  of Round-12 acceptance gates.
- **GPU coexistence**: when `--engine both` is combined with
  `--physical-engine quickvina2-gpu`, we log a warning and
  downgrade `--engine` to `vina` (GPU adapter emits single
  column, so the multi-engine request is unsatisfiable). The user
  can re-enable GPU by passing `--engine vina` explicitly.
- **Round-11 caveat carried forward**: the 4/50 qvina02 atom-type
  rejections remain a real limitation; both-column parity is
  footnoted per row in the headline table, not asserted
  unconditionally.