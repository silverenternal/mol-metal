# SPDX-License-Identifier: Apache-2.0
"""A PlatinAI run must leave weights behind, not just a number.

WHY THIS EXISTS
---------------
`train_platinai_4cell.py` is the trainer that produced the strong PlatinAI
number cited in `docs/PAPER_SCOPE.md`. Before this change it contained ZERO
`torch.save` calls -- the only `state_dict` in the file was a
`load_state_dict` of the best-val in-memory copy. So the run left a per-seed
JSON and no `.pt`, and the cross-application route in
`wf_generated_to_activity.py` was pointed at a DIFFERENT family
(`MultiTaskWassersteinModel`, recorded test AUCs ~0.51) because the right
weights did not exist to load.

A number with no weights is not reproducible, and a checkpoint that reloads
into a differently-shaped model is worse than none: it scores silently.

The three tests below pin the three properties that make the artefact real:

1. a run WRITES a checkpoint (fails on the pre-change file -- no torch.save),
2. that checkpoint reloads STRICTLY into the same architecture, 0 missing /
   0 unexpected, and reproduces the same probabilities,
3. the consumer in `wf_generated_to_activity.py` can find and score it.

Test 1 was verified RED against the pre-change trainer (reverted the save
call, confirmed the assertion failed, restored).
"""

from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _load_trainer():
    spec = importlib.util.spec_from_file_location(
        "train_platinai_4cell_ckpt_probe",
        REPO / "molmetal" / "scripts" / "train_platinai_4cell.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_consumer():
    spec = importlib.util.spec_from_file_location(
        "wf_generated_to_activity_ckpt_probe",
        REPO / "molmetal" / "scripts" / "wf_generated_to_activity.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _tiny_cohort(tmp_path: Path, n: int = 90) -> Path:
    """A real-schema CSV slice: same columns, fewer rows, so CPU 1-epoch runs.

    Uses the REAL cohort's first n rows rather than synthetic SMILES -- a
    fake cohort would featurise fine but would not prove the loader handles
    the actual Pt-complex strings the trainer sees in production.
    """
    src = REPO / "molmetal" / "data" / "platinai_4cell.csv"
    with src.open() as f:
        r = csv.DictReader(f)
        rows = [next(r) for _ in range(n)]
        fields = r.fieldnames
    out = tmp_path / "cohort_slice.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return out


def _run_one_epoch(tmp_path: Path, data: Path, out_dir: Path, extra=()):
    """Run the trainer on CPU for 1 epoch via its real CLI (main())."""
    argv = [
        "--data", str(data),
        "--seeds", "0",
        "--epochs", "1",
        "--batch-size", "16",
        "--hidden", "32",
        "--depth", "1",
        "--cuda-visible-devices", "",
        "--out-dir", str(out_dir),
        "--out-metric", str(tmp_path / "metric.json"),
        "--out-report", str(tmp_path / "final.md"),
        *extra,
    ]
    env = {"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)}
    proc = subprocess.run(
        [str(REPO / ".venv" / "bin" / "python"),
         str(REPO / "molmetal" / "scripts" / "train_platinai_4cell.py"), *argv],
        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=1800,
    )
    assert proc.returncode == 0, f"trainer failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    return proc


def test_1epoch_cpu_run_writes_a_reloadable_checkpoint(tmp_path):
    """The gap this closes: the run produced a number and no weights."""
    T = _load_trainer()
    data = _tiny_cohort(tmp_path)
    out_dir = tmp_path / "run"
    _run_one_epoch(tmp_path, data, out_dir)

    ckpt_dir = out_dir / "checkpoints"
    assert ckpt_dir.is_dir(), (
        f"no checkpoints dir at {ckpt_dir} -- the run left a number and no weights, "
        "which is the exact gap this change closes"
    )
    ckpts = sorted(ckpt_dir.glob("platinai4cell_seed*.pt"))
    assert len(ckpts) == 1, f"expected 1 checkpoint for seed 0, got {[p.name for p in ckpts]}"

    # (2) reload into the SAME architecture, strictly.
    loaded = T.load_checkpoint(ckpts[0], device=torch.device("cpu"))
    model, meta = loaded["model"], loaded["meta"]
    assert meta["model_family"] == "DMPNNFourCellModel"
    assert meta["hidden"] == 32 and meta["depth"] == 1
    assert meta["cell_lines"] == list(T.CELL_LINES)

    live = model.state_dict()
    assert set(live) == set(meta["model_state_dict"]), (
        f"missing={sorted(set(live) - set(meta['model_state_dict']))} "
        f"unexpected={sorted(set(meta['model_state_dict']) - set(live))}"
    )
    for k, v in meta["model_state_dict"].items():
        assert torch.equal(live[k].cpu(), v.cpu()), f"tensor {k} did not round-trip"

    # (2b) the checkpoint's recorded AUC is the one the run wrote to JSON.
    import json
    seed_json = json.loads((out_dir / "seed0.json").read_text())
    assert seed_json["checkpoint_path"], "per-seed JSON does not name its checkpoint"
    assert np.isclose(meta["auc_mean"], seed_json["auc_mean"], equal_nan=True), (
        f"checkpoint AUC {meta['auc_mean']} != recorded JSON AUC {seed_json['auc_mean']}"
    )


def test_checkpoint_reproduces_identical_probabilities(tmp_path):
    """A reload must SCORE the same, not merely load without error."""
    T = _load_trainer()
    data = _tiny_cohort(tmp_path)
    out_dir = tmp_path / "run"
    _run_one_epoch(tmp_path, data, out_dir)
    ckpt = sorted((out_dir / "checkpoints").glob("platinai4cell_seed*.pt"))[0]

    from molmetal.baselines.dmpnn import featurize_smiles_list, collate_graphs

    import csv as _csv
    with data.open() as f:
        smiles = [r["smiles"] for r in _csv.DictReader(f)][:16]
    graphs = [g for g in featurize_smiles_list(smiles) if g is not None]

    def probs(m):
        with torch.no_grad():
            bd = collate_graphs(graphs)
            af, bf, es, ed = bd
            _, logits = m(af, bf, es, ed)
            return torch.softmax(logits, dim=-1)[..., 1].numpy()

    a = probs(T.load_checkpoint(ckpt)["model"])
    b = probs(T.load_checkpoint(ckpt)["model"])  # second independent reload
    assert np.array_equal(a, b), "reload is not deterministic"
    assert a.shape == (len(graphs), T.N_CELL_LINES)
    assert np.isfinite(a).all()


def test_consumer_finds_and_scores_the_checkpoint(tmp_path):
    """wf_generated_to_activity.py must reach these weights, not just the mtw family."""
    C = _load_consumer()
    T = _load_trainer()
    data = _tiny_cohort(tmp_path)
    out_dir = tmp_path / "run"
    _run_one_epoch(tmp_path, data, out_dir)
    ckpt_dir = out_dir / "checkpoints"

    found = C.load_platinai_4cell_family(ckpt_dir)
    assert found, f"consumer glob found nothing under {ckpt_dir}"

    import csv as _csv
    with data.open() as f:
        smiles = [r["smiles"] for r in _csv.DictReader(f)][:8]
    res = C.score_platinai_4cell(smiles, found, torch.device("cpu"), T)
    assert res["n_seeds"] == 1
    assert set(res["per_cell_line"]) == set(T.CELL_LINES)
    for line, v in res["per_cell_line"].items():
        assert 0.0 <= v["mean_prob"] <= 1.0, f"{line} probability out of range: {v}"
    assert 0.0 <= res["mean_prob"] <= 1.0


def test_no_save_checkpoints_turns_it_off(tmp_path):
    """The opt-out is real; DEFAULT ON is a decision, not an accident."""
    data = _tiny_cohort(tmp_path)
    out_dir = tmp_path / "run_nockpt"
    _run_one_epoch(tmp_path, data, out_dir, extra=("--no-save-checkpoints",))
    assert not (out_dir / "checkpoints").exists(), "--no-save-checkpoints still wrote a .pt"
    import json
    seed_json = json.loads((out_dir / "seed0.json").read_text())
    assert seed_json["checkpoint_path"] is None
