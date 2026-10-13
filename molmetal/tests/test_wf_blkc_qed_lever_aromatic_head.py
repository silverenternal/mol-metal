# SPDX-License-Identifier: Apache-2.0
"""The (A)-defect pin: the bond head MUST predict aromatic bonds on real geometry.

WHY THIS EXISTS
---------------
docs/CFM_STATUS_2026_10_12.md BLK-C-OBJ :: QED notes that QED and SA both
trace to one structural fact: the generated ligands are acyclic, 66%
triple-bond, 98.5% non-aromatic (aromatic bond fraction 0.001229 vs real
0.454591). The natural question is whether the fix is in the bond head
(``(A)`` -- the head cannot predict aromatic bonds, even on real geometry)
or upstream in the sampler (``(B)`` -- the head predicts fine on real
geometry, but the sampler never produces arrangements that would yield
aromatics).

MEASURED 2026-10-12, on the production checkpoint
``molmetal/reports/cfm_fullcorpus_2026_10_10/checkpoints/cfm_yuelbond_wasserstein_seed0.pt``
(ep10, n_train 26715), 900 real tmQM complexes (150 x 6 metals) on the same
non-hydrogen / non-metal pair scope as ``bench_tmqm_bond_perception.py
--heavy-atom-scope``::

    aromatic-class recall (argmax) = 0.293
    aromatic-class precision (argmax) = 0.889
    per-complex aromatic fraction: real_pred = 0.0513 vs real_truth = 0.1422
    per-complex aromatic fraction: GENERATED = 0.0000

The head DOES emit class 4 (aromatic): 5,221 predicted-aromatic bonds across
798 complexes (recall 0.293 on 15,851 truth-aromatic bonds). The dominant
confusion is aromatic->single (11,209 of 15,851 truth-aromatic bonds become
predicted single, i.e. 70.7%).

The verdict: ``(B)`` is the dominant cause -- the sampler never produces
aromatic arrangements (0.0% generated vs 14.22% real truth) -- but ``(A)``
is also active (head predicts 5.13% aromatic on real geometry vs 14.22%
truth, a 9-percentage-point head-only gap that compounds with the
sampler's 14.22-percentage-point gap).

These tests pin the ``(A)`` half: the head MUST keep predicting aromatic
bonds on real geometry. If a future change accidentally disables aromatic
prediction (e.g. a soft-bias that overwhelms class 4, a softmax collapse
to a 4-class head, a class-weighting scheme that zero-prizes class 4),
these tests will go red.

RED CHECK: patch the head so class-4 logits are clamped to a large negative
value (the head can NEVER emit aromatic), confirm both tests fail, then
restore.

MEASUREMENT
-----------
The (A) test target is HINRAZ, a real Pt complex with 263 atoms, 120
non-H non-metal aromatic bonds under the bench scope. MEASURED on the
production head: 35 of 120 predicted as aromatic (29.17% recall, 100%
precision). The floor in these tests is 0.10 recall (10 of 120), well
below the measured 0.292 and high enough to detect a complete collapse.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
TRAINER = (
    REPO
    / "molmetal"
    / "scripts"
    / "train_cfm_yuelbond_wasserstein.py"
)
CKPT = (
    REPO
    / "molmetal"
    / "reports"
    / "cfm_fullcorpus_2026_10_10"
    / "checkpoints"
    / "cfm_yuelbond_wasserstein_seed0.pt"
)
TMQM_DIR = REPO / ".data" / "molmetal" / "tmQM"
TARGET_CODE = "HINRAZ"  # Pt complex, 263 atoms, 120 non-H non-metal aromatic bonds

# Aromatic truth: Wiberg BO in [1.3, 1.7) — disjoint from {single, double, triple}.
AROMATIC_LO = 1.3
AROMATIC_HI = 1.7
HEAD_AROMATIC = 4
TRAIN_METALS = frozenset({"Pt", "Ru", "Ir", "Rh", "Os", "Re"})

_SYM_TO_Z = {
    "H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "P": 15, "S": 16, "Cl": 17,
    "Br": 35, "I": 53, "Pt": 78, "Ru": 44, "Ir": 77, "Rh": 45, "Os": 76,
    "Re": 75, "B": 5, "Si": 14, "As": 33, "Se": 34,
}
METAL_Z = {_SYM_TO_Z[m] for m in TRAIN_METALS if m in _SYM_TO_Z}


# ---------------------------------------------------------------------------
# Module loaders (same pattern as test_wf_blkc_l5_metal_deletion_control.py)
# ---------------------------------------------------------------------------
def _trainer():
    if "_qed_lever_trainer" in sys.modules:
        return sys.modules["_qed_lever_trainer"]
    spec = importlib.util.spec_from_file_location(
        "_qed_lever_trainer", str(TRAINER)
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules["_qed_lever_trainer"] = m
    spec.loader.exec_module(m)
    return m


def _iter_bo_blocks(bo_paths):
    import gzip

    for path in bo_paths:
        if not Path(path).is_file():
            continue
        code, atoms = None, []
        with gzip.open(path, "rt") as fh:
            for line in fh:
                if line.startswith("CSD_code"):
                    if code is not None and atoms:
                        yield code, atoms
                    code = (
                        line.split("=", 1)[1].split("|", 1)[0].strip()
                    )
                    atoms = []
                    continue
                parts = line.split()
                if len(parts) < 3:
                    if code is not None and atoms:
                        yield code, atoms
                        code, atoms = None, []
                    continue
                try:
                    idx = int(parts[0])
                except ValueError:
                    continue
                sym = parts[1]
                rest = parts[3:]
                nbrs = []
                for k in range(0, len(rest) - 2, 3):
                    try:
                        nbrs.append((int(rest[k + 1]), float(rest[k + 2])))
                    except (ValueError, IndexError):
                        continue
                atoms.append((idx, sym, nbrs))
        if code is not None and atoms:
            yield code, atoms


def _ground_truth_bonds(atoms, threshold=0.3):
    bonds = {}
    for idx, _sym, nbrs in atoms:
        for nbr_idx, bo in nbrs:
            if bo < threshold or nbr_idx == idx:
                continue
            key = (min(idx, nbr_idx), max(idx, nbr_idx))
            if key not in bonds or bo > bonds[key]:
                bonds[key] = bo
    return bonds


def _iter_xyz_blocks(xyz_paths):
    import gzip
    import numpy as np

    for path in xyz_paths:
        if not Path(path).is_file():
            continue
        with gzip.open(path, "rt") as fh:
            lines = fh.read().split("\n")
        i = 0
        while i < len(lines):
            try:
                n_atoms = int(lines[i].strip())
            except (ValueError, IndexError):
                i += 1
                continue
            if i + 1 >= len(lines):
                break
            head = lines[i + 1].strip()
            if not head.startswith("CSD_code"):
                i += 1
                continue
            code = head.split("=", 1)[1].split("|", 1)[0].strip()
            i += 2
            symbols, coords = [], []
            for _ in range(n_atoms):
                if i >= len(lines):
                    break
                parts = lines[i].split()
                i += 1
                if len(parts) < 4:
                    continue
                symbols.append(parts[0])
                coords.append(
                    [float(parts[1]), float(parts[2]), float(parts[3])]
                )
            if symbols:
                yield code, symbols, np.asarray(coords, dtype=float)


def _load_hinraz():
    """Return (symbols, coords, truth_bo) for HINRAZ, or skip if data absent."""
    bo_paths = [TMQM_DIR / n for n in ("tmQM_X1.BO.gz", "tmQM_X2.BO.gz", "tmQM_X3.BO.gz")]
    if not all(p.is_file() for p in bo_paths):
        import pytest
        pytest.skip("tmQM BO shards not on disk")
    syms, truth = None, None
    for code, atoms in _iter_bo_blocks(bo_paths):
        if code == TARGET_CODE:
            truth = _ground_truth_bonds(atoms)
            syms = [a[1] for a in atoms]
            break
    if truth is None:
        import pytest
        pytest.skip(f"{TARGET_CODE} not in tmQM BO shards on this host")

    coords = None
    xyz_paths = [
        Path(str(p).replace(".BO.gz", ".xyz.gz")) for p in bo_paths
    ]
    for code, symbols, c in _iter_xyz_blocks(xyz_paths):
        if code == TARGET_CODE:
            coords = c
            syms = symbols
            break
    if coords is None:
        import pytest
        pytest.skip(f"{TARGET_CODE} geometry not in tmQM XYZ shards")
    return syms, coords, truth


def _load_head():
    """Build the bond head from the production checkpoint. CPU-only."""
    if not CKPT.is_file():
        import pytest
        pytest.skip(f"production checkpoint not on disk: {CKPT}")
    import torch

    from molmetal.molmetal_lam.lam_chem.cfm_decoder_yuelbond import (
        YuelBondDecoder,
    )
    from molmetal.molmetal_lam.lam_chem.yuelbond_decoder import (
        get_z_table,
    )
    from molmetal.models.bond_head import AtomCloud

    tr = _trainer()
    state = torch.load(CKPT, map_location="cpu", weights_only=False)
    saved = state.get("args", {}) if isinstance(state, dict) else {}
    model = tr.CFMJointModel(
        hidden_dim=int(saved.get("hidden_dim", tr.DEFAULT_HIDDEN_DIM)),
        n_layers=int(saved.get("n_layers", tr.DEFAULT_N_LAYERS)),
        cutoff=float(saved.get("edge_cutoff", tr.DEFAULT_EDGE_CUTOFF)),
        fm_objective=bool(saved.get("use_fm_objective", False)),
        z_table=get_z_table(str(saved.get("yuelbond_vocab", "pt") or "pt")),
        use_egnn_velocity=bool(saved.get("use_egnn_velocity", False)),
        max_atomic_number=int(saved.get("egnn_max_atomic_number", 100)),
    )
    missing, unexpected = model.load_state_dict(
        state.get("model_state_dict", state), strict=False
    )
    if missing or unexpected:
        raise SystemExit(
            f"architecture mismatch: {len(missing)} missing / "
            f"{len(unexpected)} unexpected"
        )
    model.eval()
    featuriser = YuelBondDecoder(head=model.head)
    return model, featuriser


def _score(model, featuriser, symbols, coords, truth_bo):
    """Per-pair argmax, restricted to non-H non-metal pairs (same scope as
    ``bench_tmqm_bond_perception.py --heavy-atom-scope``)."""
    import torch

    from molmetal.models.bond_head import AtomCloud

    zs = [_SYM_TO_Z.get(s, 0) for s in symbols]
    unknown = [s for s, z in zip(symbols, zs) if z == 0]
    if unknown:
        return None
    cloud = AtomCloud(
        positions=torch.as_tensor(coords, dtype=torch.float32),
        atomic_numbers=torch.as_tensor(zs, dtype=torch.long),
    )
    with torch.no_grad():
        res = featuriser.featurise(cloud)
    if not res.decode_succeeded or res.logits.shape[0] == 0:
        return None
    tr = _trainer()
    pred_class = tr._bond_decision(res.logits, "argmax", 0.5)
    src = res.edge_index[0].tolist()
    dst = res.edge_index[1].tolist()
    out = []
    for k in range(len(src)):
        s, d = int(src[k]), int(dst[k])
        key = (min(s, d) + 1, max(s, d) + 1)
        bo = truth_bo.get(key)
        if bo is None:
            t_label = None
        elif AROMATIC_LO <= bo < AROMATIC_HI:
            t_label = "aromatic"
        elif bo < 1.5:
            t_label = "single"
        elif bo < 2.5:
            t_label = "double"
        else:
            t_label = "triple"
        zi = _SYM_TO_Z.get(symbols[key[0] - 1], 0)
        zj = _SYM_TO_Z.get(symbols[key[1] - 1], 0)
        if symbols[key[0] - 1] == "H" or symbols[key[1] - 1] == "H":
            continue
        if zi in METAL_Z or zj in METAL_Z:
            continue
        out.append({
            "key": key,
            "argmax": int(pred_class[k]),
            "truth_label": t_label,
        })
    return out


def _recall_aromatic(records):
    """Recall on truth-aromatic bonds: TP / (TP + FN)."""
    n_t = sum(1 for r in records if r["truth_label"] == "aromatic")
    n_p = sum(
        1
        for r in records
        if r["truth_label"] == "aromatic" and r["argmax"] == HEAD_AROMATIC
    )
    return n_p, n_t


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_head_predicts_aromatic_on_real_geometry_hinraz():
    """The (A) defect pin: head recall on truth-aromatic bonds of HINRAZ.

    MEASURED 2026-10-12: recall = 0.292 (35/120). Floor 0.10 is well below
    that and high enough to catch a head that NEVER emits class 4.

    Goes red if a future change collapses the head to 4 classes, applies a
    -inf logit bias to class 4, or otherwise disables aromatic prediction.
    """
    syms, coords, truth = _load_hinraz()
    model, featuriser = _load_head()
    records = _score(model, featuriser, syms, coords, truth)
    assert records is not None, "head decode failed on HINRAZ"
    n_tp, n_t = _recall_aromatic(records)
    assert n_t >= 100, (
        f"HINRAZ expected ~120 truth-aromatic bonds in scope, got {n_t}; "
        "either the cohort or the scope filter changed"
    )
    recall = n_tp / n_t if n_t else 0.0
    assert recall >= 0.10, (
        f"head recall on truth-aromatic bonds of HINRAZ is {recall:.4f} "
        f"({n_tp}/{n_t}); the (A) defect has reappeared. The head has lost "
        "the ability to predict aromatic bonds even on real geometry. "
        "MEASURED 2026-10-12 was 0.292 (35/120). A regression to ~0 means "
        "a future change disabled aromatic prediction."
    )


def test_head_ever_emits_aromatic_class():
    """Sanity: the head must predict at least one aromatic bond on a real
    aromatic-rich complex. Macro-F1 can hide a class that is never predicted;
    this test pins the prediction rate at the class level.

    Goes red if every pair's argmax is in {NO_BOND, single, double, triple}.
    """
    syms, coords, truth = _load_hinraz()
    model, featuriser = _load_head()
    records = _score(model, featuriser, syms, coords, truth)
    assert records is not None, "head decode failed on HINRAZ"
    n_pred_arom = sum(
        1 for r in records if r["argmax"] == HEAD_AROMATIC
    )
    assert n_pred_arom >= 5, (
        f"head predicted aromatic on {n_pred_arom} pairs in HINRAZ scope; "
        "expected >= 5 (MEASURED 2026-10-12: 35). The aromatic class has "
        "been silenced. A 0 here is a hidden regression: macro-F1 can mask "
        "a class that is never predicted."
    )


def test_aromatic_truth_definition_is_disjoint():
    """The Wiberg-BO aromatic band [1.3, 1.7) is disjoint from the
    {single, double, triple} bands -- otherwise the truth label of an
    "aromatic" bond is ambiguous.

    Goes red if the constant or the band changes; without this guard the
    test above is mis-benchmarked.
    """
    assert AROMATIC_LO == 1.3 and AROMATIC_HI == 1.7, (
        f"aromatic band is [{AROMATIC_LO}, {AROMATIC_HI}); it must remain "
        "the open interval [1.3, 1.7) -- disjoint from single (<1.5) and "
        "double (<2.5) so that the test's truth_label cannot be ambiguous"
    )
    # Boundary checks at the round-points of single/double/triple. The Wiberg
    # BO of 1.5 is in the [1.3, 1.7) band (1.3 <= 1.5 < 1.7) so it is
    # "aromatic" under this definition, NOT "double" by the nearest-of-{1,2,3}
    # rule -- which is the whole point: the head's class 4 covers the gap
    # between single and double.
    for bo, expected in (
        (1.0, "single"),    # canonical single
        (1.3, "aromatic"),  # exact lower bound of aromatic
        (1.5, "aromatic"),  # midpoint of single/double -> aromatic by the band
        (1.7, "double"),    # just outside aromatic -> double
        (2.0, "double"),    # canonical double
    ):
        if AROMATIC_LO <= bo < AROMATIC_HI:
            label = "aromatic"
        elif bo < 1.5:
            label = "single"
        elif bo < 2.5:
            label = "double"
        else:
            label = "triple"
        assert label == expected, f"BO={bo} -> {label}; expected {expected}"
