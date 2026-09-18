"""End-to-end validation of the new joint atom-type + coord FM ``generate()``.

Loads the Ru subset of MetalCytoToxDB, trains the
:class:`LipmanFlowMatchingAdapter` for 30 CFM + atom-type-CE steps on 200
unique molecules (≤ 30 atoms), then samples 100 molecules via ``generate()``
and reports on:

* Atom-type distribution histogram (Z = 1..20 and Z > 20)
* Mean QED and fraction drug-like (QED ≥ 0.5)
* Mean MolWt and fraction in drug-like range [200, 500]
* Comparison against the training-set reference distribution

The report is written to ``molmetal/reports/generate_validation.md``.

Run via::

    source .venv/bin/activate && python -m molmetal.scripts.validate_generate
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np
import torch

from molmetal.adapters.flow_matching_lipman import LipmanFlowMatchingAdapter
from molmetal.adapters.rdkit_predictor import RDKitPropertyPredictor
from molmetal.data.cytotox import CytotoxFilter, MetalCytotoxDataset
from molmetal.domain import Molecule
from molmetal.ports import GenerationConfig
from molmetal.utils.device import verify_rocm_active


# GenerationConfig is frozen and has no `n_atoms` field; the adapter
# uses ``getattr(config, "n_atoms", 8)`` so we just need a namespace that
# satisfies the duck-typed access.  We could subclass, but that's overkill
# for a one-shot script — a SimpleNamespace is enough.
from types import SimpleNamespace

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "molmetal" / "reports"
REPORT_PATH = REPORT_DIR / "generate_validation.md"
LOG_PATH = REPORT_DIR / "generate_validation.json"

# Sampling
N_SAMPLES = 100
N_STEPS = 20
N_ATOMS = 8            # matches LipmanFlowMatchingAdapter.generate() default
N_TRAIN_MOLS = 200
MAX_ATOMS = 30
BATCH_SIZE = 8
N_TRAIN_STEPS = 30
SEED = 0
METAL = "Ru"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_unique_smiles(metal: str = METAL, n: int = N_TRAIN_MOLS,
                        max_atoms: int = MAX_ATOMS) -> List[str]:
    """Load MetalCytoToxDB, filter ``metal``, drop missing IC50, dedup
    by canonical SMILES, cap to ``n`` unique molecules with ≤ ``max_atoms``
    heavy atoms.

    Uses RDKit directly to count heavy atoms; this avoids loading the full
    3D conformer pipeline for what is ultimately a velocity-field training
    run that doesn't require accurate conformers.
    """
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    flt = CytotoxFilter(
        time_threshold=24.0,
        ic50_min=0.01,
        metal_whitelist=[metal],
        require_active_field=True,
        compute_pic50=True,
        compute_active=True,
    )
    ds = MetalCytotoxDataset.from_csv(filters=flt)
    print(f"[validate] loaded {len(ds)} rows for metal={metal}")

    seen: set[str] = set()
    picks: List[str] = []
    for i in range(len(ds)):
        row = ds[i]
        smi = row["smiles"]
        if not smi:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol)
        if canon in seen:
            continue
        if mol.GetNumHeavyAtoms() > max_atoms:
            continue
        seen.add(canon)
        picks.append(canon)
        if len(picks) >= n:
            break
    print(f"[validate] collected {len(picks)} unique SMILES (≤{max_atoms} heavy atoms)")
    return picks


def smiles_to_mol(smi: str) -> Molecule:
    """Build a :class:`Molecule` from SMILES without 3D embedding.

    The velocity field is trained on the *positions* but the CFM loss with
    random noise (Lipman 2023) is dominated by the atom-type signal at the
    scale we operate on here; using placeholder coords (heavy-atom identity
    matrix) keeps the training loop fast and removes dependence on RDKit
    3D embedding for every molecule.
    """
    from rdkit import Chem
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"RDKit failed to parse: {smi}")
    n = mol.GetNumAtoms()
    # Placeholder coords: identity-like, small variance, unit-scale.
    # The CFM path samples x_0 ~ N(0, I) so the data scale is unimportant;
    # what matters is that atom_types match the heavy-atom list.
    coords = torch.zeros(n, 3, dtype=torch.float32)
    coords += torch.randn(n, 3, dtype=torch.float32) * 0.1
    atom_types = torch.tensor(
        [a.GetAtomicNum() for a in mol.GetAtoms()], dtype=torch.long
    )
    return Molecule(
        coords=coords,
        atom_types=atom_types,
        bonds=torch.zeros(2, 0, dtype=torch.long),
        bond_types=torch.zeros(0, dtype=torch.long),
        formal_charges=torch.zeros(n, dtype=torch.long),
        smiles=smi,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def mol_to_rdkit(m: Molecule):
    """Return a sanitized molecule from declared SMILES or decoded bonds.

    Invalid SMILES do not fall back to a fabricated atom chain. Undecoded
    multi-atom samples (no SMILES or bonds) cannot support descriptors yet.
    """
    from rdkit import Chem

    try:
        if m.smiles:
            mol = Chem.MolFromSmiles(m.smiles)
        else:
            if m.n_atoms == 0 or (m.n_atoms > 1 and m.n_bonds == 0):
                return None
            mol = m.to_rdkit()
        if mol is None or mol.GetNumAtoms() == 0:
            return None
        if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
            return None
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


def evaluate_set(mols: List[Molecule]) -> dict:
    """Summarize finite descriptor rows; fractions use all input candidates.

    Mean and population std (ddof=0) use valid molecules only. Invalid or
    undecoded candidates count as failures in every range fraction. MW
    ranges are descriptive and never filter candidates out of other metrics.
    """
    from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors

    rows = []
    mw_flags = []
    for molecule in mols:
        mol = mol_to_rdkit(molecule)
        row = None
        if mol is not None:
            try:
                values = np.asarray([
                    Descriptors.qed(mol), Descriptors.MolWt(mol),
                    rdMolDescriptors.CalcTPSA(mol), Crippen.MolLogP(mol),
                    rdMolDescriptors.CalcNumRotatableBonds(mol),
                ], dtype=np.float64)
                if np.isfinite(values).all():
                    row = values
            except Exception:
                pass
        # Append only complete rows: a late descriptor failure must not
        # leave lists with different lengths or inflate n_valid.
        if row is None:
            mw_flags.append(None)
        else:
            rows.append(row)
            mw_flags.append(bool(300.0 <= row[1] <= 700.0))

    matrix = np.asarray(rows, dtype=np.float64).reshape(-1, 5)
    result = {
        "n_total": len(mols), "n_valid": len(rows),
        "n_invalid": len(mols) - len(rows),
        "fraction_denominator": "all_input_candidates",
        "std_ddof": 0,
        "mw_range_300_700_flags": mw_flags,
    }
    for i, name in enumerate(("qed", "mw", "tpsa", "logp", "rotb")):
        column = matrix[:, i]
        result[f"{name}_mean"] = float(column.mean()) if len(rows) else float("nan")
        result[f"{name}_std"] = float(column.std()) if len(rows) else float("nan")
    qed, mw, tpsa, logp, rotb = matrix.T
    result["qed_median"] = float(np.median(qed)) if len(rows) else float("nan")
    masks = {
        "frac_druglike_qed05": qed >= 0.5,
        "mw_frac_druglike": (mw >= 200.0) & (mw <= 500.0),
        "mw_frac_300_700": (mw >= 300.0) & (mw <= 700.0),
        "tpsa_frac_60_150": (tpsa >= 60.0) & (tpsa <= 150.0),
        "logp_frac_2_5": (logp >= 2.0) & (logp <= 5.0),
        "logp_frac_gt_5": logp > 5.0,
        "rotb_frac_lt_10": rotb < 10,
        # Retained for readers of older JSON reports; headline uses <10.
        "rotb_frac_le_12": rotb <= 12,
    }
    for name, mask in masks.items():
        result[name] = float(mask.sum() / len(mols)) if mols else 0.0
    return result


def atom_type_histogram(mols: List[Molecule], bins=(1, 21)) -> Counter:
    """Return a Counter of atomic-number bins across ``mols``.

    Bin 0  → Z > 20 (rare/heavy)
    Bin i  → Z = i for i in [bins[0], bins[1))
    Z = 0 (padding) is reported separately as ``"z0_padding"``.
    """
    counter: Counter = Counter()
    z0 = 0
    for m in mols:
        for z in m.atom_types.tolist():
            z = int(z)
            if z == 0:
                z0 += 1
            elif z < bins[0]:
                counter[f"Z<{bins[0]}"] += 1
            elif z >= bins[1]:
                counter["Z>20"] += 1
            else:
                counter[f"Z={z}"] += 1
    counter["z0_padding"] = z0
    return counter


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(adapter: LipmanFlowMatchingAdapter, mols: List[Molecule],
          n_steps: int, batch_size: int, seed: int) -> list[dict]:
    """Run ``n_steps`` CFM + atom-CE train steps and log per-step losses."""
    torch.manual_seed(seed)
    losses = []
    for step in range(n_steps):
        # Sample a random batch (with replacement) from the train pool.
        idx = np.random.default_rng(seed + step).choice(
            len(mols), size=min(batch_size, len(mols)), replace=False
        )
        batch = [mols[int(i)] for i in idx]
        t0 = time.time()
        loss = adapter.train_step(pocket=None, mols=batch)
        dt = time.time() - t0
        ll = adapter.last_losses
        record = {
            "step": step,
            "loss": float(loss),
            "cfm": float(ll["cfm"]),
            "atom": float(ll["atom"]),
            "dt": dt,
        }
        losses.append(record)
        if step < 5 or step == n_steps - 1 or step % 10 == 0:
            print(f"[validate] step {step:3d}: total={loss:.4f}  "
                  f"cfm={ll['cfm']:.4f}  atom={ll['atom']:.4f}  "
                  f"({dt*1000:.1f} ms)")
    return losses


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def render_report(
    losses: list[dict],
    gen_metrics: dict,
    ref_metrics: dict,
    gen_hist: Counter,
    ref_hist: Counter,
    config: dict,
    device_info: dict,
) -> str:
    """Render a Markdown report summarising the validation run."""
    lines = []
    lines.append("# Generate() Validation — Joint Atom-Type + Coord FM")
    lines.append("")
    lines.append("End-to-end smoke test of the new joint atom-type + coordinate")
    lines.append("Flow Matching ``LipmanFlowMatchingAdapter.generate()``.")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- Adapter: `LipmanFlowMatchingAdapter` (hidden_dim={config['hidden_dim']}, "
                 f"n_layers={config['n_layers']}, lr={config['lr']}, "
                 f"atom_loss_weight={config['atom_loss_weight']})")
    lines.append(f"- Device: `{config['device']}`  "
                 f"(ROCm active: {device_info.get('roc_active')}, "
                 f"HIP: {device_info.get('hip_version')}, "
                 f"name: {device_info.get('device_0_name')})")
    lines.append(f"- Train molecules: {config['n_train_mols']} unique Ru SMILES "
                 f"with ≤{config['max_atoms']} heavy atoms")
    lines.append(f"- Training: {config['n_train_steps']} steps × batch_size={config['batch_size']}")
    lines.append(f"- Generation: n_samples={config['n_samples']}, n_steps={config['n_steps']}, "
                 f"n_atoms={config['n_atoms']}")
    lines.append("")

    lines.append("## Training Loss Curve")
    lines.append("")
    lines.append("| Step | Total | CFM | Atom-CE | ms/step |")
    lines.append("|------|-------|-----|---------|---------|")
    for r in losses:
        lines.append(f"| {r['step']} | {r['loss']:.4f} | {r['cfm']:.4f} | "
                     f"{r['atom']:.4f} | {r['dt']*1000:.1f} |")
    lines.append("")
    if losses:
        first, last = losses[0], losses[-1]
        lines.append(f"- Initial loss: total={first['loss']:.4f} "
                     f"(cfm={first['cfm']:.4f}, atom={first['atom']:.4f})")
        lines.append(f"- Final loss:   total={last['loss']:.4f} "
                     f"(cfm={last['cfm']:.4f}, atom={last['atom']:.4f})")
        lines.append(f"- Loss drop (total / cfm / atom): "
                     f"{first['loss']/max(last['loss'], 1e-12):.2f}x / "
                     f"{first['cfm']/max(last['cfm'], 1e-12):.2f}x / "
                     f"{first['atom']/max(last['atom'], 1e-12):.2f}x")
    lines.append("")

    lines.append("## Generated Molecule Quality")
    lines.append("")
    lines.append("| Metric | Generated | Training Reference |")
    lines.append("|--------|-----------|--------------------|")
    lines.append(f"| Valid (RDKit-parseable) | {gen_metrics['n_valid']} / {config['n_samples']} "
                 f"| {ref_metrics['n_valid']} / {config['n_train_mols']} |")
    lines.append(f"| QED mean | {gen_metrics['qed_mean']:.4f} | {ref_metrics['qed_mean']:.4f} |")
    lines.append(f"| QED std  | {gen_metrics['qed_std']:.4f} | {ref_metrics['qed_std']:.4f} |")
    lines.append(f"| QED median | {gen_metrics['qed_median']:.4f} | {ref_metrics['qed_median']:.4f} |")
    lines.append(f"| Fraction drug-like (QED ≥ 0.5) | {gen_metrics['frac_druglike_qed05']:.3f} "
                 f"| {ref_metrics['frac_druglike_qed05']:.3f} |")
    lines.append(f"| MolWt mean | {gen_metrics['mw_mean']:.1f} | {ref_metrics['mw_mean']:.1f} |")
    lines.append(f"| MolWt in [200, 500] | {gen_metrics['mw_frac_druglike']:.3f} "
                 f"| {ref_metrics['mw_frac_druglike']:.3f} |")
    for name, label in (("logp", "logP"), ("tpsa", "TPSA (Å²)"), ("rotb", "RotB")):
        lines.append(
            f"| {label} mean ± std | {gen_metrics[name + '_mean']:.2f} ± "
            f"{gen_metrics[name + '_std']:.2f} | {ref_metrics[name + '_mean']:.2f} ± "
            f"{ref_metrics[name + '_std']:.2f} |"
        )
    for key, label in (
        ("logp_frac_2_5", "logP in [2, 5]"),
        ("logp_frac_gt_5", "logP >5 (potential liability flag)"),
        ("tpsa_frac_60_150", "TPSA in [60, 150] Å²"),
        ("rotb_frac_lt_10", "RotB <10"),
        ("mw_frac_300_700", "MW in [300, 700] Da (descriptive)"),
    ):
        lines.append(f"| {label} | {gen_metrics[key]:.3f} | {ref_metrics[key]:.3f} |")
    lines.append("")
    lines.append("Descriptor means and population std (ddof=0) use valid candidates. "
                 "Range fractions use all input candidates; invalid samples never pass. "
                 "MW flags are descriptive: high-MW candidates remain in the evaluation. "
                 "For the IV anticancer route, TPSA >75 Å² is acceptable within the "
                 "stated range. These descriptors do not measure hERG or clinical safety.")
    lines.append("")

    lines.append("## Atom-Type Distribution")
    lines.append("")
    lines.append("| Bin | Generated | Training Reference |")
    lines.append("|-----|-----------|--------------------|")
    all_bins = sorted(set(gen_hist.keys()) | set(ref_hist.keys()),
                      key=lambda k: (
                          0 if k == "z0_padding" else
                          -1 if k == "Z>20" else
                          -2 if k.startswith("Z<") else
                          int(k.split("=")[1])
                      ))
    for k in all_bins:
        lines.append(f"| {k} | {gen_hist.get(k, 0)} | {ref_hist.get(k, 0)} |")
    lines.append("")
    if gen_hist.get("z0_padding", 0) == 0:
        lines.append("- No Z=0 padding atoms in generated molecules "
                     "(the `atom_logits[..., 0] = -inf` mask is working).")
    else:
        lines.append(f"- **{gen_hist['z0_padding']} Z=0 padding atoms leaked into generated samples** "
                     f"— investigate the categorical mask in `generate()`.")
    lines.append("")

    # Qualitative judgement
    lines.append("## Verdict")
    lines.append("")
    if losses:
        ratio = losses[0]['loss'] / max(losses[-1]['loss'], 1e-12)
        if ratio >= 2.0:
            lines.append(f"- Training **converged** (total loss dropped {ratio:.2f}x over "
                         f"{config['n_train_steps']} steps).")
        else:
            lines.append(f"- Training **inconclusive** (total loss only dropped {ratio:.2f}x). "
                         "Try a higher lr or more steps.")
    lines.append(f"- Generated QED mean = {gen_metrics['qed_mean']:.3f}; "
                 f"reference = {ref_metrics['qed_mean']:.3f}. "
                 "With only 30 train steps the model is far from converged; "
                 "the distribution should sharpen as we increase the budget.")
    lines.append(f"- Atom-type histogram coverage = "
                 f"{sum(1 for k in gen_hist if k.startswith('Z=') and k != 'Z=0')} unique elements. "
                 "We expect this to grow with more training.")
    lines.append("")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    print("=" * 70)
    print("LipmanFlowMatchingAdapter.generate() — end-to-end validation")
    print("=" * 70)

    # ---- 1. Load data -----------------------------------------------------
    smiles = load_unique_smiles(metal=METAL, n=N_TRAIN_MOLS, max_atoms=MAX_ATOMS)
    if not smiles:
        print("[validate] FATAL: no molecules loaded")
        return 1
    train_mols = [smiles_to_mol(s) for s in smiles]
    print(f"[validate] built {len(train_mols)} training molecules")

    # ---- 2. Adapter setup -------------------------------------------------
    adapter = LipmanFlowMatchingAdapter(
        hidden_dim=64,
        n_layers=2,
        lr=5e-3,
        atom_loss_weight=0.1,
    )
    adapter.setup()
    device_info = adapter.device_info
    print(f"[validate] device = {adapter.device}, "
          f"roc_active = {device_info.get('roc_active')}")

    # ---- 3. Train --------------------------------------------------------
    print(f"[validate] training for {N_TRAIN_STEPS} steps "
          f"(batch={BATCH_SIZE}, lr=5e-3) ...")
    losses = train(adapter, train_mols, N_TRAIN_STEPS, BATCH_SIZE, SEED)

    # ---- 4. Generate ------------------------------------------------------
    print(f"[validate] generating {N_SAMPLES} molecules "
          f"({N_STEPS} ODE steps, {N_ATOMS} atoms each) ...")
    # GenerationConfig is frozen and has no `n_atoms` field; the adapter
    # uses ``getattr(config, "n_atoms", 8)``.  A SimpleNamespace with the
    # same attributes is enough for the duck-typed contract.
    gen_config = SimpleNamespace(
        n_samples=N_SAMPLES,
        n_steps=N_STEPS,
        temperature=1.0,
        seed=SEED,
        conditioning={},
        n_atoms=N_ATOMS,
    )
    t0 = time.time()
    gen_mols = adapter.generate(pocket=None, config=gen_config)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gen_dt = time.time() - t0
    print(f"[validate] generate() produced {len(gen_mols)} molecules in {gen_dt:.2f}s")

    # ---- 5. Reference metrics on training set ----------------------------
    print("[validate] computing training-reference metrics ...")
    ref_metrics = evaluate_set(train_mols)

    # ---- 6. Generated metrics --------------------------------------------
    print("[validate] computing generated-set metrics ...")
    gen_metrics = evaluate_set(gen_mols)

    # ---- 7. Atom-type histograms -----------------------------------------
    gen_hist = atom_type_histogram(gen_mols)
    ref_hist = atom_type_histogram(train_mols)

    # ---- 8. Report -------------------------------------------------------
    config = {
        "hidden_dim": 64,
        "n_layers": 2,
        "lr": 5e-3,
        "atom_loss_weight": 0.1,
        "n_train_mols": len(train_mols),
        "max_atoms": MAX_ATOMS,
        "n_train_steps": N_TRAIN_STEPS,
        "batch_size": BATCH_SIZE,
        "n_samples": N_SAMPLES,
        "n_steps": N_STEPS,
        "n_atoms": N_ATOMS,
        "device": str(adapter.device),
        "metal": METAL,
    }
    md = render_report(losses, gen_metrics, ref_metrics, gen_hist,
                       ref_hist, config, device_info)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(md)
    print(f"[validate] wrote Markdown report to {REPORT_PATH}")

    json_payload = {
        "config": config,
        "device_info": {k: device_info.get(k) for k in
                        ("roc_active", "hip_version", "device_0_name", "device_count")},
        "losses": losses,
        "gen_metrics": gen_metrics,
        "ref_metrics": ref_metrics,
        "gen_hist": dict(gen_hist),
        "ref_hist": dict(ref_hist),
    }
    LOG_PATH.write_text(json.dumps(json_payload, indent=2))
    print(f"[validate] wrote JSON log to {LOG_PATH}")

    # ---- 9. Print summary ------------------------------------------------
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Device              : {adapter.device}  "
          f"(roc_active={device_info.get('roc_active')})")
    print(f"  Train loss drop     : {losses[0]['loss']:.4f} -> {losses[-1]['loss']:.4f}  "
          f"({losses[0]['loss']/max(losses[-1]['loss'], 1e-12):.2f}x)")
    print(f"  Generated QED mean  : {gen_metrics['qed_mean']:.3f}  "
          f"(reference: {ref_metrics['qed_mean']:.3f})")
    print(f"  Generated MolWt mean: {gen_metrics['mw_mean']:.1f}  "
          f"(reference: {ref_metrics['mw_mean']:.1f})")
    print(f"  Drug-like fraction  : {gen_metrics['frac_druglike_qed05']:.3f}  "
          f"(reference: {ref_metrics['frac_druglike_qed05']:.3f})")
    print(f"  Z=0 padding atoms   : {gen_hist.get('z0_padding', 0)}  "
          f"(should be 0 → masking is working)")
    print(f"  Unique elements     : {sum(1 for k in gen_hist if k.startswith('Z=') and k != 'Z=0')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
