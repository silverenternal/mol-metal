"""Smoke test for facebook_fm_wrapper — 8 dummy SMILES, 100-step loss decrease.

Imports the upstream facebookresearch/flow_matching API via the wrapper
and confirms that the CFM loss decreases monotonically (median) over
100 training steps.

Run:
    /home/hugo/.local/bin/uv run --no-sync python molmetal/adapters/facebook_fm_smoke.py
"""
import sys, os, statistics, math, time
from dataclasses import dataclass

sys.path.insert(0, '/home/hugo/codes/try_triton_on_rocm')

from rdkit import Chem
from rdkit.Chem import AllChem
import torch

from molmetal.adapters.facebook_fm_wrapper import FacebookFMWrapper


SMILES_8 = [
    'CCO', 'CCN', 'CCC', 'c1ccccc1',            # 4 small organics
    'CC(=O)O', 'CC(N)C', 'c1ccncc1', 'CCS',     # 4 heteroatom-bearing
]


@dataclass
class _Mol:
    coords: torch.Tensor
    atom_types: torch.Tensor
    bonds: torch.Tensor


def smiles_to_mol(s: str):
    m = Chem.MolFromSmiles(s)
    m = Chem.AddHs(m)
    AllChem.EmbedMolecule(m, randomSeed=42)
    coords = torch.tensor(m.GetConformer().GetPositions(), dtype=torch.float32)
    z = torch.tensor([a.GetAtomicNum() for a in m.GetAtoms()], dtype=torch.long)
    return _Mol(coords=coords, atom_types=z, bonds=torch.zeros(0))


def main() -> int:
    mols = [smiles_to_mol(s) for s in SMILES_8]
    print(f'[smoke] parsed {len(mols)} SMILES -> n_atoms = {[m.coords.shape[0] for m in mols]}')

    adapter = FacebookFMWrapper(hidden_dim=32, lr=1e-3, n_atoms=8, device='cpu')
    adapter.setup(device='cpu')
    print(f'[smoke] {adapter.name} setup ok | device = {adapter.device}')

    losses = []
    t0 = time.time()
    for step in range(100):
        loss = adapter.train_step(mols, atom_loss_weight=0.1)
        losses.append(loss)
        if step < 5 or step % 20 == 19 or step == 99:
            print(f'[smoke] step {step:3d}  loss = {loss:.6f}  cfm = {adapter.last_losses["cfm"]:.6f}  atom = {adapter.last_losses["atom"]:.6f}')
    wall = time.time() - t0
    print(f'[smoke] 100 steps in {wall:.2f}s  ({wall/100*1000:.1f} ms/step)')

    first_quartile = statistics.median(losses[:25])
    last_quartile = statistics.median(losses[-25:])
    print(f'[smoke] median loss first 25 steps = {first_quartile:.4f}')
    print(f'[smoke] median loss last  25 steps = {last_quartile:.4f}')

    decreasing = last_quartile < first_quartile
    print(f'[smoke] loss decreased? {decreasing}')

    # ---- inference sanity: integrate the ODE -----------------------
    @dataclass
    class Cfg:
        n_samples: int = 4
        n_steps: int = 10
        seed: int = 7
        n_atoms: int = 8
    samples = adapter.generate(pocket=None, config=Cfg())
    print(f'[smoke] generate() returned {len(samples)} samples; first sample coords shape = {samples[0].coords.shape}')

    if decreasing and not math.isnan(last_quartile):
        print('[smoke] VERDICT = PASS  (loss decreased over 100 steps)')
        return 0
    print('[smoke] VERDICT = FAIL')
    return 1


if __name__ == '__main__':
    sys.exit(main())