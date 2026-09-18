import numpy as np
import pytest
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from molmetal.validation.gpu_molecular_metrics import molecular_diversity


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda:0', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU unavailable'))])
def test_chunked_tanimoto_matches_rdkit(device):
    smiles = ['CCO','CCN','c1ccccc1','CCO','CNC','CC(=O)O']
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = [generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]
    expected = np.mean([DataStructs.TanimotoSimilarity(a,b) for i,a in enumerate(fps) for b in fps[i+1:]])
    report = molecular_diversity(smiles, device=device, chunk_size=2)
    assert report['device'] == device
    assert report['n_pairs'] == 15 and report['n_unique'] == 5
    assert report['mean_pairwise_tanimoto'] == pytest.approx(expected, abs=1e-7)


def test_invalid_and_singleton_have_no_fabricated_diversity():
    report = molecular_diversity(['',None,'*','CCO'], device='cpu')
    assert report['n_input'] == 4 and report['n_invalid'] == 3
    assert report['diversity'] is None
