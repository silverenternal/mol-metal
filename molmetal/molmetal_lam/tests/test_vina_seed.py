"""Seed propagation must work for both native bindings and CLI engines."""
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from molmetal.ports import DockingConfig
from molmetal_lam.sbdd_env.vina_adapter import VinaDockingAdapter


def _adapter(tmp_path):
    adapter = object.__new__(VinaDockingAdapter)
    adapter._cpu = 1
    adapter._sf_name = "vina"
    adapter._tmpdir = tmp_path
    adapter._engine = "quickvina2"
    adapter._engine_binary = "/explicit/quickvina2"
    return adapter


def test_python_binding_receives_requested_seed(monkeypatch, tmp_path):
    factory = MagicMock()
    factory.return_value.energies.return_value = np.array([[-2.5, 0, 0, 0, 0]])
    factory.return_value.poses.return_value = "MODEL 1\nENDMDL\n"
    monkeypatch.setitem(sys.modules, "vina", SimpleNamespace(Vina=factory))
    _adapter(tmp_path)._dock_python_binding(
        tmp_path / "receptor.pdbqt", tmp_path / "ligand.pdbqt",
        np.zeros(3), np.full(3, 20), DockingConfig(seed=137),
    )
    assert factory.call_args.kwargs["seed"] == 137


def test_subprocess_receives_requested_seed(monkeypatch, tmp_path):
    from molmetal_lam.sbdd_env import vina_adapter
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        Path(command[command.index("--out") + 1]).write_text(
            "MODEL 1\nREMARK VINA RESULT: -2.6 0 0\nENDMDL\n"
        )
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(vina_adapter.subprocess, "run", run)
    energies, _ = _adapter(tmp_path)._dock_cli_binary(
        tmp_path / "receptor.pdbqt", tmp_path / "ligand.pdbqt",
        np.zeros(3), np.full(3, 20), DockingConfig(seed=137),
    )
    assert commands[0][commands[0].index("--seed") + 1] == "137"
    assert energies[0, 0] == -2.6


def test_zero_seed_is_deterministic_for_real_native_vina():
    import vina
    from molmetal_lam.sbdd_env.vina_adapter import native_vina_seed
    effective=native_vina_seed(0)
    assert effective != 0
    assert vina.Vina(cpu=1,seed=effective,verbosity=0).info()['seed']==effective
    assert vina.Vina(cpu=1,seed=effective,verbosity=0).info()['seed']==effective
    assert native_vina_seed(42)==42
    assert 0<native_vina_seed(2**32-1)<2**31
