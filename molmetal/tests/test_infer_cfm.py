# -*- coding: utf-8 -*-
"""Tests for infer_cfm.py standalone inference CLI."""

import json
import os
import tempfile
from pathlib import Path

import pytest

from molmetal.scripts.infer_cfm import (
    parse_args,
    make_dirs,
    load_smiles,
    infer_batch,
    DEFAULTS,
)


class TestInferCFM:
    """4+ tests for inference CLI."""

    def test_parse_args_defaults(self):
        """parse_args with no CLI overrides uses defaults."""
        import sys

        sys.argv = ["infer_cfm.py"]
        args = parse_args([])
        assert args.input_file == DEFAULTS["input_file"]
        assert args.num_cycle == DEFAULTS["num_cycle"]
        assert args.batch_size == DEFAULTS["batch_size"]
        print(f"test_parse_args_defaults PASS: num_cycle={args.num_cycle}")

    def test_parse_args_with_overrides(self):
        """parse_args accepts custom --num-cycle, --batch-size, --device."""
        args = parse_args(
            ["--num-cycle", "3", "--batch-size", "16", "--device", "cuda"]
        )
        assert args.num_cycle == 3
        assert args.batch_size == 16
        assert args.device == "cuda"
        print(
            f"test_parse_args_with_overrides PASS: num_cycle={args.num_cycle} device={args.device}"
        )

    def test_make_dirs_idempotent(self):
        """make_dirs with exist_ok=True does not raise on existing dir."""
        with tempfile.TemporaryDirectory() as tmp:
            make_dirs(tmp)  # second call should not raise
            make_dirs(tmp)
            assert os.path.isdir(tmp)
        print(f"test_make_dirs_idempotent PASS")

    def test_load_smiles_basic(self):
        """load_smiles reads SMILES file, skips blanks and comments."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".smi", delete=False) as f:
            f.write("# comment line\n")
            f.write("CCO\n")
            f.write("\n")
            f.write("c1ccccc1\n")
            path = f.name
        try:
            smiles = load_smiles(path)
            assert smiles == ["CCO", "c1ccccc1"]
        finally:
            os.unlink(path)
        print(f"test_load_smiles_basic PASS: {len(smiles)} SMILES loaded")

    def test_infer_batch_with_stub_model(self):
        """infer_batch returns confidence + predictions dict."""

        class StubModel:
            def infer(self, smiles, num_cycle=1):
                return {
                    "confidence": 0.85,
                    "predictions": [{"smiles": s} for s in smiles],
                    "num_cycle": num_cycle,
                }

        import argparse

        args = parse_args([])
        out = infer_batch(StubModel(), ["CCO", "c1ccccc1"], args)
        assert "confidence" in out
        assert 0.0 <= out["confidence"] <= 1.0
        assert len(out["predictions"]) == 2
        print(
            f"test_infer_batch_with_stub_model PASS: confidence={out['confidence']:.3f}"
        )
