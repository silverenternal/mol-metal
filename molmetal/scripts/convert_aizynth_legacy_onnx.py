# /// script
# requires-python = "==3.12.*"
# dependencies = ["numpy==1.26.4", "onnx==1.19.1", "onnxruntime==1.30.0", "h5py==3.16.0", "rdkit==2023.9.6"]
# ///
"""Convert the exact official legacy USPTO Dense/ELU/Softmax inference graph.

Weights are preserved; Dropout is disabled for inference. Validation compares
ONNX with the saved Keras graph's equations evaluated independently in NumPy.
This does not claim a TensorFlow runtime comparison or current-policy parity.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path

import h5py
import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
from rdkit import Chem
from rdkit.Chem import AllChem

DIRECTORY = Path("/mnt/storage/data/molmetal/aizynth_public/legacy_v3")
SOURCE_SHA256 = "49c3e54b280106fdc9412939b4feaa940ef6ea426461565679f69a1c826450b3"


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    source = DIRECTORY / "uspto_model.hdf5"
    assert digest(source) == SOURCE_SHA256, "Unsupported or corrupted source model"
    with h5py.File(source) as model:
        config = json.loads(model.attrs["model_config"])
        assert config["class_name"] == "Sequential"
        layers = config["config"]["layers"]
        assert [layer["class_name"] for layer in layers] == ["Dense", "Dropout", "Dense"]
        assert layers[0]["config"]["activation"] == "elu"
        assert layers[2]["config"]["activation"] == "softmax"
        assert layers[0]["config"]["batch_input_shape"] == [None, 2048]
        arrays = []
        for layer in (layers[0], layers[2]):
            name = layer["config"]["name"]
            arrays += [np.asarray(model[f"model_weights/{name}/{name}/{param}:0"], dtype=np.float32)
                       for param in ["kernel", "bias"]]
        keras_version = str(model.attrs["keras_version"])
    w1, b1, w2, b2 = arrays
    assert [a.shape for a in arrays] == [(2048, 512), (512,), (512, 46695), (46695,)]
    assert all(np.isfinite(a).all() for a in arrays)
    nodes = [
        helper.make_node("Gemm", ["fingerprint", "w1", "b1"], ["dense1"]),
        helper.make_node("Elu", ["dense1"], ["hidden"], alpha=1.0),
        helper.make_node("Gemm", ["hidden", "w2", "b2"], ["logits"]),
        helper.make_node("Softmax", ["logits"], ["probabilities"], axis=1),
    ]
    graph = helper.make_graph(nodes, "official_legacy_uspto_inference", [
        helper.make_tensor_value_info("fingerprint", TensorProto.FLOAT, [None, 2048])
    ], [helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, [None, 46695])],
        initializer=[numpy_helper.from_array(array, name) for array, name in zip(arrays, ["w1", "b1", "w2", "b2"])])
    converted = helper.make_model(graph, producer_name="molmetal audited legacy conversion",
                                  opset_imports=[helper.make_opsetid("", 17)], ir_version=8)
    onnx.checker.check_model(converted)
    destination = DIRECTORY / "uspto_legacy_converted.onnx"
    temporary = destination.with_suffix(".pending.onnx")
    onnx.save(converted, temporary)
    rng = np.random.default_rng(20260913)
    inputs = [np.zeros(2048, np.float32), *rng.binomial(1, .025, size=(24, 2048)).astype(np.float32)]
    smiles = ["CC(=O)Oc1ccccc1C(=O)O", "CCO", "Cn1ccnn1"]
    for smi in smiles:
        inputs.append(np.asarray(AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smi), 2, nBits=2048), dtype=np.float32))
    x = np.stack(inputs)
    hidden = x @ w1 + b1
    hidden = np.where(hidden > 0, hidden, np.expm1(np.minimum(hidden, 0)))
    logits = hidden @ w2 + b2
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    expected = probs / probs.sum(axis=1, keepdims=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(temporary), sess_options=options, providers=["CPUExecutionProvider"])
    observed = session.run(None, {"fingerprint": x})[0]
    np.testing.assert_allclose(observed, expected, rtol=2e-4, atol=2e-7)
    np.testing.assert_array_equal(observed.argmax(axis=1), expected.argmax(axis=1))
    temporary.replace(destination)
    np.savez(DIRECTORY / "uspto_legacy_weights.npz", w1=w1, b1=b1, w2=w2, b2=b2)
    np.save(DIRECTORY / "validation_inputs.npy", x, allow_pickle=False)
    np.save(DIRECTORY / "validation_onnx_probabilities.npy", observed, allow_pickle=False)
    report = {
        "status": "converted_and_equation_validated", "protocol": "official legacy v3 assets; CPU ONNX",
        "source_sha256": SOURCE_SHA256, "converted_sha256": digest(destination),
        "weights_npz_sha256": digest(DIRECTORY / "uspto_legacy_weights.npz"),
        "validation_inputs_sha256": digest(DIRECTORY / "validation_inputs.npy"),
        "validation_onnx_sha256": digest(DIRECTORY / "validation_onnx_probabilities.npy"),
        "source_keras_version": keras_version, "model_config": config,
        "parameter_sha256": [hashlib.sha256(a.tobytes()).hexdigest() for a in arrays],
        "seed": 20260913, "n_validation_inputs": len(x), "real_smiles": smiles,
        "comparison": "Independent NumPy evaluation of saved Keras inference equations; no TensorFlow runtime",
        "maximum_absolute_probability_error": float(np.abs(observed - expected).max()),
        "top1_agreement": float((observed.argmax(1) == expected.argmax(1)).mean()),
        "versions": {name: importlib.metadata.version(name) for name in ["onnx", "onnxruntime", "numpy", "h5py", "rdkit"]},
    }
    (DIRECTORY / "conversion_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "model_config"}, indent=2))


if __name__ == "__main__":
    main()
