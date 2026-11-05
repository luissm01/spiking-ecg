"""Tests sin red; el entrenamiento con ECG real se ejecuta por separado."""

import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from spiking_ecg.data import load_official, prepare_data
from spiking_ecg.experiment import evaluate, metrics, seed_all, train_one
from spiking_ecg.models import make_model


@pytest.fixture
def splits():
    rng = np.random.default_rng(9)
    x = rng.normal(size=(50, 140)).astype(np.float32)
    y = np.tile([0, 1], 25)
    xt = rng.normal(size=(20, 140)).astype(np.float32)
    return prepare_data(((x, y), (xt, np.tile([0, 1], 10))))


@pytest.mark.parametrize("name", ["mlp", "snn"])
def test_shapes_and_backward(name):
    seed_all(42)
    model = make_model(name, hidden=8)
    x = torch.randn(3, 140, requires_grad=True)
    logits = model(x)
    assert logits.shape == (3, 2)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 0])).backward()
    first = model.input.weight if name == "snn" else model.net[0].weight
    assert first.grad is not None and torch.isfinite(first.grad).all()
    assert first.grad.abs().sum() > 0
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert x.grad[:, 0].abs().sum() > 0
    assert x.grad[:, -1].abs().sum() > 0


def test_state_reset_and_trace():
    seed_all(42)
    model = make_model("snn", hidden=8)
    x = torch.randn(2, 140)
    with torch.no_grad():
        expected, trace = model(x, record=True)
        model(torch.full((5, 140), 10.0))  # Otro batch/tamaño/estado previo.
        actual, repeated = model(x, record=True)
        single = model(x[:1])
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    torch.testing.assert_close(single, expected[:1])
    for key in ("spikes", "membrane"):
        assert trace[key].shape == (140, 2, 8)
        torch.testing.assert_close(trace[key], repeated[key], rtol=0, atol=0)
    assert set(trace["spikes"].unique().tolist()) <= {0.0, 1.0}
    # También puede entrenar dos batches sin reutilizar el grafo anterior.
    for _ in range(2):
        model.zero_grad()
        model(x).sum().backward()


def test_split_and_no_leakage(splits):
    data, meta = splits
    assert len(data["train"][0]) == 40 and len(data["val"][0]) == 10
    assert not set(meta["train_indices"]) & set(meta["val_indices"])
    assert sorted(meta["train_indices"] + meta["val_indices"]) == list(range(50))
    assert data["train"][0].mean().abs() < 1e-6
    assert abs(data["train"][0].numpy().std() - 1) < 1e-6
    assert meta["counts"]["val"] == [5, 5]
    rng = np.random.default_rng(2)
    x = rng.normal(size=(50, 140)).astype(np.float32)
    y = np.tile([0, 1], 25)
    _, a = prepare_data(((x, y), (np.zeros((20, 140), np.float32), np.tile([0, 1], 10))))
    _, b = prepare_data(((x, y), (np.full((20, 140), 1e6, np.float32), np.tile([0, 1], 10))))
    assert a == b  # La distribución del test no afecta al preprocesado.


def test_metrics_positive_resto():
    result = metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert result["confusion_matrix"] == [[1, 1], [0, 2]]
    assert result["recall"] == 1.0
    assert result["precision"] == pytest.approx(2 / 3)


def test_corrupt_download_rejected(tmp_path):
    (tmp_path / "ECG5000.zip").write_bytes(b"not a dataset")
    with pytest.raises(ValueError, match="Checksum"):
        load_official(tmp_path)


@pytest.mark.parametrize("name", ["mlp", "snn"])
def test_training_reproducible(name):
    def run():
        seed_all(42)
        model = make_model(name, hidden=8)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for _ in range(2):
            optimizer.zero_grad()
            logits = model(torch.randn(4, 140))
            torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 0, 1])).backward()
            optimizer.step()
        return model.state_dict()
    first, second = run(), run()
    for key in first:
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)


@pytest.mark.parametrize("name", ["mlp", "snn"])
def test_minimal_training_and_checkpoint(name, splits, tmp_path):
    data, metadata = splits
    result = train_one(name, data, metadata, tmp_path, epochs=1, hidden=8, batch_size=16)
    assert 0 <= result["accuracy"] <= 1
    ckpt = torch.load(tmp_path / "best.pt", weights_only=True)
    restored = make_model(name, hidden=8)
    restored.load_state_dict(ckpt["state_dict"])
    actual, _ = evaluate(restored, data["test"], 16)
    assert actual == json.loads((tmp_path / "test_metrics.json").read_text())
    assert (tmp_path / "training.png").is_file()
    assert (tmp_path / "confusion.png").is_file()
    if name == "snn":
        assert (tmp_path / "activity_0.png").is_file()
        assert (tmp_path / "activity_1.npz").is_file()


def test_cli_help():
    subprocess.run([sys.executable, "-m", "spiking_ecg.experiment", "--help"], check=True,
                   capture_output=True)
