"""El barrido no puede abrir test ni seleccionar épocas con otra métrica."""

import hashlib
import io
import json
import zipfile

import numpy as np
import pytest
import torch

from spiking_ecg import research


def test_loader_opens_train_only(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    raw = np.column_stack([np.tile([1, 2], 250), rng.normal(size=(500, 140))])
    stream = io.StringIO()
    np.savetxt(stream, raw)
    archive = tmp_path / "ECG5000.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("ECG5000_TRAIN.txt", stream.getvalue())
        z.writestr("ECG5000_TEST.txt", "El test no se puede leer")
    monkeypatch.setattr(research, "SHA256", hashlib.sha256(archive.read_bytes()).hexdigest())
    original, opened = zipfile.ZipFile.read, []

    def guarded_read(self, member, *args, **kwargs):
        assert member == "ECG5000_TRAIN.txt"
        opened.append(member)
        return original(self, member, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "read", guarded_read)
    splits, meta = research.load_train_val(tmp_path)
    assert opened == ["ECG5000_TRAIN.txt"]
    assert set(splits) == {"train", "val"}
    assert len(splits["train"][1]) == 400 and len(splits["val"][1]) == 100
    assert not set(meta["train_indices"]) & set(meta["val_indices"])
    assert abs(splits["train"][0].mean()) < 1e-6
    assert not meta["test_member_opened"]


def test_early_stopping_uses_loss_and_restores_selection(tmp_path, monkeypatch):
    research.seed_all(42)
    train = (torch.randn(4, 140), torch.tensor([0, 1, 0, 1]))
    val = (torch.randn(4, 140), torch.tensor([0, 1, 0, 1]))
    original, val_calls = research.measure, []

    def controlled(model, split, batch_size=64):
        result = original(model, split, batch_size)
        if split is val:
            losses = [0.5, 0.4, 0.40005, 0.4001]
            result["loss"] = losses[len(val_calls)]
            result["f1"] = len(val_calls) / 4  # Aumenta aunque loss empeora.
            val_calls.append(result)
        return result

    monkeypatch.setattr(research, "measure", controlled)
    summary, history = research.train_run(
        {"train": train, "val": val}, {}, tmp_path / "run", beta=0.9, seed=42,
        max_epochs=6, min_epochs=2, patience=2,
    )
    assert summary["epochs_run"] == 4 and summary["best_epoch"] == 2
    assert summary["stop_reason"] == "patience"
    checkpoint = torch.load(tmp_path / "run/best.pt", weights_only=True)
    assert checkpoint["epoch"] == 2
    assert checkpoint["validation"]["loss"] == 0.4
    assert not summary["test_evaluated"]
    assert all("test" not in row for row in history)
    assert not list((tmp_path / "run").glob("test*"))
    research.export_results([(summary, history)], tmp_path / "report")
    assert (tmp_path / "report/comparison.csv").is_file()
    assert (tmp_path / "report/activity.png").is_file()
    assert json.loads((tmp_path / "report/aggregate.json").read_text())[0]["n"] == 1


def test_sweep_rejects_test_split(tmp_path):
    with pytest.raises(ValueError, match="sin test"):
        research.train_run({"train": None, "val": None, "test": None}, {}, tmp_path,
                           beta=0.9, seed=42)


def test_firing_statistics_count_silence():
    class SilentModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input = torch.nn.Linear(1, 32)

        def forward(self, x, record=False):
            # Una neurona dispara siempre; las otras 31 no disparan nunca.
            spikes = torch.zeros(140, len(x), 32)
            spikes[:, :, 0] = 1
            return torch.zeros(len(x), 2), {"spikes": spikes}

    stats = research.measure(SilentModel(), (torch.zeros(3, 140), torch.tensor([0, 1, 0])))
    assert stats["silent_neurons"] == 31
    assert stats["firing_rate"] == 1 / 32
    assert stats["silent_sequence_neuron_fraction"] == 31 / 32
    assert stats["neuron_rates"] == [1] + [0] * 31
