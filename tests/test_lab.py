"""Contrasta el JavaScript real con TemporalSNN; no entrena ni cambia resultados.

Node es opcional para el experimento Python. Si está disponible, pytest comprueba
forward, reset retardado, BPTT manual y SGD del laboratorio sin dependencias npm.
"""

import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import torch

from spiking_ecg.models import TemporalSNN


@pytest.fixture(scope="module")
def lab():
    if not shutil.which("node"):
        pytest.skip("El laboratorio HTML necesita Node para el contraste numérico")
    output = subprocess.run(
        ["node", str(Path(__file__).with_name("lab_snapshot.cjs"))],
        check=True, capture_output=True, text=True,
    )
    return json.loads(output.stdout)


@pytest.mark.parametrize("case", range(6))
def test_manual_lif_and_gradients_match_snntorch(lab, case):
    record = lab["cases"][case]
    weights, expected, gradient = record["before"], record["r"], record["g"]
    beta = (0.5, 0.9, 0.98)[case // 2]
    model = TemporalSNN(beta=beta)
    with torch.no_grad():
        model.input.weight.copy_(torch.tensor(weights["w"]).reshape(32, 1))
        model.input.bias.copy_(torch.tensor(weights["b"]))
        model.readout.weight.copy_(torch.tensor(weights["v"]))
        model.readout.bias.copy_(torch.tensor(weights["c"]))
    logits, trace = model(torch.tensor([expected["x"]]), record=True)
    loss = torch.nn.functional.cross_entropy(logits, torch.tensor([case % 2]))
    loss.backward()
    # JS usa float64; el experimento Python usa float32.
    np.testing.assert_allclose(trace["membrane"][:, 0].numpy().T, expected["mem"],
                               atol=2e-6, rtol=2e-5)
    np.testing.assert_array_equal(trace["spikes"][:, 0].numpy().T, expected["spk"])
    np.testing.assert_allclose(logits.detach().numpy()[0], expected["logits"], atol=2e-7)
    assert loss.item() == pytest.approx(expected["loss"], abs=2e-7)
    for parameter, key in ((model.input.weight, "gw"), (model.input.bias, "gb"),
                           (model.readout.weight, "gv"), (model.readout.bias, "gc")):
        np.testing.assert_allclose(parameter.grad.numpy().reshape(np.shape(gradient[key])),
                                   gradient[key], atol=1e-5, rtol=3e-4)
    assert record["r"] == record["repeated"]  # Nuevo estado en cada forward.
    for key, grad_key in (("w", "gw"), ("b", "gb"), ("v", "gv"), ("c", "gc")):
        np.testing.assert_allclose(record["after"][key],
                                   np.array(weights[key]) - 0.08 * np.array(gradient[grad_key]),
                                   atol=1e-14)


def test_strict_threshold_and_delayed_reset(lab):
    # I=1 constante: en U=1 no hay spike; en U=1.9 sí, sin reset inmediato.
    r = lab["boundary"]["r"]
    np.testing.assert_allclose(r["mem"][0][:3], [1, 1.9, 1.71])
    assert r["spk"][0][:3] == [0, 1, 1]
