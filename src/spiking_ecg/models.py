"""Los modelos reciben [batch, 140]; solo la SNN recorre el tiempo."""

import torch
from torch import nn
import snntorch as snn
from snntorch import surrogate


class MLP(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(140, hidden), nn.ReLU(), nn.Linear(hidden, 2))

    def forward(self, x):
        return self.net(x)


class TemporalSNN(nn.Module):
    def __init__(self, hidden=32, beta=0.9, readout_bins=7):
        super().__init__()
        if readout_bins < 1 or 140 % readout_bins:
            raise ValueError("readout_bins debe ser un divisor positivo de 140")
        self.readout_bins = readout_bins
        self.input = nn.Linear(1, hidden)
        self.lif = snn.Leaky(
            beta=beta, threshold=1.0, spike_grad=surrogate.fast_sigmoid(slope=25),
            reset_mechanism="subtract", reset_delay=True, init_hidden=False,
        )
        self.readout = nn.Linear(hidden * readout_bins, 2)

    def forward(self, x, record=False):
        # Estado explícito nuevo en CADA forward: no hay memoria entre latidos.
        mem = x.new_zeros(x.shape[0], self.input.out_features)
        spikes, membranes = [], []
        for t in range(x.shape[1]):
            current = self.input(x[:, t:t + 1])  # amplitud real, no spike de entrada
            spk, mem = self.lif(current, mem)
            spikes.append(spk)
            if record:
                membranes.append(mem.detach())
        spikes = torch.stack(spikes)  # [tiempo, batch, neuronas]
        # Tasas por ventanas: preservan posición aproximada dentro del latido.
        rates = spikes.reshape(self.readout_bins, 140 // self.readout_bins,
                               x.shape[0], -1).mean(dim=1)
        logits = self.readout(rates.permute(1, 0, 2).flatten(1))
        if record:
            return logits, {"spikes": spikes.detach(), "membrane": torch.stack(membranes)}
        return logits


def make_model(name, hidden=32, beta=0.9, readout_bins=7):
    if name == "mlp":
        return MLP(hidden)
    if name == "snn":
        return TemporalSNN(hidden, beta, readout_bins)
    raise ValueError(f"Modelo desconocido: {name}")
