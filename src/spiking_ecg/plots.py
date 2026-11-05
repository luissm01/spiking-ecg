"""Gráficos estáticos reproducibles, sin servidor ni notebook obligatorio."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def signals(x, y, path):
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    for label, ax in enumerate(axes):
        for signal in x[y == label][:5]:
            ax.plot(signal.numpy(), alpha=0.6)
        ax.set(ylabel="Amplitud normalizada", title=("Normal" if label == 0 else "Resto"))
    axes[-1].set_xlabel("Punto temporal (no milisegundos)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def training(history, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = [r["epoch"] for r in history]
    for split in ("train", "val"):
        axes[0].plot(epochs, [r[f"{split}_loss"] for r in history], label=split)
        axes[1].plot(epochs, [r[f"{split}_accuracy"] for r in history], label=split)
    for ax, ylabel in zip(axes, ("Cross entropy", "Accuracy")):
        ax.set(xlabel="Época", ylabel=ylabel)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def confusion(matrix, path):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(matrix, cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, str(value), ha="center", va="center", color="darkred")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Normal", "Resto"],
           yticklabels=["Normal", "Resto"], xlabel="Predicción", ylabel="Real")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def activity(model, x, y, directory):
    model.eval()
    # Un latido por clase, selección fija: no se escogen por acierto del modelo.
    for label in (0, 1):
        idx = (y == label).nonzero()[0].item()
        with torch.no_grad():
            logits, trace = model(x[idx:idx + 1], record=True)
        spk, mem = (trace[k][:, 0].cpu().numpy() for k in ("spikes", "membrane"))
        np.savez_compressed(directory / f"activity_{label}.npz", spikes=spk,
                            membrane=mem, input=x[idx].numpy(), index=idx,
                            target=label, prediction=logits.argmax(1).item())
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(x[idx].numpy())
        axes[0].set(ylabel="Amplitud", title=f"Test índice {idx}, clase {label}")
        time, neuron = np.nonzero(spk)
        axes[1].scatter(time, neuron, s=4, marker="|")
        axes[1].set(ylabel="Neurona LIF", ylim=(-1, spk.shape[1]), title="Spikes internos")
        # Las cuatro neuronas más activas; se documenta esta selección visual.
        selected = np.argsort(-spk.sum(0), kind="stable")[:4]
        for n in selected:
            axes[2].plot(mem[:, n], label=f"Neurona {n}", alpha=0.8)
        axes[2].axhline(1.0, color="black", linestyle="--", label="Umbral")
        axes[2].set(xlabel="Paso temporal", ylabel="Potencial", title="Membrana con reset retardado")
        axes[2].legend(ncol=3)
        fig.tight_layout()
        fig.savefig(directory / f"activity_{label}.png", dpi=140)
        plt.close(fig)
