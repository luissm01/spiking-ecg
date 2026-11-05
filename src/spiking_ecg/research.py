"""Barrido SNN aislado: solo TRAIN oficial, train/val y early stopping."""

import argparse
import csv
import hashlib
import io
from pathlib import Path
import platform
import time
import urllib.request
import zipfile

import numpy as np
from sklearn.model_selection import train_test_split
import snntorch
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import SHA256, URL
from .experiment import metrics, seed_all, write_json
from .models import TemporalSNN
from . import plots  # Configura el backend Agg antes de importar pyplot.
import matplotlib.pyplot as plt


def load_train_val(data_dir, split_seed=42):
    """No usa load_official: ni abre ni deserializa ECG5000_TEST.txt."""
    archive = Path(data_dir) / "ECG5000.zip"
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != SHA256:
            raise ValueError("Checksum de descarga inesperado")
        archive.write_bytes(content)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
        raise ValueError("Checksum incorrecto")
    with zipfile.ZipFile(archive) as z:
        raw = np.loadtxt(io.BytesIO(z.read("ECG5000_TRAIN.txt")))
    if raw.shape != (500, 141) or not np.isfinite(raw).all():
        raise ValueError("Formato de train inesperado")
    if not np.isin(raw[:, 0], [1, 2, 3, 4, 5]).all():
        raise ValueError("Etiquetas desconocidas")
    x, y = raw[:, 1:].astype(np.float32), (raw[:, 0] != 1).astype(np.int64)
    tr, va = train_test_split(np.arange(500), test_size=0.2, stratify=y, random_state=split_seed)
    mean, std = float(x[tr].mean()), max(float(x[tr].std()), 1e-8)
    splits = {name: (torch.from_numpy((x[idx] - mean) / std), torch.from_numpy(y[idx]))
              for name, idx in (("train", tr), ("val", va))}
    metadata = dict(url=URL, sha256=SHA256, member="ECG5000_TRAIN.txt", split_seed=split_seed,
                    mean=mean, std=std, train_indices=tr.tolist(), val_indices=va.tolist(),
                    counts={k: torch.bincount(v[1], minlength=2).tolist() for k, v in splits.items()},
                    test_member_opened=False)
    return splits, metadata


@torch.no_grad()
def measure(model, split, batch_size=64):
    """Estadísticas posteriores a actualizar pesos, sobre TODO el split indicado."""
    model.eval()
    n, total_loss, silent_pairs = len(split[1]), 0.0, 0
    counts = torch.zeros(model.input.out_features, dtype=torch.float64)
    predictions = []
    for x, y in DataLoader(TensorDataset(*split), batch_size=batch_size):
        logits, trace = model(x, record=True)
        spikes = trace["spikes"]
        counts += spikes.sum(dim=(0, 1)).double()
        silent_pairs += (spikes.sum(dim=0) == 0).sum().item()
        total_loss += nn.functional.cross_entropy(logits, y, reduction="sum").item()
        predictions.append(logits.argmax(1))
    result = metrics(split[1].numpy(), torch.cat(predictions).numpy())
    rates = counts / (n * split[0].shape[1])
    result.update(loss=total_loss / n, firing_rate=rates.mean().item(),
                  neuron_rates=rates.tolist(), silent_neurons=int((counts == 0).sum()),
                  silent_sequence_neuron_fraction=silent_pairs / (n * len(counts)))
    return result


def train_run(splits, metadata, directory, *, beta, seed, max_epochs=60, patience=10,
              min_delta=1e-4, min_epochs=20):
    if set(splits) != {"train", "val"}:
        raise ValueError("El barrido solo acepta train y val, sin test")
    directory.mkdir(parents=True, exist_ok=False)
    seed_all(seed)
    model = TemporalSNN(hidden=32, beta=beta, readout_bins=7)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loader = DataLoader(TensorDataset(*splits["train"]), batch_size=64, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    config = dict(beta=beta, seed=seed, max_epochs=max_epochs, patience=patience,
                  min_delta=min_delta, min_epochs=min_epochs, hidden=32, readout_bins=7,
                  batch_size=64, lr=0.001, optimizer="Adam", test_evaluated=False)
    write_json(directory / "config.json", config)
    history, best, patience_best, stale = [], float("inf"), float("inf"), 0
    start = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        online_loss = 0.0
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError("Loss no finita")
            loss.backward()
            grad = model.input.weight.grad
            if grad is None or not torch.isfinite(grad).all():
                raise RuntimeError("Gradiente de entrada ausente o no finito")
            optimizer.step()
            online_loss += loss.item() * len(y)
        row = dict(epoch=epoch, train_loss_online=online_loss / len(splits["train"][1]),
                   train=measure(model, splits["train"]), val=measure(model, splits["val"]))
        history.append(row)
        val_loss = row["val"]["loss"]
        if val_loss < best:  # Checkpoint: mínimo estricto, independiente de min_delta.
            best, best_row = val_loss, row
            torch.save(dict(state_dict=model.state_dict(), config=config, data=metadata,
                            epoch=epoch, validation=row["val"]), directory / "best.pt")
        if val_loss < patience_best - min_delta:
            patience_best, stale = val_loss, 0
        else:
            stale += 1
        if epoch == 20:
            torch.save(dict(state_dict=model.state_dict(), config=config, data=metadata,
                            epoch=epoch, validation=row["val"]), directory / "epoch20.pt")
        if epoch % 10 == 0:
            print(f"beta={beta} seed={seed} epoch={epoch} val loss={val_loss:.5f} "
                  f"F1={row['val']['f1']:.4f} recall={row['val']['recall']:.4f}", flush=True)
        if epoch >= min_epochs and stale >= patience:
            break
    # La referencia de 20 épocas también se elige SOLO por val loss.
    reference = min(history[:20], key=lambda row: row["val"]["loss"])
    summary = dict(beta=beta, seed=seed, epochs_run=len(history), best_epoch=best_row["epoch"],
                   stop_reason="patience" if epoch < max_epochs else "max_epochs",
                   elapsed_seconds=time.perf_counter() - start,
                   reference20=reference, selected=best_row,
                   val_loss_improvement=reference["val"]["loss"] - best_row["val"]["loss"],
                   test_evaluated=False)
    write_json(directory / "history.json", history)
    write_json(directory / "summary.json", summary)
    print(f"FIN beta={beta} seed={seed}: best epoch={best_row['epoch']}, "
          f"{summary['elapsed_seconds']:.1f}s", flush=True)
    return summary, history


def export_results(runs, directory):
    directory.mkdir(parents=True, exist_ok=True)
    summaries = [s for s, _ in runs]
    write_json(directory / "runs.json", summaries)
    write_json(directory / "histories.json", [h for _, h in runs])
    fields = ["beta", "seed", "epochs_run", "best_epoch", "loss20", "f1_20", "recall20",
              "loss", "f1", "recall", "firing_train", "firing_val", "silent_train", "silent_val"]
    with (directory / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for s in summaries:
            v, r = s["selected"]["val"], s["reference20"]["val"]
            writer.writerow(dict(beta=s["beta"], seed=s["seed"], epochs_run=s["epochs_run"],
                                 best_epoch=s["best_epoch"], loss20=r["loss"], f1_20=r["f1"],
                                 recall20=r["recall"], loss=v["loss"], f1=v["f1"], recall=v["recall"],
                                 firing_train=s["selected"]["train"]["firing_rate"],
                                 firing_val=v["firing_rate"],
                                 silent_train=s["selected"]["train"]["silent_neurons"],
                                 silent_val=v["silent_neurons"]))
    betas = sorted({s["beta"] for s in summaries})
    aggregate = []
    for beta in betas:
        group = [s for s in summaries if s["beta"] == beta]
        stats = {"beta": beta, "n": len(group)}
        for key in ("loss", "f1", "recall", "firing_rate", "silent_neurons"):
            for horizon in ("reference20", "selected"):
                values = [s[horizon]["val"][key] for s in group]
                stats[f"{horizon}_{key}_mean"] = float(np.mean(values))
                stats[f"{horizon}_{key}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
        aggregate.append(stats)
    write_json(directory / "aggregate.json", aggregate)
    seeds = sorted({s["seed"] for s in summaries})
    colors = dict(zip(betas, ("tab:blue", "tab:orange", "tab:green")))
    fig, axes = plt.subplots(len(seeds), 3, figsize=(14, 3.3 * len(seeds)), squeeze=False)
    for s, history in runs:
        row, color = seeds.index(s["seed"]), colors[s["beta"]]
        epochs = [h["epoch"] for h in history]
        for col, key in enumerate(("loss", "f1", "recall")):
            ax = axes[row, col]
            ax.plot(epochs, [h["val"][key] for h in history], color=color, label=f"beta={s['beta']}")
            if key == "loss":
                ax.plot(epochs, [h["train"][key] for h in history], color=color, linestyle="--", alpha=0.5)
            ax.axvline(20, color="grey", linestyle=":")
            ax.set(title=f"Seed {s['seed']}: val {key}", xlabel="Época")
            ax.legend(fontsize=8)
    fig.suptitle("Train discontinuo / validación continuo; línea vertical: 20 épocas")
    fig.tight_layout()
    fig.savefig(directory / "training.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(len(seeds), 3, figsize=(14, 3 * len(seeds)), squeeze=False)
    for s, history in runs:
        row, color = seeds.index(s["seed"]), colors[s["beta"]]
        for col, key in enumerate(("firing_rate", "silent_neurons", "silent_sequence_neuron_fraction")):
            axes[row, col].plot([h["epoch"] for h in history], [h["val"][key] for h in history],
                                color=color, label=f"beta={s['beta']}")
            axes[row, col].set(title=f"Seed {s['seed']}: val {key}", xlabel="Época")
            axes[row, col].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / "activity.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for s in summaries:
        for ax, key in zip(axes, ("loss", "f1", "recall")):
            ax.plot([20, 60], [s["reference20"]["val"][key], s["selected"]["val"][key]],
                    color=colors[s["beta"]], marker="o", alpha=0.7)
            ax.set(title=f"Validación: {key}", xticks=[20, 60], xlabel="Máximo de épocas (early stopping)")
    axes[0].legend([plt.Line2D([], [], color=colors[b]) for b in betas], [f"beta={b}" for b in betas])
    fig.tight_layout()
    fig.savefig(directory / "horizons.png", dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("results/research-beta"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/research-beta"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()
    if args.max_epochs < 20 or args.patience < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("max-epochs >= 20, patience > 0 y semillas distintas son necesarios")
    if args.output.exists() or args.report_dir.exists():
        parser.error("Usa carpetas nuevas; no se sobrescriben ejecuciones previas")
    splits, metadata = load_train_val(args.data_dir)
    args.output.mkdir(parents=True)
    write_json(args.output / "data.json", metadata)
    runs = []
    for beta in (0.7, 0.9, 0.95):
        for seed in args.seeds:
            runs.append(train_run(splits, metadata, args.output / f"beta-{beta}_seed-{seed}",
                                  beta=beta, seed=seed, max_epochs=args.max_epochs, patience=args.patience))
    export_results(runs, args.report_dir)
    write_json(args.report_dir / "protocol.json", dict(
        data=metadata, seeds=args.seeds, betas=[0.7, 0.9, 0.95], max_epochs=args.max_epochs,
        patience=args.patience, min_delta=1e-4, min_epochs=20, hidden=32, readout_bins=7,
        lr=0.001, batch_size=64, optimizer="Adam", test_evaluated=False,
        python=platform.python_version(), torch=str(torch.__version__),
        numpy=np.__version__, snntorch=snntorch.__version__, device="cpu", threads=1))


if __name__ == "__main__":
    main()
