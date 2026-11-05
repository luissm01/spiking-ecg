"""CLI CPU: selección por pérdida de validación, test al final, artefactos legibles."""

import argparse
import json
from pathlib import Path
import random

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import load_official, prepare_data
from .models import make_model
from . import plots


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def metrics(target, prediction):
    # Clase positiva = resto, etiquetas de matriz siempre [normal, resto].
    return {
        "accuracy": float(accuracy_score(target, prediction)),
        "f1": float(f1_score(target, prediction, zero_division=0)),
        "precision": float(precision_score(target, prediction, zero_division=0)),
        "recall": float(recall_score(target, prediction, zero_division=0)),
        "confusion_matrix": confusion_matrix(target, prediction, labels=[0, 1]).tolist(),
    }


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


@torch.no_grad()
def evaluate(model, split, batch_size):
    model.eval()
    total_loss, predictions = 0.0, []
    for x, y in DataLoader(TensorDataset(*split), batch_size=batch_size):
        logits = model(x)
        total_loss += nn.functional.cross_entropy(logits, y, reduction="sum").item()
        predictions.append(logits.argmax(1))
    prediction = torch.cat(predictions)
    result = metrics(split[1].numpy(), prediction.numpy())
    result["loss"] = total_loss / len(prediction)
    return result, prediction


def save_test(model, splits, batch_size, directory):
    result, prediction = evaluate(model, splits["test"], batch_size)
    write_json(directory / "test_metrics.json", result)
    np.savez_compressed(directory / "test_predictions.npz",
                        target=splits["test"][1].numpy(), prediction=prediction.numpy())
    plots.confusion(result["confusion_matrix"], directory / "confusion.png")
    if hasattr(model, "lif"):
        plots.activity(model, *splits["test"], directory)
    return result


def train_one(name, splits, metadata, directory, *, epochs=20, batch_size=64,
              hidden=32, beta=0.9, readout_bins=7, lr=1e-3, seed=42):
    if (directory / "best.pt").exists():
        raise FileExistsError(f"Ya existe un experimento en {directory}; usa otro --output")
    directory.mkdir(parents=True, exist_ok=True)
    seed_all(seed)  # Independiente de si se ejecutó otro modelo antes.
    config = dict(model=name, hidden=hidden, beta=beta, readout_bins=readout_bins, lr=lr, seed=seed,
                  epochs=epochs, batch_size=batch_size)
    environment = {"torch": str(torch.__version__), "numpy": np.__version__,
                   "device": "cpu", "threads": 1}
    write_json(directory / "config.json", {**config, "environment": environment})
    write_json(directory / "data.json", metadata)
    model = make_model(name, hidden, beta, readout_bins)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(*splits["train"]), batch_size=batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(seed))
    history, best, gradient_check = [], float("inf"), None
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, correct = 0.0, 0
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise RuntimeError("Loss no finita")
            loss.backward()  # BPTT completo en los 140 pasos de la SNN.
            if gradient_check is None:
                parameter = model.input.weight if name == "snn" else model.net[0].weight
                grad = parameter.grad
                if grad is None or not torch.isfinite(grad).all() or grad.abs().sum() == 0:
                    raise RuntimeError("No fluye un gradiente finito hasta la capa de entrada")
                gradient_check = {"first_input_weight_gradient_norm": grad.norm().item()}
                write_json(directory / "gradient_check.json", gradient_check)
            optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
        val, _ = evaluate(model, splits["val"], batch_size)
        row = dict(epoch=epoch, train_loss=total_loss / len(splits["train"][1]),
                   train_accuracy=correct / len(splits["train"][1]),
                   **{f"val_{key}": value for key, value in val.items()})
        history.append(row)
        print(f"{name} {epoch:03d}: train loss={row['train_loss']:.4f} "
              f"val loss={val['loss']:.4f} val F1={val['f1']:.4f}", flush=True)
        if val["loss"] < best:
            best = val["loss"]
            torch.save({"state_dict": model.state_dict(), "config": config,
                        "data": metadata, "epoch": epoch, "val_metrics": val,
                        "environment": environment}, directory / "best.pt")
    checkpoint = torch.load(directory / "best.pt", weights_only=True, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    write_json(directory / "history.json", history)
    plots.training(history, directory / "training.png")
    plots.signals(*splits["train"], directory / "signals.png")
    result = save_test(model, splits, batch_size, directory)
    result.update(best_epoch=checkpoint["epoch"], parameters=sum(p.numel() for p in model.parameters()))
    write_json(directory / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train")
    train.add_argument("--model", choices=["both", "mlp", "snn"], default="both")
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--val-fraction", type=float, default=0.2)
    train.add_argument("--hidden", type=int, default=32)
    train.add_argument("--beta", type=float, default=0.9)
    train.add_argument("--readout-bins", type=int, default=7,
                       help="Ventanas de lectura de la SNN; 1 = media global")
    train.add_argument("--lr", type=float, default=1e-3)
    test = sub.add_parser("evaluate")
    test.add_argument("--checkpoint", type=Path, required=True)
    for command in (train, test):
        command.add_argument("--data-dir", type=Path, default=Path("data"))
        command.add_argument("--output", type=Path, default=Path("results"))
        command.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size debe ser positivo")
    if args.command == "train":
        if args.epochs < 1 or args.hidden < 1 or args.lr <= 0 or not 0 <= args.beta <= 1:
            parser.error("epochs/hidden/lr deben ser positivos y beta debe estar en [0, 1]")
        if args.readout_bins < 1 or 140 % args.readout_bins:
            parser.error("readout-bins debe ser un divisor positivo de 140")
        splits, metadata = prepare_data(load_official(args.data_dir), args.seed, args.val_fraction)
        results = {}
        names = ("mlp", "snn") if args.model == "both" else (args.model,)
        for name in names:
            results[name] = train_one(
                name, splits, metadata, args.output / name, epochs=args.epochs,
                batch_size=args.batch_size, hidden=args.hidden, beta=args.beta,
                readout_bins=args.readout_bins, lr=args.lr, seed=args.seed,
            )
        write_json(args.output / "comparison.json", results)
        print(json.dumps(results, indent=2))
    else:
        ckpt = torch.load(args.checkpoint, weights_only=True, map_location="cpu")
        config, saved_data = ckpt["config"], ckpt["data"]
        seed_all(config["seed"])
        splits, metadata = prepare_data(load_official(args.data_dir), saved_data["seed"],
                                        saved_data["val_fraction"])
        if metadata != saved_data:
            raise ValueError("Los datos/preprocesado no coinciden con el checkpoint")
        model = make_model(config["model"], config["hidden"], config["beta"],
                           config.get("readout_bins", 1))
        model.load_state_dict(ckpt["state_dict"])
        args.output.mkdir(parents=True, exist_ok=True)
        print(json.dumps(save_test(model, splits, args.batch_size, args.output), indent=2))


if __name__ == "__main__":
    main()
