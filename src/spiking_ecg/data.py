"""Descarga verificable; validación binaria estratificada dentro del train oficial."""

import hashlib
import io
from pathlib import Path
import urllib.request
import zipfile

import numpy as np
from sklearn.model_selection import train_test_split
import torch

URL = "https://www.timeseriesclassification.com/aeon-toolkit/ECG5000.zip"
SHA256 = "41f6de20ac895e9ce31753860995518951f1ed42a405d0e51c909d27e3b3c5a4"


def load_official(data_dir):
    path = Path(data_dir) / "ECG5000.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        # No se extraen rutas del ZIP. Solo se leen dos miembros conocidos.
        with urllib.request.urlopen(URL, timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != SHA256:
            raise ValueError("Checksum de descarga inesperado; revisar fuente antes de usarla")
        path.write_bytes(content)
    if hashlib.sha256(path.read_bytes()).hexdigest() != SHA256:
        raise ValueError(f"Checksum incorrecto: {path}")
    arrays = []
    with zipfile.ZipFile(path) as archive:
        for split, size in (("TRAIN", 500), ("TEST", 4500)):
            raw = np.loadtxt(io.BytesIO(archive.read(f"ECG5000_{split}.txt")))
            if raw.shape != (size, 141) or not np.isfinite(raw).all():
                raise ValueError(f"Formato inesperado en {split}: {raw.shape}")
            if not np.isin(raw[:, 0], [1, 2, 3, 4, 5]).all():
                raise ValueError("Etiquetas originales desconocidas")
            arrays.append((raw[:, 1:].astype(np.float32), (raw[:, 0] != 1).astype(np.int64)))
    return arrays


def prepare_data(official, seed=42, val_fraction=0.2):
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction debe estar entre 0 y 1")
    (x, y), (xt, yt) = official
    train_idx, val_idx = train_test_split(
        np.arange(len(y)), test_size=val_fraction, stratify=y, random_state=seed,
    )
    # Un escalar global: misma escala para ambos modelos, sin aprender del test/val.
    mean, std = float(x[train_idx].mean()), float(x[train_idx].std())
    std = max(std, 1e-8)
    splits = {}
    for name, xx, yy in (("train", x[train_idx], y[train_idx]),
                         ("val", x[val_idx], y[val_idx]), ("test", xt, yt)):
        splits[name] = (torch.from_numpy((xx - mean) / std), torch.from_numpy(yy))
    metadata = {
        "url": URL, "sha256": SHA256, "seed": seed, "val_fraction": val_fraction,
        "mean": mean, "std": std,
        "train_indices": train_idx.tolist(), "val_indices": val_idx.tolist(),
        "labels": {"0": "normal (original 1)", "1": "resto (original 2-5)"},
        "counts": {name: torch.bincount(yy, minlength=2).tolist()
                   for name, (_, yy) in splits.items()},
    }
    return splits, metadata
