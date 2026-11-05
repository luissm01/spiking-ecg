# Ejecución local medida

Fecha: 5 de noviembre de 2026. CPU, un hilo, Python 3.12.13, semilla 42.
Dependencias fijadas por `uv.lock`; PyTorch 2.15.0+cpu, snnTorch 0.9.4.
Los JSON de esta carpeta contienen las métricas completas sin redondear.

## Protocolo y resultado principal

```bash
uv sync --frozen --group dev
uv run pytest -q
uv run spiking-ecg train --model both --epochs 20 --output results/temporal
uv run spiking-ecg evaluate --checkpoint results/temporal/snn/best.pt \
  --output results/reevaluated-temporal
```

En el entorno de ejecución se configuraron `MPLCONFIGDIR` y `UV_CACHE_DIR` en `/tmp`
por las restricciones de escritura en HOME. Se descargó el ZIP académico con `curl`
y se colocó en `data/ECG5000.zip`; el cargador verificó el checksum antes de leerlo.
El código también descarga automáticamente el mismo URL si no existe el ZIP.

Split normal/resto: train **[234, 166]**, validación **[58, 42]**, test **[2627, 1873]**.
Se utilizaron exclusivamente los 400 ejemplos de entrenamiento para ajustar el escalado.
Mejor época por pérdida de validación: **20** en ambos modelos.

| Modelo | Parámetros | Accuracy test | F1 resto | Precision resto | Recall resto |
|---|---:|---:|---:|---:|---:|
| MLP | 4578 | 0.980000 | 0.975450 | 0.997211 | 0.954618 |
| SNN, 7 ventanas | 514 | 0.872000 | 0.837838 | 0.886242 | 0.794447 |

Matrices de confusión: filas reales, columnas predichas; orden `[normal, resto]`:

```text
MLP                 SNN (7 ventanas)
[[2622,    5],      [[2436,  191],
 [  85, 1788]]       [ 385, 1488]]
```

La pérdida de validación final fue 0.011268 para el MLP y 0.362699 para la SNN
(consultar el checkpoint/historial para precisión completa). F1 de validación:
1.0 y 0.809524, respectivamente. Un 100 % sobre 100 ejemplos de validación no
demuestra rendimiento perfecto fuera de esa muestra.

Resultado principal en [temporal_20.json](temporal_20.json); configuración, gradientes
e índices del split en [run_metadata.json](run_metadata.json). En ambos modelos
se verificó un gradiente finito y no nulo en la primera capa durante entrenamiento.
La reevaluación del checkpoint SNN reprodujo exactamente todas las métricas de test.

Las **11 pruebas pasaron** (`11 passed in 8.47s`), incluyendo dos entrenamientos
mínimos sintéticos, checkpoints, figuras y repetibilidad con semilla fija.

Los PNG, checkpoints, historias y predicciones completas permanecen localmente
en `results/temporal/`. Algunos ejemplos:

- Raster y membranas, resto: `results/temporal/snn/activity_1.png`.
- Raster y membranas, normal: `results/temporal/snn/activity_0.png`.
- Curvas SNN: `results/temporal/snn/training.png`.
- Matriz de confusión SNN: `results/temporal/snn/confusion.png`.
- Curvas MLP: `results/temporal/mlp/training.png`.

Esas rutas, relativas a la raíz del repositorio, solo están disponibles tras generar
los artefactos: `results/` está ignorado por Git. No se han añadido los datos ni los
checkpoints al control de versiones.

## Ensayo inicial y decisión de arquitectura

Antes de la lectura por ventanas se probó la media global de spikes. Se conserva
como `--readout-bins 1` y se puede repetir con:

```bash
uv run spiking-ecg train --model both --epochs 5 --readout-bins 1 \
  --output results/global-5-reproduction
uv run spiking-ecg train --model both --epochs 20 --readout-bins 1 \
  --output results/global-20-reproduction
```

| Ensayo | Accuracy MLP | F1 MLP | Accuracy SNN | F1 SNN |
|---|---:|---:|---:|---:|
| 5 épocas, media global | 0.959333 | 0.949711 | 0.583778 | 0.0 |
| 20 épocas, media global | 0.980000 | 0.975450 | 0.583778 | 0.0 |

Resultados en [global_5.json](global_5.json) y [global_20.json](global_20.json).
La SNN de 130 parámetros predijo solo normal, tanto en validación como en test,
aunque el gradiente de entrada tenía norma 0.0130474 en el primer batch. La
validación con F1=0 motivó conservar algo de posición temporal en el readout:
siete tasas locales de 20 pasos, manteniendo idéntica capa LIF. Esta observación
es compatible con pérdida de información por la media global, pero no demuestra
que sea la única causa; también cambian capacidad y optimización del readout.

El test se mantuvo físicamente separado y nunca se usó para entrenar o seleccionar
épocas. Sin embargo, se observó durante la iteración de desarrollo: estos números
son exploratorios y no una evaluación confirmatoria sobre test nunca observado.
Solo se ejecutó una semilla y no se midieron energía, FPGA ni ventaja de latencia.
