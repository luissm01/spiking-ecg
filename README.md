# spiking-ecg

Proyecto educativo y reproducible en **PyTorch y snnTorch** para comparar un MLP
con una red de neuronas Leaky Integrate-and-Fire (LIF) sobre ECG5000. Incluye el
baseline, un laboratorio visual y una investigación sobre duración del entrenamiento
y retención de membrana. Permite seguir el proceso: implementación, diagnóstico,
hipótesis, controles y resultados observados. No es una herramienta médica.

La tarea es binaria: **normal (clase original 1 → 0)** frente a **resto (2–5 → 1)**.
La clase positiva de precision, recall y F1 es **resto**.

## Reproducir y probar

Se necesita [uv](https://docs.astral.sh/uv/). Python 3.12 y las dependencias CPU
están fijadas en `.python-version`, `pyproject.toml` y `uv.lock`.

```bash
uv sync --frozen --group dev
uv run pytest -q

# Baselines: entrena y después evalúa el test oficial.
uv run spiking-ecg train --model both --epochs 20 --output results/reproduction

# Reevaluar un checkpoint sin entrenar.
uv run spiking-ecg evaluate \
  --checkpoint results/reproduction/snn/best.pt --output results/evaluation

# Selección de hiperparámetros: SOLO train y validación, nunca test.
uv run python -m spiking_ecg.research \
  --seeds 42 43 44 --max-epochs 60 --patience 10 \
  --output results/research-beta-repeat --report-dir reports/research-beta-repeat
```

El primer `uv sync` y la descarga del dataset necesitan Internet. Las pruebas usan
exclusivamente datos sintéticos y no necesitan red. Incluyen shapes, reset, gradientes,
repetibilidad, checkpoints, figuras y protección del test en el barrido. Con Node
instalado también contrastan el backward del laboratorio con snnTorch; sin Node,
siete pruebas del laboratorio se omiten explícitamente.

Usa carpetas de salida nuevas para no sobrescribir experimentos. En entornos con
HOME de solo lectura, configura `MPLCONFIGDIR=/tmp/spiking-ecg-matplotlib`.
Los comandos se muestran para reproducción futura; el cierre documental no ejecuta
nuevos experimentos ECG5000.

## Dataset y protocolo

[ECG5000, archivo académico UEA/UCR](https://timeseriesclassification.com/description.php?Dataset=ECG5000)
contiene latidos interpolados a **140 puntos** de un solo paciente con insuficiencia
cardíaca, registro `chf07`. Las etiquetas se obtuvieron automáticamente. Una etiqueta
de latido «normal» no implica que el paciente esté sano.

Se conserva el split oficial: **500 train / 4500 test**. Del train se extraen
**400 entrenamiento / 100 validación**, estratificados por etiqueta binaria. El ZIP
académico se descarga de forma reproducible y se verifica mediante SHA-256. Una
media y desviación escalar se ajustan únicamente con los 400 ejemplos de entrenamiento.
Se guardan índices, checksum y normalización; no se aprende preprocesado del test.

El baseline guarda el checkpoint de menor pérdida de validación y evalúa test al
terminar. **No se debe usar ese comando para barridos de selección.** `research.py`
abre exclusivamente `ECG5000_TRAIN.txt` y acepta solo los splits train/val. El test
**ya se observó durante el desarrollo inicial**: evitar nuevas consultas no restaura
su independencia. Toda la investigación se presenta como exploratoria.

## Modelos y fundamentos SNN

| Modelo | Arquitectura | Parámetros |
|---|---|---:|
| MLP | 140 → Linear(32) → ReLU → Linear(2) | 4578 |
| SNN baseline | 140 pasos de Linear(1 → 32) → LIF; tasas en 7 ventanas → Linear(224 → 2) | 514 |

El MLP recibe el latido completo. La SNN recibe una **amplitud real por paso como
corriente continua; la entrada no es un tren de spikes**. Los spikes binarios se
generan dentro de las neuronas LIF. La membrana acumula corriente y retiene una
fracción `beta` del estado anterior:

```text
I[t] = W x[t] + b
U[t] = beta * U[t-1] + I[t] - spike[t-1] * umbral
spike[t] = 1 si U[t] > umbral; 0 en otro caso
```

El baseline usa **beta=0.9, umbral=1** y reset por sustracción retardado: un spike
reduce la membrana en la siguiente actualización. Cada latido comienza con estado
cero; la membrana no se reinicia al cambiar de ventana. Las siete tasas locales
conservan posición aproximada dentro del latido frente a una media global.

El escalón se mantiene binario en forward; en backward se sustituye su derivada
por `1 / (1 + 25 * abs(U - umbral))**2`. Se usa **BPTT completo sobre 140 pasos**,
con la decisión de reset desacoplada del gradiente. Adam (`lr=0.001`, batch 64) y
cross entropy entrenan ambos modelos. Fijar semillas y algoritmos deterministas
facilita repetirlos, sin garantizar identidad entre plataformas o versiones.

Las decisiones de arquitectura, ecuaciones y formatos de artefactos se explican
en [docs/MODEL.md](docs/MODEL.md). Ambos modelos comparten protocolo y ancho oculto,
pero no igualan capacidad ni coste computacional.

## Resultados originales: test del baseline

Resultados conservados de **20 épocas, semilla 42**, seleccionando checkpoint por
pérdida de validación. Estas cifras corresponden al **test oficial**:

| Modelo | Accuracy test | F1 resto | Precision resto | Recall resto |
|---|---:|---:|---:|---:|
| MLP | 0.9800 | 0.9755 | 0.9972 | 0.9546 |
| SNN, beta=0.9 y 7 ventanas | 0.8720 | 0.8378 | 0.8862 | 0.7944 |

El [informe original](reports/README.md) conserva métricas, matrices de confusión,
configuración y el ensayo inicial de media global, que obtuvo F1=0. Los datos
numéricos completos están en [temporal_20.json](reports/temporal_20.json).
Los baselines y estos resultados no se sustituyen con el barrido posterior.

## Investigación: beta y duración, solo validación

Se compararon **beta=0.7/0.9/0.95 × semillas 42/43/44** manteniendo el mismo split
400/100 y los demás parámetros. Las semillas cambian inicialización y orden de
batches, emparejados entre betas; **no son tres splits independientes**.

Se permitió hasta 60 épocas, con early stopping por pérdida de validación:
paciencia 10, mejora mínima 0.0001 y mínimo 20 épocas. El checkpoint conserva el
mínimo estricto de esa pérdida. Las nueve ejecuciones llegaron a 60 y seleccionaron
esa época; no se activó early stopping ni se demostró convergencia.

**Validación, media ± desviación estándar muestral de tres semillas:**

| Beta | Loss ≤20 | Loss ≤60 | F1 ≤60 | Recall ≤60 |
|---|---:|---:|---:|---:|
| 0.7 | 0.3234 ± 0.0090 | 0.1307 ± 0.0093 | 0.9643 ± 0.0207 | 0.9683 ± 0.0275 |
| 0.9 | 0.3579 ± 0.0171 | 0.1766 ± 0.0142 | 0.9280 ± 0.0252 | 0.9683 ± 0.0275 |
| 0.95 | 0.3855 ± 0.0138 | 0.2105 ± 0.0213 | 0.9053 ± 0.0151 | 0.9841 ± 0.0137 |

Ampliar de 20 a 60 mejoró loss, F1 y recall en las nueve ejecuciones. Beta=0.7 fue
el candidato por loss/F1 en este split; beta=0.95 obtuvo más recall con más falsos
positivos. Se midieron tasas por neurona/paso y 2–3 neuronas silenciosas al seleccionar
checkpoint. Estas tasas no son Hz ni medidas de eficiencia energética.

**No comparar directamente esta tabla de validación con la tabla de test anterior.**
El [informe experimental completo](reports/research-beta/README.md) distingue
observaciones, hipótesis y límites estadísticos e incluye [CSV](reports/research-beta/comparison.csv),
[protocolo](reports/research-beta/protocol.json) e historiales JSON.
Figuras: [curvas de entrenamiento](reports/research-beta/training.png),
[actividad y silencio](reports/research-beta/activity.png) y
[20 frente a 60 épocas](reports/research-beta/horizons.png).

## Interactive SNN Laboratory

Abre el [laboratorio publicado](https://luissm01.github.io/spiking-ecg/) o el
[HTML autónomo](docs/index.html), con JavaScript habilitado. **Reproducir**, **Siguiente
paso** y **Reiniciar tiempo** recorren una traza precalculada. Cambia señal, neurona
y beta para observar membrana, raster y ventanas; predicción y gradientes siempre
corresponden a los 140 pasos completos. **Entrenar 1/10 pasos** modifica pesos del
ejemplo y **Restaurar pesos** recupera sus valores iniciales.

| Laboratorio | Experimento real |
|---|---|
| Señales sintéticas y pesos de ejemplo | ECG5000 y checkpoints entrenados |
| JavaScript y backward didáctico manual | PyTorch, snnTorch y autograd |
| SGD sobre un ejemplo (`lr=0.08`) | Adam sobre batches (`lr=0.001`) |

El laboratorio ilustra la dinámica y los surrogate gradients; no carga resultados
reales ni demuestra generalización. SGD con surrogate gradients no garantiza reducir
la pérdida en cada paso. La [verificación del laboratorio](docs/VERIFICATION.md)
documenta su contraste con snnTorch y las pruebas en navegador. Pages sirve `main/docs`
con `.nojekyll`; no se necesita build adicional.

## Limitaciones y trabajo futuro

El dataset procede de **un solo paciente**, con etiquetas automáticas y latidos
interpolados; el split oficial puede contener latidos próximos temporalmente.
Agrupar clases 2–5 oculta diferencias entre tipos de latido. No hay validación
clínica ni evidencia de generalización entre pacientes.

Las tres semillas comparten solo 100 ejemplos de validación (42 positivos), y su
SD no es un intervalo de confianza. La selección reiterada puede sobreajustar esa
validación; el test históricamente observado tampoco es confirmatorio. Los modelos
no están igualados por parámetros. No se midieron energía, FPGA ni ventaja de latencia.

Trabajo futuro: acordar el compromiso entre precision y recall, contrastar el
candidato beta=0.7 con más splits y semillas, controlar capacidad y evaluar con
pacientes independientes. No se incorpora ninguno de esos experimentos en este cierre.

El código permanece pequeño: [datos](src/spiking_ecg/data.py),
[modelos](src/spiking_ecg/models.py), [baseline](src/spiking_ecg/experiment.py),
[barrido](src/spiking_ecg/research.py), [gráficos](src/spiking_ecg/plots.py) y [pruebas](tests/).
Datasets, checkpoints y salidas pesadas quedan fuera de Git en `data/` y `results/`;
`reports/` conserva solo documentación, métricas, protocolo y figuras.
