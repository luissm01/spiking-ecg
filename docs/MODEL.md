# Arquitectura, entrenamiento y artefactos

| Modelo | Recorrido | Lectura | Parámetros por defecto |
|---|---|---|---:|
| MLP | 140 → Linear(32) → ReLU | Linear(2) | 4578 |
| SNN | 140 pasos de Linear(1 → 32) → LIF(32) | Tasas en 7 ventanas de 20 pasos → Linear(224 → 2) | 514 |

El MLP ve el latido entero a la vez y proporciona un baseline sencillo con acceso
directo a la forma de onda. La SNN recibe **una amplitud real por instante**, que una
proyección aprendida convierte en corriente, también con valores negativos. **La
entrada no es un tren de spikes**: no hay codificación Poisson, rate coding de
entrada ni expansión artificial a otros 140 pasos. Los spikes se generan internamente.

Una sola capa LIF permite inspeccionar todo el estado sin introducir convoluciones,
recurrencia aprendida o varias capas difíciles de interpretar. Sus pesos se comparten
en el tiempo. El readout es convencional y produce logits, sin softmax antes de
cross entropy. La media por ventanas conserva posición temporal aproximada y es
más pequeña que concatenar los 140 estados completos. No se reinicia la membrana
en los límites de las ventanas. `--readout-bins 1` reproduce la lectura por media
global; `--readout-bins` debe dividir 140.

La comparación comparte datos, optimizador, épocas y ancho oculto, **no iguala
número de parámetros ni presupuesto computacional**. No permite atribuir diferencias
de rendimiento exclusivamente al tipo de neurona. Para una comparación de investigación
harían falta controles de capacidad, varias semillas y selección de hiperparámetros
limitada a validación.

## Dinámica LIF y gradientes

Para cada neurona y paso `t`, con estado inicial `u[-1] = 0`:

```text
i[t] = W x[t] + b
u[t] = beta * u[t-1] + i[t] - s[t-1] * theta
s[t] = 1 si u[t] > theta; 0 en otro caso
beta = 0.9; theta = 1.0
```

Se usa `snntorch.Leaky`, reset por sustracción y `reset_delay=True`: el spike se
calcula con la membrana actual y su sustracción se aplica en la siguiente actualización.
Por eso una traza puede quedar por encima del umbral en el paso del spike. No hay
periodo refractario; beta y umbral son fijos. Los 140 puntos son muestras interpoladas
del latido, no un eje en milisegundos; beta no se interpreta aquí como una constante
de tiempo fisiológica calibrada.

La función escalón no tiene una derivada útil para entrenar. En el forward sigue
produciendo spikes binarios; solo en el backward se usa la derivada sustituta
`1 / (1 + 25 * abs(u - theta))**2`, de `surrogate.fast_sigmoid(slope=25)`.
Se aplica BPTT completo sobre los 140 pasos sin separar el grafo entre instantes;
snntorch desacopla del gradiente la decisión del reset. Los tensores de visualización
se separan del grafo, pero los spikes que alimentan el readout conservan sus gradientes.
Véanse las referencias de [Leaky](https://snntorch.readthedocs.io/en/latest/snn.neurons_leaky.html)
y [surrogate gradients](https://snntorch.readthedocs.io/en/latest/snntorch.surrogate.html).

Cada `forward` crea una membrana de ceros de tamaño `[batch, hidden]` y la pasa
explícitamente a la LIF. Así cada secuencia tiene estado independiente, incluso
cuando cambia el tamaño del último batch. No se reutiliza el estado final del
latido anterior. Reiniciar entre secuencias es distinto del reset por umbral dentro
de una secuencia. Se comprueba en entrenamiento que el primer backward tiene un
gradiente finito y no nulo en los pesos de entrada; esto no garantiza convergencia.

## Artefactos

Cada modelo guarda en `results/<experimento>/<modelo>/`:

- `best.pt`: pesos, arquitectura, época, métricas de validación, normalización y split.
  Sirve para inferencia/evaluación; no incluye estado de Adam para reanudar entrenamiento.
- `config.json`, `data.json`, `gradient_check.json`, `history.json`, `summary.json`
  y `test_metrics.json`: configuración, trazabilidad, curvas y métricas numéricas.
- `test_predictions.npz`: etiqueta real y predicción en el orden del test oficial.
- `signals.png`: primeros cinco latidos de cada grupo del train normalizado.
- `training.png`: pérdida y accuracy; train se acumula durante las actualizaciones,
  validación se mide al finalizar la época.
- `confusion.png`: filas reales y columnas predichas, orden `[normal, resto]`.
- Solo SNN: `activity_0.png`, `activity_1.png` con señal, raster de todas las neuronas
  y membrana de las cuatro más activas. Se elige el primer latido de test de cada
  clase, independientemente del acierto. Los `.npz` correspondientes conservan las
  trazas completas `[140, hidden]`, índice, etiqueta y predicción.

`comparison.json` compara ambos modelos. `data/` y `results/` están ignorados por Git;
el informe numérico de la ejecución local se conserva en `reports/`.


Volver al [README principal](../README.md). El protocolo del barrido y sus resultados
están en el [informe experimental](../reports/research-beta/README.md).
