# spiking-ecg

Mini proyecto educativo en PyTorch y snnTorch: comparar un MLP con una SNN LIF
para distinguir **normal (clase original 1 → 0)** de **resto (2–5 → 1)**.
La clase positiva de precision, recall y F1 es **resto**. No es una herramienta médica.

## Reproducir

Se necesita [uv](https://docs.astral.sh/uv/). Se usa Python 3.12 y PyTorch CPU para
reducir requisitos y facilitar la repetición. `uv.lock` fija las versiones resueltas.

```bash
uv sync --frozen --group dev
uv run pytest -q

# Prueba corta real; descarga y verifica ECG5000 automáticamente.
uv run spiking-ecg train --model both --epochs 5 --output results/short-new

# Experimento de referencia: 20 épocas, semilla 42, 32 neuronas ocultas.
uv run spiking-ecg train --model both --epochs 20 --output results/reproduction

# Reevalúa el checkpoint elegido por validación, sin entrenar.
uv run spiking-ecg evaluate \
  --checkpoint results/reproduction/snn/best.pt --output results/evaluation
```

El primer `uv sync` y la descarga necesitan Internet; las pruebas usan datos
sintéticos y funcionan sin red. Los siguientes entrenamientos reutilizan el ZIP
en `data/`. Si cambia su checksum, el programa se detiene: no acepta silenciosamente
otros datos. En entornos con HOME de solo lectura se puede anteponer
`MPLCONFIGDIR=/tmp/spiking-ecg-matplotlib` a los comandos.
Cada nuevo entrenamiento necesita una carpeta sin `best.pt` para evitar sobrescribirlo.

## Interactive SNN Laboratory

Abre el [laboratorio interactivo publicado](https://luissm01.github.io/spiking-ecg/)
en un navegador con JavaScript habilitado. También puedes abrir directamente el
[HTML autónomo](docs/index.html) en `docs/index.html`. No requiere instalación,
servidor, Python ni acceso a datos. Como alternativa, puedes servirlo localmente:

```bash
uv run python -m http.server 8000 --directory docs
# Abrir http://localhost:8000
```

Pulsa **Reproducir** o **Siguiente paso** para recorrer las 140 amplitudes de una
señal sintética y observar corriente, potencial de membrana y raster de 32 neuronas
LIF. Cambia señal, neurona o beta; inspecciona las siete ventanas y la clasificación.
El cursor recorre una secuencia completa precalculada: las gráficas, probabilidades
y gradientes corresponden al latido completo, no solo al instante señalado.
**Entrenar 1 paso / 10 pasos** aplica SGD al ejemplo seleccionado; **Restaurar pesos**
recupera los pesos de ejemplo y **Reiniciar tiempo** solo devuelve el cursor al inicio.

| Laboratorio HTML | Experimento real |
|---|---|
| Dos señales sintéticas y pesos de ejemplo | ECG5000 y checkpoints entrenados |
| JavaScript, backward didáctico manual con BPTT | PyTorch y snnTorch con autograd |
| SGD sobre un ejemplo, learning rate 0.08 | Adam sobre batches, learning rate 0.001 por defecto |

Ambos usan entrada analógica (no spikes de entrada), LIF con reset por sustracción
retardado, umbral estricto `U > 1`, surrogate `fast_sigmoid(slope=25)` y lectura por
siete ventanas. El backward manual excluye la derivada de la decisión de reset,
conservando la recurrencia temporal de membrana. Los controles no cargan checkpoints
reales ni reproducen sus métricas; una actualización no garantiza reducir la pérdida
y el simulador no demuestra generalización ni eficacia clínica.

GitHub Pages está publicado desde **main /docs**, con `docs/.nojekyll` para servir
el HTML estático sin workflow de build adicional. Se verificaron el estado `built`,
la respuesta HTTP 200 y la coincidencia del HTML público con `docs/index.html`.
Los cambios posteriores en `docs/` se publican al completar el despliegue de Pages.
La configuración está en **Settings → Pages → Deploy from a branch → main → /docs**.
[Instrucciones oficiales de GitHub](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).

La revisión matemática y las comprobaciones se documentan en
[docs/VERIFICATION.md](docs/VERIFICATION.md). `uv run pytest -q` incluye el contraste
numérico del JavaScript con nuestro modelo si Node está instalado; sin Node esas
pruebas se omiten y las pruebas originales siguen disponibles.

## Datos y protocolo

La fuente es el [archivo académico UEA/UCR de ECG5000](https://timeseriesclassification.com/description.php?Dataset=ECG5000).
Se descarga su ZIP desde `https://www.timeseriesclassification.com/aeon-toolkit/ECG5000.zip`;
el SHA-256 esperado se fija en `data.py`. Se leen exclusivamente los dos archivos
oficiales `ECG5000_TRAIN.txt` y `ECG5000_TEST.txt`, nunca el archivo combinado.

- Train oficial: **500** latidos; test oficial: **4500**; longitud: **140**.
- Del train se extrae un 20 % de validación estratificada sobre la etiqueta binaria:
  **400 entrenamiento, 100 validación**. No se intenta estratificar las cinco clases
  originales, algunas de las cuales son muy escasas.
- Una media y desviación escalar se ajustan exclusivamente con los 400 latidos de
  entrenamiento. Se aplican a todos los splits y a ambos modelos. No se ajustan
  estadísticas con validación/test, ni se normaliza cada posición por separado.
- Adam, cross entropy sin ponderación, batch 64, learning rate 0.001, semilla 42.
  Se fijan las semillas Python/NumPy/PyTorch, un hilo CPU y algoritmos deterministas.
- Se guarda la época de menor **pérdida de validación**; empates conservan la primera.
  El test se evalúa después de cargar ese checkpoint, no después de cada época.
  No se ajusta el umbral con el test: se usa argmax de dos logits.

Los índices de train/validación, distribución de clases, checksum y normalización
se guardan en `data.json` y en el checkpoint. La reevaluación comprueba que coinciden.
La repetición exacta requiere el mismo software y plataforma; una semilla no garantiza
identidad numérica entre versiones o dispositivos distintos.

## Arquitectura y decisiones

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

## Pruebas y resultados reales

Las pruebas comprueban shapes, spikes binarios, reset entre batches/secuencias,
gradientes hasta entrada y pesos, dos backwards sucesivos, repetibilidad con semilla,
separación train/val, ausencia de estadísticas del test en el escalado, rechazo de un
ZIP corrupto, métricas, entrenamiento mínimo de ambos modelos, carga de checkpoint
y generación de figuras. Los datos sintéticos de las pruebas no son resultados ECG.

Los resultados medidos, comandos y observaciones están en [reports/README.md](reports/README.md).
Se conserva también el ensayo de media global: fracasó en validación (F1=0) a 20
épocas. La lectura por ventanas se añadió para reducir esa pérdida de información
temporal. Se observaron resultados de test durante este desarrollo; se reportan
como **exploración educativa**, no como una estimación independiente confirmatoria.

## Límites e ideas para una entrevista

ECG5000 procede de un único paciente, registro `chf07` de 20 horas con insuficiencia
cardíaca congestiva. Las etiquetas se obtuvieron automáticamente; los latidos fueron
extraídos e interpolados a longitud fija. Agrupar 2–5 oculta diferencias entre tipos
de latido. El split oficial no demuestra generalización a otros pacientes y puede
contener latidos cercanos temporalmente. No hay validación clínica ni diagnóstico
de personas sanas a partir de la etiqueta «normal» de un latido.

No se han medido consumo energético, latencia comparativa ni ejecución en FPGA.
Los spikes y el menor número de parámetros no prueban eficiencia energética:
esta implementación simula 140 pasos con operaciones densas en CPU y BPTT conserva
estados intermedios durante entrenamiento.

Tres conceptos que conviene poder explicar con las figuras y el código:

1. **Memoria LIF y reset:** fuga, integración, umbral y sustracción; diferencia entre
   reset por spike y reinicialización entre latidos. Entrada analógica frente a spikes internos.
2. **Surrogate gradient y BPTT:** por qué la derivada del escalón no sirve, qué
   aproximación se usa en backward y por qué un gradiente no nulo no asegura aprendizaje.
3. **Representación y evaluación temporal:** qué información elimina una media
   global, por qué usar ventanas y por qué el protocolo, capacidad y generalización
   entre pacientes condicionan cualquier conclusión sobre MLP frente a SNN.

Código: `data.py` prepara datos; `models.py` define redes; `experiment.py` entrena y
evalúa; `plots.py` visualiza; `tests/test_project.py` verifica el comportamiento.
