# Integración y verificación del laboratorio

Revisión realizada el 5 de noviembre de 2026. El HTML aportado se conserva como
archivo autónomo, con CSS y JavaScript internos y sin dependencias de red en ejecución.
Se integró sin modificar `src/`, `reports/`, los datos, los checkpoints ni las
dependencias del experimento real.

## Correspondencia matemática

Se contrastó con `TemporalSNN` en `src/spiking_ecg/models.py` y con el código instalado
de snnTorch **0.9.4**, fijado en `uv.lock`:

```text
I[t,j] = w[j] * x[t] + b[j]
U[t,j] = beta * U[t-1,j] + I[t,j] - S[t-1,j]
S[t,j] = 1 si U[t,j] > 1; 0 si no
U[-1,j] = S[-1,j] = 0
```

La implementación original ya tenía correctamente el umbral estricto y el reset
por sustracción retardado. Una prueba con corriente constante `I=1` produce
membranas `[1, 1.9, 1.71]` y spikes `[0, 1, 1]`: no hay disparo justo en el umbral
ni sustracción inmediata del spike recién calculado. El estado se inicializa de
nuevo en cada forward, sin reiniciarlo entre ventanas. La lectura concatena
`[ventana, neurona]`, el mismo orden de las 224 tasas del modelo Python.

El backward manual conserva:

```text
delta_logits = softmax(logits) - one_hot(objetivo)
delta_spike[t,j] = delta_rate[ventana(t),j] / 20
surrogate(U) = 1 / (1 + 25 * abs(U - 1))**2
delta_U[t,j] = delta_spike[t,j] * surrogate(U[t,j]) + beta * delta_U[t+1,j]
delta_w[j] = suma_t delta_U[t,j] * x[t]
delta_b[j] = suma_t delta_U[t,j]
```

La derivada de la decisión de reset es cero, coherente con `mem_reset(...).detach()`
de snnTorch. Esto no elimina el término temporal `beta * delta_U[t+1]` ni los
gradientes que llegan desde los spikes al readout. Beta y umbral no se entrenan.
No se usaron diferencias finitas del escalón como validación del surrogate: el
backward sustituto no es la derivada numérica del forward discontinuo.

Referencias: [Leaky](https://snntorch.readthedocs.io/en/latest/snn.neurons_leaky.html)
y [surrogate gradients](https://snntorch.readthedocs.io/en/latest/snntorch.surrogate.html).
El código instalado de la versión fijada fue la referencia para el contraste.

## Cambios realizados

- Se hicieron explícitos en la página y README los datos sintéticos, pesos de
  ejemplo, JavaScript, SGD de un ejemplo y backward didáctico manual, separados
  del experimento ECG5000/PyTorch/snnTorch/Adam.
- Se aclararon entrada analógica, reset retardado, estado inicial cero, umbral
  estricto y gradientes del reset desacoplados. El cursor es una reproducción de
  trazas precalculadas; no ejecuta entrenamiento ni inferencia causal en vivo.
- La cross entropy pasó de `-log(max(1e-12,p[objetivo]))` a log-sum-exp estable
  (`max_logit + log(sum_exp) - logit_objetivo`), coherente con `softmax - objetivo`
  sin introducir el recorte artificial de la pérdida anterior.
- Se aclaró que SGD con surrogate gradients puede aumentar la pérdida y que
  aprender un ejemplo artificial no demuestra generalización ni validación clínica.
- Se añadieron descripción de página, aviso sin JavaScript y etiquetas accesibles
  para los canvas; los enlaces externos abren con `noopener noreferrer`.
- Se añadió `.nojekyll` para servir HTML estático desde `main` y `/docs`, y se
  documentó el uso y la configuración de Pages en el README principal.

## Comprobaciones ejecutadas

`uv run pytest -q`: **18 passed in 8.10s**, incluyendo las 11 pruebas anteriores.
Las siete nuevas pruebas ejecutan el script real del HTML en Node mediante un
DOM simulado para no dibujar y comparan numéricamente contra `TemporalSNN`:
ambas señales y beta **0.5, 0.9, 0.98**, membranas, spikes, logits, cross entropy,
gradientes de entrada y readout, reset entre forwards y actualización SGD de todos
los pesos. Se tolera redondeo JavaScript float64 frente a PyTorch float32; para
gradientes, tolerancia absoluta `1e-5` y relativa `3e-4`. El caso de umbral/reset
se comprueba por separado. Sin Node, las siete pruebas se omiten explícitamente.

Se ejecutó además `tests/check_lab_browser.cjs` en **Chromium 141 / Playwright 1.56.1**:
apertura directa `file://` y HTTP bajo `/spiking-ecg/`, anchos **1280, 390 y 320 px**.
Pasaron reproducción/pausa, avance, reinicio temporal, selección de señal/neurona/beta,
SGD de 1 y 10 pasos, restauración de pesos, generación de siete barras y ausencia
de errores JavaScript o desbordamiento horizontal. También se inspeccionó una captura.
El entorno requirió Chromium de Ubuntu 24.04 y tres bibliotecas descargadas en `/tmp`;
no se instalaron paquetes del navegador como dependencias del proyecto.

Para repetir la comprobación opcional del navegador en un sistema compatible:

```bash
npm install --prefix /tmp/spiking-ecg-browser playwright@1.56.1
/tmp/spiking-ecg-browser/node_modules/.bin/playwright install chromium
NODE_PATH=/tmp/spiking-ecg-browser/node_modules node tests/check_lab_browser.cjs
```

## Publicación verificada

En la integración inicial Pages estaba deshabilitado y no se añadió una URL
supuesta. Tras la activación por el propietario, el 5 de noviembre de 2026 se
verificó el [laboratorio público](https://luissm01.github.io/spiking-ecg/):

- La API de Pages devolvió `status=built`, origen `branch=main`, `path=/docs`.
- La URL pública respondió HTTP 200.
- El HTML descargado coincidió byte a byte con `docs/index.html` (`cmp`).

Se añadió el enlace público al README y se retiró el aviso de activación pendiente.
No quedan pasos manuales de publicación pendientes.
