# Verificación para publicación

Cierre documental del 5 de noviembre de 2026, sobre el baseline publicado en
`c936495`. Se revisaron y prepararon los experimentos existentes; **no se ejecutaron
nuevos entrenamientos ECG5000 ni evaluaciones de test**, ni se recalcularon figuras
o resultados experimentales. Los únicos entrenamientos de esta revisión fueron
las ejecuciones mínimas con datos sintéticos que forman parte de las pruebas.

## Pruebas y aislamiento del test

- `uv run pytest -q`: **22 passed in 9.65s**, sin omisiones. Incluye las pruebas
  originales, el contraste numérico del laboratorio y las cuatro de investigación.
- El test del cargador intercepta `ZipFile.read`: solo permite abrir
  `ECG5000_TRAIN.txt` y rechaza cualquier otro miembro, incluido el test sintético.
- El barrido rechaza un diccionario con split `test`. Su código de producción no
  llama a `load_official`, `save_test` ni al entrenamiento del baseline.
- Se comprueba que el early stopping selecciona por loss aunque F1 aumente,
  y que las tasas de disparo y neuronas silenciosas se cuentan correctamente.
- La prueba opcional existente de Playwright/Chromium también pasó: apertura
  directa y por HTTP, controles y vistas de 1280/390/320 px, sin errores JavaScript
  ni desbordamiento horizontal. No se modificó el laboratorio HTML.

## Consistencia de los artefactos publicados

Una comprobación de lectura, sin inferencia ni entrenamiento, verificó:

- Nueve ejecuciones beta/semilla únicas, con 60 épocas cada una: **540 épocas**.
- Las 1080 matrices train/val concuerdan con accuracy, precision, recall y F1;
  sus tamaños son 400 y 100 ejemplos, respectivamente.
- `reference20` y `selected` coinciden con los mínimos de pérdida en sus horizontes;
  CSV y JSON contienen los mismos valores, sin redondear los archivos numéricos.
- Las medias y desviaciones muestrales de `aggregate.json` se reconstruyen a partir
  de `runs.json`; los índices y el escalado del protocolo coinciden con el baseline.
- Las tasas globales concuerdan con las 32 tasas individuales y con el número de
  neuronas de tasa cero; no hay tasas fuera de [0, 1] ni pérdidas no finitas.
- `training.png` (1960×1386), `activity.png` (1960×1260) y `horizons.png` (1960×560)
  se decodifican correctamente. Se conservan las figuras existentes.
- Los enlaces Markdown relativos resuelven a archivos o carpetas publicables,
  incluyendo el informe, protocolo, CSV, JSON y las tres figuras. Las rutas del
  informe original hacia salidas ignoradas se documentan como rutas locales, sin
  prometer enlaces de GitHub a archivos que no se versionan.

## Preservación y alcance

Se compararon por SHA-256 **154 artefactos numéricos/binarios existentes** en
`reports/`, `results/` y `data/`: todos quedaron idénticos durante el cierre.
El código original de datos, modelos, entrenamiento y gráficos, sus pruebas, las
dependencias y `docs/index.html` no difieren de la versión publicada de partida.

Se publican `research.py`, sus pruebas, el informe experimental con sus artefactos
ligeros y la documentación revisada. Los detalles técnicos del README se conservan
en `docs/MODEL.md`; los resultados de test históricos y los de validación del barrido
se presentan separados. La tabla de precision del informe deriva de los valores
ya guardados, sin cambiar los JSON/CSV.

Se revisaron diff y `.gitignore`; `data/`, `results/`, `.venv`, credenciales y
checkpoints (`*.pt`, `*.pth`, `*.ckpt`) quedan excluidos. Los JSON de investigación
y tres PNG ocupan aproximadamente 2.5 MB en total; no contienen señales ni pesos.
No se incluyen datasets, checkpoints, cachés ni salidas pesadas.

El CSV conserva byte a byte sus finales de línea CRLF originales de `csv.writer`.
`.gitattributes` evita su conversión automática y reconoce CR al final de línea
en la revisión de whitespace, sin modificar el archivo exportado.

La verificación confirma consistencia y reproducibilidad del material guardado,
no independencia estadística: las tres semillas comparten split y el test original
ya había sido observado. Véanse las [conclusiones y límites](README.md).
