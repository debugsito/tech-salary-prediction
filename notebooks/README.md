# Cuadernos

Los tres se generan con un script y **no se editan a mano**. Un cuaderno editado
acumula estado oculto y dependencias implícitas entre celdas, que es uno de los
defectos que la auditoría del trabajo previo encontró y que esta tesis no debe
repetir. Para cambiar algo, se edita el generador y se vuelve a construir.

| Cuaderno | Generador | Para qué |
| :--- | :--- | :--- |
| `tesis_trabajo.ipynb` | `build_notebook_manual.py` | Rehacer el análisis entero desde el CSV en bruto, decisión por decisión, incluido el programa de robustez (PPA, patrón oro oficial, transferencia, consistencia entre ediciones y mitigación por modelo). Produce las nueve figuras y todas las tablas |
| `tesis_colab.ipynb` | `build_notebook_colab.py` | Reejecutar el experimento y contrastar el resultado con lo que afirma el documento |
| `tesis_reproducible.ipynb` | `build_notebook.py` | Recorrer los artefactos ya calculados, sin volver a entrenar. Rápido |

El primero es el que hay que leer para entender cómo se llegó a los resultados.
Los otros dos verifican.

## Abrir en Colab

Cuaderno de trabajo:
https://colab.research.google.com/github/debugsito/tech-salary-prediction/blob/main/notebooks/tesis_trabajo.ipynb

Verificación:
https://colab.research.google.com/github/debugsito/tech-salary-prediction/blob/main/notebooks/tesis_colab.ipynb

No hace falta instalar nada ni descargar los datos por separado: la primera celda
clona el repositorio, instala lo que Colab no trae y el resto se descarga solo.

## En local

```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/jupyter lab notebooks/tesis_trabajo.ipynb
```

## Qué escribe cada uno

`tesis_trabajo.ipynb` escribe en `salida_cuaderno/`, nunca en `results/` ni en
`figures/`. Es deliberado: si una celda falla a mitad no se lleva por delante los
artefactos del proyecto. Sus nueve figuras salen byte a byte idénticas a las de
`figures/`, porque llama a las mismas funciones de dibujo.

## Reconstruir

```bash
python scripts/build_notebook_manual.py
jupyter nbconvert --execute --to notebook --inplace notebooks/tesis_trabajo.ipynb
```

La segunda orden guarda las salidas en el archivo, de modo que se pueda leer sin
ejecutarlo. Conviene hacerlo antes de publicar cualquier cambio: si una celda
falla, mejor enterarse aquí que en Colab.
