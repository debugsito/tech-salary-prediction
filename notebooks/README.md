# Cuadernos

Los dos se generan con un script y **no se editan a mano**. Un cuaderno editado
acumula estado oculto y dependencias implícitas entre celdas, que es uno de los
defectos que la auditoría del trabajo previo encontró y que esta tesis no debe
repetir. Para cambiar algo, se edita el generador y se vuelve a construir.

| Cuaderno | Generador | Para qué |
| :--- | :--- | :--- |
| `tesis_reproducible.ipynb` | `scripts/build_notebook.py` | Recorrer el experimento en local, sobre los artefactos ya calculados |
| `tesis_colab.ipynb` | `scripts/build_notebook_colab.py` | Reejecutarlo entero desde cero en Google Colab |

## Abrir en Colab

https://colab.research.google.com/github/debugsito/tech-salary-prediction/blob/main/notebooks/tesis_colab.ipynb

No hace falta instalar nada ni descargar los datos por separado. El cuaderno
clona el repositorio, instala lo que Colab no trae, baja las ediciones 2022 y
2023 de la encuesta y ejecuta el experimento completo. Entre veinte y treinta
minutos con la CPU gratuita; hay una variable `MODO_RAPIDO` en la sección 2 para
una pasada de unos cinco minutos que comprueba que la cadena funciona, aunque
sus cifras no son comparables con las del documento.

Su última sección compara once magnitudes del documento de tesis con las que
acaba de producir la ejecución, e informa de cualquier discrepancia.

## Reconstruir

```bash
python scripts/build_notebook_colab.py
jupyter nbconvert --execute --to notebook --inplace notebooks/tesis_colab.ipynb
```

La segunda orden deja las salidas guardadas en el archivo, de modo que se pueda
leer sin ejecutarlo. Conviene hacerlo antes de publicar cualquier cambio: si una
celda falla, es preferible enterarse aquí que en Colab.
