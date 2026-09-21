# Estimación salarial explicable en el sector tecnológico

Código y artefactos de mi tesis de Ingeniería de Sistemas (Universidad ESAN),
escrita junto a Arian Garay con la asesoría de Wilfredo Mamani: un modelo de
estimación de compensaciones entrenado sobre la Stack Overflow Developer Survey,
con explicabilidad SHAP, auditoría de equidad por grupos y un prototipo
desplegado que **se niega a responder cuando no tiene datos suficientes** — que
es, para mí, la parte más interesante del trabajo.

El prototipo está en línea: **https://salarios.debugsito.click** (estado en
`/api/salud`, contrato de la interfaz en `/api/docs`).

## Qué hay aquí

```
src/                  Módulos del análisis: carga de la encuesta, codificación
                      por objetivo, tubería, métricas, SHAP, equidad
scripts/              Guiones ejecutables: experimento, análisis, robustez,
                      figuras del documento, exportación del servicio
servicio/             El prototipo: FastAPI + compuerta de fiabilidad +
                      interfaz, con su Dockerfile y sus 24 pruebas
notebooks/            Cuaderno reproducible del trabajo (local y Colab)
resultados/           Artefactos de la ejecución definitiva (semilla 42)
figuras/              Figuras del documento, generadas desde los artefactos
datos/referencia/     Tablas de referencia versionadas (Banco Mundial, OEWS)
```

Los datos crudos no se versionan: `src/carga_encuesta.py` los descarga del
repositorio oficial de Stack Overflow (licencia ODbL 1.0) y aplica los criterios
de inclusión documentados en la tesis.

## Reproducir

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# La ejecución completa: ~4 minutos en una máquina de escritorio
.venv/bin/python scripts/ejecutar_experimento.py --anio 2023
.venv/bin/python scripts/ejecutar_analisis.py --anio 2023

# O el cuaderno, que recorre todo el camino con las decisiones comentadas
.venv/bin/jupyter lab notebooks/tesis_trabajo.ipynb
```

Todo corre con semilla fija (42). Los números del documento salen de
`resultados/`, nunca de una corrida suelta: si un número no está en un
artefacto, no existe.

## El servicio

```bash
cd servicio && docker compose up -d --build
```

La construcción de la imagen ejecuta dentro las 24 pruebas, incluida una de
**paridad**: las estimaciones del contenedor se contrastan contra las del
entorno de entrenamiento con tolerancia de un céntimo. Si no coinciden, la
imagen no llega a existir.

## Sobre la honestidad del modelo

El error del modelo no es uniforme: en países de renta alta el error relativo
ronda el 30 %, en los de renta media supera el 50 %. En lugar de esconder eso,
el prototipo lo usa: una compuerta evalúa el respaldo muestral y el error del
grupo antes de responder, y cuando no alcanza, explica por qué en lugar de dar
una cifra. Perú — mi país, y la razón de que me metiera en esto — es
precisamente uno de los mercados donde el sistema se abstiene.
