# Prototipo: estimación salarial explicable

Servicio que expone el modelo de la tesis con tres vistas y una compuerta que
decide si procede responder. Desplegado en **https://salarios.debugsito.click**.

## Qué lo distingue

Cualquier herramienta de estimación salarial devuelve una cifra. Esta comprueba
antes si tiene con qué, y cuando no lo tiene **se abstiene y explica por qué**.

La decisión se apoya en tres medidas exportadas del análisis: cuántas
observaciones respaldan la combinación de país y rol, si el grupo satisface la
pérdida acotada de Agarwal et al. (2019) llevada al error relativo, y cuánta
dispersión salarial hay dentro de los países de ese grupo.

El umbral de pérdida acotada no se fija a ojo: es 1.25 veces el error relativo
del grupo mejor servido, donde 1.25 es la razón de disparidad máxima que adopta
el marco teórico del documento.

## Archivos

| Archivo | Contenido |
| :--- | :--- |
| `fiabilidad.py` | La compuerta. Es la parte con contenido de tesis |
| `app.py` | Cuatro rutas sobre FastAPI |
| `static/index.html` | Interfaz, sin dependencias externas |
| `artefactos/` | Modelo serializado, contexto y estimaciones de referencia |
| `test_*.py` | 19 pruebas, que la construcción de la imagen ejecuta |

Los artefactos se generan con `scripts/exportar_modelo_servicio.py`. No se editan.

## Ejecutar

```bash
python ../scripts/exportar_modelo_servicio.py   # si los artefactos no están
python -m pytest -q
docker compose up -d --build
```

`compose.yml` se une a la red del servidor inverso que ya corre en la máquina de
despliegue; en local basta con publicar el puerto o ejecutar `uvicorn app:app`.

## Sobre las versiones

El servidor de despliegue expone una CPU virtual sin SSE3 ni AVX, y los wheels
de NumPy desde la versión 2.2 exigen la línea base `x86-64-v2`. El servicio corre
por tanto con NumPy 2.0 y Python 3.12, no con el entorno exacto del entrenamiento.

Esa divergencia se vigila: `test_paridad.py` contrasta las estimaciones del
servicio con las guardadas en `artefactos/referencia.json`, obtenidas en el
entorno de entrenamiento, y la construcción de la imagen falla si no coinciden
hasta el céntimo.

## Privacidad

No hay base de datos, ni sesión, ni registro de las consultas. El servicio recibe
un perfil, responde y lo olvida.
