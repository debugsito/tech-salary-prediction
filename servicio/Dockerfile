# El contexto de construcción es la raíz del repositorio, no este directorio:
# la tubería serializada referencia `src.encoders`, que debe viajar en la imagen.
FROM python:3.12-slim

# XGBoost necesita la biblioteca de OpenMP, que la imagen «slim» no trae.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY servicio/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Del paquete de análisis solo hace falta el codificador por objetivo, que es lo
# que el modelo serializado referencia. El resto de módulos importan bibliotecas
# de entrenamiento que el servicio no necesita.
COPY src/__init__.py src/encoders.py ./src/
COPY servicio/app.py servicio/fiabilidad.py ./
COPY servicio/test_paridad.py servicio/test_fiabilidad.py servicio/test_api.py ./
COPY servicio/artefactos/ ./artefactos/
COPY servicio/static/ ./static/

# Se comprueba dentro de la imagen que este entorno reproduce las cifras del de
# entrenamiento. Si no lo hiciera, la construcción falla y no se publica nada.
RUN pip install --no-cache-dir pytest==8.4.2 httpx==0.28.1 \
    && python -m pytest -q test_paridad.py test_fiabilidad.py test_api.py \
    && pip uninstall -y -q pytest httpx && rm -f test_*.py

RUN useradd --create-home --uid 10001 servicio && chown -R servicio /app
USER servicio

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/api/salud')"

# Un solo proceso: el modelo ocupa medio megabyte pero cada trabajador carga su
# propia copia de las bibliotecas, y el servidor tiene dos núcleos que ya
# comparte con otros servicios.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
