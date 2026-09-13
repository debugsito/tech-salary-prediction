# Contexto de construcción: la raíz del repo, porque hace falta src/codificadores.
# 3.12 y no 3.14: la CPU del servidor no soporta los wheels recientes de NumPy.
FROM python:3.12-slim

# libgomp1: OpenMP para XGBoost, la imagen slim no lo trae.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY servicio/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Solo el codificador por objetivo; el resto de src importa bibliotecas de entrenamiento.
COPY src/__init__.py src/codificadores.py ./src/
COPY servicio/app.py servicio/fiabilidad.py ./
COPY servicio/test_paridad.py servicio/test_fiabilidad.py servicio/test_api.py ./
COPY servicio/artefactos/ ./artefactos/
COPY servicio/static/ ./static/

# Las pruebas corren dentro de la construcción: si no pasan, no hay imagen.
RUN pip install --no-cache-dir pytest==8.4.2 httpx==0.28.1 \
    && python -m pytest -q test_paridad.py test_fiabilidad.py test_api.py \
    && pip uninstall -y -q pytest httpx && rm -f test_*.py

RUN useradd --create-home --uid 10001 servicio && chown -R servicio /app
USER servicio

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/api/salud')"

# Un solo trabajador: cada uno carga su copia de las bibliotecas y el servidor comparte dos núcleos.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
