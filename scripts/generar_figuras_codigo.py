#!/usr/bin/env python3
"""Genera las figuras de código del capítulo IV.

Cada figura es un render del código fuente real del repositorio —no una
transcripción—, con su archivo de procedencia como cabecera y los números de
línea verdaderos. Los extractos se localizan por marcadores de texto y no por
número de línea, de modo que sobreviven a ediciones menores; si un marcador
desaparece, el guion falla en lugar de producir una figura desactualizada.

Uso:
    python scripts/generar_figuras_codigo.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pygments import highlight
from pygments.formatters import ImageFormatter
from pygments.lexers import DockerLexer, PythonLexer, YamlLexer

RAIZ = Path(__file__).resolve().parent.parent
FIGDIR = RAIZ / "figuras"

FUENTE = "Liberation Mono"
TAMANO = 13


def _rango(lineas: list[str], inicio: str, fin: str, extra: int = 0) -> tuple[int, int]:
    """Índices (base 0, fin exclusivo) del bloque delimitado por los marcadores."""
    i = next(n for n, l in enumerate(lineas) if inicio in l)
    j = next(n for n, l in enumerate(lineas[i:], start=i) if fin in l) + 1 + extra
    return i, j


def _render(codigo: str, lexer, primera_linea: int) -> Image.Image:
    png = highlight(
        codigo,
        lexer,
        ImageFormatter(
            font_name=FUENTE,
            font_size=TAMANO,
            style="friendly",
            line_numbers=True,
            line_number_start=primera_linea,
            line_number_bg="#f6f5f0",
            line_number_fg="#8a8880",
            line_pad=4,
            image_pad=12,
        ),
    )
    import io

    return Image.open(io.BytesIO(png)).convert("RGB")


def figura(nombre: str, archivo: str, rangos: list[tuple[str, str, int]], lexer=None) -> None:
    ruta = RAIZ / archivo
    lineas = ruta.read_text(encoding="utf-8").splitlines()
    lexer = lexer or PythonLexer()

    piezas = []
    for inicio, fin, extra in rangos:
        i, j = _rango(lineas, inicio, fin, extra)
        piezas.append(_render("\n".join(lineas[i:j]), lexer, i + 1))

    ancho = max(p.width for p in piezas)
    cabecera, separador = 34, 26
    alto = cabecera + sum(p.height for p in piezas) + separador * (len(piezas) - 1) + 6
    lienzo = Image.new("RGB", (ancho, alto), "#ffffff")
    dibujo = ImageDraw.Draw(lienzo)

    fuente = ImageFont.truetype("/usr/share/fonts/liberation/LiberationMono-Regular.ttf", 14)
    dibujo.rectangle([0, 0, ancho, cabecera - 6], fill="#f6f5f0")
    dibujo.text((14, 7), archivo, fill="#52514e", font=fuente)

    y = cabecera
    for k, pieza in enumerate(piezas):
        if k:
            dibujo.text((56, y - separador + 4), "···", fill="#8a8880", font=fuente)
        lienzo.paste(pieza, (0, y))
        y += pieza.height + separador

    lienzo.save(FIGDIR / f"{nombre}.png", dpi=(200, 200))
    print(f"{nombre}.png  ({archivo})")


if __name__ == "__main__":
    figura("codigo_codificador", "src/codificadores.py", [
        ("class CodificadorPorObjetivo", "return self", 0),
    ])
    figura("codigo_cv", "scripts/ejecutar_experimento.py", [
        ("def evaluar(X, y, estratos", "return pd.DataFrame(filas)", 0),
    ])
    figura("codigo_equidad", "src/equidad.py", [
        ("def auditar(y_real", '"mediana_pred"', 1),
    ])
    figura("codigo_compuerta", "servicio/fiabilidad.py", [
        ("def evaluar(pais", '"resumen": "Sin datos para este país.",', 1),
        ("if not respalda_rol and not cumple_perdida:",
         '"La estimación se apoya en datos suficientes', 0),
    ])
    figura("codigo_api", "servicio/app.py", [
        ('@app.post("/api/estimar")', 'respuesta["factores"] = factores(fila)', 1),
    ])
    figura("codigo_dockerfile", "servicio/Dockerfile", [
        ("El contexto de construcción", 'CMD ["uvicorn"', 0),
    ], lexer=DockerLexer())
    figura("codigo_test_paridad", "servicio/test_paridad.py", [
        ("# Un céntimo", "entrenamiento {caso['estimacion']:,.2f}", 1),
    ])
    figura("codigo_compose", "servicio/compose.yml", [
        ("# Proyecto aparte", "external: true", 0),
    ], lexer=YamlLexer())
