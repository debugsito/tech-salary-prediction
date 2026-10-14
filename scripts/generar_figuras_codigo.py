#!/usr/bin/env python3
"""Renderiza extractos del código fuente como figuras, con el archivo de origen
de cabecera y los números de línea reales. Los extractos se localizan por
cadenas marcadoras, no por número de línea; si un marcador desaparece el guion
falla, que es preferible a una figura desactualizada.

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
TAMANO = 24  # px a 200 ppp: unos 8.6 pt al tamaño de inserción


def _rango(lineas: list[str], inicio: str, fin: str, extra: int = 0) -> tuple[int, int]:
    """Índices (base 0, fin exclusivo) del bloque entre los dos marcadores."""
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
            line_pad=7,
            image_pad=22,
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
    cabecera, separador = 62, 48
    alto = cabecera + sum(p.height for p in piezas) + separador * (len(piezas) - 1) + 6
    lienzo = Image.new("RGB", (ancho, alto), "#ffffff")
    dibujo = ImageDraw.Draw(lienzo)

    fuente = ImageFont.truetype("/usr/share/fonts/liberation/LiberationMono-Regular.ttf", 26)
    dibujo.rectangle([0, 0, ancho, cabecera - 11], fill="#f6f5f0")
    dibujo.text((26, 13), archivo, fill="#52514e", font=fuente)

    y = cabecera
    for k, pieza in enumerate(piezas):
        if k:
            dibujo.text((100, y - separador + 8), "···", fill="#8a8880", font=fuente)
        # Fondo del tramo a todo el ancho, para que los tramos más cortos no dejen una franja blanca.
        dibujo.rectangle([0, y, ancho, y + pieza.height - 1], fill=pieza.getpixel((pieza.width - 1, 0)))
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
        ('@app.post("/api/estimar"', 'respuesta["factores"] = factores(fila)', 1),
    ])
    figura("codigo_dockerfile", "servicio/Dockerfile", [
        ("Contexto de construcción", 'CMD ["uvicorn"', 0),
    ], lexer=DockerLexer())
    figura("codigo_test_paridad", "servicio/test_paridad.py", [
        ("# Un céntimo", "entrenamiento {caso['estimacion']:,.2f}", 1),
    ])
    figura("codigo_compose", "servicio/compose.yml", [
        ("# Proyecto aparte", "external: true", 0),
    ], lexer=YamlLexer())
