"""Comprueba que el servicio reproduce las cifras del entorno de entrenamiento.

El modelo se entrena con Python 3.14 y las versiones que fija el
`requirements.txt` del análisis, pero el servidor donde se despliega expone una
CPU virtual sin las instrucciones que exigen los wheels recientes de NumPy, de
modo que el servicio corre con Python 3.12 y NumPy 2.0.

Esa divergencia hay que vigilarla, no suponerla inocua: `referencia.json` guarda
las estimaciones obtenidas en el entorno de entrenamiento y esta prueba las
contrasta con las que produce el de ejecución. Se ejecuta al construir la
imagen, de modo que una imagen que no reproduzca las cifras no llega a existir.
"""

import json
from pathlib import Path

import app as servicio

REFERENCIA = json.loads((Path(__file__).parent / "artefactos" / "referencia.json")
                        .read_text(encoding="utf-8"))

# Un céntimo sobre cifras de decenas de miles de dólares: se admite la
# diferencia de redondeo en coma flotante, no un cambio de resultado.
TOLERANCIA = 0.01


def test_las_estimaciones_coinciden_con_el_entorno_de_entrenamiento():
    for nombre, caso in REFERENCIA.items():
        fila = servicio.construir_fila(servicio.Perfil(**caso["perfil"]))
        obtenida = servicio.estimar(fila)
        assert abs(obtenida - caso["estimacion"]) < TOLERANCIA, (
            f"{nombre}: el servicio estima {obtenida:,.2f} y el entorno de "
            f"entrenamiento {caso['estimacion']:,.2f}")


def test_las_contribuciones_coinciden():
    for nombre, caso in REFERENCIA.items():
        fila = servicio.construir_fila(servicio.Perfil(**caso["perfil"]))
        obtenidos = {f["factor"]: f["efecto"] for f in servicio.factores(fila)}
        for esperado in caso["factores"]:
            assert esperado["factor"] in obtenidos, f"{nombre}: falta {esperado['factor']}"
            assert abs(obtenidos[esperado["factor"]] - esperado["efecto"]) < 0.001
