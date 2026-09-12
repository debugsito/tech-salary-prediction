"""Pruebas de la compuerta de fiabilidad.

Se comprueban los casos que el documento cita, para que un cambio en los
umbrales no pase inadvertido.
"""

import json
from pathlib import Path

import fiabilidad as f

CONTEXTO = json.loads((Path(__file__).parent / "artefactos" / "contexto.json")
                      .read_text(encoding="utf-8"))


def rol_frecuente(pais):
    celdas = {k.split("||")[1]: v for k, v in CONTEXTO["fiabilidad"]["n_pais_rol"].items()
              if k.startswith(f"{pais}||")}
    return max(celdas, key=celdas.get) if celdas else None


def test_umbral_procede_del_grupo_mejor_servido():
    err = CONTEXTO["fiabilidad"]["error_relativo_grupo"]
    assert abs(f.umbral_zeta(err) - min(err.values()) * 1.25) < 1e-9


def test_mercado_con_datos_sale_verde():
    r = f.evaluar("Germany", rol_frecuente("Germany"), CONTEXTO)
    assert r["nivel"] == f.VERDE and r["entregar"]
    assert not r["motivos"]


def test_renta_media_con_rol_respaldado_sale_ambar():
    r = f.evaluar("Brazil", rol_frecuente("Brazil"), CONTEXTO)
    assert r["nivel"] == f.AMBAR and r["entregar"]
    assert any("pérdida acotada" in m for m in r["motivos"])


def test_peru_no_entrega_cifra():
    # 48 respuestas en el país y ninguna combinación de rol que llegue a 30.
    r = f.evaluar("Peru", "Developer, back-end", CONTEXTO)
    assert r["nivel"] == f.ROJO and not r["entregar"]
    assert r["n_pais"] == 48


def test_pais_fuera_de_la_muestra():
    r = f.evaluar("Narnia", "Developer, back-end", CONTEXTO)
    assert r["nivel"] == f.ROJO and not r["entregar"]
    assert r["n_pais"] == 0


def test_la_banda_usa_el_error_del_grupo_y_no_el_global():
    alta = f.banda(100_000, f.evaluar("Germany", rol_frecuente("Germany"), CONTEXTO))
    media = f.banda(100_000, f.evaluar("Brazil", rol_frecuente("Brazil"), CONTEXTO))
    assert media["amplitud_relativa"] > alta["amplitud_relativa"]
    assert alta["inferior"] < 100_000 < alta["superior"]


def test_sin_rol_no_puede_haber_respaldo_de_rol():
    r = f.evaluar("Germany", None, CONTEXTO)
    assert r["n_celda"] == 0
