"""Pruebas del servicio, sin levantar un servidor.

Se ejercitan los tres casos que el documento describe: un mercado con datos
suficientes, uno de renta media con el rol respaldado, y uno donde el servicio
debe negarse a dar una cifra.
"""

from fastapi.testclient import TestClient

import app as servicio

cliente = TestClient(servicio.app)

ALEMANIA = {"pais": "Germany", "rol": "Developer, back-end",
            "anios_profesionales": 8, "lenguajes": ["python", "javascript"]}
BRASIL = {**ALEMANIA, "pais": "Brazil"}
PERU = {**ALEMANIA, "pais": "Peru"}


def test_salud_declara_el_modelo():
    d = cliente.get("/api/salud").json()
    assert d["estado"] == "ok"
    assert d["modelo"]["n_muestra"] == 37622
    assert d["modelo"]["semilla"] == 42


def test_contexto_trae_los_catalogos():
    d = cliente.get("/api/contexto").json()
    assert len(d["catalogos"]["Country"]) == 78
    assert d["umbral_zeta"] > 0


def test_mercado_con_datos_entrega_banda_y_factores():
    d = cliente.post("/api/estimar", json=ALEMANIA).json()
    assert d["fiabilidad"]["nivel"] == "verde"
    assert d["banda"]["inferior"] < d["banda"]["centro"] < d["banda"]["superior"]
    factores = {f["factor"] for f in d["factores"]}
    assert "País de residencia" in factores
    assert len(d["factores"]) >= 3


def test_renta_media_entrega_con_reservas():
    d = cliente.post("/api/estimar", json=BRASIL).json()
    assert d["fiabilidad"]["nivel"] == "ambar"
    assert d["banda"] is not None
    assert d["fiabilidad"]["motivos"]


def test_peru_no_entrega_cifra_pero_explica():
    d = cliente.post("/api/estimar", json=PERU).json()
    assert d["fiabilidad"]["nivel"] == "rojo"
    assert d["banda"] is None
    assert d["fiabilidad"]["motivos"]
    # La mediana observada sí se informa: procede de datos, no del modelo.
    assert d["mediana_pais"] > 0


def test_la_banda_es_mas_ancha_donde_el_error_es_mayor():
    a = cliente.post("/api/estimar", json=ALEMANIA).json()["banda"]
    b = cliente.post("/api/estimar", json=BRASIL).json()["banda"]
    assert b["amplitud_relativa"] > a["amplitud_relativa"]


def test_comparar_devuelve_una_fila_por_pais():
    d = cliente.post("/api/comparar", json={
        "perfil": ALEMANIA,
        "paises": ["Brazil", "India", "United States of America", "Peru"]}).json()
    paises = [r["pais"] for r in d["resultados"]]
    assert paises[0] == "Germany"          # el propio siempre encabeza
    assert len(paises) == 5 and len(set(paises)) == 5
    assert any(r["banda"] is None for r in d["resultados"])   # Perú, sin respaldo


def test_comparar_sin_paises_es_un_error_de_peticion():
    assert cliente.post("/api/comparar",
                        json={"perfil": ALEMANIA, "paises": []}).status_code == 400


def test_el_mismo_perfil_cambia_de_cifra_al_cambiar_de_pais():
    d = cliente.post("/api/comparar", json={
        "perfil": ALEMANIA, "paises": ["India"]}).json()["resultados"]
    centros = [r["banda"]["centro"] for r in d if r["banda"]]
    assert max(centros) / min(centros) > 2      # la geografía domina


def test_la_portada_se_sirve():
    r = cliente.get("/")
    assert r.status_code == 200 and "Estimación salarial" in r.text


def test_roles_indica_el_respaldo_de_cada_uno():
    d = cliente.get("/api/roles", params={"pais": "Germany"}).json()
    assert d["n_pais"] == 3192 and d["con_respaldo"] > 5
    assert d["roles"][0]["n"] >= d["roles"][-1]["n"]        # ordenados por respaldo
    assert all("respaldado" in r for r in d["roles"])


def test_peru_no_tiene_ningun_rol_respaldado():
    d = cliente.get("/api/roles", params={"pais": "Peru"}).json()
    assert d["n_pais"] == 48 and d["con_respaldo"] == 0


def test_sin_rol_la_estimacion_se_apoya_en_el_pais():
    # Sin rol declarado no hay criterio de rol que incumplir.
    d = cliente.post("/api/estimar", json={"pais": "Germany",
                                           "anios_profesionales": 8}).json()
    assert d["fiabilidad"]["nivel"] == "verde"
    assert d["banda"] is not None
    assert any("No se ha indicado el rol" in m for m in d["fiabilidad"]["motivos"])
