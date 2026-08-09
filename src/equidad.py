"""Auditoría de equidad para regresión.

Métricas sobre la variable continua (Agarwal et al., 2019): error por grupo,
razón de disparidad, sesgo con signo y paridad de las predicciones (KS). No se
binariza el salario. El veredicto se apoya en el error relativo a la mediana
del grupo: el absoluto no es comparable entre mercados de escala distinta.
Definiciones en el capítulo 2 de la tesis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Análogo a la regla de los cuatro quintos.
UMBRAL_RAZON = 1.25

# Por debajo el grupo se reporta, pero solo como descriptivo.
N_MIN_GRUPO = 30


def auditar(y_real, y_pred, atributo: pd.Series, en_escala_log: bool = True,
            umbral_zeta: float | None = None) -> dict:
    """Audita el modelo sobre los grupos que define `atributo`.

    Con `en_escala_log` se vuelve a dólares antes de medir nada: en escala
    log el error no tiene lectura sustantiva.
    """
    y_real = np.asarray(y_real, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if en_escala_log:
        y_real, y_pred = np.expm1(y_real), np.expm1(y_pred)

    datos = pd.DataFrame({
        "real": y_real,
        "pred": y_pred,
        "grupo": np.asarray(atributo).ravel(),
    }).dropna(subset=["grupo"])

    por_grupo = {}
    for grupo, sub in datos.groupby("grupo", observed=True):
        error = sub["pred"] - sub["real"]
        mediana = float(sub["real"].median())
        mae = float(error.abs().mean())
        por_grupo[str(grupo)] = {
            "n": int(len(sub)),
            "suficiente_para_inferencia": bool(len(sub) >= N_MIN_GRUPO),
            "mae": mae,
            # Normalizado por la mediana del grupo: comparable entre escalas.
            "mae_relativo": float(mae / mediana) if mediana > 0 else None,
            "mape": float((error.abs() / sub["real"].clip(lower=1)).mean()),
            "rmse": float(np.sqrt((error ** 2).mean())),
            "sesgo_sistematico": float(error.mean()),
            "sesgo_relativo": float(error.mean() / mediana) if mediana > 0 else None,
            "mediana_real": mediana,
            "mediana_pred": float(sub["pred"].median()),
        }

    # Agregados solo con grupos de tamaño suficiente; uno minúsculo infla la
    # razón de disparidad por pura varianza muestral.
    validos = {g: v for g, v in por_grupo.items() if v["suficiente_para_inferencia"]}
    resultado = {
        "n_total": int(len(datos)),
        "n_grupos": len(por_grupo),
        "n_grupos_suficientes": len(validos),
        "por_grupo": por_grupo,
    }

    if len(validos) >= 2:
        maes = np.array([v["mae"] for v in validos.values()])
        rel = np.array([v["mae_relativo"] for v in validos.values()], dtype=float)
        sesgos = np.array([v["sesgo_sistematico"] for v in validos.values()])
        peor_abs = max(validos, key=lambda g: validos[g]["mae"])
        peor_rel = max(validos, key=lambda g: validos[g]["mae_relativo"])
        resultado["agregados"] = {
            # Absolutos: solo de referencia.
            "disparidad_absoluta": float(maes.max() - maes.min()),
            "razon_disparidad_absoluta": float(maes.max() / maes.min()),
            "grupo_peor_servido_absoluto": peor_abs,
            # Relativos: los que deciden el veredicto.
            "razon_disparidad_relativa": float(np.nanmax(rel) / np.nanmin(rel)),
            "coeficiente_variacion_relativo": float(np.nanstd(rel) / np.nanmean(rel)),
            "grupo_peor_servido_relativo": peor_rel,
            "supera_umbral_razon": bool(np.nanmax(rel) / np.nanmin(rel) > UMBRAL_RAZON),
            "invierte_al_normalizar": bool(peor_abs != peor_rel),
            "sesgo_maximo_absoluto": float(np.abs(sesgos).max()),
            "sesgo_todos_mismo_signo": bool(np.all(sesgos < 0) or np.all(sesgos > 0)),
            "rango_sesgo": float(sesgos.max() - sesgos.min()),
        }
        if umbral_zeta is not None:
            resultado["agregados"]["perdida_acotada_cumple"] = bool(maes.max() <= umbral_zeta)
            resultado["agregados"]["umbral_zeta"] = float(umbral_zeta)

    resultado["contrastes"] = _contrastar(datos, validos)
    return resultado


def _contrastar(datos: pd.DataFrame, validos: dict) -> dict:
    """Mann-Whitney (dos grupos) o Kruskal-Wallis (más) sobre el error con signo.

    No paramétricas porque el error no es normal. Añade la distancia KS entre
    las predicciones de los dos grupos extremos.
    """
    from scipy.stats import kruskal, ks_2samp, mannwhitneyu

    grupos = list(validos)
    if len(grupos) < 2:
        return {}

    errores = {g: (datos.loc[datos.grupo.astype(str) == g, "pred"]
                   - datos.loc[datos.grupo.astype(str) == g, "real"]).to_numpy()
               for g in grupos}

    salida = {}
    if len(grupos) == 2:
        a, b = grupos
        u, p = mannwhitneyu(errores[a], errores[b], alternative="two-sided")
        salida["prueba"] = "Mann-Whitney"
        salida["estadistico"] = float(u)
        salida["p_valor"] = float(p)
    else:
        h, p = kruskal(*errores.values())
        salida["prueba"] = "Kruskal-Wallis"
        salida["estadistico"] = float(h)
        salida["p_valor"] = float(p)
    salida["significativo"] = bool(salida["p_valor"] < 0.05)

    # Paridad estadística entre el grupo con predicción mediana más alta y el más bajo.
    medianas = {g: validos[g]["mediana_pred"] for g in grupos}
    alto = max(medianas, key=medianas.get)
    bajo = min(medianas, key=medianas.get)
    if alto != bajo:
        pred_alto = datos.loc[datos.grupo.astype(str) == alto, "pred"]
        pred_bajo = datos.loc[datos.grupo.astype(str) == bajo, "pred"]
        ks = ks_2samp(pred_alto, pred_bajo)
        salida["paridad_estadistica"] = {
            "grupo_prediccion_alta": alto,
            "grupo_prediccion_baja": bajo,
            "distancia_ks": float(ks.statistic),
            "p_valor": float(ks.pvalue),
        }
    return salida


def resumir(auditoria: dict, titulo: str = "") -> str:
    """Informe legible de una auditoría."""
    l = [f"Auditoría de equidad — {titulo}" if titulo else "Auditoría de equidad",
         f"  N = {auditoria['n_total']:,} · grupos: {auditoria['n_grupos']} "
         f"({auditoria['n_grupos_suficientes']} con tamaño suficiente)", ""]
    l.append(f"  {'grupo':26s} {'N':>6s} {'MAE':>10s} {'MAE/mediana':>12s} {'sesgo':>11s}")
    for g, v in sorted(auditoria["por_grupo"].items(),
                       key=lambda kv: -(kv[1]["mae_relativo"] or 0)):
        marca = "" if v["suficiente_para_inferencia"] else "  (descriptivo)"
        rel = f"{v['mae_relativo']*100:>11.1f} %" if v["mae_relativo"] else "          —"
        l.append(f"  {g[:26]:26s} {v['n']:>6,} {v['mae']:>10,.0f} {rel} "
                 f"{v['sesgo_sistematico']:>+11,.0f}{marca}")

    a = auditoria.get("agregados")
    if a:
        l += ["",
              f"  Razón de disparidad RELATIVA .. {a['razon_disparidad_relativa']:.3f}"
              f"   {'SUPERA el umbral de 1.25' if a['supera_umbral_razon'] else 'dentro del umbral'}",
              f"  Peor servido (relativo) ....... {a['grupo_peor_servido_relativo']}",
              f"  Coef. de variación relativo ... {a['coeficiente_variacion_relativo']:.3f}",
              "",
              f"  [Referencia] razón absoluta ... {a['razon_disparidad_absoluta']:.3f}"
              f"  · peor servido: {a['grupo_peor_servido_absoluto']}"]
        if a["invierte_al_normalizar"]:
            l.append("  ⚠ El grupo peor servido cambia al normalizar por la escala "
                     "salarial:\n    el error absoluto no es comparable entre estos grupos.")
        l.append(f"  Sesgo máximo absoluto ......... ${a['sesgo_maximo_absoluto']:,.0f}"
                 + ("  · todos los grupos en la misma dirección"
                    if a["sesgo_todos_mismo_signo"] else ""))

    c = auditoria.get("contrastes")
    if c and "p_valor" in c:
        l += ["", f"  {c['prueba']}: p = {c['p_valor']:.3e}"
              f"   {'diferencia significativa' if c['significativo'] else 'sin diferencia significativa'}"]
        pe = c.get("paridad_estadistica")
        if pe:
            l.append(f"  Paridad estadística (KS): D = {pe['distancia_ks']:.3f} "
                     f"entre «{pe['grupo_prediccion_alta'][:20]}» y «{pe['grupo_prediccion_baja'][:20]}»")
    return "\n".join(l)
