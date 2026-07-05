"""Métricas de desempeño en regresión, en ambas escalas.

El modelo se ajusta sobre el logaritmo de la compensación, pero los errores
solo admiten lectura sustantiva en dólares, de modo que cada evaluación
devuelve las dos familias a la vez (§3.6.3 de la tesis). Las claves conservan
las siglas convencionales (RMSE, MAE, MAPE) porque así se llaman también en la
literatura en español y son las columnas de los artefactos de resultados.
"""

import numpy as np
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             median_absolute_error, r2_score)


def calcular_metricas(y_real, y_pred, prefijo=""):
    rmse_log = np.sqrt(mean_squared_error(y_real, y_pred))
    mae_log = mean_absolute_error(y_real, y_pred)
    medae_log = median_absolute_error(y_real, y_pred)
    r2 = r2_score(y_real, y_pred)

    # Reconversión a dólares: expm1 es la inversa exacta de log1p.
    y_real_usd = np.expm1(y_real)
    y_pred_usd = np.expm1(y_pred)
    rmse_usd = np.sqrt(mean_squared_error(y_real_usd, y_pred_usd))
    mae_usd = mean_absolute_error(y_real_usd, y_pred_usd)
    medae_usd = median_absolute_error(y_real_usd, y_pred_usd)
    con_valor = y_real_usd > 0
    mape = float(np.mean(np.abs(
        (y_real_usd[con_valor] - y_pred_usd[con_valor]) / y_real_usd[con_valor])) * 100)

    return {
        f"{prefijo}RMSE_log": rmse_log,
        f"{prefijo}MAE_log": mae_log,
        f"{prefijo}MedAE_log": medae_log,
        f"{prefijo}R2": r2,
        f"{prefijo}RMSE_USD": rmse_usd,
        f"{prefijo}MAE_USD": mae_usd,
        f"{prefijo}MedAE_USD": medae_usd,
        f"{prefijo}MAPE": mape,
    }
