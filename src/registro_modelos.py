"""Configuraciones de modelo del experimento.

Hiperparámetros y semilla en un solo sitio, para que todos los guiones
entrenen lo mismo. Los ensamblados de terceros son opcionales: si la
biblioteca no está instalada, no se ofrecen.
"""

import warnings

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None
try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None
try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

SEMILLA = 42


def construir_modelos():
    modelos = {
        "Ridge": Ridge(alpha=1.0),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_depth=10, learning_rate=0.1, max_iter=300, random_state=SEMILLA),
    }
    if XGBRegressor is not None:
        modelos["XGBoost"] = XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=SEMILLA, n_jobs=-1)
    if LGBMRegressor is not None:
        modelos["LightGBM"] = LGBMRegressor(
            n_estimators=500, num_leaves=64, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=SEMILLA, verbose=-1)
    if CatBoostRegressor is not None:
        modelos["CatBoost"] = CatBoostRegressor(
            depth=8, learning_rate=0.05, iterations=600, loss_function="RMSE",
            verbose=False, random_seed=SEMILLA)
    return modelos
