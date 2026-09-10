import warnings
warnings.filterwarnings('ignore')
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
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, StackingRegressor

SEED = 42

def build_models():
    models = {
        'Ridge': Ridge(alpha=1.0),
        'HistGradientBoosting': HistGradientBoostingRegressor(max_depth=10, learning_rate=0.1, max_iter=300, random_state=SEED),
    }
    if XGBRegressor is not None:
        models['XGBoost'] = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1)
    if LGBMRegressor is not None:
        models['LightGBM'] = LGBMRegressor(n_estimators=500, num_leaves=64, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=SEED, verbose=-1)
    if CatBoostRegressor is not None:
        models['CatBoost'] = CatBoostRegressor(depth=8, learning_rate=0.05, iterations=600, loss_function='RMSE', verbose=False, random_seed=SEED)
    return models

def build_stacking(models=None):
    stacking_models = [('ridge', Ridge(alpha=1.0)), ('histgb', HistGradientBoostingRegressor(max_depth=10, learning_rate=0.1, max_iter=300, random_state=SEED))]
    if CatBoostRegressor is not None:
        stacking_models.append(('cat', CatBoostRegressor(depth=8, learning_rate=0.05, iterations=600, loss_function='RMSE', verbose=False, random_seed=SEED)))
    return StackingRegressor(estimators=stacking_models, final_estimator=Ridge(alpha=0.5), passthrough=True)
