import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error

def compute_metrics(y_true, y_pred, prefix=''):
    rmse_log = np.sqrt(mean_squared_error(y_true, y_pred))
    mae_log = mean_absolute_error(y_true, y_pred)
    medae_log = median_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    y_true_usd = np.expm1(y_true)
    y_pred_usd = np.expm1(y_pred)
    rmse_usd = np.sqrt(mean_squared_error(y_true_usd, y_pred_usd))
    mae_usd = mean_absolute_error(y_true_usd, y_pred_usd)
    medae_usd = median_absolute_error(y_true_usd, y_pred_usd)
    mask = y_true_usd > 0
    mape = float(np.mean(np.abs((y_true_usd[mask] - y_pred_usd[mask]) / y_true_usd[mask])) * 100)
    return {
        f'{prefix}RMSE_log': rmse_log,
        f'{prefix}MAE_log': mae_log,
        f'{prefix}MedAE_log': medae_log,
        f'{prefix}R2': r2,
        f'{prefix}RMSE_USD': rmse_usd,
        f'{prefix}MAE_USD': mae_usd,
        f'{prefix}MedAE_USD': medae_usd,
        f'{prefix}MAPE': mape
    }
