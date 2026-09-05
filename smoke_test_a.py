#!/usr/bin/env python3
"""Smoke test A: validación rápida del pipeline en <2 minutos.

Ejecutar desde root del repo:
    python smoke_test_a.py
"""
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import Pipeline

from src.feature_pipeline import load_processed, precompute_text_embeddings, build_preprocessor, NUM_FEATURES, CAT_LOW, CAT_HIGH
from src.metrics import compute_metrics

SEED = 42


def run_smoke_a(output_dir: str):
    """Ejecuta smoke test con Ridge + target encoding, 3 splits CV."""
    # Cargar solo 1000 filas
    print("Loading data (first 1000 rows)...")
    df = load_processed().head(1000)

    # Precomputar embeddings (solo datos de texto)
    df, emb_cols = precompute_text_embeddings(df)

    all_features = NUM_FEATURES + CAT_LOW + CAT_HIGH + emb_cols
    X = df[all_features].copy()
    y = df['salary_log'].astype(np.float32)

    # Split train/test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=df['region']
    )

    # Preprocesador con target encoding
    preprocessor = build_preprocessor('target', emb_cols)

    # Solo Ridge
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=1.0)
    pipe = Pipeline([('preprocessor', preprocessor), ('model', model)])

    # CV con 3 splits y n_jobs=2
    print("Running 3-fold CV with n_jobs=2...")
    kfold = KFold(n_splits=3, shuffle=True, random_state=SEED)
    scoring = {
        'rmse': 'neg_root_mean_squared_error',
        'mae': 'neg_mean_absolute_error',
        'r2': 'r2'
    }
    cv_res = cross_validate(pipe, X_train_full, y_train_full, cv=kfold, scoring=scoring, n_jobs=2)

    cv_row = {
        'encoding': 'target',
        'Modelo': 'Ridge',
        'RMSE_mean': float(-cv_res['test_rmse'].mean()),
        'RMSE_std': float(cv_res['test_rmse'].std()),
        'MAE_mean': float(-cv_res['test_mae'].mean()),
        'MAE_std': float(cv_res['test_mae'].std()),
        'R2_mean': float(cv_res['test_r2'].mean()),
        'R2_std': float(cv_res['test_r2'].std()),
    }

    # Fit final y métricas de test
    print("Fitting final model on full training set...")
    pipe.fit(X_train_full, y_train_full)
    y_pred = pipe.predict(X_test)
    test_metrics = compute_metrics(y_test, y_pred, prefix='test_')
    test_metrics['encoding'] = 'target'
    test_metrics['Modelo'] = 'Ridge'

    # Guardar resultados
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame([cv_row]).to_csv(os.path.join(output_dir, 'cv_results.csv'), index=False)
    pd.DataFrame([test_metrics]).to_csv(os.path.join(output_dir, 'test_results.csv'), index=False)

    return cv_row, test_metrics


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'results/smoke_a_{timestamp}'

    cv_row, test_metrics = run_smoke_a(output_dir)

    rmse_val = cv_row['RMSE_mean']
    rows_processed = 1000

    print(f"\nSMOKE A OK: {rows_processed} filas, Ridge, target encoding, RMSE={rmse_val:.4f}")
    print(f"Results saved in: {output_dir}/")


if __name__ == '__main__':
    main()
