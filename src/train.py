import sys
import os

# Allow direct execution: python src/train.py
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import Pipeline
from src.feature_pipeline import load_processed, precompute_text_embeddings, build_preprocessor, NUM_FEATURES, CAT_LOW, CAT_HIGH
from src.model_registry import build_models, build_stacking
from src.metrics import compute_metrics

SEED = 42

def run_variant(country_encoding, X_train_full, y_train_full, X_test, y_test, emb_cols, n_jobs=2):
    preprocessor = build_preprocessor(country_encoding, emb_cols)
    models = build_models()
    models['Stacking'] = build_stacking()
    cv_rows, test_rows = [], []
    kfold = KFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = {'rmse': 'neg_root_mean_squared_error', 'mae': 'neg_mean_absolute_error', 'r2': 'r2'}
    for name, model in models.items():
        pipe = Pipeline([('preprocessor', preprocessor), ('model', model)])
        cv_res = cross_validate(pipe, X_train_full, y_train_full, cv=kfold, scoring=scoring, n_jobs=n_jobs)
        rmse = -cv_res['test_rmse']
        mae = -cv_res['test_mae']
        r2 = cv_res['test_r2']
        cv_rows.append({'encoding': country_encoding, 'Modelo': name, 'RMSE_mean': rmse.mean(), 'RMSE_std': rmse.std(), 'MAE_mean': mae.mean(), 'MAE_std': mae.std(), 'R2_mean': r2.mean(), 'R2_std': r2.std()})
        pipe.fit(X_train_full, y_train_full)
        y_pred = pipe.predict(X_test)
        m = compute_metrics(y_test, y_pred, prefix='test_')
        m['encoding'] = country_encoding
        m['Modelo'] = name
        test_rows.append(m)
    return cv_rows, test_rows

def main():
    df = load_processed()
    df, emb_cols = precompute_text_embeddings(df)
    all_features = NUM_FEATURES + CAT_LOW + CAT_HIGH + emb_cols
    X = df[all_features].copy()
    y = df['salary_log'].astype(np.float32)
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=df['region'])
    all_cv, all_test = [], []
    for encoding in ['onehot', 'target']:
        print(f'--- Variante: {encoding} ---')
        cv_rows, test_rows = run_variant(encoding, X_train_full, y_train_full, X_test, y_test, emb_cols)
        all_cv.extend(cv_rows)
        all_test.extend(test_rows)
    os.makedirs('results', exist_ok=True)
    pd.DataFrame(all_cv).to_csv('results/cv_by_encoding.csv', index=False)
    pd.DataFrame(all_test).to_csv('results/test_by_encoding.csv', index=False)
    print('Guardado results/cv_by_encoding.csv y results/test_by_encoding.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entrenamiento de modelos de predicción de salarios tech.')
    parser.add_argument('--variant', type=str, default='target', choices=['onehot', 'target'],
                        help="Codificación para country: 'onehot' o 'target' (default: target).")
    parser.add_argument('--n_jobs', type=int, default=2,
                        help='Número de jobs para cross-validation (default: 2).')
    args = parser.parse_args()
    
    df = load_processed()
    df, emb_cols = precompute_text_embeddings(df)
    all_features = NUM_FEATURES + CAT_LOW + CAT_HIGH + emb_cols
    X = df[all_features].copy()
    y = df['salary_log'].astype(np.float32)
    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=df['region'])
    
    print(f'--- Variante: {args.variant} ---')
    cv_rows, test_rows = run_variant(args.variant, X_train_full, y_train_full, X_test, y_test, emb_cols, n_jobs=args.n_jobs)
    
    os.makedirs('results', exist_ok=True)
    pd.DataFrame(cv_rows).to_csv('results/cv_by_encoding.csv', index=False)
    pd.DataFrame(test_rows).to_csv('results/test_by_encoding.csv', index=False)
    print('Guardado results/cv_by_encoding.csv y results/test_by_encoding.csv')
