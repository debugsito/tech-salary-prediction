"""
Script de entrenamiento rápido para pipeline reducido.
- ~10k filas (configurable via --sample-size)
- Solo Ridge y Stacking (2 mejores candidatos)
- 3-fold CV
- Resultados en results/quick_run/
"""

import sys
import os

# Allow direct execution: python src/train_quick.py
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import pandas as pd
import argparse
import json
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, StackingRegressor
try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

from src.feature_pipeline import load_processed, precompute_text_embeddings, build_preprocessor, NUM_FEATURES, CAT_LOW, CAT_HIGH
from src.metrics import compute_metrics

SEED = 42


def build_stacking_quick():
    """Build Stacking with available estimators (Ridge + HistGB + CatBoost if available)."""
    estimators = [
        ('ridge', Ridge(alpha=1.0)),
        ('histgb', HistGradientBoostingRegressor(max_depth=10, learning_rate=0.1, max_iter=300, random_state=SEED))
    ]
    if CatBoostRegressor is not None:
        estimators.append(
            ('cat', CatBoostRegressor(depth=8, learning_rate=0.05, iterations=600, loss_function='RMSE', verbose=False, random_seed=SEED))
        )
    return StackingRegressor(estimators=estimators, final_estimator=Ridge(alpha=0.5), passthrough=True)


def run_quick_variant(country_encoding, X_train_full, y_train_full, X_test, y_test, emb_cols,
                      n_jobs=2, output_dir='results/quick_run'):
    """Run quick training with only Ridge and Stacking, 3-fold CV."""
    preprocessor = build_preprocessor(country_encoding, emb_cols)
    
    # Only 2 best candidates: Ridge and Stacking
    models = {
        'Ridge': Ridge(alpha=1.0),
        'Stacking': build_stacking_quick()
    }
    
    cv_rows, test_rows = [], []
    kfold = KFold(n_splits=3, shuffle=True, random_state=SEED)  # 3-fold instead of 5
    scoring = {'rmse': 'neg_root_mean_squared_error', 'mae': 'neg_mean_absolute_error', 'r2': 'r2'}

    for name, model in models.items():
        print(f"  Training {name}...")
        pipe = Pipeline([('preprocessor', preprocessor), ('model', model)])
        
        # Cross-validation
        cv_res = cross_validate(pipe, X_train_full, y_train_full, cv=kfold, scoring=scoring, n_jobs=n_jobs)
        rmse = -cv_res['test_rmse']
        mae = -cv_res['test_mae']
        r2 = cv_res['test_r2']
        
        cv_rows.append({
            'encoding': country_encoding,
            'Modelo': name,
            'RMSE_mean': rmse.mean(),
            'RMSE_std': rmse.std(),
            'MAE_mean': mae.mean(),
            'MAE_std': mae.std(),
            'R2_mean': r2.mean(),
            'R2_std': r2.std()
        })
        
        # Test evaluation
        pipe.fit(X_train_full, y_train_full)
        y_pred = pipe.predict(X_test)
        m = compute_metrics(y_test, y_pred, prefix='test_')
        m['encoding'] = country_encoding
        m['Modelo'] = name
        test_rows.append(m)
        
        print(f"    CV R²: {r2.mean():.4f} (±{r2.std():.4f})")
        print(f"    Test R²: {m.get('test_r2', float('nan')):.4f}")

    return cv_rows, test_rows


def main():
    parser = argparse.ArgumentParser(description='Entrenamiento rápido de modelos de predicción de salarios tech.')
    parser.add_argument('--variant', type=str, default='target', choices=['onehot', 'target'],
                        help="Codificación para country: 'onehot' o 'target' (default: target).")
    parser.add_argument('--sample-size', type=int, default=10000,
                        help='Número de filas a usar para entrenamiento rápido (default: 10000).')
    parser.add_argument('--n_jobs', type=int, default=2,
                        help='Número de jobs para cross-validation (default: 2).')
    args = parser.parse_args()

    output_dir = 'results/quick_run'
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading data...")
    df = load_processed()
    
    # Sample if needed
    if args.sample_size > 0 and args.sample_size < len(df):
        print(f"Sampling {args.sample_size} rows from {len(df)} total rows...")
        df = df.sample(n=args.sample_size, random_state=SEED).reset_index(drop=True)
    else:
        print(f"Using all {len(df)} rows (sample-size >= dataset size)")

    print(f"Precomputing embeddings...")
    df, emb_cols = precompute_text_embeddings(df)
    
    all_features = NUM_FEATURES + CAT_LOW + CAT_HIGH + emb_cols
    X = df[all_features].copy()
    y = df['salary_log'].astype(np.float32)
    
    print(f"Splitting data (80/20)...")
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=df['region']
    )
    
    print(f"\n=== Quick Training Run ===")
    print(f"Dataset size: {len(df)} rows")
    print(f"Train: {len(X_train_full)} | Test: {len(X_test)}")
    print(f"Encoding: {args.variant}")
    print(f"Models: Ridge, Stacking")
    print(f"CV: 3-fold")
    print(f"Output: {output_dir}/")
    print(f"----------------------------\n")

    cv_rows, test_rows = run_quick_variant(
        args.variant, X_train_full, y_train_full, X_test, y_test, emb_cols,
        n_jobs=args.n_jobs, output_dir=output_dir
    )

    # Save results
    cv_path = os.path.join(output_dir, 'quick_cv_by_encoding.csv')
    test_path = os.path.join(output_dir, 'quick_test_by_encoding.csv')
    
    pd.DataFrame(cv_rows).to_csv(cv_path, index=False)
    pd.DataFrame(test_rows).to_csv(test_path, index=False)
    
    print(f"\n=== Results saved ===")
    print(f"CV metrics: {cv_path}")
    print(f"Test metrics: {test_path}")
    
    # Print summary table
    print(f"\n=== CV Metrics Summary ===")
    print(pd.DataFrame(cv_rows)[['Modelo', 'R2_mean', 'R2_std', 'RMSE_mean', 'MAE_mean']].to_string(index=False))
    
    print(f"\n=== Test Metrics Summary ===")
    test_df = pd.DataFrame(test_rows)
    cols = ['Modelo', 'test_r2', 'test_mae', 'test_rmse']
    print(test_df[[c for c in cols if c in test_df.columns]].to_string(index=False))


if __name__ == '__main__':
    main()
