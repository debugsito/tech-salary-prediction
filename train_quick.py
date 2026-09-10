import sys
import os
import numpy as np
import pandas as pd
import argparse
import json
import shutil
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.pipeline import Pipeline
from src.feature_pipeline import load_processed, precompute_text_embeddings, build_preprocessor, NUM_FEATURES, CAT_LOW, CAT_HIGH
from src.model_registry import Ridge, HistGradientBoostingRegressor
from src.metrics import compute_metrics
from src.explainability import explain_model
from src.fairness import audit_fairness, get_fairness_summary

SEED = 42

def run_quick_variant(country_encoding, X_train_full, y_train_full, X_test, y_test, emb_cols,
                     n_jobs=2, explain=False, fairness=False, fairness_attr=None,
                     df_test=None, output_dir='results'):
    preprocessor = build_preprocessor(country_encoding, emb_cols)
    
    # Manual model selection for quick run
    models = {
        'Ridge': Ridge(alpha=1.0),
        'HistGradientBoosting': HistGradientBoostingRegressor(max_depth=10, learning_rate=0.1, max_iter=300, random_state=SEED),
    }
    
    cv_rows, test_rows = [], []
    kfold = KFold(n_splits=3, shuffle=True, random_state=SEED) # 3 folds instead of 5
    scoring = {'rmse': 'neg_root_mean_squared_error', 'mae': 'neg_mean_absolute_error', 'r2': 'r2'}

    best_model_name = None
    best_r2 = -float('inf')
    best_pipe = None

    for name, model in models.items():
        print(f"Training {name}...")
        pipe = Pipeline([('preprocessor', preprocessor), ('model', model)])
        
        # CV
        cv_res = cross_validate(pipe, X_train_full, y_train_full, cv=kfold, scoring=scoring, n_jobs=n_jobs)
        rmse = -cv_res['test_rmse']
        mae = -cv_res['test_mae']
        r2 = cv_res['test_r2']
        cv_rows.append({
            'encoding': country_encoding, 
            'Modelo': name, 
            'RMSE_mean': rmse.mean(), 'RMSE_std': rmse.std(), 
            'MAE_mean': mae.mean(), 'MAE_std': mae.std(), 
            'R2_mean': r2.mean(), 'R2_std': r2.std()
        })
        
        # Fit on full train for test set evaluation
        pipe.fit(X_train_full, y_train_full)
        y_pred = pipe.predict(X_test)
        m = compute_metrics(y_test, y_pred, prefix='test_')
        m['encoding'] = country_encoding
        m['Modelo'] = name
        test_rows.append(m)

        if m.get('test_r2', -float('inf')) > best_r2:
            best_r2 = m.get('test_r2', -float('inf'))
            best_model_name = name
            best_pipe = pipe

    # Explainability
    if explain and best_pipe is not None:
        print(f"\n--- Running SHAP explainability for best model: {best_model_name} ---")
        try:
            shap_results = explain_model(best_pipe, X_train_full, X_test, output_dir=output_dir)
            print(f"Top 5 features by SHAP importance:")
            for feat, imp in shap_results['top_5_features']:
                print(f"  {feat}: {imp:.4f}")
            
            # Rename shap file to match user requirement shap_*
            old_path = shap_results['shap_summary_path']
            new_path = os.path.join(output_dir, 'shap_summary.png')
            # explain_model already uses 'shap_summary.png' in output_dir
            # Let's just make sure it's named as requested or verify what it does.
            # The user said: "Guardar outputs en results/ con shap_* y fairness_*"
            # explain_model uses 'shap_summary.png'
            print(f"SHAP summary plot saved to: {old_path}")
            
        except Exception as e:
            print(f"Warning: SHAP explainability failed: {e}")

    # Fairness audit
    if fairness and fairness_attr is not None and best_pipe is not None:
        print(f"\n--- Running fairness audit on attribute: {fairness_attr} ---")
        try:
            y_pred_best = best_pipe.predict(X_test)
            protected = df_test[fairness_attr].reset_index(drop=True)
            fairness_results = audit_fairness(y_test.values, y_pred_best, protected)
            print(get_fairness_summary(fairness_results))
            
            # Save fairness results with required prefix
            fairness_path = os.path.join(output_dir, f'fairness_{fairness_attr}.json')
            with open(fairness_path, 'w') as f:
                json.dump(fairness_results, f, indent=2)
            print(f"Fairness results saved to: {fairness_path}")
        except Exception as e:
            print(f"Warning: Fairness audit failed: {e}")

    return cv_rows, test_rows


def main():
    parser = argparse.ArgumentParser(description='Quick training for validation.')
    parser.add_argument('--variant', type=str, default='target', choices=['onehot', 'target'])
    parser.add_argument('--n_jobs', type=int, default=2)
    parser.add_argument('--explain', action='store_true')
    parser.add_argument('--fairness', action='store_true')
    parser.add_argument('--subsample', type=int, default=20000, help='Number of rows to use.')
    args = parser.parse_args()

    os.makedirs('results', exist_ok=True)

    print(f"Loading data and subsampling to {args.subsample} rows...")
    df = load_processed()
    
    if len(df) > args.subsample:
        df = df.sample(n=args.subsample, random_state=SEED)
        
    df, emb_cols = precompute_text_embeddings(df)
    all_features = NUM_FEATURES + CAT_LOW + CAT_HIGH + emb_cols
    X = df[all_features].copy()
    y = df['salary_log'].astype(np.float32)
    
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=df['region']
    )

    df_test = df.loc[X_test.index].copy()

    print(f'--- Quick Variant: {args.variant} ---')
    cv_rows, test_rows = run_quick_variant(
        args.variant, X_train_full, y_train_full, X_test, y_test, emb_cols,
        n_jobs=args.n_jobs, explain=args.explain, fairness=args.fairness,
        fairness_attr='experience_level', df_test=df_test, output_dir='results'
    )

    pd.DataFrame(cv_rows).to_csv('results/quick_cv_by_encoding.csv', index=False)
    pd.DataFrame(test_rows).to_csv('results/quick_test_by_encoding.csv', index=False)
    print('Guardado results/quick_cv_by_encoding.csv y results/quick_test_by_encoding.csv')


if __name__ == '__main__':
    main()
