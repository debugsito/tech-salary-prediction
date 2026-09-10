import os
import sys
import pandas as pd
import numpy as np

# Add project root to sys.path
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.feature_pipeline import load_processed, precompute_text_embeddings, NUM_FEATURES, CAT_LOW, CAT_HIGH
from src.model_registry import build_models
from src.metrics import compute_metrics
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

def test_full_run_small():
    print("Starting small full run test...")
    try:
        # 1. Load and prep
        df = load_processed().head(2000) # Only 2000 rows for testing
        print(f"Loaded {len(df)} rows.")
        
        df, emb_cols = precompute_text_embeddings(df)
        all_features = NUM_FEATURES + CAT_LOW + CAT_HIGH + emb_cols
        X = df[all_features].copy()
        y = df['salary_log'].astype(np.float32)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 2. Build and test a single model
        print("Building models and testing one...")
        models = build_models()
        # Take just one model to be fast: Ridge
        model_name = 'Ridge'
        if model_name not in models:
            model_name = list(models.keys())[0]
        
        print(f"Testing model: {model_name}")
        from src.feature_pipeline import build_preprocessor
        preprocessor = build_preprocessor('target', emb_cols)
        
        pipe = Pipeline([('preprocessor', preprocessor), ('model', models[model_name])])
        pipe.fit(X_train, y_train)
        
        y_pred = pipe.predict(X_test)
        metrics = compute_metrics(y_test, y_pred)
        print(f"Success! Metrics: {metrics}")
        
        print("Test finished successfully.")
        return True
    except Exception as e:
        print(f"Test FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_full_run_small()
    if not success:
        sys.exit(1)
