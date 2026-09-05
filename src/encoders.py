import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

class TextEmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name='all-MiniLM-L6-v2', cache_path=None):
        self.model_name = model_name
        self.cache_path = cache_path
        self.model_ = None
    def fit(self, X, y=None):
        if SentenceTransformer is None:
            raise ImportError('sentence-transformers no está instalado.')
        self.model_ = SentenceTransformer(self.model_name)
        return self
    def transform(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        texts = [' | '.join([str(v) for v in row]) for row in X]
        return self.model_.encode(texts, show_progress_bar=True, normalize_embeddings=True)

class MeanTargetEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, smoothing=10.0):
        self.smoothing = float(smoothing)
        self.global_mean_ = None
        self.col_maps_ = []
    def fit(self, X, y):
        X = np.asarray(X)
        if y is None:
            raise ValueError("MeanTargetEncoder requiere y durante fit.")
        y = np.asarray(y).reshape(-1)
        self.global_mean_ = float(np.mean(y))
        self.col_maps_ = []
        n_cols = X.shape[1] if X.ndim == 2 else 1
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        for j in range(n_cols):
            col = X[:, j].astype(str)
            uniques, inv, counts = np.unique(col, return_inverse=True, return_counts=True)
            sums = np.bincount(inv, weights=y, minlength=len(uniques))
            means = (sums + self.smoothing * self.global_mean_) / (counts + self.smoothing)
            mapping = {cat: float(mean) for cat, mean in zip(uniques, means)}
            self.col_maps_.append(mapping)
        return self
    def transform(self, X):
        X = np.asarray(X)
        n_cols = X.shape[1] if X.ndim == 2 else 1
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        enc_cols = []
        for j in range(n_cols):
            col = X[:, j].astype(str)
            mapping = self.col_maps_[j]
            default = self.global_mean_
            enc = np.array([mapping.get(val, default) for val in col], dtype=float).reshape(-1, 1)
            enc_cols.append(enc)
        return np.hstack(enc_cols)
