"""Codificación de variables categóricas de alta cardinalidad.

Implementa la codificación por objetivo con suavizado de Micci-Barreca (2001):
cada nivel categórico se sustituye por la media de la variable de respuesta en
ese nivel, contraída hacia la media global en función del número de
observaciones. El parámetro `smoothing` (m en la formulación original) controla
cuántas observaciones necesita un nivel para que pese su propia media.

El estimador cumple el contrato de scikit-learn (`fit`/`transform`) a propósito:
así participa de la tubería y la validación cruzada lo reajusta en cada
partición, que es la única forma válida de usar una codificación que consume la
variable de respuesta (§3.5.3 de la tesis).
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class CodificadorPorObjetivo(BaseEstimator, TransformerMixin):
    def __init__(self, smoothing=10.0):
        self.smoothing = float(smoothing)
        self.global_mean_ = None
        self.col_maps_ = []

    def fit(self, X, y):
        X = np.asarray(X)
        if y is None:
            raise ValueError("CodificadorPorObjetivo requiere y durante fit.")
        y = np.asarray(y).reshape(-1)
        self.global_mean_ = float(np.mean(y))
        self.col_maps_ = []
        n_cols = X.shape[1] if X.ndim == 2 else 1
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        for j in range(n_cols):
            col = X[:, j].astype(str)
            unicos, inv, conteos = np.unique(col, return_inverse=True, return_counts=True)
            sumas = np.bincount(inv, weights=y, minlength=len(unicos))
            # Media suavizada: (suma + m * media_global) / (n + m).
            medias = (sumas + self.smoothing * self.global_mean_) / (conteos + self.smoothing)
            self.col_maps_.append({cat: float(m) for cat, m in zip(unicos, medias)})
        return self

    def transform(self, X):
        X = np.asarray(X)
        n_cols = X.shape[1] if X.ndim == 2 else 1
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        columnas = []
        for j in range(n_cols):
            col = X[:, j].astype(str)
            mapa = self.col_maps_[j]
            # Un nivel no visto durante el ajuste recibe la media global: es el
            # comportamiento correcto ante una categoría nueva.
            cod = np.array([mapa.get(v, self.global_mean_) for v in col],
                           dtype=float).reshape(-1, 1)
            columnas.append(cod)
        return np.hstack(columnas)
