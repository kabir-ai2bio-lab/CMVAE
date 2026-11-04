"""Leak-proof, fold-specific preprocessing for CMVAE / MAE.

MissingRateFilter -> zero-fill imputation -> TopVarianceFilter -> z-score.
Always ``fit`` on the labeled *training* partition only, then ``transform``
validation / test / censored samples.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class MissingRateFilter(BaseEstimator, TransformerMixin):
    """Drop features whose missing rate on the fit set exceeds ``threshold``."""

    def __init__(self, threshold=0.30, verbose=False):
        self.threshold = threshold
        self.verbose = verbose

    def fit(self, X, y=None):
        self.keep_mask_ = np.isnan(X).mean(axis=0) <= self.threshold
        if self.verbose:
            print(f"    MissingRateFilter: keeping {self.keep_mask_.sum()} / {X.shape[1]} features")
        return self

    def transform(self, X, y=None):
        return X[:, self.keep_mask_]


class TopVarianceFilter(BaseEstimator, TransformerMixin):
    """Keep the top-N most variable features (variance computed on the fit set)."""

    def __init__(self, n_features=5000, verbose=False):
        self.n_features = n_features
        self.verbose = verbose

    def fit(self, X, y=None):
        variances = np.nanvar(X, axis=0)
        n_keep = min(self.n_features, X.shape[1])
        self.keep_idx_ = np.argsort(variances)[::-1][:n_keep]
        if self.verbose:
            print(f"    TopVarianceFilter: keeping top {n_keep} / {X.shape[1]} features")
        return self

    def transform(self, X, y=None):
        return X[:, self.keep_idx_]


def build_preprocessor(missing_rate_thresh: float, top_n_features: int, verbose: bool = False) -> Pipeline:
    return Pipeline([
        ("missing_filter", MissingRateFilter(threshold=missing_rate_thresh, verbose=verbose)),
        ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
        ("var_filter", TopVarianceFilter(n_features=top_n_features, verbose=verbose)),
        ("scaler", StandardScaler()),
    ])
