"""Shared utilities for the benchmark methods (Raw, PCA, Gini, AE, VAE, MAE).

The benchmarks use exactly the same cohorts, labels and patient folds as CMVAE
(see ``CMVAE.data``), so differences in performance come from the representation only.

Protocol notes (kept from the original experiments):
  * classifier settings differ slightly from CMVAE's (LR: lbfgs, 1000 iter; RF: 100 trees)
  * seeds for oversampler / classifier are ``SEED + fold``
  * the F-max threshold is searched on a 0.01-0.99 grid on the evaluation fold itself
"""
import os
import random
import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from CMVAE.configs import CLASSIFIERS, OVERSAMPLERS, SEED
from CMVAE.evaluation import summarize

warnings.filterwarnings("ignore", message=".*probability.*deprecated.*") 

RF_N_TREES = 100
FEATURE_FILTER_MAX_FRAC = 0.20


# ----------------------------------------------------------------------------
# preprocessing helpers (all fitted on the training fold only)
# ----------------------------------------------------------------------------
def feature_keep_mask(X_train: np.ndarray, mode: str, max_frac: float = FEATURE_FILTER_MAX_FRAC):
    if mode == "zero_or_nan":
        frac = (np.isnan(X_train) | (X_train == 0)).mean(axis=0)
    elif mode == "nan_only":
        frac = np.isnan(X_train).mean(axis=0)
    else:
        raise ValueError(mode)
    return frac <= max_frac


def filter_and_impute(X_tr_raw, X_te_raw, mode: str):
    X_tr, X_te = X_tr_raw.astype(np.float64), X_te_raw.astype(np.float64)
    keep = feature_keep_mask(X_tr, mode)
    X_tr, X_te = X_tr[:, keep], X_te[:, keep]
    medians = np.nanmedian(X_tr, axis=0)
    for arr in (X_tr, X_te):
        r, c = np.where(np.isnan(arr))
        arr[r, c] = medians[c]
    return X_tr, X_te


def top_variance(X_tr, X_te, n: Optional[int]):
    if n is None:
        return X_tr, X_te
    idx = np.argsort(np.var(X_tr, axis=0))[::-1][:n]
    return X_tr[:, idx], X_te[:, idx]


def preprocess_deep(X_tr_raw, X_te_raw, filter_mode: str, top_n_variance: Optional[int]):
    X_tr, X_te = filter_and_impute(X_tr_raw, X_te_raw, filter_mode)
    X_tr, X_te = top_variance(X_tr, X_te, top_n_variance)
    sc = StandardScaler()
    return sc.fit_transform(X_tr).astype(np.float32), sc.transform(X_te).astype(np.float32)


# ----------------------------------------------------------------------------
# seeding / device
# ----------------------------------------------------------------------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    import torch                      
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# classifiers + metrics
# ----------------------------------------------------------------------------
def make_classifier(name: str, use_balanced: bool, seed: int):
    cw = "balanced" if use_balanced else None
    if name == "LR":
        return LogisticRegression(max_iter=1000, random_state=seed, class_weight=cw)
    if name == "RF":
        return RandomForestClassifier(n_estimators=RF_N_TREES, random_state=seed, n_jobs=-1, class_weight=cw)
    if name == "PSVM":
        return SVC(kernel="poly", degree=3, probability=True, random_state=seed, class_weight=cw)
    raise ValueError(name)


def oversample(name: str, X, y, seed: int):
    if name == "NO":
        return X, y
    if name == "ROS":
        return RandomOverSampler(random_state=seed).fit_resample(X, y)
    if name == "SMOTE":
        k = max(1, min(5, int(np.bincount(y).min()) - 1))
        return SMOTE(k_neighbors=k, random_state=seed).fit_resample(X, y)
    raise ValueError(name)


def metrics_row(y_true, y_prob) -> Dict[str, float]:
    """Same schema as ``CMVAE.evaluation.metrics_row``."""
    def block(pred):
        return dict(accuracy=accuracy_score(y_true, pred), balanced_acc=balanced_accuracy_score(y_true, pred),
                    precision=precision_score(y_true, pred, zero_division=0),
                    recall=recall_score(y_true, pred, zero_division=0), f1=f1_score(y_true, pred, zero_division=0))

    best_f1, best_t = 0.0, 0.5
    for t in np.linspace(0.01, 0.99, 99):
        f = f1_score(y_true, (y_prob >= t).astype(int), zero_division=0)
        if f >= best_f1:
            best_f1, best_t = f, t
    std, fmx = block((y_prob >= 0.5).astype(int)), block((y_prob >= best_t).astype(int))
    return dict(auroc=roc_auc_score(y_true, y_prob), fmax_threshold=round(float(best_t), 2), **std,
                **{f"{k}_fmax": v for k, v in fmx.items()})


def classifier_grid(Z: Dict[str, tuple], y_tr, y_te, fold: int, base_row: dict, verbose: bool = True):
    rows = []
    for clf_name in CLASSIFIERS:
        key = "raw" if (clf_name == "RF" and "raw" in Z) else "scaled"
        Z_tr, Z_te = Z[key]
        for os_name in OVERSAMPLERS:
            X_fit, y_fit = oversample(os_name, Z_tr.copy(), y_tr.copy(), SEED + fold)
            clf = make_classifier(clf_name, os_name == "NO", SEED + fold)
            clf.fit(X_fit, y_fit)
            m = metrics_row(y_te, clf.predict_proba(Z_te)[:, 1])
            rows.append(dict(fold=fold, **base_row, classifier=clf_name, oversampling=os_name, **m))
            if verbose:
                print(f"    {clf_name:4s}/{os_name:5s} AUROC={m['auroc']:.3f} F1={m['f1']:.3f}")
    return rows


# ----------------------------------------------------------------------------
# output
# ----------------------------------------------------------------------------
def save_results(per_fold: pd.DataFrame, out_dir: str, loss_rows: Optional[list] = None):
    res = os.path.join(out_dir, "results")
    os.makedirs(res, exist_ok=True)
    per_fold.to_csv(os.path.join(res, "per_fold_results.csv"), index=False)
    summarize(per_fold, ["method", "loss_combo", "classifier", "oversampling"]) \
        .to_csv(os.path.join(res, "display_summary.csv"), index=False)
    if loss_rows:
        pd.DataFrame(loss_rows).to_csv(os.path.join(res, "loss_curves.csv"), index=False)
    best = per_fold.groupby(["loss_combo", "classifier", "oversampling"])["auroc"].agg(["mean", "std"])
    i = best["mean"].idxmax()
    print(f"  best by mean AUROC: {i} -> {best.loc[i, 'mean']:.4f} ± {best.loc[i, 'std']:.4f}   [{res}]")
