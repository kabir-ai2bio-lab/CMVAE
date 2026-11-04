"""Downstream classifiers, oversampling and metrics for latent embeddings (CMVAE/MAE).

Three classifiers (LR, RF, polynomial-kernel SVM) x three oversampling settings
(NO, ROS, SMOTE).  Oversampling is applied to the *training* embeddings only.
"""
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)
from sklearn.svm import SVC

# scikit-learn >= 1.9 warns about SVC(probability=True); behaviour is unchanged
warnings.filterwarnings("ignore", message=".*probability.*deprecated.*")

METRIC_KEYS = ["auroc", "fmax_threshold", "accuracy", "balanced_acc", "precision", "recall", "f1",
               "accuracy_fmax", "balanced_acc_fmax", "precision_fmax", "recall_fmax", "f1_fmax"]


def build_classifier(name: str, class_weight, seed: int):
    if name == "LR":
        return LogisticRegression(max_iter=2000, solver="saga", random_state=seed,
                                  class_weight=class_weight, n_jobs=-1)
    if name == "RF":
        return RandomForestClassifier(n_estimators=200, random_state=seed,
                                      class_weight=class_weight, n_jobs=-1)
    if name == "PSVM":
        return SVC(kernel="poly", degree=3, probability=True, random_state=seed, class_weight=class_weight)
    raise ValueError(f"Unknown classifier: {name}")


def oversample(name: str, X, y, seed: int):
    if name == "NO":
        return X, y
    if name == "ROS":
        return RandomOverSampler(random_state=seed).fit_resample(X, y)
    if name == "SMOTE":
        k = max(1, min(5, int(np.bincount(y).min()) - 1))     # guard tiny minority classes
        return SMOTE(k_neighbors=k, random_state=seed).fit_resample(X, y)
    raise ValueError(f"Unknown oversampler: {name}")


def find_fmax_threshold(y_true, y_prob) -> float:
    """Threshold maximising F1 on the precision-recall curve."""
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    f1 = 2 * prec * rec / (prec + rec + 1e-10)
    return float(thr[np.argmax(f1[:-1])])


def compute_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    return dict(
        balanced_acc=balanced_accuracy_score(y_true, y_pred),
        f1=f1_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        precision=precision_score(y_true, y_pred, zero_division=0),
        accuracy=accuracy_score(y_true, y_pred),
        auroc=roc_auc_score(y_true, y_prob),
    )


def metrics_row(y_val, y_val_prob, y_test, y_test_prob) -> Dict[str, float]:
    """AUROC + metrics at t=0.5 + metrics at the validation-selected F-max threshold."""
    m_std = compute_metrics(y_test, (y_test_prob >= 0.5).astype(int), y_test_prob)
    thr = find_fmax_threshold(y_val, y_val_prob)
    m_fmax = compute_metrics(y_test, (y_test_prob >= thr).astype(int), y_test_prob)
    return dict(fmax_threshold=round(thr, 4), **m_std,
                **{f"{k}_fmax": v for k, v in m_fmax.items() if k != "auroc"})


def summarize(per_fold: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    """mean/std across folds for every metric, formatted as 'mean ± std' strings."""
    g = per_fold.groupby(group_cols, sort=False)
    mean, std = g[METRIC_KEYS].mean(), g[METRIC_KEYS].std()
    out = mean.copy().astype(object)
    for c in METRIC_KEYS:
        out[c] = [f"{m:.4f} ± {s:.4f}" for m, s in zip(mean[c], std[c])]
    return out.reset_index()
