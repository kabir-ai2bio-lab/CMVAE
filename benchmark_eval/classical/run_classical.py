"""Classical benchmarks: Raw (top-500 variance), PCA, Gini-importance selection.

Every method is fitted inside the CV training fold only (filter -> impute -> ... ) and followed by
LR/RF/PSVM x NO/ROS/SMOTE.

    python -m benchmark_eval.classical.run_classical --omic mRNA --method all --data_dir data/dummy/BRCA
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from benchmark_eval.common import (RF_N_TREES, classifier_grid, filter_and_impute, save_results, top_variance)
from CMVAE.configs import OMIC_ORDER, SEED, SURVIVAL_FILE, OmicSpec, get_omic
from CMVAE.data import build_cohort, load_or_create_folds


def preprocess_raw(X_tr, X_te, spec: OmicSpec, y_tr=None, fold=0):
    """filter -> median impute -> top-N variance (on unscaled values) -> z-score."""
    X_tr, X_te = filter_and_impute(X_tr, X_te, spec.classical_filter)
    X_tr, X_te = top_variance(X_tr, X_te, spec.raw_top_n)
    sc = StandardScaler()
    return sc.fit_transform(X_tr), sc.transform(X_te)


def preprocess_pca(X_tr, X_te, spec: OmicSpec, y_tr=None, fold=0):
    """filter -> median impute -> z-score -> PCA."""
    X_tr, X_te = filter_and_impute(X_tr, X_te, spec.classical_filter)
    sc = StandardScaler()
    X_tr, X_te = sc.fit_transform(X_tr), sc.transform(X_te)
    pca = PCA(n_components=spec.pca_components, random_state=SEED)
    return pca.fit_transform(X_tr), pca.transform(X_te)


def preprocess_gini(X_tr, X_te, spec: OmicSpec, y_tr=None, fold=0):
    """filter -> median impute -> min-max -> top-N features by random-forest Gini importance."""
    X_tr, X_te = filter_and_impute(X_tr, X_te, spec.classical_filter)
    sc = MinMaxScaler()
    X_tr, X_te = sc.fit_transform(X_tr), sc.transform(X_te)
    selector = RandomForestClassifier(n_estimators=RF_N_TREES, random_state=SEED,
                                      class_weight="balanced", n_jobs=-1).fit(X_tr, y_tr)
    idx = np.sort(np.argsort(selector.feature_importances_)[::-1][:spec.gini_top_n])
    return X_tr[:, idx], X_te[:, idx]


METHODS = {"Raw": preprocess_raw, "PCA": preprocess_pca, "Gini": preprocess_gini}


def run_method(method: str, cohort, spec, folds, out_dir, verbose=True):
    rows = []
    for fold, (tr, te) in enumerate(folds):
        X_tr, X_te = METHODS[method](cohort.X_labeled[tr], cohort.X_labeled[te], spec, cohort.y_labeled[tr], fold)
        if verbose:
            print(f"  [{method} | {spec.name}] fold {fold}: {X_tr.shape[1]} features")
        rows += classifier_grid({"scaled": (X_tr, X_te)}, cohort.y_labeled[tr], cohort.y_labeled[te], fold,
                                dict(method=method, loss_combo="-"), verbose=verbose)
    per_fold = pd.DataFrame(rows)
    save_results(per_fold, os.path.join(out_dir, method))
    return per_fold


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--omic", default="mRNA", choices=OMIC_ORDER + ["all"])
    p.add_argument("--method", default="all", choices=list(METHODS) + ["all"])
    p.add_argument("--data_dir", default="data/dummy/BRCA")
    p.add_argument("--survival_file", default=SURVIVAL_FILE)
    p.add_argument("--out_dir", default="outputs", help="results go to <out_dir>/<omic>/<method>/")
    a = p.parse_args()
    for name in (OMIC_ORDER if a.omic == "all" else [a.omic]):
        spec = get_omic(name)
        cohort = build_cohort(spec, a.data_dir, a.survival_file)
        folds = load_or_create_folds(cohort, os.path.join(a.out_dir, name, "folds"))
        for m in (METHODS if a.method == "all" else [a.method]):
            run_method(m, cohort, spec, folds, os.path.join(a.out_dir, name))


if __name__ == "__main__":
    main()
