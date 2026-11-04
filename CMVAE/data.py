"""Data loading, survival labelling and cross-validation folds.

Input files follow the UCSC Xena layout for TCGA-BRCA:
  * omic matrix : TSV, rows = features, columns = sample barcodes
  * survival    : TSV with columns ``sample``, ``OS`` (event), ``OS.time`` (days)

Label logic:
  * LTS (label 0)       : OS.time >= 5 years
  * STS (label 1)       : OS.time <  5 years AND OS == 1 (death observed)
  * censored / ambiguous: OS.time <  5 years AND OS == 0 (outcome unknown)
Censored patients never enter classifier training, evaluation or the
contrastive loss. Only CMVAE-style models may use them for reconstruction.
"""
import gc
import json
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from .configs import OmicSpec, SEED, N_FOLDS, SURVIVAL_THRESH_DAYS


@dataclass
class Cohort:
    """One omic cohort after sample matching with survival metadata."""

    omic: str
    X_labeled: np.ndarray          # (n_labeled, n_features) float32, may contain NaN
    y_labeled: np.ndarray          # (n_labeled,) int, 1 = STS, 0 = LTS
    ids_labeled: np.ndarray        # (n_labeled,) sample barcodes
    os_time_labeled: np.ndarray    # (n_labeled,) days
    os_event_labeled: np.ndarray   # (n_labeled,) 0/1
    X_ambiguous: np.ndarray        # (n_censored, n_features) float32
    ids_ambiguous: np.ndarray
    feature_names: List[str]

    @property
    def n_features(self) -> int:
        return self.X_labeled.shape[1]

    def summary(self) -> str:
        return (f"[{self.omic}] labeled={len(self.y_labeled)} "
                f"(STS={int((self.y_labeled == 1).sum())}, LTS={int((self.y_labeled == 0).sum())}) "
                f"censored={len(self.X_ambiguous)} features={self.n_features}")


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_omic_matrix(path: str) -> pd.DataFrame:
    """Read a Xena features x samples TSV and return samples x features (float32).

    Staged conversion keeps peak memory low for the 485k-probe methylation file:
    read -> float32 array -> drop DataFrame -> transpose-copy.
    """
    df = pd.read_csv(path, sep="\t", index_col=0)
    sample_ids = df.columns.tolist()
    feature_ids = df.index.tolist()
    arr = df.values.astype(np.float32)
    del df
    gc.collect()
    out = pd.DataFrame(arr.T.copy(), index=sample_ids, columns=feature_ids)
    del arr
    gc.collect()
    out.index.name = "sample"
    return out


def load_survival(path: str) -> pd.DataFrame:
    surv = pd.read_csv(path, sep="\t")
    surv = surv.dropna(subset=["OS.time", "OS"])
    surv["OS"] = surv["OS"].astype(int)
    surv["OS.time"] = surv["OS.time"].astype(float)
    return surv.set_index("sample")


def assign_labels(surv: pd.DataFrame, thresh_days: float = SURVIVAL_THRESH_DAYS) -> pd.DataFrame:
    """Add ``label`` (0 / 1 / NaN) and boolean ``ambiguous`` columns."""
    surv = surv.copy()
    lts = surv["OS.time"] >= thresh_days
    sts = (surv["OS.time"] < thresh_days) & (surv["OS"] == 1)
    amb = (surv["OS.time"] < thresh_days) & (surv["OS"] == 0)
    surv["label"] = np.nan
    surv.loc[lts, "label"] = 0
    surv.loc[sts, "label"] = 1
    surv["ambiguous"] = amb
    return surv


def build_cohort(spec: OmicSpec, data_dir: str, survival_file: str,
                 thresh_days: float = SURVIVAL_THRESH_DAYS, verbose: bool = True) -> Cohort:
    omic_path = os.path.join(data_dir, spec.file_name)
    surv_path = os.path.join(data_dir, survival_file)
    for p in (omic_path, surv_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing input file: {p}\n"
                "Use --data_dir data/dummy/BRCA for the bundled synthetic data, or see data/README.md "
                "for how to download the real UCSC Xena files.")
    omic = load_omic_matrix(omic_path)
    surv = assign_labels(load_survival(surv_path), thresh_days)

    common = omic.index.intersection(surv.index)
    omic, surv = omic.loc[common], surv.loc[common]

    lab = np.where(surv["label"].notna().values)[0]
    amb = np.where(surv["ambiguous"].values)[0]
    X_all = omic.values.astype(np.float32)

    cohort = Cohort(
        omic=spec.name,
        X_labeled=X_all[lab],
        y_labeled=surv["label"].values[lab].astype(int),
        ids_labeled=np.asarray(common[lab]),
        os_time_labeled=surv["OS.time"].values[lab].astype(float),
        os_event_labeled=surv["OS"].values[lab].astype(int),
        X_ambiguous=X_all[amb],
        ids_ambiguous=np.asarray(common[amb]),
        feature_names=omic.columns.tolist(),
    )
    if verbose:
        print(cohort.summary())
    return cohort


# ----------------------------------------------------------------------------
# Cross-validation folds
# ----------------------------------------------------------------------------
def load_or_create_folds(cohort: Cohort, folds_dir: str, n_folds: int = N_FOLDS,
                         seed: int = SEED) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Stratified K-fold over the *labeled* samples; returns [(train_idx, test_idx), ...].

    Splits are stored by sample barcode in ``fold_indices.json`` so that every
    method (CMVAE and all benchmarks) re-uses *identical* patient splits.
    """
    os.makedirs(folds_dir, exist_ok=True)
    path = os.path.join(folds_dir, "fold_indices.json")
    ids = cohort.ids_labeled
    id_to_idx = {s: i for i, s in enumerate(ids)}

    if os.path.exists(path):
        with open(path) as fh:
            records = json.load(fh)
        folds = []
        for rec in records:
            missing = [s for s in rec["train"] + rec["test"] if s not in id_to_idx]
            if missing:
                raise ValueError(f"{path}: {len(missing)} sample IDs in fold {rec['fold']} are not in the "
                                 f"cohort. Delete the folds directory to regenerate.")
            folds.append((np.array([id_to_idx[s] for s in rec["train"]]),
                          np.array([id_to_idx[s] for s in rec["test"]])))
        return folds

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = list(skf.split(cohort.X_labeled, cohort.y_labeled))
    records = [{"fold": f, "train": ids[tr].tolist(), "test": ids[te].tolist()}
               for f, (tr, te) in enumerate(folds)]
    with open(path, "w") as fh:
        json.dump(records, fh)
    pd.DataFrame({"row_index": range(len(ids)), "sample_id": ids, "label": cohort.y_labeled}) \
        .to_csv(os.path.join(folds_dir, "final_sample_order.csv"), index=False)
    return folds


def inner_split(train_idx: np.ndarray, y: np.ndarray, val_ratio: float, seed: int = SEED):
    """Stratified split of a training fold into (inner-train, inner-val) for VAE model selection."""
    tr, va = train_test_split(train_idx, test_size=val_ratio, stratify=y[train_idx], random_state=seed)
    return tr, va
