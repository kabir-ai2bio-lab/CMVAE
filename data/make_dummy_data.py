"""Generate SYNTHETIC TCGA-BRCA-like data in the UCSC Xena file format.

*** None of this is real patient data. ***  

Files written to ``--out_dir`` (same names / layout as the real Xena downloads, so the code never needs
to know whether it is looking at dummy or real data):

  TCGA.BRCA.sampleMap_HiSeqV2                                          mRNA, log2(norm_count + 1)
  TCGA.BRCA.sampleMap_HumanMethylation450                              DNA methylation, beta values in [0, 1]
  TCGA.BRCA.sampleMap_miRNA_HiSeq_gene                                 miRNA, log2(RPM + 1)
  TCGA.BRCA.sampleMap_Gistic2_CopyNumber_Gistic2_all_data_by_genes     CNV, GISTIC2 scores {-2..2}
  survival_BRCA_survival.txt                                           sample, OS, _PATIENT, OS.time

Omic files are features x samples (TSV, first header cell = "sample").  Different omics cover different
random subsets of patients, mimicking the low cross-modality overlap discussed in the paper, and the
STS / LTS / censored proportions mimic the strong imbalance of the real cohort.  A weak survival-related
signal is planted in a subset of features so that the models have something to learn.

    python data/make_dummy_data.py                       # -> data/dummy/BRCA
    python data/make_dummy_data.py --n_patients 1000 --seed 1
"""
import argparse
import os

import numpy as np
import pandas as pd

FILES = {
    "mRNA": "TCGA.BRCA.sampleMap_HiSeqV2",
    "methylation": "TCGA.BRCA.sampleMap_HumanMethylation450",
    "miRNA": "TCGA.BRCA.sampleMap_miRNA_HiSeq_gene",
    "cnv": "TCGA.BRCA.sampleMap_Gistic2_CopyNumber_Gistic2_all_data_by_genes",
}
SURVIVAL = "survival_BRCA_survival.txt"
FIVE_YEARS = 5 * 365.25
# strength of the planted survival-risk signal in the informative features (per omic).  Chosen so that the
# bundled data is learnable (AUROC clearly above 0.5) but far from perfect.
SIGNAL = {"mRNA": 0.9, "methylation": 0.7, "miRNA": 0.6, "cnv": 0.6}


def make_survival(n, rng, risk):
    """Exponential event times whose scale shrinks with risk + uniform censoring."""
    scale = 12000.0 * np.exp(-0.9 * risk)
    t_event = rng.exponential(scale)
    t_cens = rng.gamma(1.15, 1350.0, size=n)          # heavy early censoring, as in TCGA follow-up
    os_time = np.maximum(1.0, np.round(np.minimum(t_event, t_cens), 0))
    os_event = (t_event <= t_cens).astype(int)
    return os_time, os_event


def _loadings(rng, n_feat, frac_informative=0.12):
    load = np.zeros(n_feat)
    idx = rng.choice(n_feat, size=int(frac_informative * n_feat), replace=False)
    load[idx] = rng.normal(0, 1, size=len(idx))
    return load


def gen_mrna(rng, risk, subtype, n_feat=600):
    mean = rng.uniform(2, 12, n_feat)
    sd = rng.uniform(0.3, 1.5, n_feat)
    load = _loadings(rng, n_feat)
    sub_eff = rng.normal(0, 0.8, size=(3, n_feat)) * (rng.random(n_feat) < 0.3)
    X = (mean + SIGNAL['mRNA'] * np.outer(risk, load) * sd + sub_eff[subtype] * sd
         + rng.normal(0, 1, size=(len(risk), n_feat)) * sd)
    X = np.clip(X, 0, None)
    low = rng.choice(n_feat, size=int(0.05 * n_feat), replace=False)      
    for j in low:
        X[rng.random(len(risk)) < 0.5, j] = 0.0
    X[rng.random(X.shape) < 0.002] = np.nan
    return X, [f"GENE{j + 1:04d}" for j in range(n_feat)]


def gen_methylation(rng, risk, subtype, n_feat=700):
    m = rng.normal(0, 1.5, n_feat)
    load = _loadings(rng, n_feat)
    sub_eff = rng.normal(0, 0.8, size=(3, n_feat)) * (rng.random(n_feat) < 0.3)
    logit = m + SIGNAL['methylation'] * np.outer(risk, load) + sub_eff[subtype] + rng.normal(0, 0.7, size=(len(risk), n_feat))
    X = 1.0 / (1.0 + np.exp(-logit))
    bad = rng.choice(n_feat, size=int(0.04 * n_feat), replace=False)      
    for j in bad:
        X[rng.random(len(risk)) < 0.30, j] = np.nan
    X[rng.random(X.shape) < 0.01] = np.nan
    return X, [f"cg{j + 1:08d}" for j in range(n_feat)]


def gen_mirna(rng, risk, subtype, n_feat=200):
    mean = rng.uniform(0, 10, n_feat)
    sd = rng.uniform(0.4, 1.3, n_feat)
    load = _loadings(rng, n_feat, 0.10)
    sub_eff = rng.normal(0, 0.8, size=(3, n_feat)) * (rng.random(n_feat) < 0.3)
    X = (mean + SIGNAL['miRNA'] * np.outer(risk, load) * sd + sub_eff[subtype] * sd
         + rng.normal(0, 1, size=(len(risk), n_feat)) * sd)
    X = np.clip(X, 0, None)
    low = rng.choice(n_feat, size=int(0.15 * n_feat), replace=False)
    for j in low:
        X[rng.random(len(risk)) < 0.4, j] = 0.0
    X[rng.random(X.shape) < 0.002] = np.nan
    return X, [f"hsa-mir-{j + 1}" for j in range(n_feat)]


def gen_cnv(rng, risk, subtype, n_feat=500):
    load = _loadings(rng, n_feat, 0.08)
    sub_eff = rng.normal(0, 0.8, size=(3, n_feat)) * (rng.random(n_feat) < 0.2)
    c = SIGNAL['cnv'] * np.outer(risk, load) + sub_eff[subtype] + rng.normal(0, 0.7, size=(len(risk), n_feat))
    X = np.zeros_like(c)
    X[c > 0.8], X[c > 1.6] = 1, 2
    X[c < -0.8], X[c < -1.6] = -1, -2
    X[c > 1.6] = 2
    X[rng.random(X.shape) < 0.005] = np.nan
    return X, [f"CNGENE{j + 1:04d}" for j in range(n_feat)]


GENERATORS = {"mRNA": gen_mrna, "methylation": gen_methylation, "miRNA": gen_mirna, "cnv": gen_cnv}
COVERAGE = {"mRNA": 0.95, "methylation": 0.78, "miRNA": 0.70, "cnv": 0.88}   # share of patients per omic


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "dummy", "BRCA"))
    p.add_argument("--n_patients", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    rng = np.random.default_rng(a.seed)

    n = a.n_patients
    ids = np.array([f"SYNTH-{i + 1:04d}-01" for i in range(n)])
    risk = rng.normal(0, 1, n)
    subtype = rng.integers(0, 3, n)
    os_time, os_event = make_survival(n, rng, risk)

    pd.DataFrame({"sample": ids, "OS": os_event, "_PATIENT": [s[:-3] for s in ids], "OS.time": os_time.astype(int)}) \
        .to_csv(os.path.join(a.out_dir, SURVIVAL), sep="\t", index=False)

    lts = os_time >= FIVE_YEARS
    sts = (os_time < FIVE_YEARS) & (os_event == 1)
    print(f"survival: STS={sts.sum()} LTS={lts.sum()} censored={n - sts.sum() - lts.sum()}  (n={n})")

    for omic, fn in GENERATORS.items():
        keep = np.sort(rng.choice(n, size=int(COVERAGE[omic] * n), replace=False))
        X, feats = fn(rng, risk[keep], subtype[keep])
        df = pd.DataFrame(X.T, index=feats, columns=ids[keep])        # features x samples (Xena layout)
        df.to_csv(os.path.join(a.out_dir, FILES[omic]), sep="\t", index_label="sample",
                  float_format="%.4f", na_rep="NA")
        print(f"{omic:12s} {df.shape[0]:5d} features x {df.shape[1]:4d} samples -> {FILES[omic]}")

    with open(os.path.join(a.out_dir, "README_DUMMY.txt"), "w") as fh:
        fh.write("SYNTHETIC DATA - NOT REAL PATIENTS.\nGenerated by data/make_dummy_data.py "
                 f"(seed={a.seed}, n_patients={n}).\nIt only mimics the file format and class imbalance of "
                 "TCGA-BRCA (UCSC Xena); results on it are meaningless biologically.\n")


if __name__ == "__main__":
    main()
