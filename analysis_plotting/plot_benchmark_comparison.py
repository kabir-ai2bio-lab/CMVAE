"""Collect results of all methods and draw the benchmark comparison figures.

Reads ``<out_dir>/<omic>/<Method>/results/per_fold_results.csv`` for Method in
{Raw, PCA, Gini, AE, VAE, MAE, CMVAE} (missing methods are skipped) and writes, to ``<out_dir>/analysis``:

  benchmark_auroc_summary.csv   mean/std AUROC over folds for every omic x method x classifier x oversampling
  fig3_benchmark_<omic>.png     mean AUROC of the 12 representation methods (bar = mean over the 9 classifier x
                                oversampling settings, dots = the individual settings)
  fig4_cmvae_heatmap.png        CMVAE AUROC (mean ± std over folds) per omic/objective x (classifier, oversampling)

    python analysis_plotting/plot_benchmark_comparison.py --out_dir outputs
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from CMVAE.configs import CLASSIFIERS, OMIC_ORDER, OVERSAMPLERS                              

METHOD_DIRS = ["Raw", "PCA", "Gini", "AE", "VAE", "MAE", "CMVAE"]
OMIC_TITLES = {"mRNA": "A. mRNA", "methylation": "B. DNA Methylation", "miRNA": "C. miRNA", "cnv": "D. CNV"}
LOSS_NAMES = {"MSE": "L2", "MSE+KL": "L2+KL", "MSE+MMD": "L2+MMD", "Logcosh+KL": "Logcosh+KL",
              "Logcosh+MMD": "Logcosh+MMD"}
PREFIX = {"VAE": "VAE", "MAE": "MVAE", "CMVAE": "CMVAE"}     
METHOD_ORDER = ["Raw", "PCA", "Gini", "AE",
                "VAE (L2+KL)", "VAE (L2+MMD)", "VAE (Logcosh+KL)", "VAE (Logcosh+MMD)",
                "MVAE (L2+KL)", "MVAE (L2+MMD)", "CMVAE (L2+KL)", "CMVAE (L2+MMD)"]
GROUP_COLORS = {"Raw": "#8C8C8C", "PCA": "#8C8C8C", "Gini": "#8C8C8C", "AE": "#6D8FB3", "VAE": "#6D8FB3",
                "MVAE": "#5F8D8A", "CMVAE": "#B07A5B"}
FONT = 10


def load_all(out_dir: str) -> pd.DataFrame:
    frames = []
    for omic in OMIC_ORDER:
        for m in METHOD_DIRS:
            f = os.path.join(out_dir, omic, m, "results", "per_fold_results.csv")
            if os.path.exists(f):
                d = pd.read_csv(f)
                d["omic"] = omic
                frames.append(d)
    if not frames:
        raise SystemExit(f"no results found under {out_dir}/<omic>/<Method>/results/")
    df = pd.concat(frames, ignore_index=True)

    def label(r):
        if r["method"] in ("Raw", "PCA", "Gini", "AE"):
            return r["method"]
        return f"{PREFIX[r['method']]} ({LOSS_NAMES[r['loss_combo']]})"

    df["variant"] = df.apply(label, axis=1)
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["omic", "variant", "classifier", "oversampling"])["auroc"].agg(["mean", "std"]).reset_index()
    return g.rename(columns={"mean": "auroc_mean", "std": "auroc_std"})


def fig3(summary: pd.DataFrame, out_dir: str):
    for omic in OMIC_ORDER:
        s = summary[summary["omic"] == omic]
        if s.empty:
            continue
        variants = [v for v in METHOD_ORDER if v in set(s["variant"])]
        fig, ax = plt.subplots(figsize=(7.2, 3.4))
        for i, v in enumerate(variants):
            vals = s.loc[s["variant"] == v, "auroc_mean"].values
            ax.bar(i, vals.mean(), width=0.7, color=GROUP_COLORS[v.split(" ")[0]], edgecolor="#2A2A2A", linewidth=0.45, alpha=0.85)
            ax.scatter(np.full(len(vals), i) + np.random.default_rng(i).uniform(-0.18, 0.18, len(vals)), vals,
                       s=9, color="#2A2A2A", zorder=3, linewidths=0)
        ax.axhline(0.5, ls="--", lw=0.9, color="black", alpha=0.6)
        ax.set_xticks(range(len(variants)))
        ax.set_xticklabels(variants, rotation=60, ha="right", fontsize=FONT - 1)
        ax.set_ylim(0.4, 0.85); ax.set_ylabel("Average AUROC", fontsize=FONT)
        ax.set_title(OMIC_TITLES[omic], loc="left", fontsize=FONT + 1, fontweight="bold")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", alpha=0.25, lw=0.45); ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"fig3_benchmark_{omic}.png"), dpi=300, facecolor="white")
        plt.close(fig)


def fig4(summary: pd.DataFrame, out_dir: str):
    s = summary[summary["variant"].str.startswith("CMVAE")]
    if s.empty:
        return
    cols = [(c, o) for c in CLASSIFIERS for o in OVERSAMPLERS]
    rows = [(om, v) for om in reversed(OMIC_ORDER) for v in ("CMVAE (L2+KL)", "CMVAE (L2+MMD)")
            if not s[(s["omic"] == om) & (s["variant"] == v)].empty]
    M, S = np.full((len(rows), len(cols)), np.nan), np.full((len(rows), len(cols)), np.nan)
    for i, (om, v) in enumerate(rows):
        for j, (c, o) in enumerate(cols):
            r = s[(s["omic"] == om) & (s["variant"] == v) & (s["classifier"] == c) & (s["oversampling"] == o)]
            if len(r):
                M[i, j], S[i, j] = r["auroc_mean"].iloc[0], r["auroc_std"].iloc[0]
    fig, ax = plt.subplots(figsize=(8.6, 0.55 * len(rows) + 1.6))
    im = ax.imshow(M, cmap="YlGnBu", vmin=0.5, vmax=0.8, aspect="auto")
    for i in range(len(rows)):
        for j in range(len(cols)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}\n±{S[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if M[i, j] > 0.68 else "black")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([f"{c} ({o})" for c, o in cols], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([f"{om}-{v.split('(')[1][:-1]}" for om, v in rows], fontsize=8)
    ax.set_xlabel("(Classifier, Oversampling Method)", fontsize=FONT); ax.set_ylabel("(Omic Type, CMVAE Objectives)", fontsize=FONT)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("Average AUROC", fontsize=FONT)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig4_cmvae_heatmap.png"), dpi=300, facecolor="white")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default="outputs", help="root used by the run scripts")
    a = p.parse_args()
    df = load_all(a.out_dir)
    summary = summarize(df)
    dest = os.path.join(a.out_dir, "analysis")
    os.makedirs(dest, exist_ok=True)
    summary.to_csv(os.path.join(dest, "benchmark_auroc_summary.csv"), index=False)
    best = summary.loc[summary.groupby(["omic", "variant"])["auroc_mean"].idxmax()]
    print(best.round(4).to_string(index=False))
    fig3(summary, dest)
    fig4(summary, dest)
    print(f"saved -> {dest}")


if __name__ == "__main__":
    main()
