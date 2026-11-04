"""Exploratory analysis: STS/LTS/censored label distribution per omic.

Counts are computed directly from the omic + survival files (after sample-barcode matching), so the
figure can be regenerated for the bundled dummy data or for the real UCSC Xena download.

    python EDA/label_distribution.py --data_dir data/dummy/BRCA --out_dir EDA/outputs
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
from CMVAE.configs import OMIC_ORDER, SURVIVAL_FILE, SURVIVAL_THRESH_DAYS, get_omic          
from CMVAE.data import assign_labels, load_survival                                        

FONT_SIZE = 11
OMIC_COLORS = {"miRNA": "#7C6A9C", "mRNA": "#B07A5B", "methylation": "#5F8D8A", "cnv": "#A7A35C"}
GROUPS = ["STS", "LTS", "Censored"]

plt.rcParams.update({"font.size": FONT_SIZE, "axes.labelsize": FONT_SIZE, "xtick.labelsize": FONT_SIZE,
                     "ytick.labelsize": FONT_SIZE, "legend.fontsize": FONT_SIZE, "font.family": "DejaVu Sans",
                     "axes.linewidth": 0.9, "pdf.fonttype": 42, "ps.fonttype": 42})


def count_labels(data_dir: str, survival_file: str) -> pd.DataFrame:
    """One row per omic: STS/LTS/Censored sample counts and number of omic features."""
    surv = assign_labels(load_survival(os.path.join(data_dir, survival_file)), SURVIVAL_THRESH_DAYS)
    rows = []
    for name in OMIC_ORDER:
        path = os.path.join(data_dir, get_omic(name).file_name)
        header = pd.read_csv(path, sep="\t", nrows=0).columns[1:]
        n_features = sum(1 for _ in open(path)) - 1
        s = surv.loc[surv.index.intersection(header)]
        rows.append(dict(Omic=name, STS=int((s["label"] == 1).sum()), LTS=int((s["label"] == 0).sum()),
                         Censored=int(s["ambiguous"].sum()), Total=n_features, Matched_samples=len(s)))
    return pd.DataFrame(rows)


def plot_label_distribution(df: pd.DataFrame, out_base: str, dpi: int = 600):
    x = np.arange(len(GROUPS))
    width = 0.18
    offsets = (np.arange(len(df)) - (len(df) - 1) / 2) * width
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for i, r in df.iterrows():
        ax.bar(x + offsets[i], [r[g] for g in GROUPS], width=width, label=f"{r['Omic']} ({r['Matched_samples']})",
               color=OMIC_COLORS.get(r["Omic"], "#888888"), edgecolor="#2A2A2A", linewidth=0.45)
    ax.set_xticks(x)
    ax.set_xticklabels(GROUPS)
    ax.set_ylabel("Number of samples")
    ax.set_ylim(0, df[GROUPS].to_numpy().max() * 1.35)
    leg = ax.legend(loc="upper left", frameon=True, fancybox=False, edgecolor="#3A3A3A", facecolor="white",
                    framealpha=1.0, borderpad=0.55, handlelength=1.2, handletextpad=0.6)
    leg.get_frame().set_linewidth(0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#444444")
    ax.tick_params(axis="both", which="major", length=4, width=0.8, color="#444444")
    ax.grid(axis="y", linestyle="-", linewidth=0.45, alpha=0.25)
    ax.set_axisbelow(True)
    ax.margins(x=0.08)
    plt.tight_layout()
    fig.savefig(out_base + ".png", dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(out_base + ".pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default="data/dummy/BRCA")
    p.add_argument("--survival_file", default=SURVIVAL_FILE)
    p.add_argument("--out_dir", default="EDA/outputs")
    p.add_argument("--dpi", type=int, default=600)
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    df = count_labels(a.data_dir, a.survival_file)
    df.to_csv(os.path.join(a.out_dir, "label_distribution.csv"), index=False)
    print(df.to_string(index=False))
    plot_label_distribution(df, os.path.join(a.out_dir, "BRCA_label_distribution"), a.dpi)
    print(f"saved -> {a.out_dir}/BRCA_label_distribution.[png|pdf] and label_distribution.csv")


if __name__ == "__main__":
    main()
