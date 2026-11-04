"""C-index comparison with published multi-omics survival methods.

The six reference values are the C-indices reported in the DeepOmix paper for
TCGA-BRCA with four combined omic modalities - they are quoted constants, NOT recomputed here.

    python analysis_plotting/plot_cindex_comparison.py --out_dir outputs
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REFERENCE_C_INDEX = {
    "blockforest": 0.637, "CoxBoost": 0.623, "ipflasso": 0.656, "DeepSurv": 0.625, "DeepHit": 0.648,
    "DeepOmix": 0.656,
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default="outputs")
    p.add_argument("--omic", default="mRNA")
    a = p.parse_args()
    files = glob.glob(os.path.join(a.out_dir, a.omic, "analysis", "cindex_summary_*.csv"))
    if not files:
        raise SystemExit("run analysis_plotting/survival_analysis.py first")
    summ = pd.read_csv(files[0]).iloc[0]
    ours = float(summ["mean_fold_c_index"])
    names = list(REFERENCE_C_INDEX) + [f"CMVAE\n({a.omic})"]
    vals = list(REFERENCE_C_INDEX.values()) + [ours]

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=300, facecolor="white")
    bars = ax.bar(names, vals, color=["#8C8C8C"] * len(REFERENCE_C_INDEX) + ["#B07A5B"], edgecolor="#2A2A2A", linewidth=0.45)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0.58, max(vals) + 0.03); ax.set_ylabel("Concordance index", fontsize=11)
    ax.tick_params(axis="x", labelsize=8.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", ls=(0, (5, 6)), alpha=0.25); ax.set_axisbelow(True)
    fig.tight_layout()
    out = os.path.join(a.out_dir, a.omic, "analysis", "cindex_comparison_multiomics.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
