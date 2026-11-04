"""Training / validation loss curves per fold and loss combination (CMVAE, MAE, VAE, AE).

Reads ``<out_dir>/<omic>/<Method>/results/loss_curves.csv`` and writes
``<out_dir>/<omic>/analysis/loss_curves_<Method>.png`` (red dotted line = best validation epoch).

    python analysis_plotting/plot_loss_curves.py --omic mRNA --method CMVAE --out_dir outputs
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COLORS = ["steelblue", "seagreen", "darkorange", "mediumpurple"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--omic", default="mRNA")
    p.add_argument("--method", default="CMVAE", choices=["CMVAE", "MAE", "VAE", "AE"])
    p.add_argument("--out_dir", default="outputs")
    a = p.parse_args()
    df = pd.read_csv(os.path.join(a.out_dir, a.omic, a.method, "results", "loss_curves.csv"))
    combos, folds = list(dict.fromkeys(df["loss_combo"])), sorted(df["fold"].unique())
    fig, axes = plt.subplots(len(folds), len(combos), figsize=(5 * len(combos), 2.6 * len(folds)), squeeze=False)
    for i, f in enumerate(folds):
        for j, c in enumerate(combos):
            d = df[(df["fold"] == f) & (df["loss_combo"] == c)]
            ax = axes[i, j]
            ax.plot(d["epoch"], d["train_loss"], color=COLORS[j % 4], lw=1.5, label="train")
            ax.plot(d["epoch"], d["val_loss"], color=COLORS[j % 4], lw=1.5, ls="--", alpha=0.8, label="val")
            ax.axvline(d.loc[d["val_loss"].idxmin(), "epoch"], color="red", ls=":", lw=1, alpha=0.6)
            ax.set_title(f"Fold {f} | {c}", fontsize=9); ax.set_xlabel("Epoch", fontsize=8); ax.set_ylabel("Loss", fontsize=8)
            ax.grid(alpha=0.3, ls="--"); ax.legend(fontsize=7)
    fig.suptitle(f"{a.omic} - {a.method}: training / validation loss (red dotted = best val epoch)", y=1.0)
    fig.tight_layout()
    out_dir = os.path.join(a.out_dir, a.omic, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"loss_curves_{a.method}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
