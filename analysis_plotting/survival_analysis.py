"""Out-of-fold survival analysis of the best CMVAE configuration.

Steps (no re-training; everything is read from the artefacts written by ``run_cmvae.py``):
  1. pick the best CMVAE configuration by mean 5-fold AUROC (``loss_combo | classifier | oversampling``)
  2. for every fold: load the saved fold-specific preprocessor, CMVAE encoder and classifier, embed the
     held-out fold and obtain an out-of-fold predicted STS probability (= risk score)
  3. Kaplan-Meier curves for high- (p >= 0.5) vs low-risk groups + log-rank test    
  4. Harrell's C-index per fold, pooled out-of-fold C-index and bootstrap 95% CI       
    python analysis_plotting/survival_analysis.py --omic mRNA --data_dir data/dummy/BRCA --out_dir outputs
"""
import argparse
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from CMVAE.configs import SURVIVAL_FILE, CMVAEConfig, get_omic, OMIC_ORDER     
from CMVAE.data import build_cohort, load_or_create_folds                         
from CMVAE.model import CMVAE                                                       
from CMVAE.train_val import encode_features, get_device                            

F, RED, BLUE = 11, "#E53935", "#1565C0"
STS_CLASS = 1


def best_config(per_fold_csv: str):
    df = pd.read_csv(per_fold_csv)
    return df.groupby(["loss_combo", "classifier", "oversampling"])["auroc"].mean().idxmax()


def out_of_fold_scores(cohort, folds, models_dir, combo, clf_name, os_name, latent_dim, device) -> pd.DataFrame:
    tag = combo.replace("+", "_")
    rows = []
    for fold, (_, te) in enumerate(folds):
        prep = joblib.load(os.path.join(models_dir, f"fold{fold}_preprocessor.pkl"))
        clf = joblib.load(os.path.join(models_dir, f"fold{fold}_{tag}_{clf_name}_{os_name}.pkl"))
        X_pp = prep.transform(cohort.X_labeled[te])
        vae = CMVAE(X_pp.shape[1], latent_dim).to(device)
        vae.load_state_dict(torch.load(os.path.join(models_dir, f"fold{fold}_vae_{tag}_best.pt"),
                                       map_location=device, weights_only=True))
        Z = encode_features(vae, X_pp, device)
        risk = clf.predict_proba(Z)[:, list(clf.classes_).index(STS_CLASS)]     # P(STS)
        rows.append(pd.DataFrame({"fold": fold, "sample_id": cohort.ids_labeled[te],
                                  "OS.time": cohort.os_time_labeled[te], "OS": cohort.os_event_labeled[te],
                                  "label": cohort.y_labeled[te], "risk_score": risk}))
    out = pd.concat(rows, ignore_index=True)
    out["predicted_survival_score_for_cindex"] = -out["risk_score"]
    out["risk_group_0p5"] = np.where(out["risk_score"] >= 0.5, "high", "low")
    return out


def c_index_summary(oof: pd.DataFrame, n_boot: int = 1000, seed: int = 42):
    per_fold = []
    for f, g in oof.groupby("fold"):
        per_fold.append(dict(fold=f, n_test=len(g), n_events=int(g["OS"].sum()),
                             n_censored=int(len(g) - g["OS"].sum()),
                             c_index=float(concordance_index(g["OS.time"], g["predicted_survival_score_for_cindex"],
                                                             g["OS"]))))
    per_fold = pd.DataFrame(per_fold)
    T, E, S = oof["OS.time"].values, oof["OS"].values, oof["predicted_survival_score_for_cindex"].values
    pooled = float(concordance_index(T, S, E))
    rng, boot = np.random.default_rng(seed), []
    for _ in range(n_boot):
        i = rng.integers(0, len(T), len(T))
        if E[i].sum() == 0 or len(np.unique(T[i])) < 2:
            continue
        try:
            boot.append(concordance_index(T[i], S[i], E[i]))
        except Exception:
            pass
    lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan)
    return per_fold, pooled, float(lo), float(hi), len(boot)


# ----------------------------------------------------------------------------
# figures
# ----------------------------------------------------------------------------
def plot_km(oof: pd.DataFrame, path: str):
    T, E = oof["OS.time"].values.astype(float), oof["OS"].values.astype(bool)
    hi = (oof["risk_score"].values >= 0.5)
    lo = ~hi
    if hi.sum() == 0 or lo.sum() == 0:
        print("  KM skipped: one risk group is empty"); return None
    p = logrank_test(T[hi], T[lo], E[hi], E[lo]).p_value
    pstr = "p < 0.0001" if p < 1e-4 else "p < 0.001" if p < 1e-3 else f"p = {p:.3f}"
    kh, kl = KaplanMeierFitter().fit(T[hi], E[hi]), KaplanMeierFitter().fit(T[lo], E[lo])

    tmax = int(np.ceil(T.max() / 2000.0) * 2000)
    at = list(range(0, tmax + 1, 2000))
    fig = plt.figure(figsize=(8.5, 7.0), dpi=300, facecolor="white")
    gs = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[4.8, 1.35], hspace=0.22,
                           left=0.17, right=0.96, top=0.96, bottom=0.10)
    ax = fig.add_subplot(gs[0])
    nr = fig.add_subplot(gs[1], sharex=ax)

    def draw(kmf, color):
        tl, sf = kmf.timeline, kmf.survival_function_.iloc[:, 0].values
        ci = kmf.confidence_interval_
        ax.step(tl, sf, where="post", color=color, lw=1.8, zorder=3)
        ax.fill_between(tl, ci.iloc[:, 0].values, ci.iloc[:, 1].values, step="post", alpha=0.18, color=color, zorder=2)
        ct = np.asarray(kmf.durations)[~np.asarray(kmf.event_observed, dtype=bool)]
        if len(ct):
            ax.plot(ct, kmf.predict(ct).values, "+", color=color, ms=4.5, mew=1.2, zorder=4, ls="none")

    draw(kh, RED)
    draw(kl, BLUE)
    dash = (0, (5, 5))
    ax.axhline(0.5, ls=dash, lw=0.9, color="black", alpha=0.85, zorder=1)
    for k in (kh, kl):
        med = k.median_survival_time_
        if np.isfinite(med) and 0 < med < tmax:
            ax.axvline(med, ymin=0, ymax=0.5 / 1.04, ls=dash, lw=0.9, color="black", alpha=0.85, zorder=1)
    ax.text(0.02, 0.14, pstr, transform=ax.transAxes, fontsize=F, va="bottom")
    ax.legend(handles=[plt.Line2D([0], [0], color=c, lw=1.8, marker="+", ms=5, mew=1.2) for c in (RED, BLUE)],
              labels=["high", "low"], title="risk", fontsize=F, title_fontsize=F, loc="upper right",
              frameon=False, ncol=2, handlelength=2.0, columnspacing=0.9, borderaxespad=0.2)
    ax.set_xlim(0, tmax); ax.set_ylim(0, 1.04)
    ax.set_ylabel("Survival probability", fontsize=F, labelpad=16); ax.set_xlabel("Time (days)", fontsize=F, labelpad=6)
    ax.set_xticks(at); ax.set_yticks([0, .25, .5, .75, 1.0]); ax.set_yticklabels([f"{x:.2f}" for x in (0, .25, .5, .75, 1)])
    ax.tick_params(axis="both", labelsize=F)
    ax.grid(ls=(0, (5, 6)), alpha=0.25, color="#888888", linewidth=0.8)
    for s in ("top", "left", "right", "bottom"):
        ax.spines[s].set_visible(False)

    # number-at-risk table
    nr.set_xlim(0, tmax); nr.set_ylim(0, 3.2); nr.set_yticks([])
    for s in nr.spines.values():
        s.set_visible(False)
    nr.set_xticks(at); nr.tick_params(axis="x", length=0, labelsize=F, pad=3); nr.set_xlabel("Time (days)", fontsize=F, labelpad=5)
    nr.text(0.0, 0.88, "Number at risk", transform=nr.transAxes, ha="left", va="center", fontsize=F, fontweight="bold")
    nr.text(-0.115, 0.42, "risk", transform=nr.transAxes, rotation=90, ha="center", va="center", fontsize=F)
    nr.text(-0.035, 0.57, "high-", transform=nr.transAxes, ha="right", va="center", fontsize=F, color=RED)
    nr.text(-0.035, 0.29, "low-", transform=nr.transAxes, ha="right", va="center", fontsize=F, color=BLUE)
    for t in at:
        nr.axvline(t, ls=(0, (5, 6)), alpha=0.18, color="#888888", lw=0.8, zorder=0)
        x = t + (tmax * 0.018 if t == 0 else 0)
        nr.text(x, 1.85, str(int(np.sum(T[hi] >= t))), ha="center", va="center", fontsize=F, color=RED)
        nr.text(x, 0.95, str(int(np.sum(T[lo] >= t))), ha="center", va="center", fontsize=F, color=BLUE)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    return p


def _clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="both", labelsize=F)
    ax.grid(axis="y", ls=(0, (5, 6)), alpha=0.25, color="#888888", linewidth=0.8)


def plot_cindex_figures(oof, per_fold, pooled, prefix):
    # per-fold C-index
    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=300, facecolor="white")
    ax.bar(per_fold["fold"].astype(str), per_fold["c_index"], width=0.65)
    ax.axhline(0.5, ls="--", lw=1.0, color="black", alpha=0.65)
    ax.axhline(pooled, ls="-", lw=1.2, color="black", alpha=0.85)
    ax.text(0.98, 0.94, f"OOF C-index = {pooled:.3f}", transform=ax.transAxes, ha="right", va="top", fontsize=F)
    ax.set_ylim(0, 1); ax.set_xlabel("Fold", fontsize=F); ax.set_ylabel("Concordance index", fontsize=F, labelpad=10)
    _clean(ax); fig.tight_layout()
    fig.savefig(prefix + "_per_fold.png", dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)
    # risk-score distribution by event status
    sc, E, T = oof["risk_score"].values, oof["OS"].values, oof["OS.time"].values
    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=300, facecolor="white")
    ax.hist(sc[E == 1], bins=25, alpha=0.45, label="event", density=True, color=RED)
    ax.hist(sc[E == 0], bins=25, alpha=0.45, label="censored", density=True, color=BLUE)
    ax.axvline(0.5, ls="--", lw=1.0, color="black", alpha=0.65)
    ax.set_xlabel("Predicted risk score", fontsize=F); ax.set_ylabel("Density", fontsize=F, labelpad=10)
    ax.legend(frameon=False, fontsize=F); _clean(ax); fig.tight_layout()
    fig.savefig(prefix + "_risk_distribution.png", dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)
    # survival time vs risk score
    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=300, facecolor="white")
    ax.scatter(T[E == 0], sc[E == 0], s=18, alpha=0.55, label="censored", color=BLUE, edgecolors="none")
    ax.scatter(T[E == 1], sc[E == 1], s=18, alpha=0.55, label="event", color=RED, edgecolors="none")
    ax.axhline(0.5, ls="--", lw=1.0, color="black", alpha=0.65)
    ax.set_xlabel("Overall survival time (days)", fontsize=F); ax.set_ylabel("Predicted risk score", fontsize=F, labelpad=10)
    ax.legend(frameon=False, fontsize=F); _clean(ax); fig.tight_layout()
    fig.savefig(prefix + "_time_vs_risk.png", dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--omic", default="mRNA", choices=OMIC_ORDER)
    p.add_argument("--data_dir", default="data/dummy/BRCA")
    p.add_argument("--survival_file", default=SURVIVAL_FILE)
    p.add_argument("--out_dir", default="outputs", help="root used by run_cmvae.py")
    p.add_argument("--device", default="auto")
    p.add_argument("--n_boot", type=int, default=1000)
    a = p.parse_args()

    cfg = CMVAEConfig()
    method_dir = os.path.join(a.out_dir, a.omic, "CMVAE")
    models_dir, results_dir = os.path.join(method_dir, "models"), os.path.join(method_dir, "results")
    combo, clf, os_name = best_config(os.path.join(results_dir, "per_fold_results.csv"))
    print(f"best CMVAE configuration for {a.omic}: {combo} | {clf} | {os_name}")

    cohort = build_cohort(get_omic(a.omic), a.data_dir, a.survival_file, cfg.survival_thresh_days)
    folds = load_or_create_folds(cohort, os.path.join(a.out_dir, a.omic, "folds"), cfg.n_folds, cfg.seed)
    oof = out_of_fold_scores(cohort, folds, models_dir, combo, clf, os_name, cfg.latent_dim, get_device(a.device))

    name = f"{combo.replace('+', '_')}_{clf}_{os_name}"
    fig_dir = os.path.join(a.out_dir, a.omic, "analysis")
    os.makedirs(fig_dir, exist_ok=True)
    oof.to_csv(os.path.join(fig_dir, f"oof_survival_risk_scores_{name}.csv"), index=False)

    per_fold, pooled, lo, hi, nb = c_index_summary(oof, a.n_boot)
    per_fold.to_csv(os.path.join(fig_dir, f"cindex_per_fold_{name}.csv"), index=False)
    pd.DataFrame([dict(omic=a.omic, loss_combo=combo, classifier=clf, oversampling=os_name,
                       mean_fold_c_index=per_fold["c_index"].mean(), std_fold_c_index=per_fold["c_index"].std(),
                       overall_oof_c_index=pooled, bootstrap_ci_low_2p5=lo, bootstrap_ci_high_97p5=hi,
                       bootstrap_n=nb, n_total=len(oof), n_events=int(oof["OS"].sum()))]) \
        .to_csv(os.path.join(fig_dir, f"cindex_summary_{name}.csv"), index=False)
    print(f"C-index per fold: {per_fold['c_index'].mean():.3f} ± {per_fold['c_index'].std():.3f} | pooled OOF "
          f"{pooled:.3f} (95% CI [{lo:.3f}, {hi:.3f}], {nb} bootstrap draws)")

    pval = plot_km(oof, os.path.join(fig_dir, f"km_best_{name}.png"))
    if pval is not None:
        print(f"log-rank p = {pval:.3g}")
    plot_cindex_figures(oof, per_fold, pooled, os.path.join(fig_dir, f"cindex_{name}"))
    print(f"figures + tables -> {fig_dir}")


if __name__ == "__main__":
    main()
