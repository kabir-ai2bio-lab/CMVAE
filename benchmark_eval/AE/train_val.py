"""
    Train + cross-validate the AE benchmark.

    python -m benchmark_eval.AE.train_val --omic mRNA --data_dir data/dummy/BRCA
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from benchmark_eval.AE.model import Autoencoder
from benchmark_eval.common import classifier_grid, preprocess_deep, save_results, seed_everything
from CMVAE.configs import OMIC_ORDER, SEED, SURVIVAL_FILE, get_omic
from CMVAE.data import build_cohort, load_or_create_folds
from CMVAE.train_val import get_device

AE_CONFIG = dict(latent_dim=32, dropout_rate=0.3, learning_rate=1e-3, weight_decay=1e-5, batch_size=32,
                 epochs=50, patience=15, val_frac=0.15)


def train_autoencoder(model, X_scaled, cfg, device, seed):
    """MSE training with early stopping on an internal 85/15 split of the CV *training* fold.
    The CV test fold is never seen.  Best-validation weights are restored."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(X_scaled))
    n_val = max(1, int(len(X_scaled) * cfg["val_frac"]))
    X_tr, X_val = X_scaled[perm[n_val:]], X_scaled[perm[:n_val]]

    model.to(device)
    opt = optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    crit = nn.MSELoss()
    X_tr_t, X_val_t = torch.FloatTensor(X_tr).to(device), torch.FloatTensor(X_val).to(device)
    loader = DataLoader(TensorDataset(X_tr_t), batch_size=cfg["batch_size"], shuffle=True)

    best, bad, best_w, logs = float("inf"), 0, None, []
    for epoch in range(cfg["epochs"]):
        model.train()
        tr = 0.0
        for (xb,) in loader:
            opt.zero_grad()
            loss = crit(model(xb)[0], xb)
            loss.backward()
            opt.step()
            tr += loss.item() * len(xb)
        tr /= len(X_tr)
        model.eval()
        with torch.no_grad():
            va = crit(model(X_val_t)[0], X_val_t).item()
        logs.append(dict(epoch=epoch + 1, train_loss=tr, val_loss=va))
        if va < best - 1e-6:
            best, bad = va, 0
            best_w = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict({k: v.to(device) for k, v in best_w.items()})
    return model, logs


def run_ae(cohort, spec, folds, out_dir, device, cfg=AE_CONFIG, verbose=True):
    rows, loss_rows = [], []
    for fold, (tr, te) in enumerate(folds):
        seed = SEED + fold
        seed_everything(seed)
        y_tr, y_te = cohort.y_labeled[tr], cohort.y_labeled[te]
        X_tr_s, X_te_s = preprocess_deep(cohort.X_labeled[tr], cohort.X_labeled[te],
                                         spec.deep_filter, spec.deep_top_n_variance)
        model = Autoencoder(X_tr_s.shape[1], cfg["latent_dim"], cfg["dropout_rate"])
        model, logs = train_autoencoder(model, X_tr_s, cfg, device, seed)
        loss_rows += [dict(fold=fold, loss_combo="MSE", **l) for l in logs]
        if verbose:
            print(f"  [AE | {spec.name}] fold {fold}: {X_tr_s.shape[1]} features, {len(logs)} epochs, "
                  f"val_mse={logs[-1]['val_loss']:.4f}")

        model.eval()
        with torch.no_grad():
            Z_tr = model.encoder(torch.FloatTensor(X_tr_s).to(device)).cpu().numpy()
            Z_te = model.encoder(torch.FloatTensor(X_te_s).to(device)).cpu().numpy()
        sc = StandardScaler()                      # latent scaling: LR / PSVM use scaled, RF unscaled
        Z = {"scaled": (sc.fit_transform(Z_tr), sc.transform(Z_te)), "raw": (Z_tr, Z_te)}
        rows += classifier_grid(Z, y_tr, y_te, fold, dict(method="AE", loss_combo="MSE"), verbose=verbose)
    per_fold = pd.DataFrame(rows)
    save_results(per_fold, out_dir, loss_rows)
    return per_fold


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--omic", default="mRNA", choices=OMIC_ORDER + ["all"])
    p.add_argument("--data_dir", default="data/dummy/BRCA")
    p.add_argument("--survival_file", default=SURVIVAL_FILE)
    p.add_argument("--out_dir", default="outputs", help="results go to <out_dir>/<omic>/AE/")
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int, default=None)
    a = p.parse_args()
    cfg = dict(AE_CONFIG, **({"epochs": a.epochs} if a.epochs else {}))
    device = get_device(a.device)
    for name in (OMIC_ORDER if a.omic == "all" else [a.omic]):
        spec = get_omic(name)
        cohort = build_cohort(spec, a.data_dir, a.survival_file)
        folds = load_or_create_folds(cohort, os.path.join(a.out_dir, name, "folds"))
        run_ae(cohort, spec, folds, os.path.join(a.out_dir, name, "AE"), device, cfg)


if __name__ == "__main__":
    main()
