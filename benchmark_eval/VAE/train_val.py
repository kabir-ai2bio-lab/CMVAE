"""Train + cross-validate the VAE benchmark with four loss variants
(MSE+KL, Logcosh+KL, MSE+MMD, Logcosh+MMD).

    python -m benchmark_eval.VAE.train_val --omic mRNA --data_dir data/dummy/BRCA
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from benchmark_eval.VAE.model import VAE
from benchmark_eval.common import classifier_grid, preprocess_deep, save_results, seed_everything
from benchmark_eval.loss import VAE_LOSS_COMBOS, kl_divergence, mmd_loss, recon_logcosh, recon_mse
from CMVAE.configs import OMIC_ORDER, SEED, SURVIVAL_FILE, get_omic
from CMVAE.data import build_cohort, load_or_create_folds
from CMVAE.train_val import get_device

VAE_CONFIG = dict(latent_dim=32, dropout_rate=0.3, learning_rate=1e-3, weight_decay=1e-5, batch_size=32,
                  epochs=50, patience=15, val_frac=0.15,
                  beta_kl=1.0, beta_mmd=1.0, mmd_sigma=1.0,
                  warmup_epochs=10,      # linear ramp of the regulariser weight 0 -> beta (anti-collapse)
                  free_bits=0.5)         # per-dim KL floor, KL variants only


def train_vae(model, X_scaled, cfg, device, seed, recon_type, reg_type):
    """Early stopping on validation loss (internal 85/15 split of the CV training fold)."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(X_scaled))
    n_val = max(1, int(len(X_scaled) * cfg["val_frac"]))
    X_tr, X_val = X_scaled[perm[n_val:]], X_scaled[perm[:n_val]]

    model.to(device)
    opt = optim.Adam(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    X_tr_t, X_val_t = torch.FloatTensor(X_tr).to(device), torch.FloatTensor(X_val).to(device)
    loader = DataLoader(TensorDataset(X_tr_t), batch_size=cfg["batch_size"], shuffle=True)

    recon_fn = recon_mse if recon_type == "mse" else recon_logcosh
    beta = cfg["beta_kl"] if reg_type == "kl" else cfg["beta_mmd"]
    free_bits = cfg["free_bits"] if reg_type == "kl" else 0.0
    warmup = max(1, cfg["warmup_epochs"])

    def reg_fn(mu, log_var, z):
        return kl_divergence(mu, log_var, free_bits) if reg_type == "kl" else mmd_loss(z, cfg["mmd_sigma"])

    best, bad, best_w, logs = float("inf"), 0, None, []
    for epoch in range(cfg["epochs"]):
        eff_beta = beta * min(1.0, (epoch + 1) / warmup)
        model.train()
        s_rec = s_reg = nb = 0
        for (xb,) in loader:
            opt.zero_grad()
            recon, mu, lv, z = model(xb)
            r, g = recon_fn(recon, xb), reg_fn(mu, lv, z)
            (r + eff_beta * g).backward()
            opt.step()
            s_rec, s_reg, nb = s_rec + r.item(), s_reg + g.item(), nb + 1
        tr_loss = s_rec / nb + eff_beta * s_reg / nb
        model.eval()
        with torch.no_grad():
            recon, mu, lv, z = model(X_val_t)
            val_loss = recon_fn(recon, X_val_t).item() + eff_beta * reg_fn(mu, lv, z).item()
        logs.append(dict(epoch=epoch + 1, train_loss=tr_loss, val_loss=val_loss))
        if val_loss < best - 1e-6:
            best, bad = val_loss, 0
            best_w = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    model.load_state_dict({k: v.to(device) for k, v in best_w.items()})
    return model, logs


def run_vae(cohort, spec, folds, out_dir, device, cfg=VAE_CONFIG, verbose=True):
    rows, loss_rows = [], []
    for fold, (tr, te) in enumerate(folds):
        seed = SEED + fold
        seed_everything(seed)
        y_tr, y_te = cohort.y_labeled[tr], cohort.y_labeled[te]
        X_tr_s, X_te_s = preprocess_deep(cohort.X_labeled[tr], cohort.X_labeled[te],
                                         spec.deep_filter, spec.deep_top_n_variance)
        for name, recon_type, reg_type in VAE_LOSS_COMBOS:
            model = VAE(X_tr_s.shape[1], cfg["latent_dim"], cfg["dropout_rate"])
            model, logs = train_vae(model, X_tr_s, cfg, device, seed, recon_type, reg_type)
            loss_rows += [dict(fold=fold, loss_combo=name, **l) for l in logs]
            if verbose:
                print(f"  [VAE | {spec.name}] fold {fold} {name}: {X_tr_s.shape[1]} features, "
                      f"{len(logs)} epochs, val_loss={logs[-1]['val_loss']:.4f}")
            model.eval()
            with torch.no_grad():
                mu_tr, _ = model.encode(torch.FloatTensor(X_tr_s).to(device))
                mu_te, _ = model.encode(torch.FloatTensor(X_te_s).to(device))
            Z_tr, Z_te = mu_tr.cpu().numpy(), mu_te.cpu().numpy()
            sc = StandardScaler()
            Z = {"scaled": (sc.fit_transform(Z_tr), sc.transform(Z_te)), "raw": (Z_tr, Z_te)}
            rows += classifier_grid(Z, y_tr, y_te, fold, dict(method="VAE", loss_combo=name), verbose=verbose)
    per_fold = pd.DataFrame(rows)
    save_results(per_fold, out_dir, loss_rows)
    return per_fold


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--omic", default="mRNA", choices=OMIC_ORDER + ["all"])
    p.add_argument("--data_dir", default="data/dummy/BRCA")
    p.add_argument("--survival_file", default=SURVIVAL_FILE)
    p.add_argument("--out_dir", default="outputs", help="results go to <out_dir>/<omic>/VAE/")
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int, default=None)
    a = p.parse_args()
    cfg = dict(VAE_CONFIG, **({"epochs": a.epochs} if a.epochs else {}))
    device = get_device(a.device)
    for name in (OMIC_ORDER if a.omic == "all" else [a.omic]):
        spec = get_omic(name)
        cohort = build_cohort(spec, a.data_dir, a.survival_file)
        folds = load_or_create_folds(cohort, os.path.join(a.out_dir, name, "folds"))
        run_vae(cohort, spec, folds, os.path.join(a.out_dir, name, "VAE"), device, cfg)


if __name__ == "__main__":
    main()
