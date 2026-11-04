"""Training and fold-wise validation for CMVAE (and, with ``lambda_contrast=0``, the MAE benchmark).

Per CV fold (all steps fitted on the labeled TRAIN partition only):
  1. preprocess: missing-rate filter -> zero-fill -> top-variance filter -> z-score
  2. train the VAE on labeled-train (+ censored samples, reconstruction only)
  3. embed with the deterministic posterior mean mu
  4. oversample the *training* embeddings (NO / ROS / SMOTE) and fit LR / RF / PSVM
  5. score the untouched test fold
"""
import os
import random
import time
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .configs import CLASSIFIERS, OVERSAMPLERS, CMVAEConfig, OmicSpec
from .data import Cohort, inner_split, load_or_create_folds
from .evaluation import build_classifier, metrics_row, oversample, summarize
from .losses import mmd_vae_loss, supervised_contrastive_loss, vae_loss
from .model import CMVAE, MaskedDenoisingVAE
from .preprocessing import build_preprocessor


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_vae_training_set(X_tr_pp, y_train, X_amb_pp, max_amb_ratio: Optional[float], seed: int):
    """labeled-train + (capped) censored samples.

    Censored rows carry label -1 and ``is_labeled = 0``: they contribute to reconstruction only.
    """
    n_lab = len(X_tr_pp)
    if len(X_amb_pp) == 0:
        return X_tr_pp, y_train.copy(), np.ones(n_lab, dtype=np.float32)
    X_amb_use = X_amb_pp
    if max_amb_ratio is not None:
        max_amb = int(np.ceil(max_amb_ratio * n_lab))
        if len(X_amb_pp) > max_amb:
            rng = np.random.default_rng(seed)
            X_amb_use = X_amb_pp[rng.choice(len(X_amb_pp), size=max_amb, replace=False)]
    X = np.concatenate([X_tr_pp, X_amb_use], axis=0)
    y = np.concatenate([y_train, -np.ones(len(X_amb_use), dtype=y_train.dtype)])
    is_lab = np.concatenate([np.ones(n_lab, dtype=np.float32), np.zeros(len(X_amb_use), dtype=np.float32)])
    return X, y, is_lab


def encode_features(vae, X_np, device):
    vae.eval()
    with torch.no_grad():
        mu, _ = vae.encode(torch.tensor(X_np, dtype=torch.float32).to(device))
    return mu.cpu().numpy()


# ----------------------------------------------------------------------------
# VAE training
# ----------------------------------------------------------------------------
def train_vae(X_train_np, y_train_np, is_labeled_np, X_val_np, input_dim, combo, device,
              cfg: CMVAEConfig, epochs: int, batch_size: int, lambda_contrast: float,
              model_cls=CMVAE, save_prefix: Optional[str] = None, verbose: bool = True
              ) -> Tuple[nn.Module, list, list]:
    
    reg_type = combo.split("+")[1]
    X_t = torch.tensor(X_train_np, dtype=torch.float32)
    y_t = torch.tensor(y_train_np, dtype=torch.long)
    lab_t = torch.tensor(is_labeled_np, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(X_t, y_t, lab_t), batch_size=batch_size, shuffle=True,
                              drop_last=True, generator=torch.Generator().manual_seed(cfg.seed))
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val_np, dtype=torch.float32)),
                            batch_size=batch_size, shuffle=False)

    vae = model_cls(input_dim, cfg.latent_dim).to(device)
    optimizer = optim.Adam(vae.parameters(), lr=cfg.lr)
    train_losses, val_losses, best_val, best_state, step = [], [], float("inf"), None, 0

    for epoch in range(epochs):
        ramp = min(1.0, (epoch + 1) / max(1, cfg.warmup_epochs))
        beta = (cfg.max_beta_kl if reg_type == "KL" else cfg.max_beta_mmd) * ramp

        vae.train()
        ep_loss, n_b = 0.0, 0
        for batch, y_b, lab_b in train_loader:
            batch, y_b, lab_b = batch.to(device), y_b.to(device), lab_b.to(device)
            x_hat, mu, logvar, z, proj = vae(batch, mask_ratio=cfg.mask_ratio)

            if reg_type == "KL":
                base_loss, _ = vae_loss(x_hat, batch, mu, logvar, beta=beta)
            else:
                base_loss, _ = mmd_vae_loss(x_hat, batch, z, lambda_mmd=beta)

            lab_mask = lab_b.bool()
            if (lambda_contrast > 0 and proj is not None and lab_mask.sum() >= 4
                    and len(torch.unique(y_b[lab_mask])) > 1):
                con = supervised_contrastive_loss(proj[lab_mask], y_b[lab_mask], cfg.contrastive_temp)
            else:
                con = torch.tensor(0.0, device=device)
            loss = base_loss + lambda_contrast * con

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimizer.step()
            step += 1
            for pg in optimizer.param_groups:
                pg["lr"] = cfg.lr * (cfg.lr_decay_rate ** (step / cfg.lr_decay_steps))
            ep_loss += loss.item()
            n_b += 1
        avg_train = ep_loss / max(1, n_b)

        vae.eval()
        ep_val, n_v = 0.0, 0
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                x_hat, mu, logvar, z, _ = vae(batch, mask_ratio=cfg.mask_ratio)
                if reg_type == "KL":
                    v, _ = vae_loss(x_hat, batch, mu, logvar, beta=beta)
                else:
                    v, _ = mmd_vae_loss(x_hat, batch, z, lambda_mmd=beta)
                ep_val += v.item()
                n_v += 1
        avg_val = ep_val / max(1, n_v)
        train_losses.append(avg_train)
        val_losses.append(avg_val)

        if avg_val < best_val:
            best_val = avg_val
            best_state = {k: v.clone() for k, v in vae.state_dict().items()}
            if save_prefix:
                torch.save(best_state, f"{save_prefix}_best.pt")
        if avg_train < cfg.loss_threshold:
            if verbose:
                print(f"      early stop @ epoch {epoch + 1}")
            break

    vae.load_state_dict(best_state)
    vae.eval()
    if verbose:
        mu = encode_features(vae, X_train_np[:min(64, len(X_train_np))], device)
        std = float(mu.std(axis=0).mean())
        print(f"      best val loss {best_val:.5f} (epoch {int(np.argmin(val_losses)) + 1}/{len(val_losses)})"
              f" | latent mean-std {std:.4f}{'' if std >= 0.05 else '  <-- LOW, possible collapse'}")
    return vae, train_losses, val_losses


# ----------------------------------------------------------------------------
# cross-validation driver
# ----------------------------------------------------------------------------
def run_cross_validation(cohort: Cohort, spec: OmicSpec, cfg: CMVAEConfig, folds_dir: str, out_dir: str,
                         device: torch.device, method: str = "CMVAE", model_cls=CMVAE,
                         lambda_contrast: Optional[float] = None, top_n_features: Optional[int] = None,
                         save_classifiers: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Run the full leak-proof CV for one omic and write models + result tables under ``out_dir``.

    ``method`` only labels the output rows ('CMVAE' or 'MAE').
    """
    lam = cfg.lambda_contrast if lambda_contrast is None else lambda_contrast
    top_n = spec.top_n_features if top_n_features is None else top_n_features
    epochs = cfg.epochs_override or spec.vae_epochs
    batch_size = cfg.batch_size_override or spec.vae_batch_size

    models_dir = os.path.join(out_dir, "models")
    results_dir = os.path.join(out_dir, "results")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    folds = load_or_create_folds(cohort, folds_dir, cfg.n_folds, cfg.seed)
    X_lab, y_lab, X_amb = cohort.X_labeled, cohort.y_labeled, cohort.X_ambiguous
    rows, loss_rows = [], []
    t0 = time.time()

    for fold, (trainval_idx, test_idx) in enumerate(folds):
        seed_everything(cfg.seed + fold)
        train_idx, val_idx = inner_split(trainval_idx, y_lab, cfg.inner_val_ratio, cfg.seed)
        y_train, y_val, y_test = y_lab[train_idx], y_lab[val_idx], y_lab[test_idx]

        # --- 1. fold-specific preprocessing, fitted on labeled-train only --------------------
        prep = build_preprocessor(spec.missing_rate_thresh, top_n, verbose=False)
        X_tr_pp = prep.fit_transform(X_lab[train_idx])
        X_val_pp = prep.transform(X_lab[val_idx])
        X_tst_pp = prep.transform(X_lab[test_idx])
        X_amb_pp = prep.transform(X_amb) if len(X_amb) else np.empty((0, X_tr_pp.shape[1]))
        input_dim = X_tr_pp.shape[1]
        joblib.dump(prep, os.path.join(models_dir, f"fold{fold}_preprocessor.pkl"))

        # --- 2. VAE training set: labeled-train + (capped) censored -------------------------
        X_vae, y_vae, is_lab = build_vae_training_set(X_tr_pp, y_train, X_amb_pp, cfg.max_amb_ratio,
                                                      cfg.seed + fold)
        if verbose:
            print(f"\n[{method} | {cohort.omic}] fold {fold}: labeled train={len(y_train)} val={len(y_val)} "
                  f"test={len(y_test)} | censored used={len(X_vae) - len(y_train)} | features={input_dim}")

        for combo in cfg.loss_combos:
            if verbose:
                print(f"  > {combo}")
            tag = combo.replace("+", "_")
            vae, tr_l, va_l = train_vae(X_vae, y_vae, is_lab, X_val_pp, input_dim, combo, device, cfg,
                                        epochs, batch_size, lam, model_cls,
                                        save_prefix=os.path.join(models_dir, f"fold{fold}_vae_{tag}"),
                                        verbose=verbose)
            for e, (a, b) in enumerate(zip(tr_l, va_l), start=1):
                loss_rows.append(dict(fold=fold, loss_combo=combo, epoch=e, train_loss=a, val_loss=b))

            # --- 3. embeddings ---------------------------------------------------------------
            Z_tr, Z_val, Z_tst = (encode_features(vae, X, device) for X in (X_tr_pp, X_val_pp, X_tst_pp))

            # --- 4/5. classifiers x oversampling -> test-fold metrics ----------------------------
            for clf_name in CLASSIFIERS:
                for os_name in OVERSAMPLERS:
                    X_fit, y_fit = oversample(os_name, Z_tr, y_train, cfg.seed)
                    clf = build_classifier(clf_name, "balanced" if os_name == "NO" else None, cfg.seed)
                    clf.fit(X_fit, y_fit)
                    if save_classifiers:
                        joblib.dump(clf, os.path.join(models_dir, f"fold{fold}_{tag}_{clf_name}_{os_name}.pkl"),
                                    compress=3)
                    m = metrics_row(y_val, clf.predict_proba(Z_val)[:, 1], y_test, clf.predict_proba(Z_tst)[:, 1])
                    rows.append(dict(fold=fold, method=method, loss_combo=combo, classifier=clf_name,
                                     oversampling=os_name, n_latent=cfg.latent_dim, **m))
                    if verbose:
                        print(f"    {clf_name:4s}/{os_name:5s} AUROC={m['auroc']:.3f} F1={m['f1']:.3f} "
                              f"BalAcc={m['balanced_acc']:.3f}")

    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(os.path.join(results_dir, "per_fold_results.csv"), index=False)
    pd.DataFrame(loss_rows).to_csv(os.path.join(results_dir, "loss_curves.csv"), index=False)
    summarize(per_fold, ["method", "loss_combo", "classifier", "oversampling"]) \
        .to_csv(os.path.join(results_dir, "display_summary.csv"), index=False)

    best = per_fold.groupby(["loss_combo", "classifier", "oversampling"])["auroc"].agg(["mean", "std"])
    (b_combo, b_clf, b_os), b = best["mean"].idxmax(), best.loc[best["mean"].idxmax()]
    print(f"\n[{method} | {cohort.omic}] best config by mean AUROC: {b_combo} | {b_clf} | {b_os} "
          f"-> {b['mean']:.4f} ± {b['std']:.4f}   ({(time.time() - t0) / 60:.1f} min)")
    return per_fold
