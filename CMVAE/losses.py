"""CMVAE objective:  L_total = L_rec + alpha * L_reg + beta * L_supcon"""
import torch
import torch.nn.functional as F


def reconstruction_loss(x_hat, x):
    return F.mse_loss(x_hat, x, reduction="mean")


def kl_loss(mu, logvar):
    logvar_c = logvar.clamp(-4, 4)
    return -0.5 * torch.mean(torch.sum(1 + logvar_c - mu.pow(2) - logvar_c.exp(), dim=1))


def compute_mmd(z, z_prior, sigma: float = 1.0):
    k = lambda a, b: torch.exp(-torch.cdist(a, b, p=2).pow(2) / (2 * sigma * sigma))
    return k(z, z).mean() + k(z_prior, z_prior).mean() - 2 * k(z, z_prior).mean()


def vae_loss(x_hat, x, mu, logvar, beta=1.0):
    recon = reconstruction_loss(x_hat, x)
    kl = kl_loss(mu, logvar)
    return recon + beta * kl, {"recon": recon.item(), "kl": kl.item()}


def mmd_vae_loss(x_hat, x, z, lambda_mmd=1.0, sigma=1.0):
    recon = reconstruction_loss(x_hat, x)
    mmd = compute_mmd(z, torch.randn_like(z), sigma=sigma)
    return recon + lambda_mmd * mmd, {"recon": recon.item(), "mmd": mmd.item()}


def supervised_contrastive_loss(features, labels, temperature: float = 0.2):
    features = F.normalize(features, dim=1)
    sim = torch.matmul(features, features.T) / temperature         

    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.T).float()
    pos_mask.fill_diagonal_(0.0)                                    

    logits_mask = torch.ones_like(sim) - torch.eye(sim.size(0), device=sim.device)
    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    pos_count = pos_mask.sum(dim=1)
    valid = pos_count > 0
    mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / (pos_count + 1e-8)

    if valid.any():
        return -mean_log_prob_pos[valid].mean()
    return torch.tensor(0.0, device=features.device)
