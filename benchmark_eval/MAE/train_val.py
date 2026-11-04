"""Train + cross-validate the MAE (masked denoising VAE, no contrastive loss) benchmark.

It re-uses the CMVAE training loop with ``lambda_contrast = 0`` and, as in the original experiments,
trains the VAE on labeled-train + ALL censored samples (no cap) and a per-omic top-N feature count
(``OmicSpec.mae_top_n_features``).

    python -m benchmark_eval.MAE.train_val --omic mRNA --data_dir data/dummy/BRCA
"""
import argparse
import os

from CMVAE.configs import OMIC_ORDER, SURVIVAL_FILE, CMVAEConfig, get_omic
from CMVAE.data import build_cohort
from CMVAE.train_val import get_device, run_cross_validation
from benchmark_eval.MAE.model import MAE


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--omic", default="mRNA", choices=OMIC_ORDER + ["all"])
    p.add_argument("--data_dir", default="data/dummy/BRCA")
    p.add_argument("--survival_file", default=SURVIVAL_FILE)
    p.add_argument("--out_dir", default="outputs", help="results go to <out_dir>/<omic>/MAE/")
    p.add_argument("--device", default="auto")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--quick", action="store_true", help="smoke test: 5 epochs, MSE+KL only")
    a = p.parse_args()

    cfg = CMVAEConfig(lambda_contrast=0.0, max_amb_ratio=None, epochs_override=a.epochs)
    if a.quick:
        cfg = cfg.with_updates(epochs_override=a.epochs or 5, loss_combos=("MSE+KL",))
    device = get_device(a.device)
    for name in (OMIC_ORDER if a.omic == "all" else [a.omic]):
        spec = get_omic(name)
        cohort = build_cohort(spec, a.data_dir, a.survival_file)
        run_cross_validation(cohort, spec, cfg,
                             folds_dir=os.path.join(a.out_dir, name, "folds"),
                             out_dir=os.path.join(a.out_dir, name, "MAE"),
                             device=device, method="MAE", model_cls=MAE, lambda_contrast=0.0,
                             top_n_features=spec.mae_top_n_features, save_classifiers=False)

if __name__ == "__main__":
    main()
