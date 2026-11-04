"""Train and cross-validate CMVAE on one or more omic cohorts.

Examples
--------
# bundled synthetic data, quick smoke test (a few minutes on CPU)
python run_cmvae.py --omic mRNA --data_dir data/dummy/BRCA --quick

# all four omics on the bundled synthetic data and hyper-parameters.
python run_cmvae.py --omic all --data_dir data/dummy/BRCA

# real UCSC Xena files (see data/README.md)
python run_cmvae.py --omic mRNA --data_dir data/BRCA
"""
import argparse
import os

from CMVAE.configs import OMIC_ORDER, SURVIVAL_FILE, CMVAEConfig, get_omic
from CMVAE.data import build_cohort
from CMVAE.model import CMVAE
from CMVAE.train_val import get_device, run_cross_validation


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--omic", default="mRNA", choices=OMIC_ORDER + ["all"], help="omic cohort to run")
    p.add_argument("--data_dir", default="data/dummy/BRCA", help="folder with the Xena omic + survival files")
    p.add_argument("--survival_file", default=SURVIVAL_FILE)
    p.add_argument("--out_dir", default="outputs", help="results go to <out_dir>/<omic>/CMVAE/")
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=None, help="override the per-omic epoch count")
    p.add_argument("--batch_size", type=int, default=None, help="override the per-omic batch size")
    p.add_argument("--lambda_contrast", type=float, default=0.1, help="weight of the supervised contrastive loss")
    p.add_argument("--contrastive_temp", type=float, default=0.2)
    p.add_argument("--mask_ratio", type=float, default=0.15)
    p.add_argument("--loss_combos", nargs="+", default=["MSE+KL", "MSE+MMD"], choices=["MSE+KL", "MSE+MMD"])
    p.add_argument("--no_save_classifiers", action="store_true", help="do not pickle the 90 classifiers per omic")
    p.add_argument("--quick", action="store_true", help="smoke test: 5 epochs, MSE+KL only")
    return p.parse_args()


def main():
    a = parse_args()
    cfg = CMVAEConfig(seed=a.seed, n_folds=a.n_folds, lambda_contrast=a.lambda_contrast,
                      contrastive_temp=a.contrastive_temp, mask_ratio=a.mask_ratio,
                      loss_combos=tuple(a.loss_combos), epochs_override=a.epochs, batch_size_override=a.batch_size)
    if a.quick:
        cfg = cfg.with_updates(epochs_override=a.epochs or 5, loss_combos=("MSE+KL",))

    device = get_device(a.device)
    print(f"device: {device}")
    omics = OMIC_ORDER if a.omic == "all" else [a.omic]
    for name in omics:
        spec = get_omic(name)
        cohort = build_cohort(spec, a.data_dir, a.survival_file, cfg.survival_thresh_days)
        run_cross_validation(
            cohort, spec, cfg,
            folds_dir=os.path.join(a.out_dir, name, "folds"),
            out_dir=os.path.join(a.out_dir, name, "CMVAE"),
            device=device, method="CMVAE", model_cls=CMVAE,
            save_classifiers=not a.no_save_classifiers)


if __name__ == "__main__":
    main()
