"""Central configuration for CMVAE and all benchmark methods.

Every number here was taken from our experiment notebooks used for the original experimentations. Nothing is read from environment variables or hard-coded
absolute paths. Data/output locations are always passed in by the caller.
"""
from dataclasses import dataclass, replace
from typing import Optional

# ----------------------------------------------------------------------------
# Global experiment settings (shared by CMVAE and every benchmark)
# ----------------------------------------------------------------------------
SEED = 42
N_FOLDS = 5
SURVIVAL_THRESH_DAYS = 5 * 365.25  # 1826.25 days = five-year survival threshold
SURVIVAL_FILE = "survival_BRCA_survival.txt"

CLASSIFIERS = ["LR", "RF", "PSVM"]          # PSVM = SVM with polynomial kernel
OVERSAMPLERS = ["NO", "ROS", "SMOTE"]       # NO = no oversampling (class_weight="balanced")
OMIC_ORDER = ["mRNA", "methylation", "miRNA", "cnv"]


@dataclass(frozen=True)
class OmicSpec:
    """Per-omic settings.  Field groups map to the method that uses them."""

    name: str
    file_name: str                       # UCSC Xena file name (features x samples TSV)

    # ---- CMVAE (contrastive masked denoising VAE) ---------------------------
    missing_rate_thresh: float           # drop features missing in this fraction of train samples
    top_n_features: int                  # keep top-N most variable features (variance on train fold)
    vae_epochs: int
    vae_batch_size: int

    # ---- MAE benchmark (masked denoising VAE, no contrastive term) ----------
    mae_top_n_features: int

    # ---- Raw / PCA / Gini benchmarks ----------------------------------------
    classical_filter: str                # "zero_or_nan" | "nan_only"  (feature filter, frac <= 0.20)
    pca_components: int
    gini_top_n: int
    raw_top_n: int = 500

    # ---- AE / VAE benchmarks --------------------------------------------------
    deep_filter: str = "zero_or_nan"     # "zero_or_nan" | "nan_only"
    deep_top_n_variance: Optional[int] = None   # variance pre-filter before AE/VAE (None = no pre-filter)


OMICS = {
    "mRNA": OmicSpec(
        name="mRNA", file_name="TCGA.BRCA.sampleMap_HiSeqV2",
        missing_rate_thresh=0.90, top_n_features=2000, vae_epochs=80, vae_batch_size=64,
        mae_top_n_features=1000,
        classical_filter="zero_or_nan", pca_components=100, gini_top_n=200,
        deep_filter="zero_or_nan", deep_top_n_variance=2000,
    ),
    "methylation": OmicSpec(
        name="methylation", file_name="TCGA.BRCA.sampleMap_HumanMethylation450",
        missing_rate_thresh=0.10, top_n_features=5000, vae_epochs=80, vae_batch_size=64,
        mae_top_n_features=5000,
        classical_filter="nan_only", pca_components=100, gini_top_n=200,
        deep_filter="nan_only", deep_top_n_variance=2000,
    ),
    "miRNA": OmicSpec(
        name="miRNA", file_name="TCGA.BRCA.sampleMap_miRNA_HiSeq_gene",
        missing_rate_thresh=0.20, top_n_features=500, vae_epochs=150, vae_batch_size=32,
        mae_top_n_features=500,
        classical_filter="zero_or_nan", pca_components=50, gini_top_n=100,
        deep_filter="zero_or_nan", deep_top_n_variance=None,
    ),
    "cnv": OmicSpec(
        name="cnv", file_name="TCGA.BRCA.sampleMap_Gistic2_CopyNumber_Gistic2_all_data_by_genes",
        missing_rate_thresh=0.20, top_n_features=2000, vae_epochs=80, vae_batch_size=64,
        mae_top_n_features=500,
        classical_filter="nan_only", pca_components=100, gini_top_n=200,
        deep_filter="zero_or_nan", deep_top_n_variance=None,
    ),
}


def get_omic(name: str) -> OmicSpec:
    if name not in OMICS:
        raise ValueError(f"Unknown omic '{name}'. Choose from {list(OMICS)}")
    return OMICS[name]


# ----------------------------------------------------------------------------
# CMVAE / MAE hyper-parameters (identical across omics except where OmicSpec says)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class CMVAEConfig:
    seed: int = SEED
    n_folds: int = N_FOLDS
    survival_thresh_days: float = SURVIVAL_THRESH_DAYS
    inner_val_ratio: float = 0.15        

    # architecture/optimisation
    latent_dim: int = 128
    lr: float = 1e-3
    lr_decay_steps: int = 50_000
    lr_decay_rate: float = 0.96
    loss_threshold: float = 1e-4         
    warmup_epochs: int = 20             
    mask_ratio: float = 0.15          

    # regularisers: loss_combo -> maximum weight
    max_beta_kl: float = 0.01
    max_beta_mmd: float = 1.0
    loss_combos: tuple = ("MSE+KL", "MSE+MMD")

    # contrastive term
    lambda_contrast: float = 0.1         
    contrastive_temp: float = 0.2        
    max_amb_ratio: Optional[float] = 1.0                                      

    # overrides applied by run scripts (e.g. --epochs 5 for a smoke test)
    epochs_override: Optional[int] = None
    batch_size_override: Optional[int] = None

    def with_updates(self, **kw) -> "CMVAEConfig":
        return replace(self, **kw)
