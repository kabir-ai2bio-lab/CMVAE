"""Download the real TCGA-BRCA files from the UCSC Xena public data hub (needs internet access).

Files are written to ``data/BRCA/`` with the exact names the code expects:

  TCGA.BRCA.sampleMap_HiSeqV2                                        mRNA, log2(norm_count + 1), Illumina HiSeq
  TCGA.BRCA.sampleMap_HumanMethylation450                            DNA methylation beta values, Illumina 450K
  TCGA.BRCA.sampleMap_miRNA_HiSeq_gene                               miRNA, log2(RPM + 1), Illumina HiSeq
  TCGA.BRCA.sampleMap_Gistic2_CopyNumber_Gistic2_all_data_by_genes   gene-level CNV, GISTIC2
  survival_BRCA_survival.txt                                         clinical survival (OS, OS.time)

If a download fails (the hub layout can change), fetch the same datasets manually from
https://xenabrowser.net/datapages/ (cohort "GDC/TCGA Breast Cancer (BRCA)" -> legacy "TCGA Breast Cancer
(BRCA)" -> "Download"), unzip, and rename them to the names above.

    python data/download_xena_data.py            # -> data/BRCA
"""
import argparse
import gzip
import os
import shutil
import urllib.request

HUB = "https://tcga-xena-hub.s3.us-east-1.amazonaws.com/download"
DATASETS = {   # local name -> remote path on the hub
    "TCGA.BRCA.sampleMap_HiSeqV2": "TCGA.BRCA.sampleMap%2FHiSeqV2.gz",
    "TCGA.BRCA.sampleMap_HumanMethylation450": "TCGA.BRCA.sampleMap%2FHumanMethylation450.gz",
    "TCGA.BRCA.sampleMap_miRNA_HiSeq_gene": "TCGA.BRCA.sampleMap%2FmiRNA_HiSeq_gene.gz",
    "TCGA.BRCA.sampleMap_Gistic2_CopyNumber_Gistic2_all_data_by_genes":
        "TCGA.BRCA.sampleMap%2FGistic2_CopyNumber_Gistic2_all_data_by_genes.gz",
    "survival_BRCA_survival.txt": "survival%2FBRCA_survival.txt",   # plain text, not gzipped
}


def fetch(local_name: str, remote: str, out_dir: str):
    dest = os.path.join(out_dir, local_name)
    if os.path.exists(dest):
        print(f"[skip] {local_name} already present")
        return
    url = f"{HUB}/{remote}"
    tmp = dest + ".download"
    print(f"[get ] {url}")
    urllib.request.urlretrieve(url, tmp)
    if remote.endswith(".gz"):
        with gzip.open(tmp, "rb") as src, open(dest, "wb") as out:
            shutil.copyfileobj(src, out)
        os.remove(tmp)
    else:
        os.replace(tmp, dest)
    print(f"[done] {dest}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "BRCA"))
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    for name, remote in DATASETS.items():
        try:
            fetch(name, remote, a.out_dir)
        except Exception as e:   
            print(f"[FAIL] {name}: {e}")


if __name__ == "__main__":
    main()
