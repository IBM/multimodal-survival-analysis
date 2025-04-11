import pickle
from pathlib import Path

import pandas as pd

from multimodal_survival.utilities.data_processing import (
    get_dataset,
    get_fpkm_from_count,
)

SAVE_DIR = Path()
COLOTYPE_GENES_PATH = Path()
PROBEMAP_PATH = Path()
METHYLATION_GENEMAP_PATH = Path()

probemap = pd.read_csv(
    PROBEMAP_PATH,
    sep="\t",
)
probemap["length"] = probemap.chromEnd - probemap.chromStart
probemap = probemap.replace(
    {"WARS": "WARS1"}
)  # WARS == WARS1; naming inconsistency between files.
probemap = probemap.set_index("gene")

meth_gene_map = pd.read_csv(METHYLATION_GENEMAP_PATH, sep="\t")

rnaseq_counts_coad = "/data/TCGA-COAD.htseq_counts.tsv.gz"
rnaseq_counts_read = "/data/TCGA-READ.htseq_counts.tsv.gz"
dnameth_coadread_450k = "/data/dna_methylation/HumanMethylation450.gz"
mirna_coad = "/data/TCGA-COAD.mirna.tsv.gz"
mirna_read = "/data/TCGA-READ.mirna.tsv.gz"
rppa_coadread = "/data/RPPA_RBN.gz"
wsi_path = "/data/wsi_bag_of_patches_tfidf32.csv"

## Process RNA-Seq

colotype_genes = pd.read_csv(COLOTYPE_GENES_PATH, index_col=0)

rnaseq_counts_coad_df = pd.read_csv(rnaseq_counts_coad, index_col=0, sep="\t").T
rnaseq_counts_read_df = pd.read_csv(rnaseq_counts_read, index_col=0, sep="\t").T
rnaseq_counts_coadread_df = pd.concat([rnaseq_counts_coad_df, rnaseq_counts_read_df])
rnaseq_counts_coadread_df.columns = [
    col.split(".")[0] for col in rnaseq_counts_coadread_df.columns
]
idx_to_drop = [idx for idx in rnaseq_counts_coadread_df.index if "-01A" not in idx]
rnaseq_counts_coadread_df.drop(index=idx_to_drop, inplace=True)
rnaseq_counts_coadread_colotype_df = rnaseq_counts_coadread_df.loc[
    :, colotype_genes["ensemblid"]
]
rnaseq_counts_coadread_colotype_df.columns = colotype_genes["SYMBOL"]
gene_lengths = probemap["length"][rnaseq_counts_coadread_colotype_df.columns]
rnaseq_fpkm_coadread_colotype_df_stdz, rnaseq_scaler = get_fpkm_from_count(
    rnaseq_counts_coadread_colotype_df, gene_lengths, standardise=True
)
rnaseq_fpkm_coadread_colotype_df_stdz.index = [
    idx.rsplit("-", 1)[0] for idx in rnaseq_fpkm_coadread_colotype_df_stdz.index
]
rnaseq_fpkm_coadread_colotype_df_stdz.to_csv(
    SAVE_DIR / "rnaseq_fpkm_coadread_colotype_df_stdz.csv"
)

with open(SAVE_DIR / "rnaseq_scaler.pkl", "wb") as f:
    pickle.dump(rnaseq_scaler, f)

## Process RPPA

proteins_literature = [
    "BETACATENIN",
    "P53",
    "COLLAGENVI",
    "FOXO3A",
    "INPP4B",
    "PEA15",
    "PRAS40PT246",
    "RAD51",
    "S6",
    "S6PS235S236",
    "S6PS240S244",
]
rppa_coadread_literature11_df_stdz, rppa_scaler = get_dataset(
    [rppa_coadread], features_subset=proteins_literature, standardise=True
)

rppa_coadread_literature11_df_stdz.to_csv(
    SAVE_DIR / "rppa_coadread_literature11_df_stdz.csv"
)
with open(SAVE_DIR / "rppa_scaler.pkl", "wb") as f:
    pickle.dump(rppa_scaler, f)

## Process DNA Methylation
dna_probes = [
    "cg17477990",
    "cg11125249",
    "cg02827572",
    "cg04739880",
    "cg00512872",
    "cg14754494",
    "cg19107055",
    "cg05357660",
    "cg23045908",
    "cg16708174",
    "cg00901138",
    "cg00901574",
    "cg16477879",
    "cg23219253",
    "cg05211192",
    "cg12492273",
    "cg16772998",
    "cg00145955",
    "cg00097384",
    "cg27603796",
    "cg23418465",
    "cg17842966",
    "cg19335412",
    "cg23928468",
    "cg05951860",
    "cg20698769",
    "cg06786372",
    "cg17301223",
    "cg15638338",
    "cg02583465",
    "cg18065361",
    "cg06551493",
    "cg12691488",
    "cg17292758",
    "cg16170495",
    "cg21585512",
    "cg24702253",
    "cg17187762",
    "cg05983326",
    "cg11885357",
]
meth_genes = [
    "DUSP9",
    "GHSR",
    "MMP9",
    "PAQR9",
    "PRKG2",
    "PTH2R",
    "SLITRK4",
    "TIAM1",
    "SEMA7A",
    "GATA4",
    "LHX2",
    "SOST",
    "CTLA4",
    "NMNAT2",
    "ZFP42",
    "NPAS2",
    "MYLK3",
    "NUDT13",
    "KIRREL3",
    "FKBP6",
    "SOST",
    "NFATC1",
    "TLE4",
]

dna_probes_meth_genes = meth_gene_map[meth_gene_map["gene"].isin(meth_genes)][
    "#id"
].tolist()

dnameth_literature = dna_probes_meth_genes + dna_probes

dnameth_tcga_coadread_450_literature82_stdz, meth_scaler = get_dataset(
    [dnameth_coadread_450k], features_subset=dnameth_literature, standardise=True
)

dnameth_tcga_coadread_450_literature82_stdz.to_csv(
    SAVE_DIR / "dnameth_tcga_coadread_450_literature82_stdz.csv"
)
with open(SAVE_DIR / "meth_scaler.pkl", "wb") as f:
    pickle.dump(meth_scaler, f)

## Process miRNA
mirna_features = [
    "miR-17",
    "miR-18a",
    "miR-19a",
    "miR-19b",
    "miR-20a",
    "miR-21",
    "miR-27a",
    "miR-29a",
    "miR-91",
    "miR-92a-1",
    "miR-92a-2",
    "miR-135b",
    "miR-223",
    "miR-200a",
    "miR-200b",
    "miR-200c",
    "miR-141",
    "miR-143",
    "miR-145",
    "miR-221",
    "miR-222",
    "miR-99a",
    "miR-100",
    "miR-144",
    "miR-486-1",
    "miR-486-2",
    "miR-15b",
    "miR-1247",
    "miR-584",
    "miR-483",
    "miR-10a",
    "miR-425",
]
mirna_features = list(map(lambda x: ("hsa-" + x).lower(), mirna_features))

mirna_coadread_literature30_df_stdz, mirna_scaler = get_dataset(
    [mirna_coad, mirna_read], features_subset=mirna_features, standardise=True
)
mirna_coadread_literature30_df_stdz.to_csv(
    SAVE_DIR / "mirna_coadread_literature30_df_stdz.csv"
)
with open(SAVE_DIR / "mirna_scaler.pkl", "wb") as f:
    pickle.dump(mirna_scaler, f)


## Create multi-omics dataset
omics_files = [
    SAVE_DIR / "rnaseq_fpkm_coadread_colotype_df_stdz.csv",
    SAVE_DIR / "dnameth_tcga_coadread_450_literature82_stdz.csv",
    SAVE_DIR / "rppa_coadread_literature11_df_stdz.csv",
    SAVE_DIR / "mirna_coadread_literature30_df_stdz.csv",
]

df_list = []
for file in omics_files:
    df = pd.read_csv(file, index_col=0)
    df_list.append(df)
merged_cologex_literature_df_outer = df_list[0].join(df_list[1:], how="outer")
merged_cologex_literature_df_inner = df_list[0].join(df_list[1:], how="inner")

merged_cologex_literature_df_outer.dropna(axis=1, how="all", inplace=True)
merged_cologex_literature_df_outer.to_csv(
    SAVE_DIR / "merged_cologex_literature_df_stdz_v2.csv"
)

merged_cologex_literature_df_inner.dropna(axis=1, how="all", inplace=True)
merged_cologex_literature_df_inner.to_csv(
    SAVE_DIR / "merged_cologex_literature_df_stdz_v2_innerjoin.csv"
)

## Add WSI embeddings from VIT
files_to_merge = {
    "merged_cologex_literature_df_stdz_v2_innerjoin": {
        "data": merged_cologex_literature_df_inner,
        "how": "inner",
    },
    "merged_cologex_literature_df_stdz_v2": {
        "data": merged_cologex_literature_df_outer,
        "how": "outer",
    },
}


wsi_df = pd.read_csv(wsi_path, index_col=0)

for name, file_dict in files_to_merge.items():
    filename = "vit_tfidf32_" + name + ".csv"
    wsi_merged_df = file_dict["data"].join(wsi_df, how=file_dict["how"])
    wsi_merged_df.to_csv(SAVE_DIR / filename)
