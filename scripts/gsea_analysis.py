# Script to run gene set enrichment analysis.
import fnmatch
import pickle
from itertools import combinations
from pathlib import Path

import click
import gseapy as gp
import mygene
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import f_oneway
from sklearn.impute import SimpleImputer

from multimodal_survival.utilities.utils import *
from multimodal_survival.utilities.utils import flatten

PROTEIN_MAP = {
    "BETACATENIN": "CTNNB1",
    "P53": "TP53",
    "COLLAGENVI": ["COL6A1", "COL6A2", "COL6A3"],
    "FOXO3A": "FOXO3",
    "INPP4B": "INPP4B",
    "PEA15": "PEA15",
    "PRAS40PT246": "AKT1",
    "RAD51": "RAD51",
    "S6": "RPS6",
    "S6PS235S236": "RPS6",
    "S6PS240S244": "RPS6",
}

# cluster = 3


@click.command()
@click.option("--cluster", type=int, help="ID of cluster for which GSEA is needed.")
@click.option("--data_root_dir", help="Root directory of data.")
@click.option("--dataset", type=str, help="Dataset to conduct GSEA on.")
@click.option(
    "--clustering_result_dict_path",
    help="Path to clustering results stored as a dictionary.",
)
@click.option(
    "--mirna_targets_mirmap_path",
    help="Path to mirna-gene target map as compiled by MiRmap.",
)
@click.option(
    "--mirna_id_converter_path",
    help="Path to file containing mirna ids from MiRAndola database.",
)
@click.option(
    "--dna_probe_map_path",
    help="Path to DNA methylation probemap file.",
)
@click.option("save_dir", help="Path to save results.")
@click.option("--k_features", default=10, help="Top k features to visualise.")
def main(
    cluster,
    data_root_dir,
    dataset,
    clustering_result_dict_path,
    mirna_targets_mirmap_path,
    mirna_id_converter_path,
    dna_probe_map_path,
    save_dir,
    k_features,
):

    with open(clustering_result_dict_path, "rb") as f:
        kmeans_result_dict = pickle.load(f)

    mirna_targets_mirmap = pd.read_csv(
        mirna_targets_mirmap_path,
        usecols=["mirna_id", "transcript_stable_id"],
    )
    mirna_id_converter = pd.read_csv(mirna_id_converter_path, sep="\t")
    dna_probe_map = pd.read_csv(
        dna_probe_map_path,
        sep="\t",
        index_col=0,
    )

    file = dataset + ".csv"
    df = pd.read_csv(data_root_dir / file, index_col=0)
    df = df.dropna(axis=1, how="all")
    df_columns = df.columns
    df = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(df), columns=df_columns
    )
    kmeans_obj = kmeans_result_dict[dataset]["model"]
    f_df, p_df = pairwise_anova(df, kmeans_obj)

    count_df = get_topk_features_cluster_pairs(p_df, df_columns, k=k_features)
    # Get unique features
    all_features = pd.unique(count_df.values.ravel())

    # Create a new DataFrame with features as index and original columns
    count_matrix = pd.DataFrame(0, index=all_features, columns=count_df.columns)
    for col in count_df.columns:
        count_matrix.loc[list(count_df[col].values), [col]] += 1

    cluster_cols = [cols for cols in count_df.columns if cluster in cols]
    cluster_features = set(count_df[cluster_cols].values.ravel())

    # get cluster gene list
    cluster_gene_list = []
    mirna_list = fnmatch.filter(cluster_features, "*hsa*")
    mirna_list = list(map(lambda x: x.replace("r", "R"), mirna_list))
    dna_probe_list = fnmatch.filter(cluster_features, "*cg*")

    cluster_gene_list.append(
        mirna_mapper_mirmap(mirna_list, mirna_id_converter, mirna_targets_mirmap)
    )
    cluster_gene_list.append(dna_meth_mapper(dna_probe_list, dna_probe_map))

    cluster_features_protein_genes = (
        set(cluster_features) - set(mirna_list) - set(dna_probe_list)
    )

    protein_genes = [
        value
        for key, value in PROTEIN_MAP.items()
        if key in cluster_features_protein_genes
    ]
    cluster_features_genes = cluster_features_protein_genes - set(PROTEIN_MAP.keys())

    cluster_gene_list += protein_genes + list(cluster_features_genes)
    cluster_gene_list = list(flatten(cluster_gene_list))

    # get background gene list
    background_features = list(df_columns)
    background_gene_list = []
    bg_mirna_list = fnmatch.filter(background_features, "*hsa*")
    bg_mirna_list = list(map(lambda x: x.replace("r", "R"), bg_mirna_list))
    bg_dna_probe_list = fnmatch.filter(background_features, "*cg*")

    background_gene_list.append(
        mirna_mapper_mirmap(bg_mirna_list, mirna_id_converter, mirna_targets_mirmap)
    )
    background_gene_list.append(dna_meth_mapper(bg_dna_probe_list, dna_probe_map))

    background_features_protein_genes = (
        set(background_features) - set(bg_mirna_list) - set(bg_dna_probe_list)
    )

    bg_protein_genes = [
        value
        for key, value in PROTEIN_MAP.items()
        if key in background_features_protein_genes
    ]
    background_features_genes = background_features_protein_genes - set(
        PROTEIN_MAP.keys()
    )

    background_gene_list += bg_protein_genes + list(background_features_genes)
    background_gene_list = list(flatten(background_gene_list))

    # run enrichr
    enr_bg = gp.enrichr(
        gene_list=list(map(str.upper, cluster_gene_list)),
        gene_sets=["MSigDB_Hallmark_2020", "KEGG_2021_Human"],
        organism="human",
        outdir=None,
        background=set(map(str.upper, background_gene_list)),
    )
    # save results
    enr_bg.results.sort_values("Adjusted P-value").to_csv(
        save_dir / "gsea_analysis.csv"
    )


if __name__ == "__main__":
    main()
