# Script to run feature importance experiments from CLI.
import pickle

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# from multimodal_survival.clustering_analysis import *
from multimodal_survival.utilities.utils import (
    get_topk_features_cluster_pairs,
    pairwise_anova,
)
from sklearn.impute import SimpleImputer

LIST_DATASETS = [
    "rppa_coadread_literature11_df_stdz",
    "dnameth_tcga_coadread_450_literature82_stdz",
    "mirna_coadread_literature30_df_stdz",
    "rnaseq_fpkm_coadread_colotype_df_stdz",
    "merged_cologex_literature_df_stdz_v2",
]


@click.command()
@click.option("--data_root_dir", help="Root directory of data.")
@click.option(
    "--clustering_result_dict_path",
    help="Path to clustering results stored as a dictionary.",
)
@click.option("--save_dir", help="Path to save results.")
@click.option("--k_features", default="all", help="Top k features to visualise.")
def main(data_root_dir, clustering_result_dict_path, save_dir, k_features):
    with open(clustering_result_dict_path, "rb") as f:
        kmeans_result_dict = pickle.load(f)
    sns.set(context="poster", style="white")
    for dataset in LIST_DATASETS:
        _, ax = plt.subplots(figsize=(16, 9))
        file = dataset + ".csv"
        df = pd.read_csv(data_root_dir / file, index_col=0)
        df = df.dropna(axis=1, how="all")
        df_columns = df.columns
        df = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(df), columns=df_columns)
        n_features = df.shape[1]

        kmeans_obj = kmeans_result_dict[dataset]["model"]
        _, p_df = pairwise_anova(df, kmeans_obj)

        counts = get_topk_features_cluster_pairs(
            p_df, df_columns, k=k_features, n_features=n_features
        )
        if isinstance(counts, pd.DataFrame):
            count_df = counts
            all_features = pd.unique(count_df.values.ravel())

            count_matrix = pd.DataFrame(0, index=all_features, columns=count_df.columns)

            # Fill the count matrix
            for col in count_df.columns:
                count_matrix.loc[list(count_df[col].values), [col]] += 1
            if len(all_features) < 30:
                yfontsize = 22
            else:
                yfontsize = 16
        elif isinstance(counts, dict):
            count_dict = counts
            all_features = []
            for value in count_dict.values():
                all_features += value

            all_features = np.unique(all_features)

            count_matrix = pd.DataFrame(0, index=all_features, columns=count_dict.keys())

            # Fill the count matrix
            for col in count_dict.keys():
                count_matrix.loc[list(count_dict[col]), [col]] += 1
            if len(all_features) < 30:
                yfontsize = 22
            else:
                yfontsize = 16

        sns.heatmap(
            count_matrix.loc[sorted(count_matrix.index)],
            cmap="YlGnBu",
            cbar=False,
            annot=False,
            annot_kws={"size": 10},
            fmt="d",
            linewidths=1.0,
            yticklabels=1,
            ax=ax,
        )
        plt.xticks(fontsize=26, rotation=0)  # Rotate x-axis labels for better fit
        plt.yticks(
            fontsize=yfontsize, rotation=0
        )  # Keep y-axis labels horizontal with reduced font size

        plt.xlabel("Cluster Pairs", fontsize=32)
        plt.savefig(save_dir / f"{dataset}_feature_importance.pdf", bbox_inches="tight", dpi=300)


if __name__ == "__main__":
    main()
