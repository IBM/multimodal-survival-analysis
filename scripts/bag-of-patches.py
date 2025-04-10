import pickle
from collections import Counter
from pathlib import Path

import click
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize

from multimodal_survival.clustering import *


@click.command()
@click.option(
    "--data_path",
    help="path to data to be clustered.",
    type=click.Path(path_type=Path, exists=True),
)
@click.option("--n_cluster", type=int, help="number of clusters to use for kmeans.")
@click.option(
    "--save_dir",
    type=click.Path(path_type=Path, exists=True),
    help="path to save results.",
)
def main(data_path, n_cluster, save_dir):
    wsi_patches = pd.read_csv(data_path, memory_map=True, index_col=0)
    result_dict = {}
    # perform clustering on data; identify optimal number of clusters
    kmeans = KMeans(n_clusters=n_cluster, random_state=42)
    kmeans.fit(normalize(wsi_patches))

    pred_labels = kmeans.labels_

    result_dict["kmeans"] = kmeans
    result_dict["cluster_labels"] = pred_labels
    result_dict["inertia"] = kmeans.inertia_

    with open(save_dir / f"{n_cluster}_kmeans_result_dict.pkl", "wb") as f:
        pickle.dump(result_dict, f)

    patch_labels_df = pd.DataFrame(pred_labels, index=wsi_patches.index)
    patch_labels_df.to_csv(save_dir / f"{n_cluster}_predicted_labels.csv")

    # TF-IDF
    patch_labels_count = (
        patch_labels_df.groupby("patient_id")["0"]
        .apply(lambda x: Counter(x))
        .replace(np.nan, 0.0)
        .unstack()
    )
    patch_labels_count = patch_labels_count.sort_index(axis=1)
    tfidfer = TfidfTransformer()
    patch_labels_tfidf = tfidfer.fit_transform(patch_labels_count)

    pd.DataFrame(patch_labels_tfidf.todense(), index=patch_labels_count.index).to_csv(
        save_dir / "wsi_bag_of_patches_tfidf32.csv"
    )

    # Normalised count
    patch_labels_df = (
        patch_labels_df.groupby("patient_id")["0"]
        .apply(lambda x: Counter(x))
        .replace(np.nan, 0.0)
        .unstack()
    )
    patch_labels_df = patch_labels_df.sort_index(axis=1)
    patch_labels_df_norm = patch_labels_df.apply(lambda x: x / sum(x), axis=1)
    patch_labels_df_norm.to_csv(save_dir / "wsi_bag_of_patches_representation32.csv")


if __name__ == "__main__":
    main()
