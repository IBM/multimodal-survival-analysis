from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score

from multimodal_survival.clustering import Clustering


def run_clustering_analysis_with_elbow(
    data_dir: Path,
    imputer: Callable,
    embedding_method: Callable,
    n_clusters: Iterable,
    save_dir: Path,
) -> Tuple:
    """Performs clustering with k-means for different k and selects best k by elbow method.

    Args:
        data_dir: Path to data file.
        imputer: Imputer object to fill in missing values. Should have fit_transform function.
        embedding_method: Embedding method for plotting. Eg, UMAP.umap()
        n_clusters: Number of clusters k to test.
        save_dir: Directory where results are saved.

    Returns:
        Best k-means model, predicted labels of the data, and a dictionary of clustering metrics.
    """

    df = pd.read_csv(data_dir, index_col=0)
    clustering_object = Clustering()
    if df.isna().sum().sum() > 0:
        df = imputer.fit_transform(df)
    best_km, silhouette_metrics = clustering_object.kmeans_elbow(
        df, n_clusters, save_dir / f"kmeans_elbow_{data_dir.stem}.pdf"
    )
    df_pred_labels = best_km.labels_
    silhoutte_avg = clustering_object.plot_silhouette_analysis(
        df,
        embedding_method,
        df_pred_labels,
        best_km.n_clusters,
        save_dir / f"silhouette_{data_dir.stem}.pdf",
    )
    db_score = davies_bouldin_score(df, df_pred_labels)
    ch_score = calinski_harabasz_score(df, df_pred_labels)
    return (
        best_km,
        df_pred_labels,
        {
            "best_cluster_silhouette_avg": silhoutte_avg,
            "db_score": db_score,
            "ch_score": ch_score,
            "best_k": best_km.n_clusters,
            "all_silhouette": silhouette_metrics,
        },
    )


def run_clustering_analysis(
    data_dir: Path,
    imputer: Callable,
    embedding_method: Callable,
    clustering_method: str,
    n_clusters: int,
    save_dir: Path,
    **kwargs,
) -> Tuple:
    """Performs clustering with specified method.

    Args:
        data_dir: Path to data file.
        imputer: Imputer object to fill in missing values. Should have fit_transform function.
        embedding_method: Embedding method for plotting. Eg, UMAP.umap()
        clustering_method: Name of clustering method. Check registry.py for options.
        n_clusters: Number of clusters k to test.
        save_dir: Directory where results are saved.

    Returns:
        Best clustering model, predicted labels of the data, and a dictionary of clustering metrics.
    """

    df = pd.read_csv(data_dir, index_col=0)
    clustering_object = Clustering()
    if df.isna().sum().sum() > 0:
        df = imputer.fit_transform(df)
    # random_state=42
    best_model, df_pred_labels, _ = clustering_object.clustering(
        df, clustering_method, n_clusters, **kwargs
    )
    silhoutte_avg = clustering_object.plot_silhouette_analysis(
        df,
        embedding_method,
        df_pred_labels,
        best_model.n_clusters,
        save_dir / f"silhouette_{data_dir.stem}.pdf",
    )
    db_score = davies_bouldin_score(df, df_pred_labels)
    ch_score = calinski_harabasz_score(df, df_pred_labels)
    return (
        best_model,
        df_pred_labels,
        {
            "silhouette_avg": silhoutte_avg,
            "db_score": db_score,
            "ch_score": ch_score,
            "best_k": best_model.n_clusters,
        },
    )


def run_dbscan_clustering(
    data_dir: Path,
    imputer: Callable,
    embedding_method: Callable,
    save_dir: Path,
    **dbscan_args,
) -> Tuple:
    """Performs DBScan clustering.

    Args:
        data_dir: Path to data file.
        imputer: Imputer object to fill in missing values. Should have fit_transform function.
        embedding_method: Embedding method for plotting. Eg, UMAP.umap()
        save_dir: Directory where results are saved.

    Returns:
        Fit dbscan model, predicted labels of the data, and a dictionary of clustering metrics.
    """

    df = pd.read_csv(data_dir, index_col=0)
    clustering_object = Clustering()
    if df.isna().sum().sum() > 0:
        df = imputer.fit_transform(df)

    dbscan = DBSCAN(**dbscan_args)
    dbscan.fit(df)
    df_pred_labels = dbscan.labels_
    n_clusters_ = len(set(df_pred_labels)) - (1 if -1 in df_pred_labels else 0)
    silhoutte_avg = clustering_object.plot_silhouette_analysis(
        df,
        embedding_method,
        df_pred_labels,
        n_clusters_,
        save_dir / f"silhouette_{data_dir.stem}.pdf",
    )
    db_score = davies_bouldin_score(df, df_pred_labels)
    ch_score = calinski_harabasz_score(df, df_pred_labels)
    return (
        dbscan,
        df_pred_labels,
        {
            "silhouette_avg": silhoutte_avg,
            "db_score": db_score,
            "ch_score": ch_score,
            "best_k": n_clusters_,
            "n_noise": list(df_pred_labels).count(-1),
        },
    )


def analyse_cluster_targets(
    df: pd.DataFrame, targets: pd.DataFrame, target_name: str, cluster_labels: List
) -> Dict:
    """Determines the distribution of true labels across the identified clusters.

    Args:
        df: Original dataframe to retrieve the correct indices of cluster labels.
        targets: True labels.
        target_name: Name of target to index from dataframe.
        cluster_labels: Predicted cluster labels of the data.

    Returns:
        Label distribution across clusters. Eg key = OS=0, value = #0 in clusters 1,2,3.
    """

    common_labels = list(set(targets.index) & set(df.index))
    cluster_labels_df = pd.Series(cluster_labels, index=df.index)
    label_distribution_across_clusters = dict()
    for target in pd.unique(targets[target_name]):
        target_idx = (
            targets.loc[common_labels]
            .query(f"{target_name} == '{target}'")
            .index.tolist()
        )
        label_distribution_across_clusters[str(target)] = Counter(
            cluster_labels_df[target_idx]
        )

    return label_distribution_across_clusters
