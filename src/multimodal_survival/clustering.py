import random
from typing import Callable, Dict, Iterable, Tuple

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from kneed import KneeLocator
from numpy.typing import ArrayLike
from sklearn import mixture
from sklearn.cluster import KMeans

from multimodal_survival.utilities.registries import (
    CLUSTERING_METHOD_FACTORY,
    CLUSTERING_METRIC_FACTORY,
)

sns.set(context="poster", style="white")


class Clustering:
    def __init__(self):
        pass

    def compute_metric(
        self,
        true_labels: ArrayLike,
        pred_labels: ArrayLike,
        metric: dict = {
            "ari": True,
            "nmi": True,
            "ami": True,
            "homogeneity": True,
            "completeness": True,
            "vmeasure": True,
        },
    ) -> Dict:
        """Computes cluster metrics.

        Args:
            true_labels: Groundtruth labels.
            pred_labels: Predicted labels.
            metric: Boolean valued dictionary of metrics to compute. Defaults to { "ari": True, "nmi": True, "ami": True, "homogeneity": True, "completeness": True, "vmeasure": True, }.

        Returns:
            Computed metrics.
        """

        scores = {k: [] for k in metric.keys()}

        for key, value in metric.items():
            if value:
                scores[key].append(
                    CLUSTERING_METRIC_FACTORY[key](true_labels, pred_labels)
                )

        return scores

    def cluster_model_selection(
        self,
        data: ArrayLike,
        true_labels: ArrayLike,
        method: str,
        n_clusters: Iterable,
        metric: dict = {
            "ari": True,
            "nmi": True,
            "ami": True,
            "homogeneity": True,
            "completeness": True,
            "vmeasure": True,
        },
        save_here: str = "clustering_scores.svg",
        **kwargs,
    ) -> None:
        """Model selection for a clustering method based on select metrics.

        Args:
            data: Data to cluster.
            true_labels: Groundtruth labels.
            method: Name of clustering method to use.
            n_clusters: Number of clusters to test.
            metric: Boolean valued dictionary of metrics to compute.. Defaults to { "ari": True, "nmi": True, "ami": True, "homogeneity": True, "completeness": True, "vmeasure": True, }.
            save_here: File path to save plots. Defaults to "clustering_scores.svg".
        """

        score_list = []
        for k in n_clusters:
            obj, pred_lbl, _ = self.clustering(data, method, k, **kwargs)
            score_list.append(self.compute_metric(true_labels, pred_lbl, metric))

        scores = {key: [d.get(key) for d in score_list] for key in metric.keys()}

        plt.figure(figsize=(8, 8))
        for label, values in scores.items():
            plt.plot(
                n_clusters,
                values,
                "bx-",
                label=label,
                alpha=0.9,
                lw=2,
                dashes=[random.randint(1, 6), random.randint(1, 6)],
            )
        plt.legend()
        plt.xlabel("Number of Clusters")
        plt.ylabel("Clustering Score")
        plt.savefig(save_here)

    def clustering(
        self, data: ArrayLike, method: str, n_clusters: int, **kwargs
    ) -> Tuple:
        """Clustering function.

        Args:
            data: Data to cluster.
            method: Name of clustering method to use.
            n_clusters: Number of clusters k.

        Returns:
            Fit clustering model, predicted cluster labels and cluster centres.
        """
        # runs clustering from sklearn's clustering module
        clustering_obj = CLUSTERING_METHOD_FACTORY[method](
            n_clusters=n_clusters, **kwargs
        )
        clustering_obj.fit(data)
        clustering_labels = clustering_obj.labels_
        if hasattr(clustering_obj, "cluster_centers_"):
            cluster_centres = clustering_obj.cluster_centers_
        else:
            cluster_centres = None

        return clustering_obj, clustering_labels, cluster_centres

    def kmeans_elbow(
        self, data: ArrayLike, n_clusters: Iterable, filepath: str, **kwargs
    ) -> Tuple:
        """Performs k-means clustering and selects best model using elbow method.

        Args:
            data: Data to cluster.
            n_clusters: Number of clusters to test.
            filepath: Path where plots are saved.

        Returns:
            Best clustering model and dictionary of clustering metrics.
        """

        sum_of_squared_distances = []
        kmeans_objs = {}
        cluster_metrics = {}
        for k in n_clusters:
            km = KMeans(
                init="k-means++", n_clusters=k, random_state=42, n_init=10, **kwargs
            )
            km = km.fit(data)
            sum_of_squared_distances.append(km.inertia_)
            kmeans_objs[k] = km

            silhouette_avg = CLUSTERING_METRIC_FACTORY["silhouette_avg"](
                data, km.labels_
            )
            cluster_metrics[k] = silhouette_avg

        kneedle = KneeLocator(
            n_clusters,
            sum_of_squared_distances,
            S=1.0,
            curve="convex",
            direction="decreasing",
        )
        best_k = kneedle.elbow
        best_km = kmeans_objs[best_k]

        plt.figure(figsize=(16, 9))
        plt.plot(n_clusters, sum_of_squared_distances, "bx-")
        plt.axvline(x=best_k, color="red", linestyle="--")
        plt.xlabel("K Clusters", fontsize=40)
        plt.ylabel("Inertia", fontsize=40)
        plt.tick_params("both", labelsize=34)
        # plt.title("Elbow Method for Optimal k")
        plt.tight_layout()
        plt.savefig(filepath, bbox_inches="tight")
        plt.show()

        return best_km, cluster_metrics

    def plot_silhouette_analysis(
        self,
        data: ArrayLike,
        embedding_method: Callable,
        cluster_labels: ArrayLike,
        n_clusters: int,
        filepath: str,
    ) -> float:
        """Computes and plots silhouette score for each cluster.

        Args:
            data: Data to cluster.
            embedding_method: Object to embed data for clustering, must have fit_transform function.
            cluster_labels: Predicted cluster labels.
            n_clusters: Number of clusters k.
            filepath: Path where plots are saved.

        Returns:
            Average silhoutte score.
        """

        k = n_clusters
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 6)

        ax1.set_ylim([0, len(data) + (k + 1) * 10])

        silhouette_avg = CLUSTERING_METRIC_FACTORY["silhouette_avg"](
            data, cluster_labels
        )
        print(
            "For n_clusters =",
            k,
            "The average silhouette_score is :",
            silhouette_avg,
        )

        # Compute the silhouette scores for each sample
        sample_silhouette_values = CLUSTERING_METRIC_FACTORY["silhouette_sample"](
            data, cluster_labels
        )

        y_lower = 10

        colors = [
            "#fc7d0b",
            "#1f77b4",
            "#2d9f3c",
            "#9555c2",
            "#8c564b",
            "#ff7f0e",
            "#9467bd",
            "#d62728",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

        for i in range(k):

            color = colors[i % len(colors)]
            ith_cluster_silhouette_values = sample_silhouette_values[
                cluster_labels == i
            ]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=1.0,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_xlabel("Silhouette coefficient values")
        ax1.set_xlim(-0.15, 0.4)
        ax1.set_ylabel("Cluster label")

        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])

        data_embedding = embedding_method.fit_transform(data)

        for i in range(k):
            cluster_data = data_embedding[cluster_labels == i]
            ax2.scatter(
                cluster_data[:, 0],
                cluster_data[:, 1],
                marker=".",
                s=150,
                lw=0,
                alpha=0.9,
                c=[colors[i % len(colors)]],
                edgecolor="k",
                label=f"Cluster {i}",
            )

        ax2.set_xlabel(f"{embedding_method.__class__.__name__} 1")
        ax2.set_ylabel(f"{embedding_method.__class__.__name__} 2")

        plt.tight_layout()
        plt.savefig(filepath, bbox_inches="tight")

        plt.show()
        return silhouette_avg

    def plot_clustering(
        self,
        cluster_labels_pred: ArrayLike,
        data_embedding: ArrayLike,
        dataset: str,
        title: str,
        color_palette: cm = None,
        save_here: str = "clustering.svg",
    ) -> None:
        """Plots embedded data colour coded by cluster membership.

        Args:
            cluster_labels_pred: Predicted cluster labels.
            data_embedding: 2-D embedding of data for plotting.
            dataset: Name of dataset being plotted.
            title: Title for the plot.
            color_palette: Colour palette to use. Defaults to None.
            save_here: Path to save the plot.
        """
        # to get cluster centers (first, pass as argument) for methods like gmm do the following:
        # for label in np.unique(labels_pred):
        #    cluster_centers.append(X.loc[labels_pred == label].mean(axis=0))
        # for kmeans do km.cluster_centres_

        plt.figure(figsize=(5, 5), constrained_layout=True)

        n_clusters = len(set(cluster_labels_pred))
        # keeping cluster centre to maybe plot it, but also ok to remove it
        if color_palette is None:
            cmap = cm.get_cmap("nipy_spectral")
            color_palette = cmap(cluster_labels_pred.astype(float) / n_clusters)
        for k, col in zip(range(n_clusters), color_palette):
            my_members = cluster_labels_pred == k
            # cluster_center = cluster_centers[k]
            plt.plot(
                data_embedding[my_members, 0],
                data_embedding[my_members, 1],
                "w",
                markerfacecolor=col,
                marker=".",
                markersize=10,
                label=k,
            )
        plt.title(title + f"Clustering of {dataset} Dataset")
        # plt.xticks(())
        # plt.yticks(())
        # plt.subplots_adjust(hspace=0.35, bottom=0.02)
        plt.legend()
        plt.savefig(save_here)
        plt.show()

    def gaussian_mixture_model(
        self, data: ArrayLike, n_components: Iterable, cv_type: str
    ) -> Tuple:
        """Fits a Gaussian Mixture Model (GMM) on the data.

        Args:
            data: Input data.
            n_components: Number of components to use in the GMM.
            cv_type: Covariance type. One of ['tied', 'full'].

        Returns:
            Best fit GMM, list of Aikiko and Bayes Information Criterion scores.
        """
        # gmm.fit(X) takes ages so use X_PCA or similar
        # cv_types = ['tied', 'full']

        lowest_bic = np.infty
        lowest_aic = np.infty
        bic = []
        aic = []

        for n in n_components:
            # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(
                n_components=n, covariance_type=cv_type, random_state=42
            )

            gmm.fit(data)

            # bic.append(gmm.bic(X))
            bic_ = gmm.bic(data)
            aic_ = gmm.aic(data)

            if bic_ < lowest_bic:
                lowest_bic = bic_
                best_gmm = gmm

            if aic_ < lowest_aic:
                lowest_aic = aic_

            bic.append(bic_)
            aic.append(aic_)

        return best_gmm, aic, bic

    def plot_gmm_score(
        self,
        num_components: ArrayLike,
        scores: ArrayLike,
        title: str = "BIC score per model",
        save_here: str = "gmm_bic.svg",
    ) -> None:
        """Plots AIC/BIC scores from fitting a GMM model against number of components tested.

        Args:
            num_components: Number of components of different models.
            scores: Scores associated with each component.
            title: Title of plot. Defaults to "BIC score per model".
            save_here: Path where plots are saved. Defaults to "gmm_bic.svg".
        """
        plt.figure(figsize=(8, 8))
        plt.bar(num_components, scores, width=0.2)
        plt.xticks(num_components)
        plt.ylim([scores.min() * 1.01 - 0.01 * scores.max(), scores.max()])
        plt.title(title)

        xpos = (
            np.mod(scores.argmin(), len(num_components))
            + 0.65
            + 0.2 * np.floor(scores.argmin() / len(num_components))
        )
        plt.text(xpos, scores.min() * 0.97 + 0.03 * scores.max(), "*", fontsize=14)
        plt.xlabel("Number of components")
        plt.savefig(save_here)
