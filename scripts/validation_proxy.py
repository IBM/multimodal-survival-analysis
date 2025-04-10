import json
from pathlib import Path

import click
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn import impute
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier

from multimodal_survival.utilities.plotting import (
    perform_survival_analysis,
    plot_hist_clinical_cluster,
)


@click.command()
@click.option(
    "--data_root_dir",
    help="Root directory of data.",
    type=click.Path(path_type=Path, exists=True),
)
@click.option(
    "--ref_dataset",
    help="Name of reference dataset.",
    type=str,
)
@click.option(
    "--proxy_dataset",
    help="Name of proxy dataset.",
    type=str,
)
@click.option(
    "--proxy_clinical_path",
    help="Path to proxy's clinical data.",
    type=click.Path(path_type=Path, exists=True),
)
@click.option(
    "--clustering_result_dict_path",
    help="Path to clustering results stored as a dictionary.",
    type=click.Path(path_type=Path, exists=True),
)
@click.option(
    "--save_dir",
    help="Path to save results.",
    type=click.Path(path_type=Path, exists=True),
)
@click.option("--n_neighbours", default=5, help="Neighbours to consider for KNN.")
def main(
    data_root_dir,
    ref_dataset,
    proxy_dataset,
    proxy_clinical_path,
    clustering_result_dict_path,
    save_dir,
    n_neighbours=5,
):

    # load reference data
    ref_df = pd.read_csv(data_root_dir / f"{ref_dataset}.csv", index_col=0)

    # load proxy dataset
    proxy_df = pd.read_csv(data_root_dir / f"{proxy_dataset}.csv", index_col=0)

    # subset only modalities you have in ref
    ref_df_subset = ref_df[proxy_df.columns]
    imputer = SimpleImputer(strategy="median")
    ref_df_subset = imputer.fit_transform(ref_df_subset)
    proxy_df = pd.DataFrame(
        imputer.transform(proxy_df), index=proxy_df.index, columns=proxy_df.columns
    )

    # load proxy clinical
    proxy_clinical = pd.read_csv(proxy_clinical_path, index_col=0)

    # stdz? yes, using saved scaler when creating bag of patches data

    # load clustering_results to get cluster labels
    with open(clustering_result_dict_path, "rb") as f:
        kmeans_result_dict = joblib.load(f)

    cluster_labels = kmeans_result_dict[ref_dataset]["cluster_labels"]

    knn = KNeighborsClassifier(metric="cosine", n_neighbors=n_neighbours)
    knn.fit(ref_df_subset, cluster_labels)

    predicted_proxy_labels = pd.DataFrame(
        knn.predict(proxy_df),
        index=proxy_df.index,
        columns=["cluster_labels"],
    )

    # load survival information for proxy
    proxy_survival = predicted_proxy_labels.join(
        proxy_clinical[["OS_STATUS", "OS_MONTHS"]], how="inner"
    )
    proxy_survival.dropna(inplace=True)
    proxy_survival["OS_STATUS"] = proxy_survival["OS_STATUS"].astype(int)

    # plot KM
    _, log_rank = perform_survival_analysis(
        proxy_survival,
        "OS_MONTHS",
        "OS_STATUS",
        "cluster_labels",
        save_dir / f"km_{proxy_dataset}_censormarks.pdf",
    )
    np.save(save_dir / "log_rank.npy", log_rank)
    # plot clin var
    sns_plot_df = predicted_proxy_labels.join(proxy_clinical)
    plot_hist_clinical_cluster(
        sns_plot_df,
        variable="AGE",
        variable_name="Age",
        save_here=Path(save_dir / "histograms"),
        mean=True,
    )
    clin_vars = [
        "SEX",
        "TUMOR_SITE",
        "STAGE",
        "POLYPS_PRESENT",
        "microsatelite",
        "HISTOLOGY",
    ]
    clin_var_names = [
        "Gender",
        "Site",
        "Stage",
        "Colon Polyps",
        "Microsatelite",
        "Histology",
    ]
    clin_var_dict = dict(zip(clin_vars, clin_var_names))
    chi2_dict = {}
    for var, name in clin_var_dict.items():
        plot_hist_clinical_cluster(
            sns_plot_df,
            variable=var,
            variable_name=name,
            save_here=Path(save_dir / "histograms"),
            mean=False,
        )

        contingency_table = pd.crosstab(
            index=sns_plot_df[var], columns=sns_plot_df["cluster_labels"]
        )
        chi2, p, dof, _ = chi2_contingency(contingency_table)
        chi2_dict[name] = {"chi2": chi2, "p": p, "dof": dof}

    contingency_table = pd.crosstab(
        index=sns_plot_df["AGE"], columns=sns_plot_df["cluster_labels"]
    )
    chi2, p, dof, _ = chi2_contingency(contingency_table)
    chi2_dict["Age"] = {"chi2": chi2, "p": p, "dof": dof}

    with open(
        save_dir / "histograms/chi2_dict.json",
        "w",
    ) as f:
        json.dump(chi2_dict, f)


if __name__ == "__main__":
    main()
