import json
from functools import reduce
from itertools import combinations
from pathlib import Path
from typing import Callable, List, Set, Tuple

import mygene
import numpy as np
import pandas as pd
import sklearn
import sklearn.pipeline
from scipy.stats import f_oneway
from sklearn.pipeline import make_pipeline
from sksurv.metrics import concordance_index_censored, integrated_brier_score


def flatten(container):
    """Flattens a nested list or tuple. Borrowed from:
    https://stackoverflow.com/questions/10823877/what-is-the-fastest-way-to-flatten-arbitrarily-nested-lists-in-python

    Args:
        container: Iterable to flatten.

    Yields:
        Flattened container.
    """
    for i in container:
        if isinstance(i, (list, tuple)):
            # replace 2 lines below with yield(j) from flatten(i) in Python 3.3+
            for j in flatten(i):
                yield j
        else:
            yield i


def as_structured_array(
    dataframe: pd.DataFrame, target_name: str = "OS"
) -> np.rec.recarray:
    """Transforms labels from dataframe into structured array.

    Args:
        dataframe: Dataframe containing labels.
        target_name: Name of target to transform. Defaults to "OS".

    Returns:
        Structured array of labels.
    """
    target_columns = [target_name, f"{target_name}.time"]
    structured_array = dataframe[target_columns]
    structured_array.loc[:, target_name] = structured_array[target_name].astype(bool)
    structured_array = structured_array.to_records(index=False)

    return structured_array


def get_missingsurv_id(data: pd.DataFrame, targets: List[str], id_col_name: str) -> Set:
    """Retrieves set of samples that have missing targets.

    Args:
        data: Input data that also include targets.
        targets: List of targets to check missingness.
        id_col_name: Column name to set as index.

    Returns:
        Sample IDs to be dropped from data.
    """
    data_indexed = data.set_index(id_col_name, drop=True)
    drop_patients = set()
    for trgt in targets:
        patients = data_indexed[data_indexed[trgt].isna()].index.values
        if len(patients) >= 1:
            for p in patients:
                drop_patients.add(p)
    return drop_patients


def merge_df(
    dataframes_list: List, id_col_name: str = "patient_id", join: str = "outer"
) -> pd.DataFrame:
    """Merges a given list of dataframes based on join type.

    Args:
        dataframes_list: List of dataframes to merge.
        id_col_name: Column name to use as key for merging. Defaults to "patient_id".
        join: Join type. Defaults to "outer".

    Returns:
        Merged dataframe.
    """
    dataframes_merged = reduce(
        lambda left, right: pd.merge(left, right, on=id_col_name, how=join),
        dataframes_list,
    )

    return dataframes_merged


def cut_tcga_id(df: pd.DataFrame) -> pd.DataFrame:
    """Modifies TCGA patient ID to remove analyte ID (-01A).

    Args:
        df: Input data with patient ID.

    Returns:
        Input data with truncated patient ID.
    """
    if len(df.index[0]) > 12:
        df.index = list(map(lambda x: x.rsplit("-", 1)[0], df.index))

    return df


def evaluate_pipeline(
    pipeline: sklearn.pipeline.Pipeline,
    test_data,
    ytest: np.ndarray,
    ytrain: np.ndarray,
    output_path: Path,
) -> None:
    """Pipeline to evaluate and save test data results.

    Args:
        pipeline: Pipeline that predicts risk scores for given data.
        test_data: Test data to evaluate.
        ytest: Groundtruth labels of test data.
        ytrain: Groundtruth labels of training data.
        output_path: Path where results are saved.
    """
    test_risk_scores = pipeline.predict(test_data)
    eval_preprocessing = make_pipeline(*[step for _, step in pipeline[:-1].steps])
    X_transformed = eval_preprocessing.transform(test_data)
    test_survival_function = pipeline[-1].best_estimator_.predict_survival_function(
        X_transformed
    )
    test_cumulative_hazard = pipeline[
        -1
    ].best_estimator_.predict_cumulative_hazard_function(X_transformed)

    np.save(output_path / "test_survival_function.npy", test_survival_function)
    np.save(output_path / "test_cumulative_hazard.npy", test_cumulative_hazard)
    np.save(output_path / "test_risk_scores.npy", test_risk_scores)

    y_df = pd.DataFrame.from_records(ytest)
    cindex = concordance_index_censored(
        y_df.iloc[:, 0], y_df.iloc[:, 1], test_risk_scores
    )
    # time_col = [col for col in y_df.columns if col.endswith(".time")]
    # train_df = pd.DataFrame.from_records(ytrain)
    # train_time = train_df.filter(regex=".time")
    time_col = y_df.filter(regex=".time")
    min_time, max_time = time_col.min().values, time_col.max().values
    min_time, max_time = 0, 1825
    times = np.arange(min_time, max_time)
    preds = np.asarray([[fn(t) for t in times] for fn in test_survival_function])
    try:
        ibs = integrated_brier_score(ytrain, ytest, preds, times)
    except ValueError:
        ibs = "To be computed."
    metrics = {"cindex": str(cindex), "ibs": str(ibs)}

    with open(output_path / "metrics.json", "wt") as fp:
        json.dump(metrics, fp, indent=1)


def pairwise_anova(data: pd.DataFrame, clustering_obj: Callable) -> Tuple:
    """Performs pairwise one-way ANOVA test between clusters generated by clustering object of choice.

    Args:
        data: Data that was clustered by clustering_obj.
        clustering_obj: Clustering method of choice that has already been fit on data.

    Returns:
        Dataframes containing the F-score and p-values associated with each feature for every cluster pair.
    """
    cluster_labels = clustering_obj.labels_
    sorted_clusters = sorted(np.unique(cluster_labels))
    num_clusters = len(sorted_clusters)
    anova_f_df = pd.DataFrame(index=range(num_clusters), columns=range(num_clusters))
    anova_p_df = pd.DataFrame(index=range(num_clusters), columns=range(num_clusters))
    for i in sorted(np.unique(cluster_labels)):
        for j in range(i + 1, num_clusters):
            clusteri_samples = data[cluster_labels == i]
            clusterj_samples = data[cluster_labels == j]
            f_ij, p_ij = f_oneway(clusteri_samples, clusterj_samples, axis=0)
            anova_f_df.iloc[i, j] = f_ij
            anova_p_df.iloc[i, j] = p_ij
    return anova_f_df, anova_p_df


def get_topk_features_cluster_pairs(
    anova_p_df: pd.DataFrame,
    df_column_names: List,
    k: int,
    alpha=0.05,
    n_features=161,
) -> pd.DataFrame:
    """Retrieves top k contributing features that discriminate cluster pairs.

    Args:
        anova_p_df: Probability that the null hypothesis is true as generated by ANOVA test.
        df_column_names: Feature names as given in input data.
        k: Number of features to retrieve.
        alpha: Significance factor. Defaults to 0.05.
        n_features: Total number of features. Defaults to 161.

    Returns:
        Top k features ordered by increasing p-value of null hypothesis being true.
    """

    cluster_pairs = list(combinations(np.arange(anova_p_df.shape[0]), r=2))
    correction = len(cluster_pairs) * n_features
    if isinstance(k, int):
        final_topk_df = pd.DataFrame(columns=cluster_pairs)
    else:
        final_topk_df = dict()
    for i in range(anova_p_df.shape[0]):
        for j in range(i + 1, anova_p_df.shape[1]):
            if k == "all":
                all_k_p_df = (
                    pd.DataFrame(anova_p_df.iloc[i, j], index=df_column_names)
                    * correction
                ).sort_values(0)
                top_k_p_df = all_k_p_df[all_k_p_df[0] < alpha]
            else:
                top_k_p_df = (
                    pd.DataFrame(anova_p_df.iloc[i, j], index=df_column_names)
                    * correction
                ).sort_values(0)[:k]
            final_topk_df[(i, j)] = list(top_k_p_df.index)

    return final_topk_df


def mirna_mapper_mirmap(
    mirna_list: List,
    mirna_id_converter: pd.DataFrame,
    mirna_target_mirmap: pd.DataFrame,
) -> List:
    """Maps miRNA probes to gene names, if available, using mirmap and an id convertor tool to get standard accession ids.

    Args:
        mirna_list: List of miRNAs to map.
        mirna_id_converter: Tool to convert mature mirna id to standard accession ids.
        mirna_target_map: mirna to gene target map as provided by mirmap (https://mirmap.ezlab.org).


    Returns:
        Mapped list of genes.
    """

    mirna_ids = mirna_id_converter[
        mirna_id_converter["mature_mirna_from_literature"].isin(mirna_list)
    ]
    targets_mirna = mirna_target_mirmap[
        mirna_target_mirmap["mirna_id"].isin(mirna_ids["miRBase_accession"])
    ]

    mg = mygene.MyGeneInfo()
    transcript_to_gene = mg.querymany(
        set(targets_mirna["transcript_stable_id"]), scopes="ensembl.transcript"
    )

    mirna_genes_mirmap = []
    for sample in transcript_to_gene:
        try:
            mirna_genes_mirmap.append(sample["symbol"])
        except KeyError:
            pass
    return list(set(mirna_genes_mirmap))


def mirna_mapper(mirna_list: List, mirna_target_map: pd.DataFrame) -> List:
    """Maps miRNA probes to gene names, if available.

    Args:
        mirna_list: List of miRNAs to map.
        mirna_target_map: mirna to gene target map as provided by database host.

    Returns:
        Mapped list of genes.
    """
    mirna_list = list(map(lambda x: x.replace("r", "R"), mirna_list))
    filtered_targets = mirna_target_map[
        mirna_target_map["mirna"].str.contains("|".join(mirna_list))
    ]

    mg = mygene.MyGeneInfo()
    mirna_genes_dict = mg.querymany(set(filtered_targets["gene"]), scopes="refseq")

    mirna_genes = []
    for sample in mirna_genes_dict:
        try:
            mirna_genes.append(sample["symbol"])
        except KeyError:
            pass
    return list(set(mirna_genes))


def dna_meth_mapper(probe_list: List, probemap: pd.DataFrame) -> List:
    """Maps DNA Methylation probes to gene names.

    Args:
        probe_list: List of probes to map.
        probemap: Probe map as provided by database host.

    Returns:
        Mapped list of genes.
    """
    probe_to_gene_list = []
    # mg = mygene.MyGeneInfo()
    mapped_genes = probemap.loc[probe_list]["gene"].str.split(",").to_list()
    probe_to_gene_list.append(mapped_genes)

    try:
        probe_to_gene_list.remove(["."])
    except ValueError:
        try:
            probe_to_gene_list.remove(".")
        except ValueError:
            pass

    return list(flatten(probe_to_gene_list))
