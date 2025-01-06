from functools import reduce
from pathlib import Path
from typing import List, Tuple

import click
import numpy as np
import pandas as pd

from ..utilities.utils import cut_tcga_id, get_missingsurv_id


def process_unimodal(
    mode: str,
    pancan_path: str,
    cols_to_drop: List,
    label_df: pd.DataFrame,
    targets: List,
    save_dir: str,
    label_id_col: str = "_PATIENT",
) -> pd.DataFrame:
    """Pre-processing steps for data of a single modality. Retrieves and processes colorectal cancer data only.

    Args:
        mode: Modality name.
        pancan_path: Path to pancancer data of the chosen modality.
        cols_to_drop: Columns to drop from dataframe.
        label_df: Groundtruth labels associated with the data.
        targets: Targets under study/to keep.
        save_dir: Directory where processed data is saved.
        label_id_col: Name of column containing sample ids. Defaults to "_PATIENT".

    Returns:
        Processed dataframe after removing missing targets such as event status and time.
    """
    pancan_df = pd.read_csv(pancan_path, index_col=0)
    crc_df = pancan_df[
        pancan_df["_primary_disease"].isin(["colon adenocarcinoma", "rectum adenocarcinoma"])
    ]
    crc_df = crc_df.drop(columns=cols_to_drop)
    crc_df.columns = mode + "_" + crc_df.columns
    crc_df.index.name = crc_df.index.name.lower()

    crc_df = cut_tcga_id(crc_df)

    click.echo(f"{mode} before sample drop ", len(crc_df))
    drop_samples = list(get_missingsurv_id(label_df, targets, label_id_col))
    patients_with_labels = np.intersect1d(crc_df.index, label_df[label_id_col])
    patients_with_labels = np.setdiff1d(patients_with_labels, drop_samples)
    crc_df = crc_df.filter(patients_with_labels, axis=0)
    click.echo(f"{mode} after sample drop", len(crc_df))
    crc_df.to_csv(Path(save_dir) / f"crc_{mode}.csv")
    return crc_df


def merge_omics(
    omics_dir: str,
    omics_files: List,
    labels: pd.DataFrame,
    id_col: str = "patient_id",
    join: str = "outer",
) -> Tuple:
    """_summary_

    Args:
        omics_dir: Path to folder containing all omics data.
        omics_files: Filenames of omics data to merge.
        labels: Groundtruth labels associated with omics data.
        id_col: Column name corresponding to sample id. Defaults to "patient_id".
        join: Type of join to perform. Defaults to "outer".

    Returns:
        Merged and filtered omics dataframe and associated labels in corresponding order.
    """
    omics_df = []
    for f in omics_files:
        omics_df.append(pd.read_csv(Path(omics_dir) / f))
    merged_omics = reduce(lambda left, right: pd.merge(left, right, on=id_col, how=join), omics_df)

    if join == "outer":
        filename = "merged_omics.csv"
    elif join == "inner":
        filename = "merged_omics_complete.csv"

    click.echo("merged length before sample drop ", len(merged_omics))
    patients_with_labels = np.intersect1d(merged_omics.index, labels._PATIENT)
    merged_omics_filtered = merged_omics.filter(patients_with_labels, axis=0)
    click.echo("merged length after sample drop ", len(merged_omics_filtered))
    merged_omics_filtered.to_csv(Path(omics_dir) / filename)

    labels_filtered = labels[labels["_PATIENT"].isin(patients_with_labels)]
    labels_filtered = labels_filtered.set_index("_PATIENT", drop=True)
    labels_filtered.index.name = "patient_id"
    labels_filtered = labels_filtered.loc[merged_omics_filtered.index, :]
    assert all(merged_omics_filtered.index == labels_filtered.index)

    labels_filtered.to_csv(Path(omics_dir) / f"{filename}_labels.csv")

    return merged_omics_filtered, labels_filtered


def check_tsv(data_dir: Path, data_key: str) -> None:
    """Checks if a file is of tsv format. Converts to tsv if csv found.

    Args:
        data_dir: Data directory.
        data_key: Key pattern to identify files to check.
    """
    for file in data_dir.glob(data_key):
        if file.suffix == ".tsv":
            click.echo(f"{file.stem} file in tsv format.")
        elif file.suffix == ".csv":
            click.echo("csv found, converting to tsv.")
            df = pd.read_csv(file, index_col=0)
            df = cut_tcga_id(df)
            new_filename = str(file).replace(".csv", ".tsv")
            df.to_csv(new_filename, sep="\t")
        else:
            click.echo(
                "Check file format and ensure it is either csv or tsv; Also check if key pattern is reading correct files."
            )


def check_txt(filename: Path) -> None:
    """Check if a file is of txt format.

    Args:
        filename: Path to file.
    """
    assert filename.suffix == ".txt", f"Provide a text file not {filename.suffix} file"
