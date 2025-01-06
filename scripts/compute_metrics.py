import json
import os
from ast import literal_eval
from pathlib import Path

import click
import numpy as np
import pandas as pd


def get_metrics_df(
    main_dir: Path, modality: str, target: str, survival_model: str
) -> pd.DataFrame:
    """Compiles survival scores from all splits into one dataframe.

    Args:
        main_dir: Directory containing all the results.
        modality: Modality for which to compile results.
        target: Target in consideration. Eg, DSS or OS.
        survival_model: Name of the survival model.

    Returns:
        Dataframe containing survival scores indexed by split.
    """

    path = main_dir / modality / target / "repeatedkfold" / survival_model / "cindex"
    metrics_all_splits = pd.DataFrame(columns=["cindex"], index=range(30))
    for i in range(30):
        file = path / str(i) / "metrics.json"
        if not os.path.exists(file):
            continue
        with open(file, "r") as f:
            metrics = json.load(f)
        metrics_all_splits.loc[i, "cindex"] = literal_eval(metrics["cindex"])[0]

    metrics_all_splits.to_csv(path / f"metrics_{target}_{survival_model}.csv")
    return metrics_all_splits


@click.command()
@click.option("--main_dir", required=True, type=click.Path(path_type=Path, exists=True))
@click.option(
    "--target", required=True, help="name of the survival target.", default="OS"
)
@click.option(
    "--survival_model",
    required=True,
    help="name of the survival model.",
    default="coxph",
)
@click.option("--ignore_folders", default={"old", "zero_imputation", "summary_results"})
def main(
    main_dir: Path,
    target: str,
    survival_model: str,
    ignore_folders={"old", "zero_imputation", "summary_results"},
):
    modalities = [
        folder for folder in os.listdir(main_dir) if folder not in ignore_folders
    ]
    mean_metrics_df = pd.DataFrame(index=modalities, columns=["mean_cindex"])
    for modality in modalities:
        metrics_all = get_metrics_df(main_dir, modality, target, survival_model)
        mean_metric = np.mean(metrics_all["cindex"])
        mean_metrics_df.loc[modality, "mean_cindex"] = mean_metric
    mean_metrics_df.to_csv(main_dir / f"mean_metrics_{target}_{survival_model}.csv")


if __name__ == "__main__":
    main()
