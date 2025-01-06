# Script to run survival analysis.

import ast
import inspect
import json
import os
from pathlib import Path

import click
import joblib
import numpy as np
import pandas as pd
from loguru import logger
from multimodal_survival.utilities.registries import (
    CROSS_VALIDATION,
    FEATURE_IMPUTERS,
    FEATURE_SELECTORS,
    FEATURE_TRANSFORMERS,
    SCORER_WRAPPER,
    SURVIVAL_MODELS,
)
from multimodal_survival.utilities.utils import cut_tcga_id, evaluate_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline


@click.command()
@click.option(
    "--data_path",
    required=True,
    type=click.Path(path_type=Path, exists=True, help="Path to data directory."),
)
@click.option(
    "--train_filename", required=True, type=str, help="Filename of training dataset."
)
@click.option(
    "--parameters_file",
    required=True,
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="Path to the parameters json file.",
)
@click.option(
    "--output_path",
    required=True,
    type=click.Path(path_type=Path),
    help="Path where results are saved.",
)
@click.option("--target_name", required=True, help="Name of target under study.")
@click.option(
    "--model_name",
    required=True,
    help="Name of the model to use.",
    type=click.Choice(sorted(SURVIVAL_MODELS.keys())),
)
@click.option(
    "--model_parameters_grid",
    required=True,
    help="Model parameters grid for grid search as JSON-formatted string.",
)
@click.option(
    "--number_of_jobs",
    required=False,
    default=1,
    type=int,
    help="Number of jobs to run for parallel processing.",
)
@click.option(
    "--seed", required=False, default=42, type=int, help="Seed for reproducibility."
)
@click.option(
    "--save_pipeline",
    required=False,
    default=True,
    type=bool,
    help="Whether or not to save the entire data processing pipeline.",
)
@click.option(
    "--save_best",
    required=False,
    default=True,
    type=bool,
    help="Whether or not to save the best performing model.",
)
def main(
    data_path: Path,
    train_filename: str,
    parameters_file: Path,
    output_path: Path,
    target_name: str,
    model_name: str,
    model_parameters_grid: str,
    number_of_jobs: int,
    seed: int,
    save_pipeline: bool,
    save_best: bool,
) -> None:

    with open(parameters_file) as fp:
        parameters = json.load(fp)

    if "train_test_indices" in parameters.keys():
        assert (
            parameters["train_test_indices"] != ""
        ), "Please provide a non-empty filename to retrieve test indices."
        train_test_indices_path = parameters["train_test_indices"]
    else:
        raise KeyError(
            "Please provide the test indices filename with key <train_test_indices>."
        )

    merged_data = pd.read_csv(os.path.join(data_path, train_filename), index_col=0)

    label_filename = data_path / parameters["label_filename"]
    all_labels = pd.read_csv(label_filename, index_col=0)

    all_labels = cut_tcga_id(all_labels)
    merged_data = cut_tcga_id(merged_data)

    with open(data_path / train_test_indices_path, "r") as fp:
        train_test_indices = json.load(fp)

    for split, idxs in train_test_indices.items():
        output_dir = output_path / model_name / parameters["scorer"] / split
        click.echo(f"Running split {split}")
        # check if output_dir exists and contains best_estimator.pkl file
        # if yes, skip
        # if no, proceed with loop
        if (output_dir / "best_estimator.pkl").is_file():
            click.echo(f"Results for split {split} already exists, skipping.")
            continue

        train_patients = idxs["train"]
        test_patients = idxs["test"]

        train_labels = all_labels.loc[train_patients, :]
        test_labels = all_labels.loc[test_patients, :]

        click.echo(f"Length of Data = {len(merged_data)}")
        train_df_merged = merged_data[merged_data.index.isin(train_patients)]
        click.echo(f"Length of Training Data = {len(train_df_merged)}")
        train_labels_idx = np.intersect1d(train_labels.index, train_df_merged.index)
        train_labels = train_labels.loc[train_labels_idx, :]
        train_df_merged = train_df_merged.loc[train_labels_idx, :]
        assert all(train_df_merged.index == train_labels.index)

        test_df_merged = merged_data.loc[test_patients, :]
        test_labels = test_labels.loc[test_df_merged.index, :]
        assert all(test_df_merged.index == test_labels.index)

        # convert labels into structured array
        structured_train_labels = train_labels[[target_name, f"{target_name}.time"]]
        structured_train_labels.loc[:, target_name] = structured_train_labels[
            target_name
        ].astype(bool)
        structured_train_labels = structured_train_labels.to_records(index=False)

        structured_test_labels = test_labels[[target_name, f"{target_name}.time"]]
        structured_test_labels.loc[:, target_name] = structured_test_labels[
            target_name
        ].astype(bool)
        structured_test_labels = structured_test_labels.to_records(index=False)

        cross_val = parameters["cross_val"]
        cv_object = CROSS_VALIDATION[cross_val["name"]](**cross_val["args"])

        imputer_whole = parameters["imputer_whole"]

        imputer_whole_obj = FEATURE_IMPUTERS[imputer_whole["name"]](
            **imputer_whole["args"]
        )
        feature_selector = parameters["feature_selector"]
        feature_selector_obj = FEATURE_SELECTORS[feature_selector["name"]](
            **feature_selector["args"]
        )
        feature_scaler_obj = FEATURE_TRANSFORMERS[parameters["scaler"]]()

        if "random_state" in inspect.signature(SURVIVAL_MODELS[model_name]).parameters:
            survival_estimator = SURVIVAL_MODELS[model_name](random_state=seed)

        else:
            survival_estimator = SURVIVAL_MODELS[model_name]()

        estimator_param_grid = ast.literal_eval(model_parameters_grid)

        scorer = parameters["scorer"]
        if scorer == "ibs":
            scorer_wrapper = SCORER_WRAPPER[scorer]
            # need to pass time for IBS
            times = eval(parameters["scorer_wrapper_args"]["times"])
            survival_estimator = scorer_wrapper(survival_estimator, times)
        elif scorer == "cindex_ipcw":
            scorer_wrapper = SCORER_WRAPPER[scorer]
            wrapper_args = parameters["scorer_wrapper_args"]
            survival_estimator = scorer_wrapper(survival_estimator, **wrapper_args)

        # train different estimators to get estimates with different scoring functions
        # since the estimator needs to be wrapped in that specific scoring function.
        multimodal_pipeline = make_pipeline(
            imputer_whole_obj,
            feature_selector_obj,
            feature_scaler_obj,
            GridSearchCV(
                estimator=survival_estimator,
                param_grid=estimator_param_grid,
                cv=cv_object,
                refit=True,
                n_jobs=number_of_jobs,
                return_train_score=True,
            ),
        )

        logger.info(f"running train and cross validation for target={target_name}...")
        multimodal_pipeline.fit(train_df_merged, structured_train_labels)
        logger.info(f"train and cross validation for target={target_name} completed.")

        output_dir.mkdir(parents=True, exist_ok=True)

        cv_df = pd.DataFrame(multimodal_pipeline[-1].cv_results_)
        cv_df.to_csv(output_dir / "cv_results.csv")

        evaluate_pipeline(
            multimodal_pipeline,
            test_df_merged,
            structured_test_labels,
            structured_train_labels,
            output_dir,
        )

        if save_pipeline:
            joblib.dump(multimodal_pipeline, output_dir / "pipeline.pkl")
        if save_best:
            best_pipeline = make_pipeline(
                *[step for _, step in multimodal_pipeline[:-1].steps]
                + [multimodal_pipeline[-1].best_estimator_]
            )
            joblib.dump(best_pipeline, output_dir / "best_estimator.pkl")

    with open(
        output_path / model_name / parameters["scorer"] / "params.json", "w"
    ) as fp:
        json.dump(parameters, fp)


if __name__ == "__main__":
    main()
