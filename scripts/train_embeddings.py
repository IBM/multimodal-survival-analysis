# Script to train autoencoders to retrieve embeddings for different modalities.
import json
from pathlib import Path
from typing import Tuple

import click
import joblib
import lightning as pl
import matplotlib
import matplotlib_inline.backend_inline
import seaborn as sns
import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from multimodal_survival.autoencoder import AutoEncoder
from multimodal_survival.decoder import MLPDecoder
from multimodal_survival.encoder import MLPEncoder
from multimodal_survival.utilities.dataset import get_datasets
from multimodal_survival.utilities.registries import (
    FEATURE_IMPUTERS,
    FEATURE_TRANSFORMERS,
)
from sklearn.pipeline import make_pipeline
from torch.utils.data import DataLoader

matplotlib_inline.backend_inline.set_matplotlib_formats("svg", "pdf")  # For export
matplotlib.rcParams["lines.linewidth"] = 2.0
sns.reset_orig()
sns.set()


# Setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
click.echo(f"Device: {device}")


def train_autoencoder(
    model: nn.Module,
    modality: str,
    latent_dim: int,
    epochs: int,
    es_patience: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    output_path: Path,
) -> Tuple:
    """Training function for autoencoder.

    Args:
        model: Callable instance of model to train.
        modality: Modality under study.
        latent_dim: Latent size of the embeddings.
        epochs: Number of training epochs.
        es_patience: Early stopping steps.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        test_loader: Testing data loader.
        output_path: Path to save the results.

    Returns:
        Tuple of the fit model, dictionary of validation and test results, and test predictions.
    """
    trainer = pl.Trainer(
        default_root_dir=output_path / f"embeddings_{latent_dim}" / modality,
        accelerator="auto",
        devices=1,
        max_epochs=epochs,
        callbacks=[
            EarlyStopping(
                monitor="val_loss", patience=es_patience, check_on_train_epoch_end=False
            ),
            ModelCheckpoint(
                save_top_k=2,
                monitor="val_loss",
                mode="min",
                dirpath=output_path / f"embeddings_{latent_dim}" / modality,
                filename="ae-{epoch:02d}-{val_loss:.2f}",
                save_on_train_epoch_end=False,
            ),
            LearningRateMonitor(logging_interval="epoch"),
        ],
    )

    trainer.logger._log_graph = (
        True  # If True, we plot the computation graph in tensorboard
    )

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = output_path / f"embeddings_{latent_dim}" / modality / "*.ckpt"
    if pretrained_filename.is_file():
        click.echo("Found pretrained model, loading...")
        model = AutoEncoder.load_from_checkpoint(pretrained_filename)
        # trainer.fit(model, train_loader, val_loader)
    else:
        trainer.fit(model, train_loader, val_loader)
    # Test best model on validation and test set
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    result = {"test": test_result, "val": val_result}

    test_preds = trainer.predict(dataloaders=test_loader, ckpt_path="best")

    return model, result, test_preds


@click.command()
@click.option(
    "--model_name", required=True, help="name of the model to use.", default="AE"
)
@click.option(
    "--modality",
    required=True,
    help="name of the modality being embedded.",
    default="AE",
)
@click.option("--latent_dim", required=True, default=128, type=int)
@click.option(
    "--data_path", required=True, type=click.Path(path_type=Path, exists=True)
)
@click.option(
    "--parameters_file",
    required=True,
    type=click.Path(path_type=Path, exists=True),
    default=None,
)
@click.option("--output_path", required=True, type=click.Path(path_type=Path))
@click.option("--seed", required=False, default=42, type=int)
def main(
    model_name: str,
    modality: str,
    latent_dim: int,
    data_path: Path,
    parameters_file: Path,
    output_path: Path,
    seed: int = 42,
):
    with open(parameters_file) as fp:
        parameters = json.load(fp)

    output_dir = output_path / model_name

    train_filename = parameters["train_filename"]
    test_filename = parameters["test_filename"]
    es_patience = parameters.get("es_patience", 4)
    parameters["latent_size"] = latent_dim

    imputer = parameters["imputer"]
    imputer_obj = FEATURE_IMPUTERS[imputer["name"]](**imputer["args"])

    feature_scaler_obj = FEATURE_TRANSFORMERS[parameters["scaler"]]()

    transformation_pipeline = make_pipeline(
        imputer_obj,
        feature_scaler_obj,
    )

    train_dataset, test_dataset = get_datasets(
        Path(data_path) / train_filename,
        Path(data_path) / test_filename,
        transformation_pipeline,
    )

    train_set, val_set = torch.utils.data.random_split(train_dataset, [0.8, 0.2])

    train_loader = DataLoader(
        train_set,
        batch_size=parameters["batch_size"],
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=4,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=parameters["batch_size"],
        shuffle=False,
        drop_last=False,
        num_workers=4,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=parameters["batch_size"],
        drop_last=False,
        num_workers=4,
        persistent_workers=True,
    )

    example_input_array = next(iter(train_loader))

    model = AutoEncoder(MLPEncoder, MLPDecoder, parameters, example_input_array)
    print("ENCODER", model.encoder)
    print("DECODER", model.decoder)
    _, result, test_preds = train_autoencoder(
        model,
        modality,
        latent_dim,
        parameters["epochs"],
        es_patience,
        train_loader,
        val_loader,
        test_loader,
        output_dir,
    )

    with open(
        output_dir / f"embeddings_{latent_dim}" / modality / "results.json", "w"
    ) as fp:
        json.dump(result, fp)

    with open(
        output_dir / f"embeddings_{latent_dim}" / modality / "params.json", "w"
    ) as fp:
        json.dump(parameters, fp)

    # save latent embeddings
    torch.save(
        test_preds,
        output_dir / f"embeddings_{latent_dim}" / modality / "test_predictions",
    )

    # save pipeline
    joblib.dump(
        train_dataset.preprocessing_pipeline,
        output_dir
        / f"embeddings_{latent_dim}"
        / modality
        / "data_transform_pipeline.pkl",
    )
