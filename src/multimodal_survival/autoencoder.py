from typing import Callable, List, Tuple

import lightning as pl
import torch
import torch.nn as nn
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torchmetrics import MeanSquaredError

from multimodal_survival.utilities.registries import OPTIMIZER_SCHEDULERS, OPTIMIZERS


class AutoEncoder(pl.LightningModule):
    def __init__(
        self, encoder: Callable, decoder: Callable, train_params: dict
    ) -> None:
        """Constructor.

        Args:
            encoder: Encoder object.
            decoder: Decoder object.
            train_params: Parameters to train the autoencoder.
        """
        super().__init__()
        self.save_hyperparameters()

        self.train_params = train_params
        self.lr = train_params["learning_rate"]
        self.encoder = encoder(train_params)
        self.decoder = decoder(train_params)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor]:
        """Function that encodes and decodes input data.

        Args:
            x: Input tensor to encode.

        Returns:
            Latent embeddings of tensor and reconstructed input.
        """
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return z, x_recon

    def _get_reconstruction_loss(self, batch: torch.Tensor):
        """Internal function to compute reconstruction loss of the batch.

        Args:
            batch: Batch of input data.

        Returns:
            Mean squared error between reconstructed input and original input.
        """
        if len(batch.shape) > 2:
            x, _ = batch
        else:
            x = batch
        _, x_recon = self.forward(x)
        mse = MeanSquaredError()
        return mse(x_recon, x)

    def configure_optimizers(
        self,
    ) -> List[OptimizerLRScheduler] | Tuple[List[OptimizerLRScheduler]]:
        """Function to configure optimizer for training.

        Returns:
            Optimizer and/or LR scheduler for training.
        """
        optimizer = OPTIMIZERS[self.train_params["optimizer"]](
            self.parameters(), lr=self.lr, **self.train_params["optimizer_args"]
        )

        if self.train_params["lr_scheduler"] != "":
            scheduler = OPTIMIZER_SCHEDULERS[self.train_params["lr_scheduler"]](
                optimizer, **self.train_params["lr_scheduler_args"]
            )
            lr_scheduler = {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
                "name": "LR Monitor",
                "strict": False,
            }
            return [optimizer], [lr_scheduler]

        else:
            return [optimizer]

    def training_step(self, batch, batch_idx):
        """Function performing training step.

        Args:
            batch: Input batch.
            batch_idx: Index of batch.

        Returns:
            Mean squared error between reconstructed and original training input.
        """
        loss = self._get_reconstruction_loss((batch))
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        """Function performing validation step.

        Args:
            batch: Input batch.
            batch_idx: Index of batch.

        Returns:
            Mean squared error between reconstructed and original validation input.
        """
        loss = self._get_reconstruction_loss(batch)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        """Function performing testing step.

        Args:
            batch: Input batch.
            batch_idx: Index of batch.

        Returns:
            Mean squared error between reconstructed and original test input.
        """
        loss = self._get_reconstruction_loss(batch)
        self.log("test_loss", loss)

        return loss
