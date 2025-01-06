import torch
import torch.nn as nn

from multimodal_survival.utilities.layers import MLPLayer


class MLPEncoder(nn.Module):
    def __init__(self, train_params: dict) -> None:
        """Constructor of decoder class.

        Args:
            train_params: Parameters to pass to encoder.
        """
        super().__init__()

        self.enc_params = train_params.copy()
        self.latent_dim = [train_params["latent_size"]]
        self.input_size = train_params["input_size"]
        self.h_size = train_params["fc_units"]

        self.enc_params.update({
            "fc_units": train_params["fc_units"] + self.latent_dim,
        })

        self.encoder = MLPLayer(self.enc_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes input.

        Args:
            x: Input tensor to encode.

        Returns:
            Encoded input/latent embedding.
        """
        return self.encoder(x)
