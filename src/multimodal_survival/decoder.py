import torch
import torch.nn as nn

from multimodal_survival.utilities.layers import MLPLayer


class MLPDecoder(nn.Module):
    def __init__(self, train_params: dict) -> None:
        """Constructor of decoder class.

        Args:
            train_params: Parameters to pass to decoder.
        """
        super().__init__()

        self.dec_params = train_params.copy()
        self.latent_dim = [train_params["latent_size"]]
        train_params["fc_units"].reverse()

        self.dec_params.update({
            "input_size": train_params["latent_size"],
            "fc_units": train_params["fc_units"] + [train_params["input_size"]],
            "fc_layers": train_params["fc_layers"],
            "fc_activation": train_params["fc_activation"],
        })

        self.decoder = MLPLayer(self.dec_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Decodes input.

        Args:
            x: Input tensor to be decoded.

        Returns:
            reconstructed input.
        """
        return self.decoder(x)
