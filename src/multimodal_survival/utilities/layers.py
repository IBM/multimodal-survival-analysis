import ast

import torch
import torch.nn as nn

from multimodal_survival.utilities.registries import ACTIVATION_FN_FACTORY


class MLPLayer(nn.Module):
    """Generalisable DNN module to allow for flexibility in architecture."""

    def __init__(self, params: dict) -> None:
        """Constructor.

        Args:
            params (dict): DNN parameter dictionary with the following keys:
                input_size (int): Input tensor dimensions.
                fc_layers (int): Number of fully connected layers to add.
                fc_units (List[(int)]): List of hidden units for each layer.
                fc_activation (str): Activation function to apply after each
                    fully connected layer. See utils/hyperparameter.py
                    for options.

        """
        super(MLPLayer, self).__init__()

        self.input_size = params["input_size"]
        self.layers = params["fc_layers"]
        self.hidden_size = params["fc_units"]
        self.activation = params["fc_activation"]
        self.batch_norm = ast.literal_eval(params["batch_norm"])

        modules = []
        hidden_units = [self.input_size] + self.hidden_size
        for layer in range(self.layers):
            modules.append(nn.Linear(hidden_units[layer], hidden_units[layer + 1]))
            if self.activation[layer] != "None":
                modules.append(ACTIVATION_FN_FACTORY[self.activation[layer]])
            if self.batch_norm and layer < (self.layers - 1):  # ?
                modules.append(nn.BatchNorm1d(hidden_units[layer + 1]))
        self.model = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Passes input through a feed forward neural network.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size,*,input_size]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size,*, hidden_sizes[-1]].
        """

        return self.model(x)
