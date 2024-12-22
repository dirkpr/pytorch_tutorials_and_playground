#!/bin/env python3
"""Follow building neural-network model tutorial from https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html."""

import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass of the neural network."""
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def print_model_architecture(model: nn.Module) -> None:
    """Print the architecture and parameters of the model."""
    print(f"Model structure: {model}\n\n")

    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")


def get_device() -> torch.device:
    """Get the device to use for the model."""
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using {device} device")

    return device


def main() -> None:
    """Train a neural network model."""
    # Set seed to ensure reproducibility
    torch.manual_seed(0)

    device = get_device()

    # Create an instance of NeuralNetwork, and move it to the device
    model = NeuralNetwork().to(device)

    print_model_architecture(model)

    # Print the model architecture
    print(model)

    X = torch.rand(1, 28, 28, device=device)
    logits = model(X)
    pred_probab = nn.Softmax(dim=1)(logits)
    y_pred = pred_probab.argmax(1)
    print(f"Predicted class: {y_pred}")

    input_image = torch.rand(3, 28, 28)
    print(input_image.size())


if __name__ == "__main__":
    main()
