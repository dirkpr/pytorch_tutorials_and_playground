#!/bin/env python3
"""Example file to implement model regression. In parts inspired by this video: https://www.youtube.com/watch?v=YAJ5XBwlN4o."""

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

OPTIM = "custom_sgd"
LEARNING_RATE = 1e-2
EPOCHS = int(1e4)

true_function = lambda x: np.sin(x)  # noqa: E731
rng = np.random.default_rng()


def generate_data(n_samples: int = 1000, noise: Callable | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Generate sample data for regression.

    Args:
    ----
    n_samples (int): The number of samples to generate. Default is 1000.
    noise (Callable): A function that adds noise to the target values. Defaults to None.

    Returns:
    -------
    tuple: A tuple containing:
        - X (ndarray): A numpy array of shape (n_samples, 1) containing the input features.
        - y (ndarray): A numpy array of shape (n_samples, 1) containing the target values.

    """
    X = np.linspace(-2 * np.pi, 2 * np.pi, n_samples).reshape(-1, 1)  # noqa: N806
    y = true_function(X)

    if noise is not None:
        y += noise(X)

    return X, y


def create_model() -> nn.Module:
    """Create a neural network model using PyTorch's nn.Sequential.

    Returns
    -------
        nn.Module: The constructed neural network model.

    """
    return nn.Sequential(
        nn.Linear(1, 50),
        nn.ReLU(),
        nn.Linear(50, 50),
        nn.ReLU(),
        nn.Linear(50, 50),
        nn.ReLU(),
        nn.Linear(50, 1),
    )


class CustomSGD:
    """Custom implementation of the stochastic gradient descent optimizer.

    This class implements a simple version of the stochastic gradient descent
    optimizer. It takes a list of parameters and a learning rate as input and
    updates the parameters based on the gradients of the loss function.

    Args:
        params (list): A list of parameters to be optimized.
        lr (float): The learning rate for the optimizer.

    """

    def __init__(self, params, lr=0.01) -> None:
        """Initialize the regression model with given parameters and learning rate.

        Args:
        ----
            params (iterable): An iterable containing the parameters of the model.
            lr (float, optional): The learning rate for the model. Defaults to 0.01.

        """
        self.param_groups = list(params)
        self.lr = lr

    def step(self) -> None:
        """Update the parameters based on the gradients of the loss function."""
        for param in self.param_groups:
            param.data -= self.lr * param.grad

    def zero_grad(self) -> None:
        """Reset the gradients of the parameters to zero."""
        for param in self.param_groups:
            if param.grad is not None:
                param.grad.zero_()


def main() -> None:
    """Perform regression using PyTorch.

    This function generates data, creates a model, defines a loss function,
    and selects an optimizer based on the global variable OPTIM. It then
    trains the model for a specified number of epochs, printing the loss
    every 100 epochs. Finally, it plots the original data and the model's
    predictions.

    Raises:
        NotImplementedError: If the optimizer specified by OPTIM is "custom_sgd".

    Note:
        The function assumes the existence of the following global variables:
        - OPTIM: A string specifying the optimizer to use ("Adam", "SGD", or "custom_sgd").
        - LEARNING_RATE: A float specifying the learning rate for the optimizer.

    """
    X, y = generate_data(  # noqa: N806
        n_samples=1000,
        noise=lambda x: rng.normal(0, 0.2, x.shape),
    )

    data = torch.utils.data.TensorDataset(torch.Tensor(X), torch.Tensor(y))
    data_loader = DataLoader(data, batch_size=32, shuffle=True)

    model = create_model()
    loss = nn.MSELoss()

    optimizers = {
        "Adam": torch.optim.Adam,
        "SGD": torch.optim.SGD,
        "custom_sgd": CustomSGD,
    }
    optimizer = optimizers[OPTIM](model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        X_, y_ = next(iter(data_loader))  # noqa: N806
        optimizer.zero_grad()
        output = model(X_)
        loss_value = loss(output, y_)
        loss_value.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, Loss: {loss_value.item()}")

    plt.figure(figsize=(12, 6))
    plt.scatter(X, y, s=10, label="Original data")
    plt.plot(
        X,
        true_function(X),
        linestyle="-",
        label="True function",
        color="green",
        linewidth=4,
    )
    plt.plot(
        X,
        model(torch.Tensor(X)).detach().numpy(),
        linestyle="--",
        color="red",
        linewidth=4,
        label="Model prediction",
    )
    plt.legend()
    plt.grid()
    plt.xlabel("X")
    plt.ylabel("y")
    plt.show()


if __name__ == "__main__":
    main()
