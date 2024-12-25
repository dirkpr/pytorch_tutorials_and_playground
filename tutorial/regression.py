#!/bin/env python3
"""Example file to implement model regression. In parts inspired by this video: https://www.youtube.com/watch?v=YAJ5XBwlN4o."""

from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

OPTIM = "custom_sgd"
LEARNING_RATE = 1e-2
EPOCHS = int(1e4)

true_function = lambda x: np.sin(x)  # noqa: E731
rng = np.random.default_rng()


def generate_data(
    n_samples: int = 1000,
    noise: Callable | None = None,
    start: float = -2 * np.pi,
    end: float = 2 * np.pi,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate sample data for regression.

    Args:
    ----
    n_samples (int): The number of samples to generate. Default is 1000.
    noise (Callable): A function that adds noise to the target values. Defaults to None.
    start: Start of data
    end: End of data

    Returns:
    -------
    tuple: A tuple containing:
        - X (ndarray): A numpy array of shape (n_samples, 1) containing the input features.
        - y (ndarray): A numpy array of shape (n_samples, 1) containing the target values.

    """
    X = np.linspace(start, end, n_samples).reshape(-1, 1)  # noqa: N806
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


def train(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.MSELoss,
    optimizer: torch.optim.SGD,
) -> float:
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):  # noqa: N806
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            lossi, current = loss.item(), batch * dataloader.batch_size + len(X)
            print(f"loss: {lossi:>7f}  [{current:>5d}/{size:>5d}]")

    return loss.detach().numpy()


def train_dirk_fit(
    data_loader: DataLoader,
    model: nn.Module,
    loss_fn: nn.MSELoss,
    optimizer: torch.optim.SGD,
) -> float:
    model.train()
    X_, y_ = next(iter(data_loader))  # noqa: N806
    optimizer.zero_grad()
    output = model(X_)
    loss_value = loss_fn(output, y_)
    loss_value.backward()
    optimizer.step()

    return loss_value.item()


def test(
    dataloader: DataLoader,
    model: nn.Module,
    loss_fn: nn.CrossEntropyLoss,
) -> float:
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    # print(f"Avg loss: {test_loss:>8f} \n")

    return test_loss


def plot_it(
    model: nn.Module,
    orignal_data: tuple[np.ndarray, np.ndarray],
    test_data: tuple[np.ndarray, np.ndarray],
    model_line: plt.Line2D = None,
) -> plt.Figure:
    if model_line is None:
        plt.ion()
        plt.figure(figsize=(12, 6))
        plt.scatter(orignal_data[0], orignal_data[1], s=10, label="Original data")
        plt.plot(
            test_data[0],
            test_data[1],
            linestyle="-",
            label="True function",
            color="black",
            linewidth=4,
        )
        (model_line,) = plt.plot(
            test_data[0],
            model(torch.Tensor(test_data[0])).detach().numpy(),
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
        plt.pause(0.1)

    model_line.set_ydata(model(torch.Tensor(test_data[0])).detach().numpy())
    plt.pause(0.1)

    return model_line


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

    data = TensorDataset(Tensor(X), Tensor(y))
    data_loader = DataLoader(data, batch_size=32, shuffle=True)

    x_test, y_test = generate_data(n_samples=100, noise=None)
    data_loader_test = DataLoader(TensorDataset(Tensor(x_test), Tensor(y_test)), batch_size=32, shuffle=True)

    model = create_model()
    loss = nn.MSELoss()

    optimizers = {
        "Adam": torch.optim.Adam,
        "SGD": torch.optim.SGD,
        "custom_sgd": CustomSGD,
    }
    optimizer = optimizers[OPTIM](model.parameters(), lr=LEARNING_RATE)
    test_data = generate_data(1000, noise=None, start=-3 * np.pi, end=3 * np.pi)
    model_line = None

    for epoch in range(EPOCHS):
        # loss_value = train(data_loader, model, loss, optimizer)
        loss_value = train_dirk_fit(data_loader, model, loss, optimizer)
        if epoch % int(EPOCHS / 100) == 0:
            loss_test = test(data_loader_test, model, loss)
            print(f"Epoch: {epoch}, Loss: {loss_value}, Test: {loss_test}")

        if epoch % (int(EPOCHS / 10) + 1) == 0:
            model_line = plot_it(model, (X, y), test_data, model_line=model_line)

    plot_it(model, (X, y), test_data, model_line=model_line)
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
