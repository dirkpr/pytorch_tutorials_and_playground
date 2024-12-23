#!/bin/env python3
"""Follows tutorial: https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html."""

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets_and_dataloaders import load_datasets
from neural_network import NeuralNetwork
from torch import nn
from torch.utils.data import DataLoader

LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 5


def train_loop(
    dataloader: DataLoader,
    model: NeuralNetwork,
    loss_fn: nn.CrossEntropyLoss,
    optimizer: torch.optim.SGD,
) -> tuple[float, float]:
    """Train the model for one epoch.

    Args:
        dataloader (DataLoader): DataLoader for the training data.
        model (NeuralNetwork): The neural network model to be trained.
        loss_fn (nn.CrossEntropyLoss): Loss function to be used for training.
        optimizer (torch.optim.SGD): Optimizer to be used for updating model parameters.

    Returns:
        None

    Notes:
        This function sets the model to training mode, iterates over the batches of data,
        computes the predictions and loss, performs backpropagation, and updates the model
        parameters using the optimizer. It also prints the loss and progress every 100 batches.

    """
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    correct = 0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * BATCH_SIZE + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    correct /= size

    return loss.detach().numpy(), correct


def test_loop(
    dataloader: DataLoader,
    model: NeuralNetwork,
    loss_fn: nn.CrossEntropyLoss,
) -> tuple[float, float]:
    """Evaluate the performance of a neural network model on a test dataset.

    Args:
        dataloader (DataLoader): DataLoader object that provides batches of test data.
        model (NeuralNetwork): The neural network model to be evaluated.
        loss_fn (nn.CrossEntropyLoss): Loss function used to compute the loss.

    Returns:
        tuple: A tuple containing:
            - test_loss (float): The average loss over all batches.
            - correct (float): The accuracy of the model on the test dataset.

    """
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    return test_loss, correct


def main() -> None:
    """Run main function."""
    training_data, test_data = load_datasets()

    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)
    model = NeuralNetwork()
    loss_fn = nn.CrossEntropyLoss()

    loss = {"train": [], "test": []}
    accuracy = {"train": [], "test": []}

    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loss, train_accuracy = train_loop(
            dataloader=train_dataloader,
            model=model,
            loss_fn=loss_fn,
            optimizer=torch.optim.SGD(model.parameters(), lr=LEARNING_RATE),
        )
        test_loss, test_accuracy = test_loop(
            dataloader=test_dataloader,
            model=model,
            loss_fn=loss_fn,
        )

        loss["train"].append(train_loss)
        loss["test"].append(test_loss)
        accuracy["train"].append(train_accuracy)
        accuracy["test"].append(test_accuracy)

    loss["train"] = np.stack(loss["train"])

    plt.figure(figsize=(8, 4))
    plt.subplot(2, 1, 1)
    plt.plot(loss["train"], label="Train")
    plt.plot(loss["test"], label="Test")
    plt.grid()
    plt.ylabel("Cross Entropy Loss")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(accuracy["train"], label="Train")
    plt.plot(accuracy["test"], label="Test")
    plt.grid()
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy [%]")
    plt.legend()
    plt.show()

    print("Done!")


if __name__ == "__main__":
    main()
