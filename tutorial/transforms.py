#!/bin/env python3
"""Follow transforms tutorial from https://pytorch.org/tutorials/beginner/basics/transforms_tutorial.html."""

import torch
from datasets_and_dataloaders import visualize_dataset
from torchvision import datasets
from torchvision.transforms import Lambda, ToTensor


def define_hot_hot_encoder(size: int = 10) -> Lambda:
    """Define a one-hot encoder transformation.

    Args:
        size (int): The size of the one-hot encoded vector.

    Returns:
        Lambda: A Lambda transform that converts a target to a one-hot encoded tensor.

    Examples:
        >>> define_hot_hot_encoder()(3)
        tensor([0., 0., 0., 1., 0., 0., 0., 0., 0., 0.])

    """
    return Lambda(
        lambda y: torch.zeros(
            size,
            dtype=torch.float,
        ).scatter_(0, torch.tensor(y), value=1),
    )


def load_dataset(
    transform: Lambda = None,
    target_transform: Lambda = None,
) -> datasets.FashionMNIST:
    """Load the FashionMNIST dataset with optional target transformation.

    Args:
        transform (Lambda, optional): A function/transform that takes in an PIL image and returns a transformed version.
        target_transform (Lambda, optional): A function/transform that takes in the target and transforms it.

    Returns:
        datasets.FashionMNIST: The FashionMNIST dataset.

    """
    return datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=transform,
        target_transform=target_transform,
    )


def load_and_compare_target_transforms() -> None:
    """Load the dataset and display the one-hot-encoded and raw labels."""
    data = {
        "one_hot_target": load_dataset(
            transform=ToTensor(),
            target_transform=define_hot_hot_encoder(size=10),
        ),
        "raw_target": load_dataset(
            transform=ToTensor(),
            target_transform=None,
        ),
    }

    idx = 0
    print("One-hot-encoded Label:", data["one_hot_target"][idx][1])
    print("Raw Label:", data["raw_target"][idx][1])


def load_and_compare_transforms() -> None:
    """Load the dataset and compare the transformed and raw data."""
    data = {
        "tensor": load_dataset(
            transform=ToTensor(),
        ),
        "raw": load_dataset(
            transform=None,
        ),
    }

    visualize = False
    if visualize:
        visualize_dataset(data["tensor"])
        visualize_dataset(data["raw"])

    idx = 0
    print("tensor:", data["tensor"][idx][0])
    print("raw:", data["raw"][idx][0])


def main() -> None:
    load_and_compare_transforms()
    load_and_compare_target_transforms()


if __name__ == "__main__":
    main()
