#!/bin/env python3
"""Follows tutorial: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html."""

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


def load_datasets() -> tuple[datasets.FashionMNIST, datasets.FashionMNIST]:
    """Load the FashionMNIST training and test datasets."""
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    return training_data, test_data


def load_dataloaders(
    training_data: datasets.FashionMNIST,
    test_data: datasets.FashionMNIST,
) -> tuple[DataLoader, DataLoader]:
    """Create dataloaders for the training and test datasets."""
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    return train_dataloader, test_dataloader


def run_dataloader(dataloader: DataLoader) -> None:
    """Display image and label."""
    train_features, train_labels = next(iter(dataloader))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")


def visualize_dataset(dataset: datasets.FashionMNIST) -> None:
    """Visualize a sample of images from the dataset with their labels."""
    labels_map = {
        0: "T-Shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
    }
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")
    plt.show()


def check_dataset_properties(dataset: datasets.FashionMNIST) -> None:
    """Check the properties of the dataset."""
    print("Dataset length:", len(dataset))
    print("Dataset shape:", dataset[0][0].shape)
    print("Dataset labels:", dataset.classes)


def main() -> None:
    """Load datasets, check properties, and run dataloader."""
    training_data, test_data = load_datasets()
    print("Training data:", training_data)
    check_dataset_properties(training_data)
    print("Test data:", test_data)

    visualize = False
    if visualize:
        visualize_dataset(training_data)
    else:
        run_dataloader(load_dataloaders(training_data, test_data)[0])


if __name__ == "__main__":
    main()
