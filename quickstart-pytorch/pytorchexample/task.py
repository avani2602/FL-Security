"""pytorchexample: A Flower / PyTorch app."""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one batch of images through the network and return class scores.

        `x` is a batch of images shaped (batch_size, 3, 32, 32); the output
        is shaped (batch_size, 10), one raw score per CIFAR-10 class.
        """
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


fds = None  # Cache FederatedDataset

pytorch_transforms = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def apply_transforms(batch: dict[str, Any]) -> dict[str, Any]:
    """Apply the standard (non-malicious) image transforms to a batch.

    Converts each image to a tensor and normalizes it. Labels are left
    untouched. Used for every honest client and for all evaluation data.
    """
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    return batch

def apply_transforms_poison(batch: dict[str, Any]) -> dict[str, Any]:
    """Label-flipping attack: Images are transformed normally, but every label is swapped for it's
    'opposite' (e.g. 0<-->9, 1<-->8...).
    A client using this trains on deliberately wrong answers and sends a 'poisoned' update back"""
    batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
    batch["label"] = [9 - int(label) for label in batch["label"]]
    return batch


def load_data(
    partition_id: int, num_partitions: int, batch_size: int, poison: bool = False
) -> tuple[DataLoader, DataLoader]:
    """Load one client's train/test split of CIFAR-10.

    `partition_id` selects which of the `num_partitions` IID shards this
    client trains on. When `poison=True`, the training split (only) is
    wrapped with `apply_transforms_poison` instead of `apply_transforms`,
    implementing the label-flipping attack for this client; the held-out
    test split is always left un-poisoned so evaluation stays honest.
    """
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="uoft-cs/cifar10",
            partitioners={"train": partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    #Malicious clients flip their training labels, test data is not affected
    train_transform = apply_transforms_poison if poison else apply_transforms
    train_split = partition_train_test["train"].with_transform(train_transform)
    test_split = partition_train_test["test"].with_transform(apply_transforms)

    trainloader = DataLoader(train_split, batch_size = batch_size, shuffle=True)
    testloader = DataLoader(test_split, batch_size=batch_size)
    return trainloader, testloader


def load_centralized_dataset() -> DataLoader:
    """Load the full (server-side) CIFAR-10 test set as a single dataloader.

    Used by the server for global evaluation of the aggregated model, as
    opposed to `load_data`, which loads one client's local, partitioned
    slice of the data.
    """
    # Load entire test set
    test_dataset = load_dataset("uoft-cs/cifar10", split="test")
    dataset = test_dataset.with_format("torch").with_transform(apply_transforms)
    return DataLoader(dataset, batch_size=128)


def train(
    net: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
) -> float:
    """Train `net` in place on `trainloader` for `epochs` local epochs.

    Returns the average training loss per batch. If `trainloader` came from
    a malicious client with `poison=True`, its labels are already flipped,
    so this same training loop is what turns clean data into a poisoned
    update — the attack lives in the data, not in this function.
    """
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    avg_trainloss = running_loss / (epochs * len(trainloader))
    return avg_trainloss


def test(
    net: nn.Module, testloader: DataLoader, device: torch.device
) -> tuple[float, float]:
    """Evaluate `net` on `testloader` and return (loss, accuracy)."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy
