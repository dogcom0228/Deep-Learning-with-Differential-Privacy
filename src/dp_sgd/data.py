from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import DatasetConfig, TrainingConfig


@dataclass(slots=True)
class DataLoaders:
    train: DataLoader
    eval: DataLoader
    train_size: int
    eval_size: int


def _build_mnist_transforms(augment: bool) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    train_steps: list[object] = []
    if augment:
        train_steps.append(transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)))
    train_steps.extend([transforms.ToTensor(), normalize])
    eval_steps = [transforms.ToTensor(), normalize]
    return transforms.Compose(train_steps), transforms.Compose(eval_steps)


def _build_cifar10_transforms(augment: bool) -> tuple[transforms.Compose, transforms.Compose]:
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    train_steps: list[object] = []
    if augment:
        train_steps.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    train_steps.extend([transforms.ToTensor(), normalize])
    eval_steps = [transforms.ToTensor(), normalize]
    return transforms.Compose(train_steps), transforms.Compose(eval_steps)


def build_dataloaders(
    dataset_config: DatasetConfig,
    training_config: TrainingConfig,
    device_type: str,
) -> DataLoaders:
    dataset_name = dataset_config.name.lower()
    if dataset_name == "mnist":
        dataset_class = datasets.MNIST
        train_transform, eval_transform = _build_mnist_transforms(dataset_config.augment)
    elif dataset_name == "cifar10":
        dataset_class = datasets.CIFAR10
        train_transform, eval_transform = _build_cifar10_transforms(dataset_config.augment)
    else:
        raise ValueError(f"Unsupported dataset '{dataset_config.name}'.")

    train_dataset = dataset_class(
        root=dataset_config.root,
        train=True,
        download=dataset_config.download,
        transform=train_transform,
    )
    eval_dataset = dataset_class(
        root=dataset_config.root,
        train=False,
        download=dataset_config.download,
        transform=eval_transform,
    )

    loader_kwargs = {
        "num_workers": dataset_config.num_workers,
        "pin_memory": dataset_config.pin_memory and device_type == "cuda",
        "persistent_workers": dataset_config.persistent_workers and dataset_config.num_workers > 0,
    }

    train_loader = DataLoader(
        train_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=training_config.eval_batch_size,
        shuffle=False,
        **loader_kwargs,
    )

    return DataLoaders(
        train=train_loader,
        eval=eval_loader,
        train_size=len(train_dataset),
        eval_size=len(eval_dataset),
    )
