import torch
import torchvision
from torch.utils.data import DataLoader


def load_mnist(flatten=True):
    if flatten is True:
        dataset_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Lambda(lambda x: torch.flatten(x)),
            ]
        )
    else:
        dataset_transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
            ]
        )

    train = torchvision.datasets.MNIST(
        root="~/.torchdata/",
        download=False,
        # natively stored as PIL images
        transform=dataset_transform,
    )

    test = torchvision.datasets.MNIST(
        root="~/.torchdata/", download=False, train=False, transform=dataset_transform
    )

    train_loader = DataLoader(train, batch_size=100, shuffle=True)
    # If flatten
    # Returns (torch.Size([100, 784]), torch.Size([100]))
    # Else
    # Returns (torch.Size([100, 1, 28, 28]), torch.Size([100]))

    test_loader = DataLoader(test, batch_size=500, shuffle=False)

    return train_loader, test_loader
