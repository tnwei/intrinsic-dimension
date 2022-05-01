"""
Intrinsic dims of MNIST - Fully-connected and CNN
"""

from timeit import default_timer as timer

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from tqdm import trange

from data import load_mnist
from net import SubspaceConv2d, SubspaceLinear

## Data

im_train_loader, im_test_loader = load_mnist(flatten=False)
flat_train_loader, flat_test_loader = load_mnist(flatten=True)


## Util functions


def train(net, num_epochs, train_loader, device="cuda"):
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    net.train()
    loss_history = []
    acc_history = []

    # Single progress bar over all epochs
    pbar = trange(
        len(train_loader) * num_epochs,
        bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}",
        ascii=True,
    )

    for _ in range(num_epochs):
        for batch_id, (features, target) in enumerate(train_loader):
            # forward pass, calculate loss and backprop!
            opt.zero_grad()
            preds = net(features.to(device))
            loss = F.nll_loss(preds, target.to(device))
            loss.backward()
            loss_history.append(loss.item())
            opt.step()

            pbar.update()

    # Verified don't need to return the net
    return loss_history, acc_history


def eval(net, test_loader, device="cuda"):
    net.eval()
    test_loss = 0
    correct = 0

    for features, target in test_loader:
        output = net(features.to(device))
        test_loss += F.nll_loss(output, target.to(device)).item()
        pred = torch.argmax(output, dim=-1)  # get the index of the max log-probability
        correct += pred.eq(target.to(device)).cpu().sum()

    test_loss = test_loss
    test_loss /= len(test_loader)  # loss function already averages over batch size
    accuracy = 100.0 * correct / len(test_loader.dataset)
    acc_history.append(accuracy)
    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), accuracy
        )
    )

    return test_loss, correct.item()


class SubspaceConstrainedMNIST(nn.Module):
    def __init__(self, intrinsic_dim: int, device="cpu"):
        """
        Paper uses 784-200-200-10
        ref: https://arxiv.org/pdf/1804.08838.pdf

        Ref in github:
        https://github.com/uber-research/intrinsic-dimension/blob/9754ebe1954e82973c7afe280d2c59850f281dca/intrinsic_dim/model_builders.py#L81
        """
        super().__init__()
        self.theta = Parameter(torch.empty((intrinsic_dim, 1), device=device))
        self.theta.data.fill_(0)

        self.hidden1 = SubspaceLinear(
            theta=self.theta,
            in_features=784,
            out_features=200,
            device=device,
        )
        self.hidden2 = SubspaceLinear(
            theta=self.theta,
            in_features=200,
            out_features=10,
            device=device,
        )

    def forward(self, x):
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = F.log_softmax(x, dim=-1)  # (batch_size, dims)
        return x


class SubspaceConstrainedLeNet(nn.Module):
    def __init__(self, intrinsic_dim: int, device="cpu"):
        """
        Subspace constrained version of PyImageSearch's LeNet implementation
        """
        super().__init__()
        self.theta = Parameter(torch.empty((intrinsic_dim, 1), device=device))
        self.theta.data.fill_(0)

        self.conv1 = SubspaceConv2d(
            self.theta,
            in_channels=1,
            out_channels=20,
            kernel_size=(5, 5),
            stride=1,
            device=device,
        )
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = SubspaceConv2d(
            self.theta,
            in_channels=20,
            out_channels=50,
            kernel_size=(5, 5),
            stride=1,
            device=device,
        )
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.flatten1 = nn.Flatten()

        self.fc1 = SubspaceLinear(
            self.theta, in_features=800, out_features=500, device=device
        )
        self.relu3 = nn.ReLU()

        self.fc2 = SubspaceLinear(
            self.theta, in_features=500, out_features=10, device=device
        )
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.flatten1(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.logsoftmax(x)

        return x


## Config


dims = [10, 30, 50, 100, 300, 500, 1000, 3000, 5000]

num_reps = 10
device = "cuda" if torch.cuda.is_available() else "cpu"
training_epochs = 20


## Intrinsic dim for fully-connected network on MNIST

fc_corrects = {}

for count, d in enumerate(dims):
    start_ts = timer()
    print(f"Training {num_reps} repetitions for intrinsic dimension: {d}")

    corrects_per_dim = {}

    for i in range(num_reps):
        ssnet = SubspaceConstrainedMNIST(intrinsic_dim=d, device=device)

        loss_history, acc_history = train(ssnet, 20, flat_train_loader, device=device)
        test_loss, correct = eval(ssnet, flat_test_loader, device=device)

        corrects_per_dim[i] = correct / 10000 * 100

    fc_corrects[d] = corrects_per_dim

    end_ts = timer()
    print(
        f"Time taken for dim {d}: {end_ts - start_ts:.2f}s. Remaining dims: {dims[count+1:]}"
    )

# Save results
df = pd.DataFrame(fc_corrects)
df.to_csv("fc-mnist-accuracy.csv")

tidydf = df.melt()
tidydf = tidydf.rename(columns={"variable": "num_id", "value": "acc"})

# Generate and save plot
fig = plt.figure(figsize=(14, 6))
sns.boxplot(data=tidydf, x="num_id", y="acc")
plt.xlabel("Intrinsic dimension")
plt.ylabel("Accuracy")
plt.title(
    "Accuracy of fully-connected network on MNIST, by constrained intrinsic dimension"
)
plt.savefig("fc-results.PNG", bbox_inches="tight")
plt.close(fig)

## Intrinsic dim for convolutional network on MNIST

conv_corrects = {}

for count, d in enumerate(dims):
    start_ts = timer()
    print(f"Training {num_reps} repetitions for intrinsic dimension: {d}")

    corrects_per_dim = {}

    for i in range(num_reps):
        ssnet = SubspaceConstrainedLeNet(intrinsic_dim=d, device=device)

        loss_history, acc_history = train(ssnet, 20, im_train_loader, device=device)
        test_loss, correct = eval(ssnet, im_test_loader, device=device)

        corrects_per_dim[i] = correct / 10000 * 100

    conv_corrects[d] = corrects_per_dim

    end_ts = timer()
    print(
        f"Time taken for dim {d}: {end_ts - start_ts:.2f}s. Remaining dims: {dims[count+1:]}!"
    )

# Save results
df = pd.DataFrame(conv_corrects)
df.to_csv("conv-mnist-accuracy.csv")

tidydf = df.melt()
tidydf = tidydf.rename(columns={"variable": "num_id", "value": "acc"})

# Generate and save plot
fig = plt.figure(figsize=(14, 6))
sns.boxplot(data=tidydf, x="num_id", y="acc")
plt.xlabel("Intrinsic dimension")
plt.ylabel("Accuracy")
plt.title("Accuracy of conv network on MNIST, by constrained intrinsic dimension")
plt.savefig("conv-results.PNG", bbox_inches="tight")
plt.close(fig)
