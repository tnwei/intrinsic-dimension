import torch
import torch.nn as nn
import math
import torchvision
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.nn.parameter import Parameter

from data import load_mnist
from net import SubspaceLinear, SubspaceConv2d

from timeit import default_timer as timer
import matplotlib.pyplot as plt
import pandas as pd

save_output = True
device = "cpu"
outdir = Path("output/cnn/")
if not outdir.exists():
    outdir.mkdir(parents=True)


class SubspaceConstrainedLeNet(nn.Module):
    def __init__(self, intrinsic_dim: int):
        """
        Subspace constrained version of PyImageSearch's LeNet implementation
        """
        super().__init__()
        self.theta = Parameter(torch.empty((intrinsic_dim, 1)))
        self.theta.data.fill_(0)

        self.conv1 = SubspaceConv2d(
            self.theta, in_channels=1, out_channels=20, kernel_size=(5, 5), stride=1
        )
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv2 = SubspaceConv2d(
            self.theta, in_channels=20, out_channels=50, kernel_size=(5, 5), stride=1
        )
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.flatten1 = nn.Flatten()

        self.fc1 = SubspaceLinear(self.theta, in_features=800, out_features=500)
        self.relu3 = nn.ReLU()

        self.fc2 = SubspaceLinear(self.theta, in_features=500, out_features=10)
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


train_loader, test_loader = load_mnist(flatten=False)


def train(net, num_epochs, train_loader, device="cpu"):
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    net.train()
    loss_history = []
    acc_history = []

    for _ in range(num_epochs):
        for batch_id, (features, target) in enumerate(train_loader):
            # forward pass, calculate loss and backprop!
            opt.zero_grad()
            preds = net(features.to(device))
            loss = F.nll_loss(preds, target.to(device))
            loss.backward()
            loss_history.append(loss.item())
            opt.step()

            if batch_id % 100 == 0:
                pass
                # print(loss.item())

    # Verified don't need to return the net
    return loss_history, acc_history


def eval(net, test_loader, device="cpu"):
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


dims = [
    10,
    50,
    100,
    200,
    300,
    400,
    500,
    600,
    700,
    800,
    1000,
    1500,
    2000,
    3000,
    4000,
    5000,
    10000,
]

for count, d in enumerate(dims):
    start_ts = timer()
    print(f"Intrinsic dimension: {d}")

    corrects_per_dim = {}

    # 20 repetitions
    for i in range(20):
        # Use cpu since at this size, GPU with overhead is slower than using CPU directly
        ssnet = SubspaceConstrainedLeNet(intrinsic_dim=d)
        ssnet = ssnet.to(device)
        loss_history, acc_history = train(
            ssnet, 20, train_loader, device="cpu"
        )  # 20 epochs, should be enough?
        test_loss, correct = eval(ssnet, test_loader, device="cpu")
        corrects_per_dim[i] = correct / 10000 * 100

    if save_output is True:
        dim_scores = pd.Series(corrects_per_dim)
        dim_scores.to_csv(outdir / f"corrects-{d:05}.csv")

    end_ts = timer()
    print(
        f"Time taken: {end_ts - start_ts:.2f}s. Another {len(dims) - 1 - count} to go!"
    )
