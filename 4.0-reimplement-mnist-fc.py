import torch
import torch.nn as nn
import math
import torchvision
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter

from data import load_mnist
from net import SubspaceLinear

from timeit import default_timer as timer
import matplotlib.pyplot as plt
import pandas as pd

class SubspaceConstrainedMNIST(nn.Module):
    def __init__(self, intrinsic_dim: int, device="cpu"):
        """
        Paper uses 784-200-200-10
        ref: https://arxiv.org/pdf/1804.08838.pdf

        Ref in github:
        https://github.com/uber-research/intrinsic-dimension/blob/9754ebe1954e82973c7afe280d2c59850f281dca/intrinsic_dim/model_builders.py#L81
        """
        super().__init__()
        self.theta_prime = Parameter(torch.empty((intrinsic_dim, 1)))
        self.theta_prime.data.fill_(0)

        self.hidden1 = SubspaceLinear(
            in_features=784,
            out_features=200,
            theta_prime=self.theta_prime,
            device=device,
        )
        self.hidden2 = SubspaceLinear(
            in_features=200,
            out_features=10,
            theta_prime=self.theta_prime,
            device=device,
        )

    def forward(self, x):
        x = self.hidden1(x)
        x = F.relu(x)
        x = self.hidden2(x)
        x = F.relu(x)
        x = F.log_softmax(x, dim=-1)  # (batch_size, dims)
        return x


train_loader, test_loader = load_mnist()


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


# loss_histories = {}
# acc_histories = {}
# test_losses = {}
# corrects = {}

dims = [
    10, 50, 100, 200, 300, 400, 500, 
    600, 700, 750, 800, 1000, 1500, 2000, 
    3000, 4000, 5000, 10000
]

for count, d in enumerate(dims):
    start_ts = timer()
    print(f"Intrinsic dimension: {d}")

    # loss_histories_per_dim = {}
    # acc_histories_per_dim = {}
    # test_losses_per_dim = {}
    corrects_per_dim = {}

    # 20 repetitions
    for i in range(20):
        # Use cpu since at this size, GPU with overhead is slower than using CPU directly
        ssnet = SubspaceConstrainedMNIST(intrinsic_dim=d, device="cpu")
        loss_history, acc_history = train(ssnet, 20, train_loader, device="cpu") # 20 epochs, should be enough?
        test_loss, correct = eval(ssnet, test_loader, device="cpu")

        # Store everything
        # loss_histories_per_dim[i] = loss_history
        # acc_histories_per_dim[i] = acc_history
        # test_losses_per_dim[i] = test_loss
        corrects_per_dim[i] = correct / 10000 * 100

    # loss_histories[d] = loss_histories_per_dim
    # acc_histories[d] = acc_histories_per_dim
    # test_losses[d] = test_losses_per_dim
    # corrects[d] = corrects_per_dim

    dim_scores = pd.Series(corrects_per_dim)
    dim_scores.to_csv(f"output/corrects-{d:05}.csv")
    end_ts = timer()
    print(f"Time taken: {end_ts - start_ts:.2f}s. Another {len(dims) - 1 - count} to go!")
