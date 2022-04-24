import torch
import torch.nn as nn
import math
from torch.nn.parameter import Parameter
import torch.nn.functional as F


class SubspaceLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        theta_prime,
        bias: bool = True,  # the rest is by the numbers
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()

        # Mirror nn.Linear init
        self.in_features = in_features
        self.out_features = out_features
        self.subspace_features = theta_prime.shape[0]  # (intrinsic_dim, 1)
        self.theta_prime = theta_prime

        # Weight has shape (out_features, in_features)
        # Therefore P x theta_prime is:
        # (out_features, in_features, subspace_features) X (subspace_features, 1)

        # Create and init theta, save theta_zero
        self.theta = torch.empty((out_features, in_features), **factory_kwargs)
        nn.init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        self.theta_zero = self.theta.detach().clone()

        # Generate projection matrix for weights
        self.proj_mat_weights = torch.empty(
            (out_features, in_features, self.subspace_features), **factory_kwargs
        )
        nn.init.kaiming_uniform_(self.proj_mat_weights, a=math.sqrt(5))

        if bias:
            # Create and init bias, save bias zero
            self.bias = torch.empty(out_features, **factory_kwargs)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.theta_zero)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            self.bias_zero = self.bias.detach().clone()

            # Generate projection matrix for bias
            self.proj_mat_bias = torch.empty(
                (out_features, self.subspace_features), **factory_kwargs
            )
            nn.init.kaiming_uniform_(self.proj_mat_bias, a=math.sqrt(5))

        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        # in nn.Linear:
        # return F.linear(x, self.weight, self.bias)
        # torch.mm is for matrices only! torch.matmul is the one that can do broadcasting
        theta = self.theta_zero + torch.squeeze(
            torch.matmul(self.proj_mat_weights, self.theta_prime), dim=-1
        )
        bias = self.bias_zero + torch.squeeze(
            torch.matmul(self.proj_mat_bias, self.theta_prime), dim=-1
        )
        return F.linear(x, theta, bias)

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, subspace_features={}, bias={}".format(
            self.in_features,
            self.out_features,
            self.subspace_features,
            self.bias is not None,
        )
