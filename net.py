import torch
import torch.nn as nn
import math
from torch.nn.parameter import Parameter
import torch.nn.functional as F

# from torch.nn.modules.conv import _ConvNd
from typing import Union, Optional, Tuple, Union, List
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple
import torch.nn.init as init


class SubspaceLinear(nn.Module):
    def __init__(
        self,
        theta,
        in_features,
        out_features,
        bias: bool = True,  # the rest is by the numbers
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}

        super().__init__()

        # Mirror nn.Linear init
        self.in_features = in_features
        self.out_features = out_features
        self.id = theta.shape[0]  # (intrinsic_dim, 1)
        self.theta = theta

        # Weight has shape (out_features, in_features)
        # Therefore P x theta is:
        # (out_features, in_features, id) X (id, 1)

        # Create and init theta, save theta_zero
        self.weight = torch.empty((out_features, in_features), **factory_kwargs)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.weight_zero = self.weight.detach().clone()

        # Generate projection matrix for weights
        self.proj_mat_weights = torch.empty(
            (out_features, in_features, self.id), **factory_kwargs
        )
        nn.init.kaiming_uniform_(self.proj_mat_weights, a=math.sqrt(5))

        if bias:
            # Create and init bias, save bias zero
            self.bias = torch.empty(out_features, **factory_kwargs)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_zero)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            self.bias_zero = self.bias.detach().clone()

            # Generate projection matrix for bias
            self.proj_mat_bias = torch.empty((out_features, self.id), **factory_kwargs)
            nn.init.kaiming_uniform_(self.proj_mat_bias, a=math.sqrt(5))

        else:
            self.register_parameter("bias", None)

    def forward(self, x):
        # in nn.Linear:
        # return F.linear(x, self.weight, self.bias)
        # torch.mm is for matrices only! torch.matmul is the one that can do broadcasting
        weight = self.weight_zero + torch.squeeze(
            torch.matmul(self.proj_mat_weights, self.theta), dim=-1
        )
        bias = self.bias_zero + torch.squeeze(
            torch.matmul(self.proj_mat_bias, self.theta), dim=-1
        )
        return F.linear(x, weight, bias)

    def extra_repr(self) -> str:
        return "id={}, in_features={}, out_features={}, bias={}".format(
            self.id,
            self.in_features,
            self.out_features,
            self.bias is not None,
        )


class _ConvNd(nn.Module):
    # Literally the same, just that the parameters are not registered by default
    # Can't unregister Pytorch params
    # see https://discuss.pytorch.org/t/how-to-unregister-a-parameter-from-a-module/36424/7

    __constants__ = [
        "stride",
        "padding",
        "dilation",
        "groups",
        "padding_mode",
        "output_padding",
        "in_channels",
        "out_channels",
        "kernel_size",
    ]
    __annotations__ = {"bias": Optional[torch.Tensor]}

    def _conv_forward(
        self, input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        ...

    _in_channels: int
    _reversed_padding_repeated_twice: List[int]
    out_channels: int
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Union[str, Tuple[int, ...]]
    dilation: Tuple[int, ...]
    transposed: bool
    output_padding: Tuple[int, ...]
    groups: int
    padding_mode: str
    weight: torch.Tensor
    bias: Optional[torch.Tensor]

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, ...],
        stride: Tuple[int, ...],
        padding: Tuple[int, ...],
        dilation: Tuple[int, ...],
        transposed: bool,
        output_padding: Tuple[int, ...],
        groups: int,
        bias: bool,
        padding_mode: str,
        device=None,
        dtype=None,
    ) -> None:

        factory_kwargs = {"device": device, "dtype": dtype}
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        valid_padding_strings = {"same", "valid"}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    "Invalid padding string {!r}, should be one of {}".format(
                        padding, valid_padding_strings
                    )
                )
            if padding == "same" and any(s != 1 for s in stride):
                raise ValueError(
                    "padding='same' is not supported for strided convolutions"
                )

        valid_padding_modes = {"zeros", "reflect", "replicate", "circular"}
        if padding_mode not in valid_padding_modes:
            raise ValueError(
                "padding_mode must be one of {}, but got padding_mode='{}'".format(
                    valid_padding_modes, padding_mode
                )
            )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
            if padding == "same":
                for d, k, i in zip(
                    dilation, kernel_size, range(len(kernel_size) - 1, -1, -1)
                ):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                        total_padding - left_pad
                    )
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(
                self.padding, 2
            )

        if transposed:
            self.weight = torch.empty(
                (in_channels, out_channels // groups, *kernel_size), **factory_kwargs
            )
        else:
            self.weight = torch.empty(
                (out_channels, in_channels // groups, *kernel_size), **factory_kwargs
            )
        if bias:
            self.bias = torch.empty(out_channels, **factory_kwargs)
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}"
        )
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += ", padding_mode={padding_mode}"
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(_ConvNd, self).__setstate__(state)
        if not hasattr(self, "padding_mode"):
            self.padding_mode = "zeros"


class SubspaceConv2d(_ConvNd):
    # Modification of Conv2d class
    def __init__(
        self,
        theta,  # (id, 1)
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",  # TODO: refine this type
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            in_channels,
            out_channels,
            kernel_size_,
            stride_,
            padding_,
            dilation_,
            False,
            _pair(0),
            groups,
            bias,
            padding_mode,
            **factory_kwargs
        )

        self.id = theta.shape[0]
        self.theta = theta

        # Save weight_zero
        # Size is (out_channels, in_channels // groups, *kernel_size)
        # Usually for Conv2D it is (out_channels, in_channels, kernel_x, kernel_y)
        self.weight_zero = self.weight.detach().clone()

        # Generate projection matrix for weights
        self.proj_mat_weights = torch.empty(
            (out_channels, in_channels, *kernel_size, self.id), **factory_kwargs
        )
        nn.init.kaiming_uniform_(self.proj_mat_weights, a=math.sqrt(5))

        if bias:
            # Create and init bias, save bias zero
            self.bias = torch.empty(out_channels, **factory_kwargs)
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_zero)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
            self.bias_zero = self.bias.detach().clone()

            # Generate projection matrix for bias
            self.proj_mat_bias = torch.empty((out_channels, self.id), **factory_kwargs)
            nn.init.kaiming_uniform_(self.proj_mat_bias, a=math.sqrt(5))

        else:
            self.register_parameter("bias", None)

    def _conv_forward_modded(self, input: torch.Tensor):
        # Reproject weight and bias
        weight = self.weight_zero + torch.squeeze(
            torch.matmul(self.proj_mat_weights, self.theta), dim=-1
        )

        if self.bias is not None:
            bias = self.bias_zero + torch.squeeze(
                torch.matmul(self.proj_mat_bias, self.theta), dim=-1
            )
        else:
            bias = None

        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                weight,
                bias,
                self.stride,
                _pair(0),
                self.dilation,
                self.groups,
            )
        return F.conv2d(
            input, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self._conv_forward_modded(input)
