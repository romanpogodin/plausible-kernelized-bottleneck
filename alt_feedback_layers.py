import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from torch.nn.modules.utils import _reverse_repeat_tuple
from torch.nn import init
import math


class AltBackwardFunction(torch.autograd.Function):
    """
    Implements alternative feedback for a given feedback weight
    by running a one-layer forward pass with the backward weights
    to compute gradients.
    """
    @staticmethod
    def forward(ctx, input, weight_forward, weight_backward, bias=None, forward_function=F.linear, kwargs_dict=dict()):
        ctx.save_for_backward(input, weight_forward, weight_backward, bias)
        ctx.forward_function = forward_function
        ctx.kwargs = kwargs_dict
        return ctx.forward_function(input, weight_forward, bias, **ctx.kwargs)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight_forward, weight_backward, bias = ctx.saved_tensors
        grad_input = grad_weight_forward = grad_weight_backward = grad_bias = None
        grad_forward_function = grad_kwargs_dict = None
        with torch.enable_grad():
            # print(weight_backward.requires_grad)
            weight_backward.requires_grad = True
            input_had_no_grad = not input.requires_grad
            if input_had_no_grad:
                input.requires_grad = True

            alt_output = ctx.forward_function(input, weight_backward, bias, **ctx.kwargs)

            if bias is None:
                grad_input, grad_weight_forward = torch.autograd.grad(
                    alt_output, (input, weight_backward), grad_output)
            else:
                grad_input, grad_weight_forward, grad_bias = torch.autograd.grad(
                    alt_output, (input, weight_backward, bias), grad_output)

            if input_had_no_grad:
                input.requires_grad = False
                grad_input = None

            weight_backward.requires_grad = False

        # grad_weight_backward is None
        # grad_forward_function and grad_kwargs_dict are None, needed for consistency
        return grad_input, grad_weight_forward, grad_weight_backward, grad_bias, grad_forward_function, grad_kwargs_dict


def alt_backward_linear(input, weight_forward, weight_backward, bias=None):
    return AltBackwardFunction.apply(input, weight_forward, weight_backward, bias, torch.nn.functional.linear)


def alt_backward_conv2d(input, weight_forward, weight_backward, bias=None, stride=1, padding=0, dilation=1, groups=1):
    return AltBackwardFunction.apply(input, weight_forward, weight_backward, bias, torch.nn.functional.conv2d,
                                     {'stride': stride, 'padding': padding, 'dilation': dilation, 'groups': groups})


class AltBackwardLinear(nn.Module):
    def __init__(self, in_features, out_features, bias, backward_type='feedback_alignment'):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_forward = nn.Parameter(torch.Tensor(out_features, in_features))
        self.backward_type = backward_type
        if self.backward_type != 'feedback_alignment' and self.backward_type != 'sign_symmetry':
            raise NotImplementedError('backward_type must be feedback_alignment or sign_symmetry, but %s was given' %
                                      self.backward_type)
        self.register_buffer('weight_backward', torch.Tensor(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            init.kaiming_uniform_(self.weight_forward, a=math.sqrt(5))
            if self.backward_type == 'feedback_alignment':
                init.kaiming_uniform_(self.weight_backward.data, a=math.sqrt(5))
            elif self.backward_type == 'sign_symmetry':
                self.weight_backward = torch.sign(self.weight_forward.data)

            if self.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_forward)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        if self.backward_type == 'sign_symmetry':
            with torch.no_grad():
                self.weight_backward = torch.sign(self.weight_forward)

        assert not self.weight_backward.requires_grad
        assert self.weight_backward.grad is None
        return alt_backward_linear(input, self.weight_forward, self.weight_backward, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, backward_type={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.backward_type
        )


class AltBackwardConvNd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation, transposed, output_padding, groups, bias, padding_mode, backward_type='feedback_alignment'):
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))
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
        self.backward_type = backward_type
        if self.backward_type != 'feedback_alignment' and self.backward_type != 'sign_symmetry':
            raise NotImplementedError('backward_type must be feedback_alignment or sign_symmetry, but %s was given' %
                                      self.backward_type)
        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)
        if transposed:
            self.weight_forward = nn.Parameter(torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
            self.register_buffer('weight_backward', torch.Tensor(
                in_channels, out_channels // groups, *kernel_size))
        else:
            self.weight_forward = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
            self.register_buffer('weight_backward', torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            init.kaiming_uniform_(self.weight_forward, a=math.sqrt(5))
            if self.backward_type == 'feedback_alignment':
                init.kaiming_uniform_(self.weight_backward.data, a=math.sqrt(5))
            elif self.backward_type == 'sign_symmetry':
                self.weight_backward = torch.sign(self.weight_forward.data)

            if self.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_forward)
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        s += ', backward_type={backward_type}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'


class AltBackwardConv2d(AltBackwardConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, padding_mode='zeros', backward_type='feedback_alignment'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation,
                         False, _pair(0), groups, bias, padding_mode, backward_type)

    def _conv_forward(self, input, weight_forward, weight_backward):
        if self.padding_mode != 'zeros':
            return alt_backward_conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                       weight_forward, weight_backward, self.bias, self.stride,
                                       _pair(0), self.dilation, self.groups)
        return alt_backward_conv2d(input, weight_forward, weight_backward, self.bias, self.stride,
                                   self.padding, self.dilation, self.groups)

    def forward(self, input):
        if self.backward_type == 'sign_symmetry':
            with torch.no_grad():
                self.weight_backward = torch.sign(self.weight_forward)
        assert not self.weight_backward.requires_grad
        assert self.weight_backward.grad is None
        return self._conv_forward(input, self.weight_forward, self.weight_backward)
