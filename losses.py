import numpy as np
import torch
import torch.nn as nn
import kernels
import alt_feedback_layers


def center_data(x, dim=-1):
    """
    Center x along the dimension dim.
    :param x:   torch.Tensor
    :param dim: int, dim for centering
    :return: torch.Tensor, centered x
    """
    return x - x.mean(dim=dim, keepdim=True)


def compute_grouped_variance(x, grouped_dim, smoothing_delta=0.0):
    """
    Computes (smoothed) variance within groups of tensor x.
    :param x:               torch.Tensor with len(x.shape) > 1
    :param grouped_dim:     int, number of groups
    :param smoothing_delta: float, smoothing parameter s.t. var(x) = (delta + sum_{i=1}^n (x_i - mean(x))^2) / n
    :return: torch.Tensor, grouped x of shape (x.shape[0], grouped_dim)
    """
    grouped_shape = (x.shape[0], grouped_dim, -1)
    x_centered = center_data(x.view(grouped_shape))
    return ((torch.norm(x_centered, dim=-1, p=2.0)) ** 2 + smoothing_delta) / x_centered.shape[-1]


def group_neurons_with_variance(x, grouped_dim=64, group_conv=False, group_fc=False,
                                grouping_power=0.5, smoothing_delta=0.0):
    """
    Groups layer activation with some power of smoothed variance.
    :param x:               torch.Tensor, layer activation
    :param grouped_dim:     int, number of groups
    :param group_conv:      bool, whether to group conv layers or not
    :param group_fc:        bool, whether to group fc layers or not
    :param grouping_power:  float, p to group with var^p
    :param smoothing_delta: float, smoothing parameter s.t. var(x) = (delta + sum_{i=1}^n (x_i - mean(x))^2) / n
    :return: torch.Tensor, grouped activation of shape (x.shape[0], grouped_dim)
    """
    if group_conv and x.dim() == 4:
        assert x.shape[1] > 3, 'tried to group the input to the network'
        assert np.prod(list(x.shape[1:])) // grouped_dim > 1, 'grouping dim too high, will zero out the activity'
        return compute_grouped_variance(x, grouped_dim, smoothing_delta) ** grouping_power
    if group_fc and x.dim() == 2:
        assert np.prod(list(x.shape[1:])) // grouped_dim > 1, 'grouping dim too high, will zero out the activity'
        return compute_grouped_variance(x, grouped_dim, smoothing_delta) ** grouping_power
    return x


class HSICzyLoss(nn.Module):
    """
    Local loss of the form HSIC(Z,Z) - gamma HSIC(Z,Y).
    """
    def __init__(self, y_kernel, z_kernel, gamma=2.0, y_processing=None, z_processing=None, mode='biased'):
        """
        :param y_kernel:        kernels.Kernel, kernel to compare one-hot labels
        :param z_kernel:        kernels.Kernel, kernel to compare activations
        :param gamma:           balance parameter for HSIC(Z,Z) - gamma HSIC(Z,Y)
        :param y_processing:    list or None, processing for the labels
        :param z_processing:    list or None, processing for the activations
        :param mode:            str, 'biased' for HSIC or 'plausible' for pHSIC
        """
        super().__init__()

        self.y_kernel = y_kernel
        self.z_kernel = z_kernel
        self.gamma = gamma
        self.z_processing = z_processing
        self.y_processing = y_processing
        self.mode = mode

    def extra_repr(self):
        s = 'mode={mode}, y_kernel={y_kernel.__class__.__name__}, z_kernel={z_kernel.__class__.__name__}, gamma={gamma}'
        return s.format(**self.__dict__)

    @staticmethod
    def process_input(data, data_processing):
        """
        Process the input with a list of transformations.
        :param data:            torch.Tensor
        :param data_processing: list or None, a list of transformations for data
        :return: torch.Tensor,
            data_processing applied sequentially to data if data_processing is not None, data otherwise
        """
        if data_processing is not None:
            if isinstance(data_processing, list) or isinstance(data_processing, tuple):
                for transformation in data_processing:
                    data = transformation(data)
            else:
                data = data_processing(data)
        return data

    def forward(self, z, y):
        """
        Forward pass.
        :param z: torch.Tensor, activations
        :param y: torch.Tensor, one-hot labels
        :return: float, estimate of HSIC(Z,Z) - gamma HSIC(Z,Y) (or pHSIC, if self.mode == 'plausible')
        """
        return kernels.estimate_hsic_zy_objective(self.process_input(z, self.z_processing),
                                                  self.process_input(y, self.y_processing),
                                                  self.z_kernel, self.y_kernel, self.gamma, self.mode)


class LocalCrossEntropy(nn.Module):
    """
    Loss for layer-wise classification
    """
    def __init__(self, projection_size, out_size, is_conv=False, n_channels=None, alt_feedback_type=None):
        """
        :param projection_size:   int, final size (after AdaptiveAvgPool2d) for conv layers
        :param out_size:          int, number of classes
        :param is_conv:           bool, is the layer convolutional
        :param n_channels:        int, number of channels in a conv layer
        :param alt_feedback_type: str or None, type of alternative feedback (feedback_alignment or sign_symmetry)
        """
        super().__init__()

        self.projection_size = projection_size
        self.out_size = out_size
        self.is_conv = is_conv
        self.alt_feedback_type = alt_feedback_type
        projection = list()

        if self.is_conv:
            pool_width = int(np.sqrt(projection_size // n_channels))
            pool_height = projection_size // n_channels // pool_width
            projection.append(nn.AdaptiveAvgPool2d((pool_height, pool_width)))
            projection.append(nn.Flatten())
            if projection_size != pool_height * pool_width * n_channels:
                print('Adaptive projection size from %d to %d' % (projection_size, pool_height * pool_width * n_channels))
                projection_size = pool_height * pool_width * n_channels

        if alt_feedback_type is None:
            projection.append(nn.Linear(projection_size, out_size, bias=True))
        else:
            projection.append(alt_feedback_layers.AltBackwardLinear(projection_size, out_size,
                                                                    bias=True, backward_type=alt_feedback_type))

        self.projection = nn.Sequential(*projection)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def extra_repr(self):
        s = 'projection_size={projection_size}, ' \
            'out_size={out_size}, ' \
            'is_conv={is_conv}, ' \
            'alt_feedback_type={alt_feedback_type}'
        return s.format(**self.__dict__)

    def forward(self, z, y_onehot):
        """
        Forward pass.
        :param z: torch.Tensor, activations
        :param y: torch.Tensor, one-hot labels
        :return: float, cross-entropy
        """
        return self.cross_entropy_loss(self.projection(z), torch.argmax(y_onehot, dim=-1))
