import torch.nn as nn
import torch
import losses


def get_task_dimensions(task):
    """
    Get the number of channels, channel size (height * width) and number of classes of a dataset.
    :param task: str, '*MNIST' or 'CIFAR*' for any string *
    :return: int, int, int; in_channels, input_size, n_classes
    """
    if 'MNIST' in task:
        in_channels = 1
        input_size = 28 * 28
        n_classes = 10
    elif 'CIFAR' in task:
        in_channels = 3
        input_size = 32 * 32
        n_classes = int(task[5:])
    else:
        raise ValueError('Task must be either *MNIST or CIFAR*, but %s was given' % task)
    return in_channels, input_size, n_classes


class SequentialWithLocalLoss(nn.Sequential):
    """
    Adaptation of torch.nn.Sequential to pass local loss arguments.
    """
    def forward(self, x, y=None, compute_local_loss=False, update_local_loss=False):
        """
        Forward pass.
        :param x:                   torch.Tensor, input to the network
        :param y:                   torch.Tensor or None, labels
            (if None, trying to compute the local loss will result in an error)
        :param compute_local_loss:  bool, whether to compute the local loss in each layer
        :param update_local_loss:   bool, whether to invoke the optimizer on the local loss in each layer
        :return: torch.Tensor, x passed through the layers
        """
        for module in self:
            x = module(x, y, compute_local_loss, update_local_loss)
        return x


class DivisiveNorm(nn.Module):
    """
    Divisive normalization with grouping.
    """
    def __init__(self, power, grouped_dim, smoothing_delta=0.0):
        """
        :param power:           float, normalize by var^power
        :param grouped_dim:     int, normalize centered x.view(x.shape[0], grouped_dim, -1) by var w.r.t. dim=-1
        :param smoothing_delta: float, smoothing parameter s.t. var(x) = (delta + sum_{i=1}^n (x_i - mean(x))^2) / n
        """
        super().__init__()
        self.power = power
        self.grouped_dim = grouped_dim
        self.smoothing_delta = smoothing_delta

    def extra_repr(self):
        s = 'power={power}, grouped_dim={grouped_dim}, smoothing_delta={smoothing_delta}'
        return s.format(**self.__dict__)

    def forward(self, x):
        """
        Forward pass.
        :param x: torch.Tensor
        :return: torch.Tensor, centered and normalized (within the groups) x
        """
        grouped_shape = (x.shape[0], self.grouped_dim, -1)
        x_centered = losses.center_data(x.view(grouped_shape))
        var = (torch.norm(x_centered, dim=-1, p=2.0) ** 2 + self.smoothing_delta) / x_centered.shape[-1]
        return (x_centered.view(grouped_shape) / (var ** self.power).view(grouped_shape)).view(x.shape)


class Layer(nn.Module):
    """
    Layer with a local loss. Implementation idea from https://github.com/anokland/local-loss
    """
    def __init__(self):
        super().__init__()
        self.layer = None
        self.nonlinearity = None
        self.local_loss = None
        self.local_loss_value = None
        self.local_opt = None
        self.local_opt_scheduler = None
        self.pre_activation = None
        self.post_activation = None
        self.pooling = None
        self.dropout = None
        self.batch_norm = None
        self.divisive_norm = None

    def get_loss(self):
        """
        :return: current value of self.local_loss_value
        """
        return self.local_loss_value.item()

    def step_scheduler(self):
        """
        Step of the learning rate scheduler.
        """
        if self.local_opt_scheduler is not None:
            self.local_opt_scheduler.step()

    @staticmethod
    def apply_if_not_none(activation, transform):
        if transform is None:
            return activation
        return transform(activation)

    def forward(self, x, y=None, compute_local_loss=False, update_local_loss=False):
        """
        Forward pass with the following architecture
            (batch norm, pooling, divisive norm and dropout could be None, in which case they're not applied):

            1. pre-loss: x -> batch norm -> nonlinearity -> pooling -> z

            2. loss: z -> loss

            2.5 (if update_local_loss): loss -> loss.backward -> optimizer.step
                                        -> z.detach_()

            3. post_loss: z -> divisive norm -> dropout
               -> return
        :param x:                   torch.Tensor, input
        :param y:                   torch.Tensor or None, labels
            (if None, trying to compute the local loss will result in an error)
        :param compute_local_loss:  bool, whether to compute the local loss in each layer
        :param update_local_loss:   bool, whether to invoke the optimizer on the local loss in each layer
        :return: torch.Tensor, transformed x
        """
        if update_local_loss:
            self.local_opt.zero_grad()

        if update_local_loss:
            assert x.requires_grad is False
        assert (y is None and compute_local_loss) is False
        self.activation = self.layer(x)

        self.activation = self.apply_if_not_none(self.activation, self.batch_norm)
        self.activation = self.nonlinearity(self.activation)
        self.activation = self.apply_if_not_none(self.activation, self.pooling)

        if compute_local_loss and update_local_loss:
            self.local_loss_value = self.local_loss(self.activation, y)

            self.local_loss_value.backward()
            self.local_opt.step()
            self.activation.detach_()
        elif compute_local_loss:
            self.local_loss_value = self.local_loss(self.activation.detach(), y.detach())

        self.activation = self.apply_if_not_none(self.activation, self.divisive_norm)
        self.activation = self.apply_if_not_none(self.activation, self.dropout)

        return self.activation


class LinearLayer(Layer):
    """
    Linear (fully connected, or fc) layer modified to compute a local loss.
    """
    def __init__(self, num_in, num_out, nonlinearity, local_loss=None, local_opt=None, local_opt_scheduler=None,
                 bias=False, opt_arguments_dict=None, scheduler_arguments_dict=None, dropout_p=0.0, batch_norm=False,
                 divisive_norm=None):
        """
        :param num_in:                      int, input dimensionality
        :param num_out:                     int, number of neurons in the layer
        :param nonlinearity:                nn.Module, forward(x) applies an element-wise function to torch.Tensor x
            if nn.SELU, dropout becomes nn.AlphaDropout
        :param local_loss:                  nn.Module with forward(self, z, y) that returns float
        :param local_opt:                   torch.optim.Optimizer, optimizer to apply on local_loss
        :param local_opt_scheduler:         torch.optim.lr_scheduler._LRScheduler,
            learning rate scheduler for the optimizer
        :param bias:                        bool, bias for nn.Linear
        :param opt_arguments_dict:          dict, optimizer's arguments
        :param scheduler_arguments_dict:    dict, scheduler's arguments
        :param dropout_p:                   float, dropout probability
        :param batch_norm:                  bool, whether to apply batchnorm or not
        :param divisive_norm:               DivisiveNorm or None, divisive normalization
        """
        super().__init__()

        self.layer = nn.Linear(num_in, num_out, bias=bias)
        self.nonlinearity = nonlinearity

        self.local_loss = local_loss
        self.local_loss_value = 0.0

        if dropout_p > 0.0:
            if isinstance(nonlinearity, nn.SELU):
                self.dropout = nn.AlphaDropout(dropout_p, inplace=False)
            else:
                self.dropout = nn.Dropout(dropout_p, inplace=False)

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(num_out)

        if local_opt is not None:
            self.local_opt = local_opt(self.parameters(), **opt_arguments_dict)
        if local_opt_scheduler is not None:
            self.local_opt_scheduler = local_opt_scheduler(self.local_opt, **scheduler_arguments_dict)

        if divisive_norm is not None:
            self.divisive_norm = divisive_norm

    def forward(self, x, y=None, compute_local_loss=False, update_local_loss=False):
        """
        Forward pass with the following architecture
            (batch norm, divisive norm and dropout could be None, in which case they're not applied):

            1. pre-loss: x -> batch norm -> nonlinearity -> z

            2. loss: z -> loss

            2.5 (if update_local_loss): loss -> loss.backward -> optimizer.step
                                        -> z.detach_()

            3. post_loss: z -> divisive norm -> dropout
               -> return
        :param x:                   torch.Tensor, input
        :param y:                   torch.Tensor or None, labels
            (if None, trying to compute the local loss will result in an error)
        :param compute_local_loss:  bool, whether to compute the local loss in each layer
        :param update_local_loss:   bool, whether to invoke the optimizer on the local loss in each layer
        :return: torch.Tensor, transformed x
        """
        x = x.view(x.shape[0], -1)
        return super().forward(x, y, compute_local_loss, update_local_loss)


class ConvLayer(Layer):
    """
    Convolutional (conv) layer modified to compute a local loss.
    """
    def __init__(self, in_channels, out_channels, kernel_size, nonlinearity, pooling=None, local_loss=None,
                 local_opt=None, local_opt_scheduler=None, bias=False, opt_arguments_dict=None,
                 scheduler_arguments_dict=None, dropout_p=0.0, batch_norm=False, divisive_norm=None,
                 spatial_dropout=False):
        """
        :param in_channels:                 int, number of channels in the input
        :param out_channels:                int, number of channels in the layer
        :param kernel_size:                 int, size of a kernel (for s, the kernel is s by s)
        :param nonlinearity:                nn.Module, forward(x) applies an element-wise function to torch.Tensor x
            if nn.SELU, dropout becomes nn.AlphaDropout (or nn.FeatureAlphaDropout if spatial_dropout == True)
        :param pooling:
        :param local_loss:                  nn.Module with forward(self, z, y) that returns float
        :param local_opt:                   torch.optim.Optimizer, optimizer to apply on local_loss
        :param local_opt_scheduler:         torch.optim.lr_scheduler._LRScheduler,
            learning rate scheduler for the optimizer
        :param bias:                        bool, bias for nn.Conv2d
        :param opt_arguments_dict:          dict, optimizer's arguments
        :param scheduler_arguments_dict:    dict, scheduler's arguments
        :param dropout_p:                   float, dropout probability
        :param batch_norm:                  bool, whether to apply batchnorm or not
        :param divisive_norm:               DivisiveNorm or None, divisive normalization
        :param spatial_dropout:             bool, whether to use spatial dropout (which drops out whole channels)
            or the standard dropout
        """
        super().__init__()

        self.layer = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2, bias=bias)
        self.nonlinearity = nonlinearity

        self.local_loss = local_loss
        self.local_loss_value = 0.0

        if pooling == 'max':
            self.pooling = nn.MaxPool2d(2, 2)
        elif pooling == 'avg':
            self.pooling = nn.AvgPool2d(2, 2)

        if dropout_p > 0.0:
            if spatial_dropout:
                if isinstance(nonlinearity, nn.SELU):
                    self.dropout = nn.FeatureAlphaDropout(dropout_p, inplace=False)
                else:
                    self.dropout = nn.Dropout2d(dropout_p, inplace=False)
            else:
                if isinstance(nonlinearity, nn.SELU):
                    self.dropout = nn.AlphaDropout(dropout_p, inplace=False)
                else:
                    self.dropout = nn.Dropout(dropout_p, inplace=False)

        if batch_norm:
            self.batch_norm = nn.BatchNorm2d(out_channels)

        if local_opt is not None:
            self.local_opt = local_opt(self.parameters(), **opt_arguments_dict)
        if local_opt_scheduler is not None:
            self.local_opt_scheduler = local_opt_scheduler(self.local_opt, **scheduler_arguments_dict)

        if divisive_norm is not None:
            self.divisive_norm = divisive_norm


def get_divisive_norm(ind, divisive_norm_list):
    """
    Returns an element from a list of divisive norms if the list is not None.
    :param ind:                 index of the elements
    :param divisive_norm_list:  list of DivisiveNorm
    :return: DivisiveNorm or None (if divisive_norm_list == None)
    """
    if divisive_norm_list is None:
        return None
    return divisive_norm_list[ind]


class Network(nn.Module):
    def __init__(self, nonlinearity, local_loss, local_opt, local_opt_scheduler, conv_channels_sizes,
                 conv_kernels_sizes, do_pooling_by_layer, fc_layers_sizes, pooling_type='max', task='MNIST', bias=False,
                 local_opt_arguments_dict=None,  local_scheduler_arguments_dict=None, dropout_p=0.0, batch_norm=False,
                 divisive_norm_list_conv=None, spatial_dropout=False, divisive_norm_list_fc=None):
        """
        A network with convolutional layers followed by fully connected layers.
        :param nonlinearity:                    nn.Module, forward(x) applies an element-wise function to torch.Tensor x
            if nn.SELU, dropout becomes nn.AlphaDropout
            (or nn.FeatureAlphaDropout if spatial_dropout == True and the layer is conv)
        :param local_loss:                      nn.Module with forward(self, z, y) that returns float
        :param local_opt:                       torch.optim.Optimizer, optimizer to apply on local_loss
        :param local_opt_scheduler:             torch.optim.lr_scheduler._LRScheduler,
            learning rate scheduler for the optimizer
        :param conv_channels_sizes:             list of int, number of channels in each conv layers
        :param conv_kernels_sizes:              list of int, size of a kernel (for s, the kernel is s by s)
            in each conv layer
        :param do_pooling_by_layer:             list of bool, whether to do pooling in each conv layer
        :param fc_layers_sizes:                 list of int, size of the fully connected layers
        :param pooling_type:                    str, 'max' or 'avg' (average pooling); pooling is always 2 by 2
        :param task:                            str, '*MNIST' or 'CIFAR*' for any string *.
        :param bias:                            bool, bias for nn.Linear and nn.Conv2d
        :param local_opt_arguments_dict:        dict, optimizer's arguments
        :param local_scheduler_arguments_dict:  dict, scheduler's arguments
        :param dropout_p:                       float, dropout probability
        :param batch_norm:                      bool, whether to apply batchnorm or not
        :param divisive_norm_list_conv:         list of DivisiveNormalization or None, div norm for conv layers
        :param spatial_dropout:                 bool, whether to use spatial dropout (which drops out whole channels)
            or the standard dropout in convolutional layers
        :param divisive_norm_list_fc:           list of DivisiveNormalization or None, div norm for fc layers
        """
        super().__init__()

        in_channels, input_size, n_classes = get_task_dimensions(task)
        layers = []

        for ind, channel_size in enumerate(conv_channels_sizes):
            layers.append(ConvLayer(in_channels, channel_size, conv_kernels_sizes[ind], nonlinearity,
                                    (pooling_type if do_pooling_by_layer[ind] else None), local_loss, local_opt,
                                    local_opt_scheduler, bias, local_opt_arguments_dict,
                                    local_scheduler_arguments_dict, dropout_p, batch_norm,
                                    divisive_norm=get_divisive_norm(ind, divisive_norm_list_conv),
                                    spatial_dropout=spatial_dropout))
            in_channels = channel_size
            if do_pooling_by_layer[ind]:  # pooling always pools by 2 here
                input_size = input_size // 4

        in_neurons = in_channels * input_size

        for ind, layer_size in enumerate(fc_layers_sizes):
            layers.append(LinearLayer(in_neurons, layer_size, nonlinearity, local_loss, local_opt, local_opt_scheduler,
                                      bias, local_opt_arguments_dict, local_scheduler_arguments_dict,
                                      dropout_p, batch_norm,
                                      divisive_norm=get_divisive_norm(ind, divisive_norm_list_fc)))
            in_neurons = layer_size

        self.n_classes = n_classes
        self.hidden_layers = SequentialWithLocalLoss(*layers)
        self.softmax_layer = nn.Linear(in_neurons, self.n_classes, bias=True)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def get_local_losses(self):
        """
        Get values of the local losses in each layer
        :return: list of float
        """
        return [layer.get_loss() for layer in self.hidden_layers]

    def get_n_hidden_layers(self):
        return len(self.hidden_layers)

    def step_local_schedulers(self):
        """
        Step the learning rate scheduler in each layer.
        """
        for layer in self.hidden_layers:
            layer.step_scheduler()

    def make_labels_one_hot(self, y):
        """
        Take a vector of labels and make it one-hot labels.
        :param y:   torch.Tensor with 1 dimension, each entry is a class from 0 to n-1
        :return:    torch.Tensor, y_onehot such that y_onehot.shape = (y.shape[0], n) and y_onehot[i][y[i]] == 1
        """
        y_onehot = torch.zeros(y.shape[0], self.n_classes, dtype=torch.float, device=y.device)
        y_onehot.scatter_(1, y.view(-1, 1), 1)
        return y_onehot.detach()

    def forward(self, x, y, compute_local_loss=False, update_local_loss=False):
        """
        Forward pass through the network.
        :param x:                   torch.Tensor, input
        :param y:                   torch.Tensor, labels
        :param compute_local_loss:  bool, whether to compute the local loss in each layer
        :param update_local_loss:   bool, whether to invoke the optimizer on the local loss in each layer
        :return: torch.Tensor, softmax(net(x))
        :return:
        """
        y_onehot = self.make_labels_one_hot(y)
        hidden = self.hidden_layers(x, y_onehot, compute_local_loss, update_local_loss)

        if update_local_loss:
            assert hidden.requires_grad is False
        return self.softmax_layer(hidden.view(x.shape[0], -1))
