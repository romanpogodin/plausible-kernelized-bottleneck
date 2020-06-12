import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import losses
import kernels
from warnings import warn


def compute_accuracy(net, data_loader, device, print_result=True):
    """
    Computes accuracy of a given network.
    :param net:                 torch.nn.Module, network
    :param data_loader:         torch.utils.data.DataLoader
    :param device:              torch.device
    :param print_result:        bool, whether to print the found accuracy or not
    :return: float, accuracy in percentage
    """
    correct = 0
    total = 0

    net.eval()

    with torch.no_grad():
        for data in data_loader:
            images, labels = data[0].to(device), data[1].to(device)

            outputs = net(images, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    net.train()

    if print_result:
        print('Accuracy: %.1f %%' % (
                100 * correct / total))
    return 100 * correct / total


def parse_arguments(args=None):
    """
    Parse the arguments.
    :param args: None or list of str (e.g. ['--device', 'cuda:0']). If None, parses command line arguments. .
    :return: Namespace
    """
    parser = argparse.ArgumentParser(description='Configure the run')

    # general parameters
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device; default: cuda:0")
    parser.add_argument('--training-mode', type=str, default='validation',
                        help='Train with validation or test; default: validation')
    parser.add_argument('--record-train-accuracy', action='store_true', default=False,
                        help='Record accuracy on test data (with transformations); default: False')
    parser.add_argument('--validation-split', type=float, default=0.1, help='Validaiton/full train ratio; default: 0.1')
    parser.add_argument('--experiment', type=str, default='vgg', help='Experiment type: vgg or mlp; default: vgg')
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        help='Dataset: CIFAR10, MNIST, kMNIST, fMNIST; default: CIFAR10')
    parser.add_argument("--n-epochs", type=int, default=100, help="Number of epochs; default: 100")
    parser.add_argument("--batch-size", type=int, default=128, help="Mini-batch size; default: 128")

    # optimizers
    parser.add_argument('--optimizer-final', type=str, default='SGD',
                        help='Optimizer for the final layer or backprop: Adam, AdamW or SGD; default: SGD')
    parser.add_argument('--optimizer-local', type=str, default='SGD',
                        help='Optimizer for the local loss: Adam, AdamW or SGD; default: SGD')
    parser.add_argument('--backprop', action='store_true', default=False,
                        help='Do backprop instead of local losses; default: False')
    parser.add_argument("--local-sgd-momentum", type=float, default=0.9,
                        help="SGD momentum for the local loss; default: 0.9")
    parser.add_argument("--final-sgd-momentum", type=float, default=0.9,
                        help="SGD momentum for the final layer/backprop; default: 0.9")
    parser.add_argument("--epoch-decrease-lr", type=int, default=100, nargs="+",
                        help="List of epochs for lr decrease (e.g. --epoch-decrease-lr 100 200); default: [100]")
    parser.add_argument("--opt-lr-decrease", type=float, default=0.25,
                        help='Learning rate multiplier when it decreases; default: 0.25')
    parser.add_argument("--weight-decay-local", type=float, default=0.0,
                        help="Weight decay for local optimizers; default: 0.0")
    parser.add_argument("--weight-decay-final", type=float, default=1e-5,
                        help="Weight decay for final layer or backprop; default: 1e-5")
    parser.add_argument("--final-lr", type=float, default=0.005,
                        help="Learning rate for the final layer/backprop; default: 0.005")
    parser.add_argument("--local-lr", type=float, default=0.001, help="Local loss learning rate; default: 0.001")

    # network parameters
    parser.add_argument('--mlp-layer-size', type=int, default=1024,
                        help='Size of every layer (single value) for the MLP experiment; default: 1024')
    parser.add_argument('--vgg-conv-size-multiplier', type=int, default=1,
                        help='Width multiplier for the convolutional layers in the VGG experiments; default: 1')
    parser.add_argument("--nonlin", type=str, default="lrelu", help="Nonlinearity: relu, lrelu, selu; default: lrelu")
    parser.add_argument("--lrelu-negative-slope", type=float, default=0.01,
                        help="Negative slope for lrelu; default: 0.01")
    parser.add_argument("--dropout-p", type=float, default=0.0, help="Dropout probability; default: 0.0")
    parser.add_argument('--spatial-dropout', action='store_true', default=False,
                        help='Use spatial dropout (i.e. excluding whole channels) for conv layers; default: False')
    parser.add_argument('--batch-norm', action='store_true', default=False,
                        help='Use batch norm; default: False')

    # local loss parameters
    parser.add_argument("--bottleneck-gamma", type=float, default=2.0,
                        help="Bottleneck balance parameter gamma; default: 2.0")
    parser.add_argument("--hsic-kernel-z", type=str, default="cossim",
                        help="HSIC kernel on z (activations): cossim, gaussian, lin; default: cossim")
    parser.add_argument("--hsic-kernel-y", type=str, default="cossim",
                        help="HSIC kernel on y (labels): cossim, gaussian, lin; default: cossim")
    parser.add_argument('--hsic-gaussian-sigma-z', type=float, default=5.0,
                        help='Sigma for the gaussian kernel on z; default: 5.0')
    parser.add_argument('--hsic-gaussian-sigma-y', type=float, default=5.0,
                        help='Sigma for the gaussian kernel on y; default: 5.0')
    parser.add_argument('--hsic-estimate-mode', type=str, default='plausible',
                        help='HSIC estimate: biased or plausible (i.e. pHSIC); default: plausible')

    # local loss grouping and transformations
    parser.add_argument('--center-local-loss-data', action='store_true', default=False,
                        help='Center z (after grouping if any); default: False')
    parser.add_argument('--center-labels', action='store_true', default=False,
                        help='Center y for local losses; default: False')
    parser.add_argument("--divisive-norm-conv", action='store_true', default=False,
                        help='Add divisive normalization (with grouping_dim) for conv layer; default: False')
    parser.add_argument("--divisive-norm-fc", action='store_true', default=False,
                        help='Add divisive normalization (with grouping_dim) for fc layers; default: False')
    parser.add_argument("--divnorm-power", type=float, default=0.2, help='Normalize as var ^ p; default: 0.2')
    parser.add_argument("--grouping-power", type=float, default=0.5,
                        help='Group as var ^ q (should be 1 - divnorm_power for Hebbian updates); default: 0.5')
    parser.add_argument("--grouped-var-delta", type=float, default=0.0,
                        help='Smoothing constant for variance computation; default: 0.0')
    parser.add_argument('--group-conv', action='store_true', default=False,
                        help='Group neurons of conv layers; default: False')
    parser.add_argument("--group-fc", action='store_true', default=False,
                        help='Group neurons of fc layers; default: False')
    parser.add_argument("--grouping-dim", type=int, default=64,
                        help="Final dimensionality of grouped neurons; default: 64")

    if args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args)

    print("Simulation arguments:", args)

    return args


def configure_training_mode(args):
    """
    Configures test/validation runs.
    :param args: Namespace from parse_arguments
    :return: float, bool, bool, bool; validation_split, record_train_acc, record_val_acc, record_test_acc
    """
    if args.training_mode != 'test' and args.training_mode != 'validation':
        raise NotImplementedError('training-mode must be test or validation, but %s was given' % args.training_mode)

    if args.training_mode == 'test':
        args.validation_split = 0.0

    if args.validation_split == 0.0:
        warn('Validation ratio is 0.0, the validation set is the training one w/o random transformations')

    record_train_acc = args.record_train_accuracy
    record_val_acc = True
    record_test_acc = (args.training_mode == 'test')

    return args.validation_split, record_train_acc, record_val_acc, record_test_acc


def record_and_print_running_statistics(record_train_acc, record_val_acc, record_test_acc, print_results,
                                        train_acc, val_acc, test_acc, epoch, net, train_loader, validation_loader,
                                        test_loader, device):
    """
    Records and prints running statistics.
    :param record_train_acc:    bool
    :param record_val_acc:      bool
    :param record_test_acc:     bool
    :param print_results:       bool, print accuracies if recorded
    :param train_acc:           np.array, train accuracy to write to
    :param val_acc:             np.array, validation accuracy to write to
    :param test_acc:            np.array, test accuracy to write to
    :param epoch:               int, epoch number
    :param net:                 nn.Module, network
    :param train_loader:        torch.utils.data.DataLoader
    :param validation_loader:   torch.utils.data.DataLoader
    :param test_loader:         torch.utils.data.DataLoader
    :param device:              torch.device
    """
    if record_train_acc:
        if print_results:
            print('Train accuracy:')
        train_acc[epoch] = compute_accuracy(net, train_loader, device, print_result=print_results)
    if record_val_acc:
        if print_results:
            print('Validation accuracy:')
        val_acc[epoch] = compute_accuracy(net, validation_loader, device, print_result=print_results)
    if record_test_acc:
        if print_results:
            print('Test accuracy:')

        test_acc[epoch] = compute_accuracy(net, test_loader, device, print_result=print_results)


def train_network(net, device, output_loss, optimizer, scheduler, n_epochs,
                  train_loader, validation_loader, test_loader, compute_local_loss, update_local_loss,
                  record_train_acc=False, record_val_acc=True, record_test_acc=False, print_results=True):
    """
    Trains the given network
    :param net:                 nn.Module, network
    :param device:              torch.device
    :param output_loss:         nn.Module, loss for the last layer
    :param optimizer:           torch.optim.Optimizer for the last layer or backprop
    :param scheduler:           torch.optim.lr_scheduler._LRScheduler,
        learning rate scheduler for the last layer or backprop
    :param n_epochs:            int, number of training epochs
    :param train_loader:        torch.utils.data.DataLoader
    :param validation_loader:   torch.utils.data.DataLoader
    :param test_loader:         torch.utils.data.DataLoader
    :param compute_local_loss:  bool, whether to compute the local loss in each layer
    :param update_local_loss:   bool, whether to invoke the optimizer on the local loss in each layer
    :param record_train_acc:    bool
    :param record_val_acc:      bool
    :param record_test_acc:     bool
    :param print_results:       bool, print accuracies if recorded
    :return: np.array of float, np.array of float, np.array of float; train_acc, val_acc, test_acc (in percentage)
    """
    train_acc = -np.ones(n_epochs + 1)
    val_acc = -np.ones(n_epochs + 1)
    test_acc = -np.ones(n_epochs + 1)

    record_and_print_running_statistics(record_train_acc, record_val_acc, record_test_acc, print_results,
                                        train_acc, val_acc, test_acc, 0, net, train_loader, validation_loader,
                                        test_loader, device)  # zero is epoch=0

    for epoch in range(1, n_epochs + 1):
        if print_results:
            print('Epoch %d' % epoch)
        running_loss = np.zeros(net.get_n_hidden_layers() + 1)

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            # local zero_grad, backward and step are done inside
            outputs = net(inputs, labels, compute_local_loss, update_local_loss)

            loss_value = output_loss(outputs, labels)
            loss_value.backward()
            optimizer.step()

            if compute_local_loss:
                running_loss[:-1] += net.get_local_losses()
            running_loss[-1] += loss_value.item()

            if i % 50 == 49:
                if print_results:
                    print('[%d, %5d] loss: %s' % (epoch, i + 1, running_loss / 50))
                running_loss.fill(0.0)

        record_and_print_running_statistics(record_train_acc, record_val_acc, record_test_acc, print_results,
                                            train_acc, val_acc, test_acc, epoch, net, train_loader, validation_loader,
                                            test_loader, device)

        if update_local_loss:
            net.step_local_schedulers()

        scheduler.step()

    return train_acc, val_acc, test_acc


def init_kernel(kernel_name, args_dict):
    """
    Returns a kernels.Kernel.
    :param kernel_name: str, 'cossim', 'gaussian' or 'lin
    :param args_dict:   dict, arguments passed to kernels.Kernel
    :return: kernels.Kernel
    """
    if kernel_name == 'cossim':
        return kernels.CosineSimilarityKernel()
    if kernel_name == 'gaussian':
        return kernels.GaussianKernel(sigma=args_dict['hsic_gaussian_sigma'])
    if kernel_name == 'lin':
        return kernels.LinearKernel()
    else:
        raise NotImplementedError('Currently supported kernels for init_kernel:'
                                  'cossim, gaussian, lin, but %s was given' % kernel_name)


def build_kernels(args):
    """
    Returns kernels for z and y.
    :param args: Namespace from parse_arguments
    :return:     kernels.Kernel, kernels.Kernel; kernel_z, kernel_y
    """
    kernel_z = init_kernel(args.hsic_kernel_z, {'hsic_gaussian_sigma': args.hsic_gaussian_sigma_z})
    kernel_y = init_kernel(args.hsic_kernel_y, {'hsic_gaussian_sigma': args.hsic_gaussian_sigma_y})

    return kernel_z, kernel_y


def build_processing(args):
    """
    Build data processing (such as centering and grouping) for z and y.
    :param args: Namespace from parse_arguments
    :return:     list of functions or None, list of function or None; z processing, y_processing
    """
    z_processing = list()
    y_processing = list()

    if args.group_conv or args.group_fc:
        z_processing.append(lambda x: losses.group_neurons_with_variance(
            x, args.grouping_dim, args.group_conv, args.group_fc, args.grouping_power, args.grouped_var_delta))

    if args.center_local_loss_data:
        z_processing.append(losses.center_data)
    if args.center_labels:
        y_processing.append(losses.center_data)

    def return_none_if_empty(x):
        return None if len(x) == 0 else x

    return return_none_if_empty(z_processing), return_none_if_empty(y_processing)


def get_loss(args):
    """
    Builds the kernelized bottleneck loss function for the network.
    :param args: Namespace from parse_arguments
    :return: losses.HSICzyLoss
    """
    kernel_z, kernel_y = build_kernels(args)
    z_processing, y_processing = build_processing(args)

    return losses.HSICzyLoss(kernel_y, kernel_z, gamma=args.bottleneck_gamma, y_processing=y_processing,
                             z_processing=z_processing, mode=args.hsic_estimate_mode)


def get_nonlinearity(args):
    """
    Builds the nonlinearity.
    :param args: Namespace from parse_arguments
    :return: nn.Module
    """
    if args.nonlin == 'relu':
        return nn.ReLU(inplace=False)
    if args.nonlin == 'lrelu':
        return nn.LeakyReLU(negative_slope=args.lrelu_negative_slope, inplace=False)
    if args.nonlin == 'selu':
        return nn.SELU(inplace=False)

    raise NotImplementedError("Nonlinearity must be either relu or lrelu or selu, but %s was given" % args.nonlin)


def build_optimizer_and_parameters(opt_name, lr, weight_decay, sgd_momentum, epoch_decrease_lr, opt_lr_decrease):
    """
    Builds an optimizer and parameters for it and for its learning rate scheduler.
    :param opt_name:            str, 'AdamW', 'Adam' or 'SGD'
    :param lr:                  float, initial learning rate
    :param weight_decay:        float
    :param sgd_momentum:        float, momentum used if opt_name == 'SGD'
    :param epoch_decrease_lr:   list of int, epochs when the learning rate decreases
    :param opt_lr_decrease:     float, multiplier by which the learning rate decreases at epoch_decrease_lr
    :return: torch.optim.Optimizer, dict, dict; optimizer, opt_arguments_dict, scheduler_arguments_dict
    """
    opt_arguments_dict = {'lr': lr, 'weight_decay': weight_decay}

    if opt_name == 'AdamW':
        optimizer = optim.AdamW
    elif opt_name == 'Adam':
        optimizer = optim.Adam
    elif opt_name == 'SGD':
        optimizer = optim.SGD
        opt_arguments_dict.update({'momentum': sgd_momentum})
    else:
        raise NotImplementedError('optimizer_local must be either AdamW or Adam or SGD, '
                                  'but %s was passed' % opt_name)
    scheduler_arguments_dict = {'milestones': epoch_decrease_lr, 'gamma': opt_lr_decrease}

    return optimizer, opt_arguments_dict, scheduler_arguments_dict


def choose_optimizers_and_parameters(args):
    """
    Builds local and final layer optimizers, their arguments and their schedulers.
    :param args: Namespace from parse_arguments
    :return: torch.optim.Optimizer, dict, dict, torch.optim.Optimizer, dict, dict;
        optimizer_local, opt_arguments_dict_local, scheduler_arguments_dict_local,
        optimizer_final, opt_arguments_dict_final, scheduler_arguments_dict_final
    """
    return build_optimizer_and_parameters(args.optimizer_local, args.local_lr, args.weight_decay_local,
                                          args.local_sgd_momentum, args.epoch_decrease_lr, args.opt_lr_decrease) + \
           build_optimizer_and_parameters(args.optimizer_final, args.final_lr, args.weight_decay_final,
                                          args.final_sgd_momentum, args.epoch_decrease_lr, args.opt_lr_decrease)
