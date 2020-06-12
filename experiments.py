import torch
import torch.nn as nn
import numpy as np
import networks
import utils
import datasets


def run_mlp_experiment(args, device):
    """
    Runs the experiment with a 3-layers fully connected network.
    :param args:    Namespace from utils.parse_arguments
    :param device:  torch.device
    :return: np.array of float, np.array of float, np.array of float; train_acc, val_acc, test_acc (in percentage)
    """
    validation_ratio, record_train_acc, record_val_acc, record_test_acc = utils.configure_training_mode(args)

    train_loader, validation_loader, test_loader = datasets.build_loaders_by_dataset(
        args.dataset, args.batch_size, validation_ratio=validation_ratio, train_validation_split_seed=0)
    local_loss = utils.get_loss(args)
    nonlinearity = utils.get_nonlinearity(args)

    optimizer_local, local_opt_arguments_dict, local_scheduler_arguments_dict, \
        optimizer_final, final_opt_arguments_dict, final_scheduler_arguments_dict = \
        utils.choose_optimizers_and_parameters(args)

    conv_sizes = []
    do_pooling = []
    kernel_sizes = []

    fc_layers = [args.mlp_layer_size, args.mlp_layer_size, args.mlp_layer_size]

    if args.divisive_norm_fc:
        divisive_norm_list = [networks.DivisiveNorm(args.divnorm_power, args.grouping_dim,
                                                    args.grouped_var_delta)
                              for i in range(len(fc_layers))]
    else:
        divisive_norm_list = None

    net = networks.Network(nonlinearity, local_loss, optimizer_local,
                           torch.optim.lr_scheduler.MultiStepLR, conv_sizes, kernel_sizes,
                           do_pooling, fc_layers, 'max', args.dataset, bias=False,
                           local_opt_arguments_dict=local_opt_arguments_dict,
                           local_scheduler_arguments_dict=local_scheduler_arguments_dict,
                           dropout_p=args.dropout_p, batch_norm=args.batch_norm,
                           divisive_norm_list_conv=None, divisive_norm_list_fc=divisive_norm_list,
                           spatial_dropout=args.spatial_dropout)

    net = net.to(device)
    print(net)

    final_loss = nn.CrossEntropyLoss()

    if args.backprop:
        final_opt = optimizer_final(net.parameters(), **final_opt_arguments_dict)
        compute_local_loss = False
        update_local_loss = False
    else:
        final_opt = optimizer_final(net.softmax_layer.parameters(), **final_opt_arguments_dict)
        compute_local_loss = True
        update_local_loss = True

    final_scheduler = torch.optim.lr_scheduler.MultiStepLR(final_opt, **final_scheduler_arguments_dict)

    train_acc, val_acc, test_acc = utils.train_network(
        net, device, final_loss, final_opt, final_scheduler, args.n_epochs, train_loader, validation_loader,
        test_loader, compute_local_loss=compute_local_loss, update_local_loss=update_local_loss,
        record_train_acc=record_train_acc, record_val_acc=record_val_acc, record_test_acc=record_test_acc,
        print_results=True)

    return train_acc, val_acc, test_acc


def run_vgg_experiment(args, device):
    """
    Runs the experiment with a vgg-like network.
    :param args:    Namespace from utils.parse_arguments
    :param device:  torch.device
    :return: np.array of float, np.array of float, np.array of float; train_acc, val_acc, test_acc (in percentage)
    """
    validation_ratio, record_train_acc, record_val_acc, record_test_acc = utils.configure_training_mode(args)

    train_loader, validation_loader, test_loader = datasets.build_cifar10_loaders(args.batch_size,
                                                                                  validation_ratio=validation_ratio,
                                                                                  train_validation_split_seed=0)
    local_loss = utils.get_loss(args)
    nonlinearity = utils.get_nonlinearity(args)

    optimizer_local, local_opt_arguments_dict, local_scheduler_arguments_dict, \
        optimizer_final, final_opt_arguments_dict, final_scheduler_arguments_dict = \
        utils.choose_optimizers_and_parameters(args)

    conv_sizes = [128, 256, 256, 512, 512, 512]

    if args.vgg_conv_size_multiplier != 1:
        for i in range(len(conv_sizes)):
            conv_sizes[i] = conv_sizes[i] * args.vgg_conv_size_multiplier
    do_pooling = [False, True, False, True, True, True]

    if args.divisive_norm_conv:
        divisive_norm_list_conv = [networks.DivisiveNorm(args.divnorm_power, args.grouping_dim, args.grouped_var_delta)
                                   for i in range(len(conv_sizes))]
    else:
        divisive_norm_list_conv = None

    kernel_sizes = [3 for i in range(len(conv_sizes))]
    fc_layers = [1024]

    if args.divisive_norm_fc:
        divisive_norm_list_fc = [networks.DivisiveNorm(args.divnorm_power, args.grouping_dim,
                                                       args.grouped_var_delta)
                                 for i in range(len(fc_layers))]
    else:
        divisive_norm_list_fc = None

    net = networks.Network(nonlinearity, local_loss, optimizer_local,
                           torch.optim.lr_scheduler.MultiStepLR, conv_sizes, kernel_sizes,
                           do_pooling, fc_layers, 'max', 'CIFAR10', bias=False,
                           local_opt_arguments_dict=local_opt_arguments_dict,
                           local_scheduler_arguments_dict=local_scheduler_arguments_dict,
                           dropout_p=args.dropout_p, batch_norm=args.batch_norm,
                           divisive_norm_list_conv=divisive_norm_list_conv, divisive_norm_list_fc=divisive_norm_list_fc,
                           spatial_dropout=args.spatial_dropout)

    net = net.to(device)
    print(net)

    final_loss = nn.CrossEntropyLoss()

    if args.backprop:
        final_opt = optimizer_final(net.parameters(), **final_opt_arguments_dict)
        compute_local_loss = False
        update_local_loss = False
    else:
        final_opt = optimizer_final(net.softmax_layer.parameters(), **final_opt_arguments_dict)
        compute_local_loss = True
        update_local_loss = True

    final_scheduler = torch.optim.lr_scheduler.MultiStepLR(final_opt, **final_scheduler_arguments_dict)

    train_acc, val_acc, test_acc = utils.train_network(
        net, device, final_loss, final_opt, final_scheduler, args.n_epochs, train_loader, validation_loader,
        test_loader, compute_local_loss=compute_local_loss, update_local_loss=update_local_loss,
        record_train_acc=record_train_acc, record_val_acc=record_val_acc, record_test_acc=record_test_acc,
        print_results=True)

    return train_acc, val_acc, test_acc


def main():
    args = utils.parse_arguments()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(device)

    with np.printoptions(precision=4, suppress=True):
        if args.experiment == 'mlp':
            train_acc, val_acc, test_acc = run_mlp_experiment(args, device)
        elif args.experiment == 'vgg':
            train_acc, val_acc, test_acc = run_vgg_experiment(args, device)
        else:
            raise NotImplementedError('experiment must be mlp or vgg, but %s was given' % args.experiment)


if __name__ == '__main__':
    main()
