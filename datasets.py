import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler, Sampler, RandomSampler, SequentialSampler
import numpy as np


class SubsetDeterministicSampler(Sampler):
    """
    Samples elements non-randomly from a given list of indices.
    """
    def __init__(self, indices):
        """
        :param indices: list, tuple, np.array or torch.Tensor of ints, a sequence of indices
        """
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.arange(len(self.indices)))

    def __len__(self):
        return len(self.indices)


def build_cifar10_transforms():
    """
    Crops, flips and normalizes train data, normalizes test data.
    :return:    torchvision.transforms, torchvision.transforms; transform_train, transform_test
    """
    mean = (0.4914, 0.4822, 0.4465)  # https://github.com/kuangliu/pytorch-cifar/issues/19
    std = (0.247, 0.243, 0.261)

    transform_train = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean, std)])

    return transform_train, transform_test


def build_mnist_transforms():
    """
    Normalizes data.
    :return:    torchvision.transforms, torchvision.transforms; transform_train, transform_test
    """
    mean = (0.5,)
    std = (0.5,)

    transform_train = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean, std)])

    return transform_train, transform_test


def build_loaders(torch_dataset, build_transforms, batch_size=128, validation_ratio=0.1, train_validation_split_seed=0):
    """
    Builds data loaders for the given dataset.
    :param torch_dataset:               str, 'CIFAR10', 'MNIST', 'fMNIST' (fashion-MNIST) or 'kMNIST' (Kuzushiji-MNIST)
    :param build_transforms:            function that returns two lists of torchvision.transforms (train and test)
    :param batch_size:                  int, batch size
    :param validation_ratio:            float, validation size / full train set size.
        If zero, validation set is the full train set but without random transformations (e.g. crops)
    :param train_validation_split_seed: int, numpy random seed for train/validation split
    :return: torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader;
        train_loader, valid_loader, test_loader
    """
    transform_train, transform_test = build_transforms()

    train_set = torch_dataset(root='./data', train=True, download=True, transform=transform_train)
    val_set = torch_dataset(root='./data', train=True, download=True, transform=transform_test)
    test_set = torch_dataset(root='./data', train=False, download=True, transform=transform_test)

    if validation_ratio > 0.0:
        indices = np.arange(len(train_set), dtype=int)
        np.random.RandomState(train_validation_split_seed).shuffle(indices)

        split_size = int(validation_ratio * len(train_set))

        train_sampler = SubsetRandomSampler(indices[split_size:])
        valid_sampler = SubsetDeterministicSampler(indices[:split_size])  # always in the same order
    else:
        print('\nNo validation, validation data will be the training set with test transforms\n')
        train_sampler = RandomSampler(train_set)
        valid_sampler = SequentialSampler(val_set)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               num_workers=2, sampler=train_sampler, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size,
                                               num_workers=2, sampler=valid_sampler, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, valid_loader, test_loader


def build_cifar10_loaders(batch_size=128, validation_ratio=0.1, train_validation_split_seed=0):
    """
    Builds data loaders for CIFAR10.
    :param batch_size:                  int, batch size
    :param validation_ratio:            float, validation size / full train set size.
        If zero, validation set is the full train set but without random transformations (e.g. crops)
    :param train_validation_split_seed: int, numpy random seed for train/validation split
    :return: torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader;
        train_loader, valid_loader, test_loader
    """
    return build_loaders(torchvision.datasets.CIFAR10, build_cifar10_transforms,
                         batch_size, validation_ratio, train_validation_split_seed)


def build_mnist_loaders(batch_size=128, validation_ratio=0.1, train_validation_split_seed=0):
    """
    Builds data loaders for MNIST.
    :param batch_size:                  int, batch size
    :param validation_ratio:            float, validation size / full train set size.
        If zero, validation set is the full train set but without random transformations (e.g. crops)
    :param train_validation_split_seed: int, numpy random seed for train/validation split
    :return: torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader;
        train_loader, valid_loader, test_loader
    """
    return build_loaders(torchvision.datasets.MNIST, build_mnist_transforms,
                         batch_size, validation_ratio, train_validation_split_seed)


def build_fmnist_loaders(batch_size=128, validation_ratio=0.1, train_validation_split_seed=0):
    """
    Builds data loaders for fMNIST.
    :param batch_size:                  int, batch size
    :param validation_ratio:            float, validation size / full train set size.
        If zero, validation set is the full train set but without random transformations (e.g. crops)
    :param train_validation_split_seed: int, numpy random seed for train/validation split
    :return: torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader;
        train_loader, valid_loader, test_loader
    """
    return build_loaders(torchvision.datasets.FashionMNIST, build_mnist_transforms,
                         batch_size, validation_ratio, train_validation_split_seed)


def build_kmnist_loaders(batch_size=128, validation_ratio=0.1, train_validation_split_seed=0):
    """
    Builds data loaders for kMNIST.
    :param batch_size:                  int, batch size
    :param validation_ratio:            float, validation size / full train set size.
        If zero, validation set is the full train set but without random transformations (e.g. crops)
    :param train_validation_split_seed: int, numpy random seed for train/validation split
    :return: torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader;
        train_loader, valid_loader, test_loader
    """
    return build_loaders(torchvision.datasets.KMNIST, build_mnist_transforms,
                         batch_size, validation_ratio, train_validation_split_seed)


def build_loaders_by_dataset(dataset, batch_size=128, validation_ratio=0.1, train_validation_split_seed=0):
    """
    Builds loaders for a specific dataset
    :param dataset:                     str, 'CIFAR10', 'MNIST', 'fMNIST' (fashion-MNIST) or 'kMNIST' (Kuzushiji-MNIST)
    :param batch_size:                  int, batch size
    :param validation_ratio:            float, validation size / full train set size.
        If zero, validation set is the full train set but without random transformations (e.g. crops)
    :param train_validation_split_seed: int, numpy random seed for train/validation split
    :return: torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader;
        train_loader, valid_loader, test_loader
    """
    if dataset == 'CIFAR10':
        return build_cifar10_loaders(batch_size, validation_ratio, train_validation_split_seed)
    elif dataset == 'MNIST':
        return build_mnist_loaders(batch_size, validation_ratio, train_validation_split_seed)
    elif dataset == 'fMNIST':
        return build_fmnist_loaders(batch_size, validation_ratio, train_validation_split_seed)
    elif dataset == 'kMNIST':
        return build_kmnist_loaders(batch_size, validation_ratio, train_validation_split_seed)
    else:
        raise NotImplementedError('dataset must be either CIFAR10, or MNIST, or kMNIST, or fMNIST, '
                                  'but %s was given' % dataset)
