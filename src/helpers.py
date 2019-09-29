import random
import shutil
from random import Random, shuffle
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import datasets, transforms


random.seed(1234)


class Queue:
    def __init__(self, max_len):
        self.queue = list()
        self.maxlen = max_len
        self.len = 0

    def push(self, grad):

        if self.len < self.maxlen:
            self.queue.append(grad)
            self.len += 1
        else:
            ret = self.queue.pop(0)
            self.queue.append(grad)

    def sample(self, delay):

        if delay >= self.len:
            return self.queue[0]
        # print(delay)
        # i-th element in the queue is the i step delayed version of the param
        return self.queue[self.len - delay - 1]


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.classifier = nn.Sequential(
            nn.Linear(256, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10))

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def load_dataset(name, train=True):

    # Todo: custom for name in {'MNIST', 'CIFAR10', 'CIFAR100', ...}
    if name == 'CIFAR10':
        dataset = datasets.CIFAR10(
            'data/CIFAR10',
            train=train,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    else:
        dataset = datasets.MNIST(
            'data/MNIST',
            train=train,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5,), (0.5,))]))

    return dataset


def load_model(name=None):

    if name=='lenet':
        # print("loading lenet")
        net = LeNet()
    else:
        # print("loading net")
        net = Net()
    # for param in net.parameters():
    #     param.requires_grad = True

    return net


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def data_partition(num_workers, data_set, separate=True):
    """
    generates a random shuffle of the size of the dataset, and returns the indices partitioned among the workers

    :param num_workers:
    :param data_set: type torch.data
    :param separate:
    :return:
    """

    size = data_set.data.shape[0]
    ind = list(range(size))

    if separate:
        shuffle(ind)
        # worker_size is the number of samples per worker. The last worker however receives the additional samples
        worker_size = size // num_workers
        data = dict.fromkeys(list(range(num_workers)))

        for w in range(num_workers):
            if w is not num_workers - 1:
                data[w] = ind[w * worker_size: (w+1) * worker_size]
                # data[w]["X"] = X_train[ind[w * worker_size: (w + 1) * worker_size], :]
                # data[w]["Y"] = Y_train[ind[w * worker_size: (w + 1) * worker_size], :]
            else:
                data[w] = ind[w * worker_size:]
                # data[w]["X"] = X_train[ind[w * worker_size:], :]
                # data[w]["Y"] = Y_train[ind[w * worker_size:], :]

    else:
        data = dict.fromkeys(list(range(num_workers)))
        for w in range(num_workers):
            shuffle(ind)
            data[w] = ind
            # data[w]["X"] = X_train[ind, :]
            # data[w]["Y"] = Y_train[ind, :]

    return data


def load_data_from_inds(data_set, inds):

    """
    size= len(inds)
    returns data of dim size x dim of one data point, labels of dim size x 1
    :param data_set: type torch.utils.data
    :param inds: list of indices
    :return:
    """

    data = torch.cat([data_set[ind_][0].unsqueeze_(0) for ind_ in inds], 0)
    labels = torch.cat([torch.from_numpy(np.array(data_set[ind_][1])).unsqueeze_(0) for ind_ in inds], 0)

    return data, labels

