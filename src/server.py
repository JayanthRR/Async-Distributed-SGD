import pickle
import os, datetime, sys
import time
import copy
import random
import shutil
from random import Random, shuffle
import numpy as np
import types

import torch
from torch import nn

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Sampler

from . import custom_models


# random.seed(1234)


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


class SubsetSequentialSampler(Sampler):
    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        #         return (self.indices[i] for i in torch.randperm(len(self.indices)))
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


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
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '}) (({sum' + self.fmt + '}))'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, *meters, prefix=""):
        self.meters = meters
        self.prefix = prefix

    def print(self, epoch, batch):
        entries = [self.prefix + "[" + str(epoch) + str(", ") + str(batch) + "]"]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))


class Logger:
    def __init__(self, folder):
        self.logs = []
        self.folder = folder

    def update(self, vals):
        self.logs.append(vals)
        # self.logs.append([epoch, train_loss, train_accuracy, train_time, test_loss, test_accuracy, test_time])

    def save_log(self, filename):

        temp = np.asarray(self.logs)
        np.savetxt(filename, temp, delimiter=",")


def load_dataset(name, location, train=True):

    # Todo: custom for name in {'MNIST', 'CIFAR10', 'CIFAR100', ...}
    if name == 'CIFAR10':
        dataset = datasets.CIFAR10(
            location,
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

    if name == 'lenet':
        net = custom_models.LeNet()
    elif name == 'lenetsmall':
        net = custom_models.LeNetSmall()
    elif name == '3lenet':
        net = custom_models.LeNet3()
    elif name == 'vggsmall':
        net = custom_models.vgg(size='small')
    elif name == 'vggbig':
        net = custom_models.vgg(size='big')
    else:
        net = NotImplementedError

    return net


def save_checkpoint(state, is_best, foldername, epoch): # filename='checkpoint.pth.tar'):
    filename = foldername + "epoch_" + str(epoch) + '_checkpoint.pth.tar'
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, foldername + 'model_best.pth.tar')


def data_partition(num_workers, data_set):
    """
    generates a random shuffle of the size of the dataset, and returns the indices partitioned among the workers

    :param num_workers:
    :param data_set: type torch.data
    :param separate:
    :return:
    """

    size = data_set.data.shape[0]
    ind = list(range(size))

    shuffle(ind)
    # worker_size is the number of samples per worker. The last worker either receives the additional samples
    # or the last samples are dropped.
    worker_size = size // num_workers
    data = dict.fromkeys(list(range(num_workers)))

    for w in range(num_workers):
        if w is not num_workers - 1:
            data[w] = ind[w * worker_size: (w+1) * worker_size]
        else:
            # drop last
            data[w] = ind[w * worker_size:(w+1) * worker_size]

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


class ParameterServer:
    def __init__(self, **kwargs):

        # server worker related parameters
        # location, foldername added new as compared to ParameterServer
        self.max_delay = kwargs['delay']
        self.queue = Queue(self.max_delay + 1)
        self.num_workers = kwargs['num_workers']
        self.workers = []
        self.batch_size = kwargs['batch_size']
        self.inf_batch_size = 10000
        self.delaytype = kwargs['delaytype']

        # data loading
        self.dataset = kwargs['dataset']
        if 'location_' not in kwargs.keys():
            location = 'data/' + self.dataset
        else:
            location = kwargs['location_']

        self.train_data = load_dataset(self.dataset, location)
        self.test_data = load_dataset(self.dataset, location, train=False)

        self.train_loader = DataLoader(self.train_data, batch_size=10000, num_workers=8)
        self.test_loader = DataLoader(self.test_data, batch_size=10000, num_workers=8)

        self.partitions = {}

        # choosing model and loss function
        if kwargs['model']:
            self.model = load_model(name=kwargs['model'])
        else:
            if self.dataset == 'MNIST':
                self.model = load_model(name='lenet')
            else:
                raise NotImplementedError
                # self.model = load_model()

        self.loss_fn = nn.CrossEntropyLoss()

        # device to use
        if torch.cuda.is_available():
            if kwargs['device']:
                self.device = torch.device(kwargs['device'])
            else:
                self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')

        # Training related
        self.num_epochs = kwargs['num_epochs']
        self.epoch = 0
        self.init_lr = kwargs['lr']
        self.lr_schedule = kwargs['lr_schedule']
        self.lr = self.init_lr
        self.decay = kwargs['lrdecay']

        # Print related
        self.print_freq = kwargs['print_freq']   # iterations (not epochs)
        self.loss_meter = AverageMeter("loss:", ":.4e")
        self.time_meter = AverageMeter('time:', ":6.3f")
        self.compute_gradients_time_meter = AverageMeter('grads time:', "6.3f")
        self.aggregate_gradients_time_meter = AverageMeter('aggr time:', "6.3f")
        self.progress_meter = ProgressMeter(self.loss_meter, self.time_meter)

        # Logging related
        self.folder_name = kwargs['folder_name_']
        self.model_checkpoints_folder = self.folder_name + "model_checkpoints/"
        os.makedirs(self.model_checkpoints_folder)

        self.logger = Logger(self.folder_name)
        self.delaylogger = Logger(self.folder_name)     # each column corresponds to each worker
        self.grad_norm_logger = Logger(self.folder_name)    # each column corresponds to each worker
        self.loss_logger = Logger(self.folder_name)     # each column corresponds to each worker
        self.lr_logger = Logger(self.folder_name)       # logs lr, norm of weights

        self.save_checkpoint = kwargs['save_checkpoint']
        self.log = kwargs['log']

        # Functions to be called at init
        # self.initiate_workers(self.dataset, kwargs['model'])
        self.update_queue()

    def initiate_workers(self, model):

        for id_ in range(self.num_workers):

            loader = DataLoader(self.train_data, batch_size=self.batch_size,
                                sampler=SubsetSequentialSampler(self.partitions[id_]),
                                num_workers=2)

            self.workers.append(Worker(self, id_,
                                       device=self.device, loader=loader,
                                       batch_size=self.batch_size, model=model,
                                       delaytype=self.delaytype))

    def update_queue(self, reset=False):
        if reset:
            self.queue.queue = []
            self.queue.len = 0
            self.queue.push([param.clone().detach().requires_grad_().cpu() for param in self.model.parameters()])
        else:
            self.queue.push([param.clone().detach().requires_grad_().cpu() for param in self.model.parameters()])

    def nan_handler(self, msg):

        self.save_logs()
        self.save_model()
        raise SystemExit('Nan value encountered')

    def lr_update(self, itr, epoch):
        if self.lr_schedule == 'const':
            self.lr = self.init_lr
        elif self.lr_schedule == 'decay':
            # do not use this currently. lr_update is being called after each step, so it leads to lr->0.
            self.lr = self.init_lr/(1 + self.decay * self.epoch)
        elif self.lr_schedule == 't':
            # lr = c/t
            self.lr = self.init_lr / (1 + self.epoch)

        return

    def compute_norm(self, parameters):
        total_norm = 0
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]

        for p in parameters:
            param_norm = p.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm

    def clip_(self, parameters, max_norm, inplace=True):
        """
        If inplace, parameters gets changed. Else, a deepcopy of the parameters is made and which is updated and returned
        Either ways, parameters or cp is returned.
        # Note: Do not pass a generator object for parameters. Always pass a list.

        :param parameters:
        :param max_norm:
        :param inplace:
        :return:
        """

        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]

        total_norm = self.compute_norm(parameters)
        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            if inplace:
                for p in parameters:
                    p.data.mul_(clip_coef)
                return total_norm, parameters
            else:
                if isinstance(parameters, types.GeneratorType):
                    parameters = list(parameters)

                cp = copy.deepcopy(parameters)
                for p in cp:
                    p.data.mul_(clip_coef)
                return total_norm, cp
        return total_norm, parameters

    def train(self):

        best_acc = 0
        num_iter_per_epoch = len(self.partitions[0])//self.batch_size + 1
        running_itr = 0

        for epoch in range(self.num_epochs):

            self.epoch = epoch
            self.loss_meter.reset()
            self.time_meter.reset()

            for itr in range(num_iter_per_epoch):

                start_time = time.time()

                try:
                    self.step()
                except Exception as e:
                    # propagating exception
                    print('exception in step ', e)
                    raise e

                step_time = time.time() - start_time

                running_itr += 1
                self.lr_update(running_itr, epoch)

                # Keeps track of average time per iter, total time taken in epoch
                self.time_meter.update(step_time, 1)

                if itr % self.print_freq == self.print_freq - 1:
                    self.progress_meter.print(epoch, itr)

            self.progress_meter.print(epoch, itr)

            # Todo: Write a function that tracks progress, tensorboard integration (SummaryWriter)

            print("epoch train loss")
            start_time = time.time()
            train_loss, train_acc = self.inference(test=False)
            train_time = time.time() - start_time
            print("train time", train_time)

            print("epoch test loss")
            start_time = time.time()
            test_loss, test_acc = self.inference(test=True)
            test_time = time.time() - start_time
            print("test time", test_time)

            is_best = (test_acc > best_acc)
            best_acc = max(test_acc, best_acc)

            self.logger.update([epoch, train_loss, train_acc, train_time, test_loss, test_acc, test_time])

            # If flag is true, save every epoch's model. Else only save the final model
            if self.save_checkpoint:
                self.save_model(is_best)
            else:
                if epoch == self.num_epochs - 1:
                    self.save_model(is_best=True)

        self.save_logs()
        print("training completed")

    def step(self):

        grads = {}
        loss = 0
        batch_size = 0

        delays = []
        grad_norms = []
        losses = []
        start_time = time.time()

        for worker_ in self.workers:

            # worker_.get_server_weights()
            start_time = time.time()

            grads[worker_.id], worker_loss, batch_size_ = worker_.compute_gradients()

            self.compute_gradients_time_meter.update(time.time() - start_time, 1)
            start_time = time.time()

            # Check for nan values and exit the program. Also sends an email
            for wg in grads[worker_.id]:
                if torch.isnan(wg).any() or torch.isinf(wg).any():
                    # self.nan_handler(msg=str(worker_.id) + 'grad')
                    raise Exception('found Nan/Inf values')

            if torch.isnan(worker_loss) or torch.isinf(worker_loss).any():
                # self.nan_handler(msg=str(worker_.id) + 'loss')
                raise Exception('found Nan/Inf values')

            batch_size += batch_size_
            loss += worker_loss * batch_size_

            losses.append(worker_loss * batch_size)
            grad_norms.append(self.compute_norm(grads[worker_.id]))
            delays.append(worker_.delay)
            # Todo: Log worker's statistics (run time, loss, accuracy, model parameters at the end of epoch

        self.grad_norm_logger.update(grad_norms)
        self.loss_logger.update(losses)
        self.delaylogger.update(delays)
        self.aggregate_gradients(grads)
        self.aggregate_gradients_time_meter.update(time.time() - start_time, 1)

        loss /= batch_size
        self.loss_meter.update(loss.data, batch_size)

    def aggregate_gradients(self, grads):

        # average gradients across all workers (includes cached gradients)

        for id_ in range(1, len(grads)):
            for param1, param2 in zip(grads[0], grads[id_]):
                param1.data += param2.data

        for param in grads[0]:
            param.data /= len(grads)

        # norm_before_clip = nn.utils.clip_grad_norm_(grads[0], 1)
        # norm_before_clip = self.compute_norm(grads[0])
        norm_before_clip, _ = self.clip_(grads[0], max_norm=1, inplace=True)
        norm_after_clip = self.compute_norm(grads[0])

        # Assign grad data to model grad data. Update parameters of the model
        for param1, param2 in zip(self.model.parameters(), grads[0]):
            param1.data -= self.lr * param2.data

        self.lr_logger.update([self.lr, self.compute_norm(self.model.parameters()), norm_before_clip, norm_after_clip])
        self.update_queue()

    def inference(self, test=True):

        self.model.to(self.device)
        correct = 0
        total = 0
        loss = 0

        with torch.no_grad():

            if test:
                for data, labels in self.test_loader:
                    data, labels = data.to(self.device), labels.to(self.device)
                    outputs = self.model(data)
                    loss += self.loss_fn(outputs, labels) * labels.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            else:
                for data, labels in self.train_loader:
                    data, labels = data.to(self.device), labels.to(self.device)
                    outputs = self.model(data)
                    loss += self.loss_fn(outputs, labels) * labels.size(0)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

        total_loss = loss/total
        total_correct = correct/total
        print('%d images- Accuracy: %d %%, Loss: %6.3f ' % (
            total, 100 * total_correct, total_loss))
        self.model.cpu()

        return total_loss.cpu(), total_correct

    def save_model(self, is_best=False):

        state = {
            'epoch': self.epoch,
            'losses': self.loss_logger,
            'stats' : self.logger,
            'grad_norm' : self.grad_norm_logger,
            'state_dict': self.model.state_dict(),
        }
        save_checkpoint(state, is_best=is_best, foldername=self.model_checkpoints_folder, epoch=self.epoch)

    def save_logs(self, folder_name=None):
        if not folder_name:
            folder_name = self.folder_name

        self.logger.save_log(folder_name + "stats.csv")
        self.delaylogger.save_log(folder_name + "delays.csv")
        self.grad_norm_logger.save_log(folder_name + "norms.csv")
        self.loss_logger.save_log(folder_name + 'losses.csv')
        self.lr_logger.save_log(folder_name + 'lr.csv')


class Worker:
    def __init__(self, *args, **kwargs):

        self.parent = args[0]
        self.id = args[1]
        self.loader = kwargs['loader']
        self.generator = enumerate(self.loader)

        if kwargs['model']:
            self.model = load_model(name=kwargs['model'])
        else:
            raise NotImplementedError

        self.batch_size = kwargs['batch_size']
        self.delaytype = kwargs['delaytype']
        self.delay = kwargs['delay']

        self.loss_function = nn.CrossEntropyLoss()
        self.epoch = 0
        self.device = kwargs['device']
        self.model.to(self.device)

        self.data_loading_time_meter = AverageMeter('data time:', ":6.3f")
        self.model_loading_time_meter = AverageMeter('model time:', ":6.3f")
        self.nn_time_meter = AverageMeter('nn time', ":6.3f")
        self.progress_meter = ProgressMeter(self.model_loading_time_meter, self.data_loading_time_meter,
                                            self.nn_time_meter,
                                            prefix='worker:')

    def get_next_mini_batch(self):

        try:
            _, (data, labels) = next(self.generator)
        except StopIteration:
            self.generator = enumerate(self.loader)
            _, (data, labels) = next(self.generator)

        return data.to(self.device), labels.to(self.device)

    def get_server_weights(self):

        params = self.parent.queue.sample(self.delay)
        for param_1, param_2 in zip(self.model.parameters(), params):
            param_1.data = param_2.clone().detach().requires_grad_().data.to(self.device)
        del params

    def assign_weights(self, model):
        """
        Takes in a model and assigns the weights of the model to self.model.
        Skipping the check for model and self.model belonging to the same nn.Module type.

        :param model:
        :return:
        """
        for param_1, param_2 in zip(self.model.parameters(), model.parameters()):
            param_1.data = param_2.data.to(self.device)

    def compute_gradients(self):

        start_time = time.time()
        self.get_server_weights()
        self.model_loading_time_meter.update(time.time() - start_time, 1)
        start_time = time.time()

        batchdata, batchlabels = self.get_next_mini_batch()     # passes to device already

        self.data_loading_time_meter.update(time.time() - start_time, 1)

        start_time = time.time()

        output = self.model.forward(batchdata)

        loss = self.loss_function(output, batchlabels)
        self.model.zero_grad()
        loss.backward()

        # compute the gradients and return the list of gradients

        self.nn_time_meter.update(time.time() - start_time, 1)

        return [param.grad.data.cpu() for param in self.model.parameters()], loss.data, batchlabels.size(0)


def shared_randomness_partitions(n, num_workers):

    # remove last data point
    dinds1 = list(range(n-1))
    dinds2 = list(range(n-1))

    k = random.randint(0, n-1)

    dinds2[k] = n-1

    inds = list(range(n-1))
    shuffle(inds)

    dinds1 = [dinds1[i] for i in inds]
    dinds2 = [dinds2[i] for i in inds]

    worker_size = (n-1) // num_workers

    data_1 = dict.fromkeys(list(range(num_workers)))
    data_2 = dict.fromkeys(list(range(num_workers)))

    for w in range(num_workers):
        if w is not num_workers - 1:
            data_1[w] = dinds1[w * worker_size: (w+1) * worker_size]
            data_2[w] = dinds2[w * worker_size: (w+1) * worker_size]
        else:
            data_1[w] = dinds1[w * worker_size: (w+1) * worker_size]
            data_2[w] = dinds2[w * worker_size: (w+1) * worker_size]

    return data_1, data_2


def sync_data_loader(server_1, server_2):

    for id_ in range(server_1.num_workers):

        inds = list(range(len(server_1.partitions[id_])))
        shuffle(inds)

        server_1.partitions[id_] = [server_1.partitions[id_][i] for i in inds]
        server_2.partitions[id_] = [server_2.partitions[id_][i] for i in inds]

        server_1.workers[id_].loader = DataLoader(server_1.train_data, batch_size=server_1.batch_size,
                                                  sampler=SubsetSequentialSampler(server_1.partitions[id_]),
                                                  num_workers=0)

        server_2.workers[id_].loader = DataLoader(server_2.train_data, batch_size=server_1.batch_size,
                                                  sampler=SubsetSequentialSampler(server_2.partitions[id_]),
                                                  num_workers=0)

        server_1.workers[id_].generator = enumerate(server_1.workers[id_].loader)
        server_2.workers[id_].generator = enumerate(server_2.workers[id_].loader)


def compute_parameter_distances(server_1, server_2):
    normalized_distance = []  # First index is always all parameters stacked together distance. Then layer wise

    ll1 = [w.clone().detach() for w in server_1.model.parameters()]
    ll2 = [w.clone().detach() for w in server_2.model.parameters()]

    p = torch.cat([param.data.view(1, -1) for param in ll1], dim=1)
    q = torch.cat([param.data.view(1, -1) for param in ll2], dim=1)
    r = p - q
    # dist = 0.5 * r.var() / (p.var() + q.var())
    # print(p.shape, q.shape)
    dist = np.sqrt(torch.norm(r)**2/(torch.norm(p)**2 + torch.norm(q)**2))

    normalized_distance.append(dist.item())

    for ind in range(len(ll1) // 2):
        p = torch.cat((ll1[ind].view(1, -1), ll1[2 * ind + 1].view(1, -1)), dim=1)
        q = torch.cat((ll2[ind].view(1, -1), ll2[2 * ind + 1].view(1, -1)), dim=1)

        r = p - q
        # dist = 0.5 * r.var() / (p.var() + q.var())
        dist = np.sqrt(torch.norm(r) ** 2 / (torch.norm(p)**2 + torch.norm(q)**2))

        normalized_distance.append(dist.item())

    return normalized_distance


def sync_train(server_1, server_2, logger):

    best_acc_1 = 0
    best_acc_2 = 0
    num_iter_per_epoch = len(server_1.partitions[0]) // server_1.batch_size + 1
    running_itr = 0

    logger.update(compute_parameter_distances(server_1, server_2))

    for epoch in range(server_1.num_epochs):

        server_1.epoch = epoch
        server_2.epoch = epoch

        server_1.loss_meter.reset()
        server_1.time_meter.reset()
        server_2.loss_meter.reset()
        server_2.time_meter.reset()

        for itr in range(num_iter_per_epoch):

            start_time = time.time()
            try:
                server_1.step()
            except Exception as e:
                # propagating exception
                print('exception in server 1 step ', e)
                raise e
            step_time = time.time() - start_time
            # Keeps track of average time per iter, total time taken in epoch
            server_1.time_meter.update(step_time, 1)

            start_time = time.time()
            try:
                server_2.step()
            except Exception as e:
                # propagating exception
                print('exception in server 2 step ', e)
                raise e
            step_time = time.time() - start_time
            # Keeps track of average time per iter, total time taken in epoch
            server_2.time_meter.update(step_time, 1)

            running_itr += 1
            server_1.lr_update(running_itr, epoch)
            server_2.lr_update(running_itr, epoch)

            if itr % server_1.print_freq == server_1.print_freq - 1:
                server_1.progress_meter.print(epoch, itr)
                server_2.progress_meter.print(epoch, itr)

        server_1.progress_meter.print(epoch, itr)
        server_2.progress_meter.print(epoch, itr)

        # At the end of the epoch, synchronize the data loaders on both the servers
        sync_data_loader(server_1, server_2)

        # Compute the normalized parameter distance at the end of each epoch
        normalized_distance = compute_parameter_distances(server_1, server_2)
        # Logging
        logger.update(normalized_distance)

        print("epoch train loss")
        start_time = time.time()
        train_loss_1, train_acc_1 = server_1.inference(test=False)
        train_time_1 = time.time() - start_time
        print("train time 1", train_time_1)

        print("epoch train loss")
        start_time = time.time()
        train_loss_2, train_acc_2 = server_2.inference(test=False)
        train_time_2 = time.time() - start_time
        print("train time 2", train_time_2)

        print("epoch test loss")
        start_time = time.time()
        test_loss_1, test_acc_1 = server_1.inference(test=True)
        test_time_1 = time.time() - start_time
        print("test time", test_time_1)

        print("epoch test loss")
        start_time = time.time()
        test_loss_2, test_acc_2 = server_2.inference(test=True)
        test_time_2 = time.time() - start_time
        print("test time", test_time_2)

        is_best_1 = (test_acc_1 > best_acc_1)
        best_acc_1 = max(test_acc_1, best_acc_1)

        is_best_2 = (test_acc_2 > best_acc_2)
        best_acc_2 = max(test_acc_2, best_acc_2)

        server_1.logger.update([epoch, train_loss_1, train_acc_1, train_time_1, test_loss_1, test_acc_1, test_time_1])
        server_2.logger.update([epoch, train_loss_2, train_acc_2, train_time_2, test_loss_2, test_acc_2, test_time_2])

        # If flag is true, save every epoch's model. Else only save the final model
        if server_1.save_checkpoint:
            server_1.save_model(is_best_1)
            server_2.save_model(is_best_2)
        else:
            if epoch == server_1.num_epochs - 1:
                server_1.save_model(is_best=True)
                server_2.save_model(is_best=True)

    server_1.save_logs()
    server_2.save_logs()
    logger.save_log(logger.folder + 'distances.csv')
    print("training completed")


def sync_servers(**kwargs):

    folder_name = kwargs['folder_name']
    num_workers = kwargs['num_workers']
    batch_size = kwargs['batch_size']
    model = kwargs['model']
    plots_folder = kwargs['plots_folder_name']

    param_weights_logger = Logger(folder_name)

    if not os.path.exists(folder_name + 'm1/'):
        os.makedirs(folder_name + 'm1/')

    if not os.path.exists(plots_folder + 'm1/'):
        os.makedirs(plots_folder + 'm1/')

    server_1 = ParameterServer(folder_name=folder_name + 'm1/',
                               location='data/' + kwargs['dataset'] + '_1',
                               **kwargs)

    if not os.path.exists(folder_name + 'm2/'):
        os.makedirs(folder_name + 'm2/')

    if not os.path.exists(plots_folder + 'm2/'):
        os.makedirs(plots_folder + 'm2/')

    server_2 = ParameterServer(folder_name=folder_name + 'm2/',
                               location='data/' + kwargs['dataset'] + '_2',
                               **kwargs)

    cloned_weights = [w.clone().detach().requires_grad_() for w in server_1.model.parameters()]
    for p, q in zip(server_2.model.parameters(), cloned_weights):
        p.data = q.data
    del cloned_weights

    server_2.update_queue(reset=True)

    partitions_1, partitions_2 = shared_randomness_partitions(len(server_1.train_data), num_workers)
    server_1.partitions = partitions_1
    server_2.partitions = partitions_2

    # init workers together
    # currently only handles const delay
    # if not (kwargs['delaytype'] == 'const'):
    #     raise Exception('only constant delays accepted')

    # generate delays
    delays = kwargs['worker_delays']
    # for id_ in range(num_workers):
    #     if id_ == 0:
    #         delays.append(kwargs['delay'])
    #     elif id_ == 1:
    #         delays.append(0)
    #     else:
    #         delays.append(np.random.randint(0, 1 + kwargs['delay']))

    # initialize workers on each server
    for id_ in range(num_workers):
        loader_1 = DataLoader(server_1.train_data, batch_size=batch_size,
                              sampler=SubsetSequentialSampler(server_1.partitions[id_]),
                              num_workers=0)

        loader_2 = DataLoader(server_2.train_data, batch_size=batch_size,
                              sampler=SubsetSequentialSampler(server_2.partitions[id_]),
                              num_workers=0)

        server_1.workers.append(Worker(server_1, id_,
                                       device=server_1.device, loader=loader_1,
                                       batch_size=batch_size, model=model,
                                       delaytype=server_1.delaytype, delay=delays[id_]))

        server_2.workers.append(Worker(server_2, id_,
                                       device=server_2.device, loader=loader_2,
                                       batch_size=batch_size, model=model,
                                       delaytype=server_2.delaytype, delay=delays[id_]))

    print('Start training')
    sync_train(server_1, server_2, param_weights_logger)
    server_1.save_logs(plots_folder+'m1/')
    server_2.save_logs(plots_folder+'m2/')
    param_weights_logger.save_log(plots_folder+'distances.csv')


if __name__ == '__main__':

    folder = "logs/" + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "/"

    if not os.path.exists(folder):
        os.makedirs(folder)

    for dataset in ['MNIST']:
        for delay in [1]:
            folder_name = folder + dataset + "_" + str(delay)
            p = ParameterServer(num_workers=4, batch_size=16, num_epochs=30, dataset=dataset, delay=delay)

            print("######## ", dataset, " ##### ", str(delay))
            # p.train()

