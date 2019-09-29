import os
import json
import torch
import copy
import types
import numpy as np
import argparse
from . import custom_models


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


def compute_parameter_distances(server_1, server_2):
    """

    normalized_distance = 0.5 ( var(x-y)/(var(x) + var(y))
    https://stackoverflow.com/questions/38161071/how-to-calculate-normalized-euclidean-distance-on-two-vectors

    :param server_1: of type nn.model (not src.server)
    :param server_2:
    :return:
    """
    normalized_distance = []    # First index is always all parameters stacked together distance. Then layer wise

    # for p, q in zip(server_1.parameters(), server_2.parameters()):
    #     r = p.data - q.data
    #
    #     normalized_distance.append(0.5 * r.view(1, -1).var()/(p.data.view(1, -1).var() + q.data.view(1, -1).var()))
    ll1 = list(server_1.parameters())
    ll2 = list(server_2.parameters())

    p = torch.cat([param.data.view(1, -1) for param in server_1.parameters()], dim=1)
    q = torch.cat([param.data.view(1, -1) for param in server_2.parameters()], dim=1)
    r = p - q
    dist = 0.5 * r.var() / (p.var() + q.var())
    normalized_distance.append(dist.item())

    for ind in range(len(ll1)//2):
        # weight
        # p = ll1[ind].view(1, -1)
        # q = ll2[ind].view(1, -1)
        p = torch.cat((ll1[ind].view(1, -1), ll1[2*ind+1].view(1, -1)), dim=1)
        q = torch.cat((ll2[ind].view(1, -1), ll2[2*ind+1].view(1, -1)), dim=1)

        r = p - q
        dist = 0.5 * r.var()/(p.var() + q.var())
        normalized_distance.append(dist.item())

    # normalized_distance = []    # layer wise
    # for p in server_1.parameters():
    #     normalized_distance.append(p.data.norm(2))
    #     # normalized_distance.append(server_1.compute_norm(p))

    return normalized_distance


def read_(root, exp_folder):

    conf_folder = root + exp_folder

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    server_folder_1 = conf_folder + "m1/"
    server_folder_2 = conf_folder + "m2/"

    with open(conf_folder + "config.json", "r") as f:
        config = json.load(f)

    server_1 = load_model(config['model'])
    server_2 = load_model(config['model'])
    logs = []
    for epoch_ in range(config["num_epochs"]):
        file_1 = server_folder_1 + 'model_checkpoints/epoch_' + str(epoch_) + '_checkpoint.pth.tar'
        file_2 = server_folder_2 + 'model_checkpoints/epoch_' + str(epoch_) + '_checkpoint.pth.tar'

        if os.path.isfile(file_1) and os.path.isfile(file_2):
            server_1_dict = torch.load(file_1, map_location=device)
            server_2_dict = torch.load(file_2, map_location=device)

            assert(epoch_ == server_1_dict['epoch'])
            assert(epoch_ == server_2_dict['epoch'])

            server_1.load_state_dict(server_1_dict['state_dict'], strict=True)
            server_2.load_state_dict(server_2_dict['state_dict'], strict=True)

            logs.append(compute_parameter_distances(server_1, server_2))

        print(epoch_)
    temp = np.asarray(logs)
    np.savetxt('plots/' + exp_folder + 'distance.csv', temp, delimiter=",")


parser = argparse.ArgumentParser(description="meta config of experiment")

parser.add_argument('-exp', default='pexp_2MNIST_lenet_64_const_decay/', type=str, metavar='data')
parser.add_argument('-folder', default='2019-05-30-22-07-46_50_0.1/', type=str, metavar='data')

args = parser.parse_args()

root = 'logs/'
# conf_folder = 'pexp_2MNIST_lenet_64_const_decay/2019-05-30-22-07-46_50_0.1/'
read_(root, args.exp + args.folder)