import argparse
import datetime
import json
import os
import plts
from src import server
import numpy as np

config = {
    'num_workers': 1,
    'num_epochs': 60,
    'dataset': 'MNIST',
    'model': 'lenet',
    'delay': 1,
    'delaytype': 'const',  # const or random
    'batch_size': 16,
    'seed': 0,
    'lr': 0.001,
    'lr_schedule': 'decay',
    'log': True,
    'print': True,
    'save_checkpoint': True,
    'print_freq': 500,
    'device': 'cuda:0',
    'lrdecay': 0.1
}

dataset_names = ['MNIST', 'CIFAR10']
delay_types = ['const', 'random']
model_names = ['lenet', 'vggsmall', 'lenetsmall', '3lenet', 'vggbig']

models = {}
models['MNIST'] = ['lenet', 'lenetsmall']
models['CIFAR10'] = ['vggsmall', '3lenet', 'vggbig']

delays = [1, 8, 16, 32, 64]
lrs = [0.05, 0.1, 0.2, 0.4]

parser = argparse.ArgumentParser(description="meta config of experiment")

parser.add_argument('-data', default='CIFAR10', type=str, metavar='data', choices=dataset_names)
parser.add_argument('-model', default='vggsmall', type=str, metavar='model', choices=model_names)
parser.add_argument('--num-epochs', default=150, type=int, metavar='N',
                    help='number of epochs')
parser.add_argument('--num-workers', default=4, type=int, metavar='W')
parser.add_argument('--batch-size', default=64, type=int, metavar='b', help='batch size per worker')
parser.add_argument('--delay-type', default='const', type=str, choices=delay_types)
parser.add_argument('--lr-schedule', default='decay', type=str, choices=['const', 'decay', 't'])
parser.add_argument('--lr-decay', default=0.05, type=float)
parser.add_argument('--cuda-device', default=0, type=int, metavar='c')
parser.add_argument('--print-freq', default=100, type=int, metavar='p')


args = parser.parse_args()

if __name__ == "__main__":

    print(args)

    logs_folder = 'logs/'
    plots_folder = 'plots/'
    exp_folder = 'pexp_' + str(args.num_workers) + args.data + "_" + args.model + "_" + str(args.batch_size) + "_" + args.delay_type + "_" + args.lr_schedule +"/"

    if not os.path.exists(logs_folder + exp_folder):
        os.makedirs(logs_folder + exp_folder)

    with open(logs_folder + exp_folder + "meta_config.json", 'w') as fp:
        json.dump(args.__dict__, fp, indent=2)

    config['dataset'] = args.data
    config['num_epochs'] = args.num_epochs
    config['num_workers'] = args.num_workers
    if args.model in models[args.data]:
        config['model'] = args.model
    else:
        print("model not implemented for this data set")

    config['batch_size'] = args.batch_size
    config['delaytype'] = args.delay_type
    config['lr_schedule'] = args.lr_schedule
    config['lrdecay'] = args.lr_decay
    config['device'] = 'cuda:' + str(args.cuda_device)
    config['print_freq'] = args.print_freq

    for delay in delays:
        worker_delays = []
        for id_ in range(args.num_workers):
            if id_ == 0:
                worker_delays.append(delay)
            elif id_ == 1:
                worker_delays.append(0)
            else:
                worker_delays.append(np.random.randint(0, 1 + delay))

        config['worker_delays'] = worker_delays

        for trial in range(3):
            for lr in lrs:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
                exp_config_name = "_" + str(delay) + "_" + str(lr) + "/"

                config['delay'] = delay
                config['lr'] = lr   # initial learning rate

                conf_folder = logs_folder + exp_folder + timestamp + exp_config_name
                conf_plots_folder = plots_folder + exp_folder + timestamp + exp_config_name

                config['folder_name'] = conf_folder
                config['plots_folder_name'] = conf_plots_folder

                if not os.path.exists(conf_folder):
                    os.makedirs(conf_folder)
                if not os.path.exists(conf_plots_folder):
                    os.makedirs(conf_plots_folder)

                with open(conf_folder+'config.json', 'w') as fp:
                    json.dump(config, fp)

                with open(conf_plots_folder+'config.json', 'w') as fp:
                    json.dump(config, fp)

                print(conf_folder)

                server.sync_servers(**config)

                print("###### COMPLETED ########")

