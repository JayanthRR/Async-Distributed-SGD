import numpy as np
import matplotlib
# matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import os
import shutil
from matplotlib import pyplot as plt
import json
import glob
import matplotlib.colors as mcolors
from itertools import cycle

color_names = mcolors.TABLEAU_COLORS


def plot_configs_pexp(exp_folder, fig_folder='plots_gen', num_epochs=50):

    conf_folders = glob.glob(exp_folder + "*/")

    fig_folder = exp_folder + fig_folder + str(num_epochs) + "/"

    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)

    plt_stats = {}
    lrs = set()
    delays = set()

    fig1, ax1 = plt.subplots(1, 2)
    for conf_folder in conf_folders:

        if conf_folder[:4] == 'plot':
            continue

        try:
            with open(conf_folder + 'config.json', 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            continue

        for m in ['m1/', 'm2/']:
            filename = conf_folder + m + 'stats.csv'
            if os.path.isfile(filename):
                stats = np.genfromtxt(filename, delimiter=',')
                if stats.shape[0] == 0:
                    print(conf_folder)
                    shutil.rmtree(path=conf_folder)
                elif stats.shape[0] >=num_epochs:
                    stats = np.insert(stats, 0, np.zeros((1, stats.shape[1])), 0)

                    lr = config['lr']
                    delay = config['delay']
                    lrs.add(lr)
                    delays.add(delay)

                    if lr in plt_stats.keys():
                        if delay not in plt_stats[lr].keys():
                            plt_stats[lr][delay] = [stats]
                        else:
                            plt_stats[lr][delay].append(stats)
                    else:
                        plt_stats[lr] = {}
                        plt_stats[lr][delay] = [stats]

                    ax1[0].plot(stats[:, 0], abs(stats[:, 1] - stats[:, 4]), label=str(lr) + ',' + str(delay))
                    ax1[1].plot(stats[:, 0], abs(stats[:, 2] - stats[:, 5]), label=str(lr) + ',' + str(delay))
                else:
                    pass

    ax1[0].legend()
    ax1[1].legend()
    ax1[0].set_xlabel('epochs')
    ax1[0].set_ylabel('abs(train loss - test loss)')

    fig1.savefig(fig_folder + 'plot.png')
    plt.close(fig1)

    #####################################################
    # Plot for specific delay or specific lr
    print(lrs, delays)
    # plot wrt given lr

    for lr in sorted(lrs):
        fig1, ax1 = plt.subplots(1, 1)
        fig2, ax2 = plt.subplots(1, 1)

        mctk = list(color_names.keys())
        mctkl = cycle(mctk)

        for delay in sorted(plt_stats[lr].keys()):
            stats_list = plt_stats[lr][delay]
            gen_error = []
            gen_acc = []
            for stat in stats_list:
                gen_error.append(abs(stat[:num_epochs, 1]-stat[:num_epochs, 4]))
                gen_acc.append(abs(stat[:num_epochs, 2] - stat[:num_epochs, 5]))

            err_mean = np.mean(np.array(gen_error), axis=0)
            err_std = np.std(np.array(gen_error), axis=0, ddof=0)

            acc_mean = np.mean(np.array(gen_acc), axis=0)
            acc_std = np.std(np.array(gen_acc), axis=0, ddof=0)

            c_color = color_names[next(mctkl)]

            ax1.plot(stat[:num_epochs, 0], err_mean, label=str(delay), color=c_color)
            ax1.fill_between(stat[:num_epochs, 0], err_mean + err_std, err_mean - err_std, alpha=0.5)

            ax2.plot(stat[:num_epochs, 0], acc_mean, label=str(delay), color=c_color)
            ax2.fill_between(stat[:num_epochs, 0], acc_mean + acc_std, acc_mean - acc_std, alpha=0.5)

        handles, labels = ax1.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: float(t[0])))
        ax1.legend(handles, labels)
        del handles, labels

        handles, labels = ax2.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: float(t[0])))
        ax2.legend(handles, labels)
        del handles, labels

        ax1.legend()
        ax2.legend()
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('abs(train loss - test loss)')

        ax2.set_xlabel('epochs')
        ax2.set_ylabel('abs(train error - test error)')

        fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig1.savefig(fig_folder + 'loss_lr_' + str(lr) + '.png')
        fig1.savefig(fig_folder + 'loss_lr_' + str(lr) + '.pdf')

        fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig2.savefig(fig_folder + 'error_lr_' + str(lr) + '.png')
        fig2.savefig(fig_folder + 'error_lr_' + str(lr) + '.pdf')

        plt.close(fig1)
        plt.close(fig2)

    for delay in sorted(delays):
        fig1, ax1 = plt.subplots(1, 1)
        fig2, ax2 = plt.subplots(1, 1)

        mctk = list(color_names.keys())
        mctkl = cycle(mctk)

        for lr in sorted(lrs):
            if delay in plt_stats[lr].keys():
                stats_list = plt_stats[lr][delay]

                gen_error = []
                gen_acc = []
                for stat in stats_list:
                    gen_error.append(abs(stat[:num_epochs, 1]-stat[:num_epochs, 4]))
                    gen_acc.append(abs(stat[:num_epochs, 2] - stat[:num_epochs, 5]))

                c_color = color_names[next(mctkl)]

                err_mean = np.mean(np.array(gen_error), axis=0)
                err_std = np.std(np.array(gen_error), axis=0, ddof=0)

                acc_mean = np.mean(np.array(gen_acc), axis=0)
                acc_std = np.std(np.array(gen_acc), axis=0, ddof=0)

                ax2.plot(stat[:num_epochs, 0], np.mean(np.array(gen_acc), axis=0), label=str(lr), color=c_color)
                ax2.fill_between(stat[:num_epochs, 0], acc_mean+acc_std, abs(acc_mean-acc_std), alpha=0.5)
                # ax2.errorbar(stat[:num_epochs, 0], acc_mean, yerr=acc_std, label=str(lr))

                ax1.plot(stat[:num_epochs, 0], err_mean, label=str(lr), color=c_color)
                ax1.fill_between(stat[:num_epochs, 0], err_mean+err_std, abs(err_mean-err_std), alpha=0.5)

        handles, labels = ax1.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: float(t[0])))
        # print(labels)
        ax1.legend(handles, labels)
        del handles, labels

        handles, labels = ax2.get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: float(t[0])))
        # print(labels)
        ax2.legend(handles, labels)
        del handles, labels

        ax1.legend()
        ax2.legend()
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('abs(train loss - test loss)')

        ax2.set_xlabel('epochs')
        ax2.set_ylabel('abs(train error - test error)')

        fig1.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig1.savefig(fig_folder + 'loss_delay_' + str(delay) + '.png')
        fig1.savefig(fig_folder + 'loss_delay_' + str(delay) + '.pdf')
        plt.close(fig1)

        fig2.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig2.savefig(fig_folder + 'error_delay_' + str(delay) + '.png')
        fig2.savefig(fig_folder + 'error_delay_' + str(delay) + '.pdf')
        plt.close(fig2)


def plot_configs_pexp_acc(exp_folder, fig_folder='plots_acc', num_epochs=50):

    conf_folders = glob.glob(exp_folder + "*/")

    fig_folder = exp_folder + fig_folder + str(num_epochs) + "/"

    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)

    plt_stats = {}
    lrs = set()
    delays = set()

    # fig1, ax1 = plt.subplots(1, 2)
    for conf_folder in conf_folders:

        if conf_folder[:4] == 'plot':
            continue

        try:
            with open(conf_folder + 'config.json', 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            continue

        for m in ['m1/', 'm2/']:
            filename = conf_folder + m + 'stats.csv'
            if os.path.isfile(filename):
                stats = np.genfromtxt(filename, delimiter=',')
                if stats.shape[0] == 0:
                    print(conf_folder)
                    shutil.rmtree(path=conf_folder)
                elif stats.shape[0] >=num_epochs:
                    stats = np.insert(stats, 0, np.zeros((1, stats.shape[1])), 0)

                    lr = config['lr']
                    delay = config['delay']
                    lrs.add(lr)
                    delays.add(delay)

                    if lr in plt_stats.keys():
                        if delay not in plt_stats[lr].keys():
                            plt_stats[lr][delay] = [stats]
                        else:
                            plt_stats[lr][delay].append(stats)
                    else:
                        plt_stats[lr] = {}
                        plt_stats[lr][delay] = [stats]

                else:
                    pass

    #####################################################
    # Plot for specific delay or specific lr
    print(lrs, delays)
    # plot wrt given lr

    for lr in sorted(lrs):

        mctk = list(color_names.keys())
        mctkl = cycle(mctk)

        fig, ax = plt.subplots(2, 2)
        for delay in sorted(plt_stats[lr].keys()):
            stats_list = plt_stats[lr][delay]

            train_loss = []
            train_acc = []
            test_loss = []
            test_acc = []

            for stat in stats_list:

                train_loss.append(stat[:num_epochs, 1])
                train_acc.append(stat[:num_epochs, 2])
                test_loss.append(stat[:num_epochs, 4])
                test_acc.append(stat[:num_epochs, 5])

            c_color = color_names[next(mctkl)]

            train_loss_mean = np.mean(np.array(train_loss), axis=0)
            train_loss_std = np.std(np.array(train_loss), axis=0, ddof=0)

            train_acc_mean = np.mean(np.array(train_acc), axis=0)
            train_acc_std = np.std(np.array(train_acc), axis=0, ddof=0)

            test_loss_mean = np.mean(np.array(test_loss), axis=0)
            test_loss_std = np.std(np.array(test_loss), axis=0, ddof=0)

            test_acc_mean = np.mean(np.array(test_acc), axis=0)
            test_acc_std = np.std(np.array(test_acc), axis=0, ddof=0)
            # print(train_loss.shape, stat[])

            ax[0][0].plot(stat[:num_epochs, 0], train_loss_mean, label=str(delay), color=c_color)
            ax[0][0].fill_between(stat[:num_epochs, 0], train_loss_mean + train_loss_std,
                                  train_loss_mean - train_loss_std, alpha=0.4)

            ax[0][1].plot(stat[:num_epochs, 0], train_acc_mean, label=str(delay), color=c_color)
            ax[0][1].fill_between(stat[:num_epochs, 0], train_acc_mean + train_acc_std,
                                  train_acc_mean - train_acc_std, alpha=0.4)

            ax[1][0].plot(stat[:num_epochs, 0], test_loss_mean, label=str(delay), color=c_color)
            ax[1][0].fill_between(stat[:num_epochs, 0], test_loss_mean + test_loss_std,
                                  test_loss_mean - test_loss_std, alpha=0.4)

            ax[1][1].plot(stat[:num_epochs, 0], test_acc_mean, label=str(delay), color=c_color)
            ax[1][1].fill_between(stat[:num_epochs, 0], test_acc_mean + test_acc_std,
                                  test_acc_mean - test_acc_std, alpha=0.4)

        handles, labels = ax[0][0].get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: float(t[0])))
        ax[0][0].legend(handles, labels)
        del handles, labels


        handles, labels = ax[0][1].get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: float(t[0])))
        ax[0][1].legend(handles, labels)
        del handles, labels


        handles, labels = ax[1][0].get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: float(t[0])))
        ax[1][0].legend(handles, labels)
        del handles, labels


        handles, labels = ax[1][1].get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: float(t[0])))
        ax[1][1].legend(handles, labels)
        del handles, labels


        ax[0][0].set_xlabel('epochs')
        ax[0][1].set_xlabel('epochs')
        ax[1][0].set_xlabel('epochs')
        ax[1][1].set_xlabel('epochs')

        ax[0][0].set_ylabel('train loss')
        ax[0][1].set_ylabel('train acc')
        ax[1][0].set_ylabel('test loss')
        ax[1][1].set_ylabel('test acc')

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(fig_folder + 'lr_' + str(lr) + '.png')
        fig.savefig(fig_folder + 'lr_' + str(lr) + '.pdf')
        plt.close(fig)

    for delay in sorted(delays):
        fig, ax = plt.subplots(2, 2)

        mctk = list(color_names.keys())
        mctkl = cycle(mctk)

        for lr in sorted(lrs):
            if delay in plt_stats[lr].keys():
                stats_list = plt_stats[lr][delay]

                train_loss = []
                train_acc = []
                test_loss = []
                test_acc = []

                for stat in stats_list:
                    train_loss.append(stat[:num_epochs, 1])
                    train_acc.append(stat[:num_epochs, 2])
                    test_loss.append(stat[:num_epochs, 4])
                    test_acc.append(stat[:num_epochs, 5])

                c_color = color_names[next(mctkl)]

                train_loss_mean = np.mean(np.array(train_loss), axis=0)
                train_loss_std = np.std(np.array(train_loss), axis=0, ddof=0)

                train_acc_mean = np.mean(np.array(train_acc), axis=0)
                train_acc_std = np.std(np.array(train_acc), axis=0, ddof=0)

                test_loss_mean = np.mean(np.array(test_loss), axis=0)
                test_loss_std = np.std(np.array(test_loss), axis=0, ddof=0)

                test_acc_mean = np.mean(np.array(test_acc), axis=0)
                test_acc_std = np.std(np.array(test_acc), axis=0, ddof=0)
                # print(train_loss.shape, stat[])

                ax[0][0].plot(stat[:num_epochs, 0], train_loss_mean, label=str(lr), color=c_color)
                ax[0][0].fill_between(stat[:num_epochs, 0], train_loss_mean + train_loss_std,
                                      train_loss_mean - train_loss_std, alpha=0.4)

                ax[0][1].plot(stat[:num_epochs, 0], train_acc_mean, label=str(lr), color=c_color)
                ax[0][1].fill_between(stat[:num_epochs, 0], train_acc_mean + train_acc_std,
                                      train_acc_mean - train_acc_std, alpha=0.4)

                ax[1][0].plot(stat[:num_epochs, 0], test_loss_mean, label=str(lr), color=c_color)
                ax[1][0].fill_between(stat[:num_epochs, 0], test_loss_mean + test_loss_std,
                                      test_loss_mean - test_loss_std, alpha=0.4)

                ax[1][1].plot(stat[:num_epochs, 0], test_acc_mean, label=str(lr), color=c_color)
                ax[1][1].fill_between(stat[:num_epochs, 0], test_acc_mean + test_acc_std,
                                      test_acc_mean - test_acc_std, alpha=0.4)

        handles, labels = ax[0][0].get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: float(t[0])))
        ax[0][0].legend(handles, labels)
        del handles, labels


        handles, labels = ax[0][1].get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: float(t[0])))
        ax[0][1].legend(handles, labels)
        del handles, labels


        handles, labels = ax[1][0].get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: float(t[0])))
        ax[1][0].legend(handles, labels)
        del handles, labels


        handles, labels = ax[1][1].get_legend_handles_labels()
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: float(t[0])))
        ax[1][1].legend(handles, labels)
        del handles, labels


        ax[0][0].set_xlabel('epochs')
        ax[0][1].set_xlabel('epochs')
        ax[1][0].set_xlabel('epochs')
        ax[1][1].set_xlabel('epochs')

        ax[0][0].set_ylabel('train loss')
        ax[0][1].set_ylabel('train acc')
        ax[1][0].set_ylabel('test loss')
        ax[1][1].set_ylabel('test acc')

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(fig_folder + 'delay_' + str(delay) + '.png')
        fig.savefig(fig_folder + 'delay_' + str(delay) + '.pdf')
        plt.close(fig)


def plot_param_distance_all_configs(exp_folder, fig_folder='plots_stab/'):

    print("stability plots")
    conf_folders = glob.glob(exp_folder + "*/")
    fig_folder = exp_folder + fig_folder

    if not os.path.exists(fig_folder):
        os.makedirs(fig_folder)

    plt_stats = {}
    delays = set()
    lrs = set()
    comparison_stats = {}

    for conf_folder in conf_folders:
        if conf_folder[:4] == "plot":
            continue

        try:
            with open(conf_folder + 'config.json', 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            continue

        filename = conf_folder + 'distances.csv'

        if os.path.isfile(filename):
            stats = np.genfromtxt(filename, delimiter=",")

            lr = config['lr']
            delay = config['delay']
            lrs.add(lr)
            delays.add(delay)

            if lr in plt_stats.keys():
                if delay not in plt_stats[lr].keys():
                    plt_stats[lr][delay] = [stats]
                else:
                    plt_stats[lr][delay].append(stats)
            else:
                plt_stats[lr] = {}
                plt_stats[lr][delay] = [stats]

        f1 = conf_folder + 'm1/stats.csv'
        f2 = conf_folder + 'm2/stats.csv'
        if os.path.isfile(f1) and os.path.isfile(f2):
            d11 = np.genfromtxt(f1, delimiter=',')
            d21 = np.genfromtxt(f2, delimiter=',')
            d1 = np.insert(d11, 0, np.zeros((1, d11.shape[1])), 0)
            d2 = np.insert(d21, 0, np.zeros((1, d21.shape[1])), 0)

            lr = config['lr']
            delay = config['delay']
            lrs.add(lr)
            delays.add(delay)

            if lr in comparison_stats.keys():
                if delay not in comparison_stats[lr].keys():
                    comparison_stats[lr][delay] = np.hstack((np.array(abs(d1[:, 2] - d1[:, 5])).reshape(-1,1),
                                                            np.array(abs(d2[:, 2] - d2[:, 5])).reshape(-1,1)))
                else:

                    try:
                        comparison_stats[lr][delay] = np.hstack((comparison_stats[lr][delay],
                                                                 np.array(abs(d1[:, 2] - d1[:, 5])).reshape(-1,1)))
                        comparison_stats[lr][delay] = np.hstack((comparison_stats[lr][delay],
                                                                 np.array(abs(d2[:, 2] - d2[:, 5])).reshape(-1,1)))
                    except ValueError:
                        print(comparison_stats[lr][delay].shape, d1.shape)
            else:
                comparison_stats[lr] = {}
                comparison_stats[lr][delay] = np.hstack((np.array(abs(d1[:, 2] - d1[:, 5])).reshape(-1,1),
                np.array(abs(d2[:, 2] - d2[:, 5])).reshape(-1,1)))

    for lr in lrs:

        for delay in plt_stats[lr].keys():
            fig, ax = plt.subplots(1, 1)

            ax.errorbar(list(range(comparison_stats[lr][delay].shape[0])), comparison_stats[lr][delay].mean(axis=1),
                        comparison_stats[lr][delay].std(axis=1, ddof=0), label='abs(train loss - test loss)')
            try:
                dists = np.hstack([td[:,[0]] for td in plt_stats[lr][delay]])
                # print(dists.shape)
                ax.errorbar(list(range(dists.shape[0])), dists.mean(axis=1), dists.std(axis=1, ddof=0), label='stability')


                ax.legend()
                ax.set_xlabel('epochs')
                # fig.suptitle(exp_folder)
                fig.savefig(fig_folder + 'comparison_lr_' + str(lr) + "delay_" + str(delay) + '.png')
                fig.savefig(fig_folder + 'comparison_lr_' + str(lr) + "delay_" + str(delay) + '.pdf')
                plt.close(fig)
            except ValueError:
                print(len(plt_stats[lr][delay]))

    for lr in lrs:
        fig, ax = plt.subplots(1,1)

        for delay in plt_stats[lr].keys():
            # print(plt_stats[lr][delay].shape)
            dists = np.hstack([td[:,[0]] for td in plt_stats[lr][delay]])
            # print(dists.shape)
            ax.plot(dists.mean(axis=1), label="delay: "+str(delay))
            # ax.plot(plt_stats[lr][delay][:, 0], label=str(delay))

        ax.legend()
        ax.set_xlabel('epochs')
        # fig.suptitle(exp_folder)
        fig.savefig(fig_folder + 'lr_' + str(lr) + '.png')
        fig.savefig(fig_folder + 'lr_' + str(lr) + '.pdf')
        plt.close(fig)

    for delay in delays:
        fig, ax = plt.subplots(1,1)
        for lr in lrs:
            if delay in plt_stats[lr].keys():

                dists = np.hstack([td[:, [0]] for td in plt_stats[lr][delay]])
                ax.plot(dists.mean(axis=1), label="lr: "+ str(lr))

        ax.legend()
        ax.set_xlabel('epochs')
        fig.savefig(fig_folder + 'delay_' + str(delay) + '.png')
        fig.savefig(fig_folder + 'delay_' + str(delay) + '.pdf')
        plt.close(fig)


if __name__ == '__main__':


    ff = ['plots/']
    for file_folder in ff:
        pxp_folders = glob.glob(file_folder + "pexp*/")

        for pxp_folder in pxp_folders:
            print(pxp_folder)
            num_epochs=150

            plot_configs_pexp(pxp_folder, num_epochs=num_epochs)
            plot_configs_pexp_acc(pxp_folder, num_epochs=num_epochs)

            plot_param_distance_all_configs(pxp_folder)
