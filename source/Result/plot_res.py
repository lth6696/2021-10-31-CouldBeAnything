import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio


def plot_successed_route():
    SHF_FILE = './data/shf_suc.mat'
    MRU_FILE = './data/mru_suc.mat'
    SPF_FILE = './data/spf_suc.mat'
    SHF_SUCCESS = scio.loadmat(SHF_FILE)
    MRU_SUCCESS = scio.loadmat(MRU_FILE)
    SPF_SUCCESS = scio.loadmat(SPF_FILE)

    x = [(i + 1) * 0.05 for i in range(20)]
    shf_err_neg = [SHF_SUCCESS['mean'][0][i] - SHF_SUCCESS['min'][0][i] for i in range(len(SHF_SUCCESS['mean'][0]))]
    shf_err_pos = [SHF_SUCCESS['max'][0][i] - SHF_SUCCESS['mean'][0][i] for i in range(len(SHF_SUCCESS['mean'][0]))]
    mru_err_neg = [MRU_SUCCESS['mean'][0][i] - MRU_SUCCESS['min'][0][i] for i in range(len(MRU_SUCCESS['mean'][0]))]
    mru_err_pos = [MRU_SUCCESS['max'][0][i] - MRU_SUCCESS['mean'][0][i] for i in range(len(MRU_SUCCESS['mean'][0]))]
    spf_err_neg = [SPF_SUCCESS['mean'][0][i] - SPF_SUCCESS['min'][0][i] for i in range(len(SPF_SUCCESS['mean'][0]))]
    spf_err_pos = [SPF_SUCCESS['max'][0][i] - SPF_SUCCESS['mean'][0][i] for i in range(len(SPF_SUCCESS['mean'][0]))]

    excellent_val = [SHF_SUCCESS['mean'][0][i] - MRU_SUCCESS['mean'][0][i] for i in range(len(SHF_SUCCESS['mean'][0]))]
    print('SHF better than MRU - ', np.mean(excellent_val))
    excellent_val = [SPF_SUCCESS['mean'][0][i] - SHF_SUCCESS['mean'][0][i] for i in range(len(SPF_SUCCESS['mean'][0]))]
    print('SPF better than SHF - ', np.mean(excellent_val))
    print("SHF - {:4f} MRU - {:4f} SPF - {:4f}".format(np.mean(SHF_SUCCESS['mean'][0]),
                                                       np.mean(MRU_SUCCESS['mean'][0]),
                                                       np.mean(SPF_SUCCESS['mean'][0])))

    plot_style()
    plt.errorbar(x, SHF_SUCCESS['mean'][0], yerr=[shf_err_neg, shf_err_pos], label='SHF',
                 color='#4472C4', ls='-', lw=0.5,
                 marker='o', mec='#4472C4', mfc='#4472C4', mew=0.5, ms=2)
    plt.errorbar(x, MRU_SUCCESS['mean'][0], yerr=[mru_err_neg, mru_err_pos], label='MRU',
                 color='#C55A11', ls='-', lw=0.5,
                 marker='o', mec='#C55A11', mfc='#C55A11', mew=0.5, ms=2)
    plt.errorbar(x, SPF_SUCCESS['mean'][0], yerr=[spf_err_neg, spf_err_pos], label='SPF',
                 color='#FFC000', ls='-', lw=0.5,
                 marker='o', mec='#FFC000', mfc='#FFC000', mew=0.5, ms=2)
    plt.xlabel('Workload')
    plt.ylabel('Success mapping rate (%)')
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.yticks(rotation='vertical')
    plt.tight_layout()
    plt.legend()
    plt.grid(True, ls=':', lw=0.5, c='#d5d6d8')
    plt.show()


def plot_shortest_hop_first():
    SHF_FILE = './data/shf_hop.mat'
    MRU_FILE = './data/mru_hop.mat'
    SPF_FILE = './data/spf_hop.mat'
    SHF_HOP_DIS = scio.loadmat(SHF_FILE)
    MRU_HOP_DIS = scio.loadmat(MRU_FILE)
    SPF_HOP_DIS = scio.loadmat(SPF_FILE)

    x = [(i+1)*0.05 for i in range(20)]
    shf_err_neg = [SHF_HOP_DIS['mean'][0][i] - SHF_HOP_DIS['min'][0][i] for i in range(len(SHF_HOP_DIS['mean'][0]))]
    shf_err_pos = [SHF_HOP_DIS['max'][0][i] - SHF_HOP_DIS['mean'][0][i] for i in range(len(SHF_HOP_DIS['mean'][0]))]
    mru_err_neg = [MRU_HOP_DIS['mean'][0][i] - MRU_HOP_DIS['min'][0][i] for i in range(len(MRU_HOP_DIS['mean'][0]))]
    mru_err_pos = [MRU_HOP_DIS['max'][0][i] - MRU_HOP_DIS['mean'][0][i] for i in range(len(MRU_HOP_DIS['mean'][0]))]
    spf_err_neg = [SPF_HOP_DIS['mean'][0][i] - SPF_HOP_DIS['min'][0][i] for i in range(len(SPF_HOP_DIS['mean'][0]))]
    spf_err_pos = [SPF_HOP_DIS['max'][0][i] - SPF_HOP_DIS['mean'][0][i] for i in range(len(SPF_HOP_DIS['mean'][0]))]

    excellent_val = [SHF_HOP_DIS['mean'][0][i] - MRU_HOP_DIS['mean'][0][i] for i in range(len(SHF_HOP_DIS['mean'][0]))]
    print('SHF better than MRU - ', np.mean(excellent_val))
    excellent_val = [SPF_HOP_DIS['mean'][0][i] - SHF_HOP_DIS['mean'][0][i] for i in range(len(SPF_HOP_DIS['mean'][0]))]
    print('SPF better than SHF - ', np.mean(excellent_val))
    print("SHF - {:4f} MRU - {:4f} SPF - {:4f}".format(np.mean(SHF_HOP_DIS['mean'][0]),
                                                       np.mean(MRU_HOP_DIS['mean'][0]),
                                                       np.mean(SPF_HOP_DIS['mean'][0])))

    plot_style()
    plt.errorbar(x, SHF_HOP_DIS['mean'][0], yerr=[shf_err_neg, shf_err_pos], label='SHF',
                 color='#4472C4', ls='-', lw=0.5,
                 marker='o', mec='#4472C4', mfc='#4472C4', mew=0.5, ms=2)
    plt.errorbar(x, MRU_HOP_DIS['mean'][0], yerr=[mru_err_neg, mru_err_pos], label='MRU',
                 color='#C55A11', ls='-', lw=0.5,
                 marker='o', mec='#C55A11', mfc='#C55A11', mew=0.5, ms=2)
    plt.errorbar(x, SPF_HOP_DIS['mean'][0], yerr=[spf_err_neg, spf_err_pos], label='SPF',
                 color='#FFC000', ls='-', lw=0.5,
                 marker='o', mec='#FFC000', mfc='#FFC000', mew=0.5, ms=2)
    plt.xlabel('Workload')
    plt.ylabel('Hop')
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.yticks(rotation='vertical')
    plt.tight_layout()
    plt.legend()
    plt.grid(True, ls=':', lw=0.5, c='#d5d6d8')
    plt.show()


def plot_max_res_utilization():
    SHF_FILE = './data/shf_res.mat'
    MRU_FILE = './data/mru_res.mat'
    SPF_FILE = './data/spf_res.mat'
    SHF_RES = scio.loadmat(SHF_FILE)
    MRU_RES = scio.loadmat(MRU_FILE)
    SPF_RES = scio.loadmat(SPF_FILE)

    x = [(i + 1) * 0.05 for i in range(20)]
    shf_data_neg = [SHF_RES['data_mean'][0][i] - SHF_RES['data_min'][0][i] for i in range(len(SHF_RES['data_mean'][0]))]
    shf_data_pos = [SHF_RES['data_max'][0][i] - SHF_RES['data_mean'][0][i] for i in range(len(SHF_RES['data_mean'][0]))]
    # shf_key_neg = [SHF_RES['key_mean'][0][i] - SHF_RES['key_min'][0][i] for i in range(len(SHF_RES['key_mean'][0]))]
    # shf_key_pos = [SHF_RES['key_max'][0][i] - SHF_RES['key_mean'][0][i] for i in range(len(SHF_RES['key_mean'][0]))]
    mru_data_neg = [MRU_RES['data_mean'][0][i] - MRU_RES['data_min'][0][i] for i in range(len(MRU_RES['data_mean'][0]))]
    mru_data_pos = [MRU_RES['data_max'][0][i] - MRU_RES['data_mean'][0][i] for i in range(len(MRU_RES['data_mean'][0]))]
    # mru_key_neg = [MRU_RES['key_mean'][0][i] - MRU_RES['key_min'][0][i] for i in range(len(MRU_RES['key_mean'][0]))]
    # mru_key_pos = [MRU_RES['key_max'][0][i] - MRU_RES['key_mean'][0][i] for i in range(len(MRU_RES['key_mean'][0]))]
    spf_err_neg = [SPF_RES['mean'][0][i] - SPF_RES['min'][0][i] for i in range(len(SPF_RES['mean'][0]))]
    spf_err_pos = [SPF_RES['max'][0][i] - SPF_RES['mean'][0][i] for i in range(len(SPF_RES['mean'][0]))]

    excellent_val = [SHF_RES['data_mean'][0][i] - MRU_RES['data_mean'][0][i] for i in range(len(SHF_RES['data_mean'][0]))]
    print('Data: SHF better than MRU', np.mean(excellent_val))
    # excellent_val = [SHF_RES['key_mean'][0][i] - MRU_RES['key_mean'][0][i] for i in range(len(SHF_RES['key_mean'][0]))]
    # print('Key: SHF better than MRU', np.mean(excellent_val))
    excellent_val = [SPF_RES['mean'][0][i] - MRU_RES['data_mean'][0][i] for i in range(len(SPF_RES['mean'][0]))]
    print('Data: SPF better than MRU', np.mean(excellent_val))

    plot_style()
    plt.errorbar(x, SHF_RES['data_mean'][0], yerr=[shf_data_neg, shf_data_pos], label='SHF',
                 color='#4472C4', ls='-', lw=0.5,
                 marker='o', mec='#4472C4', mfc='#4472C4', mew=0.5, ms=2)
    # plt.errorbar(x, SHF_RES['key_mean'][0], yerr=[shf_key_neg, shf_key_pos], label='SHF-key',
    #              color='#B4C7E7', ls='-', lw=0.5,
    #              marker='o', mec='#B4C7E7', mfc='#B4C7E7', mew=0.5, ms=2)
    plt.errorbar(x, MRU_RES['data_mean'][0], yerr=[mru_data_neg, mru_data_pos], label='MRU',
                 color='#C55A11', ls='-', lw=0.5,
                 marker='o', mec='#C55A11', mfc='#C55A11', mew=0.5, ms=2)
    # plt.errorbar(x, MRU_RES['key_mean'][0], yerr=[mru_key_neg, mru_key_pos], label='MRU-key',
    #              color='#F4B183', ls='-', lw=0.5,
    #              marker='o', mec='#F4B183', mfc='#F4B183', mew=0.5, ms=2)
    plt.errorbar(x, SPF_RES['mean'][0], yerr=[spf_err_neg, spf_err_pos], label='SPF',
                 color='#FFC000', ls='-', lw=0.5,
                 marker='o', mec='#FFC000', mfc='#FFC000', mew=0.5, ms=2)
    plt.xlabel('Workload')
    plt.ylabel('Bandwidth utilization')
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.yticks(rotation='vertical')
    plt.tight_layout()
    plt.legend(ncol=2)
    plt.grid(True, ls=':', lw=0.5, c='#d5d6d8')
    plt.show()


def plot_algorithm_runtime():
    pass


def plot_style():
    # plt.rcParams['figure.figsize'] = (2.2, 1.65)  # figure size in inches
    plt.rcParams['figure.figsize'] = (3.44, 1.65)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['font.size'] = 6
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['patch.linewidth'] = 0.5
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['ytick.major.width'] = 0.5
    plt.rcParams['xtick.major.width'] = 0.5
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.direction'] = 'in'


if __name__ == '__main__':
    plot_successed_route()