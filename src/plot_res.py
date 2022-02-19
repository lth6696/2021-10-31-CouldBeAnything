import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio


def plot_successed_route():
    SHF_FILE = './data/shf_suc.mat'
    MRU_FILE = './data/mru_suc.mat'
    SHF_SUCCESS = scio.loadmat(SHF_FILE)
    MRU_SUCCESS = scio.loadmat(MRU_FILE)

    x = [(i + 1) * 0.05 for i in range(20)]
    shf_err_neg = [SHF_SUCCESS['mean'][0][i] - SHF_SUCCESS['min'][0][i] for i in range(len(SHF_SUCCESS['mean'][0]))]
    shf_err_pos = [SHF_SUCCESS['max'][0][i] - SHF_SUCCESS['mean'][0][i] for i in range(len(SHF_SUCCESS['mean'][0]))]
    mru_err_neg = [MRU_SUCCESS['mean'][0][i] - MRU_SUCCESS['min'][0][i] for i in range(len(MRU_SUCCESS['mean'][0]))]
    mru_err_pos = [MRU_SUCCESS['max'][0][i] - MRU_SUCCESS['mean'][0][i] for i in range(len(MRU_SUCCESS['mean'][0]))]

    excellent_val = [SHF_SUCCESS['mean'][0][i] - MRU_SUCCESS['mean'][0][i] for i in range(len(SHF_SUCCESS['mean'][0]))]
    print(np.mean(excellent_val))

    plot_style()
    plt.errorbar(x, SHF_SUCCESS['mean'][0], yerr=[shf_err_neg, shf_err_pos], label='SHF',
                 color='#eab026', ls='-', lw=0.5,
                 marker='o', mec='#05472a', mfc='#c97937', mew=0.5, ms=2)
    plt.errorbar(x, MRU_SUCCESS['mean'][0], yerr=[mru_err_neg, mru_err_pos], label='MRU',
                 color='#00a8e1', ls='-', lw=0.5,
                 marker='o', mec='#05472a', mfc='#a0febf', mew=0.5, ms=2)
    plt.xlabel('Traffic load')
    plt.ylabel('Success allocation rate (%)')
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.yticks(rotation='vertical')
    plt.tight_layout()
    plt.legend()
    plt.grid(True, ls=':', lw=0.5, c='#d5d6d8')
    plt.show()


def plot_shortest_hop_first():
    SHF_FILE = './data/shf_hop.mat'
    MRU_FILE = './data/mru_hop.mat'
    SHF_HOP_DIS = scio.loadmat(SHF_FILE)
    MRU_HOP_DIS = scio.loadmat(MRU_FILE)

    x = [(i+1)*0.05 for i in range(20)]
    shf_err_neg = [SHF_HOP_DIS['mean'][0][i] - SHF_HOP_DIS['min'][0][i] for i in range(len(SHF_HOP_DIS['mean'][0]))]
    shf_err_pos = [SHF_HOP_DIS['max'][0][i] - SHF_HOP_DIS['mean'][0][i] for i in range(len(SHF_HOP_DIS['mean'][0]))]
    mru_err_neg = [MRU_HOP_DIS['mean'][0][i] - MRU_HOP_DIS['min'][0][i] for i in range(len(MRU_HOP_DIS['mean'][0]))]
    mru_err_pos = [MRU_HOP_DIS['max'][0][i] - MRU_HOP_DIS['mean'][0][i] for i in range(len(MRU_HOP_DIS['mean'][0]))]

    excellent_val = [SHF_HOP_DIS['mean'][0][i] - MRU_HOP_DIS['mean'][0][i] for i in range(len(SHF_HOP_DIS['mean'][0]))]
    print(np.mean(excellent_val))

    plot_style()
    plt.errorbar(x, SHF_HOP_DIS['mean'][0], yerr=[shf_err_neg, shf_err_pos], label='SHF',
                 color='#eab026', ls='-', lw=0.5,
                 marker='o', mec='#05472a', mfc='#c97937', mew=0.5, ms=2)
    plt.errorbar(x, MRU_HOP_DIS['mean'][0], yerr=[mru_err_neg, mru_err_pos], label='MRU',
                 color='#00a8e1', ls='-', lw=0.5,
                 marker='o', mec='#05472a', mfc='#a0febf', mew=0.5, ms=2)
    plt.xlabel('Traffic load')
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
    SHF_RES = scio.loadmat(SHF_FILE)
    MRU_RES = scio.loadmat(MRU_FILE)

    x = [(i + 1) * 0.05 for i in range(20)]
    shf_data_neg = [SHF_RES['data_mean'][0][i] - SHF_RES['data_min'][0][i] for i in range(len(SHF_RES['data_mean'][0]))]
    shf_data_pos = [SHF_RES['data_max'][0][i] - SHF_RES['data_mean'][0][i] for i in range(len(SHF_RES['data_mean'][0]))]
    shf_key_neg = [SHF_RES['key_mean'][0][i] - SHF_RES['key_min'][0][i] for i in range(len(SHF_RES['key_mean'][0]))]
    shf_key_pos = [SHF_RES['key_max'][0][i] - SHF_RES['key_mean'][0][i] for i in range(len(SHF_RES['key_mean'][0]))]
    mru_data_neg = [MRU_RES['data_mean'][0][i] - MRU_RES['data_min'][0][i] for i in range(len(MRU_RES['data_mean'][0]))]
    mru_data_pos = [MRU_RES['data_max'][0][i] - MRU_RES['data_mean'][0][i] for i in range(len(MRU_RES['data_mean'][0]))]
    mru_key_neg = [MRU_RES['key_mean'][0][i] - MRU_RES['key_min'][0][i] for i in range(len(MRU_RES['key_mean'][0]))]
    mru_key_pos = [MRU_RES['key_max'][0][i] - MRU_RES['key_mean'][0][i] for i in range(len(MRU_RES['key_mean'][0]))]

    excellent_val = [SHF_RES['data_mean'][0][i] - MRU_RES['data_mean'][0][i] for i in range(len(SHF_RES['data_mean'][0]))]
    print(np.mean(excellent_val))
    excellent_val = [SHF_RES['key_mean'][0][i] - MRU_RES['key_mean'][0][i] for i in range(len(SHF_RES['key_mean'][0]))]
    print(np.mean(excellent_val))

    plot_style()
    plt.errorbar(x, SHF_RES['data_mean'][0], yerr=[shf_data_neg, shf_data_pos], label='SHF-data',
                 color='#b00149', ls='-', lw=0.5,
                 marker='o', mec='#ff6cb5', mfc='#80013f', mew=0.5, ms=2)
    plt.errorbar(x, SHF_RES['key_mean'][0], yerr=[shf_key_neg, shf_key_pos], label='SHF-key',
                 color='#c875c4', ls='-', lw=0.5,
                 marker='o', mec='#dd85d7', mfc='#380835', mew=0.5, ms=2)
    plt.errorbar(x, MRU_RES['data_mean'][0], yerr=[mru_data_neg, mru_data_pos], label='MRU-data',
                 color='#33b864', ls='-', lw=0.5,
                 marker='o', mec='#a0febf', mfc='#0a481e', mew=0.5, ms=2)
    plt.errorbar(x, MRU_RES['key_mean'][0], yerr=[mru_key_neg, mru_key_pos], label='MRU-key',
                 color='#cdfd02', ls='-', lw=0.5,
                 marker='o', mec='#373e02', mfc='#d0e429', mew=0.5, ms=2)
    plt.xlabel('Traffic load')
    plt.ylabel('Resource utilization')
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.yticks(rotation='vertical')
    plt.tight_layout()
    plt.legend(ncol=2)
    plt.grid(True, ls=':', lw=0.5, c='#d5d6d8')
    plt.show()


def plot_algorithm_runtime():
    pass


def plot_style():
    plt.rcParams['figure.figsize'] = (2.2, 1.65)  # figure size in inches
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
    plot_max_res_utilization()