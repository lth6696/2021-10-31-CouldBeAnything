import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


def style(width, height, fontsize=8):
    plt.rcParams['figure.figsize'] = (width * 0.39370, height * 0.39370)  # figure size in inches
    plt.rcParams['font.sans-serif'] = 'cambria'
    plt.rcParams['font.size'] = fontsize
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['patch.linewidth'] = 0.5
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['ytick.major.width'] = 0.5
    plt.rcParams['xtick.major.width'] = 0.5
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['xtick.direction'] = 'in'


def line_graph(data):
    style(8.6, 6.2)
    marker = ['o', 's', 'X', '^', 'p']
    line = ['-', '--', ':', '-.']
    map_vir = cm.get_cmap(name='gist_rainbow')
    solver = ['LBMS', 'LSMS']
    K = [4, 8, 12, 16]
    (row, col) = data.shape
    for i, rec in enumerate(ILP_MappingRate):
        plt.plot(
            K, rec,
            marker=marker[i], ms=2,
            ls=line[i], lw=0.5,
            color=map_vir(i/col),
            label=solver[i]
        )
    plt.yticks(rotation='vertical')
    plt.xlabel("Number of Traffic Matrices")
    plt.ylabel("Mapping Rate (%)")
    plt.xticks(K)
    plt.yticks([i * 5 + 80 for i in range(0, 5)])
    plt.grid(True, ls=':', lw=0.5, c='#d5d6d8')
    plt.tight_layout()
    plt.legend()
    plt.show()


def bar_graph(data):
    style(8.6, 6.2)
    colors = [
        ['#FF0309', '#B90101', '#900302'],
        ['#0FC2C0', '#0CABA8', '#008F8C']
    ]
    solver = [
        ['LBMS-L1', 'LBMS-L2', 'LBMS-L3'],
        ['LSMS-L1', 'LSMS-L2', 'LSMS-L3']
    ]
    K = [4, 8, 12, 16]
    (scheme, row, col) = data.shape
    for i in range(scheme):
        for j, rec in enumerate(data[i]):
            print(solver[i][j])
            plt.bar(
                np.array(K) - 1.05 + (i * 0.3) + 0.9 * j,
                rec,
                width=0.3,
                label=solver[i][j],
                color=colors[i][j]
            )
    plt.yticks(rotation='vertical')
    plt.xlabel("Number of Traffic Matrices")
    plt.ylabel("Number of hops")
    plt.xticks(K)
    plt.yticks([i*0.6 for i in range(6)])
    plt.grid(True, ls=':', lw=0.5, c='#d5d6d8')
    plt.tight_layout()
    plt.legend(ncol=2)
    plt.show()


if __name__ == '__main__':
    ILP_MappingRate = np.array(
        [
            [98.8, 96.2, 93.1, 91.4],
            [94.3, 90.2, 85.8, 80.4]
        ]
    )
    ILP_Throughput = np.array(
        [
            [   # LBMS
                [230, 414, 556, 731],
                [231, 478, 701, 903],
                [244, 475, 726, 948]
            ],
            [   # LSMS
                [230, 418, 616, 729],
                [223, 420, 605, 743],
                [214, 429, 591, 718]
            ]
        ]
    )
    ILP_Hops = np.array(
        [
            [
                [1.90, 1.89, 1.79, 1.71],
                [1.72, 1.75, 1.81, 1.81],
                [1.66, 1.70, 1.80, 1.92]
            ],
            [
                [1.92, 1.85, 1.86, 1.75],
                [1.89, 1.81, 1.85, 1.75],
                [1.87, 1.86, 1.80, 1.78]
            ]
        ]
    )
    ILP_LightpathUtilization = np.array(
        [
            [
                [35.7, 64.2, 78.7, 88.6],
                [15.8, 35.0, 59.5, 78.0],
                [6.5, 14.0, 28.7, 51.8]
            ],
            [
                [20.2, 37.7, 54.0, 58.1],
                [20.0, 36.3, 51.1, 59.9],
                [20.8, 36.2, 50.6, 59.0]
            ]
        ]
    )
    # line_graph(ILP_MappingRate)
    bar_graph(ILP_Hops)
