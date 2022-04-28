import matplotlib.pyplot as plt

from source.Result.plot_res import plot_style


class ResultStorage:
    def __init__(self):
        pass


if __name__ == '__main__':
    K = [2, 4, 6]
    SucML = [70.88, 66.21, 57.42]
    SucSL = [59.07, 50.69, 43.96]
    TptML = [1.59, 2.78, 3.40]
    TptSL = [1.29, 2.04, 2.56]
    plot_style()
    plt.bar(K, SucML, width=0.3, label='Multi-level')
    plt.bar([i+0.3 for i in K], SucSL, width=0.3, label='Single-level')
    plt.ylabel('SucRate')
    plt.xlabel('K')
    plt.twinx()
    plt.plot(K, TptML, label='Multi-level', ls='-', lw=0.5)
    plt.plot(K, TptSL, label='Single-level', ls='-', lw=0.5)
    plt.xticks(K)
    plt.legend()
    plt.ylabel('Throughput')
    plt.grid(True, ls=':', lw=0.5, c='#d5d6d8')
    plt.show()