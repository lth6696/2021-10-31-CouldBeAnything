import matplotlib.pyplot as plt
import numpy as np

from source.Result.plot_style import Style


class ResultPresentation(object):
    def __init__(self):
        pass

    def plot_line(self, x, y=0):
        Style.style(7, 13)
        plt.plot(x)
        plt.show()


if __name__ == '__main__':
    pass