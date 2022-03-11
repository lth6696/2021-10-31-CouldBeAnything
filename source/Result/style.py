import matplotlib.pyplot as plt


class Style:
    def __init__(self):
        pass

    @staticmethod
    def style(width, height, fontsize=8):
        plt.rcParams['figure.figsize'] = (width * 0.39370, height * 0.39370)  # figure size in inches
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['font.size'] = fontsize
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['patch.linewidth'] = 0.5
        plt.rcParams['axes.linewidth'] = 0.5
        plt.rcParams['ytick.major.width'] = 0.5
        plt.rcParams['xtick.major.width'] = 0.5
        plt.rcParams['ytick.direction'] = 'in'
        plt.rcParams['xtick.direction'] = 'in'