import numpy as np
from scipy import stats as st
import matplotlib.pyplot as plt


class MarkovChainMonteCarlo(object):

    def __init__(self):
        """
        Four methods:
        1 - classic sampler
        2 - Metropolis-Hastings sampler
        3 - Gibbs sampler
        4 - Slice sampler
        """
        pass

    def ClassicSampler(self, T):
        # 细致平稳条件：π(i)P(i,j)=π(j)P(j,i)
        # 故：P(i,j)=Q(i,j)α(i,j)
        # 通过随机转移矩阵Q进行采样，样本被保留的概率为α
        pass

    #todo only for continuous prob dist
    @staticmethod
    def MetropolisHastingsSampler(func, T=10**4):

        def dst_pbty(function, theta):
                return function.pdf(theta)

        pi = [0 for _ in range(T)]  # 初始状态pi[t]
        for t in range(T-1):
            Q = st.norm()   # 状态转移矩阵Q的条件概率分布服从正态分布，均值loc，方差scale
            pi_star = Q.rvs(loc=0, scale=1, size=1, random_state=None)  # 采取样本π*，返回列表格式
            u = np.random.uniform(0, 1)     # 从均匀分布产生u
            alpha = min(1, dst_pbty(func, pi_star[0]) / dst_pbty(func, pi[t]))  # α(i,j)=min{[π(j)Q(j,i)]/[π(i)Q(i,j)], 1}
            if u < alpha:
                pi[t+1] = pi_star[0]
            else:
                pi[t+1] = pi[t]

        # 结果展示
        # plt.scatter(pi, func.pdf(pi))
        # num_bins = 50
        # plt.hist(pi, num_bins, density=True, facecolor='red', alpha=0.7)
        # plt.show()

        return pi

    def GibbsSampler(self):
        pass