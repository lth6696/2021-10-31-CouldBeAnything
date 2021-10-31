import numpy as np
from scipy import stats as st


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

    def ClassicSampler(self, rv, T):
        # 细致平稳条件：π(i)P(i,j)=π(j)P(j,i)
        # 故：P(i,j)=Q(i,j)α(i,j)
        # 通过随机转移矩阵Q进行采样，样本被保留的概率为α
        pass

    def MetropolisHastingsSampler(self, rv, T):
        # α(i,j)=min{[π(j)Q(j,i)]/[π(i)Q(i,j)], 1}
        Q = st.binom
        naccept = 0       # 接受点数
        x = 0.1           # 初始状态
        samples = np.zeros(T+1)     # 存放采样值的列表
        samples[0] = x
        for i in range(T):
            y = x + st.norm(0, 0.3).rvs()   # 采样值
            u = np.random.uniform()     # 从均匀分布产生u
            rho = min(1, (rv.pdf(x)*Q(10, x).pmf(61))/(rv.pdf(y)*Q(10, y).pmf(61)))
            print(rho)
            if u < rho:
                naccept += 1
                x = y
            samples[i+1] = x
        # nmcmc = len(samples) // 2