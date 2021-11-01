
from scipy import stats


class DiscreteProbabilityDistributions(object):

    def __init__(self):
        pass

    @staticmethod
    def geometric_distribution(p):
        # f(k) = p*(1-p)**(k-1), k>=1, 0<p<=1
        # p - the probability of a single success
        rv = stats.geom(p)
        return rv


class ContinuousProbabilityDistributions(object):

    def __init__(self):
        pass

    @staticmethod
    def exponent_distribution():
        # f(x) = exp(-x), x>=0
        rv = stats.expon()
        return rv

    @staticmethod
    def erlang_distribution(a):
        rv = stats.erlang(a)
        return rv