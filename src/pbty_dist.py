
from scipy import stats


class DiscreteProbabilityDistributions(object):

    def __init__(self):
        pass

    def geometric_distribution(self, p):
        # f(k) = p*(1-p)**(k-1), k>=1, 0<p<=1
        # p - the probability of a single success
        rv = stats.geom(p)
        return rv


class ContinuousProbabilityDistributions(object):

    def __init__(self):
        pass

    def exponent_distribution(self):
        # f(x) = exp(-x), x>=0
        rv = stats.expon()
        return rv