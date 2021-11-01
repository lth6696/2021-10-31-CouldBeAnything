import pbty_dist as pd
import queueing_sys as qs


def function():
    rv = pd.ContinuousProbabilityDistributions.erlang_distribution(1.99)
    qs.MarkovChainMonteCarlo.MetropolisHastingsSampler(rv)


if __name__ == '__main__':
    function()