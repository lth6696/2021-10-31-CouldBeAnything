import pbty_dist as pd
import queueing_sys as qs


def function():
    rv = pd.ContinuousProbabilityDistributions().exponent_distribution()
    mcmc = qs.MarkovChainMonteCarlo()
    mcmc.MetropolisHastingsSampler(rv, 10)


if __name__ == '__main__':
    function()