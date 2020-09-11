from scipy.special import erfcinv, erfinv, erf
from scipy.optimize import bisect, minimize
import numpy as np
from functools import partial
import statsmodels.stats.api as sms

import restart_distribution as restart

# Generall the 'Q' functions denote the qunatile function of the distribution.
# The 'QS' functions are the derivative of the qunatile function.
# And the 'QG' functions are the antiderivative.

def levy_Q(p, c=1.0):
    return c/(2.0*np.power(erfcinv(p), 2))

def levy_QS(p, c=1.0):
    if p == 0:
        return np.infty
    return c*(np.sqrt(np.pi) * np.exp(np.power(erfcinv(p), 2)))/(2.0*np.power(erfcinv(p), 3))

def levy_QG(p, c=1.0):
    if p == 0:
        return c*1.0
    return c*(1 - p + (np.exp(-np.power(erfcinv(p), 2)))/(np.sqrt(np.pi)*erfcinv(p)))

def lognorm_Q(p, sigma=1.0, mu=1.0):
    x = np.sqrt(2)*sigma*erfinv(2*p-1)+mu
    return np.exp(x)

def lognorm_QS(p, sigma=1.0, mu=1.0):
    erf_inv = erfinv(2 * p - 1)
    x = np.power(erf_inv, 2) + np.sqrt(2) * sigma * erf_inv + mu
    return sigma * np.sqrt(2*np.pi) * np.exp(x)

def lognorm_QG(p, sigma=1.0, mu=1.0):
    x = erf(sigma/np.sqrt(2) - erfinv(2*p - 1))
    return - 0.5 * np.exp(mu + np.power(sigma, 2)/2.0) * x


# Return the expected runtme with (fixed-cutoff) restarts at quantile p.
def calculate_expectation(p, Q, QG):
    return (1-p)/p * Q(p) + (QG(p)-QG(0))/p

# The root of this function corresponds to an optimal restart quantile.
# c.f. https://link.springer.com/chapter/10.1007/978-3-319-73117-9_35
def minima_function(p, Q, QS, QG):
    return (p-1)*Q(p)+(1-p)*p*QS(p)-QG(p)+QG(0)

def write_to_file(filename, data_list):
    with open(filename, 'w') as f:
        f.write(','.join([str(x) for x in data_list]))

def restart_dist(t, fixed_cutoff_function, luby_function, file_prefix='./restarts', N=10000, seed=1):
    # This function takes two functions fixed_cutoff_function and luby_function as input.
    # It simulates both strategies N times (passed as parameter).
    # The fixed-cutoff strategy uses the restart time t (passed as parameter).
    # For reproducibility a seed is used.
    restart.init(seed)
    restarts = []
    luby = []
    for i in range(N):
        if i % 500 == 0:
            print(i)
        restarts.append(fixed_cutoff_function(t))
        luby.append(luby_function())
    # The data is stored in .csv files.
    write_to_file(f"{file_prefix}_fixed_seed{seed}.csv", restarts)
    write_to_file(f"{file_prefix}_luby_seed{seed}.csv", luby)
    return restarts, luby

def write_levy_to_file(scale=1.0, N=10000, levy_prefix='./levy', seed=1):
    # This functions simulates the restart process with the Levy-Smirnov distribution.
    # For reproducibility a seed is used.

    # The optimal restart quantiles of the Levy-Smirnov
    # are found numerically by finding the root of the respective minima function.
    # Finding the optimal restart quantiles is based on the methodology discussed in
    # Runtime Distributions and Criteria for Restarts
    # https://link.springer.com/chapter/10.1007/978-3-319-73117-9_35

    # Setup the necessary functions for the Levy-Smirnov distribution.
    Q = partial(levy_Q, c=scale)
    QS = partial(levy_QS, c=scale)
    QG = partial(levy_QG, c=scale)
    levy_exp = lambda p: calculate_expectation(p, Q, QG)
    levy_minima = lambda p: minima_function(p, Q, QS, QG)


    # The following line numerically approximates the root of the minima function.
    # The resulting value therefore approximates the optimal restart quantile.
    quantile_levy = bisect(levy_minima, 0.0001, 0.9999)
    t_levy = Q(quantile_levy)
    print(f"Optimal restart quantile w.r.t. Lévy-Smirnov distribution with scale {scale}: {quantile_levy}")
    print(f"This corresponds to the restart time {t_levy} and the expected value {levy_exp(quantile_levy)}")
    print("\n")

    # First, restarts regarding the Levy distributions are simulated.
    fixed_function = lambda x: restart.restart_levy(t_levy, scale=scale)
    luby_function = lambda: restart.luby_restart_levy(scale=scale)
    levy_restarts, levy_luby = restart_dist(t_levy, fixed_function, luby_function,
                                            file_prefix=levy_prefix, N=N, seed=seed)

    # Output summary data.
    print("-------------------")
    print("Lévy-Smirnov distribution")
    print(f"fixed-cutoff restarts mean {np.mean(levy_restarts)}, Luby mean {np.mean(levy_luby)}")
    print(f"95% confidence interval for fixed-cutoff mean {sms.DescrStatsW(levy_restarts).tconfint_mean()}")
    print(f"95% confidence interval for luby mean {sms.DescrStatsW(levy_luby).tconfint_mean()}")

def write_lognorm_to_file(t=None, sigma=5.0, mu=1.0, N=10000, lognorm_prefix='./lognorm', seed=1):
    # This functions simulates the restart process with the lognormal distribution.
    # The parameters of the latter two are given in the function head.
    # For reproducibility a seed is used.

    # The optimal restart quantiles of the Lognormal distribution
    # are found numerically by finding the root of the respective minima function.
    # Finding the optimal restart quantiles is based on the methodology discussed in
    # Runtime Distributions and Criteria for Restarts
    # https://link.springer.com/chapter/10.1007/978-3-319-73117-9_35
    eps = np.power(10.0, -50)
    Q = partial(lognorm_Q, sigma=sigma, mu=mu)
    QS = partial(lognorm_QS, sigma=sigma, mu=mu)
    QG = partial(lognorm_QG, sigma=sigma, mu=mu)
    lognorm_exp = lambda p: calculate_expectation(p, Q, QG) if p > eps else np.infty
    lognorm_minima = lambda p: minima_function(p, Q, QS, QG) if p > eps else -eps

    min = 0.0
    max = 0.9999999

    # The following line numerically approximates the root of the minima function.
    # The resulting value therefore approximates the optimal restart quantile.
    quantile_lognorm = bisect(lognorm_minima, min, max)
    if t is None:
        t_lognorm = Q(quantile_lognorm)
    else:
        t_lognorm = t
    print(f"Optimal restart quantile w.r.t. lognormal distribution: {quantile_lognorm}")
    print(f"This corresponds to the restart time {t_lognorm} and the expected value {lognorm_exp(quantile_lognorm)}")
    print()

    scale = np.exp(mu)
    fixed_function = lambda x: restart.restart_lognormal(x, sigma=sigma, scale=scale)
    luby_function = lambda: restart.luby_restart_lognormal(sigma=sigma, scale=scale)
    lognorm_restarts, lognorm_luby = restart_dist(t_lognorm, fixed_function, luby_function,
                                                  file_prefix=lognorm_prefix, N=N, seed=seed)

    print("-------------------")
    print("Lognormal distribution")
    print(f"fixed-cutoff restarts mean {np.mean(lognorm_restarts)}, Luby mean {np.mean(lognorm_luby)}")
    print(f"95% confidence interval for fixed-cutoff mean {sms.DescrStatsW(lognorm_restarts).tconfint_mean()}")
    print(f"95% confidence interval for luby mean {sms.DescrStatsW(lognorm_luby).tconfint_mean()}")

def write_genpareto_to_file(c=1.0, pareto_scale=1.0, t_pareto=np.power(10.0, -6), N=10000, genpareto_prefix='./genpareto', seed=1):
    # The optimal restart quantile of the generalized Pareto distribution approaches zero if c > 0.
    # Finding the optimal restart quantiles is based on the methodology discussed in
    # Runtime Distributions and Criteria for Restarts
    # https://link.springer.com/chapter/10.1007/978-3-319-73117-9_35
    fixed_function = lambda t: restart.restart_genpareto(t, c=c, scale=pareto_scale)
    luby_function = lambda: restart.luby_restart_genpareto(c=c, scale=pareto_scale)
    genpareto_restarts, genpareto_luby = restart_dist(t_pareto, fixed_function, luby_function,
                                                      file_prefix=genpareto_prefix, N=N, seed=seed)

    print("-------------------")
    print("Generalized Pareto distribution")
    print(f"fixed-cutoff restarts mean {np.mean(genpareto_restarts)}, Luby mean {np.mean(genpareto_luby)}")
    print(f"95% confidence interval for fixed-cutoff mean {sms.DescrStatsW(genpareto_restarts).tconfint_mean()}")
    print(f"95% confidence interval for luby mean {sms.DescrStatsW(genpareto_luby).tconfint_mean()}")
