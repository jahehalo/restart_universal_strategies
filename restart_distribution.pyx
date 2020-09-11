from scipy.stats import levy
from scipy.stats import genpareto
from scipy.stats import lognorm, geom
import numpy as np
import math

cimport numpy as np

# The global variable index is used for the implementation of Luby's strategy.
index = 0
luby_dict = {1:[1]}

# This makes the simulation reproducible.
def init(seed=1):
    np.random.seed(seed)

# Corresponds to the fixed-cutoff strategy.
cdef long double restart(long double t, rv):
    cdef long double p = rv.cdf(t)
    cdef long double unif = np.random.uniform(0.0, 1.0)
    # The next line samples from a geometric distribution and multiplies its result with t.
    # This corresponds to the time spent with "failed" restarts.
    cdef long double result = t * (math.ceil(np.log1p(-unif)/np.log1p(-p)) - 1.0)
    # Then a sample from a truncated distribution is generated.
    # This corresponds to the last start, which results in a success.
    unif = np.random.uniform(0.0, p)
    result += rv.ppf(unif)
    return result



# The following three functions use the fixed-cutoff strategy on certain distributions:
# The Levy distribution, the generalized Pareto distribution and the lognormal distribution.
cpdef restart_levy(t, scale=1.0):
    return restart(t, levy(scale=scale))


cpdef restart_genpareto(t, c=1.0, scale=1.0):
    return restart(t, genpareto(c, scale=scale, loc=0.0))


cpdef restart_lognormal(t, sigma=1.0, scale=1.0):
    return restart(t, lognorm(sigma, scale=scale, loc=0.0))

# This method resets the Luby sequence to its start.
def reset_luby():
    global index
    index = 0

# Returns the i-th value of the Luby sequence.
def luby(i):
    global luby_dict
    if i in luby_dict:
        return luby_dict[i]
    luby_dict[i] = 2*luby_dict[i-1]
    luby_dict[i].append(2*luby_dict[i-1][-1])
    return luby_dict[i]
# def luby(i):
#     k = (math.log10(i) / math.log10(2))
#     if math.ceil(k) == math.floor(k):
#         return int(np.power(2, k-1))
#     idx = int(np.power(2, math.floor(k)))
#     idx = i - idx + 1
#     return luby(idx)

# Returns the next value of the Luby sequence.
def next_luby():
    global index
    index += 1
    return luby(index)
# def next_luby():
#     global index
#     index += 1
#     return luby(index)

# This method uses Luby's strategy restarts on a given random variable.
def luby_restart(rv):
    reset_luby()
    result = 0.0
    luby_t = next_luby()
    randoms = rv.rvs(size=len(luby_t))
    while np.all(randoms > luby_t):
        result += np.sum(luby_t)
        luby_t = next_luby()
        randoms = rv.rvs(size=len(luby_t))
    for r, t in zip(randoms, luby_t):
        if r <= t:
            return result + r
        result += t
    # while r > luby_t:
    #     result += luby_t
    #     luby_t = next_luby()
    #     r = rv.rvs()
    return result + t


# The following three functions use Luby's strategy on certain distributions:
# The Levy distribution, the generalized Pareto distribution and the lognormal distribution.
def luby_restart_levy(scale=1.0):
    return luby_restart(levy(scale=scale))


def luby_restart_genpareto(c=1.0, scale=1.0):
    return luby_restart(genpareto(c, scale=scale, loc=0.0))


def luby_restart_lognormal(sigma=1.0, scale=1.0):
    return luby_restart(lognorm(sigma, scale=scale, loc=0.0))
