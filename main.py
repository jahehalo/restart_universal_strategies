import numpy as np
from glob import glob
import random
import pyximport; pyximport.install(setup_args={'include_dirs': np.get_include()})

import write_restarts

# Setup the seed and the number of files.
N = 1000
seed = 0
random.seed(seed)


# Setup the parameters for the distributions
# Levy
levy_scales = [100.0, 10.0, 1.0, 0.1, 0.01, 0.001]

# Lognormal
sigmas = [0.5, 1.0, 3.0, 5.0, 7.0, 9.0]
mus = [1.0, 3.0, 5.0, 7.0]
# Genpareto
cs = [0.5, 1.0, 3.0, 5.0, 7.0, 9.0]
scales = [0.001, 0.1, 1.0, 10.0]
t_pareto = np.power(10.0, -4)

# Setup the intended locations for the data.
fixed_tag = 'fixed'
luby_tag = 'luby'
for i in range(len(sigmas)):
    sigma = sigmas[i]
    c = cs[i]
    levy_scale = levy_scales[i]
    levy_prefix = f"levy_scale{levy_scale}"
    for j in range(len(mus)):
        mu = mus[j]
        scale = scales[j]
        lognorm_prefix = f"lognorm_sigma{sigma}_mu{mu}"
        genpareto_prefix = f"genpareto_c{c}_scale{scale}"

        overwrite = False
        # Start with the Levy distribution
        fixed_files = glob(f"{levy_prefix}_{fixed_tag}_seed*.csv")
        luby_files = glob(f"{levy_prefix}_{luby_tag}_seed*.csv")
        if overwrite or (len(fixed_files) == 0) or (len(luby_files) == 0):
            print("write levy")
            print(len(fixed_files), len(luby_files))
            seed = random.randint(1, 2**32-1)
            write_restarts.write_levy_to_file(scale=levy_scale, N=N, levy_prefix=levy_prefix, seed=seed)

        fixed_files = glob(f"{lognorm_prefix}_{fixed_tag}_seed*.csv")
        luby_files = glob(f"{lognorm_prefix}_{luby_tag}_seed*.csv")
        if overwrite or ((len(fixed_files) == 0) and (len(luby_files) == 0)):
            print("write lognormal")
            seed = random.randint(1, 2 ** 32 - 1)
            # The numerical estimation does not work for sigma >= 9.0.
            # This is probably a precision related problem which could be solved.
            # However, here we use an approximation of the optimal restart time.
            if sigma >= 0.0:
                t = np.exp(mu - sigma**2)
            else:
                t = None
            write_restarts.write_lognorm_to_file(t=t, sigma=sigma, mu=mu, N=N, lognorm_prefix=lognorm_prefix, seed=seed)


        fixed_files = glob(f"{genpareto_prefix}_{fixed_tag}_seed*.csv")
        luby_files = glob(f"{genpareto_prefix}_{luby_tag}_seed*.csv")
        if overwrite or ((len(fixed_files) == 0) and (len(luby_files) == 0)):
            print("write genpareto")
            seed = random.randint(1, 2 ** 32 - 1)
            write_restarts.write_genpareto_to_file(c=c, pareto_scale=scale, N=N, genpareto_prefix=genpareto_prefix,
                                                   t_pareto=t_pareto, seed=seed)

