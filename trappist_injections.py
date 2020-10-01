
import emcee
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binom
from corner import corner

from . import util_lib

planets = np.array([[1.12, 1.51],
                    [1.095, 2.42],
                    [0.784, 4.049],
                    [0.910, 6.01],
                    [1.046, 9.20],
                    [1.148, 12.35]])


def basic_insertion_test(irr, tl, log_P=True, add_t1=False, **grid_kwargs):
    """For each target, inserts the trappist planets, and
    takes the highest completeness as the detection chance
    of a full trappist system.

    Args:
        irr
        tl
    """

    # Do this the smart way.
    if 'minR' not in grid_kwargs.keys():
        grid_kwargs['minR'] = 0.5
    if 'maxR' not in grid_kwargs.keys():
        grid_kwargs['maxR'] = 1.75
    if 'binR' not in grid_kwargs.keys():
        grid_kwargs['binR'] = 0.25
    if 'minP' not in grid_kwargs.keys():
        grid_kwargs['minP'] = 1.0
    if 'maxP' not in grid_kwargs.keys():
        grid_kwargs['maxP'] = 20.0
    if 'binP' not in grid_kwargs.keys():
        grid_kwargs['binP'] = 0.10

    comp, _, _, (r_edges, P_edges) = util_lib.get_completeness(irr, tl, log_P=log_P, **grid_kwargs)

    sysdf = pd.DataFrame({'epic':tl.epic, 'campaign':tl.campaign})
    sysdf.index = tl.index.copy()

    for idx in tl.index:
        epic = tl.loc[idx, 'epic']

        subirr = irr[irr.epic == epic]

        comp, _, _, (_, _) = util_lib.get_completeness(subirr, tl,
                                                       log_P=log_P,
                                                       **grid_kwargs)

        dprobs = calc_detection_chance(planets, comp, r_edges, P_edges)
        sysdf.loc[idx, 'sys_dprob'] = max(dprobs)
        sysdf.loc[idx, 'dplanet'] = np.argmax(dprobs)

    # need binomial in python
    if add_t1:
        sysdf = add_trappist(sysdf)
    # BUG TODO

    no_sys_prob = np.prod(1 - sysdf.sys_dprob)

    sys_frac = [1.0, 0.2, 0.05, 0.01]
    sys_find_prob = [1 - np.prod(1 - fr*sysdf.sys_dprob) for fr in sys_frac]

    for frac, prob in zip(sys_frac, sys_find_prob):
        print("If {} of stars have a system,".format(frac),
              "chance of finding at least one: {}".format(prob))

    return sysdf


def fit_sys_prob(add_t1=False, *args, **insertion_kwargs):
    sysdf = basic_insertion_test(*args, **insertion_kwargs)
    sysdf = add_trappist(sysdf) if add_t1 else sysdf
    # custom epic
    sys_found = sysdf.epic == 111111111

    # Log values
    sysdf['log_dprob'] = np.log(sysdf.sys_dprob)
    sysdf['log_ndprob'] = np.log(1 - sysdf.sys_dprob)

    def log_detection_prob(sysoccr):
        if np.any(sysoccr > 1.0) or np.any(sysoccr < 0.0):
            return - np.inf

        # lnprob of the detected systems
        found_lnprob = (sysoccr*sysdf.loc[sys_found, 'log_dprob']).sum()
        # lnprob of the systems with non-detections
        nodet_lnprob = (1 - sysoccr*sysdf.loc[~sys_found, 'log_ndprob']).sum()
        return found_lnprob + nodet_lnprob

    print("Check sys_found, log_detection_prob with values, sysdf")
    import pdb; pdb.set_trace()

    nwalkers = 10
    initial_sysoccr = 0.5 * np.ones(nwalkers, dtype=float)

    sampler = emcee.EnsembleSampler(nwalkers=nwalkers,
                                    dim=1,
                                    lnpostfn=log_binom_prob)


    # def binom_prob(sysoccr):
    #     if np.any(sysoccr > 1.0) or np.any(sysoccr < 0.0):
    #         return - np.inf
    #     found_prob = sysdf.loc[sys_found, 'sys_dprob'].prod()
    #     nodet_prob = (1 - sysdf.loc[~sys_found, 'sys_dprob']).prod()
    #     return found_prob * nodet_prob
        #event_prob_array = np.empty(len(sysdf), dtype=float)
        #event_prob_array[sys_found] = sysoccr * sysdf.sys_dprob
        #event_prob_array[~sys_found] = 1 - sysoccr * sysdf.sys_dprob
        #return np.prod(event_prob_array)

    def log_binom_prob(sysoccr):
        if np.any(sysoccr > 1.0) or np.any(sysoccr < 0.0):
            return - np.inf
        # BUG TODO this ain't right
        found_lnprob = (sysoccr*sysdf.loc[sys_found, 'log_dprob']).sum()
        nodet_lnprob = (1 - sysoccr*sysdf.loc[~sys_found, 'log_ndprob']).sum()
        return found_lnprob + nodet_lnprob

    nwalkers = 10
    initial_sysoccr = 0.5 * np.ones(nwalkers, dtype=float)

    sampler = emcee.EnsembleSampler(nwalkers=nwalkers,
                                    dim=1,
                                    lnpostfn=log_binom_prob)

    pos, _, _ = sampler.run_mcmc(initial_sysoccr, 1000)
    sampler.reset()
    pos, _, _ = sampler.run_mcmc(pos, N=1000)

    samples = sampler.flatchain
    median = np.median(samples)

    fig = corner(samples, labels=['System occurrence rate'], truths=[median])
    plt.show()
    


def add_trappist(sysdf):
    dsens = 0.9
    dprob = dsens * util_lib.transit_prob(1.51, 1.12, 0.117, 0.08)
    print(dprob)
    trappist = pd.Series({'epic':111111111, 'campaign':12,
                          'sys_dprob':dprob, 'dplanet':0})
    import pdb; pdb.set_trace()
    sysdf = sysdf.append(trappist, ignore_index=True)
    return sysdf

        



def calc_detection_chance(planets, comp_grid, r_edges, P_edges):
    pparams = planets.T

    # The transpose kind of "switches" the R, P index
    # even though planets is in [[P_i, R_i], [...], ...]
    R_i = np.digitize(pparams[0], r_edges) - 1
    P_i = np.digitize(pparams[1], P_edges) - 1

    return comp_grid[R_i, P_i]




