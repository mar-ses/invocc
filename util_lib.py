"""Contains utility procedures for read and processing data."""

import os
import sys
import socket
import warnings

import numpy as np
import pandas as pd
from numpy import log10
from astropy import units, constants as const

from .__init__ import (HOME_DIR, DIST_DIR, UB_MANUAL,
        UB_DETRENDED, LATEST_CAMPAIGN, HP_LOC)
sys.path.append(HOME_DIR)

if socket.gethostname() == 'telesto':
    UB_DETRENDED = "{}/data/local_ubelix".format(HOME_DIR)

import transear.util_lib
from stellar_analysis.catalog_tools import kepler_tools

# TODO: M_star, R_star incorporated here



# Data getters
# ------------

def collect_ir_results(program, N=None, flag_col='dt_flag',
                       snr_lim=None, filter_targets=True,
                       remove_trappist=True, verbose=True,
                       remove_failures=False, snr_col='dt_snr',
                       ignore_epics=False,
                       apply_post_filter=True):
    """Aggregate the target_list results.

    Args:
        target_list_path (str)
        N (int=None): number of subsamples to take
        flag_col (str='dt_flag'): which column to use for sensitivity
            tf_flag, (bls_flag), dt_flag
        snr_lim (float=None): performs an additional snr limit cut
            on the detections
        filter_targets (bool=True): whether to filter the targets
            using kepler_tools.filter_target_list, removing
            epic duplicates, star duplicates, and campaign 19
            duplicates.
        remove_trappist (bool): whether to remove all epics
            belonging to TRAPPIST-1
        verbose (bool)
        remove_failures (bool=False): to remove the runs that crashed,
            i.e. success_flag == False
        snr_col (str='dt_snr'): if applying a post-run snr-limit, which
            column to use to compare to the SNR
        ignored_epics (bool=False): contains the epics or targets to
            specifically ignore (deprecated, contains only gaia_astro=False)

    Return:
        tl: the target_list
        irr: fully concatenated results of injection testing
            columns: found, epic, ....
    """

    if isinstance(program, (list, tuple, np.ndarray)):
        return fuse_ir_results(program, flag_col=flag_col, N=N)

    if not program.startswith('/'):
        tl = pd.read_pickle('{}/{}/target_list.pickle'.format(UB_DETRENDED, program))
    else:
        tl = pd.read_pickle("{}/target_list.pickle".format(program))

    if remove_trappist:
        tl = tl[~tl.epic.isin(kepler_tools.trappist_epics)]

    # BUG TODO: figure out what caused this
    if not tl.irm_flag.dtype == 'bool':
        tl['irm_flag'] = tl.irm_flag.astype(bool)
        warnings.warn("irm_flag was not bool dtype.")
        print("irm_flag was not bool dtype.")

    if not tl.irm_flag.all():
        print("Not all irm_flag are true. Deleting",
              (~tl.irm_flag.astype(bool)).sum())
        tl = tl[tl.irm_flag]
    #tl = tl[tl.irm_flag & ~tl.irm_flag.isnull()]

    # TODO: do I still do this? Or replace with valid_irm/valid_ultracool
    if filter_targets:
        tl = kepler_tools.filter_target_list(tl)

    if apply_post_filter:
        old_len = len(tl)
        tl = post_filter(tl)
        print("Post-filtered: {}/{}".format(
            old_len - len(tl), old_len))

    # Ignore specific epics:
    if ignore_epics:
        ignore_list = pd.read_pickle("/home/marko/data/catalogs/"
                                     "ultracool_epic_ignore.pickle")
        print("Ignoring {}/{} EPIC numbers.".format(
            tl.epic.isin(ignore_list).sum(), len(tl)))
        tl = tl[~tl.epic.isin(ignore_list)]

    # Aggregate the enormous DataFrame with *all* the pickles
    # in irm_results
    ir_result_list = []
    for path in tl[tl.irm_flag].irm_path:
        if not isinstance(path, str) or not os.path.exists(path):
            print(path, "not found or is NaN.")
            continue

        try:
            irm_results = pd.read_pickle(path)
            #print("Num results: {}".format(len(irm_results)))
            irm_results['epic'] = int(path.split('_')[-2])
            ir_result_list.append(irm_results)
        except Exception as e:
            print("Exception: {}".format(str(e)))
            continue

    irr = pd.concat(ir_result_list, ignore_index=True)

    #irr.sort_values(by='epic', inplace=True)
    if snr_lim is None:
        irr['found'] = irr[flag_col].copy().astype(bool)
    elif snr_lim > min(irr[snr_col]):
        # Should mostly be correct in identifying the snr_lim that
        # was used in the runs
        irr['found'] = irr[flag_col].astype(bool) & (irr[snr_col] > snr_lim)
    else:
        print("The artificial SNR limit may be lower than what the "
              "injection recoveries were performed with: {} vs {}"
              "".format(snr_lim, min(irr[snr_col])))
        irr['found'] = irr[flag_col].astype(bool) & (irr[snr_col] > snr_lim)

    for col in ('tf_flag', 'bls_flag'):
        if col in irr.columns:
            irr.pop(col)

    if N is not None:
        raise NotImplementedError

    if verbose and 'success_flag' in irr.columns:
        print("Number of successful runs: {}/{}"
              "".format(irr.success_flag.sum(), len(irr)))

    if 'success_flag' not in irr.columns:
        print("WARNING: success_flag column not found.")
    elif remove_failures:
        irr = irr[irr.success_flag]

    tl.index = range(len(tl))
    irr.index = range(len(irr))

    return [tl, irr]

def fuse_ir_results(program, **collect_kwargs):
    """Fuses multiple programs of irr results.

    Somewhat obsolete but will be kept.

    Args:
        program (list-like)
        **collect_kwargs:
            flag_col='dt_flag', N=None,
            snr_lim=None, filter_targets=True

    Returns:
        tl, irr
    """
    tls = []
    irrs = []

    if isinstance(program, str):
        program = [program]

    for prog in program:
        tl, irr = collect_ir_results(program=prog, **collect_kwargs)
        irrs.append(irr)
        tls.append(tl)

    # Concat them
    irr = pd.concat(irrs, ignore_index=True)

    tl = tls[0]

    # Fuse the target-lists
    targets_matched = True
    tl = tls[0].copy()
    # Check match
    for tli in tls[1:]:
        if tli.epic.isin(tl.epic).all() and tl.epic.isin(tli.epic).all():
            # Rearrange according to epics
            tl.sort_values(by='epic', inplace=True)
            tli.sort_values(by='epic', inplace=True)
            tl['irm_N'] = tl.irm_N + tli.irm_N
            tli.index = range(len(tli))
            targets_matched = True and targets_matched
        else:
            print("Adding", (~tli.epic.isin(tl.epic)).sum())
            tl = tl.append(tli[~tli.epic.isin(tl.epic)], ignore_index=True)
            targets_matched = False
            continue
            raise ValueError("Program target lists don't have matching epics.")

    # If the targets weren't matching, then the irm_N column wouldn't be
    # added properly.
    if not targets_matched:
        for idx in tl.index:
            #import pdb; pdb.set_trace()
            tl.at[idx, 'irm_N'] = ((irr.epic == tl.loc[idx, 'epic'])
                                 & (irr.success_flag)).sum()

    #tl['irm_N'] = np.sum(np.array([tli['irm_N'].values for tli in tls]),
    #                     axis=0)

    return tl, irr


# Utilities
# ---------

# def transit_prob(P, R_p, R_star, M_star):
#     A = transear.util_lib.calc_a(P, M_star, R_star)
#     return (R_star + R_p)/A

def post_filter(tl, weird_dt=True):
    """Last-minute variable filtering to remove targets for paper.
    """

    # Currently: remove the shitty targets with L_bol > 0.1
    # Also, removing all that don't have gaia_astro

    mask = tl.gg_found | ((tl.vosa_L_bol <= 0.1) & (tl.vosa_Teff <= 3000)) 
    print("Removing {} invalid ultracool dwarfs; should be 0.".format(
        (~mask).sum()))
    # tl = tl[tl.vosa_L_bol <= 0.1]
    # tl = tl[tl.vosa_Teff <= 3000]
    # tl = tl[tl.gaia_astro]

    if weird_dt:
        weird_mask = np.log10(tl.dt_rms) > (0.3*tl.gaia_magnitude - 8.1)
        # If they don't have gaia_magnitude, don't remove them
        weird_mask = weird_mask | tl.gaia_magnitude.isnull()
        print("Removing", (~weird_mask).sum(), "targets with anomalously",
              "low noise.")
        tl = tl[weird_mask]

    return tl

def transit_prob(P, R_p, R_star, M_star):
    """Calculates the transit probability of a configuration.

    NOTE - units:
    P (d), R_p (R_earth), R_star (R_star), M_star (M_sun)
    """

    A = transear.util_lib.calc_a(P, M_star, R_star)
    R_pp = (R_p * const.R_earth) / (R_star * const.R_sun)
    t_prob = (1 + R_pp)/A
    # Guards against returning object, which happened when R_star was
    # a Series of objects for some unknown reason
    return t_prob.astype(float)


def generate_grid(log_P=True, logR=False, minP=None, maxP=None,
                  minR=0.75, maxR=5.75, binP=None, binR=None,
                  force_integer_squares=True):
    """Generates a grid for binning in R,P space.

    NOTE: [i,j] gives the value in R_i, P_j

    NOTE: the grid values are ALWAYS in normal units, never log.
        log only determines the distribution of bin boundaries.

    Arguments:
        log_P (bool=False): will be in log10!
        logR (bool=False): will be in log10!
        minP (float=None), maxP (float=None): determines the range
            of the grid in P space; if None will be determined
            automatically. NEVER enter in log units;
            always in real units.
        minR (float=0.5), maxR (floatR=6.0): range of R values,
            also never give it in log; always in normal units.
        binP (int=None), binR (int=None): with of bins, sets the
            number of bins***; it is a guide and if the total range
            is incompatible with the bin-width, it will be the closest.
            If not set, will be chosen based on whether it's log or not.
            NOTE: it's the width in the space chosen, so if log_P, then
            it will be the width in log-space.
        force_integer_squares (bool=False): tries to force the bins to be
            squares in some integer fashion, even if it means that
            one bin must be a different size on the end.
            By default, in normal space, the truncated bin will be at
            the low end for period and radius
            In log units however, it is at the top end.

    Returns:
        P_edges, R_edges: including the endpoint
    """


    if not logR:
        binR = 0.5 if binR is None else binR

        if not force_integer_squares:
            R_edges = split_evenly(minR, maxR, binR)
        else:
            R_edges = split_forced(minR, maxR, binR, fix_start=False)
    else:
        raise NotImplementedError("logR not implemented yet.")

    if not log_P:
        # These are the defaults during generation
        binP = 2 if binP is None else binP
        minP = 0.5 if minP is None else minP
        maxP = 20.0 if maxP is None else maxP

        if not force_integer_squares:
            P_edges = split_evenly(minP, maxP, binP)
        else:
            P_edges = split_forced(minP, maxP, binP, fix_start=False)
    else:
        # binP in log units
        binP = 0.25 if binP is None else binP
        minP = 0.5 if minP is None else minP
        maxP = 20.0 if maxP is None else maxP

        if not force_integer_squares:
            logP_edges = split_evenly(log10(minP), log10(maxP), binP)
            P_edges = 10**logP_edges
        else:
            logP_edges = split_forced(log10(minP), log10(maxP), binP,
                                      fix_start=True)
            P_edges = 10**logP_edges

    return P_edges, R_edges

def split_evenly(min_val, max_val, step):
    """Splits with a step size roughly equal to step."""
    N_steps = int(np.round((max_val - min_val) / step))
    return np.linspace(min_val, max_val, N_steps+1, endpoint=True)

def split_forced(min_val, max_val, step,
                 fix_start=False, remove_small=True):
    """Splits a range into intervals with forced step sizes.

    Args:
        min_val, max_val, step
        fix_start (bool=False): if True, the uneven bin is at the end
        remove_small (bool=True): if True, then instead of having
            smaller bin at the end, if it's less than half the step
            size, it will be enlarged instead.
    """
    val_range = max_val - min_val

    N_steps = int(val_range // step)
    values = np.linspace(0.0, N_steps*step, N_steps+1, endpoint=True)

    if fix_start:
        values = min_val + values
        values = np.append(values, max_val)
    else:
        values = max_val - values
        values = np.insert(values, 0, min_val)

    intervals = np.sort(np.unique(values))

    if remove_small and min(np.diff(intervals)) < 0.5*step:
        if fix_start:
            intervals = np.delete(intervals, -2)
        else:
            intervals = np.delete(intervals, 1)

    return intervals

# Detection sensitivity and completeness calculation
# --------------------------------------------------

# Current fixed bin structure: R in 1.0 from 0.5, P in  5 from 0
# however P should be changed to log_P to keep density constant

def bin_irr_points(irr, weight_col=None, weight_by_N=True,
                   verbose=True, remove_low_runs=False, **grid_kwargs):
    """Bins the found and total points, and calculates the found/total ratio.

    Use as a general utility function.

    NOTE: must not contain duplicates (even if different campaigns).
    Each object_id (or epic) counts as one target.

    Args:
        irr
        weight_col (array): applies the weights to H_found (only)
            if True, uses 't_prob'
        weight_by_N (bool=True): if True, weights each target's contribution
            by the total number of injections
        verbose (bool=True): prints information on the number of runs
            per target
        remove_low_runs (bool=False): removes targets with fewers than
            a certain fraction of the maximum runs; currently 10%
            Keep in mind that there will be fewers runs than expected
            due to applying the outer grid boundaries.
        **grid_kwargs:
            log_P=False, logR=False, minP=None, maxP=None,
            minR=0.5, maxR=6.0, binP=None, binR=None,
            force_integer_squares=False

    Returns:
        completeness, H_found, H_tot, (R_edges, P_edges)
    """

    # TODO: it should already take R_boundaries and P_boundaries,
    # the grid_kwargs should be put into generate_grid in the functions
    # that call on this function.

    # Cutoff: comparing the number of runs per target to the target(s)
    # with the maximum number of runs, those that have a ratio below
    # this will get removed
    MIN_RUN_FRACTION = 0.1

    if not 'P_boundaries' in grid_kwargs.keys() and not \
           'R_boundaries' in grid_kwargs.keys():
        P_edges, R_edges = generate_grid(**grid_kwargs)
    else:
        P_edges, R_edges = (grid_kwargs['P_boundaries'],
                            grid_kwargs['R_boundaries'])

    if 'success_flag' not in irr.columns:
        print("WARNING: success_flag not foound.")
    else:
        irr = irr[irr.success_flag]

    # Remove points outside histogram before we re-weight the targets by
    # number of injections.
    irr = irr[irr.P.between(min(P_edges), max(P_edges))
            & irr.R_p.between(min(R_edges), max(R_edges))].copy()
    #            grid_kwargs['minP'] if 'minP' in grid_kwargs else 0,
    #            grid_kwargs['maxP'] if 'maxP' in grid_kwargs else 1e6) \
    #        & irr.R_p.between(
    #            grid_kwargs['minR'] if 'minR' in grid_kwargs else 0,
    #            grid_kwargs['maxR'] if 'maxR' in grid_kwargs else 1e6)].copy()
    irr_f = irr[irr.found]

    # num_weights (size len(irr)) is applied on all; it represents
    # the weighing by number of injected targets, in cases where
    # due to a job error or target additions, some of the targets
    # have different numbers of runs
    max_runs = 0        # keeps track of largest number of runs

    if 'epic' in irr.columns and 'object_id' not in irr.columns:
        if 'campaign' in irr.columns:
            # Check we don't have the same epic over multiple campaigns
            if irr[~irr[['epic', 'campaign']].duplicated()].epic.duplicated().any():
                raise ValueError("Given multiple campaigns of the same "
                                 "target to bin_irr_points")

        irr = irr.rename(columns={'epic':'object_id'})

    # Unique epic, campaign pairs
    ec_grouped = irr.groupby(by=['object_id'])

    for object_id, _ in ec_grouped:
        tmask = irr.object_id == object_id
        num_runs = tmask.sum()
        irr.loc[tmask, 'run_weight'] = num_runs

        if num_runs > max_runs:
            max_runs = num_runs

    # Turn it into a fraction of runs / max_runs
    irr['run_weight'] = max_runs / irr['run_weight']

    # Remove targets with too few successful runs
    if remove_low_runs:
        remove_mask = irr.run_weight > 1/MIN_RUN_FRACTION

        if verbose and remove_mask.any():
            print("Removing {} epics for having too few runs."
                  "".format(len(irr[remove_mask].object_id.unique())))

        irr = irr[~remove_mask]

    # weights is applying only on the planets that are found;
    # meant to represent the transit probability
    if weight_col is True:
        weights = irr_f['t_prob']
    elif weight_col is not None:
        weights = irr_f[weight_col]
    else:
        weights = np.ones_like(irr_f.P, dtype=float)

    if weight_by_N:
        run_weights = irr.run_weight
    else:
        run_weights = np.ones_like(irr.P, dtype=float)

    H_tot, _, _ = np.histogram2d(irr.P.values, irr.R_p.values,
                                 bins=(P_edges, R_edges),
                                 weights=run_weights)
    H_found, _, _ = np.histogram2d(irr_f.P.values, irr_f.R_p.values,
                                   bins=(P_edges, R_edges),
                                   weights=weights*run_weights[irr.found])

    event_ratio = H_found/H_tot
    event_ratio[np.isnan(event_ratio)] = 0.0

    return event_ratio.T, H_found.T, H_tot.T, (R_edges, P_edges)

def calc_weights(irr):
    """Assigns weights to each point in irr based on histogram."""

    _, _, H_tot, (R_edges, P_edges) = bin_irr_points(irr)

    raise NotImplementedError

def get_completeness(irr, tl=None, **grid_kwargs):
    """

    Args:
        irr (pd.DataFrame)
        tl (pd.DataFrame)
        **grid_kwargs:
            log_P=True, logR=False, minP=None, maxP=None,
            minR=0.75, maxR=5.75, binP=None, binR=None,
            force_integer_squares=True

    Returns:
        completeness, H_found, H_tot, (R_edges, P_edges)
    """

    # TODO: change get_completeness to take R_star etc from irr
    # Temporary values (TODO: after changing everything to force stellar
    # params, add this back in)
    #R_star = 0.1
    #M_star = 0.1
    # TODO: find some (pivot tables?) pandas method that will make this fast
    # notes: index: index in tl, of the epic that corresponds to row in
    # irr, must be a list or array

    if 'M_star' not in irr.columns and tl is not None:
        irr = pd.merge(left=irr, right=tl[['epic', 'M_star', 'R_star']],
                       how='left', on='epic')

    irr['t_prob'] = transit_prob(irr.P, irr.R_p, irr.R_star, irr.M_star)

    return bin_irr_points(irr, True, **grid_kwargs)


# Uncertainty bootstrapping
# -------------------------

def get_grid_overlap(R_p, R_p_lower, R_p_upper, P, P_lower, P_upper,
                     R_boundaries, P_boundaries, N_s=10000):
    """Calculates the uncertainty overlap with the bins in a grid.

    Uses bootstrap draws from the uncertainty distribution (Monte Carlo)

    Args:
        R_p, R_p_lower, R_p_upper:
        P, P_lower, P_upper:
        R_boundaries:
        P_boundaries:
        N_s (int): number of bootstrap samples

    Returns:
        overlap_array (np.array):
    """

    periods = np.random.randn(N_s)
    periods[periods >= 0] *= P_upper
    periods[periods < 0] *= P_lower
    periods += P

    radii = np.random.randn(N_s)
    radii[radii >= 0] *= R_p_upper
    radii[radii < 0] *= R_p_lower
    radii += R_p

    draws = np.array([radii, periods])

    count_grid, _, _ = np.histogram2d(draws[0, :], draws[1, :],
                                      bins=(R_boundaries, P_boundaries))

    count_grid = count_grid / N_s

    return count_grid
