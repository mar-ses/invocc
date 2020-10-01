"""For performing analysis and visualisation requiring calculations."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import matplotlib.colors as colors
#import matplotlib.ticker as ticker
#from mpl_toolkits.axes_grid1 import make_axes_locatable

from . import vislib, util_lib, inverse_tools, occr_fitter

# NOTE: figure scale is controlled in vislib.fig_scale


# Pre-inversion plots
# -------------------

def analyse_irr(irr, tl, show=True, maxR=6.0, **plot_kwargs):
    """Looks at detection sensitivity and completeness.

    Args:
        irr
        **plot_kwargs: R_bin=0.5, P_bin=0.5, log_P=False, logR=False"""

    plot_sensitivity(irr, show=False, maxR=maxR, **plot_kwargs)
    plot_completeness(irr, tl, show=show, maxR=maxR, **plot_kwargs)

def plot_sensitivity(irr, log_P=True, log_sensitivity=True, ax=None,
                     print_values=False, show=True,
                     **grid_kwargs):
    """Calculates the detection sensitivity and plots it.

    Args:
        irr
        log_P
        log_sensitivity
        print_values
        show
        **grid_kwargs: log_P=False, logR=False, minP=None, maxP=None,
            minR=0.5, maxR=6.0, binP=None, binR=None,
            force_integer_squares=False

    Returns:
        ax
    """

    (sgrid, _, _,
    (r_edges, P_edges)) = util_lib.bin_irr_points(irr, log_P=log_P,
                                                  **grid_kwargs)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax = vislib.plot_grid(grid_values=sgrid, x_edges=P_edges,
                          y_edges=r_edges, log_x=log_P,
                          log_values=log_sensitivity,
                          value_label='sensitivity',
                          print_values=print_values,
                          show=False, ax=ax)

    # Additional aesthetics
    ax.set_xlabel('Period, days')
    ax.set_ylabel(r'Radius, $R_\oplus$')

    if show:
        plt.show()
        return ax
    else:
        fig.show()
        return ax

def plot_completeness(irr, tl, log_P=True, log_completeness=True, ax=None,
                      print_mode='none', show=True, **grid_kwargs):
    """Calculates completness and plots it.

    TODO: the R_star and M_star should be in irr not tl so use those.
        The changes are to be done in util_lib.get_completeness
    TODO: at some point the more interesting analysis with SNRs as
        a function of stellar parameters and signal detection
        will need to be done somewhere in the middle of this,
        but do it as a separate function

    Args:
        irr
        tl (pd.DataFrame): temporary for the stellar parameters
            TODO: add that to irr in any case during reading so I
            can ditch this. The changes are to be done in
            util_lib.get_completeness
        log_P
        log_completness
        print_mode
        show
        **grid_kwargs: log_P=False, logR=False, minP=None, maxP=None,
            minR=0.5, maxR=6.0, binP=None, binR=None,
            force_integer_squares=False

    Returns:
        ax
    """

    (cgrid, _, _, (r_edges, P_edges)) = util_lib.get_completeness(
        irr=irr, tl=tl, log_P=log_P, **grid_kwargs)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    ax = vislib.plot_grid(grid_values=cgrid, x_edges=P_edges,
                          y_edges=r_edges, log_x=log_P,
                          log_values=log_completeness,
                          value_label='completeness',
                          print_values=False,
                          show=False, ax=ax)

    # Print the text
    if print_mode in (None, 'none', 'empty', False):
        pass
    elif print_mode in ('value', 'values'):
        vislib.add_text_grid(cgrid, P_edges, r_edges, ax)
    elif print_mode in ('full', 'complete'):
        raise NotImplementedError
    else:
        raise ValueError("print_mode [{}] not recognised.".format(print_mode))

    # Additional aesthetics
    ax.set_xlabel('Period, days')
    ax.set_ylabel(r'Radius, $R_\oplus$')

    if show:
        plt.show()
        return ax
    else:
        fig.show()
        return ax
        
# Inversion
# ---------

def analyse_occr(irr, tl, planets=None, show=False,
                 show_95=True, constrained_log_P=True,
                 burn=None, iters=None, **grid_kwargs):
    """Performs a set of quick plots to analyse the occurrences.

    Args:
        irr, tl
        planets (list=None): which planets to use. Strings also possible
            for 'trappist', 'T1', 'TRAPPIST-1', 'trappist-1' etc and
            'test', 'fake', 'none', and 'current' for the current sample
        show (bool=False): whether to show the plot_marg_occr or not
            TODO: remove this one
        show_95 (bool=True): in the resulting plots, to show the
            95% interval as shading
        constrained_log_P (bool=True): if True, the occurrence rate is
            constant across logP space. Argument to be renamed.
        burn (int=5000): number of burn iterations (if doing emcee),
            currently this is ignored for the pymc-based sample, which
            is the one automatically used when we don't constrain
            occurrence rates to logP
        iters (int=5000): number of iterations for the samplers
        **grid_kwargs: arguments into get_completeness
            Important ones:
            log_P=True, logR=False, minP=None, maxP=None,
            minR=0.75, maxR=5.75, binP=None, binR=None,
            force_integer_squares=True
        """

    tl = tl[tl.epic.isin(irr.epic)]

    if constrained_log_P:
        burn = 20000 if burn is None else burn
        iters = 20000 if iters is None else iters
        thin_factor = 5
        run_splits = 5
    else:
        burn = 50000 if burn is None else burn
        iters = 100000 if iters is None else iters
        thin_factor = 100
        run_splits = 20

    # Default is True; needs to be removed to be passed into the
    # occurrance rate fitter objects
    log_P = grid_kwargs.pop('log_P', True)

    if 'R_boundaries' in grid_kwargs and 'P_boundaries' in grid_kwargs:
        R_boundaries = grid_kwargs.pop("R_boundaries")
        P_boundaries = grid_kwargs.pop("P_boundaries")
    else:
        P_boundaries, R_boundaries = util_lib.generate_grid(**grid_kwargs)

    # comp, _, _, (R_boundaries, P_boundaries) = util_lib.get_completeness(irr, tl,
    #                                                            log_P=log_P,
    #                                                            **grid_kwargs)

    if not isinstance(planets, str):
        pass
    elif planets in ('trappist', 'trappist-1', 'T1', 'TRAPPIST-1', True):
        raise NotImplementedError("Not added for uncertain parameter case.")
        planets = trappist_planets
    elif planets in ('test', 'fake', 'simulate'):
        raise NotImplementedError("Not added for uncertain parameter case.")
        planets = test_planets
    elif planets in ('current', 'latest', 'draft', 'final', 'main'):
        planets = current_planets
    elif planets is None or planets == 'none':
        planets = None
    else:
        raise ValueError("planets code:", planets, "not recognised.")

    if constrained_log_P:
        L = occr_fitter.ConstrainedOCCR(
            irr, planets, R_boundaries, P_boundaries, log_r=False, log_p=log_P)
        L.sample_occr_emcee(burn=burn, iters=iters, threads=2)
    else:
        L = occr_fitter.UncertainBinnedPoissonProcess(
            irr, planets, R_boundaries, P_boundaries, log_r=False, log_p=log_P)
        L.sample_occr_emcee(burn=burn, iters=iters, threads=2)
    L.plot_marg_occr(show=show, show_95=show_95, fix_limits=True)
    L.plot_2d_occr(show=False)

    return L

# Useful data
# -----------

current_planets = pd.DataFrame({'epic':[210490365], 'R_p':[3.43],
                                'R_p_lower':[0.31], 'R_p_upper':[0.95],
                                'P':[3.4846], 'P_lower':[0.0001],
                                'P_upper':[0.0001], 'planet_id':[0]})

# Old precise planet parameters

# planet_dict = {
#     # This one is an issue, 3.4R_e in discovery, like 1.4 with us...
#     # Half the size; why is their stellar estimate so huge
#     210490365:[3.4, 3.48],
#     #'211926133b':[1.99, 11.9],
#     #'211926133c':[1.89, 15.3],
#     # Candidate for removal
#     #246036782:[1.48, 13.4]
# }

# current_planets = np.array([v for v in planet_dict.values()])

trappist_planets = np.array([[1.12, 1.51],
                             [1.095, 2.42],
                             [0.784, 4.049],
                             [0.910, 6.01],
                             [1.046, 9.20],
                             [1.148, 12.35]])

test_planets = np.array([[1.62, 2.45], [1.90, 7.80], [2.42, 4.56],
                         [2.45, 1.43], [2.66, 9.70], [2.94, 15.4]])
test_planets = np.concatenate(
    [[[1.90, 7.80], [2.42, 4.56], [2.45, 1.43], [2.66, 9.70], [2.94, 15.4]],
     list(zip(np.random.uniform(3.0, 4.0, 4), np.random.uniform(1.0, 20, 4))),
     list(zip(np.random.uniform(4.0, 6.0, 8), np.random.uniform(1.0, 20, 8)))],
    )
