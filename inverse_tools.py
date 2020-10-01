"""Statistical model and fitter for occurrence rates."""

import warnings

import emcee
import numpy as np
from numpy import random
import pymc3 as pm
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from mpl_toolkits.axes_grid1 import make_axes_locatable

from corner import corner

from . import vislib

# Jeffreys prior for a Poisson rate parameter:
# 
# p(rate) = sqrt(1/rate)
# 
# Or equivalently, sqrt(rate) has an unnormalised uniform distribution
#
# See:
# PAPERS
# https://en.wikipedia.org/wiki/Jeffreys_prior relevant section
#
# When transformed to log_rate, then it is:
#
# p(rate) = rate**(-3/2)
#

# TODO: more accurate name is OccurrenceModel or OccurrenceModelPoisson
# update this in analysis and other places too
class ObservableOccurrenceBasic(object):
    """Probability object defines the observable occurrence pdf.

    The completeness is treated as binned in a set number of bins
    in period and planet radius. The true occurence is treated as
    binned only in radius in a set number of intervals.

    NOTE: regarding logP vs P; this object doesn't care, at least
          in the calculations. That's an outside consideration.

    NOTE: regarding log_occr, the only place where this is considered
          is during sampling. Thus, likelihood and probability
          calculations will switch between taking log_occr or just
          occr. Everywhere else (i.e volumise_occr), the parameter
          is the occr, and never log_occr.
          Additionally, all output, including MCMC samples, and
          sample_storage, are always in normal form (occr). Never
          in log form. So log_occr only affects the sampling. Really
          apart from potentially convergence issues, it should be
          completely the same.

    TODO: potentially source of error, what happens outside of our grid.
          For example, what if I give an event value outside our grid,
          or in general it would be nice if there was an outside value.
    """

    def __init__(self, R_boundaries, P_boundaries, cpf_value_grid,
                 N_stars, planets=None, fit_log_occr=False,
                 log_p=True, log_r=False):
        """Creates the probability from array of completeness values.

        Completeness grid has a N x M dimension; with N R-bins,
        and M P-bins.

        Occurence rate is binned in radius, in 1 Rj intervals,
        for now.

        Args:
            R_boundaries (np.array, (N+1)-dim): the boundaries of
                the bins in R space (inclusive of top bound)
            P_boundaries (np.array, (M+1)-dim): the boundaries of
                the bins in P space (inclusive of top bound)
            cpf_value_grid (np.ndarray, shape: NxM): the values of the
                *completeness* in each bin,
                [i,j]: i indexes radii, j index period
            N_stars
            planets
            fit_log_occr
            log_p
            log_r
        """

        grid_shape = (len(R_boundaries) - 1, len(P_boundaries) - 1)

        self._R_boundaries = np.array(R_boundaries)
        self._P_boundaries = np.array(P_boundaries)
        self._N_stars = N_stars
        self._grid_shape = grid_shape
        self._log_p_flag = log_p
        self._log_r_flag = log_r
        self._log_occr_flag = fit_log_occr

        # Enter completeness array
        if np.shape(cpf_value_grid) == grid_shape:
            self._cpf_grid = np.array(cpf_value_grid)
        elif np.shape(cpf_value_grid.T) == grid_shape:
            print("Reshaping completeness array.")
            self._cpf_grid = np.array(cpf_value_grid.T)
        else:
            raise ValueError("Array shape of  completeness_value_grid "
                             "doesn't match the expected internal shape "
                             "based on the bin boundary values.")

        # Check that shape of completeness grid is the same as grid_shape
        if not all(dim in np.shape(self._cpf_grid) for dim in self.shape):
            raise ValueError("The completeness grid doesn't match "
                             "the grid shape.")

        # Initiate the occurrence rate in same bins of R as cpf
        # NOTE: never start them at zero, it will crash
        # This initialisation isn't checked for shape
        self.occr_array = np.ones(np.shape(self))

        # Flag; if True, the parameters have been unchanged since
        # the last time the rate integral was calculated; therefore
        # the cached integral is safe to use.
        self._int_cache_flag = False
        self._int_cache = 0.0
        self._sample_storage = None

        # If uncertainties are given, bootstrap the events and reduce
        # the weights
        if np.ndim(planets) == 2 and np.shape(planets)[1] == 2:
            # Where only hard values are given
            event_weight = 1
            event_values = planets
        else:
            # Otherwise, we need a df in the form:
            # R_p, R_p_err_low, R_p_err_high, P, P_err (optional)
            N_s = 1000
            event_weight = 1/N_s
            planet_samples = []

            for idx in planets.index:
                radii, periods = [], []

                if 'P_err' not in planets or planets.P_err.isnull()[idx]:
                    periods = N_s * [planets.loc[idx, 'P']]
                else:
                    periods = random.randn(N_s) * planets.loc[idx, 'P_err']
                    periods += planets.loc[idx, 'P']

                # The radii are taken from a "split" normal
                radii = random.randn(N_s)
                radii[radii >= 0] *= planets.loc[idx, 'R_p_err_high']
                radii[radii < 0] *= planets.loc[idx, 'R_p_err_low']
                radii += planets.loc[idx, 'R_p']

                planet_samples.append(np.column_stack([radii, periods]))

            event_values = np.concatenate(planet_samples)

        # event_values must not be a pandas DataFrame
        self._event_values = event_values
        self._event_weights = event_weight
        # if event_values is None or len(event_values) == 0:
        #     self._event_values = event_values
        # else:
        #     filtered_events = []
        #     for i in range(len(event_values)):
        #         if (self._R_boundaries[0] < event_values[i][0])
        #             and (event_values[i][0] < self._R_boundaries[-1]):
        #             filtered_events.append(event_values[i])
        #     self._event_values = np.concatenate(filtered_events)

    # this may be un-necessary. Avoid using this and perhaps delete it.
    # It's entirely ambiguous what the object call should really return.
    def __call__(self, *args, **kwargs):
        """Calculates the value of the likelihood.

        Args:
            event_values (np.array or float): the R, P pairs to calculate
                the likelihood. If None or empty, calculates it
                assuming zero events. Otherwise, expects a single
                (R, P) coordinate, or N values of (R, P), i.e
                shape (N x 2).
        """

        return self.log_likelihood(*args, **kwargs)

    # Properties and internals
    # ------------------------

    @property
    def shape(self):
        """Gives the shape of the completeness array."""
        # TODO: perhaps this should be a list? That's what the shape
        # normally is.
        return np.array([len(self._R_boundaries)-1, len(self._P_boundaries)-1])

    @property
    def occr_r_names(self):
        """The string names (ranges) of the radius bins."""

        occr_names = []

        for i in range(len(self._R_boundaries) - 1):
            occr_names.append("{} - {}".format(self._R_boundaries[i],
                                               self._R_boundaries[i+1]))

        return occr_names

    @property
    def occr_p_names(self):
        """The string names (ranges) of the period bins."""

        occr_names = []

        for i in range(len(self._P_boundaries) - 1):
            occr_names.append("{:.3g} - {:.3g}".format(
                self._P_boundaries[i], self._P_boundaries[i+1]))

        return occr_names

    def get_occr(self):
        return self.occr_array

    def set_occr(self, array):
        if not np.array_equal(np.shape(array), np.shape(self)):
            # np.all(np.shape(array) == np.shape(self)):
            import pdb; pdb.set_trace()
            raise ValueError("Input array is the wrong shape.")
        elif (array < 0.0).any():
            raise InvalidOccurrenceRate("Negative occurrence rate is invalid.")
        self._int_cache_flag = False
        self.occr_array = array

    def get_log_occr(self):
        return np.log10(self.occr_array)

    def set_log_occr(self, array):
        if not np.array_equal(np.shape(array), np.shape(self)):
            # np.all(np.shape(array) == np.shape(self)):
            import pdb; pdb.set_trace()
            raise ValueError("Input array is the wrong shape.")
        elif not np.isfinite(array).all():
            raise InvalidOccurrenceRate("Negative occurrence rate is invalid.")
        self._int_cache_flag = False
        self.occr_array = 10**array


    def get_event_values(self):
        return self._event_values

    def set_event_values(self, array):
        self._int_cache_flag = False

        if array is None:
            self._event_values = None
        elif np.ndim(array) == 1:
            self._event_values = np.array(array).reshape([1, 2])
        elif (np.ndim(array) == 2) and (np.shape(array)[1] == 2):
            self._event_values = np.array(array)
        else:
            raise ValueError('Invalid entry to event values '
                             '(check the shape).')

    occr = property(get_occr, set_occr)
    log_occr = property(get_log_occr, set_log_occr)
    event_values = property(get_event_values, set_event_values)

    # Probabilistic methods
    # ---------------------

    def likelihood(self, occr_array=None, event_values=None):
        """Calculates the value of the likelihood.

        NOTE: likelihood, not log-likelihood

        Args:
            occr_array (np.array): occurrence rates. If _log_occr_flag
                then give them as log occurrence rates.
            event_values (np.array or float): the R, P pairs to calculate
                the likelihood. If None or empty, calculates it
                assuming zero events. Otherwise, expects a single
                (R, P) coordinate, or N values of (R, P), i.e
                shape (N x 2).
        """

        # BUG UPDATE
        if occr_array is not None and not self._log_occr_flag:
            # Normal occurrence rates
            try:
                self.occr = occr_array
            except InvalidOccurrenceRate:
                # Catch invalid occurrence rates for zero likelihood
                return 0.0
        elif occr_array is not None and self._log_occr_flag:
            # Log occurrence
            try:
                self.log_occr = occr_array
            except InvalidOccurrenceRate:
                # Catch invalid occurrence rates for zero likelihood
                return 0.0

        # if occr_array is not None and not (occr_array < 0.0).any():
        #     self.occr = occr_array
        # elif occr_array is not None and (occr_array < 0.0).any():
        #     # Prevent it at this stage so we don't trying an error
        #     return 0.0

        if event_values is not None:
            self.event_values = event_values

        I = self.calc_integral() * self._N_stars

        # Case of no events
        if self.event_values is None or \
           not hasattr(self.event_values, '__len__'):
            value = np.exp(-I)
        else:
            value = np.exp(-I) * np.prod(self.rate_density(self.event_values))

        # BUG
        if np.isnan(value):
            import pdb; pdb.set_trace()

        # A nan value is possible when some of the occr are too high
        return value if not np.isnan(value) else 0.0

    def log_likelihood(self, occr_array=None, event_values=None):
        """Calculates the value of the likelihood.

        Args:
            occr_array (np.array): occurrence rates. If _log_occr_flag
                then give them as log occurrence rates.
            event_values (np.array or float): the R, P pairs to calculate
                the likelihood. If None or empty, calculates it
                assuming zero events. Otherwise, expects a single
                (R, P) coordinate, or N values of (R, P), i.e
                shape (N x 2).
        """

        # BUG UPDATE
        if occr_array is not None and not self._log_occr_flag:
            # Normal occurrence rates
            try:
                self.occr = occr_array
            except InvalidOccurrenceRate:
                # Catch invalid occurrence rates for zero likelihood
                return -np.inf
        elif occr_array is not None and self._log_occr_flag:
            # Log occurrence
            try:
                self.log_occr = occr_array
            except InvalidOccurrenceRate:
                # Catch invalid occurrence rates for zero likelihood
                return -np.inf

        # if occr_array is not None and not (occr_array < 0.0).any():
        #     self.occr = occr_array
        # elif occr_array is not None and (occr_array < 0.0).any():
        #     # Prevent it at this stage so we don't try an error
        #     return - np.inf

        if event_values is not None:
            self.event_values = event_values

        I = self.calc_integral() * self._N_stars

        # Case of no events
        if self.event_values is None or \
           not hasattr(self.event_values, '__len__'):
            value = - I
        else:
            value = np.sum(np.log(self.rate_density(self.event_values))) - I

        # BUG
        if np.isnan(value):
            import pdb; pdb.set_trace()

        return value if not np.isnan(value) else -np.inf

    def prior(self, occr_array=None):
        """Calculates the prior pdf of the occurrence rates.

        Automatically transforms in the case where log_occr is used.

        Args:
            occr_array (np.array): occurrence rates. If _log_occr_flag
                then give them as log occurrence rates.
        """

        # Deal with the actual input/read it
        if occr_array is not None and not self._log_occr_flag:
            # Normal occurrence rates
            try:
                self.occr = occr_array
            except InvalidOccurrenceRate:
                # Catch invalid occurrence rates for zero likelihood
                return 0.0
        elif occr_array is not None and self._log_occr_flag:
            # Log occurrence
            try:
                self.log_occr = occr_array
            except InvalidOccurrenceRate:
                # Catch invalid occurrence rates for zero likelihood
                return 0.0

        if not self._log_occr_flag:
            return np.prod(np.sqrt(1/self.occr))
        else:
            # Written in terms of occr for ease, also same as:
            # sqrt(occr) = sqrt(10**self.log_occr)
            return np.prod(np.sqrt(self.occr))

    def log_prior(self, occr_array=None):
        """Calculates the prior pdf of the occurrence rates.

        Automatically transforms in the case where log_occr is used.

        Args:
            occr_array (np.array): occurrence rates. If _log_occr_flag
                then give them as log occurrence rates.
        """

        # Deal with the actual input/read it
        if occr_array is not None and not self._log_occr_flag:
            # Normal occurrence rates
            try:
                self.occr = occr_array
            except InvalidOccurrenceRate:
                # Catch invalid occurrence rates for zero likelihood
                return -np.inf
        elif occr_array is not None and self._log_occr_flag:
            # Log occurrence
            try:
                self.log_occr = occr_array
            except InvalidOccurrenceRate:
                # Catch invalid occurrence rates for zero likelihood
                return -np.inf

        if not self._log_occr_flag:
            # NOTE: Ah ah ah! We take log10 for log_occr, but log
            # for the log of a prior or probability density function
            return np.sum(-0.5*np.log(self.occr))
        else:
            # Written in terms of occr for ease.
            value = np.sum(0.5*np.log(self.occr))
            # At this point, it's still possible for occr to be so
            # small that it's underflowing, where value will be nan
            # This should not be possible if _log_occr_flag = False
            return value if not np.isnan(value) else -np.inf

    def log_posterior(self, occr_array=None, event_values=None,
                      flattened_occr=False):
        """Calculates the log posterior of a value array of occr.

        occr_array can be normal or log, which must be reflected in
        self._log_occr_flag

        occr_array can also be flattened so that emcee can use it,
        however we then need to set flattened_occr to True.

        Args:
            occr_array: can be normal or log, which must be reflected in
                self._log_occr_flag. Can also be flattened so that
                emcee can use it, however we then need to set
                flattened_occr to True.
            event_values (np.array or float): the R, P pairs to calculate
                the likelihood. If None or empty, calculates it
                assuming zero events. Otherwise, expects a single
                (R, P) coordinate, or N values of (R, P), i.e
                shape (N x 2).
            flattened_occr (bool=False): if True, it will assume that
                that passed occr was flattened, and will attempt to
                unflatted (ravel) it, through np.reshape.

        Returns:
            log(p(occr|data))
        """

        if flattened_occr and occr_array is not None:
            # Unflatten it into the occurrence rate grid
            occr_array = np.reshape(occr_array, np.shape(self.occr))

        # Let the likelihood and prior actually sub the values in
        log_likelihood = self.log_likelihood(occr_array)
        log_prior = self.log_prior(occr_array)

        # BUG:
        if np.isnan(log_likelihood + log_prior) \
            or (log_likelihood + log_prior) is None:
            import pdb; pdb.set_trace()

        return log_likelihood + log_prior


    # Calculations
    # ------------

    def calc_integral(self):
        """Calculates the in TODO"""

        # Return the cached value if possible
        if self._int_cache_flag:
            return self._int_cache

        #I = np.sum(self.occr * np.sum(self._cpf_grid*self.calc_bin_volumes(),
        #                              axis=1))

        I = np.sum(self.volumise_occr() * self._cpf_grid)

        self._int_cache_flag = True
        self._int_cache = I
        return I

    def calc_bin_volumes(self):
        """Calculates the array of areas (Lebesque measure) per bin.

        If logP, the volume will be in log-space for P.
        """

        if self._log_p_flag:
            P_diffs = np.diff(np.log10(self._P_boundaries))
        else:
            P_diffs = np.diff(self._P_boundaries)

        if self._log_r_flag:
            R_diffs = np.diff(np.log10(self._R_boundaries))
        else:
            R_diffs = np.diff(self._R_boundaries)

        return np.outer(R_diffs, P_diffs)

        #    return np.outer(np.diff(self._R_boundaries),
        #                    np.diff(np.log10(self._P_boundaries)))
        #else:
        #   return np.outer(np.diff(self._R_boundaries),
        #                    np.diff(self._P_boundaries))

    # TODO: should this take occr as a parameter and should it be
    # allowed to take log_occr
    def volumise_occr(self, occr=None):
        """Integrates the occurrence rate over volume.

        NOTE:should be the only thing that's changed between,
        different hyperparametrisations.

        Returns:
            volumised_occr: must be the same shape as the grid shape.
        """
        occr = self.occr if occr is None else occr
        volumised_occr = occr * self.calc_bin_volumes()

        assert np.all(np.shape(volumised_occr)[-2:] == self.shape)
        assert np.array_equal(np.shape(volumised_occr)[-2:], np.shape(self))

        return volumised_occr

    def integrate_over_volume(self, value_array):
        """Multiplies input by self.calc_bin_volumes.

        TODO: currently do not use, DEPRECATED."""
        return value_array * self.calc_bin_volumes()

    def rate_density(self, value):
        """Returns the rate density at a particular value of (R, P).

        Returns: occurrence rate x completeness
        """

        # TODO: analyse for certain that log units cancel out
        # with the change in occr

        if value.ndim == 2:
            value = value.T

        # The transpose kind of "switches" the R, P index
        # even though planets is in [[P_i, R_i], [...], ...]
        R_i = np.digitize(value[0], self._R_boundaries) - 1
        P_i = np.digitize(value[1], self._P_boundaries) - 1

        # Remove the ones out of bounds (oob_mask = out of bounds mask)
        oob_mask = np.zeros_like(R_i, dtype=bool)
        oob_mask = oob_mask | ((R_i < 0) | (R_i >= np.shape(self.occr)[0]))
        oob_mask = oob_mask | ((P_i < 0) | (P_i >= np.shape(self.occr)[1]))

        R_i = R_i[~oob_mask]
        P_i = P_i[~oob_mask]

        return self.occr[R_i, P_i] * self._cpf_grid[R_i, P_i]


    # Estimators
    # ----------

    def predict_rate_grid(self, occr=None, N_stars=None,
                          expected_detection=False):
        """Predicts the rate at each cpf-grid bin."""

        occr = self.occr if occr is None else occr
        N_stars = self._N_stars if N_stars is None else N_stars
        
        if expected_detection:
            # Includes detection sensitivity and geometric transit,
            # i.e the entire completeness
            rate_grid = self.volumise_occr(occr) * self._cpf_grid
            #rate_grid = (self._cpf_grid * self.calc_bin_volumes()).T * occr
        else:
            rate_grid = self.volumise_occr(occr)

        return N_stars * rate_grid

    def marginalise_occr_period(self, occr=None):
        """Marginalises the occurence rate over the range of periods."""

        occr = self.occr if occr is None else occr

        #return occr * self.calc_bin_volumes().sum(axis=1)
        # TODO: check if this is True.
        # i.e: in an inhomogeneous Poisson process, is the total
        # rate in a volume of the space equal to the integral
        # of the rate-function across that volume of space?
        return self.volumise_occr(occr).sum(axis=-1)


    # Sampling and inversion
    # ----------------------

    def sample_occr(self, burn=2000, iters=2000, nwalkers=None,
                    save=True, plot=False):

        flat_shape = np.prod(np.shape(self.occr))
        nwalkers = 2*(flat_shape+1) if nwalkers is None else nwalkers
        occr_initial = random.rand(nwalkers, flat_shape)

        if self._log_occr_flag:
            occr_initial = np.log10(occr_initial)

        sampler = emcee.EnsembleSampler(nwalkers=nwalkers,
                                        dim=flat_shape,
                                        lnpostfn=self.log_posterior,
                                        kwargs={'event_values':None,
                                                'flattened_occr':True})

        # Burn
        pos, _, _ = sampler.run_mcmc(occr_initial, N=burn)
        sampler.reset()

        # BUG
        print("Burn complete without error.")

        # Run
        pos, _, _ = sampler.run_mcmc(pos, N=iters)

        # BUG
        print("Run complete without error.")

        # Extract chains
        #samples = sampler.chain[:, burn:, :].reshape((-1, len(self.occr)))
        samples = sampler.flatchain
        # Need to reshape back into the occr shape
        samples = np.reshape(samples, (-1, *np.shape(self.occr)))

        # Crucial: the samples and output are always in normal form,
        # never in log form
        if self._log_occr_flag:
            samples = 10**samples

        if save:
            self._sample_storage = samples

        # TODO: rmeove this.
        if plot:
            medians = np.median(samples, axis=0)
            hfig = corner(samples, labels=self.occr_r_names, truths=medians)
            hfig.suptitle("Occurrence hyperparameters")
            hfig.show()

            # The occurrences marginalised over period bins
            moccr_samples = self.volumise_occr(samples).sum(axis=-1)
            moccr_medians = np.median(moccr_samples, axis=0)
            mfig = corner(moccr_samples, labels=self.occr_r_names,
                          truths=moccr_medians)
            mfig.suptitle("Marginalised occurrences")
            mfig.show()

        return samples

    def sample_occr_individual(self, iters=2000, nwalkers=2,
                               save=True, plot=False):
        """Samples the occurrence rates individually in each bin."""

        # TODO: make an invert_poisson function that does the MCMC.
        # will need to be transposed to samples of 2d grid
        # But this shape will make it easier to sub the data in
        samples = np.zeros([np.shape(self)[0],
                            np.shape(self)[1],
                            iters*nwalkers])

        volumised_cpf = self._cpf_grid * self.calc_bin_volumes()

        for ix, iy in np.ndindex(*np.shape(self)):
            rm = volumised_cpf[ix, iy] * self._N_stars

            if self._event_values is not None:
                nevents = np.sum(
                      (self._event_values.T[0] > self._R_boundaries[ix]) \
                    & (self._event_values.T[0] < self._R_boundaries[ix+1]) \
                    & (self._event_values.T[1] > self._P_boundaries[iy]) \
                    & (self._event_values.T[1] < self._P_boundaries[iy+1]))
            else:
                nevents = 0

            samples[ix, iy, :] = sample_poisson_rate_pymc(rate_multiplier=rm,
                                                          num_events=nevents,
                                                          iters=iters,
                                                          nchains=nwalkers)

        samples = samples.swapaxes(0, -1).swapaxes(-1, 1)

        if save:
            self._sample_storage = samples

        if plot:
            medians = np.median(samples, axis=0)
            hfig = corner(samples, labels=self.occr_r_names, truths=medians)
            hfig.suptitle("Occurrence hyperparameters")
            hfig.show()

            # The occurrences marginalised over period bins
            moccr_samples = self.volumise_occr(samples).sum(axis=-1)
            moccr_medians = np.median(moccr_samples, axis=0)
            mfig = corner(moccr_samples, labels=self.occr_r_names,
                          truths=moccr_medians)
            mfig.suptitle("Marginalised occurrences")
            mfig.show()

        return samples

    # Plotting and additional
    # -----------------------

    def plot_2d_occr(self, samples=None, show=True, percentage_flag=True,
                     **sampler_kwargs):
        """Plots the 2d occurrence rate. Focus on the distribution.

        Args:
            samples (np.ndarray=None)
            show (bool=True)
            upper_limits (bool=True)
            print_mode (str='dist'): What to print on the array
                None, 'none', False: nothing is printed
                'dist', 'norm', 'uncertainties': gaussian uncertainties
                'detail', 'full', 'predict': paper plot, prints
                    predicted detections, 95% limit, etc...

        Returns:
            fig, ax
        """

        # We still get this error:
        # TypeError: Dimensions of C (5, 11) are incompatible with X (6) and/or Y (12); see help(pcolormesh)

        # ~/invocc/inverse_tools.py in plot_2d_occr(self, samples, show, **sampler_kwargs)
        #     681         else:
        #     682             im = ax.pcolormesh(self._P_boundaries, self._R_boundaries,
        # --> 683                                occr_grid.T)
        #     684             ax.set_xscale('log')
        #
        # TODO: for that, check how it's done in vislib, perhaps we need
        # one less boundary at the end for example.
        #
        # TODO: in any case, update this to use vislib
        #
        # TODO: one option is: to plot the upper bounds or the median
        #       this should be an argument

        # Multiplies by 100 to get percentages
        pfac = 100 if percentage_flag else 1

        if percentage_flag:
            cbar_text = 'occurrence rate limit (%)'
            pfac = 100
        else:
            cbar_text = 'occurrence rate limit'
            pfac = 1

        if samples is None and self._sample_storage is not None:
            samples = self._sample_storage
        elif samples is None:
            samples = self.sample_occr(**sampler_kwargs)

        occr_median = self.predict_rate_grid(np.median(samples, axis=0),
                                             N_stars=1,
                                             expected_detection=False)
        occr_lower = self.predict_rate_grid(np.percentile(samples, 16, axis=0),
                                            N_stars=1,
                                            expected_detection=False)
        occr_upper = self.predict_rate_grid(np.percentile(samples, 84, axis=0),
                                            N_stars=1,
                                            expected_detection=False)
        occr_limit = self.predict_rate_grid(np.percentile(samples, 95, axis=0),
                                            N_stars=1,
                                            expected_detection=False)

        ax = vislib.plot_grid(grid_values=occr_limit*pfac,
                              x_edges=self._P_boundaries,
                              y_edges=self._R_boundaries,
                              log_x=self._log_p_flag,
                              log_values=True,
                              value_label=cbar_text,
                              print_values=False,
                              show=False,
                              truncated_cmap=True)

        # Add the text
        # ------------

        # Upper limits
        ulim_text = np.empty_like(occr_limit, dtype=object)
        for ix, iy in np.ndindex(*np.shape(ulim_text)):
            # ulim_text[ix, iy] = r"{:.2g}".format(occr_limit[ix, iy]*pfac)
            ulim_text[ix, iy] = np.format_float_positional(
                occr_limit[ix, iy]*pfac, precision=2, fractional=False)
        vislib.add_text_grid(ulim_text,
                             self._P_boundaries,
                             self._R_boundaries,
                             ax=ax, square_offset=[0.98, 0.95],
                             horizontalalignment='right',
                             verticalalignment='top',
                             size_factor=1,
                             color='red')

        # Median
        med_text = np.empty_like(occr_median, dtype=object)
        for ix, iy in np.ndindex(*np.shape(med_text)):
            # med_text[ix, iy] = r"{:.2g}".format(occr_median[ix, iy]*pfac)
            med_text[ix, iy] = np.format_float_positional(
                occr_median[ix, iy]*pfac, precision=2, fractional=False)
        vislib.add_text_grid(med_text,
                             self._P_boundaries,
                             self._R_boundaries,
                             ax=ax, square_offset=[0.45, 0.05],
                             horizontalalignment='right',
                             verticalalignment='bottom',
                             size_factor=1)

        # +-
        pm_text = np.empty_like(occr_upper, dtype=object)
        for ix, iy in np.ndindex(*np.shape(pm_text)):
            pm_text[ix, iy] = r"$^{{\,+{}}} _{{\,-{}}}$".format(
                np.format_float_positional(occr_upper[ix, iy]*pfac,
                                           precision=1, fractional=False),
                np.format_float_positional(occr_lower[ix, iy]*pfac,
                                           precision=1, fractional=False))
        vislib.add_text_grid(pm_text,
                             self._P_boundaries,
                             self._R_boundaries,
                             ax=ax, square_offset=[0.42, 0.01],
                             horizontalalignment='left',
                             verticalalignment='bottom',
                             size_factor=1*1)

        # Predicted detections with occr=1
        pred_text = np.empty_like(occr_median, dtype=object)
        for ix, iy in np.ndindex(*np.shape(pred_text)):
            # pred_text[ix, iy] = r"{:.2g}".format(
            #     self._cpf_grid[ix, iy]*self._N_stars)
            pred_text[ix, iy] = np.format_float_positional(
                self._cpf_grid[ix, iy]*self._N_stars,
                precision=2, fractional=False)
        vislib.add_text_grid(pred_text,
                             self._P_boundaries,
                             self._R_boundaries,
                             ax=ax, square_offset=[0.03, 0.95],
                             horizontalalignment='left',
                             verticalalignment='top',
                             size_factor=1)

        # occr_text = np.empty_like(occr_median, dtype=object)

        # for ix, iy in np.ndindex(*np.shape(occr_text)):
        #     occr_text[ix, iy] = r"${:.2g} ^{{\,{:.1g}}} _{{\,{:.1g}}}$".format(
        #         occr_median[ix, iy], occr_upper[ix, iy], occr_lower[ix, iy])

        # vislib.add_text_grid(occr_text,
        #                      self._P_boundaries,
        #                      self._R_boundaries,
        #                      ax=ax, square_offset=[0.1, 0.05])

        # Additional aesthetics
        ax.set_xlabel('Period, days')
        ax.set_ylabel(r'Radius, $R_\oplus$')

        fig = ax.figure

        if show:
            plt.show()
        else:
            fig.show()

        return fig, ax

    def plot_2d_occr_detail(self, samples=None, show=True,
                            upper_limits=True, **sampler_kwargs):
        """Plots the 2d occurrence rate with different details.

        Focus on upper limit, expected number of detections, etc...

        Returns:
            fig, ax
        """

        # We still get this error:
        # TypeError: Dimensions of C (5, 11) are incompatible with X (6) and/or Y (12); see help(pcolormesh)

        # ~/invocc/inverse_tools.py in plot_2d_occr(self, samples, show, **sampler_kwargs)
        #     681         else:
        #     682             im = ax.pcolormesh(self._P_boundaries, self._R_boundaries,
        # --> 683                                occr_grid.T)
        #     684             ax.set_xscale('log')
        #
        # TODO: for that, check how it's done in vislib, perhaps we need
        # one less boundary at the end for example.
        #
        # TODO: in any case, update this to use vislib
        #
        # TODO: one option is: to plot the upper bounds or the median
        #       this should be an argument

        if samples is None and self._sample_storage is not None:
            samples = self._sample_storage
        elif samples is None:
            samples = self.sample_occr(**sampler_kwargs)

        occr_median = np.median(samples, axis=0)
        occr_lower = np.percentile(samples, 16, axis=0)
        occr_upper = np.percentile(samples, 84, axis=0)
        occr_limit = np.percentile(samples, 95, axis=0)

        occr_grid = self.predict_rate_grid(occr_median,
                                           N_stars=1,
                                           expected_detection=False)

        fig, ax = plt.subplots()

        if not self._log_p_flag:
            im = ax.matshow(occr_grid, origin='lower',
                            extent=(self._P_boundaries[0], 
                                    self._P_boundaries[-1],
                                    self._R_boundaries[0],
                                    self._R_boundaries[-1]))
        else:
            import pdb; pdb.set_trace()
            im = ax.pcolormesh(self._P_boundaries, self._R_boundaries,
                               occr_grid)
            ax.set_xscale('log')

        # Colour bar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="6%", pad=0.3)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("occurrence rate")

        fig.suptitle('Occurrence rate (within bin)')
        #ax.set_xticklabels(self._R_boundaries - 0.25)
        #ax.set_yticklabels(self._P_boundaries - 2.50)
        ax.set_xlabel('Period, days')
        ax.set_ylabel(r'Radius, $R_\oplus$')
        ax.set_aspect('auto')
        ax.tick_params('both', reset=True, which='major', direction='inout',
                       bottom=True, top=False, left=True, right=False,
                       length=12, width=1, zorder=10)
        ax.tick_params('both', reset=True, which='minor', direction='inout',
                       bottom=True, top=False, left=True, right=False,
                       length=8, width=0.5, zorder=10)

        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

        if show:
            plt.show()
        else:
            fig.show()

        return fig, ax

    def plot_marg_occr(self, samples=None, show_95=True,
                       show=True, fix_limits=True,
                       print_values=True, **sampler_kwargs):
        """Plots the marginalised occurrence rate in R-space.

        Returns:
            fig, ax
        """
        if samples is None and self._sample_storage is not None:
            samples = self._sample_storage
        elif samples is None:
            samples = self.sample_occr(**sampler_kwargs)

        #moccr_samples = samples * self.calc_bin_volumes().sum(axis=1)
        moccr_samples = self.volumise_occr(samples).sum(axis=-1)
        # better way
        moccr_samples = self.marginalise_occr_period(samples)

        moccr_med = np.median(moccr_samples, axis=0)
        moccr_low = np.percentile(moccr_samples, 16, axis=0)
        moccr_high = np.percentile(moccr_samples, 84, axis=0)

        # Prepare axis centering
        R_mid = list(self._R_boundaries) + [max(self._R_boundaries) + 0.5]
        moccr_mplt = [1e-5] + list(moccr_med) + [1e-5]
        moccr_yerr = np.array([moccr_med - moccr_low, moccr_high - moccr_med])

        def fillbounds(x, R_bounds, moccr):
            y = np.zeros_like(x)
            for i in range(len(moccr)):
                try:
                    mask = ((x > R_bounds[i]) & (x <= R_bounds[i+1]))
                except TypeError:
                    import pdb; pdb.set_trace()
                y[mask] = moccr[i]
            return y

        # Do the histograms-type plot
        plt.rcParams['figure.figsize'] = [5*vislib.fig_scale,
                                          3*vislib.fig_scale]
        fig, ax = plt.subplots()

        ax.set_yscale('log')
        ax.step(R_mid, moccr_mplt, 'k')
        #ax.errorbar(np.array(R_mid[1:-1]) - 0.25, moccr_med,
        #			xerr=None, yerr=moccr_yerr, ecolor='0.5',
        #			linestyle="None", marker="None")
        x = np.array(list(self._R_boundaries - 0.0001)
                   + list(self._R_boundaries + 0.0001))
        x.sort()

        ax.fill_between(x, fillbounds(x, self._R_boundaries, moccr_low),
                        fillbounds(x, self._R_boundaries, moccr_high),
                        color='0.5', alpha=0.5)

        if show_95:
            # BUG: if we want 95% confidence on upper limit, we don't
            # need lower limit and upper limit is at 95th percentile
            moccr_2low = np.percentile(moccr_samples, 5, axis=0)
            moccr_2high = np.percentile(moccr_samples, 95, axis=0)
            ax.fill_between(x, fillbounds(x, self._R_boundaries, moccr_2low),
                            fillbounds(x, self._R_boundaries, moccr_2high),
                            color='0.8', alpha=0.5, zorder=-4)

        if fix_limits:
            ax.set_ylim(0.008, 5)
            ax.set_xlim(self._R_boundaries[0], self._R_boundaries[-1])

        ax.set_xlabel(r"Radius, $R_\oplus$")
        ax.set_ylabel("Occurrence rate, per star")
        ax.tick_params('both', reset=True, which='major', direction='inout',
                       bottom=True, top=False, left=True, right=False,
                       length=12, width=1, zorder=10)
        ax.tick_params('both', reset=True, which='minor', direction='out',
                       bottom=True, top=False, left=True, right=False,
                       length=5, width=0.5, zorder=10)

        # Print the values
        if print_values:
            moccr_df = pd.DataFrame(columns=self.occr_r_names, data=moccr_samples)
            print(moccr_df.describe(percentiles=[0.25, 0.50, 0.75, 0.95]))

        if show:
            plt.show()
        else:
            fig.show()

        return fig, ax

    def split_into_classes(self, bounds=None, percentile=95):
        """Calculates the 95% upper limit in different planet classes.

        Classes:
        - super-Earth: 1.00 - 2.00
        - sub-Neptune: 2.00 - 4.00
        - gas giant: 4.00+

        Give in periods, and integrated over periods.

        Args:
            bounds (list-like=[1.25, 2.25, 3.75, 5.75]): the boundaries
                between the planet classes

        Returns:
            fully_integrated_df, split_df
        """

        if self._sample_storage is not None:
            occr = self.volumise_occr(self._sample_storage)
            moccr = self.marginalise_occr_period(self._sample_storage)
        else:
            raise AttributeError("No stored samples found.")

        if bounds is None:
            bounds = []
            for b in [1.25, 2.25, 3.75, 5.75]:
                if b in self._R_boundaries:
                    bounds.append(b)
            for b in [1.0, 2.0, 4.0, 6.0]:
                if b in self._R_boundaries:
                    bounds.append(b)

        bound_pairs = []
        bound_args = []

        for i in range(len(bounds) - 1):
            bound_pairs.append([bounds[i], bounds[i+1]])
            bound_args.append(
                [np.argwhere(self._R_boundaries == bound_pairs[-1][0])[0, 0],
                 np.argwhere(self._R_boundaries == bound_pairs[-1][1])[0, 0]])

            if pd.isnull(bound_args[-1]).any():
                raise ValueError("Boundary not found; given: {}, found:{}"
                                 "".format(bound_pairs[-1], bound_args[-1]))

        # Add two more bound pairs
        if 1.5 in self._R_boundaries:
            bound_pairs.append([1.5, 2.0])
            bound_args.append(
                [np.argwhere(self._R_boundaries == 1.5)[0, 0],
                 np.argwhere(self._R_boundaries == 2.0)[0, 0]])
            bound_pairs.append([1.5, 4.0])
            bound_args.append(
                [np.argwhere(self._R_boundaries == 1.5)[0, 0],
                 np.argwhere(self._R_boundaries == 4.0)[0, 0]])

        bound_pairs.append([bound_pairs[0][0], bound_pairs[1][1]])
        bound_args.append([bound_args[0][0], bound_args[1][1]])

        # period_pairs = []
        # for i in range(len(self._P_boundaries) - 1):
        period_pairs = self.occr_p_names

        full_df = pd.DataFrame(index=range(np.shape(moccr)[0]))
        #split_df = pd.DataFrame()
        #    columns=["{}-{}".format(*bp) for bp in bound_pairs])

        for i, ((b1, b2), (a1, a2)) in enumerate(zip(bound_pairs, bound_args)):
            full_df["{}-{}".format(b1, b2)] = moccr[:, a1:a2].sum(axis=-1)

        full_df = full_df.describe(percentiles=[0.16, 0.50, 0.84,
                                                0.25, 0.75, 0.95])

        for i, ((b1, b2), (a1, a2)) in enumerate(zip(bound_pairs, bound_args)):
            int_occr = occr[:, a1:a2, :].sum(axis=1)

            for j, pp in enumerate(period_pairs):
                ulim = np.percentile(int_occr[:,j], percentile)
                full_df.loc[pp, "{}-{}".format(b1, b2)] = ulim

        # The med+-sig rate at the planet discovery
        if len(self.occr_p_names) > 1:
            for i, ((b1, b2), (a1, a2)) in enumerate(zip(bound_pairs,
                                                         bound_args)):
                int_occr = occr[:, a1:a2, :].sum(axis=1)

                j = 0
                for check_P in self._P_boundaries[1:]:
                    if check_P > 3.49:
                        break
                    else:
                        j += 1

                pp = 'mean: ' + self.occr_p_names[j]

                # try:
                #     j = np.argwhere(
                #         np.isclose(self._P_boundaries, 5, atol=0.001))[0,0] - 1
                #     pp = 'mean' + self.occr_p_names[j]
                # except IndexError:
                #     continue

                med = np.percentile(int_occr[:,j], 50)
                low = np.percentile(int_occr[:,j], 16)
                hi = np.percentile(int_occr[:,j], 84)

                full_df.loc[pp, '{}-{}'.format(b1, b2)] = \
                    "{:.3g}-{:.3g}+{:.3g}".format(med, med-low, hi-med) 


        #full_df = pd.concat([full_df, split_df])

        return full_df


# Answer: yes, it's actually the one that was being used. And it has
# the correct model for period (as in integrating in log period).
class ObservableOccurrenceLogP(ObservableOccurrenceBasic):
    """Assumes constant occurrence across logP.

    NOTE: the mechanism is that the internally saved occr,
    in occr_array, is now saved as a 1D array (across radii).
    So internally, there is no more 2D occurrence rate.
    However, what happens is that when volumised, or when
    rates are calculated, they are done as if the occr is 2D,
    and for example volumised occr is returned as a 2D array.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        #raise NotImplementedError("Check whether this needs updating.")

        # This initialisation isn't checked for shape
        # TODO: Isn't this already inherited?
        self.occr_array = np.ones(self.shape[0])

    # Aren't the following inherited?
    def get_occr(self):
        return self.occr_array

    def set_occr(self, array):
        if not np.array_equal(np.shape(array), np.shape(self)[:1]):
            # np.all(np.shape(array) == np.shape(self)[0]):
            import pdb; pdb.set_trace()
            raise ValueError("Input array is the wrong shape.")
        elif (array < 0.0).any():
            raise InvalidOccurrenceRate("Negative occurrence rate is invalid.")
        self._int_cache_flag = False
        self.occr_array = array

    def get_log_occr(self):
        return np.log10(self.occr_array)

    def set_log_occr(self, array):
        if not np.array_equal(np.shape(array), np.shape(self)[:1]):
            # np.all(np.shape(array) == np.shape(self)):
            import pdb; pdb.set_trace()
            raise ValueError("Input array is the wrong shape.")
        elif not np.isfinite(array).all():
            raise InvalidOccurrenceRate("Negative occurrence rate is invalid.")
        self._int_cache_flag = False
        self.occr_array = 10**array

    occr = property(get_occr, set_occr)
    log_occr = property(get_log_occr, set_log_occr)

    def volumise_occr(self, occr=None):
        """Integrates the occurrence rate over volume.

        NOTE:should be the only thing that's changed between,
        different hyperparametrisations.

        Args:
            occr: this is not and never should be the log_occr!

        Returns:
            volumised_occr: must be the same shape as the grid shape.
        """
        occr = self.occr if occr is None else occr

        if np.ndim(occr) == 1:
            volumised_occr = occr[:,None] * self.calc_bin_volumes()
        elif np.ndim(occr) == 2:
            volumised_occr = occr[:,:,None] * self.calc_bin_volumes()
        else:
            raise ValueError("Unexpected shape of occr.")

        assert np.array_equal(np.shape(volumised_occr)[-2:], np.shape(self))
        return volumised_occr

    def rate_density(self, value):
        """Returns the rate density at a particular value of (R, P)."""

        # TODO: analyse for certain that log units cancel out
        # with the change in occr

        if value.ndim == 2:
            value = value.T

        R_i = np.digitize(value[0], self._R_boundaries) - 1
        P_i = np.digitize(value[1], self._P_boundaries) - 1

        # Remove the ones out of bounds (oob_mask = out of bounds mask)
        oob_mask = np.zeros_like(R_i, dtype=bool)
        oob_mask = oob_mask | ((R_i < 0) | (R_i >= np.shape(self.occr)[0]))
        oob_mask = oob_mask | ((P_i < 0) | (P_i >= len(self._P_boundaries)-1))

        R_i = R_i[~oob_mask]
        P_i = P_i[~oob_mask]

        return self.occr[R_i] * self._cpf_grid[R_i, P_i]

    def sample_occr_individual(self, iters=2000, nwalkers=2,
                               save=True, plot=False):
        raise NotImplementedError("Not implemented for the fixed "
                                  "log(P) occurrence rate model. "
                                  "It is not needed.")


# Utility and work functions
# --------------------------

def sample_poisson_rate_emcee(rate_multiplier, num_events,
                              burn=5000, iters=5000,
                              nwalkers=6, log_rate=True):
    """Fits a single poisson rate (with a multiplier), in a single bin.

    NOTE: this filth doesn't work. No clue, check affine invariant
    technique, perhaps it requires more than 1 dimension, though
    if that were the case it would be nice if they provided at
    least a warning. In any case, this would have been overkill anyway,
    can practically use a basic rejection sampler for this.
    What a waste of time.

    Args:
        rate_multiplier (float): completeness x area_integral x N_stars
            actual event rate = lamba*rate_multiplier.
            Should be a constant.
        num_events (int): number of detected events
        burn (int): number of burn iterations
        iters (int): number of iterations
        nwalkers (int): number of emcee walkers
        log_rate (bool=True): performs the sampling in log-space,
            but never returns the log of occurrence rate.

    Returns:
        rate_samples (np.array): never in log-form
    """

    warnings.warn("This doesn't work!")

    sampler = emcee.EnsembleSampler(nwalkers=nwalkers,
                                    dim=1,
                                    lnpostfn=poisson_log_posterior,
                                    kwargs={'rate_multiplier':rate_multiplier,
                                            'num_events':num_events,
                                            'log_rate':log_rate})

    # Burn
    pos, _, _ = sampler.run_mcmc(np.ones([nwalkers, 1])*0.1, N=burn)
    sampler.reset()
    pos, _, _ = sampler.run_mcmc(pos, N=iters)

    #import pdb; pdb.set_trace()

    samples = sampler.flatchain[:, 0]
    samples = np.exp(samples) if log_rate else samples

    return samples

def sample_poisson_rate_pymc(rate_multiplier, num_events,
                             iters=1000, nchains=2):
    """Fits a single poisson rate (with a multiplier), in a single bin.

    NOTE: always sampling by log(rate), always returning rate.
    NOTE: actually maybe not; HMC shouldn't need to have it
    transformed into log-space.

    Args:
        rate_multiplier (float): completeness x area_integral x N_stars
            actual event rate = lamba*rate_multiplier.
            Should be a constant.
        num_events (int): number of detected events
        burn (int): number of burn iterations
        iters (int): number of iterations
        nwalkers (int): number of emcee walkers

    Returns:
        rate_samples (np.array): never in log-form
    """

    pmodel = pm.Model()

    with pmodel:
        # Prior for rate, at the moment, use a transformation trick
        # I want rate distributed as rate**-0.5
        # So if u ~ Uniform, then v = Cu**2 is v ~ v**-0.5.
        # Should recheck the maths.
        u = pm.Uniform('u', 0, 10)
        rate = u**2 * rate_multiplier
        events = pm.Poisson('events', mu=rate, observed=num_events)

        # u = pm.Uniform('u', 0, 10)

        # events = pm.Poisson('events', mu=u, observed=num_events)

    map_estimate = pm.find_MAP(model=pmodel)

    with pmodel:
        trace = pm.sample(iters, chains=nchains)

    return trace['u']**2

def poisson_log_posterior(rate, rate_multiplier, num_events, log_rate):
    """Calculates a single poisson posterior pdf.

    If log_rate, then we are sampling rate in log_space and the prior
    is transformed. It is them assumed that rate is actually log(rate).
    NOTE: here it is assumed that we are doing log(rate) not log10(rate)
    """

    if log_rate:
        log_prior = 0.5*rate
    else:
        log_prior = -0.5*np.log(rate)

    rate_total = rate_multiplier * (np.exp(rate) if log_rate else rate)

    log_likelihood = -rate_total + num_events * np.log(rate_total) \
                   - np.log(np.math.factorial(num_events))

    log_post = log_likelihood + log_prior

    if np.isscalar(log_post):
        log_post = log_post if np.isfinite(log_post) else -np.inf
    else:
        log_post[~np.isfinite(log_post)] = -np.inf

    return log_post


# Exceptions
# ----------

class InvalidOccurrenceRate(ValueError):
    pass


