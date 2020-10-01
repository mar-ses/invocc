"""Statistical model for poisson point process with uncertain events."""

import warnings

import emcee
import numpy as np
import pymc3 as pm
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from scipy import optimize

from corner import corner

from . import vislib, analysis, util_lib

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

# Objects leads the creation of the F_ijk and H_ij arrays, calling on
# external functions.
# Always fit log_occr.
# Allow choice in whether to assume constant in log_p/log_r; within a
# bin only.

# Changes to make from previous object:
# _log_occr_flag: not present, assume it's True in all cases
# _cpf_grid -> _int_cpf_grid (?)
#

# Tensorflow
# ----------
#
# Working with tensorflow is currently incomplete.
# The idea:
#
# When using tensorflow for NUTS/autodiff or whatever, the .occr array
# will be a tf.Tensor. Therefore, all the object internals need to work
# with this possiblity (the results from sampling will be converted back
# to numpy arrays).
# 
# However; need to make sure that in that case, the
# operations done on .occr (i.e in likelihoods, priors, calc_integral,
# volumise_occr, and so on...) work well with tensorflow. The
# multiplying arrays may need to be turned into tf.constants.
# 
# I also need to make sure that the operations don't add an ever
# increasing number of nods to the operations graph.
#
# Another thing is that hopefully, if log_likelihood is called multiple
# times, we don't add multiple versions of the same operations to the
# same graph. Figure out what it means to make the GradientTape
# persistent.

class UncertainBinnedPoissonProcess(object):
    """Poisson process model for occurrence rates.

    NOTE regarding occr and log-occr: for the probability functions,
    i.e likelihood, prior, and so on; we always use only log-occr as
    an input parameter. For internal operations like calculating N_exp,
    volumising the rate, calculate rates at points, and so on,
    use log_occr for now, although keep track of this.

    NOTE sampling is done in log-occr, results are given in occr. Same
    for storage.
    """

    def __init__(self, irr, planets, R_boundaries, P_boundaries,
                 log_r=False, log_p=True):
        """Sets up grid, completeness values and pre-calculated arrays.

        Completeness grid has a N x M dimension; with N R-bins,
        and M P-bins.

        Args:
            irr (pd.DataFrame): injection-recovery results; must include
                columns: 'object_id', 'R_p', 'P',
            planets (pd.DataFrame): planet detections, must include
                columns: 'object_id', 'planet_id', 'R_p', 'P', '
            R_boundaries (np.array, (N+1)-dim): the boundaries of
                the bins in R space (inclusive of top bound)
            P_boundaries (np.array, (M+1)-dim): the boundaries of
                the bins in P space (inclusive of top bound)
            fit_log_occr
            log_p
            log_r
        """

        # There must be an 'object_id' column to count individual targets,
        # although 'epic' will also be accepted and substituted, if
        # 'object_id' is not found in irr. Planets must also contain the
        # matching 'object' id for each target.

        # Potentially cpf_grid to H_array, and overlap_array to F_array

        # Set up the grid and attributes
        # ------------------------------

        if 'object_id' not in irr.columns:
            irr = irr.rename(columns={'epic':'object_id'})
        if 'object_id' not in planets.columns:
            planets = planets.rename(columns={'epic':'object_id'})
        if 'planet_id' not in planets.columns:
            planets['planet_id'] = range(len(planets))

        self._R_boundaries = np.array(R_boundaries)
        self._P_boundaries = np.array(P_boundaries)
        self._grid_shape = (len(R_boundaries) - 1, len(P_boundaries) - 1)
        self._log_p_flag = log_p
        self._log_r_flag = log_r
        self._N_stars = len(irr.object_id.unique())

        planets = planets[planets.object_id.isin(irr.object_id)]
        planet_indexes = pd.DataFrame({'object_id':planets.object_id,
                                       'planet_id':planets.planet_id})

        H_array = np.empty((len(planets),) + self._grid_shape, dtype=float)
        F_array = np.empty((len(planets),) + self._grid_shape, dtype=float)

        # Precalculate completeness grids (H_ij)
        # --------------------------------------

        # Integrated completeness grid
        integrated_cpf_grid, _, _, _ = util_lib.get_completeness(
            irr=irr, R_boundaries=R_boundaries, P_boundaries=P_boundaries)

        # Individual planet-hosting target grids
        for i, idx in enumerate(planet_indexes.index):
            planet_indexes.loc[idx, 'H_index'] = i
            H_array[i], _, _, _ = util_lib.get_completeness(
                irr=irr, R_boundaries=R_boundaries, P_boundaries=P_boundaries)

        # Precalculated uncertainty overlaps (F_ijk)
        # ------------------------------------------

        for i, idx in enumerate(planet_indexes.index):
            planet_indexes.loc[idx, 'F_index'] = i
            F_array[i] = util_lib.get_grid_overlap(
                R_boundaries=R_boundaries, P_boundaries=P_boundaries,
                **planets.loc[idx, ['R_p', 'P', 'R_p_lower', 'R_p_upper',
                                    'P_lower', 'P_upper']])

        planet_indexes['F_index'] = planet_indexes.F_index.astype(int)
        planet_indexes['H_index'] = planet_indexes.H_index.astype(int)
        # This is currently unused; I guess I wrote it all previously
        # with the assumption that the H, F, etc... arrays were all
        # aligned as arrays, and that they remained unchanged.
        self._planet_indexes = planet_indexes

        # Initiate the occurrence rate in same bins of R as cpf
        # NOTE: never start them at zero, it will crash
        # This initialisation isn't checked for shape, and must not be
        # referred to by the object; always use self.occr,
        # self.log_occr
        self._occr_array = np.ones(np.shape(self))
        self.H_bar_array = np.array(integrated_cpf_grid)
        self.H_array = H_array
        self.F_array = F_array

        # Flag; if True, the parameters have been unchanged since
        # the last time the rate integral was calculated; therefore
        # the cached integral is safe to use.
        self._int_cache_flag = False
        self._int_cache = 0.0
        self._sample_storage = None

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
        """Returns the occurrence rate grid."""
        return self._occr_array

    def set_occr(self, array):
        """Set the occurrence rate grid."""
        if not np.array_equal(np.shape(array), np.shape(self)):
            # np.all(np.shape(array) == np.shape(self)):
            import pdb; pdb.set_trace()
            raise ValueError("Input array is the wrong shape.")
        elif tf.is_tensor(array):
            # Placeholder: not sure how to check for finiteness with a
            # tensorflow Variable (TODO)
            pass
        elif (array < 0.0).any():
            raise InvalidOccurrenceRate("Negative occurrence rate is invalid.")
        self._int_cache_flag = False
        self._occr_array = array

    def get_log_occr(self):
        """Returns the log-occurrence rate grid."""
        if tf.is_tensor(self._occr_array):
            return tf.log(self._occr_array) / tf.log(10)
        else:
            return np.log10(self._occr_array)

    def set_log_occr(self, array):
        """Set the log-occurrence rates grid."""
        if not np.array_equal(np.shape(array), np.shape(self)):
            # np.all(np.shape(array) == np.shape(self)):
            import pdb; pdb.set_trace()
            raise ValueError("Input array is the wrong shape.")
        elif tf.is_tensor(array):
            # Placeholder: not sure how to check for finiteness with a
            # tensorflow Variable (TODO)
            pass
        elif not np.isfinite(array).all():
            raise InvalidOccurrenceRate("Negative occurrence rate is invalid.")
        self._int_cache_flag = False
        self._occr_array = 10**array

    # Preferentially use log-occurrence rates, especially for fitting
    occr = property(get_occr, set_occr)
    log_occr = property(get_log_occr, set_log_occr)

    # Probabilistic methods
    # ---------------------

    def likelihood(self, log_occr_array=None):
        """Calculates the value of the likelihood.

        Args:
            log_occr_array (np.array): log-occurrence rates. Optional;
                if not given, they will be taken from the stored values.
        """

        if log_occr_array is not None:
            # Catch invalid occurrence rates for zero likelihood
            try:
                self.log_occr = log_occr_array
            except InvalidOccurrenceRate:
                return 0.0

        # Procedure: calculate N_exp in all cases, calculate detection
        # term if there are detections

        # N_exp
        N_exp = self.calc_integral() * self._N_stars

        # Product terms
        # TODO:Check that the array broadcasting works here
        # Shape of s_terms should be [N_planets, NR, NP]
        s_terms = self.H_array * self.F_array * self.occr

        if tf.is_tensor(self.occr):
            ps_terms = tf.reduce_sum(s_terms, axis=(-1, -2))
            product_term = tf.reduce_prod(ps_terms)
            ll_value = product_term * tf.exp(product_term)
        else:
            product_term = s_terms.sum(axis=(-1, -2)).prod()
            ll_value = product_term * np.exp(-N_exp)

            # BUG
            if np.isnan(ll_value):
                warnings.warn(".likelihood value is nan.")
                import pdb; pdb.set_trace()

        # A nan value is possible when some of the occr are too high
        return ll_value if not np.isnan(ll_value) else 0.0

    def log_likelihood(self, log_occr_array=None):
        """Calculates the value of the log-likelihood.

        Args:
            log_occr_array (np.array): log-occurrence rates. Optional;
                if not given, they will be taken from the stored values.
        """

        if log_occr_array is not None:
            # Catch invalid occurrence rates for zero likelihood
            try:
                self.log_occr = log_occr_array
            except InvalidOccurrenceRate:
                return -np.inf

        # N_exp
        N_exp = self.calc_integral() * self._N_stars

        # Product terms
        # TODO:Check that the array broadcasting works here
        # Shape of s_terms should be [N_planets, NR, NP]
        s_terms = self.H_array * self.F_array * self.occr

        if tf.is_tensor(self.occr):
            ps_terms = tf.reduce_sum(s_terms, axis=(-1, -2))
            product_term = tf.reduce_sum(tf.math.log(ps_terms))
            log_ll_value = product_term - N_exp
        else:
            product_term = np.log(s_terms.sum(axis=(-1, -2))).sum()
            log_ll_value = product_term - N_exp

            # BUG TODO
            if np.isnan(log_ll_value):
                warnings.warn(".likelihood value is nan.")
                import pdb; pdb.set_trace()

            # A nan value is possible when some of the occr are too high
            log_ll_value = -np.inf if np.isnan(log_ll_value) else log_ll_value

        return log_ll_value

    def grad_log_likelihood(self, log_occr_array=None):
        """Calculates the gradient of the log-likelihood.

        NOTE: gradient w.r.t occr, NOT log-occr. TODO: implement w.r.t
        log-occr.

        TODO: changing to w.r.t log-occr could just be done with the
        chain rule actually.

        Args:
            log_occr_array (np.array): log-occurrence rates. Optional;
                if not given, they will be taken from the stored values.
        """

        if log_occr_array is not None:
            # Catch invalid occurrence rates for zero likelihood
            try:
                self.log_occr = log_occr_array
            except InvalidOccurrenceRate:
                return -np.inf * np.ones_like(self.occr, dtype=float)

        # Calculate components first
        N_exp = self.calc_integral() * self._N_stars   # perhaps not needed
        nexp_terms = self._N_stars * self.calc_bin_volumes() * self.H_bar_array
        s_terms = self.H_array * self.F_array * self.occr
        numerator_terms = self.H_array * self.F_array

        if not tf.is_tensor(self.occr):
            # Checking shapes of intermediate terms,
            # numerator_terms vs s_terms.sum(-1, -2) and vs v factors.
            intermediate_terms = numerator_terms / s_terms.sum(axis=(-1, -2))
            # TODO: v_factor changed to negative, I think a minus
            # sign had been missed
            grad_log_array = - nexp_terms + intermediate_terms.sum(axis=0)

            # BUG TODO
            if np.isnan(grad_log_array).any():
                warnings.warn(".grad_log_likelihood value is nan.")
                import pdb; pdb.set_trace()
                grad_log_array = -np.inf * grad_log_array
        else:
            raise NotImplementedError("Manual gradient calculate with "
                                      "tensorflow objects isn't "
                                      "implemented, and seems a bit "
                                      "redundant.")

        return grad_log_array

    def prior(self, log_occr_array=None):
        """Calculates the prior pdf of the occurrence rates.

        Args:
            log_occr_array (np.array): occurrence rates in log form.
        """

        if log_occr_array is not None:
            # Catch invalid occurrence rates for zero likelihood
            try:
                self.log_occr = log_occr_array
            except InvalidOccurrenceRate:
                return 0.0

        # Written in terms of occr for ease, also same as:
        # sqrt(occr) = sqrt(10**self.log_occr)
        return np.prod(np.sqrt(self.occr))

        if tf.is_tensor(self.occr):
            # Written in terms of occr for ease.
            value = tf.reduce_prod(tf.math.sqrt(self.occr))
        else:
            # Written in terms of occr for ease.
            value = np.prod(np.sqrt(self.occr))
            # At this point, it's still possible for occr to be so
            # small that it's underflowing, where value will be nan
            value = value if not np.isnan(value) else 0.0

        return value

    def log_prior(self, log_occr_array=None):
        """Calculates the prior pdf of the occurrence rates.

        Args:
            log_occr_array (np.array): occurrence rates in log form.
        """

        if log_occr_array is not None:
            # Catch invalid occurrence rates for zero likelihood
            try:
                self.log_occr = log_occr_array
            except InvalidOccurrenceRate:
                return -np.inf

        if tf.is_tensor(self.occr):
            # Written in terms of occr for ease.
            value = tf.reduce_sum(0.5*tf.math.log(self.occr))
        else:
            # Written in terms of occr for ease.
            value = np.sum(0.5*np.log(self.occr))
            # At this point, it's still possible for occr to be so
            # small that it's underflowing, where value will be nan
            value = value if not np.isnan(value) else -np.inf

        return value

    def grad_log_prior(self, log_occr_array=None):
        """Calculates the gradient of the prior pdf.

        NOTE: gradient w.r.t occr, NOT log-occr. TODO: implement w.r.t
        log-occr.

        Args:
            log_occr_array (np.array): occurrence rates in log form.
        """

        if log_occr_array is not None:
            # Catch invalid occurrence rates for zero likelihood
            try:
                self.log_occr = log_occr_array
            except InvalidOccurrenceRate:
                return -np.inf * np.ones_like(self.occr, dtype=float)

        if tf.is_tensor(self.occr):
            raise NotImplementedError("Manual gradient calculate with "
                                      "tensorflow objects isn't "
                                      "implemented, and seems a bit "
                                      "redundant.")
        else:
            # Written in terms of occr for ease.
            grad = 1 / (2*self.occr)
            # At this point, it's still possible for occr to be so
            # small that it's underflowing, where value will be nan
            grad = grad if not np.isnan(grad).any() else -np.inf * grad

        return grad

    def log_posterior(self, log_occr_array=None, event_values=None,
                      flattened_occr=False):
        """Calculates the log posterior of a value array of occr.

        log_occr_array can also be flattened so that emcee can use it,
        however we then need to set flattened_occr to True.

        Args:
            log_occr_array: can be normal or log, which must be reflected in
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

        if flattened_occr and log_occr_array is not None:
            # Unflatten it into the occurrence rate grid
            log_occr_array = np.reshape(log_occr_array, np.shape(self.occr))

        # Let the likelihood and prior actually sub the values in
        log_likelihood = self.log_likelihood(log_occr_array)
        log_prior = self.log_prior(log_occr_array)

        return log_likelihood + log_prior

    def grad_log_posterior(self, log_occr_array=None, event_values=None,
                      flattened_occr=False):
        """Calculates the gradient of the log posterior.

        NOTE: gradient w.r.t occr, not log-occr, at the moment.

        log_occr_array can also be flattened so that emcee can use it,
        however we then need to set flattened_occr to True.

        Args:
            log_occr_array: can be normal or log, which must be reflected in
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
            d(log(p(occr|data)))/d(eta)
        """

        if flattened_occr and log_occr_array is not None:
            # Unflatten it into the occurrence rate grid
            log_occr_array = np.reshape(log_occr_array, np.shape(self.occr))

        # Let the likelihood and prior actually sub the values in
        grad_log_likelihood = self.grad_log_likelihood(log_occr_array)
        grad_log_prior = self.grad_log_prior(log_occr_array)

        glp = grad_log_likelihood + grad_log_prior

        if flattened_occr:
            try:
                return glp.reshape(-1)
            except Exception:
                import pdb; pdb.set_trace()
        else:
            return glp

    # Calculations
    # ------------

    def calc_integral(self, occr_array=None, H_array=None):
        """Calculates the integral N_exp / N_star.

        Args:
            log_occr_array (np.ndarray=None): default is stored values
            H_array (np.ndarray=None): default is self.H_bar_array
        """

        # Return the cached value if possible
        if self._int_cache_flag:
            return self._int_cache

        if occr_array is not None:
            self.occr = occr_array

        H_array = self.H_bar_array if H_array is None else H_array

        #I = np.sum(self.occr * np.sum(self._cpf_grid*self.calc_bin_volumes(),
        #                              axis=1))

        I = np.sum(self.volumise_occr() * H_array)

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
    def volumise_occr(self, occr_array=None):
        """Integrates the occurrence rate over volume.

        NOTE:should be the only thing that's changed between,
        different hyperparametrisations.

        Returns:
            volumised_occr: must be the same shape as the grid shape.
        """
        occr_array = self.occr if occr_array is None else occr_array
        volumised_occr = occr_array * self.calc_bin_volumes()

        assert np.all(np.shape(volumised_occr)[-2:] == self.shape)
        assert np.array_equal(np.shape(volumised_occr)[-2:], np.shape(self))

        return volumised_occr

    def integrate_over_volume(self, value_array):
        """Multiplies input by self.calc_bin_volumes.

        TODO: currently do not use, DEPRECATED."""
        return value_array * self.calc_bin_volumes()

    def rate_density(self, value, H_array=None):
        """Returns the rate density at a particular value of (R, P).

        Returns: occurrence rate x completeness
        """

        # Not used during the fitting/likelihood calculations

        H_array = self.H_bar_array if H_array is None else H_array

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

        return self.occr[R_i, P_i] * H_array[R_i, P_i]


    # Estimators
    # ----------

    def predict_rate_grid(self, occr=None, H_array=None, N_stars=None):
        """Predicts the detection rate in each bin over whole sample.

        NOTE: gives the lambda-rate, i.e occurrence rate x completeness;
        if we want the actual rate, use volumise occr, and volumise occr
        times N_stars.

        Args:
            occr (np.ndarray=None): occurrence rate, default is stored
            H_array (np.ndarray=None): completeness array; default is
                to use the integrated mean completeness (H_bar_array)
            N_stars (int=self._N_stars): number of stars to use,
                assuming the completeness array is a mean completeness
        """

        occr = self.occr if occr is None else occr
        H_array = self.H_bar_array if H_array is None else H_array
        N_stars = self._N_stars if N_stars is None else N_stars

        rate_grid = self.volumise_occr(occr) * H_array

        return N_stars * rate_grid

    def marginalise_occr_period(self, occr_array=None):
        """Marginalises the occurence rate over the range of periods."""

        occr_array = self.occr if occr_array is None else occr_array

        #return occr_array * self.calc_bin_volumes().sum(axis=1)
        # TODO: check if this is True.
        # i.e: in an inhomogeneous Poisson process, is the total
        # rate in a volume of the space equal to the integral
        # of the rate-function across that volume of space?
        return self.volumise_occr(occr_array).sum(axis=-1)

    # Sampling and inversion
    # ----------------------

    def sample_occr_emcee(self, burn=2000, iters=2000, nwalkers=None,
                          thin_factor=50, save=True, plot=False,
                          pre_optimise=True, run_splits=1, threads=1):
        """Sample the occurrence rate posterior with emcee.

        First optimises to the MAP with a gradient-based-sampler, then
        uses the Affine-Invariant Sampler, implemented in emcee by DFM.
        Not the most efficient in some ways, needs many iterations.

        Args:
            burn (int=2000): number of iterations to burn out,
                per walker
            iters (int=2000): number of posterior samples per walker
            save (bool=True): stored the MCMC samples; overwriting
                previous stored samples
            plot (bool=False): plots the triangle plot of the posterior

        Returns:
            samples (np.ndarray): gives the occurrence rate, not
                log-occr
        """

        flat_shape = np.prod(np.shape(self.occr))
        nwalkers = 2*(flat_shape+1) if nwalkers is None else nwalkers

        if pre_optimise:
            # Use a gradient-based optimiser.
            # NOTE: at the moment, the gradient is only defined w.r.t
            # occr, not log-occr. So this optimisation will work on
            # occr, while the MCMC will naturally work on log-occr.

            # Negative log-posterior and gradient of the log-posterior
            nlp = lambda x: -self.log_posterior(np.log10(x),
                                                flattened_occr=True)
            nglp = lambda x: -self.grad_log_posterior(np.log10(x),
                                                      flattened_occr=True)

            occr_0 = np.ones(flat_shape, dtype=float)
            occr_opt = optimize.minimize(fun=nlp, x0=occr_0,
                                         method='BFGS', jac=nglp).x
            occr_rand = occr_opt * (1 + 0.001*np.random.rand(nwalkers,
                                                             flat_shape))

            log_occr_initial = np.log10(occr_rand)
        else:
            log_occr_initial = np.log10(np.random.rand(nwalkers, flat_shape))

        sampler = emcee.EnsembleSampler(nwalkers=nwalkers,
                                        dim=flat_shape,
                                        lnpostfn=self.log_posterior,
                                        threads=threads,
                                        kwargs={'event_values':None,
                                                'flattened_occr':True})

        # Burn
        pos, _, _ = sampler.run_mcmc(log_occr_initial, N=burn)
        sampler.reset()

        # Run over multiple stages to prevent memory errors
        iters_per_run = int(np.round(iters))/run_splits
        sample_list = []

        # Make the temporary file
        # if temp_file is not None:
        #     f = open(temp_file, "w")
        #     f.close()

        for i in range(run_splits):
            pos, _, _ = sampler.run_mcmc(pos, N=iters_per_run,
                                         thin=thin_factor)

            # Extract chains
            samples = sampler.flatchain
            sampler.reset()

            # if thin_factor > 1:
            #     N_resample = int(np.round(len(samples) / thin_factor))
            #     np.random.shuffle(samples)
            #     samples = samples[:N_resample]

            sample_list.append(samples)

        samples = np.concatenate(sample_list)

        # Need to reshape back into the occr shape
        samples = np.reshape(samples, (-1, *np.shape(self.occr)))

        # Crucial: the samples and output are always in normal form,
        # never in log form
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

        # Clean-up
        # os.remove(temp_file)

        return samples

    def sample_occr_individual(self, iters=2000, nwalkers=2,
                               save=True, plot=False):
        """Samples occurrence rates individually in each bin with PyMC3.

        Args:
            burn (int=2000): number of iterations to burn out,
                per walker
            iters (int=2000): number of posterior samples per walker
            save (bool=True): stored the MCMC samples; overwriting
        
        
                previous stored samples
            plot (bool=False): plots the triangle plot of the posterior

        Returns:
            samples (np.ndarray): gives the occurrence rate, not
                log-occr
        """

        
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

        occr_median = self.volumise_occr(np.median(samples, axis=0))
        occr_lower = self.volumise_occr(np.percentile(samples, 16, axis=0))
        occr_upper = self.volumise_occr(np.percentile(samples, 84, axis=0))
        occr_limit = self.volumise_occr(np.percentile(samples, 95, axis=0))

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
                self.H_bar_array[ix, iy]*self._N_stars,
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

        occr_grid = self.volumise_occr(occr_median)

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
class ConstrainedOCCR(UncertainBinnedPoissonProcess):
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

        # This initialisation isn't checked for shape
        self._occr_array = np.ones(self.shape[0])

        # Must calculate the Jacobian once
        # Do the calculations in non-flattened form (3rd order matrix)
        jacobian = np.zeros([self.shape[0], self.shape[1], self.shape[0]])

        for i in range(self.shape[0]):
            jacobian[i, :, i] = 1

        self._occr_jacobian = jacobian

    def set_occr(self, array):
        """This is overloaded so that the shape-check works."""

        if not np.array_equal(np.shape(array), np.shape(self)[:1]):
            # np.all(np.shape(array) == np.shape(self)[0]):
            import pdb; pdb.set_trace()
            raise ValueError("Input array is the wrong shape.")
        elif tf.is_tensor(array):
            # Placeholder: not sure how to check for finiteness with a
            # tensorflow Variable (TODO)
            pass
        elif (array < 0.0).any():
            raise InvalidOccurrenceRate("Negative occurrence rate is invalid.")
        self._int_cache_flag = False
        self._occr_array = array

    def set_log_occr(self, array):
        """This is overloaded so that the shape-check works."""

        if not np.array_equal(np.shape(array), np.shape(self)[:1]):
            # np.all(np.shape(array) == np.shape(self)):
            import pdb; pdb.set_trace()
            raise ValueError("Input array is the wrong shape.")
        elif not np.isfinite(array).all():
            raise InvalidOccurrenceRate("Negative occurrence rate is invalid.")
        self._int_cache_flag = False
        self._occr_array = 10**array

    # TODO: instead of having to overload these operators to use only
    # dimension of shape, distinguish between grid_shape and
    # parameter_shape
    occr = property(UncertainBinnedPoissonProcess.get_occr,
                    set_occr)
    log_occr = property(UncertainBinnedPoissonProcess.get_log_occr,
                        set_log_occr)

    # TODO: implement log_occr_grid and the Jacobian of that

    def get_occr_grid(self, flattened=False):
        """This gives the actual occurrence rates in the original grid.

        The distinction is that .occr are the parameters that
        parametrise the occurrence rate grid, while occr_grid is the
        actual occr grid.

        This is needed to make the chain-rule application easier. In
        fact, this should be implemented for the main object,
        which would make adding new parametrisations as child objects
        much easier and more intuitive (we would only have to change,
        the occr_grid getter method, and its Jacobian). TODO this.

        TODO: if doing the above, in the general object, having
        large arrays and so on, and having a large matrix for the
        jacobian (it will be quite large actually) will slow things
        down. Solution: make hte occr_grid just return self.occr, and
        make the Jacobian equal to 1 in that case. TODO: in fact, the
        occr_jacobian getter method need only be defined in the parent
        object; in this object, the only difference will be the actual
        calculation of the self._occr_jacobian, which is done once in
        the __init__ function.
        
        Args:
            flattened (bool=False): flattened into a vector. The matrix
                shape of the occr grid is unimportant for calculating
                the likelihood.
        """

        if tf.is_tensor(self.occr):
            raise NotImplementedError("This doesn't work with tensors.")

        occr_grid = np.array([self.occr] * np.shape(self)[1]).T

        if flattened:
            return np.reshape(occr_grid, -1)
        else:
            return occr_grid

    def get_occr_jacobian(self, flattened=True):
        """The Jacobian for the parametrisation of the occr grid.

        The occr_grid object is an vector with nxm elements, and is a
        function of some parametrisation (self.occr).
        The Jacobian of that function is required to calculate the
        gradients of the likelihood/prior etc...

        This is needed to make the chain-rule application easier. In
        fact, this should be implemented for the main object,
        which would make adding new parametrisations as child objects
        much easier and more intuitive (we would only have to change,
        the occr_grid getter method, and its Jacobian). TODO this.

        TODO: if doing the above, in the general object, having
        large arrays and so on, and having a large matrix for the
        jacobian (it will be quite large actually) will slow things
        down. Solution: make hte occr_grid just return self.occr, and
        make the Jacobian equal to 1 in that case. TODO: in fact, the
        occr_jacobian getter method need only be defined in the parent
        object; in this object, the only difference will be the actual
        calculation of the self._occr_jacobian, which is done once in
        the __init__ function.
        
        Args:
            flattened (bool=True): flattened into a 2-dim matrix. This
                equivalent to treating the occr_grid as vector. It makes
                more sense to have the Jacobian as a 2-dim matrix. If
                this is false, the Jacobian will be a 3-dimensional
                matrix,because the dependent parameter will be treated
                as being a matrix, and not a vector.  The matrix shape
                of the occr grid is unimportant for calculating the
                likelihood.
        """

        if flattened:
            return np.reshape(a=self._occr_jacobian,
                              newshape=(np.prod(self.shape), self.shape[0]))
        else:
            return self._occr_jacobian

    occr_grid = property(get_occr_grid)
    occr_jacobian = property(get_occr_jacobian)

    # Main operations on the grid
    # ---------------------------

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

    # Probabilistic methods
    # ---------------------

    def log_likelihood(self, log_occr_array=None):
        """Calculates the value of the log-likelihood.

        Args:
            log_occr_array (np.array): log-occurrence rates. Optional;
                if not given, they will be taken from the stored values.
        """

        if log_occr_array is not None:
            # Catch invalid occurrence rates for zero likelihood
            try:
                self.log_occr = log_occr_array
            except InvalidOccurrenceRate:
                return -np.inf

        # N_exp
        N_exp = self.calc_integral() * self._N_stars

        # Product terms
        # TODO:Check that the array broadcasting works here
        # Shape of s_terms should be [N_planets, NR, NP]
        s_terms = self.H_array * self.F_array * self.occr_grid

        if tf.is_tensor(self.occr):
            ps_terms = tf.reduce_sum(s_terms, axis=(-1, -2))
            product_term = tf.reduce_sum(tf.math.log(ps_terms))
            log_ll_value = product_term - N_exp
        else:
            product_term = np.log(s_terms.sum(axis=(-1, -2))).sum()
            log_ll_value = product_term - N_exp

            # BUG TODO
            if np.isnan(log_ll_value):
                warnings.warn(".likelihood value is nan.")
                import pdb; pdb.set_trace()

            # A nan value is possible when some of the occr are too high
            log_ll_value = -np.inf if np.isnan(log_ll_value) else log_ll_value

        return log_ll_value

    def grad_log_likelihood(self, log_occr_array=None):
        """Calculates the gradient of the log-likelihood.

        Overloaded due to the different shape of the occr and thus the
        Jacobian. TODO: figure out where the crux of the difference is
        and try to make it universal for the base object.

        TODO: currently self.occr is one-dimensional. However,
        H_arrays and F_array are 2-dimensional. F_array could be made
        one-dimensional actually.

        TODO: this is more of a mutli-dim chain rule issue.
        The Jacobian w.r.t to the occr grid remains the same. However,
        here, the occr_grid is dependent on a smaller number of
        parameters, the occrs (by equality).

        NOTE: gradient w.r.t occr, NOT log-occr. TODO: implement w.r.t
        log-occr.

        Args:
            log_occr_array (np.array): log-occurrence rates. Optional;
                if not given, they will be taken from the stored values.
        """

        if log_occr_array is not None:
            # Catch invalid occurrence rates for zero likelihood
            try:
                self.log_occr = log_occr_array
            except InvalidOccurrenceRate:
                return -np.inf * np.ones_like(self.occr, dtype=float)

        # The Poisson Jacobian
        # --------------------
        # Calculate components first
        N_exp = self.calc_integral() * self._N_stars   # perhaps not needed
        v_factors = self._N_stars * self.calc_bin_volumes() * self.H_bar_array
        # TODO: broadcast self.occr into the same shape as self.F_array
        # Reduce it later through the chain-rule
        s_terms = self.H_array * self.F_array * self.occr_grid
        numerator_terms = self.H_array * self.F_array

        if not tf.is_tensor(self.occr):
            # Checking shapes of intermediate terms,
            # numerator_terms vs s_terms.sum(-1, -2) and vs v factors.
            intermediate_terms = numerator_terms / s_terms.sum(axis=(-1, -2))
            # TODO: v_factor changed to negative, I think a minus
            # sign had been missed
            poisson_jacobian = intermediate_terms.sum(axis=0) - v_factors
            poisson_jacobian = poisson_jacobian.reshape(-1)
            param_jacobian = self.get_occr_jacobian(flattened=True)
            grad_log_array = np.matmul(poisson_jacobian, param_jacobian)

            # BUG TODO
            if np.isnan(grad_log_array).any():
                warnings.warn(".grad_log_likelihood value is nan.")
                import pdb; pdb.set_trace()
                grad_log_array = -np.inf * grad_log_array
        else:
            raise NotImplementedError("Manual gradient calculate with "
                                      "tensorflow objects isn't "
                                      "implemented, and seems a bit "
                                      "redundant.")

        return grad_log_array

    # def grad_log_prior(self, log_occr_array=None):
    #     """Calculates the gradient of the prior pdf.

    #     NOTE: gradient w.r.t occr, NOT log-occr. Also, w.r.t to the
    #     actually paramtrisation, not the grid. TODO: implement w.r.t
    #     log-occr.

    #     Args:
    #         log_occr_array (np.array): occurrence rates in log form.
    #     """

    #     if log_occr_array is not None:
    #         # Catch invalid occurrence rates for zero likelihood
    #         try:
    #             self.log_occr = log_occr_array
    #         except InvalidOccurrenceRate:
    #             return -np.inf * np.ones_like(self.occr, dtype=float)

    #     if tf.is_tensor(self.occr):
    #         raise NotImplementedError("Manual gradient calculate with "
    #                                   "tensorflow objects isn't "
    #                                   "implemented, and seems a bit "
    #                                   "redundant.")
    #     else:
    #         # Written in terms of occr for ease.
    #         grad = 1 / (2*self.occr)
    #         # At this point, it's still possible for occr to be so
    #         # small that it's underflowing, where value will be nan
    #         grad = grad if not np.isnan(grad).any() else -np.inf * grad

    #     param_jacobian = self.occr_jacobian(flattened=True)
    #     # TODO
    #     print("Check if the multiplication is valid (shapes).")
    #     grad = np.matmul(grad, param_jacobian)

    #     return grad

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
