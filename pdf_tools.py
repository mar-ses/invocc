"""Defines basic forms of probability distributions used in the packages.

TODO: since occurrence rate must be binned in radius; one option
	  is to have one bin per complenetess bin (i.e same radius
	  intervals), so that it's more general.
	  For this to work however; then I would set a Gaussian prior
	  on the radius.

TODO: consider the possibility of calculating the integral in the
	  Poisson likelihood simply through basic Monte-Carlo.

TODO: create bin-grid object.

TODO: decide on logP vs P, and same for R

TODO: change HP (occr) to log
"""

import numpy as np



class BinnedCompleteness(object):
	"""Probability object that defines the binned completeness function.

	TODO: determine if this is the observable or real occurrence rate;
		  
		  Better: this is a general function that doesn't care; instead
		  observable completeness will be a combination of these and-or
		  others.

		  Alternatively: this is the observable occurrence (or that
		  inherits this), and is a composition of the real binned
		  completenes and the occurrence rate function (which is a callable
		  that is set during initiation).

	TODO: create internal properties; binning parameters, plus a function
		  to go from (R or P) value to bin-index (independent and not).

	TODO: auto-integrate function (i.e get N_expected)
	"""

	def __init__(self, R_boundaries, P_boundaries, value_grid,
				 occurrence_rate_func, outside_value=0.0):
		"""Creates the probability from a binned array of completeness values.

		Completeness grid has a N x M dimension; with N R-bins,
		and M P-bins.

		Args:
			R_boundaries (np.array, (N+1)-dim): the boundaries of the bins
				in R space
			P_boundaries (np.array, (M+1)-dim): the boundaries of the bins
				in P space
			value_grid (np.ndarray, shape: NxM): the values of the
				*completeness* in each bin

		"""

		raise NotImplementedError


class ObservableOccurrenceBasic(object):
	"""Probability object defines the observable occurrence pdf.

	The completeness is treated as binned in a set number of bins
	in period and planet radius. The true occurence is treated as
	binned only in radius in a set number of intervals.

	NOTE: regarding logP vs P; this object doesn't care, at least
		  in the calculations. That's an outside consideration.

	-------------

	TODO: determine if this is the observable or real occurrence rate;

		  Better: this is a general function that doesn't care; instead
		  observable completeness will be a combination of these and-or
		  others.

		  Alternatively: this is the observable occurrence (or that
		  inherits this), and is a composition of the real binned
		  completenes and the occurrence rate function (which is a callable
		  that is set during initiation).

	TODO: create internal properties; binning parameters, plus a function
		  to go from (R or P) value to bin-index (independent and not).

	TODO: auto-integrate function (i.e get N_expected)

	TODO: for the above; cache the value, so that it isn't
		  recalculated unless the hyperparameters (theta or
		  something) change.
		  In order for this to work; we need strong setter
		  and getter functions on the hyperparameters and
		  completeness (or have their values hidden).
	"""

	def __init__(self, R_boundaries, P_boundaries, cpf_value_grid,
				 N_stars, outside_value=0.0, event_values=None):
		"""Creates the probability from array of completeness values.

		Completeness grid has a N x M dimension; with N R-bins,
		and M P-bins.

		Occurence rate is binned in radius, in 1 Rj intervals,
		for now.

		Args:
			R_boundaries (np.array, (N+1)-dim): the boundaries of
				the bins in R space
			P_boundaries (np.array, (M+1)-dim): the boundaries of
				the bins in P space
			value_grid (np.ndarray, shape: NxM): the values of the
				*completeness* in each bin
		"""

		self._R_boundaries = np.array(R_boundaries)
		self._P_boundaries = np.array(P_boundaries)
		self._cpf_grid = np.array(cpf_value_grid)
		self._N_stars = N_stars

		# Flag; if True, the parameters have been unchanged since
		# the last time the rate integral was calculated; therefore
		# the cached integral is safe to use.
		self._int_cache_flag = False
		self._int_cache = 0.0

		# Initiate the occurrence rate in same bins of R as cpf
		self.occr_array = np.zeros(np.shape(self)[0])

		# Initiate event values
		self._event_values = event_values

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

	def likelihood(self, occr=None, event_values=None):
		"""Calculates the value of the likelihood.

		Args:
			event_values (np.array or float): the R, P pairs to calculate
				the likelihood. If None or empty, calculates it
				assuming zero events. Otherwise, expects a single
				(R, P) coordinate, or N values of (R, P), i.e
				shape (N x 2).
		"""

		if occr is not None:
			if (occr < 0.0).any():
				return 0.0
			self.occr = occr
		if event_values is not None:
			self.event_values = event_values

		I = self.calc_integral() * self._N_stars

		# Case of no events
		if self.event_values is None or not hasattr(self.event_values,
													'__len__'):
			return np.exp(-I)
		else:
			return np.exp(-I) * np.prod(self.rate_density(self.event_values))

	def log_likelihood(self, occr=None, event_values=None):
		"""Calculates the value of the likelihood.

		Args:
			event_values (np.array or float): the R, P pairs to calculate
				the likelihood. If None or empty, calculates it
				assuming zero events. Otherwise, expects a single
				(R, P) coordinate, or N values of (R, P), i.e
				shape (N x 2).
		"""

		if occr is not None:
			if (occr < 0.0).any():
				return -np.inf
			self.occr = occr
		if event_values is not None:
			self.event_values = event_values

		I = self.calc_integral() * self._N_stars

		# Case of no events
		if self.event_values is None or not hasattr(self.event_values,
													'__len__'):
			return - I
		else:
			return np.sum(np.log(self.rate_density(self.event_values))) - I

	def calc_integral(self):
		"""Calculates the in TODO"""

		# Return the cached value if possible
		if self._int_cache_flag:
			return self._int_cache

		I = np.sum(self.occr * np.sum(self._cpf_grid * self.calc_bin_volumes(),
									  axis=1)
		)

		self._int_cache_flag = True
		self._int_cache = I
		return I

	def calc_bin_volumes(self):
		"""Calculates the array of areas (Lebesque measure) per bin."""

		return np.outer(np.diff(self._R_boundaries),
						np.diff(self._P_boundaries))

	def rate_density(self, value):
		"""Returns the rate density at a particular value of (R, P)."""

		if value.ndim == 2:
			value = value.T

		R_i = np.digitize(value[0], self._R_boundaries) - 1
		P_i = np.digitize(value[1], self._P_boundaries) - 1

		return self.occr[R_i] * self._cpf_grid[R_i, P_i]


	# Estimators
	# ----------

	def predict_rate_grid(self, N_stars=1):
		"""Predicts the rate at each cpf-grid bin."""

		rate_grid = (self._cpf_grid * self.calc_bin_volumes()).T * self.occr
		return N_stars * rate_grid

	def marginalise_occr_period(self):
		"""Marginalises the occurence rate over the range of periods."""

		return self.occr * self.calc_bin_volumes().sum(axis=1)


	# Properties
	# ----------

	# TODO: make sure object references work as expected (i.e copies)

	@property
	def shape(self):
		"""Gives the shape of the completeness array."""
		return np.array([len(self._R_boundaries)-1, len(self._P_boundaries)-1])

	def get_occr(self):
		return self.occr_array

	def set_occr(self, array):
		if np.shape(array) != np.shape(self)[0]:
			raise ValueError("Input array is the wrong shape.")
		elif (array < 0.0).any():
			raise ValueError("Negative occurrence rate is invalid.")
		self._int_cache_flag = False
		self.occr_array = array

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
	event_values = property(get_event_values, set_event_values)

class BinGrid(object):
	"""Class that overwrites a pandas DataFrame and acts as
	bin-grid function.

	Should contain a callable (funct-at); and properly define
	intervals and boundaries.
	"""

	pass
