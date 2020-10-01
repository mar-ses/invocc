"""Simple initial iteration of the inverse process with max-likelihood.

Non-HBM model; doesn't include planet parameter uncertainties.

General idea:
- Decide which probabilistic software to use.
- Split layers (completeness, occurence rate function) of observable
  occurence rate; and other layers too. Thus can pick and choose the
  occurence rate function.
- Should include various testing-suites (in another module perhaps);
  include fake data production etc...

Given the non-heriarchical set-up, the MCMC will explore a
low-dimensional space; potentially even one-dimensional in the case
of a constant occurence rate.

NOTE: if occurence rate is constant; it should be constant in the
space logR, logP, at least in the initial assumption (or should it).
This must be decided.
"""

import numpy as np
import pandas as pd
import emcee

# An object-based implementation may be the easiest;
# especially for HBM

# The probability distributions
# -----------------------------



# # Occurence rates
# def occr_const():
# 	pass

# # Occurence rate log-likelihood (current test)
# def occr_likelihood(PR_array, occr, completeness_func):
# 	"""

# 	Args:
# 		PR_array (np.ndarray): n discovered planets with P and R
# 		occr (float): the occurence rate parameter
# 		completeness_func (callable): with a 
# 	"""
