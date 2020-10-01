"""Package containing processes for inverting occurrence rate statistics."""

import os

# Main file structure
HOME_DIR = os.environ['HOME']
PACKAGES_DIR = HOME_DIR
DATA_DIR = "{}/data".format(HOME_DIR)

# K2GP FILE STRUCTURE
K2GP_DIR = '%s/k2gp'%PACKAGES_DIR
K2GP_DATA = '%s/data'%K2GP_DIR
DIST_DIR = "{}/k2gp_dist".format(PACKAGES_DIR)
TRANSEAR_DIR = "{}/transear".format(PACKAGES_DIR)
INVOCC_DIR = "{}/invocc".format(PACKAGES_DIR)
CATALOG_DIR = "{}/catalogs/ultracool_occr".format(DATA_DIR)

# UBELIX DATA FILE STRUCTURE
UB_HOME_DIR = "/home/ubelix/csh/sestovic"
UB_MANUAL = "{}/k2_manual_lightcurves".format(UB_HOME_DIR)
UB_MAST_LCF = "{}/k2_mast_lightcurves".format(UB_HOME_DIR)
UB_MAST_TPF = "{}/k2_mast_tpfs".format(UB_HOME_DIR)
UB_DETRENDED = "{}/k2_detrended".format(UB_HOME_DIR)

# MAST VARIABLES
LATEST_CAMPAIGN = 19		# the latest fully calibrated campaign.

# Hyperparameter logging table on ubelix
HP_LOC = "{}/hp_table.pickle".format(HOME_DIR)

# Directory for saving the figures for the paper
FIG_DIR = '{}/Paper Production/Occurrence Rates/figures'.format(HOME_DIR)
