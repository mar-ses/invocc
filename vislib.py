"""Library of pure visualisation routines."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from corner import corner
from astropy import units, constants as const

from . import DATA_DIR


# Formatting
# ----------

# A scale factor for figure sizes to allow changing between
# powerpoint size figures, paper size figures, and poster size
# first (and thus to allow the axes to change accordingly)
fig_scale = 1.0
font_size = 11
# For the paper, around 1.3/1.25 at half-scale is a good compromise,
# though perhaps it produces text that is too small.

# TODO: add figure sizes and final formatting


# Grid plots
# ----------

def plot_grid(grid_values, x_edges, y_edges, log_x=True,
              log_values=True, value_label=None, colour_map=None,
              print_values=False, ax=None, show=False,
              truncated_cmap=False):
    """Main work-function for plotting the colour-grids.

    NOTE: doesn't set the x and y axis labels, do this externally

    Args:
        sgrid: grid of sensitivity values. Shape as produced by
            util_lib.bin_irr_points
        y_edges: the edges of the bins in x/radius
        x_edges: the edges of the bins in y/periods
        log_x (bool=True): if True, the x/P axis is in log-form, which
            requires a bit of tinkering
        log_values (bool=True): if True, plots the log of the
            colour-value as the third dimension (i.e the colour)
        value_label (str=None): the name of the colour-dimension,
            i.e sensitivity or completeness or occurrence rate,
            to be put in the colourbar axis label
        colour_map: which cmap to use
        print_values (bool=False): to also add the text as values
            to the grid
        ax (plt.Axes=None): if plotting on an existing axis is desired
        show (bool=False)

    Returns:
        ax
    """

    plt.rcParams['figure.figsize'] = [5*fig_scale, 5*fig_scale]

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    norm = colors.LogNorm() if log_values else None

    # The color-map
    cmap = plt.get_cmap('viridis') if colour_map is None else colour_map
    cmap = truncate_colormap(cmap, 0.2, 1.0) if truncated_cmap else cmap

    # The actual plot
    if not log_x:
        im = ax.matshow(grid_values,
                        origin='lower',
                        norm=norm,
                        cmap=cmap,
                        extent=(x_edges[0], x_edges[-1],
                                y_edges[0], y_edges[-1]))
    else:
        im = ax.pcolormesh(x_edges, y_edges, grid_values, norm=norm, cmap=cmap)
        ax.set_xscale('log')

    # Colour bar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="6%", pad=0.3)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(value_label)

    # Print the text
    if print_values:
        add_text_grid(grid_values, x_edges, y_edges, ax)

    # Aesthetics
    # ax.set_xlabel('Period, days')
    # ax.set_ylabel('Radius, $R_\oplus$')
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
        return ax
    else:
        fig.show()
        return ax


# Theoretical and scientific background plots
# -------------------------------------------

def get_eu_catalog():
    catalog = pd.read_csv("{}/catalogs/exoplanet_eu_catalog_23_12_19.csv"
                          "".format(DATA_DIR))
    catalog.rename(columns={'# name':'name'}, inplace=True)
    catalog = catalog[(catalog.orbital_period < 365) \
                    & ((catalog.mass < 13) | (catalog.radius < 0.8))]

    catalog['T_eq'] = np.sqrt((catalog.star_radius.values * const.R_sun / (2 * catalog.semi_major_axis.values * units.AU)).to('')) * catalog.star_teff

    return catalog

def plot_catalog():
    eu_catalog = get_eu_catalog()

    fig, ax = plt.subplots()

    no_temp = eu_catalog[eu_catalog.T_eq.isnull()]
    eu_catalog = eu_catalog[~eu_catalog.T_eq.isnull()]

    colourscale = np.log10(eu_catalog.T_eq)
    colourscale[colourscale > np.log10(2000)] = np.log10(2000)
    colourscale[colourscale < 0] = 0

    _  = ax.scatter(no_temp.orbital_period, no_temp.star_teff,
                    marker='.', c='0.5', alpha=0.8)
    cc = ax.scatter(eu_catalog.orbital_period, eu_catalog.star_teff,
                    marker='.', c=colourscale)

    # Colour bar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="6%", pad=0.3)
    cbar = fig.colorbar(cc, cax=cax)
    cbar.set_label(r"Planet eq. temperature, $log_{10} T_{eq}$")

    #ax.axhline(2600, 0.1, 0.6, color='r', linestyle='--')

    ax.set_ylabel("Stellar temperature, K")
    ax.set_xlabel("Orbital period, days")
    ax.set_xscale('log')
    #ax.set_yscale('log')
    ax.set_xlim(0.5, 365)
    ax.set_ylim(2000, 7000)

    fig.show()

    return fig, ax

def plot_catalog_new():
    eu_catalog = get_eu_catalog()

    fig, ax = plt.subplots()

    no_temp = eu_catalog[eu_catalog.T_eq.isnull()]
    eu_catalog = eu_catalog[~eu_catalog.T_eq.isnull()]
    eu_catalog.loc[eu_catalog.T_eq > 2000, 'T_eq'] = 2000
    
    colourscale = np.log10(eu_catalog.T_eq)
    colourscale[colourscale > np.log10(2000)] = np.log10(2000)
    colourscale[colourscale < 0] = 0

    _  = ax.scatter(no_temp.orbital_period, no_temp.star_teff,
                    marker='.', c='0.5', alpha=0.8)
    cc = ax.scatter(eu_catalog.orbital_period, eu_catalog.star_teff,
                    marker='.', c=eu_catalog.T_eq, norm=colors.LogNorm(),)

    # Colour bar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="6%", pad=0.3)
    cbar = fig.colorbar(cc, cax=cax)
    cbar.set_label(r"Planet eq. temperature, $T_{eq}$")
    # This currently doesn't work, need to set minor ticks
    # cax.set_yticklabels([200,300,400,500,600,700,800,900,1000,2000])

    #ax.axhline(2600, 0.1, 0.6, color='r', linestyle='--')

    ax.set_ylabel("Stellar temperature, K")
    ax.set_xlabel("Orbital period, days")
    ax.set_xscale('log')
    #ax.set_yscale('log')
    ax.set_xlim(0.5, 365)
    ax.set_ylim(2000, 7000)

    fig.show()

    return fig, ax


# Utilities
# ---------

def add_text_grid(text_array, x_edges, y_edges, ax,
                  size_factor=1.0, square_offset=[0.1, 0.2],
                  **text_kwargs):
    """Adds a text grid to an axis.

    NOTE: text size gets multiplied by fig_scale so that it
    remains the same relative size

    Args:
        text_array (np.ndarray): array of text values to add in
            each grid square. Must match the shape
        x_edges (np.array): the x-locations of the grid-square
            edges, giving both inner and outer bin edges
        y_edges (np.array): the y-locations of the grid-square
            edges, giving both inner and outer bin edges
        ax (plt.Axes): the axes to draw on
        size_factor (float=1.0): the size of the text as a factor
            of the default size, which is to be determined but
            currently 8
        square_offset (list or np.array = [0,0]): the offset from
            bottom left in each grid square in some sort of units;
            make it standardised units (i.e 1.0 = at the next edge)
        **text_kwargs (dict): examples:
            verticalalignment ('center', 'top', 'bottom')
            horizontalalignment ('center', 'right', 'left')

    Returns:
    """

    # The text size needs to remain constant whatever the figure size
    # so that it takes the same portion of each square
    size_factor = size_factor * fig_scale

    x_locs = x_edges[:-1] + np.diff(x_edges) * square_offset[0]
    y_locs = y_edges[:-1] + np.diff(y_edges) * square_offset[1]

    # To hold the plt.Text objects so they can be returned
    text_elements = np.empty_like(text_array.T, dtype=object)

    #import pdb; pdb.set_trace()

    # NOTE crucial point: in an array, the first index counts through
    # rows, i.e it goes down and corresponds to the y-like axis.
    # Therefore, the indexes must be swapped. A matrix layed onto a
    # grid must be transposed, IFF you want the first index to be
    # corresponding to the x axis and the second to the y axis.
    for idx, txt in np.ndenumerate(text_array.T):
        if not isinstance(txt, str):
            txt = "{:.2g}".format(txt)

        text_elements[idx[0], idx[1]] = ax.text(x=x_locs[idx[0]],
                                                y=y_locs[idx[1]],
                                                s=txt,
                                                fontsize=size_factor*8,
                                                **text_kwargs)

    return text_elements

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=300):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
