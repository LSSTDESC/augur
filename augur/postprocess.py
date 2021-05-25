"""
Postprocessing module
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import astropy.table
from scipy.stats import norm
import os


def postprocess(config):
    """
    Generates postprocessing plots and Figure
    of Merit based on the configuration passed

    Parameters:
    -------
    config : dict
        The yaml parsed dictional of the input yaml file
    """
    pconfig = config["postprocess"]
    outdir = config["analyze"]["cosmosis"]["output_dir"]
    input = os.path.join(outdir, "chain.txt")
    fisher = np.loadtxt(input)
    npars = fisher.shape[0]
    keys = astropy.table.Table.read(input, format='ascii').keys()
    keys = np.array(keys)
    if "labels" not in pconfig.keys():
        labels = keys
    else:
        labels = pconfig["labels"]
    xsize, ysize = pconfig["size"] if "size" in pconfig.keys() else (48, 48)
    if "centers" in pconfig.keys():
        centers = pconfig["centers"]
    else:
        centers = np.zeros(npars)
    nsigma = pconfig["nsigma"] if "nsigma" in pconfig.keys() else 1
    legend = pconfig["legend"] if "legend" in pconfig.keys() else '__nolegend__'
    lw = pconfig["linewidth"] if "linewidth" in pconfig.keys() else 1
    ls = pconfig["linestyle"] if "linestyle" in pconfig.keys() else '-'
    color = pconfig["linecolor"] if "linecolor" in pconfig.keys() else 'k'
    f, ax = plt.subplots(npars, npars, figsize=(xsize, ysize))
    inv_cache = None
    for i in range(npars):
        for j in range(i+1, npars):
            i_key = labels[i]
            j_key = labels[j]
            vals, theta, inv_cache = get_ellipse(fisher, i, j, inv_cache=inv_cache)
            x0 = 2*np.sqrt(vals[0])*np.cos(np.radians(theta))
            x1 = 2*np.sqrt(vals[1])*np.cos(np.radians(theta))
            y0 = 2*np.sqrt(vals[0])*np.sin(np.radians(theta))
            y1 = 2*np.sqrt(vals[1])*np.sin(np.radians(theta))
            xlim = 3*np.sqrt(x0**2+x1**2)
            ylim = 3*np.sqrt(y0**2+y1**2)
            # Create the 1D histogram for i, i
            if j == i+1:
                xarr = np.linspace(centers[i]-5*np.sqrt(vals[0]),
                                   centers[i]+5*np.sqrt(vals[0]), 200)
                ax[i, i].plot(xarr, norm.pdf(xarr, centers[i], np.sqrt(vals[0])))
            plot_ellipse(vals, theta, color, legend, nsigma,
                         lw, ls, centers[i], centers[j], ax=ax[j, i])
            ax[j, i].set_xlim(-xlim+centers[i], xlim+centers[i])
            ax[j, i].set_ylim(-ylim+centers[j], ylim+centers[j])
            ax[i, j].set_visible(False)
            if j == npars-1:
                ax[j, i].set_xlabel(j_key)
            if i == 0:
                ax[j, i].set_ylabel(i_key)
    plt.tight_layout()
    f.savefig(pconfig["triangle_plot"])

    pairplots = pconfig["pairplots"]
    npairplots = int(len(pairplots)/2)
    for i in range(npairplots):
        pair0 = pairplots[2*i].split('(')[1]
        pair1 = pairplots[2*i+1].split(')')[0]
        key1 = f'params--{pair0.lower()}'
        key2 = f'params--{pair1.lower()}'
        if (key1 not in keys) or (key2 not in keys):
            # The selected set of parameters is not in the forecast
            continue
        else:
            f, ax = plt.subplots(1, 1)
            i = np.where(keys == key1)[0]
            j = np.where(keys == key2)[0]
            vals, theta, inv_cache = get_ellipse(fisher, i, j, inv_cache=inv_cache)
            x0 = 2*np.sqrt(vals[0])*np.cos(np.radians(theta))
            x1 = 2*np.sqrt(vals[1])*np.cos(np.radians(theta))
            y0 = 2*np.sqrt(vals[0])*np.sin(np.radians(theta))
            y1 = 2*np.sqrt(vals[1])*np.sin(np.radians(theta))
            xlim = 3*np.sqrt(x0**2+x1**2)
            ylim = 3*np.sqrt(y0**2+y1**2)
            plot_ellipse(vals, theta, color, legend, nsigma,
                         lw, ls, centers[i], centers[j], ax=ax)
            ax.set_xlim(-xlim+centers[i], xlim+centers[i])
            ax.set_ylim(-ylim+centers[j], ylim+centers[j])
            ax.set_xlabel(pair0)
            ax.set_ylabel(pair1)
            plt.tight_layout()
            f.savefig(os.path.join(outdir, f'{pair0}--{pair1}.pdf'))


def eigsorted(cov):
    """
    Auxiliary function to return the eigenvalues, and eigenvectors
    of the Fisher matrix passsed.

    Parameters:
    -------
    cov : ndarray
        Array with the Fisher information matrix.

    Returns:
    -------
    vals: ndarray
        eigenvalues of the matrix
    vecs: ndarray
        eigenvectors of the matrix
    """
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    return vals[order], vecs[:, order]


def get_all_fish(fisher):
    """
    Auxiliary function that returns the values and angles corresponding
    to the eigenvalues and eigenvectors of a Fisher information matrix.

    Parameters:
    -----
    fisher : ndarray
        Array containing the Fisher information matrix.

    Returns:
    ------
    vals : ndarray
        Array with the eigenvalues of the Fisher matrix
    theta : ndarray
        Angle of the ellipse describing the degeneracy between a pair of parameters
    """
    inv_fisher = np.linalg.inv(fisher)
    vals, vecs = eigsorted(inv_fisher)
    theta = []
    for i in range(len(vecs)):
        for j in range(i+1, len(vecs)):
            theta.append(np.degrees(np.arctan2(vecs[i, j], vecs[i, i])))
    return vals, np.array(theta)


def get_ellipse(fisher, par1, par2, inv_cache=None):
    """
    Routine to obtain the ellipse parameters
    of a pair of parameters given a Fisher matrix.
    This routine is similar to `get_all_fish` but slices for a given pair.
    It may not give the same results as `get_all_fish` if the Fisher matrix is
    not easily invertible.

    Parameters:
    -----
    fisher : ndarray
        Array containing the Fisher matrix.
    par1 : int
        Index of the first parameter to consider
    par2: int
        Index of the second paramter to consider
    inv_cache:
        Cached inverse Fisher matrix.
    Returns:
    -----
    vals : ndarray
        eigenvalues of the particular pair.
    theta : float
        degeneracy angle of the pair.
    """
    inv_fisher = np.zeros((2, 2))
    if inv_cache is None:
        inv_cache = np.linalg.inv(fisher)
    inv_fisher[0, 0] = inv_cache[par1, par1]
    inv_fisher[0, 1] = inv_cache[par1, par2]
    inv_fisher[1, 0] = inv_cache[par2, par1]
    inv_fisher[1, 1] = inv_cache[par2, par2]
    fisher_red = np.linalg.inv(inv_fisher)
    smallC = np.linalg.inv(fisher_red)

    vals, vecs = eigsorted(smallC)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    return vals, theta, inv_cache


def plot_ellipse(vals, theta, color, label, nsigma, lw, ls, cent0, cent1, ax=None):
    """
    Routine to generate plots of a pair of Parameters

    Parameters:
    ------
    vals : ndarray
        Array of eigenvalues
    theta : float
        Angle of degeneracy (in degrees).
    color : sting or matplotlib.color
        Color of the line of the Ellipse object drawn.
    label: string
        Label describing the Ellipse to plot.
    lw : float
        Linewidth parameter from matplotlib.
    ls : string
        Linestyle parameter from matplotlib.
    cent0 : float
        x-coordinate of the centroid of the Ellipse.
    cent1 : float
        y-coordinate of the centroid of the Ellipse.
    ax : matplotlib.pyplot.axis
        Axis in which to plot the Ellipse

    Returns:
    -----
    Ellipse : matplotlib.patches.Ellipse
        Ellipse for the selected values
    """
    for i in range(nsigma):
        width, height = 2 * (i+1) * np.sqrt(vals)
        if i == (nsigma-1):
            ellip = Ellipse(xy=[cent0, cent1], width=width, height=height,
                            angle=theta, facecolor='none', label=label,
                            edgecolor=color, lw=lw, ls=ls)
        else:
            ellip = Ellipse(xy=[cent0, cent1], width=width, height=height,
                            angle=theta, facecolor='none', label=label,
                            edgecolor=color, lw=lw, ls=ls)
        if ax is None:
            plt.gca().add_patch(ellip)
        else:
            ax.add_patch(ellip)
    return

    def get_FoM_all(fisher, par1, par2):
        """
        Two alternative ways to get the FoM for a pair of parameters
        (code stolen from https://github.com/CosmoLike/DESC_SRD/blob/master/fisher.py#L274)

        Parameters:
        -----
        fisher : ndarray
            Fisher matrix
        par1 : int
            Index of first parameter
        par2 : int
            Index of second parameter

        Returns:
        -----
        FOM : float
            Estimate of FoM.
        FOM2 : float
            Alternative estimate of FoM. Choose wisely.
        """
        fisher_inv = np.linalg.inv(fisher)
        cov_de = fisher_inv[np.ix_([par1, par2], [par1, par2])]
        FOM = 1./np.sqrt(fisher_inv[par1, par1]*fisher_inv[par2, par2] -
                         fisher_inv[par1, par2]*fisher_inv[par2, par1])
        FOM2 = np.sqrt(np.linalg.det(np.linalg.inv(cov_de)))
        return FOM, FOM2
