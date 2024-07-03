import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
from astropy.table import Table
from scipy.stats import norm, chi2
from augur.utils.config_io import parse_config
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
    config = parse_config(config)
    pconfig = config["postprocess"]
    try:
        fid_params = Table.read(config["fisher"]["fid_output"], format='ascii')
    except FileNotFoundError:
        print("Obtain a fiducial file first by running get_fisher_matrix \
               or provide your own text file.")
    var_params = config["fisher"]["var_pars"]
    try:
        fisher = np.loadtxt(config["fisher"]["output"])
    except FileNotFoundError:
        print("Obtain a Fisher matrix first via analyze or provide your own text file.")
    npars = fisher.shape[0]
    keys = np.array(var_params)
    outdir = config["postprocess"]["outdir"]

    if "labels" not in pconfig.keys():
        labels = keys
    else:
        labels = pconfig["labels"]
    xsize, ysize = pconfig["size"] if "size" in pconfig.keys() else (48, 48)

    if "centers" in pconfig.keys():
        centers = pconfig["centers"]
    else:
        # Setting the centers at the fiducial values of the parameters
        centers = np.zeros(npars)
        ind = 0
        for key in fid_params.keys():
            if key in var_params:
                centers[ind] = fid_params[key]
                ind += 1

    lw = pconfig["linewidth"] if "linewidth" in pconfig.keys() else 1
    ls = pconfig["linestyle"] if "linestyle" in pconfig.keys() else "-"
    edgecolors = pconfig["linecolor"] if "linecolor" in pconfig.keys() else "k"
    facecolors = pconfig["facecolor"] if "facecolor" in pconfig.keys() else "none"
    CL = pconfig["CL"] if "CL" in pconfig.keys() else 0.68
    f, ax = plt.subplots(npars, npars, figsize=(xsize, ysize))

    try:
        inv_cache = np.linalg.inv(fisher)
    except np.linalg.LinAlgError:
        print("Fisher matrix non-invertible -- quitting...")

    for i in range(npars):
        i_key = labels[i]
        for j in range(i+1, npars):
            j_key = labels[j]
            print(i_key, j_key)
            inv_fisher = np.zeros((2, 2))
            inv_fisher[0, 0] = inv_cache[i, i]
            inv_fisher[0, 1] = inv_cache[i, j]
            inv_fisher[1, 0] = inv_cache[j, i]
            inv_fisher[1, 1] = inv_cache[j, j]
            sig0, sig1 = draw_fisher_ellipses(ax[j, i], inv_fisher, facecolors,
                                              edgecolors, ls, lw,
                                              mu=(centers[i], centers[j]), CL=CL)
            # Create the 1D histogram for i, i
            if j == i+1:
                xarr = np.linspace(centers[i]-5*sig0,
                                   centers[i]+5*sig0, 200)
                ax[i, i].plot(xarr, norm.pdf(xarr, centers[i], sig0))
            ax[j, i].set_xlim(-sig0+centers[i], sig0+centers[i])
            ax[j, i].set_ylim(-sig1+centers[j], sig1+centers[j])
            ax[i, j].set_visible(False)
            if j == npars-1:
                ax[j, i].set_xlabel(i_key)
            if i == 0:
                ax[j, i].set_ylabel(j_key)
    plt.tight_layout()
    f.savefig(pconfig["triangle_plot"])
    if "pairplots" in pconfig.keys():
        pairplots = pconfig["pairplots"]
        npairplots = int(len(pairplots)/2)
    else:
        npairplots = 0
    for i in range(npairplots):
        pair0 = pairplots[2*i].split("(")[1]
        pair1 = pairplots[2*i+1].split(")")[0]
        key1 = f"{pair0}"
        key2 = f"{pair1}"
        if (key1 not in keys) or (key2 not in keys):
            # The selected set of parameters is not in the forecast
            continue
        else:
            f, ax = plt.subplots(1, 1)
            ii = np.where(keys == key1)[0][0]
            jj = np.where(keys == key2)[0][0]
            inv_fisher = np.zeros((2, 2))
            inv_fisher[0, 0] = inv_cache[ii, ii]
            inv_fisher[0, 1] = inv_cache[ii, jj]
            inv_fisher[1, 0] = inv_cache[jj, ii]
            inv_fisher[1, 1] = inv_cache[jj, jj]
            sig0, sig1 = draw_fisher_ellipses(ax, inv_fisher, facecolors,
                                              edgecolors, ls, lw,
                                              mu=(centers[ii], centers[jj]), CL=CL)
            ax.set_xlim(-sig0+centers[ii], sig0+centers[ii])
            ax.set_ylim(-sig1+centers[jj], sig1+centers[jj])
            ax.set_xlabel(pair0)
            ax.set_ylabel(pair1)
            plt.tight_layout()
            f.savefig(os.path.join(outdir, f"{pair0}--{pair1}.pdf"))

    # w0 -- wa plots are always made
    iw = np.where(keys == "w0")[0][0]
    iwa = np.where(keys == "wa")[0][0]
    sig_w0 = np.sqrt(inv_cache[iw, iw])
    sig_wa = np.sqrt(inv_cache[iwa, iwa])
    FOM, FOM2 = get_FoM_all(fisher, iw, iwa, CL)
    fisher_table = Table([[CL], [FOM], [FOM2], [sig_w0], [sig_wa]],
                         names=("CL", "FoM", "FoM (alt.)",
                                "sigma_w0", "sigma_wa"))
    fisher_table.write(pconfig["latex_table"], format="latex", overwrite=True)


def draw_fisher_ellipses(ax, inv_F, facecolors, edgecolors, linestyles, linewidth,
                         mu=[0, 0], CL=0.95):
    """
    Draw uncertainty ellipses for a set of Fisher matrices.

    All ellipses are drawn with the same center. Contours are scaled
    to enclose a specified probability (CL) assuming Gaussian errors.

    Script copied from: https://github.com/LSSTDESC/Requirements/blob/v1.0.1/notebooks/DETF.ipynb
    Parameters
    ----------
    ax : matplotlib axes
        Where the ellipses should be drawn.
    inv_F : array
        Array of dimensions (2,2) encoding 2x2 inverse of Fisher sub-matrices.
    facecolors : matplotlib.colors.Color
        Matplotlib colors for filling the ellipse interior.
    edgecolors : matplotlib.colors.Color
        Matplotlib colors for drawing the ellipse boundary.
    linestyles : matpolotlib.linestyles
        Matplotlib line styles for drawing the ellipse boundary.
    linewidth : float
        Linewidth of the ellipse
    mu : tuple
        Center coordinates for the ellipse
    CL : float
        Confidence level in the range [0,1] that the ellipses should enclose.
    """
    # Convert CL into covariance scale factor.
    assert CL > 0 and CL < 1
    scale = chi2.ppf(q=CL, df=2)

    # Calculate scaled covariances as inverse of Fisher matrices.
    C = scale * inv_F

    # Calculate the discriminant of each scaled covariance.
    D = np.sqrt((C[0, 0] - C[1, 1]) ** 2 + 4 * C[0, 1] ** 2)

    # Calculate ellipse geometric parameters.
    widths = 2 * np.sqrt(0.5 * (C[0, 0] + C[1, 1] + D))
    heights = 2 * np.sqrt(0.5 * (C[0, 0] + C[1, 1] - D))
    angles = np.rad2deg(0.5 * np.arctan2(2 * C[0, 1], C[0, 0] - C[1, 1]))

    # Build a collection of the ellipses to display.
    ellipses = EllipseCollection(
        widths, heights, angles, units="xy", offsets=mu,
        transOffset=ax.transData, facecolors=facecolors,
        edgecolors=edgecolors, linestyles=linestyles, linewidth=linewidth)
    ax.add_collection(ellipses)

    sig0 = np.sqrt(C[0, 0])
    sig1 = np.sqrt(C[1, 1])
    return sig0, sig1


def get_FoM_all(fisher, par1, par2, CL):
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
    CL : float
        Confidence level (between 0 and 1)
    Returns:
    -----
    FOM : float
        Estimate of FoM.
    FOM2 : float
        Alternative estimate of FoM. Choose wisely.
    """
    assert CL > 0 and CL < 1
    scale = chi2.ppf(q=CL, df=2)
    fisher_inv = scale*np.linalg.inv(fisher)
    cov_de = fisher_inv[np.ix_([par1, par2], [par1, par2])]
    FOM = 1./np.sqrt(fisher_inv[par1, par1]*fisher_inv[par2, par2] -
                     fisher_inv[par1, par2]*fisher_inv[par2, par1])
    FOM2 = np.sqrt(np.linalg.det(np.linalg.inv(cov_de)))
    return FOM, FOM2
