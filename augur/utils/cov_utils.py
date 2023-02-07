import numpy as np
import pyccl as ccl

def get_noise_power(config, src):
    """ Returns noise power for tracer

    Parameters:
    ----------
    config : dict
        The dictinary containt the relevant two_point section of the config

    src : str
        Tracer ID

    Returns:
    -------
    noise : float
       Noise power for Cls for that particular tracer. That is 1/nbar for
       number counts and e**2/nbar for weak lensing tracer.

    Note:
    -----
    The input number_densities are #/arcmin.
    The output units are in steradian.
    """

    d = config["sources"][src]
    nbar = d["number_density"] * (180 * 60 / np.pi) ** 2  # per steradian
    kind = d["kind"]
    if kind == "WLSource":
        noise_power = d["ellipticity_error"] ** 2 / nbar
    elif kind == "NumberCountsSource":
        noise_power = 1 / nbar
    else:
        print("Cannot do error for source of kind %s." % (kind))
        raise NotImplementedError
    return noise_power

def get_gaus_cov(S, lk, cosmo, fsky):
    """
    Basic implementation of Gaussian covariance using the mode-counting formula
    and fsky approximation.
    
    Parameters:
    -----------
    S : Sacc object. Sacc object containing where the matrix will be stored
    lk : firecrown likelihood object containing the statistics for which we want to compute
        the covariance matrix.
    cosmo : ccl.Cosmology object. Fiducial cosmology in which to evaluate the covariance matrix.
    fsky : float. Fraction of the sky observed.

    Returns:
    --------
    S : Sacc object with the covariance updated
    """
    # Initialize big matrix
    cov_all = np.zeros((len(S.data), len(S.data)))
    # Loop over statistic in the likelihood (assuming 3x2pt so far)
    for i, myst1 in enumerate(lk.statistics):
        tr1 = myst1.source0.tracer  # Pulling out the tracers
        tr2 = myst1.source1.tracer
        ell12 = myst1._ell_or_theta
        # Loop over upper-triangle and fill lower-triangle by symmetry
        for j in range(i, len(lk.statistics)):
            myst2 = lk.statistics[j]
            tr3 = myst2.source0.tracer
            tr4 = myst2.source1.tracer
            ell34 = myst2._ell_or_theta
            # Assuming that everything has the same ell-edges and we are just changing the length
            # TODO update this for a more general case
            if len(ell34) < len(ell12):
                ells_here = ell34.astype(np.int16)
            else:
                ells_here = ell12.astype(np.int16)
            # Get the necessary Cls 
            cls13 = ccl.angular_cl(cosmo, tr1, tr3, ells_here)
            cls24 = ccl.angular_cl(cosmo, tr2, tr4, ells_here)
            cls14 = ccl.angular_cl(cosmo, tr1, tr4, ells_here)
            cls23 = ccl.angular_cl(cosmo, tr2, tr3, ells_here)
            # Normalization factor
            norm = np.gradient(ells_here)*(2*ells_here+1)*fsky
            cov_here = cls13*cls24 + cls14*cls23
            cov_here /= norm
            # The following lines only work if the ell-edges are constant across the probes, and we just vary the length
            n_ells = min(len(ell12), len(ell34))
            # Use the sacc indices to write the matrix in the correct order
            cov_all[myst1.sacc_inds[:n_ells], myst2.sacc_inds[:n_ells]] = cov_here[:n_ells]
            cov_all[myst2.sacc_inds[:n_ells], myst1.sacc_inds[:n_ells]] = cov_here[:n_ells]
    return cov_all
