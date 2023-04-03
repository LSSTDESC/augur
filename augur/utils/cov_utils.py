import numpy as np
import pyccl as ccl


def get_noise_power(config, S, tracer_name):
    """ Returns noise power for tracer

    Parameters:
    ----------
    config : dict
        The dictinary containt the relevant two_point section of the config

    S : Sacc
        Sacc file containing all tracers

    tracer_name : str
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
    nz_all = dict()
    nz_all['src'] = []
    nz_all['lens'] = []
    for tr in S.tracers:
        trobj = S.get_tracer(tracer_name)
        nz_all[tr[:-1]].append(trobj.nz)  # This assumes 10 or less bins
    nz_all['src'] = np.array(nz_all['src'])
    nz_all['lens'] = np.array(nz_all['lens'])
    norm = dict()
    norm['src'] = np.sum(nz_all['src'], axis=1)/np.sum(nz_all['src'])
    norm['lens'] = np.sum(nz_all['lens'], axis=1)/np.sum(nz_all['lens'])

    if 'src' in tracer_name:
        ndens = config['sources']['ndens']
    elif 'lens' in tracer_name:
        ndens = config['lenses']['ndens']
    nbar = ndens * (180 * 60 / np.pi) ** 2  # per steradian
    nbar *= norm['src'][int(tracer_name[-1])]
    if 'src' in tracer_name:
        noise_power = config['sources']['ellipticity_error'] ** 2 / nbar
    elif 'lens' in tracer_name:
        noise_power = 1 / nbar
    else:
        print("Cannot do error for source of kind %s." % (tracer_name[:-1]))
        raise NotImplementedError
    return noise_power


def get_gaus_cov(S, lk, cosmo, fsky, config):
    """
    Basic implementation of Gaussian covariance using the mode-counting formula
    and fsky approximation.

    Parameters:
    -----------
    S : Sacc
        Sacc object containing where the matrix will be stored
    lk : firecrown.likelihood
        Likelihood object containing the statistics for which we want to compute
        the covariance matrix.
    cosmo : ccl.Cosmology object
        Fiducial cosmology in which to evaluate the covariance matrix.
    fsky : float
        Fraction of the sky observed.
    config : dict
        Configuration dictionary

    Returns:
    --------
    S : Sacc
        Sacc object with the covariance updated
    """
    # Initialize big matrix
    cov_all = np.zeros((len(S.data), len(S.data)))
    # Loop over statistic in the likelihood (assuming 3x2pt so far)
    for i, myst1 in enumerate(lk.statistics):
        tr1 = myst1.source0.tracers[0].ccl_tracer  # Pulling out the tracers
        tr2 = myst1.source1.tracers[0].ccl_tracer
        ell12 = myst1._ell_or_theta
        # Loop over upper-triangle and fill lower-triangle by symmetry
        for j in range(i, len(lk.statistics)):
            myst2 = lk.statistics[j]
            tr3 = myst2.source0.tracers[0].ccl_tracer
            tr4 = myst2.source1.tracers[0].ccl_tracer
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
            if (i == j) & (myst1.source0.sacc_tracer == myst1.source1.sacc_tracer):
                cov_here += get_noise_power(config, S, myst1.source0.sacc_tracer)**2
            # The following lines only work if the ell-edges are constant across the probes
            # and we just vary the length
            n_ells = min(len(ell12), len(ell34))
            # Use the sacc indices to write the matrix in the correct order
            cov_all[myst1.sacc_indices[:n_ells], myst2.sacc_indices[:n_ells]] = cov_here[:n_ells]
            cov_all[myst2.sacc_indices[:n_ells], myst1.sacc_indices[:n_ells]] = cov_here[:n_ells]
    return cov_all
