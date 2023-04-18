import numpy as np
import pyccl as ccl
from numpy.linalg import LinAlgError
from tjpcov.covariance_gaussian_fsky import FourierGaussianFsky


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
        trobj = S.get_tracer(tr)
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
        Sacc object containing the data-vector for which we want the covariance
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
    cov_all : np.ndarray
        Numpy array containing the covariance matrix.
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


def get_SRD_cov(config, S, return_inv=True):
    """
    Read covariance file from SRD v1:
    https://github.com/LSSTDESC/Requirements/tree/master/forecasting/WL-LSS-CL/cov

    Parameters:
    -----------
    config : dict
        The dictinary containt the relevant two_point section of the config.
    S : sacc.Sacc
        Sacc object containing the data vector for which to compute the covariance.
    return_inv : bool
        If `True` it returns the inverse covariance rather than the covariance matrix.
    Returns:
    --------
    cov_all : np.ndarray
        Numpy array containing the covariance matrix (or its inverse if `return_inv == True`).
    """
    if 'SRD_inv_cov_path' not in config.keys():
        raise ValueError('SRD_inv_cov_path is needed to use SRD covariance.')
    inv_cov_in = np.loadtxt(config['SRD_inv_cov_path'])
    nx = int(np.sqrt(inv_cov_in.shape[0]))
    # This might be inefficient but helps to visualize the object
    inv_cov = np.zeros((nx, nx))
    inv_cov[inv_cov_in[:, 0].astype(np.int16), inv_cov_in[:, 1].astype(np.int16)] = inv_cov_in[:, 2]
    # Data combinations for Y1 as per SRD v1
    data_combs_y1 = [('src0', 'src0'), ('src0', 'src1'), ('src0', 'src2'), ('src0', 'src3'),
                     ('src0', 'src4'), ('src1', 'src1'), ('src1', 'src2'), ('src1', 'src3'),
                     ('src1', 'src4'), ('src2', 'src2'), ('src2', 'src3'), ('src2', 'src4'),
                     ('src3', 'src3'), ('src3', 'src4'), ('src4', 'src4'),
                     ('lens0', 'src2'), ('lens0', 'src3'), ('lens0', 'src4'),
                     ('lens1', 'src3'), ('lens1', 'src4'), ('lens2', 'src4'),
                     ('lens3', 'src4'), ('lens0', 'lens0'), ('lens1', 'lens1'),
                     ('lens2', 'lens2'), ('lens3', 'lens3'), ('lens4', 'lens4')]
    # Data combinations for Y10 as per SRD v1
    data_combs_y10 = [('src0', 'src0'), ('src0', 'src1'), ('src0', 'src2'), ('src0', 'src3'),
                      ('src0', 'src4'), ('src1', 'src1'), ('src1', 'src2'), ('src1', 'src3'),
                      ('src1', 'src4'), ('src2', 'src2'), ('src2', 'src3'), ('src2', 'src4'),
                      ('src3', 'src3'), ('src3', 'src4'), ('src4', 'src4'),
                      ('lens0', 'src1'), ('lens0', 'src2'), ('lens0', 'src3'), ('lens0', 'src4'),
                      ('lens1', 'src1'), ('lens1', 'src2'), ('lens1', 'src3'), ('lens1', 'src4'),
                      ('lens2', 'src2'), ('lens2', 'src3'), ('lens2', 'src4'),
                      ('lens3', 'src2'), ('lens3', 'src3'), ('lens3', 'src4'),
                      ('lens4', 'src2'), ('lens4', 'src4'),
                      ('lens5', 'src3'), ('lens5', 'src4'),
                      ('lens6', 'src3'), ('lens6', 'src4'),
                      ('lens7', 'src3'), ('lens7', 'src4'),
                      ('lens8', 'src4'),
                      ('lens9', 'src4'),
                      ('lens0', 'lens0'), ('lens1', 'lens1'), ('lens2', 'lens2'),
                      ('lens3', 'lens3'), ('lens4', 'lens4'), ('lens5', 'lens5'),
                      ('lens6', 'lens6'), ('lens7', 'lens7'), ('lens8', 'lens8'),
                      ('lens9', 'lens9')]

    if 'Y10' in config['SRD_inv_cov_path']:
        data_combs = data_combs_y10
    else:
        data_combs = data_combs_y1

    inv_cov_sacc_all = np.zeros((len(S.mean), len(S.mean)))
    for i, comb1 in enumerate(data_combs):
        dtype_here1 = S.get_data_types(tracers=comb1)[0]
        inds1 = S.indices(data_type=dtype_here1, tracers=comb1)
        for j, comb2 in enumerate(data_combs):
            dtype_here2 = S.get_data_types(tracers=comb2)[0]
            inds2 = S.indices(data_type=dtype_here2, tracers=comb2)
            inds_all = np.meshgrid(inds1, inds2)
            inv_cov_sacc_all[inds_all[0].T, inds_all[1].T] = inv_cov[20*i:20*i+len(inds1),
                                                                     20*j:20*j+len(inds2)]
    if return_inv:
        return inv_cov_sacc_all
    else:
        try:
            return np.linalg.inv(inv_cov_sacc_all)
        except LinAlgError:
            return np.linalg.pinv(inv_cov_sacc_all)


class TJPCovGaus(FourierGaussianFsky):
    """
    Class to patch FourierGaussianFsky to work with Augur
    """
    def __init__(self, config):
        super().__init__(config)
        self.tracer_Noise = self.tracer_Noise_coupled

    def get_binning_info(self):
        ell_eff = self.get_ell_eff()
        if 'ell_edges' in self.config['tjpcov']['binning_info'].keys():
            ell_edges = self.config['tjpcov']['binning_info']['ell_edges']
        ell_min = np.min(ell_edges)
        ell_max = np.max(ell_edges)
        nbpw = ell_max - ell_min
        ell = np.linspace(ell_min, ell_max, nbpw+1).astype(np.int32)
        return ell, ell_eff, ell_edges

    def get_tracer_info(self, return_noise_coupled=False):
        _ = super().get_tracer_info(return_noise_coupled=True)
        self.tracer_Noise = self.tracer_Noise_coupled
        if return_noise_coupled:
            return (
                self.ccl_tracers,
                self.tracer_Noise,
                self.tracer_Noise_coupled)
        else:
            return self.ccl_tracers, self.tracer_Noise
