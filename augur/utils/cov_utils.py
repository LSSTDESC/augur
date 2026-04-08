import numpy as np
import pyccl as ccl
from tjpcov.covariance_gaussian_fsky import FourierGaussianFsky
from tjpcov.covariance_cluster_counts_gaussian import ClusterCountsGaussian as BaseClusterGaussian
from tjpcov.covariance_cluster_counts_ssc import ClusterCountsSSC as BaseClusterSSC


def get_noise_power(config, S, tracer_name, return_ndens=False):
    """ Returns noise power for tracer

    Parameters:
    ----------
    config : dict
        The dictinary containt the relevant two_point section of the config

    S : Sacc
        Sacc file containing all tracers

    tracer_name : str
        Tracer ID

    return_ndens : bool
        If `True` return the number density (in arcmin^-2)

    Returns:
    -------
    noise : float
       Noise power for Cls for that particular tracer. That is 1/nbar for
       number counts and e**2/nbar for weak lensing tracer.

    Note:
    -----
    The input number_densities are #/arcmin. The output noise has no units.
    The output number density is in arcmin^-2.
    """
    if 'src' not in tracer_name and 'lens' not in tracer_name:
        # Not a galaxy source/lens tracer; may be cluster or other (skip gracefully).
        if return_ndens:
            return 0.0, 0.0
        return 0.0

    nz_all = dict()
    nz_all['src'] = []
    nz_all['lens'] = []
    for tr in S.tracers:
        trobj = S.get_tracer(tr)
        # obtain some arbitrary number of bins
        prefix = tr.rstrip('0123456789')
        if prefix in nz_all:
            nz_all[prefix].append(trobj.nz)
    tracer_prefix = tracer_name.rstrip('0123456789')
    tracer_bin_str = tracer_name[len(tracer_prefix):]
    if tracer_bin_str == '':
        raise ValueError(
            f"Tracer name '{tracer_name}' must end with a bin index."
        )
    tracer_bin = int(tracer_bin_str)

    norm = dict()
    if 'src' in tracer_name:
        nz_all['src'] = np.array(nz_all['src'])
        norm['src'] = np.sum(nz_all['src'], axis=1)/np.sum(nz_all['src'])

        ndens = config['sources']['ndens']
        ndens *= norm['src'][tracer_bin]
    elif 'lens' in tracer_name:
        nz_all['lens'] = np.array(nz_all['lens'])
        norm['lens'] = np.sum(nz_all['lens'], axis=1)/np.sum(nz_all['lens'])
        ndens = config['lenses']['ndens']
        ndens *= norm['lens'][tracer_bin]
    nbar = ndens * (180 * 60 / np.pi) ** 2  # per steradian

    if 'src' in tracer_name:
        noise_power = config['sources']['ellipticity_error'] ** 2 / nbar
    elif 'lens' in tracer_name:
        noise_power = 1 / nbar
    else:
        print("Cannot do error for source of kind %s." % tracer_prefix)
        raise NotImplementedError
    if return_ndens:
        return noise_power, ndens
    else:
        return noise_power


def get_cluster_noise_power(config, S, cluster_tracer_name, return_cluster_density=False):
    """
    Compute cluster shot noise (Poisson fluctuation noise) for TJPCov.

    Cluster counts have Poisson shot noise proportional to 1/N_cluster.
    TJPCov's ClusterCountsGaussian computes this automatically from redshift
    and richness bin definitions in the SACC tracer metadata.

    This function provides a placeholder interface for Augur to extract or
    configure cluster-specific noise parameters. Currently, TJPCov handles
    the full computation internally via the Halo Mass Function and Mass-Richness
    relation, so this mainly returns a flag indicating that cluster noise is
    handled separately.

    Parameters:
    -----------
    config : dict
        Augur configuration dictionary.
    S : sacc.Sacc
        SACC object containing cluster tracers.
    cluster_tracer_name : str
        Name of the cluster tracer (e.g., 'cluster_0').
    return_cluster_density : bool
        If True, also return an effective cluster density estimate.

    Returns:
    --------
    noise : float
        Placeholder noise value (TJPCov computes actual shot noise).
        Returns 0.0 as a signal that cluster noise is handled by TJPCov.
    cluster_density : float, optional
        Estimated cluster number density if return_cluster_density=True.
    """
    # Cluster noise is handled internally by TJPCov via ClusterCountsGaussian.
    # This function exists as an interface for future cluster-specific
    # configuration (e.g., if Augur needs to override TJPCov defaults).

    tracer_obj = S.get_tracer(cluster_tracer_name)

    # Estimate cluster density if available from config
    cluster_density = config.get('cluster_counts', {}).get('ndens', 0.0)

    if return_cluster_density:
        return 0.0, cluster_density
    return 0.0


def get_cluster_properties(config, S):
    """
    Extract cluster tracer metadata and properties from SACC for TJPCov.

    TJPCov's ClusterCountsGaussian and ClusterCountsSSC extract
    cluster properties automatically via load_from_sacc(). This function
    provides a reference point for Augur to cross-check or override
    cluster-specific settings (e.g., min_halo_mass, has_mproxy).

    Parameters:
    -----------
    config : dict
        Augur configuration dictionary with optional cluster_counts section.
    S : sacc.Sacc
        SACC object with cluster tracers and data points.

    Returns:
    --------
    cluster_info : dict
        Dictionary with cluster metadata:
        - 'tracer_names': list of cluster tracer names in SACC
        - 'z_bins': list of (z_min, z_max) tuples
        - 'richness_bins': list of (log10_lambda_min, log10_lambda_max) tuples
        - 'survey_area': survey area in steradians
        - 'min_halo_mass': minimum mass for integration (from config or default 1e13)
    """
    cluster_info = {}

    # Find all cluster tracers and bin definitions
    cluster_tracers = []
    z_bins = []
    richness_bins = []

    for tracer in S.tracers:
        tracer_obj = S.get_tracer(tracer)

        # Identify bin_z tracers
        if hasattr(tracer_obj, 'quantity') and tracer_obj.quantity == 'redshift':
            # Assuming tracer has (z_min, z_max) or similar metadata
            z_bins.append(tracer)

        # Identify richness/lambda bin tracers
        if hasattr(tracer_obj, 'quantity') and 'richness' in str(tracer_obj.quantity).lower():
            richness_bins.append(tracer)

        # Identify cluster data tracers (those in cluster_counts data points)
        # This is detected via SACC data type, done at covariance.py level

    cluster_info['z_bins'] = z_bins
    cluster_info['richness_bins'] = richness_bins
    cluster_info['min_halo_mass'] = config.get('cluster_counts', {}).get('min_halo_mass', 1e13)
    cluster_info['has_mproxy'] = config.get('cluster_counts', {}).get('has_mproxy', True)

    # Extract survey area from survey tracer or assume full sky
    survey_area = 4 * np.pi  # Full sky in steradians
    for tracer in S.tracers:
        tracer_obj = S.get_tracer(tracer)
        if hasattr(tracer_obj, 'quantity') and tracer_obj.quantity == 'survey_area':
            survey_area = tracer_obj.scale  # If SACC stores it as a scale factor
            break

    cluster_info['survey_area'] = survey_area

    return cluster_info


def get_gaus_cov(S, lk, cosmo, fsky, config):
    """
    Basic implementation of Gaussian covariance using the mode-counting formula
    and fsky approximation. Assumes that all ell-edges are the same for (3x)2pt
    statistics and uniform bin-widths in ell.

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

    ell_edges_by_stat = {
        stat_name: np.asarray(eval(stat_cfg['ell_edges']))
        for stat_name, stat_cfg in config['statistics'].items()
    }

    def _noise_between(tr_a_name, tr_b_name):
        """Uncorrelated noise term N_ab (non-zero only for auto-tracer pairs)."""
        if tr_a_name != tr_b_name:
            return 0.0
        return get_noise_power(config, S, tr_a_name)

    # Loop over statistic in the likelihood (assuming 3x2pt so far)
    for i, myst1 in enumerate(lk.statistics):
        myst1 = myst1.statistic
        tr1_name = myst1.source0.sacc_tracer
        tr2_name = myst1.source1.sacc_tracer
        tr1 = myst1.source0.tracers[0].ccl_tracer  # Pulling out the tracers
        tr2 = myst1.source1.tracers[0].ccl_tracer
        ell12 = myst1.ells
        # Loop over upper-triangle and fill lower-triangle by symmetry
        for j in range(i, len(lk.statistics)):
            myst2 = lk.statistics[j]
            myst2 = myst2.statistic
            tr3_name = myst2.source0.sacc_tracer
            tr4_name = myst2.source1.sacc_tracer
            tr3 = myst2.source0.tracers[0].ccl_tracer
            tr4 = myst2.source1.tracers[0].ccl_tracer
            ell34 = myst2.ells
            # Assuming that everything has the same ell-edges and we are just changing the length
            # TODO update this for a more general case
            if len(ell34) < len(ell12):
                ells_here = ell34
                stat_here = myst2.sacc_data_type
            else:
                ells_here = ell12
                stat_here = myst1.sacc_data_type
            # Get the necessary Cls
            cls13 = ccl.angular_cl(cosmo, tr1, tr3, ells_here)
            cls24 = ccl.angular_cl(cosmo, tr2, tr4, ells_here)
            cls14 = ccl.angular_cl(cosmo, tr1, tr4, ells_here)
            cls23 = ccl.angular_cl(cosmo, tr2, tr3, ells_here)

            # Add uncorrelated noise (shot/shape) terms for auto-tracer pairs:
            # C_ab -> C_ab + N_ab, where N_ab != 0 only if a == b.
            cls13 += _noise_between(tr1_name, tr3_name)
            cls24 += _noise_between(tr2_name, tr4_name)
            cls14 += _noise_between(tr1_name, tr4_name)
            cls23 += _noise_between(tr2_name, tr3_name)

            # Normalization factor
            dell = np.diff(ell_edges_by_stat[stat_here])[:len(ells_here)]
            norm = dell*(2*ells_here+1)*fsky
            cov_here = cls13*cls24 + cls14*cls23
            cov_here /= norm
            # The following lines only work if the ell-edges are constant across the probes
            # and we just vary the length
            n_ells = min(len(ell12), len(ell34))
            # Use the sacc indices to write the matrix in the correct order
            cov_all[myst1.sacc_indices[:n_ells], myst2.sacc_indices[:n_ells]] = cov_here[:n_ells]
            cov_all[myst2.sacc_indices[:n_ells], myst1.sacc_indices[:n_ells]] = cov_here[:n_ells]
    return cov_all


def get_SRD_cov(config, S):
    """
    Read covariance file from SRD v1:
    https://github.com/LSSTDESC/Requirements/tree/master/forecasting/WL-LSS-CL/cov

    Parameters:
    -----------
    config : dict
        Dictionary containing the relevant two_point section of the config.
    S : sacc.Sacc
        Sacc object containing the data vector for which to compute the covariance.

    Returns:
    --------
    cov_all : np.ndarray
        Numpy array containing the SRD covariance matrix with the relevant shape as established
        by the input sacc file S.
    """
    if 'SRD_cov_path' not in config.keys():
        raise ValueError('SRD_cov_path is needed to use SRD covariance.')
    cov_in = np.load(config['SRD_cov_path'])
    ncls = 20
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
                      ('lens4', 'src2'), ('lens4', 'src3'), ('lens4', 'src4'),
                      ('lens5', 'src3'), ('lens5', 'src4'),
                      ('lens6', 'src3'), ('lens6', 'src4'),
                      ('lens7', 'src3'), ('lens7', 'src4'),
                      ('lens8', 'src4'),
                      ('lens9', 'src4'),
                      ('lens0', 'lens0'), ('lens1', 'lens1'), ('lens2', 'lens2'),
                      ('lens3', 'lens3'), ('lens4', 'lens4'), ('lens5', 'lens5'),
                      ('lens6', 'lens6'), ('lens7', 'lens7'), ('lens8', 'lens8'),
                      ('lens9', 'lens9')]

    if 'Y10' in config['SRD_cov_path']:
        data_combs = data_combs_y10
    else:
        data_combs = data_combs_y1

    cov_sacc_all = np.zeros((len(S.mean), len(S.mean)))
    for i, comb1 in enumerate(data_combs):
        dtype_here1 = S.get_data_types(tracers=comb1)[0]
        inds1 = S.indices(data_type=dtype_here1, tracers=comb1)
        for j, comb2 in enumerate(data_combs):
            dtype_here2 = S.get_data_types(tracers=comb2)[0]
            inds2 = S.indices(data_type=dtype_here2, tracers=comb2)
            inds_all = np.meshgrid(inds1, inds2)
            cov_sacc_all[inds_all[0].T, inds_all[1].T] = cov_in[ncls*i:ncls*i+len(inds1),
                                                                ncls*j:ncls*j+len(inds2)]
    return cov_sacc_all


class TJPCovGaus(FourierGaussianFsky):
    """
    Class to patch FourierGaussianFsky to work with Augur
    """

    def __init__(self, config):
        super().__init__(config)
        # self.tracer_Noise = self.tracer_Noise_coupled

    def get_tracer_info(self, return_noise_coupled=False):
        """Return tracer info while ensuring CMB noise keys exist.

        TJPCov's default Fourier tracer parser creates CMB tracers but may not
        populate ``tracer_Noise`` entries for them. For covariance blocks that
        include CMB auto/cross terms this can cause missing-key failures.

        Augur provides an optional scalar ``tjpcov.cmb_noise`` (default 0.0)
        and applies it to all ``quantity == 'cmb_convergence'`` tracers.
        """
        ccl_tracers, tracer_Noise, tracer_Noise_coupled = super().get_tracer_info(
            return_noise_coupled=True
        )

        cmb_noise = float(self.config['tjpcov'].get('cmb_noise', 0.0))
        sacc_file = self.io.get_sacc_file()
        for tracer in sacc_file.tracers:
            tracer_dat = sacc_file.get_tracer(tracer)
            if getattr(tracer_dat, 'quantity', None) == 'cmb_convergence':
                tracer_Noise.setdefault(tracer, cmb_noise)

        self.ccl_tracers = ccl_tracers
        self.tracer_Noise = tracer_Noise
        self.tracer_Noise_coupled = tracer_Noise_coupled

        if return_noise_coupled:
            return ccl_tracers, tracer_Noise, tracer_Noise_coupled
        return ccl_tracers, tracer_Noise

    def get_binning_info(self):
        ell_eff = self.get_ell_eff()
        if 'ell_edges' not in self.config['tjpcov']['binning_info'].keys():
            raise ValueError(
                "`tjpcov.binning_info.ell_edges` must be defined in the configuration "
                "when using TJPCov with Augur."
            )
        ell_edges = self.config['tjpcov']['binning_info']['ell_edges']
        ell_min = np.min(ell_edges)
        ell_max = np.max(ell_edges)
        nbpw = int(ell_max - ell_min)
        ell = np.linspace(ell_min, ell_max, nbpw+1).astype(np.int32)
        return ell, ell_eff, ell_edges


class TJPCovClusterGaus(BaseClusterGaussian):
    """
    Wrapper around TJPCov's ClusterCountsGaussian for Augur integration.

    Provides a minimal pass-through interface for cluster number-count shot noise
    (Poisson + mass-richness relation) covariance computation.

    Configuration keys (in config['tjpcov']):
        - sacc_file: SACC object (required)
        - cosmo: pyccl.Cosmology (required)
        - fsky: float in [0, 1]
        - min_halo_mass: float (default 1e13 M_sun)
        - has_mproxy: bool (default True, include mass-richness scatter)

    Example:
        config = {'tjpcov': {'sacc_file': S, 'cosmo': cosmo, 'fsky': 0.5}}
        cov_gauss = TJPCovClusterGaus(config)
        cov_matrix = cov_gauss.get_covariance()  # Full covariance matrix
    """

    def __init__(self, config):
        """
        Initialize cluster Gaussian covariance wrapper.

        Parameters
        ----------
        config : dict
            Configuration dict with 'tjpcov' key containing TJPCov setup.
            Will be passed directly to parent ClusterCountsGaussian.
        """
        tjpcov_config = config.get('tjpcov', {})
        super().__init__(tjpcov_config)


class TJPCovClusterSSC(BaseClusterSSC):
    """
    Wrapper around TJPCov's ClusterCountsSSC for Augur integration.

    Provides a minimal pass-through interface for cluster number-count super-sample
    covariance (large-scale modes) computation.

    Configuration keys (in config['tjpcov']):
        - sacc_file: SACC object (required)
        - cosmo: pyccl.Cosmology (required)
        - fsky: float in [0, 1]
        - min_halo_mass: float (default 1e13 M_sun)
        - has_mproxy: bool (default True, include mass-richness scatter)

    Example:
        config = {'tjpcov': {'sacc_file': S, 'cosmo': cosmo, 'fsky': 0.5}}
        cov_ssc = TJPCovClusterSSC(config)
        cov_matrix = cov_ssc.get_covariance()  # Full covariance matrix
    """

    def __init__(self, config):
        """
        Initialize cluster SSC covariance wrapper.

        Parameters
        ----------
        config : dict
            Configuration dict with 'tjpcov' key containing TJPCov setup.
            Will be passed directly to parent ClusterCountsSSC.
        """
        tjpcov_config = config.get('tjpcov', {})
        super().__init__(tjpcov_config)
