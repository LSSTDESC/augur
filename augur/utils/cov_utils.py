import logging
import warnings
import numpy as np
import pyccl as ccl
import sacc
from firecrown.likelihood.two_point import TwoPoint
from tjpcov.covariance_gaussian_fsky import FourierGaussianFsky
from tjpcov.covariance_cluster_counts_gaussian import ClusterCountsGaussian as BaseClusterGaussian
from tjpcov.covariance_cluster_counts_ssc import ClusterCountsSSC as BaseClusterSSC
from augur.utils.config_io import parse_array

logger = logging.getLogger(__name__)


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
    The input number_densities are #/arcmin^2.
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
    keyw, section = ('src', 'sources') if 'src' in tracer_name else ('lens', 'lenses')
    ndens = parse_array(config[section]['ndens'])
    nbins = len(nz_all[keyw])
    if ndens.shape == ():
        # Split between the tomographic bins
        if ndens <= 0:
            raise ValueError(f"ndens for {keyw} must be positive.")
        nz = np.array(nz_all[keyw])
        norm = np.sum(nz, axis=1)/np.sum(nz)
        ndens = float(ndens * norm[tracer_bin])
    elif ndens.shape == (nbins,):
        # Individually-defined number densities for each tomographic bin
        if not np.all(ndens > 0):
            raise ValueError(f"ndens for {keyw} must be positive.")
        ndens = float(ndens[tracer_bin])
    else:
        raise ValueError(
            f"ndens for {section} must be a scalar or 1D array of length {nbins}."
        )

    nbar = ndens * (180 * 60 / np.pi) ** 2  # per steradian

    if 'src' in tracer_name:
        noise_power = config['sources']['ellipticity_error'] ** 2 / nbar
    elif 'lens' in tracer_name:
        noise_power = 1 / nbar
    else:
        logger.error("Cannot do error for source of kind %s.", tracer_prefix)
        raise NotImplementedError
    if return_ndens:
        return noise_power, ndens
    else:
        return noise_power


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

    # Merge ell_edges from all harmonic-space config sections:
    # standard galaxy statistics and CMB lensing statistics.
    ell_edges_by_stat = {}
    for stat_name, stat_cfg in config.get('statistics', {}).items():
        ell_edges_by_stat[stat_name] = parse_array(stat_cfg['ell_edges'])
    cmb_stats = config.get('cmb_lensing', {}).get('statistics', {})
    for stat_name, stat_cfg in cmb_stats.items():
        ell_edges_by_stat[stat_name] = parse_array(stat_cfg['ell_edges'])

    # CMB convergence noise (reconstruction noise floor)
    cmb_noise_cl = float(
        config.get('cmb_lensing', {}).get('noise_cl', 0.0)
    )

    def _noise_between(tr_a_name, tr_b_name):
        """Uncorrelated noise term N_ab (non-zero only for auto-tracer pairs)."""
        if tr_a_name != tr_b_name:
            return 0.0
        # CMB convergence reconstruction noise
        if tr_a_name == 'cmb_convergence':
            return cmb_noise_cl
        return get_noise_power(config, S, tr_a_name)

    # Loop over the two-point statistics; cluster blocks are filled separately.
    stats = [g.statistic for g in lk.statistics if isinstance(g.statistic, TwoPoint)]
    for i, myst1 in enumerate(stats):
        tr1_name = myst1.source0.sacc_tracer
        tr2_name = myst1.source1.sacc_tracer
        tr1 = myst1.source0.tracers[0].ccl_tracer  # Pulling out the tracers
        tr2 = myst1.source1.tracers[0].ccl_tracer
        ell12 = myst1.ells
        # Loop over upper-triangle and fill lower-triangle by symmetry
        for j in range(i, len(stats)):
            myst2 = stats[j]
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
                     ('src2', 'lens0'), ('src3', 'lens0'), ('src4', 'lens0'),
                     ('src3', 'lens1'), ('src4', 'lens1'), ('src4', 'lens2'),
                     ('src4', 'lens3'), ('lens0', 'lens0'), ('lens1', 'lens1'),
                     ('lens2', 'lens2'), ('lens3', 'lens3'), ('lens4', 'lens4')]
    # Data combinations for Y10 as per SRD v1
    data_combs_y10 = [('src0', 'src0'), ('src0', 'src1'), ('src0', 'src2'), ('src0', 'src3'),
                      ('src0', 'src4'), ('src1', 'src1'), ('src1', 'src2'), ('src1', 'src3'),
                      ('src1', 'src4'), ('src2', 'src2'), ('src2', 'src3'), ('src2', 'src4'),
                      ('src3', 'src3'), ('src3', 'src4'), ('src4', 'src4'),
                      ('src1', 'lens0'), ('src2', 'lens0'), ('src3', 'lens0'), ('src4', 'lens0'),
                      ('src1', 'lens1'), ('src2', 'lens1'), ('src3', 'lens1'), ('src4', 'lens1'),
                      ('src2', 'lens2'), ('src3', 'lens2'), ('src4', 'lens2'),
                      ('src2', 'lens3'), ('src3', 'lens3'), ('src4', 'lens3'),
                      ('src2', 'lens4'), ('src3', 'lens4'), ('src4', 'lens4'),
                      ('src3', 'lens5'), ('src4', 'lens5'),
                      ('src3', 'lens6'), ('src4', 'lens6'),
                      ('src3', 'lens7'), ('src4', 'lens7'),
                      ('src4', 'lens8'),
                      ('src4', 'lens9'),
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


class _SpectroscopicRedshiftKernel:
    """Top-hat observed-redshift kernel for ``photo-z: sigma_0: 0``.

    Matches crow's SpectroscopicRedshift; TJPCov's Gaussian photo-z kernel
    would divide by zero there. Other values of sigma_0 keep TJPCov's kernel.
    """

    def observed_photo_z(self, z_true, z_i, sigma_0):
        if sigma_0 > 0:
            return super().observed_photo_z(z_true, z_i, sigma_0)
        return float(self.z_bins[z_i] <= z_true < self.z_bins[z_i + 1])


class TJPCovClusterSSC(_SpectroscopicRedshiftKernel, BaseClusterSSC):
    """TJPCov cluster-count SSC with the spectroscopic-redshift kernel."""


class TJPCovClusterGaussian(_SpectroscopicRedshiftKernel, BaseClusterGaussian):
    """TJPCov cluster counts (shot noise) with the spectroscopic-redshift kernel."""

    def shot_noise(self, z_i, lbd_i):
        if self.sigma_0 > 0:
            return super().shot_noise(z_i, lbd_i)
        # With the top-hat kernel, integrate over the bin only: same value, but
        # quad over TJPCov's wide range resolves the kernel steps ~100x slower.

        def integrand(z):
            return (self.mass_richness_integral(z, lbd_i, remove_bias=True)
                    * self.comoving_volume_element(z, z_i, self.sigma_0))
        return self.survey_area * self._quad_integrate(
            integrand, self.z_bins[z_i], self.z_bins[z_i + 1])


def get_cluster_ssc(S, recipe, cosmo, counts, halo_bias='Tinker10'):
    """Super-sample covariance of the cluster counts in *S*, from TJPCov.

    The mass function, mass range and Murata mass-richness relation (at its
    fiducial values) are read from the crow *recipe* of the firecrown
    statistic, so the covariance describes the model of the data vector.

    TJPCov integrates its own counts, without crow's completeness and purity
    models. Following CLPipe (LSSTDESC/CLPipe, ``clpipe/clp_covariance.py``,
    ``CLPCovariance.replace_crow_counts``, commit 4a6128d), the SSC is rescaled
    to crow's counts *N*: ``SSC_ij * N_i N_j / (G_i G_j)``, where *G* are
    TJPCov's own counts (its Gaussian term). Since SSC_ij is proportional to
    the bias-weighted counts of both bins, this carries the selection over to
    the SSC. It assumes the selection leaves the mean halo bias of each bin
    unchanged.

    Parameters
    ----------
    counts : array
        crow's fiducial counts, in the order of
        ``S.indices(data_type='cluster_counts')``.

    Returns
    -------
    np.ndarray
        (n_counts, n_counts) matrix in the order of
        ``S.indices(data_type='cluster_counts')``.
    """
    counts_type = sacc.standard_types.cluster_counts
    S_cl = S.copy()
    S_cl.covariance = None
    S_cl.keep_selection(data_type=counts_type)
    S_cl.remove_tracers([n for n, t in S_cl.tracers.items()
                         if t.tracer_type not in ('survey', 'bin_z', 'bin_richness')])

    z_max = max(t.upper for t in S_cl.tracers.values() if t.tracer_type == 'bin_z')
    if z_max > 0.91:
        warnings.warn(
            "TJPCov ClusterCountsSSC stops its true-redshift integral at z=0.91; "
            f"the SSC of cluster bins above it (z_max={z_max}) is underestimated."
        )

    mor = recipe.mass_distribution
    hmf = recipe.cluster_theory.halo_mass_function
    lo, hi = recipe.mass_interval
    mor_parameters = {
        'mass_func': hmf.name, 'mass_def': hmf.mass_def.name, 'halo_bias': halo_bias,
        'min_halo_mass': 10**lo, 'max_halo_mass': 10**hi,
        'm_pivot': mor.pivot_ln_mass / np.log(10), 'z_pivot': mor.pivot_redshift,
    }
    for i in range(3):
        mor_parameters[f'mu_p{i}'] = mor.parameters[f'mu{i}']
        mor_parameters[f'sigma_p{i}'] = mor.parameters[f'sigma{i}']

    def config():
        return {
            'tjpcov': {'sacc_file': S_cl, 'cosmo': cosmo, 'use_mpi': False},
            'parameters': {'h': cosmo['h']},
            'photo-z': {'sigma_0': 0.0},
            'mor_parameters': mor_parameters,
        }
    ssc = TJPCovClusterSSC(config()).get_covariance()
    tjpcov_counts = np.diag(TJPCovClusterGaussian(config()).get_covariance())
    ratio = np.asarray(counts) / tjpcov_counts
    return ssc * np.outer(ratio, ratio)
