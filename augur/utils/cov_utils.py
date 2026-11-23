import logging
import os
import shutil
import tempfile
import warnings

import numpy as np
import pyccl as ccl
from tjpcov.covariance_gaussian_fsky import FourierGaussianFsky
from tjpcov.covariance_fourier_gaussian_nmt import FourierGaussianNmt
from tjpcov.covariance_cluster_counts_gaussian import ClusterCountsGaussian as BaseClusterGaussian
from tjpcov.covariance_cluster_counts_ssc import ClusterCountsSSC as BaseClusterSSC
from augur.utils.config_io import parse_array
from augur.generate_utils.namaster import make_nmt_bin, tracer_probe

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
    if ndens.ndim == 0:
        # Split between the tomographic bins
        if ndens <= 0:
            raise ValueError(f"ndens for {keyw} must be positive.")
        nz = np.array(nz_all[keyw])
        norm = np.sum(nz, axis=1)/np.sum(nz)
        ndens = float(ndens * norm[tracer_bin])
    elif ndens.ndim == (nbins,):
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

    # tracer_obj = S.get_tracer(cluster_tracer_name)

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
    # cluster_tracers = []
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

    # Merge ell_edges from all harmonic-space config sections:
    # standard galaxy statistics and CMB lensing statistics.
    ell_edges_by_stat = {}
    for stat_name, stat_cfg in config.get('statistics', {}).items():
        ell_edges_by_stat[stat_name] = parse_array(stat_cfg['ell_edges'])
    cmb_stats = config.get('cmb_lensing', {}).get('statistics', {})
    for stat_name, stat_cfg in cmb_stats.items():
        ell_edges_by_stat[stat_name] = parse_array(stat_cfg['ell_edges'])

    # CMB convergence noise (reconstruction noise floor)
    cmb_noise_cl = load_noise_cl(config.get('cmb_lensing', {}).get('noise_cl', 0.0),
                                 'cmb_lensing.noise_cl')

    def _noise_between(tr_a_name, tr_b_name, ells):
        """Uncorrelated noise term N_ab (non-zero only for auto-tracer pairs)."""
        if tr_a_name != tr_b_name:
            return 0.0
        # CMB convergence reconstruction noise
        if tr_a_name == 'cmb_convergence':
            return cmb_noise_cl(ells)
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
            cls13 += _noise_between(tr1_name, tr3_name, ells_here)
            cls24 += _noise_between(tr2_name, tr4_name, ells_here)
            cls14 += _noise_between(tr1_name, tr4_name, ells_here)
            cls23 += _noise_between(tr2_name, tr3_name, ells_here)

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


def load_noise_cl(spec, where='noise_cl'):
    """
    Noise power spectrum from a config entry.

    Parameters
    ----------
    spec : float, str or dict
        A number (white noise); a text file whose first two columns are ell and
        N_ell; or ``{file: path, column: 1, ell_column: 0}`` to pick the columns,
        e.g. ``column: 7`` (minimum-variance N_L^kk) for the Simons Observatory
        lensing-noise files. ``.csv`` files are read as comma-separated.
    where : str
        Config location, for error messages.

    Returns
    -------
    callable
        ``noise(ell)``: the number, or the file's column interpolated linearly in
        ell and held at its end values outside the file's range.
    """
    try:
        value = float(spec)
    except (TypeError, ValueError):
        pass
    else:
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{where} must be a non-negative number, got {spec!r}.")
        return lambda ell: value

    if isinstance(spec, (str, os.PathLike)):
        spec = {'file': spec}
    if not isinstance(spec, dict) or 'file' not in spec \
            or set(spec) - {'file', 'column', 'ell_column'}:
        raise ValueError(
            f"{where} must be a number, a file path or {{file, column, ell_column}}; "
            f"got {spec!r}."
        )
    path = os.fspath(spec['file'])
    table = np.loadtxt(path, delimiter=',' if path.endswith('.csv') else None, ndmin=2)
    ell_in = table[:, int(spec.get('ell_column', 0))]
    nl_in = table[:, int(spec.get('column', 1))]
    if np.any(np.diff(ell_in) <= 0) or not np.all(np.isfinite(nl_in)) or np.any(nl_in < 0):
        raise ValueError(f"{where}: {path} needs increasing ell and finite, non-negative noise.")

    def noise(ell):
        if np.max(ell) > ell_in[-1]:
            warnings.warn(f"{where}: {path} stops at ell = {ell_in[-1]:g}; holding its last "
                          f"value up to ell = {np.max(ell):g}.")
        return np.interp(ell, ell_in, nl_in)
    return noise


def _set_cmb_noise(calc, ells):
    """
    Give a TJPCov calculator Augur's CMB-lensing noise, ``tjpcov.cmb_noise``.

    TJPCov has no CMB-lensing noise and caches its tracer info on first use, so the
    cached noise dict is filled in place, on the ells the calculator uses. Returns
    the noise and coupled-noise dicts.
    """
    with warnings.catch_warnings():
        # Augur never stores n_ell_coupled in the SACC; TJPCovNmt sets it instead.
        warnings.filterwarnings('ignore', message='Missing n_ell_coupled')
        _, noise, noise_coupled = calc.get_tracer_info(return_noise_coupled=True)
    cmb_noise = load_noise_cl(calc.config['tjpcov'].get('cmb_noise', 0.0), 'cmb_lensing.noise_cl')
    sacc_file = calc.io.get_sacc_file()
    for tracer in sacc_file.tracers:
        if getattr(sacc_file.get_tracer(tracer), 'quantity', None) == 'cmb_convergence':
            noise[tracer] = cmb_noise(ells)
    return noise, noise_coupled


class TJPCovGaus(FourierGaussianFsky):
    """
    TJPCov's ``FourierGaussianFsky`` with Augur's ell binning and CMB-lensing noise.
    """

    def __init__(self, config):
        super().__init__(config)
        _set_cmb_noise(self, self.get_binning_info()[0])

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


def _has_field(tracer_name):
    try:
        tracer_probe(tracer_name)
    except ValueError:
        return False
    return True


class TJPCovNmt(FourierGaussianNmt):
    """
    TJPCov's ``FourierGaussianNmt`` using a :class:`NaMasterWorkspaces` object.

    Fields, workspaces and covariance workspaces come from ``namaster``, where they
    are computed once and shared with the bandpower windows, instead of being
    rebuilt or re-read from disk for every block.

    Noise enters as coupled noise (the noise pseudo-C_ell on the tracer's mask):
    ``tjpcov.noise_coupled[tracer]`` (anything :func:`load_noise_cl` reads) where
    given, else the white-noise value ``N * mean(mask**2)``, which gives exactly
    TJPCov's uncoupled-noise covariance.

    TJPCov saves each block to its output directory and reloads it by tracer name
    on later runs, so unless an ``outdir`` is given this class uses a temporary
    directory removed by :meth:`cleanup` (or on leaving a ``with`` block).

    Parameters
    ----------
    config : dict
        TJPCov config ``{'tjpcov': {...}}``, as built for :class:`TJPCovGaus`.
    namaster : NaMasterWorkspaces
        Shared NaMaster products. Every mask used must have a file.
    ell_edges : array_like
        Band edges shared by every harmonic statistic in the SACC.
    """

    def __init__(self, config, namaster, ell_edges):
        tj = dict(config['tjpcov'])
        footprint = namaster.footprint
        names = {t: footprint.tracer_mask_name(t)
                 for t in tj['sacc_file'].tracers if _has_field(t)}
        tj['mask_names'] = names
        tj['mask_file'] = {t: footprint.mask(n) for t, n in names.items()}
        tj['binning_info'] = make_nmt_bin(ell_edges, namaster.lmax)
        given = dict(tj.get('noise_coupled') or {})
        unknown = sorted(set(given) - set(names))
        if unknown:
            raise ValueError(f"cov_options.noise_coupled names {unknown}, which are not "
                             f"NaMaster tracers of the SACC ({sorted(names)}).")
        self._tmpdir = None
        if tj.get('outdir') is None:
            self._tmpdir = tj['outdir'] = tempfile.mkdtemp(prefix='augur_tjpcov_')
        try:
            super().__init__(dict(config, tjpcov=tj))
            self.namaster = namaster
            self.ell_edges = np.asarray(ell_edges, dtype=float)
            self._concise_sacc = None
            self._check_all_bands()

            ells = np.arange(namaster.lmax + 1)
            noise, coupled = _set_cmb_noise(self, ells)
            for tracer, mask_name in names.items():
                if tracer in given:
                    where = f"cov_options.noise_coupled.{tracer}"
                    coupled[tracer] = load_noise_cl(given[tracer], where)(ells)
                else:
                    coupled[tracer] = noise[tracer] * footprint.mean_overlap(mask_name,
                                                                             mask_name)
        except BaseException:
            self.cleanup()
            raise

    def _check_all_bands(self):
        """TJPCov reshapes every block with the band count of the first pair."""
        sacc_file = self.io.get_sacc_file()
        nbands = self.ell_edges.size - 1
        for dtype in sacc_file.get_data_types():
            for comb in sacc_file.get_tracer_combinations(data_type=dtype):
                if not all(t in self.mask_names for t in comb):
                    continue
                n = len(sacc_file.indices(data_type=dtype, tracers=comb))
                if n != nbands:
                    raise ValueError(
                        f"{dtype} {comb} has {n} bandpowers in the SACC but the NaMaster "
                        f"binning has {nbands}. The NaMaster covariance needs every band "
                        "stored for every pair (general.ignore_scale_cuts: True); scale "
                        "cuts are then applied by the likelihood filters."
                    )

    def get_fields_dict(self, tracer_names, cache=None, masks=None, **kwargs):
        names = self.get_mask_names_dict(tracer_names)
        spins = self.get_tracers_spin_dict(tracer_names)
        return {i: self.namaster.field(names[i], spins[i]) for i in range(1, 5)}

    def get_workspaces_dict(self, tracer_names, bins, cache=None, fields=None, masks=None,
                            **kwargs):
        names = self.get_mask_names_dict(tracer_names)
        overlap = self.namaster.footprint.mean_overlap
        workspaces = {}
        for i in (12, 34, 13, 23, 14, 24):
            a, b = divmod(i, 10)
            if i not in (12, 34) and overlap(names[a], names[b]) == 0:
                workspaces[i] = None  # TJPCov's convention for non-overlapping masks
            else:
                workspaces[i] = self.namaster.workspace(tracer_names[a], tracer_names[b],
                                                        self.ell_edges)
        return workspaces

    def get_covariance_workspace(self, f1, f2, f3, f4, m1, m2, m3, m4, **kwargs):
        return self.namaster.covariance_workspace_from_masks(m1, m2, m3, m4)

    def get_sacc_with_concise_dtypes(self):
        # TJPCov copies the whole SACC for every block; one copy is enough.
        if self._concise_sacc is None:
            self._concise_sacc = super().get_sacc_with_concise_dtypes()
        return self._concise_sacc

    def precompute(self, pairs):
        """
        Build every workspace and covariance workspace needed by ``pairs``.

        Call this before forking workers, which would otherwise each build (and
        discard) their own copies.
        """
        self.get_sacc_with_concise_dtypes()
        for comb1, comb2 in pairs:
            tracer_names = dict(zip((1, 2, 3, 4), (*comb1, *comb2)))
            if all(t in self.mask_names for t in tracer_names.values()):
                self.get_workspaces_dict(tracer_names, self.binning_info)
                names = self.get_mask_names_dict(tracer_names)
                self.namaster.covariance_workspace_from_masks(*(names[i] for i in range(1, 5)))

    def cleanup(self):
        """Remove the temporary output directory, if this object made one."""
        if self._tmpdir is not None:
            shutil.rmtree(self._tmpdir, ignore_errors=True)
            self._tmpdir = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cleanup()


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
