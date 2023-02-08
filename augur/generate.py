"""Data Generation module

This code generates synthetic dataset by cobbling
together a suitable configuration file for firecrown
and then convincing it to generate data.

"""

import numpy as np
import pyccl as ccl
import sacc
from augur.tracers.two_point import ZDist, LensSRD2018, SourceSRD2018
from augur.utils.cov_utils import get_gaus_cov
import firecrown.likelihood.gauss_family.statistic.source.weak_lensing as wl
import firecrown.likelihood.gauss_family.statistic.source.number_counts as nc
from firecrown.likelihood.gauss_family.statistic.two_point import TwoPoint
from firecrown.likelihood.gauss_family.gaussian import ConstGaussian

implemented_nzs = [ZDist, LensSRD2018, SourceSRD2018]


def _get_tracers(statistic, comb):
    """
    Auxiliary function to get the tracers in a given statistic
    in the configuration file.
    """
    if 'galaxy_density_cl' in statistic:
        tr1 = f'lens{comb[0]}'
        tr2 = f'lens{comb[1]}'
    elif 'galaxy_shear_cl_ee' in statistic:
        tr1 = f'src{comb[0]}'
        tr2 = f'src{comb[1]}'
    elif 'galaxy_shearDensity_cl_e' in statistic:
        tr1 = f'lens{comb[0]}'
        tr2 = f'src{comb[1]}'
    else:
        return NotImplementedError('Only C_ls available')
    return tr1, tr2


def _parse_config(config):
    """
    Parse configuration file
    """
    if isinstance(config, str):
        import yaml
        with open(config) as f:
            config = yaml.safe_load(f)
    elif isinstance(config, dict):
        pass
    else:
        raise ValueError('config must be a dictionary or path to a config file')
    return config


def generate_sacc_and_stats(config):
    """
    Routine to generate a placeholder SACC file containing the data-vector
    combinations and tracers specified in the configuration. It also generates
    the required TwoPoint statistics to feed the firecrown likelihood.

    Note: Systematics implemented are PhotoZShift, MultiplicativeShearBias,
    and LinearAlignmentSystematic.

    Parameters:
    -----------

    config : dict or path
             Dictionary containing the analysis configuration or path to configuration file.

    Returns:
    --------
    S : sacc.Sacc
         Placeholder sacc file
    cosmo : ccl.Cosmology
         Fiducial cosmology
    stats : firecrown.likelihood.gauss_family.statistic.two_point.TwoPoint
         List of TwoPoint statistics that will enter the likelihood
    """

    config = _parse_config(config)
    cosmo_cfg = config['cosmo']
    # Set up ccl.Cosmology object
    cosmo = ccl.Cosmology(Omega_b=cosmo_cfg['Omega_b'],
                          Omega_c=cosmo_cfg['Omega_c'],
                          n_s=cosmo_cfg['n_s'], sigma8=cosmo_cfg['sigma8'],
                          h=cosmo_cfg['h'])

    # First we generate the placeholder SACC file with the correct N(z) and ell-binning
    # TODO add real-space

    S = sacc.Sacc()

    # Read sources from config file
    src_cfg = config['sources']
    sources = {}
    dndz = {}
    z = np.linspace(0, 4, 400)  # z to probe the dndz distribution

    # Set up intrinsic alignment systematics
    if 'ia_class' in src_cfg.keys():
        ia_sys = eval(src_cfg['ia_class'])(sacc_tracer="")
    else:
        ia_sys = None
    # TODO -- fix this in subsequent releases
    if isinstance(ia_sys, wl.LinearAlignmentSystematic):
        ia_sys = None  # the LinearAlignmentSystematic class seems to be a bit buggy
    if 'sources' in config.keys():
        nbins = src_cfg['nbins']  # Number of bins for shear sources
        src_root = 'src'  # Root of sacc tracer name

        # Set up multiplicative bias (if present in config)
        if 'mult_bias' in src_cfg.keys():
            if np.isscalar(src_cfg['mult_bias']):
                mult_bias = np.ones(nbins)*src_cfg['mult_bias']
            else:
                mult_bias = src_cfg['mult_bias']
                if len(mult_bias) != nbins:
                    raise ValueError('Expected scalar or list of length=nbins')
        # Set up photo-z shift (if present in config)
        if 'delta_z' in src_cfg.keys():
            if np.isscalar(src_cfg['delta_z']):
                delta_z = np.ones(nbins)*src_cfg['delta_z']
            else:
                delta_z = src_cfg['delta_z']

        # Loop over bins
        for i in range(nbins):
            sacc_tracer = f'{src_root}{i}'
            if isinstance(src_cfg['Nz_type'], list):
                if eval(src_cfg['Nz_type'][i]) in implemented_nzs:
                    dndz[sacc_tracer] = eval(src_cfg['Nz_type'][i])(z, Nz_nbins=nbins, Nz_ibin=i,
                                                                    **src_cfg['Nz_kwargs'])
                else:
                    raise NotImplementedError('The selected N(z) is yet not implemented')
            else:
                if eval(src_cfg['Nz_type']) in implemented_nzs:
                    dndz[sacc_tracer] = eval(src_cfg['Nz_type'])(z, Nz_nbins=nbins, Nz_ibin=i,
                                                                 **src_cfg['Nz_kwargs'])
                else:
                    raise NotImplementedError('The selected N(z) is yet not implemented')
            S.add_tracer('NZ', sacc_tracer, dndz[sacc_tracer].z, dndz[sacc_tracer].Nz)
            # Set up the WeakLensing objects for firecrown
            mbias = wl.MultiplicativeShearBias(sacc_tracer=sacc_tracer)
            mbias.mult_bias = mult_bias[i]
            pzshift = wl.PhotoZShift(sacc_tracer=sacc_tracer)
            pzshift.delta_z = delta_z[i]
            sys_list = [mbias, pzshift]
            if ia_sys is not None:
                sys_list.append(ia_sys)
            sources[sacc_tracer] = wl.WeakLensing(sacc_tracer=sacc_tracer,
                                                  systematics=sys_list)

    # Read lenses from config file
    if 'lenses' in config.keys():
        lns_cfg = config['lenses']
        nbins = lns_cfg['nbins']
        lns_root = 'lens'
        Nz_centers = eval(lns_cfg['Nz_kwargs']['Nz_center'])
        lns_cfg['Nz_kwargs'].pop('Nz_center')

        if np.isscalar(Nz_centers):
            Nz_centers = [Nz_centers]
            if nbins != 1:
                raise ValueError('Nz_centers should have the same length as the number of bins')
        else:
            if len(Nz_centers) != nbins:
                raise ValueError('Nz_centers should have the same length as the number of bins')
        # Checking if there's photo-z shift systematics in the config
        if 'delta_z' in lns_cfg.keys():
            if np.isscalar(lns_cfg['delta_z']):
                delta_z = np.ones(nbins)*lns_cfg['delta_z']
            else:
                delta_z = lns_cfg['delta_z']

        for i in range(nbins):
            sacc_tracer = f'{lns_root}{i}'
            if isinstance(lns_cfg['Nz_type'], list):
                if eval(lns_cfg['Nz_type'][i]) in implemented_nzs:
                    dndz[sacc_tracer] = eval(lns_cfg['Nz_type'][i])(z, Nz_center=Nz_centers[i],
                                                                    Nz_nbins=nbins,
                                                                    **lns_cfg['Nz_kwargs'])
                else:
                    raise NotImplementedError('The selected N(z) is yet not implemented')
            else:
                if eval(lns_cfg['Nz_type']) in implemented_nzs:
                    dndz[sacc_tracer] = eval(lns_cfg['Nz_type'])(z, Nz_center=Nz_centers[i],
                                                                 Nz_nbins=nbins,
                                                                 **lns_cfg['Nz_kwargs'])
                else:
                    raise NotImplementedError('The selected N(z) is yet not implemented')
            S.add_tracer('NZ', sacc_tracer, dndz[sacc_tracer].z, dndz[sacc_tracer].Nz)
            # Set up the NumberCounts objects for firecrown
            # Start by retrieving the bias
            if 'inverse_growth' in lns_cfg['bias_type']:  # b(z) = b0/D+(z)
                bias = lns_cfg['bias_kwargs']['b0'] / \
                       ccl.growth_factor(cosmo, 1/(1+dndz[sacc_tracer].zav))
            elif 'custom' in lns_cfg['bias_type']:  # b(z) = list of values
                bias = lns_cfg['bias_kwargs']['b']
                if len(bias) != nbins:
                    raise ValueError('bias_type==custom requires a bias value per bin')
                bias = bias[i]
            else:
                raise NotImplementedError('bias_type implemented are custom or inverse_growth')
            pzshift = nc.PhotoZShift(sacc_tracer=sacc_tracer)
            pzshift.delta_z = delta_z[i]
            sources[sacc_tracer] = nc.NumberCounts(sacc_tracer=sacc_tracer, systematics=[pzshift],
                                                   derived_scale=True)
            sources[sacc_tracer].bias = bias

    # Read data vector combinations
    if 'statistics' not in config.keys():
        raise ValueError('statistics key is required in config file')
    stat_cfg = config['statistics']
    stats = []
    for key in stat_cfg.keys():
        tracer_combs = stat_cfg[key]['tracer_combs']
        kmax = stat_cfg[key]['kmax']
        ell_edges = eval(stat_cfg[key]['ell_edges'])
        ells = 0.5*(ell_edges[:-1] + ell_edges[1:])
        for comb in tracer_combs:
            tr1, tr2 = _get_tracers(key, comb)
            if (kmax is not None) and (kmax != 'None'):
                zmean1 = dndz[tr1].zav
                zmean2 = dndz[tr2].zav
                a12 = np.array([1./(1+zmean1), 1./(1+zmean2)])
                ell_max = np.min(kmax * ccl.comoving_radial_distance(cosmo, a12))
                ells_here = ells[ells < ell_max]
            for ell in ells_here:
                S.add_data_point(key, (tr1, tr2), 0.0, ell=ell, error=1e30)
            # Now create TwoPoint objects for firecrown
            _aux_stat = TwoPoint(source0=sources[tr1], source1=sources[tr2],
                                 sacc_data_type=key)
            stats.append(_aux_stat)
    S.add_covariance(np.ones_like(S.data))
    return S, cosmo, stats


def generate(config, return_outputs=False, write_sacc=True):
    """
    Generate likelihood object and sacc file with fiducial cosmology

    Parameters:
    -----------

    config : dict or path
        Dictionary containing the analysis configuration or path to configuration file.
    return_outputs : bool
        If `True` it returns the likelihood object (so it can be used later) and the modified
        Sacc object.
    write_sacc : bool
        If `True` it writes a sacc file with fiducial data vector.

    Returns:
    --------
    S : sacc.Sacc
         Placeholder sacc file
    cosmo : ccl.cosmology
    """
    config = _parse_config(config)
    # Generate placeholders
    S, cosmo, stats = generate_sacc_and_stats(config)
    # Generate likelihood object
    lk = ConstGaussian(statistics=stats)
    # Pass the correct binning/tracers
    lk.read(S)
    # Run the likelihood (to get the theory)
    lk.compute_loglike(cosmo)
    # Empty the placeholder Sacc's covariance and data vector so we can "overwrite"
    S.covariance = None
    S.data = []
    # Fill out the data-vector with the theory predictions for the fiducial
    # cosmology/parameters
    for st in lk.statistics:
        S.add_ell_cl(st.sacc_data_type, st.sacc_tracers[0], st.sacc_tracers[1],
                     st.ell_or_theta_, st.predicted_statistic_)
    if config['cov_options']['cov_type'] == 'gaus_internal':
        fsky = config['cov_options']['fsky']
        cov = get_gaus_cov(S, lk, cosmo, fsky, config)
        S.add_covariance(cov)
    else:
        raise Warning('''Currently only internal Gaussian covariance has been implemented,
                         cov_type is not understood. Using identity matrix as covariance.''')
    if write_sacc:
        print(config['fiducial_sacc_path'])
        S.save_fits(config['fiducial_sacc_path'], overwrite=True)
    # Update covariance and inverse -- TODO need to update cholesky!!
    # lk.read(S)  # This would update everything but Cholesky is failing
    lk.cov = S.covariance.covmat
    lk.inv_cov = np.linalg.inv(lk.cov)
    if return_outputs:
        return lk, S
