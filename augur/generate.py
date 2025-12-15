"""Data Generation module

This code generates synthetic dataset by cobbling
together a suitable configuration file for firecrown
and then convincing it to generate data.

"""

import numpy as np
import pyccl as ccl
import sacc
from augur.tracers.two_point import ZDist, LensSRD2018, SourceSRD2018
from augur.tracers.two_point import ZDistFromFile
from augur.utils.cov_utils import get_gaus_cov, get_SRD_cov, get_noise_power
from augur.utils.cov_utils import TJPCovGaus
from augur.utils.theory_utils import compute_new_theory_vector
from packaging.version import Version
import firecrown

import firecrown.likelihood.weak_lensing as wl
import firecrown.likelihood.number_counts as nc
from firecrown.likelihood.two_point import TwoPoint
from firecrown.likelihood.gaussian import ConstGaussian
from firecrown.modeling_tools import ModelingTools
from firecrown.parameters import ParamsMap
from augur.utils.config_io import parse_config
from augur.utils.firecrown_interface import create_modeling_tools
from firecrown.ccl_factory import (
    CCLFactory,
    CCLCreationMode,
    CCLPureModeTransferFunction,
    CAMBExtraParams,
    PoweSpecAmplitudeParameter, 
)


implemented_nzs = [ZDist, LensSRD2018, SourceSRD2018, ZDistFromFile]

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
'''
def _get_correlation_space(name: str):
    key = (name or "harmonic").lower()
    cs = CORRELATION_SPACE_MAP.get(key)
    if cs is None:
        raise ValueError(f"Unsupported correlation_space '{name}'")
    return cs
'''

#TODO: not correct, check this later
def add_nz(config, tracer_name, S):
    """
    Auxiliary function to get the n(z) distribution for a given tracer
    in the configuration file.
    """
    src_cfg = config[tracer_name]
    nbins = src_cfg['nbins']
    if 'src' in tracer_name:
        Nz_ibin = int(tracer_name.replace('src', ''))
    elif 'lens' in tracer_name:
        Nz_ibin = int(tracer_name.replace('lens', ''))
    else:
        raise ValueError('Tracer name not recognized')
    if eval(src_cfg['Nz_type']) in implemented_nzs:
        if 'ZDistFromFile' in src_cfg['Nz_type']:
            dndz = ZDistFromFile(**src_cfg['Nz_kwargs'], ibin=Nz_ibin)
        else:
            z = np.linspace(0.0, 4.0, 1000)
            dndz = eval(src_cfg['Nz_type'])(z, Nz_nbins=nbins,
                                           Nz_ibin=Nz_ibin,
                                           **src_cfg['Nz_kwargs'])
        S.add_tracer('NZ', sacc_tracer, dndz[sacc_tracer].z, dndz[sacc_tracer].Nz)

    else:
        raise NotImplementedError('The selected N(z) is yet not implemented')
    return dndz


'''
def create_dummy_sacc(config):
    S = sacc.Sacc()
    S = add_nz(config, tracer_name, S)

    #TODO: add coavariance!

    # this will add n(z)'s, the specific tracer combos, etc.
    # and get it ready to be used by the firecrown factory

    return S


def create_firecrown_factory(config, S, tools):
    # read in factories from config and follow general_y1_likelihood
    # structure of setting up TwoPointFactory, TwoPointExperiment, etc.
    # this will create a DUMMY likelihood that will compute the initial datavector

    factory_cfg = config.get("Firecrown_Factory", None)
    lk = None
    for key in factory_cfg.keys():
        # parse each factory type
        if key=='TwoPointFactory':
            tpf_cfg = factory_cfg[key]
            nc_factory = []
            wl_factory = []
            if 'WeakLensingFactory' in tpf_cfg.keys():
                wl_factory_cfg = tpf_cfg['WeakLensingFactory']
                wl_factory = [base_model_from_yaml(WeakLensingFactory, wl_factory_cfg)]
            elif 'NumberCountsFactory' in tpf_cfg.keys():
                nc_factory_cfg = tpf_cfg['NumberCountsFactory']
                nc_factory = [base_model_from_yaml(NumberCountsFactory, nc_factory_cfg)]
            else:
                raise NotImplementedError('Only WeakLensingFactory and NumberCountsFactory are implemented so far.')

            tp_factory = TwoPointFactory(eval(tpf_cfg['correlation_space']),
                                         weak_lensing_factories=wl_factory,
                                         number_counts_factories=nc_factory,
                                         )
            
            two_point_experiment = TwoPointExperiment(
                two_point_factory=tp_factory,
                ccl_factory=tools.ccl_factory,
                data_source=DataSourceSacc(
                sacc_data_file=S,
                ),
            )
            
            lk = two_point_experiment.make_likelihood()
        elif key=='Fiducial':
            pass
        else:
            raise NotImplementedError(f'Factory type {key} not implemented yet.')

    if lk is not None:
        return lk
    else:
        raise ValueError('No valid Firecrown factory found in configuration.')
    return lk

def overwrite_sacc(dummy_S, dummy_lk, cosmo, tools):
    # NOW compute the theory vector, copy sacc and overwrite data

    return None


def generate_sacc_and_stats_refactored(config):

    config = parse_config(config)
    tools, cosmo = create_modeling_tools(config)
    dummy_S = create_dummy_sacc(config)

    dummy_lk = create_firecrown_factory(config, dummy_S, tools)

    S = overwrite_sacc(dummy_S, dummy_lk, cosmo, tools)

    lk = create_firecrown_factory(config, S, tools)

    return S, cosmo, lk, tools

    # read in factories from config and follow general_y1_likelihood
    # structure of setting up TwoPointFactory, TwoPointExperiment, etc.
    # this will create a DUMMY likelihood that will compute the initial datavector

    # NOW ccompute the theory vector, copy sacc and overwrite data

    #return final sacc, cosmo, lk, ccl_factory, tools, etc.
'''


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

    config = parse_config(config)

    # Initialize cosmology
    cosmo_cfg = config['cosmo']

    if cosmo_cfg.get("transfer_function") is None:
        cosmo_cfg['transfer_function'] = 'boltzmann_camb'

    if cosmo_cfg.get('extra_parameters') is None:
        cosmo_cfg['extra_parameters'] = dict()

    if 'ccl_accuracy' in config.keys():
        # Pass along spline control parameters
        if 'spline_params' in config['ccl_accuracy'].keys():
            for key in config['ccl_accuracy']['spline_params'].keys():
                try:
                    type_here = type(ccl.spline_params[key])
                    value = config['ccl_accuracy']['spline_params'][key]
                    ccl.spline_params[key] = type_here(value)
                except KeyError:
                    print(f'The selected spline keyword `{key}` is not recognized.')
                except ValueError:
                    print(f'The selected value `{value}` could not be casted to `{type_here}`.')
        # Pass along GSL control parameters
        if 'gsl_params' in config['ccl_accuracy'].keys():
            for key in config['ccl_accuracy']['gsl_params'].keys():
                try:
                    type_here = type(ccl.gsl_params[key])
                    value = config['ccl_accuracy']['gsl_params'][key]
                    ccl.gsl_params[key] = type_here(value)
                except KeyError:
                    print(f'The selected GSL keyword `{key}` is not recognized.')
                except ValueError:
                    print(f'The selected value `{value}` could not be casted to `{type_here}`.')

    try:
        cosmo = ccl.Cosmology(**cosmo_cfg)
    except (KeyError, TypeError, ValueError) as e:
        print('Error in cosmology configuration. Check the config file.')
        # Reraise the exception to see the full traceback
        raise e

    # First we generate the placeholder SACC file with the correct N(z) and ell-binning
    # TODO add real-space

    S = sacc.Sacc()

    # I think there is no reason to set systematic parameters directly through the sacc framework here
    # Instead, I believe we can make a "template" sacc file with the relevant n(z) distributions, 
    # compute a theory datavector with the firecrown likelihood machinery, 
    # and utilize that in the final sacc returned to the user. 


    # Read sources from config file
    src_cfg = config['sources']
    sources = {}
    dndz = {}
    # These are to match the N(z)s from the fits file in the firecrown repo
    z = np.linspace(0.004004004004004004,
                    4.004004004004004004, 1000)  # z to probe the dndz distribution
    sys_params = {}
    # Set up intrinsic alignment systematics
    if 'ia_class' in src_cfg.keys():
        ia_sys = eval(src_cfg['ia_class'])(sacc_tracer="")
    else:
        ia_sys = None
    # TODO -- fix this in subsequent releases
    # if isinstance(ia_sys, wl.LinearAlignmentSystematic):
    #    ia_sys = None  # the LinearAlignmentSystematic class seems to be a bit buggy
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
                    if 'ZDistFromFile' not in src_cfg['Nz_type'][i]:
                        dndz[sacc_tracer] = eval(src_cfg['Nz_type'][i])(z, Nz_nbins=nbins,
                                                                        Nz_ibin=i,
                                                                        **src_cfg['Nz_kwargs'])
                    else:
                        dndz[sacc_tracer] = ZDistFromFile(**src_cfg['Nz_kwargs'], ibin=i)
                else:
                    raise NotImplementedError('The selected N(z) is yet not implemented')
            else:
                if eval(src_cfg['Nz_type']) in implemented_nzs:
                    if 'ZDistFromFile' not in src_cfg['Nz_type']:
                        dndz[sacc_tracer] = eval(src_cfg['Nz_type'])(z, Nz_nbins=nbins,
                                                                     Nz_ibin=i,
                                                                     **src_cfg['Nz_kwargs'])
                    else:
                        dndz[sacc_tracer] = ZDistFromFile(**src_cfg['Nz_kwargs'], ibin=i)
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
                sys_params['ia_bias'] = src_cfg['ia_bias']
                sys_params['alphaz'] = src_cfg['alphaz']
                sys_params['z_piv'] = src_cfg['z_piv']
            sources[sacc_tracer] = wl.WeakLensing(sacc_tracer=sacc_tracer,
                                                  systematics=sys_list)
            sys_params[f'{sacc_tracer}_delta_z'] = delta_z[i]
            sys_params[f'{sacc_tracer}_mult_bias'] = mult_bias[i]

    # Read lenses from config file
    if 'lenses' in config.keys():
        lns_cfg = config['lenses']
        nbins = lns_cfg['nbins']
        lns_root = 'lens'
        if 'Nz_center' in lns_cfg['Nz_kwargs'].keys():
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
                    if 'ZDistFromFile' not in lns_cfg['Nz_type'][i]:
                        dndz[sacc_tracer] = eval(lns_cfg['Nz_type'][i])(z,
                                                                        Nz_center=Nz_centers[i],
                                                                        Nz_nbins=nbins,
                                                                        **lns_cfg['Nz_kwargs'])
                    else:
                        dndz[sacc_tracer] = ZDistFromFile(**lns_cfg['Nz_kwargs'], ibin=i)
                else:
                    raise NotImplementedError('The selected N(z) is yet not implemented')
            else:
                if eval(lns_cfg['Nz_type']) in implemented_nzs:
                    if 'ZDistFromFile' not in lns_cfg['Nz_type']:
                        dndz[sacc_tracer] = eval(lns_cfg['Nz_type'])(z,
                                                                     Nz_center=Nz_centers[i],
                                                                     Nz_nbins=nbins,
                                                                     **lns_cfg['Nz_kwargs'])
                    else:
                        dndz[sacc_tracer] = ZDistFromFile(**lns_cfg['Nz_kwargs'], ibin=i)
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
            sys_params[f'{sacc_tracer}_bias'] = bias
            sys_params[f'{sacc_tracer}_delta_z'] = delta_z[i]

    # Read data vector combinations
    if 'statistics' not in config.keys():
        raise ValueError('statistics key is required in config file')
    stat_cfg = config['statistics']
    stats = []
    ignore_sc = config['general'].get('ignore_scale_cuts', False)
    Bandpower = config['general'].get('bandpower_windows', 'None')
    for key in stat_cfg.keys():
        tracer_combs = stat_cfg[key]['tracer_combs']
        kmax = stat_cfg[key]['kmax']
        ell_edges = eval(stat_cfg[key]['ell_edges'])
        ells = np.sqrt(ell_edges[:-1]*ell_edges[1:])  # Geometric average
        print(ell_edges)
        for comb in tracer_combs:
            tr1, tr2 = _get_tracers(key, comb)
            if (kmax is not None) and (kmax != 'None') and (not ignore_sc):
                zmean1 = dndz[tr1].zav
                zmean2 = dndz[tr2].zav
                a12 = np.array([1./(1+zmean1), 1./(1+zmean2)])
                ell_max = np.min(kmax * ccl.comoving_radial_distance(cosmo, a12))
                ells_here = ells[ells < ell_max]
                print('ell_max', ell_max)
            else:
                ells_here = ells
            # add bandpower windows
            if Bandpower == 'TopHat':
                ells_aux = np.arange(0, np.max(ells_here)+1)
                wgt = np.zeros((len(ells_aux), len(ells_here)))
                for i in range(len(ells_here)):
                    in_win = (ells_aux > ell_edges[i]) & (ells_aux < ell_edges[i+1])
                    wgt[in_win, i] = 1.0
                win = sacc.BandpowerWindow(ells_aux, wgt)
                print(win.nv, win.nell, len(ells_here))
                S.add_ell_cl(key, tr1, tr2,
                             ells_here, np.zeros(len(ells_here)), window=win)

            else:
                S.add_ell_cl(key, tr1, tr2,
                             ells_here, np.zeros(len(ells_here)))                

            # Now create TwoPoint objects for firecrown
            _aux_stat = TwoPoint(source0=sources[tr1], source1=sources[tr2],
                                 sacc_data_type=key)
            stats.append(_aux_stat)
    S.add_covariance(np.ones_like(S.data))
    sys_params = ParamsMap(sys_params)
    return S, cosmo, stats, sys_params


def generate(config, return_all_outputs=False, write_sacc=True):
    """
    Generate likelihood object and sacc file with fiducial cosmology

    Parameters:
    -----------

    config : dict or path
        Dictionary containing the analysis configuration or path to configuration file.
    return_all_outputs : bool
        If `True` it returns the likelihood object (so it can be used later) and the modified
        Sacc object, as well as the modeling tools object used. If `False` it only returns the
        likelihood object.
    write_sacc : bool
        If `True` it writes a sacc file with fiducial data vector.

    Returns:
    -------
    lk : firecrown.likelihood.ConstGaussian
        Likelihood object, only returned if `return_all_outputs` is True.
    S : sacc.Sacc
        Sacc object with fake data vector and covariance. It is only returned if
        `return_all_outputs` is True.
    tools : firecrown.modeling.ModelingTools
        Modeling tools, only returned if `return_all_outputs` is True.
    sys_params : dict
        Dictionary containing the modeling systematic parameters. It is only returned if
        `return_all_outputs` is True.

    """

    config = parse_config(config)
    # Generate placeholders
    S, cosmo, stats, sys_params = generate_sacc_and_stats(config)

    # config needs to specify likelihood yaml.
    # alternatively, can pass likelihood and tools objects at input paramters.
    # choose objects to take precedence. 
    # Generate likelihood object
    lk = ConstGaussian(statistics=stats)
    # Pass the correct binning/tracers
    lk.read(S)
    tools, cosmo = create_modeling_tools(config)

    cosmo.compute_nonlin_power()
    _pars = cosmo.__dict__['_params_init_kwargs']
    # Populate ModelingTools and likelihood
    # tools = firecrown.modeling_tools.ModelingTools()
    _, lk, tools = compute_new_theory_vector(lk, tools, sys_params, _pars, return_all=True)

    # Get all bandpower windows before erasing the placeholder sacc
    win_dict = {}
    ell_dict = {}
    for st in lk.statistics:
        st = st.statistic
        tr1 = st.source0.sacc_tracer
        tr2 = st.source1.sacc_tracer
        dtype = st.sacc_data_type
        idx = S.indices(tracers=(tr1, tr2))
        win_dict[(tr1, tr2)] = S.get_bandpower_windows(idx)
        ell_dict[(tr1, tr2)], _ = S.get_ell_cl(dtype, tr1, tr2)
    # Empty the placeholder Sacc's covariance and data vector so we can "overwrite"
    S.covariance = None
    S.data = []
    # Fill out the data-vector with the theory predictions for the fiducial
    # cosmology/parameters
    for st in lk.statistics:
        # Hack to be able to reuse the statistics
        st = st.statistic
        tr1 = st.source0.sacc_tracer
        tr2 = st.source1.sacc_tracer
        st.ready = False
        S.add_ell_cl(st.sacc_data_type, tr1, tr2,
                     ell_dict[(tr1, tr2)], st.get_theory_vector(),  # Only valid for harmonic space
                     window=win_dict[(tr1, tr2)])
    if config['cov_options']['cov_type'] == 'gaus_internal':
        fsky = config['cov_options']['fsky']
        cov = get_gaus_cov(S, lk, cosmo, fsky, config)
        S.add_covariance(cov)
    elif config['cov_options']['cov_type'] == 'SRD':
        cov = get_SRD_cov(config['cov_options'], S)
        S.add_covariance(cov)
    # The option using TJPCov takes a while. TODO: Use some sort of parallelization.
    elif config['cov_options']['cov_type'] == 'tjpcov':
        tjpcov_config = dict()  # Create a config dictionary to instantiate TJPCov
        tjpcov_config['tjpcov'] = dict()
        tjpcov_config['tjpcov']['cosmo'] = tools.ccl_cosmo
        ccl_tracers = dict()
        bias_all = dict()
        for i, myst1 in enumerate(lk.statistics):
            trname1 = myst1.statistic.source0.sacc_tracer
            trname2 = myst1.statistic.source1.sacc_tracer
            tr1 = myst1.statistic.source0.tracers[0].ccl_tracer  # Pulling out the tracers
            tr2 = myst1.statistic.source1.tracers[0].ccl_tracer
            ccl_tracers[trname1] = tr1
            ccl_tracers[trname2] = tr2
            if 'lens' in trname1:
                bias_all[trname1] = myst1.statistic.source0.bias
            if 'lens' in trname2:
                bias_all[trname2] = myst1.statistic.source1.bias
        for key in bias_all.keys():
            tjpcov_config['tjpcov'][f'bias_{key}'] = bias_all[key]
        tjpcov_config['tjpcov']['sacc_file'] = S
        tjpcov_config['tjpcov']['IA'] = config['cov_options'].get('IA', None)
        tjpcov_config['tjpcov']['cov_type'] = ['FourierGaussianFsky']
        tjpcov_config['tjpcov']['fsky'] = config['cov_options']['fsky']
        tjpcov_config['tjpcov']['binning_info'] = dict()
        tjpcov_config['tjpcov']['binning_info']['ell_edges'] = \
            eval(config['cov_options']['binning_info']['ell_edges'])
        for tr in S.tracers:
            _, ndens = get_noise_power(config, S, tr, return_ndens=True)
            tjpcov_config['tjpcov'][f'Ngal_{tr}'] = ndens
            if 'src' in tr:
                tjpcov_config['tjpcov'][f'sigma_e_{tr}'] = config['sources']['ellipticity_error']

        cov_calc = TJPCovGaus(tjpcov_config)
        if config['general']['ignore_scale_cuts']:
            cov_all = cov_calc.get_covariance()
        else:
            ndata = len(S.mean)
            cov_all = np.zeros((ndata, ndata))
            for i, trcombs1 in enumerate(S.get_tracer_combinations()):
                ii = S.indices(tracers=trcombs1)
                for trcombs2 in S.get_tracer_combinations()[i:]:
                    jj = S.indices(tracers=trcombs2)
                    ii_all, jj_all = np.meshgrid(ii, jj, indexing='ij')
                    cov_here = cov_calc.get_covariance_block(trcombs1, trcombs2)
                    cov_all[ii_all, jj_all] = cov_here[:len(ii), :len(jj)]
                    cov_all[jj_all.T, ii_all.T] = cov_here[:len(ii), :len(jj)].T
        S.add_covariance(cov_all)
    else:
        raise Warning('''Currently only internal Gaussian covariance and SRD has been implemented,
                         cov_type is not understood. Using identity matrix as covariance.''')
    if write_sacc:
        print(config['fiducial_sacc_path'])
        S.save_fits(config['fiducial_sacc_path'], overwrite=True)
    # Update covariance and inverse -- TODO need to update cholesky!!
    lk = ConstGaussian(statistics=stats)
    lk.read(S)
    if return_all_outputs:
        return lk, S, tools, sys_params
