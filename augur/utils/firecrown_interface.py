import pyccl as ccl

from firecrown.modeling_tools import ModelingTools
from firecrown.likelihood.factories import (
    DataSourceSacc,
    TwoPointExperiment,
    TwoPointFactory,
)

from firecrown.ccl_factory import (
    CCLFactory,
    CCLCreationMode,
    CCLPureModeTransferFunction,
    CAMBExtraParams,
    PoweSpecAmplitudeParameter,
)
from firecrown.metadata_types import Galaxies
from firecrown.data_functions import TwoPointBinFilterCollection, TwoPointBinFilter

import warnings

TRANSFER_FUNCTION_REGISTRY = {
    "boltzmann_camb": CCLPureModeTransferFunction.BOLTZMANN_CAMB,
    "boltzmann_class": CCLPureModeTransferFunction.BOLTZMANN_CLASS,
    "eisenstein_hu": CCLPureModeTransferFunction.EISENSTEIN_HU,
    "eh": CCLPureModeTransferFunction.EISENSTEIN_HU,
    "bbks": CCLPureModeTransferFunction.BBKS,
}

PT_CALCULATOR_REGISTRY = {
    "eulerian_pt_calculator": ccl.nl_pt.EulerianPTCalculator,
    "lagrangian_pt_calculator": ccl.nl_pt.LagrangianPTCalculator,
    "bacco_lbias_calculator": ccl.nl_pt.BaccoLbiasCalculator,
}

BARYON_HM_REGISTRY = ['mead2020_feedback', 'mead', 'mead2015', 'mead2016']

TP_FILTER_REGISTRY = {'galaxy_shear_cl_ee': [[Galaxies.SHEAR_E, Galaxies.SHEAR_E]],
                      'galaxy_density_cl': [[Galaxies.COUNTS, Galaxies.COUNTS]],
                      'galaxy_shearDensity_cl_e': [[Galaxies.COUNTS, Galaxies.SHEAR_E]],
                      }


def _create_ccl_factory(config):
    """
    Build and return (ccl_factory, ccl_cosmo) using entries in the
    'cosmo' and optional 'ccl_accuracy' sections of the config.

    Assumptions:
      - Pure CCL mode (optional CAMB nonlinear sampling extras).
      - Transfer function chosen via config['cosmo']['transfer_function'].
      - Optional accuracy tweaks in config['ccl_accuracy'], applied directly to ccl.
    """
    # Copy to avoid mutating original
    cosmo_cfg = dict(config['cosmo'])
    camb_extra = None
    # get vs. pop is to allow passing the rest of the cosmo_cfg to ccl.Cosmology
    # Transfer function selection
    tf_name = cosmo_cfg.pop("transfer_function", "boltzmann_camb").lower()
    tf_enum = TRANSFER_FUNCTION_REGISTRY.get(tf_name, CCLPureModeTransferFunction.BOLTZMANN_CAMB)

    # Optional flag to require nonlinear P(k)
    require_nl = bool(cosmo_cfg.pop("require_nonlinear_pk", True))

    # Handle extra parameters for CAMB matter power spectrum
    camb_baryon = False
    if tf_name == 'boltzmann_camb':
        extra_params = cosmo_cfg.get("extra_parameters", None)
        if extra_params is not None:
            extra_params_camb = extra_params.get("camb", None)
            if extra_params_camb is not None:
                camb_extra = CAMBExtraParams(**extra_params_camb)
                if 'halofit_version' in extra_params_camb.keys() and require_nl:
                    halofit_version = extra_params_camb['halofit_version']
                    if halofit_version in BARYON_HM_REGISTRY:
                        camb_baryon = True
                else:
                    raise ValueError('When using CAMB transfer function, \
                                     halofit_version must be specified \
                                     in extra_parameters.camb.')

    amplitude = None
    if cosmo_cfg.get("A_s", None) is not None:
        amplitude = PoweSpecAmplitudeParameter.AS
    elif cosmo_cfg.get("sigma8", None) is not None:
        amplitude = PoweSpecAmplitudeParameter.SIGMA8
    else:
        raise ValueError("Either A_s or sigma8 must be specified in cosmology config")

    # Apply accuracy overrides (global pyccl settings) before building cosmology
    acc_cfg = config.pop("ccl_accuracy", {})
    for k, v in acc_cfg.get("spline_params", {}).items():
        if hasattr(ccl.spline_params, k):
            ccl.spline_params[k] = type(getattr(ccl.spline_params, k))(v)
    for k, v in acc_cfg.get("gsl_params", {}).items():
        if hasattr(ccl.gsl_params, k):
            ccl.gsl_params[k] = type(getattr(ccl.gsl_params, k))(v)

    # Build cosmology
    cosmo = ccl.Cosmology(**cosmo_cfg)

    factory = CCLFactory(
        creation_mode=CCLCreationMode.PURE_CCL_MODE,
        pure_ccl_transfer_function=tf_enum,
        require_nonlinear_pk=require_nl,
        camb_extra_params=camb_extra,
        use_camb_hm_sampling=camb_baryon,
        amplitude_parameter=amplitude,
    )
    factory.cosmo = cosmo
    return factory, cosmo


def _create_pt_calculator(config, cosmo):
    """
    Build and return a pyccl PTCalculator using entries in the
    'pt_calculator' section of the config.

    Assumptions:
      - no nested parameter settings in config.
    """
    # I think we should generally pop these to avoid passing them again
    pt_cfg = config.pop("pt_calculator", None)
    if pt_cfg is None or pt_cfg == {}:
        return None

    cfg = dict(pt_cfg)  # shallow copy
    calc_type = cfg.pop("type", "eulerian_pt_calculator").lower()
    cls = PT_CALCULATOR_REGISTRY.get(calc_type)
    if cls is None:
        raise ValueError(f"Unknown PT calculator type '{calc_type}'")
    try:
        return cls(cosmo=cosmo, **cfg)
    except TypeError as e:
        raise ValueError(f"Invalid PT calculator parameters for '{calc_type}': {e}")


def _create_hm_calculator(config, cosmo):
    """
    Build and return a pyccl HaloModelCalculator using entries in the
    'hm_calculator' section of the config.

    Assumptions:
      - no nested parameter settings in config.
    """
    # I think we should generally pop these to avoid passing them again
    hm_cfg = config.pop("hm_calculator", None)
    if hm_cfg is None or hm_cfg == {}:
        return None

    cfg = dict(hm_cfg)  # shallow copy
    try:
        hm_calculator = ccl.hm.HaloModelCalculator(cosmo=cosmo, **cfg)
        return hm_calculator
    except TypeError as e:
        raise ValueError(f"Invalid Halo Model calculator parameters: {e}")


def _create_cM_relation(config):
    """
    Add 'cM_relation' section of the config to be parsed by firecrown.

    Assumptions:
      - type of cM relation set via config['cM_relation']['type'].
    """

    # I think we should generally pop these to avoid passing them again
    cm_cfg = config.pop("cM_relation", None)
    if cm_cfg is None or cm_cfg == {}:
        return None
    if type(cm_cfg) is not str:
        raise ValueError("cM_relation config must be a string specifying the type")
    return cm_cfg


# NOT IMPLEMENTED YET, probably should not?
def _create_pk_modifiers(config):
    """
    Build and return a list of pyccl PowerSpectrumModifier using entries in the
    'pk_modifiers' section of the config.
    """
    pkm_cfg = config.pop("pk_modifiers", None)
    if pkm_cfg is not None:
        warnings.warn("PowerSpectrumModifier is not yet implemented in Augur")

    return None


# NOT IMPLEMENTED YET, probably should not?
def _create_powerspectra(config):
    """
    Build and return a list of pyccl PowerSpectrumCalculator using entries in the
    'powerspectra' section of the config.
    """
    ps_cfg = config.pop("powerspectra", None)
    if ps_cfg is not None:
        warnings.warn("PowerSpectrumCalculator is not yet implemented in Augur")

    return None


# NOT IMPLEMENTED YET
def _create_cluster_abundance(config, cosmo):
    """
    Build and return a firecrown ClusterAbundance using entries in the
    'cluster_abundance' section of the config.

    Assumptions:
      - mass definition and halo mass function set via
        config['cluster_abundance'].
    """
    cab_cfg = config.pop("cluster_abundance", None)
    if cab_cfg is not None:
        warnings.warn("ClusterAbundance is not yet implemented in Augur")

    return None


# NOT IMPLEMENTED YET
def _create_clusterdeltasigma(config, cosmo):
    """
    Build and return a firecrown ClusterDeltaSigma using entries in the
    'cluster_deltasigma' section of the config.

    Assumptions:
      - mass definition and halo mass function set via
        config['cluster_deltasigma'].
    """
    cds_cfg = config.pop("cluster_deltasigma", None)
    if cds_cfg is not None:
        warnings.warn("ClusterDeltaSigma is not yet implemented in Augur")

    return None


def create_modeling_tools(config):

    factory, cosmo = _create_ccl_factory(config)
    pt_calculator = _create_pt_calculator(config, cosmo)
    hm_calculator = _create_hm_calculator(config, cosmo)
    cluster_abundance = _create_cluster_abundance(config, cosmo)
    cluster_deltasigma = _create_clusterdeltasigma(config, cosmo)
    tools = ModelingTools(ccl_factory=factory,
                          pt_calculator=pt_calculator,
                          hm_calculator=hm_calculator,
                          cluster_abundance=cluster_abundance,
                          cluster_deltasigma=cluster_deltasigma,
                          )
    return tools, cosmo


# Leave possibility open for multi-probe
FC_FACTORY_REGISTRY = {'TwoPointFactory': TwoPointFactory}


def load_likelihood_from_yaml(config, ccl_factory, S, filters=[]):
    lk_config = config.get("Firecrown_Factory", None)
    if (lk_config is None) or (lk_config == {}):
        raise ValueError("Firecrown_Factory must have contents to produce a Firecrown Likelihood")
    keys = lk_config.keys()

    if len(keys) != 1:
        raise ValueError("Augur can only support one Factory definition.")
    keys = list(keys)
    like_dict = lk_config[keys[0]]
    probes = FC_FACTORY_REGISTRY.get(keys[0])
    if probes is None:
        raise NameError(str(keys[0])+" is not a valid Firecrown Factory")

    tpf = probes.model_validate(like_dict)

    tpbfc = TwoPointBinFilterCollection(
            require_filter_for_all=False,
            allow_empty=True,
            filters=filters,
        )

    two_point_experiment = TwoPointExperiment(
                                two_point_factory=tpf,
                                ccl_factory=ccl_factory,
                                data_source=DataSourceSacc(
                                    sacc_data_file=S,
                                    filters=tpbfc,
                                    ),
                                )
    lk = two_point_experiment.make_likelihood()

    return lk


def create_twopoint_filter(combo_name, tr1, tr2, cut_low, cut_high):
    info = TP_FILTER_REGISTRY.get(combo_name, None)
    if info is None:
        raise ValueError(f"Unknown two point combination '{combo_name}'")
    m1, m2 = info[0]
    filter = TwoPointBinFilter.from_args(
                name1=tr1,
                name2=tr2,
                measurement1=m1,
                measurement2=m2,
                lower=cut_low,
                upper=cut_high,
            )
    return filter
