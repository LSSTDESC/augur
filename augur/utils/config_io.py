import ast
import types
import warnings
import numpy as np


# CMB lensing source redshift.
#
# This mirrors firecrown's own default (`CMBConvergenceFactory.z_source`, and
# `_DEFAULT_CMB_Z_LSS` used when a sacc Map tracer carries no `z_lss`). It is
# defined here so augur has one value rather than a literal repeated in
# generate.py and generate_utils/cmb_lensing.py, which is what let the two
# drift apart silently.
CMB_Z_SOURCE_DEFAULT = 1100.0

# TJPCov hard-codes this in `ccl.CMBLensingTracer` and offers no configuration
# hook (tjpcov/covariance_builder.py, the `cmb_convergence` branch of
# `get_tracer_info`). Deliberately a separate constant from the one above: the
# two are equal today by coincidence, not by construction, and collapsing them
# would make this guard silently follow a future change to firecrown's default.
#
# TEMPORARY -- remove this and the guard that uses it once
# https://github.com/LSSTDESC/TJPCov/issues/124 lands, together with the
# TJPCovGaus.get_tracer_info override in augur/utils/cov_utils.py, which
# carries the same note.
TJPCOV_CMB_Z_SOURCE = 1100.0


# Restricted namespace for safe evaluation of array expressions in configs.
# Only numpy array-construction helpers and dtypes are exposed — no builtins, no I/O.
_np_ns = types.SimpleNamespace(
    **{attr: getattr(np, attr) for attr in (
        'linspace', 'logspace', 'geomspace', 'arange',
        'array', 'zeros', 'ones', 'concatenate',
        'pi', 'inf',
        # dtypes users may reference in config strings
        'int32', 'int64', 'float32', 'float64',
    )}
)
_SAFE_NUMPY_NS = {'np': _np_ns}


def parse_array(value):
    """Safely evaluate a config string to a numpy array.

    Accepts either a plain list literal (``"[1, 2, 3]"``) or a limited
    set of numpy expressions (``"np.linspace(20, 15000, 21)"``).

    Parameters
    ----------
    value : str or array-like
        If already an array/list, returned as ``np.asarray(value)``.

    Returns
    -------
    np.ndarray
    """
    if not isinstance(value, str):
        return np.asarray(value)

    # Fast-path: plain list literal → use ast.literal_eval (no code exec)
    stripped = value.strip()
    if stripped.startswith('['):
        try:
            return np.asarray(ast.literal_eval(stripped))
        except (ValueError, SyntaxError):
            pass  # Fall through to restricted eval

    # Restricted eval with only safe numpy helpers
    try:
        result = eval(value, {"__builtins__": {}}, _SAFE_NUMPY_NS)  # noqa: S307
    except Exception as exc:
        raise ValueError(
            f"Cannot safely evaluate array expression: {value!r}"
        ) from exc
    return np.asarray(result)


def validate_amplitude_parameter(cosmo_cfg):
    """
    Validate that exactly one of sigma8 or A_s is specified in the cosmology config.
    Raises ValueError if both are defined as non-None values, which is ambiguous.
    """
    has_sigma8 = cosmo_cfg.get('sigma8') is not None
    has_A_s = cosmo_cfg.get('A_s') is not None
    if has_sigma8 and has_A_s:
        raise ValueError(
            'Both sigma8 and A_s are specified in the cosmo config. '
            'These parameters are mutually exclusive: use sigma8 (RMS matter '
            'fluctuation in 8 h^-1 Mpc spheres) OR A_s (scalar amplitude of '
            'the primordial power spectrum), not both.'
        )
    if not has_sigma8 and not has_A_s:
        raise ValueError(
            'Neither sigma8 nor A_s is specified in the cosmo config. '
            'Exactly one amplitude parameter must be provided.'
        )


def _cmb_factories(config):
    """
    Return the `cmb_factories` list from the Firecrown_Factory config block.

    Returns None when there is no Firecrown_Factory at all, which is the
    ConstGaussian path: there augur's own CMBConvergence object is used and it
    does carry the configured z_source, so there is no second value to
    disagree with. Returns an empty list when the block exists but declares no
    CMB factories.
    """
    fc_cfg = config.get('Firecrown_Factory', None)
    if not fc_cfg:
        return None
    # TwoPointFactory is the only factory augur supports (FC_FACTORY_REGISTRY);
    # anything else is rejected by load_likelihood_from_yaml with its own error,
    # so stay quiet here rather than producing a confusing second message.
    if 'TwoPointFactory' not in fc_cfg:
        return None
    return (fc_cfg['TwoPointFactory'] or {}).get('cmb_factories', []) or []


def validate_cmb_z_source(config, z_source=None, origin=None, check_cov_type=True):
    """
    Check that the CMB lensing source redshift agrees everywhere it is declared.

    Only `Firecrown_Factory.TwoPointFactory.cmb_factories[].z_source` reaches
    the theory prediction -- firecrown rebuilds the CMB convergence source from
    that list and `CMBConvergenceFactory.create` ignores the `z_lss` metadata
    stored in the sacc. So a disagreement between it and whatever augur built
    the kappa tracer from gives a data vector and a theory vector at different
    source redshifts, with no error raised anywhere.

    This validates and never writes. Silently syncing one config value from
    another would hide a typo rather than surface it, and augur re-parses the
    config from disk partway through `generate`, so an in-place fix would be
    discarded anyway.

    :param config: the full augur config dict.
    :param z_source: the value augur actually built the kappa tracer from. Pass
        it explicitly on the `use_sacc` path, where it comes from the sacc
        tracer's `z_lss` metadata rather than from the config. When None, it is
        read from `cmb_lensing.z_source`.
    :param origin: human-readable description of where `z_source` came from,
        used in the error messages.
    :param check_cov_type: whether to apply the TJPCov guard. False on the
        `use_sacc` path, which returns before any covariance is computed and
        never reads `cov_options.cov_type`.
    """
    cmb_cfg = config.get('cmb_lensing', None)
    factories = _cmb_factories(config)

    if z_source is None:
        # Generation path: cmb_lensing.z_source is what add_cmb_tracer used.
        if cmb_cfg is None:
            if factories:
                warnings.warn(
                    '`Firecrown_Factory.TwoPointFactory.cmb_factories` is configured but '
                    'the config has no `cmb_lensing` section, so no CMB convergence tracer '
                    'will be written to the sacc and the CMB factories are inert.'
                )
            return
        z_source = float(cmb_cfg.get('z_source', CMB_Z_SOURCE_DEFAULT))
        origin = origin or '`cmb_lensing.z_source`'
    else:
        # use_sacc path: the sacc wins, and cmb_lensing.z_source is inert.
        z_source = float(z_source)
        origin = origin or 'the sacc CMB tracer metadata `z_lss`'
        if cmb_cfg is not None and 'z_source' in cmb_cfg:
            z_cfg = float(cmb_cfg['z_source'])
            if z_cfg != z_source:
                warnings.warn(
                    f'{origin} = {z_source!r} but `cmb_lensing.z_source` = {z_cfg!r}. '
                    'When a sacc is supplied the sacc value is used and the config value '
                    'is ignored; the mismatch suggests one of the two is stale.'
                )

    if factories is None:
        # No Firecrown_Factory: the ConstGaussian path uses augur's own
        # CMBConvergence, which carries z_source. Nothing to cross-check.
        return

    if not factories:
        raise ValueError(
            'CMB lensing is configured but '
            '`Firecrown_Factory.TwoPointFactory.cmb_factories` is empty or absent. '
            'Firecrown rebuilds the CMB convergence source from this list rather than '
            'from the sacc, and will fail to find a factory for the CMB tracer. '
            'Please add an entry, for example '
            f'`- {{type_source: default, z_source: {z_source}}}`.'
        )

    for i, factory in enumerate(factories):
        factory = factory or {}
        declared = 'z_source' in factory
        z_fac = float(factory.get('z_source', CMB_Z_SOURCE_DEFAULT))
        if z_fac == z_source:
            continue
        note = '' if declared else (
            " (firecrown's default -- no `z_source` key is present in that entry)"
        )
        raise ValueError(
            'CMB lensing source redshift disagrees between the augur and firecrown '
            f'configurations: {origin} = {z_source!r} but '
            f'`Firecrown_Factory.TwoPointFactory.cmb_factories[{i}].z_source` = '
            f'{z_fac!r}{note}. Only the firecrown value reaches the theory prediction, '
            'so the data vector and the theory would use different source redshifts. '
            'Please set `cmb_factories[].z_source` to match.'
        )

    if check_cov_type and z_source != TJPCOV_CMB_Z_SOURCE:
        cov_type = (config.get('cov_options', None) or {}).get('cov_type', None)
        if cov_type == 'tjpcov':
            raise ValueError(
                f'TJPCov hard-codes `z_source={TJPCOV_CMB_Z_SOURCE}` for the CMB lensing '
                'tracer and offers no configuration hook, but '
                f'{origin} = {z_source!r}. The covariance would be computed for a '
                'different source redshift than the data vector. Please set '
                f'`cmb_lensing.z_source` to {TJPCOV_CMB_Z_SOURCE}, or use '
                '`cov_options.cov_type: gaus_internal`. '
                'See https://github.com/LSSTDESC/TJPCov/issues/124.'
            )


def parse_config(config):
    """
    Utility to parse configuration file
    """
    if isinstance(config, str):
        from augur.parser import parse
        config = parse(config)
    elif isinstance(config, dict):
        pass
    else:
        raise ValueError('config must be a dictionary or path to a config file')
    return config


def read_fisher_from_file(base):
    '''
    Helper function to add external Fisher evaluations to the computed Fisher matrix in Augur.
    Requires the path have two files in the Augur format: fiducial and Fisher.

    This function does not check the compatibility of the fiducial systematic parameters
     or cosmology with the current Augur run; that is the user's responsibility.
    :param base: base path to the files (without _fiducials.dat or _fisher.dat)
    :return: fisher matrix and fiducial vector as numpy arrays
    '''
    try:
        fiducials = np.loadtxt(f"{base}_fiducials.dat")
        fisher = np.loadtxt(f"{base}_fisher.dat")
        # TODO: If we wind up changing the format of the Analyze object, we may want
        # to do some additional processing here to ensure compatibility/ease of use.
        return fisher, fiducials
    except Exception as e:
        raise RuntimeError(f"Could not read files at {base}. Exception: {e}")
