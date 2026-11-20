import ast
import types
import numpy as np
from pathlib import Path


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


def read_fisher_from_file(fisher_path, fid_path, rename=None):
    """
    Read a Fisher matrix saved in Augur's output format.

    Parameters:
    -----------
    fisher_path : str
        Path to the Fisher matrix, as written to `fisher: output`.
    fid_path : str
        Path to the matching parameter file, as written to `fisher: fid_output`:
        a line of parameter names followed by a line of fiducial values.
    rename : dict, optional
        Maps parameter names in `fid_path` to the names to use instead.

    Returns:
    --------
    fisher : np.ndarray
        Fisher matrix, in the order of `names`.
    names : list of str
        Parameter names, after renaming.
    fiducials : dict
        Fiducial value of each parameter, keyed by its name after renaming.
    """
    from astropy.table import Table

    fisher = np.atleast_2d(np.loadtxt(fisher_path))
    tab = Table.read(fid_path, format='ascii')
    rename = rename or {}
    unused = sorted(set(rename) - set(tab.colnames))
    if unused:
        raise ValueError(f'rename lists {unused}, which are not parameters in {fid_path}.')
    names = [rename.get(name, name) for name in tab.colnames]
    _check_fisher_shape(fisher, names, fisher_path)
    if not np.allclose(fisher, fisher.T):
        raise ValueError(f'{fisher_path} is not symmetric.')
    fiducials = {name: float(tab[col][0]) for name, col in zip(names, tab.colnames)}
    return fisher, names, fiducials


def write_fisher_to_file(fisher, names, fiducials, fisher_path, fid_path):
    """
    Write a Fisher matrix in Augur's output format, which `read_fisher_from_file` reads.
    Use it to turn a Fisher matrix from elsewhere into one that can be added to an
    Augur run with `external_fishers`.

    Parameters:
    -----------
    fisher : array_like
        Fisher matrix, in the order of `names`.
    names : list of str
        Parameter names.
    fiducials : dict
        Fiducial value of each parameter, keyed by name.
    fisher_path : str
        Path for the Fisher matrix.
    fid_path : str
        Path for the parameter file: a line of names followed by a line of values.
    """
    from astropy.table import Table

    fisher, names = np.atleast_2d(np.asarray(fisher, dtype=np.float64)), list(names)
    _check_fisher_shape(fisher, names, fisher_path)
    missing = [name for name in names if name not in fiducials]
    if missing:
        raise ValueError(f'No fiducial values given for {missing}.')
    np.savetxt(fisher_path, fisher)
    values = np.array([fiducials[name] for name in names], dtype=np.float64)
    Table(values, names=names).write(fid_path, format='ascii', overwrite=True)


def _check_fisher_shape(fisher, names, source):
    """Check that the parameter names are unique and match the size of the Fisher matrix."""
    if len(set(names)) != len(names):
        raise ValueError(f'{source}: the parameter names are not unique: {names}.')
    if fisher.shape != (len(names), len(names)):
        raise ValueError(f'{source} has shape {fisher.shape}, but there are {len(names)} '
                         'parameter names.')


def _check_path_exists(path):
    """
    Check if a given path exists.
    Creates it if it does not exist, and raises an error if the path is invalid.

    Parameters
    ----------
    path : str or Path
        The path to check.
    """
    if not Path(path).exists():
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise FileNotFoundError(f"The specified path does not exist: {path}") from e
    return path
