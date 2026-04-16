import ast
import types
import numpy as np


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
