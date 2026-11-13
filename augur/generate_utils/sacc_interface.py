"""
SACC interface dispatcher for Augur.

Maps SACC data types to the correct ``add_*`` / ``get_*`` methods so that
:func:`augur.generate.generate` can fill theory predictions for harmonic-space,
real-space, and (eventually) other probe types without hard-coding a single
SACC method.

The classification relies on the SACC naming convention:

* ``_cl`` suffix (or ``_cl_`` component) → harmonic space
* ``_xi`` suffix (or ``_xi_`` component) → real space
* everything else (``cluster_counts``, ``cluster_mean_log_mass``, …)
  → non-two-point
"""

import re

# ── data-type classification ────────────────────────────────────────── #

_HARMONIC_RE = re.compile(r'_cl(?:_[a-zA-Z]+)?$')
_REAL_SPACE_RE = re.compile(r'_xi(?:_[a-zA-Z]+)?$')


def classify_data_type(data_type):
    """Return ``'harmonic'``, ``'real_space'``, or ``'other'``."""
    if _HARMONIC_RE.search(data_type):
        return 'harmonic'
    if _REAL_SPACE_RE.search(data_type):
        return 'real_space'
    return 'other'


def is_harmonic(data_type):
    return classify_data_type(data_type) == 'harmonic'


def is_real_space(data_type):
    return classify_data_type(data_type) == 'real_space'


# ── unified getters ─────────────────────────────────────────────────── #

def get_data_points(S, data_type, tracer1, tracer2, **kwargs):
    """Dispatch to ``S.get_ell_cl`` or ``S.get_theta_xi``."""
    space = classify_data_type(data_type)
    if space == 'harmonic':
        return S.get_ell_cl(data_type, tracer1, tracer2, **kwargs)
    if space == 'real_space':
        return S.get_theta_xi(data_type, tracer1, tracer2, **kwargs)
    raise ValueError(
        f"Data type '{data_type}' is not a two-point statistic; "
        f"cannot retrieve data points with get_ell_cl / get_theta_xi."
    )


# ── unified adder ───────────────────────────────────────────────────── #

def add_data_points(S, data_type, tracer1, tracer2, x, y, window=None):
    """Dispatch to ``S.add_ell_cl`` or ``S.add_theta_xi``."""
    space = classify_data_type(data_type)
    if space == 'harmonic':
        S.add_ell_cl(data_type, tracer1, tracer2, x, y, window=window)
        return
    if space == 'real_space':
        S.add_theta_xi(data_type, tracer1, tracer2, x, y, window=window)
        return
    raise ValueError(
        f"Data type '{data_type}' is not a two-point statistic; "
        f"cannot add data points with add_ell_cl / add_theta_xi."
    )


# ── extract placeholder x-values & windows from an existing SACC ──── #

def extract_x_and_windows(S, data_type, tracer1, tracer2):
    """Return ``(x_values, window_or_None)`` from a placeholder SACC.

    For harmonic data types this returns (ell, bandpower_window),
    for real-space it returns (theta, None).
    """
    space = classify_data_type(data_type)
    if space == 'harmonic':
        idx = S.indices(tracers=(tracer1, tracer2))
        window = S.get_bandpower_windows(idx)
        x_vals, _ = S.get_ell_cl(data_type, tracer1, tracer2)
        return x_vals, window
    if space == 'real_space':
        x_vals, _ = S.get_theta_xi(data_type, tracer1, tracer2)
        return x_vals, None
    raise ValueError(
        f"Data type '{data_type}' is not a two-point statistic."
    )
