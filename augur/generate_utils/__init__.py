"""
Generate utilities for Augur.

This package contains the modular building blocks used by
:func:`augur.generate.generate` and :func:`augur.generate.generate_sacc_and_stats`
to assemble synthetic data vectors for different probes.

Submodules
----------
cosmology
    Cosmology initialisation from a config dict.
tracers
    Tracer / N(z) setup and naming helpers.
harmonic
    Harmonic-space (C_ell) two-point statistics.
real_space
    Real-space (xi) two-point correlation functions.
cmb_lensing
    CMB lensing statistics (stub).
cluster_counts
    Cluster-count statistics (stub).
covariance
    Covariance-matrix computation (Gaussian, SRD, TJPCov).
"""

from .cosmology import initialize_cosmology          # noqa: F401
from .tracers import (                                # noqa: F401
    get_tracers,
    add_nz,
    setup_sources,
    setup_lenses,
)
from .harmonic import add_harmonic_two_point          # noqa: F401
from .real_space import add_real_space_two_point      # noqa: F401
from .cmb_lensing import add_cmb_lensing              # noqa: F401
from .cluster_counts import add_cluster_counts        # noqa: F401
from .covariance import compute_covariance            # noqa: F401
from .sacc_interface import (                          # noqa: F401
    classify_data_type,
    is_harmonic,
    is_real_space,
    get_data_points,
    add_data_points,
    extract_x_and_windows,
)
