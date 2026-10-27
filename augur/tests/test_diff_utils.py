from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_five_pt_stencil():
    """Load five_pt_stencil without importing augur/__init__.py.

    Importing any `augur.*` module executes `augur/__init__.py`, which pulls in
    optional heavy dependencies (e.g. firecrown). For this unit test we only
    need the pure-numpy finite difference routine.
    """

    diff_utils_path = Path(__file__).resolve().parents[1] / "utils" / "diff_utils.py"
    spec = importlib.util.spec_from_file_location("_augur_diff_utils", diff_utils_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.five_pt_stencil


five_pt_stencil = _load_five_pt_stencil()


def _vectorized_fun(x):
    """R^3 -> R^2 test function that supports both 1D and batched 2D inputs."""
    x = np.asarray(x, dtype=float)

    if x.ndim == 1:
        x0, x1, x2 = x
        return np.array(
            [x0**2 + np.sin(x1), np.exp(x2) + x0 * x1],
            dtype=float,
        )

    if x.ndim == 2:
        x0 = x[:, 0]
        x1 = x[:, 1]
        x2 = x[:, 2]
        return np.column_stack(
            [x0**2 + np.sin(x1), np.exp(x2) + x0 * x1]
        ).astype(float)

    raise ValueError(f"Unsupported x shape for test function: {x.shape}")


def test_five_pt_stencil_matches_numdifftools_scalar():
    nd = pytest.importorskip("numdifftools")

    def f(x):
        x = np.asarray(x, dtype=float)
        return np.sin(x) + x**2

    x0 = 0.37
    h = 1.0e-5

    augur_der = five_pt_stencil(f, x0, h=h)
    nd_der = nd.Derivative(f, step=h, method="central", order=4)(x0)

    np.testing.assert_allclose(augur_der, nd_der, rtol=1e-10, atol=1e-12)


def test_five_pt_stencil_matches_numdifftools_jacobian_vector_to_vector():
    nd = pytest.importorskip("numdifftools")

    x0 = np.array([0.1, -0.3, 0.2], dtype=float)
    h = 1.0e-5

    augur_jac_t = five_pt_stencil(_vectorized_fun, x0, h=h)
    nd_jac = nd.Jacobian(_vectorized_fun, step=h, method="central", order=4)(x0)

    # Augur's five_pt_stencil returns shape (n_params, n_outputs).
    # numdifftools returns shape (n_outputs, n_params).
    np.testing.assert_allclose(augur_jac_t, nd_jac.T, rtol=1e-9, atol=1e-11)
