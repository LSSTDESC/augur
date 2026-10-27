from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_numdifftools_derivative_args_forwarded_from_yaml(monkeypatch):
    """Ensure Augur forwards YAML-provided numdifftools kwargs.

    This matters because Augur's `Analyze.get_derivatives(method='numdifftools')`
    uses `nd.Jacobian(..., **kwargs)` and the canonical way to request a 5-point
    central stencil in numdifftools is `method='central', order=4`.

    We don't actually need to compute the Jacobian here; we only verify that the
    kwargs make it to `numdifftools.Jacobian`.
    """

    pytest.importorskip("firecrown")
    nd = pytest.importorskip("numdifftools")

    from augur.analyze import Analyze
    from augur.utils.config_io import parse_config

    cfg_path = Path(__file__).parent / "test.yaml"
    config = parse_config(str(cfg_path))

    config.setdefault("fisher", {})
    config["fisher"]["derivative_method"] = "numdifftools"
    config["fisher"]["derivative_args"] = {"method": "central", "order": 4}

    captured: dict[str, object] = {}

    def spy_jacobian(function, step=None, **kwargs):
        captured.update(kwargs)

        class _DummyJacobian:
            def __init__(self, f):
                self._f = f

            def __call__(self, x):
                y = np.asarray(self._f(x), dtype=float).ravel()
                x = np.asarray(x, dtype=float).ravel()
                # numdifftools returns shape (n_outputs, n_inputs)
                return np.zeros((y.size, x.size), dtype=float)

        return _DummyJacobian(function)

    monkeypatch.setattr(nd, "Jacobian", spy_jacobian)

    fish = Analyze(config)
    fish.get_derivatives(method="numdifftools", force=True)

    assert captured.get("method") == "central"
    assert captured.get("order") == 4
