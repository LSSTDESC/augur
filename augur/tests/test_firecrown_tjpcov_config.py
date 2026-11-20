from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from augur.utils.config_io import parse_array


def test_test_yaml_tjpcov_firecrown_inputs_are_valid_and_runnable():
    """Integration-style unit test for our test YAML.

    Verifies that the YAML config used in tests:
      - can be parsed by Augur
      - can build a CCL cosmology + Firecrown likelihood
      - can compute a TJPCov Gaussian covariance

    This intentionally exercises the Firecrown/tjpcov plumbing so that future
    config changes fail loudly.
    """

    pytest.importorskip("firecrown")
    pytest.importorskip("tjpcov")
    pyccl = pytest.importorskip("pyccl")

    from augur.utils.config_io import parse_config
    from augur.generate import generate

    cfg_path = Path(__file__).parent / "test_tjpcov.yaml"
    config = parse_config(str(cfg_path))

    # Basic config sanity: required sections for the plumbing we want to test.
    assert "cosmo" in config
    assert "Firecrown_Factory" in config
    assert "statistics" in config
    assert "cov_options" in config
    assert config["cov_options"]["cov_type"] == "tjpcov"
    assert isinstance(config["cov_options"]["fsky"], (int, float))
    assert "binning_info" in config["cov_options"]
    assert "ell_edges" in config["cov_options"]["binning_info"]

    # Ensure the test YAML expressions are evaluable in the same way Augur expects.
    ell_edges = parse_array(config["cov_options"]["binning_info"]["ell_edges"])
    assert isinstance(ell_edges, np.ndarray)
    assert ell_edges.ndim == 1 and len(ell_edges) >= 3
    assert np.all(np.diff(ell_edges) > 0)

    # Run generation end-to-end (but do not write outputs).
    lk, S, tools, _sys_params = generate(str(cfg_path), return_all_outputs=True, write_sacc=False)

    assert tools.ccl_cosmo is not None
    assert isinstance(tools.ccl_cosmo, pyccl.Cosmology)

    # Check tracers exist for both lenses and sources.
    assert "lens0" in S.tracers
    assert "src0" in S.tracers

    cov = S.covariance.covmat
    assert cov.shape == (len(S.mean), len(S.mean))
    assert np.all(np.isfinite(cov))
    np.testing.assert_allclose(cov, cov.T, rtol=0, atol=0)

    # Likelihood should have computed theory vector (S.mean is the fiducial prediction).
    assert len(S.mean) > 0
    assert lk is not None
