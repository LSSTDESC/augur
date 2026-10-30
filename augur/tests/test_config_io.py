import numpy as np
import pytest
from augur.utils.config_io import parse_config, read_fisher_from_file, validate_amplitude_parameter
from augur.utils.config_io import (
    validate_cmb_z_source,
    CMB_Z_SOURCE_DEFAULT,
    TJPCOV_CMB_Z_SOURCE,
)


class TestParseConfig:
    """Tests for the parse_config function"""

    def test_parse_config_with_dict(self):
        """Test that parse_config accepts and returns a dictionary unchanged"""
        config_dict = {"key1": "value1", "key2": 42, "nested": {"inner": "data"}}
        result = parse_config(config_dict)
        assert result is config_dict
        assert result == config_dict

    def test_parse_config_with_empty_dict(self):
        """Test parse_config with an empty dictionary"""
        config_dict = {}
        result = parse_config(config_dict)
        assert result == {}

    def test_parse_config_with_complex_dict(self):
        """Test parse_config with a complex nested dictionary"""
        config_dict = {
            "cosmology": {"Omega_m": 0.3, "Omega_l": 0.7},
            "bins": [1, 2, 3],
            "strings": "test",
            "nested": {"level2": {"level3": "deep"}},
        }
        result = parse_config(config_dict)
        assert result == config_dict

    def test_parse_config_with_yaml_file(self, tmp_path):
        """Test that parse_config can read from a YAML file"""
        config_file = tmp_path / "test_config.yml"
        config_content = """
cosmology:
  Omega_m: 0.3
  Omega_l: 0.7
model: test_model
parameters:
  - param1
  - param2
"""
        config_file.write_text(config_content)

        result = parse_config(str(config_file))
        assert isinstance(result, dict)
        assert result["cosmology"]["Omega_m"] == 0.3
        assert result["model"] == "test_model"
        assert result["parameters"] == ["param1", "param2"]

    def test_parse_config_with_yaml_file_jinja_template(self, tmp_path, monkeypatch):
        """Test parse_config with YAML file containing Jinja2 template variables"""
        monkeypatch.setenv("TEST_VALUE", "substituted_value")
        config_file = tmp_path / "test_config_jinja.yml"
        config_content = """
test_key: "{{ env.TEST_VALUE }}"
other_key: 123
"""
        config_file.write_text(config_content)

        result = parse_config(str(config_file))
        assert result["test_key"] == "substituted_value"
        assert result["other_key"] == 123

    def test_parse_config_rejects_none(self):
        """Test that parse_config rejects None input"""
        with pytest.raises(ValueError):
            parse_config(None)

    def test_parse_config_rejects_integer(self):
        """Test that parse_config rejects integer input"""
        with pytest.raises(ValueError):
            parse_config(123)

    def test_parse_config_rejects_list(self):
        """Test that parse_config rejects list input"""
        with pytest.raises(ValueError):
            parse_config([1, 2, 3])

    def test_parse_config_rejects_string_nonexistent_file(self):
        """Test that parse_config rejects a string path to nonexistent file"""
        with pytest.raises(Exception):  # FileNotFoundError or similar
            parse_config("/nonexistent/path/to/config.yml")

    def test_parse_config_rejects_invalid_yaml(self, tmp_path):
        """Test that parse_config rejects invalid YAML"""
        config_file = tmp_path / "invalid.yml"
        config_file.write_text("invalid: yaml: content: [")

        with pytest.raises(Exception):  # yaml.YAMLError or similar
            parse_config(str(config_file))


class TestReadFisherFromFile:
    """Tests for the read_fisher_from_file function"""

    def test_read_fisher_from_file_basic(self, tmp_path):
        """Test basic reading of Fisher matrix and fiducials"""
        base = str(tmp_path / "test")
        fiducials = np.array([1.0, 2.0, 3.0])
        fisher = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])

        np.savetxt(f"{base}_fiducials.dat", fiducials)
        np.savetxt(f"{base}_fisher.dat", fisher)

        fisher_read, fid_read = read_fisher_from_file(base)

        assert np.allclose(fisher_read, fisher)
        assert np.allclose(fid_read, fiducials)

    def test_read_fisher_from_file_1d_arrays(self, tmp_path):
        """Test reading 1D arrays"""
        base = str(tmp_path / "test_1d")
        fiducials = np.array([0.5, 1.5, 2.5, 3.5])
        fisher = np.array([10.0, 20.0, 30.0, 40.0])

        np.savetxt(f"{base}_fiducials.dat", fiducials)
        np.savetxt(f"{base}_fisher.dat", fisher)

        fisher_read, fid_read = read_fisher_from_file(base)

        assert np.allclose(fisher_read, fisher)
        assert np.allclose(fid_read, fiducials)

    def test_read_fisher_from_file_2d_array(self, tmp_path):
        """Test reading a 2D Fisher matrix"""
        base = str(tmp_path / "test_2d")
        fiducials = np.array([1.0, 2.0])
        fisher = np.array([[1.0, 0.5], [0.5, 2.0]])

        np.savetxt(f"{base}_fiducials.dat", fiducials)
        np.savetxt(f"{base}_fisher.dat", fisher)

        fisher_read, fid_read = read_fisher_from_file(base)

        assert fisher_read.shape == (2, 2)
        assert np.allclose(fisher_read, fisher)
        assert np.allclose(fid_read, fiducials)

    def test_read_fisher_from_file_large_matrix(self, tmp_path):
        """Test reading a larger Fisher matrix"""
        base = str(tmp_path / "test_large")
        n = 10
        fiducials = np.linspace(0.1, 1.0, n)
        # Create a symmetric positive definite matrix
        A = np.random.randn(n, n)
        fisher = A @ A.T

        np.savetxt(f"{base}_fiducials.dat", fiducials)
        np.savetxt(f"{base}_fisher.dat", fisher)

        fisher_read, fid_read = read_fisher_from_file(base)

        assert fisher_read.shape == (n, n)
        assert np.allclose(fisher_read, fisher)
        assert np.allclose(fid_read, fiducials)

    def test_read_fisher_from_file_missing_fisher(self, tmp_path):
        """Test error handling when fisher file is missing"""
        base = str(tmp_path / "missing_fisher")
        fiducials = np.array([1.0, 2.0, 3.0])

        np.savetxt(f"{base}_fiducials.dat", fiducials)
        # Intentionally don't create the fisher file

        with pytest.raises(RuntimeError, match="Could not read files"):
            read_fisher_from_file(base)

    def test_read_fisher_from_file_missing_fiducials(self, tmp_path):
        """Test error handling when fiducials file is missing"""
        base = str(tmp_path / "missing_fiducials")
        fisher = np.array([[1.0, 0.0], [0.0, 2.0]])

        np.savetxt(f"{base}_fisher.dat", fisher)
        # Intentionally don't create the fiducials file

        with pytest.raises(RuntimeError, match="Could not read files"):
            read_fisher_from_file(base)

    def test_read_fisher_from_file_missing_both(self, tmp_path):
        """Test error handling when both files are missing"""
        base = str(tmp_path / "missing_both")

        with pytest.raises(RuntimeError, match="Could not read files"):
            read_fisher_from_file(base)

    def test_read_fisher_from_file_corrupted_fisher(self, tmp_path):
        """Test error handling when fisher file is corrupted"""
        base = str(tmp_path / "corrupted")
        fiducials = np.array([1.0, 2.0])

        np.savetxt(f"{base}_fiducials.dat", fiducials)
        # Write corrupted data to fisher file
        with open(f"{base}_fisher.dat", "w") as f:
            f.write("this is not valid data")

        with pytest.raises(RuntimeError, match="Could not read files"):
            read_fisher_from_file(base)

    def test_read_fisher_from_file_preserves_values(self, tmp_path):
        """Test that read values are correctly preserved"""
        base = str(tmp_path / "preserve")
        fiducials = np.array([0.1, 0.5, 0.9])
        fisher = np.array([[1.5, 0.2, -0.1], [0.2, 2.5, 0.3], [-0.1, 0.3, 1.8]])

        np.savetxt(f"{base}_fiducials.dat", fiducials)
        np.savetxt(f"{base}_fisher.dat", fisher)

        fisher_read, fid_read = read_fisher_from_file(base)

        # Check specific values
        assert fid_read[0] == pytest.approx(0.1)
        assert fid_read[1] == pytest.approx(0.5)
        assert fid_read[2] == pytest.approx(0.9)
        assert fisher_read[0, 0] == pytest.approx(1.5)
        assert fisher_read[1, 2] == pytest.approx(0.3)
        assert fisher_read[2, 1] == pytest.approx(0.3)


class TestValidateAmplitudeParameter:
    """Tests for the validate_amplitude_parameter function."""

    def test_sigma8_only_passes(self):
        cosmo_cfg = {'Omega_c': 0.27, 'sigma8': 0.8, 'A_s': None}
        validate_amplitude_parameter(cosmo_cfg)  # Should not raise

    def test_A_s_only_passes(self):
        cosmo_cfg = {'Omega_c': 0.27, 'sigma8': None, 'A_s': 2.1e-9}
        validate_amplitude_parameter(cosmo_cfg)  # Should not raise

    def test_neither_defined_raises(self):
        cosmo_cfg = {'Omega_c': 0.27}
        with pytest.raises(ValueError, match='Neither sigma8 nor A_s'):
            validate_amplitude_parameter(cosmo_cfg)

    def test_both_defined_raises(self):
        cosmo_cfg = {'Omega_c': 0.27, 'sigma8': 0.8, 'A_s': 2.1e-9}
        with pytest.raises(ValueError, match='mutually exclusive'):
            validate_amplitude_parameter(cosmo_cfg)

    def test_error_message_mentions_both_parameters(self):
        cosmo_cfg = {'sigma8': 0.831, 'A_s': 2.1e-9}
        with pytest.raises(ValueError, match='sigma8') as exc_info:
            validate_amplitude_parameter(cosmo_cfg)
        assert 'A_s' in str(exc_info.value)

    def test_sigma8_none_A_s_defined_passes(self):
        # Explicit None for sigma8 is treated as not defined
        cosmo_cfg = {'sigma8': None, 'A_s': 2.1e-9}
        validate_amplitude_parameter(cosmo_cfg)  # Should not raise

    def test_A_s_none_sigma8_defined_passes(self):
        # Explicit None for A_s is treated as not defined
        cosmo_cfg = {'sigma8': 0.8, 'A_s': None}
        validate_amplitude_parameter(cosmo_cfg)  # Should not raise

    def test_both_none_raises(self):
        cosmo_cfg = {'sigma8': None, 'A_s': None}
        with pytest.raises(ValueError, match='Neither sigma8 nor A_s'):
            validate_amplitude_parameter(cosmo_cfg)

    def test_read_fisher_return_order(self, tmp_path):
        """Test that read_fisher_from_file returns (fisher, fiducials) in correct order"""
        base = str(tmp_path / "order_test")
        fiducials = np.array([1.0, 2.0])
        fisher = np.array([[3.0, 4.0], [5.0, 6.0]])

        np.savetxt(f"{base}_fiducials.dat", fiducials)
        np.savetxt(f"{base}_fisher.dat", fisher)

        result1, result2 = read_fisher_from_file(base)

        # First return value should be the Fisher matrix
        assert np.allclose(result1, fisher)
        # Second return value should be fiducials
        assert np.allclose(result2, fiducials)

    def test_read_fisher_with_base_path_with_special_chars(self, tmp_path):
        """Test read_fisher_from_file with special characters in path"""
        special_dir = tmp_path / "test_dir_with_underscores"
        special_dir.mkdir()
        base = str(special_dir / "my_test_file")

        fiducials = np.array([1.0, 2.0])
        fisher = np.array([[1.0, 0.5], [0.5, 2.0]])

        np.savetxt(f"{base}_fiducials.dat", fiducials)
        np.savetxt(f"{base}_fisher.dat", fisher)

        fisher_read, fid_read = read_fisher_from_file(base)

        assert np.allclose(fisher_read, fisher)
        assert np.allclose(fid_read, fiducials)


_SENTINEL = object()


def _cmb_cfg(z_cfg=_SENTINEL, z_fac=_SENTINEL, n_fac=1, cov_type='gaus_internal',
             cmb_lensing=True, factory=True):
    """
    Build a minimal config exercising the z_source cross-check.

    `_SENTINEL` means "omit the key entirely", which is distinct from setting
    it -- omission is what makes firecrown's own default apply, and that
    distinction is the whole point of several of these tests.
    """
    config = {'cov_options': {'cov_type': cov_type}}
    if cmb_lensing:
        config['cmb_lensing'] = {} if z_cfg is _SENTINEL else {'z_source': z_cfg}
    if factory:
        facs = []
        for _ in range(n_fac):
            facs.append({'type_source': 'default'} if z_fac is _SENTINEL
                        else {'type_source': 'default', 'z_source': z_fac})
        config['Firecrown_Factory'] = {'TwoPointFactory': {'cmb_factories': facs}}
    return config


class TestValidateCMBZSource:
    """Tests for validate_cmb_z_source -- see the case table in the D5 plan."""

    # -- agreement -------------------------------------------------------
    def test_matching_z_source_passes(self):
        validate_cmb_z_source(_cmb_cfg(z_cfg=1100.0, z_fac=1100.0))

    def test_matching_non_default_z_source_passes(self):
        validate_cmb_z_source(_cmb_cfg(z_cfg=950.0, z_fac=950.0,
                                       cov_type='gaus_internal'))

    def test_int_and_float_z_source_compare_equal(self):
        # YAML happily yields an int here; it must not read as a mismatch.
        validate_cmb_z_source(_cmb_cfg(z_cfg=1100, z_fac=1100.0))

    # -- disagreement ----------------------------------------------------
    def test_mismatched_z_source_raises(self):
        with pytest.raises(ValueError, match='cmb_factories'):
            validate_cmb_z_source(_cmb_cfg(z_cfg=950.0, z_fac=1100.0))

    def test_error_message_names_both_values(self):
        with pytest.raises(ValueError) as exc_info:
            validate_cmb_z_source(_cmb_cfg(z_cfg=950.0, z_fac=1100.0))
        msg = str(exc_info.value)
        assert '950.0' in msg and '1100.0' in msg

    def test_factory_omitting_z_source_matches_default_passes(self):
        # Case 3b: absent key means firecrown's default, which agrees here.
        validate_cmb_z_source(_cmb_cfg(z_cfg=CMB_Z_SOURCE_DEFAULT))

    def test_factory_omitting_z_source_raises_against_non_default(self):
        # Case 3: the headline failure -- z_source moved, factory forgotten.
        with pytest.raises(ValueError, match="firecrown's default"):
            validate_cmb_z_source(_cmb_cfg(z_cfg=950.0))

    def test_cmb_lensing_omitting_z_source_matches_factory_default_passes(self):
        validate_cmb_z_source(_cmb_cfg(z_fac=CMB_Z_SOURCE_DEFAULT))

    def test_cmb_lensing_omitting_z_source_raises_against_non_default(self):
        with pytest.raises(ValueError, match='cmb_factories'):
            validate_cmb_z_source(_cmb_cfg(z_fac=950.0))

    # -- factory list shape ----------------------------------------------
    def test_multiple_factories_all_agreeing_passes(self):
        validate_cmb_z_source(_cmb_cfg(z_cfg=950.0, z_fac=950.0, n_fac=3))

    def test_multiple_factories_one_disagreeing_raises(self):
        config = _cmb_cfg(z_cfg=950.0, z_fac=950.0, n_fac=3)
        config['Firecrown_Factory']['TwoPointFactory']['cmb_factories'][2]['z_source'] = 1100.0
        with pytest.raises(ValueError) as exc_info:
            validate_cmb_z_source(config)
        assert 'cmb_factories[2]' in str(exc_info.value)

    def test_empty_cmb_factories_raises(self):
        config = _cmb_cfg(z_cfg=1100.0, n_fac=0)
        with pytest.raises(ValueError, match='empty or absent'):
            validate_cmb_z_source(config)

    def test_missing_cmb_factories_key_raises(self):
        config = _cmb_cfg(z_cfg=1100.0, factory=False)
        config['Firecrown_Factory'] = {'TwoPointFactory': {}}
        with pytest.raises(ValueError, match='empty or absent'):
            validate_cmb_z_source(config)

    def test_no_firecrown_factory_passes(self):
        # Case 6: the ConstGaussian path uses augur's own CMBConvergence,
        # which does carry the configured value. Nothing to disagree with.
        validate_cmb_z_source(_cmb_cfg(z_cfg=950.0, factory=False))

    def test_unknown_factory_type_stays_quiet(self):
        # load_likelihood_from_yaml raises its own NameError for this; a
        # second, more confusing message here would not help.
        config = _cmb_cfg(z_cfg=950.0, factory=False)
        config['Firecrown_Factory'] = {'SomeOtherFactory': {}}
        validate_cmb_z_source(config)

    # -- kappa configured on only one side --------------------------------
    def test_cmb_factories_without_cmb_lensing_warns(self):
        # Case 5: no kappa tracer is written, so the factories are inert.
        # Warn -- raising would break configs that work today.
        with pytest.warns(UserWarning, match='inert'):
            validate_cmb_z_source(_cmb_cfg(z_fac=1100.0, cmb_lensing=False))

    def test_no_cmb_lensing_and_no_cmb_factories_passes(self):
        validate_cmb_z_source(_cmb_cfg(cmb_lensing=False, n_fac=0))

    # -- the TJPCov guard --------------------------------------------------
    def test_tjpcov_with_default_z_source_passes(self):
        validate_cmb_z_source(_cmb_cfg(z_cfg=TJPCOV_CMB_Z_SOURCE,
                                       z_fac=TJPCOV_CMB_Z_SOURCE,
                                       cov_type='tjpcov'))

    def test_tjpcov_with_non_default_z_source_raises(self):
        with pytest.raises(ValueError, match='124'):
            validate_cmb_z_source(_cmb_cfg(z_cfg=950.0, z_fac=950.0,
                                           cov_type='tjpcov'))

    def test_tjpcov_without_cmb_lensing_passes(self):
        # Case 9c: TJPCov never reaches its kappa branch without a kappa tracer.
        validate_cmb_z_source(_cmb_cfg(cmb_lensing=False, n_fac=0,
                                       cov_type='tjpcov'))

    def test_gaus_internal_with_non_default_z_source_passes(self):
        # Case 9d: proves the guard is scoped to tjpcov and does not quietly
        # narrow which configs augur accepts.
        validate_cmb_z_source(_cmb_cfg(z_cfg=950.0, z_fac=950.0,
                                       cov_type='gaus_internal'))

    def test_srd_with_non_default_z_source_passes(self):
        validate_cmb_z_source(_cmb_cfg(z_cfg=950.0, z_fac=950.0, cov_type='SRD'))

    def test_check_cov_type_false_skips_tjpcov_guard(self):
        # The use_sacc path returns before any covariance is computed.
        validate_cmb_z_source(_cmb_cfg(z_cfg=950.0, z_fac=950.0, cov_type='tjpcov'),
                              check_cov_type=False)

    # -- the use_sacc signature -------------------------------------------
    def test_explicit_z_source_overrides_config(self):
        # On the use_sacc path the sacc wins; cmb_lensing.z_source is inert.
        config = _cmb_cfg(z_cfg=1100.0, z_fac=950.0)
        validate_cmb_z_source(config, z_source=950.0, check_cov_type=False)

    def test_explicit_z_source_mismatch_raises(self):
        config = _cmb_cfg(z_cfg=950.0, z_fac=950.0)
        with pytest.raises(ValueError, match='cmb_factories'):
            validate_cmb_z_source(config, z_source=1100.0, check_cov_type=False)

    def test_explicit_z_source_disagreeing_with_config_warns(self):
        # Case 13: the sacc wins, but a user who set both is confused.
        config = _cmb_cfg(z_cfg=1100.0, z_fac=950.0)
        with pytest.warns(UserWarning, match='stale'):
            validate_cmb_z_source(config, z_source=950.0, check_cov_type=False)

    def test_origin_appears_in_the_message(self):
        config = _cmb_cfg(z_cfg=950.0, z_fac=950.0)
        with pytest.raises(ValueError) as exc_info:
            validate_cmb_z_source(config, z_source=1100.0,
                                  origin='the sacc metadata', check_cov_type=False)
        assert 'the sacc metadata' in str(exc_info.value)
