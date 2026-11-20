import numpy as np
import pytest
from astropy.table import Table
from augur.utils.config_io import (parse_config, read_fisher_from_file, write_fisher_to_file,
                                   validate_amplitude_parameter)


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


def _write_augur_fisher(tmp_path, fisher, names, values):
    """Write a Fisher matrix and parameter file the way Analyze.get_fisher_matrix does."""
    fisher_path, fid_path = str(tmp_path / 'fisher.dat'), str(tmp_path / 'pars.dat')
    np.savetxt(fisher_path, fisher)
    Table(np.array(values, dtype=float), names=names).write(fid_path, format='ascii',
                                                            overwrite=True)
    return fisher_path, fid_path


class TestReadFisherFromFile:
    """Tests for read_fisher_from_file, which reads Augur's output format"""

    def test_reads_matrix_names_and_fiducials(self, tmp_path):
        fisher = np.array([[1.5, 0.2, -0.1], [0.2, 2.5, 0.3], [-0.1, 0.3, 1.8]])
        names = ['Omega_c', 'sigma8', 'lens0_bias']
        paths = _write_augur_fisher(tmp_path, fisher, names, [0.27, 0.83, 1.4])
        fisher_read, names_read, fid_read = read_fisher_from_file(*paths)
        assert np.array_equal(fisher_read, fisher)
        assert names_read == names
        assert fid_read == pytest.approx({'Omega_c': 0.27, 'sigma8': 0.83, 'lens0_bias': 1.4})

    def test_single_parameter(self, tmp_path):
        paths = _write_augur_fisher(tmp_path, np.array([[4.0]]), ['h'], [0.7])
        fisher_read, names_read, fid_read = read_fisher_from_file(*paths)
        assert fisher_read.shape == (1, 1)
        assert names_read == ['h']
        assert fid_read['h'] == pytest.approx(0.7)

    def test_rename(self, tmp_path):
        paths = _write_augur_fisher(tmp_path, np.eye(2), ['Om', 'h'], [0.3, 0.7])
        _, names_read, fid_read = read_fisher_from_file(*paths, rename={'Om': 'Omega_m'})
        assert names_read == ['Omega_m', 'h']
        assert fid_read['Omega_m'] == pytest.approx(0.3)

    def test_rename_of_unknown_parameter_raises(self, tmp_path):
        paths = _write_augur_fisher(tmp_path, np.eye(2), ['Om', 'h'], [0.3, 0.7])
        with pytest.raises(ValueError, match='not parameters'):
            read_fisher_from_file(*paths, rename={'Omega_c': 'Omega_m'})

    def test_rename_to_duplicate_raises(self, tmp_path):
        paths = _write_augur_fisher(tmp_path, np.eye(2), ['Om', 'h'], [0.3, 0.7])
        with pytest.raises(ValueError, match='not unique'):
            read_fisher_from_file(*paths, rename={'Om': 'h'})

    def test_shape_mismatch_raises(self, tmp_path):
        paths = _write_augur_fisher(tmp_path, np.eye(3), ['Om', 'h'], [0.3, 0.7])
        with pytest.raises(ValueError, match='shape'):
            read_fisher_from_file(*paths)

    def test_asymmetric_matrix_raises(self, tmp_path):
        fisher = np.array([[1.0, 0.5], [0.2, 1.0]])
        paths = _write_augur_fisher(tmp_path, fisher, ['Om', 'h'], [0.3, 0.7])
        with pytest.raises(ValueError, match='symmetric'):
            read_fisher_from_file(*paths)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_fisher_from_file(str(tmp_path / 'fisher.dat'), str(tmp_path / 'pars.dat'))


class TestWriteFisherToFile:
    """Tests for write_fisher_to_file, the inverse of read_fisher_from_file"""

    def test_round_trip(self, tmp_path):
        fisher = np.array([[2.0, 0.3], [0.3, 1.0]])
        paths = str(tmp_path / 'fisher.dat'), str(tmp_path / 'pars.dat')
        write_fisher_to_file(fisher, ['Omega_m', 'h'], {'h': 0.7, 'Omega_m': 0.3}, *paths)
        fisher_read, names_read, fid_read = read_fisher_from_file(*paths)
        assert np.array_equal(fisher_read, fisher)
        assert names_read == ['Omega_m', 'h']
        assert fid_read == pytest.approx({'Omega_m': 0.3, 'h': 0.7})

    def test_parameter_file_is_augurs_format(self, tmp_path):
        paths = str(tmp_path / 'fisher.dat'), str(tmp_path / 'pars.dat')
        write_fisher_to_file(np.eye(2), ['Omega_m', 'h'], {'Omega_m': 0.3, 'h': 0.7}, *paths)
        with open(paths[1]) as f:
            assert f.read() == 'Omega_m h\n0.3 0.7\n'

    def test_missing_fiducial_raises(self, tmp_path):
        paths = str(tmp_path / 'fisher.dat'), str(tmp_path / 'pars.dat')
        with pytest.raises(ValueError, match='No fiducial values'):
            write_fisher_to_file(np.eye(2), ['Omega_m', 'h'], {'Omega_m': 0.3}, *paths)

    def test_shape_mismatch_raises(self, tmp_path):
        paths = str(tmp_path / 'fisher.dat'), str(tmp_path / 'pars.dat')
        with pytest.raises(ValueError, match='shape'):
            write_fisher_to_file(np.eye(3), ['Omega_m', 'h'], {'Omega_m': 0.3, 'h': 0.7},
                                 *paths)


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
