import numpy as np
import pytest
from augur.utils.config_io import parse_config, read_fisher_from_file


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
        with pytest.raises(ValueError, match="""config must be a dictionary or path to
                            a config file"""):
            parse_config(None)

    def test_parse_config_rejects_integer(self):
        """Test that parse_config rejects integer input"""
        with pytest.raises(ValueError, match="""config must be a dictionary or path to
                            a config file"""):
            parse_config(123)

    def test_parse_config_rejects_list(self):
        """Test that parse_config rejects list input"""
        with pytest.raises(ValueError, match="""config must be a dictionary or path to
                            a config file"""):
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
