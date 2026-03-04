import numpy as np
import pytest

from augur.utils.diff_utils import five_pt_stencil
from augur.utils.config_io import parse_config, read_fisher_from_file


def test_five_pt_stencil_scalar():
    def f(x):
        return x**3
    x0 = 1.5
    der = five_pt_stencil(f, x0, h=1e-6)
    assert pytest.approx(der, rel=1e-6) == 3.0 * x0**2


def test_five_pt_stencil_vector():
    # Define a function of a 2-component vector that returns one value per row
    def f(X):
        # X has shape (n, 2) where each row is a perturbed vector
        return np.array([row[0]**2 + 3.0 * row[1] for row in X])

    x0 = np.array([1.0, 2.0])
    der = five_pt_stencil(f, x0, h=1e-6)
    # Expect partial derivatives [2*x0[0], 3.0]
    assert der.shape[0] == 2
    assert pytest.approx(der[0], rel=1e-6) == 2.0 * x0[0]
    assert pytest.approx(der[1], rel=1e-6) == 3.0


def test_parse_config_accepts_dict():
    d = {'a': 1}
    assert parse_config(d) is d


def test_parse_config_rejects_other_types():
    with pytest.raises(ValueError):
        parse_config(123)


def test_read_fisher_from_file(tmp_path):
    base = tmp_path / "basefile"
    fid = np.array([1.0, 2.0, 3.0])
    fisher = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    np.savetxt(str(base) + "_fiducials.dat", fid)
    np.savetxt(str(base) + "_fisher.dat", fisher)

    f_read, fid_read = read_fisher_from_file(str(base))
    # The function returns (fisher, fiducials)
    assert np.allclose(f_read, fisher)
    assert np.allclose(fid_read, fid)


def test_read_fisher_from_file_missing(tmp_path):
    base = tmp_path / "does_not_exist"
    with pytest.raises(RuntimeError):
        read_fisher_from_file(str(base))
