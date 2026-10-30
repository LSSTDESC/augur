"""Tests for the SRD covariance guards in augur.utils.cov_utils.

`get_SRD_cov` had no test coverage at all, which is how it kept the failure
mode these tests pin: it loops only over its own hard-coded combination lists
and writes into a zeros array, so any data the lists do not name -- CMB lensing
above all -- is left at zero with no error, giving a singular covariance and a
NaN Fisher matrix much later and somewhere else.
"""
import numpy as np
import pytest
import sacc

from augur.generate_utils.cmb_lensing import CMB_TRACER_NAME, add_cmb_tracer
from augur.utils.cov_utils import cmb_lensing_tracers, get_SRD_cov, _srd_data_type


def _sacc_3x2pt(n_src=5, n_lens=5):
    """A sacc with the SRD Y1 tracer names and one data point per pair."""
    S = sacc.Sacc()
    z = np.linspace(0.1, 2.0, 16)
    nz = np.exp(-0.5 * ((z - 0.8) / 0.3) ** 2)
    for i in range(n_src):
        S.add_tracer('NZ', f'src{i}', z, nz, quantity='galaxy_shear')
    for i in range(n_lens):
        S.add_tracer('NZ', f'lens{i}', z, nz, quantity='galaxy_density')
    return S


def _add_kappa(S, z_source=1100.0):
    add_cmb_tracer(S, {}, z_source=z_source)
    return S


# ---------------------------------------------------------------------------
# cmb_lensing_tracers
# ---------------------------------------------------------------------------

def test_cmb_lensing_tracers_finds_nothing_in_a_3x2pt_sacc():
    assert cmb_lensing_tracers(_sacc_3x2pt()) == []


def test_cmb_lensing_tracers_finds_the_kappa_tracer():
    S = _add_kappa(_sacc_3x2pt())
    assert cmb_lensing_tracers(S) == [CMB_TRACER_NAME]


def test_cmb_lensing_tracers_dispatches_on_quantity_not_name():
    """A sacc from another tool may name its kappa tracer anything.

    Detecting on `quantity` is what makes the guard work on saccs augur did
    not write -- which is exactly the case where nobody has checked.
    """
    S = _sacc_3x2pt()
    ells = np.array([0.0, 3000.0])
    S.add_tracer('Map', 'kappa_planck', 0, ells, np.ones_like(ells),
                 quantity='cmb_convergence', metadata={'z_lss': 1100.0})
    assert cmb_lensing_tracers(S) == ['kappa_planck']


# ---------------------------------------------------------------------------
# get_SRD_cov: the kappa guard
# ---------------------------------------------------------------------------

def test_get_SRD_cov_refuses_a_kappa_sacc():
    """The whole point of D9: refuse, rather than silently return zero rows."""
    S = _add_kappa(_sacc_3x2pt())
    with pytest.raises(ValueError, match='CMB lensing') as exc_info:
        get_SRD_cov({'SRD_cov_path': 'unused.npy'}, S)
    msg = str(exc_info.value)
    assert CMB_TRACER_NAME in msg
    # The message has to say what to do instead, not just what is wrong.
    assert 'gaus_internal' in msg


def test_get_SRD_cov_refuses_before_reading_the_file():
    """The guard must fire before np.load, so a kappa run fails fast."""
    S = _add_kappa(_sacc_3x2pt())
    # A path that does not exist: reaching np.load would raise FileNotFoundError.
    with pytest.raises(ValueError, match='CMB lensing'):
        get_SRD_cov({'SRD_cov_path': '/nonexistent/Y1_cov.npy'}, S)


def test_get_SRD_cov_still_requires_its_path():
    with pytest.raises(ValueError, match='SRD_cov_path'):
        get_SRD_cov({}, _sacc_3x2pt())


# ---------------------------------------------------------------------------
# get_SRD_cov: the release is chosen by sniffing the filename
# ---------------------------------------------------------------------------

def test_get_SRD_cov_detects_a_y10_sacc_read_as_y1(tmp_path):
    """A Y10 covariance saved without 'Y10' in its name silently used the Y1 list.

    The Y1 tracer names are a subset of the Y10 ones, so a missing-tracer check
    alone cannot see this direction -- and it is the more likely mistake.
    """
    cov_path = tmp_path / 'srd_cov.npy'          # deliberately no 'Y10'
    np.save(cov_path, np.eye(4))
    S = _sacc_3x2pt(n_src=5, n_lens=10)          # a Y10-shaped sacc

    with pytest.raises(ValueError, match='full Y10 tracer set') as exc_info:
        get_SRD_cov({'SRD_cov_path': str(cov_path)}, S)
    assert 'rename' in str(exc_info.value)


def test_get_SRD_cov_detects_a_y1_sacc_read_as_y10(tmp_path):
    cov_path = tmp_path / 'Y10_3x2_SRD_cov.npy'
    np.save(cov_path, np.eye(4))
    S = _sacc_3x2pt(n_src=5, n_lens=5)           # only Y1 has 5 lens bins

    with pytest.raises(ValueError, match='Y10') as exc_info:
        get_SRD_cov({'SRD_cov_path': str(cov_path)}, S)
    msg = str(exc_info.value)
    assert 'lens5' in msg          # names what is actually missing
    assert 'Y1 layout' in msg      # and points at the likely cause


# ---------------------------------------------------------------------------
# _srd_data_type: a named error instead of a bare IndexError
# ---------------------------------------------------------------------------

def test_srd_data_type_names_the_missing_pair():
    """`S.get_data_types(...)[0]` used to raise a bare IndexError naming nothing."""
    S = _sacc_3x2pt()
    with pytest.raises(ValueError, match='src0') as exc_info:
        _srd_data_type(S, ('src0', 'src1'))
    assert 'src1' in str(exc_info.value)


def test_srd_data_type_returns_the_type_when_present():
    S = _sacc_3x2pt()
    ell = np.array([100.0, 200.0])
    cl = np.array([1e-8, 5e-9])
    S.add_ell_cl('galaxy_shear_cl_ee', 'src0', 'src1', ell, cl)
    assert _srd_data_type(S, ('src0', 'src1')) == 'galaxy_shear_cl_ee'
