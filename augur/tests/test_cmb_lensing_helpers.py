"""Unit tests for the CMB lensing generation helpers."""

import numpy as np
import pytest
import sacc

from augur.generate_utils.cmb_lensing import (
    CMB_TRACER_NAME,
    _convert_noise_convention,
    _tracer_pair,
    add_cmb_tracer,
    get_cmb_noise,
)
from augur.utils.cov_utils import get_noise_power


# ---------------------------------------------------------------------------
# The kappa sacc tracer
# ---------------------------------------------------------------------------

def test_add_cmb_tracer_is_a_map_tracer_with_both_conventions():
    """kappa needs `quantity` for TJPCov and `z_lss` metadata for firecrown.

    It must be a MapTracer, not an NZ tracer: firecrown dispatches on the sacc
    tracer class, so an NZ kappa tracer is read back as a tomographic bin.
    """
    S, sources = sacc.Sacc(), {}
    add_cmb_tracer(S, sources, z_source=1100.0, lmax=3000.0)

    tracer = S.get_tracer(CMB_TRACER_NAME)
    assert isinstance(tracer, sacc.tracers.MapTracer)
    assert tracer.quantity == 'cmb_convergence'
    assert tracer.metadata['z_lss'] == 1100.0
    assert tracer.spin == 0
    assert sources[CMB_TRACER_NAME].z_source == 1100.0


def test_add_cmb_tracer_is_idempotent():
    S, sources = sacc.Sacc(), {}
    add_cmb_tracer(S, sources, z_source=1100.0)
    add_cmb_tracer(S, sources, z_source=1100.0)
    assert list(S.tracers) == [CMB_TRACER_NAME]


# ---------------------------------------------------------------------------
# Tracer pairing and tracer_combs arity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('key, comb, expected', [
    ('cmb_convergence_cl', [], (CMB_TRACER_NAME, CMB_TRACER_NAME)),
    ('cmbGalaxy_convergenceDensity_cl', [3], (CMB_TRACER_NAME, 'lens3')),
    ('cmbGalaxy_convergenceShear_cl_e', [2], (CMB_TRACER_NAME, 'src2')),
])
def test_tracer_pair_puts_kappa_first(key, comb, expected):
    """firecrown orders CMB < Clusters < Galaxies; kappa second auto-swaps."""
    assert _tracer_pair(key, comb) == expected


@pytest.mark.parametrize('key, comb', [
    ('cmb_convergence_cl', [0]),
    ('cmbGalaxy_convergenceDensity_cl', []),
    ('cmbGalaxy_convergenceDensity_cl', [0, 1]),
    ('cmbGalaxy_convergenceShear_cl_e', [0, 1]),
])
def test_tracer_pair_rejects_wrong_arity(key, comb):
    with pytest.raises(ValueError):
        _tracer_pair(key, comb)


# ---------------------------------------------------------------------------
# Reconstruction noise
# ---------------------------------------------------------------------------

def test_noise_convention_kappa_is_identity():
    ells = np.array([10.0, 100.0])
    n_ell = np.array([1e-8, 2e-8])
    np.testing.assert_allclose(
        _convert_noise_convention(ells, n_ell, 'kappa'), n_ell
    )


def test_noise_convention_phi_and_dd():
    ells = np.array([10.0, 100.0])
    n_ell = np.ones(2)
    fac = ells * (ells + 1.0)
    np.testing.assert_allclose(
        _convert_noise_convention(ells, n_ell, 'phi'), (fac / 2.0) ** 2
    )
    np.testing.assert_allclose(
        _convert_noise_convention(ells, n_ell, 'dd'), fac / 4.0
    )


def test_noise_convention_unknown_raises():
    with pytest.raises(ValueError, match='convention'):
        _convert_noise_convention(np.array([10.0]), np.array([1.0]), 'bananas')


def test_scalar_noise_broadcasts():
    ells = np.array([10.0, 100.0, 1000.0])
    np.testing.assert_allclose(
        get_cmb_noise({'noise_cl': 3e-8}, ells), np.full(3, 3e-8)
    )


def test_missing_noise_defaults_to_zero():
    np.testing.assert_allclose(get_cmb_noise({}, np.array([10.0])), np.zeros(1))


def _write_noise_file(tmp_path, ells, n_ell):
    path = tmp_path / 'nlkk.dat'
    np.savetxt(path, np.column_stack([ells, n_ell]))
    return str(path)


def test_tabulated_noise_is_interpolated_not_indexed(tmp_path):
    """These tables are (L, N_L) pairs; the row index is not the multipole."""
    ell_tab = np.array([100.0, 200.0, 400.0])
    n_tab = np.array([1e-8, 2e-8, 4e-8])
    cfg = {'noise': {'file': _write_noise_file(tmp_path, ell_tab, n_tab),
                     'ell_col': 0, 'nl_col': 1, 'convention': 'kappa'}}

    # On the tabulated points the curve is reproduced exactly...
    np.testing.assert_allclose(get_cmb_noise(cfg, ell_tab), n_tab, rtol=1e-10)
    # ...and a point between them is interpolated, not taken from a row.
    mid = get_cmb_noise(cfg, np.array([200.0 * np.sqrt(2)]))[0]
    assert 2e-8 < mid < 4e-8


def test_noise_is_inf_outside_the_reconstruction_band(tmp_path):
    """Zero there would silently make the reconstruction look noiseless."""
    cfg = {'noise': {'file': _write_noise_file(
        tmp_path, np.array([100.0, 200.0]), np.array([1e-8, 2e-8]))}}
    with pytest.warns(UserWarning, match='outside the reconstruction band'):
        n_ell = get_cmb_noise(cfg, np.array([10.0, 150.0, 5000.0]))
    assert np.isinf(n_ell[0])
    assert np.isfinite(n_ell[1])
    assert np.isinf(n_ell[2])


def test_noise_file_rejects_out_of_range_column(tmp_path):
    cfg = {'noise': {'file': _write_noise_file(
        tmp_path, np.array([100.0, 200.0]), np.array([1e-8, 2e-8])),
        'nl_col': 7}}
    with pytest.raises(ValueError, match='columns'):
        get_cmb_noise(cfg, np.array([150.0]))


def test_noise_file_wins_over_scalar_with_a_warning(tmp_path):
    cfg = {'noise_cl': 1.0,
           'noise': {'file': _write_noise_file(
               tmp_path, np.array([100.0, 200.0]), np.array([1e-8, 2e-8]))}}
    with pytest.warns(UserWarning, match='ignoring `noise_cl`'):
        n_ell = get_cmb_noise(cfg, np.array([150.0]))
    assert n_ell[0] < 1.0


# ---------------------------------------------------------------------------
# Regression: a kappa tracer in the sacc must not break galaxy noise
# ---------------------------------------------------------------------------

def test_galaxy_noise_still_works_with_a_kappa_tracer_present():
    """get_noise_power loops every tracer regardless of which was asked about.

    Without the guard, the mere presence of a Map tracer (no `.nz`, and a
    prefix that is neither 'src' nor 'lens') raises for `lens0` as well.
    """
    S = sacc.Sacc()
    z = np.linspace(0.1, 2.0, 20)
    nz = np.exp(-0.5 * ((z - 0.5) / 0.2) ** 2)
    S.add_tracer('NZ', 'lens0', z, nz)
    S.add_tracer('NZ', 'src0', z, nz)
    config = {'sources': {'ndens': 10, 'ellipticity_error': 0.26},
              'lenses': {'ndens': 18}}

    without_kappa = get_noise_power(config, S, 'lens0')

    add_cmb_tracer(S, {}, z_source=1100.0)
    with_kappa = get_noise_power(config, S, 'lens0')

    assert with_kappa == without_kappa


def test_noise_power_still_refuses_the_kappa_tracer_itself():
    """kappa noise comes from the config, never from an n(z)."""
    S = sacc.Sacc()
    add_cmb_tracer(S, {}, z_source=1100.0)
    with pytest.raises(NotImplementedError):
        get_noise_power({}, S, CMB_TRACER_NAME)
