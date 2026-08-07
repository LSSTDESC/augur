import logging
import pytest
import numpy as np
import pyccl as ccl
import sacc

from augur.generate import _get_tracers, _get_scale_cuts, _build_tp_filters_from_sacc
from augur.generate_utils.real_space import _get_scale_cuts_real_space


def test_get_tracers_galaxy_density():
    assert _get_tracers('galaxy_density_cl', (0, 1)) == ('lens0', 'lens1')


def test_get_tracers_shear_ee():
    assert _get_tracers('galaxy_shear_cl_ee', (2, 3)) == ('src2', 'src3')


def test_get_tracers_shear_density():
    assert _get_tracers('galaxy_shearDensity_cl_e', (1, 4)) == ('lens1', 'src4')


def test_get_tracers_unrecognized():
    with pytest.raises(NotImplementedError):
        _get_tracers('unknown_stat', (0, 1))


def test_get_tracers_rejects_cmb_statistics():
    with pytest.raises(NotImplementedError):
        _get_tracers('cmbGalaxy_convergenceDensity_cl', [0])


def test_get_scale_cuts_scalar_lmax():
    stat_cfg = {'lmax': 500, 'tracer_combs': [('lens0', 'src0'), ('lens1', 'src1')]}
    lmax, kmax = _get_scale_cuts(stat_cfg, ('lens0', 'src0'))
    assert lmax == 500
    assert kmax is None


def test_get_scale_cuts_list_lmax():
    stat_cfg = {'lmax': [100, 200], 'tracer_combs': [('lens0', 'src0'), ('lens1', 'src1')]}
    lmax, _ = _get_scale_cuts(stat_cfg, ('lens1', 'src1'))
    assert lmax == 200


def test_get_scale_cuts_list_length_mismatch():
    stat_cfg = {'lmax': [100], 'tracer_combs': [('lens0', 'src0'), ('lens1', 'src1')]}
    with pytest.raises(ValueError):
        _get_scale_cuts(stat_cfg, ('lens1', 'src1'))


def test_get_scale_cuts_both_kmax_lmax():
    stat_cfg = {'lmax': 100, 'kmax': 0.2}
    with pytest.raises(ValueError):
        _get_scale_cuts(stat_cfg, ('lens0', 'src0'))


def test_get_scale_cuts_invalid_kmax_type():
    stat_cfg = {'kmax': 'not_a_number'}
    with pytest.raises(ValueError):
        _get_scale_cuts(stat_cfg, ('lens0', 'src0'))


def test_get_scale_cuts_list_kmax():
    stat_cfg = {'kmax': [0.2, 0.4], 'tracer_combs': [('lens0', 'src0'), ('lens1', 'src1')]}
    _, kmax = _get_scale_cuts(stat_cfg, ('lens1', 'src1'))
    assert kmax == 0.4


def test_get_scale_cuts_real_space_list_theta():
    stat_cfg = {
        'theta_min': [2.0, 5.0],
        'theta_max': [100.0, 200.0],
        'tracer_combs': [('lens0', 'src0'), ('lens1', 'src1')],
    }
    theta_min, theta_max = _get_scale_cuts_real_space(stat_cfg, ('lens1', 'src1'))
    assert theta_min == 5.0
    assert theta_max == 200.0


def test_get_scale_cuts_real_space_list_r():
    class FakeDndz:
        def __init__(self, z):
            self.zav = z

    stat_cfg = {
        'r_min': [6.0, 12.0],
        'r_max': [120.0, 240.0],
        'tracer_combs': [('lens0', 'src0'), ('lens1', 'src1')],
    }
    cosmo = ccl.CosmologyVanillaLCDM()
    dndz = {'lens1': FakeDndz(0.5), 'src1': FakeDndz(1.0)}
    theta_min, theta_max = _get_scale_cuts_real_space(
        stat_cfg,
        ('lens1', 'src1'),
        cosmo=cosmo,
        dndz=dndz,
        tr1='lens1',
        tr2='src1',
    )
    assert theta_min is not None
    assert theta_max is not None
    assert theta_max > theta_min


def test_get_scale_cuts_per_bin_lmax_conservative_min():
    # Symmetric rectangular grid: 3 lens bins + 3 src bins = 6 values (not 9 pairs)
    # Per-bin list: [lens0, lens1, lens2, src0, src1, src2]
    # For (lens0, src1): lens bin=0→offset 0+0=0→800, src bin=1→offset 3+1=4→600
    # min(800, 600) = 600
    stat_cfg = {
        'lmax': [800, 600, 300, 700, 600, 550],
        'tracer_combs': [
            ('lens0', 'src0'), ('lens0', 'src1'), ('lens0', 'src2'),
            ('lens1', 'src0'), ('lens1', 'src1'), ('lens1', 'src2'),
            ('lens2', 'src0'), ('lens2', 'src1'), ('lens2', 'src2'),
        ]
    }
    lmax, _ = _get_scale_cuts(stat_cfg, ('lens0', 'src1'))
    assert lmax == 600


def test_get_scale_cuts_per_bin_kmax_conservative_min():
    # Symmetric rectangular grid: 3 lens bins + 3 src bins = 6 values
    # Per-bin list: [lens0, lens1, lens2, src0, src1, src2]
    # For (lens0, src1): lens bin=0→offset 0+0=0→0.4, src bin=1→offset 3+1=4→0.3
    # min(0.4, 0.3) = 0.3
    stat_cfg = {
        'kmax': [0.4, 0.2, 0.5, 0.35, 0.3, 0.25],
        'tracer_combs': [
            ('lens0', 'src0'), ('lens0', 'src1'), ('lens0', 'src2'),
            ('lens1', 'src0'), ('lens1', 'src1'), ('lens1', 'src2'),
            ('lens2', 'src0'), ('lens2', 'src1'), ('lens2', 'src2'),
        ]
    }
    _, kmax = _get_scale_cuts(stat_cfg, ('lens0', 'src1'))
    assert kmax == 0.3


def test_get_scale_cuts_per_bin_asymmetric_lmax():
    # Asymmetric rectangular grid: 3 lens bins + 2 src bins = 5 values
    # Per-bin list: [3000, 2500, 2000, 4000, 3500]
    # For (lens1, src1): lens bin=1→offset 0+1=1→2500, src bin=1→offset 3+1=4→3500
    # min(2500, 3500) = 2500
    stat_cfg = {
        'lmax': [3000, 2500, 2000, 4000, 3500],
        'tracer_combs': [
            ('lens0', 'src0'), ('lens0', 'src1'),
            ('lens1', 'src0'), ('lens1', 'src1'),
            ('lens2', 'src0'), ('lens2', 'src1'),
        ]
    }
    lmax, _ = _get_scale_cuts(stat_cfg, ('lens1', 'src1'))
    assert lmax == 2500


def test_get_scale_cuts_real_space_per_bin_theta_min_conservative_max():
    # Symmetric rectangular grid: 3 lens bins + 3 src bins = 6 values
    # Per-bin list: [lens0, lens1, lens2, src0, src1, src2]
    # For (lens0, src1): lens bin=0→offset 0+0=0→2.0, src bin=1→offset 3+1=4→8.0
    # max(2.0, 8.0) = 8.0
    stat_cfg = {
        'theta_min': [2.0, 5.0, 8.0, 3.0, 8.0, 9.0],
        'tracer_combs': [
            ('lens0', 'src0'), ('lens0', 'src1'), ('lens0', 'src2'),
            ('lens1', 'src0'), ('lens1', 'src1'), ('lens1', 'src2'),
            ('lens2', 'src0'), ('lens2', 'src1'), ('lens2', 'src2'),
        ],
    }
    theta_min, theta_max = _get_scale_cuts_real_space(stat_cfg, ('lens0', 'src1'))
    assert theta_min == 8.0
    assert theta_max is None


def test_get_scale_cuts_real_space_per_bin_theta_max_conservative_min():
    # Symmetric rectangular grid: 3 lens bins + 3 src bins = 6 values
    # Per-bin list: [lens0, lens1, lens2, src0, src1, src2]
    # For (lens0, src1): lens bin=0→offset 0+0=0→120.0, src bin=1→offset 3+1=4→80.0
    # min(120.0, 80.0) = 80.0
    stat_cfg = {
        'theta_max': [120.0, 80.0, 150.0, 110.0, 80.0, 140.0],
        'tracer_combs': [
            ('lens0', 'src0'), ('lens0', 'src1'), ('lens0', 'src2'),
            ('lens1', 'src0'), ('lens1', 'src1'), ('lens1', 'src2'),
            ('lens2', 'src0'), ('lens2', 'src1'), ('lens2', 'src2'),
        ],
    }
    theta_min, theta_max = _get_scale_cuts_real_space(stat_cfg, ('lens0', 'src1'))
    assert theta_min is None
    assert theta_max == 80.0


# ---------------------------------------------------------------------------
# Helpers shared by _build_tp_filters_from_sacc tests
# ---------------------------------------------------------------------------

def _make_sacc_shear():
    """Minimal sacc with two shear tracers and two EE combinations."""
    S = sacc.Sacc()
    z = np.linspace(0.1, 2.0, 20)
    nz0 = np.exp(-0.5 * ((z - 0.5) / 0.2) ** 2)
    nz1 = np.exp(-0.5 * ((z - 1.0) / 0.2) ** 2)
    S.add_tracer('NZ', 'src0', z, nz0)
    S.add_tracer('NZ', 'src1', z, nz1)
    ells = np.array([10., 50., 100., 200., 500., 1000.])
    cls = np.zeros(len(ells))
    S.add_ell_cl('galaxy_shear_cl_ee', 'src0', 'src1', ells, cls)
    S.add_ell_cl('galaxy_shear_cl_ee', 'src0', 'src0', ells, cls)
    return S


def _make_cosmo():
    return ccl.Cosmology(Omega_c=0.27, Omega_b=0.045, h=0.67, sigma8=0.8, n_s=0.96)


# _make_sacc_shear() contains two combinations: (src0, src1) and (src0, src0).
# _build_tp_filters_from_sacc returns two kinds of filters:
#   * "real" filters (interval floored at 0.0) for configured combos present in
#     the sacc; and
#   * "exclusion" filters (interval placed just above the last ell, so no data
#     survives) for sacc combos that are NOT listed in the config, which drops
#     them from the likelihood.
# The last ell in the test sacc is 1000, so exclusion intervals are (1001, 1002).

def _real_filters(filters):
    """Filters that actually keep data: lower bound floored at 0.0."""
    return [f for f in filters if f.interval[0] == 0.0]


def _exclusion_filters(filters):
    """Drop filters for unlisted sacc combos: lower bound above the data range."""
    return [f for f in filters if f.interval[0] > 0.0]


class TestBuildTpFiltersFromSacc:

    def test_no_scale_cuts_returns_one_filter_per_comb(self):
        S = _make_sacc_shear()
        cosmo = _make_cosmo()
        stat_cfg = {'galaxy_shear_cl_ee': {'tracer_combs': [[0, 1]]}}
        filters = _build_tp_filters_from_sacc(stat_cfg, S, cosmo, ignore_sc_likelihood=False)
        assert len(_real_filters(filters)) == 1

    def test_no_scale_cuts_cut_high_is_last_ell(self):
        S = _make_sacc_shear()
        cosmo = _make_cosmo()
        stat_cfg = {'galaxy_shear_cl_ee': {'tracer_combs': [[0, 1]]}}
        filters = _build_tp_filters_from_sacc(stat_cfg, S, cosmo, ignore_sc_likelihood=False)
        # cut_low is floored at 0.0 so the lowest bin's window is fully retained.
        assert _real_filters(filters)[0].interval == (0.0, 1000.0)

    def test_lmax_sets_cut_high(self):
        S = _make_sacc_shear()
        cosmo = _make_cosmo()
        stat_cfg = {'galaxy_shear_cl_ee': {'tracer_combs': [[0, 1]], 'lmax': 300.0}}
        filters = _build_tp_filters_from_sacc(stat_cfg, S, cosmo, ignore_sc_likelihood=False)
        real = _real_filters(filters)
        assert len(real) == 1
        assert real[0].interval[1] == pytest.approx(300.0)

    def test_ignore_sc_likelihood_overrides_lmax(self):
        S = _make_sacc_shear()
        cosmo = _make_cosmo()
        stat_cfg = {'galaxy_shear_cl_ee': {'tracer_combs': [[0, 1]], 'lmax': 300.0}}
        filters = _build_tp_filters_from_sacc(stat_cfg, S, cosmo, ignore_sc_likelihood=True)
        real = _real_filters(filters)
        assert len(real) == 1
        # lmax is ignored; cut_high should be last ell in sacc
        assert real[0].interval[1] == pytest.approx(1000.0)

    def test_kmax_produces_finite_cut_high(self):
        S = _make_sacc_shear()
        cosmo = _make_cosmo()
        stat_cfg = {'galaxy_shear_cl_ee': {'tracer_combs': [[0, 1]], 'kmax': 0.2}}
        filters = _build_tp_filters_from_sacc(stat_cfg, S, cosmo, ignore_sc_likelihood=False)
        real = _real_filters(filters)
        assert len(real) == 1
        cut_high = real[0].interval[1]
        assert np.isfinite(cut_high)
        assert cut_high > 0.0

    def test_ignore_sc_likelihood_overrides_kmax(self):
        S = _make_sacc_shear()
        cosmo = _make_cosmo()
        stat_cfg = {'galaxy_shear_cl_ee': {'tracer_combs': [[0, 1]], 'kmax': 0.2}}
        filters = _build_tp_filters_from_sacc(stat_cfg, S, cosmo, ignore_sc_likelihood=True)
        real = _real_filters(filters)
        assert len(real) == 1
        assert real[0].interval[1] == pytest.approx(1000.0)

    def test_multiple_combs_returns_multiple_filters(self):
        S = _make_sacc_shear()
        cosmo = _make_cosmo()
        stat_cfg = {'galaxy_shear_cl_ee': {'tracer_combs': [[0, 1], [0, 0]]}}
        filters = _build_tp_filters_from_sacc(stat_cfg, S, cosmo, ignore_sc_likelihood=False)
        # Both sacc combos are configured -> two real filters, no exclusions.
        assert len(_real_filters(filters)) == 2
        assert _exclusion_filters(filters) == []

    def test_unlisted_sacc_combo_gets_exclusion_filter(self):
        S = _make_sacc_shear()
        cosmo = _make_cosmo()
        # Only (src0, src1) configured; (src0, src0) present in sacc must be dropped.
        stat_cfg = {'galaxy_shear_cl_ee': {'tracer_combs': [[0, 1]]}}
        filters = _build_tp_filters_from_sacc(stat_cfg, S, cosmo, ignore_sc_likelihood=False)
        excl = _exclusion_filters(filters)
        assert len(excl) == 1
        # Exclusion interval sits above the last ell so no data survives it.
        assert excl[0].interval[0] > 1000.0

    def test_missing_combo_skipped_with_warning(self, caplog):
        S = _make_sacc_shear()
        cosmo = _make_cosmo()
        # src5, src6 not in sacc
        stat_cfg = {'galaxy_shear_cl_ee': {'tracer_combs': [[5, 6]]}}
        with caplog.at_level(logging.WARNING, logger='augur.generate'):
            filters = _build_tp_filters_from_sacc(stat_cfg, S, cosmo, ignore_sc_likelihood=False)
        # No configured combo is present -> no real filters (only exclusions).
        assert _real_filters(filters) == []
        assert 'not found in sacc' in caplog.text.lower()

    def test_present_and_missing_combo_mixed(self, caplog):
        S = _make_sacc_shear()
        cosmo = _make_cosmo()
        stat_cfg = {'galaxy_shear_cl_ee': {'tracer_combs': [[0, 1], [5, 6]]}}
        with caplog.at_level(logging.WARNING, logger='augur.generate'):
            filters = _build_tp_filters_from_sacc(stat_cfg, S, cosmo, ignore_sc_likelihood=False)
        # Only the present combination produces a real filter
        assert len(_real_filters(filters)) == 1
        assert 'not found in sacc' in caplog.text.lower()

    def test_cut_low_is_zero(self):
        S = _make_sacc_shear()
        cosmo = _make_cosmo()
        stat_cfg = {'galaxy_shear_cl_ee': {'tracer_combs': [[0, 1]]}}
        filters = _build_tp_filters_from_sacc(stat_cfg, S, cosmo, ignore_sc_likelihood=False)
        # cut_low is floored at 0.0 (not the first ell) so the lowest bin's full
        # bandpower window is retained under the SUPPORT filter method.
        assert _real_filters(filters)[0].interval[0] == 0.0

    def test_returns_empty_list_when_no_valid_combs(self, caplog):
        S = _make_sacc_shear()
        cosmo = _make_cosmo()
        stat_cfg = {'galaxy_shear_cl_ee': {'tracer_combs': [[9, 9]]}}
        with caplog.at_level(logging.WARNING, logger='augur.generate'):
            filters = _build_tp_filters_from_sacc(stat_cfg, S, cosmo, ignore_sc_likelihood=False)
        # No configured combo is present -> no real filters.
        assert _real_filters(filters) == []
